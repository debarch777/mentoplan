"""Integration tests on a processed patient case (skipped when no processed CT case is available).

They check physical and clinical consistency of the full pipeline rather than exact numbers:
monotonic response to projection, zero implant -> zero change, left/right symmetry, plausible
soft:hard ratios, sensible uncertainty bands and healing course, and the HTTP API."""
import os
from pathlib import Path

import numpy as np
import pytest

from mentoplan.case import CaseStore
from mentoplan.implant.design import ImplantDesign

ROOT = Path(os.environ.get("MENTOPLAN_DATA", Path(__file__).resolve().parents[2] / "data" / "cases"))


def _case():
    if not ROOT.exists():
        return None
    store = CaseStore(ROOT)
    for m in store.list():
        if m.get("status", {}).get("stage") == "ready" and m.get("volume_input") and \
                m.get("anatomy_source") != "estimated":
            return store.get(m["id"])
    return None


CASE = _case()
pytestmark = pytest.mark.skipif(CASE is None, reason="no processed CT/CBCT case available")


@pytest.fixture(scope="module")
def engine():
    from mentoplan.tissue.predict import PredictionEngine
    return PredictionEngine(CASE)


def _pog(eng, sol, t=365.0, calibrate=True):
    r = eng.predict(sol, t, calibrate=calibrate, vertices=False)
    L = r["landmarks"]["soft_pogonion"]
    return np.array(L["median"]) - np.array(L["pre"]), r


def test_anatomy_plausible():
    L = CASE.anatomy.landmarks
    for k in ("soft_pogonion", "pogonion", "menton", "stomion", "subnasale", "mental_foramen_right",
              "mental_foramen_left"):
        assert k in L
    th = np.linalg.norm(L["soft_pogonion"] - L["pogonion"])
    assert 3.0 < th < 25.0                                        # chin pad thickness
    assert L["soft_pogonion"][1] > L["pogonion"][1]               # skin in front of bone
    assert L["menton"][2] < L["pogonion"][2] < L["b_point"][2]    # vertical order on the symphysis
    assert L["mental_foramen_right"][0] > 10 and L["mental_foramen_left"][0] < -10
    m = CASE.model
    assert m.info["min_jacobian_quality"] > 0.1
    assert len(m.sets["pocket"]) > 200


def test_zero_implant_no_change(engine):
    sol = engine.solve_design(ImplantDesign(projection=0.0, vertical_extension=0.0, inferior_wrap=0.0))
    assert np.abs(sol.fields["impl"]).max() < 1e-6


def test_projection_monotonic_and_ratio(engine):
    adv = []
    for P in (3.0, 6.0, 9.0):
        sol = engine.solve_design(ImplantDesign(projection=P))
        d, r = _pog(engine, sol, calibrate=False)
        adv.append(d[1])
        ratio = d[1] / sol.t_pog
        assert 0.3 < ratio < 1.1                                  # physically plausible soft:hard ratio
    assert adv[0] < adv[1] < adv[2]


def test_symmetric_design_symmetric_response(engine):
    sol = engine.solve_design(ImplantDesign(projection=6.0))
    d, _ = _pog(engine, sol, calibrate=False)
    assert abs(d[0]) < 0.25 * abs(d[1])                           # no lateral drift of the midline


def test_asymmetric_design_shifts_chin(engine):
    s0 = engine.solve_design(ImplantDesign(projection=6.0))
    s1 = engine.solve_design(ImplantDesign(projection=6.0, asymmetry=0.4))
    v = CASE.meshes["skin"].vertices[engine.active]
    f0 = np.asarray(engine.W_act @ s0.fields["impl"])
    f1 = np.asarray(engine.W_act @ s1.fields["impl"])
    right = (v[:, 0] > 8) & (v[:, 0] < 25)
    left = (v[:, 0] < -8) & (v[:, 0] > -25)
    gain_r = np.linalg.norm(f1[right], axis=1).mean() - np.linalg.norm(f0[right], axis=1).mean()
    gain_l = np.linalg.norm(f1[left], axis=1).mean() - np.linalg.norm(f0[left], axis=1).mean()
    assert gain_r > 0 > gain_l


def test_uncertainty_and_healing(engine):
    sol = engine.solve_design(ImplantDesign(projection=6.0))
    early, r_early = _pog(engine, sol, 7.0)
    late, r_late = _pog(engine, sol, 365.0)
    assert early[1] > late[1] + 1.0                               # edema at one week
    L = r_late["landmarks"]["soft_pogonion"]
    assert L["p05"][1] < L["median"][1] < L["p95"][1]
    assert L["p05"][1] - L["pre"][1] > 0                          # an implant never retrudes the chin
    assert L["p05_total"][1] <= L["p05"][1] and L["p95_total"][1] >= L["p95"][1]   # posture adds spread
    width = L["p95"][1] - L["p05"][1]
    assert 0.8 < width < 10.0                                     # honest but informative band
    r = engine.predict(sol, 365.0)
    act = engine.active
    assert np.all(r["low"][act] <= 1e-6) and np.all(r["high"][act] >= -1e-6)


def test_material_affects_settled_result(engine):
    a = engine.solve_design(ImplantDesign(projection=6.0, material="silicone"))
    b = engine.solve_design(ImplantDesign(projection=6.0, material="titanium"))
    da, _ = _pog(engine, a)
    db, _ = _pog(engine, b)
    assert db[1] > da[1]                                          # less resorption under titanium


def test_risks(engine):
    from mentoplan.service import CaseRuntime
    rt = CaseRuntime(CASE)
    rt._engine = engine
    big = rt.plan(ImplantDesign(projection=12.0, width=70, vertical_extension=6), [("Year 1", 365)])
    small = rt.plan(ImplantDesign(projection=3.0), [("Year 1", 365)])
    lv = {"low": 0, "moderate": 1, "high": 2}
    def level(res, rid):
        return next(lv[r["level"]] for r in res["summary"]["risks"] if r["id"] == rid)
    assert level(big, "tissue_tension") >= level(small, "tissue_tension")
    assert level(big, "lip_competence") >= level(small, "lip_competence")
    ids = {r["id"] for r in big["summary"]["risks"]}
    assert {"mental_nerve_right", "mental_nerve_left", "tooth_roots", "bone_resorption", "migration"} <= ids


def test_implant_solid_watertight():
    import trimesh
    from mentoplan.implant.design import build_implant
    a = CASE.anatomy
    sol = build_implant(ImplantDesign(projection=6.0), a.arch, CASE.bonefield.sample_mandible, a.landmarks)
    tm = trimesh.Trimesh(sol.mesh.vertices, sol.mesh.faces)
    assert tm.is_watertight
    assert 0.5 < sol.metrics["volume_cc"] < 8
    assert sol.metrics["max_thickness_mm"] == pytest.approx(6.0, abs=0.3)


def test_api_smoke():
    from fastapi.testclient import TestClient
    from mentoplan.api.bundle import unpack
    from mentoplan.api.server import create_app
    client = TestClient(create_app(CaseStore(ROOT)))
    assert client.get("/api/health").json()["ok"]
    cid = CASE.id
    d = client.get(f"/api/cases/{cid}").json()
    assert d["preop_metrics"] and "soft_pogonion" in d["anatomy"]["landmarks"]
    meta, arr = unpack(client.get(f"/api/cases/{cid}/meshes/skin").content)
    assert arr["position"].shape[1] == 3 and arr["index"].size % 3 == 0
    stl = client.post(f"/api/cases/{cid}/implant.stl", json={"design": {"projection": 5}}).content
    n = int.from_bytes(stl[80:84], "little")
    assert len(stl) == 84 + 50 * n and n > 100
    bad = client.post(f"/api/cases/{cid}/plan", json={"design": {"projection": 40}})
    assert bad.status_code == 422
