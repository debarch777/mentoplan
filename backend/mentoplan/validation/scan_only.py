"""Validation of face-scan-only planning against the CT of the same person.

A face scan is simulated from a processed CT case (camera-visible frontal skin, random pose,
0.15 mm noise, synthetic colour), planned as a scan-only case using a *different* processed case
as anatomical template, and compared with the CT-based anatomy and prediction of the same person.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np


def run(store, subject_id: str, template_id: str, out: Path, seed: int = 3, log=print) -> dict:
    import trimesh
    from scipy.spatial.transform import Rotation

    from ..imaging.surface import submesh
    from ..implant.design import ImplantDesign
    from ..tissue.predict import PredictionEngine

    subj = store.get(subject_id)
    tmpl = store.get(template_id)
    skin = subj.meshes["skin"]
    L = subj.anatomy.landmarks
    fn = skin.face_normals()
    fc = skin.vertices[skin.faces].mean(1)
    vis = (fn[:, 1] > 0.05) & (np.abs(fc[:, 0]) < 75) & (fc[:, 2] > L["soft_menton"][2] - 25)
    face = submesh(skin, vis)
    rng = np.random.default_rng(seed)
    R = Rotation.from_rotvec(rng.normal(0, 0.15, 3)).as_matrix()
    t = rng.normal(0, 20, 3)
    V = face.vertices @ R.T + t + rng.normal(0, 0.15, face.vertices.shape)
    cols = np.clip(np.c_[200 + 10 * np.sin(V[:, 0] / 7), 160 + 8 * np.cos(V[:, 2] / 5), np.full(len(V), 140)], 0, 255)
    tm = trimesh.Trimesh(V, face.faces, vertex_colors=np.c_[cols, np.full(len(V), 255)].astype(np.uint8),
                         process=False)
    d = Path(tempfile.mkdtemp(prefix="mentoplan_scanval_"))
    tm.export(d / "face_scan.ply")
    case = store.create(f"{subj.meta['name']} - face scan only (validation)", None, sex=subj.meta.get("sex") or "M",
                        surface_path=d / "face_scan.ply")
    case.meta["scan_up"] = (R @ np.array([0, 0, 1.0])).tolist()
    case.save_meta()
    case.process(template=tmpl, log=log)
    Le = case.anatomy.landmarks
    keys = ["subnasale", "labrale_superius", "stomion", "labrale_inferius", "soft_b_point", "soft_pogonion",
            "soft_menton"]
    P = np.array([Le[k] for k in keys])
    Q = np.array([L[k] for k in keys])
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rr = Vt.T @ D @ U.T
    tr = lambda x: (x - cp) @ Rr.T + cq
    res = {"subject_case": subject_id, "template_case": template_id, "scan_case": case.id,
           "soft_landmark_fit_rms_mm": float(np.sqrt(np.mean(np.sum((tr(P) - Q) ** 2, 1))))}
    for k in ("pogonion", "b_point", "menton", "mental_foramen_right", "mental_foramen_left", "anterior_root_apex"):
        res[f"error_{k}_mm"] = float(np.linalg.norm(tr(Le[k]) - L[k]))
    res["soft_thickness_pog_true_mm"] = float(np.linalg.norm(L["soft_pogonion"] - L["pogonion"]))
    res["soft_thickness_pog_est_mm"] = float(np.linalg.norm(Le["soft_pogonion"] - Le["pogonion"]))
    pred = {}
    for name, c in (("ct", subj), ("scan_only", case)):
        eng = PredictionEngine(c)
        sol = eng.solve_design(ImplantDesign(projection=6.0))
        r = eng.predict(sol, 365.0, vertices=False)
        Lp = r["landmarks"]["soft_pogonion"]
        pre = np.array(Lp["pre"])
        pred[name] = {"soft_pogonion_advance_mm": Lp["median"][1] - pre[1], "p05": Lp["p05"][1] - pre[1],
                      "p95": Lp["p95"][1] - pre[1], "ratio": r["calibration"].get("ratio_posterior")}
    res["prediction_6mm_implant"] = pred
    Path(out).write_text(json.dumps(res, indent=2))
    log(json.dumps(res, indent=2))
    return res
