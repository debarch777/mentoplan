"""Physics-based facial expressions.

A second, coarser finite-element model of the lower two thirds of the face
(cheeks, lips, chin, submental region) is driven by

* active muscle contraction, modelled Hill-style as an active fibre tension
  sigma_a = T f (x) f added to the passive stress inside a tube around each
  muscle's line of action (origin on bone -> insertion in the lips /
  modiolus / chin skin), and
* jaw opening, a rigid rotation of the mandible about the condylar axis
  prescribed on the mandible-attached tissue nodes.

The *shape* of every expression comes from the patient's own anatomy and the
mechanics; the overall magnitude of each expression is normalised to typical
values measured in 3-D facial motion studies (e.g. commissure excursion of
~9 mm in a social smile), because activation levels cannot be read from a
static scan.  Fields are linear and are superimposed on the pre- and
post-operative surfaces; with an implant in place the chin pad is pushed
forward but moves with the same muscle pattern (the mentalis is re-suspended).
"""
from __future__ import annotations

import numpy as np

from .model import MAT_FAT, ModelOptions, build_tissue_model
from ..imaging.surface import lip_side_labels, smooth_vertex_field
from .hexfem import stress_load
from .simulate import Simulator, TissueParams

# extends well below the chin: the fixed ROI floor must not hold the submental skin when the jaw opens
EXPR_OPTIONS = ModelOptions(h=3.0, roi_half_width=72.0, roi_posterior=95.0, roi_below_menton=48.0,
                            roi_above_subnasale=38.0, pocket_half_arc=0.1, pocket_top=0.0, pocket_under=0.0)


def face_model(rt):
    """Coarse whole-lower-face model, cached on disk next to the case."""
    from ..case import load_model, save_model
    case = rt.case
    path = case.cache / "face_model.npz"
    if getattr(rt, "_face_model", None) is not None:
        return rt._face_model
    if path.exists():
        m = load_model(path)
    else:
        from ..case import BoneField
        a = case.anatomy
        seg = case.seg
        bf = BoneField(seg, a.frame)          # short-lived: released right after the model is built
        m = build_tissue_model(seg, a.frame, a.landmarks, a.arch, bf.normals, EXPR_OPTIONS, surface_fn=bf.surface)
        save_model(path, m)
        del bf, seg
        case._seg = None
    rt._face_model = m
    return m


def _tube(centers, p0, p1, radius):
    """Weight (0..1) of element centres inside a tube around segment p0-p1, and the fibre direction."""
    d = p1 - p0
    L = np.linalg.norm(d)
    f = d / max(L, 1e-9)
    t = np.clip((centers - p0) @ f / L, 0, 1)
    closest = p0 + t[:, None] * d
    r = np.linalg.norm(centers - closest, axis=1)
    w = np.clip(1.0 - (r / radius) ** 2, 0, 1) * np.clip(np.minimum(t, 1 - t) / 0.08, 0, 1)
    return w, np.broadcast_to(f, centers.shape)


ACTIVE_STRESS_KPA = 60.0   # fibre tension at activation 1.0 (facial muscle, order of magnitude)


def _eig(f, a):
    """Active fibre stress a * T (f x f), (E,3,3); a = activation in [0, 1]."""
    return ACTIVE_STRESS_KPA * a[:, None, None] * np.einsum("ei,ej->eij", f, f)


def condyle_axis(rt, ramus_height: float = 56.0):
    """Approximate condylar hinge axis.

    The separated mandible may lack the condylar heads (they sit in the
    fossa and are often merged with the temporal bone in a threshold mask), so
    the ramus is followed from gonion along its principal direction to a
    typical adult gonion-condylion height (Co-Go ~ 52-60 mm)."""
    mand = rt.case.meshes["mandible"].vertices
    lms = rt.case.anatomy.landmarks
    pts = []
    for sgn, key in ((1, "gonion_right"), (-1, "gonion_left")):
        go = lms[key] if key in lms else None
        side = mand[sgn * mand[:, 0] > 20]
        if go is None:
            go = side[np.argmin(side[:, 1] + side[:, 2])]
        ramus = side[(side[:, 1] < go[1] + 15) & (side[:, 2] > go[2] + 15)]
        d = np.array([0.0, -0.15, 1.0])
        if len(ramus) > 50:
            c0 = ramus.mean(0)
            _, _, Vt = np.linalg.svd(ramus - c0, full_matrices=False)
            cand = Vt[0] if Vt[0][2] > 0 else -Vt[0]
            # accept only anatomically plausible ramus inclinations (< 25 deg from vertical)
            if cand[2] / np.linalg.norm(cand) > np.cos(np.radians(25)):
                d = cand
        d = d / np.linalg.norm(d)
        top = side[np.argmax(side[:, 2])]
        cond = top if top[2] - go[2] >= ramus_height - 5 else go + ramus_height * d
        pts.append(cond)
    c = 0.5 * (pts[0] + pts[1])
    ax = pts[0] - pts[1]
    return c, ax / np.linalg.norm(ax), pts


def muscles(lms, hw):
    """Lines of action (head frame) from landmarks; returns {name: [(p0, p1, radius), ...]}"""
    g = lambda k: np.asarray(lms[k], float)
    sto, sn, li, ls, pg, me = g("stomion"), g("subnasale"), g("labrale_inferius"), g("labrale_superius"), \
        g("soft_pogonion"), g("menton")
    b = g("b_point")
    out = {}
    for s, side in ((1, "R"), (-1, "L")):
        mod = np.array([s * (hw + 1.5), sto[1] - 7, sto[2]])
        out.setdefault("zygomaticus", []).append((np.array([s * (hw + 30), sto[1] - 32, sn[2] + 24]), mod, 5.5))
        out.setdefault("risorius", []).append((np.array([s * (hw + 38), sto[1] - 38, sto[2] - 2]), mod, 4.0))
        out.setdefault("levator", []).append((np.array([s * 19, sn[1] - 10, sn[2] + 20]),
                                              np.array([s * 10, ls[1] - 4, ls[2] + 1]), 4.5))
        out.setdefault("dao", []).append((np.array([s * (hw + 6), me[1] - 22, me[2] + 6]), mod, 5.0))
        out.setdefault("dli", []).append((np.array([s * 14, me[1] - 6, me[2] + 9]),
                                          np.array([s * 9, li[1] - 4, li[2] - 2]), 4.5))
        out.setdefault("mentalis", []).append((np.array([s * 5, b[1] - 2, b[2] - 3]),
                                               np.array([s * 7, pg[1] - 3, pg[2] - 7]), 4.5))
    return out


EXPRESSIONS = {
    "smile": {"label": "Smile (lips closed)", "mus": {"zygomaticus": 0.3, "risorius": 0.1}, "jaw": 0.0,
              "norm": ("commissure", 9.0)},
    "broad_smile": {"label": "Broad smile", "mus": {"zygomaticus": 0.45, "levator": 0.3, "risorius": 0.15}, "jaw": 7.0,
                    "norm": ("commissure", 12.0)},
    "mouth_open": {"label": "Mouth open (speech)", "mus": {"dli": 0.1}, "jaw": 16.0, "norm": None},
    "pucker": {"label": "Pucker / pout", "mus": {"orbicularis": 0.35, "mentalis": 0.1}, "jaw": 0.0,
               "norm": ("lips_forward", 3.5)},
    "chin_raise": {"label": "Chin raise (mentalis)", "mus": {"mentalis": 0.4, "orbicularis": 0.05}, "jaw": 0.0,
                   "norm": ("lower_lip_up", 4.0)},
    "frown": {"label": "Frown / grimace", "mus": {"dao": 0.35, "dli": 0.25}, "jaw": 2.0, "norm": ("commissure", 4.5)},
}


def expression_fields(rt) -> dict:
    if getattr(rt, "_expr", None) is not None:
        return rt._expr
    case = rt.case
    lms = case.anatomy.landmarks
    m = face_model(rt)
    sim = Simulator(m, TissueParams())
    hw = (case.model.info.get("fissure") or {}).get("half_width_mm", 22.0) if isinstance(
        case.model.info.get("fissure"), dict) else 22.0
    centers = m.elem_centers()
    soft = np.ones(len(m.elems), bool)
    lines = muscles(lms, hw)
    sto = lms["stomion"]
    # orbicularis oris: circumferential fibres in an elliptical ring around the mouth
    q = centers - sto
    ell = (q[:, 0] / (hw + 4)) ** 2 + (q[:, 2] / 11.0) ** 2
    ring = np.clip(1 - np.abs(ell - 0.65) / 0.45, 0, 1) * (centers[:, 1] > sto[1] - 14)
    tang = np.stack([-q[:, 2] / 11.0 ** 2, np.zeros(len(q)), q[:, 0] / (hw + 4) ** 2], 1)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)

    def muscle_eig(name, a):
        E = np.zeros((len(m.elems), 3, 3))
        if name == "orbicularis":
            return _eig(tang, a * ring)
        for p0, p1, r in lines[name]:
            w, f = _tube(centers, p0, p1, r)
            E += _eig(f, a * w * soft)
        return E

    skin = case.meshes["skin"]
    W = m.interpolation_matrix(skin.vertices, surface=True).tocsr()
    active = np.where(np.diff(W.indptr) > 0)[0]
    Wa = W[active]
    probe_names = {"commissure": [np.array([hw + 1.5, sto[1] - 3, sto[2]]), np.array([-(hw + 1.5), sto[1] - 3, sto[2]])],
                   "lips_forward": [lms["labrale_superius"], lms["labrale_inferius"]],
                   "lower_lip_up": [lms["labrale_inferius"]]}
    from scipy.spatial import cKDTree
    tree = cKDTree(skin.vertices)
    probe_vertex = {k: tree.query(np.array(v))[1] for k, v in probe_names.items()}
    c_ax, ax, condyles = condyle_axis(rt)
    side = fissure_sides(case, skin.vertices)
    mand_nodes = m.sets["mandible_iface"]
    fields = {}
    meta = {}
    for key, ex in EXPRESSIONS.items():
        E = np.zeros((len(m.elems), 3, 3))
        for mus, a in ex["mus"].items():
            E += muscle_eig(mus, np.full(len(m.elems), a))
        f = stress_load(m.geom, m.asm.dofs, sim.ndof, E)
        u_m = sim.solve(None, f_extra=f, interface="tied")
        u = u_m
        if ex["norm"]:
            kind, target = ex["norm"]
            # measure on the actual skin surface: nearest render vertices to the probe points
            idx = probe_vertex[kind]
            d = np.asarray(W[idx] @ u_m)
            mag = float(np.linalg.norm(d, axis=1).mean()) if kind != "lips_forward" else float(d[:, 1].mean())
            if mag > 1e-6:
                u = u_m * (target / mag)
        if ex["jaw"]:
            u = u + _jaw_solution(sim, m, mand_nodes, c_ax, ax, ex["jaw"])
        fields[key] = smooth_vertex_field(skin, np.asarray(Wa @ u, np.float32), active, 12, 0.5, side)
        meta[key] = {"label": ex["label"]}
        if ex["jaw"]:
            meta[key].update({"jaw_deg": -ex["jaw"], "axis_point": c_ax.tolist(), "axis_dir": ax.tolist()})
    rt._expr = {"active": active, "fields": fields, "meta": meta,
                "condyles": [c.tolist() for c in condyles]}
    return rt._expr


def _jaw_solution(sim: Simulator, m, mand_nodes, c, ax, deg):
    """Tissue response to opening the jaw by ``deg`` degrees about the condylar axis (opening = chin down/back)."""
    th = np.radians(-deg)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    p = m.nodes[mand_nodes]
    disp = (p - c) @ R.T + c - p
    ls = sim.system("tied")
    u = np.zeros(sim.ndof)
    for k in range(3):
        u[3 * mand_nodes + k] = disp[:, k]
    fixed_vals = u[ls.fixed]
    rhs = -(ls.K[ls.free][:, ls.fixed] @ fixed_vals)
    u[ls.free] = ls.solver.solve(rhs)
    return u.reshape(-1, 3)


def fissure_sides(case, vertices):
    lms = case.anatomy.landmarks
    fis = case.model.info.get("fissure")
    if not isinstance(fis, dict):
        return None
    y_teeth = lms["lower_incisor_tip"][1] if "lower_incisor_tip" in lms else lms["stomion"][1] - 15
    return lip_side_labels(vertices, lms["stomion"], fis["z"], fis["half_width_mm"], y_teeth)
