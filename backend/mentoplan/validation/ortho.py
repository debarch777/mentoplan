"""Retrospective validation of the soft-tissue model on a pre/post-operative scan pair.

Procedure (blind to the post-operative soft tissue):

1. Segment both scans; register the post-op scan to the pre-op scan on the
   cranium (forehead, orbits, skull - unchanged by jaw surgery) with trimmed ICP.
2. Measure the movement of every bony segment by registering the pre-op
   segment surfaces to the post-op bone: maxilla, mandible and - if it moved
   independently - the genioplasty chin segment.
3. Prescribe these movements on the pre-op FE model (bone-attached nodes) and
   solve for the soft tissue; include the material-uncertainty ensemble to
   obtain 90% bands.
4. Compare the predicted face with the actual post-op face (signed distance
   along the skin normal) in the lower face, and with a "no change" baseline.

Only bone information from the post-op scan is used for the prediction; the
post-op skin is used solely for scoring.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..imaging.segment import segment_head
from ..imaging.surface import decimate, mask_to_mesh, skin_surface
from ..io.volume import read_volume
from ..surface_scan import _normals


def rigid_icp(src, dst, T0=None, iters=60, trim=0.8, max_dist=None):
    """Point-to-point trimmed ICP (src -> dst).  Returns (4x4, rms of kept matches)."""
    tree = cKDTree(dst)
    T = np.eye(4) if T0 is None else T0.copy()
    rms = np.inf
    for _ in range(iters):
        p = src @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(p)
        keep = d <= np.quantile(d, trim)
        if max_dist is not None:
            keep &= d < max_dist
        a, b = p[keep], dst[j[keep]]
        ca, cb = a.mean(0), b.mean(0)
        H = (a - ca).T @ (b - cb)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = cb - R @ ca
        T = dT @ T
        new = float(np.sqrt(np.mean(d[keep] ** 2)))
        if abs(rms - new) < 1e-5:
            rms = new
            break
        rms = new
    return T, rms


def _apply(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def run_validation(pre_path: Path, post_path: Path, out: Path, case=None, log=print) -> dict:
    """``case``: an already processed pre-op Case (else one is created in a temp store)."""
    from ..case import CaseStore
    from ..service import CaseRuntime
    from ..tissue.simulate import SENS_PARAMS, Simulator

    t0 = time.time()
    if case is None:
        import tempfile
        store = CaseStore(Path(tempfile.mkdtemp(prefix="mentoplan_val_")))
        case = store.create("validation pre-op", pre_path)
        case.process(log=log)
    a = case.anatomy
    fr = a.frame
    lms = a.landmarks
    log("segmenting post-operative scan")
    post_seg = segment_head(read_volume(post_path), 0.5)
    post_skin_w = skin_surface(post_seg)
    post_bone_w = decimate(mask_to_mesh(post_seg.bone, post_seg.volume, 0.6), 300_000)

    # ---- 1. cranial-base registration (post world -> pre head frame)
    pre_bone = case.meshes["bone"].vertices
    pre_mand = case.meshes["mandible"].vertices
    sn = lms["subnasale"]
    tree_m = cKDTree(pre_mand)
    dm, _ = tree_m.query(pre_bone)
    cranial = pre_bone[(pre_bone[:, 2] > sn[2] + 25) & (dm > 5)]
    pb = fr.to_head(post_bone_w.vertices)
    ps = fr.to_head(post_skin_w.vertices)
    # coarse: align centroids of the upper skull, then refine with trimmed ICP
    post_up = pb[pb[:, 2] > np.percentile(pb[:, 2], 55)]
    T0 = np.eye(4)
    T0[:3, 3] = cranial.mean(0) - post_up.mean(0)
    rng = np.random.default_rng(0)
    sub = post_up[rng.choice(len(post_up), min(40000, len(post_up)), replace=False)]
    T_reg, rms_reg = rigid_icp(sub, cranial, T0, trim=0.7)
    # second pass using all post bone above the occlusal level
    T_reg, rms_reg = rigid_icp(pb[(_apply(T_reg, pb)[:, 2] > sn[2] + 25)], cranial, T_reg, trim=0.75)
    post_bone = _apply(T_reg, pb)
    post_skin = _apply(T_reg, ps)
    log(f"cranial registration rms {rms_reg:.2f} mm")

    # ---- 2. bony segment movements
    tip = lms["lower_incisor_tip"] if "lower_incisor_tip" in lms else lms["stomion"]
    # Le Fort I segment: upper dentition and alveolar process up to ~5 mm above the nasal floor level
    max_pts = pre_bone[(pre_bone[:, 2] > tip[2] - 2) & (pre_bone[:, 2] < sn[2] + 5) &
                       (pre_bone[:, 1] > lms["gonion_right"][1] + 10 if "gonion_right" in lms else True) & (dm > 3)]
    T_max, rms_max = rigid_icp(max_pts, post_bone, trim=0.7, max_dist=8)
    T_mand, rms_mand = rigid_icp(pre_mand, post_bone, trim=0.7, max_dist=10)
    apex = lms["anterior_root_apex"][2] if "anterior_root_apex" in lms else lms["menton"][2] + 20
    chin = pre_mand[(pre_mand[:, 2] < apex - 2) & (pre_mand[:, 1] > lms["menton"][1] - 25)]
    T_chin, rms_chin = rigid_icp(chin, post_bone, T_mand, trim=0.7, max_dist=8)
    d_chin = np.linalg.norm(_apply(T_chin, chin) - _apply(T_mand, chin), axis=1).mean()
    genio = d_chin > 1.5
    bone_lm = {k: (_apply(T_chin if k in ("pogonion", "menton", "gnathion") else T_mand, lms[k][None])[0] - lms[k]).tolist()
               for k in ("pogonion", "b_point", "menton", "gnathion", "lower_incisor_tip") if k in lms}
    moves = {"bony_landmark_displacement_mm": bone_lm,
        "maxilla": _movement(T_max, max_pts, rms_max),
        "mandible": _movement(T_mand, pre_mand, rms_mand),
        "chin_segment": {**_movement(T_chin, chin, rms_chin), "independent_genioplasty": bool(genio),
                         "difference_from_mandible_mm": float(d_chin)},
    }
    log(json.dumps(moves, indent=1))

    # ---- 3. FE prediction driven by the bone movements only
    rt = CaseRuntime(case)
    model = case.model
    sim = Simulator(model)
    ls = sim.system("tied")
    nodes = model.nodes
    fixed_nodes = np.unique(ls.fixed // 3)
    disp = np.zeros_like(nodes)
    # which bone does each fixed interface node rest on?
    mand_set = set(model.sets["mandible_iface"].tolist())
    is_mand = np.array([n in mand_set for n in fixed_nodes])
    boundary = np.isin(fixed_nodes, model.sets["boundary_box"])
    p = nodes[fixed_nodes]
    in_chin = is_mand & (p[:, 2] < apex - 2) & (p[:, 1] > lms["menton"][1] - 25)
    Tm = T_chin if genio else T_mand
    d_m = np.where(in_chin[:, None], _apply(Tm, p) - p, _apply(T_mand, p) - p)
    near_max = (~is_mand) & (p[:, 2] > tip[2] - 4) & (p[:, 2] < sn[2] + 15) & (p[:, 1] > lms["menton"][1] - 40)
    d_x = _apply(T_max, p) - p
    disp_f = np.where(is_mand[:, None], d_m, np.where(near_max[:, None], d_x, 0.0))
    disp_f[boundary] = 0.0
    disp[fixed_nodes] = disp_f
    # lip mucosa resting on the teeth follows the teeth in the normal direction (slides tangentially)
    ts = model.sets["teeth_slide"]
    tn = model.normals["teeth_slide"][: len(ts)]
    pt = nodes[ts]
    on_mand = pt[:, 2] < lms["stomion"][2]
    d_t = np.where(on_mand[:, None], _apply(T_mand, pt) - pt, _apply(T_max, pt) - pt)
    gap = np.einsum("ij,ij->i", d_t, tn)
    f_slide = np.zeros(sim.ndof)
    keep = ~np.isin(ts, fixed_nodes)
    for c in range(3):
        np.add.at(f_slide, 3 * ts[keep] + c, ls.k_pen * tn[keep, c] * gap[keep])
    u = _solve_prescribed(sim, ls, disp, f_slide)
    sens = sim.sensitivities(u, "tied")

    skin = case.meshes["skin"]
    V = skin.vertices
    N = skin.vertex_normals()
    W = model.interpolation_matrix(V, surface=True).tocsr()
    act = np.where(np.diff(W.indptr) > 0)[0]
    pred = V.copy()
    pred[act] += W[act] @ u
    # ensemble along the normal
    S = 300
    p_ = sim.params
    th = np.stack([rng.normal(0, sd, S) for sd in (p_.sd_log_fat, p_.sd_log_muscle, p_.sd_log_dermis, p_.sd_log_bulk)], 1)
    Fs = np.stack([np.einsum("nj,nj->n", np.asarray(W[act] @ sens[k]), N[act]) for k in SENS_PARAMS], 0)
    base_n = np.einsum("nj,nj->n", np.asarray(W[act] @ u), N[act])
    from ..tissue.predict import ClinicalPriors
    pr = ClinicalPriors()
    sig = np.hypot(pr.discrepancy_const_mm * np.clip(np.abs(base_n), 0, 1) + pr.discrepancy_rel * np.abs(base_n),
                   pr.segmentation_sd_mm)
    samples = base_n[None] + th @ Fs + rng.normal(0, 1, (S, 1)) * sig[None]
    lo = np.percentile(samples, 5, 0) - base_n
    hi = np.percentile(samples, 95, 0) - base_n

    # ---- 4. scoring in the lower face (below subnasale, frontal, inside the FE ROI)
    pg = lms["soft_pogonion"]
    region = act[(V[act, 2] < sn[2] - 3) & (V[act, 2] > lms["soft_menton"][2] - 6) &
                 (N[act, 1] > 0.3) & (np.abs(V[act, 0]) < 45)]
    ridx = np.searchsorted(act, region)
    post_tree = cKDTree(post_skin)
    post_n = _normals(post_skin[rng.choice(len(post_skin), min(len(post_skin), 200000), replace=False)])

    def signed_dist(points, normals):
        d, j = post_tree.query(points)
        # sign: positive if the actual surface lies outside (along +normal) of the points
        return np.einsum("ij,ij->i", post_skin[j] - points, normals), d

    # whole-face offset between the two scans measured where surgery cannot act (forehead,
    # nasal dorsum, upper cheeks): scanner calibration / weight change, reported and removed
    nas = lms["soft_nasion"]
    ref = act_all = np.where((V[:, 2] > sn[2] + 8) & (N[:, 1] > 0.3) & (np.abs(V[:, 0]) < 55))[0]
    e_ref, d_ref = signed_dist(V[ref], N[ref])
    offset = float(np.median(e_ref[d_ref < 6])) if (d_ref < 6).any() else 0.0
    err_pred, dpred = signed_dist(pred[region], N[region])
    err_none, dnone = signed_dist(V[region], N[region])
    err_pred_raw, err_none_raw = err_pred.copy(), err_none.copy()
    err_pred = err_pred - offset
    err_none = err_none - offset
    valid = (dpred < 8) & (dnone < 12)          # ignore vertices without a post-op counterpart (FOV)
    e_p, e_0 = err_pred[valid], err_none[valid]
    inside = (e_p >= lo[ridx][valid] - 0.0) & (e_p <= hi[ridx][valid])
    changed = np.abs(e_0) > 1.0

    def stats(e):
        a_ = np.abs(e)
        return {"mean_abs_mm": float(a_.mean()), "rms_mm": float(np.sqrt(np.mean(e ** 2))),
                "median_abs_mm": float(np.median(a_)), "p90_abs_mm": float(np.percentile(a_, 90)),
                "within_1mm": float((a_ < 1).mean()), "within_2mm": float((a_ < 2).mean())}

    lm_err = {}
    for k in ("soft_pogonion", "labrale_inferius", "soft_b_point", "soft_menton", "labrale_superius", "subnasale"):
        if k in lms:
            q = lms[k]
            Wq = model.interpolation_matrix(q[None], surface=True)
            pq = q + (Wq @ u)[0]
            nq = N[np.argmin(np.linalg.norm(V - q, axis=1))]
            e1, _ = signed_dist(pq[None], nq[None])
            e0, _ = signed_dist(q[None], nq[None])
            lm_err[k] = {"actual_change_mm": float(e0[0]), "predicted_change_mm": float(e0[0] - e1[0]),
                         "error_mm": float(e1[0])}
    report = {
        "registration_rms_mm": rms_reg, "bone_movements": moves,
        "whole_face_scan_offset_mm": offset,
        "raw_without_offset_correction": {"prediction": stats(err_pred_raw[valid]),
                                          "no_change_baseline": stats(err_none_raw[valid])},
        "n_vertices_scored": int(valid.sum()),
        "prediction": stats(e_p), "no_change_baseline": stats(e_0),
        "changed_region": {"n": int(changed.sum()), "prediction": stats(e_p[changed]),
                           "no_change_baseline": stats(e_0[changed])},
        "interval_coverage_90": float(inside.mean()),
        "landmarks": lm_err, "seconds": round(time.time() - t0, 1),
    }
    Path(out).write_text(json.dumps(report, indent=2))
    np.savez_compressed(Path(out).with_suffix(".npz"), region=region, err_pred=err_pred, err_none=err_none,
                        valid=valid, pred=pred.astype(np.float32), post_skin=post_skin.astype(np.float32))
    log(json.dumps({k: report[k] for k in ("prediction", "no_change_baseline", "interval_coverage_90")}, indent=1))
    return report


def _movement(T, pts, rms):
    c = pts.mean(0)
    R = T[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return {"translation_at_centroid_mm": (_apply(T, c[None])[0] - c).tolist(), "rotation_deg": ang,
            "fit_rms_mm": float(rms)}


def _solve_prescribed(sim, ls, disp, f=None):
    u = np.zeros(sim.ndof)
    uf = disp.ravel()
    u[ls.fixed] = uf[ls.fixed]
    rhs = -(ls.K[ls.free][:, ls.fixed] @ u[ls.fixed])
    if f is not None:
        rhs = rhs + f[ls.free]
    u[ls.free] = ls.solver.solve(rhs)
    return u.reshape(-1, 3)
