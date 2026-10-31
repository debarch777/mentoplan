"""Optical / photogrammetric face scans.

The scan is rigidly registered to the CT/CBCT skin surface and its colour
(texture or vertex colours) is transferred onto the CT-derived skin mesh,
which remains the geometric reference for the finite-element model.  The
registration residual is reported; a large residual usually means the scan
was taken with a different expression or head posture.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .imaging.surface import Mesh


def _normals(points: np.ndarray, k: int = 12) -> np.ndarray:
    tree = cKDTree(points)
    _, idx = tree.query(points, k)
    nb = points[idx] - points[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    w, v = np.linalg.eigh(cov)
    return v[:, :, 0]


def icp_point_to_plane(src: np.ndarray, dst: np.ndarray, dst_n: np.ndarray, T0: np.ndarray, iters: int = 40,
                       trim: float = 0.85):
    """Rigid ICP (src -> dst).  Returns (4x4, trimmed rms)."""
    tree = cKDTree(dst)
    T = T0.copy()
    rms = np.inf
    for _ in range(iters):
        p = src @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(p)
        keep = d <= np.quantile(d, trim)
        p, q, n = p[keep], dst[j[keep]], dst_n[j[keep]]
        r = np.einsum("ij,ij->i", p - q, n)
        A = np.c_[np.cross(p, n), n]
        x, *_ = np.linalg.lstsq(A, -r, rcond=None)
        a, b, c = x[:3]
        R = np.array([[1, -c, b], [c, 1, -a], [-b, a, 1]])
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = x[3:]
        T = dT @ T
        new = float(np.sqrt(np.mean(r ** 2)))
        if abs(rms - new) < 1e-4:
            rms = new
            break
        rms = new
    return T, rms


def register_scan(scan_pts: np.ndarray, skin_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Automatic rigid registration of a face scan onto the CT skin (both point clouds, mm).

    Scale is fixed by detecting metres vs millimetres.  Initialisation: PCA
    axes of the scan aligned to the PCA axes of the frontal CT skin, trying the
    four proper sign combinations, each refined with trimmed point-to-plane ICP."""
    rng = np.random.default_rng(0)
    s = scan_pts
    if np.ptp(s, axis=0).max() < 2.0:      # scan in metres
        s = s * 1000.0
    src = s[rng.choice(len(s), min(len(s), 20000), replace=False)]
    # frontal part of the CT skin (scans only see the face)
    sk = skin_pts[skin_pts[:, 1] > np.percentile(skin_pts[:, 1], 45)]
    dst = sk[rng.choice(len(sk), min(len(sk), 40000), replace=False)]
    dst_n = _normals(dst)
    cs, cd = src.mean(0), dst.mean(0)
    _, _, Vs = np.linalg.svd(src - cs, full_matrices=False)
    _, _, Vd = np.linalg.svd(dst - cd, full_matrices=False)
    best = (None, np.inf)
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        D = np.diag([sx, sy, sx * sy])
        R = Vd.T @ D @ Vs
        if np.linalg.det(R) < 0:
            continue
        T0 = np.eye(4)
        T0[:3, :3] = R
        T0[:3, 3] = cd - R @ cs
        T, rms = icp_point_to_plane(src, dst, dst_n, T0)
        if rms < best[1]:
            best = (T, rms)
    T = best[0]
    if np.ptp(scan_pts, axis=0).max() < 2.0:
        T = T @ np.diag([1000.0, 1000.0, 1000.0, 1.0])
    return T, best[1]


def transfer_colors(skin: Mesh, scan: Mesh, T: np.ndarray, uv=None, texture=None, colors=None,
                    max_dist: float = 2.5):
    """Per-vertex RGB (0..1) for the CT skin from the registered scan, plus a 0..1 coverage mask."""
    sv = scan.vertices @ T[:3, :3].T + T[:3, 3]
    tree = cKDTree(sv)
    d, j = tree.query(skin.vertices)
    rgb = np.zeros((skin.n_vertices, 3), np.float32)
    if texture is not None and uv is not None:
        h, w = texture.shape[:2]
        u = np.clip(uv[j, 0], 0, 1) * (w - 1)
        v = (1 - np.clip(uv[j, 1], 0, 1)) * (h - 1)
        rgb = texture[v.astype(int), u.astype(int)].astype(np.float32) / 255.0
    elif colors is not None:
        rgb = colors[j].astype(np.float32) / 255.0
    mask = np.clip(1.0 - (d - 0.5 * max_dist) / (0.5 * max_dist), 0, 1).astype(np.float32)
    # sRGB -> linear for physically based shading
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)
    return lin, mask, float(np.median(d[mask > 0.5])) if (mask > 0.5).any() else float("nan")


def attach_surface_scan(case, log=print):
    from .io.meshio import read_surface
    from .case import save_mesh_dict
    path = case.root / "input" / case.meta["surface_input"]
    scan, uv, tex, cols = read_surface(path)
    skin = case.meshes["skin"]
    T, rms = register_scan(scan.vertices, skin.vertices)
    rgb, mask, med = transfer_colors(skin, scan, T, uv, tex, cols)
    skin.attributes["albedo_r"] = rgb[:, 0]
    skin.attributes["albedo_g"] = rgb[:, 1]
    skin.attributes["albedo_b"] = rgb[:, 2]
    skin.attributes["albedo_mask"] = mask
    np.savez_compressed(case.cache / "meshes.npz", **save_mesh_dict(case.meshes))
    case.meta["surface_registration"] = {"rms_mm": rms, "median_distance_mm": med,
                                         "coverage": float((mask > 0.5).mean())}
    case.save_meta()
    log(f"surface scan registered: rms {rms:.2f} mm, coverage {(mask > 0.5).mean() * 100:.0f}% of skin vertices")
    return case.meta["surface_registration"]
