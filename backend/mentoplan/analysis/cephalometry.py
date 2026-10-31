"""Soft-tissue profile and facial-balance measurements.

All measurements are taken in the head frame (RAS, mm): +y anterior, +z
superior, the x = 0 plane is the mid-sagittal plane.  Norms are the
commonly used clinical references; ``depends_on_head_pose`` marks values that
change with the head pitch (true-vertical / Frankfort based).
"""
from __future__ import annotations

import numpy as np

# name: (label, unit, norm_lo, norm_hi, depends_on_head_pose, reference)
NORMS = {
    "facial_convexity": ("Facial convexity G'-Sn-Pog'", "deg", 8.0, 16.0, False,
                         "Legan & Burstone 1980: 12 +/- 4; negative = concave profile"),
    "lower_lip_e_line": ("Lower lip to E-line (Prn-Pog')", "mm", -4.0, 0.0, False, "Ricketts: -2 +/- 2"),
    "upper_lip_e_line": ("Upper lip to E-line", "mm", -6.0, -2.0, False, "Ricketts: -4 +/- 2"),
    "mentolabial_angle": ("Mentolabial angle Li-B'-Pog'", "deg", 110.0, 135.0, False, "120-130 typical"),
    "labiomental_depth": ("Labiomental fold depth", "mm", 3.0, 7.0, False, "4-6 mm typical"),
    "chin_to_zero_meridian": ("Pog' to zero meridian (N' vertical)", "mm", -3.0, 1.0, True,
                              "Gonzalez-Ulloa: Pog' on or slightly behind"),
    "z_angle": ("Merrifield Z-angle", "deg", 75.0, 85.0, True, "Merrifield 1966: 80 +/- 5"),
    "lower_third_ratio": ("Sn-Sto : Sto-Me' ratio", "", 0.42, 0.58, False, "1 : 2 (0.5)"),
    "lower_face_height": ("Lower face height Sn-Me'", "mm", None, None, False, ""),
    "chin_height": ("Chin height Sto-Me'", "mm", None, None, False, ""),
    "chin_projection_sn_vertical": ("Pog' to subnasale vertical", "mm", -5.0, 0.0, True,
                                    "Arnett TVL: about -3 (female) / -2 (male)"),
    "chin_width": ("Chin prominence width", "mm", None, None, False, "relative to mouth / alar width"),
    "chin_midline_deviation": ("Chin midline deviation", "mm", -1.5, 1.5, False, "< 2 mm usually imperceptible"),
}


def _angle(a, b, c):
    """Angle ABC in degrees (sagittal projection y-z)."""
    u = np.asarray(a)[1:] - np.asarray(b)[1:]
    v = np.asarray(c)[1:] - np.asarray(b)[1:]
    cs = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)
    return float(np.degrees(np.arccos(np.clip(cs, -1, 1))))


def _line_dist(p, a, b):
    """Signed distance of p from line a->b in the sagittal plane; + = anterior of the line."""
    p, a, b = (np.asarray(x)[1:] for x in (p, a, b))
    t = b - a
    n = np.array([t[1], -t[0]]) / max(np.linalg.norm(t), 1e-9)
    if n[0] < 0:
        n = -n
    return float(np.dot(p - a, n))


def profile_metrics(L: dict) -> dict:
    """L: landmark name -> position (3,).  Missing landmarks yield missing metrics."""
    out = {}
    g = lambda k: np.asarray(L[k]) if k in L else None
    G, N, Prn, Sn, Ls, Sto, Li, Bs, Pg, Me = (g(k) for k in ("soft_glabella", "soft_nasion", "pronasale",
                                                             "subnasale", "labrale_superius", "stomion",
                                                             "labrale_inferius", "soft_b_point", "soft_pogonion",
                                                             "soft_menton"))
    if G is not None and Sn is not None and Pg is not None:
        # signed: + convex (chin behind the glabella-subnasale line), - concave (prognathic)
        mag = 180.0 - _angle(G, Sn, Pg)
        out["facial_convexity"] = mag if _line_dist(Pg, G, Sn) < 0 else -mag
    if Prn is not None and Pg is not None:
        if Li is not None:
            out["lower_lip_e_line"] = _line_dist(Li, Prn, Pg)
        if Ls is not None:
            out["upper_lip_e_line"] = _line_dist(Ls, Prn, Pg)
    if Li is not None and Bs is not None and Pg is not None:
        out["mentolabial_angle"] = _angle(Li, Bs, Pg)
        out["labiomental_depth"] = -_line_dist(Bs, Li, Pg)
    if N is not None and Pg is not None:
        out["chin_to_zero_meridian"] = float(Pg[1] - N[1])
    if Sn is not None and Pg is not None:
        out["chin_projection_sn_vertical"] = float(Pg[1] - Sn[1])
    if Pg is not None and (Li is not None or Ls is not None):
        lip = Li if Ls is None or (Li is not None and Li[1] > Ls[1]) else Ls
        d = lip[1:] - Pg[1:]
        out["z_angle"] = float(np.degrees(np.arctan2(d[1], d[0])))
    if Sn is not None and Sto is not None and Me is not None:
        a = abs(Sn[2] - Sto[2])
        b = abs(Sto[2] - Me[2])
        out["lower_third_ratio"] = a / max(b, 1e-6)
        out["lower_face_height"] = float(abs(Sn[2] - Me[2]))
        out["chin_height"] = float(b)
    if Pg is not None:
        out["chin_midline_deviation"] = float(Pg[0])
    return out


def chin_width(vertices: np.ndarray, pog_soft: np.ndarray, depth: float = 10.0) -> float | None:
    """Width of the chin prominence: axial skin section through soft pogonion, measured
    ``depth`` mm behind the most anterior point."""
    sel = np.abs(vertices[:, 2] - pog_soft[2]) < 0.8
    v = vertices[sel]
    if len(v) < 20:
        return None
    y_cut = v[:, 1].max() - depth
    band = v[np.abs(v[:, 1] - y_cut) < 0.6]
    if len(band) < 2:
        return None
    left = band[band[:, 0] < 0]
    right = band[band[:, 0] > 0]
    if len(left) == 0 or len(right) == 0:
        return None
    # outermost points on each side
    return float(right[:, 0].max() - left[:, 0].min())


def asymmetry_index(vertices: np.ndarray, region_z: tuple, region_y_min: float) -> float:
    """RMS distance (mm) between the lower-face skin and its mirror image about x = 0."""
    from scipy.spatial import cKDTree
    sel = (vertices[:, 2] > region_z[0]) & (vertices[:, 2] < region_z[1]) & (vertices[:, 1] > region_y_min)
    v = vertices[sel]
    if len(v) < 100:
        return float("nan")
    m = v * np.array([-1, 1, 1])
    d, _ = cKDTree(v).query(m)
    return float(np.sqrt(np.mean(d ** 2)))


def describe(metrics: dict) -> list[dict]:
    rows = []
    for k, val in metrics.items():
        label, unit, lo, hi, pose, ref = NORMS.get(k, (k, "", None, None, False, ""))
        status = "n/a"
        if lo is not None and hi is not None:
            status = "within" if lo <= val <= hi else ("below" if val < lo else "above")
        rows.append({"key": k, "label": label, "value": val, "unit": unit, "norm": [lo, hi], "status": status,
                     "depends_on_head_pose": pose, "reference": ref})
    return rows
