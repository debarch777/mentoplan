"""Patient head coordinate frame.

The *head frame* is an anatomical RAS frame (mm): +X patient right, +Y
anterior, +Z superior, with the X=0 plane being the mid-sagittal plane
estimated by reflective symmetry of the skeleton and skin.  Pitch defaults to
the scanner's superior axis (CBCT is normally acquired seated in natural head
position) and can be corrected by the clinician (``pitch_deg``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class HeadFrame:
    world_to_head: np.ndarray            # 4x4, LPS world -> head RAS
    symmetry_rms_mm: float = 0.0
    pitch_deg: float = 0.0               # user correction applied about the X axis
    info: dict = field(default_factory=dict)

    @property
    def head_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_head)

    def to_head(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        return p @ self.world_to_head[:3, :3].T + self.world_to_head[:3, 3]

    def to_world(self, p: np.ndarray) -> np.ndarray:
        T = self.head_to_world
        p = np.asarray(p, float)
        return p @ T[:3, :3].T + T[:3, 3]

    def with_pitch(self, pitch_deg: float) -> "HeadFrame":
        d = np.radians(pitch_deg - self.pitch_deg)
        R = np.eye(4)
        R[1:3, 1:3] = [[np.cos(d), -np.sin(d)], [np.sin(d), np.cos(d)]]
        return HeadFrame(R @ self.world_to_head, self.symmetry_rms_mm, pitch_deg, dict(self.info))

    def to_dict(self) -> dict:
        return {"world_to_head": self.world_to_head.tolist(), "symmetry_rms_mm": self.symmetry_rms_mm,
                "pitch_deg": self.pitch_deg, "info": self.info}

    @staticmethod
    def from_dict(d: dict) -> "HeadFrame":
        return HeadFrame(np.array(d["world_to_head"]), d.get("symmetry_rms_mm", 0.0), d.get("pitch_deg", 0.0),
                         d.get("info", {}))


def _rigid_icp(src: np.ndarray, dst_tree: cKDTree, dst: np.ndarray, iters: int = 150, trim: float = 0.8):
    """Trimmed point-to-point ICP; returns R, t mapping src onto dst."""
    R = np.eye(3)
    t = np.zeros(3)
    prev = np.inf
    for _ in range(iters):
        p = src @ R.T + t
        d, j = dst_tree.query(p)
        keep = d <= np.quantile(d, trim)
        a, b = p[keep], dst[j[keep]]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        dR = Vt.T @ D @ U.T
        R, t = dR @ R, dR @ (t - ca) + cb
        err = float(np.sqrt(np.mean(d[keep] ** 2)))
        if abs(prev - err) < 1e-6:
            break
        prev = err
    return R, t, prev


def fit_symmetry_plane(points: np.ndarray, n0: np.ndarray, d0: float, iters: int = 10,
                       trim: float = 0.8) -> tuple[np.ndarray, float, float]:
    """Mid-sagittal plane by mirror-and-register.

    The cloud is mirrored across the current plane estimate and rigidly
    registered back onto itself (trimmed ICP, robust to genuine asymmetry).
    The composite map x -> R(H x + 2 d n) + t is an improper isometry that
    equals the reflection about the true symmetry plane: its normal is the
    eigenvector with eigenvalue -1 and the offset follows from the fixed
    points.  Repeating once or twice removes the residual linearisation error.
    Returns (unit normal, offset d with plane {x: x.n = d}, rms residual mm).
    """
    tree = cKDTree(points)
    n = n0 / np.linalg.norm(n0)
    d = float(d0)
    rms = np.inf
    for _ in range(iters):
        H = np.eye(3) - 2 * np.outer(n, n)
        mirrored = points @ H.T + 2 * d * n
        R, t, rms = _rigid_icp(mirrored, tree, points, trim=trim)
        A = R @ H
        b = R @ (2 * d * n) + t
        w, V = np.linalg.eigh(0.5 * (A + A.T))
        n_new = V[:, np.argmin(w)]
        if n_new @ n < 0:
            n_new = -n_new
        d_new = float(b @ n_new) / 2.0
        done = np.degrees(np.arccos(np.clip(n_new @ n, -1, 1))) < 0.01 and abs(d_new - d) < 0.01
        n, d = n_new, d_new
        if done:
            break
    return n, d, rms


def estimate_head_frame(bone_pts_lps: np.ndarray, skin_pts_lps: np.ndarray, superior_hint=(0, 0, 1.0),
                        lateral_hint=(1.0, 0, 0)) -> HeadFrame:
    """Estimate the head frame from bone and skin vertices (LPS world coordinates).

    The hints are the scanner's patient axes; they only need to be within
    ~20 deg of the truth.
    """
    rng = np.random.default_rng(0)
    pts = np.concatenate([bone_pts_lps[rng.choice(len(bone_pts_lps), min(60000, len(bone_pts_lps)), replace=False)],
                          skin_pts_lps[rng.choice(len(skin_pts_lps), min(30000, len(skin_pts_lps)), replace=False)]])
    n0 = np.asarray(lateral_hint, float)
    d0 = float(np.median(pts @ n0))
    n, d, rms = fit_symmetry_plane(pts, n0, d0)
    left = n if n @ np.array(lateral_hint) > 0 else -n  # LPS +x = left
    d = d if left is n else -d
    right = -left
    sup = np.asarray(superior_hint, float)
    sup = sup - (sup @ right) * right
    sup /= np.linalg.norm(sup)
    ant = np.cross(sup, right)          # RAS: A = S x R
    R = np.stack([right, ant, sup])     # rows: head axes in world coords
    # origin: on the symmetry plane, at the skin centroid's projection
    c = skin_pts_lps.mean(0)
    c = c - ((c @ left) - d) * left
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ c
    return HeadFrame(T, rms, 0.0, {"plane_normal_world": left.tolist(), "plane_offset": d})
