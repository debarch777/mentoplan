"""Automatic anatomical landmarking.

Soft-tissue and bony mid-line landmarks are located on mid-sagittal contours
(ordered outlines from the segmentation masks), which is far more robust than
picking extreme mesh vertices.  Bilateral structures (mental foramina, gonion,
canine root apices) are found on the separated mandible.

All coordinates returned are in the head frame (RAS, mm).  Every landmark
carries a confidence in [0, 1] and a ``method`` string so the UI can show
which points need manual review.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from ..imaging.segment import Segmentation, _dilate, _erode, _largest_component
from .frame import HeadFrame


@dataclass
class Landmark:
    name: str
    position: np.ndarray       # head frame mm
    confidence: float = 1.0
    method: str = "auto"

    def to_dict(self):
        return {"name": self.name, "position": [float(v) for v in self.position],
                "confidence": float(self.confidence), "method": self.method}


@dataclass
class LandmarkSet:
    points: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)   # name -> (n, 2) arrays of (y, z)
    notes: list = field(default_factory=list)

    def __getitem__(self, k) -> np.ndarray:
        return self.points[k].position

    def __contains__(self, k):
        return k in self.points

    def add(self, name, pos, conf=1.0, method="auto"):
        self.points[name] = Landmark(name, np.asarray(pos, float), conf, method)

    def to_dict(self):
        return {"points": {k: v.to_dict() for k, v in self.points.items()},
                "profiles": {k: np.asarray(v).round(3).tolist() for k, v in self.profiles.items()},
                "notes": self.notes}

    @staticmethod
    def from_dict(d):
        ls = LandmarkSet()
        for k, v in d["points"].items():
            ls.points[k] = Landmark(k, np.array(v["position"]), v.get("confidence", 1.0), v.get("method", "auto"))
        ls.profiles = {k: np.array(v) for k, v in d.get("profiles", {}).items()}
        ls.notes = list(d.get("notes", []))
        return ls


# --------------------------------------------------------------------------- plane sampling

class PlaneSampler:
    """Samples smoothed segmentation masks on planes of the head frame."""

    def __init__(self, seg: Segmentation, frame: HeadFrame, sigma_mm: float = 0.4):
        self.seg = seg
        self.frame = frame
        sp = seg.volume.spacing
        self._fields = {}
        self._sigma = sigma_mm / sp[0]

    def field(self, name: str) -> np.ndarray:
        if name not in self._fields:
            m = getattr(self.seg, name)
            self._fields[name] = ndi.gaussian_filter(m.astype(np.float32), self._sigma)
        return self._fields[name]

    def sample(self, name: str, pts_head: np.ndarray) -> np.ndarray:
        w = self.frame.to_world(pts_head.reshape(-1, 3))
        ijk = self.seg.volume.world_to_voxel(w)
        vals = ndi.map_coordinates(self.field(name), ijk.T, order=1, mode="constant", cval=0.0)
        return vals.reshape(pts_head.shape[:-1])

    def sagittal(self, name: str, x: float, yr, zr, res: float = 0.25):
        ys = np.arange(yr[0], yr[1], res)
        zs = np.arange(zr[0], zr[1], res)
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        P = np.stack([np.full_like(Y, x), Y, Z], -1)
        return self.sample(name, P), ys, zs

    def contours(self, name: str, x: float, yr, zr, res: float = 0.25, level: float = 0.5):
        img, ys, zs = self.sagittal(name, x, yr, zr, res)
        cs = measure.find_contours(img, level)
        out = []
        for c in cs:
            y = ys[0] + c[:, 0] * res
            z = zs[0] + c[:, 1] * res
            out.append(np.stack([y, z], 1))
        return out


# --------------------------------------------------------------------------- contour helpers

def _resample_closed(c: np.ndarray, step: float = 0.25) -> np.ndarray:
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    n = max(int(s[-1] / step), 10)
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])], 1)


def _arc_walk(c: np.ndarray, start: int, direction: int, length_mm: float, step: float = 0.25) -> np.ndarray:
    n = int(length_mm / step)
    return (start + direction * np.arange(n)) % len(c)


def _chord_extreme(c: np.ndarray, idx: np.ndarray, outward_ref: np.ndarray, mode: str) -> int:
    """Index (into c) of the point on the arc idx that is farthest from the chord
    between its endpoints, on the outward ('convex') or inward ('concave') side.
    ``outward_ref`` is a point inside the head used to decide which side is outward."""
    a, b = c[idx[0]], c[idx[-1]]
    t = b - a
    nrm = np.array([t[1], -t[0]]) / max(np.linalg.norm(t), 1e-9)
    if (a - outward_ref) @ nrm < 0:
        nrm = -nrm
    s = (c[idx] - a) @ nrm
    k = np.argmax(s) if mode == "convex" else np.argmin(s)
    return int(idx[k])


def _local_extreme(c: np.ndarray, idx: np.ndarray, axis: int, mode: str, smooth: float = 4) -> int:
    v = ndi.gaussian_filter1d(c[idx, axis], smooth)
    k = np.argmax(v) if mode == "max" else np.argmin(v)
    return int(idx[k])


# --------------------------------------------------------------------------- main detector

def detect_landmarks(seg: Segmentation, frame: HeadFrame, skin_pts_head: np.ndarray) -> LandmarkSet:
    ls = LandmarkSet()
    ps = PlaneSampler(seg, frame)
    lo = skin_pts_head.min(0) - 5
    hi = skin_pts_head.max(0) + 5

    # ---------------- soft-tissue mid-sagittal profile
    cs = ps.contours("head", 0.0, (lo[1], hi[1]), (lo[2], hi[2]))
    if not cs:
        raise RuntimeError("no mid-sagittal skin contour found")
    c = _resample_closed(max(cs, key=len))
    fov_val = ps.sample("fov", np.c_[np.zeros(len(c)), c])
    valid = (fov_val > 0.9) & _inside_scan_hull(seg, frame, np.c_[np.zeros(len(c)), c], margin=1.5)
    soft_landmarks_from_contour(ls, c, valid, lo)
    ls.notes.append(f"symmetry plane rms {frame.symmetry_rms_mm:.2f} mm")
    return ls


def _inside_scan_hull(seg: Segmentation, frame: HeadFrame, pts_head: np.ndarray, margin: float = 1.5) -> np.ndarray:
    """Points lying at least ``margin`` mm inside the convex hull of all non-padding voxels."""
    from scipy.spatial import ConvexHull
    d = seg.volume.data
    sub = d[::3, ::3, ::3]
    idx = np.argwhere(sub > float(d.min()) + 1e-3) * 3
    if len(idx) < 100:
        return np.ones(len(pts_head), bool)
    rng = np.random.default_rng(0)
    hull = ConvexHull(seg.volume.voxel_to_world(idx[rng.choice(len(idx), min(len(idx), 300000), replace=False)]))
    w = frame.to_world(pts_head)
    inside = np.all(w @ hull.equations[:, :3].T + hull.equations[:, 3] <= -margin, axis=1)
    # tissue truncated at the edge of the image array (field of view smaller than the head)
    ijk = seg.volume.world_to_voxel(w)
    edge = np.minimum(ijk, np.array(d.shape) - 1 - ijk).min(1) * seg.volume.spacing.min()
    return inside & (edge > 3.0 + margin)


def soft_landmarks_from_contour(ls: LandmarkSet, c: np.ndarray, valid: np.ndarray, lo: np.ndarray) -> LandmarkSet:
    """Soft-tissue mid-line landmarks on a closed, uniformly resampled (0.25 mm) profile contour
    c (n, 2) = (y anterior, z superior) in the head frame."""
    centroid = c.mean(0)
    # pronasale: most anterior valid point in the upper 2/3 of the face
    cand = np.where(c[:, 1] > np.percentile(c[:, 1], 25))[0]   # a cut nose is still the tip (low confidence)
    i_prn = int(cand[np.argmax(c[cand, 0])])
    # walking direction that goes down
    down = 1 if c[(i_prn + 100) % len(c), 1] < c[(i_prn - 100) % len(c), 1] else -1
    ls.add("pronasale", [0, *c[i_prn]], 0.9 if valid[i_prn] else 0.1,
           "auto" if valid[i_prn] else "outside field of view")
    if not valid[i_prn]:
        ls.notes.append("nose partly outside the scan field of view: nose-based measurements are unavailable")

    # upward: soft nasion and glabella
    up_arc = _arc_walk(c, i_prn, -down, 75)
    i_n = _local_extreme(c, up_arc[int(10 / 0.25):], 0, "min")
    ok_n = valid[i_n] and valid[i_prn]
    ls.add("soft_nasion", [0, *c[i_n]], 0.85 if ok_n else 0.1, "auto" if ok_n else "outside field of view")
    g_arc = _arc_walk(c, i_n, -down, 35)
    i_g = _local_extreme(c, g_arc, 0, "max")
    ok_g = valid[i_g] and ok_n
    ls.add("soft_glabella", [0, *c[i_g]], 0.8 if ok_g else 0.1, "auto" if ok_g else "outside field of view")

    # downward profile
    arc = _arc_walk(c, i_prn, down, 160)
    zs = c[arc, 1]
    # subnasale: junction of columella and upper lip = where the profile, after running back
    # along the columella (>= 6 mm behind the nose tip), turns downward (tangent > 45 deg).  This
    # holds whether or not the under-side of the nose was captured (face scans often miss it).
    walk = _arc_walk(c, i_prn, down, 60)
    pts = c[walk]
    tang = np.gradient(ndi.gaussian_filter1d(pts, 4, axis=0), axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    back = pts[:, 0] < c[i_prn, 0] - 6.0
    steep = tang[:, 1] < -0.7
    cand_sn = np.where(back & steep)[0]
    if len(cand_sn):
        i_sn = int(walk[cand_sn[0]])
    else:
        end = arc[np.argmax(zs < c[i_prn, 1] - 30)]
        seg_sn = _arc_walk(c, i_prn, down, ((end - i_prn) * down) % len(c) * 0.25)
        i_sn = _chord_extreme(c, seg_sn, centroid, "concave")
    ls.add("subnasale", [0, *c[i_sn]], 0.85)
    # stomion first (the most reliable lip feature): deepest point 12-35 mm below subnasale
    a_lip = _arc_walk(c, i_sn, down, 70)
    dz = c[i_sn, 1] - c[a_lip, 1]
    win = a_lip[(dz > 12) & (dz < 35)]
    if len(win) < 8:
        win = a_lip[(dz > 6) & (dz < 40)]
    i_sto = _local_extreme(c, win, 0, "min", smooth=3)
    ls.add("stomion", [0, *c[i_sto]], 0.8)
    # labrale superius: most anterior point between subnasale and stomion
    k_sto = int(np.where(a_lip == i_sto)[0][0])
    a_ls = a_lip[4:max(k_sto - 2, 6)]
    i_ls = _local_extreme(c, a_ls, 0, "max", smooth=2)
    ls.add("labrale_superius", [0, *c[i_ls]], 0.85)
    # labrale inferius: most anterior point within 25 mm below the stomion
    a_li = _arc_walk(c, i_sto, down, 30)
    a_li = a_li[(c[i_sto, 1] - c[a_li, 1] > 2) & (c[i_sto, 1] - c[a_li, 1] < 25)]
    i_li = _local_extreme(c, a_li, 0, "max")
    ls.add("labrale_inferius", [0, *c[i_li]], 0.85)
    # chin: soft B point (labiomental fold) and soft pogonion
    a_chin = _arc_walk(c, i_li, down, 45)
    i_pog_s = _local_extreme(c, a_chin[int(8 / 0.25):], 0, "max")
    a_b = _arc_walk(c, i_li, down, ((i_pog_s - i_li) * down % len(c)) * 0.25)
    i_bs = _chord_extreme(c, a_b, centroid, "concave")
    ls.add("soft_b_point", [0, *c[i_bs]], 0.85)
    # soft pogonion: most anterior chin point within 30 mm below the labiomental fold
    a_pg = _arc_walk(c, i_bs, down, 30)
    i_pog_s = _local_extreme(c, a_pg[8:], 0, "max", smooth=3)
    ls.add("soft_pogonion", [0, *c[i_pog_s]], 0.9)
    ls.profiles["soft_midline"] = c
    ls.profiles["soft_midline_valid"] = valid.astype(float)
    ls._contour_state = (c, down, i_pog_s, centroid, valid, lo)  # used by detect_bony_midline
    return ls


def mesh_midline_contour(vertices: np.ndarray, faces: np.ndarray, x: float = 0.0, res: float = 0.25) -> np.ndarray:
    """Outer mid-sagittal profile of an (open) face-scan mesh as a closed 0.25 mm contour (y, z).

    The section may contain pieces inside the mouth or nostrils.  As for CT, the profile is the
    boundary of the air in front of the face: the section is rasterised, its open ends are closed
    towards the back, air is flooded from the front and narrow openings (parted lips, nostrils)
    are sealed with a 3 mm closing before the outer contour is traced."""
    import trimesh
    tm = trimesh.Trimesh(vertices, faces, process=False)
    sec = tm.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
    if sec is None:
        raise RuntimeError("the surface does not cross the mid-sagittal plane")
    pieces = [np.asarray(e.discrete(sec.vertices))[:, 1:] for e in sec.entities]
    pieces = [p for p in pieces if len(p) >= 2]
    if not pieces:
        raise RuntimeError("empty mid-sagittal section")
    allp = np.vstack(pieces)
    lo = allp.min(0) - 8.0
    hi = allp.max(0) + 8.0
    shape = tuple(np.ceil((hi - lo) / res).astype(int) + 1)
    img = np.zeros(shape, bool)

    def draw(p0, p1):
        n = int(np.ceil(np.linalg.norm(p1 - p0) / (0.5 * res))) + 1
        t = np.linspace(0, 1, n)[:, None]
        q = np.rint(((p0 + t * (p1 - p0)) - lo) / res).astype(int)
        img[np.clip(q[:, 0], 0, shape[0] - 1), np.clip(q[:, 1], 0, shape[1] - 1)] = True

    for p in pieces:
        for i in range(len(p) - 1):
            draw(p[i], p[i + 1])
    # bridge holes of the scan (e.g. the under-side of the nose is often not captured): join
    # vertically consecutive pieces at their closest endpoints when the gap is < 30 mm
    order = sorted(pieces, key=lambda p: -p[:, 1].max())
    for upper, lower in zip(order[:-1], order[1:]):
        ends_u = np.array([upper[0], upper[-1]])
        ends_l = np.array([lower[0], lower[-1]])
        d = np.linalg.norm(ends_u[:, None] - ends_l[None], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] < 30.0:
            draw(ends_u[i], ends_l[j])
    # close the scan's open top and bottom towards the back of the image
    top = allp[np.argmax(allp[:, 1])]
    bot = allp[np.argmin(allp[:, 1])]
    draw(top, np.array([lo[0], top[1]]))
    draw(bot, np.array([lo[0], bot[1]]))
    img = ndi.binary_dilation(img)
    seed = np.zeros(shape, bool)
    seed[-1, :] = True                                    # anterior border
    air = ndi.binary_propagation(seed & ~img, mask=~img)
    head = ~air
    r = int(round(3.0 / res))
    yy, zz = np.mgrid[-r:r + 1, -r:r + 1]
    disk = yy ** 2 + zz ** 2 <= r * r
    head = ndi.binary_closing(np.pad(head, r + 1), structure=disk)[r + 1:-r - 1, r + 1:-r - 1] | head
    cs = measure.find_contours(head.astype(float), 0.5)
    c = max(cs, key=len)
    c = lo + c * res
    return _resample_closed(c)


def detect_bony_midline(seg: Segmentation, frame: HeadFrame, ls: LandmarkSet, mandible: np.ndarray) -> None:
    """Symphyseal landmarks from the mid-sagittal contour of the separated mandible,
    then the soft-tissue landmarks that depend on them (soft menton/gnathion, cervical point)."""
    c, down, i_pog_s, centroid, valid, lo = ls._contour_state
    ps = PlaneSampler(seg, frame)
    ps._fields["mandible"] = ndi.gaussian_filter(mandible.astype(np.float32), ps._sigma)
    pog_s = c[i_pog_s]
    yr = (pog_s[0] - 60, pog_s[0] + 10)
    zr = (pog_s[1] - 30, pog_s[1] + 70)
    cb = [k for k in ps.contours("mandible", 0.0, yr, zr) if len(k) >= 40]
    if not cb:
        raise RuntimeError("mandibular symphysis not found in mid-sagittal plane")
    sym = _resample_closed(max(cb, key=lambda k: len(k)))
    sym_c = sym.mean(0)
    i_me = int(np.argmin(sym[:, 1]))
    i_tip = int(np.argmax(sym[:, 1]))
    n = len(sym)
    arc1 = np.arange(i_tip, i_tip + ((i_me - i_tip) % n) + 1) % n
    arc2 = (np.arange(i_me, i_me + ((i_tip - i_me) % n) + 1) % n)[::-1]
    ant = arc1 if sym[arc1, 0].mean() > sym[arc2, 0].mean() else arc2
    # pogonion: most anterior point of the anterior outline below the alveolar region
    ant_low = ant[sym[ant, 1] < sym[i_tip, 1] - 12]
    if len(ant_low) < 8:
        raise RuntimeError("symphysis outline too short")
    i_pog = _local_extreme(sym, ant_low[len(ant_low) // 3:], 0, "max", smooth=3)
    k_pog = int(np.where(ant == i_pog)[0][0])
    k_top = int(np.where(ant == ant_low[0])[0][0])
    a_b = ant[k_top:k_pog + 1]
    i_b = _chord_extreme(sym, a_b, sym_c, "concave") if len(a_b) > 4 else int(ant_low[0])
    a_gn = ant[k_pog:]
    i_gn = _chord_extreme(sym, a_gn, sym_c, "convex") if len(a_gn) > 4 else i_me
    ls.add("lower_incisor_tip", [0, *sym[i_tip]], 0.7)
    ls.add("b_point", [0, *sym[i_b]], 0.85)
    ls.add("pogonion", [0, *sym[i_pog]], 0.9)
    ls.add("gnathion", [0, *sym[i_gn]], 0.85)
    ls.add("menton", [0, *sym[i_me]], 0.9)
    ls.profiles["bone_symphysis"] = sym

    # soft menton: on the submental outline vertically below bony menton
    a_sub = _arc_walk(c, i_pog_s, down, 60)
    below = a_sub[c[a_sub, 1] < sym[i_me, 1]]
    if below.size:
        i_me_s = int(below[np.argmin(np.abs(c[below, 0] - sym[i_me, 0]))])
    else:
        i_me_s = int(a_sub[np.argmin(c[a_sub, 1])])
    ls.add("soft_menton", [0, *c[i_me_s]], 0.8)
    a_gs = _arc_walk(c, i_pog_s, down, ((i_me_s - i_pog_s) * down % len(c)) * 0.25)
    i_gn_s = _chord_extreme(c, a_gs, centroid, "convex") if len(a_gs) > 4 else i_me_s
    ls.add("soft_gnathion", [0, *c[i_gn_s]], 0.8)
    a_cv = _arc_walk(c, i_me_s, down, 55)
    i_cv = _chord_extreme(c, a_cv, centroid, "concave")
    cv_valid = bool(np.all(valid[a_cv])) and c[i_cv, 1] > lo[2] + 3
    ls.add("cervical_point", [0, *c[i_cv]], 0.7 if cv_valid else 0.15,
           "auto" if cv_valid else "outside field of view")
    if not cv_valid:
        ls.notes.append("submental/neck region partly outside the scan field of view")
    th = np.linalg.norm(c[i_pog_s] - sym[i_pog])
    ls.notes.append(f"soft-tissue thickness at pogonion {th:.1f} mm")


# --------------------------------------------------------------------------- mandible

def separate_mandible(seg: Segmentation, frame: HeadFrame, ls: LandmarkSet) -> np.ndarray:
    """Binary mask of the mandible (incl. lower teeth).

    Teeth in occlusion (often bridged further by orthodontic brackets) and the
    TMJ fuse the mandible to the cranium in a threshold mask, and the contacts
    can be wider than any safe erosion.  We therefore fit the occlusal plane to
    the tooth voxels and place watershed markers well below it (mandibular
    body, connected to the symphysis) and well above it (maxilla / cranium).
    Flooding the inverted distance map from these markers splits the bone at
    the narrowest necks between them: the cusp contacts and the joint space.
    """
    from skimage.segmentation import watershed

    sp = seg.volume.spacing
    pog_s = ls["soft_pogonion"]
    sn = ls["subnasale"]
    sto = ls["stomion"]
    box_h = np.array([[-80, pog_s[1] - 130, pog_s[2] - 25], [80, pog_s[1] + 15, sn[2] + 55]])
    corners = np.array([[x, y, z] for x in box_h[:, 0] for y in box_h[:, 1] for z in box_h[:, 2]])
    vc = seg.volume.world_to_voxel(frame.to_world(corners))
    lo = np.clip(np.floor(vc.min(0)).astype(int), 0, None)
    hi = np.minimum(np.ceil(vc.max(0)).astype(int) + 1, seg.bone.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    bone = seg.bone[sl]
    teeth = seg.teeth[sl]
    A = frame.world_to_head @ seg.volume.affine
    ii, jj, kk = np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij", sparse=True)
    X = A[0, 0] * ii + A[0, 1] * jj + A[0, 2] * kk + A[0, 3]
    Y = A[1, 0] * ii + A[1, 1] * jj + A[1, 2] * kk + A[1, 3]
    Z = A[2, 0] * ii + A[2, 1] * jj + A[2, 2] * kk + A[2, 3]

    # occlusal surface z = c0 + c1*y + c2*x^2 fitted to tooth voxels near the stomion level
    t_idx = np.argwhere(teeth)
    th = np.c_[X[t_idx[:, 0], 0, 0] * 0 + (A[0, :3] @ (t_idx + lo).T + A[0, 3]),
               A[1, :3] @ (t_idx + lo).T + A[1, 3], A[2, :3] @ (t_idx + lo).T + A[2, 3]]
    sel = (np.abs(th[:, 2] - sto[2]) < 22) & (np.abs(th[:, 0]) < 45)
    th = th[sel]
    if len(th) < 200:
        occ = lambda x, y: np.full(np.broadcast(x, y).shape, sto[2] - 2.0)
        ls.notes.append("occlusal plane: too few tooth voxels, using stomion level")
        y_teeth_min = pog_s[1] - 45
    else:
        Mx = np.c_[np.ones(len(th)), th[:, 1], th[:, 0] ** 2]
        coef = np.linalg.lstsq(Mx, th[:, 2], rcond=None)[0]
        for _ in range(3):   # re-weight towards the contact zone (median of upper+lower crowns)
            r = th[:, 2] - Mx @ coef
            w = np.abs(r) < 8
            coef = np.linalg.lstsq(Mx[w], th[w, 2], rcond=None)[0]
        occ = lambda x, y: coef[0] + coef[1] * y + coef[2] * x ** 2
        y_teeth_min = float(np.percentile(th[:, 1], 1))
    zocc = occ(X, Y)
    below = bone & (Z < zocc - 7.0)
    lab, _ = ndi.label(below)
    seed_h = ls["soft_pogonion"] + np.array([0.0, -10.0, 0.0])
    seed = np.round(seg.volume.world_to_voxel(frame.to_world(seed_h[None]))[0]).astype(int) - lo
    seed = np.clip(seed, 0, np.array(bone.shape) - 1)
    l0 = lab[tuple(seed)]
    if l0 == 0:
        d, inds = ndi.distance_transform_edt(lab == 0, return_indices=True)
        l0 = lab[tuple(inds[:, seed[0], seed[1], seed[2]])]
    mand_marker = lab == l0
    dentate = Y > y_teeth_min - 5
    upper_marker = bone & (((Z > zocc + 7.0) & dentate) | (Z > zocc + 38.0))
    markers = np.zeros(bone.shape, np.int32)
    markers[upper_marker] = 2
    markers[mand_marker] = 1
    edt = ndi.distance_transform_edt(bone, sampling=sp)
    ws = watershed(-edt, markers, mask=bone)
    mand_c = _largest_component(ws == 1)
    mand_c = _reassign_teeth(mand_c, bone, teeth, sp)
    mand = np.zeros_like(seg.bone)
    mand[sl] = mand_c
    ls.notes.append("mandible separated by occlusal-plane watershed; teeth assigned by alveolar anchorage")
    return mand


def _reassign_teeth(mand: np.ndarray, bone: np.ndarray, teeth: np.ndarray, sp) -> np.ndarray:
    """Attribute each tooth to the jaw whose (non-dental) alveolar bone anchors it.

    Crowns in contact - deep bite, anterior crossbite, brackets - can make the watershed hand an
    upper incisor to the mandible.  Tooth components (after a 0.4 mm erosion that breaks bracket
    bridges) are counted against the surrounding 1.5 mm shell of mandibular vs maxillary bone."""
    core = _erode(teeth, 0.4, sp)
    lab, n = ndi.label(core)
    if n == 0:
        return mand
    shell_m = bone & ~teeth & mand
    shell_x = bone & ~teeth & ~mand
    grown = ndi.grey_dilation(lab, size=(int(np.ceil(1.5 / sp[0])) * 2 + 1,) * 3)
    cm = np.bincount(grown[shell_m], minlength=n + 1)
    cx = np.bincount(grown[shell_x], minlength=n + 1)
    lower = cm >= cx
    lower[0] = False
    # nearest component for every tooth voxel (eroded-away rim included)
    _, ind = ndi.distance_transform_edt(lab == 0, return_indices=True)
    nearest = lab[tuple(ind)]
    tooth_lower = teeth & lower[nearest]
    return (mand & ~teeth) | tooth_lower


def _heights(seg: Segmentation, frame: HeadFrame) -> np.ndarray:
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in seg.bone.shape], indexing="ij", sparse=True)
    A = frame.world_to_head @ seg.volume.affine
    return A[2, 0] * ii + A[2, 1] * jj + A[2, 2] * kk + A[2, 3]


def detect_bilateral(seg: Segmentation, frame: HeadFrame, lms: LandmarkSet, mandible_mesh_head) -> None:
    """Mental foramina, gonia and canine/incisor root apices (adds to ``lms``)."""
    v = mandible_mesh_head.vertices
    nrm = mandible_mesh_head.vertex_normals()
    me = lms["menton"]
    tip = lms["lower_incisor_tip"]
    body_h = tip[2] - me[2]

    # ---- gonion: most posterior-inferior point of each half (angle of the mandible)
    for side, sgn in (("right", 1), ("left", -1)):
        sel = (sgn * v[:, 0] > 25) & (v[:, 1] < me[1] - 40)
        if sel.sum() < 20:
            lms.notes.append(f"gonion {side}: not found")
            continue
        p = v[sel]
        score = -p[:, 1] - p[:, 2]
        lms.add(f"gonion_{side}", p[np.argmax(score)], 0.6)

    # ---- root apices of the anterior teeth: lowest point of tooth components that reach the crowns
    # (dense symphyseal cortex can exceed the tooth threshold; it is not connected to the crowns)
    t_idx = np.argwhere(seg.teeth)
    if len(t_idx):
        th = frame.to_head(seg.volume.voxel_to_world(t_idx))
        sel = (np.abs(th[:, 0]) < 20) & (th[:, 1] > me[1] - 25) & (th[:, 2] > me[2]) & (th[:, 2] < tip[2] + 3)
        if sel.sum() > 20:
            sub = np.zeros(seg.teeth.shape, bool)
            sub[tuple(t_idx[sel].T)] = True
            lab, nl = ndi.label(sub)
            labs = lab[tuple(t_idx[sel].T)]
            tops = ndi.maximum(th[sel, 2], labs, index=np.arange(1, nl + 1)) if nl else []
            crown = [i + 1 for i, z in enumerate(np.atleast_1d(tops)) if z > tip[2] - 10]
            ok = np.isin(labs, crown)
            if ok.sum() > 20:
                apex_z = float(np.percentile(th[sel][ok, 2], 2))
                lms.add("anterior_root_apex", [0, float(np.median(th[sel][ok, 1])), apex_z], 0.55,
                        "lowest dentin of crown-connected teeth |x|<20mm")
    if "anterior_root_apex" not in lms:
        # population prior: incisor/canine roots end ~ 60% of the way from the incisal edge to menton
        lms.add("anterior_root_apex", [0, tip[1] - 6, tip[2] - 0.6 * (tip[2] - me[2])], 0.2, "population prior")
        lms.notes.append("root apices placed from a population prior; please verify")
    # ---- mental foramina: holes in the buccal cortex
    hu = seg.volume
    for side, sgn in (("right", 1), ("left", -1)):
        pog_y = lms["pogonion"][1]
        sel = (sgn * v[:, 0] > 14) & (sgn * v[:, 0] < 34) & (v[:, 1] > pog_y - 36) & (v[:, 2] > me[2] + 0.2 * body_h) & \
              (v[:, 2] < me[2] + 0.6 * body_h) & (sgn * nrm[:, 0] > 0.3) & (v[:, 1] > me[1] - 40)
        if sel.sum() < 30:
            continue
        p, nn = v[sel], nrm[sel]
        # sample HU just inside the outer surface; foramen = no cortical bone along the normal
        depths = np.linspace(0.6, 3.0, 7)
        samples = []
        for dd in depths:
            q = frame.to_world(p - nn * dd)
            samples.append(ndi.map_coordinates(hu.data, hu.world_to_voxel(q).T, order=1))
        cort = np.max(np.stack(samples), axis=0)
        thr = np.percentile(cort, 8)
        hole = cort <= thr
        if hole.sum() < 5:
            continue
        # cluster the hole points and take the most compact large cluster
        from scipy.cluster.hierarchy import fcluster, linkage
        hp = p[hole]
        if len(hp) > 1:
            Z = linkage(hp, "single")
            labels = fcluster(Z, 2.0, "distance")
            best = max(set(labels), key=lambda l: (labels == l).sum())
            cpts = hp[labels == best]
        else:
            cpts = hp
        contrast = float(np.median(cort[~hole]) - np.median(cort[hole]))
        conf = float(np.clip(contrast / 800.0, 0.1, 0.8)) * float(np.clip(len(cpts) / 15, 0.3, 1.0))
        lms.add(f"mental_foramen_{side}", cpts.mean(0), conf, "cortical gap search")
    # bilateral consistency: the foramina are near-symmetric; re-place an outlier at the mirror of
    # the other side (snapped to the bone surface) with reduced confidence
    if "mental_foramen_right" in lms and "mental_foramen_left" in lms:
        r, l = lms["mental_foramen_right"], lms["mental_foramen_left"]
        mir = l * np.array([-1, 1, 1])
        if np.linalg.norm(r - mir) > 8.0:
            cr, cl = lms.points["mental_foramen_right"].confidence, lms.points["mental_foramen_left"].confidence
            bad, good, sgn = ("right", l, -1) if cr < cl else ("left", r, 1)
            if cr == cl:      # equal confidence: keep the one closer to the typical position
                typ_y = lms["pogonion"][1] - 27
                bad, good = ("right", l) if abs(r[1] - typ_y) > abs(l[1] - typ_y) else ("left", r)
            target = good * np.array([-1, 1, 1])
            j = np.argmin(np.linalg.norm(v - target, axis=1))
            lms.add(f"mental_foramen_{bad}", v[j], 0.3, "mirrored from the opposite side")
            lms.notes.append(f"mental foramen {bad}: inconsistent detection, mirrored from the other side - verify")
    for side, sgn in (("right", 1), ("left", -1)):
        if f"mental_foramen_{side}" not in lms:
            # population prior: below the 2nd premolar, ~24 mm from the midline, mid-body height
            prior = np.array([sgn * 24.0, me[1] - 12.0, me[2] + 0.45 * body_h])
            j = np.argmin(np.linalg.norm(v - prior, axis=1))
            lms.add(f"mental_foramen_{side}", v[j], 0.15, "population prior")
            lms.notes.append(f"mental foramen {side} placed from population prior; please verify")
