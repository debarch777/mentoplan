"""Parametric, patient-specific chin implant design.

The implant is described by a thickness field t(u, v) laid over the external
surface of the anterior mandible:

* ``u`` – signed arc length (mm) along the mandibular arch, 0 at the
  mid-sagittal plane, positive towards the patient's right;
* ``v`` – height (mm) above the inferior border at that arch position
  (negative values are below/under the inferior border).

The solid is built volumetrically: a voxel belongs to the implant if it lies
outside the bone, its nearest bone point is on the labial/inferior cortex and
its distance from a *smoothed* bone surface is below t(u, v).  This gives a
watertight part whose posterior surface is an exact negative of the patient's
bone (custom fit) while the anterior surface stays smooth even over bony
irregularities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage as ndi

from ..imaging.surface import Mesh, taubin_smooth

SHAPES = ("round", "anatomical", "square", "custom")
MATERIALS = ("silicone", "porous_polyethylene", "peek", "titanium")


@dataclass
class ImplantDesign:
    projection: float = 6.0          # mm, max anterior thickness (sagittal augmentation)
    width: float = 50.0              # mm, total arch length of the footprint (incl. wings)
    height: float = 18.0             # mm, vertical extent of the footprint on the labial face
    inferior_offset: float = 0.0     # mm, lower footprint edge above (+) / below (-) the inferior border
    vertical_extension: float = 0.0  # mm, thickness wrapped under the inferior border (chin lengthening)
    peak_height: float = 0.45        # 0..1, vertical position of maximum projection within the footprint
    base_fill: float = 0.75          # 0..1, thickness at the inferior border relative to the peak
    inferior_wrap: float = 1.5       # mm, thickness of the lip wrapping under the inferior border
    shape: str = "anatomical"
    squareness: float = 0.3          # 0 = pointed/round, 1 = flat broad plateau
    wing_ratio: float = 0.35         # anatomical: wing thickness as fraction of projection
    body_fraction: float = 0.5       # anatomical: central body width as fraction of total width
    asymmetry: float = 0.0           # -1..1, relative extra projection on patient right (+) / left (-)
    midline_shift: float = 0.0       # mm, lateral shift of the implant centre (+ = patient right)
    cant_deg: float = 0.0            # degrees, coronal rotation of the footprint (+ raises the right side)
    edge_feather: float = 6.0        # mm, edge taper length
    material: str = "silicone"
    fixation: str = "screws"         # "screws", "sutures", "none"
    custom_grid: list | None = None  # optional user-sculpted delta thickness on a (u, v) grid
    name: str = "Variant"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ImplantDesign":
        known = {k: v for k, v in d.items() if k in ImplantDesign.__dataclass_fields__}
        return ImplantDesign(**known)

    def validate(self) -> list[str]:
        errs = []
        if not 0 <= self.projection <= 20:
            errs.append("projection must be within 0..20 mm")
        if not 10 <= self.width <= 110:
            errs.append("width must be within 10..110 mm")
        if not 6 <= self.height <= 40:
            errs.append("height must be within 6..40 mm")
        if not 0 <= self.vertical_extension <= 12:
            errs.append("vertical extension must be within 0..12 mm")
        if self.shape not in SHAPES:
            errs.append(f"shape must be one of {SHAPES}")
        if self.material not in MATERIALS:
            errs.append(f"material must be one of {MATERIALS}")
        return errs


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _taper(d, feather):
    """0 at the footprint edge rising smoothly (C1) to 1 at ``feather`` mm inside."""
    return _smoothstep(d / max(feather, 1e-6))


def thickness(u: np.ndarray, v: np.ndarray, d: ImplantDesign) -> np.ndarray:
    """Implant thickness (mm) at arch coordinates (u, v)."""
    u = np.asarray(u, float) - d.midline_shift
    v = np.asarray(v, float) - np.tan(np.radians(d.cant_deg)) * u
    half = d.width / 2.0
    au = np.abs(u)
    # ---- lateral profile L(u) in [0, 1]
    s = au / half
    if d.shape == "round":
        L = np.clip(1 - s ** 2, 0, None) ** 1.5
    elif d.shape == "square":
        plateau = 0.25 + 0.6 * d.squareness
        L = np.where(s < plateau, 1.0, np.cos(np.clip((s - plateau) / (1 - plateau), 0, 1) * np.pi / 2) ** 2)
    else:  # anatomical / custom: central body blended smoothly (C1) into thinner extended wings
        bh = d.body_fraction
        sb = au / (half * bh)
        plateau = 0.1 + 0.5 * d.squareness
        body = np.cos(np.clip((sb - plateau) / (1 - plateau), 0, 1) * np.pi / 2) ** 2
        wing = np.cos(np.clip(s, 0, 1) * np.pi / 2) ** 0.8
        w = d.wing_ratio
        L = (1 - w) * body + w * wing
    L = L * _taper(half - au, d.edge_feather)
    # ---- vertical profile V(v) in [0, 1] over [v0, v0 + height]
    v0 = d.inferior_offset
    vh = d.height
    t_rel = (v - v0) / vh
    pk = np.clip(d.peak_height, 0.1, 0.9)
    V0 = np.clip(d.base_fill, 0.0, 1.0)
    rise = V0 + (1 - V0) * np.sin(np.pi / 2 * np.clip(t_rel / pk, 0, 1))
    fall = np.cos(np.pi / 2 * np.clip((t_rel - pk) / (1 - pk), 0, 1)) ** 1.5
    V = np.where(t_rel < pk, rise, fall) * (t_rel <= 1)
    V = V * _taper(v0 + vh - v, d.edge_feather)
    if v0 > 0.5:
        # footprint raised above the border: feathered lower edge
        V = V * _taper(v - v0, d.edge_feather * 0.6) * (v >= v0)
    else:
        # footprint seated on the border: wrap around it, fading out 4 mm underneath
        V = np.where(v < v0, V0 * _taper(v - (v0 - 4.0), 4.0), V)
    t = d.projection * L * V
    # ---- asymmetry: linear left/right modulation (keeps the midline value)
    t = t * np.clip(1 + d.asymmetry * np.clip(u / half, -1, 1), 0, None)
    # ---- under the border only a thin lip wraps around (stabilises the implant without lengthening)
    if v0 <= 0.5:
        t = np.where(v < v0, np.minimum(t, d.inferior_wrap), t)
    # ---- wrap under the inferior border (vertical lengthening)
    if d.vertical_extension > 0:
        Lw = np.clip(1 - (au / (half * 0.85)) ** 2, 0, None) ** 1.2
        under = np.clip(1 - np.abs(v) / (d.edge_feather + 4.0), 0, 1) * (v <= 2.0)
        t = np.maximum(t, d.vertical_extension * Lw * _smoothstep(under))
    if d.custom_grid:
        g = d.custom_grid
        uu = np.asarray(g["u"], float)
        vv = np.asarray(g["v"], float)
        dt = np.asarray(g["delta"], float)
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((uu, vv), dt, bounds_error=False, fill_value=0.0)
        t = np.clip(t + f(np.stack([np.broadcast_to(u + d.midline_shift, t.shape), np.broadcast_to(v, t.shape)], -1)),
                    0, None)
    return t


def surface_thickness(u, v, n_z, behind, d: ImplantDesign) -> np.ndarray:
    """Implant thickness on the actual bone surface.

    The arch height ``v`` is measured from the inferior border, so points on the *under*-side of
    the chin also have small v; there the implant is only the thin under-border wrap (plus any
    deliberate vertical lengthening), not the labial thickness.  The inferior surface is recognised
    by its downward outward normal (``n_z``); the wrap is feathered towards the lingual side over
    3 mm (``behind`` = mm behind the labial cortex)."""
    t = thickness(u, v, d)
    under = np.clip((-np.asarray(n_z) - 0.45) / 0.3, 0.0, 1.0)
    half = d.width / 2.0
    au = np.abs(np.asarray(u, float) - d.midline_shift)
    t_under = np.minimum(t, d.inferior_wrap)
    if d.vertical_extension > 0:
        Lw = np.clip(1 - (au / (half * 0.85)) ** 2, 0, None) ** 1.2
        t_under = np.maximum(t_under, d.vertical_extension * Lw * (au < half))
    t_under = t_under * _taper(6.0 + d.vertical_extension - np.asarray(behind, float), 3.0)
    return (1.0 - under) * t + under * t_under


def node_thickness(nodes: np.ndarray, normals: np.ndarray, arch, d: ImplantDesign, du: float = 0.0,
                   dv: float = 0.0) -> np.ndarray:
    """Implant thickness at bone-interface points (head frame) with outward bone normals;
    ``du``/``dv`` shift the evaluation (placement perturbations)."""
    u, v = arch.uv(nodes)
    th = arch.angle(nodes)
    behind = np.interp(th, arch.theta, arch.r_outer) - np.hypot(nodes[:, 0], nodes[:, 1] - arch.center_y)
    return surface_thickness(np.nan_to_num(u, nan=1e6) - du, v - dv, normals[:, 2], np.clip(behind, 0, None), d)


# --------------------------------------------------------------------------- arch parameterisation

@dataclass
class ArchParam:
    """Maps head-frame points to arch coordinates (u, v) around the anterior mandible."""
    center_y: float                 # arch centre (head frame y), behind the symphysis
    center_z: float
    theta: np.ndarray               # sample angles (rad), 0 = anterior, + = patient right
    arc: np.ndarray                 # arc length at each angle (mm, signed)
    z_inf: np.ndarray               # inferior border height at each angle (mm)
    r_outer: np.ndarray             # radial distance of the labial cortex at each angle (mm)

    def angle(self, p: np.ndarray) -> np.ndarray:
        return np.arctan2(p[..., 0], p[..., 1] - self.center_y)

    def uv(self, p: np.ndarray):
        th = self.angle(p)
        u = np.interp(th, self.theta, self.arc, left=np.nan, right=np.nan)
        zi = np.interp(th, self.theta, self.z_inf)
        return u, p[..., 2] - zi

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}

    @staticmethod
    def from_dict(d):
        return ArchParam(d["center_y"], d["center_z"], np.array(d["theta"]), np.array(d["arc"]),
                         np.array(d["z_inf"]), np.array(d["r_outer"]))


def build_arch_param(mand_vertices_head: np.ndarray, lms) -> ArchParam:
    pog = lms["pogonion"]
    me = lms["menton"]
    v = mand_vertices_head
    # outer arch contour in a slab around mid-symphysis height
    z_mid = 0.5 * (pog[2] + me[2]) + 4.0
    slab = v[np.abs(v[:, 2] - z_mid) < 2.0]
    # centre: fit a circle to the anterior labial points
    cy = pog[1] - 30.0   # angular reference only; arc lengths are measured on the actual contour
    th_grid = np.radians(np.linspace(-95, 95, 191))
    th = np.arctan2(slab[:, 0], slab[:, 1] - cy)
    r = np.hypot(slab[:, 0], slab[:, 1] - cy)
    r_out = np.full(len(th_grid), np.nan)
    for i, t in enumerate(th_grid):
        sel = np.abs(th - t) < np.radians(1.5)
        if sel.sum() >= 3:
            r_out[i] = r[sel].max()
    ok = ~np.isnan(r_out)
    r_out = np.interp(th_grid, th_grid[ok], r_out[ok])
    r_out = ndi.gaussian_filter1d(r_out, 2)
    x = r_out * np.sin(th_grid)
    y = cy + r_out * np.cos(th_grid)
    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.r_[0, np.cumsum(ds)]
    s = s - np.interp(0.0, th_grid, s)
    # inferior border height per angle (lowest outer-half vertex)
    thv = np.arctan2(v[:, 0], v[:, 1] - cy)
    zi = np.full(len(th_grid), np.nan)
    rv = np.hypot(v[:, 0], v[:, 1] - cy)
    for i, t in enumerate(th_grid):
        sel = (np.abs(thv - t) < np.radians(2)) & (rv > r_out[i] - 18) & (v[:, 2] < pog[2] + 5)
        if sel.sum() >= 3:
            zi[i] = v[sel, 2].min()
    ok = ~np.isnan(zi)
    zi = np.interp(th_grid, th_grid[ok], zi[ok])
    zi = ndi.gaussian_filter1d(zi, 3)
    # the mid-line value must equal menton height
    zi = zi - (np.interp(0, th_grid, zi) - me[2])
    return ArchParam(cy, z_mid, th_grid, s, zi, r_out)


# --------------------------------------------------------------------------- solid construction

@dataclass
class ImplantSolid:
    mesh: Mesh                       # head frame
    voxel_size: float
    metrics: dict = field(default_factory=dict)


def build_implant(design: ImplantDesign, arch: ArchParam, bone_sampler, lms, voxel: float = 0.35) -> ImplantSolid:
    """Construct the implant solid.

    ``bone_sampler(points_head) -> values in [0,1]`` samples the (smoothed)
    mandible mask in head coordinates.
    """
    from skimage import measure

    pog = lms["pogonion"]
    me = lms["menton"]
    half = design.width / 2.0 + 4.0
    th_max = np.interp(half, arch.arc, arch.theta)
    th_min = np.interp(-half, arch.arc, arch.theta)
    # bounding box (head frame)
    rr = np.interp([th_min, th_max], arch.theta, arch.r_outer)
    xs = [rr[0] * np.sin(th_min), rr[1] * np.sin(th_max)]
    ys = [arch.center_y + rr[0] * np.cos(th_min), arch.center_y + rr[1] * np.cos(th_max)]
    pad = design.projection + design.vertical_extension + 6.0
    lo = np.array([min(xs) - pad, min(ys) - pad - 4, me[2] - design.vertical_extension - 8.0])
    hi = np.array([max(xs) + pad, pog[1] + pad, me[2] + design.inferior_offset + design.height + 14.0 + 15.0])
    n = np.ceil((hi - lo) / voxel).astype(int) + 1
    gx, gy, gz = (lo[i] + np.arange(n[i]) * voxel for i in range(3))
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    # memory-lean construction: float32 volumes, per-slab evaluation, no nearest-index arrays
    b = np.empty(tuple(n), np.float32)
    for k in range(n[2]):
        pts = np.stack([GX.ravel(), GY.ravel(), np.full(GX.size, gz[k])], 1)
        b[:, :, k] = bone_sampler(pts).reshape(GX.shape)
    bone = b > 0.5
    # smoothed bone surface for the anterior face (removes bony irregularities < ~1.5 mm)
    b_s = ndi.gaussian_filter(b, 1.5 / voxel)
    del b
    sdf_s = ndi.distance_transform_edt(b_s <= 0.5, sampling=voxel).astype(np.float32)
    dist = ndi.distance_transform_edt(~bone, sampling=voxel).astype(np.float32)
    wrap_depth = 6.0 + design.vertical_extension
    sol = np.zeros(tuple(n), bool)
    t_max = 0.0
    for k in range(n[2]):
        k0, k1 = max(k - 1, 0), min(k + 1, n[2] - 1)
        gxk = np.gradient(b_s[:, :, k], voxel, axis=0)
        gyk = np.gradient(b_s[:, :, k], voxel, axis=1)
        gzk = (b_s[:, :, k1] - b_s[:, :, k0]) / (max(k1 - k0, 1) * voxel)
        nb = -np.stack([gxk, gyk, gzk], -1)                 # outward normal of the bone surface
        nb /= np.maximum(np.linalg.norm(nb, axis=-1, keepdims=True), 1e-9)
        P = np.stack([GX, GY, np.full(GX.shape, gz[k])], -1)
        near = P - dist[:, :, k, None] * nb                 # closest bone point along the normal
        th = arch.angle(near)
        radial = np.stack([np.sin(th), np.cos(th), np.zeros_like(th)], -1)
        r_near = np.hypot(near[..., 0], near[..., 1] - arch.center_y)
        r_out = np.interp(th, arch.theta, arch.r_outer)
        external = ((np.einsum("...i,...i", nb, radial) > 0.25) | (nb[..., 2] < -0.55)) & \
            (r_near > r_out - wrap_depth)
        # (u, v) of the voxel itself: the nearest bone point is ambiguous over concavities
        # (B-point region) and would make the implant fill them
        u, v = arch.uv(P)
        behind = np.clip(r_out - r_near, 0, None)
        t = surface_thickness(np.nan_to_num(u, nan=1e6), v, nb[..., 2], behind, design)
        sk = (~bone[:, :, k]) & external & (sdf_s[:, :, k] < t) & (t > 0.05)
        sol[:, :, k] = sk
        if sk.any():
            t_max = max(t_max, float(t[sk].max()))
    del sdf_s, dist, b_s
    sol = _largest(sol)
    if sol.sum() < 10:
        raise ValueError("implant design produces an empty solid")
    f = ndi.gaussian_filter(sol.astype(np.float32), 0.6)
    # keep the bone-contact face exact: inside bone the field is forced to 0
    f[bone] = np.minimum(f[bone], 0.2)
    verts, faces, _, _ = measure.marching_cubes(np.pad(f, 1), 0.5)
    verts = (verts - 1) * voxel + lo
    mesh = Mesh(verts, faces[:, ::-1].astype(np.int32))
    if mesh.volume() < 0:
        mesh.faces = mesh.faces[:, ::-1].copy()
    mesh = taubin_smooth(mesh, 6)
    vol_mm3 = float(sol.sum() * voxel ** 3)
    # contact area: implant voxels face-adjacent to bone
    contact = sol & ndi.binary_dilation(bone)
    metrics = {
        "volume_cc": vol_mm3 / 1000.0,
        "max_thickness_mm": t_max,
        "contact_area_cm2": float(contact.sum() * voxel ** 2 / 100.0),
        "footprint_width_mm": design.width,
        "bbox_mm": (verts.max(0) - verts.min(0)).tolist(),
    }
    return ImplantSolid(mesh, voxel, metrics)


def _largest(mask):
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == np.argmax(sizes)
