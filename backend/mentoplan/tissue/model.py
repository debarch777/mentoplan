"""Patient-specific soft-tissue finite-element model of the lower face.

Built from the segmentation on a regular grid in the head frame:

* cells are classified by 2x2x2 super-sampling of the masks (tissue, bone,
  teeth, internal air, outside);
* cells outside the scanner's field of view are *extrapolated* from the
  nearest measured cell (so a scan that cuts just below the chin does not
  clamp the submental tissue) and flagged;
* tissue cells get a layer label from the CT: dermis (outermost layer),
  subcutaneous/deep fat (HU < -30), muscle/glandular, oral mucosa;
* boundary node sets: bone attachment (fixed), subperiosteal pocket over the
  anterior mandible (sliding, implant prescribes the normal displacement),
  lip mucosa on the teeth (sliding), ROI cut faces (fixed), outer skin;
* the lips are split along the oral fissure between the commissures
  (duplicated nodes with unilateral contact).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi

from ..imaging.segment import Segmentation
from ..anatomy.frame import HeadFrame
from scipy.spatial import cKDTree

from .hexfem import Assembler, ElementGeom

MAT_FAT, MAT_MUSCLE, MAT_SKIN, MAT_MUCOSA = 0, 1, 2, 3
MAT_NAMES = ["fat", "muscle", "dermis", "mucosa"]

C_OUT, C_TISSUE, C_BONE, C_CAVITY = 0, 1, 2, 3


@dataclass
class TissueModel:
    h: float
    origin: np.ndarray            # head-frame position of grid node (0,0,0)
    shape: tuple                  # nodes per axis
    nodes: np.ndarray             # (n, 3) head frame
    node_ijk: np.ndarray          # (n, 3)
    elems: np.ndarray             # (E, 8)
    elem_ijk: np.ndarray          # (E, 3)
    mat_id: np.ndarray            # (E,)
    extrapolated: np.ndarray      # (E,) bool, outside the scan FOV
    sets: dict = field(default_factory=dict)      # node index arrays
    normals: dict = field(default_factory=dict)   # per-set outward bone normals
    pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))  # (upper, lower) fissure pairs
    cell_class: np.ndarray | None = None          # full grid cell classes (for QA)
    elem_uv: np.ndarray | None = None             # arch coordinates of element centres
    info: dict = field(default_factory=dict)

    # cached helpers
    _geom: ElementGeom | None = None
    _asm: Assembler | None = None

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def geom(self) -> ElementGeom:
        if self._geom is None:
            self._geom = ElementGeom.from_nodes(self.nodes, self.elems)
        return self._geom

    @property
    def asm(self) -> Assembler:
        if self._asm is None:
            self._asm = Assembler(self.elems, self.n_nodes)
        return self._asm

    def elem_centers(self):
        return self.nodes[self.elems].mean(1)

    # ------------------------------------------------------------------ interpolation
    def interpolation_matrix(self, pts: np.ndarray, surface: bool = True, k: int = 8,
                             max_dist: float = 4.5) -> sp.csr_matrix:
        """Interpolation weights (sparse, n_pts x n_nodes) from FE nodes to arbitrary points.

        Inverse-distance weighting over the k nearest nodes; ``surface=True``
        restricts the support to outer-skin nodes, which lie exactly on the
        skin surface after boundary fitting.  Points above / below the oral
        fissure only see nodes of their own lip.  Weights fade to zero for
        points farther than 1.5 mm from the mesh, so regions outside the
        model (forehead, ears) are untouched and the transition is smooth
        (the ROI cut faces are fixed, hence zero displacement there too).
        """
        cand = self.sets["skin"] if surface else np.arange(self.n_nodes)
        z_cut = self.info.get("fissure", {}).get("z") if isinstance(self.info.get("fissure"), dict) else None
        rows, cols, vals, fades = [], [], [], np.zeros(len(pts))
        groups = [(np.ones(len(pts), bool), cand)]
        if z_cut is not None and len(self.pairs):
            up_c = cand[~np.isin(cand, self.pairs[:, 1])]
            lo_c = cand[~np.isin(cand, self.pairs[:, 0])]
            above = pts[:, 2] >= z_cut
            groups = [(above, up_c), (~above, lo_c)]
        for sel, cset in groups:
            idx = np.where(sel)[0]
            if len(idx) == 0 or len(cset) == 0:
                continue
            tree = cKDTree(self.nodes[cset])
            kk = min(k, len(cset))
            d, j = tree.query(pts[idx], kk)
            d = d.reshape(len(idx), kk)
            j = j.reshape(len(idx), kk)
            w = 1.0 / (d + 0.4) ** 2
            w[d > max_dist] = 0
            ws = w.sum(1, keepdims=True)
            w = np.where(ws > 0, w / np.maximum(ws, 1e-12), 0)
            fades[idx] = np.clip(1.0 - np.maximum(d[:, 0] - 1.5, 0) / (max_dist - 1.5), 0, 1)
            rows.append(np.repeat(idx, kk))
            cols.append(cset[j].ravel())
            vals.append((w * fades[idx, None]).ravel())
        W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(pts), self.n_nodes))
        W.eliminate_zeros()
        return W

    def to_dict(self):
        """Compact serialisable form (np arrays stored by the caller)."""
        return {"h": self.h, "origin": self.origin.tolist(), "shape": list(self.shape), "info": self.info}


# --------------------------------------------------------------------------- builder

@dataclass
class ModelOptions:
    h: float = 2.0
    roi_half_width: float = 58.0       # mm lateral
    roi_posterior: float = 72.0        # mm behind soft pogonion
    roi_below_menton: float = 26.0     # mm
    roi_above_subnasale: float = 6.0   # mm
    mouth_half_width: float | None = None  # mm; default from facial width
    pocket_half_arc: float = 38.0      # mm of arch length either side of the midline
    pocket_top: float = 24.0           # mm above the inferior border
    pocket_under: float = 9.0          # mm of the inferior surface (wrap) included
    dermis_layers: int = 1


def _supersample(seg: Segmentation, frame: HeadFrame, origin, h, nc, sub=2):
    """Fractions of each mask in every grid cell (sub^3 samples per cell)."""
    offs = (np.arange(sub) + 0.5) / sub
    names = ["head", "bone", "teeth", "air_inside", "fat", "fov", "mandible"]
    acc = {k: np.zeros(tuple(nc), np.float32) for k in names}
    cnt = np.zeros(tuple(nc), np.float32)
    gi, gj, gk = np.meshgrid(*[np.arange(n) for n in nc], indexing="ij")
    base = np.stack([gi, gj, gk], -1).reshape(-1, 3).astype(float)
    Tinv = np.linalg.inv(seg.volume.affine)
    H2W = frame.head_to_world
    # planes of the array that contain no data at all (every voxel equal to the padding value) were
    # not measured, even if the padding value equals the air value (CBCT spherical/cylindrical FOV)
    # (detected by the absence of fine-scale variation: a smooth bias correction may have been applied)
    live = _live_planes(seg.volume.data)
    for ox in offs:
        for oy in offs:
            for oz in offs:
                p = origin + (base + [ox, oy, oz]) * h
                w = p @ H2W[:3, :3].T + H2W[:3, 3]
                ijk = np.rint(w @ Tinv[:3, :3].T + Tinv[:3, 3]).astype(int)
                inb = np.all((ijk >= 0) & (ijk < np.array(seg.head.shape)), axis=1)
                ijk = np.clip(ijk, 0, np.array(seg.head.shape) - 1)
                cnt += inb.reshape(tuple(nc))
                alive = live[0][ijk[:, 0]] & live[1][ijk[:, 1]] & live[2][ijk[:, 2]]
                for k in names:
                    m = getattr(seg, k)
                    v = m[ijk[:, 0], ijk[:, 1], ijk[:, 2]] & inb
                    if k == "fov":
                        v = v & alive
                    acc[k] += v.reshape(tuple(nc))
    # fractions over the *measured* sub-samples; out-of-array samples count as unmeasured
    out = {k: v / np.maximum(cnt, 1) for k, v in acc.items()}
    out["fov"] = acc["fov"] / sub ** 3
    return out


def build_tissue_model(seg: Segmentation, frame: HeadFrame, lms, arch, bone_normal_fn,
                       opts: ModelOptions | None = None, surface_fn=None) -> TissueModel:
    """``bone_normal_fn(points_head) -> (n,3)`` outward unit normals of the bone surface.
    ``surface_fn(kind, points_head) -> values`` samples the smoothed 'skin' (head) and 'bone'
    masks (0..1); when given, boundary nodes are snapped onto the true surfaces.

    ``seg`` must carry a ``mandible`` attribute (bool mask)."""
    opts = opts or ModelOptions()
    h = opts.h
    pog_s = lms["soft_pogonion"]
    me = lms["menton"]
    sn = lms["subnasale"]
    lo = np.array([-opts.roi_half_width, pog_s[1] - opts.roi_posterior, me[2] - opts.roi_below_menton])
    hi = np.array([opts.roi_half_width, max(pog_s[1], lms["labrale_inferius"][1], lms["labrale_superius"][1]) + 6.0,
                   sn[2] + opts.roi_above_subnasale])
    nc = np.ceil((hi - lo) / h).astype(int)
    origin = lo
    fr = _supersample(seg, frame, origin, h, nc)

    # ---- field of view: extrapolate unmeasured cells from the nearest measured cell
    measured = fr["fov"] >= 0.5
    cls = np.full(tuple(nc), C_OUT, np.int8)
    tissue_f = fr["head"] - fr["bone"] - fr["air_inside"]
    stack = np.stack([1 - fr["head"], tissue_f, fr["bone"], fr["air_inside"]], 0)
    cls[:] = np.argmax(stack, 0)
    fat = fr["fat"] >= 0.5 * np.maximum(tissue_f, 1e-6)
    teeth = fr["teeth"] >= 0.5
    mand = fr["mandible"] >= 0.5
    extrap = ~measured
    if extrap.any():
        _, inds = ndi.distance_transform_edt(~measured, return_indices=True)
        src = tuple(inds[:, extrap])
        cls[extrap] = cls[src]
        fat[extrap] = fat[src]
        teeth[extrap] = teeth[src]
        mand[extrap] = mand[src]

    # ---- scan cut through the neck: cells outside the scanner field of view (convex hull of
    # the non-padding voxels - tissue fills the FOV cap at the neck) that lie below the chin are
    # filled from the nearest measured cell, i.e. the neck continues instead of ending in a
    # free surface.  Air in front of the face is never touched (restricted to below/behind Me').
    me_s = lms["soft_menton"]
    hull_in = _fov_hull_mask(seg, frame, origin, h, nc)
    centers_z = origin[2] + (np.arange(nc[2]) + 0.5) * h
    centers_y = origin[1] + (np.arange(nc[1]) + 0.5) * h
    below = (centers_z[None, None, :] < me_s[2] - 1.0) & (centers_y[None, :, None] < me_s[1] + 3.0)
    neck_cut = None
    valid = measured & hull_in
    solid = (cls == C_TISSUE) | (cls == C_BONE)
    occ_cols = solid.any(2)
    k_t = np.where(occ_cols, np.argmax(solid, axis=2), -1)          # lowest tissue cell per column
    layer_n = solid.sum((0, 1))
    nz = np.nonzero(layer_n)[0]
    if len(nz):
        k_floor = int(nz[0])
        # (b) abrupt flat floor: a real body surface never ends with a large horizontal cross-section
        flat = layer_n[k_floor] >= 0.35 * layer_n.max()
        ii, jj = np.nonzero(occ_cols)
        kt = k_t[ii, jj]
        below_invalid = np.zeros(len(ii), bool)
        has_below = kt > 0
        below_invalid[has_below] = ~valid[ii[has_below], jj[has_below], kt[has_below] - 1]
        cut_col = below_invalid | (flat & (kt <= k_floor + 1))
        # only under the chin / neck, behind the soft-tissue chin
        cy = origin[1] + (jj + 0.5) * h
        cz = origin[2] + (kt + 0.5) * h
        cut_col &= (cz < me_s[2] + 4.0) & (cy < me_s[1] + 3.0) & (kt > 0)
        for i, j, k in zip(ii[cut_col], jj[cut_col], kt[cut_col]):
            src = cls[i, j, k]
            cls[i, j, :k] = C_TISSUE if src == C_BONE else src
            fat[i, j, :k] = fat[i, j, k]
            mand[i, j, :k] = False
            teeth[i, j, :k] = False
            extrap[i, j, :k] = True
        if cut_col.any():
            neck_cut = float(origin[2] + (kt[cut_col].max() + 0.5) * h)

    # ---- tissue component (face-connected), dermis/mucosa labels
    tissue = cls == C_TISSUE
    lab, n = ndi.label(tissue)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        tissue = lab == np.argmax(sizes)
    outside = cls == C_OUT
    cavity = cls == C_CAVITY
    # cells removed from the tissue component become "outside" for the purpose of surfaces
    outside |= (cls == C_TISSUE) & ~tissue
    struct = ndi.generate_binary_structure(3, 1)
    dermis = tissue & ndi.binary_dilation(outside, struct, iterations=opts.dermis_layers)
    mucosa = tissue & ndi.binary_dilation(cavity, struct) & ~dermis
    mat = np.full(tuple(nc), MAT_MUSCLE, np.int8)
    if seg.volume.modality == "CT":
        mat[fat] = MAT_FAT
    else:
        # CBCT grey values are not reliable enough to separate fat from muscle (scatter, cupping):
        # use the anatomical layering instead - subcutaneous fat band under the dermis, muscle /
        # glandular tissue deeper - and only keep CT-derived fat where it lies in that band.
        depth = ndi.distance_transform_edt(~outside) * h
        band = (depth > 1.5) & (depth <= 8.0)
        mat[tissue & band] = MAT_FAT
        mat[tissue & ~band & fat & (depth <= 12.0)] = MAT_FAT
    mat[mucosa] = MAT_MUCOSA
    mat[dermis] = MAT_SKIN

    elem_ijk = np.argwhere(tissue)
    node_grid = -np.ones(tuple(nc + 1), np.int64)
    corner = LOCAL_OFF = np.array([[a & 1, (a >> 1) & 1, (a >> 2) & 1] for a in range(8)])
    all_nodes = (elem_ijk[:, None, :] + corner[None]).reshape(-1, 3)
    uniq = np.unique(all_nodes, axis=0)
    node_grid[tuple(uniq.T)] = np.arange(len(uniq))
    elems = np.stack([node_grid[tuple((elem_ijk + o).T)] for o in LOCAL_OFF], 1)
    nodes = origin + uniq * h
    node_ijk = uniq

    # ---- node classification via neighbour cells
    def touches(mask):
        """nodes that are a corner of at least one cell in mask"""
        hit = np.zeros(tuple(nc + 1), bool)
        for o in LOCAL_OFF:
            sl_dst = tuple(slice(o[d], o[d] + nc[d]) for d in range(3))
            hit[sl_dst] |= mask
        return hit[tuple(node_ijk.T)]

    bone_cells = cls == C_BONE
    t_bone = touches(bone_cells)
    t_teeth = touches(bone_cells & teeth)
    t_mand = touches(bone_cells & mand)
    t_out = touches(outside)
    t_cav = touches(cavity)
    boundary_box = np.any((node_ijk == 0) | (node_ijk == nc), axis=1)

    snap_info = {}
    if surface_fn is not None:
        node_grid_full = -np.ones(tuple(nc + 1), np.int64)
        node_grid_full[tuple(node_ijk.T)] = np.arange(len(node_ijk))
        nodes, snap_info = _snap_boundary(nodes, node_ijk, node_grid_full, elems, h,
                                          skin_idx=np.where(t_out & ~t_bone & ~boundary_box)[0],
                                          bone_idx=np.where(t_bone & ~t_out & ~boundary_box)[0],
                                          fixed_idx=np.where(boundary_box)[0], surface_fn=surface_fn)
    # outward bone normals on the interface
    iface = np.where(t_bone)[0]
    nrm = bone_normal_fn(nodes[iface])
    # arch coordinates
    u, v = arch.uv(nodes)
    r_node = np.hypot(nodes[:, 0], nodes[:, 1] - arch.center_y)
    r_out = np.interp(arch.angle(nodes), arch.theta, arch.r_outer)
    # external (labial / inferior) cortex only: the implant never lies on the lingual side
    all_n = np.zeros((len(nodes), 3))
    all_n[iface] = nrm
    th_n = arch.angle(nodes)
    radial = np.stack([np.sin(th_n), np.cos(th_n), np.zeros_like(th_n)], 1)
    external = (np.einsum("ij,ij->i", all_n, radial) > 0.2) | (all_n[:, 2] < -0.5)
    pocket_mask = (t_mand & ~t_teeth & external & np.isfinite(u) & (np.abs(np.nan_to_num(u, nan=1e9)) < opts.pocket_half_arc)
                   & (v < opts.pocket_top) & (v > -opts.pocket_under) & (r_node > r_out - 12.0))
    # never let the pocket reach the alveolar bone around the roots
    if "anterior_root_apex" in lms:
        pocket_mask &= nodes[:, 2] < lms["anterior_root_apex"][2] - 1.0
    # lips rest on the crowns: bone-interface nodes in the dental band in front of the arch slide,
    # whether or not every crown was segmented as tooth (CBCT dentin vs cortex is ambiguous and
    # brackets dominate the bright end of the histogram)
    sto = lms["stomion"]
    r_arch = np.hypot(nodes[:, 0], nodes[:, 1] - arch.center_y)
    r_lab = np.interp(arch.angle(nodes), arch.theta, arch.r_outer)
    crown_band = (np.abs(nodes[:, 2] - sto[2]) < 12.0) & (r_arch > r_lab - 10.0) & (np.abs(nodes[:, 0]) < 35)
    teeth_slide = (t_teeth | (t_bone & crown_band)) & ~boundary_box
    pocket = pocket_mask & t_bone & ~boundary_box
    fixed = (t_bone & ~pocket & ~teeth_slide) | boundary_box
    skin = t_out & ~boundary_box

    sets = {
        "bone_iface": iface,
        "pocket": np.where(pocket)[0],
        "teeth_slide": np.where(teeth_slide)[0],
        "fixed": np.where(fixed)[0],
        "skin": np.where(skin)[0],
        "cavity": np.where(t_cav)[0],
        "mandible_iface": np.where(t_mand & t_bone)[0],
        "external_iface": np.where(t_mand & t_bone & external & ~t_teeth & (r_node > r_out - 12.0))[0],
        "boundary_box": np.where(boundary_box)[0],
    }
    nmap = -np.ones(len(nodes), int)
    nmap[iface] = np.arange(len(iface))
    normals = {k: nrm[nmap[sets[k]]] for k in ("pocket", "teeth_slide", "mandible_iface")}

    model = TissueModel(h, origin, tuple(nc + 1), nodes, node_ijk, elems, elem_ijk,
                        mat[tuple(elem_ijk.T)], extrap[tuple(elem_ijk.T)], sets, normals,
                        cell_class=cls)
    cu, cv = arch.uv(model.elem_centers())
    model.elem_uv = np.stack([cu, cv], 1)
    _split_oral_fissure(model, lms, opts, tissue, outside, cavity, bone_cells)
    model.info.update({
        "n_nodes": int(model.n_nodes), "n_elems": int(len(model.elems)),
        "n_extrapolated_elems": int(model.extrapolated.sum()),
        "material_counts": {MAT_NAMES[i]: int((model.mat_id == i).sum()) for i in range(4)},
        "set_sizes": {k: int(len(v)) for k, v in model.sets.items()},
        "roi_lo": lo.tolist(), "roi_hi": hi.tolist(), "snap": snap_info, "neck_cut_z": neck_cut,
        "min_jacobian_quality": float(ElementGeom.jacobian_quality(model.nodes, model.elems).min()),
    })
    return model


def _split_oral_fissure(model: TissueModel, lms, opts: ModelOptions, tissue, outside, cavity, bone_cells):
    """Duplicate the nodes on the oral fissure so the lips can separate and slide."""
    h = model.h
    sto = lms["stomion"]
    k_cut = int(round((sto[2] - model.origin[2]) / h))
    if k_cut <= 0 or k_cut >= model.shape[2] - 1:
        model.info["fissure"] = "stomion outside ROI"
        return
    hw = opts.mouth_half_width
    if hw is None:
        # intercommissural width ~ 0.36-0.40 x bigonial width; fall back to 25 mm
        if "gonion_left" in lms and "gonion_right" in lms:
            hw = 0.5 * 0.40 * abs(lms["gonion_right"][0] - lms["gonion_left"][0])
            hw = float(np.clip(hw, 20.0, 30.0))
        else:
            hw = 25.0
    y_teeth = lms["lower_incisor_tip"][1] - 2.0 if "lower_incisor_tip" in lms else sto[1] - 15.0
    nc = np.array(model.shape) - 1
    # cut faces between cell layers k_cut-1 (below) and k_cut (above)
    below = tissue[:, :, k_cut - 1]
    above = tissue[:, :, k_cut]
    xi = model.origin[0] + (np.arange(nc[0]) + 0.5) * h
    yj = model.origin[1] + (np.arange(nc[1]) + 0.5) * h
    X, Y = np.meshgrid(xi, yj, indexing="ij")
    # the lips curve back around the arch: depth limit follows the teeth arch approx.
    face = below & above & (np.abs(X) < hw) & (Y > y_teeth - 0.012 * X ** 2)
    if face.sum() == 0:
        model.info["fissure"] = "no lip contact cells"
        return
    # nodes of the cut faces strictly inside the commissures
    node_grid = -np.ones(tuple(model.shape), np.int64)
    node_grid[tuple(model.node_ijk.T)] = np.arange(model.n_nodes)
    fi = np.argwhere(face)
    cand = set()
    for dx in (0, 1):
        for dy in (0, 1):
            ii = fi[:, 0] + dx
            jj = fi[:, 1] + dy
            for a, b in zip(ii, jj):
                cand.add((a, b))
    cand = np.array(sorted(cand))
    xn = model.origin[0] + cand[:, 0] * h
    cand = cand[np.abs(xn) < hw - 0.5 * h]
    ids = node_grid[cand[:, 0], cand[:, 1], k_cut]
    ok = ids >= 0
    cand, ids = cand[ok], ids[ok]
    # only nodes that border at least one cut face (not just the corner of a lip cell)
    n0 = model.n_nodes
    dup = np.arange(n0, n0 + len(ids))
    model.nodes = np.vstack([model.nodes, model.nodes[ids]])
    model.node_ijk = np.vstack([model.node_ijk, model.node_ijk[ids]])
    # elements in layer k_cut-1 (below the cut) whose top nodes are duplicated -> remap
    remap = dict(zip(ids.tolist(), dup.tolist()))
    below_e = np.where(model.elem_ijk[:, 2] == k_cut - 1)[0]
    E = model.elems
    for a in range(4, 8):   # top nodes of a cell (dz = 1)
        col = E[below_e, a]
        hit = np.isin(col, ids)
        if hit.any():
            E[below_e[hit], a] = [remap[c] for c in col[hit]]
    model.elems = E
    model.pairs = np.stack([ids, dup], 1)
    # sets: duplicated nodes inherit membership of skin / teeth / cavity / fixed
    for k in ("skin", "cavity", "teeth_slide", "fixed", "boundary_box"):
        s = model.sets[k]
        extra = dup[np.isin(ids, s)]
        model.sets[k] = np.r_[s, extra]
    if len(model.normals.get("teeth_slide", [])):
        s = model.sets["teeth_slide"]
        base = s[: len(s) - int(np.isin(ids, s[: len(s)]).sum())]
    # recompute teeth normals for duplicated entries by copying
    ts = model.sets["teeth_slide"]
    nmap = {int(n): i for i, n in enumerate(ts[: len(model.normals["teeth_slide"])])}
    add = [model.normals["teeth_slide"][nmap[int(o)]] for o in ids if int(o) in nmap]
    if add:
        model.normals["teeth_slide"] = np.vstack([model.normals["teeth_slide"], np.array(add)])
    model._asm = None
    model.info["fissure"] = {"k": k_cut, "z": float(model.origin[2] + k_cut * h), "half_width_mm": hw,
                             "pairs": int(len(ids))}


def _snap_boundary(nodes, node_ijk, node_grid, elems, h, skin_idx, bone_idx, fixed_idx, surface_fn,
                   max_snap_frac: float = 0.75, min_quality: float = 0.2):
    """Boundary-fit the voxel mesh: project skin / bone-interface nodes onto the 0.5
    iso-surfaces of the smoothed masks, spread the motion into the interior by
    Laplacian relaxation, and back off wherever element quality would drop
    below ``min_quality`` (det J relative to a cube)."""
    X0 = nodes.copy()
    disp = np.zeros_like(nodes)
    eps = 0.25
    for idx, kind in ((skin_idx, "skin"), (bone_idx, "bone")):
        if len(idx) == 0:
            continue
        p = X0[idx].copy()
        for _ in range(5):
            f = surface_fn(kind, p) - 0.5
            g = np.stack([(surface_fn(kind, p + eps * e) - surface_fn(kind, p - eps * e)) / (2 * eps)
                          for e in np.eye(3)], 1)
            gn2 = np.maximum((g * g).sum(1), 1e-6)
            step = -(f / gn2)[:, None] * g
            sl = np.linalg.norm(step, axis=1, keepdims=True)
            step *= np.minimum(1.0, 0.5 * h / np.maximum(sl, 1e-9))
            p += step
        d = p - X0[idx]
        dl = np.linalg.norm(d, axis=1, keepdims=True)
        d *= np.minimum(1.0, max_snap_frac * h / np.maximum(dl, 1e-9))
        disp[idx] = d
    boundary = np.zeros(len(nodes), bool)
    boundary[skin_idx] = True
    boundary[bone_idx] = True
    boundary[fixed_idx] = True
    # interior relaxation: Jacobi smoothing of the displacement over grid neighbours
    nbrs = []
    for e in np.vstack([np.eye(3, dtype=int), -np.eye(3, dtype=int)]):
        q = node_ijk + e
        ok = np.all((q >= 0) & (q < np.array(node_grid.shape)), axis=1)
        j = -np.ones(len(nodes), np.int64)
        j[ok] = node_grid[tuple(q[ok].T)]
        nbrs.append(j)
    nbrs = np.stack(nbrs, 1)
    interior = np.where(~boundary)[0]
    for _ in range(6):
        nb = nbrs[interior]
        valid = nb >= 0
        acc = np.where(valid[..., None], disp[np.maximum(nb, 0)], 0).sum(1)
        cnt = np.maximum(valid.sum(1), 1)
        disp[interior] = 0.6 * acc / cnt[:, None]
    # quality guard
    scale = np.ones(len(nodes))
    for _ in range(8):
        X = X0 + disp * scale[:, None]
        q = ElementGeom.jacobian_quality(X, elems)
        bad = q < min_quality
        if not bad.any():
            break
        bn = np.unique(elems[bad])
        scale[bn] *= 0.5
    else:
        X = X0 + disp * scale[:, None]
        q = ElementGeom.jacobian_quality(X, elems)
        bad = q < 0.02
        scale[np.unique(elems[bad])] = 0
        X = X0 + disp * scale[:, None]
    moved = np.linalg.norm(disp * scale[:, None], axis=1)
    info = {"skin_nodes": int(len(skin_idx)), "bone_nodes": int(len(bone_idx)),
            "mean_snap_mm": float(moved[boundary].mean()) if boundary.any() else 0.0,
            "max_snap_mm": float(moved.max()), "backed_off_nodes": int((scale < 1).sum())}
    return X, info


def _live_planes(d: np.ndarray, min_sd: float = 2.0) -> list:
    """Per axis, which array planes contain measured data: the standard deviation of neighbouring-voxel
    differences inside the plane exceeds ``min_sd`` (measured slices always carry noise / anatomy)."""
    out = []
    for ax in range(3):
        inplane = [a for a in range(3) if a != ax][0]
        sd = np.empty(d.shape[ax], np.float32)
        for i in range(d.shape[ax]):
            plane = np.take(d, i, axis=ax)
            sd[i] = np.std(np.diff(plane, axis=0 if inplane < ax else 0))
        out.append(sd > min_sd)
    return out


def _fov_hull_mask(seg: Segmentation, frame: HeadFrame, origin, h, nc, step: int = 3) -> np.ndarray:
    """Cells (centres) inside the convex hull of all non-padding voxels."""
    from scipy.spatial import ConvexHull
    d = seg.volume.data
    pad = float(d.min())
    sub = d[::step, ::step, ::step]
    idx = np.argwhere(sub > pad + 1e-3) * step
    if len(idx) < 100:
        return np.ones(tuple(nc), bool)
    # boundary voxels suffice for the hull
    hull = ConvexHull(seg.volume.voxel_to_world(idx[np.random.default_rng(0).choice(len(idx), min(len(idx), 400000),
                                                                                    replace=False)]))
    gi, gj, gk = np.meshgrid(*[np.arange(n) for n in nc], indexing="ij")
    c = origin + (np.stack([gi, gj, gk], -1).reshape(-1, 3) + 0.5) * h
    w = frame.to_world(c)
    inside = np.all(w @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1.0, axis=1)
    return inside.reshape(tuple(nc))
