"""Planning from an optical face scan alone (no CT / CBCT).

Without radiological data the bone and the soft-tissue layers are not
measured, so they are *estimated* by transferring the anatomy of a template
head (a processed CT/CBCT case) onto the patient's face:

1. The scan is put into a head frame (scale detection, reflective-symmetry
   plane, anterior = mean surface normal, superior = scanner up axis).
2. Soft-tissue mid-line landmarks are detected on the scan's mid-sagittal
   section with the same algorithm as for CT.
3. A thin-plate-spline deformation maps the template head frame onto the
   patient: correspondences are the soft landmarks, dense surface points
   (non-rigid ICP between template skin and scan) and *bone targets* that
   impose population soft-tissue thickness norms at pogonion, B point and
   menton (sex specific).
4. The template segmentation is resampled through the inverse deformation to
   create a synthetic segmentation in the patient frame; meshes, mandible,
   nerves and roots are warped forward.  The regular FE pipeline then runs
   unchanged.

All estimated structures carry reduced confidence and the prediction engine
widens its uncertainty (``anatomy_source = "estimated"``).  A CT/CBCT always
supersedes this path.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from .anatomy.frame import HeadFrame, fit_symmetry_plane
from .anatomy.landmarks import LandmarkSet, mesh_midline_contour, soft_landmarks_from_contour
from .imaging.calibrate import Calibration
from .imaging.segment import Segmentation
from .imaging.surface import Mesh, decimate, largest_connected_submesh, taubin_smooth
from .io.volume import Volume

# soft-tissue thickness norms (mm, mean, sd) for adults; Holdaway / Utsuno-type cephalometric data
THICKNESS_NORMS = {
    "M": {"pogonion": (12.5, 2.0), "b_point": (12.0, 2.0), "menton": (8.0, 1.8)},
    "F": {"pogonion": (11.5, 2.0), "b_point": (11.0, 1.8), "menton": (7.3, 1.6)},
}
SOFT_KEYS = ("soft_glabella", "soft_nasion", "pronasale", "subnasale", "labrale_superius", "stomion",
             "labrale_inferius", "soft_b_point", "soft_pogonion", "soft_gnathion", "soft_menton")


def scan_frame(v: np.ndarray, n: np.ndarray, up_hint=(0, 1, 0)) -> HeadFrame:
    """Head frame (RAS) for a face scan given its vertices/normals (mm).

    The scanner's up axis is trusted (all common face scanners export Y-up; override with
    ``scan_up``); anterior is the mean surface normal with its vertical part removed, because
    submental and neck surfaces tilt the raw mean normal downwards."""
    up = np.asarray(up_hint, float)
    up /= np.linalg.norm(up)
    ant = n.mean(0)
    ant = ant - (ant @ up) * up
    ant /= np.linalg.norm(ant)
    lat0 = np.cross(up, ant)      # patient left in RAS terms (S x A = -R ... see below)
    c = v.mean(0)
    nrm, d, rms = fit_symmetry_plane(v[np.random.default_rng(0).choice(len(v), min(40000, len(v)), replace=False)],
                                     lat0, float(c @ lat0))
    if nrm @ lat0 < 0:
        nrm, d = -nrm, -d
    right = -nrm                                  # R = -(S x A) since A = S x R
    sup = up - (up @ right) * right
    sup /= np.linalg.norm(sup)
    ant = np.cross(sup, right)
    R = np.stack([right, ant, sup])
    origin = c - ((c @ nrm) - d) * nrm
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ origin
    return HeadFrame(T, rms, 0.0, {"source": "face scan"})


def _nonrigid_pairs(src: np.ndarray, dst: np.ndarray, fixed_src, fixed_dst, iters=5, n_pts=500):
    """Dense correspondences template -> scan.

    Both surfaces are already in head frames sharing the mid-sagittal plane, so an isotropic
    scale + translation from the (coplanar) mid-line landmarks initialises the alignment; closest
    points over the whole face then provide the bilateral correspondences the thin-plate spline
    needs, refined by alternating TPS fitting and closest-point matching (non-rigid ICP)."""
    rng = np.random.default_rng(1)
    pick = src[rng.choice(len(src), min(n_pts, len(src)), replace=False)]
    tree = cKDTree(dst)
    cs, cd = fixed_src.mean(0), fixed_dst.mean(0)
    scale = np.sqrt(((fixed_dst - cd) ** 2).sum() / max(((fixed_src - cs) ** 2).sum(), 1e-9))
    warp = lambda p: (p - cs) * scale + cd
    for it in range(iters):
        moved = warp(pick)
        dist, j = tree.query(moved)
        ok = dist < (12 if it == 0 else 6)
        S = np.vstack([fixed_src, pick[ok]])
        D = np.vstack([fixed_dst, dst[j[ok]]])
        f = RBFInterpolator(S, D - S, kernel="thin_plate_spline", smoothing=20.0 / (it + 1))
        warp = lambda p, f=f: p + f(p)
    return S, D


def build_from_scan(case, template, log=print):
    """Populate ``case`` (which has only a surface input) using ``template`` (a processed CT case)."""
    from .case import Anatomy, BoneField, save_mesh_dict, save_model, split_teeth_meshes
    from .implant.design import build_arch_param
    from .io.meshio import read_surface
    from .surface_scan import transfer_colors
    from .tissue.model import build_tissue_model

    t0 = time.time()
    case.set_status("loading", 0.05, "reading face scan")
    scan, uv, tex, cols = read_surface(case.root / "input" / case.meta["surface_input"])
    if np.ptp(scan.vertices, axis=0).max() < 2.0:
        scan.vertices = scan.vertices * 1000.0
    scan = largest_connected_submesh(scan)
    up = case.meta.get("scan_up", [0, 1, 0])
    fr = scan_frame(scan.vertices, scan.vertex_normals(), up)
    skin_h = scan.transformed(fr.world_to_head)
    skin_h = taubin_smooth(decimate(skin_h, 300_000), 5)

    case.set_status("landmarks", 0.15, "detecting soft-tissue landmarks on the scan")
    c = mesh_midline_contour(skin_h.vertices, skin_h.faces)
    lms = LandmarkSet()
    soft_landmarks_from_contour(lms, c, np.ones(len(c), bool), skin_h.vertices.min(0))
    del lms._contour_state

    case.set_status("template", 0.25, "estimating bone from the template head")
    ta = template.anatomy
    tl = ta.landmarks
    common = [k for k in SOFT_KEYS if k in lms and k in tl and tl.points[k].confidence >= 0.3
              and lms.points[k].confidence >= 0.3]
    src_fixed = np.array([tl[k] for k in common])
    dst_fixed = np.array([lms[k] for k in common])
    # bone targets from population thickness norms (patient-specific soft surface normal)
    sex = (case.meta.get("sex") or "F").upper()[:1]
    norms = THICKNESS_NORMS.get(sex, THICKNESS_NORMS["F"])
    skin_n = skin_h.vertex_normals()
    tree_s = cKDTree(skin_h.vertices)
    bone_src, bone_dst = [], []
    for soft, bone, key in (("soft_pogonion", "pogonion", "pogonion"), ("soft_b_point", "b_point", "b_point"),
                            ("soft_menton", "menton", "menton")):
        if soft in lms and bone in tl:
            n = skin_n[tree_s.query(lms[soft])[1]]
            bone_src.append(tl[bone])
            bone_dst.append(lms[soft] - norms[key][0] * n)
    tmpl_skin = template.meshes["skin"].vertices
    face_t = tmpl_skin[(tmpl_skin[:, 1] > tl["soft_pogonion"][1] - 45) & (tmpl_skin[:, 2] < tl["soft_nasion"][2] + 10)]
    S, D = _nonrigid_pairs(face_t, skin_h.vertices, np.vstack([src_fixed, bone_src]), np.vstack([dst_fixed, bone_dst]))
    fwd = RBFInterpolator(S, D - S, kernel="thin_plate_spline", smoothing=5.0)
    inv = RBFInterpolator(D, S - D, kernel="thin_plate_spline", smoothing=5.0)
    warp = lambda p: p + fwd(p)
    unwarp = lambda p: p + inv(p)

    # ---- synthetic segmentation on a patient grid (head frame, 0.6 mm) around the lower 2/3 face
    case.set_status("segmentation", 0.4, "building the estimated tissue volume")
    pog = lms["soft_pogonion"]
    lo = np.array([-85, pog[1] - 150, pog[2] - 50])
    hi = np.array([85, skin_h.vertices[:, 1].max() + 4, lms["soft_nasion"][2] + 30])
    h = 0.6
    shape = tuple(np.ceil((hi - lo) / h).astype(int) + 1)
    aff = np.eye(4)
    aff[:3, :3] = np.diag([h, h, h])
    aff[:3, 3] = lo
    tseg = template.seg
    tfr = ta.frame
    gi = [np.arange(n) for n in shape]
    masks = {k: np.zeros(shape, bool) for k in ("head", "bone", "teeth", "air_inside", "fat", "mandible")}
    hu = np.full(shape, -1000.0, np.float32)
    # inverse warp evaluated on a 4x coarser grid and interpolated (the TPS is smooth)
    cs_ = 4
    cshape = tuple((np.array(shape) - 1) // cs_ + 2)
    cg = np.stack(np.meshgrid(*[lo[i] + np.arange(cshape[i]) * h * cs_ for i in range(3)], indexing="ij"), -1)
    cdisp = (unwarp(cg.reshape(-1, 3)) - cg.reshape(-1, 3)).reshape(cshape + (3,))
    for k in range(shape[2]):
        I, J = np.meshgrid(gi[0], gi[1], indexing="ij")
        P = np.stack([lo[0] + I * h, lo[1] + J * h, np.full(I.shape, lo[2] + k * h)], -1).reshape(-1, 3)
        cc = np.stack([I.ravel() / cs_, J.ravel() / cs_, np.full(I.size, k / cs_)])
        Q = P + np.stack([ndi.map_coordinates(cdisp[..., c], cc, order=1, mode="nearest") for c in range(3)], 1)
        ijk = np.rint(tseg.volume.world_to_voxel(tfr.to_world(Q))).astype(int)
        inb = np.all((ijk >= 0) & (ijk < np.array(tseg.head.shape)), axis=1)
        ijk = np.clip(ijk, 0, np.array(tseg.head.shape) - 1)
        for name in masks:
            m = getattr(tseg, name)
            masks[name][:, :, k] = (m[ijk[:, 0], ijk[:, 1], ijk[:, 2]] & inb).reshape(I.shape)
        hu[:, :, k] = np.where(inb, tseg.volume.data[ijk[:, 0], ijk[:, 1], ijk[:, 2]], -1000).reshape(I.shape)
    # the outer surface is the patient's own: inside = behind the scan along the anterior axis
    depth = _front_depth(skin_h, lo, h, shape)
    J = lo[1] + np.arange(shape[1]) * h
    behind = J[None, :, None] < depth[:, None, :]
    covered = np.isfinite(depth)[:, None, :]
    head = np.where(covered, behind & (masks["head"] | (J[None, :, None] > depth[:, None, :] - 25)), masks["head"])
    head = ndi.binary_fill_holes(head)
    masks["head"] = head
    for name in ("bone", "teeth", "air_inside", "fat", "mandible"):
        masks[name] &= head
    vol = Volume(hu, aff, "CT", {"source": "template transfer"})
    seg = Segmentation(vol, Calibration(1, 0, -1000, 40, True), np.ones(shape, bool), head, masks["air_inside"],
                       masks["bone"], masks["teeth"], np.zeros(shape, bool), masks["fat"],
                       {"source": "estimated from template"}, masks["mandible"])
    frame = HeadFrame(np.eye(4), fr.symmetry_rms_mm, 0.0, {"source": "face scan"})

    # ---- bony landmarks / meshes warped from the template
    for k, lm in tl.points.items():
        if k not in lms:
            lms.add(k, warp(lm.position[None])[0], min(lm.confidence, 0.2), "template estimate")
    lms.notes.append("Bone, teeth, nerve and root positions are ESTIMATED from a template head and population "
                     "soft-tissue thickness norms - obtain a CT/CBCT before surgery.")
    meshes = {"skin": skin_h}
    for k in ("bone", "mandible", "cranium", "teeth_lower", "teeth_upper"):
        if k in template.meshes:
            m = template.meshes[k]
            meshes[k] = Mesh(warp(m.vertices), m.faces.copy())
    if tex is not None or cols is not None:
        T = np.linalg.inv(fr.world_to_head)
        rgb, mask, _ = transfer_colors(skin_h, scan.transformed(fr.world_to_head), np.eye(4), uv, tex, cols)
        for i, ch in enumerate("rgb"):
            skin_h.attributes[f"albedo_{ch}"] = rgb[:, i]
        skin_h.attributes["albedo_mask"] = mask
    arch = build_arch_param(meshes["mandible"].vertices, lms)
    th_pog = float(np.linalg.norm(lms["soft_pogonion"] - lms["pogonion"]))
    patient = {"soft_tissue_thickness_pog_mm": th_pog,
               "soft_tissue_thickness_menton_mm": float(np.linalg.norm(lms["soft_menton"] - lms["menton"])),
               "thickness_norm_used": norms, "sex": sex, "template_case": template.id}
    anatomy = Anatomy(frame, lms, arch, patient)
    case.set_status("model", 0.6, "building the soft-tissue finite-element model")
    case._seg, case._anatomy, case._meshes = seg, anatomy, meshes
    bf = BoneField(seg, frame)
    model = build_tissue_model(seg, frame, lms, arch, bf.normals, None, surface_fn=bf.surface)
    case._model, case._bonefield = model, bf
    case._save_seg(seg)
    from .case import _json_default
    (case.cache / "anatomy.json").write_text(json.dumps(anatomy.to_dict(), default=_json_default))
    np.savez_compressed(case.cache / "meshes.npz", **save_mesh_dict(meshes))
    save_model(case.cache / "model.npz", model)
    case.meta.update({"anatomy_source": "estimated", "template_case": template.id, "skin_smoothed": True,
                      "model_info": model.info, "processing_seconds": round(time.time() - t0, 1)})
    case.set_status("ready", 1.0, "ready (anatomy estimated from face scan)")
    log(f"scan-only case {case.id} built in {time.time() - t0:.0f}s")


def _front_depth(skin: Mesh, lo, h, shape) -> np.ndarray:
    """Most anterior skin y for every (x, z) grid column (NaN where the scan has no data)."""
    v = skin.vertices
    f = skin.faces
    # dense surface samples so every column is hit
    from .render.preview import surface_samples
    pts, _, _ = surface_samples(v, f, None, density=3.0)
    i = np.rint((pts[:, 0] - lo[0]) / h).astype(int)
    k = np.rint((pts[:, 2] - lo[2]) / h).astype(int)
    ok = (i >= 0) & (i < shape[0]) & (k >= 0) & (k < shape[2])
    depth = np.full((shape[0], shape[2]), -np.inf)
    np.maximum.at(depth, (i[ok], k[ok]), pts[ok, 1])
    depth[np.isinf(depth)] = np.nan
    # close pinholes
    filled = ndi.generic_filter(np.nan_to_num(depth, nan=-1e9), np.max, size=3)
    depth = np.where(np.isnan(depth) & (filled > -1e8), filled, depth)
    return depth
