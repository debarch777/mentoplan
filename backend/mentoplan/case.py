"""Patient case: ingest, processing pipeline and on-disk cache.

Directory layout (``<root>/<case_id>/``)::

    case.json            metadata + processing status
    input/               original uploads
    cache/seg.npz        segmentation masks (bit-packed) + pseudo-HU volume
    cache/anatomy.json   head frame, landmarks, arch parameterisation
    cache/meshes.npz     skin / bone / mandible / teeth surfaces (head frame)
    cache/model.npz      soft-tissue FE model
    variants.json        implant design variants
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .anatomy.frame import HeadFrame, estimate_head_frame
from .anatomy.landmarks import (LandmarkSet, detect_bilateral, detect_bony_midline, detect_landmarks,
                                separate_mandible)
from .imaging.calibrate import Calibration
from .imaging.segment import Segmentation, segment_head
from .imaging.surface import Mesh, decimate, largest_connected_submesh, mask_to_mesh, skin_surface, taubin_smooth
from .implant.design import ArchParam, build_arch_param
from .io.volume import Volume, read_volume
from .tissue.model import ModelOptions, TissueModel, build_tissue_model

PIPELINE_VERSION = 4


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


@dataclass
class Anatomy:
    frame: HeadFrame
    landmarks: LandmarkSet
    arch: ArchParam
    patient: dict = field(default_factory=dict)

    def to_dict(self):
        return {"frame": self.frame.to_dict(), "landmarks": self.landmarks.to_dict(), "arch": self.arch.to_dict(),
                "patient": self.patient}

    @staticmethod
    def from_dict(d):
        return Anatomy(HeadFrame.from_dict(d["frame"]), LandmarkSet.from_dict(d["landmarks"]),
                       ArchParam.from_dict(d["arch"]), d.get("patient", {}))


class BoneField:
    """Smoothed bone / mandible masks sampled in head coordinates (normals, implant fit)."""

    def __init__(self, seg: Segmentation, frame: HeadFrame):
        self.vol = seg.volume
        self.frame = frame
        self.bone = ndi.gaussian_filter(seg.bone.astype(np.float32), 1.0)
        self._surf = {}
        self._seg = seg
        self.mand = ndi.gaussian_filter(seg.mandible.astype(np.float32), 0.8) if seg.mandible is not None else None

    def _ijk(self, p_head):
        return self.vol.world_to_voxel(self.frame.to_world(p_head)).T

    def surface(self, kind: str, p_head):
        """Smoothed (0.6 mm) skin / bone masks for boundary fitting of the FE mesh."""
        if kind not in self._surf:
            m = self._seg.head if kind == "skin" else self._seg.bone
            self._surf[kind] = ndi.gaussian_filter(m.astype(np.float32), 0.6 / self.vol.spacing[0])
        return ndi.map_coordinates(self._surf[kind], self._ijk(p_head), order=1, mode="nearest")

    def sample_mandible(self, p_head):
        return ndi.map_coordinates(self.mand, self._ijk(p_head), order=1, mode="constant")

    def normals(self, p_head):
        """Outward unit bone normals at the given points: central differences of the smoothed bone
        field sampled locally (no full-volume gradient arrays)."""
        p_head = np.asarray(p_head, float)
        eps = 0.5
        g = np.zeros_like(p_head)
        for a in range(3):
            d = np.zeros(3)
            d[a] = eps
            fp = ndi.map_coordinates(self.bone, self._ijk(p_head + d), order=1, mode="nearest")
            fm = ndi.map_coordinates(self.bone, self._ijk(p_head - d), order=1, mode="nearest")
            g[:, a] = (fp - fm) / (2 * eps)
        return -g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)

    def release_segmentation(self):
        """Drop references to the full segmentation (only the smoothed fields are kept)."""
        self._seg = None
        self._surf = {}


def save_mesh_dict(meshes: dict[str, Mesh]) -> dict:
    out = {}
    for k, m in meshes.items():
        out[f"{k}__v"] = m.vertices.astype(np.float32)
        out[f"{k}__f"] = m.faces.astype(np.int32)
        for a, arr in m.attributes.items():
            out[f"{k}__a__{a}"] = arr
    return out


def load_mesh_dict(z) -> dict[str, Mesh]:
    meshes = {}
    for key in z.files:
        if key.endswith("__v"):
            name = key[:-3]
            attrs = {k.split("__a__")[1]: z[k] for k in z.files if k.startswith(name + "__a__")}
            meshes[name] = Mesh(z[key].astype(np.float64), z[name + "__f"], attrs)
    return meshes


def save_model(path: Path, m: TissueModel):
    np.savez_compressed(path, h=m.h, origin=m.origin, shape=np.array(m.shape), nodes=m.nodes, node_ijk=m.node_ijk,
                        elems=m.elems, elem_ijk=m.elem_ijk, mat_id=m.mat_id, extrapolated=m.extrapolated,
                        pairs=m.pairs, elem_uv=m.elem_uv,
                        **{f"set__{k}": v for k, v in m.sets.items()},
                        **{f"nrm__{k}": v for k, v in m.normals.items()},
                        info=json.dumps(m.info, default=_json_default))


def load_model(path: Path) -> TissueModel:
    z = np.load(path, allow_pickle=False)
    sets = {k[5:]: z[k] for k in z.files if k.startswith("set__")}
    normals = {k[5:]: z[k] for k in z.files if k.startswith("nrm__")}
    return TissueModel(float(z["h"]), z["origin"], tuple(int(v) for v in z["shape"]), z["nodes"], z["node_ijk"],
                       z["elems"], z["elem_ijk"], z["mat_id"], z["extrapolated"], sets, normals, z["pairs"],
                       None, z["elem_uv"], json.loads(str(z["info"])))


def teeth_render_mask(seg: Segmentation) -> np.ndarray:
    """Dentition for display. ``seg.teeth`` keeps only the densest enamel (robust for the mechanics),
    which on noisy CBCT and around brackets leaves fragmented crowns. Here the enamel seeds grow by up to
    1.5 mm into tooth-density voxels (dentin, > bone threshold), orthodontic metal touching the teeth is
    kept (it is really there), and holes < 1 mm (metal streaks) are closed."""
    sp = seg.volume.spacing
    if seg.teeth.sum() < 100:
        return seg.teeth
    smooth = ndi.gaussian_filter(seg.volume.data.astype(np.float32), 0.5 / sp[0])
    near = ndi.distance_transform_edt(~seg.teeth, sampling=sp) <= 1.5
    grown = seg.teeth | (near & (smooth > seg.thresholds.get("bone_hu", 400.0)) & seg.head)
    del smooth, near
    metal = seg.metal & (ndi.distance_transform_edt(~grown, sampling=sp) <= 1.0)
    grown = grown | metal
    r = max(1, int(round(0.5 / sp[0])))
    ball = ndi.generate_binary_structure(3, 1)
    closed = ndi.binary_erosion(ndi.binary_dilation(grown, ball, iterations=r), ball, iterations=r)
    return grown | (closed & seg.head)


def split_teeth_meshes(seg: Segmentation, frame: HeadFrame) -> dict:
    """Upper (fixed) and lower (move with the jaw) dentition, head frame."""
    out = {}
    if seg.teeth.sum() < 100:
        return out
    # the watershed mandible label already splits the dentition at the occlusal contacts; dilating it
    # would capture upper incisors that overlap the lower ones (deep bite / anterior crossbite)
    near_mand = seg.mandible if seg.mandible is not None else np.zeros_like(seg.teeth)
    teeth = teeth_render_mask(seg)
    for name, m in (("teeth_lower", teeth & near_mand), ("teeth_upper", teeth & ~near_mand)):
        if m.sum() > 50:
            out[name] = decimate(mask_to_mesh(m, seg.volume, sigma_mm=0.4), 40_000).transformed(frame.world_to_head)
    return out


CHIN_REST_NOTE = ("chin rest removed from the scan: the skin under the chin is shown as scanned, i.e. compressed by "
                  "the rest; the submental contour is not representative of the relaxed face")


def remove_chin_rest(seg: Segmentation, frame: HeadFrame, lms) -> bool:
    """Remove chin cups / bite blocks: tissue-density structures below the soft-tissue chin that are
    thinner than ~7 mm (plates resting under the chin): a 3.5 mm opening restricted to the submental
    region, where all anatomy is thick, so thin facial parts - alae, lips, ears - are untouched."""
    sp = seg.volume.spacing
    pog_s = lms["soft_pogonion"]
    A = frame.world_to_head @ seg.volume.affine
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in seg.head.shape], indexing="ij", sparse=True)
    Zh = A[2, 0] * ii + A[2, 1] * jj + A[2, 2] * kk + A[2, 3]
    Yh = A[1, 0] * ii + A[1, 1] * jj + A[1, 2] * kk + A[1, 3]
    region = (Zh < pog_s[2] - 2.0) & (Yh > pog_s[1] - 45.0)
    tissue = seg.head & ~seg.air_inside
    core = ndi.distance_transform_edt(tissue, sampling=sp) > 3.5
    core = ndi.distance_transform_edt(~core, sampling=sp) <= 3.5
    lab, n = ndi.label(core)
    if n == 0:
        return False
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = lab == np.argmax(sizes)
    keep = ndi.distance_transform_edt(~main, sampling=sp) <= 3.6
    drop = region & tissue & ~keep
    lab2, n2 = ndi.label(drop)
    if n2 == 0:
        return False
    sizes2 = np.bincount(lab2.ravel())
    big = sizes2 >= int(200 / np.prod(sp))      # > 0.2 cm^3: a device, not noise
    big[0] = False
    remove = big[lab2]
    if not remove.any():
        return False
    # the 6 mm closing that seals nostrils / parted lips also seals the air gap between the skin and
    # a device resting against it ("air_inside"); once the device is gone that gap is outside air
    gap = region & seg.air_inside
    lab3, n3 = ndi.label(gap)
    if n3:
        near = ndi.distance_transform_edt(~remove, sampling=sp) <= 1.5
        touching = np.unique(lab3[near & gap])
        open_gap = np.isin(lab3, touching[touching > 0])
        seg.air_inside = seg.air_inside & ~open_gap
        remove = remove | open_gap
    seg.head = seg.head & ~remove
    # second, finer pass next to the removed device: with the gap open, the device wall that touched
    # the skin is a thin flap (< ~4 mm, air on both sides) standing off the thick tissue; real skin
    # there is backed by thick tissue and lies within 2.2 mm of the 2 mm-opened core
    tissue2 = seg.head & ~seg.air_inside
    core2 = ndi.distance_transform_edt(tissue2, sampling=sp) > 2.0
    core2 = ndi.distance_transform_edt(~core2, sampling=sp) <= 2.0
    lab4, n4 = ndi.label(core2)
    if n4:
        sizes4 = np.bincount(lab4.ravel())
        sizes4[0] = 0
        keep2 = ndi.distance_transform_edt(lab4 != np.argmax(sizes4), sampling=sp) <= 2.2
        del lab4, core2
        near = ndi.distance_transform_edt(~remove, sampling=sp) <= 6.0
        flap = near & region & tissue2 & ~keep2
        seg.head = seg.head & ~flap
    return True


def fov_edge_distance(seg: Segmentation, world_vertices: np.ndarray) -> np.ndarray:
    """Distance (mm) of each point to the nearest unmeasured region: CBCT padding, slices without
    data and everything beyond the image array. Skin surface where this is ~0 is the flat cap the scan
    field of view cuts through the head, not real skin; the renderer hides it."""
    from .tissue.model import _live_planes
    live = _live_planes(seg.volume.data)
    measured = seg.fov.copy()
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = ~live[ax]
        measured[tuple(sl)] = False
    measured = np.pad(measured, 1, constant_values=False)     # beyond the array is unmeasured
    d = ndi.distance_transform_edt(measured, sampling=seg.volume.spacing).astype(np.float32)
    ijk = seg.volume.world_to_voxel(world_vertices) + 1.0
    return ndi.map_coordinates(d, ijk.T, order=1, mode="nearest").astype(np.float32)


class Case:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "case.json").read_text())
        self._seg = None
        self._anatomy = None
        self._meshes = None
        self._model = None
        self._bonefield = None

    @property
    def id(self):
        return self.meta["id"]

    @property
    def cache(self) -> Path:
        p = self.root / "cache"
        p.mkdir(exist_ok=True)
        return p

    def save_meta(self):
        (self.root / "case.json").write_text(json.dumps(self.meta, indent=2, default=_json_default))

    def set_status(self, stage: str, progress: float, message: str = ""):
        self.meta["status"] = {"stage": stage, "progress": round(progress, 3), "message": message,
                               "time": time.time()}
        self.save_meta()

    # ------------------------------------------------------------------ processing
    def process(self, log=print, model_options: ModelOptions | None = None, template: "Case | None" = None):
        if not self.meta.get("volume_input"):
            if not self.meta.get("surface_input"):
                raise ValueError("case has neither a CT/CBCT volume nor a surface scan")
            if template is None:
                raise ValueError("a processed CT/CBCT template case is required for scan-only planning")
            from .scan_only import build_from_scan
            return build_from_scan(self, template, log)
        t0 = time.time()
        src = self.meta["volume_input"]
        for stale in ("face_model.npz",):
            (self.cache / stale).unlink(missing_ok=True)
        self.set_status("loading", 0.02, "reading volume")
        vol = read_volume(self.root / "input" / src)
        if self.meta.get("modality"):
            vol.modality = self.meta["modality"]
        self.set_status("segmentation", 0.08, "calibrating intensities and segmenting tissues")
        seg = segment_head(vol, spacing_mm=self.meta.get("working_spacing_mm", 0.5))
        log(f"segmentation done {time.time() - t0:.0f}s; thresholds {seg.thresholds}")
        self.set_status("surfaces", 0.30, "extracting skin and bone surfaces")
        skin = skin_surface(seg)
        bone = mask_to_mesh(seg.bone, seg.volume, sigma_mm=0.6)
        self.set_status("frame", 0.40, "estimating mid-sagittal plane")
        bone_d = decimate(bone, 250_000)
        skin_d = decimate(skin, 300_000)
        frame = estimate_head_frame(bone_d.vertices, skin_d.vertices)
        self.set_status("landmarks", 0.45, "detecting landmarks")
        skin_h = frame.to_head(skin_d.vertices)
        lms = detect_landmarks(seg, frame, skin_h)
        if remove_chin_rest(seg, frame, lms):
            log("positioning device under the chin removed")
            skin = skin_surface(seg)
            skin_d = decimate(skin, 300_000)
            lms = detect_landmarks(seg, frame, frame.to_head(skin_d.vertices))   # on the cleaned skin
            lms.notes.append(CHIN_REST_NOTE)
        seg.mandible = separate_mandible(seg, frame, lms)
        detect_bony_midline(seg, frame, lms, seg.mandible)
        mand_mesh = decimate(mask_to_mesh(seg.mandible, seg.volume, sigma_mm=0.6), 150_000).transformed(
            frame.world_to_head)
        detect_bilateral(seg, frame, lms, mand_mesh)
        del lms._contour_state
        arch = build_arch_param(mand_mesh.vertices, lms)
        teeth_meshes = split_teeth_meshes(seg, frame)
        # skin attributes: distance to the edge of the scan field of view (renderer fades cut surfaces)
        skin_h_mesh = skin_d.transformed(frame.world_to_head)
        skin_h_mesh.attributes["fov_dist"] = fov_edge_distance(seg, skin_d.vertices)
        meshes = {"skin": skin_h_mesh, "bone": bone_d.transformed(frame.world_to_head), "mandible": mand_mesh}
        meshes.update(teeth_meshes)
        # maxilla+cranium without mandible (for jaw-motion rendering)
        upper = seg.bone & ~ndi.binary_dilation(seg.mandible, iterations=1)
        meshes["cranium"] = decimate(mask_to_mesh(upper, seg.volume, sigma_mm=0.6), 200_000).transformed(
            frame.world_to_head)
        patient = {
            "soft_tissue_thickness_pog_mm": float(np.linalg.norm(lms["soft_pogonion"] - lms["pogonion"])),
            "soft_tissue_thickness_menton_mm": float(np.linalg.norm(lms["soft_menton"] - lms["menton"])),
            "calibration": {"scale": seg.calibration.scale, "offset": seg.calibration.offset},
            "thresholds": seg.thresholds,
            "sex": self.meta.get("sex", ""), "age": self.meta.get("age"),
        }
        anatomy = Anatomy(frame, lms, arch, patient)
        self.set_status("model", 0.6, "building the soft-tissue finite-element model")
        bf = BoneField(seg, frame)
        model = build_tissue_model(seg, frame, lms, arch, bf.normals, model_options, surface_fn=bf.surface)
        log(f"FE model: {model.info}")
        bf._surf = {}            # large surface volumes are only needed for model building
        self.set_status("saving", 0.9, "writing cache")
        self._save_seg(seg)
        (self.cache / "anatomy.json").write_text(json.dumps(anatomy.to_dict(), default=_json_default))
        np.savez_compressed(self.cache / "meshes.npz", **save_mesh_dict(meshes))
        save_model(self.cache / "model.npz", model)
        self._seg, self._anatomy, self._meshes, self._model, self._bonefield = seg, anatomy, meshes, model, bf
        self.set_status("motion", 0.95, "precomputing expressions and aging")
        bf.release_segmentation()
        from .tissue.motion_cache import compute_and_store
        compute_and_store(self)
        self.meta["pipeline_version"] = PIPELINE_VERSION
        self.meta["skin_smoothed"] = True
        self.meta["model_info"] = model.info
        self.meta["processing_seconds"] = round(time.time() - t0, 1)
        self.set_status("ready", 1.0, "ready")
        log(f"case {self.id} processed in {time.time() - t0:.0f}s")

    def _save_seg(self, seg: Segmentation):
        packed = {k: np.packbits(getattr(seg, k)) for k in
                  ("fov", "head", "air_inside", "bone", "teeth", "metal", "fat", "mandible")}
        np.savez_compressed(self.cache / "seg.npz", hu=np.clip(seg.volume.data, -1100, 3000).astype(np.int16),
                            affine=seg.volume.affine, shape=np.array(seg.head.shape),
                            calib=np.array([seg.calibration.scale, seg.calibration.offset]),
                            thresholds=json.dumps(seg.thresholds), modality=seg.volume.modality, **packed)

    # ------------------------------------------------------------------ cached accessors
    @property
    def ready(self) -> bool:
        return self.meta.get("status", {}).get("stage") == "ready"

    @property
    def seg(self) -> Segmentation:
        if self._seg is None:
            z = np.load(self.cache / "seg.npz")
            shape = tuple(z["shape"])
            n = int(np.prod(shape))
            unpack = lambda k: np.unpackbits(z[k], count=n).reshape(shape).astype(bool)
            vol = Volume(z["hu"].astype(np.float32), z["affine"], str(z["modality"]))
            cal = Calibration(float(z["calib"][0]), float(z["calib"][1]), 0, 0, False)
            self._seg = Segmentation(vol, cal, unpack("fov"), unpack("head"), unpack("air_inside"), unpack("bone"),
                                     unpack("teeth"), unpack("metal"), unpack("fat"), json.loads(str(z["thresholds"])),
                                     unpack("mandible"))
        return self._seg

    @property
    def anatomy(self) -> Anatomy:
        if self._anatomy is None:
            self._anatomy = Anatomy.from_dict(json.loads((self.cache / "anatomy.json").read_text()))
        return self._anatomy

    def save_anatomy(self):
        (self.cache / "anatomy.json").write_text(json.dumps(self.anatomy.to_dict(), default=_json_default))

    @property
    def meshes(self) -> dict[str, Mesh]:
        if self._meshes is None:
            self._meshes = load_mesh_dict(np.load(self.cache / "meshes.npz"))
            if not self.meta.get("skin_smoothed"):
                self._meshes["skin"] = taubin_smooth(self._meshes["skin"], iterations=20)
                self.meta["skin_smoothed"] = True
                self.save_meta()
                np.savez_compressed(self.cache / "meshes.npz", **save_mesh_dict(self._meshes))
            if "teeth_lower" not in self._meshes and self.seg.teeth.sum() > 100:
                # cache upgrade from earlier pipeline versions
                self._meshes.pop("teeth", None)
                self._meshes.update(split_teeth_meshes(self.seg, self.anatomy.frame))
                np.savez_compressed(self.cache / "meshes.npz", **save_mesh_dict(self._meshes))
        return self._meshes

    @property
    def model(self) -> TissueModel:
        if self._model is None:
            self._model = load_model(self.cache / "model.npz")
        return self._model

    @property
    def bonefield(self) -> BoneField:
        if self._bonefield is None:
            self._bonefield = BoneField(self.seg, self.anatomy.frame)
        return self._bonefield

    def redetect(self, log=print):
        """Re-run landmarking, mandible separation, arch and FE model from the cached segmentation."""
        seg = self.seg
        frame = self.anatomy.frame
        skin_h = self.meshes["skin"].vertices
        lms = detect_landmarks(seg, frame, skin_h)
        seg.mandible = separate_mandible(seg, frame, lms)
        detect_bony_midline(seg, frame, lms, seg.mandible)
        mand_mesh = decimate(mask_to_mesh(seg.mandible, seg.volume, sigma_mm=0.6), 150_000).transformed(
            frame.world_to_head)
        detect_bilateral(seg, frame, lms, mand_mesh)
        del lms._contour_state
        arch = build_arch_param(mand_mesh.vertices, lms)
        self._anatomy = Anatomy(frame, lms, arch, {**self.anatomy.patient,
                                "soft_tissue_thickness_pog_mm": float(np.linalg.norm(lms["soft_pogonion"] - lms["pogonion"])),
                                "soft_tissue_thickness_menton_mm": float(np.linalg.norm(lms["soft_menton"] - lms["menton"]))})
        self._meshes["mandible"] = mand_mesh
        self._meshes.update(split_teeth_meshes(seg, frame))
        np.savez_compressed(self.cache / "meshes.npz", **save_mesh_dict(self._meshes))
        self._save_seg(seg)
        self.save_anatomy()
        for f in ("face_model.npz",):
            (self.cache / f).unlink(missing_ok=True)
        self._bonefield = None
        self.rebuild_model()
        log(f"landmarks re-detected: {lms.notes}")

    def rebuild_model(self, opts: ModelOptions | None = None):
        """Rebuild the FE model (e.g. after landmark edits)."""
        a = self.anatomy
        self._model = build_tissue_model(self.seg, a.frame, a.landmarks, a.arch, self.bonefield.normals, opts,
                                         surface_fn=self.bonefield.surface)
        save_model(self.cache / "model.npz", self._model)
        self.meta["model_info"] = self._model.info
        self.save_meta()
        return self._model

    # ------------------------------------------------------------------ variants
    def variants(self) -> list[dict]:
        p = self.root / "variants.json"
        return json.loads(p.read_text()) if p.exists() else []

    def save_variants(self, v: list[dict]):
        (self.root / "variants.json").write_text(json.dumps(v, indent=2, default=_json_default))


class CaseStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def default_template(self) -> "Case | None":
        """Processed CT/CBCT case used as the anatomical template for scan-only planning
        (MENTOPLAN_TEMPLATE env var, else the first ready radiological case)."""
        import os
        tid = os.environ.get("MENTOPLAN_TEMPLATE")
        for m in self.list():
            if (tid and m["id"] == tid) or (not tid and m.get("status", {}).get("stage") == "ready"
                                            and m.get("anatomy_source") != "estimated" and m.get("volume_input")):
                return self.get(m["id"])
        return None

    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            if (d / "case.json").exists():
                out.append(json.loads((d / "case.json").read_text()))
        return out

    def get(self, case_id: str) -> Case:
        d = self.root / case_id
        if not (d / "case.json").exists():
            raise KeyError(case_id)
        return Case(d)

    def create(self, name: str, volume_path: Path | None = None, *, move: bool = False, sex: str = "",
               age: int | None = None, modality: str | None = None, surface_path: Path | None = None) -> Case:
        cid = time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:8]
        d = self.root / cid
        (d / "input").mkdir(parents=True)
        meta = {"id": cid, "name": name, "created": time.time(), "sex": sex, "age": age, "modality": modality,
                "status": {"stage": "queued", "progress": 0.0, "message": "queued"}}
        if volume_path is not None:
            volume_path = Path(volume_path)
            dst = d / "input" / volume_path.name
            if volume_path.is_dir():
                shutil.copytree(volume_path, dst)
            elif move:
                shutil.move(str(volume_path), dst)
            else:
                shutil.copy2(volume_path, dst)
            meta["volume_input"] = volume_path.name
        if surface_path is not None:
            surface_path = Path(surface_path)
            shutil.copy2(surface_path, d / "input" / surface_path.name)
            for sib in surface_path.parent.iterdir():   # OBJ material and texture files
                if sib.suffix.lower() in (".mtl", ".jpg", ".jpeg", ".png") and sib.is_file():
                    shutil.copy2(sib, d / "input" / sib.name)
            meta["surface_input"] = surface_path.name
        (d / "case.json").write_text(json.dumps(meta, indent=2))
        return Case(d)
