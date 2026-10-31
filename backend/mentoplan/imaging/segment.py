"""Tissue segmentation of a (pseudo-)HU head volume.

All morphology is done with Euclidean distance transforms so radii are in
millimetres and independent of voxel size.

Output masks (same grid as the input volume):

* ``fov``      – voxels that contain measured data (not CBCT padding)
* ``head``     – everything inside the skin envelope; internal air cavities
                 (airway, sinuses, oral cavity) are excluded, but the nostril
                 and lip openings are sealed so the skin surface is closed
* ``bone``     – solid bone, medullary cavities filled
* ``teeth``    – enamel/dentin (sub-set of bone)
* ``metal``    – orthodontic brackets, restorations, plates (streak-artefact sources)
* ``fat``      – subcutaneous/deep fat (HU < -30) inside soft tissue
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from ..io.volume import Volume
from .calibrate import Calibration, estimate_calibration


@dataclass
class Segmentation:
    volume: Volume          # pseudo-HU volume on the working grid
    calibration: Calibration
    fov: np.ndarray
    head: np.ndarray
    air_inside: np.ndarray  # internal air cavities (airway, oral cavity, sinuses)
    bone: np.ndarray
    teeth: np.ndarray
    metal: np.ndarray
    fat: np.ndarray
    thresholds: dict
    mandible: np.ndarray | None = None   # filled in after landmarking (anatomy.landmarks.separate_mandible)

    @property
    def soft(self) -> np.ndarray:
        return self.head & ~self.bone & ~self.air_inside


def _dilate(mask: np.ndarray, r_mm: float, spacing) -> np.ndarray:
    return ndi.distance_transform_edt(~mask, sampling=spacing) <= r_mm


def _erode(mask: np.ndarray, r_mm: float, spacing) -> np.ndarray:
    return ndi.distance_transform_edt(mask, sampling=spacing) > r_mm


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == np.argmax(sizes)


def _remove_small(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    return keep[lab]


def _fill_holes_2d(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for ax in range(3):
        filled = np.zeros_like(mask)
        for i in range(mask.shape[ax]):
            sl = [slice(None)] * 3
            sl[ax] = i
            filled[tuple(sl)] = ndi.binary_fill_holes(mask[tuple(sl)])
        out |= filled
    return out


def correct_bias(hu: np.ndarray, sp, sigma_mm: float = 12.0, iterations: int = 3):
    """Remove the smooth additive intensity non-uniformity of CBCT (scatter, cupping, heel effect).

    The local soft-tissue level is estimated by normalised convolution over voxels whose value is
    close to the current estimate (robust to air and bone), then shifted to +40 HU everywhere.
    Works on a 2 mm grid; the field is interpolated back."""
    f = max(1, int(round(2.0 / sp[0])))
    small = hu[::f, ::f, ::f]
    sig = sigma_mm / (sp[0] * f)
    local = np.full(small.shape, 40.0, np.float32)
    m = None
    for _ in range(iterations):
        m = ((np.abs(small - local) < 350) & (small > -700)).astype(np.float32)
        den = ndi.gaussian_filter(m, sig)
        num = ndi.gaussian_filter(small * m, sig)
        ok = den > 0.05
        est = np.where(ok, num / np.maximum(den, 1e-6), 0)
        # extend into regions without tissue (air) from the nearest estimate
        _, ind = ndi.distance_transform_edt(~ok, return_indices=True)
        local = est[tuple(ind)].astype(np.float32)
    bias = np.clip(local - 40.0, -600, 600)
    zoom = np.array(hu.shape) / np.array(bias.shape)
    full = ndi.zoom(bias, zoom, order=1)[:hu.shape[0], :hu.shape[1], :hu.shape[2]]
    pad = [(0, max(0, hu.shape[i] - full.shape[i])) for i in range(3)]
    full = np.pad(full, pad, mode="edge")
    out = (hu - full).astype(np.float32)
    soft = out[::f, ::f, ::f][m > 0]
    info = {"bias_min": float(bias.min()), "bias_max": float(bias.max()),
            "soft_sd": float(np.std(soft[np.abs(soft - 40) < 300])) if soft.size else 60.0}
    return out, info


def resample_isotropic(vol: Volume, spacing_mm: float = 0.5) -> Volume:
    """Resample (trilinear) onto an isotropic grid aligned with the volume axes.

    Integer down-sampling is done by block averaging (exact and fast); other
    ratios use map_coordinates.
    """
    sp = vol.spacing
    ratio = spacing_mm / sp
    if np.allclose(ratio, np.round(ratio), atol=0.02) and np.all(np.round(ratio) >= 1) and \
            len(set(np.round(ratio).astype(int))) == 1:
        return vol.resampled(int(round(ratio[0])))
    new_shape = np.maximum(1, np.floor((np.array(vol.shape) - 1) * sp / spacing_mm).astype(int) + 1)
    coords = [np.arange(n) * spacing_mm / s for n, s in zip(new_shape, sp)]
    out = np.empty(tuple(new_shape), dtype=np.float32)
    gi, gj = np.meshgrid(coords[0], coords[1], indexing="ij")
    for k, ck in enumerate(coords[2]):
        out[:, :, k] = ndi.map_coordinates(vol.data, [gi, gj, np.full_like(gi, ck)], order=1, mode="nearest")
    aff = vol.affine.copy()
    aff[:3, :3] = vol.affine[:3, :3] @ np.diag(spacing_mm / sp)
    return Volume(out, aff, vol.modality, dict(vol.meta))


def segment_head(vol: Volume, spacing_mm: float = 0.5, calibration: Calibration | None = None,
                 bone_hu: float | None = None) -> Segmentation:
    vol = resample_isotropic(vol, spacing_mm)
    sp = vol.spacing
    raw = vol.data
    pad_value = float(raw.min())
    cal = calibration or estimate_calibration(raw, vol.modality)
    hu = cal.apply(raw)
    bias_info = {}
    if vol.modality != "CT":
        hu, bias_info = correct_bias(hu, sp)
    vol = Volume(hu, vol.affine, vol.modality, dict(vol.meta))

    # --- field of view.  Voxels outside the reconstruction cylinder are padded with a constant.
    # Only when that constant is distinguishable from real air (noisy, different mean) can we
    # tell "not measured" from "air"; otherwise everything inside the array counts as measured
    # (out-of-array regions are always treated as unmeasured downstream).
    pad = raw == pad_value
    fov = np.ones_like(pad)
    if pad.mean() > 0.01:
        near_air = raw[(raw > pad_value) & (raw < cal.air_value + 0.25 * (cal.soft_value - cal.air_value))]
        air_mean = float(np.median(near_air)) if near_air.size > 1000 else pad_value
        if abs(air_mean - pad_value) > 0.08 * (cal.soft_value - cal.air_value):
            fov = ~_remove_small(pad, 10000)
            fov = _erode(fov, 1.0, sp)

    smooth = ndi.gaussian_filter(hu, 0.6 / sp[0])

    # --- tissue / air boundary at the half-way value between air and soft tissue
    tissue = (smooth > -450) & fov
    # remove positioning devices (chin rest, head band) and noise: open with 3 mm, keep the
    # largest component, then grow back into the original tissue.
    opened = _dilate(_erode(tissue, 3.0, sp), 3.0, sp) & tissue
    core = _largest_component(opened)
    head0 = tissue & _dilate(core, 3.5, sp)
    head0 = _largest_component(head0)

    # outside air: flood from the volume border after sealing narrow openings (nostrils,
    # parted lips) with a 6 mm closing; then recover concavities of the true skin surface
    # (a 6 mm ball fills < 1 mm of a 120 deg fold, which the 4.5 mm recovery restores)
    closed = _erode(_dilate(head0, 6.0, sp), 6.0, sp) | head0
    border_seed = np.zeros_like(closed)
    border_seed[[0, -1], :, :] = True
    border_seed[:, [0, -1], :] = True
    border_seed[:, :, [0, -1]] = True
    outside_sealed = ndi.binary_propagation(border_seed & ~closed, mask=~closed)
    outside = _dilate(outside_sealed, 4.5, sp) & ~head0
    head = ~outside
    air_inside = head & ~head0

    # --- bone: threshold between soft tissue and cortical bone, adapted to the soft-tissue noise
    if bone_hu is None:
        bone_hu = 300.0 if vol.modality == "CT" else max(350.0, 40.0 + 5.0 * bias_info.get("soft_sd", 60.0))
    # nothing within 1.5 mm of the skin is bone (excludes chin rests / bite blocks touching the face)
    depth = ndi.distance_transform_edt(head, sampling=sp)
    bone = (smooth > bone_hu) & head & (depth > 1.5)
    bone = _remove_small(bone, int(500 / np.prod(sp)))  # >0.5 cm^3
    bone_closed = _erode(_dilate(bone, 1.0, sp), 1.0, sp) | bone
    bone_solid = _fill_holes_2d(bone_closed) & head

    # --- teeth and metal: data-driven high-density thresholds
    bvals = hu[bone_solid]
    teeth_hu = max(float(np.percentile(bvals, 97)), bone_hu + 600) if bvals.size else 1500.0
    metal_hu = max(2500.0, float(np.percentile(bvals, 99.9))) if bvals.size else 3000.0
    teeth = (smooth > teeth_hu) & bone_solid
    teeth = _remove_small(teeth, int(20 / np.prod(sp)))
    metal = (hu > metal_hu) & head
    metal = _remove_small(metal, 3)

    soft = head & ~bone_solid & ~air_inside
    # fat vs. muscle/glandular tissue: CBCT noise requires stronger smoothing
    fat = soft & (ndi.gaussian_filter(hu, 1.2 / sp[0]) < -30)

    return Segmentation(volume=vol, calibration=cal, fov=fov, head=head, air_inside=air_inside,
                        bone=bone_solid, teeth=teeth, metal=metal, fat=fat,
                        thresholds={"tissue_hu": -450.0, "bone_hu": bone_hu, "teeth_hu": teeth_hu,
                                    "metal_hu": metal_hu, "fat_hu": -30.0, **bias_info})
