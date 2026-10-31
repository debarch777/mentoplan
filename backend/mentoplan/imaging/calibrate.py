"""Intensity calibration.

CBCT grey values are not Hounsfield units and differ between scanners (the
two sample CBCTs used in development have air at 0 and -2254 respectively).
All downstream thresholds are expressed in HU, so every volume is first
mapped to a *pseudo-HU* scale by a linear map that sends the air mode to
-1000 HU and the soft-tissue mode to +40 HU.  For calibrated CT already in
HU this map is (nearly) the identity and is skipped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

AIR_HU = -1000.0
SOFT_HU = 40.0


@dataclass
class Calibration:
    scale: float
    offset: float
    air_value: float
    soft_value: float
    already_hu: bool

    def apply(self, data: np.ndarray) -> np.ndarray:
        if self.already_hu:
            return data.astype(np.float32, copy=False)
        return (data * self.scale + self.offset).astype(np.float32)


def _otsu(values: np.ndarray, bins: int = 512) -> float:
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    w = hist.astype(float)
    w0 = np.cumsum(w)
    w1 = w0[-1] - w0
    m = np.cumsum(w * centers)
    mu0 = m / np.maximum(w0, 1e-12)
    mu1 = (m[-1] - m) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[np.argmax(between[:-1])])


def estimate_calibration(data: np.ndarray, modality: str = "CBCT") -> Calibration:
    """Map the air and soft-tissue intensity modes to -1000 / +40 HU.

    Otsu's threshold separates air from tissue; air is the median of the low
    class and soft tissue the histogram mode of the high class (soft tissue
    dominates bone by volume in any head scan).  Field-of-view padding (a
    constant at the global minimum) falls into the air class, which is
    harmless because its value is close to true air in all scanners seen.
    """
    sub = data[::2, ::2, ::2].ravel()
    vmin = float(sub.min())
    vals = sub
    if np.count_nonzero(sub > vmin) < 1000:
        raise ValueError("calibration failed: volume is empty")
    lo, hi = np.percentile(vals, [0.01, 99.9])
    vals = vals[(vals >= lo) & (vals <= hi)]
    t = _otsu(vals)
    low, high = vals[vals < t], vals[vals >= t]
    air = float(np.median(low)) if low.size else vmin
    hist, edges = np.histogram(high, bins=256)
    centers = 0.5 * (edges[1:] + edges[:-1])
    soft = float(centers[np.argmax(ndi.gaussian_filter1d(hist.astype(float), 2))])
    if modality == "CT" and -1100 < air < -850 and -80 < soft < 120:
        return Calibration(1.0, 0.0, air, soft, True)
    if soft <= air:
        raise ValueError("calibration failed: soft tissue mode not above air mode")
    scale = (SOFT_HU - AIR_HU) / (soft - air)
    offset = AIR_HU - air * scale
    return Calibration(scale, offset, air, soft, False)
