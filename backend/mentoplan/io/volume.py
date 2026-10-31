"""Volumetric image I/O.

Every reader returns a :class:`Volume` whose ``affine`` maps voxel indices
(i, j, k) -> patient coordinates in millimetres, LPS convention (DICOM:
+x = patient Left, +y = Posterior, +z = Superior).  Array axis order is
(i, j, k), i.e. ``data[i, j, k]`` where i is the fastest-varying index on
disk (columns in DICOM terms).

Supported formats: DICOM series (directory or list of files), NRRD/NHDR,
NIfTI (.nii/.nii.gz via nibabel), GIPL (.gipl/.gipl.gz) and MetaImage
(.mha/.mhd).
"""
from __future__ import annotations

import gzip
import os
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Volume:
    data: np.ndarray            # (ni, nj, nk) intensities; CT in Hounsfield units where known
    affine: np.ndarray          # 4x4 voxel -> LPS mm
    modality: str = "CT"        # "CT" or "CBCT" (CBCT grey values are not calibrated HU)
    meta: dict = field(default_factory=dict)

    @property
    def spacing(self) -> np.ndarray:
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def shape(self):
        return self.data.shape

    def voxel_to_world(self, ijk: np.ndarray) -> np.ndarray:
        ijk = np.asarray(ijk, dtype=float)
        return ijk @ self.affine[:3, :3].T + self.affine[:3, 3]

    def world_to_voxel(self, xyz: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(self.affine)
        xyz = np.asarray(xyz, dtype=float)
        return xyz @ inv[:3, :3].T + inv[:3, 3]

    def resampled(self, factor: int) -> "Volume":
        """Integer down-sampling by block averaging (anti-aliased)."""
        if factor <= 1:
            return self
        d = self.data
        ni, nj, nk = (s // factor for s in d.shape)
        d = d[: ni * factor, : nj * factor, : nk * factor].astype(np.float32)
        d = d.reshape(ni, factor, nj, factor, nk, factor).mean(axis=(1, 3, 5))
        aff = self.affine.copy()
        aff[:3, :3] *= factor
        # voxel centre of the new grid is the centre of each block
        aff[:3, 3] = self.affine[:3, 3] + self.affine[:3, :3] @ np.full(3, (factor - 1) / 2.0)
        return Volume(d, aff, self.modality, dict(self.meta))


# --------------------------------------------------------------------------- NRRD

_NRRD_TYPES = {
    "signed char": "i1", "int8": "i1", "int8_t": "i1",
    "uchar": "u1", "unsigned char": "u1", "uint8": "u1", "uint8_t": "u1",
    "short": "i2", "short int": "i2", "signed short": "i2", "int16": "i2", "int16_t": "i2",
    "ushort": "u2", "unsigned short": "u2", "uint16": "u2", "uint16_t": "u2",
    "int": "i4", "signed int": "i4", "int32": "i4", "int32_t": "i4",
    "uint": "u4", "unsigned int": "u4", "uint32": "u4", "uint32_t": "u4",
    "float": "f4", "double": "f8",
}


def _parse_vec(s: str) -> list[float]:
    return [float(v) for v in s.strip().strip("()").split(",")]


def read_nrrd(path: str | os.PathLike) -> Volume:
    path = Path(path)
    raw = path.read_bytes()
    sep = raw.find(b"\n\n")
    if sep < 0:
        raise ValueError("NRRD: header terminator not found")
    header = raw[:sep].decode("latin-1").splitlines()
    if not header[0].startswith("NRRD"):
        raise ValueError("not a NRRD file")
    fields: dict[str, str] = {}
    for line in header[1:]:
        if line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        fields[k.strip().lower()] = v.lstrip("=").strip()
    dim = int(fields["dimension"])
    if dim != 3:
        raise ValueError(f"NRRD: only 3D volumes supported (got {dim}D)")
    sizes = [int(s) for s in fields["sizes"].split()]
    dtype = np.dtype(_NRRD_TYPES[fields["type"].lower()])
    endian = fields.get("endian", "little")
    dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    enc = fields.get("encoding", "raw").lower()

    if "data file" in fields or "datafile" in fields:
        df = fields.get("data file", fields.get("datafile"))
        payload = (path.parent / df).read_bytes()
    else:
        payload = raw[sep + 2:]
    if enc in ("gzip", "gz"):
        payload = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else zlib.decompress(payload)
    elif enc in ("bzip2", "bz2"):
        import bz2
        payload = bz2.decompress(payload)
    elif enc != "raw":
        raise ValueError(f"NRRD: unsupported encoding {enc}")
    n = int(np.prod(sizes))
    arr = np.frombuffer(payload[len(payload) - n * dtype.itemsize:], dtype=dtype, count=n)
    data = arr.reshape(sizes[::-1]).transpose(2, 1, 0).astype(np.float32)

    space = fields.get("space", "left-posterior-superior").lower()
    if "space directions" in fields:
        dirs = re.findall(r"\(([^)]*)\)", fields["space directions"])
        M = np.array([_parse_vec(d) for d in dirs]).T  # columns are axis vectors
    else:
        sp = [float(s) for s in fields.get("spacings", "1 1 1").split()]
        M = np.diag(sp)
    origin = np.array(_parse_vec(fields["space origin"])) if "space origin" in fields else np.zeros(3)
    aff = np.eye(4)
    aff[:3, :3] = M
    aff[:3, 3] = origin
    if space in ("right-anterior-superior", "ras"):
        flip = np.diag([-1, -1, 1, 1])
        aff = flip @ aff
    return Volume(data, aff, meta={"format": "nrrd", "source": str(path)})


# --------------------------------------------------------------------------- GIPL

_GIPL_TYPES = {1: ">u1", 7: ">i1", 8: ">u1", 15: ">i2", 16: ">u2", 31: ">u4", 32: ">i4", 64: ">f4", 65: ">f8"}


def read_gipl(path: str | os.PathLike) -> Volume:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        raw = f.read()
    dims = struct.unpack(">4H", raw[0:8])
    itype = struct.unpack(">H", raw[8:10])[0]
    pix = struct.unpack(">4f", raw[10:26])
    origin = struct.unpack(">4d", raw[204:236])
    if itype not in _GIPL_TYPES:
        raise ValueError(f"GIPL: unsupported image type {itype}")
    dtype = np.dtype(_GIPL_TYPES[itype])
    ni, nj, nk = dims[0], dims[1], max(dims[2], 1)
    n = ni * nj * nk
    arr = np.frombuffer(raw[256:256 + n * dtype.itemsize], dtype=dtype, count=n)
    data = arr.reshape(nk, nj, ni).transpose(2, 1, 0).astype(np.float32)
    aff = np.eye(4)
    aff[:3, :3] = np.diag(pix[:3])
    aff[:3, 3] = origin[:3]
    return Volume(data, aff, meta={"format": "gipl", "source": str(path)})


# --------------------------------------------------------------------------- MetaImage

def read_metaimage(path: str | os.PathLike) -> Volume:
    path = Path(path)
    raw = path.read_bytes()
    fields: dict[str, str] = {}
    offset = 0
    for line in raw.split(b"\n"):
        offset += len(line) + 1
        k, _, v = line.decode("latin-1").partition("=")
        fields[k.strip()] = v.strip()
        if k.strip() == "ElementDataFile":
            break
    sizes = [int(s) for s in fields["DimSize"].split()]
    types = {"MET_SHORT": "i2", "MET_USHORT": "u2", "MET_UCHAR": "u1", "MET_CHAR": "i1",
             "MET_INT": "i4", "MET_UINT": "u4", "MET_FLOAT": "f4", "MET_DOUBLE": "f8"}
    dtype = np.dtype(types[fields["ElementType"]])
    dtype = dtype.newbyteorder(">" if fields.get("BinaryDataByteOrderMSB", "False") == "True" else "<")
    edf = fields["ElementDataFile"]
    payload = raw[offset:] if edf == "LOCAL" else (path.parent / edf).read_bytes()
    if fields.get("CompressedData", "False") == "True":
        payload = zlib.decompress(payload)
    n = int(np.prod(sizes))
    data = np.frombuffer(payload, dtype=dtype, count=n).reshape(sizes[::-1]).transpose(2, 1, 0).astype(np.float32)
    sp = [float(s) for s in fields.get("ElementSpacing", "1 1 1").split()]
    tm = [float(s) for s in fields.get("TransformMatrix", "1 0 0 0 1 0 0 0 1").split()]
    R = np.array(tm).reshape(3, 3).T
    aff = np.eye(4)
    aff[:3, :3] = R @ np.diag(sp)
    aff[:3, 3] = [float(s) for s in fields.get("Offset", fields.get("Position", "0 0 0")).split()]
    return Volume(data, aff, meta={"format": "metaimage", "source": str(path)})


# --------------------------------------------------------------------------- NIfTI

def read_nifti(path: str | os.PathLike) -> Volume:
    import nibabel as nib
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    ras = img.affine
    aff = np.diag([-1, -1, 1, 1]) @ ras  # RAS -> LPS
    return Volume(data, aff, meta={"format": "nifti", "source": str(path)})


# --------------------------------------------------------------------------- DICOM

def read_dicom_series(path_or_files) -> Volume:
    """Read a single CT/CBCT series. ``path_or_files`` is a directory or list of files.

    If the directory contains several series, the one with most slices is used.
    Slices are sorted along the slice normal using ImagePositionPatient, and
    rescale slope/intercept are applied so CT values are in HU.
    """
    import pydicom

    if isinstance(path_or_files, (str, os.PathLike)) and Path(path_or_files).is_dir():
        files = [p for p in Path(path_or_files).rglob("*") if p.is_file()]
    else:
        files = [Path(p) for p in path_or_files]
    series: dict[str, list] = {}
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False, force=True)
        except Exception:
            continue
        if not hasattr(ds, "PixelData") or not hasattr(ds, "ImagePositionPatient"):
            continue
        series.setdefault(str(getattr(ds, "SeriesInstanceUID", "unknown")), []).append(ds)
    if not series:
        raise ValueError("no DICOM image slices found")
    slices = max(series.values(), key=len)
    iop = np.array(slices[0].ImageOrientationPatient, dtype=float)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    slices.sort(key=lambda s: float(np.dot(normal, np.array(s.ImagePositionPatient, dtype=float))))
    if len(slices) == 1 and int(getattr(slices[0], "NumberOfFrames", 1)) > 1:
        raise ValueError("enhanced multi-frame DICOM not supported; export as classic series")
    imgs = []
    for s in slices:
        a = s.pixel_array.astype(np.float32)
        a = a * float(getattr(s, "RescaleSlope", 1.0)) + float(getattr(s, "RescaleIntercept", 0.0))
        imgs.append(a)
    vol = np.stack(imgs, axis=0)  # (k, rows=j, cols=i)
    data = vol.transpose(2, 1, 0)
    ps = [float(v) for v in slices[0].PixelSpacing]  # row spacing, column spacing
    p0 = np.array(slices[0].ImagePositionPatient, dtype=float)
    if len(slices) > 1:
        p1 = np.array(slices[-1].ImagePositionPatient, dtype=float)
        kvec = (p1 - p0) / (len(slices) - 1)
    else:
        kvec = normal * float(getattr(slices[0], "SliceThickness", 1.0))
    aff = np.eye(4)
    aff[:3, 0] = row * ps[1]
    aff[:3, 1] = col * ps[0]
    aff[:3, 2] = kvec
    aff[:3, 3] = p0
    modality = str(getattr(slices[0], "Modality", "CT"))
    desc = (str(getattr(slices[0], "Manufacturer", "")) + " " + str(getattr(slices[0], "SeriesDescription", ""))).lower()
    if "cbct" in desc or "cone" in desc:
        modality = "CBCT"
    return Volume(data, aff, modality=modality,
                  meta={"format": "dicom", "n_slices": len(slices),
                        "patient_sex": str(getattr(slices[0], "PatientSex", "")),
                        "patient_age": str(getattr(slices[0], "PatientAge", ""))})


def read_volume(path: str | os.PathLike) -> Volume:
    p = Path(path)
    name = p.name.lower()
    if p.is_dir():
        return read_dicom_series(p)
    if name.endswith((".nrrd", ".nhdr")):
        return read_nrrd(p)
    if name.endswith((".gipl", ".gipl.gz")):
        return read_gipl(p)
    if name.endswith((".mha", ".mhd")):
        return read_metaimage(p)
    if name.endswith((".nii", ".nii.gz")):
        return read_nifti(p)
    if name.endswith(".dcm") or name.endswith(".zip"):
        if name.endswith(".zip"):
            import tempfile
            import zipfile
            tmp = tempfile.mkdtemp(prefix="mentoplan_dicom_")
            with zipfile.ZipFile(p) as z:
                z.extractall(tmp)
            return read_dicom_series(tmp)
        return read_dicom_series(p.parent)
    raise ValueError(f"unrecognised volume format: {p.name}")
