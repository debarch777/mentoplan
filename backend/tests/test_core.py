"""Unit tests: I/O, calibration, geometry, implant design, analysis helpers."""
import gzip
import struct

import numpy as np
import pytest

from mentoplan.analysis import cephalometry as ceph
from mentoplan.anatomy.frame import fit_symmetry_plane
from mentoplan.api.bundle import pack, unpack
from mentoplan.imaging.calibrate import estimate_calibration
from mentoplan.imaging.surface import Mesh, smooth_vertex_field, taubin_smooth
from mentoplan.implant.design import ImplantDesign, thickness
from mentoplan.io.meshio import stl_bytes
from mentoplan.io.volume import read_volume
from mentoplan.tissue.predict import swelling_fraction


def _phantom(shape=(40, 50, 30)):
    rng = np.random.default_rng(0)
    d = np.full(shape, 0.0)
    d[8:32, 10:40, 5:25] = 400.0           # soft tissue
    d[16:24, 20:30, 10:20] = 1400.0        # bone
    return d + rng.normal(0, 8, shape)


def test_nrrd_roundtrip(tmp_path):
    d = _phantom().astype(np.int16)
    hdr = (f"NRRD0004\ntype: short\ndimension: 3\nspace: left-posterior-superior\nsizes: {d.shape[0]} {d.shape[1]} {d.shape[2]}\n"
           "space directions: (0.5,0,0) (0,0.5,0) (0,0,0.8)\nkinds: domain domain domain\nendian: little\n"
           "encoding: gzip\nspace origin: (-10,-20,5)\n\n").encode()
    p = tmp_path / "v.nrrd"
    p.write_bytes(hdr + gzip.compress(d.transpose(2, 1, 0).tobytes()))
    v = read_volume(p)
    assert v.shape == d.shape
    assert np.allclose(v.data, d)
    assert np.allclose(v.spacing, [0.5, 0.5, 0.8])
    assert np.allclose(v.voxel_to_world(np.array([[2, 4, 10]]))[0], [-9, -18, 13])


def test_gipl_and_metaimage(tmp_path):
    d = _phantom().astype(np.int16)
    hdr = bytearray(256)
    hdr[0:8] = struct.pack(">4H", *d.shape, 1)
    hdr[8:10] = struct.pack(">H", 15)
    hdr[10:26] = struct.pack(">4f", 0.5, 0.5, 0.5, 1.0)
    p = tmp_path / "v.gipl.gz"
    p.write_bytes(gzip.compress(bytes(hdr) + d.transpose(2, 1, 0).astype(">i2").tobytes()))
    v = read_volume(p)
    assert np.allclose(v.data, d)
    m = tmp_path / "v.mha"
    m.write_bytes((f"ObjectType = Image\nNDims = 3\nDimSize = {d.shape[0]} {d.shape[1]} {d.shape[2]}\n"
                   "ElementType = MET_SHORT\nElementSpacing = 1 1 2\nOffset = 1 2 3\nElementDataFile = LOCAL\n").encode()
                  + d.transpose(2, 1, 0).astype("<i2").tobytes())
    v2 = read_volume(m)
    assert np.allclose(v2.data, d) and np.allclose(v2.spacing, [1, 1, 2])


def test_dicom_series(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    d = _phantom().astype(np.int16)
    series = generate_uid()
    for k in range(d.shape[2]):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(tmp_path / f"s{k}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SeriesInstanceUID = series
        ds.Modality = "CT"
        ds.Rows, ds.Columns = d.shape[1], d.shape[0]
        ds.PixelSpacing = [0.5, 0.4]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, 1.25 * k]
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelData = (d[:, :, k].T + 1024).astype(np.int16).tobytes()
        # write slices in reverse order: the reader must sort along the slice normal
        ds.save_as(str(tmp_path / f"s{d.shape[2] - k:03d}.dcm"), enforce_file_format=True)
    v = read_volume(tmp_path)
    assert v.shape == d.shape
    assert np.allclose(v.data, d)
    assert np.allclose(v.spacing, [0.4, 0.5, 1.25])


def test_calibration_maps_air_and_soft_tissue():
    rng = np.random.default_rng(1)
    raw = np.concatenate([rng.normal(-2000, 30, 600000), rng.normal(300, 20, 300000), rng.normal(1500, 80, 60000)])
    raw = raw.reshape(60, 80, 200).astype(np.float32)
    cal = estimate_calibration(raw)
    hu = cal.apply(np.array([-2000.0, 300.0, 1500.0]))
    assert abs(hu[0] + 1000) < 40 and abs(hu[1] - 40) < 25 and hu[2] > 500


def test_symmetry_plane_recovery():
    import trimesh
    # head-like symmetric surface: ellipsoid with a nose and two (mirror) cheek bumps
    s = trimesh.creation.icosphere(5, 1.0)
    v = np.asarray(s.vertices) * [70, 90, 110]
    for c, r, h in (([0, 90, 0], 25, 22), ([35, 70, -30], 20, 10), ([-35, 70, -30], 20, 10)):
        d = np.linalg.norm(v - np.array(c, float) * [1, 1, 1], axis=1)
        v = v + (h * np.exp(-(d / r) ** 2))[:, None] * v / np.linalg.norm(v, axis=1, keepdims=True)
    pts = v
    ang = np.radians(8)
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    pts = pts @ R.T + [3, -2, 1]
    n, d, rms = fit_symmetry_plane(pts, np.array([1.0, 0, 0]), 0.0)
    true_n = R @ np.array([1.0, 0, 0])
    assert abs(abs(n @ true_n) - 1) < 1e-3
    assert rms < 2.0          # nearest-neighbour residual, limited by the ~3 mm point spacing


@pytest.mark.parametrize("shape", ["round", "anatomical", "square"])
def test_implant_thickness_properties(shape):
    d = ImplantDesign(projection=7, width=50, height=18, shape=shape)
    u = np.linspace(-40, 40, 321)
    v = np.linspace(-8, 30, 153)
    U, V = np.meshgrid(u, v, indexing="ij")
    t = thickness(U, V, d)
    assert np.isclose(t.max(), 7.0, atol=0.05)
    assert np.all(t >= 0)
    assert np.all(t[np.abs(U) > 25.01] == 0)                  # nothing outside the footprint width
    assert np.all(t[V > 18.01] == 0)                           # nor above it
    assert np.allclose(t, t[::-1], atol=1e-9)                  # symmetric without asymmetry
    under = V < -0.01
    assert t[under].max() <= d.inferior_wrap + 1e-9            # only a thin lip under the border


def test_implant_asymmetry_and_cant():
    d = ImplantDesign(projection=6, asymmetry=0.3)
    tl = thickness(np.array([-12.0]), np.array([8.0]), d)[0]
    tr = thickness(np.array([12.0]), np.array([8.0]), d)[0]
    assert tr > tl * 1.3
    d2 = ImplantDesign(projection=6, cant_deg=8, inferior_offset=3)
    # with cant the right side footprint is raised: at low v the right side is thinner
    assert thickness(np.array([15.0]), np.array([3.5]), d2)[0] < thickness(np.array([-15.0]), np.array([3.5]), d2)[0]


def test_vertical_extension_adds_under_border():
    base = ImplantDesign(projection=5)
    ext = ImplantDesign(projection=5, vertical_extension=4)
    v = np.array([-1.0])
    assert thickness(np.zeros(1), v, ext)[0] > thickness(np.zeros(1), v, base)[0] + 1.5


def test_design_validation():
    assert ImplantDesign(projection=30).validate()
    assert not ImplantDesign().validate()


def test_profile_metrics_signs():
    L = {"soft_glabella": [0, 60, 60], "subnasale": [0, 55, 0], "soft_pogonion": [0, 45, -55],
         "pronasale": [0, 75, 5], "labrale_inferius": [0, 52, -30], "labrale_superius": [0, 58, -15],
         "soft_b_point": [0, 45, -42], "stomion": [0, 55, -22], "soft_menton": [0, 35, -70], "soft_nasion": [0, 52, 30]}
    m = ceph.profile_metrics({k: np.array(v, float) for k, v in L.items()})
    assert m["facial_convexity"] > 5               # retrusive chin -> convex profile
    assert m["lower_lip_e_line"] < 0               # lip behind the Prn-Pog' line
    L["soft_pogonion"] = [0, 70, -55]              # prognathic chin
    m2 = ceph.profile_metrics({k: np.array(v, float) for k, v in L.items()})
    assert m2["facial_convexity"] < 0


def test_swelling_course():
    assert swelling_fraction(0) == 0
    peak = max(swelling_fraction(t) for t in np.linspace(0.5, 5, 50))
    assert peak > 0.98
    assert 0.25 < swelling_fraction(14) < 0.5
    assert swelling_fraction(42) < 0.2
    assert swelling_fraction(180) < 0.02


def test_bundle_roundtrip():
    a = {"x": np.arange(10, dtype=np.float32), "i": np.arange(6, dtype=np.uint32).reshape(2, 3)}
    meta, arr = unpack(pack(a, {"k": 1}))
    assert meta == {"k": 1}
    assert np.array_equal(arr["x"], a["x"]) and arr["i"].shape == (2, 3)


def _sphere(n=20):
    import trimesh
    s = trimesh.creation.icosphere(3, 10.0)
    return Mesh(np.asarray(s.vertices), np.asarray(s.faces, np.int32))


def test_taubin_preserves_volume_and_stl():
    m = _sphere()
    rng = np.random.default_rng(0)
    noisy = m.copy()
    noisy.vertices = noisy.vertices + rng.normal(0, 0.2, noisy.vertices.shape)
    sm = taubin_smooth(noisy, 20)
    assert abs(sm.volume() - m.volume()) / m.volume() < 0.03
    b = stl_bytes(m)
    assert len(b) == 84 + 50 * len(m.faces)


def test_field_smoothing_respects_fissure():
    m = _sphere()
    side = np.where(m.vertices[:, 2] > 0, 1, -1).astype(np.int8)
    f = np.zeros((m.n_vertices, 3), np.float32)
    f[m.vertices[:, 2] > 0, 2] = 1.0
    act = np.arange(m.n_vertices)
    out = smooth_vertex_field(m, f, act, 30, 0.5, side)
    assert np.allclose(out[m.vertices[:, 2] < 0], 0)           # nothing leaks across the cut
