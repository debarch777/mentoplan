"""Surface extraction and mesh processing."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from skimage import measure

from ..io.volume import Volume


@dataclass
class Mesh:
    vertices: np.ndarray            # (n, 3) float64, mm
    faces: np.ndarray               # (m, 3) int32
    attributes: dict = field(default_factory=dict)  # per-vertex arrays

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy(), {k: v.copy() for k, v in self.attributes.items()})

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    def face_normals(self) -> np.ndarray:
        v = self.vertices
        f = self.faces
        n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def vertex_normals(self) -> np.ndarray:
        v = self.vertices
        f = self.faces
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])  # area weighted
        vn = np.zeros_like(v)
        for c in range(3):
            np.add.at(vn, f[:, c], fn)
        return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)

    def transformed(self, T: np.ndarray) -> "Mesh":
        v = self.vertices @ T[:3, :3].T + T[:3, 3]
        f = self.faces if np.linalg.det(T[:3, :3]) > 0 else self.faces[:, ::-1].copy()
        return Mesh(v, f, dict(self.attributes))

    def area(self) -> float:
        v = self.vertices
        f = self.faces
        return float(0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum())

    def volume(self) -> float:
        v = self.vertices
        f = self.faces
        return float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6.0)


def mask_to_mesh(mask: np.ndarray, vol: Volume, sigma_mm: float = 0.7, step: int = 1,
                 crop_margin: int = 4) -> Mesh:
    """Smooth iso-surface (level 0.5) of a binary mask, in world (LPS mm) coordinates.

    Outward-facing normals (triangle winding is fixed so that the surface is
    oriented away from the masked region).
    """
    idx = np.argwhere(mask)
    if idx.size == 0:
        raise ValueError("empty mask")
    lo = np.maximum(idx.min(0) - crop_margin, 0)
    hi = np.minimum(idx.max(0) + crop_margin + 1, mask.shape)
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    sub = np.pad(sub, 2)
    sp = vol.spacing
    field_ = ndi.gaussian_filter(sub, sigma_mm / sp)
    verts, faces, _, _ = measure.marching_cubes(field_, 0.5, step_size=step, allow_degenerate=False)
    verts = verts - 2 + lo
    world = vol.voxel_to_world(verts)
    m = Mesh(world, faces.astype(np.int32))
    # orient outward: gradient of the field points inward (towards mask), so normals should
    # point opposite to the mask; skimage's winding depends on the affine handedness
    if np.linalg.det(vol.affine[:3, :3]) < 0:
        m.faces = m.faces[:, ::-1].copy()
    if m.volume() < 0:
        m.faces = m.faces[:, ::-1].copy()
    return m


def largest_connected_submesh(mesh: Mesh) -> Mesh:
    f = mesh.faces
    n = mesh.n_vertices
    adj = sparse.coo_matrix((np.ones(len(f) * 3), (np.r_[f[:, 0], f[:, 1], f[:, 2]],
                                                    np.r_[f[:, 1], f[:, 2], f[:, 0]])), shape=(n, n))
    ncomp, lab = sparse.csgraph.connected_components(adj, directed=False)
    if ncomp == 1:
        return mesh
    counts = np.bincount(lab[f[:, 0]], minlength=ncomp)
    keep_lab = np.argmax(counts)
    return submesh(mesh, lab[f[:, 0]] == keep_lab)


def submesh(mesh: Mesh, face_mask: np.ndarray) -> Mesh:
    f = mesh.faces[face_mask]
    used = np.unique(f)
    remap = -np.ones(mesh.n_vertices, dtype=np.int64)
    remap[used] = np.arange(len(used))
    attrs = {k: v[used] for k, v in mesh.attributes.items()}
    out = Mesh(mesh.vertices[used], remap[f].astype(np.int32), attrs)
    out.attributes["_source_index"] = used
    return out


def vertex_adjacency(mesh: Mesh) -> sparse.csr_matrix:
    f = mesh.faces
    n = mesh.n_vertices
    i = np.r_[f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]]
    j = np.r_[f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]]
    A = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def taubin_smooth(mesh: Mesh, iterations: int = 10, lam: float = 0.5, mu: float = -0.53,
                  fixed: np.ndarray | None = None) -> Mesh:
    A = vertex_adjacency(mesh)
    deg = np.asarray(A.sum(1)).ravel()
    W = sparse.diags(1.0 / np.maximum(deg, 1)) @ A
    v = mesh.vertices.copy()
    for _ in range(iterations):
        for f in (lam, mu):
            d = W @ v - v
            if fixed is not None:
                d[fixed] = 0
            v = v + f * d
    out = mesh.copy()
    out.vertices = v
    return out


def decimate(mesh: Mesh, target_faces: int) -> Mesh:
    if len(mesh.faces) <= target_faces:
        return mesh
    import fast_simplification
    reduction = 1.0 - target_faces / len(mesh.faces)
    v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                                        target_reduction=reduction, agg=7)
    return Mesh(v.astype(np.float64), f.astype(np.int32))


def sample_volume(vol: Volume, points: np.ndarray, data: np.ndarray | None = None, order: int = 1,
                  cval: float = 0.0) -> np.ndarray:
    ijk = vol.world_to_voxel(points)
    return ndi.map_coordinates(vol.data if data is None else data, ijk.T, order=order, mode="constant", cval=cval)


def to_trimesh(mesh: Mesh):
    import trimesh
    return trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)


def lip_side_labels(vertices: np.ndarray, sto: np.ndarray, z_cut: float, half_width: float,
                    y_teeth: float) -> np.ndarray:
    """+1 upper lip side, -1 lower lip side, 0 elsewhere (vertices in the oral fissure zone)."""
    zone = (np.abs(vertices[:, 0]) < half_width + 3) & (np.abs(vertices[:, 2] - z_cut) < 12) & \
        (vertices[:, 1] > y_teeth - 8)
    lab = np.zeros(len(vertices), np.int8)
    lab[zone & (vertices[:, 2] >= z_cut)] = 1
    lab[zone & (vertices[:, 2] < z_cut)] = -1
    return lab


def smooth_vertex_field(mesh: Mesh, field: np.ndarray, active: np.ndarray, iterations: int = 10,
                        lam: float = 0.5, side: np.ndarray | None = None) -> np.ndarray:
    """Umbrella-operator smoothing of a per-vertex vector field over the surface.

    ``field`` holds values for ``active`` vertices (others are zero).  Edges joining
    vertices on opposite sides of the oral fissure (``side`` = +1 / -1) are cut so
    upper and lower lip motion never blur into each other."""
    n = mesh.n_vertices
    f = mesh.faces
    i = np.r_[f[:, 0], f[:, 1], f[:, 2]]
    j = np.r_[f[:, 1], f[:, 2], f[:, 0]]
    if side is not None:
        keep = side[i] * side[j] >= 0
        i, j = i[keep], j[keep]
    A = sparse.coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    Wm = sparse.diags(1.0 / np.maximum(deg, 1)) @ A
    full = np.zeros((n, field.shape[1]), np.float64)
    full[active] = field
    for _ in range(iterations):
        full = full + lam * (Wm @ full - full)
    return full[active].astype(field.dtype)


def skin_surface(seg, sigma_mm: float = 0.5, taubin_iters: int = 10) -> Mesh:
    """Sub-voxel skin surface: iso-surface (-450 HU, half-way air/tissue) of the smoothed intensity,
    constrained to the segmented head (outside air forced to -1000, internal cavities to tissue).
    Avoids the 0.5 mm terracing of a binary-mask surface under grazing light."""
    vol = seg.volume
    sp = vol.spacing
    f = np.where(seg.head, np.maximum(vol.data, -400.0), -1000.0).astype(np.float32)
    f[seg.air_inside] = 40.0
    idx = np.argwhere(seg.head)
    lo = np.maximum(idx.min(0) - 4, 0)
    hi = np.minimum(idx.max(0) + 5, seg.head.shape)
    sub = f[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub = np.pad(sub, 2, constant_values=-1000.0)
    sub = ndi.gaussian_filter(sub, sigma_mm / sp)
    verts, faces, _, _ = measure.marching_cubes(sub, -450.0)
    verts = verts - 2 + lo
    m = Mesh(vol.voxel_to_world(verts), faces.astype(np.int32))
    if np.linalg.det(vol.affine[:3, :3]) < 0:
        m.faces = m.faces[:, ::-1].copy()
    m = largest_connected_submesh(m)
    if m.volume() < 0:
        m.faces = m.faces[:, ::-1].copy()
    return taubin_smooth(m, taubin_iters)
