"""Lightweight headless preview renderer (numpy z-buffer point splatting).

Used for server-side thumbnails, PDF reports and automated visual checks; the
photorealistic renderer is the WebGL client.
"""
from __future__ import annotations

import numpy as np


def _rot(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Rx @ Ry


def render_points(layers, view_dir_basis: np.ndarray, size=(512, 512), mm_per_px: float | None = None,
                  center: np.ndarray | None = None, yaw: float = 0.0, pitch: float = 0.0,
                  light=(0.3, 0.5, 1.0), background=(1.0, 1.0, 1.0)):
    """Render a list of (vertices, normals, rgb) layers.

    ``view_dir_basis`` is a 3x3 matrix whose rows are the camera right, up and
    towards-viewer axes expressed in the data frame.  Returns an (H, W, 3) uint8 image.
    """
    H, W = size
    R = _rot(yaw, pitch) @ view_dir_basis
    allv = np.concatenate([l[0] for l in layers])
    if center is None:
        center = 0.5 * (allv.min(0) + allv.max(0))
    pv = (allv - center) @ R.T
    if mm_per_px is None:
        ext = np.abs(pv[:, :2]).max() * 2.1
        mm_per_px = ext / min(H, W)
    img = np.empty((H, W, 3))
    img[:] = background
    zbuf = np.full(H * W, -np.inf)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    cols = []
    pix = []
    depth = []
    for v, n, rgb in layers:
        p = (v - center) @ R.T
        nn = n @ R.T
        diff = np.clip(nn @ L, 0, 1)
        h = (L + np.array([0, 0, 1.0]))
        h /= np.linalg.norm(h)
        spec = np.clip(nn @ h, 0, 1) ** 40 * 0.25
        c = np.asarray(rgb, float)
        if c.ndim == 1:
            c = np.broadcast_to(c, (len(v), 3))
        shade = c * (0.25 + 0.75 * diff[:, None]) + spec[:, None]
        facing = nn[:, 2] > -0.05
        u = np.round(p[:, 0] / mm_per_px + W / 2).astype(int)
        w = np.round(-p[:, 1] / mm_per_px + H / 2).astype(int)
        ok = facing & (u >= 0) & (u < W) & (w >= 0) & (w < H)
        cols.append(shade[ok]); pix.append(w[ok] * W + u[ok]); depth.append(p[ok, 2])
    cols = np.concatenate(cols); pix = np.concatenate(pix); depth = np.concatenate(depth)
    # splat 2x2 to close gaps
    for du, dw in ((0, 0), (1, 0), (0, 1), (1, 1)):
        pp = pix + du + dw * W
        ok = pp < H * W
        np.maximum.at(zbuf, pp[ok], depth[ok])
    flat = img.reshape(-1, 3)
    for du, dw in ((0, 0), (1, 0), (0, 1), (1, 1)):
        pp = pix + du + dw * W
        ok = (pp < H * W)
        win = ok.copy()
        win[ok] = depth[ok] >= zbuf[pp[ok]] - 1e-9
        flat[pp[win]] = cols[win]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def surface_samples(vertices: np.ndarray, faces: np.ndarray, colors=None, density: float = 6.0,
                    max_points: int = 4_000_000, seed: int = 0):
    """Area-uniform surface samples (points, normals, colours) for gap-free splatting.

    ``density`` is samples per mm^2.  Normals are interpolated vertex normals.
    """
    rng = np.random.default_rng(seed)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(fn, axis=1)
    vn = np.zeros_like(vertices)
    for c in range(3):
        np.add.at(vn, faces[:, c], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    n = int(min(max_points, area.sum() * density))
    fi = rng.choice(len(faces), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = 1 - s, s * (1 - r2), s * r2
    f = faces[fi]
    pts = a[:, None] * vertices[f[:, 0]] + b[:, None] * vertices[f[:, 1]] + c[:, None] * vertices[f[:, 2]]
    nrm = a[:, None] * vn[f[:, 0]] + b[:, None] * vn[f[:, 1]] + c[:, None] * vn[f[:, 2]]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    if colors is None:
        col = None
    else:
        colors = np.asarray(colors, float)
        if colors.ndim == 1:
            col = colors
        else:
            col = a[:, None] * colors[f[:, 0]] + b[:, None] * colors[f[:, 1]] + c[:, None] * colors[f[:, 2]]
    return pts, nrm, col


def render_mesh_layers(meshes, view_basis, **kw):
    """meshes: list of (vertices, faces, colour or per-vertex colours)."""
    layers = [surface_samples(v, f, c) for v, f, c in meshes]
    return render_points(layers, view_basis, **kw)
