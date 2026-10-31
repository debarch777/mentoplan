"""Surface mesh I/O: STL (binary) export; OBJ/PLY/STL import with optional texture/colours."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ..imaging.surface import Mesh


def stl_bytes(mesh: Mesh, header: str = "MentoPlan") -> bytes:
    v = mesh.vertices
    f = mesh.faces
    n = mesh.face_normals()
    out = bytearray(header.encode()[:80].ljust(80, b" "))
    out += struct.pack("<I", len(f))
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = n
    rec["v"] = v[f]
    out += rec.tobytes()
    return bytes(out)


def read_surface(path: str | Path):
    """Read a surface scan.  Returns (Mesh, uv or None, texture image (H,W,3) uint8 or None, colors or None)."""
    import trimesh
    path = Path(path)
    tm = trimesh.load(str(path), process=False, force="mesh")
    mesh = Mesh(np.asarray(tm.vertices, float), np.asarray(tm.faces, np.int32))
    uv = tex = colors = None
    vis = getattr(tm, "visual", None)
    if vis is not None:
        if getattr(vis, "kind", None) == "texture" and getattr(vis, "uv", None) is not None:
            uv = np.asarray(vis.uv, np.float32)
            img = getattr(getattr(vis, "material", None), "image", None)
            if img is None and hasattr(vis.material, "baseColorTexture"):
                img = vis.material.baseColorTexture
            if img is not None:
                tex = np.asarray(img.convert("RGB"))
        elif getattr(vis, "kind", None) == "vertex":
            colors = np.asarray(vis.vertex_colors)[:, :3].astype(np.uint8)
    return mesh, uv, tex, colors
