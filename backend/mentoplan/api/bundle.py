"""Compact binary bundles for meshes and fields (consumed by the WebGL client).

Layout:  b"MPB1" | uint32 header_len | header JSON (utf-8, padded to 8) | array blobs (8-aligned)
Header:  {"meta": {...}, "arrays": [{"name", "dtype", "shape", "offset", "nbytes"}]}
"""
from __future__ import annotations

import json
import struct

import numpy as np

_DT = {np.dtype("float32"): "f32", np.dtype("uint32"): "u32", np.dtype("int32"): "i32", np.dtype("uint8"): "u8",
       np.dtype("uint16"): "u16"}


def pack(arrays: dict, meta: dict | None = None) -> bytes:
    entries = []
    blobs = []
    off = 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        elif a.dtype == np.int64:
            a = a.astype(np.uint32 if a.min(initial=0) >= 0 else np.int32)
        elif a.dtype == bool:
            a = a.astype(np.uint8)
        b = a.tobytes()
        pad = (-len(b)) % 8
        entries.append({"name": name, "dtype": _DT[a.dtype], "shape": list(a.shape), "offset": off, "nbytes": len(b)})
        blobs.append(b + b"\0" * pad)
        off += len(b) + pad
    header = json.dumps({"meta": meta or {}, "arrays": entries}, default=_json).encode()
    header += b" " * ((-len(header) - 8) % 8)
    return b"MPB1" + struct.pack("<I", len(header)) + header + b"".join(blobs)


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def unpack(data: bytes) -> tuple[dict, dict]:
    assert data[:4] == b"MPB1"
    hl = struct.unpack("<I", data[4:8])[0]
    header = json.loads(data[8:8 + hl])
    base = 8 + hl
    inv = {v: k for k, v in _DT.items()}
    arrays = {}
    for e in header["arrays"]:
        a = np.frombuffer(data, dtype=inv[e["dtype"]], count=int(np.prod(e["shape"])) if e["shape"] else 1,
                          offset=base + e["offset"])
        arrays[e["name"]] = a.reshape(e["shape"])
    return header["meta"], arrays
