"""Disk cache of the expression and aging displacement fields.

They are computed once - during case processing, inside the memory-capped processing job - so the
interactive server only loads small arrays and never builds the whole-face model itself."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np


def _path(case):
    return case.cache / "motion_fields.npz"


def compute_and_store(case) -> None:
    from .aging import aging_fields
    from .expressions import expression_fields
    holder = SimpleNamespace(case=case)
    ex = expression_fields(holder)
    ag = aging_fields(holder)
    arrays = {"expr_active": ex["active"].astype(np.int64), "age_active": ag["active"].astype(np.int64)}
    for k, v in ex["fields"].items():
        arrays["expr__" + k] = v.astype(np.float32)
    for k, v in ag["fields"].items():
        arrays["age__" + k] = v.astype(np.float32)
    np.savez_compressed(_path(case), meta=json.dumps({"expressions": ex["meta"], "aging": ag["meta"]}), **arrays)


def load(case):
    p = _path(case)
    if not p.exists():
        return None
    z = np.load(p)
    meta = json.loads(str(z["meta"]))
    ex = {"active": z["expr_active"], "meta": meta["expressions"],
          "fields": {k[6:]: z[k] for k in z.files if k.startswith("expr__")}}
    ag = {"active": z["age_active"], "meta": meta["aging"],
          "fields": {k[5:]: z[k] for k in z.files if k.startswith("age__")}}
    return ex, ag


def motion_job(case_root: str) -> bool:
    """Child-process entry for cases processed before the cache existed."""
    from pathlib import Path
    from ..case import Case
    compute_and_store(Case(Path(case_root)))
    return True
