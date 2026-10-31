"""Soft-tissue aging of the lower face (illustrative, wide uncertainty).

Two mechanisms are simulated on the whole-lower-face model:

1. Laxity under gravity - the progressive loss of dermal elasticity and
   retaining-ligament support is represented by the additional sag of the
   tissue under its own weight when its stiffness drops (upright posture).
   The response shape comes from the FE model; its magnitude is scaled to
   about 1 mm of soft-menton descent per decade, in line with longitudinal
   3-D facial aging studies of adults.
2. Volume loss - atrophy of the subcutaneous fat compartments (about 3% of
   their volume per decade) as a negative volumetric eigenstrain.

The same fields apply with and without an implant; the implant result is
superimposed, so the viewer shows "you in N years" for both the current
and the planned face.  These are population-average trends, not individual
predictions, and the UI labels them as such.
"""
from __future__ import annotations

import numpy as np

from .expressions import face_model
from .model import MAT_FAT
from .simulate import Simulator, TissueParams

TISSUE_WEIGHT = 1.05e-3 * 9.81      # mN / mm^3  (density 1.05 g/cm^3)
DESCENT_MM_PER_DECADE = 1.0
FAT_VOLUME_LOSS_PER_DECADE = 0.03


def aging_fields(rt) -> dict:
    if getattr(rt, "_aging", None) is not None:
        return rt._aging
    case = rt.case
    m = face_model(rt)
    sim = Simulator(m, TissueParams())
    lms = case.anatomy.landmarks
    from .hexfem import body_force_load
    f_g = body_force_load(m.geom, m.elems, m.n_nodes, np.full(len(m.elems), TISSUE_WEIGHT), np.array([0, 0, -1.0]))
    u_g = sim.solve(None, f_extra=f_g, interface="tied")
    Wme = m.interpolation_matrix(np.array([lms["soft_menton"], lms["soft_pogonion"]]), surface=True)
    dz = -(Wme @ u_g)[:, 2].mean()
    u_sag = u_g * (DESCENT_MM_PER_DECADE / max(dz, 1e-6))
    vol = np.where(m.mat_id == MAT_FAT, -FAT_VOLUME_LOSS_PER_DECADE, 0.0)
    u_vol = sim.solve(None, f_extra=sim.edema_load(vol), interface="tied")
    u = u_sag + u_vol
    skin = case.meshes["skin"]
    W = m.interpolation_matrix(skin.vertices, surface=True).tocsr()
    active = np.where(np.diff(W.indptr) > 0)[0]
    from ..imaging.surface import smooth_vertex_field
    from .expressions import fissure_sides
    field = smooth_vertex_field(skin, np.asarray(W[active] @ u, np.float32), active, 15, 0.5,
                                fissure_sides(case, skin.vertices))
    rt._aging = {"active": active, "fields": {"decade": field},
                 "meta": {"descent_mm_per_decade": DESCENT_MM_PER_DECADE,
                          "fat_volume_loss_per_decade": FAT_VOLUME_LOSS_PER_DECADE,
                          "note": "population-average trend; individual aging varies widely"}}
    return rt._aging
