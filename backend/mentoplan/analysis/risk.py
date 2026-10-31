"""Surgical risk assessment for a chin implant design.

Every check reports a level (low / moderate / high), the quantitative basis
and the rule that produced it, so the clinician can see *why*.  Thresholds
are conservative planning heuristics drawn from the craniofacial surgery
literature; they are decision support, not guarantees.
"""
from __future__ import annotations

import numpy as np

from ..implant.design import ImplantDesign, thickness
from ..tissue.model import MAT_SKIN

LEVELS = ("low", "moderate", "high")


def _lvl(v, moderate, high, reverse=False):
    if reverse:
        return "high" if v <= high else ("moderate" if v <= moderate else "low")
    return "high" if v >= high else ("moderate" if v >= moderate else "low")


def implant_footprint_points(design: ImplantDesign, arch, mand_vertices: np.ndarray, min_t: float = 0.3,
                             mand_normals: np.ndarray | None = None):
    """Mandible surface points covered by the implant (thickness >= min_t) and their thickness."""
    from ..implant.design import node_thickness
    if mand_normals is None:
        mand_normals = np.tile([0.0, 1.0, 0.0], (len(mand_vertices), 1))
    th = arch.angle(mand_vertices)
    r = np.hypot(mand_vertices[:, 0], mand_vertices[:, 1] - arch.center_y)
    r_out = np.interp(th, arch.theta, arch.r_outer)
    labial = r > r_out - 6.0 - design.vertical_extension
    t = node_thickness(mand_vertices, mand_normals, arch, design)
    sel = (t >= min_t) & labial
    return mand_vertices[sel], t[sel]


def assess(design: ImplantDesign, anatomy, model, sol_fields: dict, mand_vertices: np.ndarray,
           implant_mesh=None, implant_metrics: dict | None = None, predicted_landmarks: dict | None = None,
           penalty_k: float | None = None, pocket_nodes=None, pocket_normals=None, pocket_g=None,
           mand_normals: np.ndarray | None = None) -> list[dict]:
    lms = anatomy.landmarks
    arch = anatomy.arch
    out = []
    fp, ft = implant_footprint_points(design, arch, mand_vertices, mand_normals=mand_normals)
    pts = implant_mesh.vertices if implant_mesh is not None else fp

    # ---- 1. mental nerve
    for side in ("right", "left"):
        k = f"mental_foramen_{side}"
        if k not in lms or len(pts) == 0:
            continue
        fpos = lms[k]
        conf = lms.points[k].confidence
        d = float(np.min(np.linalg.norm(pts - fpos, axis=1)))
        # does the footprint pass above the foramen (nerve compression / traction)?
        fu, fv = arch.uv(fpos[None])
        t_at = float(thickness(np.nan_to_num(fu, nan=1e6), fv, design)[0])
        level = _lvl(d, 6.0, 3.0, reverse=True)
        if t_at > 0.3:
            level = "high"
        msg = (f"Closest implant point {d:.1f} mm from the {side} mental foramen"
               + (f"; implant covers the foramen ({t_at:.1f} mm thick there)" if t_at > 0.3 else ""))
        if conf < 0.3:
            msg += " - foramen position estimated with low confidence, verify on the scan"
            if level == "low":
                level = "moderate"
        out.append({"id": f"mental_nerve_{side}", "category": "Nerve", "title": f"Mental nerve ({side})",
                    "level": level, "value": d, "unit": "mm", "detail": msg,
                    "rule": ">= 6 mm clearance low; 3-6 mm moderate (dissect below the nerve, avoid traction); "
                            "< 3 mm or covering the foramen high (hypoesthesia of lip/chin)"})

    # ---- 2. tooth roots (screw fixation and alveolar pressure)
    if "anterior_root_apex" in lms and len(fp):
        apex_z = lms["anterior_root_apex"][2]
        top = float(fp[:, 2].max())
        margin = apex_z - top
        level = _lvl(margin, 3.0, 0.0, reverse=True)
        detail = f"Upper implant edge {abs(margin):.1f} mm {'below' if margin >= 0 else 'ABOVE'} the anterior root apices"
        if design.fixation == "screws":
            # proposed screws: +/-8 mm from midline at 35% of the footprint height
            screw_z = arch.z_inf[np.argmin(np.abs(arch.arc))] + design.inferior_offset + 0.35 * design.height
            sm = apex_z - (screw_z + 1.0)
            detail += f"; proposed screw level {sm:.1f} mm below the apices"
            level = max(level, _lvl(sm, 4.0, 2.0, reverse=True), key=LEVELS.index)
        out.append({"id": "tooth_roots", "category": "Teeth", "title": "Incisor / canine roots", "level": level,
                    "value": margin, "unit": "mm", "detail": detail,
                    "rule": "keep implant and screws >= 3-4 mm below root apices; implants over thin "
                            "alveolar bone resorb more"})

    # ---- 3. pressure on bone / resorption
    if penalty_k is not None and pocket_nodes is not None and "impl" in sol_fields:
        u = sol_fields["impl"][pocket_nodes]
        gap = pocket_g - np.einsum("ij,ij->i", u, pocket_normals)
        force = penalty_k * gap                                          # mN per node (contact reaction)
        area = model.h ** 2
        p = force / area                                                 # kPa
        act = pocket_g > 0.3
        if act.any():
            pm = float(np.mean(p[act]))
            px = float(np.percentile(p[act], 95))
            alveolar = 0.0
            if "anterior_root_apex" in lms and len(fp):
                alveolar = float(np.mean(fp[:, 2] > lms["anterior_root_apex"][2] - 2.0))
            mat_factor = {"silicone": 1.0, "porous_polyethylene": 0.7, "peek": 0.6, "titanium": 0.5}.get(design.material, 1.0)
            score = (px / 4.0) * mat_factor * (1 + 2 * alveolar)
            level = _lvl(score, 1.0, 2.0)
            out.append({"id": "bone_resorption", "category": "Bone", "title": "Pressure on bone / resorption",
                        "level": level, "value": px, "unit": "kPa",
                        "detail": f"Soft-tissue contact pressure mean {pm:.1f} kPa, 95th pct {px:.1f} kPa; "
                                  f"{alveolar * 100:.0f}% of the footprint on alveolar bone; material {design.material}",
                        "rule": "resorption under alloplastic chin implants is usually < 2 mm and increases with "
                                "envelope tension, alveolar placement and silicone (systematic review)"})

    # ---- 4. migration / stability
    contact = (implant_metrics or {}).get("contact_area_cm2", 0.0)
    fix_score = {"screws": 0, "sutures": 1, "none": 2}.get(design.fixation, 1)
    shape_score = 0
    if design.projection > 0 and contact > 0:
        slender = design.projection / np.sqrt(contact * 100.0)
        shape_score = 1 if slender > 0.35 else 0
    wraps = design.inferior_offset <= 0.5
    total = fix_score + shape_score + (0 if wraps else 1)
    level = "low" if total == 0 else ("moderate" if total <= 2 else "high")
    out.append({"id": "migration", "category": "Implant", "title": "Migration / malposition", "level": level,
                "value": contact, "unit": "cm2",
                "detail": f"Bone-conforming custom fit, contact area {contact:.1f} cm2, fixation: {design.fixation}, "
                          f"{'seated on' if wraps else 'raised above'} the inferior border",
                "rule": "screw fixation of a bone-conforming implant seated on the inferior border minimises "
                        "migration; unfixed stock implants in a loose pocket migrate most"})

    # ---- 5. soft-tissue stretch / perfusion / extrusion
    if "impl" in sol_fields:
        u = sol_fields["impl"]
        G = model.geom
        grad = np.einsum("eai,eaJ->eiJ", u[model.elems], G.dNc)        # du/dX at element centre
        eps = 0.5 * (grad + grad.swapaxes(1, 2))
        w = np.linalg.eigvalsh(eps)
        emax = w[:, -1]
        skin = model.mat_id == MAT_SKIN
        e_skin = float(np.percentile(emax[skin], 99.5)) if skin.any() else 0.0
        level = _lvl(e_skin, 0.15, 0.25)
        out.append({"id": "tissue_tension", "category": "Soft tissue", "title": "Skin / envelope tension",
                    "level": level, "value": e_skin * 100, "unit": "% strain",
                    "detail": f"Peak dermal stretch {e_skin * 100:.0f}% (99.5th percentile over the skin layer)",
                    "rule": "< 15% low; 15-25% moderate (tight closure, edge visibility); > 25% high "
                            "(perfusion compromise, wound dehiscence, extrusion)"})
        # covering tissue thickness over the thickest part of the implant
        th = anatomy.patient.get("soft_tissue_thickness_pog_mm", None)
        if th is not None:
            cover = th * (1 - 0.5 * min(e_skin, 0.4))
            level = _lvl(cover, 6.0, 4.0, reverse=True)
            out.append({"id": "palpability", "category": "Soft tissue", "title": "Implant palpability / visibility",
                        "level": level, "value": cover, "unit": "mm",
                        "detail": f"Estimated soft-tissue cover over the implant {cover:.1f} mm "
                                  f"(pre-op {th:.1f} mm at pogonion)",
                        "rule": "< 4 mm cover: edges palpable/visible and higher extrusion risk; prefer feathered "
                                "edges or a smaller projection"})

    # ---- 6. mentalis strain / lip competence
    if predicted_landmarks is not None:
        Li = predicted_landmarks.get("labrale_inferius")
        Me = predicted_landmarks.get("soft_menton")
        sto = lms["stomion"] if "stomion" in lms else None
        detail = []
        score = 0
        if design.vertical_extension >= 4:
            score += 1
            detail.append(f"vertical lengthening {design.vertical_extension:.0f} mm")
        if Me is not None and sto is not None:
            pre_h = abs(lms["soft_menton"][2] - sto[2])
            post_h = abs(Me["median"][2] - sto[2])
            if pre_h > 0 and (post_h - pre_h) / pre_h > 0.12:
                score += 1
                detail.append(f"chin height +{(post_h - pre_h):.1f} mm ({(post_h - pre_h) / pre_h * 100:.0f}%)")
        if Li is not None:
            dz = Li["median"][2] - Li["pre"][2]
            if dz < -1.5:
                score += 1
                detail.append(f"lower lip drops {abs(dz):.1f} mm")
        level = LEVELS[min(score, 2)]
        out.append({"id": "lip_competence", "category": "Function", "title": "Lip competence / mentalis strain",
                    "level": level, "value": float(score), "unit": "",
                    "detail": "; ".join(detail) if detail else "no lengthening of the lower face that would strain lip closure",
                    "rule": "vertical lengthening > ~4 mm or > 12% chin height increase can cause mentalis strain "
                            "and lip incompetence"})
    ind = indication(anatomy, predicted_landmarks, design)
    if ind:
        out.insert(0, ind)
    return out


def indication(anatomy, predicted_landmarks: dict | None, design: ImplantDesign) -> dict:
    """Is chin augmentation aesthetically indicated, and does the plan over-project?

    Uses the head-pose independent soft-tissue facial convexity (G'-Sn-Pog')."""
    from .cephalometry import profile_metrics
    L = {k: anatomy.landmarks[k] for k, v in anatomy.landmarks.points.items() if v.confidence >= 0.3}
    pre = profile_metrics(L).get("facial_convexity")
    post = None
    if predicted_landmarks:
        L2 = dict(L)
        L2.update({k: np.asarray(v["median"]) for k, v in predicted_landmarks.items()})
        post = profile_metrics(L2).get("facial_convexity")
    if pre is None:
        return {}
    level, msg = "low", f"Facial convexity {pre:.0f} deg now"
    if post is not None:
        msg += f", {post:.0f} deg planned (norm 12 +/- 4; lower = more chin projection)"
    if pre < 8 and design.projection > 0.5:
        level = "high" if pre < 2 else "moderate"
        msg += ". The chin is not retrusive relative to the mid-face; augmentation will flatten or reverse " \
               "the profile - consider orthognathic assessment or a smaller / purely vertical / width design"
    elif post is not None and post < 6:
        level = "moderate" if post >= 2 else "high"
        msg += ". The plan over-projects the chin relative to the mid-face"
    return {"id": "indication", "category": "Aesthetics", "title": "Indication / over-projection",
            "level": level, "value": pre, "unit": "deg", "detail": msg,
            "rule": "soft-tissue convexity < 8 deg means the chin already projects; planned convexity < 6 deg "
                    "is over-projection for most faces"}


def overall(risks: list[dict]) -> str:
    if any(r["level"] == "high" for r in risks):
        return "high"
    if sum(r["level"] == "moderate" for r in risks) >= 2:
        return "moderate"
    return "low" if not any(r["level"] == "moderate" for r in risks) else "moderate"
