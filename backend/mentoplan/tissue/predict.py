"""Probabilistic post-operative soft-tissue prediction.

For one implant design the engine solves the patient FE model and builds a
small set of displacement *fields* (all linear in their coefficient):

    impl      median-material solution, sliding pocket interface
    tied      difference to the adhered-interface hypothesis
    sens_*    d u / d log(theta) for each material group (fat, muscle, dermis, bulk)
    place_*   response to 1 mm vertical / 1 mm lateral / 1 deg cant placement error
    edema     response to post-operative edema, normalised to 1 mm at soft pogonion
    nl        nonlinear (large-deformation) correction of ``impl`` (optional, computed async)

A Monte-Carlo ensemble then samples the uncertain coefficients:

    u_s(t) = c_s k_s(t) [impl + nl + sum th_si sens_i + b_s tied + sum p_sj place_j]
             + a_s h(t) edema + e_s n

    c_s   evidence calibration (Bayesian fusion of the FE soft/hard ratio with
          clinical cephalometric ratios conditioned on this patient's soft-tissue thickness)
    k_s   1 - resorption_s(t)/projection  (material-specific bone resorption / settling)
    a_s   edema magnitude, h(t) the swelling time course
    e_s   surface (segmentation) error along the skin normal

and returns the likely displacement plus a 90% interval along the skin normal,
for every render vertex and for every soft-tissue landmark.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..implant.design import ImplantDesign, node_thickness, thickness
from .simulate import SENS_PARAMS, Simulator, TissueParams


# --------------------------------------------------------------------------- clinical priors

@dataclass
class ClinicalPriors:
    """Evidence used to calibrate and bound the physics model (see docs/evidence.md)."""
    # soft-tissue pogonion advancement / implant thickness vs soft-tissue thickness at pogonion
    # (Park; Ko - silicone chin augmentation: <7 mm: 0.8, 8-11 mm: 0.66, >12 mm: 0.5)
    ratio_thickness_mm: tuple = (7.0, 9.5, 12.5)
    ratio_values: tuple = (0.80, 0.66, 0.50)
    ratio_sd: float = 0.12
    # bone resorption / settling under the implant at 12 months (mm): mean, sd
    # (systematic review: mean < 2 mm, silicone most variable)
    resorption_mm: dict = field(default_factory=lambda: {
        "silicone": (1.0, 0.7), "porous_polyethylene": (0.6, 0.5), "peek": (0.4, 0.3), "titanium": (0.3, 0.3)})
    resorption_tau_days: float = 120.0
    # surgical placement error SD: (vertical mm, lateral mm, cant deg) by fixation
    placement_sd: dict = field(default_factory=lambda: {
        "screws": (0.8, 0.6, 1.5), "sutures": (1.5, 1.0, 3.0), "none": (2.5, 1.5, 4.0)})
    # edema: peak swelling at soft pogonion (mm), log-SD, time-to-peak (days)
    edema_peak_mm: float = 3.5
    edema_log_sd: float = 0.35
    edema_tp_days: float = 2.5
    segmentation_sd_mm: float = 0.3
    # model discrepancy: additive SD along the skin normal (const * w, w -> 0 where the predicted
    # change is < 1 mm) and relative SD of the implant-driven change.  "validated": fitted for 90%
    # coverage on the retrospective pre/post-operative surgery case, which includes a change of lip /
    # mentalis posture between the scans (validation/ortho_case_A_discrepancy.json); "relaxed":
    # same relative term, additive term = rest-posture repeatability (~1 mm), assuming both scans
    # are taken relaxed with lips together.
    uncertainty_profile: str = "validated"
    discrepancy_const_mm: float = 2.86
    discrepancy_rel: float = 0.36

    PROFILES = {"validated": (2.86, 0.36), "relaxed": (1.0, 0.36)}

    def use_profile(self, name: str):
        if name not in self.PROFILES:
            raise ValueError(f"unknown uncertainty profile {name}")
        self.uncertainty_profile = name
        self.discrepancy_const_mm, self.discrepancy_rel = self.PROFILES[name]
    model_form_rel_sd: float = 0.08

    def ratio_prior(self, soft_thickness_mm: float) -> tuple[float, float]:
        r = float(np.interp(soft_thickness_mm, self.ratio_thickness_mm, self.ratio_values))
        # mild extrapolation outside the reported thickness classes
        if soft_thickness_mm < self.ratio_thickness_mm[0]:
            r = min(0.88, r + 0.02 * (self.ratio_thickness_mm[0] - soft_thickness_mm))
        if soft_thickness_mm > self.ratio_thickness_mm[-1]:
            r = max(0.40, r - 0.03 * (soft_thickness_mm - self.ratio_thickness_mm[-1]))
        return r, self.ratio_sd


def swelling_fraction(t_days: float, tp: float = 2.5) -> float:
    """Normalised edema magnitude: peak at ~48-72 h, ~60% resolved by 2 weeks,
    ~85% by 6 weeks, >95% by 3 months, residual < 1% at 6 months."""
    t = float(t_days)
    if t <= 0:
        return 0.0
    if t <= tp:
        return float(np.sin(0.5 * np.pi * t / tp))
    d = t - tp
    return float(0.65 * np.exp(-d / 7.0) + 0.35 * np.exp(-d / 42.0))


TIMEPOINTS = [("Day 2", 2), ("Week 1", 7), ("Week 2", 14), ("Week 6", 42), ("Month 3", 90), ("Month 6", 180),
              ("Year 1 (settled)", 365)]


# --------------------------------------------------------------------------- engine

SOFT_LANDMARKS = ("soft_pogonion", "soft_gnathion", "soft_menton", "soft_b_point", "labrale_inferius", "stomion",
                  "labrale_superius", "subnasale", "cervical_point")


@dataclass
class DesignSolution:
    design: ImplantDesign
    fields: dict                     # name -> (n_nodes, 3)
    t_pog: float                     # implant thickness at bony pogonion (mm)
    pocket_coverage: float           # fraction of implant thickness inside the dissection pocket
    warnings: list
    solve_seconds: float
    nonlinear: bool = False


def _mesh_normals(m, nodes: np.ndarray) -> np.ndarray:
    """Outward normal of the bone interface at mesh nodes: direction from the centroid of the
    node's tissue elements towards the node is into the bone, so the normal is its opposite."""
    sel = np.isin(m.elems, nodes)
    e_idx, a_idx = np.nonzero(sel)
    cen = m.nodes[m.elems[e_idx]].mean(1)
    node_ids = m.elems[e_idx, a_idx]
    acc = np.zeros((m.n_nodes, 3))
    for c in range(3):
        acc[:, c] = np.bincount(node_ids, weights=cen[:, c] - m.nodes[node_ids, c], minlength=m.n_nodes)
    v = acc[nodes]
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


class PredictionEngine:
    def __init__(self, case, params: TissueParams | None = None, priors: ClinicalPriors | None = None):
        self.case = case
        self.model = case.model
        self.anat = case.anatomy
        self.lms = self.anat.landmarks
        self.arch = self.anat.arch
        self.params = params or TissueParams()
        self.priors = priors or ClinicalPriors()
        self.sim = Simulator(self.model, self.params)
        m = self.model
        self.pocket = self.sim.pocket
        self.pocket_xyz = m.nodes[self.pocket]
        self.pocket_n = self.sim.pocket_normals
        # bone-interface nodes outside the pocket (to detect implants larger than the pocket)
        ext = m.sets.get("external_iface", m.sets["mandible_iface"])
        outside = np.setdiff1d(ext, self.pocket)
        self.out_xyz = m.nodes[outside]
        # outward normals of these interface nodes from the hexahedral mesh itself (no volume access)
        self.out_n = _mesh_normals(m, outside) if len(outside) else np.zeros((0, 3))
        # render skin mesh
        skin = case.meshes["skin"]
        self.skin_v = skin.vertices
        self.skin_n = skin.vertex_normals()
        self.W = m.interpolation_matrix(self.skin_v, surface=True).tocsr()
        self.active = np.where(np.diff(self.W.indptr) > 0)[0]
        self.W_act = self.W[self.active]
        names = [n for n in SOFT_LANDMARKS if n in self.lms]
        pts = np.array([self.lms[n] for n in names])
        W = m.interpolation_matrix(pts, surface=True).tocsr()
        # only landmarks that lie on real, modelled skin are predicted (a landmark on a surface cut by
        # the scan's field of view has no skin support and is reported as unavailable, not as 0 mm)
        keep = np.asarray(W.sum(1)).ravel() >= 0.5
        self.unavailable = [n for n, k in zip(names, keep) if not k]
        self.lm_names = [n for n, k in zip(names, keep) if k]
        self.lm_pts = pts[keep]
        self.W_lm = W[np.where(keep)[0]]
        self.i_pog = self.lm_names.index("soft_pogonion")
        self.soft_thickness = float(np.linalg.norm(self.lms["soft_pogonion"] - self.lms["pogonion"]))
        if case.meta.get("anatomy_source") == "estimated":
            # bone and tissue layers estimated from a face scan: unknown chin-pad thickness (SD ~2 mm)
            # widens the evidence prior and the model discrepancy
            pr = self.priors
            pr.ratio_sd = float(np.hypot(pr.ratio_sd, 0.055 * 2.0))
            pr.PROFILES = {k: (c + 0.8, r + 0.1) for k, (c, r) in pr.PROFILES.items()}
            pr.use_profile(pr.uncertainty_profile)
            pr.placement_sd = {k: tuple(x * 1.3 for x in v) for k, v in pr.placement_sd.items()}
        self._edema = None

    # ------------------------------------------------------------------ helpers
    def pocket_thickness(self, d: ImplantDesign, du: float = 0.0, dv: float = 0.0) -> np.ndarray:
        return node_thickness(self.pocket_xyz, self.pocket_n, self.arch, d, du, dv)

    def implant_t_at_pogonion(self, d: ImplantDesign) -> float:
        me = self.lms["menton"]
        pog = self.lms["pogonion"]
        u, v = self.arch.uv(pog[None])
        return float(thickness(np.nan_to_num(u, nan=0.0), v, d)[0])

    def edema_field(self) -> np.ndarray:
        """Unit edema response: volumetric eigenstrain concentrated around the dissected pocket,
        normalised to 1 mm anterior swelling at soft pogonion."""
        if self._edema is None:
            m = self.model
            c = m.elem_centers()
            tree = cKDTree(m.nodes[self.pocket])
            d, _ = tree.query(c)
            w = np.exp(-(d / 10.0) ** 2)
            # intra-oral approach: the labiomental / lower-lip region swells as well
            li = self.lms["labrale_inferius"]
            w += 0.5 * np.exp(-np.sum(((c - li) / np.array([18.0, 10.0, 10.0])) ** 2, axis=1))
            f = self.sim.edema_load(0.1 * w)
            u = self.sim.solve(None, f_extra=f, interface="sliding")
            dy = (self.W_lm @ u)[self.i_pog]
            self._edema = u / max(abs(dy[1]), 1e-6)
        return self._edema

    # ------------------------------------------------------------------ solve
    def solve_design(self, d: ImplantDesign, with_uncertainty: bool = True) -> DesignSolution:
        t0 = time.time()
        warnings = []
        g = self.pocket_thickness(d)
        g_out = node_thickness(self.out_xyz, self.out_n, self.arch, d) if len(self.out_xyz) else np.zeros(1)
        cov = float(g.sum() / max(g.sum() + g_out.sum(), 1e-9))
        if g_out.max() > 0.5:
            warnings.append(f"implant extends beyond the planned subperiosteal pocket "
                            f"(max {g_out.max():.1f} mm outside); the pocket will need a wider dissection")
        fields = {"impl": self.sim.solve(g, interface="sliding")}
        if with_uncertainty:
            fields["tied"] = self.sim.solve_adhered(g) - fields["impl"]
            for k, v in self.sim.sensitivities(fields["impl"]).items():
                fields["sens_" + k] = v
            # placement perturbations (finite differences of the thickness field)
            gv = self.pocket_thickness(d, dv=1.0)
            gl = self.pocket_thickness(d, du=1.0)
            dc = ImplantDesign.from_dict({**d.to_dict(), "cant_deg": d.cant_deg + 1.0})
            gc = self.pocket_thickness(dc)
            G = np.stack([gv - g, gl - g, gc - g], 1)
            ls = self.sim.system("sliding")
            f = np.zeros((self.sim.ndof, 3))
            for c in range(3):
                for j in range(3):
                    np.add.at(f[:, j], 3 * self.pocket + c, ls.k_pen * self.sim.pocket_normals[:, c] * G[:, j])
            X = ls.solver.solve(f[ls.free])
            for j, name in enumerate(("place_vertical", "place_lateral", "place_cant")):
                u = np.zeros(self.sim.ndof)
                u[ls.free] = X[:, j]
                fields[name] = u.reshape(-1, 3)
            fields["edema"] = self.edema_field()
        return DesignSolution(d, fields, self.implant_t_at_pogonion(d), cov, warnings, time.time() - t0)

    def refine_nonlinear(self, sol: DesignSolution, log=None) -> DesignSolution:
        g = self.pocket_thickness(sol.design)
        # the large-deformation correction is a few % of the linear field: 2 load steps, 1e-4 tolerance
        u_nl, info = self.sim.solve_nonlinear(g, interface="sliding", n_steps=2, log=log, tol=1e-4)
        sol.fields["nl"] = u_nl - sol.fields["impl"]
        sol.nonlinear = True
        return sol

    # ------------------------------------------------------------------ ensemble
    def _coefficients(self, sol: DesignSolution, t_days: float, n: int, seed: int, calibrate: bool):
        rng = np.random.default_rng(seed)
        pr = self.priors
        p = self.params
        d = sol.design
        th = {k: rng.normal(0, sd, n) for k, sd in zip(SENS_PARAMS, (p.sd_log_fat, p.sd_log_muscle,
                                                                     p.sd_log_dermis, p.sd_log_bulk))}
        b = (rng.random(n) < p.p_tied).astype(float)
        psd = pr.placement_sd.get(d.fixation, pr.placement_sd["sutures"])
        place = np.stack([rng.normal(0, s, n) for s in psd], 1)
        rm, rs = pr.resorption_mm.get(d.material, (1.0, 0.7))
        P = max(d.projection, d.vertical_extension, 1.0)
        R = np.clip(rng.normal(rm, rs, n), -0.3, 4.0) * np.sqrt(np.clip(P / 6.0, 0.3, 2.5))
        def kres(t):
            return 1.0 - R * (1 - np.exp(-t / pr.resorption_tau_days)) / P
        a = np.exp(rng.normal(np.log(pr.edema_peak_mm), pr.edema_log_sd, n)) * swelling_fraction(t_days, pr.edema_tp_days)
        eps = rng.normal(0, 1.0, n)        # standard normal driver of the surface discrepancy
        coef = {"th": th, "b": b, "place": place, "k": kres(t_days), "a": a, "eps": eps, "k365": kres(365.0)}
        # ---- evidence calibration on the settled (1 year) soft-pogonion ratio
        cal = {"enabled": calibrate}
        Fp = {k: (self.W_lm @ v)[self.i_pog, 1] for k, v in sol.fields.items()}
        impl_y = (Fp["impl"] + Fp.get("nl", 0.0) + sum(th[k] * Fp.get("sens_" + k, 0.0) for k in SENS_PARAMS)
                  + b * Fp.get("tied", 0.0) + place @ np.array([Fp.get("place_vertical", 0.0),
                                                                Fp.get("place_lateral", 0.0),
                                                                Fp.get("place_cant", 0.0)]))
        c = np.ones(n)
        if sol.t_pog > 0.5:
            r_s = impl_y * coef["k365"] / sol.t_pog
            r_fem = float(np.mean(r_s))
            sd_fem = float(np.hypot(np.std(r_s), pr.model_form_rel_sd * abs(r_fem)))
            r_ev, sd_ev = pr.ratio_prior(self.soft_thickness)
            w_f, w_e = 1 / sd_fem ** 2, 1 / sd_ev ** 2
            r_post = (r_fem * w_f + r_ev * w_e) / (w_f + w_e)
            sd_post = (w_f + w_e) ** -0.5
            cal.update({"ratio_fem": r_fem, "ratio_fem_sd": sd_fem, "ratio_evidence": r_ev,
                        "ratio_evidence_sd": sd_ev, "ratio_posterior": r_post, "ratio_posterior_sd": sd_post,
                        "soft_tissue_thickness_mm": self.soft_thickness})
            if calibrate and r_fem > 0.05:
                s_mean = r_post / r_fem
                extra = max(sd_post ** 2 - np.var(r_s * s_mean), 0.0) ** 0.5 / r_fem
                c = rng.normal(s_mean, extra, n) if extra > 0 else np.full(n, s_mean)
                cal["scale"] = s_mean
        # model discrepancy (validation-calibrated): relative part acts multiplicatively on the
        # implant-driven change (log-normal, never reverses its direction); the share of it already
        # represented by the evidence-calibration spread is removed to avoid double counting
        rel_cal = (cal.get("ratio_posterior_sd", 0.0) / max(cal.get("ratio_posterior", 1.0), 1e-6)) if calibrate else 0.0
        rel = float(np.sqrt(max(pr.discrepancy_rel ** 2 - rel_cal ** 2, 0.0)))
        coef["c"] = c * np.exp(rng.normal(-0.5 * rel ** 2, rel, n))
        cal["discrepancy_rel_effective"] = rel
        return coef, cal

    def _project(self, sol: DesignSolution, W):
        """Evaluate every field at the points of interpolation matrix W -> dict name -> (n_pts, 3)."""
        return {k: np.asarray(W @ v) for k, v in sol.fields.items()}

    @staticmethod
    def _combine(F, coef, idx=None):
        """Per-sample displacement (S, n_pts, 3) for the points in F (dict of (n,3))."""
        sl = slice(None) if idx is None else idx
        th, b, place = coef["th"], coef["b"], coef["place"]
        base = F["impl"][sl] + (F["nl"][sl] if "nl" in F else 0)
        impl = base[None] + 0.0
        for k in SENS_PARAMS:
            if "sens_" + k in F:
                impl = impl + th[k][:, None, None] * F["sens_" + k][sl][None]
        if "tied" in F:
            impl = impl + b[:, None, None] * F["tied"][sl][None]
        for j, nm in enumerate(("place_vertical", "place_lateral", "place_cant")):
            if nm in F:
                impl = impl + place[:, j, None, None] * F[nm][sl][None]
        out = (coef["c"] * coef["k"])[:, None, None] * impl
        if "edema" in F:
            out = out + coef["a"][:, None, None] * F["edema"][sl][None]
        return out

    def _sigma(self, u_abs: np.ndarray) -> np.ndarray:
        """Additive model-discrepancy SD (mm) along the skin normal (soft-tissue state / posture and
        surface error); the relative part is applied multiplicatively in ``_coefficients``."""
        pr = self.priors
        w = np.clip(u_abs / 1.0, 0.0, 1.0)
        return np.hypot(pr.discrepancy_const_mm * w, pr.segmentation_sd_mm)

    FIELD_ORDER = ("impl", "sens_fat", "sens_muscle", "sens_dermis", "sens_bulk", "tied",
                   "place_vertical", "place_lateral", "place_cant", "edema")

    def _coef_matrix(self, sol: DesignSolution, coef) -> tuple[list, np.ndarray]:
        """(names, A) with A (S, K) so that sample displacement = A @ stacked fields."""
        ck = coef["c"] * coef["k"]
        names, cols = [], []
        for nm in self.FIELD_ORDER:
            if nm not in sol.fields:
                continue
            if nm == "impl":
                col = ck
            elif nm.startswith("sens_"):
                col = ck * coef["th"][nm[5:]]
            elif nm == "tied":
                col = ck * coef["b"]
            elif nm.startswith("place_"):
                col = ck * coef["place"][:, ("place_vertical", "place_lateral", "place_cant").index(nm)]
            else:
                col = coef["a"]
            names.append(nm)
            cols.append(col)
        if "nl" in sol.fields:
            names.append("nl")
            cols.append(ck)
        return names, np.stack(cols, 1)

    def vertex_fields(self, sol: DesignSolution) -> dict:
        """Fields at the active render vertices (float32), cached on the solution."""
        if getattr(sol, "_vf", None) is None or set(sol._vf) != set(sol.fields):
            from ..imaging.surface import smooth_vertex_field
            from .expressions import fissure_sides
            if not hasattr(self, "_sides"):
                self._sides = fissure_sides(self.case, self.skin_v)
            skin = self.case.meshes["skin"]
            sol._vf = {k: smooth_vertex_field(skin, np.asarray(self.W_act @ v, dtype=np.float32), self.active, 5, 0.5,
                                              self._sides) for k, v in sol.fields.items()}
        return sol._vf

    def predict(self, sol: DesignSolution, t_days: float = 365.0, n_samples: int = 600, seed: int = 7,
                calibrate: bool = True, vertices: bool = True, n_vertex_samples: int = 240) -> dict:
        coef, cal = self._coefficients(sol, t_days, n_samples, seed, calibrate)
        # ---- landmarks (full sample set, kept for metric intervals)
        Fl = self._project(sol, self.W_lm)
        U = self._combine(Fl, coef)                             # (S, L, 3)
        if not hasattr(self, "_lm_normals"):
            from scipy.spatial import cKDTree
            self._lm_normals = self.skin_n[cKDTree(self.skin_v).query(self.lm_pts)[1]]
        un = np.abs(np.einsum("slj,lj->sl", U, self._lm_normals)).mean(0)
        sig = self._sigma(un)
        # P: outcome of the surgery (implant effect incl. all surgical/biological uncertainty);
        # Pt: additionally the resting-posture / surface variability between two occasions
        P = self.lm_pts[None] + U
        Pt = P + (coef["eps"][:, None] * sig[None])[..., None] * self._lm_normals[None]
        lm = {}
        for i, n in enumerate(self.lm_names):
            lm[n] = {"pre": self.lm_pts[i].tolist(), "median": np.median(P[:, i], 0).tolist(),
                     "p05": np.percentile(P[:, i], 5, 0).tolist(), "p95": np.percentile(P[:, i], 95, 0).tolist(),
                     "p05_total": np.percentile(Pt[:, i], 5, 0).tolist(),
                     "p95_total": np.percentile(Pt[:, i], 95, 0).tolist(),
                     "samples": P[:, i].astype(np.float32)}
        out = {"t_days": t_days, "calibration": cal, "landmarks": lm, "warnings": list(sol.warnings),
               "unavailable_landmarks": self.unavailable,
               "swelling_fraction": swelling_fraction(t_days), "n_samples": n_samples,
               "posture_sd_mm": float(self.priors.discrepancy_const_mm),
               "uncertainty_profile": self.priors.uncertainty_profile}
        if vertices:
            Fv = self.vertex_fields(sol)
            names, A = self._coef_matrix(sol, coef)
            nrm = self.skin_n[self.active].astype(np.float32)
            Fstack = np.stack([Fv[k] for k in names], 0)                       # (K, N, 3)
            Nk = np.einsum("knj,nj->kn", Fstack, nrm)                          # (K, N)
            mean_coef = A.mean(0).astype(np.float32)
            mean = np.einsum("k,knj->nj", mean_coef, Fstack)                   # likely surface
            As = A[:n_vertex_samples].astype(np.float32)
            mn = Nk.T @ mean_coef
            # surface bands show the surgical outcome range; posture variability is reported apart
            Sn = As @ Nk + coef["eps"][:n_vertex_samples, None].astype(np.float32) * \
                self.priors.segmentation_sd_mm                                     # (S, N)
            lo = np.percentile(Sn, 5, axis=0) - mn
            hi = np.percentile(Sn, 95, axis=0) - mn
            V = len(self.skin_v)
            disp = np.zeros((V, 3), np.float32)
            disp[self.active] = mean
            low = np.zeros(V, np.float32)
            high = np.zeros(V, np.float32)
            low[self.active] = lo
            high[self.active] = hi
            out.update({"disp": disp, "low": low, "high": high})
        return out
