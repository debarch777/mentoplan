"""Solvers on a patient TissueModel.

Units: lengths in mm, moduli in kPa (forces in mN).

The implant is applied as a prescribed *normal* displacement g of the bone
interface nodes inside the subperiosteal pocket.  Two interface hypotheses
are kept:

* ``sliding`` – the elevated periosteum/mentalis slides freely on the implant
  and bone (penalty on the normal component only); this is the acute
  post-operative state;
* ``tied``    – full adhesion (displacement g*n prescribed); represents a
  tight pocket / capsule adhesion.

Material parameters are uncertain; first-order sensitivities of the solution
with respect to the log-moduli of each tissue class and the log of the
bulk-to-shear ratio are obtained by direct differentiation
(K du/dtheta = -dK/dtheta u) using the same factorisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .hexfem import (ElementGeom, Material, SPDSolver, eigenstrain_load, newton_solve, nonlinear_element_terms,
                     pair_penalty_matrix, penalty_matrix)
from .model import MAT_FAT, MAT_MUCOSA, MAT_MUSCLE, MAT_SKIN, TissueModel


@dataclass
class TissueParams:
    """Median material parameters (kPa) and log-normal spreads.

    Sources: in-vivo indentation / elastography of facial tissues and
    parameter ranges used in published facial soft-tissue FE models
    (e.g. Chabanas 2003; Barbarino 2009; Mollemans 2007 mass-tensor model).
    Displacement-driven predictions depend only on stiffness *ratios* and on
    compressibility, which is why these spreads matter more than the absolute
    values.
    """
    E_fat: float = 2.5
    E_muscle: float = 9.0
    E_dermis: float = 60.0
    E_mucosa: float = 15.0
    nu: float = 0.475
    sd_log_fat: float = 0.6
    sd_log_muscle: float = 0.5
    sd_log_dermis: float = 0.5
    sd_log_bulk: float = 0.6        # on log(kappa/mu); covers nu ~ 0.45 .. 0.49
    p_tied: float = 0.25            # prior probability of the adhered interface hypothesis

    def material(self, model: TissueModel, scale: dict | None = None) -> Material:
        s = scale or {}
        E = np.empty(len(model.elems))
        E[model.mat_id == MAT_FAT] = self.E_fat * s.get("fat", 1.0)
        E[model.mat_id == MAT_MUSCLE] = self.E_muscle * s.get("muscle", 1.0)
        E[model.mat_id == MAT_SKIN] = self.E_dermis * s.get("dermis", 1.0)
        E[model.mat_id == MAT_MUCOSA] = self.E_mucosa * s.get("muscle", 1.0)
        m = Material.from_E_nu(E, self.nu)
        m.kappa = m.kappa * s.get("bulk", 1.0)
        return m


SENS_PARAMS = ("fat", "muscle", "dermis", "bulk")


@dataclass
class LinearSystem:
    interface: str
    K: sp.csr_matrix            # full stiffness incl. penalties
    free: np.ndarray
    fixed: np.ndarray
    k_pen: float
    slide_nodes: np.ndarray
    slide_normals: np.ndarray
    _solver: SPDSolver | None = None

    _Kff: sp.csr_matrix | None = None

    @property
    def Kff(self) -> sp.csr_matrix:
        if self._Kff is None:
            self._Kff = self.K[self.free][:, self.free].tocsr()
        return self._Kff

    @property
    def solver(self) -> SPDSolver:
        """Factorisation of the free-free block, built on first use (the nonlinear path never needs it)."""
        if self._solver is None:
            self._solver = SPDSolver(self.Kff)
        return self._solver


class Simulator:
    def __init__(self, model: TissueModel, params: TissueParams | None = None):
        self.model = model
        self.params = params or TissueParams()
        self.mat = self.params.material(model)
        self.geom = model.geom
        self.asm = model.asm
        self.ndof = 3 * model.n_nodes
        self.k_pen = 1e3 * float(np.max(self.mat.mu)) * model.h
        self._systems: dict[str, LinearSystem] = {}
        pocket = model.sets["pocket"]
        self.pocket = pocket
        self.pocket_normals = model.normals["pocket"]

    # ------------------------------------------------------------------ assembly
    def _Klin(self, mat: Material) -> sp.csr_matrix:
        return self.asm.linear_matrix(self.geom, mat)

    def system(self, interface: str = "sliding") -> LinearSystem:
        if interface in self._systems:
            return self._systems[interface]
        m = self.model
        K = self._Klin(self.mat)
        fixed_nodes = m.sets["fixed"]
        slide_nodes = m.sets["teeth_slide"]
        slide_normals = m.normals["teeth_slide"]
        if interface == "sliding":
            slide_nodes = np.r_[slide_nodes, self.pocket]
            slide_normals = np.vstack([slide_normals, self.pocket_normals])
        else:
            fixed_nodes = np.r_[fixed_nodes, self.pocket]
        # a node can only be in one constraint class
        slide_keep = ~np.isin(slide_nodes, fixed_nodes)
        slide_nodes, slide_normals = slide_nodes[slide_keep], slide_normals[slide_keep]
        K = K + penalty_matrix(m.n_nodes, slide_nodes, slide_normals, self.k_pen)
        fixed = np.unique((3 * fixed_nodes[:, None] + np.arange(3)).ravel())
        free = np.setdiff1d(np.arange(self.ndof), fixed)
        ls = LinearSystem(interface, K.tocsr(), free, fixed, self.k_pen, slide_nodes, slide_normals)
        self._systems[interface] = ls
        return ls

    def dK_matvec(self, name: str, u: np.ndarray) -> np.ndarray:
        """(dK / d log theta) @ u for a parameter group, element by element (never assembled:
        four assembled derivative matrices would cost >1 GB on a typical model)."""
        m = self.model
        if name == "bulk":
            idx = np.arange(len(m.elems))
            mu = np.zeros_like(self.mat.mu)
        else:
            ids = {"fat": [MAT_FAT], "muscle": [MAT_MUSCLE, MAT_MUCOSA], "dermis": [MAT_SKIN]}[name]
            idx = np.where(np.isin(m.mat_id, ids))[0]
            mu = self.mat.mu
        ka = self.mat.kappa
        uf = u.ravel()
        out = np.zeros(self.ndof)
        g = self.geom
        for s0 in range(0, len(idx), 6000):
            e = idx[s0:s0 + 6000]
            sub = ElementGeom(g.dN[e], g.w[e], g.dNc[e], g.wc[e])
            Ke = sub.linear_K(mu[e], ka[e])
            dofs = self.asm.dofs[e]
            fe = np.einsum("eij,ej->ei", Ke, uf[dofs])
            out += np.bincount(dofs.ravel(), weights=fe.ravel(), minlength=self.ndof)
        return out

    # ------------------------------------------------------------------ linear solves
    def _rhs(self, ls: LinearSystem, g_pocket: np.ndarray | None, f_extra: np.ndarray | None):
        u_fixed = np.zeros(len(ls.fixed))
        f = np.zeros(self.ndof) if f_extra is None else f_extra.copy()
        if g_pocket is not None:
            if ls.interface == "sliding":
                # penalty rhs: k n g on pocket nodes
                for c in range(3):
                    np.add.at(f, 3 * self.pocket + c, ls.k_pen * self.pocket_normals[:, c] * g_pocket)
            else:
                ufull = np.zeros(self.ndof)
                for c in range(3):
                    ufull[3 * self.pocket + c] = self.pocket_normals[:, c] * g_pocket
                u_fixed = ufull[ls.fixed]
        return f, u_fixed

    def solve(self, g_pocket: np.ndarray | None = None, f_extra: np.ndarray | None = None,
              interface: str = "sliding") -> np.ndarray:
        """Linear static solution; returns nodal displacements (n_nodes, 3)."""
        ls = self.system(interface)
        f, u_fixed = self._rhs(ls, g_pocket, f_extra)
        u = np.zeros(self.ndof)
        u[ls.fixed] = u_fixed
        rhs = f[ls.free] - ls.K[ls.free][:, ls.fixed] @ u_fixed if np.any(u_fixed) else f[ls.free]
        u[ls.free] = ls.solver.solve(rhs)
        return u.reshape(-1, 3)

    def solve_adhered(self, g_pocket: np.ndarray, k_rel: float = 20.0, rtol: float = 1e-5) -> np.ndarray:
        """Adhered-pocket hypothesis without a second factorisation: the sliding system plus
        tangential springs (k_rel x stiffest shear modulus x h) at the pocket nodes, solved by CG
        preconditioned with the sliding factorisation, started from the sliding solution."""
        from scipy.sparse.linalg import LinearOperator, cg
        ls = self.system("sliding")
        n = self.pocket_normals
        a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
        t1 = np.cross(n, a)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(n, t1)
        kt = k_rel * float(np.max(self.mat.mu)) * self.model.h
        Pt = (penalty_matrix(self.model.n_nodes, self.pocket, t1, kt) +
              penalty_matrix(self.model.n_nodes, self.pocket, t2, kt))[ls.free][:, ls.free].tocsr()
        f, _ = self._rhs(ls, g_pocket, None)
        x0 = self.solve(g_pocket, interface="sliding").ravel()[ls.free]
        A = LinearOperator(ls.Kff.shape, matvec=lambda x: ls.Kff @ x + Pt @ x, dtype=float)
        M = LinearOperator(ls.Kff.shape, matvec=lambda x: ls.solver.solve(np.ascontiguousarray(x)), dtype=float)
        x, _ = cg(A, f[ls.free], x0=x0, M=M, rtol=rtol, maxiter=400)
        u = np.zeros(self.ndof)
        u[ls.free] = x
        return u.reshape(-1, 3)

    def sensitivities(self, u: np.ndarray, interface: str = "sliding") -> dict:
        """du/dlog(theta) for every material parameter group, given the solution u."""
        ls = self.system(interface)
        out = {}
        uf = u.ravel()
        rhs = np.stack([-self.dK_matvec(n, uf)[ls.free] for n in SENS_PARAMS], 1)
        X = ls.solver.solve(rhs)
        for i, n in enumerate(SENS_PARAMS):
            d = np.zeros(self.ndof)
            d[ls.free] = X[:, i]
            out[n] = d.reshape(-1, 3)
        return out

    # ------------------------------------------------------------------ loads
    def edema_load(self, vol_strain: np.ndarray) -> np.ndarray:
        return eigenstrain_load(self.geom, self.asm.dofs, self.ndof, self.mat, vol_strain=vol_strain)

    def active_load(self, dev_strain: np.ndarray) -> np.ndarray:
        """dev_strain: (E, 3, 3) traceless eigenstrain tensors."""
        return eigenstrain_load(self.geom, self.asm.dofs, self.ndof, self.mat, dev_strain=dev_strain)

    # ------------------------------------------------------------------ nonlinear
    def solve_nonlinear(self, g_pocket: np.ndarray, interface: str = "sliding", n_steps: int = 4,
                        contact: bool = True, f_extra: np.ndarray | None = None, log=None,
                        tol: float = 1e-5) -> tuple[np.ndarray, dict]:
        """Total-Lagrangian neo-Hookean solution with penalty sliding and lip contact."""
        m = self.model
        ls = self.system(interface)
        P_slide = penalty_matrix(m.n_nodes, ls.slide_nodes, ls.slide_normals, self.k_pen)
        f_slide_full, u_fixed = self._rhs(ls, g_pocket, None)
        pairs = m.pairs if contact else np.zeros((0, 2), int)
        ez = np.array([0, 0, 1.0])
        state = {"active": np.zeros(len(pairs), bool)}
        f_ext = np.zeros(self.ndof) if f_extra is None else f_extra
        lam = {"s": 1.0}

        def residual_tangent(u, want_tangent=True):
            Ue = u[self.asm.dofs].reshape(-1, 8, 3)
            fe, Ke = nonlinear_element_terms(self.geom, Ue, self.mat, want_tangent)
            r = self.asm.vector(fe) + P_slide @ u - lam["s"] * (f_slide_full + f_ext)
            K = self.asm.matrix(Ke) + P_slide if want_tangent else None
            if len(pairs):
                uu = u.reshape(-1, 3)
                gap = (uu[pairs[:, 1]] - uu[pairs[:, 0]]) @ ez     # >0: lower lip penetrates upper
                act = gap > 0
                state["active"] = act
                if act.any():
                    pa = pairs[act]
                    Pc = pair_penalty_matrix(m.n_nodes, pa, ez, self.k_pen)
                    r = r + Pc @ u
                    if K is not None:
                        K = K + Pc
            return r, K

        # Newton-Krylov with load continuation.  Each tangent system is solved by preconditioned CG
        # using the *linear* stiffness factorisation (computed once, reused) as preconditioner, so the
        # memory stays at about one factorisation; the linear solution, scaled by the load factor,
        # is an excellent initial guess (the nonlinear correction is a few % of the displacement).
        from scipy.sparse.linalg import LinearOperator, cg, minres
        free, fixed = ls.free, ls.fixed
        u_lin = self.solve(g_pocket, f_extra=f_extra, interface=interface).ravel()
        pre = ls.solver
        M = LinearOperator((len(free), len(free)), matvec=lambda x: pre.solve(np.ascontiguousarray(x)),
                           dtype=float)
        hist_all = []

        def newton_krylov(u, lam_s, max_it=20):
            lam["s"] = lam_s
            r, K = residual_tangent(u)
            fref = max(float(np.linalg.norm(lam_s * (f_slide_full + f_ext))), float(np.linalg.norm(r[fixed])), 1.0)
            for it in range(max_it):
                rn = float(np.linalg.norm(r[free]))
                hist_all.append((lam_s, it, rn / fref))
                if rn <= tol * fref:
                    return u, True
                Kff = K[free][:, free].tocsr()
                du, info = cg(Kff, -r[free], M=M, rtol=1e-3, maxiter=60)
                if info != 0:
                    du, info = minres(Kff, -r[free], M=M, rtol=1e-3, maxiter=100)
                del Kff, K
                alpha, accepted = 1.0, False
                while alpha > 1e-3:
                    trial = u.copy()
                    trial[free] += alpha * du
                    try:
                        r2, _ = residual_tangent(trial, False)
                        if np.linalg.norm(r2[free]) < rn * (1 - 1e-4 * alpha):
                            accepted = True
                            break
                    except FloatingPointError:
                        pass
                    alpha *= 0.5
                if not accepted:
                    return u, False
                u = trial
                r, K = residual_tangent(u)
            return u, False

        steps = [1.0]
        for attempt in range(4):
            u = np.zeros(self.ndof)
            lam_prev = 0.0
            ok = True
            for lam_s in steps:
                guess = u + (lam_s - lam_prev) * u_lin
                try:
                    u, conv = newton_krylov(guess, lam_s)
                except FloatingPointError:
                    ok = False
                    break
                if not conv and lam_s < 1.0:
                    ok = False
                    break
                lam_prev = lam_s
            if ok:
                break
            steps = list(np.linspace(0, 1, 2 * len(steps) + 1)[1:])
        else:
            raise RuntimeError("nonlinear solution failed: the design deforms the tissue beyond what the "
                               "model can represent (element inversion)")
        if hist_all and hist_all[-1][2] > tol and log:
            log(f"nonlinear: stopped at relative residual {hist_all[-1][2]:.1e}")
        info = {"history": hist_all, "contact_active": int(state["active"].sum()), "method": "Newton-Krylov"}
        return u.reshape(-1, 3), info
