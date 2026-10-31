"""Voxel hexahedral finite elements for facial soft tissue.

Element: trilinear 8-node hexahedron on a regular grid (side ``h``) with
*selective reduced integration* – deviatoric terms at 2x2x2 Gauss points,
volumetric terms at the element centre – which removes volumetric locking
for nearly incompressible tissue (Poisson's ratio 0.45-0.495) while keeping
the element free of hourglass modes.

Two constitutive paths share the same mesh:

* linear elasticity (small strain) – used for the reduced response basis,
  edema/expression eigenstrain loads and Monte-Carlo sampling;
* compressible decoupled neo-Hookean, total Lagrangian, Newton-Raphson with
  load stepping – used for the full nonlinear solution of the implant
  displacement problem (large deformation of the chin pad).

Constraints:
* Dirichlet (fixed bone attachment, ROI cut boundary) by elimination;
* sliding (normal displacement prescribed, tangential free) by penalty –
  subperiosteal pocket over the implant, lip mucosa on the teeth;
* unilateral node-to-node contact (penalty, active set) across the oral
  fissure, where the lips are split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

# local node a <-> (dx, dy, dz) with a = dx + 2*dy + 4*dz
LOCAL = np.array([[a & 1, (a >> 1) & 1, (a >> 2) & 1] for a in range(8)])
XI = 2.0 * LOCAL - 1.0
GP = XI / np.sqrt(3.0)          # 2x2x2 Gauss points in natural coordinates


def natural_grads():
    """dN/dxi at the 8 Gauss points (8, 8, 3) and at the centre (8, 3)."""
    def grads(xi):
        g = np.empty((8, 3))
        for a in range(8):
            for i in range(3):
                t = 0.125 * XI[a, i]
                for j in range(3):
                    if j != i:
                        t *= 1 + XI[a, j] * xi[j]
                g[a, i] = t
        return g
    return np.stack([grads(x) for x in GP]), grads(np.zeros(3))


DNXI, DNXI_C = natural_grads()


@dataclass
class ElementGeom:
    """Per-element reference geometry: physical shape-function gradients and weights.

    dN : (E, 8 gp, 8 nodes, 3)  dN/dX at the Gauss points
    w  : (E, 8)                 Gauss weight x det J
    dNc: (E, 8, 3)              volume-averaged dN/dX (mean dilatation)
    wc : (E,)                   element volume
    """
    dN: np.ndarray
    w: np.ndarray
    dNc: np.ndarray
    wc: np.ndarray

    @property
    def n(self):
        return len(self.w)

    @staticmethod
    def from_nodes(X: np.ndarray, elems: np.ndarray, min_rel_det: float = 0.0) -> "ElementGeom":
        Xe = X[elems]                                              # (E, 8, 3)
        J = np.einsum("eai,gaj->egij", Xe, DNXI)                   # (E, 8, 3, 3)
        det = np.linalg.det(J)
        if np.any(det <= min_rel_det):
            raise ValueError(f"{int((det <= 0).any(1).sum())} inverted elements")
        Jinv = np.linalg.inv(J)
        dN = np.einsum("gaj,egjk->egak", DNXI, Jinv)
        # mean-dilatation (uniform gradient, Flanagan & Belytschko 1981): volume-averaged
        # gradients make the one-point volumetric term pass the patch test on distorted hexes
        V = det.sum(1)
        dNc = np.einsum("eg,egak->eak", det, dN) / V[:, None, None]
        return ElementGeom(dN, det, dNc, V)

    @staticmethod
    def jacobian_quality(X: np.ndarray, elems: np.ndarray) -> np.ndarray:
        """min over Gauss points and centre of det J / (mean edge length/2)^3 (1 for a cube)."""
        Xe = X[elems]
        J = np.einsum("eai,gaj->egij", Xe, DNXI)
        det = np.linalg.det(J)
        Jc = np.linalg.det(np.einsum("eai,aj->eij", Xe, DNXI_C))
        edges = np.linalg.norm(Xe[:, [1, 3, 5, 7, 2, 3, 6, 7, 4, 5, 6, 7]] -
                               Xe[:, [0, 2, 4, 6, 0, 1, 4, 5, 0, 1, 2, 3]], axis=2).mean(1)
        return np.minimum(det.min(1), Jc) / (edges / 2) ** 3

    def linear_K(self, mu: np.ndarray, kappa: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """Element stiffness (E, 24, 24): SRI isotropic elasticity."""
        E = self.n
        out = np.empty((E, 24, 24))
        I3 = np.eye(3)
        for s in range(0, E, chunk):
            sl = slice(s, min(E, s + chunk))
            G = self.dN[sl]
            w = self.w[sl]
            gg = np.einsum("egak,egbk,eg->eab", G, G, w)
            K = np.einsum("eab,ij->eaibj", gg, I3)
            K += np.einsum("egaj,egbi,eg->eaibj", G, G, w)
            K -= (2.0 / 3.0) * np.einsum("egai,egbj,eg->eaibj", G, G, w)
            K *= mu[sl, None, None, None, None]
            Gc = self.dNc[sl]
            K += (kappa[sl] * self.wc[sl])[:, None, None, None, None] * np.einsum("eai,ebj->eaibj", Gc, Gc)
            out[sl] = K.reshape(-1, 24, 24)
        return out

    @staticmethod
    def regular(h: float, n_elems: int) -> "ElementGeom":
        X = np.array([[a & 1, (a >> 1) & 1, (a >> 2) & 1] for a in range(8)], float) * h
        g = ElementGeom.from_nodes(X, np.arange(8)[None])
        rep = lambda a: np.broadcast_to(a, (n_elems,) + a.shape[1:])
        return ElementGeom(rep(g.dN), rep(g.w), rep(g.dNc), rep(g.wc))


@dataclass
class Material:
    """Per-element shear modulus mu (Pa) and bulk modulus kappa (Pa)."""
    mu: np.ndarray
    kappa: np.ndarray

    @staticmethod
    def from_E_nu(E: np.ndarray, nu) -> "Material":
        nu = np.broadcast_to(np.asarray(nu, float), np.shape(E))
        mu = E / (2 * (1 + nu))
        kappa = E / (3 * (1 - 2 * nu))
        return Material(np.asarray(mu, float), np.asarray(kappa, float))


@dataclass
class Constraints:
    fixed_dofs: np.ndarray                       # Dirichlet dofs
    fixed_values: np.ndarray                     # prescribed values (same length)
    slide_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    slide_normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    slide_gaps: np.ndarray = field(default_factory=lambda: np.zeros(0))   # prescribed normal displacement
    contact_pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))  # (upper, lower)
    contact_normal: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1.0]))
    penalty: float = 0.0


class Assembler:
    """Sparse assembly with a precomputed CSR pattern (fast repeated assembly)."""

    def __init__(self, elems: np.ndarray, n_nodes: int):
        self.elems = elems
        self.n_nodes = n_nodes
        self.ndof = 3 * n_nodes
        dofs = (3 * elems[:, :, None] + np.arange(3)[None, None, :]).reshape(len(elems), 24)
        self.dofs = dofs
        # one int64 key per element-matrix entry (row-major); unique -> CSR order.  Built in one pass
        # and the index map stored as int32 to keep the transient memory of large models low.
        keys = (dofs.astype(np.int64)[:, :, None] * self.ndof + dofs[:, None, :]).ravel()
        uniq, inv = np.unique(keys, return_inverse=True)
        del keys
        self.map = inv.astype(np.int32 if len(uniq) < 2 ** 31 else np.int64)
        del inv
        self.nnz = len(uniq)
        r = uniq // self.ndof
        self.indices = (uniq - r * self.ndof).astype(np.int32)
        self.indptr = np.r_[0, np.cumsum(np.bincount(r, minlength=self.ndof))].astype(np.int32)
        del uniq, r

    def matrix(self, Ke: np.ndarray) -> sp.csr_matrix:
        """Ke: (E, 24, 24)."""
        data = np.bincount(self.map, weights=Ke.reshape(-1), minlength=self.nnz)
        return sp.csr_matrix((data, self.indices.copy(), self.indptr.copy()), shape=(self.ndof, self.ndof))

    def linear_matrix(self, geom: "ElementGeom", mat: Material) -> sp.csr_matrix:
        return self.matrix(geom.linear_K(mat.mu, mat.kappa))

    def vector(self, fe: np.ndarray) -> np.ndarray:
        return np.bincount(self.dofs.ravel(), weights=fe.reshape(-1), minlength=self.ndof)


# --------------------------------------------------------------------------- loads

def eigenstrain_load(geom: ElementGeom, elems_dofs: np.ndarray, ndof: int, mat: Material,
                     vol_strain: np.ndarray | None = None, dev_strain: np.ndarray | None = None) -> np.ndarray:
    """Equivalent nodal forces of an element-wise eigenstrain (f = int B^T C eps* dV).

    vol_strain: (E,) isotropic volumetric eigenstrain (e.g. edema fluid volume fraction)
    dev_strain: (E, 3, 3) traceless eigenstrain tensor (e.g. active muscle contraction).
    """
    E = len(elems_dofs)
    fe = np.zeros((E, 8, 3))
    if vol_strain is not None:
        fe += (mat.kappa * vol_strain * geom.wc)[:, None, None] * geom.dNc
    if dev_strain is not None:
        sig = 2.0 * mat.mu[:, None, None] * dev_strain
        fe += np.einsum("eij,egaj,eg->eai", sig, geom.dN, geom.w)
    return np.bincount(elems_dofs.ravel(), weights=fe.reshape(-1), minlength=ndof)


def body_force_load(geom: ElementGeom, elems: np.ndarray, n_nodes: int, density: np.ndarray,
                    g: np.ndarray) -> np.ndarray:
    """Nodal forces of a uniform body force density*g per element (lumped)."""
    fe = (density * geom.wc / 8.0)[:, None, None] * np.broadcast_to(g, (len(elems), 8, 3))
    f = np.zeros((n_nodes, 3))
    for c in range(3):
        f[:, c] = np.bincount(elems.ravel(), weights=fe[:, :, c].ravel(), minlength=n_nodes)
    return f.ravel()


# --------------------------------------------------------------------------- nonlinear neo-Hookean

def _neohooke(F: np.ndarray, mu: np.ndarray, want_tangent=True):
    """Isochoric neo-Hookean first PK stress and tangent. F: (..., 3, 3), mu broadcastable (...)."""
    J = np.linalg.det(F)
    G = np.linalg.inv(F).swapaxes(-1, -2)               # F^-T
    a = J ** (-2.0 / 3.0)
    I1 = np.einsum("...ij,...ij->...", F, F)
    mu_a = (mu * a)[..., None, None]
    P = mu_a * (F - (I1 / 3.0)[..., None, None] * G)
    if not want_tangent:
        return P, None, J
    I = np.eye(3)
    m = (mu * a)[..., None, None, None, None]
    A = (np.einsum("ik,JL->iJkL", I, I)[None]
         - (2.0 / 3.0) * np.einsum("...iJ,...kL->...iJkL", F, G)
         - (2.0 / 3.0) * np.einsum("...iJ,...kL->...iJkL", G, F)
         + (2.0 / 9.0) * I1[..., None, None, None, None] * np.einsum("...iJ,...kL->...iJkL", G, G)
         + (I1 / 3.0)[..., None, None, None, None] * np.einsum("...iL,...kJ->...iJkL", G, G))
    return P, m * A, J


def _volumetric(F: np.ndarray, kappa: np.ndarray, want_tangent=True):
    """U(J) = kappa/2 (ln J)^2 : P = kappa lnJ F^-T, A = kappa (G x G - lnJ G (x) G)."""
    J = np.linalg.det(F)
    if np.any(J <= 0):
        raise FloatingPointError("element inversion (J <= 0)")
    G = np.linalg.inv(F).swapaxes(-1, -2)
    lnJ = np.log(J)
    P = (kappa * lnJ)[..., None, None] * G
    if not want_tangent:
        return P, None
    k = kappa[..., None, None, None, None]
    A = k * (np.einsum("...iJ,...kL->...iJkL", G, G)
             - lnJ[..., None, None, None, None] * np.einsum("...iL,...kJ->...iJkL", G, G))
    return P, A


def nonlinear_element_terms(geom: ElementGeom, Ue: np.ndarray, mat: Material, want_tangent=True, chunk=5000):
    """Internal forces (E, 24) and tangents (E, 24, 24) for element displacements Ue (E, 8, 3)."""
    E = len(Ue)
    fe = np.empty((E, 24))
    Ke = np.empty((E, 24, 24)) if want_tangent else None
    I = np.eye(3)
    for s in range(0, E, chunk):
        sl = slice(s, min(E, s + chunk))
        U = Ue[sl]
        dN = geom.dN[sl]
        w = geom.w[sl]
        dNc = geom.dNc[sl]
        wc = geom.wc[sl]
        mu = mat.mu[sl]
        kap = mat.kappa[sl]
        # deviatoric part at 8 Gauss points
        F = I + np.einsum("eai,egaJ->egiJ", U, dN)
        if np.any(np.linalg.det(F) <= 0):
            raise FloatingPointError("element inversion (J <= 0)")
        P, A, J = _neohooke(F, mu[:, None], want_tangent)
        f = np.einsum("egiJ,egaJ,eg->eai", P, dN, w)
        # volumetric part at the centre
        Fc = I + np.einsum("eai,eaJ->eiJ", U, dNc)
        Pc, Ac = _volumetric(Fc, kap, want_tangent)
        f += np.einsum("eiJ,eaJ,e->eai", Pc, dNc, wc)
        fe[sl] = f.reshape(-1, 24)
        if want_tangent:
            n = U.shape[0]
            # T[e,g,a,(i,k,L)] = sum_J dN[e,g,a,J] A[e,g,i,J,k,L] w   -- batched GEMM over (e,g)
            Aj = (A * w[:, :, None, None, None, None]).transpose(0, 1, 3, 2, 4, 5).reshape(n * 8, 3, 27)
            T = np.matmul(dN.reshape(n * 8, 8, 3), Aj).reshape(n, 8, 8, 3, 3, 3)          # e,g,a,i,k,L
            # K[e,(a,i,k),b] = sum_{g,L} T[e,g,a,i,k,L] dN[e,g,b,L]
            T2 = T.transpose(0, 2, 3, 4, 1, 5).reshape(n, 72, 24)
            D2 = dN.transpose(0, 1, 3, 2).reshape(n, 24, 8)
            K = np.matmul(T2, D2).reshape(n, 8, 3, 3, 8).transpose(0, 1, 2, 4, 3)            # e,a,i,b,k
            Acw = (Ac * wc[:, None, None, None, None]).transpose(0, 2, 1, 3, 4).reshape(n, 3, 27)
            Tc = np.matmul(dNc, Acw).reshape(n, 8, 3, 3, 3)                                  # e,a,i,k,L
            Kc = np.matmul(Tc.reshape(n, 72, 3), dNc.transpose(0, 2, 1)).reshape(n, 8, 3, 3, 8)
            K = K + Kc.transpose(0, 1, 2, 4, 3)
            Ke[sl] = K.reshape(-1, 24, 24)
    return fe, Ke


# --------------------------------------------------------------------------- linear solver back-end

class SPDSolver:
    """Factorise once, solve many right-hand sides.  PARDISO when available, else SuperLU."""

    def __init__(self, A: sp.csr_matrix, symmetric_indefinite: bool = False):
        self.n = A.shape[0]
        try:
            import pypardiso
            self._A = sp.triu(A, format="csr")
            self._A.sort_indices()
            self._s = pypardiso.PyPardisoSolver(mtype=2)
            try:
                # SPD Cholesky first (2-3x faster); fall back to Bunch-Kaufman LDL^T if not definite
                self._s.factorize(self._A)
            except Exception:
                if not symmetric_indefinite:
                    raise
                self._s = pypardiso.PyPardisoSolver(mtype=-2)
                self._s.factorize(self._A)
            self.kind = "pardiso"
        except Exception:
            from scipy.sparse.linalg import splu
            self._lu = splu(A.tocsc())
            self.kind = "superlu"

    def solve(self, b: np.ndarray) -> np.ndarray:
        if self.kind == "pardiso":
            x = self._s.solve(self._A, np.ascontiguousarray(b))
        else:
            x = self._lu.solve(b)
        return x

    def free(self):
        if self.kind == "pardiso":
            try:
                self._s.free_memory(everything=True)
            except Exception:
                pass


def penalty_matrix(n_nodes: int, nodes: np.ndarray, normals: np.ndarray, k: float) -> sp.csr_matrix:
    """sum_i k n_i n_i^T on the 3x3 diagonal blocks of the given nodes."""
    rows, cols, vals = [], [], []
    for a in range(3):
        for b in range(3):
            rows.append(3 * nodes + a)
            cols.append(3 * nodes + b)
            vals.append(k * normals[:, a] * normals[:, b])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(3 * n_nodes, 3 * n_nodes))


def pair_penalty_matrix(n_nodes: int, pairs: np.ndarray, normal: np.ndarray, k: float) -> sp.csr_matrix:
    """k (n.(u_b - u_a))^2 / 2 for each pair (a, b)."""
    if len(pairs) == 0:
        return sp.csr_matrix((3 * n_nodes, 3 * n_nodes))
    a_, b_ = pairs[:, 0], pairs[:, 1]
    nn = np.outer(normal, normal)
    rows, cols, vals = [], [], []
    for p, q, s in ((a_, a_, 1), (b_, b_, 1), (a_, b_, -1), (b_, a_, -1)):
        for i in range(3):
            for j in range(3):
                rows.append(3 * p + i)
                cols.append(3 * q + j)
                vals.append(np.full(len(p), s * k * nn[i, j]))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(3 * n_nodes, 3 * n_nodes))


# --------------------------------------------------------------------------- nonlinear driver

def newton_solve(residual_tangent, ndof: int, fixed: np.ndarray, fixed_target: np.ndarray,
                 u0: np.ndarray | None = None, n_steps: int = 4, tol: float = 1e-6, max_iter: int = 25,
                 log=None):
    """Incremental Newton-Raphson with Dirichlet predictor and backtracking line search.

    ``residual_tangent(u, want_tangent=True) -> (r, K)`` returns the out-of-balance force
    vector (internal - external) and its consistent tangent (None if not wanted); it may raise
    FloatingPointError on element inversion, which triggers step halving.
    Convergence: ||r_free|| <= tol * max(||f_ref||, 1).
    """
    u = np.zeros(ndof) if u0 is None else u0.copy()
    free = np.setdiff1d(np.arange(ndof), fixed)
    start = u[fixed].copy()
    history = []
    for s in range(1, n_steps + 1):
        target = start + (fixed_target - start) * s / n_steps
        r, K = residual_tangent(u)
        dup = target - u[fixed]
        # predictor: spread the prescribed increment into the interior with the current tangent
        rhs = -(r[free] + K[free][:, fixed] @ dup)
        solver = SPDSolver(K[free][:, free].tocsr(), symmetric_indefinite=True)
        duf = solver.solve(rhs)
        solver.free()
        alpha = 1.0
        while True:
            trial = u.copy()
            trial[fixed] = u[fixed] + alpha * dup
            trial[free] = u[free] + alpha * duf
            try:
                r, K = residual_tangent(trial)
                break
            except FloatingPointError:
                alpha *= 0.5
                if alpha < 1e-3:
                    raise
        if alpha < 1.0:
            # could not take the whole increment: recurse with smaller steps
            return newton_solve(residual_tangent, ndof, fixed, fixed_target, u, n_steps * 2, tol, max_iter, log)
        u = trial
        fref = max(np.linalg.norm(r[fixed]), 1.0)
        for it in range(max_iter):
            rn = np.linalg.norm(r[free])
            history.append((s, it, rn / fref))
            if rn <= tol * fref:
                break
            solver = SPDSolver(K[free][:, free].tocsr(), symmetric_indefinite=True)
            du = solver.solve(-r[free])
            solver.free()
            alpha = 1.0
            accepted = False
            while alpha > 1e-4:
                trial = u.copy()
                trial[free] += alpha * du
                try:
                    r2, K2 = residual_tangent(trial, False)
                    accepted = True
                    if np.linalg.norm(r2[free]) < max(rn, 1e-30) * (1 - 1e-4 * alpha) or alpha < 0.13:
                        break
                except FloatingPointError:
                    accepted = False
                alpha *= 0.5
            if not accepted:
                raise FloatingPointError("Newton line search failed (element inversion)")
            u = trial
            r, K = residual_tangent(u)
        else:
            if log:
                log(f"newton: step {s} did not fully converge (rel residual {rn / fref:.2e})")
    return u, history


def stress_load(geom: ElementGeom, elems_dofs: np.ndarray, ndof: int, sigma: np.ndarray) -> np.ndarray:
    """Nodal forces of a prescribed element stress field (E, 3, 3) acting *in addition* to the
    elastic stress (e.g. active muscle fibre tension): returns f = -int B^T sigma dV, i.e. the
    right-hand side of K u = f."""
    fe = -np.einsum("eij,egaj,eg->eai", sigma, geom.dN, geom.w)
    return np.bincount(elems_dofs.ravel(), weights=fe.reshape(-1), minlength=ndof)
