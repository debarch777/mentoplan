"""Verification of the hexahedral FE implementation against analytic solutions."""
import numpy as np
import pytest
import scipy.sparse as sp

from mentoplan.tissue.hexfem import (Assembler, ElementGeom, Material, SPDSolver, newton_solve,
                                     nonlinear_element_terms)


def block(nx, ny, nz, h):
    idx = -np.ones((nx + 1, ny + 1, nz + 1), int)
    ijk = np.argwhere(np.ones((nx + 1, ny + 1, nz + 1), bool))
    idx[tuple(ijk.T)] = np.arange(len(ijk))
    X = ijk * h
    cells = np.argwhere(np.ones((nx, ny, nz), bool))
    off = np.array([[a & 1, (a >> 1) & 1, (a >> 2) & 1] for a in range(8)])
    elems = np.stack([idx[tuple((cells + o).T)] for o in off], 1)
    return X, elems


def solve_dirichlet(K, f, fixed, vals):
    n = K.shape[0]
    free = np.setdiff1d(np.arange(n), fixed)
    u = np.zeros(n)
    u[fixed] = vals
    rhs = f[free] - K[free][:, fixed] @ vals
    u[free] = SPDSolver(K[free][:, free].tocsr()).solve(rhs)
    return u


@pytest.mark.parametrize("distort", [0.0, 0.25])
def test_patch_test_linear(distort):
    h = 2.0
    X, elems = block(4, 3, 3, h)
    rng = np.random.default_rng(0)
    interior = ~((X == X.min(0)).any(1) | (X == X.max(0)).any(1))
    X = X + distort * h * rng.uniform(-1, 1, X.shape) * interior[:, None]
    ref = ElementGeom.from_nodes(X, elems)
    asm = Assembler(elems, len(X))
    E = len(elems)
    mat = Material.from_E_nu(np.full(E, 10e3), 0.49)
    K = asm.linear_matrix(ref, mat)
    A = np.array([[0.01, 0.003, -0.002], [0.004, -0.02, 0.001], [0.0, 0.002, 0.015]])
    uex = X @ A.T
    bnd = np.where((X == X.min(0)).any(1) | (X == X.max(0)).any(1))[0]
    fixed = (3 * bnd[:, None] + np.arange(3)).ravel()
    u = solve_dirichlet(K, np.zeros(K.shape[0]), fixed, uex.ravel()[fixed])
    assert np.allclose(u.reshape(-1, 3), uex, atol=1e-9)


def test_tangent_matches_finite_differences():
    h = 1.5
    X, el = block(1, 1, 1, h)
    X = X + np.random.default_rng(3).uniform(-0.2, 0.2, X.shape)   # distorted element
    ref = ElementGeom.from_nodes(X, el)
    rng = np.random.default_rng(1)
    U = rng.normal(scale=0.15, size=(1, 8, 3))
    mat = Material(np.array([3000.0]), np.array([60000.0]))
    f0, K = nonlinear_element_terms(ref, U, mat)
    eps = 1e-6
    Kfd = np.zeros((24, 24))
    for j in range(24):
        dU = np.zeros(24)
        dU[j] = eps
        fp, _ = nonlinear_element_terms(ref, U + dU.reshape(1, 8, 3), mat, want_tangent=False)
        fm, _ = nonlinear_element_terms(ref, U - dU.reshape(1, 8, 3), mat, want_tangent=False)
        Kfd[:, j] = (fp[0] - fm[0]) / (2 * eps)
    assert np.allclose(K[0], Kfd, rtol=1e-5, atol=1e-6 * np.abs(K).max())
    assert np.allclose(K[0], K[0].T, atol=1e-8 * np.abs(K).max())


def test_nonlinear_reduces_to_linear_at_small_strain():
    h = 2.0
    X, el = block(3, 1, 1, h)
    X = X + np.random.default_rng(4).uniform(-0.3, 0.3, X.shape)
    ref = ElementGeom.from_nodes(X, el)
    rng = np.random.default_rng(2)
    U = rng.normal(scale=1e-7, size=(3, 8, 3))
    mat = Material(np.full(3, 2000.0), np.full(3, 50000.0))
    f, _ = nonlinear_element_terms(ref, U, mat)
    Klin = ref.linear_K(mat.mu, mat.kappa)
    flin = np.einsum("eij,ej->ei", Klin, U.reshape(3, 24))
    assert np.allclose(f, flin, rtol=1e-4, atol=1e-12)


def test_uniaxial_neohookean_against_analytic():
    """Unconfined uniaxial stretch of a block: compare the reaction force with the analytic
    nearly-incompressible neo-Hookean solution (lateral stretch from the volumetric balance)."""
    h = 1.0
    n = 3
    X, elems = block(n, n, n, h)
    ref = ElementGeom.from_nodes(X, elems)
    asm = Assembler(elems, len(X))
    mu, kappa = 1000.0, 50000.0
    mat = Material(np.full(len(elems), mu), np.full(len(elems), kappa))
    lam = 1.3
    L = n * h
    # boundary: x=0 face fixed in x, x=L face prescribed; symmetry planes y=0 (uy=0), z=0 (uz=0)
    x0 = np.where(np.isclose(X[:, 0], 0))[0]
    xL = np.where(np.isclose(X[:, 0], L))[0]
    y0 = np.where(np.isclose(X[:, 1], 0))[0]
    z0 = np.where(np.isclose(X[:, 2], 0))[0]
    fixed = np.unique(np.r_[3 * x0, 3 * xL, 3 * y0 + 1, 3 * z0 + 2])
    target = np.zeros(3 * len(X))
    target[3 * xL] = (lam - 1) * L

    def rt(u, want_tangent=True):
        fe, Ke = nonlinear_element_terms(ref, u[asm.dofs].reshape(-1, 8, 3), mat, want_tangent)
        return asm.vector(fe), (asm.matrix(Ke) if want_tangent else None)
    u, hist = newton_solve(rt, 3 * len(X), fixed, target[fixed], n_steps=4, tol=1e-9)
    fe, _ = nonlinear_element_terms(ref, u[asm.dofs].reshape(-1, 8, 3), mat, want_tangent=False)
    r = asm.vector(fe)
    force = r[3 * xL].sum()
    # analytic: F = diag(lam, l, l); lateral traction-free -> solve for l
    from scipy.optimize import brentq

    def lateral(l):
        J = lam * l * l
        I1 = lam ** 2 + 2 * l ** 2
        a = J ** (-2 / 3)
        P22 = mu * a * (l - I1 / 3 / l) + kappa * np.log(J) / l
        return P22
    l = brentq(lateral, 0.5, 1.2)
    J = lam * l * l
    I1 = lam ** 2 + 2 * l ** 2
    P11 = mu * J ** (-2 / 3) * (lam - I1 / 3 / lam) + kappa * np.log(J) / lam
    assert force == pytest.approx(P11 * L * L, rel=2e-3)


def test_no_volumetric_locking_in_bending():
    """Cantilever with nu -> 0.4999: SRI must not lock (tip deflection stays near the nu=0.3 value
    scaled by the plane-strain factor rather than collapsing)."""
    h = 1.0
    X, elems = block(16, 2, 4, h)
    ref = ElementGeom.from_nodes(X, elems)
    asm = Assembler(elems, len(X))
    tips = []
    for nu in (0.3, 0.4999):
        mat = Material.from_E_nu(np.full(len(elems), 1e4), nu)
        K = asm.linear_matrix(ref, mat)
        root = np.where(np.isclose(X[:, 0], 0))[0]
        fixed = (3 * root[:, None] + np.arange(3)).ravel()
        f = np.zeros(K.shape[0])
        tip = np.where(np.isclose(X[:, 0], 16))[0]
        f[3 * tip + 2] = -1.0 / len(tip)
        u = solve_dirichlet(K, f, fixed, np.zeros(len(fixed)))
        tips.append(-u[3 * tip + 2].mean())
    # Euler-Bernoulli: deflection depends on E only; mild Poisson effect expected
    assert tips[1] / tips[0] > 0.75
