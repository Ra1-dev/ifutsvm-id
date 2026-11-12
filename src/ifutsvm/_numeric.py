"""Numerical primitives; no data or experiment assumptions."""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist
import clarabel


def rbf(X, Y, mu):
    nx, ny = (X * X).sum(1), (Y * Y).sum(1)
    # The norm expansion can lose all pair-distance information under a large
    # common offset. A conservative floating-point error estimate, propagated
    # through exp(-d/(2*mu**2)), selects direct pair differences when needed.
    # The threshold is an absolute kernel-accuracy budget, independent of data
    # identities, experiment seeds, training shapes or fitted results.
    factor = 8 * np.finfo(float).eps * (X.shape[1] + 2)
    uncertainty = factor * (nx.max(initial=0) + ny.max(initial=0))
    if not np.isfinite(uncertainty):
        d = cdist(X, Y, metric='sqeuclidean')
    else:
        d = nx[:, None] + ny[None, :] - 2 * X @ Y.T
        if uncertainty > 2 * mu**2 * 1e-9:
            error = factor * (nx[:, None] + ny[None, :])
            # A far pair can have inaccurate squared distance but negligible
            # kernel error. The mean-value bound uses the largest derivative
            # over the possible distance interval, including clipping at zero.
            kernel_error = error * np.exp(-np.maximum(d - error, 0) / (2 * mu**2)) / (2 * mu**2)
            repair = kernel_error > 1e-9
            if np.any(repair):
                direct = cdist(X, Y, metric='sqeuclidean')
                d = np.where(repair, direct, d)
    return np.exp(-np.maximum(d, 0) / (2 * mu**2))


def aug(X):
    return np.column_stack((X, np.ones(len(X))))


def scores(X, y, mu, eta=1e-4):
    K = rbf(X, X, mu)
    theta = np.empty(len(X))
    radii = []
    for label in np.unique(y):
        mask = y == label
        Kc = K[np.ix_(mask, mask)]
        d = np.sqrt(np.maximum(0, 1 - 2 * Kc.mean(1) + Kc.mean()))
        radii.append(d.max())
        theta[mask] = 1 - d / (d.max() + eta)
    nb = np.sqrt(np.maximum(0, 2 - 2 * K)) < max(radii)
    # Degenerate identical samples have radius zero: retain self neighbourhood.
    np.fill_diagonal(nb, True)
    rho = (nb & (y[:, None] != y[None, :])).sum(1) / nb.sum(1)
    sigma = (1 - theta) * rho
    return np.where(sigma == 0, theta, np.where(theta <= sigma, 0,
                    (1 - sigma) / (2 - theta - sigma)))


def qp(P, q, A, b, *, equilibrate=True):
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = 1e-12
    settings.tol_gap_rel = 1e-12
    settings.tol_feas = 1e-12
    settings.max_iter = 300
    settings.equilibrate_enable = equilibrate
    solver = clarabel.DefaultSolver(sp.triu(sp.csc_matrix(P)).tocsc(), q,
                                    sp.csc_matrix(A), b,
                                    [clarabel.NonnegativeConeT(len(b))], settings)
    sol = solver.solve()
    if str(sol.status) not in ("Solved", "AlmostSolved"):
        raise RuntimeError(f"QP did not converge: {sol.status}")
    return np.asarray(sol.x)


def hinge_plane(P, Q, rhs, penalty, ridge):
    """min .5||Pz||² + ridge/2||z||² + penalty' max(0, rhs-Qz).

    Explicit slack primal keeps the numerical system well-conditioned and
    avoids constructing an O(number of constraints squared) dual Hessian.
    """
    n, m = P.shape[1], len(Q)
    h = P.T @ P + ridge * np.eye(n)
    # v = L.T z makes the quadratic term identity. This invertible change of
    # variables reduces ridge-dependent conditioning without changing the
    # objective, penalties, or any comparison tolerance.
    L = cholesky(h, lower=True)
    transformed = solve_triangular(L, Q.T, lower=True).T
    H = sp.block_diag((sp.eye(n), sp.csc_matrix((m, m))), format="csc")
    A = sp.vstack((sp.hstack((-sp.csc_matrix(transformed), -sp.eye(m))),
                   sp.hstack((sp.csc_matrix((m, n)), -sp.eye(m)))), format="csc")
    v = qp(H, np.r_[np.zeros(n), penalty], A, np.r_[-rhs, np.zeros(m)])[:n]
    return solve_triangular(L.T, v, lower=False)
