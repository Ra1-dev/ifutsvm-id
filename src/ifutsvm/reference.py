"""Independent Python port of the published demo's executable semantics.

Original source: https://github.com/abdulquadir1999/IFUTSVM-ID
This module deliberately preserves defects; use IFUTSVMID for corrected mode.
Octave rounded indexing and MATLAB-semantic box QPs follow docs/SPECIFICATION.md.
"""
import numpy as np
import scipy.sparse as sp
from ._numeric import qp


def octave_indices(base, step, limit, size):
    """Positive Octave colon indices converted to zero-based rounded integers."""
    if step == 0 or np.isnan(step) or np.isnan(limit):
        raise ValueError("invalid colon range")
    if limit < base:
        return np.empty(0, dtype=int)
    if np.isinf(step):
        values = np.array([base])
    else:
        # Positive-range specialization of the documented Octave range-length
        # rule (Hagerty tolerant floor, then adjacent-endpoint correction).
        # https://docs.octave.org/doxygen/stable/d7/d53/Range_8cc_source.html
        tolerance = 3 * np.finfo(float).eps
        quotient = (limit - base + step) / step
        cap = 1 / (2 - tolerance)
        fuzz = max(tolerance, min(cap, tolerance * abs(1 + np.floor(quotient))))
        rounded = np.floor(quotient + fuzz)
        count = int(rounded - (quotient > 0 and rounded - quotient >= cap))
        close = lambda a, b: abs(a-b) < max(abs(a), abs(b)) * tolerance
        if not close(base + (count - 1) * step, limit):
            if close(base + (count - 2) * step, limit):
                count -= 1
            elif close(base + count * step, limit):
                count += 1
        values = base + np.arange(count) * step
        if count and values[-1] >= limit:
            values[-1] = limit
    indices = np.floor(values + 0.5).astype(int) - 1
    if (indices < 0).any() or (indices >= size).any():
        raise IndexError("Octave index out of bounds")
    return indices


def _kernel(X, Y, scale):
    d = -2 * X @ Y.T
    d = d + np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :]
    return np.exp(-d / scale)


def _scores(train, mu):
    X, y = train[:, :-1], train[:, -1]
    A, B = X[y == 1], X[y != 1]
    theta = []
    radii = []
    for part in (A, B):
        K = _kernel(part, part, mu**2)
        radius = np.sqrt(np.maximum(0, 1 - 2 * K.mean(1) + K.mean()))
        radii.append(radius.max())
        theta.append(1 - radius / (radius.max() + 1e-4))
    K = _kernel(X, X, mu**2)
    near = np.sqrt(np.maximum(0, 2 * (1 - K))) < max(radii)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (near & (y[:, None] != y[None, :])).sum(1) / near.sum(1)
    out = []
    for mem, mask in zip(theta, (y != -1, y == -1)):
        non = (1 - mem) * rho[mask]
        out.append(np.where(non == 0, mem, np.where(mem <= non, 0, (1 - non) / (2 - mem - non))))
    return out


def evaluate(actual, predicted):
    """Evaluate.m conventions, including NaNs and its precision-based gmean."""
    pos = actual == 1
    p, n = pos.sum(), (~pos).sum()
    tp, tn = (predicted[pos] == actual[pos]).sum(), (predicted[~pos] == actual[~pos]).sum()
    sensitivity = tp / p if p else 0.0
    specificity = tn / n if n else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.divide(float(tp), tp + n - tn)
        f = np.divide(2 * precision * sensitivity, precision + sensitivity)
    vals = [(sensitivity + specificity) / 2, (tp + tn) / len(actual),
            sensitivity, specificity, precision, sensitivity, f, np.sqrt(precision * sensitivity)]
    return dict(zip(("eval_" + k for k in ("AUC", "accuracy", "sensitivity", "specificity",
                     "precision", "recall", "f_measure", "gmean")), vals))


def _box(H, f, ub):
    H = (H + H.T) / 2
    n = len(f)
    try:
        x = qp(H, f, sp.vstack((sp.eye(n), -sp.eye(n))), np.r_[ub, np.zeros(n)])
    except RuntimeError:
        # Restart the SAME box QP in unit-box coordinates x=diag(ub)t.
        # This removes disparate bound scales without changing penalties,
        # objective scale, feasible vectors, or solver tolerances. Zero-bound
        # variables are fixed exactly and eliminated from the restart.
        active = ub > 0
        x = np.zeros(n)
        if active.any():
            bound = ub[active]
            size = len(bound)
            reduced = H[np.ix_(active, active)] * bound[:, None] * bound[None, :]
            try:
                x[active] = bound * qp(reduced, f[active] * bound,
                    sp.vstack((sp.eye(size), -sp.eye(size))), np.r_[np.ones(size), np.zeros(size)])
            except RuntimeError:
                # Equilibration can itself stagnate on rank-deficient duals.
                # Retry the original coordinates with that internal heuristic
                # disabled, still at the identical accuracy tolerances.
                x[active] = qp(H[np.ix_(active, active)], f[active],
                    sp.vstack((sp.eye(size), -sp.eye(size))), np.r_[bound, np.zeros(size)],
                    equilibrate=False)
    # Recover the active-set stationary solution; avoids barrier bias amplified
    # by the demo's inverse ridge. Bound classification is gradient-checked.
    for _ in range(10):
        grad = H @ x + f
        low = (x < 1e-7) & (grad > 1e-8)
        high = (ub - x < 1e-7) & (grad < -1e-8)
        fixed = low | high | (ub == 0)
        candidate = x.copy()
        candidate[low | (ub == 0)] = 0
        candidate[high] = ub[high]
        free = ~fixed
        if free.any():
            candidate[free] = np.linalg.lstsq(H[np.ix_(free, free)],
                -f[free] - H[np.ix_(free, fixed)] @ candidate[fixed], rcond=1e-14)[0]
        if np.any(candidate < -1e-9) or np.any(candidate > ub + 1e-9):
            break
        candidate = np.clip(candidate, 0, ub)
        if np.linalg.norm(candidate - x) < 1e-14:
            x = candidate
            break
        x = candidate
    return x


def reference_prediction(X, C, w1, w2, mu):
    """Original demo test kernel, plane distances and hardwired +/-1 rule."""
    K = np.column_stack((np.exp(-np.sum((X[:, None, :] - C[None, :, :])**2, axis=2) / (2 * mu**2)), np.ones(len(X))))
    with np.errstate(divide="ignore", invalid="ignore"):
        y1, y2 = K @ w1 / np.linalg.norm(w1[:-1]), K @ w2 / np.linalg.norm(w2[:-1])
    classifier = np.where(np.abs(y1) < np.abs(y2), 1, -1)
    return K,y1,y2,classifier


def trace_reference(train, test, c, mu, eps):
    train, test = np.asarray(train, float), np.asarray(test, float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("train/test must have matching feature and label columns")
    A, B = train[train[:, -1] == 1, :-1], train[train[:, -1] != 1, :-1]
    d = abs(len(A) - len(B))
    if not len(A) or not len(B) or d <= 1:
        raise ValueError("reference sampling is undefined for these class counts")
    Au = A[octave_indices(1, len(A) / d, len(A), len(A))]
    Bu = B[octave_indices(1, len(B) / d, len(B), len(B))]
    diff = len(Au) - len(Bu)
    if diff > 0:
        if diff > len(Bu):
            raise IndexError("reference padding exceeds available rows")
        Bu = np.vstack((Bu, Bu[:diff]))
    elif diff < 0:
        if -diff > len(Au):
            raise IndexError("reference padding exceeds available rows")
        Au = np.vstack((Au, Au[:-diff]))
    main_U = (Au + Bu) / 2
    s1, s2 = _scores(train, mu)
    if len(A) > len(B):
        A, B, s1, s2 = B, A, s2, s1
    m1, m2 = len(A), len(B)
    U = main_U[octave_indices(1, (len(main_U) - 1) / d,
                               len(main_U) - (len(main_U) - 1) / d, len(main_U))]
    cl = (m1 + 1) // 2
    ib = octave_indices(1, (m2 - 1) / m1, m2 - (m2 - 1) / m1, m2)
    iu = octave_indices(1, (len(U) - 1) / cl, len(U) - (len(U) - 1) / cl, len(U))
    B1 = B[ib]
    C = np.vstack((A, B1))
    K1, K2, K3 = [_kernel(V, C, 2 * mu) for V in (A, B, U)]
    H, G, O = [np.column_stack((V, np.ones(len(V)))) for V in (K1, K2, K3)]
    invH = np.linalg.inv(H.T @ H + 1e-5 * np.eye(2 * m1 + 1))
    invG = np.linalg.inv(G.T @ G + 1e-5 * np.eye(2 * m1 + 1))
    G1 = np.vstack((G[ib], -O[iu]))
    HO = np.vstack((H, O))
    GOINVGOT, HOINVHOT = G1 @ invH @ G1.T, HO @ invG @ HO.T
    f1, f2 = -np.r_[np.ones(m1), np.full(cl, eps - 1)], -np.r_[np.ones(m1), np.full(d, 1 - eps)]
    lowb1, lowb2 = np.zeros(m1 + cl), np.zeros(m1 + d)
    upb1, upb2 = c * np.r_[s2[ib], np.ones(cl)], c * np.r_[s1, np.ones(d)]
    u1, u2 = _box(GOINVGOT, f1, upb1), _box(HOINVHOT, f2, upb2)
    w1 = -invH @ (G[ib].T @ u1[:m1] - O[iu].T @ u1[m1:])
    w2 = invG @ (H.T @ u2[:m1] + O.T @ u2[m1:])
    # Testing uses explicit Euclidean distance and the squared bandwidth.
    K,y1,y2,classifier = reference_prediction(test[:, :-1],C,w1,w2,mu)
    tr = {k: v for k, v in locals().items() if k in (
        "main_U", "s1", "s2", "U", "A", "B", "B1", "C", "K1", "K2", "K3", "H", "G", "O", "G1",
        "GOINVGOT", "HOINVHOT", "f1", "f2", "lowb1", "upb1", "lowb2", "upb2", "w1", "w2", "K", "y1", "y2", "classifier")}
    tr.update(evaluate(test[:, -1], classifier))
    # IFUTSVM's absent-negative convention differs from Evaluate's.
    sensitivity = tr["eval_sensitivity"]
    negatives = test[:, -1] == -1
    false_positive_rate = np.mean(classifier[negatives] != -1) if negatives.any() else 0
    tr["acc_returned"] = 50 * (1 + sensitivity - false_positive_rate)
    for key in (*evaluate(test[:, -1], classifier), "acc_returned"):
        tr[key] = np.atleast_1d(tr[key])
    return tr
