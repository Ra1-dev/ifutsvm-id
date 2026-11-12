"""Corrected, explicitly specified paper-mode binary classifier."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.multiclass import check_classification_targets
from ._numeric import aug, rbf, scores, hinge_plane


class IFUTSVMID(ClassifierMixin, BaseEstimator):
    """Intuitionistic fuzzy Universum twin SVM with independent penalties.

    Sampling uses only fit data. ``trace_`` records the full construction.
    Plane 1 represents the minority (first sorted label on count ties).
    See docs/PORT_DISCREPANCIES.md for differences from the original demo.
    """
    def __init__(self, C=1.0, C3=1.0, Cu=1.0, eps=0.3, mu=1.0, eta=1e-4,
                 universum="random_average", universum_side2="symmetric",
                 kernel_basis="reduced", random_state=None):
        self.C = C
        self.C3 = C3
        self.Cu = Cu
        self.eps = eps
        self.mu = mu
        self.eta = eta
        self.universum = universum
        self.universum_side2 = universum_side2
        self.kernel_basis = kernel_basis
        self.random_state = random_state

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        if len(counts) != 2:
            raise ValueError("IFUTSVMID requires exactly two classes")
        for key in ("C", "C3", "mu", "eta"):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not np.isfinite(self.Cu) or self.Cu < 0 or not np.isfinite(self.eps):
            raise ValueError("Cu must be nonnegative and eps finite")
        if self.kernel_basis not in ("reduced", "full"):
            raise ValueError("kernel_basis must be reduced or full")
        sides = {"symmetric": (-1, self.eps - 1),
                 "paper_dual": (1, 1 - self.eps),
                 "paper_primal": (1, self.eps - 1)}
        if self.universum_side2 not in sides:
            raise ValueError("invalid universum_side2")
        self.minority_ = self.classes_[int(np.argmin(counts))]
        self.majority_ = self.classes_[1 - int(np.argmin(counts))]
        X1, X2 = X[y == self.minority_], X[y == self.majority_]
        m1, m2 = len(X1), len(X2)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(m2, m1, replace=False)
        X2_star = X2[idx]
        g = (m1 + 1) // 2
        u = max(m2 - m1, g)
        if self.universum == "random_average":
            U = (X1[rng.integers(m1, size=u)] + X2[rng.integers(m2, size=u)]) / 2
        elif callable(self.universum):
            U = np.asarray(self.universum(X1.copy(), X2.copy(), u, rng), float)
            if U.shape != (u, X.shape[1]) or not np.isfinite(U).all():
                raise ValueError("universum callable returned invalid shape/values")
        else:
            raise ValueError("universum must be random_average or a callable")
        U_star = U[rng.choice(len(U), g, replace=False)]
        D = np.vstack((X1, X2_star if self.kernel_basis == "reduced" else X2))
        s = scores(np.vstack((X1, X2)), np.r_[np.ones(m1), np.zeros(m2)], self.mu, self.eta)
        s_min, s_maj = s[:m1], s[m1:]
        s_maj_star = s_maj[idx]
        E, Fs, Gs, F, G = [aug(rbf(a, D, self.mu)) for a in (X1, X2_star, U_star, X2, U)]
        z1 = hinge_plane(E, np.vstack((-Fs, Gs)), np.r_[np.ones(m1), np.full(g, self.eps - 1)],
                         np.r_[self.C * s_maj_star, np.full(g, self.Cu)], self.C3)
        sign, rhs = sides[self.universum_side2]
        z2 = hinge_plane(F, np.vstack((E, sign * G)), np.r_[np.ones(m1), np.full(u, rhs)],
                         np.r_[self.C * s_min, np.full(u, self.Cu)], self.C3)
        self.trace_ = {k: v.copy() for k, v in locals().items() if k in (
            "X1", "X2", "X2_star", "U", "U_star", "D", "s_min", "s_maj", "s_maj_star", "z1", "z2")}
        return self

    def _distances(self, X):
        check_is_fitted(self, "trace_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        t = self.trace_
        K = aug(rbf(X, t["D"], self.mu))
        return tuple(np.abs(K @ t[k]) / max(np.linalg.norm(t[k][:-1]), np.finfo(float).tiny)
                     for k in ("z1", "z2"))

    def predict(self, X):
        a, b = self._distances(X)
        return np.where(a < b, self.minority_, self.majority_)

    def decision_function(self, X):
        a, b = self._distances(X)
        return b - a if self.minority_ == self.classes_[1] else a - b
