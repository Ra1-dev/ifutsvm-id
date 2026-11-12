"""Readable default-method implementation emitted literally into the notebook.

This direct-primal teaching version is checked against the production estimator;
it does not replace any fitted benchmark or its pre-specified numerical code.
"""
import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf


def teaching_rbf(X, Z, mu):
    # One bandwidth is used in training, fuzzy scores and prediction.
    squared_distance = np.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    return np.exp(-squared_distance / (2 * mu ** 2))


def teaching_if_scores(X, y, mu, eta=1e-4):
    # Kernel distances to each class centre define memberships and radii.
    K = teaching_rbf(X, X, mu)
    membership = np.zeros(len(X))
    radii = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        block = K[np.ix_(indices, indices)]
        distance = np.sqrt(np.maximum(0, 1 - 2 * block.mean(axis=1) + block.mean()))
        radius = distance.max()
        membership[indices] = 1 - distance / (radius + eta)
        radii.append(radius)
    # Opposite-label neighbours quantify local ambiguity. Include self even
    # in the zero-radius case so that the neighbourhood fraction is defined.
    kernel_distance = np.sqrt(np.maximum(0, 2 - 2 * K))
    neighbours = kernel_distance < max(radii)
    np.fill_diagonal(neighbours, True)
    opposite = y[:, None] != y[None, :]
    fraction = (neighbours & opposite).sum(axis=1) / neighbours.sum(axis=1)
    nonmembership = (1 - membership) * fraction
    score = np.where(nonmembership == 0, membership,
        np.where(membership <= nonmembership, 0,
            (1 - nonmembership) / (2 - membership - nonmembership)))
    return membership, nonmembership, score


def teaching_random_average(minority, majority, count, rng):
    # Pairs are sampled independently with replacement from training classes.
    left = minority[rng.integers(len(minority), size=count)]
    right = majority[rng.integers(len(majority), size=count)]
    return (left + right) / 2


def teaching_conditional_gaussian(minority, majority, count, rng):
    # Estimate a shared, equally class-weighted covariance on training rows.
    p = minority.shape[1]
    covariances = [LedoitWolf().fit(rows).covariance_ if len(rows) > 1
                   else np.zeros((p, p)) for rows in (minority, majority)]
    sigma = (covariances[0] + covariances[1]) / 2
    sigma += 1e-8 * max(np.trace(sigma) / p, np.finfo(float).eps) * np.eye(p)
    midpoint = (minority.mean(axis=0) + majority.mean(axis=0)) / 2
    direction = majority.mean(axis=0) - minority.mean(axis=0)
    normal = np.linalg.solve(sigma, direction)
    separation = float(direction @ normal)
    # Scaling preserves the same neutral hyperplane if products underflow.
    magnitude = np.max(np.abs(direction))
    if magnitude > 0 and (separation == 0 or magnitude < np.sqrt(np.finfo(float).tiny)):
        direction = direction / magnitude
        normal = np.linalg.solve(sigma, direction)
        separation = float(direction @ normal)
    conditional = sigma - np.outer(direction, direction) / separation if separation > 0 else sigma.copy()
    conditional = (conditional + conditional.T) / 2
    # Gaussian conditioning gives midpoint mean and this singular covariance.
    values, vectors = np.linalg.eigh(conditional)
    root = vectors * np.sqrt(np.maximum(values, 0))[None, :]
    U = midpoint + rng.normal(size=(count, p)) @ root.T
    if separation > 0:
        U -= np.outer(((U - midpoint) @ normal) / separation, direction)
    return U


def teaching_plane(P, Q, rhs, penalties, ridge):
    # Each slack belongs to one displayed linear inequality Q z >= rhs - slack.
    # Solving the untransformed primal makes the paper objective visible.
    z = cp.Variable(P.shape[1])
    slack = cp.Variable(len(Q), nonneg=True)
    objective = .5 * cp.sum_squares(P @ z) + .5 * ridge * cp.sum_squares(z) + penalties @ slack
    problem = cp.Problem(cp.Minimize(objective), [Q @ z >= rhs - slack])
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-12, tol_gap_rel=1e-12,
                  tol_feas=1e-12, max_iter=1000)
    assert problem.status == 'optimal', problem.status
    return z.value


def teaching_fit(X, y, C=.7, C3=.4, Cu=.3, eps=.3, mu=1.2,
                 random_state=47, generative=False):
    # This compact example implements the pre-specified default: reduced basis,
    # symmetric second-plane Universum sign, and full-training IF scores.
    labels, counts = np.unique(y, return_counts=True)
    minority, majority = labels[np.argmin(counts)], labels[1 - np.argmin(counts)]
    A, B = X[y == minority], X[y == majority]
    m1, m2 = len(A), len(B)
    rng = np.random.default_rng(random_state)
    selected = rng.choice(m2, m1, replace=False)
    B_star = B[selected]
    g = (m1 + 1) // 2
    u = max(m2 - m1, g)
    sampler = teaching_conditional_gaussian if generative else teaching_random_average
    U = sampler(A, B, u, rng)
    U_star = U[rng.choice(u, g, replace=False)]
    D = np.vstack([A, B_star])
    _, _, weights = teaching_if_scores(np.vstack([A, B]), np.r_[np.ones(m1), np.zeros(m2)], mu)
    s_min, s_majority = weights[:m1], weights[m1:]
    def augmented(rows):
        return np.column_stack([teaching_rbf(rows, D, mu), np.ones(len(rows))])
    E, F_star, G_star, F, G = [augmented(rows) for rows in [A, B_star, U_star, B, U]]
    # Plane 1: majority classification slack and reduced-Universum slack.
    z1 = teaching_plane(E, np.vstack([-F_star, G_star]),
        np.r_[np.ones(m1), np.full(g, eps - 1)],
        np.r_[C * s_majority[selected], np.full(g, Cu)], C3)
    # Plane 2: minority classification slack and symmetric full-Universum slack.
    z2 = teaching_plane(F, np.vstack([E, -G]),
        np.r_[np.ones(m1), np.full(u, eps - 1)],
        np.r_[C * s_min, np.full(u, Cu)], C3)
    return dict(D=D, z1=z1, z2=z2, minority=minority, majority=majority, mu=mu, U=U)


def teaching_predict(model, X):
    # Eq. 32 divides by the Euclidean coefficient norm, excluding the bias.
    K = np.column_stack([teaching_rbf(X, model['D'], model['mu']), np.ones(len(X))])
    distances = [np.abs(K @ model[key]) / max(np.linalg.norm(model[key][:-1]), np.finfo(float).tiny)
                 for key in ['z1', 'z2']]
    return np.where(distances[0] < distances[1], model['minority'], model['majority'])
