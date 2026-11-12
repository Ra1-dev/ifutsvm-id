"""Zero Universum penalty equals the independently reduced primal."""
import cvxpy as cp
import numpy as np
import pytest

from ifutsvm import IFUTSVMID
from ifutsvm.universum import ConditionalGaussianUniversum
from experiments.research_model import ResearchIFUTSVMID


@pytest.mark.parametrize('basis', ['reduced', 'full'])
@pytest.mark.parametrize('side', ['symmetric', 'paper_dual', 'paper_primal'])
def test_zero_penalty_removes_universum_effect(basis, side):
    rng = np.random.default_rng(863)
    X = rng.normal(size=(22, 3))
    y = np.repeat(['minority-z', 'majority-a'], [5, 17])
    query = rng.normal(size=(31, 3))
    params = dict(C=.7, C3=.4, Cu=0, eps=.27, mu=1.3,
                  kernel_basis=basis, universum_side2=side, random_state=991)
    models = [
        IFUTSVMID(**params),
        IFUTSVMID(**params, universum=ConditionalGaussianUniversum()),
        ResearchIFUTSVMID(**params, size_multiplier=.25),
        ResearchIFUTSVMID(**params, size_multiplier=2,
                         universum=ConditionalGaussianUniversum()),
    ]
    for model in models:
        model.fit(X, y)
    trace = models[0].trace_
    D = trace['D']

    def design(points):
        # Independent direct differences, without production kernel helpers.
        K = np.exp(-np.sum((points[:, None]-D[None, :])**2, axis=2)/(2*params['mu']**2))
        return np.column_stack((K, np.ones(len(points))))

    E, F, Fs = (design(trace[key]) for key in ('X1', 'X2', 'X2_star'))
    for key, P, Q, weights in [('z1', E, -Fs, trace['s_maj_star']),
                                ('z2', F, E, trace['s_min'])]:
        z = cp.Variable(P.shape[1])
        xi = cp.Variable(len(Q), nonneg=True)
        # No Universum points, variables, constraints or penalty in this oracle.
        problem = cp.Problem(cp.Minimize(.5*cp.sum_squares(P@z)
            +params['C3']/2*cp.sum_squares(z)+params['C']*weights@xi),
            [Q@z >= 1-xi])
        problem.solve(solver='OSQP', eps_abs=1e-10, eps_rel=1e-10,
                      max_iter=100000, polishing=True)
        assert problem.status == 'optimal'
        for model in models:
            actual = model.trace_[key]
            assert np.linalg.norm(actual-z.value)/max(1, np.linalg.norm(z.value)) <= 1e-6
    # The different laws/counts really changed U while leaving class geometry fixed.
    assert len({len(model.trace_['U']) for model in models}) == 3
    assert not np.array_equal(models[0].trace_['U'], models[1].trace_['U'])
    for model in models[1:]:
        for key in ('D', 'X2_star', 's_min', 's_maj', 's_maj_star'):
            np.testing.assert_array_equal(model.trace_[key], trace[key])
        np.testing.assert_array_equal(model.predict(query), models[0].predict(query))
