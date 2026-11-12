"""Contracts beyond visible fixtures: degeneracy, labels, sampling and validation."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from sklearn.base import clone
from ifutsvm import IFUTSVMID
from ifutsvm.reference import octave_indices, trace_reference, evaluate


@pytest.mark.parametrize("counts", [(1, 3), (4, 4), (4, 5), (3, 30)])
@pytest.mark.parametrize("constant", [False, True])
def test_shapes_and_degeneracy(counts, constant):
    r = np.random.default_rng(82)
    X = np.zeros((sum(counts), 2)) if constant else r.normal(size=(sum(counts), 2))
    y = np.repeat(["a", "b"], counts)
    est = IFUTSVMID(random_state=9).fit(X, y)
    assert np.isfinite(est.decision_function(X)).all()
    assert set(est.predict(X)) <= set(y)
    assert len(est.trace_["U"]) == max(counts[1] - counts[0], (counts[0] + 1) // 2)
    assert_array_equal(clone(est).fit(X, y).predict(X), est.predict(X))


def test_sampling_exact_rng_order():
    X = np.arange(68.).reshape(34, 2)
    y = np.r_[np.zeros(9), np.ones(25)]
    r = np.random.default_rng(191)
    idx = r.choice(25, 9, replace=False)
    U = (X[:9][r.integers(9, size=16)] + X[9:][r.integers(25, size=16)]) / 2
    U_star = U[r.choice(16, 5, replace=False)]
    tr = IFUTSVMID(random_state=191).fit(X, y).trace_
    assert_array_equal(tr["X2_star"], X[9:][idx])
    assert_array_equal(tr["U"], U)
    assert_array_equal(tr["U_star"], U_star)


@pytest.mark.parametrize("kwargs", [dict(mu=0), dict(C3=0), dict(C=-1), dict(Cu=-1),
                                      dict(eta=0), dict(kernel_basis="bad"), dict(universum_side2="bad")])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        IFUTSVMID(**kwargs).fit([[0], [1], [2]], [0, 1, 1])


def test_octave_rounds_halves_up():
    assert_array_equal(octave_indices(1, 1.5, 7, 7), [0, 2, 3, 5, 6])
    assert_array_equal(octave_indices(1, .1, 1.3, 2), [0, 0, 0, 0])
    assert len(octave_indices(1.8, .05, 1.9, 3)) == 3


def test_reference_subsample_range_lengths():
    for minority in range(2, 30):
        for majority in range(minority+2, 120):
            step = (majority-1)/minority
            assert len(octave_indices(1, step, majority-step, majority)) == minority


def test_faithful_error_and_metrics():
    with pytest.raises(ValueError):
        trace_reference(np.array([[0, 1], [1, -1], [2, -1]]), np.array([[0, 1]]), 1, 1, .3)
    tr = evaluate(np.array([1, 1, -1, -1, -1]), np.array([1, -1, -1, -1, -1]))
    assert_allclose(tr["eval_AUC"], .75)
    assert_allclose(tr["eval_accuracy"], .8)
    assert_allclose(tr["eval_gmean"], np.sqrt(.5))


@pytest.mark.parametrize("scale", [1e-4, 1, 100])
def test_faithful_box_qp_against_independent_solver(scale):
    # Non-unique duals may differ; objectives and projected KKT residuals must agree.
    import cvxpy as cp
    from ifutsvm.reference import _box
    rng = np.random.default_rng(411)
    R = rng.normal(size=(5, 9))
    H = R.T @ R
    f = rng.normal(size=9)
    ub = scale * rng.uniform(.1, 1, 9)
    ub[0] = 0
    got = _box(H, f, ub)
    x = cp.Variable(9)
    problem = cp.Problem(cp.Minimize(.5*cp.quad_form(x, cp.psd_wrap(H))+f@x), [x>=0,x<=ub])
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12)
    assert_allclose(.5*got@H@got+f@got, problem.value, atol=1e-8, rtol=1e-8)
    assert np.max(np.abs(got-np.clip(got-(H@got+f),0,ub))) < 1e-7
