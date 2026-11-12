"""Contracts for Gaussian distances under offsets and unusual array shapes."""
import numpy as np
import pytest
from ifutsvm import IFUTSVMID
from ifutsvm._numeric import rbf


@pytest.mark.parametrize('features', [1, 7, 31])
@pytest.mark.parametrize('mu', [.125, 1., 8.])
def test_offset_kernel_against_direct_long_double(features, mu):
    rng = np.random.default_rng(314159)
    X = rng.normal(size=(13, features)) + 2.**24
    Y = np.vstack((X[:3], rng.normal(size=(8, features)) + 2.**24))
    difference = X.astype(np.longdouble)[:, None] - Y.astype(np.longdouble)[None]
    expected = np.exp(-np.sum(difference**2, axis=2) / (2 * np.longdouble(mu)**2))
    np.testing.assert_allclose(rbf(X, Y, mu), expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize('basis', ['reduced', 'full'])
@pytest.mark.parametrize('side', ['symmetric', 'paper_dual', 'paper_primal'])
def test_exact_binary_translation_preserves_fitted_planes(basis, side):
    rng = np.random.default_rng(271828)
    X = rng.integers(-16, 17, size=(18, 3)) / 8
    y = np.r_[np.zeros(6), np.ones(12)]
    query = rng.integers(-16, 17, size=(9, 3)) / 8
    shift = 2.**30 * np.array([1., -2., 3.])
    options = dict(C=.7, C3=.4, Cu=.2, mu=1., random_state=12,
                   kernel_basis=basis, universum_side2=side)
    first = IFUTSVMID(**options).fit(X, y)
    second = IFUTSVMID(**options).fit(X + shift, y)
    for key in ['s_min', 's_maj', 's_maj_star', 'z1', 'z2']:
        np.testing.assert_allclose(first.trace_[key], second.trace_[key], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(first.predict(query), second.predict(query + shift))
    np.testing.assert_allclose(first.decision_function(query), second.decision_function(query + shift), atol=1e-6, rtol=0)


@pytest.mark.parametrize('rows', [(0, 0), (0, 3), (3, 0)])
def test_empty_kernel_shape(rows):
    a, b = rows
    assert rbf(np.zeros((a, 2)), np.zeros((b, 2)), 1.).shape == (a, b)
