"""Callback isolation, RNG protocol and explicit invalid-output errors."""
import numpy as np
import pytest
from ifutsvm import IFUTSVMID


@pytest.mark.parametrize('counts,constant', [((4, 5), False), ((2, 23), False), ((4, 10), True)])
def test_mutating_sampler_preserves_training_data_and_exact_default_trace(counts, constant):
    rng = np.random.default_rng(20261421)
    X = np.zeros((sum(counts), 3)) if constant else rng.normal(size=(sum(counts), 3))
    y = np.repeat(['minority', 'majority'], counts)
    before = X.copy()
    calls = []
    buffers = []

    def sampler(minority, majority, count, random):
        # Callback receives only the supplied training classes, in their
        # original row order, and at the documented point in the RNG stream.
        np.testing.assert_array_equal(minority, before[:counts[0]])
        np.testing.assert_array_equal(majority, before[counts[0]:])
        calls.append((len(minority), len(majority), count))
        U = (minority[random.integers(len(minority), size=count)]
             + majority[random.integers(len(majority), size=count)]) / 2
        # A third-party strategy may use its input arrays as scratch buffers.
        minority[:] = 173
        majority[:] = -181
        buffers.append(U)
        return U

    expected = IFUTSVMID(random_state=71).fit(X, y)
    got = IFUTSVMID(universum=sampler, random_state=71).fit(X, y)
    assert calls == [(counts[0], counts[1], max(counts[1]-counts[0], (counts[0]+1)//2))]
    np.testing.assert_array_equal(X, before)
    for name, value in expected.trace_.items():
        np.testing.assert_array_equal(got.trace_[name], value)
    probe = np.vstack([before, rng.normal(size=(13, 3))])
    predictions = expected.predict(probe)
    margins = expected.decision_function(probe)
    # Later edits to caller-owned inputs or the callback's output buffer must
    # not corrupt the already-fitted model or its recorded construction.
    X[:] = 211
    buffers[0][:] = -223
    for name, value in expected.trace_.items():
        np.testing.assert_array_equal(got.trace_[name], value)
    np.testing.assert_array_equal(got.predict(probe), predictions)
    np.testing.assert_array_equal(got.decision_function(probe), margins)


@pytest.mark.parametrize('failure', ['rows', 'columns', 'vector', 'nan', 'inf'])
def test_invalid_sampler_output_is_rejected(failure):
    X = np.arange(36, dtype=float).reshape(12, 3)/10
    y = np.r_[np.ones(3), np.zeros(9)]

    def sampler(minority, majority, count, random):
        if failure == 'rows':
            return np.zeros((count+1, X.shape[1]))
        if failure == 'columns':
            return np.zeros((count, X.shape[1]+1))
        if failure == 'vector':
            return np.zeros(count)
        return np.full((count, X.shape[1]), np.nan if failure == 'nan' else np.inf)

    with pytest.raises(ValueError, match='universum callable returned invalid shape/values'):
        IFUTSVMID(universum=sampler, random_state=71).fit(X, y)
