"""Estimator contract across equivalent NumPy storage and prediction batches."""
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal

from ifutsvm import IFUTSVMID
from ifutsvm.universum import ConditionalGaussianUniversum


def layout(X,kind):
    if kind=='fortran':return np.asfortranarray(X)
    if kind=='strided':
        storage=np.zeros((2*len(X),2*X.shape[1]))
        view=storage[::2,::2];view[:]=X
        assert not view.flags.c_contiguous and not view.flags.f_contiguous
        return view
    if kind=='readonly':
        result=X.copy();result.flags.writeable=False
        return result
    if kind=='float32':return X.astype(np.float32)
    return X.astype('>f8')


@pytest.mark.parametrize('kind',['fortran','strided','readonly','float32','big_endian'])
@pytest.mark.parametrize('counts',[(3,9),(6,6)])
@pytest.mark.parametrize('generative',[False,True])
def test_storage_layout_preserves_fit_predictions_and_inputs(kind,counts,generative):
    # Binary fractions are exactly representable in both float32 and float64;
    # this tests storage conversion, not loss from input quantization.
    rng=np.random.default_rng(20261127)
    X=rng.integers(-12,13,size=(sum(counts),4))/8
    y=np.repeat(['z-positive','a-negative'],counts)
    order=rng.permutation(len(X));X=X[order];y=y[order]
    test=rng.integers(-16,17,size=(17,4))/8

    def estimator():
        return IFUTSVMID(random_state=47,
            universum=ConditionalGaussianUniversum() if generative else 'random_average')

    expected=estimator().fit(X,y)
    stored=layout(X,kind);labels=y[::-1][::-1];query=layout(test,kind)
    before=stored.tobytes();query_before=query.tobytes();labels_before=labels.tobytes()
    got=estimator().fit(stored,labels)
    assert got.n_features_in_==X.shape[1]
    assert_array_equal(got.classes_,expected.classes_)
    assert got.minority_==expected.minority_ and got.majority_==expected.majority_
    for name,value in expected.trace_.items():
        assert_allclose(got.trace_[name],value,rtol=0,atol=1e-7,err_msg=name)
    assert_array_equal(got.predict(query),expected.predict(test))
    assert_allclose(got.decision_function(query),expected.decision_function(test),rtol=0,atol=1e-7)
    predictions=np.concatenate([got.predict(query[:1]),got.predict(query[1:8]),got.predict(query[8:])])
    margins=np.concatenate([got.decision_function(query[:1]),got.decision_function(query[1:8]),got.decision_function(query[8:])])
    assert_array_equal(predictions,got.predict(query))
    assert_allclose(margins,got.decision_function(query),rtol=0,atol=1e-7)
    assert stored.tobytes()==before and query.tobytes()==query_before
    assert labels.tobytes()==labels_before
    assert_array_equal(labels,y)
