"""faithful adapter must retain the trace's label and kernel behaviour."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal,assert_allclose
from sklearn.base import clone
from ifutsvm import FaithfulIFUTSVMID
from ifutsvm.reference import trace_reference


@pytest.mark.parametrize('orientation',[1,-1])
def test_adapter_reuses_planes_and_preserves_original_labels(orientation):
    rng=np.random.default_rng(731)
    X=rng.normal(size=(34,3)); y=orientation*np.r_[np.ones(9),-np.ones(25)]
    test=rng.normal(size=(11,3))
    model=FaithfulIFUTSVMID(C=.8,mu=1.4,eps=.3).fit(X,y)
    trace=trace_reference(np.column_stack((X,y)),np.column_stack((test,np.ones(len(test)))),.8,1.4,.3)
    assert_array_equal(model.predict(test),trace['classifier'])
    assert_allclose(model.decision_function(test),np.abs(trace['y2'])-np.abs(trace['y1']))
    assert clone(model).get_params()==model.get_params()
    # Mutating evaluation data does not alter the stored fitted coefficients.
    before=model.trace_['w1'].copy()
    model.predict(test+5)
    assert_array_equal(before,model.trace_['w1'])
