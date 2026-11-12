"""Label orientation, standard metrics and saved evidence."""
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ifutsvm.evaluation import minority_label, minority_scores, binary_metrics, save_run


def test_minority_positive_metrics():
    # tn=3, fp=1, fn=1, tp=2: precision differs from specificity.
    y=np.array(['rare','rare','rare','common','common','common','common'])
    pred=np.array(['rare','rare','common','rare','common','common','common'])
    scores=np.array([.9,.8,-.5,.6,-.3,-.8,-.9])
    result=binary_metrics(y,pred,scores,'rare',.25)
    assert result['confusion_matrix']==[[3,1],[1,2]]
    assert_allclose([result[k] for k in ('accuracy','precision','recall','f1','specificity','gmean','balanced_accuracy')],
                    [5/7,2/3,2/3,2/3,3/4,np.sqrt(.5),17/24])
    assert result['roc_auc']==pytest.approx(5/6)


def test_orientation_and_tie():
    assert minority_label(['a','a','b','b'])=='a'
    class Est:
        classes_=np.array(['a','b'])
        def decision_function(self,X):
            return np.array([2.,-1.])
    assert_allclose(minority_scores(Est(),np.zeros((2,1)),'a'),[-2,1])


def test_saved_run_recomputes_and_rejects_leakage(tmp_path):
    args=dict(y=[0,1],prediction=[0,0],scores=[-.3,.2],positive_label=1,training_seconds=0,
              train_indices=[0,2,4],test_indices=[1,3],metadata=dict(seed=7))
    path=tmp_path/'run.json'
    save_run(path,**args)
    result=json.loads(path.read_text())
    again=binary_metrics(result['y'],result['prediction'],result['scores'],result['positive_label'],result['training_seconds'])
    assert result['roc_auc']==again['roc_auc']==1
    assert result['precision']==0 and result['f1']==0
    assert result['roc']['thresholds'][0] is None
    with pytest.raises(ValueError,match='overlap'):
        save_run(path,**{**args,'train_indices':[1,2,4]})


def test_missing_class_has_explicit_null_auc():
    result=binary_metrics([0,0],[0,0],[0.,0.],1,0)
    assert result['roc_auc'] is None and result['roc'] is None
