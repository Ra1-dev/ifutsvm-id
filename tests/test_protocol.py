"""contracts: identical split plans and outer-test isolation."""
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from experiments.protocol import split_plan,evaluate_split


def test_shared_splits_are_disjoint_and_outer_cv_covers_every_row():
    y=np.r_[np.zeros(50),np.ones(20)]
    for _,_,train,test in split_plan(y,'holdout'):
        assert not np.intersect1d(train,test).size
        assert len(test)==21
    plan=list(split_plan(y,'nested_cv'))
    assert np.array_equal(np.sort(np.concatenate([x[3] for x in plan])),np.arange(len(y)))


def test_tuning_and_scaling_never_see_outer_test(tmp_path):
    rng=np.random.default_rng(17)
    X=rng.normal(size=(60,2))
    y=np.r_[np.zeros(40),np.ones(20)]
    train=np.r_[np.arange(30),np.arange(40,55)]
    test=np.setdiff1d(np.arange(60),train)
    X[test]+=1e6  # would dominate scaler statistics if leaked into training
    est=make_pipeline(StandardScaler(),SVC(kernel='linear',class_weight='balanced'))
    model,result=evaluate_split(est,{'svc__C':[.1,1]},X,y,train,test,tmp_path,
                                seed=0,provenance={'dataset':'synthetic_contract'})
    np.testing.assert_allclose(model[0].mean_,X[train].mean(0))
    assert model[-1].random_state==0
    paths=list((tmp_path/'inner').glob('*.json'))
    assert len(paths)==10
    for path in paths:
        inner=json.loads(path.read_text())
        assert set(inner['train_indices'])<=set(train)
        assert set(inner['test_indices'])<=set(train)
        assert not set(inner['train_indices'])&set(inner['test_indices'])
    assert result['metadata']['parameters'] in ({'svc__C':.1},{'svc__C':1})
