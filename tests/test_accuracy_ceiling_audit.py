"""The independent auditor must reject contaminated recorded evidence."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.accuracy_ceiling import fit
from experiments.audit_accuracy_ceiling import check_record,exact_accuracy_mean


class MemoryStore:
    def get(self,key): return None
    def put(self,key,record): return record


@pytest.fixture
def record():
    X=np.random.default_rng(8357).normal(size=(16,3))
    y=np.array(['negative']*8+['positive']*4+['negative','positive']*2)
    train=np.arange(12);test=np.arange(12,16);params=dict(n_neighbors=3,weights='uniform',p=2)
    context=dict(role='outer_evaluation',candidate=0)
    r=fit(MemoryStore(),'fixture','knn',params,0,X,y,train,test,context)
    return r,(X,y,train,test,params,0,0,dict(role='outer_evaluation',model='knn'))


def test_auditor_accepts_unmodified_record(record):
    r,args=record
    assert check_record(r,*args)


@pytest.mark.parametrize('field',['accuracy','scaler','indices','parameters'])
def test_auditor_rejects_evidence_corruption(record,field):
    original,args=record;r=deepcopy(original)
    if field=='accuracy': r['accuracy']+=.01
    elif field=='scaler': r['metadata']['scaler']['mean'][0]+=.01
    elif field=='indices': r['train_indices'][0]=r['test_indices'][0]
    elif field=='parameters': r['metadata']['parameters']['n_neighbors']=5
    with pytest.raises(AssertionError): check_record(r,*args)


def test_exact_count_ties_do_not_depend_on_float_summation():
    def rows(counts):
        return [dict(y=['a']*346,prediction=['a']*correct+['b']*(346-correct)) for correct in counts]
    assert exact_accuracy_mean(rows([260,269,273]))==exact_accuracy_mean(rows([264,267,271]))
    assert exact_accuracy_mean(rows([260,269,274]))>exact_accuracy_mean(rows([264,267,271]))
