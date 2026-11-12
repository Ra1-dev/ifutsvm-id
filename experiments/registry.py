"""One model registry and one shared, seeded candidate list for all baselines."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from ifutsvm import IFUTSVMID,FaithfulIFUTSVMID
from ifutsvm.baselines import TwinSVM

MODELS=('svm_balanced','smote_svm','tsvm','utsvm','faithful_raw','faithful_scaled','paper')


class AdaptiveSMOTE(BaseEstimator):
    def __init__(self,random_state=None):
        self.random_state=random_state
    def fit_resample(self,X,y):
        _,counts=np.unique(y,return_counts=True)
        if min(counts)<2:
            raise ValueError('SMOTE requires at least two minority training rows')
        sampler=SMOTE(k_neighbors=min(5,int(min(counts))-1),random_state=self.random_state)
        return sampler.fit_resample(X,y)


def candidates(dataset_index,split_index,protocol,budget=40):
    rng=np.random.default_rng(np.random.SeedSequence([20261110,dataset_index,split_index,int(protocol=='nested_cv')]))
    return [dict(C=float(10.**rng.integers(-3,4)),Cu=float(10.**rng.integers(-3,4)),
                 C3=float(10.**rng.integers(-3,4)),eps=float(rng.choice([.1,.3,.5,.6])),
                 mu=float(2.**rng.integers(-3,4))) for _ in range(budget)]


def make_model(name,schema,parameters,seed):
    numeric=[i for i,a in enumerate(schema) if not a['nominal']]
    nominal=[i for i,a in enumerate(schema) if a['nominal']]
    parts=[]
    if numeric:
        parts.append(('numeric','passthrough',numeric))
    if nominal:
        parts.append(('nominal',OneHotEncoder(handle_unknown='ignore',sparse_output=False),nominal))
    encoder=ColumnTransformer(parts,sparse_threshold=0)
    p=parameters
    if name in ('svm_balanced','smote_svm'):
        model=SVC(C=p['C'],gamma=1/(2*p['mu']**2),kernel='rbf',
                  class_weight='balanced' if name=='svm_balanced' else None,random_state=seed)
    elif name in ('tsvm','utsvm'):
        model=TwinSVM(C=p['C'],mu=p['mu'],Cu=p['Cu'],eps=p['eps'],universum=name=='utsvm',random_state=seed)
    elif name in ('faithful_raw','faithful_scaled'):
        model=FaithfulIFUTSVMID(C=p['C'],mu=p['mu'],eps=p['eps'])
    elif name=='paper':
        model=IFUTSVMID(**p,universum_side2='symmetric',random_state=seed)
    else:
        raise ValueError(name)
    steps=[('encode',encoder),('scale','passthrough' if name=='faithful_raw' else StandardScaler())]
    if name=='smote_svm':
        steps.append(('smote',AdaptiveSMOTE(random_state=seed)))
    return Pipeline([*steps,('model',model)])
