"""Kernel twin-SVM baselines with explicit, testable primal contracts."""
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.utils.validation import validate_data,check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from ._numeric import rbf,aug,hinge_plane


class TwinSVM(ClassifierMixin,BaseEstimator):
    """TSVM or symmetric UTSVM with a fixed numerical ridge.

    Configurable training basis; no fuzzy weights. The small ridge is
    explicit because coefficient uniqueness is needed for reproducible solves.
    See docs/BASELINES.md for the exact problems and original references.
    """
    def __init__(self,C=1.0,mu=1.0,Cu=1.0,eps=.3,universum=False,ridge=1e-5,random_state=None,kernel_basis='reduced'):
        self.C=C; self.mu=mu; self.Cu=Cu; self.eps=eps
        self.universum=universum; self.ridge=ridge; self.random_state=random_state
        self.kernel_basis=kernel_basis

    def fit(self,X,y):
        X,y=validate_data(self,X,y,dtype=np.float64)
        check_classification_targets(y)
        self.classes_,counts=np.unique(y,return_counts=True)
        if len(counts)!=2:
            raise ValueError('TwinSVM requires two classes')
        for name in ('C','mu','ridge'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(self.Cu) or self.Cu<0 or not np.isfinite(self.eps):
            raise ValueError('invalid Universum penalty/margin')
        i=int(np.argmin(counts))
        self.minority_=self.classes_[i]; self.majority_=self.classes_[1-i]
        X1,X2=X[y==self.minority_],X[y==self.majority_]
        if self.kernel_basis not in ('reduced','full'):
            raise ValueError('kernel_basis must be reduced or full')
        rng=np.random.default_rng(self.random_state)
        X2_star=X2[rng.choice(len(X2),len(X1),replace=False)]
        D=np.vstack((X1,X2_star if self.kernel_basis=='reduced' else X2))
        E=aug(rbf(X1,D,self.mu)); F=aug(rbf(X2,D,self.mu))
        Q1,Q2=-F,E
        rhs1,rhs2=np.ones(len(F)),np.ones(len(E))
        c1,c2=np.full(len(F),self.C),np.full(len(E),self.C)
        U=np.empty((0,X.shape[1]))
        if self.universum:
            count=max(len(X2)-len(X1),(len(X1)+1)//2)
            U=(X1[rng.integers(len(X1),size=count)]+X2[rng.integers(len(X2),size=count)])/2
            G=aug(rbf(U,D,self.mu))
            Q1=np.vstack((Q1,G)); Q2=np.vstack((Q2,-G))
            rhs1=np.r_[rhs1,np.full(count,self.eps-1)]; rhs2=np.r_[rhs2,np.full(count,self.eps-1)]
            c1=np.r_[c1,np.full(count,self.Cu)]; c2=np.r_[c2,np.full(count,self.Cu)]
        z1=hinge_plane(E,Q1,rhs1,c1,self.ridge)
        z2=hinge_plane(F,Q2,rhs2,c2,self.ridge)
        self.trace_=dict(X1=X1.copy(),X2=X2.copy(),X2_star=X2_star.copy(),D=D,U=U,z1=z1,z2=z2)
        return self

    def _distances(self,X):
        check_is_fitted(self,'trace_')
        X=validate_data(self,X,reset=False,dtype=np.float64)
        K=aug(rbf(X,self.trace_['D'],self.mu))
        return tuple(np.abs(K@self.trace_[k])/max(np.linalg.norm(self.trace_[k][:-1]),np.finfo(float).tiny)
                     for k in ('z1','z2'))

    def predict(self,X):
        a,b=self._distances(X)
        return np.where(a<b,self.minority_,self.majority_)

    def decision_function(self,X):
        a,b=self._distances(X)
        return b-a if self.minority_==self.classes_[1] else a-b
