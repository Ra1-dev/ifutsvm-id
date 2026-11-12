"""Controlled ablations of paper mode; no benchmark implementation is modified.

The default is numerically identical to IFUTSVMID. Only the declared Universum
law/count, fuzzy weights and side-two sign vary. Common random draws precede
strategy sampling, keeping the reduced basis identical across alternatives.
"""
import numpy as np
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import check_classification_targets
from ifutsvm import IFUTSVMID
from ifutsvm._numeric import aug,rbf,scores,hinge_plane
from ifutsvm.reference import octave_indices
from ifutsvm.universum import ConditionalGaussianUniversum
from experiments.registry import make_model


class IsotropicNoise:
    """N(midpoint of training means, equal-class mean feature variance * I)."""
    def __call__(self,A,B,count,rng):
        variance=(np.var(A,axis=0).mean()+np.var(B,axis=0).mean())/2
        return (A.mean(0)+B.mean(0))/2+np.sqrt(variance)*rng.normal(size=(count,A.shape[1]))


class PublishedUniversum:
    """Original spaced pairs, prefix padding, then internal resampling.

    This returns the actual code count; the ablation estimator accepts it.
    Near-balance exceptions are retained as failed fits, never silently replaced.
    It isolates the U construction, retaining paper mode's random X2* and U*.
    """
    def __call__(self,A,B,count,rng):
        d=abs(len(B)-len(A))
        if d<=1: raise ValueError('published Universum undefined at count difference <=1')
        parts=[X[octave_indices(1,len(X)/d,len(X),len(X))] for X in (A,B)]
        maximum=max(map(len,parts))
        for i,X in enumerate(parts):
            missing=maximum-len(X)
            if missing>len(X): raise ValueError('published prefix padding exceeds available rows')
            if missing: parts[i]=np.vstack((X,X[:missing]))
        U=(parts[0]+parts[1])/2
        return U[octave_indices(1,(len(U)-1)/d,len(U)-(len(U)-1)/d,len(U))]


class ResearchIFUTSVMID(IFUTSVMID):
    def __init__(self,C=1.,C3=1.,Cu=1.,eps=.3,mu=1.,eta=1e-4,
                 universum='random_average',universum_side2='symmetric',
                 kernel_basis='reduced',random_state=None,size_multiplier=1.,if_scores=True):
        super().__init__(C,C3,Cu,eps,mu,eta,universum,universum_side2,kernel_basis,random_state)
        self.size_multiplier=size_multiplier
        self.if_scores=if_scores

    def fit(self,X,y):
        X,y=validate_data(self,X,y,dtype=np.float64)
        check_classification_targets(y)
        self.classes_,counts=np.unique(y,return_counts=True)
        if len(counts)!=2: raise ValueError('exactly two classes required')
        for key in ('C','C3','mu','eta','size_multiplier'):
            if not np.isfinite(getattr(self,key)) or getattr(self,key)<=0:
                raise ValueError(f'{key} must be finite and positive')
        if not np.isfinite(self.Cu) or self.Cu<0 or not np.isfinite(self.eps):
            raise ValueError('invalid penalty or epsilon')
        if self.kernel_basis not in ('reduced','full'): raise ValueError('invalid basis')
        sides={'symmetric':(-1,self.eps-1),'paper_dual':(1,1-self.eps),'paper_primal':(1,self.eps-1)}
        if self.universum_side2 not in sides: raise ValueError('invalid side2')
        self.minority_=self.classes_[np.argmin(counts)]
        self.majority_=self.classes_[1-np.argmin(counts)]
        X1,X2=X[y==self.minority_],X[y==self.majority_]
        m1,m2=len(X1),len(X2); rng=np.random.default_rng(self.random_state)
        idx=rng.choice(m2,m1,replace=False); X2_star=X2[idx]
        g=(m1+1)//2; u=max(1,int(np.ceil(max(m2-m1,g)*self.size_multiplier)))
        if self.universum=='random_average':
            U=(X1[rng.integers(m1,size=u)]+X2[rng.integers(m2,size=u)])/2
        elif callable(self.universum):
            U=np.asarray(self.universum(X1.copy(),X2.copy(),u,rng),float)
            if U.ndim!=2 or U.shape[1]!=X.shape[1] or not len(U) or not np.isfinite(U).all():
                raise ValueError('invalid Universum output')
            if not isinstance(self.universum,PublishedUniversum) and len(U)!=u:
                raise ValueError('wrong Universum count')
        else: raise ValueError('invalid Universum strategy')
        u=len(U); g=min(g,u)
        U_star=U[rng.choice(u,g,replace=False)]
        D=np.vstack((X1,X2_star if self.kernel_basis=='reduced' else X2))
        s=scores(np.vstack((X1,X2)),np.r_[np.ones(m1),np.zeros(m2)],self.mu,self.eta) if self.if_scores else np.ones(m1+m2)
        s_min,s_maj=s[:m1],s[m1:]; s_maj_star=s_maj[idx]
        E,Fs,Gs,F,G=[aug(rbf(a,D,self.mu)) for a in (X1,X2_star,U_star,X2,U)]
        z1=hinge_plane(E,np.vstack((-Fs,Gs)),np.r_[np.ones(m1),np.full(g,self.eps-1)],
                       np.r_[self.C*s_maj_star,np.full(g,self.Cu)],self.C3)
        sign,rhs=sides[self.universum_side2]
        z2=hinge_plane(F,np.vstack((E,sign*G)),np.r_[np.ones(m1),np.full(u,rhs)],
                       np.r_[self.C*s_min,np.full(u,self.Cu)],self.C3)
        self.trace_={k:v.copy() for k,v in locals().items() if k in
                     ('X1','X2','X2_star','U','U_star','D','s_min','s_maj','s_maj_star','z1','z2')}
        return self


def variants():
    """Predeclared one-factor controls around random and generated baselines."""
    result={'no_universum':dict(universum='random_average',Cu=0.),
            'code_universum':dict(universum='published'),
            'random_average':dict(universum='random_average'),
            'isotropic_noise':dict(universum='isotropic'),
            'conditional_gaussian':dict(universum='conditional')}
    for law in ('random_average','conditional_gaussian'):
        base=result[law]
        for size in (.25,.5,2.): result[f'{law}_size_{size:g}']={**base,'size_multiplier':size}
        result[f'{law}_if_off']={**base,'if_scores':False}
        result[f'{law}_paper_dual']={**base,'universum_side2':'paper_dual'}
    return result


def research_pipeline(name,schema,p,seed):
    if name=='svm_balanced': return make_model(name,schema,p,seed)
    specification=variants()[name].copy()
    law=specification.pop('universum')
    strategy={'random_average':'random_average','published':PublishedUniversum(),
              'isotropic':IsotropicNoise(),'conditional':ConditionalGaussianUniversum()}[law]
    estimator=ResearchIFUTSVMID(**{**p,**specification},universum=strategy,random_state=seed)
    pipeline=make_model('paper',schema,p,seed)
    pipeline.set_params(model=estimator)
    return pipeline
