"""compare TSVM/UTSVM to independent direct cvxpy primals."""
import numpy as np
import pytest
import cvxpy as cp
from ifutsvm.baselines import TwinSVM
from ifutsvm._numeric import aug,rbf
from ifutsvm import IFUTSVMID


@pytest.mark.parametrize('universum',[False,True])
@pytest.mark.parametrize('seed',[43,71])
@pytest.mark.parametrize('kernel_basis',['reduced','full'])
def test_baseline_primal_exactness(universum,seed,kernel_basis):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(28,4)); y=np.r_[np.ones(8),-np.ones(20)]
    est=TwinSVM(C=.7,mu=.8,Cu=.3,eps=.4,universum=universum,random_state=seed,kernel_basis=kernel_basis).fit(X,y)
    t=est.trace_; E=aug(rbf(t['X1'],t['D'],.8)); F=aug(rbf(t['X2'],t['D'],.8))
    for key,P,Q,sign in [('z1',E,-F,1),('z2',F,E,-1)]:
        z=cp.Variable(P.shape[1]); xi=cp.Variable(len(Q),nonneg=True)
        objective=.5*cp.sum_squares(P@z)+.5*est.ridge*cp.sum_squares(z)+est.C*cp.sum(xi)
        constraints=[Q@z>=1-xi]
        if universum:
            G=aug(rbf(t['U'],t['D'],.8)); lam=cp.Variable(len(G),nonneg=True)
            objective+=est.Cu*cp.sum(lam)
            constraints.append(sign*G@z>=est.eps-1-lam)
        problem=cp.Problem(cp.Minimize(objective),constraints)
        problem.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12,max_iter=1000)
        assert np.linalg.norm(t[key]-z.value)/np.linalg.norm(z.value)<=1e-5
    if universum:
        paper=IFUTSVMID(random_state=seed).fit(X,y)
        np.testing.assert_array_equal(t['U'],paper.trace_['U'])
        np.testing.assert_array_equal(t['X2_star'],paper.trace_['X2_star'])
        if kernel_basis=='reduced': np.testing.assert_array_equal(t['D'],paper.trace_['D'])
