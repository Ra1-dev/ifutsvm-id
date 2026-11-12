"""Faithful numerical recovery must retain the exact box-QP objective."""
import numpy as np
import pytest
import cvxpy as cp
from ifutsvm.reference import _box
import ifutsvm.reference as reference


@pytest.mark.parametrize('failures',[1,2])
def test_unit_box_restart_matches_independent_primal(monkeypatch,failures):
    rng=np.random.default_rng(981)
    B=rng.normal(size=(12,19)); H=B.T@B
    f=rng.normal(size=19); bound=np.geomspace(.001,100,19); bound[3]=0
    original=reference.qp
    count=0
    def initial_failure(P,q,A,b,**kwargs):
        nonlocal count
        count+=1
        if count<=failures:
            raise RuntimeError('forced initial-coordinate failure')
        return original(P,q,A,b,**kwargs)
    monkeypatch.setattr(reference,'qp',initial_failure)
    actual=_box(H,f,bound)
    x=cp.Variable(19)
    problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(B@x)+f@x),[x>=0,x<=bound])
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12)
    assert count==failures+1
    assert np.linalg.norm(B@(actual-x.value))<=1e-7
    assert abs((.5*actual@H@actual+f@actual)-problem.value)<=1e-8
    assert actual[3]==0
