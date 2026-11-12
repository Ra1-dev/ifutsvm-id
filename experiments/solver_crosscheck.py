"""Independent untransformed primal QPs solved by OSQP.

Cases and 1e-6 plane/objective limits are fixed before execution. This checks
numerical contracts, not predictive superiority or new research hypotheses.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import cvxpy as cp
import numpy as np
from threadpoolctl import threadpool_limits
from ifutsvm import IFUTSVMID
from ifutsvm.universum import ConditionalGaussianUniversum
from experiments.run_all import source_hash

CASES=[
    (1,3,3,'random','symmetric','reduced',.3,False),
    (4,4,2,'duplicates','paper_primal','full',.7,True),
    (3,4,4,'constant','paper_dual','reduced',.7,True),
    (3,30,3,'random','symmetric','full',0.,False),
    (5,9,1,'random','paper_dual','full',2.,True),
    (4,11,4,'duplicates','paper_primal','reduced',2.,False),
]
TOLERANCE=1e-6


def check_case(index):
    m1,m2,features,geometry,side,basis,penalty,generative=CASES[index]
    rng=np.random.default_rng(20261121+index)
    X=rng.normal(size=(m1+m2,features))
    if geometry=='constant':X[:]=0
    elif geometry=='duplicates':X[1::2]=X[::2][:len(X[1::2])]
    labels=['minority-z','majority-a'] if index%2 else [1,-1]
    y=np.repeat(labels,[m1,m2])
    model=IFUTSVMID(C=.7,C3=.4,Cu=penalty,eps=.27,mu=1.3,
        universum=ConditionalGaussianUniversum() if generative else 'random_average',
        universum_side2=side,kernel_basis=basis,random_state=20261131+index).fit(X,y)
    trace=model.trace_;D=trace['D']
    def design(points):
        # Independent distance expansion: do not call production kernel helpers.
        kernel=np.exp(-np.sum((points[:,None,:]-D[None,:,:])**2,axis=2)/(2*model.mu**2))
        return np.column_stack((kernel,np.ones(len(points))))
    E,F,Fs,G,Gs=[design(trace[key]) for key in ['X1','X2','X2_star','U','U_star']]
    sign=-1 if side=='symmetric' else 1
    rhs=1-model.eps if side=='paper_dual' else model.eps-1
    planes=[]
    for key,P,Q,U,weights,sense,target in [
        ('z1',E,-Fs,Gs,trace['s_maj_star'],1,model.eps-1),
        ('z2',F,E,G,trace['s_min'],sign,rhs)]:
        z=cp.Variable(P.shape[1]);xi=cp.Variable(len(Q),nonneg=True);lam=cp.Variable(len(U),nonneg=True)
        problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(P@z)+model.C3/2*cp.sum_squares(z)
            +model.C*weights@xi+model.Cu*cp.sum(lam)),[Q@z>=1-xi,sense*U@z>=target-lam])
        problem.solve(solver='OSQP',eps_abs=1e-10,eps_rel=1e-10,max_iter=100000,polishing=True)
        assert problem.status=='optimal',(index,key,problem.status)
        actual=trace[key]
        plane_error=float(np.linalg.norm(actual-z.value)/max(1,np.linalg.norm(z.value)))
        objective=.5*np.sum((P@actual)**2)+model.C3/2*np.sum(actual**2)
        objective+=model.C*weights@np.maximum(0,1-Q@actual)+model.Cu*np.maximum(0,target-sense*U@actual).sum()
        objective_error=float(abs(objective-problem.value)/max(1,abs(problem.value)))
        assert plane_error<=TOLERANCE and objective_error<=TOLERANCE,(index,key,plane_error,objective_error)
        planes.append(dict(plane=key,scaled_plane_error=plane_error,scaled_objective_error=objective_error,
            osqp_status=problem.status,osqp_iterations=problem.solver_stats.num_iters))
    return dict(case=index,counts=[m1,m2],features=features,geometry=geometry,side=side,basis=basis,
        Cu=penalty,conditional_gaussian=generative,planes=planes)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/verification/solver_crosscheck.json')
    args=parser.parse_args();start=time.perf_counter()
    with threadpool_limits(limits=1):rows=[check_case(index) for index in range(len(CASES))]
    result=dict(complete=True,source_hash=source_hash(),cases=rows,plane_comparisons=sum(len(r['planes']) for r in rows),
        tolerance=TOLERANCE,seconds=time.perf_counter()-start,
        versions={name:version(name) for name in ['cvxpy','osqp','clarabel']},
        maximum_scaled_plane_error=max(p['scaled_plane_error'] for r in rows for p in r['planes']),
        maximum_scaled_objective_error=max(p['scaled_objective_error'] for r in rows for p in r['planes']),
        statement='Fixed synthetic numerical contracts; independent RBF construction and untransformed primal solved with OSQP; no benchmark refit or performance selection.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__=='__main__':main()
