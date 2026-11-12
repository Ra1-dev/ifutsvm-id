"""Training-only numerical investigation of failed faithful box solves."""
import json
from pathlib import Path
import time
import numpy as np
import scipy.sparse as sp
import clarabel
from threadpoolctl import threadpool_limits
from sklearn.model_selection import train_test_split,StratifiedKFold
from ifutsvm.data import load
from experiments.registry import make_model,candidates
import ifutsvm.reference as reference


def main():
    # Recreate the observed training-only failure without any cached arrays.
    X,y=load('abalone9-18')
    train,_=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=3)
    it,_=next(StratifiedKFold(5,shuffle=True,random_state=3).split(X[train],y[train]))
    schema=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets']['abalone9-18']['attributes']
    captured=[]
    original=reference.qp
    def capture(H,f,A,b,**kwargs):
        captured.append((H.copy(),f.copy(),b[:len(f)].copy()))
        return np.zeros(len(f))
    try:
        reference.qp=capture
        model=make_model('faithful_scaled',schema,candidates(0,3,'holdout',4)[0],3)
        with threadpool_limits(limits=1):
            model.fit(X[train[it]],y[train[it]])
    finally:
        reference.qp=original
    records=[]
    with threadpool_limits(limits=1):
        for plane in (1,2):
            H,f,ub=captured[plane-1]
            n=len(f)
            eigen=np.linalg.eigvalsh(H)
            print('PLANE',plane,'eigenrange',eigen[[0,-1]],flush=True)
            for method in ('original','unit_box','no_equilibration','reduced_regularization','scaled_objective'):
                d=ub if method=='unit_box' else np.ones(n)
                scale=max(np.max(np.abs(H)),np.max(np.abs(f))) if method=='scaled_objective' else 1
                P=H*d[:,None]*d[None,:]/scale
                q=f*d/scale
                settings=clarabel.DefaultSettings(); settings.verbose=False
                settings.tol_gap_abs=1e-12; settings.tol_gap_rel=1e-12; settings.tol_feas=1e-12
                settings.max_iter=1000
                if method=='no_equilibration': settings.equilibrate_enable=False
                if method=='reduced_regularization':
                    settings.static_regularization_constant=1e-12
                start=time.perf_counter()
                solver=clarabel.DefaultSolver(sp.triu(sp.csc_matrix(P)).tocsc(),q,
                    sp.vstack((sp.eye(n),-sp.eye(n)),format='csc'),np.r_[ub/d,np.zeros(n)],
                    [clarabel.NonnegativeConeT(2*n)],settings)
                sol=solver.solve(); x=np.array(sol.x)*d
                grad=H@x+f
                gap=float(grad@x-np.minimum(grad,0)@ub)
                row=dict(plane=plane,method=method,status=str(sol.status),iterations=sol.iterations,
                         objective=float(.5*x@H@x+f@x),box_gap=gap,
                         feasibility=max(float(np.max(-x)),float(np.max(x-ub)),0),
                         seconds=time.perf_counter()-start,
                         solver_primal_residual=sol.r_prim,solver_dual_residual=sol.r_dual)
                records.append(row); print(json.dumps(row),flush=True)
    Path('results/verification/box_diagnostic.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__': main()
