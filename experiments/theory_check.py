"""Falsifiable checks of the conditional law and its transfer to actual IFUTSVM."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ifutsvm import IFUTSVMID
from ifutsvm.evaluation import save_run,minority_scores
from ifutsvm.universum import ConditionalGaussianUniversum
from experiments.run_all import write_json,source_hash,digest

RATIOS=(1,3,9,30)
PENALTIES=(.1,1.,10.)
SEEDS=tuple(range(12))


def analytical():
    generator=ConditionalGaussianUniversum()
    rng=np.random.default_rng(20261111)
    A=rng.normal(size=(31,3))-np.array([1.,.2,-.1])
    B=rng.normal(size=(233,3))+np.array([1.,.2,-.1])
    t=generator.parameters(A,B)
    U=generator(A,B,50000,np.random.default_rng(20261112))
    target=t['conditional_covariance']
    moments=dict(samples=len(U),
        maximum_absolute_log_density_ratio=float(np.max(np.abs((U-t['midpoint'])@t['normal']))),
        mean_error_l2=float(np.linalg.norm(U.mean(0)-t['midpoint'])),
        relative_covariance_error=float(np.linalg.norm(np.cov(U,rowvar=False)-target)/np.linalg.norm(target)))
    rows=[]
    d=2.; alpha=1.; strength=1.
    for ratio in RATIOS:
        prior=-np.log(ratio)/d; anchored=alpha*prior/(alpha+strength)
        risk=lambda q:.5*(1-norm.cdf(q+d/2)+norm.cdf(q-d/2))
        rows.append(dict(ratio=ratio,prior_threshold=float(prior),anchored_threshold=float(anchored),
                         prior_balanced_error=float(risk(prior)),anchored_balanced_error=float(risk(anchored)),
                         balanced_optimal_error=float(risk(0))))
    return dict(conditional_moments=moments,threshold_surrogate=dict(separation=d,alpha=alpha,lambda_=strength,rows=rows))


def empirical(directory,fingerprint):
    rows=[]
    covariance=np.array([[1.,.25],[.25,1.5]])
    center=np.array([1.,0.]); delta=2*center
    population_separation=np.sqrt(delta@np.linalg.solve(covariance,delta))
    for ratio in RATIOS:
        for seed in SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([20261113,ratio,seed]))
            A=rng.multivariate_normal(-center,covariance,20)
            B=rng.multivariate_normal(center,covariance,20*ratio)
            X=np.vstack((A,B)); y=np.r_[np.zeros(len(A),int),np.ones(len(B),int)]
            test_rng=np.random.default_rng(np.random.SeedSequence([20261114,seed]))
            Xt=np.vstack((test_rng.multivariate_normal(-center,covariance,1000),
                          test_rng.multivariate_normal(center,covariance,1000)))
            yt=np.r_[np.zeros(1000,int),np.ones(1000,int)]
            configurations=[('none',0.)]+[(name,cu) for cu in PENALTIES for name in ('random_average','conditional_gaussian')]
            for name,cu in configurations:
                path=directory/f'ratio-{ratio}'/f'seed-{seed}'/f'{name}-Cu-{cu:g}.json'
                if path.exists():
                    record=json.loads(path.read_text())
                    assert record['metadata']['fingerprint']==fingerprint
                else:
                    strategy=ConditionalGaussianUniversum() if name=='conditional_gaussian' else 'random_average'
                    estimator=IFUTSVMID(C=1,C3=1,Cu=cu,eps=.3,mu=1,universum=strategy,random_state=seed)
                    pipeline=make_pipeline(StandardScaler(),estimator)
                    start=time.perf_counter(); pipeline.fit(X,y); seconds=time.perf_counter()-start
                    grid=np.linspace(-4,4,513)
                    grid_X=np.outer(grid,delta/population_separation)
                    response=pipeline.decision_function(grid_X)
                    changes=np.flatnonzero(np.signbit(response[:-1])!=np.signbit(response[1:]))
                    roots=[float(grid[i]-response[i]*(grid[i+1]-grid[i])/(response[i+1]-response[i])) for i in changes]
                    prediction=pipeline.predict(Xt); scores=minority_scores(pipeline,Xt,0)
                    record=save_run(path,y=yt,prediction=prediction,scores=scores,positive_label=0,
                        training_seconds=seconds,train_indices=np.arange(len(X)),test_indices=np.arange(len(X),len(X)+len(Xt)),
                        metadata=dict(fingerprint=fingerprint,ratio=ratio,seed=seed,strategy=name,Cu=cu,
                            C=1,C3=1,eps=.3,mu=1,training_minority=20,training_majority=20*ratio,
                            mean1=(-center).tolist(),mean2=center.tolist(),covariance=covariance.tolist(),
                            decision_line_roots=roots,nearest_boundary_abs=min(map(abs,roots)) if roots else None),csv_output=True)
                rows.append(dict(ratio=ratio,seed=seed,strategy=name,Cu=cu,balanced_accuracy=record['balanced_accuracy'],
                                 nearest_boundary_abs=record['metadata']['nearest_boundary_abs'],
                                 roots=len(record['metadata']['decision_line_roots']),file=path.as_posix()))
            print(f'Theory transfer completed ratio={ratio}, seed={seed}',flush=True)
    contrasts=[]
    for ratio in RATIOS:
        for cu in PENALTIES:
            changes=[]; boundary=[]
            for seed in SEEDS:
                get=lambda name:next(r for r in rows if r['ratio']==ratio and r['Cu']==cu and r['seed']==seed and r['strategy']==name)
                random,generated=get('random_average'),get('conditional_gaussian')
                changes.append(generated['balanced_accuracy']-random['balanced_accuracy'])
                if random['nearest_boundary_abs'] is not None and generated['nearest_boundary_abs'] is not None:
                    boundary.append(generated['nearest_boundary_abs']-random['nearest_boundary_abs'])
            contrasts.append(dict(ratio=ratio,Cu=cu,seeds=len(changes),
                mean_balanced_accuracy_delta=float(np.mean(changes)),std_delta=float(np.std(changes,ddof=1)),
                boundary_pairs=len(boundary),mean_boundary_distance_delta=float(np.mean(boundary)) if boundary else None))
    high=[r for r in contrasts if r['ratio'] in (9,30)]
    performance=float(np.mean([r['mean_balanced_accuracy_delta'] for r in high]))
    boundary_values=[r['mean_boundary_distance_delta'] for r in high if r['mean_boundary_distance_delta'] is not None]
    movement=float(np.mean(boundary_values)) if boundary_values else None
    return dict(runs=rows,contrasts=contrasts,high_imbalance_mean_accuracy_delta=performance,
                high_imbalance_mean_boundary_distance_delta=movement,
                transfer_prediction_supported=performance>0 and movement is not None and movement<0,
                interpretation='The actual estimator transfer is supported only if both predeclared directions hold; no penalty or seed is discarded.')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--analytical-only',action='store_true')
    parser.add_argument('--output',default='results/research'); args=parser.parse_args()
    root=Path(args.output)
    fingerprint=digest(dict(product=source_hash(),script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           generator=hashlib.sha256(Path('src/ifutsvm/universum/__init__.py').read_bytes()).hexdigest()))
    plan=dict(ratios=RATIOS,penalties=PENALTIES,seeds=SEEDS,fingerprint=fingerprint,
              criterion='Across ratios 9 and 30 and all fixed penalties/seeds: positive mean balanced-accuracy difference and negative mean absolute-boundary-distance difference versus random averaging.',
              negative_control='ratio=1',selection='none',scope='The threshold surrogate is not the IFUTSVM objective; transfer is an empirical hypothesis.')
    plan_path=root/'theory_plan.json'
    if plan_path.exists():
        registered=json.loads(plan_path.read_text())
        for key in ('ratios','penalties','seeds','criterion','negative_control','selection','scope'):
            assert json.dumps(registered[key])==json.dumps(plan[key]),'Pre-pre-specified prediction changed'
        # Preserve the original fingerprint and prediction file verbatim. A
        # fresh replay has its own operational fingerprint in each run record.
    else:
        write_json(plan_path,plan)
    with threadpool_limits(limits=1):
        result=analytical()
        write_json(root/'theory_analytical.json',result)
        print(json.dumps(result,indent=2),flush=True)
        if not args.analytical_only:
            result=empirical(root/'theory_runs',fingerprint)
            write_json(root/'theory_empirical.json',result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('runs','contrasts')},indent=2),flush=True)


if __name__=='__main__': main()
