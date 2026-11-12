"""Course learning comparator and training-selected validation diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from ifutsvm.data import load
from ifutsvm.evaluation import binary_metrics,minority_label,minority_scores
from experiments.registry import make_model
from experiments.protocol import split_plan
from experiments.run_all import write_json,source_hash,digest

PLAN=Path('results/course_curve_plan.json')


def git(*args):
    return subprocess.check_output(['git','-c',f'safe.directory={Path.cwd().as_posix()}',*args])


def proposal():
    path=Path('results/artifact_plan.json');p=json.loads(path.read_text())
    return dict(learning=p['learning'],validation=p['validation'],limitation=p['limitation'],
        metrics=['accuracy','balanced_accuracy'],source_hash=source_hash(),
        artifact_plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        artifact_registration_commit=git('log','-1','--format=%H','--',path.as_posix()).decode().strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def run(plan):
    registration=git('log','-1','--format=%H','--',PLAN.as_posix()).decode().strip()
    assert git('show',registration+':'+PLAN.as_posix())==PLAN.read_bytes()
    assert git('show',registration+':experiments/course_curves.py')==Path(__file__).read_bytes()
    l=plan['learning'];v=plan['validation'];X,y=load(l['dataset'])
    schema=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets'][l['dataset']]['attributes']
    split,seed,outer_train,outer_test=next(iter(split_plan(y,'holdout')))
    assert split==l['outer_split']==v['outer_split']
    folds=list(StratifiedKFold(l['inner_folds'],shuffle=True,random_state=seed).split(X[outer_train],y[outer_train]))
    jobs=[(name,'learning',fraction,l['parameters'][name]) for name in l['models'] for fraction in l['fractions']]
    jobs += [(v['model'],parameter,value,{**v['parameters'],parameter:value}) for parameter,values in v['sweeps'].items() for value in values]
    path=Path('results/course_curves.json');fingerprint=digest(plan)
    result=json.loads(path.read_text()) if path.exists() else dict(plan=plan,fingerprint=fingerprint,
        preregistration_commit=registration,records={},complete=False)
    assert result['fingerprint']==fingerprint
    for name,family,value,parameters in jobs:
        for fold,(it,iv) in enumerate(folds):
            train=outer_train[it];validation=outer_train[iv]
            if family=='learning':
                rng=np.random.default_rng(np.random.SeedSequence([l['subset_seed'],seed,fold]));selected=[]
                for label in np.unique(y[train]):
                    indices=rng.permutation(train[y[train]==label])
                    selected.extend(indices[:max(1,int(np.floor(value*len(indices))))])
                train=np.array(sorted(selected))
            assert not np.intersect1d(train,validation).size
            assert not np.intersect1d(np.r_[train,validation],outer_test).size
            key=f'{name}/{family}/{value:g}/fold-{fold}'
            if key in result['records']:continue
            model=make_model(name,schema,parameters,seed);start=time.perf_counter()
            model.fit(X[train],y[train]);seconds=time.perf_counter()-start
            positive=minority_label(y[train]);record={}
            for role,indices in [('training',train),('validation',validation)]:
                prediction=model.predict(X[indices]);scores=minority_scores(model,X[indices],positive)
                record[role]=dict(indices=indices.tolist(),y=y[indices].tolist(),prediction=prediction.tolist(),scores=scores.tolist(),
                    metrics=binary_metrics(y[indices],prediction,scores,positive,seconds))
            scale=model.named_steps['scale']
            record.update(model=name,family=family,value=value,parameters=parameters,fold=fold,seed=seed,
                positive_label=str(positive),outer_train_indices=outer_train.tolist(),outer_test_indices=outer_test.tolist(),
                scaler=dict(mean=scale.mean_.tolist(),var=scale.var_.tolist(),n=int(scale.n_samples_seen_)))
            result['records'][key]=record;write_json(path,result)
        print(f'Course curve {name}/{family}/{value:g}: {len(result["records"])} fits',flush=True)
    groups={}
    for name,family,value,parameters in jobs:
        rows=[result['records'][f'{name}/{family}/{value:g}/fold-{f}'] for f in range(len(folds))]
        groups.setdefault(name,{}).setdefault(family,[]).append(dict(value=value,
            training_rows_mean=float(np.mean([len(r['training']['indices']) for r in rows])),
            metrics={metric:{role:dict(mean=float(np.mean([r[role]['metrics'][metric] for r in rows])),
                std=float(np.std([r[role]['metrics'][metric] for r in rows],ddof=1)))
                for role in ('training','validation')} for metric in plan['metrics']}))
    result.update(complete=True,groups=groups,fit_count=len(result['records']))
    write_json(path,result);print(f'Course curves complete: {result["fit_count"]} fits; original outer-test rows excluded')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    plan=proposal()
    if PLAN.exists():assert json.loads(PLAN.read_text())==plan,'Course-curve registration changed'
    else:
        assert args.register,'Register and commit before fitting';write_json(PLAN,plan)
    if args.register:print('Pre-specified course curve extension; no fits');return
    with threadpool_limits(limits=1):run(plan)


if __name__=='__main__':main()
