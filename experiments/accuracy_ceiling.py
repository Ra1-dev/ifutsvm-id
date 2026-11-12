"""Single pre-specified accuracy diagnostic; never changes the main benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from ifutsvm.data import load
from ifutsvm.evaluation import binary_metrics, minority_label
from experiments.protocol import split_plan
from experiments.run_all import digest, source_hash, write_json
from experiments.storage import RunStore, read_table, table_path

ROOT = Path('results/accuracy_ceiling')
PLAN = ROOT/'plan.json'


def git(*args):
    return subprocess.check_output(['git', '-c', f'safe.directory={Path.cwd().as_posix()}', *args])


def proposed_plan():
    spaces = {
        'svm_rbf': {'C': [2.**i for i in range(-5,16,2)], 'gamma': [2.**i for i in range(-15,4,2)]},
        'random_forest': {'n_estimators': [300], 'max_features': ['sqrt', .5, 1.],
            'max_depth': [None,3,6,12], 'min_samples_leaf': [1,2,5,10], 'min_samples_split': [2,5,10]},
        'gradient_boosting': {'n_estimators': [100,200,400], 'learning_rate': [.01,.03,.1,.3],
            'max_depth': [1,2,3], 'min_samples_leaf': [1,5,10], 'subsample': [.7,1.]},
        'knn': {'n_neighbors': list(range(1,52,2)), 'weights': ['uniform','distance'], 'p': [1,2]},
    }
    return dict(dataset='yeast1', protocol='holdout', seeds=list(range(5)), inner_folds=3,
        budget=40, candidate_seed_base=20261110, spaces=spaces,
        models=list(spaces), selection_metric='accuracy', tie_break='lowest candidate index',
        class_weights='None for all families; selection is ordinary accuracy',
        preprocessing='StandardScaler fitted only on each inner/outer training fold',
        score_definition='minority-positive decision function if available; otherwise minority probability',
        random_state='outer seed for all stochastic estimators; no seed tuning',
        failure_policy='failed inner candidates consume budget; any failed selected outer fit blocks completion; no test-based reselection',
        criterion=dict(suspect_if_any_mean_accuracy_at_least=.9, published_accuracy_percent=92.14,
            otherwise='Published value unattained by this pre-specified diagnostic on the official KEEL file under protocol A; this finite probe is not a universal mathematical upper bound.'),
        candidates={name:{str(seed):list(ParameterSampler(space,n_iter=40,random_state=20261110+seed))
            for seed in range(5)} for name,space in spaces.items()},
        dataset_sha256=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets']['yeast1']['sha256'],
        source_hash=source_hash(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={p:version(p) for p in ('numpy','scipy','scikit-learn','pyarrow')})


def make_model(name, params, seed):
    if name == 'svm_rbf': model=SVC(kernel='rbf',class_weight=None,random_state=seed,**params)
    elif name == 'random_forest': model=RandomForestClassifier(random_state=seed,n_jobs=1,class_weight=None,**params)
    elif name == 'gradient_boosting': model=GradientBoostingClassifier(random_state=seed,**params)
    elif name == 'knn': model=KNeighborsClassifier(n_jobs=1,**params)
    else: raise ValueError(name)
    return make_pipeline(StandardScaler(),model)


def fit(store, key, name, params, seed, X, y, train, test, metadata):
    metadata={**metadata,'parameters':params,'seed':seed,'model':name}
    cache_key=digest(dict(metadata=metadata,train=train.tolist(),test=test.tolist()))
    cached=store.get(key)
    if cached is not None:
        assert cached['metadata']['cache_key']==cache_key, 'Stale diagnostic cache'
        return cached
    metadata['cache_key']=cache_key
    model=make_model(name,params,seed);start=time.perf_counter();stage='fit'
    try:
        model.fit(X[train],y[train]);seconds=time.perf_counter()-start;stage='evaluation'
        scaler=model[0];metadata['scaler']=dict(mean=scaler.mean_.tolist(),var=scaler.var_.tolist(),n=int(scaler.n_samples_seen_))
        positive=minority_label(y[train]);pred=model.predict(X[test])
        if hasattr(model,'decision_function'):
            scores=model.decision_function(X[test])*(1 if positive==model.classes_[1] else -1)
        else: scores=model.predict_proba(X[test])[:,np.flatnonzero(model.classes_==positive)[0]]
        record=binary_metrics(y[test],pred,scores,positive,seconds)
        record.update(positive_label=str(positive),y=y[test].tolist(),prediction=pred.tolist(),scores=scores.tolist())
    except Exception as exc:
        record=dict(error=f'{type(exc).__name__}: {exc}',error_stage=stage,training_seconds=time.perf_counter()-start)
    record.update(train_indices=train.tolist(),test_indices=test.tolist(),metadata=metadata)
    return store.put(key,record)


def run(plan):
    # Refuse to fit an uncommitted or modified registration/runner.
    revision=git('log','-1','--format=%H','--',PLAN.as_posix()).decode().strip()
    assert revision and git('show',f'{revision}:{PLAN.as_posix()}')==PLAN.read_bytes()
    script=Path(__file__).relative_to(Path.cwd()).as_posix()
    assert git('show',f'{revision}:{script}')==Path(__file__).read_bytes()
    X,y=load(plan['dataset']);fingerprint=digest(plan)
    provenance=dict(fingerprint=fingerprint,preregistration_commit=revision,
        preregistration_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),dataset_sha256=plan['dataset_sha256'])
    run_path=ROOT/'execution.json'
    previous=json.loads(run_path.read_text()) if run_path.exists() else dict(segments=[])
    if previous.get('current_segment'): previous['segments'].append(previous['current_segment'])
    start=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    for name in plan['models']:
        store=RunStore(ROOT,plan['dataset'],name,'holdout',fingerprint)
        try:
            for split,seed,train,test in split_plan(y,'holdout',seeds=plan['seeds']):
                context={**provenance,'split':split,'outer_train_indices':train.tolist(),'outer_test_indices':test.tolist()}
                folds=list(StratifiedKFold(plan['inner_folds'],shuffle=True,random_state=seed).split(X[train],y[train]))
                summaries=[]
                for ci,params in enumerate(plan['candidates'][name][str(seed)]):
                    records=[fit(store,f'{split}/inner/candidate-{ci}-fold-{fi}',name,params,seed,X,y,train[it],train[iv],
                        {**context,'role':'inner_validation','candidate':ci,'fold':fi}) for fi,(it,iv) in enumerate(folds)]
                    valid=all('error' not in r for r in records)
                    summaries.append(dict(candidate=ci,parameters=params,valid=valid,
                        accuracy=float(np.mean([r['accuracy'] for r in records])) if valid else None,
                        errors=[r['error'] for r in records if 'error' in r]))
                candidates=[s for s in summaries if s['valid']]
                if not candidates: store.export();raise RuntimeError('No valid accuracy candidate')
                best=min(candidates,key=lambda s:(-s['accuracy'],s['candidate']))
                store.put(f'{split}/selection',dict(**context,candidates=summaries,selected_candidate=best['candidate'],
                    selection_metric='accuracy',candidate_seed=plan['candidate_seed_base']+seed))
                outer=fit(store,f'{split}/evaluation',name,best['parameters'],seed,X,y,train,test,
                    {**context,'role':'outer_evaluation','candidate':best['candidate']})
                store.export()
                if 'error' in outer: raise RuntimeError(outer['error'])
                write_json(run_path,dict(**provenance,segments=previous['segments'],
                    current_segment=dict(started_utc=started,wall_seconds=time.perf_counter()-start,workers=1),
                    latest_completed=dict(model=name,split=split),complete=False))
                print(f'Diagnostic {name} {split} complete',flush=True)
        finally: store.close()
    summary=aggregate(plan,provenance)
    write_json(ROOT/'summary.json',summary)
    write_json(run_path,dict(**provenance,segments=previous['segments'],
        current_segment=dict(started_utc=started,wall_seconds=time.perf_counter()-start,workers=1),complete=True))
    for name,row in summary['models'].items():
        print(f"{name}: accuracy={row['accuracy_mean_percent']:.9f} +/- {row['accuracy_std_pp']:.9f}; best_split={row['best_split_accuracy_percent']:.9f}")
    print(f"Outer evaluations: {summary['outer_evaluations']}; threshold_triggered={summary['implementation_suspect']}")


def aggregate(plan, provenance):
    models={}
    for name in plan['models']:
        path=table_path(ROOT,plan['dataset'],name,'holdout');records=read_table(path)
        outer=[records[f'holdout-{s}/evaluation'] for s in plan['seeds']]
        assert all('error' not in r for r in outer)
        values=np.array([r['accuracy'] for r in outer])
        models[name]=dict(accuracy_mean_percent=float(values.mean()*100),accuracy_std_pp=float(values.std(ddof=1)*100),
            best_split_accuracy_percent=float(values.max()*100),per_seed_accuracy_percent=(values*100).tolist(),
            selected_candidates=[r['metadata']['candidate'] for r in outer],
            file=path.as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    suspect=any(r['accuracy_mean_percent']>=100*plan['criterion']['suspect_if_any_mean_accuracy_at_least'] for r in models.values())
    return dict(**provenance,complete=True,models=models,outer_evaluations=len(models)*len(plan['seeds']),
        inner_evaluations=len(models)*len(plan['seeds'])*plan['budget']*plan['inner_folds'],
        implementation_suspect=suspect,criterion=plan['criterion'],
        conclusion='Implementation remains suspect; the reproduction question unresolved.' if suspect else plan['criterion']['otherwise'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    proposal=proposed_plan()
    if PLAN.exists():
        plan=json.loads(PLAN.read_text());assert plan==proposal,'Pre-specified diagnostic plan/source changed'
    else:
        assert args.plan_only,'Register and commit the plan before any fits'
        plan=proposal;write_json(PLAN,plan)
    if args.plan_only:
        print(f"Pre-specified diagnostic: {len(plan['models'])} families, {len(plan['seeds'])} seeds, {plan['budget']} candidates, {plan['inner_folds']} folds; no fits")
        return
    with threadpool_limits(limits=1): run(plan)


if __name__=='__main__': main()
