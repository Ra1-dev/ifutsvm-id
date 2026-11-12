"""Training-only learning/validation curves with fixed illustrative parameters.

These are diagnostics, not another parameter-selection stage. The original
outer test split is excluded from every fit and every plotted metric.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from ifutsvm.data import load
from ifutsvm.evaluation import binary_metrics,minority_label,minority_scores
from experiments.registry import make_model
from experiments.protocol import split_plan
from experiments.run_all import write_json,source_hash,digest


def plan():
    config=json.loads(Path('results/experiment_config.json').read_text())
    return dict(dataset=config['datasets'][0],outer_split='holdout-0',seed=0,inner_folds=config['inner_folds'],
        model='paper',parameters=dict(C=1.,C3=1.,Cu=1.,eps=.3,mu=1.),
        learning_fractions=[.2,.4,.6,.8,1.],validation={'C':[10.**i for i in range(-3,4)],'mu':[2.**i for i in range(-3,4)]},
        metrics=['accuracy','balanced_accuracy'],selection='None: default parameters are illustrative; curves never select a main comparison model.',
        dataset_rule='First dataset in the pre-specified list, not selected for favorable curves.',
        split_rule='Only outer training rows; three stratified validation folds. Learning subsets use nested class-stratified prefixes within each training fold.',
        uncertainty='Mean and sample SD over validation folds; SD is descriptive fold variation, not a confidence interval.',
        source_hash=source_hash(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def run(config,root):
    X,y=load(config['dataset']);schema=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets'][config['dataset']]['attributes']
    split,seed,outer_train,outer_test=next(iter(split_plan(y,'holdout')))
    assert split==config['outer_split'] and seed==config['seed']
    folds=list(StratifiedKFold(config['inner_folds'],shuffle=True,random_state=seed).split(X[outer_train],y[outer_train]))
    fingerprint=digest(config);path=root/'curves.json'
    result=json.loads(path.read_text()) if path.exists() else dict(fingerprint=fingerprint,plan=config,records={},complete=False)
    assert result['fingerprint']==fingerprint,'Curve protocol/source changed; retain the original evidence separately'
    jobs=[('learning',fraction,config['parameters']) for fraction in config['learning_fractions']]
    jobs += [(name,value,{**config['parameters'],name:value}) for name,values in config['validation'].items() for value in values]
    for family,value,parameters in jobs:
        for fold,(it,iv) in enumerate(folds):
            train=outer_train[it];validation=outer_train[iv]
            if family=='learning':
                rng=np.random.default_rng(np.random.SeedSequence([20261116,seed,fold]))
                selected=[]
                for label in np.unique(y[train]):
                    indices=rng.permutation(train[y[train]==label])
                    selected.extend(indices[:max(1,int(np.floor(value*len(indices))))])
                train=np.array(sorted(selected))
            assert not np.intersect1d(train,validation).size
            assert not np.intersect1d(np.r_[train,validation],outer_test).size
            key=f'{family}/{value:g}/fold-{fold}'
            if key in result['records']: continue
            model=make_model('paper',schema,parameters,seed)
            start=time.perf_counter();model.fit(X[train],y[train]);seconds=time.perf_counter()-start
            positive=minority_label(y[train]);record={}
            for name,indices in [('training',train),('validation',validation)]:
                pred=model.predict(X[indices]);scores=minority_scores(model,X[indices],positive)
                record[name]=dict(metrics=binary_metrics(y[indices],pred,scores,positive,seconds),
                    indices=indices.tolist(),y=y[indices].tolist(),prediction=pred.tolist(),scores=scores.tolist())
            scale=model.named_steps['scale']
            record.update(family=family,value=value,fold=fold,parameters=parameters,seed=seed,positive_label=str(positive),
                outer_train_indices=outer_train.tolist(),outer_test_indices=outer_test.tolist(),
                scaler=dict(mean=scale.mean_.tolist(),var=scale.var_.tolist(),n=int(scale.n_samples_seen_)))
            result['records'][key]=record;write_json(path,result)
        print(f'Curve {family}={value:g}: {len(result["records"])} fits',flush=True)
    groups={}
    for family,value,parameters in jobs:
        rows=[result['records'][f'{family}/{value:g}/fold-{fold}'] for fold in range(len(folds))]
        groups.setdefault(family,[]).append(dict(value=value,training_rows_mean=float(np.mean([len(r['training']['indices']) for r in rows])),
            metrics={metric:{part:dict(mean=float(np.mean([r[part]['metrics'][metric] for r in rows])),
                std=float(np.std([r[part]['metrics'][metric] for r in rows],ddof=1)))
                for part in ('training','validation')} for metric in config['metrics']}))
    result.update(complete=True,groups=groups,fit_count=len(result['records']))
    write_json(path,result);print(f'Curves complete: {result["fit_count"]} training-only fits; no outer test observations')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    root=Path('results');path=root/'curve_plan.json';proposed=plan()
    if path.exists():
        config=json.loads(path.read_text());assert config==proposed,'Curve plan changed'
    else:
        config=proposed;write_json(path,config)
    if args.plan_only:
        print(f'Curve plan: {config["dataset"]}; only outer training; fixed defaults; no model selection');return
    with threadpool_limits(limits=1):run(config,root)


if __name__=='__main__': main()
