"""Train-only, equal-budget ablations and label-noise experiments."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from ifutsvm.data import DATASETS,load
from ifutsvm.evaluation import binary_metrics,minority_label,minority_scores
from experiments.protocol import split_plan
from experiments.registry import candidates
from experiments.research_model import research_pipeline,variants
from experiments.storage import RunStore
from experiments.run_all import write_json,source_hash,digest
from experiments.checksums import source_file_digest

NOISE_DATASETS=('abalone9-18','ecoli4','haberman','pima','yeast3')
NOISE_LEVELS=(0.,.05,.10,.20)


def fingerprint():
    paths=[Path('experiments')/n for n in ('research_model.py','research_run.py')]
    paths.append(Path('src/ifutsvm/universum/__init__.py'))
    return digest(dict(product=source_hash(),research={p.as_posix():source_file_digest(p) for p in paths}))


def plan():
    config=json.loads(Path('results/experiment_config.json').read_text())
    return dict(fingerprint=fingerprint(),product_source_hash=source_hash(),budget=config['budget'],
        versions={name:version(name) for name in config['versions']},
        inner_folds=config['inner_folds'],seeds=list(range(5)),datasets=list(DATASETS),
        candidate_seed_entropy=20261110,variants=variants(),
        design='One-factor ablations around random_average and conditional_gaussian; not a factorial interaction study.',
        size_rule='ceil(max(m2-m1,ceil(m1/2))*multiplier), at least 1; U_star=min(ceil(m1/2),u).',
        code_rule='Published spaced pairs, padding and internal U resampling; all other paper components fixed. Actual code U count retained.',
        generator_hyperparameters='None tuned; numerical covariance ridge fixed at 1e-8.',
        selection_metric='balanced_accuracy',failure_policy='Failed inner fit consumes candidate; rank valid candidates on training CV only. Outer evaluation failure never triggers reselection.',
        noise_datasets=list(NOISE_DATASETS),noise_levels=list(NOISE_LEVELS),
        noise_models=['random_average','conditional_gaussian','svm_balanced'],
        noise_selection='Fixed coverage of nominal/high-IR, small high-IR, small low-IR, larger low-IR, larger moderate-IR data; not selected on strategy outcomes.',
        noise_rule='Flip floor(rate*n_outer_train) rows from a fixed random permutation, shared by models; nested inner validation sees the noisy training labels. Outer test labels always clean.',
        expected_ablation_outer=len(DATASETS)*5*len(variants()),
        expected_noise_outer=len(NOISE_DATASETS)*5*len(NOISE_LEVELS)*3,
        comparison='Paired seed differences against random_average; mean/std over five seeds. Dataset means are the units for across-dataset tests.',
        theory='Original refuted synthetic prediction retained. KEEL transfer and IR relation are external-validity evidence, not a retrospective replacement hypothesis.')


def fit_record(store,key,name,p,seed,X,y_fit,y_eval,train,test,schema,metadata):
    metadata={**metadata,'variant':name,'parameters':p,'seed':seed}
    cache_key=digest({**metadata,'train':train.tolist(),'test':test.tolist()})
    old=store.get(key)
    if old is not None:
        assert old['metadata']['cache_key']==cache_key, f'stale research record {key}'
        return old
    metadata['cache_key']=cache_key
    pipeline=research_pipeline(name,schema,p,seed)
    start=time.perf_counter(); stage='fit'
    try:
        pipeline.fit(X[train],y_fit[train]); seconds=time.perf_counter()-start; stage='evaluation'
        scale=pipeline.named_steps['scale']
        metadata['scaler']=dict(mean=scale.mean_.tolist(),var=scale.var_.tolist(),n=int(scale.n_samples_seen_))
        if any(a['nominal'] for a in schema):
            metadata['categories']=[c.tolist() for c in pipeline.named_steps['encode'].named_transformers_['nominal'].categories_]
        estimator=pipeline.named_steps['model']
        if hasattr(estimator,'trace_'):
            metadata['construction_counts']={k:len(estimator.trace_[k]) for k in ('X1','X2','X2_star','U','U_star')}
        positive=minority_label(y_fit[train]); prediction=pipeline.predict(X[test]); scores=minority_scores(pipeline,X[test],positive)
        result=binary_metrics(y_eval[test],prediction,scores,positive,seconds)
        result.update(positive_label=str(positive),train_indices=train.tolist(),test_indices=test.tolist(),
                      y=y_eval[test].tolist(),prediction=prediction.tolist(),scores=scores.tolist(),metadata=metadata)
    except Exception as ex:
        result=dict(error=f'{type(ex).__name__}: {ex}',error_stage=stage,training_seconds=time.perf_counter()-start,
                    train_indices=train.tolist(),test_indices=test.tolist(),metadata=metadata)
    return store.put(key,result)


def run_dataset(job):
    dataset,wave,root,config=job
    X,y=load(dataset); schema=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets'][dataset]
    store=RunStore(root,dataset,'research',wave,config['fingerprint'])
    outcomes=[]; errors=[]
    names=list(variants()) if wave=='ablation' else config['noise_models']
    levels=[0.] if wave=='ablation' else config['noise_levels']
    try:
        with threadpool_limits(limits=1):
            for split_index,(split_id,seed,train,test) in enumerate(split_plan(y,'holdout')):
                proposals=candidates(DATASETS.index(dataset),split_index,'holdout',config['budget'])
                for rate in levels:
                    noisy=y.copy()
                    rng=np.random.default_rng(np.random.SeedSequence([20261115,DATASETS.index(dataset),seed]))
                    flips=rng.permutation(train)[:int(np.floor(rate*len(train)))]
                    labels=np.unique(y[train]); noisy[flips]=np.where(y[flips]==labels[0],labels[1],labels[0])
                    inner=list(StratifiedKFold(config['inner_folds'],shuffle=True,random_state=seed).split(X[train],noisy[train]))
                    for name in names:
                        prefix=f'{name}/{split_id}/noise-{rate:g}'
                        meta=dict(dataset=dataset,wave=wave,split=split_id,noise_rate=rate,flipped_training_indices=flips.tolist(),
                            source_hash=config['fingerprint'],dataset_sha256=schema['sha256'],
                            outer_train_indices=train.tolist(),outer_test_indices=test.tolist())
                        summary=[]
                        for ci,p in enumerate(proposals):
                            records=[fit_record(store,f'{prefix}/inner/candidate-{ci}-fold-{fold}',name,p,seed,X,noisy,noisy,
                                train[it],train[iv],schema['attributes'],{**meta,'candidate':ci,'fold':fold,'role':'inner_validation'})
                                for fold,(it,iv) in enumerate(inner)]
                            valid=all('error' not in r for r in records)
                            summary.append(dict(candidate=ci,parameters=p,valid=valid,
                                balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in records])) if valid else None,
                                errors=[r['error'] for r in records if 'error' in r]))
                        store.put(prefix+'/selection',dict(metadata=meta,candidates=summary,inner_folds=config['inner_folds'],
                            candidate_seed=[20261110,DATASETS.index(dataset),split_index,0]))
                        ranking=sorted((r for r in summary if r['valid']),key=lambda r:(-r['balanced_accuracy'],r['candidate']))
                        rejected=[]; selected=None
                        for candidate in ranking:
                            ci=candidate['candidate']
                            result=fit_record(store,f'{prefix}/outer_candidates/candidate-{ci}',name,candidate['parameters'],seed,
                                X,noisy,y,train,test,schema['attributes'],{**meta,'candidate':ci,'role':'outer_evaluation'})
                            if 'error' not in result:
                                selected=store.put(prefix+'/evaluation',{**result,'selected_candidate':ci,
                                    'candidate_budget':len(proposals),'selection_metric':'balanced_accuracy','rejected_final_fits':rejected})
                                break
                            rejected.append(dict(candidate=ci,error=result['error']))
                            if result['error_stage']!='fit': break
                        if selected is None:
                            errors.append(dict(dataset=dataset,variant=name,split=split_id,noise_rate=rate,failures=rejected,
                                valid_inner_candidates=len(ranking)))
                        else:
                            outcomes.append(dict(dataset=dataset,variant=name,split=split_id,noise_rate=rate))
                        store.export()
                        print(f'RESEARCH {dataset} {prefix} completed={len(outcomes)} errors={len(errors)}',flush=True)
        return dict(outcomes=outcomes,errors=errors,file=store.path.as_posix())
    finally: store.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--wave',choices=['ablation','noise'],default='ablation')
    parser.add_argument('--workers',type=int,default=2); parser.add_argument('--output',default='results/research')
    parser.add_argument('--datasets'); args=parser.parse_args()
    proposed=plan(); root=Path(args.output); plan_path=root/'ablation_plan.json'
    if plan_path.exists():
        config=json.loads(plan_path.read_text()); assert config==proposed,'Saved research plan changed'
    else: config=proposed; write_json(plan_path,config)
    if args.plan_only:
        print(f"Research plan saved: {len(variants())} variants; {config['expected_ablation_outer']} ablation and {config['expected_noise_outer']} noise evaluations; {config['budget']} shared candidates")
        return
    datasets=args.datasets.split(',') if args.datasets else config['datasets'] if args.wave=='ablation' else config['noise_datasets']
    assert set(datasets)<=set(config['datasets'] if args.wave=='ablation' else config['noise_datasets'])
    start=time.perf_counter(); rows=[]; errors=[]; groups=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_dataset,(d,args.wave,root,config)) for d in datasets]
        for future in as_completed(futures):
            result=future.result(); rows.extend(result['outcomes']); errors.extend(result['errors']); groups.append(result['file'])
            write_json(root/f'{args.wave}_execution.json',dict(completed=len(rows),errors=errors,outcomes=rows,
                expected=config[f'expected_{args.wave}_outer'],wall_seconds=time.perf_counter()-start,fingerprint=config['fingerprint']))
            write_json(root/f'{args.wave}_index.json',dict(files=groups,fingerprint=config['fingerprint']))
    print(f'Research {args.wave}: completed={len(rows)} errors={len(errors)} seconds={time.perf_counter()-start:.3f}')
    return int(bool(errors))


if __name__=='__main__': raise SystemExit(main())
