"""Resumable shared-budget benchmark with compact typed evidence tables."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from ifutsvm.data import DATASETS,load
from ifutsvm.evaluation import binary_metrics,minority_label,minority_scores
from experiments.protocol import split_plan
from experiments.registry import MODELS,candidates,make_model
from experiments.storage import RunStore
from experiments.checksums import source_file_digest


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_hash():
    paths=sorted(Path('src/ifutsvm').glob('*.py'))+[Path('experiments')/n for n in
        ('registry.py','run_all.py','protocol.py','storage.py')]
    return digest({p.as_posix():source_file_digest(p) for p in paths})


def write_json(path,record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    temp.replace(path)


def fit_evaluate(store,key,name,schema,p,seed,X,y,train,test,metadata):
    metadata={**metadata,'parameters':p,'seed':seed,'model':name}
    cache_key=digest({**metadata,'train':train.tolist(),'test':test.tolist()})
    cached=store.get(key)
    if cached is not None:
        if cached['metadata']['cache_key']!=cache_key: raise ValueError(f'stale cache: {key}')
        return cached
    metadata['cache_key']=cache_key
    model=make_model(name,schema,p,seed)
    metadata['effective_estimator_parameters']=model.named_steps['model'].get_params(deep=False)
    start=time.perf_counter(); stage='fit'
    try:
        model.fit(X[train],y[train]); seconds=time.perf_counter()-start; stage='evaluation'
        scale=model.named_steps['scale']
        metadata['scaler']=None if isinstance(scale,str) else dict(mean=scale.mean_.tolist(),var=scale.var_.tolist(),n=int(scale.n_samples_seen_))
        if any(a['nominal'] for a in schema):
            metadata['categories']=[c.tolist() for c in model.named_steps['encode'].named_transformers_['nominal'].categories_]
        positive=minority_label(y[train]); pred=model.predict(X[test]); scores=minority_scores(model,X[test],positive)
        record=binary_metrics(y[test],pred,scores,positive,seconds)
        record.update(positive_label=str(positive),y=y[test].tolist(),prediction=pred.tolist(),scores=scores.tolist(),
                      train_indices=train.tolist(),test_indices=test.tolist(),metadata=metadata)
    except Exception as ex:
        record=dict(error=f'{type(ex).__name__}: {ex}',error_stage=stage,training_seconds=time.perf_counter()-start,
                    train_indices=train.tolist(),test_indices=test.tolist(),metadata=metadata)
    return store.put(key,record)


def run_split(store,dataset,protocol,split_index,split_id,seed,train,test,name,config,X,y,manifest):
    proposals=candidates(DATASETS.index(dataset),split_index,protocol,config['budget'])
    inner=list(StratifiedKFold(config['inner_folds'],shuffle=True,random_state=seed).split(X[train],y[train]))
    provenance=dict(dataset=dataset,protocol=protocol,split=split_id,source_hash=config['source_hash'],
        dataset_sha256=manifest['sha256'],versions=config['versions'],outer_train_indices=train.tolist(),outer_test_indices=test.tolist())
    summaries=[]
    for ci,p in enumerate(proposals):
        records=[fit_evaluate(store,f'{split_id}/inner/candidate-{ci}-fold-{fold}',name,manifest['attributes'],p,seed,
            X,y,train[it],train[iv],{**provenance,'candidate':ci,'fold':fold,'role':'inner_validation'})
            for fold,(it,iv) in enumerate(inner)]
        valid=all('error' not in r for r in records)
        summaries.append(dict(candidate=ci,parameters=p,valid=valid,
            balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in records])) if valid else None,
            accuracy=float(np.mean([r['accuracy'] for r in records])) if valid else None,
            errors=[r['error'] for r in records if 'error' in r],fit_seconds=sum(r['training_seconds'] for r in records)))
    store.put(f'{split_id}/selection',dict(candidate_seed=[20261110,DATASETS.index(dataset),split_index,int(protocol=='nested_cv')],
              candidates=summaries,inner_folds=config['inner_folds'],**provenance))
    def selected(metric,label):
        ranking=sorted((r for r in summaries if r['valid']),key=lambda r:(-r[metric],r['candidate']))
        rejected=[]
        for candidate in ranking:
            result=fit_evaluate(store,f"{split_id}/outer_candidates/candidate-{candidate['candidate']}",name,
                manifest['attributes'],candidate['parameters'],seed,X,y,train,test,
                {**provenance,'role':'outer_evaluation','candidate':candidate['candidate']})
            if 'error' not in result:
                return store.put(f'{split_id}/{label}',{**result,'selection_metric':metric,
                    'candidate_budget':len(proposals),'rejected_final_fits':rejected,'selected_candidate':candidate['candidate']})
            if result.get('error_stage')!='fit':
                raise RuntimeError('outer evaluation failed; no test-based reselection: '+result['error'])
            rejected.append(dict(candidate=candidate['candidate'],error=result['error']))
        raise RuntimeError('no valid training-selected fit: '+json.dumps(rejected))
    main=selected('balanced_accuracy','evaluation')
    if name=='paper' and protocol=='holdout':
        selected('accuracy','reproduction')
        if seed==0:
            table=json.loads(Path('data/published_table_sii.json').read_text())
            p={k:table['datasets'][dataset][k] for k in ('C','Cu','eps','mu')}; p['C3']=table['unreported_C3_default']
            fit_evaluate(store,f'{split_id}/published_config',name,manifest['attributes'],p,seed,X,y,train,test,
                {**provenance,'role':'published_config','C3_unreported_default':True})
    return dict(dataset=dataset,protocol=protocol,split=split_id,model=name,training_seconds=main['training_seconds'])


def run_group(job):
    dataset,protocol,name,config=job
    X,y=load(dataset); manifest=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets'][dataset]
    store=RunStore(config['output'],dataset,name,protocol,config['source_hash'])
    outcomes=[]; errors=[]
    try:
        with threadpool_limits(limits=1):
            for si,(split_id,seed,train,test) in enumerate(split_plan(y,protocol)):
                try:
                    result=run_split(store,dataset,protocol,si,split_id,seed,train,test,name,config,X,y,manifest)
                    outcomes.append(result)
                except Exception as ex:
                    errors.append(dict(dataset=dataset,protocol=protocol,split=split_id,model=name,error=f'{type(ex).__name__}: {ex}'))
                records=store.export()
                print(f'SPLIT {dataset} {protocol} {split_id} {name} completed={len(outcomes)} errors={len(errors)}',flush=True)
        index=dict(dataset=dataset,protocol=protocol,model=name,file=store.path.relative_to(config['output']).as_posix(),
                   records=len(records),outer_evaluations=len(outcomes),keys=sorted(records),
                   sha256=hashlib.sha256(store.path.read_bytes()).hexdigest())
        return dict(outcomes=outcomes,errors=errors,index=index)
    finally:
        store.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--budget',type=int,default=40)
    parser.add_argument('--inner-folds',type=int,default=3); parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',default='results'); parser.add_argument('--datasets',default=','.join(DATASETS))
    parser.add_argument('--models',default=','.join(MODELS)); parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.budget<1 or args.workers<1 or args.inner_folds<2: parser.error('invalid budget/workers/folds')
    if not set(args.datasets.split(','))<=set(DATASETS) or not set(args.models.split(','))<=set(MODELS): parser.error('unknown dataset or model')
    root=Path.cwd()
    revision=(subprocess.check_output(['git','-c',f'safe.directory={root.as_posix()}','rev-parse','HEAD'],text=True).strip()
              if (root/'.git').exists() else source_hash())
    config=dict(budget=args.budget,inner_folds=args.inner_folds,workers=args.workers,logical_cpus=os.cpu_count(),
        output=args.output,source_hash=source_hash(),revision=revision,datasets=args.datasets.split(','),models=args.models.split(','),
        baseline_basis='reduced',versions={p:version(p) for p in ('numpy','scipy','scikit-learn','clarabel','imbalanced-learn','pyarrow')})
    config_path=Path(args.output)/'experiment_config.json'
    if config_path.exists():
        old=json.loads(config_path.read_text())
        if any(old[k]!=config[k] for k in ('budget','inner_folds','source_hash','versions','datasets','models')):
            raise ValueError('configuration changed; preserve the old attempt and choose a fresh output')
        config=old
    else: write_json(config_path,config)
    jobs=[(d,p,m,config) for d in config['datasets'] for p in ('holdout','nested_cv') for m in config['models']]
    if args.limit: jobs=jobs[:args.limit]
    execution_path=Path(args.output)/'execution.json'; previous=json.loads(execution_path.read_text()) if execution_path.exists() else {}
    segments=previous.get('segments',[])
    if previous.get('current_segment'): segments.append(previous['current_segment'])
    start=time.perf_counter(); started=datetime.now(timezone.utc).isoformat(); outcomes=[]; errors=[]; indices=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_group,j):j[:3] for j in jobs}
        for future in as_completed(futures):
            try:
                result=future.result(); outcomes.extend(result['outcomes']); errors.extend(result['errors']); indices.append(result['index'])
            except Exception as ex: errors.append(dict(group=futures[future],error=f'{type(ex).__name__}: {ex}'))
            elapsed=time.perf_counter()-start
            write_json(execution_path,dict(config=config,completed=len(outcomes),expected=len(jobs)*5,errors=errors,
                wall_seconds=elapsed+sum(s['wall_seconds'] for s in segments),segments=segments,
                current_segment=dict(started_utc=started,wall_seconds=elapsed,workers=args.workers),
                updated_utc=datetime.now(timezone.utc).isoformat(),outcomes=outcomes))
            write_json(Path(args.output)/'runs/index.json',dict(schema_version=1,source_hash=config['source_hash'],groups=indices))
            print(f'GROUPS {len(indices)}/{len(jobs)}; outer runs {len(outcomes)}/{len(jobs)*5}; errors={len(errors)}',flush=True)
    print(f'Completed {len(outcomes)}/{len(jobs)*5}; errors={len(errors)}; wall_seconds={time.perf_counter()-start:.3f}',flush=True)
    return int(bool(errors))


if __name__=='__main__': raise SystemExit(main())
