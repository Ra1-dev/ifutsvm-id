"""Predeclared Table S.II diagnostics, never used for model or seed selection.

All variants use published parameters with explicit C3=1 unless the variant
changes C3. Repeated probes use the issued seeds; expensive/decision-rule and
feature diagnostics use seed zero. Every result, including failures, persists.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from ifutsvm.data import DATASETS,load
from ifutsvm.evaluation import binary_metrics,minority_label,minority_scores
from ifutsvm._numeric import rbf,aug
from experiments.registry import make_model
from experiments.run_all import source_hash,write_json,digest
from experiments.storage import read_table,write_table,table_path

VARIANTS={
    'published_scaled':dict(seeds=list(range(5)),discrepancies=['DSC-34']),
    'raw_features':dict(seeds=list(range(5)),discrepancies=['DSC-01','DSC-02']),
    'paper_dual':dict(seeds=list(range(5)),discrepancies=['DSC-11']),
    'paper_primal':dict(seeds=list(range(5)),discrepancies=['DSC-11']),
    'demo_ridge':dict(seeds=list(range(5)),discrepancies=['DSC-09','DSC-34']),
    'unstratified':dict(seeds=list(range(5)),discrepancies=['DSC-29']),
    'full_basis':dict(seeds=[0],discrepancies=['DSC-30']),
    'rkhs_norm':dict(seeds=[0],discrepancies=['DSC-31']),
    'drop_nominal':dict(seeds=[0],datasets=['abalone9-18'],discrepancies=['DSC-38']),
}


def run_probe(job,records):
    dataset,variant,seed,output,fingerprint=job
    key=f'{variant}/seed-{seed}'
    path=Path(output)/'reproduction_probes'/f'{dataset}.parquet'
    def save(**kwargs):
        record=binary_metrics(kwargs['y'],kwargs['prediction'],kwargs['scores'],
                              kwargs['positive_label'],kwargs['training_seconds'])
        record.update({k:np.asarray(kwargs[k]).tolist() for k in
                       ('y','prediction','scores','train_indices','test_indices')})
        record.update(positive_label=str(kwargs['positive_label']),metadata=kwargs['metadata'])
        records[key]=record
        write_table(path,records)
    if key in records:
        record=records[key]
        assert record['metadata']['fingerprint']==fingerprint
        return dict(dataset=dataset,variant=variant,seed=seed,error=record.get('error'),cached=True)
    table=json.loads(Path('data/published_table_sii.json').read_text())
    manifest=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets'][dataset]
    X,y=load(dataset)
    train,test=train_test_split(np.arange(len(y)),test_size=.3,random_state=seed,
                               stratify=None if variant=='unstratified' else y)
    p={k:table['datasets'][dataset][k] for k in ('C','Cu','eps','mu')}
    p['C3']=table['unreported_C3_default']
    metadata=dict(dataset=dataset,variant=variant,seed=seed,parameters=p,
                  fingerprint=fingerprint,dataset_sha256=manifest['sha256'],
                  discrepancies=VARIANTS[variant]['discrepancies'],role='fixed_reproduction_probe')
    # The mandatory published-config probe already touches this configuration.
    # Reuse that exact evaluation rather than evaluating it again.
    if variant=='published_scaled':
        main_path=table_path(output,dataset,'paper','holdout')
        main_records=read_table(main_path) if main_path.exists() else {}
        prefix=f'holdout-{seed}/'
        possible=[prefix+'published_config',*sorted(k for k in main_records if k.startswith(prefix+'outer_candidates/'))]
        for prior in possible:
            if prior not in main_records: continue
            record=main_records[prior]
            if record['metadata']['parameters']==p and 'error' not in record:
                assert record['train_indices']==train.tolist() and record['test_indices']==test.tolist()
                metadata.update(reused_from=main_path.as_posix()+'#'+prior,scaler=record['metadata']['scaler'])
                save(y=record['y'],prediction=record['prediction'],scores=record['scores'],
                    positive_label=record['positive_label'],training_seconds=record['training_seconds'],
                    train_indices=train,test_indices=test,metadata=metadata)
                return dict(dataset=dataset,variant=variant,seed=seed,error=None,reused=True)
    schema=manifest['attributes']
    if variant=='drop_nominal':
        keep=[i for i,a in enumerate(schema) if not a['nominal']]
        X=X[:,keep]; schema=[schema[i] for i in keep]; metadata['kept_features']=keep
    model=make_model('paper',schema,p,seed)
    if variant=='raw_features': model.set_params(scale='passthrough')
    if variant in ('paper_dual','paper_primal'): model.set_params(model__universum_side2=variant)
    if variant=='demo_ridge': model.set_params(model__C3=1e-5)
    if variant=='full_basis': model.set_params(model__kernel_basis='full')
    metadata['effective_C3']=model.named_steps['model'].C3
    metadata['kernel_basis']=model.named_steps['model'].kernel_basis
    metadata['universum_side2']=model.named_steps['model'].universum_side2
    start=time.perf_counter(); stage='fit'
    with threadpool_limits(limits=1):
        try:
            model.fit(X[train],y[train]); seconds=time.perf_counter()-start; stage='evaluation'
            scale=model.named_steps['scale']
            metadata['scaler']=None if isinstance(scale,str) else dict(mean=scale.mean_.tolist(),var=scale.var_.tolist(),n=int(scale.n_samples_seen_))
            positive=minority_label(y[train])
            if variant=='rkhs_norm':
                est=model.named_steps['model']; trace=est.trace_
                features=model[:-1].transform(X[test]); K=aug(rbf(features,trace['D'],est.mu))
                gram=rbf(trace['D'],trace['D'],est.mu)
                distances=[]
                for plane_key in ('z1','z2'):
                    w=trace[plane_key][:-1]
                    norm=np.sqrt(max(float(w@gram@w),np.finfo(float).tiny))
                    distances.append(np.abs(K@trace[plane_key])/norm)
                a,b=distances
                pred=np.where(a<b,est.minority_,est.majority_)
                scores=b-a  # minority-positive by construction
            else:
                pred=model.predict(X[test]); scores=minority_scores(model,X[test],positive)
            save(y=y[test],prediction=pred,scores=scores,positive_label=positive,
                training_seconds=seconds,train_indices=train,test_indices=test,metadata=metadata)
            error=None
        except Exception as ex:
            error=f'{type(ex).__name__}: {ex}'
            records[key]=dict(error=error,error_stage=stage,training_seconds=time.perf_counter()-start,
                             train_indices=train.tolist(),test_indices=test.tolist(),metadata=metadata)
            write_table(path,records)
    return dict(dataset=dataset,variant=variant,seed=seed,error=error)


def run_dataset(job):
    dataset,output,fingerprint=job
    path=Path(output)/'reproduction_probes'/f'{dataset}.parquet'
    records=read_table(path) if path.exists() else {}
    rows=[]
    for variant,setup in VARIANTS.items():
        if dataset not in setup.get('datasets',DATASETS): continue
        for seed in setup['seeds']:
            row=run_probe((dataset,variant,seed,output,fingerprint),records)
            rows.append(row)
            print(f"PROBE {dataset} {variant} seed={seed} error={row['error']}",flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results')
    parser.add_argument('--workers',type=int,default=2); parser.add_argument('--datasets',default=','.join(DATASETS))
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--repair-keys',action='store_true',help='Repair the recorded RKHS row-key shadowing error without refitting')
    args=parser.parse_args()
    fingerprint=digest(dict(product=source_hash(),probe=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    plan=dict(variants=VARIANTS,fingerprint=fingerprint,selection='none; all fixed variants retained',
              datasets=args.datasets.split(','),C3='unreported in paper; explicit default, except demo_ridge')
    if args.repair_keys:
        previous=json.loads((Path(args.output)/'reproduction_probe_plan.json').read_text())
        assert previous['variants']==plan['variants']
        renamed=0; migrated=0
        for dataset in plan['datasets']:
            path=Path(args.output)/'reproduction_probes'/f'{dataset}.parquet'
            if not path.exists(): continue
            original=read_table(path); repaired={}
            for key,record in original.items():
                meta=record['metadata']; assert meta['dataset']==dataset
                assert meta['fingerprint']==previous['fingerprint']
                expected=f"{meta['variant']}/seed-{meta['seed']}"
                assert key==expected or (key=='z2' and expected=='rkhs_norm/seed-0')
                assert expected not in repaired
                renamed+=int(key!=expected); migrated+=1
                meta['storage_migration']=dict(original_fingerprint=previous['fingerprint'],
                    original_key=key,reason='RKHS plane loop shadowed record key; predictions, parameters and fit evidence unchanged')
                meta['fingerprint']=fingerprint; repaired[expected]=record
            write_table(path,repaired)
        plan['storage_repair']=dict(records=migrated,renamed=renamed,original_fingerprint=previous['fingerprint'])
        write_json(Path(args.output)/'reproduction_probe_plan.json',plan)
        print(f'Repaired {renamed} row keys; preserved {migrated} fixed probe records without refits')
        return
    write_json(Path(args.output)/'reproduction_probe_plan.json',plan)
    if args.plan_only:
        print('Fixed reproduction probe plan saved; no fits or test evaluations performed')
        return
    jobs=[(d,args.output,fingerprint) for d in plan['datasets']]
    expected=sum(len(v['seeds']) for d in plan['datasets'] for v in VARIANTS.values()
                 if d in v.get('datasets',plan['datasets']))
    start=time.perf_counter(); rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_dataset,j) for j in jobs]
        for future in as_completed(futures):
            rows.extend(future.result())
            print(f"PROBES {len(rows)}/{expected}",flush=True)
            write_json(Path(args.output)/'reproduction_probe_execution.json',
                       dict(expected=expected,completed=len(rows),wall_seconds=time.perf_counter()-start,outcomes=rows))
            write_json(Path(args.output)/'reproduction_probes'/'index.json',
                       dict(fingerprint=fingerprint,groups=[dict(dataset=d,file=f'{d}.parquet')
                            for d in plan['datasets'] if (Path(args.output)/'reproduction_probes'/f'{d}.parquet').exists()]))


if __name__=='__main__': main()
