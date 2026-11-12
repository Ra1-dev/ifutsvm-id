"""Generate comparison tables and one canonical numerical source of truth."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from experiments.audit import METRICS
from experiments.run_all import write_json
from experiments.storage import table_path
from experiments.evidence import read_table,table_exists
from experiments.findings import headline,write_headline
from experiments.display import MODEL_NAMES


def summarize(root,partial=False):
    root=Path(root)
    config=json.loads((root/'experiment_config.json').read_bytes())
    result=dict(schema_version=1,config=config,complete=True,protocols={})
    missing=[]
    for protocol in ('holdout','nested_cv'):
        datasets={}; ranks={m:{k:[] for k in METRICS} for m in config['models']}
        for dataset in config['datasets']:
            models={}
            for model in config['models']:
                path=table_path(root,dataset,model,protocol)
                group=read_table(path) if table_exists(path) else {}
                keys=sorted(k for k in group if k.endswith('/evaluation'))
                if len(keys)!=5:
                    missing.append(dict(dataset=dataset,protocol=protocol,model=model,found=len(keys)))
                    continue
                records=[group[k] for k in keys]
                models[model]=dict(runs=[path.relative_to(root).as_posix()+'#'+k for k in keys],n=len(records),
                    metrics={k:dict(values=[r[k] for r in records],mean=float(np.mean([r[k] for r in records])),
                                   std=float(np.std([r[k] for r in records],ddof=1))) for k in METRICS})
            if len(models)==len(config['models']):
                for metric in METRICS:
                    means=[models[m]['metrics'][metric]['mean'] for m in config['models']]
                    order=rankdata(np.array(means)*(1 if metric=='training_seconds' else -1),method='average')
                    best=min(means) if metric=='training_seconds' else max(means)
                    for m,rank in zip(config['models'],order):
                        entry=models[m]['metrics'][metric]
                        entry['rank']=float(rank); entry['best']=entry['mean']==best
                        ranks[m][metric].append(float(rank))
            datasets[dataset]=models
        result['protocols'][protocol]=dict(datasets=datasets,average_ranks={
            m:{k:dict(mean=float(np.mean(v)) if v else None,datasets=len(v)) for k,v in values.items()}
            for m,values in ranks.items()})
        common=[d for d,group in datasets.items() if all(m in group for m in config['models'])]
        result['protocols'][protocol]['aggregate']=dict(datasets=common,
            definition='Equal-weight mean of dataset means; SD describes variation between dataset means, not uncertainty of the grand mean.',
            models={m:{metric:dict(mean=float(np.mean([datasets[d][m]['metrics'][metric]['mean'] for d in common])),
                std=float(np.std([datasets[d][m]['metrics'][metric]['mean'] for d in common],ddof=1)) if len(common)>1 else None,
                median_outer_fit_seconds=float(np.median([value for d in common for value in datasets[d][m]['metrics']['training_seconds']['values']]))
                    if metric=='training_seconds' else None)
                for metric in METRICS} for m in config['models']} if common else {})
    result['complete']=not missing; result['missing']=missing
    result['headline']=headline(result)
    result['model_display_names']=MODEL_NAMES
    result['data_manifest']=json.loads(Path('data/keel/MANIFEST.json').read_bytes())
    for key,filename in [('environment','environment.json'),('reproduction','reproduction.json'),
                         ('literature','literature.json')]:
        if (root/filename).exists():result[key]=json.loads((root/filename).read_bytes())
    # A deterministic illustration is selected by dataset order and split,
    # never by observed performance. Figures consume only this summary JSON.
    example_dataset=config['datasets'][0]
    illustrations={}
    for model in config['models']:
        path=table_path(root,example_dataset,model,'holdout')
        record=read_table(path).get('holdout-0/evaluation') if table_exists(path) else None
        if record and 'error' not in record:
            illustrations[model]={k:record[k] for k in ('confusion_matrix','roc','roc_auc','positive_label','test_indices')}
            illustrations[model].update(parameters=record['metadata']['parameters'],seed=record['metadata']['seed'],
                prediction=record['prediction'],scores=record['scores'])
            illustrations[model]['source']=path.relative_to(root).as_posix()+'#holdout-0/evaluation'
    result['illustration']=dict(dataset=example_dataset,split='holdout-0',models=illustrations,
        selection='First dataset in the pre-specified list, seed zero; deterministic illustration rule, not a best-run selection.',
        confusion_order=['majority (negative)','minority (positive)'])
    curves=root/'curves.json'
    if curves.exists(): result['curves']=json.loads(curves.read_bytes())
    course_curves=root/'course_curves.json'
    if course_curves.exists():
        # Raw observations remain independently auditable; plots need only
        # the source-bound plan and aggregate values in the canonical summary.
        record=json.loads(course_curves.read_bytes())
        result['course_curves']={k:v for k,v in record.items() if k!='records'}
    artifact_plan=root/'artifact_plan.json'
    if artifact_plan.exists():result['artifact_plan']=json.loads(artifact_plan.read_bytes())
    if result['complete'] and 'artifact_plan' in result:
        from experiments.presentation_data import illustrations
        result['presentation_illustrations']=illustrations(result,root)
    main_stats=root/'research/main_statistics.json'
    if main_stats.exists():result['main_statistics']=json.loads(main_stats.read_bytes())
    execution=root/'execution.json'
    result['execution']=json.loads(execution.read_bytes()) if execution.exists() else None
    if missing and not partial:
        raise RuntimeError(f'{len(missing)} dataset/protocol/model groups are incomplete')
    if partial:
        write_json(root/'progress_summary.json',result)
    else:
        research=root/'research/summary.json'
        if research.exists():
            result['research']=json.loads(research.read_bytes())
        write_json(root/'summary.json',result)
        if root.resolve()==Path('results').resolve() and result['headline']['complete']:
            write_headline(result)
        tables=root/'tables'; tables.mkdir(exist_ok=True)
        for protocol,suffix in [('holdout','holdout'),('nested_cv','cv')]:
            columns=['dataset','model','n']+[f'{k}_{s}' for k in METRICS
                for s in ('mean','std','rank','best','display','average_rank')]
            with (tables/f'main_{suffix}.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n'); writer.writeheader()
                for dataset,models in result['protocols'][protocol]['datasets'].items():
                    for model,data in models.items():
                        row=dict(dataset=dataset,model=model,n=data['n'])
                        for metric,value in data['metrics'].items():
                            for stat in ('mean','std','rank','best'):
                                row[f'{metric}_{stat}']=value[stat]
                            scale=1 if metric=='training_seconds' else 100
                            display=f"{scale*value['mean']:.2f} ± {scale*value['std']:.2f}"
                            row[f'{metric}_display']=f'**{display}**' if value['best'] else display
                            row[f'{metric}_average_rank']=result['protocols'][protocol]['average_ranks'][model][metric]['mean']
                        writer.writerow(row)
    completed=sum(len(models) for p in result['protocols'].values() for models in p['datasets'].values())
    print(f'Summarized complete five-run groups: {completed}; missing groups: {len(missing)}')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results')
    parser.add_argument('--partial',action='store_true'); args=parser.parse_args()
    summarize(args.output,args.partial)
