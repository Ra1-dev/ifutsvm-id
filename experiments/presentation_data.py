"""Assemble the predeclared presentation illustrations from persisted runs."""
import numpy as np
from experiments.evidence import read_table
from experiments.storage import table_path


def resolve(document, pointer):
    """Read an RFC 6901 JSON pointer used by a figure or table."""
    value=document
    for key in pointer.strip('/').split('/'):
        key=key.replace('~1','/').replace('~0','~')
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value


def illustrations(summary,root):
    plan=summary['artifact_plan'];config=summary['config']
    assert plan['main_source_hash']==config['source_hash']
    metadata=summary['data_manifest']['datasets']
    order=sorted(config['datasets'],key=lambda n:(metadata[n]['imbalance_ratio'],n))
    assert plan['representative_datasets']==[order[0],order[(len(order)-1)//2],order[-1]]
    baselines=[m for m in config['models'] if m not in ('paper','faithful_raw','faithful_scaled')]
    best=max(baselines,key=lambda m:summary['protocols']['holdout']['aggregate']['models'][m]['balanced_accuracy']['mean'])
    datasets={}
    for dataset in plan['representative_datasets']:
        models={}
        for model in config['models']:
            path=table_path(root,dataset,model,'holdout');table=read_table(path)
            records=[table[f'holdout-{seed}/evaluation'] for seed in range(5)]
            assert all('error' not in r for r in records)
            first=records[0]
            models[model]=dict(roc=first['roc'],roc_auc=first['roc_auc'],scores=first['scores'],
                y=first['y'],prediction=first['prediction'],positive_label=first['positive_label'],
                test_indices=first['test_indices'],
                summed_confusion_matrix=np.sum([r['confusion_matrix'] for r in records],axis=0).tolist(),
                sources=[path.relative_to(root).as_posix()+f'#holdout-{seed}/evaluation' for seed in range(5)])
        datasets[dataset]=dict(models=models,imbalance_ratio=metadata[dataset]['imbalance_ratio'])
    return dict(datasets=datasets,representative_rule=plan['representative_rule'],roc_rule=plan['roc'],
        confusion_rule=plan['confusion'],confusion_models=['paper',best],confusion_seeds=list(range(5)),
        roc_seed=0,best_baseline=best,best_baseline_metric='equal-weight holdout balanced accuracy over all twelve datasets')
