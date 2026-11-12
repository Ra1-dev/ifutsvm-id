"""Train-only provenance and independently recomputed fixed-probe metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,confusion_matrix,f1_score,precision_score,recall_score
from ifutsvm.data import load,DATASETS
from experiments.reproduction_probes import VARIANTS
from experiments.run_all import write_json
from experiments.storage import read_table


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results')
    parser.add_argument('--partial',action='store_true'); args=parser.parse_args(); root=Path(args.output)
    manifest=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets']
    table=json.loads(Path('data/published_table_sii.json').read_text())
    checked=0; failures=[]; missing=[]
    for dataset in DATASETS:
        X,y=load(dataset); schema=manifest[dataset]['attributes']
        path=root/'reproduction_probes'/f'{dataset}.parquet'
        records=read_table(path) if path.exists() else {}
        for variant,setup in VARIANTS.items():
            if dataset not in setup.get('datasets',DATASETS): continue
            for seed in setup['seeds']:
                key=f'{variant}/seed-{seed}'
                if key not in records: missing.append(path.as_posix()+'#'+key); continue
                r=records[key]; m=r['metadata']
                train,test=train_test_split(np.arange(len(y)),test_size=.3,random_state=seed,
                    stratify=None if variant=='unstratified' else y)
                np.testing.assert_array_equal(r['train_indices'],train)
                np.testing.assert_array_equal(r['test_indices'],test)
                assert m['dataset_sha256']==manifest[dataset]['sha256']
                assert m['dataset']==dataset and m['variant']==variant and m['seed']==seed
                p={k:table['datasets'][dataset][k] for k in ('C','Cu','eps','mu')}; p['C3']=table['unreported_C3_default']
                assert m['parameters']==p
                if 'error' in r:
                    failures.append(dict(path=path.as_posix()+'#'+key,error=r['error'])); continue
                numeric=[i for i,a in enumerate(schema) if not a['nominal']]
                nominal=[i for i,a in enumerate(schema) if a['nominal']]
                parts=[X[train][:,numeric].astype(float)] if numeric else []
                if nominal and variant!='drop_nominal':
                    parts.append(OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit_transform(X[train][:,nominal]))
                if variant=='raw_features':
                    assert m['scaler'] is None
                else:
                    scale=StandardScaler().fit(np.column_stack(parts))
                    np.testing.assert_allclose(m['scaler']['mean'],scale.mean_,rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(m['scaler']['var'],scale.var_,rtol=1e-12,atol=1e-12)
                    assert m['scaler']['n']==len(train)
                np.testing.assert_array_equal(r['y'],y[test])
                labels,counts=np.unique(y[train],return_counts=True)
                assert r['positive_label']==labels[np.argmin(counts)]
                truth=np.asarray(r['y'])==r['positive_label']; pred=np.asarray(r['prediction'])==r['positive_label']
                recall=recall_score(truth,pred,zero_division=0); specificity=recall_score(~truth,~pred,zero_division=0)
                metrics=dict(accuracy=accuracy_score(truth,pred),balanced_accuracy=balanced_accuracy_score(truth,pred),
                    roc_auc=roc_auc_score(truth,r['scores']),f1=f1_score(truth,pred,zero_division=0),
                    precision=precision_score(truth,pred,zero_division=0),recall=recall,specificity=specificity,
                    gmean=np.sqrt(recall*specificity))
                for k,v in metrics.items(): np.testing.assert_allclose(r[k],v,rtol=1e-12,atol=1e-12)
                np.testing.assert_array_equal(r['confusion_matrix'],confusion_matrix(truth,pred,labels=[False,True]))
                checked+=1
    result=dict(checked=checked,failed_probes=failures,missing=missing,complete=not missing,
                statement='Every available probe checked against raw-data training preprocessing, fixed parameters, exact indices and independent metrics.')
    write_json(root/'reproduction_audit.json',result)
    print(f'Reproduction audit: {checked} checked; {len(failures)} failed fits; {len(missing)} missing probes')
    if missing and not args.partial: raise AssertionError('Missing fixed diagnostic probes')


if __name__=='__main__': main()
