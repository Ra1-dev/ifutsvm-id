"""Independently audit persisted split, preprocessing, selection and metric evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
    balanced_accuracy_score,roc_auc_score,confusion_matrix,roc_curve)
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from ifutsvm.data import DATASETS,load
from experiments.protocol import split_plan
from experiments.registry import candidates
from experiments.run_all import write_json,source_hash
from experiments.storage import table_path
from experiments.evidence import read_table,table_exists

METRICS=('accuracy','precision','recall','specificity','f1','roc_auc','gmean','balanced_accuracy','training_seconds')


def audit(root,partial=False):
    root=Path(root)
    config=json.loads((root/'experiment_config.json').read_text())
    assert config['source_hash']==source_hash(),'source hash changed'
    manifest=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets']
    counts=dict(selections=0,inner_records=0,failed_inner_fits=0,outer_records=0,
                reproduction_records=0,published_records=0,scaler_checks=0,metric_checks=0,typed_table_checks=0)
    missing=[]; preprocessing={}; checked_records=set()
    for dataset in config['datasets']:
        X,y=load(dataset)
        schema=manifest[dataset]['attributes']
        numeric=[i for i,a in enumerate(schema) if not a['nominal']]
        nominal=[i for i,a in enumerate(schema) if a['nominal']]
        for protocol in ('holdout','nested_cv'):
            groups={m:read_table(table_path(root,dataset,m,protocol)) if table_exists(table_path(root,dataset,m,protocol)) else {} for m in config['models']}
            for si,(split_id,seed,train,test) in enumerate(split_plan(y,protocol)):
                proposals=candidates(DATASETS.index(dataset),si,protocol,config['budget'])
                folds=list(StratifiedKFold(config['inner_folds'],shuffle=True,random_state=seed).split(X[train],y[train]))

                def check_record(path,fit_indices,eval_indices,outer=False):
                    r=group_records[path]
                    np.testing.assert_array_equal(r['train_indices'],fit_indices)
                    np.testing.assert_array_equal(r['test_indices'],eval_indices)
                    assert not set(fit_indices)&set(eval_indices)
                    m=r['metadata']
                    np.testing.assert_array_equal(m['outer_train_indices'],train)
                    np.testing.assert_array_equal(m['outer_test_indices'],test)
                    assert m['source_hash']==config['source_hash']
                    assert m['dataset_sha256']==manifest[dataset]['sha256']
                    assert m['seed']==seed
                    if not outer:
                        assert set(fit_indices)|set(eval_indices)==set(train)
                        assert not (set(fit_indices)|set(eval_indices))&set(test)
                    if 'error' in r:
                        return r
                    np.testing.assert_array_equal(r['y'],y[eval_indices])
                    labs,freq=np.unique(y[fit_indices],return_counts=True)
                    assert r['positive_label']==labs[np.argmin(freq)]
                    key=(dataset,tuple(fit_indices))
                    if key not in preprocessing:
                        parts=[]; cats=None
                        if numeric:
                            parts.append(X[fit_indices][:,numeric].astype(float))
                        if nominal:
                            enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
                            parts.append(enc.fit_transform(X[fit_indices][:,nominal]))
                            cats=[v.tolist() for v in enc.categories_]
                        matrix=np.column_stack(parts)
                        scaler=StandardScaler().fit(matrix)
                        preprocessing[key]=(scaler.mean_,scaler.var_,cats)
                    mean,var,cats=preprocessing[key]
                    if m['model']=='faithful_raw':
                        assert m['scaler'] is None
                    else:
                        assert m['scaler']['n']==len(fit_indices)
                        np.testing.assert_allclose(m['scaler']['mean'],mean,rtol=1e-12,atol=1e-12)
                        np.testing.assert_allclose(m['scaler']['var'],var,rtol=1e-12,atol=1e-12)
                        counts['scaler_checks']+=1
                    if cats is not None:
                        assert m['categories']==cats
                    truth=np.array(r['y'])==r['positive_label']
                    prediction=np.array(r['prediction'])==r['positive_label']
                    scores=np.array(r['scores'])
                    # Independent count/rank derivation avoids invoking eight
                    # sklearn validation pipelines for every inner candidate.
                    # Tied score ranks implement the Mann-Whitney AUC exactly.
                    cm=np.bincount(2*truth.astype(int)+prediction.astype(int),minlength=4).reshape(2,2)
                    tn,fp,fn,tp=cm.ravel(); positive=tp+fn; negative=tn+fp
                    sensitivity=tp/positive if positive else 0.
                    specificity=tn/negative if negative else 0.
                    auc=(rankdata(scores)[truth].sum()-positive*(positive+1)/2)/(positive*negative)
                    expected=dict(accuracy=(tp+tn)/len(truth),
                        precision=tp/(tp+fp) if tp+fp else 0.,
                        recall=sensitivity,specificity=specificity,
                        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                        balanced_accuracy=(sensitivity+specificity)/2,
                        roc_auc=auc,gmean=np.sqrt(sensitivity*specificity))
                    for name,value in expected.items():
                        np.testing.assert_allclose(r[name],value,rtol=1e-12,atol=1e-12)
                    assert np.isfinite(r['training_seconds']) and r['training_seconds']>=0
                    np.testing.assert_array_equal(r['confusion_matrix'],cm)
                    fpr,tpr,thresholds=roc_curve(truth,scores)
                    np.testing.assert_array_equal(r['roc']['fpr'],fpr)
                    np.testing.assert_array_equal(r['roc']['tpr'],tpr)
                    assert r['roc']['thresholds']==[float(v) if np.isfinite(v) else None for v in thresholds]
                    counts['metric_checks']+=1
                    if outer:
                        counts['typed_table_checks']+=1
                    checked_records.add(str(path))
                    return r

                for model in config['models']:
                    group_records=groups[model]
                    selection_path=f'{split_id}/selection'
                    if selection_path not in group_records:
                        missing.append(f'{dataset}/{model}/{protocol}/{selection_path}'); continue
                    selection=group_records[selection_path]
                    assert len(selection['candidates'])==config['budget']
                    assert selection['inner_folds']==config['inner_folds']
                    assert selection['candidate_seed']==[20261110,DATASETS.index(dataset),si,int(protocol=='nested_cv')]
                    summaries=[]
                    for ci,p in enumerate(proposals):
                        records=[]
                        for fold,(it,iv) in enumerate(folds):
                            path=f'{split_id}/inner/candidate-{ci}-fold-{fold}'
                            r=check_record(path,train[it],train[iv])
                            assert r['metadata']['parameters']==p
                            assert r['metadata']['candidate']==ci and r['metadata']['fold']==fold
                            records.append(r); counts['inner_records']+=1
                            counts['failed_inner_fits']+=int('error' in r)
                        valid=all('error' not in r for r in records)
                        s=selection['candidates'][ci]
                        assert s['valid']==valid and s['parameters']==p and s['candidate']==ci
                        for metric in ('balanced_accuracy','accuracy'):
                            if valid:
                                np.testing.assert_allclose(s[metric],np.mean([r[metric] for r in records]),rtol=1e-13)
                            else:
                                assert s[metric] is None
                        summaries.append(s)
                    counts['selections']+=1
                    targets=[('evaluation','balanced_accuracy','outer_records')]
                    if model=='paper' and protocol=='holdout':
                        targets.append(('reproduction','accuracy','reproduction_records'))
                    for label,metric,counter in targets:
                        path=f'{split_id}/{label}'
                        if path not in group_records:
                            missing.append(f'{dataset}/{model}/{protocol}/{path}'); continue
                        r=check_record(path,train,test,outer=True)
                        assert 'error' not in r
                        assert r['candidate_budget']==config['budget'] and r['selection_metric']==metric
                        ranking=sorted((s for s in summaries if s['valid']),key=lambda s:(-s[metric],s['candidate']))
                        for s in ranking:
                            candidate_file=f"{split_id}/outer_candidates/candidate-{s['candidate']}"
                            fitted=group_records[candidate_file]
                            if 'error' not in fitted:
                                assert r['selected_candidate']==s['candidate']
                                assert r['prediction']==fitted['prediction'] and r['scores']==fitted['scores']
                                break
                            assert fitted['error_stage']=='fit'
                        else:
                            raise AssertionError('selected candidate absent')
                        counts[counter]+=1
                    if model=='paper' and protocol=='holdout' and seed==0:
                        path=f'{split_id}/published_config'
                        if path not in group_records:
                            missing.append(f'{dataset}/{model}/{protocol}/{path}'); continue
                        r=check_record(path,train,test,outer=True)
                        assert 'error' not in r
                        table=json.loads(Path('data/published_table_sii.json').read_text())
                        p={k:table['datasets'][dataset][k] for k in ('C','Cu','eps','mu')}
                        p['C3']=table['unreported_C3_default']
                        assert r['metadata']['parameters']==p
                        counts['published_records']+=1
    result=dict(**counts,expected_main=len(config['datasets'])*10*len(config['models']),
                missing=missing,complete=not missing,source_hash=config['source_hash'],
                unique_training_preprocessing_sets=len(preprocessing),
                statement='All persisted completed selections checked against original indices, '
                          'raw-data training-only preprocessing, independent count/rank metrics and sklearn ROC points.')
    write_json(root/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='missing'},indent=2))
    if missing and not partial:
        raise AssertionError(f'{len(missing)} required outputs missing')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results')
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    audit(args.output,args.partial)
