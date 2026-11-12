"""Independent reconstruction of research labels, tuning, scaling and metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from ifutsvm.data import DATASETS,load
from experiments.protocol import split_plan
from experiments.registry import candidates
from experiments.research_run import fingerprint
from experiments.storage import table_path
from experiments.evidence import read_table,table_exists
from experiments.run_all import write_json


def independent_metrics(record):
    truth=np.asarray(record['y'])==record['positive_label']
    pred=np.asarray(record['prediction'])==record['positive_label']
    scores=np.asarray(record['scores'])
    assert truth.shape==pred.shape==scores.shape and np.isfinite(scores).all()
    cm=np.bincount(2*truth.astype(int)+pred.astype(int),minlength=4).reshape(2,2)
    tn,fp,fn,tp=cm.ravel(); positive=tp+fn; negative=tn+fp
    sensitivity=tp/positive if positive else 0.; specificity=tn/negative if negative else 0.
    expected=dict(accuracy=(tp+tn)/len(truth),precision=tp/(tp+fp) if tp+fp else 0.,
        recall=sensitivity,specificity=specificity,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        balanced_accuracy=(sensitivity+specificity)/2,gmean=np.sqrt(sensitivity*specificity),
        roc_auc=(rankdata(scores)[truth].sum()-positive*(positive+1)/2)/(positive*negative))
    for k,v in expected.items(): np.testing.assert_allclose(record[k],v,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(record['confusion_matrix'],cm)
    fpr,tpr,thresholds=roc_curve(truth,scores)
    np.testing.assert_array_equal(record['roc']['fpr'],fpr)
    np.testing.assert_array_equal(record['roc']['tpr'],tpr)
    assert record['roc']['thresholds']==[float(v) if np.isfinite(v) else None for v in thresholds]
    assert record['training_seconds']>=0 and np.isfinite(record['training_seconds'])


def audit(root,wave,partial=False):
    root=Path(root); plan=json.loads((root/'ablation_plan.json').read_text())
    assert plan['fingerprint']==fingerprint(),'research implementation changed'
    manifest=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets']
    datasets=plan['datasets'] if wave=='ablation' else plan['noise_datasets']
    names=list(plan['variants']) if wave=='ablation' else plan['noise_models']
    levels=[0.] if wave=='ablation' else plan['noise_levels']
    count=dict(outer_records=0,inner_records=0,failed_inner_fits=0,scaler_checks=0,metric_checks=0,main_paper_matches=0)
    missing=[]; failed=[]; preprocessing={}
    for dataset in datasets:
        X,y=load(dataset); schema=manifest[dataset]['attributes']
        path=table_path(root,dataset,'research',wave)
        records=read_table(path) if path.exists() else {}
        main_path=table_path(root.parent,dataset,'paper','holdout')
        main=read_table(main_path) if table_exists(main_path) else {}
        for si,(split_id,seed,train,test) in enumerate(split_plan(y,'holdout')):
            proposals=candidates(DATASETS.index(dataset),si,'holdout',plan['budget'])
            for rate in levels:
                rng=np.random.default_rng(np.random.SeedSequence([20261115,DATASETS.index(dataset),seed]))
                flips=rng.permutation(train)[:int(np.floor(rate*len(train)))]
                noisy=y.copy(); labels=np.unique(y[train]); noisy[flips]=np.where(y[flips]==labels[0],labels[1],labels[0])
                assert not set(flips)&set(test)
                folds=list(StratifiedKFold(plan['inner_folds'],shuffle=True,random_state=seed).split(X[train],noisy[train]))

                def check(key,fit_indices,eval_indices,outer=False):
                    r=records[key]; m=r['metadata']
                    np.testing.assert_array_equal(r['train_indices'],fit_indices)
                    np.testing.assert_array_equal(r['test_indices'],eval_indices)
                    np.testing.assert_array_equal(m['outer_train_indices'],train)
                    np.testing.assert_array_equal(m['outer_test_indices'],test)
                    np.testing.assert_array_equal(m['flipped_training_indices'],flips)
                    assert m['noise_rate']==rate and m['seed']==seed
                    assert m['source_hash']==plan['fingerprint'] and m['dataset_sha256']==manifest[dataset]['sha256']
                    assert m['dataset']==dataset and m['wave']==wave and m['split']==split_id
                    assert not set(fit_indices)&set(eval_indices)
                    if not outer:
                        assert set(fit_indices)|set(eval_indices)==set(train)
                        assert not (set(fit_indices)|set(eval_indices))&set(test)
                    if 'error' in r: return r
                    np.testing.assert_array_equal(r['y'],(y if outer else noisy)[eval_indices])
                    labels_,counts=np.unique(noisy[fit_indices],return_counts=True)
                    assert r['positive_label']==labels_[np.argmin(counts)]
                    cache=(dataset,tuple(fit_indices))
                    if cache not in preprocessing:
                        numeric=[i for i,a in enumerate(schema) if not a['nominal']]
                        nominal=[i for i,a in enumerate(schema) if a['nominal']]
                        parts=[X[fit_indices][:,numeric].astype(float)] if numeric else []
                        categories=None
                        if nominal:
                            encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
                            parts.append(encoder.fit_transform(X[fit_indices][:,nominal]))
                            categories=[a.tolist() for a in encoder.categories_]
                        scale=StandardScaler().fit(np.column_stack(parts))
                        preprocessing[cache]=(scale.mean_,scale.var_,categories)
                    mean,var,categories=preprocessing[cache]
                    np.testing.assert_allclose(m['scaler']['mean'],mean,rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(m['scaler']['var'],var,rtol=1e-12,atol=1e-12)
                    assert m['scaler']['n']==len(fit_indices)
                    if categories is not None: assert m['categories']==categories
                    independent_metrics(r); count['metric_checks']+=1; count['scaler_checks']+=1
                    return r

                for name in names:
                    prefix=f'{name}/{split_id}/noise-{rate:g}'; selection=records.get(prefix+'/selection')
                    if selection is None:
                        missing.append(f'{dataset}/{prefix}/selection'); continue
                    assert selection['inner_folds']==plan['inner_folds']
                    assert selection['candidate_seed']==[20261110,DATASETS.index(dataset),si,0]
                    assert len(selection['candidates'])==len(proposals)
                    summary=[]
                    for ci,p in enumerate(proposals):
                        inner=[]
                        for fold,(it,iv) in enumerate(folds):
                            r=check(f'{prefix}/inner/candidate-{ci}-fold-{fold}',train[it],train[iv])
                            assert r['metadata']['parameters']==p and r['metadata']['variant']==name
                            assert r['metadata']['candidate']==ci and r['metadata']['fold']==fold
                            inner.append(r); count['inner_records']+=1; count['failed_inner_fits']+=int('error' in r)
                        valid=all('error' not in r for r in inner)
                        saved=selection['candidates'][ci]
                        assert saved['candidate']==ci and saved['parameters']==p and saved['valid']==valid
                        if valid:
                            np.testing.assert_allclose(saved['balanced_accuracy'],np.mean([r['balanced_accuracy'] for r in inner]),atol=1e-14)
                            summary.append(saved)
                        else: assert saved['balanced_accuracy'] is None
                    key=prefix+'/evaluation'
                    if key not in records:
                        missing.append(f'{dataset}/{key}'); continue
                    r=check(key,train,test,outer=True)
                    if 'error' in r:
                        failed.append(f'{dataset}/{key}'); continue
                    assert r['candidate_budget']==plan['budget'] and r['selection_metric']=='balanced_accuracy'
                    assert r['metadata']['variant']==name
                    assert r['metadata']['parameters']==proposals[r['selected_candidate']]
                    ranking=sorted(summary,key=lambda row:(-row['balanced_accuracy'],row['candidate']))
                    for candidate in ranking:
                        prior=records[f"{prefix}/outer_candidates/candidate-{candidate['candidate']}"]
                        if 'error' in prior:
                            assert prior['error_stage']=='fit'; continue
                        assert r['selected_candidate']==candidate['candidate']
                        assert r['prediction']==prior['prediction'] and r['scores']==prior['scores']
                        break
                    else: raise AssertionError('no valid training-selected result')
                    count['outer_records']+=1
                    if name=='random_average' and rate==0 and split_id+'/evaluation' in main:
                        original=main[split_id+'/evaluation']
                        assert r['selected_candidate']==original['selected_candidate']
                        np.testing.assert_array_equal(r['prediction'],original['prediction'])
                        np.testing.assert_allclose(r['scores'],original['scores'],rtol=1e-10,atol=1e-10)
                        count['main_paper_matches']+=1
    result=dict(**count,wave=wave,expected=plan[f'expected_{wave}_outer'],missing=missing,failed=failed,
                complete=not missing and not failed,fingerprint=plan['fingerprint'],
                statement='All completed selections reconstructed from prescribed candidates, noisy training labels, clean outer labels, raw training preprocessing and independent count/rank metrics.')
    write_json(root/f'{wave}_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('missing','failed')},indent=2))
    if not partial and not result['complete']: raise AssertionError('research evidence incomplete')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results/research')
    parser.add_argument('--wave',choices=['ablation','noise'],default='ablation')
    parser.add_argument('--partial',action='store_true'); args=parser.parse_args()
    audit(args.output,args.wave,args.partial)
