"""Independent counts/ranks, splits, scalers and accuracy-selection audit."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve
from ifutsvm.data import load
from experiments.storage import read_table, table_path
from experiments.run_all import digest, source_hash, write_json

ROOT=Path('results/accuracy_ceiling')


def exact_accuracy_mean(records):
    """Average fold proportions as rationals, retaining exact accuracy ties."""
    return sum((Fraction(sum(a==b for a,b in zip(r['y'],r['prediction'])),len(r['y']))
                for r in records),Fraction())/len(records)


def check_record(r, X, y, train, test, params, seed, candidate, context):
    np.testing.assert_array_equal(r['train_indices'],train)
    np.testing.assert_array_equal(r['test_indices'],test)
    assert not set(train)&set(test)
    meta=r['metadata']
    for key,value in context.items(): assert meta[key]==value,(key,meta[key],value)
    assert meta['parameters']==params and meta['seed']==seed and meta['candidate']==candidate
    cache_meta={k:v for k,v in meta.items() if k not in ('cache_key','scaler')}
    assert meta['cache_key']==digest(dict(metadata=cache_meta,train=train.tolist(),test=test.tolist()))
    if 'error' in r:
        assert r['error_stage']=='fit','Evaluation errors cannot drive reselection'
        return False
    np.testing.assert_array_equal(r['y'],y[test])
    labels,counts=np.unique(y[train],return_counts=True)
    assert r['positive_label']==labels[np.argmin(counts)]
    matrix=X[train].astype(float)
    assert meta['scaler']['n']==len(train)
    np.testing.assert_allclose(meta['scaler']['mean'],matrix.mean(axis=0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(meta['scaler']['var'],matrix.var(axis=0),rtol=1e-12,atol=1e-12)
    truth=np.array(r['y'])==r['positive_label'];pred=np.array(r['prediction'])==r['positive_label']
    scores=np.asarray(r['scores']);assert np.isfinite(scores).all()
    cm=np.bincount(2*truth.astype(int)+pred.astype(int),minlength=4).reshape(2,2)
    tn,fp,fn,tp=cm.ravel();recall=tp/(tp+fn);specificity=tn/(tn+fp)
    positive=truth.sum();negative=len(truth)-positive
    expected=dict(accuracy=(tp+tn)/len(truth),precision=tp/(tp+fp) if tp+fp else 0.,recall=recall,
        specificity=specificity,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        balanced_accuracy=(recall+specificity)/2,gmean=np.sqrt(recall*specificity),
        roc_auc=(rankdata(scores)[truth].sum()-positive*(positive+1)/2)/(positive*negative))
    for key,value in expected.items(): np.testing.assert_allclose(r[key],value,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(r['confusion_matrix'],cm)
    fpr,tpr,threshold=roc_curve(truth,scores)
    np.testing.assert_array_equal(r['roc']['fpr'],fpr);np.testing.assert_array_equal(r['roc']['tpr'],tpr)
    assert r['roc']['thresholds']==[float(v) if np.isfinite(v) else None for v in threshold]
    assert np.isfinite(r['training_seconds']) and r['training_seconds']>=0
    return True


def audit():
    plan=json.loads((ROOT/'plan.json').read_text());summary=json.loads((ROOT/'summary.json').read_text())
    assert plan['source_hash']==source_hash()
    assert hashlib.sha256(Path('experiments/accuracy_ceiling.py').read_bytes()).hexdigest()==plan['script_sha256']
    assert summary['preregistration_sha256']==hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest()
    assert summary['fingerprint']==digest(plan)
    # Verify registration exists before the result history; no result is used to reconstruct proposals.
    from experiments.accuracy_ceiling import git
    revision=summary['preregistration_commit']
    assert git('show',f'{revision}:results/accuracy_ceiling/plan.json')==(ROOT/'plan.json').read_bytes()
    manifest=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets'][plan['dataset']]
    assert summary['dataset_sha256']==plan['dataset_sha256']==manifest['sha256']
    X,y=load(plan['dataset']);indices=np.arange(len(y))
    provenance={key:summary[key] for key in ('fingerprint','preregistration_commit','preregistration_sha256','dataset_sha256')}
    counts=dict(outer_records=0,inner_records=0,failed_inner_fits=0,metric_checks=0,scaler_checks=0,selections=0)
    family_means=[]
    for name in plan['models']:
        path=table_path(ROOT,plan['dataset'],name,'holdout');records=read_table(path)
        assert summary['models'][name]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        values=[];selected=[];expected_keys=set()
        for seed in plan['seeds']:
            split=f'holdout-{seed}'
            train,test=train_test_split(indices,test_size=.3,stratify=y,random_state=seed)
            context={**provenance,'split':split,'outer_train_indices':train.tolist(),'outer_test_indices':test.tolist(),'model':name}
            proposals=list(ParameterSampler(plan['spaces'][name],n_iter=plan['budget'],random_state=plan['candidate_seed_base']+seed))
            assert proposals==plan['candidates'][name][str(seed)]
            inner=list(StratifiedKFold(plan['inner_folds'],shuffle=True,random_state=seed).split(X[train],y[train]))
            key=f'{split}/selection';selection=records[key];expected_keys.add(key)
            assert selection['candidate_seed']==plan['candidate_seed_base']+seed
            assert selection['selection_metric']=='accuracy' and len(selection['candidates'])==plan['budget']
            candidate_values=[]
            for ci,params in enumerate(proposals):
                results=[]
                for fold,(it,iv) in enumerate(inner):
                    key=f'{split}/inner/candidate-{ci}-fold-{fold}';r=records[key];expected_keys.add(key)
                    assert not (set(train[it])|set(train[iv]))&set(test)
                    valid=check_record(r,X,y,train[it],train[iv],params,seed,ci,{**context,'role':'inner_validation','fold':fold})
                    counts['inner_records']+=1;counts['failed_inner_fits']+=not valid
                    counts['metric_checks']+=valid;counts['scaler_checks']+=valid;results.append(r)
                s=selection['candidates'][ci];valid=all('error' not in r for r in results)
                assert s['valid']==valid and s['parameters']==params and s['candidate']==ci
                assert s['errors']==[r['error'] for r in results if 'error' in r]
                if valid:
                    value=exact_accuracy_mean(results)
                    np.testing.assert_allclose(s['accuracy'],float(value),atol=1e-14,rtol=1e-14)
                    candidate_values.append((value,ci))
                else: assert s['accuracy'] is None
            best=max(candidate_values,key=lambda v:(v[0],-v[1]))[1]
            assert selection['selected_candidate']==best
            key=f'{split}/evaluation';r=records[key];expected_keys.add(key)
            assert check_record(r,X,y,train,test,proposals[best],seed,best,{**context,'role':'outer_evaluation'})
            values.append(100*r['accuracy']);selected.append(best)
            for key in ('outer_records','metric_checks','scaler_checks','selections'): counts[key]+=1
        assert set(records)==expected_keys,'Unregistered extra/missing records'
        row=summary['models'][name]
        for key,value in dict(accuracy_mean_percent=mean(values),accuracy_std_pp=stdev(values),best_split_accuracy_percent=max(values)).items():
            np.testing.assert_allclose(row[key],value,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(row['per_seed_accuracy_percent'],values,rtol=1e-13)
        assert row['selected_candidates']==selected;family_means.append(mean(values))
    assert counts['outer_records']==summary['outer_evaluations']==len(plan['models'])*len(plan['seeds'])
    assert counts['inner_records']==summary['inner_evaluations']==counts['outer_records']*plan['budget']*plan['inner_folds']
    suspect=any(value>=plan['criterion']['suspect_if_any_mean_accuracy_at_least']*100 for value in family_means)
    assert summary['implementation_suspect']==suspect and summary['complete']
    result=dict(complete=True,**counts,implementation_suspect=suspect,preregistration_commit=revision,
        summary_sha256=hashlib.sha256((ROOT/'summary.json').read_bytes()).hexdigest(),
        exact_rational_accuracy_selection=True,
        statement='All splits, training-only means/variances, eight count/rank metrics, ROC points, candidate draws and accuracy-selected winners independently checked. Accuracy ties reconstructed from exact rational correct-count proportions.')
    write_json(ROOT/'audit.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__': audit()
