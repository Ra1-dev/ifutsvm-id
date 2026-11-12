"""Independently rederive training-only curve indices, scalers and metrics."""
import json
import argparse
import hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ifutsvm.data import load
from experiments.protocol import split_plan
from experiments.run_all import digest, source_hash, write_json


def audit(root,course=False):
    root = Path(root)
    result = json.loads((root/('course_curves.json' if course else 'curves.json')).read_text())
    plan = json.loads((root/('course_curve_plan.json' if course else 'curve_plan.json')).read_text())
    assert result['complete'] and result['plan'] == plan
    assert result['fingerprint'] == digest(plan) and plan['source_hash'] == source_hash()
    config=plan['learning'] if course else plan
    if course:
        assert plan['script_sha256']==hashlib.sha256(Path('experiments/course_curves.py').read_bytes()).hexdigest()
        assert plan['artifact_plan_sha256']==hashlib.sha256((root/'artifact_plan.json').read_bytes()).hexdigest()
        from experiments.evidence import read_table
        from experiments.storage import table_path
        for name in config['models']:
            original=read_table(table_path(root,config['dataset'],name,'holdout'))['holdout-0/evaluation']
            assert config['parameters'][name]==original['metadata']['parameters']
        assert plan['validation']['parameters']==config['parameters']['paper']
    X, y = load(config['dataset'])
    schema = json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets'][config['dataset']]['attributes']
    numeric = [i for i, a in enumerate(schema) if not a['nominal']]
    nominal = [i for i, a in enumerate(schema) if a['nominal']]
    split_id, seed, train, test = next(iter(split_plan(y, 'holdout')))
    assert split_id == config['outer_split'] and seed == config.get('seed',0)
    folds = list(StratifiedKFold(config['inner_folds'], shuffle=True, random_state=seed).split(X[train], y[train]))
    if course:
        jobs=[(name,'learning',value,config['parameters'][name]) for name in config['models'] for value in config['fractions']]
        validation_plan=plan['validation']
        jobs += [(validation_plan['model'],name,value,{**validation_plan['parameters'],name:value})
                 for name,values in validation_plan['sweeps'].items() for value in values]
    else:
        jobs = [(None,'learning', v, plan['parameters']) for v in plan['learning_fractions']]
        jobs += [(None,name, v, {**plan['parameters'], name: v}) for name, values in plan['validation'].items() for v in values]
    seen = set(); metric_checks = 0
    for model,family, value, parameters in jobs:
        records = []
        for fold, (it, iv) in enumerate(folds):
            fit = train[it]; validation = train[iv]
            if family == 'learning':
                rng = np.random.default_rng(np.random.SeedSequence([config.get('subset_seed',20261116), seed, fold]))
                pieces = []
                for label in np.unique(y[fit]):
                    indices = rng.permutation(fit[y[fit] == label])
                    pieces.extend(indices[:max(1, int(np.floor(value*len(indices))))])
                fit = np.array(sorted(pieces))
            key = (model+'/' if course else '')+f'{family}/{value:g}/fold-{fold}'; seen.add(key)
            record = result['records'][key]; records.append(record)
            assert (record['family'], record['value'], record['fold'], record['seed']) == (family, value, fold, seed)
            assert record['parameters'] == parameters
            if course:assert record['model']==model
            np.testing.assert_array_equal(record['outer_train_indices'], train)
            np.testing.assert_array_equal(record['outer_test_indices'], test)
            assert not np.intersect1d(fit, validation).size
            assert not np.intersect1d(np.r_[fit, validation], test).size
            blocks = [X[fit][:, numeric].astype(float)] if numeric else []
            if nominal:
                blocks.append(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(X[fit][:, nominal]))
            scaler = StandardScaler().fit(np.column_stack(blocks))
            assert record['scaler']['n'] == len(fit)
            np.testing.assert_allclose(record['scaler']['mean'], scaler.mean_, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(record['scaler']['var'], scaler.var_, rtol=1e-12, atol=1e-12)
            labels, counts = np.unique(y[fit], return_counts=True)
            positive = labels[np.argmin(counts)]; assert record['positive_label'] == positive
            for part, indices in [('training', fit), ('validation', validation)]:
                observation = record[part]; measured = observation['metrics']
                np.testing.assert_array_equal(observation['indices'], indices)
                np.testing.assert_array_equal(observation['y'], y[indices])
                truth = y[indices] == positive
                prediction = np.array(observation['prediction']) == positive
                scores = np.array(observation['scores'])
                cm = confusion_matrix(truth, prediction, labels=[False, True])
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp/(tp+fn); specificity = tn/(tn+fp)
                expected = dict(accuracy=accuracy_score(truth, prediction),
                    precision=precision_score(truth, prediction, zero_division=0),
                    recall=recall_score(truth, prediction, zero_division=0),
                    specificity=specificity, f1=f1_score(truth, prediction, zero_division=0),
                    balanced_accuracy=balanced_accuracy_score(truth, prediction),
                    roc_auc=roc_auc_score(truth, scores), gmean=np.sqrt(sensitivity*specificity))
                for name, number in expected.items():
                    np.testing.assert_allclose(measured[name], number, rtol=1e-12, atol=1e-12)
                assert np.isfinite(measured['training_seconds']) and measured['training_seconds'] >= 0
                np.testing.assert_array_equal(measured['confusion_matrix'], cm)
                fpr, tpr, thresholds = roc_curve(truth, scores)
                np.testing.assert_array_equal(measured['roc']['fpr'], fpr)
                np.testing.assert_array_equal(measured['roc']['tpr'], tpr)
                assert measured['roc']['thresholds'] == [float(v) if np.isfinite(v) else None for v in thresholds]
                metric_checks += 1
        groups=result['groups'][model] if course else result['groups']
        aggregate = next(r for r in groups[family] if r['value'] == value)
        np.testing.assert_allclose(aggregate['training_rows_mean'], np.mean([len(r['training']['indices']) for r in records]))
        for metric in plan['metrics']:
            for part in ['training', 'validation']:
                values = [r[part]['metrics'][metric] for r in records]
                np.testing.assert_allclose(aggregate['metrics'][metric][part]['mean'], np.mean(values), rtol=1e-12)
                np.testing.assert_allclose(aggregate['metrics'][metric][part]['std'], np.std(values, ddof=1), rtol=1e-12, atol=1e-12)
    assert set(result['records']) == seen and result['fit_count'] == len(seen)
    audit_result = dict(complete=True, fits=len(seen), scaler_checks=len(seen), metric_records=metric_checks,
        outer_test_rows_used=0, parameters_selected_from_curves=False, fingerprint=result['fingerprint'])
    write_json(root/('course_curve_audit.json' if course else 'curve_audit.json'), audit_result)
    print(json.dumps(audit_result))
    return audit_result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--course',action='store_true');args=parser.parse_args()
    audit('results',course=args.course)
