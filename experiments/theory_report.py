"""Summarize and independently verify the synthetic theory experiments."""
from contextlib import nullcontext
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np


def audit_archive(root,empirical,plan,local_records=False):
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
    archive=Path('archive/theory-preregistered.zip')
    with (nullcontext(None) if local_records else zipfile.ZipFile(archive)) as bundle:
        if bundle is not None:assert bundle.testzip() is None
        for r in empirical['runs']:
            if bundle is not None:record=json.loads(bundle.read(r['file']))
            else:
                parts=Path(r['file']).parts
                relative=Path(*parts[parts.index('theory_runs'):])
                path=(root/relative).resolve()
                assert path.is_relative_to(root.resolve())
                record=json.loads(path.read_bytes())
            assert record['metadata']['fingerprint']==plan['fingerprint']
            actual=np.asarray(record['y'])==record['positive_label']
            prediction=np.asarray(record['prediction'])==record['positive_label']
            score=balanced_accuracy_score(actual,prediction)
            np.testing.assert_allclose(score,r['balanced_accuracy'],rtol=1e-12)
            np.testing.assert_allclose(roc_auc_score(actual,record['scores']),record['roc_auc'],rtol=1e-12)
            np.testing.assert_allclose(f1_score(actual,prediction,zero_division=0),record['f1'],rtol=1e-12)
            assert r['nearest_boundary_abs']==record['metadata']['nearest_boundary_abs']
    paired=[]; boundaries=[]
    for ratio in (9,30):
        for cu in plan['penalties']:
            for seed in plan['seeds']:
                get=lambda name:next(r for r in empirical['runs'] if
                    (r['ratio'],r['Cu'],r['seed'],r['strategy'])==(ratio,cu,seed,name))
                a,b=get('conditional_gaussian'),get('random_average')
                paired.append(a['balanced_accuracy']-b['balanced_accuracy'])
                if a['nearest_boundary_abs'] is not None and b['nearest_boundary_abs'] is not None:
                    boundaries.append(a['nearest_boundary_abs']-b['nearest_boundary_abs'])
    accuracy=float(np.mean(paired)); movement=float(np.mean(boundaries))
    np.testing.assert_allclose(accuracy,empirical['high_imbalance_mean_accuracy_delta'],atol=1e-15)
    np.testing.assert_allclose(movement,empirical['high_imbalance_mean_boundary_distance_delta'],atol=1e-15)
    result=dict(records_checked=len(empirical['runs']),high_imbalance_pairs=len(paired),
        balanced_accuracy_delta=accuracy,boundary_distance_delta=movement,supported=accuracy>0 and movement<0,
        archive_sha256=None if local_records else hashlib.sha256(archive.read_bytes()).hexdigest(),
        preregistration_sha256=hashlib.sha256((root/'theory_plan.json').read_bytes()).hexdigest())
    (root/'theory_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--audit-archive',action='store_true')
    parser.add_argument('--audit-records',action='store_true');parser.add_argument('--output',default='results/research')
    args=parser.parse_args();root=Path(args.output)
    analytical=json.loads((root/'theory_analytical.json').read_text())
    empirical=json.loads((root/'theory_empirical.json').read_text())
    plan=json.loads((root/'theory_plan.json').read_text())
    if args.audit_archive: audit_archive(root,empirical,plan)
    if args.audit_records:audit_archive(root,empirical,plan,local_records=True)
    moments=analytical['conditional_moments']
    text=['# Theory-check evidence', '',
          'Numerical inputs: theory_analytical.json and theory_empirical.json. The criterion, seeds and penalties are fixed in theory_plan.json. Verify per-example predictions with the --audit-records option after running experiments.theory_check in the same output directory.', '',
          f"Predeclared actual-estimator transfer criterion supported: **{empirical['transfer_prediction_supported']}**. The criterion was: {plan['criterion']}", '',
          f"High-imbalance mean balanced-accuracy change: {100*empirical['high_imbalance_mean_accuracy_delta']:+.6f} percentage points. Mean absolute-boundary-distance change: {empirical['high_imbalance_mean_boundary_distance_delta']:+.8f} in the population separation coordinate.", '',
          'The conditional-law derivation holds under its assumptions. Its proposed transfer to the actual IFUTSVM estimator is a separate empirical claim. The measured boundary motion fails the stated direction; the small performance change is not evidence of practical superiority. The original hypothesis, seeds, penalties and all outputs are retained.', '',
          '## Conditional-law check', '',
          f"Generated samples: {moments['samples']}. Maximum absolute fitted log density ratio: {moments['maximum_absolute_log_density_ratio']:.12g}. Mean error L2: {moments['mean_error_l2']:.12g}. Relative covariance error: {moments['relative_covariance_error']:.12g}.", '',
          '## Threshold surrogate (not the actual IFUTSVM objective)', '',
          '| Count ratio | Prior threshold | Anchored threshold | Prior balanced error | Anchored balanced error | Balanced optimum |',
          '|---|---:|---:|---:|---:|---:|']
    for r in analytical['threshold_surrogate']['rows']:
        text.append(f"| {r['ratio']} | {r['prior_threshold']:.6f} | {r['anchored_threshold']:.6f} | {r['prior_balanced_error']:.6f} | {r['anchored_balanced_error']:.6f} | {r['balanced_optimal_error']:.6f} |")
    text += ['', '## Actual-estimator contrasts', '',
             'Conditional generation minus random averaging; all fixed seeds and penalties are included. Positive accuracy change and negative distance change favor the proposed mechanism.', '',
             '| Count ratio | Cu | Seeds | Δbalanced accuracy pp | Paired std pp | Boundary pairs | Δabsolute boundary distance |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in empirical['contrasts']:
        distance='unavailable' if r['mean_boundary_distance_delta'] is None else f"{r['mean_boundary_distance_delta']:+.8f}"
        text.append(f"| {r['ratio']} | {r['Cu']:g} | {r['seeds']} | {100*r['mean_balanced_accuracy_delta']:+.6f} | {100*r['std_delta']:.6f} | {r['boundary_pairs']} | {distance} |")
    text += ['', '## Absolute performance and diagnostic limits', '',
             '| Count ratio | Strategy | Cu | Balanced accuracy mean % | Std pp | Multiple-boundary runs | No-boundary runs |',
             '|---|---|---:|---:|---:|---:|---:|']
    groups=sorted({(r['ratio'],r['strategy'],r['Cu']) for r in empirical['runs']})
    for ratio,strategy,cu in groups:
        group=[r for r in empirical['runs'] if (r['ratio'],r['strategy'],r['Cu'])==(ratio,strategy,cu)]
        values=[100*r['balanced_accuracy'] for r in group]
        text.append(f"| {ratio} | {strategy} | {cu:g} | {np.mean(values):.6f} | {np.std(values,ddof=1):.6f} | {sum(r['roots']>1 for r in group)} | {sum(r['roots']==0 for r in group)} |")
    text += ['', 'A nonlinear classifier can cross the population separation line more than once. The declared diagnostic uses the crossing nearest the balanced boundary and reports missing/multiple crossings. This line diagnostic is not a full decision-surface theorem. Population test samples are balanced, overlapping and fixed independently of every fit. The Gaussian experiment tests a mechanism; KEEL evidence is a separate requirement.']
    (root/'theory_details.md').write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    print(f"Theory evidence generated: {len(empirical['runs'])} runs; transfer criterion supported={empirical['transfer_prediction_supported']}")


if __name__=='__main__': main()
