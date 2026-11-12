"""Generate factual reproduction notes and CSV directly from persisted evidence."""
from experiments.display import model_name,CONTROL_NAMES
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from experiments.reproduction_probes import VARIANTS
from experiments.run_all import write_json
from experiments.storage import table_path
from experiments.evidence import read_table,table_exists
from experiments.findings import headline


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results')
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args(); root=Path(args.output)
    config=json.loads((root/'experiment_config.json').read_bytes())
    published=json.loads(Path('data/published_table_sii.json').read_bytes())
    manifest=json.loads(Path('data/keel/MANIFEST.json').read_bytes())['datasets']
    arithmetic={r['dataset']:r for r in json.loads((root/'table_arithmetic.json').read_bytes())}
    literature=json.loads((root/'literature.json').read_bytes())
    ceiling_path=root/'accuracy_ceiling/summary.json'
    ceiling=json.loads(ceiling_path.read_bytes()) if ceiling_path.exists() else None
    if ceiling:
        audit=json.loads((root/'accuracy_ceiling/audit.json').read_bytes())
        assert ceiling['complete'] and audit['complete']
        assert audit['summary_sha256']==hashlib.sha256(ceiling_path.read_bytes()).hexdigest()
    rows=[]; evidence={}
    for dataset in config['datasets']:
        main_path=table_path(root,dataset,'paper','holdout')
        probe_path=root/'reproduction_probes'/f'{dataset}.parquet'
        main_records=read_table(main_path) if table_exists(main_path) else {}
        probes=read_table(probe_path) if probe_path.exists() else {}
        required=[f'holdout-{s}/reproduction' for s in range(5)]+['holdout-0/published_config']
        required_probes=[f'{v}/seed-{s}' for v,setup in VARIANTS.items()
                         if dataset in setup.get('datasets',config['datasets']) for s in setup['seeds']]
        complete=all(k in main_records for k in required) and all(k in probes for k in required_probes)
        if args.partial and not complete:
            continue
        if not complete: raise ValueError(f'Incomplete reproduction evidence: {dataset}')
        records=[main_records[f'holdout-{s}/reproduction'] for s in range(5)]
        fixed=main_records['holdout-0/published_config']
        values=np.array([r['accuracy'] for r in records])*100
        table=published['datasets'][dataset]
        mean=float(values.mean()); gap=mean-table['accuracy_percent']
        counts=manifest[dataset]['class_counts']
        row=dict(dataset=dataset,published_accuracy_percent=table['accuracy_percent'],
            tuned_accuracy_mean_percent=mean,tuned_accuracy_std_pp=float(values.std(ddof=1)),
            gap_pp=gap,within_two_points=abs(gap)<=2,
            tuned_balanced_accuracy_mean_percent=float(np.mean([r['balanced_accuracy'] for r in records])*100),
            published_config_seed0_accuracy_percent=None if 'error' in fixed else 100*fixed['accuracy'],
            published_config_seed0_balanced_accuracy_percent=None if 'error' in fixed else 100*fixed['balanced_accuracy'],
            published_config_error=fixed.get('error',''),
            full_data_majority_rate_percent=100*max(counts.values())/sum(counts.values()),
            raw_feature_count=manifest[dataset]['features'],published_feature_count=table['features'],
            unreported_C3_default=published['unreported_C3_default'])
        rows.append(row)
        variants={}
        for variant,setup in VARIANTS.items():
            if dataset not in setup.get('datasets',config['datasets']): continue
            results=[probes[f'{variant}/seed-{s}'] for s in setup['seeds']]
            valid=[r for r in results if 'error' not in r]
            variants[variant]=dict(expected=len(results),valid=len(valid),errors=[r['error'] for r in results if 'error' in r],
                accuracy_mean_percent=float(np.mean([r['accuracy'] for r in valid])*100) if valid else None,
                accuracy_min_percent=min(100*r['accuracy'] for r in valid) if valid else None,
                accuracy_max_percent=max(100*r['accuracy'] for r in valid) if valid else None,
                balanced_accuracy_mean_percent=float(np.mean([r['balanced_accuracy'] for r in valid])*100) if valid else None,
                accuracy_std_pp=float(np.std([100*r['accuracy'] for r in valid],ddof=1)) if len(valid)>1 else None,
                results={str(r['metadata']['seed']):r for r in results},discrepancies=setup['discrepancies'])
        base=variants['published_scaled']['results']
        for variant,v in variants.items():
            pairs=[(r,base[s]) for s,r in v['results'].items() if 'error' not in r and 'error' not in base[s]]
            v['paired_seeds']=len(pairs)
            v['accuracy_delta_pp']=float(np.mean([100*(a['accuracy']-b['accuracy']) for a,b in pairs])) if pairs else None
            v['balanced_accuracy_delta_pp']=float(np.mean([100*(a['balanced_accuracy']-b['balanced_accuracy']) for a,b in pairs])) if pairs else None
            v['same_test_rows']=all(a['test_indices']==b['test_indices'] for a,b in pairs)
            # Complete prediction evidence stays in the typed Parquet tables.
            v.pop('results')
        baseline=variants['published_scaled']['accuracy_mean_percent']
        fixed_records=[probes[f'published_scaled/seed-{s}'] for s in VARIANTS['published_scaled']['seeds']]
        row.update(published_parameter_balanced_accuracy_mean_percent=variants['published_scaled']['balanced_accuracy_mean_percent'],
            published_probe_all_majority=all('error' not in r and len(set(r['prediction']))==1 and r['positive_label'] not in r['prediction'] for r in fixed_records),
            published_below_full_majority=table['accuracy_percent']<row['full_data_majority_rate_percent'],
            published_equals_IF_USVM=table['accuracy_percent']==literature['datasets'][dataset]['accuracy_percent']['IF-USVM'],
            accuracy_ceiling_complete=None,accuracy_ceiling_max_family_mean_percent=None,
            accuracy_ceiling_suspect=None)
        row['published_parameter_mean_percent']=baseline
        row['tuning_configuration_contrast_pp']=mean-baseline if baseline is not None else None
        row['published_parameter_gap_pp']=baseline-table['accuracy_percent'] if baseline is not None else None
        unstratified=variants['unstratified']
        row['unstratified_predeclared_min_percent']=unstratified['accuracy_min_percent']
        row['unstratified_predeclared_max_percent']=unstratified['accuracy_max_percent']
        row['metric_definition_contrast_pp']=mean-row['tuned_balanced_accuracy_mean_percent']
        row['balanced_accuracy_gap_against_published_pp']=row['tuned_balanced_accuracy_mean_percent']-table['accuracy_percent']
        if row['within_two_points']:
            row['evidence_category']='mean_within_tolerance'
        elif baseline is not None and abs(row['published_parameter_gap_pp'])<=2:
            row['evidence_category']='configuration_contrast_leaves_residual_within_tolerance'
        elif dataset=='vehicle1' and row['published_probe_all_majority'] and row['published_below_full_majority'] and row['published_equals_IF_USVM']:
            row['evidence_category']='published_configuration_all_majority_published_below_majority_and_equals_IF_USVM'
        elif abs(row['balanced_accuracy_gap_against_published_pp'])<=2:
            row['evidence_category']='metric_definition_contrast_leaves_residual_within_tolerance'
        else:
            row['evidence_category']='residual_unresolved'
        if dataset=='yeast1' and ceiling:
            row.update(accuracy_ceiling_complete=True,
                accuracy_ceiling_max_family_mean_percent=max(v['accuracy_mean_percent'] for v in ceiling['models'].values()),
                accuracy_ceiling_suspect=ceiling['implementation_suspect'])
            if not ceiling['implementation_suspect']:
                row['evidence_category']='published_accuracy_unattained_by_preregistered_four_family_probe'
        row['gap_explained']=row['evidence_category']!='residual_unresolved'
        evidence[dataset]=dict(comparison=row,variants=variants,table_arithmetic=arithmetic[dataset],
                               reproduced=abs(gap)<=2,authors_configuration_identified=False,
                               selected_parameters=[r['metadata']['parameters'] for r in records])
        if dataset=='yeast1':
            ordered=sorted(literature['datasets'][dataset]['accuracy_percent'].items(),key=lambda x:x[1])
            split=max(range(1,len(ordered)),key=lambda i:ordered[i][1]-ordered[i-1][1])
            evidence[dataset]['published_bimodality']=dict(lower=dict(ordered[:split]),upper=dict(ordered[split:]),
                separating_gap_pp=ordered[split][1]-ordered[split-1][1],source='results/literature.json#/datasets/yeast1/accuracy_percent')
            evidence[dataset]['accuracy_ceiling']=ceiling
    if args.partial:
        write_json(root/'reproduction_progress.json',dict(datasets=evidence,complete=False,
            within_tolerance=sum(r['within_two_points'] for r in rows),datasets_available=len(rows),
            datasets_expected=len(config['datasets'])))
        print(f"Reproduction progress: {len(rows)}/{len(config['datasets'])} datasets have complete diagnostics; {sum(r['within_two_points'] for r in rows)} within tolerance")
        return
    with (root/'reproduction.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    write_json(root/'reproduction.json',dict(datasets=evidence,
        within_tolerance=sum(r['within_two_points'] for r in rows),datasets_total=len(rows),
        matched_or_explained=sum(r['gap_explained'] for r in rows),e4_complete=all(r['gap_explained'] for r in rows),
        note='A measured diagnostic effect is not proof of the authors unpublished configuration. Unreproduced numbers remain explicitly unreproduced.'))
    text=['# Reproduction evidence', '',
          'Recompute these comparisons with `python -m experiments.reproduction_report` from persisted predictions, the fixed diagnostic plan, and Table S.II.', '',
          'Table S.II supplies C, Cu, epsilon and bandwidth but omits the independent C3. Its split indices, random seed and complete preprocessing are unavailable. We use the stated default for the missing parameter and do not infer it from test scores.', '',
          'Main reproduction uses accuracy-selected inner CV on each prescribed stratified holdout. Reference-configuration diagnostics use the published parameters and fixed seeds. The separately pre-specified yeast1 accuracy-ceiling diagnostic tunes four other model families on accuracy. None changes the main comparison, which uses balanced accuracy for selection. Metrics are ordinary accuracy unless explicitly named otherwise.', '',
          f"Accuracy means within the prescribed tolerance: {sum(r['within_two_points'] for r in rows)}/{len(rows)}. Values outside tolerance are not claimed as reproduced.", '',
          headline(json.loads((root/'summary.json').read_bytes()))['text'], '',
          '| Dataset | Published ACC % | Reproduction mean ± std % | Gap pp | Published config, seed 0 % | Within tolerance |',
          '|---|---:|---:|---:|---:|---|']
    for r in rows:
        fixed='FIT FAILED' if r['published_config_seed0_accuracy_percent'] is None else f"{r['published_config_seed0_accuracy_percent']:.4f}"
        text.append(f"| {r['dataset']} | {r['published_accuracy_percent']:.4f} | {r['tuned_accuracy_mean_percent']:.4f} ± {r['tuned_accuracy_std_pp']:.4f} | {r['gap_pp']:+.4f} | {fixed} | {r['within_two_points']} |")
    text += ['', '## Controlled diagnostic effects', '',
             'Changes below are relative to the published-parameter, standardized, symmetric, reduced-basis probe on matching seeds. A protocol contrast may change test rows and is labeled accordingly. Full-basis and normalization diagnostics have a single predeclared seed. Missing or failed probes are shown, never replaced by favorable cases. Effects can identify sensitivity to implementation choices; they do not establish which choice produced the authors number.', '']
    for dataset,data in evidence.items():
        r=data['comparison']
        text += [f'### {dataset}', '',
                 f"Official raw features: {r['raw_feature_count']}; Table S.II features: {r['published_feature_count']}. Full-data majority-class rate: {r['full_data_majority_rate_percent']:.4f}%. Unreported C3 default: {r['unreported_C3_default']:.6g}.", '',
                 '| Fixed variant | Valid/expected | ACC mean % | Balanced ACC mean % | ΔACC pp | Δbalanced accuracy pp | Same test rows | Register |',
                 '|---|---:|---:|---:|---:|---:|---|---|']
        for variant,v in data['variants'].items():
            fmt=lambda x:'unavailable' if x is None else f'{x:.4f}'
            text.append(f"| {CONTROL_NAMES[variant]} | {v['valid']}/{v['expected']} | {fmt(v['accuracy_mean_percent'])} | {fmt(v['balanced_accuracy_mean_percent'])} | {fmt(v['accuracy_delta_pp'])} | {fmt(v['balanced_accuracy_delta_pp'])} | {v['same_test_rows']} | {', '.join(v['discrepancies'])} |")
            for error in v['errors']: text.append(f'\nProbe failure: `{error}`.\n')
        if not data['table_arithmetic']['compatible_with_single_split_accuracy']:
            nearest=min(data['table_arithmetic']['candidates'],key=lambda v:v['rounding_distance_pp'])
            text += ['', f"Table arithmetic: nearest possible ordinary accuracy for either rounded holdout size differs by {nearest['rounding_distance_pp']:.10g} pp. This checks the single-split interpretation only; it does not explain a larger performance gap or rule out an unstated aggregation."]
        if not data['reproduced']:
            text += ['', 'Status: not reproduced. The table above measures effects of the pre-specified ambiguities and differences. The authors exact configuration remains unidentified; any residual difference is not assigned a fabricated cause.']
            if r['evidence_category']=='configuration_contrast_leaves_residual_within_tolerance':
                text += ['', f"Measured configuration decomposition: changing from the fixed published-parameter probe to the training-selected configurations shifts the mean by {r['tuning_configuration_contrast_pp']:+.4f} pp. The published-parameter mean is {r['published_parameter_mean_percent']:.4f}%, leaving {r['published_parameter_gap_pp']:+.4f} pp against the paper, within tolerance. This contrast measures the effect of configuration choice, including the explicit missing-C3 convention (DSC-34); it does not identify the authors unpublished settings."]
            elif dataset=='vehicle1':
                params=published['datasets'][dataset]
                text += ['', f"Measured published-configuration facts (DSC-13/DSC-34): C={params['C']:g}, Cu={params['Cu']:g}, epsilon={params['eps']:g}, sigma={params['mu']:g}. All {data['variants']['published_scaled']['expected']} fixed scaled probes predict only the majority class: balanced accuracy {r['published_parameter_balanced_accuracy_mean_percent']:.4f}%, ordinary accuracy {r['published_parameter_mean_percent']:.4f}%, versus the full-data majority rate {r['full_data_majority_rate_percent']:.4f}%. The published {r['published_accuracy_percent']:.2f}% is below that majority rate and equals the IF-USVM column ({literature['datasets'][dataset]['accuracy_percent']['IF-USVM']:.2f}%). These are measured facts; neither a duplication cause nor the authors split identity is established."]
            elif r['evidence_category']=='metric_definition_contrast_leaves_residual_within_tolerance':
                text += ['', f"Leading supported but unproven hypothesis (DSC-13): the published ACC used balanced accuracy, as IFUTSVM.m returns balanced accuracy under the name accuracy. On the identical accuracy-selected predictions, ordinary accuracy exceeds that definition by {r['metric_definition_contrast_pp']:+.4f} pp. Balanced accuracy leaves {r['balanced_accuracy_gap_against_published_pp']:+.4f} pp against the published number, within tolerance. This supports the metric-interpretation hypothesis without proving which metric the authors used; ordinary accuracy remains unmatched."]
        if dataset=='yeast1':
            bimodal=data['published_bimodality']
            text += ['', 'Published Table S.II yeast1 row (as published):', '',
                '| Lower group | ACC % | Upper group | ACC % |','|---|---:|---|---:|']
            for (left,lv),(right,rv) in zip(bimodal['lower'].items(),bimodal['upper'].items()):
                text.append(f'| {left} | {lv:.2f} | {right} | {rv:.2f} |')
            text += ['', f"Visible bimodality: {len(bimodal['lower'])} methods at {min(bimodal['lower'].values()):.2f}–{max(bimodal['lower'].values()):.2f}%, and {len(bimodal['upper'])} at {min(bimodal['upper'].values()):.2f}–{max(bimodal['upper'].values()):.2f}%; separating gap {bimodal['separating_gap_pp']:.2f} pp. No explanation for that clustering is inferred from the row alone."]
            if ceiling:
                plan=json.loads((root/'accuracy_ceiling/plan.json').read_bytes())
                text += ['', f"Accuracy-ceiling diagnostic fixed in advance at `{ceiling['preregistration_commit']}`; plan SHA-256 `{ceiling['preregistration_sha256']}`. Official KEEL file, protocol A, {len(plan['seeds'])} seeds, {plan['budget']} candidates per family/split, {plan['inner_folds']} inner folds; StandardScaler fitted on training rows only; accuracy selection with first-candidate tie breaking. Threshold fixed before fits: any family mean >= {100*plan['criterion']['suspect_if_any_mean_accuracy_at_least']:.0f}% makes our implementation suspect and leaves the reproduction question unresolved.", '',
                    '| Family | Mean ACC % | Sample SD pp | Best single split % |','|---|---:|---:|---:|']
                for name,v in ceiling['models'].items():
                    text.append(f"| {model_name(name)} | {v['accuracy_mean_percent']:.6f} | {v['accuracy_std_pp']:.6f} | {v['best_split_accuracy_percent']:.6f} |")
                text += ['', ceiling['conclusion'], '',
                    'All families, splits and losses retained. This diagnostic does not alter the balanced-accuracy-selected main benchmark. Commands: `python -m experiments.accuracy_ceiling`; `python -m experiments.audit_accuracy_ceiling`. Predictions, scores, scalers, candidates and indices: `results/accuracy_ceiling/runs/*.parquet`.']
        text += ['', f"Metric-definition control (DSC-13): the same accuracy-selected configurations average {r['tuned_accuracy_mean_percent']:.4f}% ordinary accuracy and {r['tuned_balanced_accuracy_mean_percent']:.4f}% balanced accuracy. The executable returns the latter under its accuracy name. This establishes the size of the metric discrepancy on these outputs, not which metric the authors used for this table row."]
        text += ['']
    text += ['## Reproduction limits', '',
             'The timing-limited search budget is recorded in results/compute_budget.json and is smaller than the paper grid. C3 is not published. The manuscript and executable disagree on kernels, signs, basis, normalization interpretation and reported metrics; the register distinguishes each. Abalone includes a nominal feature absent from the published feature count. A matched accuracy alone does not establish equivalent fitted models. Statistical variation is shown over the fixed runs, not selected seeds.', '',
             'Commands: `python -m experiments.reproduction_probes`; `python -m experiments.table_audit`; `python -m experiments.reproduction_report`. Raw per-configuration predictions, scores and indices are in results/reproduction_probes/*.parquet. Main accuracy-selected outputs are reproduction records in the *__paper__holdout.parquet members of results/runs/<dataset>.zip. The evidence reader opens these bundles directly; `python -m experiments.evidence unpack` restores exact loose tables when needed.']
    Path('docs/REPRODUCTION.md').write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    print(f"Reproduction within tolerance: {sum(r['within_two_points'] for r in rows)}/{len(rows)}; matched or explained: {sum(r['gap_explained'] for r in rows)}/{len(rows)}; analysis complete={all(r['gap_explained'] for r in rows)}")


if __name__=='__main__': main()
