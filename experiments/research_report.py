"""Derive ablation/noise tables and dataset-level statistics from predictions."""
from experiments.display import model_name
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import numpy as np
from experiments.storage import read_table,table_path
from experiments.run_all import write_json
from experiments.statistics import friedman_nemenyi,paired_family

METRICS=('gmean','balanced_accuracy','f1','roc_auc')
METRIC_LABELS=dict(gmean='G-mean',balanced_accuracy='Balanced accuracy',f1='F1',roc_auc='ROC-AUC')


def collect(root,plan,wave):
    datasets=plan['datasets'] if wave=='ablation' else plan['noise_datasets']
    names=list(plan['variants']) if wave=='ablation' else plan['noise_models']
    levels=[0.] if wave=='ablation' else plan['noise_levels']
    result={}; missing=[]; completed=0
    for dataset in datasets:
        path=table_path(root,dataset,'research',wave)
        table=read_table(path) if path.exists() else {}
        result[dataset]={}
        for rate in levels:
            groups={}; result[dataset][str(rate)]=groups
            for name in names:
                keys=[f'{name}/holdout-{seed}/noise-{rate:g}/evaluation' for seed in plan['seeds']]
                present=[key for key in keys if key in table and 'error' not in table[key]]
                completed+=len(present)
                if len(present)!=len(keys):
                    missing.append(dict(dataset=dataset,variant=name,noise_rate=rate,found=len(present),expected=len(keys)))
                    continue
                records=[table[key] for key in keys]
                groups[name]=dict(n=len(records),seeds=plan['seeds'],
                    records=[path.as_posix()+'#'+key for key in keys],
                    metrics={metric:dict(values=[r[metric] for r in records],
                        mean=float(np.mean([r[metric] for r in records])),
                        std=float(np.std([r[metric] for r in records],ddof=1))) for metric in METRICS})
            if 'random_average' in groups:
                baseline=groups['random_average']
                for name,group in groups.items():
                    for metric in METRICS:
                        delta=np.array(group['metrics'][metric]['values'])-baseline['metrics'][metric]['values']
                        group['metrics'][metric]['paired_vs_random']=dict(values=delta.tolist(),
                            mean=float(delta.mean()),std=float(delta.std(ddof=1)))
    return dict(datasets=result,completed_outer=completed,expected_outer=plan[f'expected_{wave}_outer'],
                missing=missing,complete=not missing)


def rows_from(data):
    rows=[]
    for dataset,levels in data['datasets'].items():
        for rate,groups in levels.items():
            for name,group in groups.items():
                for metric,value in group['metrics'].items():
                    delta=value.get('paired_vs_random')
                    rows.append(dict(dataset=dataset,variant=name,noise_rate=float(rate),metric=metric,n=group['n'],
                        mean=value['mean'],std=value['std'],
                        paired_delta_mean=None if delta is None else delta['mean'],
                        paired_delta_std=None if delta is None else delta['std']))
    return rows


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main_statistics(main,plan):
    assert main['complete']
    omnibus={}; family_omnibus={}
    families={}
    for dataset in plan['datasets']:
        family='ecoli' if dataset.startswith('ecoli') else 'yeast' if dataset.startswith('yeast') else dataset
        families.setdefault(family,[]).append(dataset)
    for protocol in ('holdout','nested_cv'):
        names=main['config']['models']; datasets=main['config']['datasets']
        assert datasets==plan['datasets']
        omnibus[protocol]={metric:friedman_nemenyi(
            [[main['protocols'][protocol]['datasets'][d][name]['metrics'][metric]['mean'] for name in names] for d in datasets],names)
            for metric in ('gmean','balanced_accuracy')}
        family_omnibus[protocol]={metric:friedman_nemenyi(
            [[np.mean([main['protocols'][protocol]['datasets'][d][name]['metrics'][metric]['mean'] for d in group])
              for name in names] for group in families.values()],names)
            for metric in ('gmean','balanced_accuracy')}
    return dict(complete=True,source_hash=main['config']['source_hash'],friedman_nemenyi=omnibus,
        family_sensitivity=dict(families=families,friedman_nemenyi=family_omnibus,
            reason='Ecoli and yeast derived tasks share a source family, so the required twelve-task analysis need not contain twelve independent datasets. This sensitivity gives each source family equal weight.'),
        omnibus_families='Each predeclared protocol/metric has a separate omnibus and Nemenyi family; no global across-outcome significance claim.')


def statistics(main,ablation,plan):
    result=main_statistics(main,plan)
    families=result['family_sensitivity']['families']
    names=['conditional_gaussian','random_average','paper_main']
    matrices={metric:[[ablation['datasets'][d]['0.0']['conditional_gaussian']['metrics'][metric]['mean'],
        ablation['datasets'][d]['0.0']['random_average']['metrics'][metric]['mean'],
        main['protocols']['holdout']['datasets'][d]['paper']['metrics'][metric]['mean']] for d in plan['datasets']] for metric in METRICS}
    for matrix in matrices.values(): np.testing.assert_allclose(np.array(matrix)[:,1],np.array(matrix)[:,2],rtol=1e-12,atol=1e-12)
    family_matrices={metric:[np.mean([matrices[metric][plan['datasets'].index(d)] for d in group],axis=0).tolist()
                             for group in families.values()] for metric in METRICS}
    result.update(paired=paired_family(matrices,names,METRICS),
        paper_identity=f'{model_name("paper")} uses random averaging. The separately run ablation random control must match it; this duplicate control is not a third scientific method.')
    result['family_sensitivity']['paired']=paired_family(family_matrices,names,METRICS)
    return result


def notes(summary):
    plan=summary['plan']; theory=summary['theory']
    lines=['# Research evidence notes','',
        'The compact release includes these summaries. To recompute them from predictions,',
        'first run the complete main benchmark, both research experiments and the theory',
        "check described in README, putting the research outputs in the main output",
        "directory's `research/` subdirectory. Then run",
        '`python -m experiments.research_report --output <that-directory>`.','',
        summary['main_finding']['text'],'',
        '## Universum construction','',
        '- Objective: `Q_U = Law(X | a^T(X-m)=0), X ~ N(mu_min, Sigma)`.',
        '- Training-only means and equally weighted class covariance estimates; fixed numerical ridge; no tuned strategy hyperparameters.',
        '- Support, moments, default equivalence, and modified primals: `tests/test_generative.py`.',
        '- Gaussian class assumption and threshold-surrogate scope: `docs/THEORY.md`.',
        '- No population theorem of superiority over random averaging.','',
        '## Protocol and ablation','',
        f"- Shared candidates per split: {plan['budget']}; inner folds: {plan['inner_folds']}; seeds: {plan['seeds']}.",
        f"- One-factor variants: {len(plan['variants'])}; completed outer cells: {summary['ablation']['completed_outer']}.",
        '- Size, IF-score, and side-two controls are separate main-effect contrasts; interactions not estimated.',
        '- Full dataset/variant/metric means, sample SDs and paired changes: `results/ablation.csv`.',
        '- Values below: equal-weight averages of dataset mean paired differences against random averaging.','',
        '| Variant | G-mean change pp | Balanced accuracy change pp | F1 change pp | ROC-AUC change pp |',
        '|---|---:|---:|---:|---:|']
    for name in plan['variants']:
        values=[np.mean([summary['ablation']['datasets'][d]['0.0'][name]['metrics'][metric]['paired_vs_random']['mean']
                        for d in plan['datasets']])*100 for metric in METRICS]
        lines.append('| '+model_name(name)+' | '+' | '.join(f'{v:+.6f}' for v in values)+' |')
    lines += ['', '## Pre-specified theory outcome','',
        f"- Original synthetic records audited: {theory['records_checked']}; transfer supported: {theory['supported']}.",
        f"- High-IR balanced-accuracy change: {100*theory['balanced_accuracy_delta']:+.9f} pp.",
        f"- Absolute-boundary-distance change: {theory['boundary_distance_delta']:+.12g}.",
        f"- Original registration SHA-256: `{theory['preregistration_sha256']}`.",
        '- All penalties/seeds retained. Detailed contrasts and absolute performance: `results/research/theory_empirical.json`.',
        '- KEEL results are external-validity evidence, not a replacement pre-registration.','',
        '## Inference and label noise','',
        '- Statistical unit: dataset mean over five fixed seeds/folds.',
        '- Friedman and Nemenyi rank/critical-difference data: `results/research/statistics.json`.',
        '- Exact two-sided signed-rank enumeration; zero differences removed; fixed twelve-decimal rounding; Holm correction across every declared metric/pair contrast.',
        '- Four identity contrasts (random averaging versus the main paper run) stay in the declared Holm family to preserve its original scope and saved values. Their p=1 contributes no evidence of a method difference. This bookkeeping is conservative; the smallest raw p already exceeds 0.05, so removing duplicates would not produce significance.',
        f'- {model_name("paper")} and its random-averaging ablation control are identical by design; their identity is audited, not presented as two independent competitors.',
        '- Signed-rank interpretation assumes independent datasets and symmetric paired differences. Non-significance does not prove equivalence.',
        '- Ecoli/yeast derivatives may be correlated: the required twelve-task statistics are accompanied by equal-weight source-family sensitivity in statistics.json. Neither analysis is a random sample of every classification domain.',
        '- Statistical method source: [Demšar, JMLR 2006](https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf).',
        f"- Noise datasets: {', '.join(plan['noise_datasets'])}.",
        f"- Noise rates: {plan['noise_levels']}; completed outer cells: {summary['noise']['completed_outer']}.",
        '- Labels flipped only in outer training, before inner tuning; outer test labels stay clean. Same flip permutation and candidates for all models.',
        '- All noise means and paired changes: `results/label_noise.csv`.',
        '- Saved artifact destinations: `figures/fig10_critical_difference`, `figures/fig11_ablation` and `figures/fig12_label_noise`. Order and parameters are fixed.','',
        '### Main-model omnibus and source-family sensitivity','',
        '| Scope | Protocol | Metric | Units | Friedman statistic | p | Nemenyi CD | Source |',
        '|---|---|---|---:|---:|---:|---:|---|']
    stats=summary['statistics']
    for scope,prefix,record in [('Twelve tasks','',stats),
            ('Source families','/family_sensitivity',stats['family_sensitivity'])]:
        for protocol,metrics in record['friedman_nemenyi'].items():
            for metric,row in metrics.items():
                pointer=f'results/research/statistics.json#{prefix}/friedman_nemenyi/{protocol}/{metric}'
                lines.append(f"| {scope} | {'Hold-out' if protocol=='holdout' else 'Nested CV'} | {METRIC_LABELS[metric]} | {row['datasets']} | {row['friedman_statistic']:.9f} | {row['friedman_p']:.12g} | {row['critical_difference']:.9f} | `{pointer}` |")
    lines += ['', '### Exact paired comparisons','',
        '- Difference direction: first model minus second model; percentage points. All declared contrasts remain visible, including duplicate-control identities.',
        '- Run labels identify the ablation control and the independently executed main-protocol control; they use the same corrected model.','',
        '| Scope | Metric | First model / run | Second model / run | Mean change pp | Median change pp | W | Raw p | Holm p | Significant | Source |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---|---|']
    for scope,prefix,record in [('Twelve tasks','',stats),
            ('Source families','/family_sensitivity',stats['family_sensitivity'])]:
        for i,row in enumerate(record['paired']['comparisons']):
            def label(name):
                return model_name(name)+' / '+('main protocol' if name=='paper_main' else 'ablation')
            pointer=f'results/research/statistics.json#{prefix}/paired/comparisons/{i}'
            lines.append(f"| {scope} | {METRIC_LABELS[row['metric']]} | {label(row['first'])} | {label(row['second'])} | {100*row['mean_difference']:+.6f} | {100*row['median_difference']:+.6f} | {row['statistic']:g} | {row['pvalue']:.12g} | {row['holm_p']:.12g} | {row['significant']} | `{pointer}` |")
    lines += ['', '### Label-noise aggregate','',
        '- Percent means and sample SD across dataset means, after averaging the fixed seeds within each dataset. This SD describes dataset heterogeneity; it is not a confidence interval.',
        '- Each source row is `results/research/summary.json#/noise/datasets/<dataset>/<rate>/<model>/metrics/<metric>`; all dataset/seed summaries are retained in `results/label_noise.csv`.','',
        '| Training-label flips | Model | G-mean mean ± SD | Balanced accuracy mean ± SD | F1 mean ± SD | ROC-AUC mean ± SD |',
        '|---:|---|---:|---:|---:|---:|']
    for rate in plan['noise_levels']:
        for name in plan['noise_models']:
            cells=[]
            for metric in METRICS:
                values=np.array([summary['noise']['datasets'][d][str(float(rate))][name]['metrics'][metric]['mean']
                    for d in plan['noise_datasets']])*100
                cells.append(f'{values.mean():.6f} ± {values.std(ddof=1):.6f}')
            lines.append(f'| {100*rate:g}% | {model_name(name)} | '+' | '.join(cells)+' |')
    lines.append('')
    return '\n'.join(lines)


def theory_outcome(summary):
    """Factual outcome block; the original proposition and prediction stay intact."""
    assert summary['complete'], 'Theory outcome requires the complete research experiment'
    theory=summary['theory'];plan=summary['plan']
    lines=['<!-- EMPIRICAL OUTCOME -->','## Recorded empirical outcome','',
        f"- Original pre-specified transfer criterion: {'supported' if theory['supported'] else 'refuted'}; {theory['records_checked']} retained synthetic records.",
        f"- High-IR balanced-accuracy change: {100*theory['balanced_accuracy_delta']:+.9f} percentage points.",
        f"- Absolute-boundary-distance change: {theory['boundary_distance_delta']:+.12g}; the pre-specified criterion requires movement toward the balanced boundary.",
        '- Synthetic numerical source: `results/research/theory_audit.json`; all original contrasts: `results/research/theory_empirical.json`.',
        f"- Original pre-registration SHA-256: `{theory['preregistration_sha256']}`.",
        f"- Complete KEEL ablation: {summary['ablation']['completed_outer']} outer cells over {len(plan['datasets'])} datasets.",
        f'- Comparison below: {model_name("conditional_gaussian")} minus {model_name("random_average")}; dataset means over the fixed seeds.',
        '- KEEL contrasts are external-validity evidence, not a replacement prediction or a proof that the threshold surrogate transfers to the estimator.',
        '- Signed-rank tests assume independent, symmetric paired differences; source-family sensitivity and every declared contrast remain in `docs/RESEARCH.md` and `results/research/statistics.json`.','',
        '| Metric | Mean change pp | Median change pp | W | Raw p | Holm p | Significant | Source |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    selected=[]
    for i,row in enumerate(summary['statistics']['paired']['comparisons']):
        if (row['first'],row['second'])!=('conditional_gaussian','random_average'):continue
        selected.append(row['metric'])
        lines.append(f"| {METRIC_LABELS[row['metric']]} | {100*row['mean_difference']:+.6f} | {100*row['median_difference']:+.6f} | {row['statistic']:g} | {row['pvalue']:.12g} | {row['holm_p']:.12g} | {row['significant']} | `results/research/statistics.json#/paired/comparisons/{i}` |")
    assert selected==list(METRICS)
    lines += ['', '<!-- END EMPIRICAL OUTCOME -->','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--partial',action='store_true')
    mode.add_argument('--main-statistics-only',action='store_true')
    args=parser.parse_args(); root=Path(args.output)
    research=root/'research'; plan=json.loads((research/'ablation_plan.json').read_bytes())
    if args.main_statistics_only:
        stats=main_statistics(json.loads((root/'summary.json').read_bytes()),plan)
        write_json(research/'main_statistics.json',stats)
        for protocol,metrics in stats['friedman_nemenyi'].items():
            for metric,row in metrics.items():
                print(f"{protocol}/{metric}: N={row['datasets']}; Friedman={row['friedman_statistic']:.9f}; p={row['friedman_p']:.12g}; CD={row['critical_difference']:.9f}")
        return
    ablation=collect(research,plan,'ablation'); noise=collect(research,plan,'noise')
    summary=dict(plan=plan,ablation=ablation,noise=noise,complete=ablation['complete'] and noise['complete'])
    if args.partial:
        write_json(research/'progress.json',summary)
        print(f"Research progress: ablation {ablation['completed_outer']}/{ablation['expected_outer']}; noise {noise['completed_outer']}/{noise['expected_outer']}")
        return
    if not summary['complete']: raise ValueError('Research cells incomplete; only --partial output is allowed')
    main_summary=json.loads((root/'summary.json').read_bytes()); assert main_summary['complete']
    stats=statistics(main_summary,ablation,plan)
    main_only=deepcopy(stats)
    main_only.pop('paired');main_only.pop('paper_identity')
    main_only['family_sensitivity'].pop('paired')
    write_json(research/'main_statistics.json',main_only)
    main_summary['main_statistics']=main_only
    from experiments.findings import headline
    summary.update(statistics=stats,theory=json.loads((research/'theory_audit.json').read_bytes()),main_finding=headline(main_summary))
    write_csv(root/'ablation.csv',rows_from(ablation)); write_csv(root/'label_noise.csv',rows_from(noise))
    write_json(research/'statistics.json',stats); write_json(research/'summary.json',summary)
    main_summary['research']=summary; write_json(root/'summary.json',main_summary)
    Path('docs/RESEARCH.md').write_text(notes(summary),encoding='utf-8',newline='\n')
    theory_path=Path('docs/THEORY.md')
    text=theory_path.read_text(encoding='utf-8').split('<!-- EMPIRICAL OUTCOME -->',1)[0].rstrip()
    text=text.replace('The KEEL ablation will compare','The KEEL ablation compares')
    text=text.replace('KEEL ablation evidence is still pending.','The complete KEEL contrasts are recorded below.')
    theory_path.write_text(text+'\n\n'+theory_outcome(summary),encoding='utf-8',newline='\n')
    print(f"Research summary complete: {ablation['completed_outer']} ablation and {noise['completed_outer']} noise cells; all tables/statistics generated")


if __name__=='__main__': main()
