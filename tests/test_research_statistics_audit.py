"""an independent audit must reject plausible corrupted inference."""
from copy import deepcopy
import pytest
from experiments.statistics import paired_family
from experiments.audit_research_statistics import audit_paired,METRICS,MODELS


def fixture():
    # Ties, zeros, both signs and one completely identical control pair.
    differences=[.05,-.03,.02,.02,0.,-.01,.06,-.04,.01,0.,.02,.03]
    matrices={metric:[[.5+d,.5,.5] for d in differences] for metric in METRICS}
    return matrices,paired_family(matrices,MODELS,METRICS)


def test_independent_subset_sum_agrees_with_declared_exact_test():
    matrices,record=fixture()
    assert audit_paired(record,matrices)==12


@pytest.mark.parametrize('field',['pvalue','holm_p','mean_difference','significant'])
def test_research_audit_rejects_changed_inference(field):
    matrices,record=fixture();record=deepcopy(record)
    row=record['comparisons'][0]
    row[field]=(not row[field]) if field=='significant' else row[field]+.01
    with pytest.raises(AssertionError):audit_paired(record,matrices)


def test_persisted_research_tables_are_audited_end_to_end(tmp_path,monkeypatch):
    import csv
    from pathlib import Path
    from experiments.run_all import write_json
    from experiments.storage import write_table,table_path
    from experiments.research_report import collect,rows_from,write_csv
    from experiments.audit_research_statistics import audit
    monkeypatch.chdir(tmp_path)
    root=Path('results');research=root/'research'
    datasets=['ecoli3','ecoli4','pima'];names=['conditional_gaussian','random_average']
    plan=dict(datasets=datasets,noise_datasets=datasets,seeds=list(range(5)),variants=dict.fromkeys(names,{}),
        noise_models=[*names,'svm_balanced'],noise_levels=[0.,.1],expected_ablation_outer=30,expected_noise_outer=90)
    for wave in ['ablation','noise']:
        models=names if wave=='ablation' else plan['noise_models']
        for d,dataset in enumerate(datasets):
            records={}
            for rate in ([0.] if wave=='ablation' else plan['noise_levels']):
                for i,name in enumerate(models):
                    for seed in plan['seeds']:
                        records[f'{name}/holdout-{seed}/noise-{rate:g}/evaluation']={
                            metric:.55+.02*d+((d%2)*2-1)*.01*i+.001*seed-rate for metric in METRICS}
            write_table(table_path(research,dataset,'research',wave),records)
    ablation=collect(research,plan,'ablation');noise=collect(research,plan,'noise')
    matrices={metric:[[ablation['datasets'][d]['0.0'][name]['metrics'][metric]['mean']
                       for name in [*names,'random_average']] for d in datasets] for metric in METRICS}
    family_matrices={metric:[[sum(values[i] for values in matrices[metric][:2])/2 for i in range(3)],
                              matrices[metric][2]] for metric in METRICS}
    stats=dict(paired=paired_family(matrices,MODELS,METRICS),family_sensitivity=dict(
        families=dict(ecoli=datasets[:2],pima=datasets[2:]),paired=paired_family(family_matrices,MODELS,METRICS)))
    summary=dict(complete=True,plan=plan,ablation=ablation,noise=noise,statistics=stats)
    main=dict(research=summary,protocols=dict(holdout=dict(datasets={d:dict(paper=dict(metrics={
        metric:dict(mean=matrices[metric][i][2]) for metric in METRICS})) for i,d in enumerate(datasets)})))
    for path,value in [(research/'ablation_plan.json',plan),(research/'summary.json',summary),
                       (research/'statistics.json',stats),(root/'summary.json',main)]:write_json(path,value)
    write_csv(root/'ablation.csv',rows_from(ablation));write_csv(root/'label_noise.csv',rows_from(noise))
    result=audit(root)
    assert result['outer_records']==120 and result['persisted_metric_values']==480
    assert result['csv_rows']==96 and result['paired_comparisons']==24
    with (root/'ablation.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    rows[0]['mean']=float(rows[0]['mean'])+.01;write_csv(root/'ablation.csv',rows)
    with pytest.raises(AssertionError):audit(root)
