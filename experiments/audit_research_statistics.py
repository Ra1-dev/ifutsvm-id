"""Independently trace research aggregates, CSVs and signed-rank/Holm tests."""
import argparse
from collections import Counter
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean,median,stdev

import numpy as np
from experiments.storage import read_table,table_path

METRICS=('gmean','balanced_accuracy','f1','roc_auc')
MODELS=('conditional_gaussian','random_average','paper_main')


def close(actual,expected):
    assert math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12),(actual,expected)


def signed_rank(differences):
    """Integer doubled ranks and subset-sum counts, without sign enumeration."""
    values=np.round(np.asarray(differences,float),12).tolist()
    values=[v for v in values if v!=0]
    ranks=[2*sum(abs(other)<abs(v) for other in values)+sum(abs(other)==abs(v) for other in values)+1
           for v in values]
    total=sum(ranks);observed=sum(r for r,v in zip(ranks,values) if v>0)
    statistic=min(observed,total-observed)
    counts=Counter({0:1})
    for rank in ranks:
        shifted=Counter({value+rank:count for value,count in counts.items()})
        counts.update(shifted)
    assignments=2**len(values)
    tail=sum(count for value,count in counts.items() if min(value,total-value)<=statistic)
    assert sum(counts.values())==assignments
    return dict(statistic=statistic/2,pvalue=tail/assignments,nonzero=len(values),assignments=assignments)


def audit_paired(record,matrices):
    assert record['alpha']==.05 and record['rounding_decimal_places']==12
    rows=record['comparisons'];assert len(rows)==len(METRICS)*3
    expected_keys=[(metric,MODELS[i],MODELS[j]) for metric in METRICS for i,j in combinations(range(3),2)]
    assert [(r['metric'],r['first'],r['second']) for r in rows]==expected_keys
    probabilities=[]
    for row,(metric,first,second) in zip(rows,expected_keys):
        i,j=MODELS.index(first),MODELS.index(second)
        delta=[values[i]-values[j] for values in matrices[metric]]
        np.testing.assert_allclose(row['differences'],delta,rtol=0,atol=1e-12)
        close(row['mean_difference'],mean(delta));close(row['median_difference'],median(delta))
        expected=signed_rank(delta)
        for key,value in expected.items():close(row[key],value)
        assert row['zero_method']=='wilcox'
        probabilities.append(expected['pvalue'])
    ordered=sorted(range(len(rows)),key=lambda i:(probabilities[i],i))
    for position,index in enumerate(ordered):
        adjusted=min(1,max((len(rows)-j)*probabilities[ordered[j]] for j in range(position+1)))
        close(rows[index]['holm_p'],adjusted)
        assert rows[index]['significant']==(adjusted<record['alpha'])
    return len(rows)


def audit(root='results'):
    root=Path(root);research=root/'research'
    read=lambda path:json.loads(path.read_bytes())
    main=read(root/'summary.json');summary=read(research/'summary.json');stats=read(research/'statistics.json')
    plan=read(research/'ablation_plan.json')
    assert summary['complete'] and summary['plan']==plan and summary['statistics']==stats
    assert main['research']==summary
    sources=[root/'summary.json',research/'summary.json',research/'statistics.json',research/'ablation_plan.json']
    aggregates={};outer_count=0;metric_values=0;csv_rows=0;control_checks=0
    for wave,filename in [('ablation','ablation.csv'),('noise','label_noise.csv')]:
        datasets=plan['datasets'] if wave=='ablation' else plan['noise_datasets']
        models=list(plan['variants']) if wave=='ablation' else plan['noise_models']
        levels=[0.] if wave=='ablation' else plan['noise_levels']
        data=summary[wave];assert data['complete'] and not data['missing']
        assert data['completed_outer']==data['expected_outer']==plan[f'expected_{wave}_outer']
        expected_rows=[];wave_count=0
        for dataset in datasets:
            path=table_path(research,dataset,'research',wave);sources.append(path)
            table=read_table(path)
            for rate in levels:
                groups=data['datasets'][dataset][str(rate)];assert set(groups)==set(models)
                observations={name:[table[f'{name}/holdout-{seed}/noise-{rate:g}/evaluation'] for seed in plan['seeds']]
                              for name in models}
                for name,records in observations.items():
                    assert all('error' not in r for r in records)
                    wave_count+=len(records)
                    group=groups[name];assert group['n']==len(records) and group['seeds']==plan['seeds']
                    expected_paths=[path.as_posix()+'#'+f'{name}/holdout-{seed}/noise-{rate:g}/evaluation' for seed in plan['seeds']]
                    assert group['records']==expected_paths
                    for metric in METRICS:
                        values=[r[metric] for r in records];metric_values+=len(values)
                        delta=[v-r[metric] for v,r in zip(values,observations['random_average'])]
                        row=group['metrics'][metric]
                        assert row['values']==values
                        np.testing.assert_allclose(row['paired_vs_random']['values'],delta,rtol=0,atol=1e-12)
                        for key,value in [('mean',mean(values)),('std',stdev(values))]:close(row[key],value)
                        for key,value in [('mean',mean(delta)),('std',stdev(delta))]:close(row['paired_vs_random'][key],value)
                        aggregates[wave,dataset,rate,name,metric]=mean(values)
                        expected_rows.append(dict(dataset=dataset,variant=name,noise_rate=rate,metric=metric,n=len(records),
                            mean=mean(values),std=stdev(values),paired_delta_mean=mean(delta),paired_delta_std=stdev(delta)))
        assert wave_count==plan[f'expected_{wave}_outer'];outer_count+=wave_count
        csv_path=root/filename;sources.append(csv_path)
        with csv_path.open(encoding='utf-8',newline='') as stream:actual_rows=list(csv.DictReader(stream))
        assert len(actual_rows)==len(expected_rows)
        for actual,expected in zip(actual_rows,expected_rows):
            assert set(actual)==set(expected)
            for key,value in expected.items():
                if isinstance(value,str):assert actual[key]==value
                else:close(float(actual[key]),value)
            csv_rows+=1
    matrices={metric:[] for metric in METRICS}
    for metric in METRICS:
        for dataset in plan['datasets']:
            row=[aggregates['ablation',dataset,0.,name,metric] for name in MODELS[:2]]
            row.append(main['protocols']['holdout']['datasets'][dataset]['paper']['metrics'][metric]['mean'])
            close(row[1],row[2]);control_checks+=1;matrices[metric].append(row)
    paired_checks=audit_paired(stats['paired'],matrices)
    families={}
    for dataset in plan['datasets']:
        name='ecoli' if dataset.startswith('ecoli') else 'yeast' if dataset.startswith('yeast') else dataset
        families.setdefault(name,[]).append(dataset)
    assert stats['family_sensitivity']['families']==families
    family_matrices={metric:[[mean(matrices[metric][plan['datasets'].index(d)][i] for d in group)
                              for i in range(3)] for group in families.values()] for metric in METRICS}
    paired_checks+=audit_paired(stats['family_sensitivity']['paired'],family_matrices)
    record=dict(complete=True,outer_records=outer_count,persisted_metric_values=metric_values,csv_rows=csv_rows,
        paired_comparisons=paired_checks,paper_control_mean_checks=control_checks,failures=0,
        source_sha256={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        statement='Every aggregate and CSV row traced to persisted evaluation metrics; paired ranks and exact probabilities independently computed by integer subset-sum counts, then Holm adjusted.')
    (research/'statistics_audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in record.items() if k!='source_sha256'}));return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results')
    audit(parser.parse_args().output)
