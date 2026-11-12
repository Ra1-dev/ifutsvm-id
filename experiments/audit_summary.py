"""Check every displayed main-table number against persisted outer evaluations."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

from experiments.audit import METRICS
from experiments.evidence import read_table
from experiments.storage import table_path
from experiments.run_all import write_json


def close(actual, expected):
    assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def audit(root, partial=False):
    root = Path(root)
    summary = json.loads((root / ('progress_summary.json' if partial else 'summary.json')).read_text())
    config = json.loads((root / 'experiment_config.json').read_text())
    assert summary['config'] == config
    if not partial:
        assert summary['complete'] and not summary['missing']
    counts = dict(groups=0, outer_values=0, metric_cells=0, csv_rows=0, fit_time_medians=0)
    failures = []
    for protocol, suffix in [('holdout', 'holdout'), ('nested_cv', 'cv')]:
        section = summary['protocols'][protocol]
        assert list(section['datasets']) == config['datasets']
        ranks = {model: {metric: [] for metric in METRICS} for model in config['models']}
        common = []
        for dataset, models in section['datasets'].items():
            for model, item in models.items():
                assert model in config['models'] and item['n'] == 5
                table = read_table(table_path(root, dataset, model, protocol))
                keys = sorted(key for key in table if key.endswith('/evaluation'))
                assert len(keys) == 5
                path = table_path(root, dataset, model, protocol).relative_to(root).as_posix()
                for key, record in table.items():
                    if '/inner/' in key and record.get('error'):
                        failures.append(dict(table=path, record_id=key, error=record['error'],
                            error_stage=record.get('error_stage'), parameters=record['metadata']['parameters']))
                assert item['runs'] == [path + '#' + key for key in keys]
                for metric in METRICS:
                    values = [table[key][metric] for key in keys]
                    value = item['metrics'][metric]
                    assert value['values'] == values
                    close(value['mean'], mean(values))
                    close(value['std'], stdev(values))
                    counts['metric_cells'] += 1
                    counts['outer_values'] += len(values)
                counts['groups'] += 1
            if len(models) != len(config['models']):
                assert partial
                continue
            common.append(dataset)
            for metric in METRICS:
                values = [models[m]['metrics'][metric]['mean'] for m in config['models']]
                best = min(values) if metric == 'training_seconds' else max(values)
                for model, value in zip(config['models'], values):
                    better = sum(other < value if metric == 'training_seconds' else other > value for other in values)
                    rank = 1 + better + (sum(other == value for other in values) - 1) / 2
                    item = models[model]['metrics'][metric]
                    close(item['rank'], rank)
                    assert item['best'] is (value == best)
                    ranks[model][metric].append(rank)
        assert section['aggregate']['datasets'] == common
        for model in config['models']:
            for metric in METRICS:
                rank = section['average_ranks'][model][metric]
                assert rank['datasets'] == len(common)
                if not common:
                    assert rank['mean'] is None
                    continue
                close(rank['mean'], mean(ranks[model][metric]))
                values = [section['datasets'][d][model]['metrics'][metric]['mean'] for d in common]
                aggregate = section['aggregate']['models'][model][metric]
                close(aggregate['mean'], mean(values))
                if len(values) > 1:
                    close(aggregate['std'], stdev(values))
                else:
                    assert aggregate['std'] is None
                if metric == 'training_seconds':
                    times = [v for d in common for v in section['datasets'][d][model]['metrics'][metric]['values']]
                    close(aggregate['median_outer_fit_seconds'], median(times))
                    counts['fit_time_medians'] += 1
        if partial:
            continue
        with (root / 'tables' / f'main_{suffix}.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        expected = {(d, m) for d in config['datasets'] for m in config['models']}
        assert len(rows) == len(expected) and {(r['dataset'], r['model']) for r in rows} == expected
        for row in rows:
            item = section['datasets'][row['dataset']][row['model']]
            assert int(row['n']) == item['n']
            for metric in METRICS:
                value = item['metrics'][metric]
                for stat in ('mean', 'std', 'rank'):
                    close(float(row[f'{metric}_{stat}']), value[stat])
                assert row[f'{metric}_best'] == str(value['best'])
                scale = 1 if metric == 'training_seconds' else 100
                display = f"{scale * value['mean']:.2f} ± {scale * value['std']:.2f}"
                assert row[f'{metric}_display'] == (f'**{display}**' if value['best'] else display)
                close(float(row[f'{metric}_average_rank']), section['average_ranks'][row['model']][metric]['mean'])
            counts['csv_rows'] += 1
    result = dict(**counts, failed_inner_fits=len(failures), retained_failures=failures,
                  complete=summary['complete'], partial=partial,
                  statement='Independent arithmetic over persisted values; ranks from pairwise comparisons; '
                  + ('partial preview has no canonical CSV to audit.' if partial else 'CSV units and best markers checked.'))
    write_json(root / ('progress_summary_audit.json' if partial else 'summary_audit.json'), result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    audit(args.output, args.partial)
