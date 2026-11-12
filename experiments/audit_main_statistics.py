"""Re-derive omnibus ranks, tie correction, probabilities and every Nemenyi pair."""
import argparse
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import chi2, studentized_range
from experiments.run_all import write_json


def close(actual, expected):
    assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def audit(root):
    root = Path(root)
    summary = json.loads((root / 'summary.json').read_bytes())
    result = json.loads((root / 'research/main_statistics.json').read_bytes())
    assert summary['complete'] and result['complete']
    assert result['source_hash'] == summary['config']['source_hash']
    names = summary['config']['models']; datasets = summary['config']['datasets']
    families = {}
    for dataset in datasets:
        family = 'ecoli' if dataset.startswith('ecoli') else 'yeast' if dataset.startswith('yeast') else dataset
        families.setdefault(family, []).append(dataset)
    assert result['family_sensitivity']['families'] == families
    checked = []
    for scope, groups, tables in [
        ('datasets', [[d] for d in datasets], result['friedman_nemenyi']),
        ('source_families', list(families.values()), result['family_sensitivity']['friedman_nemenyi'])]:
        for protocol in ['holdout', 'nested_cv']:
            for metric in ['gmean', 'balanced_accuracy']:
                row = tables[protocol][metric]
                n, k = len(groups), len(names)
                assert row['models'] == names and row['datasets'] == n and row['alpha'] == .05
                expected = [[mean(summary['protocols'][protocol]['datasets'][d][model]['metrics'][metric]['mean']
                                  for d in group) for model in names] for group in groups]
                values = row['per_dataset_values']
                np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)
                ranks = [[1 + sum(other > value for other in observations)
                          + (sum(other == value for other in observations)-1)/2 for value in observations]
                         for observations in values]
                assert row['per_dataset_ranks'] == ranks
                rank_means = [mean(r[j] for r in ranks) for j in range(k)]
                ties = 0
                for observations in values:
                    for value in set(observations):
                        size = observations.count(value)
                        ties += size**3 - size
                correction = 1 - ties / (n*k*(k*k-1))
                raw = 12*n/(k*(k+1)) * sum((value-(k+1)/2)**2 for value in rank_means)
                statistic = raw/correction if correction else 0.
                probability = float(chi2.sf(statistic, k-1)) if correction else 1.
                close(row['friedman_statistic'], statistic)
                close(row['friedman_p'], probability)
                se = math.sqrt(k*(k+1)/(6*n))
                # Check the quantile through its CDF, independently of the ppf call.
                close(float(studentized_range.cdf(row['q_alpha']*math.sqrt(2), k, np.inf)), 1-row['alpha'])
                close(row['critical_difference'], row['q_alpha']*se)
                assert len(row['pairs']) == k*(k-1)//2
                for model, rank in zip(names, rank_means):
                    close(row['average_ranks'][model], rank)
                pairs = {(pair['first'], pair['second']): pair for pair in row['pairs']}
                assert len(pairs) == len(row['pairs'])
                for i, j in combinations(range(k), 2):
                    pair = pairs[names[i], names[j]]
                    difference = abs(rank_means[i]-rank_means[j])
                    close(pair['rank_difference'], difference)
                    close(pair['nemenyi_p'], float(1-studentized_range.cdf(difference/se*math.sqrt(2), k, np.inf)))
                    exceeds = difference > row['critical_difference']
                    assert pair['exceeds_critical_difference'] == exceeds
                    assert pair['significant'] == (probability < row['alpha'] and exceeds)
                checked.append(dict(scope=scope, protocol=protocol, metric=metric, units=n, pairs=len(pairs)))
    audit_result = dict(complete=True, omnibus_checks=len(checked),
        pairwise_checks=sum(row['pairs'] for row in checked), checks=checked,
        source_hash=result['source_hash'],
        statement='Means checked against summary; ranks and tie correction from comparisons/counts; chi-square probabilities and studentized-range CDF checked independently.')
    write_json(root / 'research/main_statistics_audit.json', audit_result)
    print(json.dumps(audit_result, indent=2))
    return audit_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='results')
    audit(parser.parse_args().output)
