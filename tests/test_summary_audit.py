"""reject numeric and presentation corruption in generated tables."""
from pathlib import Path
import json
import pytest
from experiments.audit import METRICS
from experiments.audit_summary import audit
from experiments.run_all import write_json
from experiments.storage import table_path, write_table
from experiments.summarize import summarize


@pytest.mark.parametrize('corruption', ['mean', 'rank', 'marker', 'median'])
def test_summary_audit_rejects_corruption(tmp_path, corruption):
    config = dict(datasets=['ecoli4'], models=['svm_balanced', 'tsvm'])
    write_json(tmp_path / 'experiment_config.json', config)
    literature={'citation':'Demšar: comparison – fuzzy “scores”'}
    (tmp_path/'literature.json').write_text(json.dumps(literature,ensure_ascii=False),encoding='utf-8')
    for protocol in ('holdout', 'nested_cv'):
        for model_index, model in enumerate(config['models']):
            rows = {}
            for seed in range(5):
                # Non-perfect, unequal, nonconstant outcomes discriminate every marker/rank.
                values = {metric: .45 + .04 * seed + .08 * model_index for metric in METRICS}
                values['training_seconds'] = 1 + seed + 2 * model_index
                key = f"{'holdout-' if protocol == 'holdout' else 'outer-'}{seed}/evaluation"
                rows[key] = dict(**values, confusion_matrix=[[2, 1], [1, 2]], roc={}, positive_label='positive',
                    test_indices=[0, 1], prediction=['negative', 'positive'], scores=[-.1, .1],
                    metadata=dict(parameters={}, seed=seed))
            write_table(table_path(tmp_path, 'ecoli4', model, protocol), rows)
    canonical_docs={path:path.read_bytes() if path.exists() else None
                    for path in (Path('README.md'),Path('docs/MAIN_FINDINGS.md'))}
    summary = summarize(tmp_path)
    assert summary['literature']==literature
    for path,before in canonical_docs.items():
        assert (path.read_bytes() if path.exists() else None)==before
    result = audit(tmp_path)
    assert result['metric_cells'] == 36 and result['csv_rows'] == 4
    if corruption in ('mean', 'rank'):
        summary['protocols']['holdout']['datasets']['ecoli4']['svm_balanced']['metrics']['accuracy'][corruption] += .1
    elif corruption == 'median':
        summary['protocols']['holdout']['aggregate']['models']['svm_balanced']['training_seconds']['median_outer_fit_seconds'] += .1
    else:
        path = tmp_path / 'tables/main_holdout.csv'
        path.write_text(path.read_text(encoding='utf-8').replace('**', ''), encoding='utf-8')
    write_json(tmp_path / 'summary.json', summary)
    with pytest.raises(AssertionError):
        audit(tmp_path)
