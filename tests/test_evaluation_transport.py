"""independently check tied ROC evidence and strict persisted row identity."""
import csv
import json

import numpy as np
import pytest

from ifutsvm.evaluation import binary_metrics, save_run


def test_tied_auc_matches_pairwise_probability_and_roc_integral():
    rng = np.random.default_rng(447031)
    observed = set()
    for positives in (1, 2, 4, 7):
        for negatives in (1, 3, 9, 13):
            truth = np.r_[np.ones(positives, int), np.zeros(negatives, int)]
            for scores in (np.zeros(len(truth)), rng.integers(-2, 3, len(truth))):
                prediction = (scores > 0).astype(int)
                measured = binary_metrics(truth, prediction, scores, 1, 0)
                # Mann-Whitney probability uses every positive/negative pair;
                # ties contribute one half, with no rank/ROC library call.
                expected = sum(float(a > b) + .5*float(a == b)
                               for a in scores[:positives]
                               for b in scores[positives:])/(positives*negatives)
                assert measured['roc_auc'] == pytest.approx(expected, abs=1e-14)
                roc = measured['roc']
                integral = sum((x1-x0)*(y1+y0)/2 for x0, x1, y0, y1 in zip(
                    roc['fpr'], roc['fpr'][1:], roc['tpr'], roc['tpr'][1:]))
                assert integral == pytest.approx(expected, abs=1e-14)
                transformed = binary_metrics(truth, prediction, scores*3+11, 1, 0)
                assert transformed['roc_auc'] == measured['roc_auc']
                assert transformed['roc']['fpr'] == roc['fpr']
                assert transformed['roc']['tpr'] == roc['tpr']
                reversed_score = binary_metrics(truth, prediction, -scores, 1, 0)
                assert reversed_score['roc_auc'] == pytest.approx(1-expected, abs=1e-14)
                observed.add(expected)
    assert len(observed) > 10  # The corpus includes distinct nontrivial outcomes.


def test_unicode_label_json_csv_round_trip(tmp_path):
    labels = ['rare,"\u03bb"\nsecond line', 'common,\u00e9']
    truth = [labels[i] for i in (0, 1, 1, 0, 1)]
    prediction = [labels[i] for i in (1, 1, 0, 0, 1)]
    scores = [-.5, 0., 0., .75, -1.]
    path = tmp_path/'nested'/'evaluation.json'
    record = save_run(path, y=truth, prediction=prediction, scores=scores,
        positive_label=labels[0], training_seconds=.125,
        train_indices=[0, 2, 4], test_indices=[11, 7, 9, 1, 3],
        metadata={'description': labels[0]}, csv_output=True)
    # Disallow the nonstandard JSON NaN/Infinity extensions while decoding.
    def reject_constant(value):
        raise AssertionError(value)
    restored = json.loads(path.read_bytes(), parse_constant=reject_constant)
    assert restored == record
    assert restored['roc']['thresholds'][0] is None
    with path.with_suffix('.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['row_index']) for row in rows] == [11, 7, 9, 1, 3]
    assert [row['actual'] for row in rows] == truth
    assert [row['prediction'] for row in rows] == prediction
    assert [float(row['score']) for row in rows] == scores
    assert b'\r\n' not in path.read_bytes()


@pytest.mark.parametrize('replacement', [
    {'train_indices': [0, 0, 4]}, {'test_indices': [1, 1]},
    {'train_indices': [-1, 2, 4]}, {'test_indices': [1, -3]},
    {'train_indices': [0., 2., 4.]}, {'test_indices': [1., 3.]},
    {'train_indices': [[0, 2, 4]]}, {'test_indices': [[1, 3]]},
    {'test_indices': [1]}, {'train_indices': [0, 1, 4]},
    {'scores': [np.nan, .2]}, {'scores': [-.3, np.inf]},
    {'training_seconds': -1}, {'training_seconds': np.inf},
    {'prediction': [0, 2]},
])
def test_rejected_save_preserves_existing_evidence(tmp_path, replacement):
    path = tmp_path/'record.json'
    args = dict(y=[0, 1], prediction=[0, 0], scores=[-.3, .2],
        positive_label=1, training_seconds=0, train_indices=[0, 2, 4],
        test_indices=[1, 3], metadata={'seed': 7}, csv_output=True)
    save_run(path, **args)
    before = {p: p.read_bytes() for p in (path, path.with_suffix('.csv'))}
    with pytest.raises(ValueError):
        save_run(path, **{**args, **replacement})
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not path.with_suffix('.json.tmp').exists()
