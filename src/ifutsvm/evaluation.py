"""Minority-positive evaluation and lossless per-run records.

The caller supplies the positive label from training data. Never infer it from
the evaluation labels: a test split need not have the training class balance.
"""
import json
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def minority_label(y):
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise ValueError('exactly two training classes are required')
    return labels[np.argmin(counts)]


def minority_scores(estimator, X, positive_label):
    """Convert sklearn's classes_[1]-positive score to a training-defined label."""
    labels = estimator.classes_
    if len(labels) != 2 or positive_label not in labels:
        raise ValueError('positive label must be one of two fitted classes')
    score = np.asarray(estimator.decision_function(X), float)
    if score.shape != (len(X),):
        raise ValueError('binary decision_function must return one score per row')
    return score if positive_label == labels[1] else -score


def binary_metrics(y, prediction, scores, positive_label, training_seconds):
    """All course/experiment metrics; undefined ROC is explicitly null.

    Precision/recall/F1 use zero for zero denominators. The confusion matrix is
    ordered [negative, positive]. Scores must already favour positive_label.
    """
    y, prediction, scores = np.asarray(y), np.asarray(prediction), np.asarray(scores, float)
    if y.ndim != 1 or not len(y) or prediction.shape != y.shape or scores.shape != y.shape:
        raise ValueError('nonempty matching one-dimensional arrays are required')
    if not np.isfinite(scores).all() or not np.isfinite(training_seconds) or training_seconds < 0:
        raise ValueError('scores and nonnegative training time must be finite')
    if len(set(y.tolist()) | set(prediction.tolist()) | {positive_label}) > 2:
        raise ValueError('evaluation requires binary labels and valid predictions')
    truth, predicted = y == positive_label, prediction == positive_label
    cm = confusion_matrix(truth, predicted, labels=[False, True])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    specificity = tn/(tn+fp) if tn+fp else 0.0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    if truth.any() and (~truth).any():
        auc = float(roc_auc_score(truth, scores))
        fpr, tpr, thresholds = roc_curve(truth, scores)
        # sklearn's initial +inf threshold is represented as null in strict JSON.
        roc = dict(fpr=fpr.tolist(), tpr=tpr.tolist(),
                   thresholds=[float(v) if np.isfinite(v) else None for v in thresholds])
    else:
        auc, roc = None, None
    return dict(accuracy=(tp+tn)/len(y), precision=precision, recall=recall,
                f1=f1, roc_auc=auc, gmean=float(np.sqrt(recall*specificity)),
                balanced_accuracy=(recall+specificity)/2, specificity=specificity,
                training_seconds=float(training_seconds), confusion_matrix=cm.tolist(), roc=roc)


def save_run(path, *, y, prediction, scores, positive_label, training_seconds,
             train_indices, test_indices, metadata, csv_output=False):
    """Persist sufficient predictions, indices and provenance to recompute metrics."""
    path = Path(path)
    train_indices, test_indices = np.asarray(train_indices), np.asarray(test_indices)
    for indices in (train_indices,test_indices):
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or (indices<0).any() or len(np.unique(indices))!=len(indices):
            raise ValueError('split indices must be unique nonnegative integers')
    if np.intersect1d(train_indices,test_indices).size:
        raise ValueError('training and test indices overlap')
    if len(test_indices) != len(y):
        raise ValueError('test indices and labels differ in length')
    result = binary_metrics(y,prediction,scores,positive_label,training_seconds)
    result['positive_label'] = positive_label.item() if isinstance(positive_label,np.generic) else positive_label
    result['metadata'] = metadata
    path.parent.mkdir(parents=True,exist_ok=True)
    # Labels are encoded in strict JSON, avoiding pickle for object/string data.
    result['y'] = np.asarray(y).tolist()
    result['prediction'] = np.asarray(prediction).tolist()
    result['scores'] = np.asarray(scores).tolist()
    result['train_indices'] = train_indices.tolist()
    result['test_indices'] = test_indices.tolist()
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    temporary.replace(path)
    if csv_output:
        csv_path=path.with_suffix('.csv')
        with csv_path.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream,lineterminator='\n')
            writer.writerow(['row_index','actual','prediction','score'])
            writer.writerows(zip(result['test_indices'],result['y'],result['prediction'],result['scores']))
    return result
