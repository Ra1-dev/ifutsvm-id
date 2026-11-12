"""Shared stratified split plans and train-only model selection.

No dataset download or benchmark defaults live here. Callers supply dataset
and grid configuration. Each candidate's inner-fold evidence is persisted.
"""
import time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from ifutsvm.evaluation import minority_label, minority_scores, save_run


def split_plan(y, protocol, seeds=(0,1,2,3,4)):
    indices=np.arange(len(y))
    if protocol=='holdout':
        for seed in seeds:
            train,test=train_test_split(indices,test_size=.3,stratify=y,random_state=seed)
            yield f'holdout-{seed}',seed,train,test
    elif protocol=='nested_cv':
        outer=StratifiedKFold(5,shuffle=True,random_state=0)
        for fold,(train,test) in enumerate(outer.split(indices,y)):
            yield f'outer-{fold}',0,train,test
    else:
        raise ValueError('protocol must be holdout or nested_cv')


def evaluate_split(estimator, grid, X, y, train, test, directory, *, seed, provenance):
    """Tune on train only; use test exactly once after candidate selection.

    Estimator should be a complete preprocessing Pipeline. Selection uses mean
    balanced accuracy; the first ParameterGrid candidate wins exact ties.
    Every inner evaluation and the final evaluation retain indices and scores.
    """
    X,y=np.asarray(X),np.asarray(y)
    train,test=np.asarray(train),np.asarray(test)
    if np.intersect1d(train,test).size:
        raise ValueError('outer train and test overlap')
    labels,counts=np.unique(y[train],return_counts=True)
    if len(labels)!=2 or min(counts)<5:
        raise ValueError('five-fold tuning requires at least five training rows in each class')
    directory=Path(directory)
    inner=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X[train],y[train]))
    start_tuning=time.perf_counter()
    candidates=[]
    for candidate,params in enumerate(ParameterGrid(grid)):
        if any(k.endswith('random_state') for k in params):
            raise ValueError('random seeds are fixed protocol inputs, not tuning parameters')
        values=[]
        for fold,(it,iv) in enumerate(inner):
            fit_indices,validation_indices=train[it],train[iv]
            model=clone(estimator).set_params(**params)
            model.set_params(**{k:seed for k in model.get_params(deep=True) if k.endswith('random_state')})
            start=time.perf_counter()
            model.fit(X[fit_indices],y[fit_indices])
            fit_seconds=time.perf_counter()-start
            positive=minority_label(y[fit_indices])
            prediction=model.predict(X[validation_indices])
            scores=minority_scores(model,X[validation_indices],positive)
            result=save_run(directory/'inner'/f'candidate-{candidate}-fold-{fold}.json',
                y=y[validation_indices],prediction=prediction,scores=scores,positive_label=positive,
                training_seconds=fit_seconds,train_indices=fit_indices,test_indices=validation_indices,
                metadata=dict(provenance=provenance,seed=seed,parameters=params,role='inner_validation'))
            values.append(result['balanced_accuracy'])
        candidates.append(dict(parameters=params,mean_balanced_accuracy=float(np.mean(values)),fold_scores=values))
    selected=int(np.argmax([r['mean_balanced_accuracy'] for r in candidates]))
    tuning_seconds=time.perf_counter()-start_tuning
    model=clone(estimator).set_params(**candidates[selected]['parameters'])
    model.set_params(**{k:seed for k in model.get_params(deep=True) if k.endswith('random_state')})
    start=time.perf_counter()
    model.fit(X[train],y[train])
    fit_seconds=time.perf_counter()-start
    positive=minority_label(y[train])
    # This is the first evaluation of the outer test data.
    prediction=model.predict(X[test])
    scores=minority_scores(model,X[test],positive)
    result=save_run(directory/'evaluation.json',y=y[test],prediction=prediction,scores=scores,
        positive_label=positive,training_seconds=fit_seconds,train_indices=train,test_indices=test,
        metadata=dict(provenance=provenance,seed=seed,parameters=candidates[selected]['parameters'],
                      selected_candidate=selected,candidates=candidates,tuning_seconds=tuning_seconds,
                      role='outer_evaluation'))
    return model,result
