"""unseen categories and extreme held-out values cannot enter fit state."""
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_validate
from experiments.registry import make_model


@pytest.mark.parametrize('name', ['paper', 'smote_svm'])
def test_categorical_encoder_scaler_and_model_are_training_only(name):
    rng = np.random.default_rng(20261461)
    X = np.empty((36, 3), dtype=object)
    X[:, 0] = rng.normal(size=36)
    X[:, 1] = np.where(np.arange(36) % 2, 'red', 'blue')
    X[:, 2] = rng.normal(size=36)
    X[0, 1] = 'single-row-category'
    y = np.r_[np.ones(8), np.zeros(28)]
    before = X.copy()
    schema = [dict(nominal=False), dict(nominal=True), dict(nominal=False)]
    params = dict(C=.7, C3=.4, Cu=.3, eps=.3, mu=1.2)
    estimator = make_model(name, schema, params, seed=47)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=91).split(X, y))
    result = cross_validate(estimator, X, y, cv=splits,
        scoring=['balanced_accuracy', 'roc_auc'], return_estimator=True, error_score='raise')
    folds_with_unseen_category = 0
    for model, (train, validation) in zip(result['estimator'], splits):
        encoder, scaler = model.named_steps['encode'], model.named_steps['scale']
        categories = encoder.named_transformers_['nominal'].categories_[0]
        np.testing.assert_array_equal(categories, np.unique(X[train, 1]))
        encoded = np.asarray(encoder.transform(X[train]), dtype=float)
        np.testing.assert_allclose(scaler.mean_, encoded.mean(axis=0), rtol=0, atol=1e-14)
        assert scaler.n_samples_seen_ == len(train)
        folds_with_unseen_category += int(bool(set(X[validation, 1])-set(categories)))
        if name == 'paper':
            standardized = scaler.transform(encoded)
            fitted = model.named_steps['model']
            np.testing.assert_array_equal(fitted.trace_['X1'], standardized[y[train] == fitted.minority_])
            np.testing.assert_array_equal(fitted.trace_['X2'], standardized[y[train] == fitted.majority_])
        # Prediction transforms unknown categories; it must not refit them.
        saved_mean, saved_categories = scaler.mean_.copy(), categories.copy()
        assert np.isfinite(model.decision_function(X[validation])).all()
        np.testing.assert_array_equal(scaler.mean_, saved_mean)
        np.testing.assert_array_equal(encoder.named_transformers_['nominal'].categories_[0], saved_categories)
    assert folds_with_unseen_category == 1
    # Outer held-out data are deliberately far outside both learned domains.
    outer = np.empty((7, 3), dtype=object)
    outer[:, 0], outer[:, 1], outer[:, 2] = 1e6, 'outer-only-category', -1e6
    fitted = estimator.fit(X, y)
    encoder, scaler = fitted.named_steps['encode'], fitted.named_steps['scale']
    categories = encoder.named_transformers_['nominal'].categories_[0].copy()
    means = scaler.mean_.copy()
    assert 'outer-only-category' not in categories
    assert np.isfinite(fitted.decision_function(outer)).all()
    np.testing.assert_array_equal(encoder.named_transformers_['nominal'].categories_[0], categories)
    np.testing.assert_array_equal(scaler.mean_, means)
    np.testing.assert_array_equal(X, before)
