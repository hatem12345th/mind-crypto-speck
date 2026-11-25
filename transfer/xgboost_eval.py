"""
XGBoost training, hyperparameter tuning, and evaluation module for TransferAnalyst.
"""

from typing import Dict, Any, Optional
import numpy as np
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from models.resnet_distinguisher import compute_metrics


# Appendix D (Table 8) search space adapted for scikit-learn RandomizedSearchCV
APPENDIX_D_PARAM_DISTRIBUTIONS = {
    'reg_lambda': loguniform(0.001, 10.0),
    'reg_alpha': loguniform(0.001, 10.0),
    'max_depth': randint(3, 10),           # [3 .. 9]
    'n_estimators': randint(50, 1001),     # [50 .. 1000]
    'learning_rate': loguniform(0.01, 1.0),
    'subsample': uniform(0.5, 0.5),        # [0.5 .. 1.0]
    'colsample_bytree': uniform(0.5, 0.5), # [0.5 .. 1.0]
    'gamma': loguniform(1e-8, 1.0),
    'min_child_weight': loguniform(1.0, 10.0),
}


def _default_xgb_params() -> Dict[str, Any]:
    """Return baseline XGBoost hyperparameters."""
    return {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
    }


def _evaluate_fitted_xgb(
    clf: xgb.XGBClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run inference with a fitted XGBClassifier and compute MIND-Crypt metrics."""
    preds = clf.predict(x_test)
    metrics = compute_metrics(y_test, preds)
    if verbose:
        print(
            f"XGBoost Test Accuracy: {metrics['accuracy']:.4f} | "
            f"TPR: {metrics['tpr']:.4f} | TNR: {metrics['tnr']:.4f}"
        )
    return {
        'metrics': metrics,
        'model': clf,
        'preds': preds,
    }


def tune_and_eval_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_iter: int = 12,
    cv: int = 3,
    param_distributions: Optional[Dict[str, Any]] = None,
    base_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV over the Appendix D
    search space and evaluate the best estimator on x_test.
    """
    params = _default_xgb_params()
    params['random_state'] = random_state
    if base_params is not None:
        params.update(base_params)
    # Avoid thread oversubscription between CV workers and XGBoost trees
    params['n_jobs'] = 1

    base_clf = xgb.XGBClassifier(**params)
    search_space = param_distributions or APPENDIX_D_PARAM_DISTRIBUTIONS

    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=1 if verbose else 0,
    )
    search.fit(x_train, y_train)

    result = _evaluate_fitted_xgb(search.best_estimator_, x_test, y_test, verbose=verbose)
    result['best_params'] = dict(search.best_params_)
    result['cv_best_score'] = float(search.best_score_)
    return result


def train_and_eval_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    xgb_params: Optional[Dict[str, Any]] = None,
    tune: bool = False,
    tune_n_iter: int = 12,
    tune_cv: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Trains an XGBoost classifier on x_train and evaluates on x_test.

    When tune=True, delegates to tune_and_eval_xgboost using Appendix D search space.
    """
    if tune:
        return tune_and_eval_xgboost(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            n_iter=tune_n_iter,
            cv=tune_cv,
            base_params=xgb_params,
            verbose=verbose,
        )

    params = _default_xgb_params()
    if xgb_params is not None:
        params.update(xgb_params)

    clf = xgb.XGBClassifier(**params)
    clf.fit(x_train, y_train, verbose=verbose)
    return _evaluate_fitted_xgb(clf, x_test, y_test, verbose=verbose)
