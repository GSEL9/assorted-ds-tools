# -*- coding: utf-8 -*-
#
# model_selection.py
#
# This module is part of skxtend.
#

"""
Tools for algorithm selection tools.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


from skxtend.prep import train_test_scaling

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


def model_selection(X, y, estimators, test_size=0.3, folds=10, verbose=True):
    """Compares estimators through a nested k-fold stratified cross-validation
    scheme across random states."""

    results = {}
    for name, (estimator, params) in estimators.items():
        # Address model stochasticity across multiple random states.
        train_scores, test_scores = [], []
        for random_state in range(10):
            # Collect training and test scores from nested cross-validation.
            grid, train_score, test_score = _eval_estimator(
                X, y, estimator, params, random_state, test_size, folds
            )
            train_scores.append(train_score)
            test_scores.append(test_score)

        results[name] = np.mean(test_scores)

        if verbose:
            performance_report(name, train_scores, test_scores)

    if verbose:
        best_model(results)

    return results


def _eval_estimator(X, y, estimator, params, random_state, test_size, folds):
    # Construct a compression pipline (with PCA), standardize training and test
    # data, and perform nested cross-validation of estimator performance.

    pipe = make_pipeline(
            PCA(n_components=0.98, random_state=random_state),
            estimator(random_state=random_state)
    )
    # Construct training and test splits including scaling of feature data.
    X_train_std, X_test_std, y_train, y_test = train_test_scaling(
        X, y, test_size, random_state
    )
    # NB: Evaluated by negative mean squared error.
    grid = GridSearchCV(
        estimator=pipe, param_grid=params, scoring='neg_mean_squared_error',
        cv=folds
    )
    grid.fit(X_train_std, y_train)
    # Array of scores of the estimator for each run of the cross validation.
    train_score = cross_val_score(grid, X_train_std, y_train)
    test_score = cross_val_score(grid, X_test_std, y_test)

    return grid, -train_score, -test_score


def performance_report(name, train_scores, test_scores):
    """Prints a model performance report including training and test scores,
    and the difference between the training and test scores."""

    print('Model: {}\nTrain scores: {} +/- {}\nTest scores: {} +/- {}'
          ''.format(name,
                    np.round(np.mean(train_scores), decimals=3),
                    np.round(np.std(train_scores), decimals=3),
                    np.round(np.mean(test_scores), decimals=3),
                    np.round(np.std(test_scores), decimals=3)))

    print('Train-test deviation: {}\n'
          ''.format(np.mean(train_scores) - np.mean(test_scores))
    )


def best_model(results):
    """Prints the label of the maximum score model."""

        opt_model, avg_score = max(
        results.items(), key=(lambda name_avgscore: name_avgscore[1])
    )
    print('Best model: `{}`\nAverage score: {}'.format(opt_model, avg_score))


if __name__ = '__main__':
    # NOTE: Demo run.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet

    models_and_parameters = {
        'elnet': (
            ElasticNet, {
                 'elasticnet__alpha': [1e-5, 1e-3, 1, 1e2],
                 'elasticnet__l1_ratio': [0.3, 0.5, 0.7, 0.9]
            }
        ),
        'rand_forest': (
            RandomForestRegressor, {
                'randomforestregressor__max_depth': [5, 10, 50, 100, 200, 500]
            }
        )
    }
