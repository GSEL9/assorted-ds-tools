# -*- coding: utf-8 -*-
#
# model_selection.py
#
# This module is part of dskit.
#

"""
Tools for comparing algorithms and hyperparameter optimization.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from dstools.utils import score_stats
from dstools.prep import train_test_scaling


def compare_estimators(X, y, models_and_params, scoring, test_size=0.3, folds=10):
    """Compares estimators through a nested k-fold stratified cross-validation
    scheme across random states.

    Args:
        X (array-like): An (n x m) array of feature samples.
        y (array-like): An (n x 1) array of target samples.
        models_and_params (tuple): A nested iterable of
            (`model`, `grid parameters`) where `model` referes to the <>
            learning algorithm and `grid parameters` represents a <dict> of
            hyperparameter name and values in iterable as key-value pairs.
        test_size (float): The fraction of data used in validation.
        folds (int): The number of folds to generate in cross validation.
        scoring (): The estimator evaluation scoring metric.

    Returns:
        (dict): The training and test scores of each estimator averaged across
            each random state.

    """

    results = {}
    for model_name, (model, param_grid) in models_and_params.items():

        results[model_name] = {'train_scores': [], 'test_scores': []}
        # Address model stochasticity by eval across multiple random states.
        for random_state in range(2):

            if isinstance(model, Pipeline):
                estimator = model
            else:
                estimator = model(random_state=random_state)
            # Collect training and test scores from nested cross-validation.
            train_scores, test_scores = nested_cross_val(
                X, y, test_size, random_state, estimator, param_grid, scoring, folds
            )
            results[model_name]['train_scores'].append(np.mean(train_scores))
            results[model_name]['test_scores'].append(np.mean(test_scores))

        # Print model training and test performance.
        model_performance_report(model_name, train_scores, test_scores)

    return results


def nested_cross_val(*args):
    """Perform nested cross validation of estimator performance.

    Args:
        X (array-like): An (n x m) array of feature samples.
        y (array-like): An (n x 1) array of target samples.
        estimator (class): The learning algorithm.
        params (dict): The hyperparameter grid with parameter name and iterable
            parameter values as key-value pairs.
        random_state (int): The random number generator intiator.
        test_size (float): The fraction of data used in validation.
        folds (int): The number of folds to generate in cross validation.
        scoring ():

    Returns:
        (tuple): Cross validated training and test scores.

    """

    # Construct training and test splits including scaling of feature data.
    X_train_std, X_test_std, y_train, y_test = train_test_scaling(
        args[0], args[1], test_size=args[2], random_state=args[3]
    )
    # Perform cross-validated hyperparameter search.
    cv_grid = GridSearchCV(
        estimator=args[4], param_grid=args[5], scoring=args[6], cv=args[7]
    )
    cv_grid.fit(X_train_std, y_train)
    # Array of scores of the estimator for each run of the cross validation.
    cv_train_score = cross_val_score(cv_grid, X_train_std, y_train)
    cv_test_score = cross_val_score(cv_grid, X_test_std, y_test)

    return cv_train_score, cv_test_score


def model_performance_report(name, train_scores, test_scores):
    """Prints a model performance report including training and test scores,
    and the difference between the training and test scores."""

    print('Model performance report', '\n{}'.format('-' * 25))
    print('Name: {}\nTraining scores: {} +/- {}\nTest scores: {} +/- {}'
          ''.format(name,
                    np.round(np.mean(train_scores), decimals=3),
                    np.round(np.std(train_scores), decimals=3),
                    np.round(np.mean(test_scores), decimals=3),
                    np.round(np.std(test_scores), decimals=3)))

    print('Train-test difference: {}\n'
          ''.format(np.mean(train_scores) - np.mean(test_scores))
    )


def report_best_model(results, criteria='variance'):
    """Determines the optimal model by evaluating the model performance results
    according to a specified criteria.

    Args:
        results ():
        criteria (str, {bias, variance}): Decision rule for model performance comparison.
            Is `variance` criteria: Selects the model corresponding to the minimum difference
            between the training and test scores. If `bias` criteria: Selects the model
            corresponding to the maximum test score.

    """

    if criteria == 'bias':
        init_score = -np.float('inf')
    elif criteria == 'variance':
        init_score = np.float('inf')
    else:
        raise ValueError('Invalid evaluation criteria: `{}`'.format(criteria))

    best_model, best_score = None, init_score
    for model_name, scores in results.items():

        keys, (train, test) = list(scores.keys()), list(scores.values())

        if criteria == 'bias':
            score = np.max(test)
            if score > best_score:
                best_model, best_score = model_name, score
            else:
                continue

        elif criteria == 'variance':
            score = np.min(np.squeeze(train) - np.squeeze(test))
            if score < best_score:
                best_model, best_score = model_name, score
            else:
                continue

    print('Best model report', '\n{}'.format('-' * 20))
    print('Name: {}\nCriteria: {}\nBest scores: {}'.format(best_model,
                                                           criteria,
                                                           best_score))

    return None


def parameter_grid(grid_specs, pipeline=False):
    """Transforms a set of grid search specifications into a pipeline compatible
    parameter grid.

    Args:
        grid_specs (tuple): A nested iterable of (`model`, `grid parameters`)
            where `model` referes to the learning algorithm and
            `grid parameters` represents a <dict> of hyperparameter name and
            values in iterable as key-value pairs.

    Returns:
        (dict): The formatted grid search package containing the learning
            algorithms and correpsonding hyperparameters.

    """

    models_and_parameters = {}
    for model, params in grid_specs:

        if pipeline:
            model_name = str(model.steps[-1][0])
        else:
            model_name = str(model.__name__).lower()

        models_and_parameters[model_name] = (model, {})
        for key, value in params.items():

            if pipeline:
                param_name = ('__').join((model_name, str(key)))
            else:
                param_name = str(key)

            models_and_parameters[model_name][1][param_name] = value

    return models_and_parameters


if __name__ == '__main__':
    # Demo run:

    pass
