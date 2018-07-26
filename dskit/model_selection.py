# -*- coding: utf-8 -*-
#
# model_selection.py
#
# This module is part of dskit.
#

"""
Tools for algorithm and hyperparameter selection.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from dskit.prep import train_test_scaling


def compare_models(X, y, grid_specs, test_size=0.3, folds=10, scoring=None):
    """Compares estimators through a nested k-fold stratified cross-validation
    scheme across random states.

    Args:
        X (array-like): An (n x m) array of feature samples.
        y (array-like): An (n x 1) array of target samples.
        grid_specs (tuple): A nested iterable of (`model`, `grid parameters`)
            where `model` referes to the learning algorithm and
            `grid parameters` represents a <dict> of hyperparameter name and
            values in iterable as key-value pairs.
        test_size (float): The fraction of data used in validation.
        folds (int):
        scoring (): The estimator evaluation scoring metric.

    Returns:
        (dict): The training and test scores of each estimator.

    """

    # Format container of models and hyperparameter specifications.
    grid = parameter_grid(grid_specs)

    results = {}
    for model_name, (model, params) in grid.items():

        # Address model stochasticity by eval across multiple random states.
        results[model_name] = {'train_scores': [], 'test_scores': []}
        for random_state in range(10):

            # Collect training and test scores from nested cross-validation.
            grid, train_score, test_score = _eval_estimator(
                X, y, model, params, random_state, test_size, folds, scoring
            )
            train_scores.extend(np.mean(train_score))
            test_scores.extend(np.mean(test_score))

        results[model_name]['train_scores'] = train_scores
        results[model_name]['test_scores'] = test_scores

        # Print model training and test performance.
        model_performance_report(model_name, train_scores, test_scores)

    return results


def _eval_estimator(*args):
    # Perform nested cross-validation of estimator performance.
    # Args: X, y, model, params, random_state, test_size, folds, scoring
    # Returns: grid, training and test scores

    # Construct training and test splits including scaling of feature data.
    X_train_std, X_test_std, y_train, y_test = train_test_scaling(
        args[0], args[1], args[3], args[2]
    )
    # Perform cross-validated hyperparameter search.
    grid = GridSearchCV(
        estimator=args[4], param_grid=args[5], scoring=args[7],
        cv=args[6]
    )
    grid.fit(X_train_std, y_train)
    # Array of scores of the estimator for each run of the cross validation.
    train_score = cross_val_score(grid, X_train_std, y_train)
    test_score = cross_val_score(grid, X_test_std, y_test)

    return grid, train_score, test_score


def model_performance_report(name, train_scores, test_scores):
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


def report_best_model(results):
    """Determines the model corresponding to the maximum average test score."""

    max_score, best_model = 0, None
    for model, (_, test_scores) in results.items():

        score = np.mean(test_scores)
        if score > max_score:
            max_score = score
            best_model = model

    #opt_model, avg_score = max(
    #    results.items(), key=(lambda name_avgscore: name_avgscore[1])
    #)
    print('Best model: `{}`\nAverage score: {}'.format(best_model, best_score))


def parameter_grid(grid_specs):
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

        model_name = str(model.__name__).lower()
        models_and_parameters[model_name] = (model, {})

        for key, value in params.items():

            param_name = ('__').join((model_name, key))
            models_and_parameters[model_name][1][param_name] = value


if __name__ == '__main__':
    # Demo run:

    pass
