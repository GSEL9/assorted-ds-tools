# -*- coding: utf-8 -*-
#
# ensemble.py
#
# This module is part of dstools.
#

"""
Tools for combining base estimators to build a learning algorithm with
improved robustness over a single estimator.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import numpy as np

from dstools.utils import BaseLearner
from dstools.prep import train_test_scaling


class Stacking:
    """Stacking is a way of combining multiple models, that introduces the
    concept of a meta learner.

    Enables combining models of different types.

    Algorithm:
        1. Split the training set into sub training and test sets.
        2. Train base learners on the sub training set.
        3. Generate predictions from the base learners on sub test set.
        4. Consider the generated base learner perdictions as training data,
           and use the ground truth samples as reference for the end learner

    """

    def __init__(self, learners, random_state=None):

        self.base_learners = learners[:-1]
        self.end_learner = learners[-1]
        self.random_state = random_state

    @property
    def base_learners(self):

        return self._base_learners

    @base_learners.setter
    def base_learners(self, value):

        if isinstance(value, (list, tuple, np.ndarray)):
            self._base_learners = [BaseLearner(learner) for learner in value]
        else:
            self._base_learners = BaseLearner(value)

    @property
    def end_learner(self):

        return self._end_learner

    @end_learner.setter
    def end_learner(self, value):

        # TODO: Type checking.
        self._end_learner = value

    def fit(self, X, y, test_size=0.2, n_folds=10, scale=True):
        """Train an end learner from a stack of base learners."""

        # Train base learners on the sub training set and generate predictions
        # from the base learners on sub test set.
        stack_X_train, stack_X_test = _gen_stack_feature_data(
            X, y, test_size, n_folds, scale=scale
        )
        # Train end learner on aggregated base learner predictions.
        self.end_learner.fit(stack_X_train, y_train)

    def _gen_train_subsets(self, X, y, test_size, scale):
        # Divide training data into stratified training and test sub splits.

        if scale:
            X_train, X_test, y_train, y_test = train_test_scaling(
                X, y, test_size, self.random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

        return X_train, X_test, y_train, y_test

    def _gen_stack_feature_data(self, X, y, test_size, n_folds, scale):
        # Aggregate predictions from base learners.

        X_train, X_test, y_train, y_test = self._gen_train_subsets(
            X, y, test_size, scale
        )
        # Collect base learner predictions as new feature data.
        train_preds, test_preds = [], []
        for learner in self.base_learners:
            estimator = learner(random_state=self.random_state)
            train_preds.append(
                cross_val_predict(estimator, X_train, y_train, cv=n_folds)
            )
            test_preds.append(
                cross_val_predict(estimator, X_test, cv=n_folds)
            )
        stack_X_train = np.concatenate(train_preds, axis=1).astype(float)
        stack_X_test = np.concatenate(test_preds, axis=1).astype(float)

        return stack_X_train, stack_X_test

    def predict(self, X):

        return self.end_learner.predict(X)

    # TEMP:
    def _gen_base_predictions(self, clf, Xtrain, ytrain, Xtest, n_folds):

        ntrain, ntest = Xtrain.shape[0], Xtest.shape[0]

        kfold_test_skf = np.empty((n_folds, ntest))
        kfold_train, kfold_test = np.zeros((ntrain,)), np.zeros((ntest,))

        splits = StratifiedKFold(n_splits=n_folds).split(Xtrain, ytrain)

        for idx, (train_idx, test_idx) in enumerate(splits):
            X_test_sub = X_train[test_idx]
            X_train_sub, y_train_sub = X_train[train_idx], y_train[train_idx]

            clf.fit(X_train_sub, y_train_sub)
            kfold_train[test_idx] = clf.predict(X_test_sub)
            kfold_test_skf[idx, :] = clf.predict(X_test)
        kfold_test[:] = kfold_test_skf.mean(axis=0)

        return kfold_train.reshape(-1, 1), kfold_test.reshape(-1, 1)
