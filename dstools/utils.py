# -*- coding: utf-8 -*-
#
# utils.py
#
# This module is part of dstools.
#

"""
The dskit utilities module.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import numpy as np


def score_stats(scores):
    # Computes mean and standard deviation of score values.

    return np.mean(scores, axis=1), np.std(scores, axis=1)


class BaseLearner:
    """An estimator wrapper."""

    # NOTE: Used in stacking.
    def __init__(self, estimator):

        self.estimator = estimator

    def train(self, X_train, y_train):

        self.estimator.fit(X_train, y_train)

    def predict(self, X):

        return self.estimator.predict(X)

    def fit(self, X, y):

        return self.estimator.fit(X, y)

    def feature_importances(self, X, y):

        print(self.estimator.fit(X, y).feature_importances_)
