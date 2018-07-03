# -*- coding: utf-8 -*-
#
# regression.py
#
# This module is part of skxtend.
#

"""
Various scikit-learn compatible regression algorithms.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array


class TikhonovCV(BaseEstimator):
    """Tikhonov regularized least-squares regression by QR-decomposition with
    leave-one-out cross-validation of prediction estimates.

    Args:
        D (array like): Regularization matrix.
        alpha (int, float): Regularization parameter.

    Kwargs:
        fit_intercept (bool): Includes an intercept coefficient to the
                              regression model parameters.
        normalize (bool): Normalizes the data matrix.

    """

    def __init__(self, reg_mat, alpha, fit_intercept=True, normalize=False):

        self.reg_mat = np.sqrt(alpha) * reg_mat
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        # NOTE: Variable set with instance.
        self.coef_ = None
        self._msecv = None
        self._rescv_ = None
        self._orthog = None
        self._upper_tri = None

    @property
    def mescv(self):
        """Returns cross-validated Mean Square Error"""

        return self._mescv

    @property
    def rescv(self):
        """Returns cross-validated model residuals."""

        return self._rescv

    @staticmethod
    def _check_X_y(X, y):
        # Checks feature matrix X and target vector y array. Converts y to
        # (n x 1) array.

        X, y = check_X_y(X, y)

        if len(np.shape(y)) < 2:
            return X, y[:, np.newaxis]
        else:
            return X, y

    @staticmethod
    def _normalize(X, y):
        # Divides feature matrix and target vector by column sums.

        X_col_sums = np.sum(X, axis=0)
        X_norm = np.divide(X, X_col_sums)

        y_col_sum = np.sum(y, axis=0)
        y_norm = y / y_col_sum

        return X_norm, y_norm

    def _stacked_matrices(self, X, y):
        # Stacks regularization and feature matrix, and expands target data.

        num_data_samples, _ = np.shape(X)
        num_reg_samples, _ = np.shape(self.reg_mat)

        reg_dummies = np.zeros((num_reg_samples, 1))

        if self.fit_intercept:
            intercept = np.ones((num_data_samples, 1))
            variables = np.hstack((intercept, X))
            regularization = np.hstack((reg_dummies, self.reg_mat))

            stacked_data = np.vstack((variables, regularization))

        else:
            stacked_data = np.vstack((X, self.reg_mat))

        stacked_target = np.vstack((y, reg_dummies))

        return stacked_data, stacked_target

    def _decompose(self, X):
        # Performs QR decomposition of feature matrix.

        return np.linalg.qr(X)

    def fit(self, X, y):
        """Estimate least-squares regression coefficients.

        Args:
            X (array like): Feature matrix as (samples x features).
            y (array like): Target vector.

        """

        X, y = self._check_X_y (X, y)

        if self.normalize:
            _data_stack, _target_stack = self._stacked_matrices(X, y)
            data_stack, target_stack = self._normalize(
                _data_stack, _target_stack
            )
        else:
            data_stack, target_stack = self._stacked_matrices(X, y)

        self._orthog, self._upper_tri = self._decompose(data_stack)

        proj = np.dot(np.transpose(self._orthog), target_stack)
        self.coef_ = np.dot(np.linalg.inv(self._upper_tri), proj)

        return self

    def predict(self, X):
        """Predict observations for each input data point.

        Args:
            X (array like): Feature matrix as (samples x features).

        """

        X = check_array(X)

        if self.fit_intercept:
            return np.dot(X, self.coef_[1:]) + self.coef_[0]

        else:
            return np.dot(X, self.coef_)

    def cross_val_predict(self, X, y):
        """Perform cross-validated prediction for each input data point.

        Args:
            X (array like): Feature matrix as (samples x features).
            y (array like): Target vector.

        """

        self.fit(X, y)

        pred = self.predict(X)
        res = y - pred[:, 0]

        diag = np.sum(np.power(self._orthog[:X.shape[0], :], 2), axis=1)
        self._rescv = res / (1 - diag)
        self._mescv = np.dot(self._rescv, self._rescv) / self._rescv.size

        return pred[:, 0] - self._rescv
