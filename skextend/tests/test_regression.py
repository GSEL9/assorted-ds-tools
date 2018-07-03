# -*- coding: utf-8 -*-
#
# test_regression.py
#
# This module is part of skxtend.
#

"""
Testing of skxtend regression algorithms.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


import pytest

import numpy as np

from .. import regression
from scipy.io import loadmatcd


class TestTikhonovCV:
    """Tests for TikhonovCV regression model."""

    testing_model = regression.TikhonovCV

    reg_coef = 1.0
    reg_mat = np.eye(3)

    THRESH = 1e-10

    @pytest.fixture
    def test_dataset(self, path='./datasets/tikhonov.mat'):
        """Generates training and test data."""

        try:
            testing_data = loadmat(path)

        except:
            raise RuntimeError('Unable to import test data from {}.'
                               ''.format(path))

        # Use only one sugar sample as target variable
        X_train, y_train = testing_data['Xtrain'], testing_data['Ytrain'][:, 0]
        X_test, y_test = testing_data['Xtest'], testing_data['Ytest'][:, 0]

        return X_train, X_test, y_train, y_test

    def test_no_intercept(self, test_dataset):
        """Tests regression coefficient estimates regression excluding
        intercept term."""

        X_train, X_test, _, _ = test_dataset

        model = self.testing_model(self.reg_mat, self.reg_coef,
                                   fit_intercept=False)
        model.fit(X_train, y_train)

        assert len(model.coef_) == X_train.shape[1]

    def test_intercept(self, test_dataset):
        """Tests regression coefficient estimates regression including
        intercept term."""

        X_train, X_test, _, _ = test_dataset

        model = self.testing_model(self.reg_mat, self.reg_coef)
        model.fit(X_train, y_train)

        assert len(model.coef_) == (X_train.shape[1] + 1)

    def test_coef_norm(self, test_dataset):
        """Tests Frobenius norm of regression coefficient estimates."""

        X_train, X_test, _, _ = test_dataset

        model = self.testing_model(self.reg_mat, self.reg_coef)
        model.fit(X_train, y_train)

        model_coef_norm = np.linalg.norm(model.coef_)

        assert model_coef_norm == pytest.approx(3.61267028102, rel=self.THRESH)

    def test_mescv(self, test_dataset):
        """Tests cross-validated mean Square Error."""

        X_train, X_test, y_train, y_test = test_dataset

        model = self.testing_model(self.reg_mat, self.reg_coef)
        model.fit(X_train, y_train)

        model.cross_val_predict(X_test, y_test)

        assert model.mescv == pytest.approx(443.279553552, rel=self.THRESH)

    def test_rescv_sum(self, test_dataset):
        """Tests sum or cross-validated predictions."""

        X_train, X_test, y_train, y_test = test_dataset

        model = self.testing_model(self.reg_mat, self.reg_coef)
        model.fit(X_train, y_train)

        model.cross_val_predict(X_test, y_test)

        assert model.rescv == pytest.approx(295.778167806, rel=self.THRESH)
