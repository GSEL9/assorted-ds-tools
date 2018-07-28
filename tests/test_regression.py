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
__status__ = 'Passed 07.28.2018'


import pytest

import numpy as np
import pandas as pd

from dstools import regression


class TestTikhonovCV:
    """Testing the TikhonovCV regression algorithm."""

    num_vars = None
    testing_model = regression.TikhonovCV

    THRESH = 1e-10

    @pytest.fixture
    def test_dataset(self):
        """Generates training and test data."""

        data_path = './test_data/tikhonov_data.csv'
        target_path = './test_data/tikhonov_target.csv'

        try:
            X = pd.read_csv(data_path).values
            y = pd.read_csv(target_path).values[:, 0]
        except:
            raise RuntimeError('Unable to import test data from {}.'
                               ''.format(data_path))

        self.num_vars = int(X.shape[1])

        X_train, X_test, y_train, y_test = X[:125], X[125:], y[:125], y[125:]

        return X_train, X_test, y_train, y_test

    def setup(self):
        """Executed before each test."""

        self.reg_mat = np.eye(self.num_vars)
        self.reg_coef = 1.0

    def test_no_intercept(self, test_dataset):
        """Tests regression coefficient estimates regression excluding
        intercept term."""

        X_train, X_test, y_train, _ = test_dataset

        model = self.testing_model(self.reg_mat, self.reg_coef,
                                   fit_intercept=False)
        model.fit(X_train, y_train)

        assert len(model.coef_) == X_train.shape[1]

    def test_intercept(self, test_dataset):
        """Tests regression coefficient estimates regression including
        intercept term."""

        X_train, X_test, y_train, _ = test_dataset

        model = self.testing_model(self.reg_mat, self.reg_coef)
        model.fit(X_train, y_train)

        assert len(model.coef_) == (X_train.shape[1] + 1)

    def test_coef_norm(self, test_dataset):
        """Tests Frobenius norm of regression coefficient estimates."""

        X_train, X_test, y_train, _ = test_dataset

        model = self.testing_model(self.reg_mat, self.reg_coef)
        model.fit(X_train, y_train)

        model_coef_norm = np.linalg.norm(model.coef_)

        assert model_coef_norm == pytest.approx(43.873011993328831,
                                                rel=self.THRESH)

    def test_mescv(self, test_dataset):
        """Tests cross-validated mean Square Error."""

        X_train, X_test, y_train, y_test = test_dataset

        model = self.testing_model(self.reg_mat, self.reg_coef)
        model.fit(X_train, y_train)

        model.cross_val_predict(X_test, y_test)

        assert model.mescv == pytest.approx(90.32844292622984, rel=self.THRESH)

    def test_rescv_sum(self, test_dataset):
        """Tests sum or cross-validated predictions."""

        X_train, X_test, y_train, y_test = test_dataset

        model = self.testing_model(self.reg_mat, self.reg_coef)
        model.fit(X_train, y_train)

        model.cross_val_predict(X_test, y_test)

        sum_rescv = np.sum(model.rescv)

        assert sum_rescv == pytest.approx(0.2086234041320018, rel=self.THRESH)
