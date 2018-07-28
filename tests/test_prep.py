# -*- coding: utf-8 -*-
#
# test_prep.py
#
# This module is part of dstools.
#

"""
Testing of data pre-processing tools.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Passed {}'


import pytest

import numpy as np
import pandas as pd

from . import mocking
from pytest import approx
from dstools import prep


class TestStandardizer:

    # Arbitrary initiator to random number generator.
    SEED = 123

    THRESH = 1e-10

    @pytest.fixture
    def data(self):

        np.random.seed(self.SEED)

        num_samples, num_features = 100, 10
        return np.random.random((num_samples, num_features))

    @pytest.fixture
    def scaler(self):

        scaler = prep.Standardizer

        return scaler

    def test_default_scaler(self, scaler):

        assert isinstance(scaler(), prep.Standardizer)

    def test_custom_scaler(self, scaler):

        assert isinstance(scaler(scaler=mocking.MockScaler), prep.Standardizer)

    def test_fit(self, scaler, data):

        scaler().fit(data)

    def test_transform(self, scaler, data):

        standardizer = scaler()
        standardizer.fit(data)
        standardizer.transform(data)

    def test_mean(self, data, scaler):

        standardizer = scaler()
        standardizer.fit(data)
        data_trans = standardizer.transform(data)

        data_avg = np.mean(data_trans, axis=0)
        assert data_avg == approx(np.zeros((data_avg.shape)), rel=self.THRESH)

    def test_std(self, data, scaler):

        standardizer = scaler()
        standardizer.fit(data)
        data_trans = standardizer.transform(data)

        data_std = np.std(data_trans, axis=0)
        assert data_std == approx(np.ones((data_std.shape)), rel=self.THRESH)

    def test_fit_transform(self, data, scaler):

        # Stepwise fitting and tarnsforming of data.
        data_trans = scaler().fit(data).transform(data)
        # Direct fitting and transforming of data.
        data_direct_trans = scaler().fit_transform(data)

        assert data_direct_trans == approx(data_trans, rel=self.THRESH)


class TestTrainTestScaling:

    # Arbitrary initiator to random number generator.
    SEED = 123

    THRESH = 1e-10

    SPLIT_SIZE = 0.2

    @pytest.fixture
    def data(self):

        np.random.seed(self.SEED)

        num_samples, num_features = 100, 10
        X = np.random.random((num_samples, num_features))
        y = np.random.random((num_samples, 1))

        return X, y

    def test_mean(self, data):

        # NOTE: Test data is standardized from train data which excludes mean
        # and std test properties.

        X, y = data
        X_train, _, _, _ = prep.train_test_scaling(
            X, y, self.SPLIT_SIZE, self.SEED
        )
        train_avg = np.mean(X_train, axis=0)
        assert train_avg == approx(np.zeros((train_avg.shape)), rel=self.THRESH)

    def test_std(self, data):

        # NOTE: Test data is standardized from train data which excludes mean
        # and std test properties.

        X, y = data
        X_train, _, _, _ = prep.train_test_scaling(
            X, y, self.SPLIT_SIZE, self.SEED
        )
        train_std = np.std(X_train, axis=0)
        assert train_std == approx(np.ones((train_std.shape)), rel=self.THRESH)

    def test_split(self, data):

        X, y = data
        org_num_rows, org_num_cols = X.shape

        X_train, X_test, y_train, y_test = prep.train_test_scaling(
            X, y, self.SPLIT_SIZE, self.SEED
        )
        assert X_train.shape[1] == org_num_cols
        assert X_test.shape[1] == org_num_cols
        
        joined_X_rows = int(X_train.shape[0]) + int(X_test.shape[0])
        assert joined_X_rows == org_num_rows

        joined_y_rows = int(y_train.shape[0]) + int(y_test.shape[0])
        assert joined_y_rows == org_num_rows


class TestDiscardOutliers:

    pass
