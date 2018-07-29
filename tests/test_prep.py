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
__status__ = 'Passed 07.28.2018'


import pytest

import numpy as np
import pandas as pd

from . import mocking

from pytest import approx
from dstools import prep
from pandas.util.testing import assert_frame_equal


class TestFeatureScaling:

    # Arbitrary initiator to random number generator.
    SEED = 123

    THRESH = 1.e-10

    @pytest.fixture
    def data(self):

        np.random.seed(self.SEED)

        num_samples, num_features = 100, 10
        return np.random.random((num_samples, num_features))

    @pytest.fixture
    def scaler(self):

        return prep.FeatureScaling

    def test_default_scaler(self, scaler):

        assert isinstance(scaler(), prep.FeatureScaling)

    def test_custom_scaler(self, scaler):

        assert isinstance(scaler(scaler=mocking.MockScaler), prep.FeatureScaling)

    def test_fit(self, scaler, data):

        scaler().fit(data)

    def test_transform(self, scaler, data):

        scaler = scaler()
        scaler.fit(data)
        scaler.transform(data)

    def test_mean(self, data, scaler):

        scaler = scaler()
        scaler.fit(data)
        data_trans = scaler.transform(data)

        data_avg = np.mean(data_trans, axis=0)
        assert data_avg == approx(np.zeros((data_avg.shape)), rel=self.THRESH)

    def test_std(self, data, scaler):

        scaler = scaler()
        scaler.fit(data)
        data_trans = scaler.transform(data)

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

    THRESH = 1.e-10

    TEST_SIZE = 0.2

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
            X, y, self.TEST_SIZE, self.SEED
        )
        train_avg = np.mean(X_train, axis=0)
        assert train_avg == approx(np.zeros((train_avg.shape)), rel=self.THRESH)

    def test_std(self, data):

        # NOTE: Test data is standardized from train data which excludes mean
        # and std test properties.

        X, y = data
        X_train, _, _, _ = prep.train_test_scaling(
            X, y, self.TEST_SIZE, self.SEED
        )
        train_std = np.std(X_train, axis=0)
        assert train_std == approx(np.ones((train_std.shape)), rel=self.THRESH)

    def test_split_shapes(self, data):

        X, y = data
        org_num_rows, org_num_cols = X.shape

        X_train, X_test, y_train, y_test = prep.train_test_scaling(
            X, y, test_size=self.TEST_SIZE, random_state=self.SEED
        )
        assert X_train.shape[1] == org_num_cols
        assert X_test.shape[1] == org_num_cols

        joined_X_rows = int(X_train.shape[0]) + int(X_test.shape[0])
        assert joined_X_rows == org_num_rows

        joined_y_rows = int(y_train.shape[0]) + int(y_test.shape[0])
        assert joined_y_rows == org_num_rows

    def test_split_size(self, data):

        X, y = data
        org_num_rows, org_num_cols = X.shape

        X_train, X_test, y_train, y_test = prep.train_test_scaling(
            X, y, test_size=self.TEST_SIZE, random_state=self.SEED
        )
        assert int(X_test.shape[0]) / int(X.shape[0]) == self.TEST_SIZE
        assert int(X_train.shape[0]) / int(X.shape[0]) == 1 - self.TEST_SIZE


class TestClone:

    SEED = 123

    @pytest.fixture
    def data(self):

        np.random.seed(self.SEED)

        num_samples, num_features = 100, 10
        X = np.random.random((num_samples, num_features))

        return X

    @pytest.fixture
    def cloner(self):

        return prep.Clone()

    def test_init(self, cloner):

        assert isinstance(cloner, prep.Clone)

    def test_fit(self, data, cloner):

        cloner.fit(data)

    def test_transform(self, data, cloner):

        cloner.fit(data)
        cloner.transform(data)

    def test_fit_transform(self, data, cloner):

        cloner.fit_transform(data)

    def test_copying(self, data, cloner):

        copy = cloner.fit_transform(data)

        assert np.array_equal(data, copy)
        assert copy is not data


class TestFeatureImputer:

    SEED = 123

    IMPUTE_VAL = np.nan

    @pytest.fixture
    def data(self):

        np.random.seed(self.SEED)

        df = pd.DataFrame(np.random.random((10, 4)))
        df.iloc[3:5, 0] = self.IMPUTE_VAL
        df.iloc[6:8, 2] = self.IMPUTE_VAL
        df.iloc[2:4, 3] = self.IMPUTE_VAL

        return df

    @pytest.fixture
    def imputer(self):

        imputer = prep.FeatureImputer

        return imputer

    def test_default_imputer(self, imputer):

        assert isinstance(imputer(), prep.FeatureImputer)

    def test_custom_imputer(self, imputer):

        _imputer = imputer(imputer=mocking.MockImputer)
        assert isinstance(_imputer, prep.FeatureImputer)

    def test_fit(self, imputer, data):

        _imputer = imputer()
        assert _imputer.targets is None

        _imputer.fit(data)
        assert isinstance(_imputer.targets, list)

    def test_transform(self, imputer, data):

        _imputer = imputer()
        _imputer.fit(data)
        imputed_data = _imputer.transform(data)

        assert np.all(imputed_data.isnull().sum() == 0)

    def test_fit_transform(self, imputer, data):

        imputed_data_direct = imputer().fit_transform(data)

        _imputer = imputer()
        _imputer.fit(data)
        imputed_data = _imputer.transform(data)

        assert np.array_equal(imputed_data_direct, imputed_data)

    def test_impute_strategies(self, imputer, data):

        strategies = ['mean', 'median', 'most_frequent']
        for strategy in strategies:
            _imputer = imputer(method=strategy)
            _imputer.fit(data)
            imputed_data = _imputer.transform(data)

            assert isinstance(imputed_data, pd.DataFrame)


class TestRemoveOutliers:

    SEED = 123

    @pytest.fixture
    def data(self):

        np.random.seed(self.SEED)

        return pd.DataFrame(np.random.random((10, 4)))

    @pytest.fixture
    def filter(self):

        return prep.RemoveOutliers

    def test_init(self, filter):

        assert isinstance(filter(), prep.RemoveOutliers)

    def test_fit(self, data, filter):

        _filter = filter()

        assert _filter.quantiles is None
        _filter.fit(data)
        assert isinstance(_filter.quantiles, pd.DataFrame)

    def test_transform(self, data, filter):

        _filter = filter()

        _filter.fit(data)
        data_clean = _filter.transform(data)

        assert np.any(data_clean.isnull().sum())

    def test_fit_transform(self, data, filter):

        direct_trans = filter().fit_transform(data)

        _filter = filter()
        _filter.fit(data)
        trans = _filter.transform(data)

        assert_frame_equal(direct_trans, trans, check_dtype=True)


class TestFeatureEncoder:

    SEED = 123

    @pytest.fixture
    def data(self):

        data = pd.DataFrame(
            {
                'patient': [1, 1, 1, 2, 2],
                'treatment': [0, 1, 0, 1, 0],
                'score': ['strong', 'weak', 'normal', 'weak', 'strong']
            },
            columns=['patient', 'treatment', 'score']
        )

        return data

    @pytest.fixture
    def encoder(self):

        return prep.FeatureEncoder

    def test_default_encoder(self, encoder):

        assert isinstance(encoder(), prep.FeatureEncoder)

    def test_custom_encoder(self, encoder):

        _encoder = encoder(encoder=mocking.MockEncoder)
        assert isinstance(_encoder, prep.FeatureEncoder)

    def test_fit(self, data, encoder):

        _encoder = encoder()

        assert _encoder.targets is None
        _encoder.fit(data)
        assert isinstance(_encoder.targets, list)

    def test_transform(self, data, encoder):

        _encoder = encoder()

        _encoder.fit(data)
        data_trans = _encoder.transform(data)

        assert any(data_trans.dtypes != 'object')

    def test_fit_transform(self, data, encoder):

        direct_trans = encoder().fit_transform(data)

        _encoder = encoder()
        _encoder.fit(data)
        trans = _encoder.transform(data)

        assert_frame_equal(direct_trans, trans, check_dtype=True)
