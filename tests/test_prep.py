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


def train_test_scaling():

    pass


class TestStandardizer:

    # Arbitrary initiator to random number generator.
    SEED = 123

    THRESH = 1e-13

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


class TestDiscardOutliers:

    pass
