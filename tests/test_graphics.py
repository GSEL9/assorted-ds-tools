# -*- coding: utf-8 -*-
#
# test_graphics.py
#
# This module is part of skxtend.
#

"""
Testing of skxtend graphics tools.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


import pytest

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples

from .. import clustering
from .. import graphics


class TestGraphics:
    """Testing plotting functions."""

    RANDOM_STATE = 0

    TOL = 1e-04
    MAX_ITER = 300
    NUM_CLUSTERS = 3

    @pytest.fixture
    def blob_data(self, n_samples=150, n_features=2, centers=4, std=0.5):

        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=std,
            shuffle=True,
            random_state=self.RANDOM_STATE
        )

        return X, y

    @pytest.fixture
    def fuzzy_c_means(self):
        """Instantiates a fuzzy c-means clustering estimator."""

        estimator = clustering.FuzzyCMeans(n_clusters=self.NUM_CLUSTERS,
                                           max_iter=self.MAX_ITER,
                                           tol=self.TOL,
                                           random_state=self.RANDOM_STATE)

        return estimator


    def test_silhouette_plot(self, blob_data, fuzzy_c_means):
        """Test call to silhouette plot function."""

        X, _ = blob_data
        estimator = fuzzy_c_means
        tool = graphics.silhouette_plot

        y_pred = estimator.fit_predict(X)

        tool(X, y_pred)
