# -*- coding: utf-8 -*-
#
# test_clustering.py
#
# This module is part of skxtend.
#

"""
Testing of skxtend clustering algorithms.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Passed 07.28.2018'


import pytest

import numpy as np

from dstools import clustering

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class TestFuzzyCMeans:
    """Testing the fuzzy c-means clustering algorithm."""

    testing_model = clustering.FuzzyCMeans

    THRESH = 1e-10
    TEST_SIZE = 0.3
    RANDOM_STATE = 0

    NUM_CLUSTERS = 3

    @pytest.fixture
    def test_dataset(self):
        """Generates training and test data."""

        iris = load_iris()

        X = iris.data
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE
        )

        return X_train, X_test, y_train, y_test

    def test_num_epochs(self, test_dataset):

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=self.NUM_CLUSTERS,
                                   random_state=self.RANDOM_STATE)
        model.fit(X_train)

        assert model.epochs == 201

    def test_ini_centroids(self, test_dataset):
        """Test generation of initial centroids."""

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=self.NUM_CLUSTERS,
                                   random_state=self.RANDOM_STATE)

        assert model.centroids is None

        model.fit(X_train)

        assert model.centroids is not None

    def test_centroids(self, test_dataset, mocker):
        """Test update of centroids."""

        np.random.seed(self.RANDOM_STATE)

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=self.NUM_CLUSTERS,
                                   random_state=self.RANDOM_STATE)
        model.fit(X_train)

        ini_centroids = model.centroids

        mocker.patch('skxtend.clustering.FuzzyCMeans.memberships',
                     np.random.random(np.shape(model.memberships)))

        new_centroids = model.update_centroids(X_train)

        assert not np.array_equal(ini_centroids, new_centroids)

    def test_predict(self, test_dataset):
        """Test index corr to highest prob of membership is selected as class
        label."""

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=self.NUM_CLUSTERS,
                                   random_state=self.RANDOM_STATE)
        model.fit(X_train)

        y_pred = model.predict(X_train)

        targets = np.argmax(model.memberships, axis=1)

        assert np.array_equal(targets, y_pred)

    def test_fit_predict(self, test_dataset):
        """Test model is not previously fitted and index corr to highest prob of
        membership is selected as class label."""

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=self.NUM_CLUSTERS,
                                   random_state=self.RANDOM_STATE)

        ini_state = [model.epochs, model.memberships, model.centroids,
                     model.inertia]

        for state in ini_state:
            assert state is None

        model.fit_predict(X_train)

        fitted_state = [model.epochs, model.memberships, model.centroids,
                        model.inertia]

        for state in fitted_state:
            assert state is not None

    def test_inertia(self, test_dataset):
        """Tests model inertia as status evaluation of model objective."""

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=self.NUM_CLUSTERS,
                                   random_state=self.RANDOM_STATE)
        model.fit(X_train)

        centroids, memberships = model.centroids, model.memberships

        loss = 0
        for num in range(centroids.shape[0]):
            cum_dists = np.sum(np.power(X_train - centroids[num, :], 2), axis=1)
            loss += np.dot(memberships[:, num].T, cum_dists)

        assert model.inertia == pytest.approx(loss, rel=self.THRESH)
        assert model.inertia == pytest.approx(156.733390720795, rel=self.THRESH)

    def test_membership_probs(self, test_dataset):
        """Tests all sample membership values sum to one."""

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=self.NUM_CLUSTERS,
                                   random_state=self.RANDOM_STATE)
        model.fit(X_train)

        for membership in model.memberships:
            assert np.sum(membership) == pytest.approx(1.0, rel=self.THRESH)
