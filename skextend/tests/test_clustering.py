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
__status__ = 'Operational'


import pytest

from .. import clustering

from unittest.mock import patch

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class TestFuzzyCMeans:

    testing_model = clustering.FuzzyCMeans
    random_state = 0

    THRESH = 1e-10

    test_epochs = mocks.MockEpochs(201)

    @pytest.fixture
    def test_dataset(self):
        """Generates training and test data."""

        iris = load_iris()

        X = iris.data
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )

        return X_train, X_test, y_train, y_test

    @mock.patch('skxtend.clustering.FuzzyCMeans.memberships')
    def assign_test_memberships(self, mock_memberships, shape):
        """Assigns random class membership values to each data sample in
        algorithm."""

        mock_memberships.return_value = np.random.random(shape))

        mock_memberships.assert_called()

    def test_num_epochs(self, test_dataset):

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=3, random_state=self.random_state)
        model.fit(X_train)

        assert model.epochs == self.test_epochs()

    def test_ini_centroids(self):
        """Test generation of initial centroids."""

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=3, max_iter=-1,
                                   random_state=self.random_state)

        assert model.centroids is None

        model.fit(X_train)

        assert model.centroids is not None

    @patch('skxtend.clustering.FuzzyCMeans.memberships')
    def test_centroids(test_dataset, mock_memberships):
        """Test update of centroids."""

        np.random.seed(self.random_state)

        X_train, _, _, _ = test_dataset

        model = FuzzyCMeans(n_clusters=3, random_state=0)
        model.fit(X_train)

        ini_centroids = model.centroids

        random_memberships = np.random.random(np.shape(model.memberships))
        mock_memberships.return_value = random_memberships

        FuzzyCMeans.memberships = mock_memberships()

        new_centroids = model.update_centroids(X_train)

        assert not np.array_equal(ini_centroids, new_centroids)

    def test_predict(self, test_dataset):
        """Test index corr to highest prob of membership is selected as class
        label."""

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=3, random_state=self.random_state)
        model.fit(X_train)

        y_pred = model.predict(X_train)

        targets = np.argmax(model.memberships, axis=1)

        assert np.array_equal(targets, y_pred)

    def test_accuracy_score(self, test_dataset):
        """Test accuracy score of predictions from test data."""

        X_train, X_test, _, y_test = test_dataset

        model = self.testing_model(n_clusters=3, random_state=self.random_state)
        model.fit(X_train)

        score = accuracy_score(y_test, model.predict(X_test))

        assert score == pytest.approx(0.5, rel=self.THRESH)

    def test_fit_predict(self, test_dataset):
        """Test model is not previously fitted and index corr to highest prob of
        membership is selected as class label."""

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=3, random_state=self.random_state)

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

        model = self.testing_model(n_clusters=3, random_state=self.random_state)
        model.fit(X_train)

        centroids, memberships = model.centroids, model.memberships

        loss = 0
        for num in range(centroids.shape[0]):
            cum_dists = np.sum(np.power(X_train - centroids[num, :], 2), axis=1)
            loss += np.dot(memberships[:, num].T, cum_dists)

        assert model.inertia == pytest.approx(loss, rel=self.THRESH)
        assert model.inertia == pytest.approx(168.992424032, rel=self.THRESH)

    def test_membership_probs(self, test_dataset):
        """Tests all sample membership values sum to one."""

        X_train, _, _, _ = test_dataset

        model = self.testing_model(n_clusters=3, random_state=self.random_state)
        model.fit(X_train)

        for membership in model.memberships:
            assert np.sum(membership) == pytest.approx(1.0, rel=self.THRESH)
