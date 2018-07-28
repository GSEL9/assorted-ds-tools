# -*- coding: utf-8 -*-
#
# clustering.py
#
# This module is part of skxtend.
#

"""
Various scikit-learn compatible clustering algorithms.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


import numpy as np


class FuzzyCMeans:
    """The fuzzy c-means clustering algorithm.

    Args:
        n_clusters (int): The number of clusters to form as well as the number
                          of centroids to generate.
        fuzz (int, float): Parameter regulating the degree of fuzziness.
        max_iter (int): Maximum number of iterations of the fuzzy c-means
                        algorithm for a single run.
        tol (int, float): Relative tolerance with regards to inertia to declare
                          convergence.
        random_state (int): The seed used by the random number generator.

    """

    def __init__(self, n_clusters, fuzz=2, max_iter=200, tol=1e-04,
                 random_state=None):

        self.n_clusters = n_clusters
        self.fuzz = fuzz
        self.max_iter = max_iter
        self.tol = 1.0 / tol
        self.random_state = random_state

        # NOTE: Variables set with instance
        self._n_epochs = None
        self._inertia = None
        self._centroids = None
        self._memberships = None

    @property
    def epochs(self):
        """Returns the number of iterations necessary for the algorithm to
        converge according to the given tolerance or abort according to the
        number of max iterations."""

        return self._n_epochs

    @property
    def centroids(self):
        """Returns estimated centroids."""

        return self._centroids

    @property
    def memberships(self):
        """Returns class membership probabilities for each input sample."""

        return self._memberships

    @property
    def inertia(self):
        """Returns global inertia as accumulated within-cluster sum-of-squares
        of fitted model."""

        return self._inertia

    def fit(self, X):
        """Estimate the probabilities of cluster membership for each input data
        point.

        Args:
            X (array like): Feature matrix as (samples x features).

        """

        # Initial centroids.
        self._centroids = self._gen_ini_centroids(X)
        # Initial memberships.
        self._memberships = self.eval_memberships(X)

        # Maximum inertia change for convergence
        inertia, p_inertia = self.tol, -self.tol

        self._n_epochs = 0
        while abs(inertia - p_inertia) > self.tol and self._n_epochs <= self.max_iter:

            self._centroids = self.update_centroids(X)
            self._memberships = self.eval_memberships(X)
            self._inertia = self.eval_objective(X)

            self._n_epochs += 1

        return self

    def predict(self, X):
        """Predict the closest cluster of each input data point.

        Args:
            X (array like): Feature matrix as (samples x features).

        Returns:
            np.ndarray: Predicted cluster labels of each input data point.

        """

        memberships = self.eval_memberships(X)

        return np.argmax(memberships, axis=1)

    def fit_predict(self, X):
        """Compute cluster centers and predict the closest cluster of each input
        data point.

        Args:
            X (array like): Feature matrix as (samples x features).

        Returns:
            np.ndarray: Predicted cluster labels of each input data point.

        """

        self.fit(X)

        return self.predict(X)

    def _gen_ini_centroids(self, X):
        # Generate and return numpy.ndarray of randomly selected centroids.

        rand_gen = np.random.RandomState(self.random_state)

        return rand_gen.random_sample(size=(self.n_clusters, X.shape[1]))

    def update_centroids(self, X):
        """Compute centroids of each cluster.

        Args:
            X (array like): Feature matrix as (samples x features).

        Returns:
            numpy.ndarray: Centroid coordinate values.

        """

        numerator = np.dot(self.memberships.T, X)
        denominator = np.sum(self.memberships, axis=0)

        return np.divide(numerator, denominator[:, np.newaxis])

    def eval_memberships(self, X):
        """Estimate the probabilities of cluster membership for each input data
        point.

        Args:
            X (array like): Feature matrix as (samples x features).

        Returns:
            numpy.ndarray: Membership probabilities of each data point as
                           (samples x centroids).

        """

        num_samples, num_centroids = X.shape[0], self.centroids.shape[0]

        memberships = np.zeros((num_samples, num_centroids))
        dists = np.zeros((num_samples, num_centroids))

        # Computes distances between data points and centroids.
        for num in range(num_centroids):
            # Sqrt of sum of power due to dealing with small numbers.
            sqrd_eucl_dist = np.power(X - self.centroids[num, :], 2)
            dists[:, num] = np.sqrt(np.sum(sqrd_eucl_dist, axis=1))

        # Computes probabilities of cluster memberships.
        for num in range(num_centroids):
            rel_dists = np.divide(dists, dists[:, num][:, np.newaxis])
            memberships += np.power(rel_dists, (2 / (self.fuzz - 1)))

        return np.power(memberships, -1)

    def eval_objective(self, X):
        """Computes global cluster inertia.

        Args:
            X (array like): Feature matrix as (samples x features).

        Returns:
            float: The global inertia value of all clusters.

        """

        loss = 0
        for num in range(self._centroids.shape[0]):
            cum_dists = np.sum(np.power(X - self.centroids[num, :], 2), axis=1)
            loss += np.dot(self.memberships[:, num].T, cum_dists)

        return loss
