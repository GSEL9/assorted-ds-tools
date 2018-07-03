# -*- coding: utf-8 -*-

# Response Optimal Sequential alternation.


import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import euclidean_distances


class ROSA(BaseEstimator):
    """

    Reference:
        Liland, K. H., Næs, T., and Indahl, U. G. (2016),
        ROSA—a fast extension of par- tial least squares regression for
        multiblock data analy- sis, J. Chemometrics, doi: 10.1002/cem.2824

    """

    def __init__(self, n_components=None):

        self.n_components = n_components

        # NOTE: Variables set during instance.
        self._org_X = None
        self._org_y = None

        self.coef_ = None

    def fit(self, X, y):
        """

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.

        """

        A = self.n_components

        for Xi in X:
            _, _ = check_X_y(Xi, y)

        self._org_X, self._org_y = X, y

        nb = len(X)
        n, _ = np.shape(X[0])

        pk = [np.shape(X[num])[1] for num in range(nb)]

        count = np.zeros(nb)
        order = np.zeros(A)
        T = np.zeros((n, A))
        q = np.zeros(A)

        Pb = [0] * nb

        Wb = [np.zeros((x, n)) for x in pk]
        W = np.zeros((sum(pk), A))

        X_cent, y_cent = X, y#self.centering(X, y)

        inds = [np.arange(var) for var in pk]

        for i in range(1, nb):
            inds[i] += int(np.sum(pk[:i]))

        v = [0] * nb
        t = np.zeros((n, nb))
        r = np.zeros((n, nb))

        for a in range(A):

            for k in range(nb):
                v[k] = np.dot(X_cent[k].T, y_cent)
                t[:, k] = np.dot(X_cent[k], v[k])

            if a > 0:
                t = t - T[:, :a].dot(np.dot(T[:, :a].T, t))

            for k in range(nb):
                t[:, k] = t[:, k] / np.linalg.norm(t[:, k])

                offset = np.dot(np.dot(t[:, k], t[:, k].T), y_cent)
                r[:, k] = y_cent - offset

            i = np.argmin(np.sum(r ** 2))
            count[i] += 1
            order[a] = i

            T[:, a] = t[:, i]
            q[a] = np.dot(np.transpose(y_cent), T[:, a])

            y_cent = r[:, i]

            weight = Wb[i][:, :int(count[i])]

            v[i] -= weight.dot(np.dot(weight.T, v[i]))
            normalized = v[i] / np.linalg.norm(v[i])
            Wb[i][:, int(count[i])] = normalized
            W[inds[i], a] = Wb[i][:, int(count[i])]

        ## Postprocessing
        for k in range(nb):
            Pb[k] = np.dot(X_cent[k].T, T)

        PtW = np.triu(np.dot((np.concatenate(Pb, axis=0)).T, W))

        # PtW_ext = self._broadcast(PtW, W)

        Beta = np.cumsum(np.divide(W, PtW) * q, axis=1)

        #Beta = np.cumsum(W / PtW * q, axis=1)

    @staticmethod
    def centering(X, y):

        y_cent = y - np.mean(y, axis=0)
        X_cent = [Xi - np.mean(Xi, axis=0) for Xi in X]
        return X_cent, y_cent[:, np.newaxis]
    
    @staticmethod
    def _broadcast(array1, array2):

        if np.size(array2) > np.size(array1):
            largest = array2
        else:
            largest = array1

        print(largest)
        print(array1)


    def predict(self, X):
        """

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        X = check_array(X)

if __name__ == '__main__':

    X1 = np.ones((10, 4))
    X2 = np.ones((10, 2))
    X3 = np.ones((10, 3))
    X = [X1, X2, X3]

    y = np.ones(10)

    rosa = ROSA(n_components=2)
    rosa.fit(X, y)

