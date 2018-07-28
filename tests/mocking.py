# -*- coding: utf-8 -*-
#
# mocking.py
#
# This module is part of dstools.
#

"""
Various mockers for test purposes.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


class MockScaler:
    """Mocking a data standardizer object."""

    def __init__(self, dummy_param='dummy_param'):

        self.dummy_param = dummy_param

        # NOTE: Variable set with instance.
        self._feature_std = None

    def fit(self, X, y=None):

        self._feature_std = np.std(X, axis=0)

        return self

    def transform(self, X):

        return np.divide(X, self._feature_std)
