# -*- coding: utf-8 -*-
#
# prep.py
#
# This module is part of dstools.
#

"""
Data pre-processing tools.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_X_y

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# QUESTION: Rename (e.g. scaler)?
class Standardizer:
    """Standardize feature data by subtracting mean and dividing by feature
    standard deviation."""

    # NOTE: Fit standard scaler to training data to learn training data
    # parameters. Transform both training data and test data with training
    # data parameters. Thus, standardized training and test data are comparable.

    def __init__(self, scaler=StandardScaler, **kwargs):

        self._scaler = scaler(**kwargs)

    def fit(self, X, y=None, **kwargs):

        self._scaler.fit(X)

        return self

    def transform(self, X):

        return self._scaler.transform(X)

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X)


# QUESTION: Rename (e.g. train_test_scale/split_and_scale/split_and_standardize
# /split_center_scale)?
def train_test_scaling(X, y, test_size, random_state):
    """Split original feature data into training and test splits including
    standardization.

    Args:
        X (array-like): An (n x m) array of feature samples.
        y (array-like): An (n x 1) array of target samples.
        test_size (float): The fraction of data used in validation.
        random_state (int): The random number generator intiator.

    Returns:
        (tuple): Standardized training and test sets of feature and target
            data.

    """

    # NOTE: Should be function not class since dependent of random numer
    # intiator and thus cannot be included in pipeline.

    # Generate training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # Standardize feature data.
    scaler = Standardizer()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test


class Clone(BaseEstimator, TransformerMixin):
    """Clone a dataset."""

    def __init__(self):

        self._data = None

    def fit(self, X, y=None):

        # TODO: Type checking.
        self._data = np.empty(np.shape(X))

        return self

    def transform(self, X=None):

        self._data[:] = X

        return self._data

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)


# ERROR: Cannot query with numpy array. Must convert data to correct dtype.
class DiscardOutliers(BaseEstimator, TransformerMixin):
    """Remove outliers based on minimum and maximum criteria of log error
    value.

    Attributes:
        query (str):
        min_val (float): The minimum field value.
        max_val (float): The maximum field value.

    """

    def __init__(self, field='logerror', min_val=-0.4, max_val=0.4):

        self.field = field
        self.min_val = min_val
        self.max_val = max_val

        self.query = None

    def fit(self, X, y=None):
        """Define query that determines outliers."""

        self.query = '{field} > {min_val} and {field} < {max_val}'.format(
            field=self.field,
            min_val=self.min_val,
            max_val=self.max_val
        )
        return self

    # ERROR:
    def transform(self, X):
        """Remove outliers from data."""

        return X.query(self.query)


# WIP:
class ReplaceNans(BaseEstimator, TransformerMixin):
    """Replace NaN values by specified fill value or method."""

    def __init__(self, fill_val=-1, features=[], method=None):

        self.fill_val = fill_val
        self.features = features
        self.method = method

        self._data = None

    def fit(self, X, y=None):

        self._data = check_array(X)

        if self.features is None:
            self.features = self._data.columns

        return self

    def transform(self):

        if self.method is None:
            self._data.fillna(self.fill_val)

        elif self.method == 'mean':
            for feature in self.features:
                filled = self._data[feature].fillna(self._data[feature].mean)
                self._data[feature] = filled

        elif self.method == 'std':
            for feature in self.features:
                filled = self._data[feature].fillna(self._data[feature].std)
                self._data[feature] = filled

        else:
            raise NotImplementedError('Not implemented fill method `{}`'
                                      ''.format(self.method))

        return self._data


# WIP:
class DropFeatures(BaseEstimator, TransformerMixin):
    """Remove features from dataset."""

    def __init__(self, features=[]):

        self.features = features

        self._data = None

    def fit(self, X, y=None):

        self._data = check_array(X)

        return self

    def transform(self):

        return self._data.drop(self.features, axis=1)


# WIP:
class FeatureEncoder(BaseEstimator, TransformerMixin):
    """Transform the feature data of a target data type according to a
    specified encoding procedure.

    Attributes:
        targets (list of str): The feature labels of target data type.
        encoder (object): An encoder with a `fit_transform()` method.

    """

    def __init__(self, target_dtype='object', encoder=LabelEncoder):

        self.target_dtype = target_dtype
        self.encoder = encoder()

        # NOTE: Variable set with instance.
        self.targets = None

    def fit(self, X, y=None):
        """Determine which features are not numerical."""

        # Determine which features that will be encoded.
        self.targets = []
        for feature in X.columns:
            if X[feature].dtype == self.target_dtype:
                self.targets.append(feature)

        return self

    def transform(self, X, copy=True):
        """Transform features of target data type with specified encoder
        procedure.

        Returns:
            (): The label encoded dataset.

        """

        if copy:
            # TODO: Type checking
            data = X.copy()
        else:
            # TODO: Type checking
            data = X

        # Encode selected features according to specified procedure.
        for feature in self.targets:
            feature_data = list(data[feature].values)
            data[feature] = self.encoder.fit_transform(feature_data)

        return data


if __name__ == '__main__':

    """
    DOTO:
        * Check how LabelEncoder actually works.
        * Mimick LabelEncoder in mock and complete tests.
    """

    from pytest import approx

    def data(seed=123):

        np.random.seed(seed)

        num_samples, num_features = 100, 10
        X = np.random.random((num_samples, num_features))
        y = np.random.random((num_samples, 1))

        return X, y

    data, _ = data()


    class MockEncoder:
        """Mocking a feature encoder object."""

        def __init__(self):

            self.old_keys = None
            self.new_keys = None

        def fit(self, X, y=None, **kwargs):

            pass

        def transform(self, X):

            pass

        def fit_transform(self, X, y=None, **kwargs):

            self.fit(X, y=y, **kwargs)

            return self.transform(X)
