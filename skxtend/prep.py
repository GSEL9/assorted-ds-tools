# -*- coding: utf-8 -*-
#
# prep.py
#
# This module is part of skxtend.
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


class DiscardOutliers(BaseEstimator, TransformerMixin):
    """Remove outliers based on minimum and maximum criteria of log error
    value."""

    def __init__(self, field='logerror', min_val=-0.4, max_val=0.4):

        self.field = field
        self.min_val = min_val
        self.max_val = max_val

        self._query = None
        self._data = None

    def fit(self, X, y=None):
        """Construct query to remove outliers."""

        self._data = check_array(X)

        self._query = '{field} > {min_val} and {field} < {max_val}'.format(
            field=self.field, min_val=self.min_val, max_val=self.max_val
        )
        return self

    def transform(self):
        """Execute outlier removing query."""

        return self._data.query(self._query)


class TrainTestSplitter:
    """Generate training and test splits from original feature data.
    Optional to standardize training and test data."""

    def __init__(self, test_size=0.3, scale=True, random_state=None):

        self.test_size = test_size
        self.scale = scale
        self.random_state = random_state

        # NOTE: Variables set with instance.
        self.X = None
        self.y = None
        self.scaler = None

    def fit(self, X, y):

        self.X, self.y = check_X_y(X, y)
        self.scaler = self.Standardizer()

        return self

    def transform(self):

        # Split org data into training and test data.
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size,
            random_state=self.random_state
        )

        # Standardize training and test data splits.
        if self.scale:
            self.scaler.fit(X_train)
            X_train_std = self.scaler.transform(self.X_train)
            X_test_std = self.scaler.transform(self.X_test)

            return X_train_std, X_test_std, y_train, y_test

        else:
            return X_train, X_test, y_train, y_test


class Standardizer:
    """Standardize feature data by subtracting mean and dividing by
    standard deviation."""

    # NOTE: Fit standard scaler to training data to learn training data
    # parameters. Transform both training data and test data with training
    # data parameters. Thus, standardized training and test data are comparable.

    def __init__(self, scaler=StandardScaler):

        self._scaler = scaler

    def fit(self, X, y=None, **kwargs):

        self._scaler(**kwargs)
        self._scaler.fit(X)

        return self

    def transform(self, X):

        X_trans = self._scaler.transform(X)

        return X_trans


class LabelEncodeObjects(BaseEstimator, TransformerMixin):
    """Transform features of data type object by label encoding."""

    def __init__(self, object_dtype='object'):

        self.object_dtype = object_dtype

        self._data = None
        self._object_columns = None

    def fit(self, X, y=None):
        """Determine which features are not numerical."""

        self._data = check_array(X)
        # Check which features (non-numerical) that must be encoded.
        self._object_columns = []
        for feature in self.data.columns:
            if self._data[feature].dtype == self.object_dtype:
                self._object_columns.append(feature)

        return self

    def transform(self):
        """Transform features of data type object by label encoding.

        Returns:
            (numpy.ndarray): The label encoded dataset.

        """

        for feature in self._object_feature:
            encoder = LabelEncoder()
            feature_data = list(self._data[feature].values)
            self._data[feature] = encoder.fit_transform(feature_data)

        return np.ndarray(self._data, dtype=float)


class LabelEncodeFeatures(BaseEstimator, TransformerMixin):
    """Transform specified features by label encoding."""

    def __init__(self, features=None):

        self.features = features

        self._data = None

    def fit(self, X, y=None):

        self._data = check_array(X)

        if self.features is None:
            self.features = self._data.columns

        return self

    def transform(self):
        """Transform specified features by label encoding.

        Returns:
            (numpy.ndarray): The label encoded dataset.

        """

        for feature in self.features:

            encoder = LabelEncoder()
            feature_data = list(self._data[feature].values)
            self._data[feature] = encoder.fit_transform(feature_data)

        return np.array(self._data, dtype=float)


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


class Clone(BaseEstimator, TransformerMixin):
    """Clone a dataset."""

    def __init__(self):

        self._data = None

    def fit(self, X, y=None):

        self._data = check_array(X)

        return self

    def transform(self):

        return self._data.copy()
