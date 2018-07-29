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


class FeatureScaling:
    """Scale feature data."""

    # NOTE: Fit standard scaler to training data to learn training data
    # parameters. Transform both training data and test data with training
    # data parameters. Thus, standardized training and test data are comparable.

    # NOTE: The objective functions of some learning algorithms objective will
    # not work properly without normalization.

    def __init__(self, scaler=StandardScaler, **kwargs):

        self.scaler = scaler(**kwargs)

    def fit(self, X, y=None, **kwargs):

        self.scaler.fit(X)

        return self

    def transform(self, X, y=None, **kwargs):

        return self.scaler.transform(X)

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=None, **kwargs)


class Clone(BaseEstimator, TransformerMixin):
    """Clone dataset."""

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


class FeatureImputer(BaseEstimator, TransformerMixin):
    """Impute missing values by specified fill value or method.

    Args:
        method (str, {mean, median, most_frequent}): The imputation strategy.

    Attributes:
        targets (list of str): The labels of all features containing the value
            to impute.

    """

    def __init__(self, method='mean', impute_val=np.nan, imputer=None):

        self.method = method
        self.impute_val = impute_val

        if imputer is None:
            self.imputer = Imputer(
                missing_values=self.impute_val, strategy=self.method, axis=1
            )
        else:
            self.imputer = imputer

        # NOTE: Variable set with instace.
        self.targets = None

    def fit(self, X, y=None, **kwargs):

        # Determine which features that will be imputed.
        self.targets = []
        for feature in X.columns:
            if np.any(X[feature].isnull()):
                self.targets.append(feature)

        return self

    def transform(self, X, y=None, **kwargs):

        # TODO: Type checking

        data = X.copy()
        for feature in self.targets:
            feature_data = np.atleast_2d(data[feature].values)
            data[feature] = self.imputer.fit_transform(feature_data)[0]

        return data

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


class RemoveOutliers(BaseEstimator, TransformerMixin):
    """Remove outliers from dataset using percentiles.

    Attributes:
        quantiles (pandas.DataFrame): The feature values considered outside the
            boundaries of the percentiles.

    """

    def __init__(self, low=0.05, high=0.95):

        self.low = low
        self.high = high

        # NOTE: Variable set with instance.
        self.quantiles = None

    def fit(self, X, y=None, **kwargs):
        """Define query that determines outliers."""

        self.quantiles = X.quantile([self.low, self.high])

        return self

    def transform(self, X, y=None, **kwargs):
        """Remove outliers from data."""

        def quantile_filter(x):

            lower_limit = self.quantiles.loc[self.low, x.name]
            upper_limit = self.quantiles.loc[self.high, x.name]

            return x[(x > lower_limit) & (x < upper_limit)]

        return X.apply(quantile_filter, axis=0)

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)



def train_test_scaling(X, y, test_size=0.2, random_state=None, scaler=None):
    """Split original feature data into training and test splits including
    standardization.

    Args:
        X (array-like): An (n x m) array of feature samples.
        y (array-like): An (n x 1) array of target samples.
        test_size (float): The fraction of data used in validation.
        scaler (object): A feature data scaler object.
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
    # Scale feature data with default parameters.
    if scaler is None:
        _scaler = FeatureScaling()
    else:
        _scaler = FeatureScaling(scaler=scaler)
    X_train_std = _scaler.fit_transform(X_train)
    X_test_std = _scaler.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """Transform the feature data of a target data type according to a
    specified encoding procedure.

    Attributes:
        targets (list of str): The feature labels of target data type.


    """

    def __init__(self, target_dtype='object', encoder=LabelEncoder):

        self.target_dtype = target_dtype
        self.encoder = encoder()

        # NOTE: Variable set with instance.
        self.targets = None
        self.target_labels = None

    def fit(self, X, y=None, **kwargs):
        """Determine which features are not numerical."""

        # Determine which features that will be encoded.
        self.targets, self.target_labels = [], {}
        for feature in X.columns:
            if X[feature].dtype == self.target_dtype:
                self.targets.append(feature)
                self.target_labels[feature] = np.unique(X[feature])

        return self

    def transform(self, X, y=None, **kwargs):
        """Transform features of target data type with specified encoder
        procedure.

        Returns:
            (): The label encoded dataset.

        """

        # TODO: Type checking

        data = X.copy()
        for feature in self.targets:
            feature_data = list(data[feature].values)
            data[feature] = self.encoder.fit_transform(feature_data)

        return data

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


class DropFeatures(BaseEstimator, TransformerMixin):
    """Remove features from dataset.

    Attributes:
        targets (list):

    """

    def __init__(self, features):

        self.features = features

        # NOTE: Variable set with instance.
        self.targets = None

    def fit(self, X, y=None, **kwargs):

        self.targets = []
        for feature in X.columns:
            if feature in self.features:
                self.targets.append(feature)

        return self

    def transform(self, X, y=None, **kwargs):

        data = X.copy()
        for feature in self.targets:
            data.drop(labels=feature, axis=1, inplace=True)

        return data

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)
