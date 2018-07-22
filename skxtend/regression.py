# -*- coding: utf-8 -*-
#
# regression.py
#
# This module is part of skxtend.
#

"""
Various scikit-learn compatible regression algorithms.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import torch
import inspect
import datetime

import numpy as np
import pandas as pd
import torch.nn.init as init
import torch.utils.data as data_utils

from tempfile import mkdtemp
from dateutil.parser import parse
from torch.autograd import Variable
from sklearn.externals.joblib import Memory
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array


# WIP:
class TorchRegressor(BaseEstimator, RegressorMixin):
    """A scikit-learn compatible pytorch regressor.

    From Wiki: CUDA is a parallel computing platform and application
        programming interface (API) model created by Nvidia.

    Args:
        input_dim (int): The number of original features.
        output_dim (int): The number of features ...


        num_epochs (int): The number of iterations
        batch_size (int):
        shuffle (bool, {True, False}): Determines whether to
        learning_rate (float):

    """

    # QUESTION: Can replace input_dim with X.shape[1]???
    def __init__(self, output_dim=1, input_dim=100, hidden_layer_dims=[100, 100],
                 num_epochs=1, learning_rate=0.01, batch_size=128, shuffle=False,
                 callbacks=[], use_gpu=True, verbose=1):

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.callbacks = callbacks
        self.use_gpu = use_gpu
        self.verbose = verbose

        # NOTE: Variables set during instance.
        self._gpu = None
        self._model = None
        self._history = None
        self._shape_layers = None

    def _gpu_dependency(self):
        # Determine if CUDA is available and should use GPU.

        if torch.cuda.is_available():
            self._gpu = use_gpu
        else:
            self._gpu = False

        if self._gpu:
            self._model = self._model.cuda()

        return self

    def _construct_model(self):
        # Build the torch NN model (???).

        self._layer_dims = [self.input_dim] + self.hidden_layer_dims + [self.output_dim]

        # A container of modules in each model layer.
        self._model = torch.nn.Sequential()

        # Loop through the layer dimensions and add to the sequential container
        # (1) an input layer (2) a hidden layer with relu activation.
        for idx, dim in enumerate(self._layer_dims):

            if (idx < len(self._layer_dims) - 1):

                module = torch.nn.Linear(dim, self._layer_dims[idx + 1])
                init.xavier_uniform_(module.weight)

                self._model.add_module(
                    ('').join(('linear', str(idx))), module
                )
            if (idx < len(self._layer_dims) - 2):
                self._model.add_module(
                    ('').join(('relu', str(idx))), torch.nn.ReLU()
                )

        # Setup following the use of GPU.
        self._gpu_dependency()

        return self

    def transform_X_y(self, X, y):
        # Prepare data for BGD optimizer.

        # Convert numpy arrays to torch tensor object.
        torch_X = torch.from_numpy(X).float()
        torch_y = torch.from_numpy(y).float()

        # Create CUDA device objects.
        if self._gpu:
            torch_X, torch_y = torch_X.cuda(), torch_y.cuda()

        # Create torch.utils.data.dataset.TensorDataset objects of X and y
        # tensors.
        train = data_utils.TensorDataset(torch_X, torch_y)

        # Combines a dataset and a sampler, and provides single- or
        # multi-process iterators over the dataset.
        train_loader = data_utils.DataLoader(
            train, batch_size=self.batch_size, shuffle=self.shuffle
        )

        return train_loader

    def optimizer(self, method='adam'):

        # Selects Adam Optimization Algorithm which is an extension to SGD.
        if method == 'adam':
            algorithm = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )
        else:
            raise NotImplementedError('')

        return algorithm

    def loss_function(self, method='mse'):

        if method == 'mse':
            function = torch.nn.MSELoss(size_average=False)
        else:
            raise NotImplementedError('')

        return function

    def _eval_loss(self, y_train, y_pred, function):

        _loss = function(
            y_pred,
            Variable(y_train.cuda().float() if self._gpu else y_train.float())
        )

        return _loss

    def _train_model(self, X, y, optimizer='adam', loss='mse'):

        train_loader = self.transform_X_y(X, y)
        optimizer = self.optimizer(method=optimizer)
        loss_function = self.loss_function(method=loss)

        self._history = {'loss': [], 'val_loss': [], 'mse_loss': []}

        finished = False
        while not finished:

            for epoch in range(self.num_epochs):

                loss = None
                for num, (minibatch, y_train) in enumerate(train_loader):
                    y_train, y_pred, loss = self._forward(
                        minibatch, y_train, loss_function, optimizer
                    )

                error = mean_absolute_error(y_train, y_pred)

                self._history['mse_loss'].append(loss.data[0])
                self._history['loss'].append(error)

                if self.verbose > 0:
                    self._status_report(epoch, error, loss.data[0])

            finished = self._eval_callbacks()

    def _forward(self, minibatch, y_train, loss_function, optimizer):

        y_pred = np.squeeze(self._model(Variable(minibatch)))

        loss = self._eval_loss(y_train, y_pred, loss_function)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self._gpu:
            y_train, y_pred = y_train.cpu().numpy(), y_pred.cpu().data.numpy()
        else:
            y_train, y_pred = y_train.numpy(), y_pred.data.numpy()

        return y_train, y_pred, loss

    def _status_report(self, epoch, error, mse_loss):

        print('Results for epoch {}, loss {}, mse_loss {}'
              ''.format(epoch + 1, error, mse_loss))

    def _eval_callbacks(self):

        for callback in self.callbacks:
            callback.call(self._model, self._history)

            if callback.finish:
                return True
            else:
                return False

    def fit(self, X, y):
        """
        Trains the pytorch regressor.
        """

        # NOTE: Replace with @property.
        assert (type(self.input_dim) == int), 'input_dim parameter must be defined'
        assert (type(self.output_dim) == int), 'output_dim must be defined'

        self._construct_model()
        self._train_model(X, y)

        return self

    def predict(self, X, y=None):
        """
        Makes a prediction using the trained pytorch model
        """

        if self._history == None:
            raise RuntimeError('Regressor has not been fit')

        results = []
        split_size = np.ceil(len(X) / self.batch_size)

        # In case the requested size of prediction is too large for memory (especially gpu)
        # split into batchs, roughly similar to the original training batch size. Not
        # particularly scientific but should always be small enough.
        for batch in np.array_split(X, split_size):
            x_pred = Variable(torch.from_numpy(batch).float())
            y_pred = self._model(x_pred.cuda() if self._gpu else x_pred)
            y_pred_formatted = y_pred.cpu().data.numpy() if self._gpu else y_pred.data.numpy()
            results = np.append(results, y_pred_formatted)

        return results

    def score(self, X, y, sample_weight=None):
        """Scores the data using the trained pytorch model. Returns negative
        Mean Squared Error.
        """

        y_pred = self.predict(X, y)

        return mean_absolute_error(y, y_pred) * (-1)


class TikhonovCV(BaseEstimator):
    """Tikhonov regularized least-squares regression by QR-decomposition with
    leave-one-out cross-validation of prediction estimates.

    Args:
        reg_mat (array like): Regularization matrix. Assigns identity matrix by
                              default.
        alpha (int, float): Regularization parameter. Assigns 1.0 by default.

    Kwargs:
        fit_intercept (bool): Includes an intercept coefficient to the
                              regression model parameters.
        normalize (bool): Normalizes the data matrix.

    """

    def __init__(self, reg_mat=None, alpha=None, fit_intercept=True,
                 normalize=False):

        self.reg_mat = reg_mat
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        # NOTE: Variable set with instance.
        self.coef_ = None
        self._msecv = None
        self._rescv_ = None
        self._orthog = None
        self._upper_tri = None

    @property
    def mescv(self):
        """Returns cross-validated Mean Square Error"""

        return self._mescv

    @property
    def rescv(self):
        """Returns cross-validated model residuals."""

        return self._rescv

    def _check_setup(self, X, y):
        # Checks feature matrix X and target vector y array.
        # Constructs regularization matrix.
        # Converts y to (n x 1) array.

        X, y = check_X_y(X, y)

        if not self.alpha:
            self.alpha = 1.0

        if self.reg_mat is None:
            _tikh_mat = np.eye(X.shape[1]) * np.sqrt(self.alpha)
        else:
            _tikh_mat = self.reg_mat * np.sqrt(self.alpha)

        if not _tikh_mat.shape[1] == X.shape[1]:
            self.tikh_mat = _tikh_mat[:, :X.shape[1]]
        else:
            self.tikh_mat = _tikh_mat

        if len(np.shape(y)) < 2:
            return X, y[:, np.newaxis]
        else:
            return X, y

    @staticmethod
    def _normalize(X, y):
        # Divides feature matrix and target vector by column sums.

        X_col_sums = np.sum(X, axis=0)
        X_norm = np.divide(X, X_col_sums)

        y_col_sum = np.sum(y, axis=0)
        y_norm = y / y_col_sum

        return X_norm, y_norm

    def _stacked_matrices(self, X, y):
        # Stacks regularization and feature matrix, and expands target data.

        num_data_samples, _ = np.shape(X)
        num_reg_samples, _ = np.shape(self.tikh_mat)

        reg_dummies = np.zeros((num_reg_samples, 1))

        if self.fit_intercept:
            intercept = np.ones((num_data_samples, 1))
            variables = np.hstack((intercept, X))
            regularization = np.hstack((reg_dummies, self.tikh_mat))

            stacked_data = np.vstack((variables, regularization))

        else:
            stacked_data = np.vstack((X, self.tikh_mat))

        stacked_target = np.vstack((y, reg_dummies))

        return stacked_data, stacked_target

    def _decompose(self, X):
        # Performs QR decomposition of feature matrix.

        return np.linalg.qr(X)

    def fit(self, X, y):
        """Estimate least-squares regression coefficients.

        Args:
            X (array like): Feature matrix as (samples x features).
            y (array like): Target vector.

        """

        _X, _y = self._check_setup(X, y)

        if self.normalize:
            _data_stack, _target_stack = self._stacked_matrices(_X, _y)

            data_stack, target_stack = self._normalize(
                _data_stack, _target_stack
            )
        else:
            data_stack, target_stack = self._stacked_matrices(_X, _y)

        self._orthog, self._upper_tri = self._decompose(data_stack)

        proj = np.dot(np.transpose(self._orthog), target_stack)
        self.coef_ = np.dot(np.linalg.inv(self._upper_tri), proj)

        return self

    def predict(self, X):
        """Predict observations for each input data point.

        Args:
            X (array like): Feature matrix as (samples x features).

        """

        _X = check_array(X)

        if self.fit_intercept:
            return np.dot(_X, self.coef_[1:]) + self.coef_[0]

        else:
            return np.dot(_X, self.coef_)

    def cross_val_predict(self, X, y):
        """Perform cross-validated prediction for each input data point.

        Args:
            X (array like): Feature matrix as (samples x features).
            y (array like): Target vector.

        """

        self.fit(X, y)

        pred = self.predict(X)
        res = y - pred[:, 0]

        diag = np.sum(np.power(self._orthog[:X.shape[0], :], 2), axis=1)
        self._rescv = res / (1 - diag)
        self._mescv = np.dot(self._rescv, self._rescv) / self._rescv.size

        return pred[:, 0] - self._rescv
