# -*- coding: utf-8 -*-
#
# graphics.py
#
# This module is part of skxtend.
#

"""
Tools for data visualization and exploration.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

from sklearn.metrics import silhouette_samples
from sklearn.model_selection import learning_curve, validation_curve


def plot_learning_curve(estimator, X, y, train_sizes, cv):
    """Collects model learning score data score data, and constructs a model
    learning curve from training and test scores. Graphs the model performance
    as a function of the number training samples."""

    # NOTE: Used to check model over-/underfitting across range of training
    # data size. Typically used after optimal hyperparameters are selected.
    _, train_scores, valid_scores = learning_curve(
        estimator=estimator, X=X, y=y, train_sizes=train_sizes, cv=cv
    )
    plt = _gen_plot_learning_curves(train_scores, valid_scores)

    return plt


def plot_validation_curve(estimator, X, y, param_name, param_range, cv):
    """Collects model validation score data score data, and constructs a model
    validation curve from training and test scores. Graphs the model performance
    as a function of a specific hyperparameter."""

    # NOTE: Used to check model over-/underfitting across range of
    # hyperparameter value.
    train_scores, test_scores = validation_curve(
        estimator=estimator, X=X, y=y, param_name=param_name,
        param_range=param_range, cv=cv
    )
    plt = _gen_plot_validation_curve(train_scores, valid_scores)

    return plt


def _gen_plot_learning_curve(train_scores, test_scores, **kwargs):
    # Constructs a model learning curve from training and test scores. Graphs
    # the model performance as a function of the number training samples.

    train_mean, train_std = _scores_stats(train_scores)
    test_mean, test_std = _scores_stats(test_scores)

    plt.figure(figsize=(8, 6))
    plt.title('Learning curve')
    plt.grid()
    plt.plot(
        train_sizes, train_mean, color='blue', marker='o', markersize=5
    )
    plt.fill_between(
        train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15,
        color='blue'
    )
    plt.plot(
        train_sizes, test_mean, color='green', linestyle='--', marker='s',
        markersize=5
    )
    plt.fill_between(
        train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15,
        color='green'
    )
    plt.xlabel('Number of training samples')
    plt.ylabel('Score')
    plt.legend(['training score', 'validation score'], loc='best')
    # NOTE: Limit scores interest area to be between 0.5 and 1.05 (overfitting).
    plt.ylim([0.5, 1.05])

    return plt


def _gen_plot_validation_curve(train_scores, test_scores, **kwargs):
    # Constructs a model validation curve from training and test scores. Graphs
    # the model performance as a function of a specified hyperparameter.

    # NOTE: Collect score data from sklearn validation_curve()
    train_mean, train_std = _scores_stats(train_scores)
    test_mean, test_std = _scores_stats(test_scores)

    plt.figure(figsize=(8, 6))
    plt.title('Validation curve')
    # NOTE: Assign logarithmic scale to enhance small differences.
    plt.grid(), plt.xscale('log')
    plt.plot(
        param_range, train_mean, color='blue', marker='o', markersize=5
    )
    plt.fill_between(
        param_range, train_mean + train_std, train_mean - train_std, alpha=0.15,
        color='blue'
    )
    plt.plot(
        param_range, test_mean, color='green', linestyle='--', marker='s',
        markersize=5
    )
    plt.fill_between(
        param_range, test_mean + test_std, test_mean - test_std, alpha=0.15,
        color='green'
    )
    plt.legend(['training score', 'validation score'], loc='best')
    plt.xlabel('Parameter value')
    plt.ylabel('Score')
    # NOTE: Limit scores interest area to be between 0.5 and 1.
    plt.ylim([0.5, 1.0])

    return plt


def _scores_stats(scores):
    # Computes mean and standard deviation of score values.

    return np.mean(scores, axis=1), np.std(scores, axis=1)


def silhouette_plot(X, ypred, **kwargs):
    """Generates silhouette plot of cluster samples."""

    # Plot confidence levels for each cluster
    cluster_labels = np.unique(ypred)
    n_clusters = cluster_labels.shape[0]

    # Compute the Silhouette Coefficient for each sample.
    silhouette_vals = silhouette_samples(X, ypred, metric='euclidean')

    y_ax_lower, y_ax_upper = 0, 0

    yticks = []
    for num, cluster in enumerate(cluster_labels):

        cluster_silhouette_vals = silhouette_vals[ypred == cluster]
        cluster_silhouette_vals.sort()

        y_ax_upper += len(cluster_silhouette_vals)

        color = cm.jet(float(num) / n_clusters)

        # Barplot with fixed y-range and sorted silhouette values for class
        plt.barh(range(y_ax_lower, y_ax_upper), cluster_silhouette_vals,
                 height=1.0, edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(cluster_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)

    # Vertical line at mean silhouette value
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    return plt
