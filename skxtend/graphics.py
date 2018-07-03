# -*- coding: utf-8 -*-
#
# graphics.py
#
# This module is part of skxtend.
#

"""
Tools to visualize data and create plots.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

from sklearn.metrics import silhouette_samples


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
