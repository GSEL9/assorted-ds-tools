# -*- coding: utf-8 -*-
#
# utils.py
#
# This module is part of dstools.
#

"""
The dskit utilities module.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import numpy as np


def score_stats(scores):
    # Computes mean and standard deviation of score values.

    return np.mean(scores, axis=1), np.std(scores, axis=1)
