# -*- coding: utf-8 -*-
#
# base.py
#
# This module is part of dstools.
#

"""
The dstools base module.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.base import ClassifierMixin, TransformerMixin


class StackingBase(BaseEstimator, ClassifierMixin, TransformerMixin):
    """The stacking meta estimator base."""

    def __init__(self, learners, random_state=None):

        self.base_learners = learners[:-1]
        self.end_learner = learners[-1]
        self.random_state = random_state

    @property
    def base_learners(self):

        return self._base_learners

    @base_learners.setter
    def base_learners(self, value):

        if isinstance(value, (list, tuple, np.ndarray)):
            self._base_learners = value
        else:
            self._base_learners = [value]

    @property
    def end_learner(self):

        return self._end_learner

    @end_learner.setter
    def end_learner(self, value):

        # TODO: Type checking.
        self._end_learner = value

    # ERROR: Must specify attributes to check_is_fitted()
    def _check_learners_fitted(self):

        #for learner in self.base_learners:
        #    check_is_fitted(
        #        learner, msg='Estimator {} not fitted'.format(learner)
        #    )

        pass
