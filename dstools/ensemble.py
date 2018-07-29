# -*- coding: utf-8 -*-
#
# ensemble.py
#
# This module is part of dstools.
#

"""
Tools for combining base estimators to build a learning algorithm with
improved robustness over a single estimator.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import numpy as np

from scipy import sparse
from dstools.base import StackBase
from sklearn.utils.validation import check_X_y, check_array


class ClassifierStack(StackBase):
    """An ensemble learning technique to combine multiple classification models
    via a meta-classifier."""

    def __init__(self, learners, random_state=None, use_probas=False):

        super().__init__(learners, random_state)

        self.use_probas = use_probas

    def fit(self, X, y, verbose=True, **kwargs):

        _X, _y = check_X_y(X, y)

        for leaner in self.base_learners:

            if verbose:
                print('Training classifier: `{}`'.format(type(leaner)))

            leaner.fit(_X, _y)

        meta_features = self.gen_meta_features(_X)
        if sparse.issparse(_X):
            self.end_learner.fit(sparse.hstack((_X, meta_features)), _y)
        else:
            self.end_learner.fit(np.hstack((_X, meta_features)), _y)

        return self

    def gen_meta_features(self, X):

        self._check_learners_fitted()

        if self.use_probas:
            probas = np.asarray(
                [learner.predict_proba(X) for learner in self.base_learners]
            )
            meta_features = np.concatenate(probas, axis=1)

        else:
            meta_features = np.column_stack(
                [learner.predict(X) for learner in self.base_learners]
            )

        return meta_features

    def predict(self, X):

        self._check_learners_fitted()

        _X = check_array(X)
        meta_features = self.gen_meta_features(_X)
        if sparse.issparse(_X):
            return self.end_learner.predict(sparse.hstack((_X, meta_features)))
        else:
            return self.end_learner.predict(np.hstack((_X, meta_features)))

    def predict_proba(self, X):

        self._check_learners_fitted()

        _X = check_array(X)
        meta_features = self.gen_meta_features(_X)
        if sparse.issparse(_X):
            return self.end_learner.predict_proba(
                sparse.hstack((_X, meta_features))
            )
        else:
            return self.end_learner.predict_proba(
                np.hstack((_X, meta_features))
            )


class RegressionStack(StackBase):

    def __init__(self, learners, random_state=None):

        super().__init__(learners, random_state)

    @property
    def coef_(self):

        return self.meta_regr_.coef_

    @property
    def intercept_(self):

        return self.meta_regr_.intercept_

    def fit(self, X, y, verbose=True, **kwargs):

        _X, _y = check_X_y(X, y)

        for learner in self.base_learners:

            if verbose:
                print('Training classifier: `{}`'.format(type(learner)))

            learner.fit(_X, _y)

        meta_features = self.gen_meta_features(_X)
        if sparse.issparse(_X):
            self.end_learner.fit(sparse.hstack((_X, meta_features)), _y)
        else:
            self.end_learner.fit(np.hstack((_X, meta_features)), _y)

        return self

    def gen_meta_features(self, X):

        self._check_learners_fitted()

        return np.column_stack(
            [learner.predict(X) for learner in self.base_learners]
        )

    def predict(self, X):

        self._check_learners_fitted()

        _X = check_array(X)
        meta_features = self.gen_meta_features(X)

        if sparse.issparse(_X):
            return self.end_learner.predict(sparse.hstack((_X, meta_features)))
        else:
            return self.end_learner.predict(np.hstack((_X, meta_features)))


if __name__ == '__main__':

    from sklearn import datasets

    iris = datasets.load_iris()
    X, y = iris.data[:, 1:3], iris.target

    # Classification
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier

    classifiers = [RandomForestClassifier(), LogisticRegression()]
    clf_stack = ClassifierStack(learners=classifiers)
    clf_stack.fit(X, y)
    clf_pred = clf_stack.predict(X)
    clf_stack.predict_proba(X)

    # Regression
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge

    regressors = [Ridge(), LinearRegression()]
    reg_stack = RegressionStack(learners=regressors)
    reg_stack.fit(X, y)
    reg_pred = reg_stack.predict(X)


    # NOTE: Make Grid search example. Use parameter_grid() function.
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from mlxtend.classifier import StackingClassifier

    # Initializing models

    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              meta_classifier=lr)

    params = {'kneighborsclassifier__n_neighbors': [1, 5],
              'randomforestclassifier__n_estimators': [10, 50],
              'meta-logisticregression__C': [0.1, 10.0]}

    grid = GridSearchCV(estimator=sclf,
                        param_grid=params,
                        cv=5,
                        refit=True)
    grid.fit(X, y)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')

    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))

    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)


    # NOTE: Make Grid search example. Use parameter_grid() function.

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Lasso

    # Initializing models

    lr = LinearRegression()
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    lasso = Lasso(random_state=1)
    svr_rbf = SVR(kernel='rbf')
    regressors = [svr_lin, lr, ridge, lasso]
    stregr = StackingRegressor(regressors=regressors,
                               meta_regressor=svr_rbf)

    params = {'lasso__alpha': [0.1, 1.0, 10.0],
              'ridge__alpha': [0.1, 1.0, 10.0],
              'svr__C': [0.1, 1.0, 10.0],
              'meta-svr__C': [0.1, 1.0, 10.0, 100.0],
              'meta-svr__gamma': [0.1, 1.0, 10.0]}

    grid = GridSearchCV(estimator=stregr,
                        param_grid=params,
                        cv=5,
                        refit=True)
    grid.fit(X, y)

    for params, mean_score, scores in grid.grid_scores_:
            print("%0.3f +/- %0.2f %r"
                  % (mean_score, scores.std() / 2.0, params))
