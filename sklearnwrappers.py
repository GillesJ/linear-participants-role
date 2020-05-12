"""
Wrapper classes for making several external resources work with sklearn.
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.base import SamplerMixin
from imblearn import under_sampling, over_sampling
from sklearn.utils.estimator_checks import check_estimator, check_estimator_sparse_data
import logging


class FeatureMatcher(BaseEstimator, ClassifierMixin):
    """An estimator that predicts true if a feature is present. Works for wordlist matching"""

    def __init__(self, feature_index=None):
        self.feature_index = feature_index

    def fit(self, X, y):
        """This doesn't require any training so just set X, y"""
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self):
        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])

        X = self.X_  # because this only uses own data no training

        # slice X based on feature column
        y = X[:, self.feature_index].toarray().flatten()
        # check if binary feature
        uniq_vals = np.unique(y)
        assert np.array_equal(uniq_vals, np.array([0.0, 1.0])) or np.array_equal(
            uniq_vals, np.array([-1.0, 1.0])
        )
        # normalize to negative label -1
        if np.array_equal(uniq_vals, np.array([0.0, 1.0])):
            y[y == 0] = -1

        return y


class ImblearnWrapper(BaseEstimator, ClassifierMixin):
    """An estimator that predicts true if a feature is present. Works for wordlist matching"""

    def __init__(self, imblearnclassifier):
        if isinstance(imblearnclassifier, SamplerMixin):
            self.imblearn_inst = imblearnclassifier
            self.__name__ = type(imblearnclassifier)
            self.__type__ = type(self.imblearn_inst)
        else:
            raise TypeError("The passed classifier is not an Imblearn Sampler.")

    def __call__(self):
        print(self)

    def __repr__(self):
        return self.imblearn_inst.__repr__()

    def fit(self, X, y):
        """This doesn't require any training so just set X, y"""
        # Store the classes seen during fit
        self.X = X
        self.y = y
        self.classes_ = unique_labels(y)

        self.fitted_X_, self.fitted_y_ = self.imblearn_inst.fit(self.X, self.y)
        # Return the classifier
        self.fit_ = True

        return self

    def transform(self, X, y):
        # X and y are required by sklearn but not used, this is perfectly fine.
        if self.fit_:
            self.X = self.fitted_X_
            self.y = self.fitted_y_
        else:
            raise UserWarning("The estimator has to be fit before it is transformed.")

    def fit_transform(self, X, y):

        return self.fit(X, y).transform(X, y)

    def get_params(self, deep=False):

        print(deep)
        return self.imblearn_inst.get_params(deep=deep)

    def set_params(self, **params):

        return self.imblearn_inst.set_params(**params)


def main():
    # test the estimators
    n_jobs = 10
    imblearn_est = under_sampling.InstanceHardnessThreshold(n_jobs=n_jobs)
    wrapped_est = ImblearnWrapper(imblearn_est)
    # check_estimator(wrapped_est)


if __name__ == "__main__":
    main()
