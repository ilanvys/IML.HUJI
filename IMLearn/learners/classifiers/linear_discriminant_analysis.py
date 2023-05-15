from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        classes_num = len(self.classes_)
        self.pi_ = np.zeros(classes_num)
        self.mu_ = np.zeros([classes_num, len(X[0])])

        for i in range(classes_num):
            indices = np.where(y == self.classes_[i])
            self.pi_[i] = len(indices[0]) / len(y)
            self.mu_[i] = np.mean(X[indices], axis=0)

        # Calculate the centered data matrix
        centered_X = X - self.mu_[y]
        self.cov_ = np.dot(centered_X.T, centered_X) / (len(X)-len(self.classes_))
        self._cov_inv = inv(self.cov_)

        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihood = self.likelihood(X)
        argmax_indices = np.argmax(likelihood, axis=1)
        return self.classes_[argmax_indices]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        z = np.sqrt((2 * np.pi) ** len(X[0]) * det(self.cov_))
        diff = X[:, np.newaxis, :] - self.mu_
        exp = np.exp(-0.5 * np.sum(((diff @ self._cov_inv) * diff), axis=2))
        return self.pi_ * (1 / z) * exp

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
