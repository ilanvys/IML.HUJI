from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

#TODO: fix _find_threshold to work faster

class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        thr_err = np.inf
        self.threshold_ = 0
        self.j_ = 0
        for index, feature in enumerate(X.T):
            for sign in [-1, 1]:
                feature_thr, feature_thr_err = self._find_threshold(feature, y, sign)
            
                if feature_thr_err < thr_err:
                    thr_err = feature_thr_err
                    self.threshold_ = feature_thr
                    self.j_ = index
                    self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        #TODO: fix this func
        # thr_err = np.inf
        # thr = None
        # possible_thresholds = np.sort(values)


        # for thr_index, thr_val in enumerate(values):
        #     predictions = np.where(values >= thr_val, sign, -sign)

        #     # classification_err = np.mean(predictions != labels)
        #     classification_err = np.sum(np.abs(labels)[np.sign(labels) == sign])
        #     if classification_err < thr_err:
        #         thr_err = classification_err
        #         thr = thr_val
        
        # a = (thr, thr_err)

        # for thr_index, thr_value in enumerate(possible_thresholds):
        #     temp_classification = np.zeros(len(values))
        #     for i in range(len(values)):
        #         if values[i] < thr_value:
        #             temp_classification[i] = -sign
        #         else:
        #             temp_classification[i] = sign
        #     calssification_err = np.count_nonzero(temp_classification != labels) / len(labels)
        #     if calssification_err < thr_err:
        #         thr_err = calssification_err
        #         thr = thr_index
        
        # return (thr, thr_err)

        # Sort values such that search of threshold below is in O(nlogn) for n the number of samples
        # instead of O(n^2)
        possible_thresholds = np.sort(values)
        ids = np.argsort(values)
        values, labels = values[ids], labels[ids]

        # Loss for classifying all as `sign` - namely, if threshold is smaller than values[0]
        loss = np.sum(np.abs(labels)[np.sign(labels) == sign])

        # Loss of classifying threshold being each of the values given
        loss = np.append(loss, loss - np.cumsum(labels * sign))

        id = np.argmin(loss)
        return np.concatenate([[-np.inf], values[1:], [np.inf]])[id], loss[id]
    
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
