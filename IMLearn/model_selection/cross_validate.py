from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score = []
    validation_score = []
    indices = np.arange(X.shape[0])
    # shuffled_indices = np.random.shuffle(shuffled_indices) 
    folds_indices = np.array_split(indices, cv)

    for i in range(cv):
        # Exclude the indices of the i'th fold
        curr_train_indices = np.concatenate(folds_indices[:i] + folds_indices[i+1:])
        X_train, y_train = X[curr_train_indices], y[curr_train_indices]

        # Fit model over all the train except the i'th fold
        fitted_model = deepcopy(estimator).fit(X_train, y_train)

        # save train_score for current train and validation_score for the i'th fold
        train_score.append(scoring(y_train, fitted_model.predict(X_train)))
        validation_score.append(scoring(y[folds_indices[i]], 
                                        fitted_model.predict(X[folds_indices[i]])))
    
    return np.mean(train_score), np.mean(validation_score)
