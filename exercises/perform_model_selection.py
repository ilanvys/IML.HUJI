from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X = X[:n_samples]
    train_y = y[:n_samples]
    test_X = X[n_samples:]
    test_y = y[n_samples:]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_lambdas = np.linspace(0, 0.3, num=n_evaluations)
    lasso_lambdas = np.linspace(0, 2, num=n_evaluations)
    
    ridge_train_err, ridge_validation_err = np.zeros(n_evaluations), np.zeros(n_evaluations)
    lasso_train_err, lasso_validation_err = np.zeros(n_evaluations), np.zeros(n_evaluations)
    for i in range(n_evaluations):
        ridge_train_err[i], ridge_validation_err[i] = cross_validate(RidgeRegression(ridge_lambdas[i]), train_X, train_y, mean_square_error, 5)
        lasso_train_err[i], lasso_validation_err[i] = cross_validate(Lasso(lasso_lambdas[i]), train_X, train_y, mean_square_error, 5)

    ridge_train_err_line = go.Scatter(x=ridge_lambdas, y=ridge_train_err, mode='lines', name="Ridge Train Error")
    ridge_validation_err_line = go.Scatter(x=ridge_lambdas, y=ridge_validation_err, mode='lines', name="Ridge Validation Error")
    layout = go.Layout(
                title=f'Train and Validation Error as a Function of Ridge Regularization Parameter', 
                xaxis_title=f'Regularization Parameter Values', 
                yaxis_title='Error')
    go.Figure(data=[ridge_train_err_line, ridge_validation_err_line], layout=layout).show()
    
    lasso_train_err_line = go.Scatter(x=lasso_lambdas, y=lasso_train_err, mode='lines', name="Lasso Train Error")
    lasso_validation_err_line = go.Scatter(x=lasso_lambdas, y=lasso_validation_err, mode='lines', name="Lasso Validation Error")
    layout2 = go.Layout(
                title=f'Train and Validation Error as a Function of Lasso Regularization Parameter', 
                xaxis_title=f'Regularization Parameter Values', 
                yaxis_title='Error')
    go.Figure(data=[lasso_train_err_line, lasso_validation_err_line], layout=layout2).show()

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    # Handle Lasso Model
    best_lasso_validation_lambda = lasso_lambdas[np.argmin(lasso_validation_err)]
    best_lasso_model = Lasso(best_lasso_validation_lambda)
    best_lasso_model.fit(train_X, train_y)
    lasso_loss = mean_square_error(test_y, best_lasso_model.predict(test_X))

    # Handle Ridge Model
    best_ridge_validation_lambda = ridge_lambdas[np.argmin(ridge_validation_err)]
    best_ridge_model = RidgeRegression(best_ridge_validation_lambda)
    ridge_loss = best_ridge_model.fit(train_X, train_y).loss(test_X, test_y)

    # Handle Linear Model
    linear_model = LinearRegression(best_ridge_validation_lambda)
    linear_loss = linear_model.fit(train_X, train_y).loss(test_X, test_y)
    
    print("Best Regularization Parameter for Ridge: ", best_ridge_validation_lambda)
    print("Best Regularization Parameter for Lasso: ", best_lasso_validation_lambda)
    
    print("Lasso Model Loss: ", lasso_loss)
    print("Ridge Model Loss: ", ridge_loss)
    print("Linear Model Loss: ", linear_loss)

if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
