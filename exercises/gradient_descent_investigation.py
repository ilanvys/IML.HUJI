import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
import copy

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_arr = []
    weights_arr = []
    def gd_state_recorder_callback(**kwargs):
        values_arr.append(kwargs['val'])
        weights_arr.append(kwargs['weights'])

    return gd_state_recorder_callback, values_arr, weights_arr

def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    loss_values = []
    # Question 1
    for eta in etas:
        base_lr = FixedLR(eta)
        l1_norm_model = L1(weights=copy.deepcopy(init))
        l2_norm_model = L2(weights=copy.deepcopy(init))
        l1_gd_callback, l1_values, l1_weights = get_gd_state_recorder_callback()
        l2_gd_callback, l2_values, l2_weights = get_gd_state_recorder_callback()
    
        l1_gd_model = GradientDescent(learning_rate=base_lr, callback=l1_gd_callback, out_type="best")
        l2_gd_model = GradientDescent(learning_rate=base_lr, callback=l2_gd_callback, out_type="best")

        l1_gd_model.fit(l1_norm_model, None, None)
        l2_gd_model.fit(l2_norm_model, None, None)

        l1_fig = plot_descent_path(L1, np.array(l1_weights), f"L1 Norm Descent for eta = {eta}")
        l2_fig = plot_descent_path(L2, np.array(l2_weights), f"L2 Norm Descent for eta = {eta}")

        l1_fig.show()
        l2_fig.show()

        # Question 3
        l1_trace = go.Scatter(x=list(range(len(l1_values))), 
                                   y=l1_values, 
                                   mode='lines', 
                                   showlegend=False)
        l1_layout= go.Layout(
                    title=f'Convergence Rate for eta = {eta}', 
                    xaxis_title=f'Iterations', 
                    yaxis_title='L1 Norm')
        
        l2_trace = go.Scatter(x=list(range(len(l2_values))), 
                                   y=l2_values, 
                                   mode='lines', 
                                   showlegend=False)
        l2_layout= go.Layout(
                    title=f'Convergence Rate for eta = {eta}', 
                    xaxis_title=f'Iterations', 
                    yaxis_title='L2 Norm')
        
        l1_convergence_fig = go.Figure(data=[l1_trace], layout=l1_layout)
        l2_convergence_fig = go.Figure(data=[l2_trace], layout=l2_layout)

        l1_convergence_fig.show()
        l2_convergence_fig.show()

        # Question 4
        loss_values.append({"eta": eta, "norm": "L1", "value": np.min(l1_values)})
        loss_values.append({"eta": eta, "norm": "L2", "value": np.min(l2_values)})

    # Find the minimum values for L1 and L2 norms
    min_l1 = min(filter(lambda x: x["norm"] == "L1", loss_values), key=lambda x: x["value"])
    min_l2 = min(filter(lambda x: x["norm"] == "L2", loss_values), key=lambda x: x["value"])

    # Print the results
    print("Min Loss with L1 Norm for eta={}: {}".format(min_l1["eta"], min_l1["value"]))
    print("Min Loss with L2 Norm for eta={}: {}".format(min_l2["eta"], min_l2["value"]))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()

def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Necessary imports for this function
    from sklearn.metrics import roc_curve, auc
    from IMLearn.model_selection import cross_validate
    from IMLearn.metrics import misclassification_error
    
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Question 8 - Plotting convergence rate of logistic regression over SA heart disease data
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    fpr, tpr, thresholds = roc_curve(y_test, lr_model.predict_proba(X_test))

    go.Figure(
    data=[go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(color="black", dash='dash'), name="Random Class Assignment"),
          go.Scatter(x=fpr, y=tpr, mode='markers+lines',text=thresholds, name="", showlegend=False, marker_size=5,
                     hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
    layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$", 
                                 xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                                 yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()
    # Question 9
    alpha_star = thresholds[np.argmax(tpr - fpr)]
    lr_model_alpha_star = LogisticRegression(alpha=alpha_star)
    lr_model_alpha_star.fit(X_train, y_train)
    test_error_alpha_star = lr_model_alpha_star.loss(X_test, y_test)
    
    print("Best alpha is: {}\nTest Error for Logistic Regression is: {}"
          .format(np.round(alpha_star, 4), np.round(test_error_alpha_star, 4)))

    
    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    # Question 10 - L1
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    max_iter = 20000
    cutoff_alpha = 0.5
    lr=1e-4
    l1_train_err, l1_validation_err = np.zeros(len(lambdas)), np.zeros(len(lambdas))
    for i in range(len(lambdas)):
        gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)
        lr_model_lambda = LogisticRegression(solver=gd, penalty="l1", lam=lambdas[i], alpha=cutoff_alpha)
        l1_train_err[i], l1_validation_err[i] = cross_validate(lr_model_lambda, X_train.to_numpy(), y_train.to_numpy(), misclassification_error)

    l1_best_lambda = lambdas[np.argmin(l1_validation_err)]
    
    l1_best_gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)
    l1_best_model = LogisticRegression(solver=l1_best_gd, penalty="l1", lam=l1_best_lambda, alpha=cutoff_alpha)
    
    l1_best_model.fit(X_train, y_train)
    test_error_l1_best_model = l1_best_model.loss(X_test, y_test)

    print("L1 - Best lambda is: {}, Test Error is: {}"
        .format(np.round(l1_best_lambda, 4), np.round(test_error_l1_best_model, 4)))
    

    # Question 11 - L2
    l2_train_err, l2_validation_err = np.zeros(len(lambdas)), np.zeros(len(lambdas))
    for i in range(len(lambdas)):
        gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)
        lr_model_lambda = LogisticRegression(solver=gd, penalty="l2", lam=lambdas[i], alpha=cutoff_alpha)
        l2_train_err[i], l2_validation_err[i] = cross_validate(lr_model_lambda, X_train.to_numpy(), y_train.to_numpy(), misclassification_error)

    l2_best_lambda = lambdas[np.argmin(l2_validation_err)]
    
    l2_best_gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)
    l2_best_model = LogisticRegression(solver=l2_best_gd, penalty="l2", lam=l2_best_lambda, alpha=cutoff_alpha)
    
    l2_best_model.fit(X_train, y_train)
    test_error_l2_best_model = l2_best_model.loss(X_test, y_test)

    print("L2 - Best lambda is: {}, Test Error is: {}"
        .format(np.round(l2_best_lambda, 4), np.round(test_error_l2_best_model, 4)))

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
