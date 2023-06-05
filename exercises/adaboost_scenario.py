import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners)
    model.fit(train_X, train_y)

    values_arr = list(range(1, n_learners + 1))
    test_err = np.zeros(n_learners)
    train_err = np.zeros(n_learners)
    for i in range(n_learners):
        test_err[i] = model.partial_loss(test_X, test_y, i + 1)
        train_err[i] = model.partial_loss(train_X, train_y, i + 1)
    
    train_err_line = go.Scatter(x=values_arr, y=train_err, mode='lines', name="Train Loss")
    test_err_line = go.Scatter(x=values_arr, y=test_err, mode='lines', name="Test Loss")
    layout = go.Layout(
                title=f'Loss as a Function of The Number of Fitted Learners', 
                xaxis_title=f'Amount of Fitted Learners', 
                yaxis_title='Loss')
    go.Figure(data=[train_err_line, test_err_line], layout=layout)\
        .write_image(f"ex4_plots/q1_{noise}_AdaBoost_in_noiseless_case.png") #TODO: change to show
    
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    
    fig = make_subplots(rows=1, cols=4, subplot_titles=[rf"$\textbf{{Size: {t}}}$" for t in T],
                    horizontal_spacing = 0.01, vertical_spacing=.03)
    for index, value in enumerate(T):
        fig.add_traces([decision_surface(lambda X: model.partial_predict(X, value), lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                               marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "square"),
                                           line=dict(color="black", width=1)))], 
                    rows=1, cols=index+1)
        
    fig.update_layout(title=f"Decision Boundaries Of Models With Different Ensamble Size", 
                      margin=dict(t=100), height=350, width=1200)\
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"ex4_plots/q2_{noise}_AdaBoost_decision_surface.png") #TODO: change to show

    # Question 3: Decision surface of best performing ensemble
    min_err = np.argmin(test_err) + 1
    min_err_accuracy = round(1 - test_err[min_err - 1], 2)

    layout = go.Layout(title=rf"$\textbf{{Ensemble Size With Lowest Test Error. Size: {min_err}, Accuarcy: {min_err_accuracy}}}$",
                       xaxis=dict(visible=False), yaxis=dict(visible=False),
                       height=650, width=650)
    fig = go.Figure([decision_surface(lambda X: model.partial_predict(X, min_err), lims[0], lims[1], showscale=False),
                go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                           marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "square"),
                                           line=dict(color="black", width=1)))], layout=layout)
    fig.write_image(f"ex4_plots/q3_{noise}_best_score.png") #TODO: change to show

    # Question 4: Decision surface with weighted samples
    normalized_D = 5 * (model.D_/np.max(model.D_)) #TODO: 5 or 20
    
    layout = go.Layout(title=f"Samples Weights And Decision Surface of Biggest Ensemble Size",
                       xaxis=dict(visible=False), yaxis=dict(visible=False),
                       height=650, width=650)
    fig = go.Figure([decision_surface(model.predict, lims[0], lims[1], showscale=False),
                go.Scatter(x=train_X[:,0], y=train_X[:,1], mode="markers", showlegend=False,
                           marker=dict(color=train_y, size=normalized_D, 
                                        symbol=np.where(test_y == 1, "circle", "square"),
                                        line=dict(color="black", width=1)))], layout=layout)
    fig.write_image(f"ex4_plots/q4_{noise}_best_score.png") #TODO: change to show

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
