from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    ug = UnivariateGaussian().fit(samples)
    print((round(ug.mu_, 3), round(ug.var_, 3)))

    # Question 2 - Empirically showing sample mean is consistent
    q2_mat = np.zeros((2, 100))
    for i in range(1, 101):
        distance_calc = abs(ug.fit(samples[0:(i*10)]).mu_ - 10)
        q2_mat[0][i - 1] = distance_calc
        q2_mat[1][i - 1] = i * 10

    go.Figure(go.Scatter(x=q2_mat[1], y=q2_mat[0], mode='markers'),
                layout=go.Layout(
                title=r"$\text{Absolute Distance Between Estimated and True Expectation}$", 
                xaxis_title="$\\text{Number of Samples}$", 
                yaxis_title="r$|\hat\mu - \mu|$",
                height=400, width=700)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    samples_pdf = ug.pdf(samples)
    go.Figure(go.Scatter(x=samples, y=samples_pdf, mode='markers'),
                layout=go.Layout(
                title=r"$\text{Probability Density Function of The Normal Distribution}$", 
                xaxis_title="$\\text{X}$", 
                yaxis_title="r$PDF (X)$",
                height=400, width=700)).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    cov = np.array([[1, 0.2, 0, 0.5], 
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, cov, 1000)
    mg = MultivariateGaussian().fit(samples)
    print(np.round(mg.mu_, 3))
    print(np.round(mg.cov_, 3))

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    ll_values = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            ll_ij = mg.log_likelihood(np.array([f1[i], 0, f3[j], 0]), cov, samples)
            ll_values[i][j] = ll_ij
    
    go.Figure(go.Heatmap(x=f1, y=f3,z=ll_values), 
                layout=go.Layout(
                title="Multivariate Gaussian Log-Likelihood With Expectation: (f1,0,f3,0)", 
                xaxis_title="f1 value", 
                yaxis_title="f3 value",
                height=700, width=800)).show()
    
    # Question 6 - Maximum likelihood
    max_index = np.argmax(ll_values)
    max_row_index = max_index // ll_values.shape[1]
    max_col_index = max_index % ll_values.shape[1]

    print((round(f1[max_row_index], 3), round(f3[max_col_index], 3)))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
