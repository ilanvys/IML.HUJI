
import math
import numpy as np
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian


def test_univariate_pdf():
    sample = np.random.normal(10, 1, 10000)
    ug = UnivariateGaussian().fit(sample)
    
    assert math.isclose(ug.mu_, 10, rel_tol=0.1)
    assert math.isclose(ug.var_, 1, rel_tol=0.1)

    # Test 1: Test the function with mean = 0, variance = 1, and x = 0
    ug.mu_ = 0
    ug.var_ = 1
    assert math.isclose(ug.pdf(np.array([0])), 0.3989422804014327, rel_tol=1e-9)
    
    # Test 2: Test the function with mean = 1, variance = 4, and x = 0
    ug.mu_ = 1
    ug.var_ = 4
    assert math.isclose(ug.pdf(np.array([0])), 0.17603266338214976, rel_tol=1e-9)
    
    # Test 3: Test the function with mean = 0, variance = 1, and x = 1
    ug.mu_ = 0
    ug.var_ = 1
    assert math.isclose(ug.pdf(np.array([1])), 0.24197072451914337, rel_tol=1e-9)
    
    # Test 4: Test the function with mean = -1, variance = 0.25, and x = -0.5
    ug.mu_ = -1
    ug.var_ = 0.25
    assert math.isclose(ug.pdf(np.array([-0.5])), 0.4839414490382867, rel_tol=1e-9)
    
    # Test 5: Test the function with mean = 2, variance = 1, and x = 2
    ug.mu_ = 2
    ug.var_ = 1
    assert math.isclose(ug.pdf(np.array([2])), 0.3989422804014327, rel_tol=1e-9)
    
    # Test 6: Test the function with mean = -1, variance = 4, and x = -3
    ug.mu_ = -1
    ug.var_ = 4
    assert math.isclose(ug.pdf(np.array([-3])), 0.120985362259571675, rel_tol=1e-9)

def test_log_likelihood():
    ug = UnivariateGaussian()
    # Test 1: Test the function with mean = 0, variance = 1, and data = [0, 1, 2]
    mu = 0
    sigma = 1
    X = np.array([0, 1, 2])
    assert math.isclose(ug.log_likelihood(mu, sigma, X), -5.2568155996140185, rel_tol=0.001)
    
    # Test 2: Test the function with mean = -1, variance = 4, and data = [0, 0, 0, 0]
    mu = -1
    sigma = 2
    X = np.array([0, 0, 0, 0])
    assert math.isclose(ug.log_likelihood(mu, sigma, X), -6.0620484939385815, rel_tol=0.001)
    
    # Test 3: Test the function with mean = 2, variance = 1, and data = [3, 3, 3]
    mu = 2
    sigma = 1
    X = np.array([3, 3, 3])
    assert math.isclose(ug.log_likelihood(mu, sigma, X), -4.2568155996140185, rel_tol=0.001)
    # Test 4: Test the function with mean = -1, variance = 0.25, and data = [-1, -1, -1, -1]
    mu = -1
    sigma = 0.5
    X = np.array([-1, -1, -1, -1])
    assert math.isclose(ug.log_likelihood(mu, sigma, X), -2.2894597716988003, rel_tol=0.001)
    
    # Test 5: Test the function with mean = 0, variance = 1, and data = [0]
    mu = 0
    sigma = 1
    X = np.array([0])
    assert math.isclose(ug.log_likelihood(mu, sigma, X), -0.9189385332046727, rel_tol=0.001)
    
    # Test 6: Test the function with mean = -1, variance = 4, and data = [-2, -2, -2]
    mu = -1
    sigma = 2
    X = np.array([-2, -2, -2])
    assert math.isclose(ug.log_likelihood(mu, sigma, X), -4.546536370453936, rel_tol=0.001)

def test_multivariate_pdf():
    mg = MultivariateGaussian()
    mg.fitted_ = True
    # Test 1: Test the function with mean = [0, 0], covariance matrix = [[1, 0], [0, 1]], and data = [[0, 0], [1, 1]]
    mg.mu_ = np.array([0, 0])
    mg.cov_ = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 0], [1, 1]])
    assert np.allclose(mg.pdf(X), [0.15915494, 0.05854983])
    
    # Test 2: Test the function with mean = [1, 2], covariance matrix = [[2, 1], [1, 2]], and data = [[1, 2], [2, 3]]
    mg.mu_ = np.array([1, 2])
    mg.cov_ = np.array([[2, 1], [1, 2]])
    X = np.array([[1, 2], [2, 3]])
    assert np.allclose(mg.pdf(X), [0.09188815, 0.06584074])

    # Test 3: Test the function with mean = [0, 0], covariance matrix = [[1, 0.5], [0.5, 2]], and data = [[0, 0], [1, 1]]
    mg.mu_ = np.array([0, 0])
    mg.cov_ = np.array([[1, 0.5], [0.5, 2]])
    X = np.array([[0, 0], [1, 1]])
    assert np.allclose(mg.pdf(X), [0.12030983, 0.06794114])
    
    # Test 4: Test the function with mean = [-1, 1], covariance matrix = [[2, 1], [1, 3]], and data = [[-1, 1], [0, 2]]
    mg.mu_ = np.array([-1, 1])
    mg.cov_ = np.array([[2, 1], [1, 3]])
    X = np.array([[-1, 1], [0, 2]])
    assert np.allclose(mg.pdf(X), [0.07117625, 0.05272867])
    
    # Test 5: Test the function with mean = [1, 0], covariance matrix = [[1, 0], [0, 1]], and data = [[1, 0], [2, 0]]
    mg.mu_ = np.array([1, 0])
    mg.cov_ = np.array([[1, 0], [0, 1]])
    X = np.array([[1, 0], [2, 0]])
    assert np.allclose(mg.pdf(X), [0.15915494, 0.09653235])

def test_multivariate_log_likelihood():
    mg = MultivariateGaussian()

    # Test 1: Test the function with mean = [0, 0], covariance matrix = [[1, 0], [0, 1]], and data = [[0, 0], [1, 1]]
    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 0], [1, 1]])
    assert math.isclose(mg.log_likelihood(mu, cov, X), -4.675754132818691, rel_tol=0.001)

    # Test 2: Test the function with mean = [1, 2], covariance matrix = [[2, 1], [1, 2]], and data = [[1, 2], [2, 3]]
    mu = np.array([1, 2])
    cov = np.array([[2, 1], [1, 2]])
    X = np.array([[1, 2], [2, 3]])
    assert math.isclose(mg.log_likelihood(mu, cov, X), -5.107699754820134, rel_tol=0.001)

    # Test 3: Test the function with mean = [0, 0], covariance matrix = [[1, 0.5], [0.5, 2]], and data = [[0, 0], [1, 1]]
    mu = np.array([0, 0])
    cov = np.array([[1, 0.5], [0.5, 2]])
    X = np.array([[0, 0], [1, 1]])
    assert math.isclose(mg.log_likelihood(mu, cov, X), -4.8067984921826845, rel_tol=0.001)

    # Test 4: Test the function with mean = [-1, 1], covariance matrix = [[2, 1], [1, 3]], and data = [[-1, 1], [0, 2]]
    mu = np.array([-1, 1])
    cov = np.array([[2, 1], [1, 3]])
    X = np.array([[-1, 1], [0, 2]])
    assert math.isclose(mg.log_likelihood(mu, cov, X), -5.585192045252792, rel_tol=0.001)

    # Test 5: Test the function with mean = [1, 0], covariance matrix = [[1, 0], [0, 1]], and data = [[1, 0], [2, 0]]
    mu = np.array([1, 0])
    cov = np.array([[1, 0], [0, 1]])
    X = np.array([[1, 0], [2, 0]])
    assert math.isclose(mg.log_likelihood(mu, cov, X), -4.175754132818691, rel_tol=0.001)
      
if __name__ == "__main__":
    test_univariate_pdf()
    # test_log_likelihood() 
    # test_multivariate_pdf()
    # test_multivariate_log_likelihood()