from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional, Tuple
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

# global variables to use in data prepreocess after learning
columns_averages = {}
columns_order = None

numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 
                    'sqft_lot', 'floors', 'sqft_above',
                    'sqft_basement', 'yr_built', 'yr_renovated']
features_to_drop = ['id', 'date', 'lat', 'long']


def preprocess_data_learning(X: pd.DataFrame, y: pd.Series = None):
    global columns_averages, columns_order
    # Drop irrelevant rows
    X = X.dropna()
    
    # Make sure the response vector is numeric
    y = pd.to_numeric(y)

    # Check valid data for conditions features
    valid_data_conditions = (X['bedrooms'] > 0) & (X['bathrooms'] >= 0) & (X['sqft_living15'] > 0) &\
                            (X['sqft_lot15'] > 0) & (X['sqft_living'] > 0) & (X['sqft_lot'] >= 0) &\
                            (X['floors'] >= 0) & (X['sqft_above'] > 0) & (X['sqft_basement'] >= 0) &\
                            (X['yr_built'] > 0) & (X['yr_renovated'] >= 0)
    X = X[valid_data_conditions]
    
    # Check features with range
    range_conditions = X['waterfront'].isin([0,1]) & X['view'].isin(range(5)) & \
                       X['condition'].isin(range(1,6)) & X['grade'].isin(range(1,14))
    X = X[range_conditions]

    # Bound features to avoid abnormal samples
    X = X[X['bedrooms'] <= 15]
    X = X[X['sqft_living'] > 10]
    X = X[X['sqft_lot'] > 10]

    # Check valid prices
    y = y.loc[X.index]
    y = y[(y > 0)]
    X = X.loc[y.index]

    # Save the average value for each feature in training set
    columns_averages = X.mean()

    # Handle categorical features
    X['zipcode'] = X['zipcode'].astype(int)
    X = pd.get_dummies(X, prefix='zipcode', columns=['zipcode'])
    
    # Save the order of the X columns
    columns_order = X.columns

    return X, y
    
def preprocess_data_test(X: pd.DataFrame):
    global columns_averages, columns_order
    # Handle categorical features
    X['zipcode'].fillna(0, inplace=True)
    X['zipcode'] = X['zipcode'].astype(int)
    X = pd.get_dummies(X, prefix='zipcode', columns=['zipcode'])

    # Re-oreder X columns
    X = X.reindex(columns=columns_order, fill_value=0)

    # Change invalid features in the test set with their average value in the training set
    for index, row in X.iterrows():
        for feature in X.columns:
            if feature in ['bedrooms', 'sqft_living', 'sqft_above', 
                           'yr_built', 'sqft_living15', 'sqft_lot15']:
                if not (row[feature] > 0):
                    X.loc[index, feature] = columns_averages[feature] or 1

            if feature in ['bathrooms', 'sqft_lot', 'floors', 
                           'sqft_basement', 'yr_renovated']:
                if not (row[feature] >= 0):
                    X.loc[index, feature] = columns_averages[feature] or 0
        
        if not (row['waterfront'] in [0,1]):
            X.loc[index, 'waterfront'] = round(columns_averages['waterfront']) or 0
        
        if not (row['view'] in range(5)):
            X.loc[index, 'view'] = round(columns_averages['view']) or 0
        
        if not (row['condition'] in range(1,6)):
            X.loc[index, 'condition'] = round(columns_averages['condition']) or 1
        
        if not (row['grade'] in range(1,14)):
            X.loc[index, 'grade'] = round(columns_averages['grade']) or 1

    return X

def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # Drop irrelevant features
    X = X.drop(features_to_drop, axis=1)

    # Make sure the numeric features are indeed numeric
    X[numeric_features] = X[numeric_features].apply(pd.to_numeric)

    if y is None:
        return preprocess_data_test(X)
    
    return preprocess_data_learning(X, y)

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_y = np.std(y)
    for column in X.columns:
        if not column.startswith('zipcode_'):
            # Calculate the Pearson Correlation for the current feature
            pearson_corr = np.cov(X[column], y)[0,1] / (np.std(X[column]) * std_y)
            
            # Calculate the trendline using Linear Regression
            feature = X[column]
            coefs = LinearRegression(include_intercept=True).fit(feature, y).coefs_
            trendline = coefs[0] + feature * coefs[1]

            # Plot the result
            scatter_trace = go.Scatter(x=feature, y=y, mode='markers', showlegend=False)
            trendline_trace = go.Scatter(x=feature, y=trendline, mode='lines', 
                                        line=dict(color='orange'), name='Trendline', showlegend=False)
            layout= go.Layout(
                        title=f'Pearson Correlation Between {column} and the Response<br>Pearson Correlation is {pearson_corr}', 
                        xaxis_title=f'Values of {column}', 
                        yaxis_title='Response')
            fig = go.Figure(data=[scatter_trace, trendline_trace], layout=layout)\
                    .write_image(output_path + f'/Pearson_Correlation_{column}.png')
            
def remove_test_samples_with__invalid_response(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    y = y.dropna()
    X = X.loc[y.index]
    
    return X, y

if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("datasets/house_prices.csv")
    output_path = 'ex2_plots'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Question 1 - split data into train and test sets
    price = df['price']
    df.drop(columns=['price'], inplace=True)
    train_X, train_y, test_X, test_y = split_train_test(df, price)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y, output_path)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    test_X, test_y = remove_test_samples_with__invalid_response(test_X, test_y)
    test_X = preprocess_data(test_X)
    values = np.zeros((101, 10))
    for p in range(10,101):
        for i in range(10):
            sampled_train_X = train_X.sample(frac=(p/100))
            sampled_train_y = train_y.loc[sampled_train_X.index]
            model = LinearRegression(include_intercept=True).fit(sampled_train_X, sampled_train_y)
            values[p-10, i] = model.loss(test_X, test_y)
    
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    percentage_values = list(range(10,101))
    upper = go.Scatter(x=percentage_values, y=mean-2*std, fill=None, mode='lines', 
                            line=dict(color='orange'), showlegend=False)
    lower = go.Scatter(x=percentage_values, y=mean+2*std, fill='tonexty', mode='lines', 
                        line=dict(color='orange'), showlegend=False)
    mean = go.Scatter(x=percentage_values, y=mean, mode='markers+lines', 
                        line=dict(color='blue'), showlegend=False)
    layout = go.Layout(
                title=f'Effect of Training Size on Test MSE', 
                xaxis_title=f'Training Data Percentage', 
                yaxis_title='Test Data MSE')
    go.Figure(data=[upper, lower, mean], layout=layout)\
        .write_image(output_path + '/Effect_of_Training_Size_on_Test_MSE.png')
    