import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).drop_duplicates()

    # Check valid data for feature
    df = df[(df['Temp'] > -15) & (df['Temp'] < 50) & (df['Year'] > 0)]
    
    # Check features with range
    df = df[df['Month'].isin(range(1,13)) & df['Day'].isin(range(1,32))]

    # Add DayOfYear column
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    df['Year'] = df['Year'].astype(str)

    return df

if __name__ == '__main__':
    np.random.seed(0)
    output_path = 'ex2_plots'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    isreal_subset = X[X['Country'] == 'Israel']

    px.scatter(isreal_subset, x='DayOfYear', y='Temp', color='Year')\
     .write_image(output_path + '/q2_Israel_Temperature.png')
    
    grouped_by_month = isreal_subset.groupby('Month')
    std_of_temp = grouped_by_month['Temp'].agg(['std'])

    px.bar(std_of_temp.reset_index(), x='Month', y='std',
           title='Standard Deviation of the Daily Temperatures for Each Month')\
            .write_image(output_path + '/q2_Standard_Deviation_by_Month.png')

    # Question 3 - Exploring differences between countries
    grouped_by_month_country = X.groupby(['Month','Country'])
    avg_and_std = grouped_by_month_country['Temp'].agg(['std','mean'])

    fig = px.line(avg_and_std.reset_index(), x='Month', y='mean', error_y='std',
                    color='Country', line_group='Country')
    fig.update_yaxes(title_text='Average Temperature')
    fig.update_layout(title='Average Monthly Temperatures in Different Countries')
    fig.write_image(output_path + '/q3_Average_Monthly_Temperatures.png')

    # Question 4 - Fitting model for different values of `k`
    temp = isreal_subset['Temp']
    isreal_subset.drop(columns=['Temp'], inplace=True)
    train_X, train_y, test_X, test_y = split_train_test(isreal_subset, temp)
    loss = []
    for k in range(1, 11):
        model = PolynomialFitting(k).fit(train_X['DayOfYear'], train_y)
        test_err = round(model.loss(test_X['DayOfYear'], test_y), 2)
        loss.append(test_err)
        print(f'Loss for the value {k} - {test_err}')

    fig = px.bar(x=list(range(1,11)), y=loss,  text_auto=True,
           title='Test Error as a Function of the k Value')
    fig.update_yaxes(title_text='Loss')
    fig.update_xaxes(title_text='Value of K')
    fig.write_image(output_path + '/q4_Loss_for_k_values.png')

    # Question 5 - Evaluating fitted model on different countries
    countries = X['Country'].unique()
    countries = np.delete(countries, np.where(countries == 'Israel'))
    loss = []
    model = PolynomialFitting(5).fit(isreal_subset['DayOfYear'], temp)
    for country in countries:
        country_subset = X[(X['Country'] == country)]
        country_loss = round(model.loss(country_subset['DayOfYear'], country_subset['Temp']), 2)
        loss.append(country_loss)

    fig = px.bar(x=countries, y=loss, text_auto=True,
           title='Calculated Loss for Selected Countries of Fitted Model over Israel')
    fig.update_yaxes(title_text='Loss')
    fig.update_xaxes(title_text='Country')
    fig.write_image(output_path + '/q5_Loss_over_Israel_model.png')
