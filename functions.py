# import models
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as po
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8
from prophet import Prophet

# load data
df = pd.read_csv('forex.csv', parse_dates=['date']).set_index('date')

# function to view 113 unique currency 
def view_cur():
    uniq = df['currency'].unique()
    return uniq

# function to filter for a currency
def select(cur: str):
    currency = df[df['currency'] == cur]
    uniq_xchng= currency['slug'].nunique()
    print(f'The number of unique exchange (slug) is {uniq_xchng}')
    return currency

"""
funtion to create a dataframe of each exchange in the slug colum for a selected currency.
it also plot the ohlc of each exchange if ohlc is set to true.

example: if you selected EUR as the currency. EUR has 6 xchange INR/EUR, AUD/EUR,
JPY/EUR, CHF/EUR, USD/EUR, and GBP/EUR. The below function will filter for each of
this xchange and return a dataframe for each xchange, all in a list. 
So list[0] = INR/EUR > dataframe 
If ohlc is set to True, the ohlc plot for each xchange will be displayed
"""
df_map = []
def slug_df_lst(cur_df, ohlc=False, start=0, end=0):
    uniq_xchng = cur_df['slug'].unique()
    global df_map
    df_map = cur_df['slug'].unique().tolist()
    slug_df_lst = [cur_df[cur_df['slug'] == x ]for x in uniq_xchng]
    if ohlc == True:
        x = 0
        for df in slug_df_lst[start:end]:
                fig = go.Figure()
                fig.add_trace(go.Ohlc(x=df.index, 
                        open=df.open,
                        high=df.high,
                        low=df.low,
                        close=df.close,
                        name='Price'))
                fig.update(layout_xaxis_rangeslider_visible=False, layout_width=1000,
                        layout_title=df_map[x])
                x += 1
                fig.show()
    return slug_df_lst

"""
The function below plot the seasonal decomposition for each exchange.
"""
def seasonal_decompos(slug_df_lst):
    wkly_df_lst = [df.resample('W').mean().ffill() for df in slug_df_lst]
    x = 0
    for wkly_df in wkly_df_lst:
        decompose_series = sm.tsa.seasonal_decompose(wkly_df['close'], model='multiplicative')
        decompose_series.plot()
        print(df_map[x])
        x += 1
        return plt.show()
        
"""
The function below checks for stationarity in the time series.
It displays ADFuller Test Statistics for the first xchange in the list,
then print results for the rest of the xchange
"""    
def stationary_check(lst):
    adf_result = adfuller(lst[0]['close'])
    print(f'ADF Statistics: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    print(f'No. of lags used: {adf_result[2]}')
    print(f'No. of observations used: {adf_result[3]}')
    print('Critical Values')
    for k,v in adf_result[4].items():
        print(f'  {k}: {v}')
    
    print("="*35)
    
    x = 0
    for df in lst:
        adf_result = adfuller(df['close'])
        if adf_result[1] < 0.05:
            print(f'For {df_map[x]} the series is Stationary')
        else:
            print(f'For {df_map[x]} the series is Non-Stationary')
        x += 1

"""
The function below converts non-stationary series to stationary series.
In the process, all xchanges are transform using numpy log.
If plot is set to true, it will display the stationnary time series for the 
first xchange. To view other xchange plots, alter the value of x.
"""
def convt_to_stat(lst, plot=False, x=0):
    stat_df_lst = []
    for df in lst:
        dd = df.drop(columns=['slug', 'currency'])
        adf_result = adfuller(dd['close'])
        if adf_result[1] > 0.05:
            df_log = np.log(dd)
            df_diff = df_log.diff().bfill()
            stat_df_lst.append(df_diff)
        else:
            stat_df_lst.append(dd)
            
    if plot == True:
        plt.plot(stat_df_lst[x].index, stat_df_lst[x].close, '-')
        plt.plot(stat_df_lst[x].rolling(12).mean(), color='blue')
        plt.show()
    return stat_df_lst

"""
Functions to ealuate the model
"""
def MAPE(y, y_hat):
        return np.mean(np.abs((y - y_hat)/y)) * 100

def RMSE(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))

"""
Funtion to perform a univariate forecast and evaluate performance
"""
def uni_forecast(slug_df_lst, slug=0, plot=False, plot_comp=False):
    print(f'Forecast for {df_map[slug]} exchange')
    print('='*40)
    new = slug_df_lst[slug].reset_index()[['date', 'close']]
    new = new.rename(columns={'date': 'ds', 'close': 'y'})
    
    train_data = new.sample(frac=0.7, random_state=2)
    test_data = new.drop(train_data.index)
    print(f'Training data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')
    
    model = Prophet(seasonality_mode='multiplicative', daily_seasonality=True,)
    model.fit(train_data)
    
    future = test_data[['ds']]
    forecast = model.predict(future)
    print(f'Forecast data shape: {forecast.shape}')
    
    if plot == True:
        model.plot(forecast)
        plt.show()
    if plot_comp == True:
        model.plot_components(forecast)
        plt.show()

    mape = str(round(MAPE(test_data['y'], forecast['yhat']),2)) + "%"
    rmse = round(RMSE(test_data['y'], forecast['yhat']), 5)
    print(f'MAPE: {mape}')
    print(f'RMSE: {rmse}')
    
"""
Function to perform a multivariate forecaast and evaluate model
"""    
def mul_forecast(slug_df_lst, slug=0, plot=False, plot_comp=False):
    new_mul = slug_df_lst[slug].reset_index().drop(columns=['slug', 'currency', 'open'])
    new_mul = new_mul.rename(columns={'date': 'ds', 'high': 'add1', 
                                    'low': 'add2', 'close': 'y'})
    
    mul_train_data = new_mul.sample(frac=0.7, random_state=2)
    mul_test_data = new_mul.drop(mul_train_data.index)
    
    n_model = Prophet(seasonality_mode='multiplicative', daily_seasonality=True,)
    n_model.add_regressor('add1')
    n_model.add_regressor('add2')
    n_model.fit(mul_train_data)
    future = mul_test_data.drop('y', axis=1)
    n_forecast = n_model.predict(future)
    
    if plot == True:
        n_model.plot(n_forecast)
        plt.show()
    if plot_comp == True:
        n_model.plot_components(n_forecast)
        plt.show()
    
    mape = str(round(MAPE(mul_test_data['y'], n_forecast['yhat']),2)) + "%"
    rmse = round(RMSE(mul_test_data['y'], n_forecast['yhat']), 5)
    print(f'MAPE: {mape}')
    print(f'RMSE: {rmse}')