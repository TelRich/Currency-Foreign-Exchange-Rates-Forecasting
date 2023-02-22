# Importing required library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as po
po.init_notebook_mode(connected=True)
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from pylab import rcParams
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

"""Loading the data"""
data = pd.read_csv("forex.csv")

"""data preview"""
def data_preview():
    return data.head()

"""
 The function below split the exchange currency in the slug column 
and separate the left and right currency to a new column respectively.
"""
def slug_split():
    global data
    data["A"]=[str(a).split("/")[0] for a in data["slug"]]
    data["B"]=[str(a).split("/")[1] for a in data["slug"]]
    return data


"""
The function below displays the unique currency,left and right after splitting.
"""
def uniq_cur():
    data = slug_split()
    curA = data["A"].unique()
    curB = data['B'].unique()
    return curA, curB

"""
The function below, with the help of the Ohlc method from Plotly graphs objects, plot the open, high, low and 
close of a currency pair(exchange) If Ohlc is not set to True, the function will return the first five row of    
the data frame. Because of the high number of unique currencies on the 
right, start and end was implemented to help slice the number of visuals to display.E.g if you select USD >> USD/x. 
There are more than 15 currencies for 
x, thus you may want to display just a few to save memory and reduce running time.
"""
def plot_viz(currencyA, ohlc=False, start = 0, end=1):
    data = slug_split()
    cur = str(currencyA)
    curA = data[data['A'] == cur] 
    curA = curA.set_index(pd.to_datetime(curA.date))
    xch = curA['slug'].unique().tolist()
    xch_lst = [curA[curA['slug'] == x] for x in xch]
    if ohlc == True:
        for df in xch_lst[start:end]:
            name = df['slug'].unique()[0]
            fig = go.Figure()
            fig.add_trace(go.Ohlc(x=df.index, 
                    open=df.open,
                    high=df.high,
                    low=df.low,
                    close=df.close,
                    name='Price',
                    showlegend=True))
            fig.update(layout_xaxis_rangeslider_visible=False, layout_width=1000,
                    layout_title=f'{name} Candle Stick Chart', layout_yaxis_title='Open, High, Low, and Close')
            fig.show()
    else:
        return curA.head()


"""
The function below processes currency A inserted by the user.
It filters the main data for that currency, and displays the possible 
currency B that you can choose for the selected currency A. You can preview 
the data frame by setting prev to True, then using index 0 to display it.
"""
def select_curA(prev=False):
    data = slug_split()
    curr_A = input()
    A_data = data[data["A"]==curr_A]
    preview = A_data.head()
    pos_curB = A_data["B"].unique()
    print(f"Below are the possible currency B for {curr_A} \n {'='*100} {pos_curB}")
    if prev == True:
        return preview, A_data
    else:
        return A_data

"""
The function below forms a dictionary of data frame for currency A 
selected above, with all possible currency B. It prints the shape of 
the data frame if the shape is set o True. x is used to control the 
number of data frame shapes it displays.
"""
def posB_dfs(A_data, shape=False, x=5):
    Alist=[]
    for b in A_data["B"].unique():
        Alist.append(A_data[A_data["B"]==b])
    cur_map = {x:y for y,x in zip(Alist,A_data["B"].unique())}
    
    if shape == True:
        for b in A_data["B"].unique()[:x]:
            print(b,A_data[A_data["B"]==b].shape)
    return cur_map

# setting start date and end date for global use
start_date = 0
end_date = 0

"""
The function below takes currency B from the user and filters the dictionary 
from above and return the data frame for the currency pair.
"""
def select_curB(curA, prev=False):
    cur_B=input()
    preview = curA[cur_B].head()
    A_B = curA[cur_B]
    #Check the number of missing entries
    A_B = A_B.set_index(A_B.date)
    A_B.index = pd.to_datetime(A_B.date)
    global start_date, end_date
    start_date = A_B.iloc[0].date
    end_date = A_B.iloc[-1].date
    print(f"Missing entries \n{'='*100}")
    print(pd.date_range(start=start_date, end=end_date).difference(A_B.index))
    if prev == True:
        return preview, A_B
    else:
        return A_B

"""
The function below is used to resample the data frame to weekly, showing the mean.
It also prints out the days with no entries. Because of the data leakage, only the close
variable will be used for prediction, thus missing value for only the close variable was handled.
"""
def upsample(A_B):
    xch = A_B['slug'].unique()
    #downsample to weekly records using mean
    weekly = A_B.resample('W', label='left',closed = 'left').mean()
    print(pd.date_range(start=start_date, end=end_date,freq="W").difference(weekly.index))
    #use ffil to fill null values
    weekly["close"]=weekly["close"].ffill()
    print('='*20)
    print(weekly.isnull().sum())
    print('='*20)
    print(f'Weekly Data Shape: {weekly.shape}')
    return weekly, xch
        
"""
The function below plots the weekly data showing the time series. 
With closing, kde, and auto_corr set to True, it will plot the closing value,
density plot and auto correlation plot for the close feature.
"""
cur_name = ''
# Plot all features to check if any independent features are present
def weekly_plot(week_data, closing=False, kde=False, auto_corr=False):
    weekly = week_data[0]
    global cur_name
    cur_name = week_data[1][0]
    df_close = weekly['close']
    fig = go.Figure()
    fig.add_trace(go.Ohlc(x=weekly.index,
                        open=weekly['open'],
                        high=weekly['high'],
                        low=weekly['low'],
                        close=weekly['close'],
                        name='Price',
                        showlegend=True))
    fig.update_layout(width=1000, title=f'{week_data[1][0]} Candle Stick Chart',
                      yaxis_title='Open, High, Low, and Close')
    fig.show()
    
    if closing == True:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weekly.index, y=weekly['close'], name='Closing value'
        ))
        fig.update_layout(title=f'{week_data[1][0]} closing price over the years', showlegend=True,
                          yaxis_title='Closing Price')
        fig.show()
        
    if kde == True:
        rcParams['figure.figsize'] = 10, 6
        # Analyse the KDE plot of the time series to check for shape, spread, modes and outliers
        df_close.plot(kind='kde')
        plt.title('Close price density plot')
        plt.show()
        
    if auto_corr == True:
        rcParams['figure.figsize'] = 10, 6
        #check for autocorrelation with historic values
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(df_close)
        plt.title('Close price auto correlation plot')
        plt.show()


"""
The function below plots the rolling mean and rolling stationarity of the series.
It also prints the Dickey Fuller test statistics.
"""    
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(52).mean()
    rolstd = timeseries.rolling(52).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeseries.index, y=timeseries.values, name='Original'
    ))
    fig.add_trace(go.Scatter(
        x=rolmean.index, y=rolmean.values, name='Rolling Mean'
    ))
    fig.add_trace(go.Scatter(
        x=rolstd.index, y=rolstd.values, name='Rolling Std'
    ))
    fig.update_layout(title=f'{cur_name} Rolling Mean and Standard Deviation', width=900,
                      yaxis_title='Closing Price')
    fig.show()
    
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    # hence we manually write what values it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)

"""The function below plot the seasonal decompose of the weekly closing price."""
def seasonal_decomp(df_close):
    result = seasonal_decompose(df_close, model='multiplicative')
    fig = plt.figure()  
    fig = result.plot()  
    fig.set_size_inches(16, 9)
    plt.show()

"""
The function below converts the non-stationary series to stationary series.
It plots the chart to display the new rolling mean and standard deviation 
"""
def conv_to_statn(df_close):
    df_log = np.log(df_close)
    moving_avg = df_log.rolling(52).mean()
    std_dev = df_log.rolling(52).std()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=moving_avg.index, y=moving_avg.values, name='Mean'
    ))
    fig.add_trace(go.Scatter(
        x=std_dev.index, y=std_dev.values, name='Standard Deviation'
    ))
    fig.update_layout(title='Moving Average and Standard average', width=900,
                      yaxis_title = 'Closing Price')
    fig.show()
    return df_log

"""
The function below splits the data and plots the visual. It also returns the train and test data.
"""
def train_test_split(df_log):
    train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_log.index, y=df_log.values, name='Train data'
    ))
    fig.add_trace(go.Scatter(
        x=test_data.index, y=test_data.values, name='Test data'
    ))
    fig.update_layout(width=900, yaxis_title='Closing Price', title=f'{cur_name} Train data and Test Data')
    fig.show()
    return train_data, test_data

"""
The function below takes in the train data and prints the autoARIMA summary.
It also plots the diagnostics of the model.
"""
p = 0
d = 0
q = 0
def aut_arima(train_data):
    #create an instance of Auto arima
    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=False,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
    print(model_autoARIMA.summary())
    model_autoARIMA.plot_diagnostics(figsize=(15,8))
    global p, d, q
    p, d, q = model_autoARIMA.order
    plt.show()
    
"""
The function below takes in the train and test data which is used in the ARIMA model
for forecasting. It also prints out the model Summary and displays the data plot. 
It prints out the model evaluation at the end.
"""
A_MAPE = 0
P_MAPE = 0

def arima(train_data, test_data, plot=False):
    model = ARIMA(train_data, order=(p,d,q))  
    fitted = model.fit()  
    print(fitted.summary())
    samples=len(test_data)
    fc=fitted.forecast(samples, alpha=0.05)
    fc_series = pd.Series(fc, index=test_data.index)
    
    if plot == True:
        #plot predicted vs actual
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_data.index, y=train_data.values, name='Training data'
        ))
        fig.add_trace(go.Scatter(
            x=test_data.index, y=test_data.values, name='Actual Forex rates'
        ))
        fig.add_trace(go.Scatter(
            x=fc_series.index, y=fc_series.values, name='Predicted Forex rates'
        ))
        fig.update_layout(title=f'{cur_name} Train, Actual, and Prediction Chart', width=900,
                          yaxis_title='Closing Price')
        fig.show()
        
        mse = mean_squared_error(test_data, fc)
        print('MSE: '+str(mse))
        mae = mean_absolute_error(test_data, fc)
        print('MAE: '+str(mae))
        rmse = math.sqrt(mean_squared_error(test_data, fc))
        print('RMSE: '+str(rmse))
        mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
        global A_MAPE
        A_MAPE = mape
        print('MAPE: '+str(mape))

"""
The function below takes in the train and test data which is used in the prophet model
for forecasting. It plots the splitting and prediction chart the displays the model evaluation
at the end.
"""
def prophet_model(train_data, test_data, plot=False):
    train_df=pd.DataFrame(train_data)
    train_df["ds"]=train_df.index
    train_df["y"]=train_df["close"]
    model = Prophet(seasonality_mode='additive', weekly_seasonality=True, daily_seasonality=True)
    model.fit(train_df)
    future = model.make_future_dataframe(periods=len(test_data), freq='W-SUN',include_history=False)
    forecast = model.predict(future)
    fs=pd.Series(forecast["yhat"])
    fs.index=forecast.ds
    
    if plot == True:
        train_data1 = np.exp(train_data)
        test_data1 = np.exp(test_data)
        fs1 = np.exp(fs)
        #plot predicted vs actual
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_data1.index, y=train_data1.values, name='Training data'
        ))
        fig.add_trace(go.Scatter(
            x=test_data1.index, y=test_data1.values, name='Actual Forex rates'
        ))
        fig.add_trace(go.Scatter(
            x=fs1.index, y=fs1.values, name='Predicted Forex rates'
        ))
        fig.update_layout(title=f'{cur_name} Train, Actual, and Prediction chart', width=900,
                          yaxis_title='Closing Price')
        fig.show()
    
    mse = mean_squared_error(test_data, fs)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(test_data, fs)
    print('MAE: '+str(mae))
    rmse = math.sqrt(mean_squared_error(test_data, fs))
    print('RMSE: '+str(rmse))
    mape = np.mean(np.abs(fs - test_data)/np.abs(test_data))
    global P_MAPE
    P_MAPE = mape
    print('MAPE: '+str(mape))
    
    
def performance():
    diff = A_MAPE-P_MAPE
    pct = (diff/A_MAPE)*100
    if pct < 0:
        print(f'With Prophet Model, ARIMA error {round(A_MAPE, 5)} increased by {round(abs(pct), 2)}% ({round(P_MAPE, 5)})')
    else:
        print(f'With Prophet Model, ARIMA error {round(A_MAPE, 5)} reduced by {round(abs(pct), 2)}% ({round(P_MAPE, 5)})')