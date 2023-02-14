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

from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

data = pd.read_csv("forex.csv")

# data preview
def data_preview():
    return data.head()

# slug splitting
def slug_split():
    global data
    data["A"]=[str(a).split("/")[0] for a in data["slug"]]
    data["B"]=[str(a).split("/")[1] for a in data["slug"]]
    return data

# unique currencies
def uniq_cur():
    data = slug_split()
    curA = data["A"].unique()
    curB = data['B'].unique()
    return curA, curB

# Data Visualization
def plot_viz(currencyA, ohlc=False, start = 0, end=0):
    data = slug_split()
    cur = str(currencyA)
    curA = data[data['A'] == cur] 
    curA = curA.set_index(pd.to_datetime(curA.date))
    xch = curA['slug'].unique().tolist()
    xch_lst = [curA[curA['slug'] == x] for x in xch]
    if ohlc == True:
        x = 0
        for df in xch_lst[start:end]:
            fig = go.Figure()
            fig.add_trace(go.Ohlc(x=df.index, 
                    open=df.open,
                    high=df.high,
                    low=df.low,
                    close=df.close,
                    name='Price',
                    showlegend=True))
            fig.update(layout_xaxis_rangeslider_visible=False, layout_width=1000,
                    layout_title=xch[x])
            x += 1
            fig.show()
    else:
        return curA[0].head()

# select currency
def select_curA(prev=False):
    data = slug_split()
    curr_A = input()
    A_data = data[data["A"]==curr_A]
    preview = A_data.head()
    pos_curB = A_data["B"].unique()
    print(f"Possible Currency B for {curr_A} \n {'='*100} {pos_curB}")
    if prev == True:
        return preview, A_data
    else:
        return A_data


def posB_dfs(A_data, shape=False, x=5):
    Alist=[]
    for b in A_data["B"].unique():
        Alist.append(A_data[A_data["B"]==b])
    cur_map = {x:y for y,x in zip(Alist,A_data["B"].unique())}
    
    if shape == True:
        for b in A_data["B"].unique()[:x]:
            print(b,A_data[A_data["B"]==b].shape)
    return cur_map

start_date = 0
end_date = 0

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
    print(f"Missing entries \n {'='*100}")
    print(pd.date_range(start=start_date, end=end_date).difference(A_B.index))
    if prev == True:
        return preview, A_B
    else:
        return A_B

def upsample(A_B):
    xch = A_B['slug'].unique()
    #upsample to weekly records using mean
    weekly = A_B.resample('W', label='left',closed = 'left').mean()
    print(pd.date_range(start=start_date, end=end_date,freq="W").difference(weekly.index))
    #use ffil to fill null values
    weekly["close"]=weekly["close"].ffill()
    print('='*20)
    print(weekly.isnull().sum())
    print('='*20)
    print(f'Weekly Data Shape: {weekly.shape}')
    return weekly, xch
        

# Plot all features to check if any independant features are present
def weekly_plot(week_data, closing=False, kde=False, auto_corr=False):
    weekly = week_data[0]
    df_close = weekly['close']
    fig = go.Figure()
    fig.add_trace(go.Ohlc(x=weekly.index,
                        open=weekly['open'],
                        high=weekly['high'],
                        low=weekly['low'],
                        close=weekly['close'],
                        name='Price',
                        showlegend=True))
    fig.update_layout(width=1000, title=f'{week_data[1][0]}')
    fig.show()
    
    if closing == True:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weekly.index, y=weekly['close']
        ))
        fig.update_layout(title=f'{week_data[1][0]} closing price over the years')
        fig.show()
        
    if kde == True:
        #Analyse the KDE plot of the time series to checks for shape, spread, modes and ouliers
        df_close.plot(kind='kde')
        plt.show()
        
    if auto_corr == True:
        #check for autocorrelation with historic values
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(df_close)
        plt.show()
        
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
    fig.update_layout(title='Rolling Mean and Standard Deviation', width=900)
    fig.show()
    
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)

def seasonal_decomp(df_close):
    result = seasonal_decompose(df_close, model='multiplicative')
    fig = plt.figure()  
    fig = result.plot()  
    fig.set_size_inches(16, 9)
    plt.show()

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
    fig.update_layout(title='Moving Average', width=900)
    fig.show()
    return df_log

def train_test_split(df_log):
    train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_log.index, y=df_log.values, name='Train data'
    ))
    fig.add_trace(go.Scatter(
        x=test_data.index, y=test_data.values, name='Test data'
    ))
    fig.update_layout(width=900, yaxis_title='Closing Price')
    fig.show()
    return train_data, test_data

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
    plt.show()
    
def arima(train_data, test_data, plot=False):
    model = ARIMA(train_data, order=(1,1,2))  
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
        fig.update_layout(title='Train, Actual, and Prediction', width=900)
        fig.show()
        
        mse = mean_squared_error(test_data, fc)
        print('MSE: '+str(mse))
        mae = mean_absolute_error(test_data, fc)
        print('MAE: '+str(mae))
        rmse = math.sqrt(mean_squared_error(test_data, fc))
        print('RMSE: '+str(rmse))
        mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
        print('MAPE: '+str(mape))


def prophet_model(train_data, test_data, plot=False):
    train_df=pd.DataFrame(train_data)
    train_df["ds"]=train_df.index
    train_df["y"]=train_df["close"]
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=True)
    model.fit(train_df)
    future = model.make_future_dataframe(periods=len(test_data), freq='W-SUN',include_history=False)
    forecast = model.predict(future)
    fs=pd.Series(forecast["yhat"])
    fs.index=forecast.ds
    
    if plot == True:
        train_data = np.exp(train_data)
        test_data = np.exp(test_data)
        fs = np.exp(fs)
        #plot predicted vs actual
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_data.index, y=train_data.values, name='Training data'
        ))
        fig.add_trace(go.Scatter(
            x=test_data.index, y=test_data.values, name='Actual Forex rates'
        ))
        fig.add_trace(go.Scatter(
            x=fs.index, y=fs.values, name='Predicted Forex rates'
        ))
        fig.update_layout(title='Train, Actual, and Prediction', width=900)
        fig.show()
    
    mse = mean_squared_error(test_data, fs)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(test_data, fs)
    print('MAE: '+str(mae))
    rmse = math.sqrt(mean_squared_error(test_data, fs))
    print('RMSE: '+str(rmse))
    mape = np.mean(np.abs(fs - test_data)/np.abs(test_data))
    print('MAPE: '+str(mape))