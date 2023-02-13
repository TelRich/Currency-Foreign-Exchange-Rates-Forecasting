import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as po
po.init_notebook_mode(connected=True)
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
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
                    name='Price'))
            fig.update(layout_xaxis_rangeslider_visible=False, layout_width=1000,
                    layout_title=xch[x])
            x += 1
            fig.show()
    else:
        return curA[0].head()

# select currency
def select_curA(prev=False):
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
    #upsample to weekly records using mean
    weekly = A_B.resample('W', label='left',closed = 'left').mean()
    print(pd.date_range(start=start_date, end=end_date,freq="W").difference(weekly.index))
    #use ffil to fill null values
    weekly["close"]=weekly["close"].ffill()
    print(weekly.isnull().sum())
    print(weekly.shape)
    return weekly
        

