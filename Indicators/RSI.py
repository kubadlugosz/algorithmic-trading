# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt
import time
from datetime import datetime
import plotly.graph_objects as go
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import yfinance as yf
import plotly.express as px
from plotly.subplots import make_subplots
import cufflinks as cf
from plotly.offline import download_plotlyjs, plot,iplot
cf.go_offline()
# we will build a function that will calculate RSI

# key = 'ZT6j9cVi03zdNaVZaq8anCvIF86eH5vyJhAfh4YCFRCgRRra6zP297hOxfRX9Zwc'
# secret = 'ABzCXOJ41tI0OAOKBUu8hPqDbLKP1JYxRc6jsYjgPmy5RTXF6q7IYYtbWalVkQnL'
# client =  Client(key, secret,tld='us',testnet=True) #connect to testnet server
def get_data(symbol, start,intervals, end = None):
    """
    This function will generate time series data for any ticker specified

    Args:
        symbol (str): this will be the ticker symbol you want to look up
        start (str): start date
        end (str): end date
        interval (str): interval you want your data in 1m,5m,15m, 1hr, 1d...etc
    """
    client = Client()  # make an instance for the binance
    #condition if no end date is specified
    if end == None:
        end = 'now UTC'
    if intervals == '1m':
        intervals = Client.KLINE_INTERVAL_1MINUTE
    # get historical
    df =client.get_historical_klines(symbol, intervals, start_str=start,end_str=end)
    df = pd.DataFrame(df)  # convert it to a dataframe
    # name each column for this data frame
    df.columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close_time",
        "qav",
        "num_trades",
        "taker_base_vol",
        "taker_quote_vol",
        "ignore",
    ]
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"],unit='ms')
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column])
    return df

def RSI(data, time_period):
    """This function will take historical date and calculate the RSI score

    Args:
        data (data frame): historical data in the form of a pandas data frame
        time_period (_type_): specify trading period to calculate on 
    """
    # https://www.macroption.com/rsi-calculation/ -> rsi calculation
    #RSI = 100 - 100/(1+RS)
    
    #Step 1: calculate up and down moves
    data['Price_Diff'] = data['Close'].diff(1)
    
    data['Up_Moves'] = data['Price_Diff'].apply(lambda x: x if x > 0 else 0)
    data['Down_Moves'] = data['Price_Diff'].apply(lambda x: abs(x) if x < 0 else 0)
    data['Avg_Up'] = data['Up_Moves'].ewm(span=time_period).mean()
    data['Avg_Down'] = data['Down_Moves'].ewm(span=time_period).mean()
    data['RS'] = data['Avg_Up']/data['Avg_Down'] 
    data['RSI'] = data['RS'].apply(lambda x: 100-(100/(1+x)))
    
    data = data.dropna()
    # data = data.iloc[1: , :]
    return data

def plot_RSI(df):
    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(211)
    ax1.plot(df['Date'],df['Close'])


    ax1 = plt.subplot(212)
    ax1.plot(df['Date'],df['RSI'])
    plt.axhline(y=50, color='r', linestyle='-')
    plt.axhline(y=70, color='b', linestyle='-')
    plt.axhline(y=30, color='g', linestyle='-')
    plt.show()

def RSI_Strategy(data,upper_limit,lower_limit):
    """Implement the RSI strategy
    buy - when the closing prices reaches above the upper threshold 
    sell - when the closing price reaches below the lower threshold

    Args:
        data (dataframe): historical data with rsi computed column
        upper_limit (numeric): upper threshold for over bought
        lower_limit (numeric): lower threshold for over sold

    Returns:
        dataframe: return updated data frame with buy/sell prices and signals 
    """
    #create strategy rules for RSI -> long when rsi 
    buy_price = []
    print(data)
    sell_price = []
    rsi_signal = []
    signal = 0
    entry = False
    rsi = [x for x in data['RSI']]
    prices = [x for x in data['Close']]
    print(len(rsi),len(prices))
    for i in range(len(rsi)):
        
        if rsi[i-1] > lower_limit and rsi[i] < lower_limit:
            if signal != 1 :
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                rsi_signal.append(signal)
                
               
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)
                #entry = False
        elif rsi[i-1] < upper_limit and rsi[i] > upper_limit:
            if signal != -1 :
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                rsi_signal.append(signal)
               
              
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)
                #entry == True
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            rsi_signal.append(0)
            #entry == False
   
            
            
    data['Buy Price'] = buy_price
    data['Sell Price'] = sell_price
    data['Signal'] = rsi_signal
    
    return data
    


def back_test(data):
   
    data['MA20'] = data['Close'].rolling(window=20).mean()
    fig = go.Figure()
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Candlestick(x=data['Date'],
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'], 
                             showlegend=False), row=1, col=1)
    # fig.add_trace(go.Scatter(x=data['Date'], 
    #                  y=data['Close']
    #                 ), row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.add_trace({"mode": "markers", "name": "Sell", 
        "type": "scatter", 
        "x": data['Date'], 
        "y": data['Sell Price'], 
        "xaxis": "x1", 
        "yaxis": "y1", 
        "marker": {
            "size": 12, 
            "color": "red"
        }})
    fig.add_trace({"mode": "markers", "name": "Sell", 
        "type": "scatter", 
        "x": data['Date'], 
        "y": data['Buy Price'], 
        "xaxis": "x1", 
        "yaxis": "y1", 
        "marker": {
            "size": 12, 
            "color": "green"
        }})
        
    
    fig.add_trace(go.Scatter(x=data['Date'], 
                     y=data['RSI']
                    ), row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", row=2, col=1, line_color="#000000", line_width=2)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red", line_width=2)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="red", line_width=2)
    
    plot(fig)
    

            

df = pd.DataFrame(columns=['Open','High','Low','Close','Volume','Complete'])     
   
def stream_candles(msg):
    
    
    
    event_time = pd.to_datetime(msg['E'],unit='ms')
    start_time = pd.to_datetime(msg['k']['t'],unit='ms')
    first = float(msg['k']['o'])
    high = float(msg['k']['h'])
    low = float(msg['k']['l'])
    close = float(msg['k']['c'])
    volume = float(msg['k']['v'])
    complete = float(msg['k']['x'])
    
    print("Time: {} | Price: {}".format(event_time, close))
    df.loc[start_time] = [first, high, low, close, volume, complete]
    

def trader(symbol,interval,start,end,rsi_period):
    # df = get_data(symbol, start,interval, end )
    # df = RSI(df,rsi_period)
    # df = RSI_Strategy(df,65,35)
    print(df)
    twm = ThreadedWebsocketManager()
    twm.start()
    stream = twm.start_kline_socket(callback=stream_candles,symbol=symbol,interval =interval)
    print(stream)
    time.sleep(1)
    
    
    
def main():
    import time
    interval ='1h'
    symbol = 'BTCUSDT'
    start = '1 Feb,2022'
    end = None
    rsi_period = 14
 
    df = get_data(symbol, start,interval, end )
    df = RSI(df,14)
    df = RSI_Strategy(df,65,35)
    #back_test(df)
    #trader(symbol,interval,start,end,rsi_period)
    print()
    
main()
