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


# we will build a function that will calculate RSI



def get_data(symbol, start,intervals, end):
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
    # if end == None:
    #     end = 'now UTC'
    
    # get historical
    df =client.get_historical_klines("BNBBTC", intervals, "1 day ago UTC")
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
    df["Date"] = pd.to_datetime(df["Date"],utc=True)
    return df

    def RSI(data, days):
        """
        This function will take a dataframe and calculate

        Args:
            data (_type_): _description_
            days (_type_): _description_
        """


def main():
    interval ='1d'
    symbol = 'BTCUSDT'
    start = '25 Feb,2022'
    end = '27 Feb,2022'
 
    df = get_data(symbol, start, interval,end)
    print(df)


main()
