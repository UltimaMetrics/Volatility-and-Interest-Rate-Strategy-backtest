# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 13:16:41 2022

@author: sigma
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 13:16:41 2022

@author: sigma
"""

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import pyfolio as pf

class FinancialData:
   # initialize function
   def __init__(self, symbol='^VIX', end=dt.datetime.today(), days=500):
       self.symbol = symbol
       self.start = end - pd.Timedelta(days=days)
       self.end = end
       self.retrieve_data(self.symbol, self.start, self.end)
       self.prepare_data()
  
   # function call to retrieve daily data
   def retrieve_data(self, symbol, start, end):
       self.data = yf.download(symbol, start=start, end=end)
  
   # preparing data - adding daily returns and buy/hold returns column
   def prepare_data(self):
       self.data['daily_returns'] = np.log(self.data['Adj Close'] / self.data['Adj Close'].shift(1))
       self.data['buy&hold_returns'] =  self.data['daily_returns'].cumsum()
       self.data.dropna(inplace=True)
      
   # function to plot a list of attributes in the pandas data frame
   def plot_data(self, attribute_list):
       self.data[attribute_list].plot()
       plt.show()
        
   # plotting strategy returns
   def plot_strategy_returns(self):
       self.plot_data(['buy&hold_returns', 'strategy_returns'])
      
   # function to create a simple tear sheet using pyfolio
   def create_simple_tear_sheet(self):
       pf.display(pf.create_simple_tear_sheet(self.data['strategy_returns'].diff()))

class BollingerBandBacktester(FinancialData):
    def prepare_indicators(self, window):
        self.data['moving_avg'] = \
        self.data['Adj Close'].rolling(window=window).mean()
        self.data['moving_std'] = \
        self.data['Adj Close'].rolling(window=window).std()

    def backtest_strategy(self, window, start=None):
        self.prepare_indicators(window)
        self.data['upper_band'] = \
        self.data['moving_avg'] + 2 * self.data['moving_std']
        self.data['lower_band'] = \
        self.data['moving_avg'] - 2 * self.data['moving_std']

        if start is None:
            start = window

        # BUY condition
        self.data['signal'] = \
        np.where((self.data['Adj Close'] < self.data['lower_band']) &
                 (self.data['Adj Close'].shift(1) >= self.data['lower_band']), 1, 0)

        # SELL condition
        self.data['signal'] = \
        np.where((self.data['Adj Close'] > self.data['upper_band']) &
                 (self.data['Adj Close'].shift(1) <= self.data['upper_band']), -1,
                  self.data['signal'])

        self.data['position'] = self.data['signal'].replace(to_replace=0, method='ffill')
        self.data['position'] = self.data['position'].shift()

        self.data['strategy_returns'] = self.data['position'] * self.data['daily_returns']

        performance = self.data[['daily_returns', 'strategy_returns']].iloc[start:].sum()

        self.data['strategy_returns'] = self.data['strategy_returns'].cumsum()
        return performance

    def optimize_bollinger_band_parameters(self, windows):
        start = max(windows)
        self.results = pd.DataFrame()
        for window in windows:
            perf = self.backtest_strategy(window=window, start=start)
            self.result = pd.DataFrame({'Window': window,
                                        'buy&hold returns': perf['daily_returns'],
                                        'strategy returns': perf['strategy_returns']}, 
                                         index=[0, ])
            self.results = self.results.append(self.result, ignore_index=True)
        self.results.sort_values(by='strategy returns', inplace=True, ascending=False)
        self.results = self.results.reset_index()
        self.results = self.results.drop("index", axis=1)
        print(self.results.head())

    def plot_optimized_bollinger_strategy_returns(self):
        if (len(self.results)) > 0:
            window = self.results.loc[0, 'Window']
            print("Window:", window)
            self.backtest_strategy(window=window)
            self.plot_strategy_returns()



ticker= "CVOL-USD"

today = dt.datetime.today()
n_days = 1000
Bollinger = BollingerBandBacktester(symbol=ticker, end =today, days = n_days)
Bollinger.optimize_bollinger_band_parameters(range(0, 50, 1))
Bollinger.plot_optimized_bollinger_strategy_returns()


Bollinger.create_simple_tear_sheet()


from unitroot import stationarity

asset=yf.Ticker("CVOL-USD")
cvol = asset.history(period="2y")
cvol['Close'].plot(title="Crypto Volatility")

stationarity(cvol['Close'])

from HalfLife import estimate_half_life
estimate_half_life(cvol['Close'])
