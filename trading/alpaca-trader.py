from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alpaca.data.live.crypto import CryptoDataStream

import time
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    DEFAULT = '\033[39m'

os.chdir(os.path.dirname(os.path.abspath(__file__))) #set cwd as script folder

import json
data = json.load(open('keys.json'))

API_KEY_ID = data['key']
API_SECRET_KEY = data['secret']

class TradingEngine:
    def __init__(self,SYMBOL, API_KEY_ID, API_SECRET_KEY):
        self.SYMBOL = SYMBOL
        self.API_KEY_ID = API_KEY_ID
        self.API_SECRET_KEY = API_SECRET_KEY

        self.trading_client = TradingClient(API_KEY_ID, API_SECRET_KEY, paper=True)
        self.data_client = CryptoHistoricalDataClient(API_KEY_ID, API_SECRET_KEY)
        
        try:
            os.system('cls')
            self.portfolio = self.trading_client.get_all_positions()
            print(bcolors.OKGREEN + "KEYS CLEAR", end=bcolors.DEFAULT + "\n")
            print()
        except:
            print(bcolors.FAIL + "ERROR", end=bcolors.DEFAULT + "\n")
            raise Exception("BAD KEYS")

### INTIAL DATA LOAD
        self.load_data()

        #fun display
        print("-"*55)
        self.get_account_info(verbose=True)
        print("-"*55)

    def load_data(self, verbose=False):
        DATE_RANGE = 14
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[self.SYMBOL],       #pair to get
            #REMEMBER TO CHANGE ALGORITHM TRADING TIMEFRAMES
            timeframe=TimeFrame.Hour,            # Timeframe (e.g., Minute, Hour, Day)
            start=datetime.now() - timedelta(DATE_RANGE),          #start date
            end=datetime.now(),            #end date
            limit=10000                          # Max number of bars to retrieve
        )
        
        ### DATA COLLECTION
        crypto_bars = self.data_client.get_crypto_bars(request_params)
        bars = []
        for bar in crypto_bars[self.SYMBOL]:
            bars.append({
                "timestamp": bar.timestamp.replace(tzinfo=None),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            })
        self.ohlcv = pd.DataFrame(bars)

        print("DATA LOADED - OHLCV REFRESHED")
    
    def compute_indicators(self):
        ma_window = 20  # Moving average window size
        num_std = 2  # Number of standard deviations for the bands
        self.ohlcv['SMA'] = self.ohlcv['close'].rolling(window=ma_window).mean()
        self.ohlcv['STD'] = self.ohlcv['close'].rolling(window=ma_window).std()
        self.ohlcv['upper_band'] = self.ohlcv['SMA'] + (num_std * self.ohlcv['STD'])
        self.ohlcv['lower_band'] = self.ohlcv['SMA'] - (num_std * self.ohlcv['STD'])

        self.ohlcv['delta'] = self.ohlcv['close'].diff() #self.ohlcv['close'][x]-self.ohlcv['close'][x-1]
        self.ohlcv['gain'] = self.ohlcv['delta'].where(self.ohlcv['delta'] > 0, 0)
        self.ohlcv['loss'] = -self.ohlcv['delta'].where(self.ohlcv['delta'] < 0, 0)
        window = 14 # Calculate the rolling averages of gains and losses
        self.ohlcv['avg_gain'] = self.ohlcv['gain'].rolling(window=window, min_periods=1).mean()
        self.ohlcv['avg_loss'] = self.ohlcv['loss'].rolling(window=window, min_periods=1).mean()
        self.ohlcv['rs'] = self.ohlcv['avg_gain'] / self.ohlcv['avg_loss']
        self.ohlcv['rsi'] = 100 - (100 / (1 + self.ohlcv['rs']))
### TA FUNCTIONS
    def strategy_BB(self):
        #BOLLINGER BAND CROSSOVER - OVERBOUGHT
        results = np.greater(self.ohlcv['high'],self.ohlcv['upper_band'])
        sell_points = []
        for x in range(len(results)):
            if results[x]:
                sell_points.append(self.ohlcv['timestamp'][x])
        #BOLLINGER BAND CROSSUNDER - OVERSOLD
        buy_points = []
        results = np.less(self.ohlcv['low'],self.ohlcv['lower_band'])
        for x in range(len(results)):
            if results[x]:
                buy_points.append(self.ohlcv['timestamp'][x])

        return {"buy_points":buy_points, "sell_points":sell_points}

    def strategy_RSI(self):
        limit = 30
        low_rsi = np.less(self.ohlcv['rsi'], limit, ) #low rsi - oversold
        high_rsi = np.greater(self.ohlcv['rsi'], 100-limit, ) #high rsi - overbought
        buy_points = []
        sell_points = []
        #1 - buy 
        #-1 - sell 
        #0 - none
        for count, b, s in zip(range(len(self.ohlcv['rsi'])), low_rsi, high_rsi):
            if b:
                buy_points.append(self.ohlcv['timestamp'][count])
            elif s:
                sell_points.append(self.ohlcv['timestamp'][count])

        return {"buy_points":buy_points, "sell_points":sell_points}

    def update_graph(self):
        self.compute_indicators() #indicator columns don't exist yet

        ohlcv = self.ohlcv #no more eye sore
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])

        candlestick = go.Candlestick(
            x=ohlcv['timestamp'],
            open=ohlcv['open'],
            high=ohlcv['high'],
            low=ohlcv['low'],
            close=ohlcv['close'],
            name="Candlesticks"
        )

        rsi = go.Scatter(
            x=ohlcv['timestamp'],
            y=ohlcv['rsi'],
            line=dict(color='rgba(104, 22, 200, 0.75)', width=1),
            name="RSI"
        )

        self.fig.add_trace(candlestick)
        self.fig.add_trace(rsi, secondary_y=True)

        ### BOLLINGER BANDS
        self.fig.add_trace(go.Scatter(
            x=ohlcv['timestamp'],
            y=ohlcv['upper_band'],
            line=dict(color='rgba(255, 0, 0, 0.75)', width=1),
            name="Upper"
        ))
        self.fig.add_trace(go.Scatter(
            x=ohlcv['timestamp'],
            y=ohlcv['lower_band'],
            line=dict(color='rgba(0, 255, 0, 0.75)', width=1),
            name="Lower"
        ))
        self.fig.add_trace(go.Scatter(
            x=ohlcv['timestamp'],
            y=ohlcv['SMA'],
            line=dict(color='rgba(0, 191, 255, 0.75)', width=1),
            name="SMA"
        ))

        self.fig.update_layout(
            title=self.SYMBOL,
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_dark"  # Optional: Dark theme for the chart
        )
    
    def show_graph(self):
        try:
            self.fig.show()
        except:
            print(bcolors.FAIL + "GRAPH NOT CREATED YET" + bcolors.DEFAULT)
                                                                                                                                                                
    def get_positions(self, verbose=False):
        positions = []
        for position in self.portfolio:
            positions.append({
                "symbol":position.symbol,
                "shares":position.qty
                              })
            if verbose:
                print(f"{position.symbol} - {position.qty}")

        return positions
    def get_account_info(self, verbose=False):
        raw_info = self.trading_client.get_account()
        info_dict = {
            "account_number" : raw_info.account_number,
            "account_id" : raw_info.id,
            "currency" : raw_info.currency,
            "currency_amount" : raw_info.cash,
            "portfolio_value" : raw_info.portfolio_value,
            "equity" : raw_info.equity
        }
        if verbose:
            print(f"Account Number: {info_dict['account_number']}\nAccount ID: {info_dict['account_id']}\nCurrency Amount: {info_dict['currency_amount']} {info_dict['currency']}\nPortfolio Value: {info_dict['portfolio_value']}\nEquity: {info_dict['equity']}")
        return info_dict

### TRADING FUNCTIONS
    def place_order(self, value, type, use_shares=False): #trading in usd by default
        #notional means trading in USD
        #qty means trading in shares
        if use_shares:
            market_order_data = MarketOrderRequest(
                symbol=self.SYMBOL,
                qty=value,
                side=type, #OrderSide.BUY OrderSide.SELL
                time_in_force=TimeInForce.GTC
            )
        else:
            market_order_data = MarketOrderRequest(
                symbol=self.SYMBOL,
                notional=value,
                side=type, #OrderSide.BUY OrderSide.SELL
                time_in_force=TimeInForce.GTC
            )
        
        try:
            self.trading_client.submit_order(order_data=market_order_data)
            print(bcolors.OKGREEN if type == OrderSide.BUY else bcolors.FAIL + f"Trade placed at {datetime.now()} - {self.SYMBOL.split("/")[0]+' ' if use_shares else "$"}{value} - {type}" + bcolors.DEFAULT)
        except:
            print(bcolors.WARNING + "Order Failed; Out of currency or order is too large." + bcolors.DEFAULT)
        
        
    def buy_max(self):
        self.place_order(self.get_account_info()['currency_amount'],OrderSide.BUY,use_shares=False)
    def sell_max(self):
        try:
            self.place_order(trader.get_positions()[0]['shares'],OrderSide.SELL,use_shares=True)
        except:
            print(bcolors.WARNING + "Order Failed; No assets?" + bcolors.DEFAULT)

    def rsi_bb_auto(self):
        while True:
            # current_time = datetime.now().replace(microsecond=0, second=0) #trading by minutes
            current_time = datetime.now().replace(microsecond=0, second=0, minute=0) #trading by hours

            print(current_time)
            self.load_data()
            self.compute_indicators()
            print(f"RSI: {self.ohlcv['rsi'].iloc[-1]}")
            

            ### OVERSOLD
            if (self.strategy_RSI()['buy_points'][-1] == current_time) or (self.strategy_BB()['buy_points'][-1] == current_time): #if the last entry point is the hour, buy
                self.buy_max()
            ### OVERBOUGHT
            if (self.strategy_RSI()['sell_points'][-1] == current_time) or (self.strategy_BB()['sell_points'][-1] == current_time):
                self.sell_max()

            print()
            time.sleep(3600)
        

trader = TradingEngine("BTC/USD", API_KEY_ID=API_KEY_ID,API_SECRET_KEY=API_SECRET_KEY)

trader.compute_indicators()

print(trader.strategy_RSI()['sell_points'][-1])
print(datetime.now().replace(microsecond=0, second=0, minute=0))

trader.rsi_bb_auto()