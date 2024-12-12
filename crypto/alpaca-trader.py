from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import datetime
import pandas as pd
import numpy as np
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alpaca.data.live.crypto import CryptoDataStream

import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    DEFAULT = '\033[39m'

API_KEY_ID = "PK0A3PJBDB887VC8BO53"
API_SECRET_KEY = "hWeFc9JiICNSIrvfJWBIUZ6UmZ64SDaCUpqBNsyP"

class trading_bot:
    def __init__(self,SYMBOL, API_KEY_ID, API_SECRET_KEY):
        self.SYMBOL = SYMBOL
        self.API_KEY_ID = API_KEY_ID
        self.API_SECRET_KEY = API_SECRET_KEY

        self.trading_client = TradingClient(API_KEY_ID, API_SECRET_KEY, paper=True)
        try:
            self.portfolio = self.trading_client.get_all_positions()
            print(bcolors.OKGREEN + "KEYS CLEAR", end=bcolors.DEFAULT + "\n")
        except:
            print(bcolors.FAIL + "ERROR", end=bcolors.DEFAULT + "\n")
            raise Exception("BAD KEYS")
        data_client = CryptoHistoricalDataClient(API_KEY_ID, API_SECRET_KEY)
        DATE_RANGE = 14
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[self.SYMBOL],       # The crypto pair to fetch
            timeframe=TimeFrame.Hour,            # Timeframe (e.g., Minute, Hour, Day)
            start=datetime.datetime.now() - datetime.timedelta(DATE_RANGE),          # Start date for historical data
            end=datetime.datetime.now(),            # End date for historical data
            limit=1000                           # Max number of bars to retrieve
        )

        crypto_bars = data_client.get_crypto_bars(request_params)
        bars = []
        for bar in crypto_bars[SYMBOL]:
            bars.append({
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            })
        self.ohlcv = pd.DataFrame(bars)
    
    def compute_indicators(self):
        ma_window = 20  # Moving average window size
        num_std = 2  # Number of standard deviations for the bands
        self.ohlcv['SMA'] = self.ohlcv['close'].rolling(window=ma_window).mean()
        self.ohlcv['STD'] = self.ohlcv['close'].rolling(window=ma_window).std()
        self.ohlcv['Upper_Band'] = self.ohlcv['SMA'] + (num_std * self.ohlcv['STD'])
        self.ohlcv['Lower_Band'] = self.ohlcv['SMA'] - (num_std * self.ohlcv['STD'])

        self.ohlcv['delta'] = self.ohlcv['close'].diff() #self.ohlcv['close'][x]-self.ohlcv['close'][x-1]
        self.ohlcv['gain'] = self.ohlcv['delta'].where(self.ohlcv['delta'] > 0, 0)
        self.ohlcv['loss'] = -self.ohlcv['delta'].where(self.ohlcv['delta'] < 0, 0)
        window = 14 # Calculate the rolling averages of gains and losses
        self.ohlcv['avg_gain'] = self.ohlcv['gain'].rolling(window=window, min_periods=1).mean()
        self.ohlcv['avg_loss'] = self.ohlcv['loss'].rolling(window=window, min_periods=1).mean()
        self.ohlcv['rs'] = self.ohlcv['avg_gain'] / self.ohlcv['avg_loss']
        self.ohlcv['rsi'] = 100 - (100 / (1 + self.ohlcv['rs']))
    
    def open_graph(self):
        self.compute_indicators() #indicator columns don't exist yet

        ohlcv = self.ohlcv #no more eye sore
        fig = make_subplots(specs=[[{"secondary_y": True}]])

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

        fig.add_trace(candlestick)
        fig.add_trace(rsi, secondary_y=True)

        ### BOLLINGER BANDS
        fig.add_trace(go.Scatter(
            x=ohlcv['timestamp'],
            y=ohlcv['Upper_Band'],
            line=dict(color='rgba(255, 0, 0, 0.75)', width=1),
            name="Upper"
        ))
        fig.add_trace(go.Scatter(
            x=ohlcv['timestamp'],
            y=ohlcv['Lower_Band'],
            line=dict(color='rgba(0, 255, 0, 0.75)', width=1),
            name="Lower"
        ))
        fig.add_trace(go.Scatter(
            x=ohlcv['timestamp'],
            y=ohlcv['SMA'],
            line=dict(color='rgba(0, 191, 255, 0.75)', width=1),
            name="SMA"
        ))

        fig.update_layout(
            title=self.SYMBOL,
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_dark"  # Optional: Dark theme for the chart
        )
        fig.show()

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
        
        market_order = self.trading_client.submit_order(
                        order_data=market_order_data
                    )
        
        print(f"Trade placed at {datetime.datetime.now()} - {self.SYMBOL.split("/")[0]+" " if use_shares else "$"}{value} - {type}")
    
    def algo_trading(self):
        #trading algorithm asyncronous with data stream
        async def quote_handler(data):
            print(data)
        
        #stream data
        stream_client = CryptoDataStream(API_KEY_ID,API_SECRET_KEY)
        stream_client.subscribe_quotes(quote_handler, "BTC/USD")
        print('streamer ready')
        stream_client.run()

trader = trading_bot("BTC/USD", API_KEY_ID=API_KEY_ID,API_SECRET_KEY=API_SECRET_KEY)

trader.open_graph()
