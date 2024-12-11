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

from alpaca.data.live.crypto import CryptoDataStream



API_KEY_ID = "PKSUCH8LLQDX4CJVXCGD"
API_SECRET_KEY = "4l0zZRsUcoDSrZJxJtvveamRBeI1QusMIvKsugNl"


#data collection
data_client = CryptoHistoricalDataClient(API_KEY_ID, API_SECRET_KEY)
SYMBOL = "BTC/USD"
DATE_RANGE = 14
request_params = CryptoBarsRequest(
    symbol_or_symbols=[SYMBOL],       # The crypto pair to fetch
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
ohlcv = pd.DataFrame(bars)

graph_enabled = False


ma_window = 20  # Moving average window size
num_std = 2  # Number of standard deviations for the bands
ohlcv['SMA'] = ohlcv['close'].rolling(window=ma_window).mean()
ohlcv['STD'] = ohlcv['close'].rolling(window=ma_window).std()
ohlcv['Upper_Band'] = ohlcv['SMA'] + (num_std * ohlcv['STD'])
ohlcv['Lower_Band'] = ohlcv['SMA'] - (num_std * ohlcv['STD'])

ohlcv['delta'] = ohlcv['close'].diff() #ohlcv['close'][x]-ohlcv['close'][x-1]
ohlcv['gain'] = ohlcv['delta'].where(ohlcv['delta'] > 0, 0)
ohlcv['loss'] = -ohlcv['delta'].where(ohlcv['delta'] < 0, 0)
window = 14 # Calculate the rolling averages of gains and losses
ohlcv['avg_gain'] = ohlcv['gain'].rolling(window=window, min_periods=1).mean()
ohlcv['avg_loss'] = ohlcv['loss'].rolling(window=window, min_periods=1).mean()
ohlcv['rs'] = ohlcv['avg_gain'] / ohlcv['avg_loss']
ohlcv['rsi'] = 100 - (100 / (1 + ohlcv['rs']))

print(ohlcv)

ohlcv.drop(columns=['delta', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], inplace=True)


if graph_enabled:
    fig = go.Figure(data=[go.Candlestick(
        x=ohlcv['timestamp'],
        open=ohlcv['open'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        close=ohlcv['close'],
        name="Candlesticks"
    )])

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
        title="Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"  # Optional: Dark theme for the chart
    )
    fig.show()

# async def quote_handler(data):
#     print(data)

# stream_client = CryptoDataStream(API_KEY_ID,API_SECRET_KEY)
# stream_client.subscribe_quotes(quote_handler, "BTC/USD")
# print('streamer ready')
# stream.run()

trading_client = TradingClient(API_KEY_ID, API_SECRET_KEY, paper=True)
portfolio = trading_client.get_all_positions()

# market_order_data = MarketOrderRequest(
#                     symbol="BTC/USD",
#                     qty=1,
#                     side=OrderSide.BUY,
#                     time_in_force=TimeInForce.GTC
#                     )
# market_order = trading_client.submit_order(
#                 order_data=market_order_data
#                )
# for position in portfolio:
#     print("{} shares of {}".format(position.qty, position.symbol))