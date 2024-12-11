from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import datetime
import pandas as pd
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
request_params = CryptoBarsRequest(
    symbol_or_symbols=[SYMBOL],       # The crypto pair to fetch
    timeframe=TimeFrame.Hour,            # Timeframe (e.g., Minute, Hour, Day)
    start=datetime.datetime.now() - datetime.timedelta(1),          # Start date for historical data
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

fig = go.Figure(data=[go.Candlestick(x=ohlcv['timestamp'],
                open=ohlcv['open'],
                high=ohlcv['high'],
                low=ohlcv['low'],
                close=ohlcv['close'])])

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