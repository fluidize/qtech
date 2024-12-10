from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import requests
API_KEY_ID = "PKACVJJ8HOHIPE5EFP0Z"
API_SECRET_KEY = "gRXq5qom5Oxjb0dvMBwPmJLNlUrUhTtE3JMKeiHn"

trading_client = TradingClient(API_KEY_ID, API_SECRET_KEY, paper=True)
account = trading_client.get_account()

# market_order_data = MarketOrderRequest(
#                     symbol="BTC/USD",
#                     qty=1,
#                     side=OrderSide.BUY,
#                     time_in_force=TimeInForce.GTC
#                     )
# market_order = trading_client.submit_order(
#                 order_data=market_order_data
#                )

def latestquote(symbol):
    url = f"https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols={symbol}&feed=opra"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": "PKACVJJ8HOHIPE5EFP0Z",
        "APCA-API-SECRET-KEY": "gRXq5qom5Oxjb0dvMBwPmJLNlUrUhTtE3JMKeiHn"
    }
    response = requests.get(url, headers=headers)
    
    return response.text

portfolio = trading_client.get_all_positions()

for position in portfolio:
    print("{} shares of {}".format(position.qty, position.symbol))
