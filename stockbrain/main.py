from alpaca.trading.client import TradingClient
API_KEY_ID = "PKACVJJ8HOHIPE5EFP0Z"
API_SECRET_KEY = "gRXq5qom5Oxjb0dvMBwPmJLNlUrUhTtE3JMKeiHn"

trading_client = TradingClient(API_KEY_ID, API_SECRET_KEY, paper=True)
account = trading_client.get_account()
print(account)