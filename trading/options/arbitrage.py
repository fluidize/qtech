import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil import parser
from rich import print

# long expiration short put positions can have arbitrage opportunity is if 
# strike - premium - price < 0
# additionally, money market funds, can pay or yield yearly on the collateral.
# goal is to identify stocks with lespa (long experiation short puts arbitrage)
# long expiration short put positions in the money

class Arbitrage:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.option_expiries = {}
    
    def load(self):
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            expirations = [parser.parse(x).date() for x in expirations]
            self.option_expiries[ticker] = expirations #keep ticker as yf.ticker obj
        return self.option_expiries
    
    def analyze(self):
        output = {}
        for symbol in self.option_expiries:
            best_profit = 0
            best_expiry = None
            best_strike = None
            best_premium = None

            market_price = symbol.history(period="1d").iloc[-1]['Close']

            for expiry in self.option_expiries[symbol]:
                options = symbol.option_chain(str(expiry))
                puts = options.puts #ARBITRAGING PUTS
                for index, option in puts.iterrows():
                    strike = option['strike']
                    premium = option['bid'] #bid ask midpoint
                    factor = 100

                    arbitrage_value = (strike - premium - market_price) * 100

                    if (arbitrage_value < best_profit) and (premium > 0):
                        best_profit = arbitrage_value
                        best_expiry = expiry
                        best_strike = strike
                        best_premium = premium
                        

            output[symbol] = {
                "best_expiry" : best_expiry,
                "best_strike" : best_strike,
                "best_profit" : best_profit,
                "best_premium" : best_premium
                              }

        return output
                
    def run(self):
        self.load()
        print(self.analyze())

arbitrage = Arbitrage(["JOBY"])
arbitrage.run()