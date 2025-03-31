import yfinance as yf
import pandas as pd
import requests
import asyncio

from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta

from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich import print

from tqdm import tqdm

from dataclasses import dataclass
from typing import Dict

# long expiration short put positions can have arbitrage opportunity is if 
# strike - premium - price < 0
# additionally, money market funds, can pay or yield yearly on the collateral.
# goal is to identify stocks with lespa (long expIriation short puts arbitrage)
# long expiration short put positions in the money

def find_stocks():
    output = []

    LOW_CAP_MAX = 2_000_000_000  
    MID_CAP_MAX = 10_000_000_000

    symbols = requests.get("https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt").text.strip().split("\n")
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            market_cap = ticker.info.get("marketCap", 0)

            if (market_cap and market_cap < LOW_CAP_MAX) or (market_cap and market_cap < MID_CAP_MAX):
                output.append(symbol)
                print(symbol)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    df = pd.DataFrame(output, columns=["Ticker"])
    df.to_csv("low_mid_cap.csv", index=False)
    
    return output

@dataclass
class Option:
    type: str
    expiry: datetime
    strike: float
    premium: float
    market_price: float
    intrinsic_value: float = 0.0


class Arbitrage:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.option_expiries = {}
        self.best_options = {}
    
    def load(self):
        print("[yellow]Loading Expiries...[/yellow]")

        progress_bar = tqdm(total=len(self.symbols))
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                expirations = ticker.options
                expirations = [parser.parse(x).date() for x in expirations]
                self.option_expiries[ticker] = expirations 
            except:
                print(f"\nFAILED TO LOAD {symbol}")
            progress_bar.update(1)
        progress_bar.close()
        return self.option_expiries
    
    def _find_best_arbitrage(self, ITM=True, collateral_yield=True):
        output = {}
        progress_bar = tqdm(total=len(self.option_expiries))
        for symbol in self.option_expiries:
            best_profit = 0
            best_option = None

            try:
                market_price = symbol.history(period="1d").iloc[-1]['Close']

                for expiry in self.option_expiries[symbol]:
                    options = symbol.option_chain(str(expiry))
                    puts = options.puts #ARBITRAGING PUTS
                    for index, option in puts.iterrows():
                        strike = option['strike']
                        premium = option['bid']

                        factor = 100

                        option_arbitrage_value = (strike - premium - market_price) * factor
                        
                        collateral = strike*factor

                        expiry_time_delta = relativedelta(expiry, datetime.now())
                        time_delta_months = expiry_time_delta.years * 12 + expiry_time_delta.months

                        collateral_yielded_profit = (collateral*0.0033)*time_delta_months if collateral_yield else 0 #assuming naked option + SIMPLE monthly compounding of 0.33% on collat

                        total_arbitrage_value = -option_arbitrage_value + collateral_yielded_profit #flip option_arbitrage_value as it is an EFFECTIVE COST BASIS
                        
                        #OTM options typically have weird numbers leading to large profits
                        if (total_arbitrage_value > best_profit) and (premium > 0) and ( ((strike-market_price)>0) if ITM else True ): #if market_price-strike > 0, put is ITM
                            best_profit = total_arbitrage_value
                            best_option = Option(type='put', expiry=expiry, strike=strike, premium=premium, market_price=market_price, intrinsic_value=max(0, strike - market_price))

                best = {
                    "best_option" : best_option,
                    "best_profit" : best_profit,
                }
                output[symbol] = best
                self.best_options[symbol] = best
            except:
                print(f"\nFAILED TO SCAN {symbol}")

            progress_bar.update(1)
        progress_bar.close()

        return output

    def display_arbitrage(self, input_dict: Dict[yf.Ticker, Option] = {}):
        console = Console()
        table = Table()
        
        table.add_column("Symbol", justify="center", style="bold"),
        table.add_column("Market Price", justify="center")
        table.add_column("Expiry", justify="center")
        table.add_column("Strike", justify="center")
        table.add_column("Premium", justify="center")
        table.add_column("Intrinsic Value", justify="center")
        table.add_column("Profit", justify="center", style="bold green")
        
        if input_dict:
            best_options = input_dict
        elif not self.best_options:
            raise ValueError("Best options not calculated yet.")
        else:
            best_options = self.best_options

        for symbol, data in best_options.items():
            option = data["best_option"]
            if option:
                last_price = symbol.history(period='1d')['Close'].iloc[-1]
                table.add_row(
                    symbol.ticker,
                    f"{last_price:.2f}",
                    str(option.expiry),
                    f"${option.strike:.2f}",
                    f"${option.premium:.2f}",
                    f"${option.intrinsic_value:.2f}",
                    f"${data['best_profit']:.2f}",
                )
        
        console.print(Panel(table, title="Arbitrage Opportunities", expand=False))

        return True
    
    def save_arbitrage_to_csv(self, filename="arbitrage_opportunities.csv"):
        print("[green]SAVING...[/green]")
        data = []
        for symbol, data_info in self.best_options.items():
            option = data_info["best_option"]
            if option:
                last_price = symbol.history(period='1d')['Close'].iloc[-1]
                
                if option.type == 'call':
                    intrinsic_value = max(0, last_price - option.strike) 
                elif option.type == 'put':
                    intrinsic_value = max(0, option.strike - last_price)

                data.append([symbol.ticker, last_price, option.type, str(option.expiry), option.strike, option.premium, intrinsic_value, data_info['best_profit']])

        # Create the DataFrame and save to CSV
        df = pd.DataFrame(data, columns=["Symbol", "Market Price", "Type", "Expiry", "Strike", "Premium", "Intrinsic Value", "Profit"])
        df.to_csv(filename, index=False)
    
    def run(self):
        self.load()
        self._find_best_arbitrage()
        self.save_arbitrage_to_csv()


arbitrage = Arbitrage(pd.read_csv("trading\options\low_mid_cap.csv")["Ticker"].tolist())
arbitrage.run()