from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live.crypto import CryptoDataStream

from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import traceback
import json
from typing import Dict, List, Optional, Tuple, Union

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    DEFAULT = '\033[39m'

def countdown(seconds: int) -> None:
    interval = 0.1
    total_steps = round(seconds / interval)
    for i in range(total_steps, -1, -1):
        print(f"{round(i*interval, 2)} seconds remaining...", end="\r")
        time.sleep(interval)

class RealTrading:
    def __init__(self, 
                 symbol: str, 
                 api_key_id: str, 
                 api_secret_key: str,
                 paper_trading: bool = True):
        self.symbol = symbol
        self.api_key_id = api_key_id
        self.api_secret_key = api_secret_key
        self.paper_trading = paper_trading
        
        # Initialize API clients
        self.trading_client = TradingClient(api_key_id, api_secret_key, paper=paper_trading)
        self.data_client = CryptoHistoricalDataClient(api_key_id, api_secret_key)
        
        # Initialize data structures
        self.ohlcv: Optional[pd.DataFrame] = None
        self.portfolio: Optional[List[Dict]] = None
        self.fig: Optional[go.Figure] = None
        
        self._initialize_trading()
        self._load_initial_data()

    def _initialize_trading(self) -> None:
        try:
            os.system('cls')
            self.portfolio = self.trading_client.get_all_positions()
            print(bcolors.OKGREEN + "API credentials verified successfully" + bcolors.DEFAULT)
        except Exception as e:
            print(bcolors.FAIL + f"Failed to initialize trading: {str(e)}" + bcolors.DEFAULT)
            raise Exception("Invalid API credentials or connection error")

    def _load_initial_data(self) -> None:
        self.load_data(verbose=True)
        print("-" * 55)
        self.get_account_info(verbose=True)
        print("-" * 55)

    def load_data(self, verbose: bool = False) -> None:
        try:
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[self.symbol],
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(days=1),
                end=datetime.now(),
                limit=5000
            )
            
            crypto_bars = self.data_client.get_crypto_bars(request_params)
            bars = []
            for bar in crypto_bars[self.symbol]:
                bars.append({
                    "Datetime": bar.timestamp.replace(tzinfo=None),
                    "Open": bar.open,
                    "High": bar.high,
                    "Low": bar.low,
                    "Close": bar.close,
                    "Volume": bar.volume
                })
            self.ohlcv = pd.DataFrame(bars)
            
            if verbose:
                print("Market data loaded successfully")
        except Exception as e:
            print(bcolors.FAIL + f"Failed to load market data: {str(e)}" + bcolors.DEFAULT)
            raise

    def _calculate_std(self, period: int = 5) -> pd.Series:
        raw_std = self.ohlcv['Close'].rolling(window=period).std()
        return (raw_std - raw_std.min()) / (raw_std.max() - raw_std.min())

    def custom_scalper(self) -> None:
        try:
            current_close = self.ohlcv['Close'].iloc[-1]
            prev_close = self.ohlcv['Close'].iloc[-2]
            
            std = self._calculate_std(5)
            current_std = std.iloc[-1]
            
            price_change_pct = ((current_close - prev_close) / prev_close) * 100
            
            buy_conditions = [
                price_change_pct > 0.05,
                current_std > 0.01
            ]
            
            sell_conditions = [
                price_change_pct < -0.05
            ]
            
            if all(buy_conditions):
                self.buy_max()
            elif all(sell_conditions):
                self.sell_max()
                
        except Exception as e:
            print(bcolors.FAIL + f"Error in custom_scalper: {str(e)}" + bcolors.DEFAULT)
            traceback.print_exc()

    def get_positions(self, verbose: bool = False) -> List[Dict]:
        positions = []
        for position in self.portfolio:
            pos_dict = {
                "symbol": position.symbol,
                "shares": float(position.qty),
                "value": float(position.market_value),
                "asset_price": float(position.current_price)
            }
            positions.append(pos_dict)
            if verbose:
                print(f"{position.symbol} - {position.qty}")
        return positions

    def get_account_info(self, verbose: bool = False) -> Dict:
        raw_info = self.trading_client.get_account()
        info_dict = {
            "account_number": raw_info.account_number,
            "account_id": raw_info.id,
            "currency": raw_info.currency,
            "currency_amount": raw_info.cash,
            "portfolio_value": raw_info.portfolio_value,
            "equity": raw_info.equity
        }
        if verbose:
            print(f"Account Number: {info_dict['account_number']}")
            print(f"Account ID: {info_dict['account_id']}")
            print(f"Currency Amount: {info_dict['currency_amount']} {info_dict['currency']}")
            print(f"Portfolio Value: {info_dict['portfolio_value']}")
            print(f"Equity: {info_dict['equity']}")
        return info_dict

    def place_order(self, value: float, order_type: OrderSide, use_shares: bool = False) -> bool:
        try:
            # Round the value to appropriate decimal places
            if use_shares:
                value = round(value, 9)  # 8 decimal places for BTC
            else:
                value = round(value, 3)  # 2 decimal places for USD

            market_order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=value if use_shares else None,
                notional=value if not use_shares else None,
                side=order_type,
                time_in_force=TimeInForce.GTC
            )
            
            # Submit the order and get the order response
            
            order = self.trading_client.submit_order(order_data=market_order_data)
            
            time.sleep(1)
            
            order_status  = self.trading_client.get_order_by_client_id(order.client_order_id)
            if order_status.status != 'filled':
                print(bcolors.WARNING + f"Order not filled. Status: {order_status.status}" + bcolors.DEFAULT)
                return False
            
            pcolor = bcolors.OKGREEN if order_type == OrderSide.BUY else bcolors.FAIL
            pcurrency = f"{self.symbol.split('/')[0]} " if use_shares else "$"
            print(f"{pcolor}Trade placed at {datetime.now()} - {pcurrency}{value} - {order_type}{bcolors.DEFAULT}")
            
            return True
            
        except Exception as e:
            print(bcolors.FAIL + f"Failed to place order: {str(e)}" + bcolors.DEFAULT)
            return False

    def buy_max(self) -> None:
        try:
            # Update portfolio positions before checking available cash
            self.portfolio = self.trading_client.get_all_positions()
            account_info = self.get_account_info()
            available_cash = float(account_info['currency_amount'])
            
            if available_cash >= 10:  # Minimum order size
                # Place the order and check if it was successful
                if self.place_order(
                    available_cash,
                    OrderSide.BUY,
                    use_shares=False
                ):
                    # Only update portfolio if order was successful
                    self.portfolio = self.trading_client.get_all_positions()
                else:
                    print(bcolors.WARNING + "Buy order was not successful" + bcolors.DEFAULT)
            else:
                print(bcolors.WARNING + f"Insufficient funds for buy order. Available: ${available_cash}" + bcolors.DEFAULT)
        except Exception as e:
            print(bcolors.WARNING + f"Buy order failed: {str(e)}" + bcolors.DEFAULT)

    def sell_max(self) -> None:
        try:
            current_position = self.get_positions()[0]
            try:
                # Round down to 8 decimal places to ensure we don't exceed available balance
                shares = round(current_position['shares'], 9)
                if shares <= 0:
                    print(bcolors.WARNING + "No shares available to sell" + bcolors.DEFAULT)
                    return
                    
                self.place_order(
                    shares,
                    OrderSide.SELL,
                    use_shares=True
                )
            except Exception as e:
                if current_position['value'] >= 200000:
                    print(bcolors.WARNING + "Order too large, selling maximum allowed amount" + bcolors.DEFAULT)
                    # Round down to 8 decimal places for the maximum sell amount
                    shares_to_sell = round(200000 / current_position['asset_price'], 8)
                    if shares_to_sell <= 0:
                        print(bcolors.WARNING + "No shares available to sell" + bcolors.DEFAULT)
                        return
                    self.place_order(shares_to_sell, OrderSide.SELL, use_shares=True)
                    print(bcolors.FAIL + "Sold maximum amount of shares" + bcolors.DEFAULT)
        except Exception as e:
            print(bcolors.WARNING + f"Sell order failed: {str(e)}" + bcolors.DEFAULT)

    def create_ohlcv_chart(self) -> None:
        fig = go.Figure(data=[go.Candlestick(x=self.ohlcv['Datetime'],
                                            open=self.ohlcv['Open'],
                                            high=self.ohlcv['High'],
                                            low=self.ohlcv['Low'],
                                            close=self.ohlcv['Close'])])
        
        fig.update_layout(
            title=f'{self.symbol} Price Chart',
            yaxis_title='Price',
            xaxis_title='Time',
            template='plotly_dark'
        )
        
        fig.show()

    def auto_trade(self) -> None:
        print("Starting automated trading\n")
        while True:
            try:
                current_time = datetime.now()
                print(f"Current time: {current_time} | Last Updated: {self.ohlcv['Datetime'].iloc[-1]}")
                
                self.load_data()
                self.custom_scalper()
                countdown(5)
                print('\n')
                
            except Exception as e:
                print(bcolors.FAIL + f"Error in auto_trade: {str(e)}" + bcolors.DEFAULT)
                traceback.print_exc()
                countdown(45)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with open('keys.json') as f:
        data = json.load(f)
    
    trader = RealTrading(
        "BTC/USD",
        data['key'],
        data['secret'],
        paper_trading=True
    )
    trader.auto_trade()