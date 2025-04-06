import os
import time
import json
import asyncio
import websockets
import pandas as pd
import numpy as np
from rich import print
from jupiter_api import JupiterAPI, token

import sys
sys.path.append(rf"C:\Users\{os.getlogin()}\Documents\GitHub\fintech\trading")
from single_pytorch_model import load_model
from model_tools import fetch_data

class LiveTrader:
    def __init__(
        self,
        wallet_address: str,
        private_key: str,
        model_path: str,
        interval: str = "5min",
        chunks: int = 1,
        age_days: int = 0,
        sol_threshold: float = 0.09,
        usd_threshold: float = 39,
        position_size: float = 0.25,
        max_drawdown: float = 0.1,
    ):
        self.jupiter = JupiterAPI(wallet_address, private_key)
        self.model = load_model(model_path)
        self.current_position = 0  # 0: no position, 1: long position
        self.data = None
        self.initial_value = self.jupiter.get_wallet_value(self.jupiter.wallet_address)
        self.sol_threshold = sol_threshold
        self.usd_threshold = usd_threshold
        self.position_size = position_size
        self.max_drawdown = max_drawdown

    def fetch_latest_data(self, ticker, chunks, interval, age_days):
        """Fetch the latest market data"""
        return fetch_data(ticker, chunks, interval, age_days, kucoin=True)

    def get_prediction(self, data):
        print("[bold]PREDICTING...[/bold]")
        predictions = self.model.predict(self.model, data[['Open', 'High', 'Low', 'Close', 'Volume']])
        return predictions

    def execute_trade(self, prediction, previous_prediction):
        if prediction > previous_prediction and self.current_position == 0:
            try:
                result = self.jupiter.buy_max(token.SOL)
                print(f"[green]Buy executed at price KU: {self.data['Close'].iloc[-1]} JUP: {result}[/green]")
                self.current_position = 1
            except Exception as e:
                print(f"[red]Error executing buy: {e}[/red]")

        elif prediction < previous_prediction and self.current_position == 1:
            try:
                result = self.jupiter.sell(token.SOL, 0.25)
                print(f"[green]Sell executed at price KU: {self.data['Close'].iloc[-1]} JUP: {result}[/green]")
                self.current_position = 0
            except Exception as e:
                print(f"[red]Error executing sell: {e}[/red]")

    def log_wallet_value(self): 
        current_value = self.jupiter.get_wallet_value(self.jupiter.wallet_address)
        current_time = pd.Timestamp.now()
        with open("log.csv", "a") as f:
            f.write(f"{current_time}, {current_value}, {self.current_position}\n")
    
    def log_trade(self, action, size, price):
        current_time = pd.Timestamp.now()
        with open("trade_log.csv", "a") as f:
            f.write(f"{current_time}, {action}, {size}, {price}, {self.current_position}\n")

    def check_safety_limits(self):
        current_value = self.jupiter.get_wallet_value(self.jupiter.wallet_address)
        initial_value = self.initial_value if hasattr(self, 'initial_value') else current_value
        drawdown = (initial_value - current_value) / initial_value
        
        if drawdown > self.max_drawdown:
            raise Exception(f"Maximum drawdown exceeded: {drawdown:.2%}")
        
        if self.jupiter.get_wallet_token_amount(self.jupiter.wallet_address, token.SOL) < self.sol_threshold:
            raise Exception("SOL balance below threshold")
        
        if current_value < self.usd_threshold:
            raise Exception("USD value below threshold")

    def run(self, ticker, chunks, interval):
        print("[bold]Starting live trading...[/bold]")
        self.data = self.fetch_latest_data(ticker, chunks, interval, 0)
        previous_prediction = self.get_prediction(self.data)[-1]

        while True:
            new_data = self.fetch_latest_data(ticker, chunks, interval, 0)
            if not (new_data['Datetime'].iloc[-1] == self.data['Datetime'].iloc[-1]):
                print(new_data['Close'].iloc[-1], self.jupiter.get_price(token.SOL))
                self.check_safety_limits()

                print("-"*50)
                self.data = new_data
                current_prediction = self.get_prediction(self.data)[-1]

                self.execute_trade(current_prediction, previous_prediction)
                print("Long Prediction" if current_prediction > previous_prediction else "Short Prediction")

                previous_prediction = current_prediction

                print(f"[cyan]Current Position: {'Long' if self.current_position == 1 else 'No Position'}[/cyan]")
                print(f"[cyan]Wallet Value: ${self.jupiter.get_wallet_value(self.jupiter.wallet_address):.2f}[/cyan]")
                print("-"*50)

                self.log_wallet_value()

            time.sleep(30)

if __name__ == "__main__":
    # WALLET_ADDRESS = input("Enter your wallet address: ")
    PRIVATE_KEY = input("Enter your private key: ")

    trader = LiveTrader(
        wallet_address="",
        private_key=PRIVATE_KEY,
        model_path=r"btc1m.pth",
        interval="1min",
        chunks=1,
        age_days=0,
        sol_threshold=0.09,
        usd_threshold=39,
        position_size=0.25,
        max_drawdown=0.1
    )

    trader.run("SOL-USDT", 5, "1min")
