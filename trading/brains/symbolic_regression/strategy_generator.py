import pandas as pd
import numpy as np
from pysr import PySRRegressor

from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("trading")
import technical_analysis as ta
import model_tools as mt
sys.path.append("trading/backtesting")
from backtesting import VectorizedBacktesting
import strategy


class StrategyGenerator:
    def __init__(self, 
                 symbol: str = "ETH-USDT",
                 chunks: int = 10,
                 interval: str = "15m",
                 age_days: int = 0,
                 data_source: str = "binance",
                 initial_capital: float = 10000,
                 slippage_pct: float = 0.001,
                 commission_fixed: float = 1.0):
        """
        Generate trading strategies using symbolic regression.
        
        Args:
            symbol: Trading pair symbol
            chunks: Number of data chunks
            interval: Time interval
            age_days: Data age in days
            data_source: Data source
            initial_capital: Initial capital for backtesting
            slippage_pct: Slippage percentage
            commission_fixed: Fixed commission per trade
        """
        self.symbol = symbol
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.data_source = data_source
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_fixed = commission_fixed
        
        self.backtest = VectorizedBacktesting(
            instance_name="StrategyGenerator",
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            commission_fixed=commission_fixed,
            reinvest=False,
            leverage=1.0
        )
        
        self.data = self.backtest.fetch_data(symbol=symbol, chunks=chunks, interval=interval, age_days=age_days, data_source=data_source)
        
    def generate_strategy(self):
        pass

if __name__ == "__main__":
    generator = StrategyGenerator(
        symbol="ETH-USDT",
        chunks=20,
        interval="15m",
        age_days=1,
        data_source="binance",
        initial_capital=10000,
        slippage_pct=0.00,
        commission_fixed=0.00
    )