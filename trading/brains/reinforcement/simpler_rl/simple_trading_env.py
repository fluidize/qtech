import pandas as pd
import numpy as np
import sys
import os

sys.path.append("trading")
import model_tools
from typing import Dict, List, Tuple
import brains.gbm.feature_selector as fs

class TradingEnv:
    """
    A simple trading environment for reinforcement learning.
    
    This environment allows an agent to:
    - Buy, sell, or hold a single asset
    - Observe market state (prices and indicators)
    - Receive rewards based on profit/loss
    """
    
    def __init__(self, 
                 symbol: str = "BTC-USD", 
                 initial_balance: float = 10000.0,
                 window_size: int = 10,  # Number of past observations available to the agent
                 commission: float = 0.001):  # 0.1% commission
        
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.window_size = window_size
        self.commission = commission
        
        # Trading state
        self.position = 0  # How much of the asset we own
        self.current_step = 0
        
        # Two separate datasets
        self.price_data = None  # Raw price data for the environment
        self.feature_data = None  # Processed feature data for observations
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = 3
        
        # Performance tracking
        self.trades = []
        self.portfolio_values = []
    
    def fetch_data(self, chunks=1, age_days=60, interval="1h"):
        # Fetch raw price data
        self.price_data = model_tools.fetch_data(
            ticker=self.symbol,
            chunks=chunks,
            interval=interval,
            age_days=age_days,
            kucoin=True
        )
        
        indicator_states = model_tools.prepare_data_reinforcement(
            self.price_data, 
            lagged_length=0,
            extra_features=False, 
            elapsed_time=False
        )
        y = np.zeros(len(indicator_states))
        y[indicator_states['Close'] > indicator_states['Close'].shift(1)] = 1
        feature_selector = fs.FeatureSelector(indicator_states,
                                               y,
                                               indicator_states.columns.tolist(),
                                               importance_threshold=5,
                                               max_features=-1
                                            )
        important_features = feature_selector.get_important_features()
        self.feature_data = indicator_states[important_features]
        length_diff = len(self.price_data) - len(self.feature_data)
        if length_diff > 0:
            self.price_data = self.price_data.iloc[length_diff:]
        
        print(f"Processed price data: {len(self.price_data)} bars")
        print(f"Processed feature data: {len(self.feature_data)} bars with {len(self.feature_data.columns)} features")
        
        return self.price_data, self.feature_data
    
    def reset(self):
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.position = 0
        
        # Make sure we don't start at the beginning but after window_size
        # to have enough history for the observation window
        self.current_step = self.window_size
        self.trades = []
        self.portfolio_values = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """Return the current state observation for the agent using post-processed data."""
        obs_data = self.feature_data.iloc[self.current_step - self.window_size:self.current_step]
        
        obs = obs_data.values
        
        position = np.array([self.position])
        
        return {
            'market_data': obs,
            'position': position,
        }
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment using the agent's action.
        
        Args:
            action: 0 = Hold, 1 = Buy, 2 = Sell
            
        Returns:
            observation: The new state
            reward: The reward for the action
            done: Whether the episode is complete
            info: Additional information
        """
        # Record portfolio value before action
        current_close_price = self.price_data.iloc[self.current_step]['Close']
        prev_portfolio_value = self.balance + (self.position * current_close_price)
        
        # Default reward is 0
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            if self.position == 0 and self.balance > 0:
                buy_amount = self.balance / current_close_price
                buy_amount *= (1 - self.commission)  # Account for commission
                
                self.position = buy_amount
                cost = buy_amount * current_close_price
                self.balance -= cost
                
                self.trades.append({
                    'type': 'buy',
                    'step': self.current_step,
                    'price': current_close_price,
                    'amount': buy_amount,
                    'cost': cost
                })
                
        elif action == 2:  # Sell
            if self.position > 0:
                sale_value = self.position * current_close_price
                sale_value *= (1 - self.commission)  # Account for commission
                
                cost_basis = sum([t['cost'] for t in self.trades if t['type'] == 'buy'])
                profit = sale_value - cost_basis
                
                self.balance += sale_value
                self.position = 0
                
                self.trades.append({
                    'type': 'sell',
                    'step': self.current_step,
                    'price': current_close_price,
                    'amount': self.position,
                    'value': sale_value,
                    'profit': profit
                })
        
        new_portfolio_value = self.balance + (self.position * current_close_price)
        self.portfolio_values.append(new_portfolio_value)
        
        portfolio_pct_change = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        benchmark_pct_change = (current_close_price - self.price_data.iloc[self.current_step - 1]['Close']) / self.price_data.iloc[self.current_step - 1]['Close']
        
        reward = (portfolio_pct_change - benchmark_pct_change) * 100 #reward alpha*100
        
        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1
        
        # Return updated state and information
        observation = self._get_observation()
        info = {
            'portfolio_value': new_portfolio_value,
            'step': self.current_step,
            'position': self.position,
            'balance': self.balance
        }
        
        return observation, reward, done, info
    
    def render(self):
        """Print current environment state."""
        print(f"\nStep: {self.current_step}")
        
        if 'Datetime' in self.price_data.columns:
            print(f"Date: {self.price_data.iloc[self.current_step]['Datetime']}")
        
        print(f"Price: ${self.price_data.iloc[self.current_step]['Close']:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position:.6f}")
        print(f"Portfolio Value: ${self.balance + (self.position * self.price_data.iloc[self.current_step]['Close']):.2f}")
        print(f"Trades: {len(self.trades)}")
        
    def get_performance_summary(self):
        """Return a summary of trading performance."""
        if not self.portfolio_values:
            return "No trading has occurred yet."
            
        start_value = self.initial_balance
        final_value = self.portfolio_values[-1]
        profit = final_value - start_value
        profit_percent = (profit / start_value) * 100
        
        # Count buy and sell trades
        buys = sum(1 for t in self.trades if t['type'] == 'buy')
        sells = sum(1 for t in self.trades if t['type'] == 'sell')
        
        return {
            'initial_balance': start_value,
            'final_balance': final_value,
            'profit': profit,
            'profit_percent': profit_percent,
            'trade_count': len(self.trades),
            'buys': buys,
            'sells': sells
        } 