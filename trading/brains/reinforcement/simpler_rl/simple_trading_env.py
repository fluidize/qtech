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
        
        # Process and add indicators for feature data
        X, y = model_tools.prepare_data_classifier(
            self.price_data, 
            lagged_length=5, 
            extra_features=False, 
            elapsed_time=False
        )
        
        # Select important features
        feature_selector = fs.FeatureSelector(X, y, X.columns.tolist())
        important_features = feature_selector.get_important_features()
        
        # Create feature dataset with only important features
        self.feature_data = X[important_features]
        
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
        # Get a window of data up to the current step
        obs_data = self.feature_data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Convert to numpy array
        obs = obs_data.values
        
        # Add position information
        position = np.array([[self.position]])
        
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
        # Get current price from price_data
        current_price = self.price_data.iloc[self.current_step]['Close']
        
        # Initialize reward
        reward = 0
        
        # Process the action
        if action == 1:  # Buy
            if self.position == 0 and self.balance > 0:
                # Calculate amount to buy (all available balance)
                amount = self.balance / current_price
                # Apply commission
                amount *= (1 - self.commission)
                
                self.position = amount
                cost = amount * current_price
                self.balance -= cost
                
                self.trades.append({
                    'type': 'buy',
                    'step': self.current_step,
                    'price': current_price,
                    'amount': amount,
                    'cost': cost
                })
                
                # Small penalty to discourage excessive trading
                reward = -self.commission
                
        elif action == 2:  # Sell
            if self.position > 0:
                # Calculate sale value
                sale_value = self.position * current_price
                # Apply commission
                sale_value *= (1 - self.commission)
                
                # Calculate profit/loss
                cost_basis = sum([t['cost'] for t in self.trades if t['type'] == 'buy'])
                profit = sale_value - cost_basis
                
                # Update balance and position
                self.balance += sale_value
                self.position = 0
                
                self.trades.append({
                    'type': 'sell',
                    'step': self.current_step,
                    'price': current_price,
                    'amount': self.position,
                    'value': sale_value,
                    'profit': profit
                })
                
                # Reward is proportional to profit percentage
                reward = profit / cost_basis if cost_basis > 0 else 0
        
        # Calculate portfolio value (cash + assets)
        portfolio_value = self.balance + (self.position * current_price)
        self.portfolio_values.append(portfolio_value)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1
        
        # If no trades were made, give a small negative reward to encourage action
        if not self.trades and self.current_step % 10 == 0:
            reward -= 0.001
            
        # If we're holding a position and price went up, give a small reward
        if self.position > 0:
            prev_price = self.price_data.iloc[self.current_step - 1]['Close']
            price_change_pct = (current_price - prev_price) / prev_price
            reward += price_change_pct
            
        observation = self._get_observation()
        info = {
            'portfolio_value': portfolio_value,
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