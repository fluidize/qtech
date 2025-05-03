import pandas as pd
import numpy as np
import sys
import os

# Add trading directory to path to import model_tools
current_dir = os.path.dirname(os.path.abspath(__file__))
trading_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(trading_dir)

# Import from trading directory
import model_tools
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

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
        self.data = None
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = 3
        
        # Performance tracking
        self.trades = []
        self.portfolio_values = []
    
    def fetch_data(self, chunks=1, age_days=60, interval="1h"):
        self.data = model_tools.fetch_data(
            ticker=self.symbol,
            chunks=chunks,
            interval=interval,
            age_days=age_days,
            kucoin=True
        )
        
        # Save the original datetime column
        date_col_name = 'Datetime' if 'Datetime' in self.data.columns else 'Date'
        datetime_col = self.data[date_col_name].copy() if date_col_name in self.data.columns else None
        
        # Process and add indicators
        self._add_indicators()
        
        # Make sure we have the date column for rendering
        if datetime_col is not None and date_col_name not in self.data.columns:
            self.data[date_col_name] = datetime_col
        
        print(f"Processed data: {len(self.data)} bars with {len(self.data.columns)} features")
        return self.data
    
    def _add_indicators(self):
        """Use model_tools to prepare data with indicators."""
        # Use model_tools prepare_data_classifier which returns X, y
        # We only need X (the feature dataframe)
        processed_data, _ = model_tools.prepare_data_classifier(self.data,  lagged_length=5, extra_features=False, elapsed_time=False)
        self.data = processed_data
        
        print(f"Data shape after adding indicators: {self.data.shape}")
    
    def reset(self):
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.position = 0
        
        # Store datetime column before preprocessing if it exists
        datetime_col = None
        date_col_name = 'Datetime' if 'Datetime' in self.data.columns else 'Date'
        if date_col_name in self.data.columns:
            datetime_col = self.data[date_col_name].copy()
            
        # Make sure we don't start at the beginning but after window_size
        # to have enough history for the observation window
        self.current_step = self.window_size
        self.trades = []
        self.portfolio_values = []
        
        # Check if we saved the datetime column and add it back if needed
        if datetime_col is not None and date_col_name not in self.data.columns:
            self.data[date_col_name] = datetime_col
        
        return self._get_observation()
    
    def _get_observation(self):
        """Return the current state observation for the agent."""
        # Get date column name
        date_col_name = 'Datetime' if 'Datetime' in self.data.columns else 'Date'
        
        # Get a window of data up to the current step
        obs_data = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Drop date column if it exists
        if date_col_name in obs_data.columns:
            obs_data = obs_data.drop(columns=[date_col_name])
        
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
        # Get current price
        current_price = self.data.iloc[self.current_step]['Close']
        
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
        done = self.current_step >= len(self.data) - 1
        
        # If no trades were made, give a small negative reward to encourage action
        if not self.trades and self.current_step % 10 == 0:
            reward -= 0.001
            
        # If we're holding a position and price went up, give a small reward
        if self.position > 0:
            prev_price = self.data.iloc[self.current_step - 1]['Close']
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
        
        # Handle both 'Date' and 'Datetime' column names
        date_col = 'Datetime' if 'Datetime' in self.data.columns else 'Date'
        if date_col in self.data.columns:
            print(f"Date: {self.data.iloc[self.current_step][date_col]}")
        
        print(f"Price: ${self.data.iloc[self.current_step]['Close']:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position:.6f}")
        print(f"Portfolio Value: ${self.balance + (self.position * self.data.iloc[self.current_step]['Close']):.2f}")
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