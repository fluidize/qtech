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
        obs_data = self.price_data.iloc[self.current_step - self.window_size:self.current_step].drop(columns=['Datetime'])
        
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
        current_price = self.price_data.iloc[self.current_step]['Close']
        prev_price = self.price_data.iloc[self.current_step - 1]['Close'] if self.current_step > 0 else current_price
        price_change_pct = (current_price - prev_price) / prev_price
        
        # Calculate portfolio value before action
        prev_portfolio_value = self.balance + (self.position * prev_price)
        
        reward = 0
        if action == 1:  # Buy
            if self.position > 0:
                # Penalty for redundant buy when already holding
                reward = -1.0
            elif self.balance > 0:
                # Execute buy
                buy_amount = self.balance / current_price
                buy_amount *= (1 - self.commission)
                
                self.position = buy_amount
                cost = buy_amount * current_price
                self.balance -= cost
                
                self.trades.append({
                    'type': 'buy',
                    'step': self.current_step,
                    'price': current_price,
                    'amount': buy_amount,
                    'cost': cost
                })
                
                # Small cost for transaction
                reward = -self.commission * 2
                
                # Reward for buying during uptrend, penalize for buying during downtrend
                reward += price_change_pct * 2
                
        elif action == 2:  # Sell
            if self.position == 0:
                # Penalty for redundant sell when not holding
                reward = -1.0
            else:
                # Execute sell
                sale_value = self.position * current_price
                sale_value *= (1 - self.commission)
                
                # Get cost basis of current position
                cost_basis = sum([t['cost'] for t in self.trades if t['type'] == 'buy'])
                profit = sale_value - cost_basis
                profit_pct = profit / cost_basis if cost_basis > 0 else 0
                
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
                
                # Reward based on profit percentage, scaled for better learning
                reward = profit_pct * 5
                
                # Additional reward for selling during downtrend, penalty for selling during uptrend
                reward -= price_change_pct * 2
                
        else:  # Hold
            # For holding, give small reward/penalty based on price movement relative to position
            if self.position > 0:  # If holding asset
                reward = price_change_pct * 3  # Reward for holding during uptrend
            else:  # If holding cash
                reward = -price_change_pct * 1.5  # Penalty for holding cash during uptrend (opportunity cost)
        
        # Calculate new portfolio value
        portfolio_value = self.balance + (self.position * current_price)
        self.portfolio_values.append(portfolio_value)
        
        # Add portfolio change component to reward
        portfolio_change_pct = (portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
        reward += portfolio_change_pct * 2
        
        # Normalize rewards to a more consistent scale
        reward = np.clip(reward, -2.0, 2.0)
        
        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1
        
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