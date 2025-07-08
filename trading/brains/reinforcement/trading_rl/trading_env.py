import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, Any
import sys
sys.path.append("trading")
sys.path.append("trading/backtesting")

import model_tools as mt
import technical_analysis as ta
from backtesting import VectorizedBacktesting

class TradingEnvironment:
    """
    Trading environment that interfaces with VectorizedBacktesting for PPO training.
    Ensures no lookahead bias by starting signals at context_length.
    """
    
    def __init__(self, symbol: str = "BTC-USDT", chunks: int = 10, interval: str = "5m", 
                 age_days: int = 0, context_length: int = 20, data_source: str = "binance",
                 initial_capital: float = 10000, slippage_pct: float = 0.001, 
                 commission_fixed: float = 0.0):
        
        self.symbol = symbol
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.context_length = context_length
        self.data_source = data_source
        
        # Initialize VectorizedBacktesting
        self.vb = VectorizedBacktesting(
            instance_name="PPO_Training",
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            commission_fixed=commission_fixed,
            reinvest=False,
            leverage=1.0
        )
        
        # Load data
        self.data = self.vb.fetch_data(
            symbol=symbol,
            chunks=chunks,
            interval=interval,
            age_days=age_days,
            data_source=data_source
        )
        
        # Calculate technical indicators for state representation
        self._calculate_features()
        
        # State and action dimensions
        self.state_dim = self.features.shape[1]
        self.action_dim = 3  # short=1, flat=2, long=3
        
        # Episode parameters
        self.current_step = 0
        self.start_step = context_length  # No lookahead
        self.max_steps = len(self.data) - context_length - 1
        
        # Tracking variables
        self.signals = pd.Series(2, index=self.data.index)  # Start with flat
        self.episode_start_value = initial_capital
        
    def _calculate_features(self):
        """Calculate technical indicators for state representation"""
        df = self.data.copy()
        
        # Price-based features (normalized)
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages (ratios)
        df['SMA_5'] = ta.sma(df['Close'], 5) / df['Close']
        df['SMA_10'] = ta.sma(df['Close'], 10) / df['Close']
        df['SMA_20'] = ta.sma(df['Close'], 20) / df['Close']
        df['EMA_5'] = ta.ema(df['Close'], 5) / df['Close']
        df['EMA_10'] = ta.ema(df['Close'], 10) / df['Close']
        
        # Momentum indicators
        df['RSI'] = ta.rsi(df['Close'], 14) / 100.0  # Normalize to 0-1
        df['ROC'] = ta.roc(df['Close'], 10) / 100.0  # Normalize
        df['MOM'] = ta.mom(df['Close'], 10) / df['Close']
        
        # Volatility indicators
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14) / df['Close']
        df['BB_Width'] = (ta.bbands(df['Close'])[0] - ta.bbands(df['Close'])[2]) / df['Close']
        
        # Trend indicators
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])[0] / 100.0  # Normalize
        
        # Market structure
        df['ZScore'] = ta.zscore(df['Close'], 20)
        df['Hurst'] = ta.hurst_exponent(df['Close'], max_lag=10)
        
        # Volume indicators (if available)
        if 'Volume' in df.columns:
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['Volume_Change'] = df['Volume'].pct_change()
        else:
            df['Volume_Ratio'] = 1.0
            df['Volume_Change'] = 0.0
        
        # Select features for state representation
        feature_cols = [
            'Returns', 'Log_Returns', 'Price_Position', 'Price_Change',
            'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10',
            'RSI', 'ROC', 'MOM', 'ATR', 'BB_Width', 'ADX',
            'ZScore', 'Hurst', 'Volume_Ratio', 'Volume_Change'
        ]
        
        self.features = df[feature_cols].fillna(0).values
        
    def reset(self) -> torch.Tensor:
        """Reset environment to start of episode"""
        self.current_step = 0
        self.signals = pd.Series(2, index=self.data.index)  # Reset to flat
        self.episode_start_value = self.vb.initial_capital
        
        # Return initial state
        state = self._get_state()
        return torch.tensor(state, dtype=torch.float32)
    
    def _get_state(self) -> np.ndarray:
        """Get current state (features at current step)"""
        step_idx = self.start_step + self.current_step
        if step_idx >= len(self.features):
            step_idx = len(self.features) - 1
        return self.features[step_idx]
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Take action and return next state, reward, done, info
        
        Args:
            action: Trading action (1=short, 2=flat, 3=long)
        """
        # Store action in signals
        step_idx = self.start_step + self.current_step
        if step_idx < len(self.signals):
            self.signals.iloc[step_idx] = action
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Calculate reward
        reward = self._calculate_reward(action, done)
        
        # Get next state
        next_state = self._get_state()
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        
        # Additional info
        info = {
            'step': self.current_step,
            'action': action,
            'current_price': self.data['Close'].iloc[step_idx] if step_idx < len(self.data) else 0
        }
        
        return next_state_tensor, reward, done, info
    
    def _calculate_reward(self, action: int, done: bool) -> float:
        """Calculate reward for current action"""
        if self.current_step < 2:
            return 0.0
        
        # Get recent return
        step_idx = self.start_step + self.current_step - 1
        if step_idx >= len(self.data) - 1:
            return 0.0
            
        # Calculate single-step return based on action
        price_change = (self.data['Open'].iloc[step_idx + 1] - self.data['Open'].iloc[step_idx]) / self.data['Open'].iloc[step_idx]
        
        # Reward based on action alignment with price movement
        if action == 3:  # Long
            reward = price_change
        elif action == 1:  # Short
            reward = -price_change
        else:  # Flat
            reward = 0.0
        
        # Small penalty for trading costs
        if action != 2:  # Not flat
            reward -= self.vb.slippage_pct
        
        # Bonus reward at episode end based on overall performance
        if done:
            # Run full backtest to get final performance
            self._run_backtest()
            metrics = self.vb.get_performance_metrics()
            total_return = metrics.get('Total_Return', 0.0)
            
            # Bonus reward based on total return
            reward += total_return * 10  # Scale up total return reward
        
        return reward
    
    def _run_backtest(self):
        """Run backtest with current signals"""
        # Create strategy function from signals
        def signal_strategy(data):
            return self.signals
        
        # Run backtest
        self.vb.run_strategy(signal_strategy, verbose=False)
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final performance metrics after episode"""
        self._run_backtest()
        return self.vb.get_performance_metrics()
    
    def render(self):
        """Render current state (optional)"""
        pass 