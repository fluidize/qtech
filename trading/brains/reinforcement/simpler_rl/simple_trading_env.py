import numpy as np
import gym
from gym import spaces

import sys
sys.path.append("trading")
import model_tools as mt
import brains.gbm.feature_selector as fs

class TradingEnv(gym.Env):
    def __init__(self, initial_balance=10000.0, commission=0.001, window_size=10, max_position_size=0.5):
        super().__init__()
        self.price_data = None
        self.feature_data = None
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.max_position_size = max_position_size  # Maximum position size as fraction of balance
        self.action_space = spaces.Discrete(3)  # 0: short, 1: flat, 2: long
        if self.feature_data is not None:
            market_shape = (window_size, self.feature_data.shape[1])
        else:
            market_shape = (window_size, 1)
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(low=-np.inf, high=np.inf, shape=market_shape, dtype=np.float32),
            'position': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })
        self._reset_internal_state()

    def fetch_data(self, symbol="BTC-USD", chunks=1, age_days=60, interval="1h"):
        self.price_data = mt.fetch_data(
            ticker=symbol,
            chunks=chunks,
            interval=interval,
            age_days=age_days,
            kucoin=True
        )
        indicator_states = mt.prepare_data_reinforcement(
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
                                              max_features=-1)
        important_features = feature_selector.get_important_features()
        self.feature_data = indicator_states[important_features].reset_index(drop=True)
        length_diff = len(self.price_data) - len(self.feature_data)
        if length_diff > 0:
            self.price_data = self.price_data.iloc[length_diff:].reset_index(drop=True)
        print(f"Processed price data: {len(self.price_data)} bars")
        print(f"Processed feature data: {len(self.feature_data)} bars with {len(self.feature_data.columns)} features")
        # Update observation space
        market_shape = (self.window_size, self.feature_data.shape[1])
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(low=-np.inf, high=np.inf, shape=market_shape, dtype=np.float32),
            'position': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })
        self._reset_internal_state()

        return self.price_data, self.feature_data

    def _reset_internal_state(self):
        self.balance = self.initial_balance
        self.position = 0.0
        self.prev_position = 0
        self.current_step = self.window_size
        self.trades = []
        self.portfolio_values = []

    def reset(self):
        self._reset_internal_state()
        return self._get_observation()

    def _get_observation(self):
        obs_data = self.feature_data.iloc[self.current_step - self.window_size:self.current_step].values.astype(np.float32)
        position = np.array([self.position], dtype=np.float32)
        return {'market_data': obs_data, 'position': position}

    def step(self, action):
        # Map action: 0=short, 1=flat, 2=long => -1, 0, 1
        action_map = {0: -1, 1: 0, 2: 1}
        target_position = action_map[action]
        current_price = self.price_data.iloc[self.current_step]['Close']
        
        # Calculate current portfolio value
        prev_portfolio_value = self.balance + self.position * current_price
        
        trade_penalty = 0
        if target_position != self.prev_position:
            # Close previous position if any
            if self.prev_position != 0:
                sale_value = self.position * current_price
                sale_value *= (1 - self.commission)
                self.balance += sale_value
                self.position = 0.0
                self.trades.append({
                    'type': 'close',
                    'step': self.current_step,
                    'price': current_price,
                    'amount': self.prev_position,
                    'value': sale_value
                })
            
            # Open new position if not flat
            if target_position != 0:
                # Calculate maximum position size based on current balance
                max_position_value = self.balance * self.max_position_size
                max_position_units = max_position_value / current_price
                
                # Apply position sizing
                position_units = max_position_units * abs(target_position)
                position_units *= (1 - self.commission)  # Account for commission
                
                self.position = position_units * np.sign(target_position)
                cost = abs(position_units * current_price)
                self.balance -= cost
                
                self.trades.append({
                    'type': 'open',
                    'step': self.current_step,
                    'price': current_price,
                    'amount': self.position,
                    'cost': cost
                })
            
            trade_penalty = -0.0001 * abs(target_position - self.prev_position)  # Reduced penalty
        
        # Calculate new portfolio value and return
        new_portfolio_value = self.balance + self.position * current_price
        self.portfolio_values.append(new_portfolio_value)
        
        # Clip portfolio change to prevent extreme values
        portfolio_pct_change = np.clip(
            (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value,
            -0.5,  # Max 50% loss per step
            0.5    # Max 50% gain per step
        )
        
        reward = portfolio_pct_change + trade_penalty
        
        # Check for bankruptcy
        done = (new_portfolio_value <= 0) or (self.current_step >= len(self.price_data) - 1)
        
        self.current_step += 1
        self.prev_position = target_position
        
        obs = self._get_observation()
        info = {
            'portfolio_value': new_portfolio_value,
            'step': self.current_step,
            'position': self.position,
            'balance': self.balance
        }
        
        return obs, reward, done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Price: ${self.price_data.iloc[self.current_step]['Close']:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position:.6f}")
        print(f"Portfolio Value: ${self.balance + (self.position * self.price_data.iloc[self.current_step]['Close']):.2f}")
        print(f"Trades: {len(self.trades)}")

    def close(self):
        pass

    def get_performance_summary(self):
        if not self.portfolio_values:
            return "No trading has occurred yet."
        start_value = self.initial_balance
        final_value = self.portfolio_values[-1]
        profit = final_value - start_value
        profit_percent = (profit / start_value) * 100
        buys = sum(1 for t in self.trades if t['type'] == 'open' and t['amount'] > 0)
        sells = sum(1 for t in self.trades if t['type'] == 'close' and t['amount'] > 0)
        return {
            'initial_balance': start_value,
            'final_balance': final_value,
            'profit': profit,
            'profit_percent': profit_percent,
            'trade_count': len(self.trades),
            'buys': buys,
            'sells': sells
        } 