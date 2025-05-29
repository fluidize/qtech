import pandas as pd
from typing import Dict
import inspect

import sys
sys.path.append("trading")
import model_tools as mt
import vectorized_backtesting as vb

# start off with algo optimization

class AlgorithmConstructionEnv:
    """
    Environment for RL-based black-box optimization of trading strategy parameters.
    The action is a parameter set; the reward is the strategy's performance.
    Observation is always None (stateless optimization).
    """
    def __init__(self, symbol: str, chunks: int, interval: str, age_days: int):
        self.data = None

        self.symbol = symbol
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days

        self.engine = vb.VectorizedBacktesting(instance_name='default', initial_capital=10000)
        self.basis_algorithm = self.engine.scalper_strategy1
        self.params = list(inspect.signature(self.basis_algorithm).parameters.keys())

        # Parameter space for optimization (example ranges, adjust as needed)
        self.param_space = {
            'fast_period': (5, 20),
            'slow_period': (10, 50),
            'adx_threshold': (10, 40),
            'momentum_period': (5, 20),
            'momentum_threshold': (0.1, 1.0),
            'wick_threshold': (0.1, 1.0)
        }

        self._initialize_data(self.symbol, self.chunks, self.interval, self.age_days)

        self.current_step = 0
        self.done = False

        self.episode_reward = 0
        self.returns_history = []

    def _initialize_data(self, symbol: str, chunks: int, interval: str, age_days: int):
        self.data = self.engine.fetch_data(symbol=symbol, chunks=chunks, interval=interval, age_days=age_days)

    def reset(self):
        self._initialize_data(self.symbol, self.chunks, self.interval, self.age_days)
        self.current_step = 0
        self.done = False
        self.episode_reward = 0
        self.returns_history = []
        # Optionally reset engine state if needed
        # Observation is always None for RL-based optimization
        return self._get_observation()

    def step(self, params: Dict):
        self.engine.run_strategy(self.basis_algorithm, verbose=False, **params)
        metrics = self.engine.get_performance_metrics()
        reward = metrics['Total_Return']
        self.episode_reward += reward
        self.returns_history.append(reward)

        self.current_step += 1
        self.done = self.current_step >= len(self.data)
        obs = self._get_observation()  # Always None
        info = {'metrics': metrics}
        return obs, reward, self.done, info

    def _get_observation(self):
        # Observation is always None for RL-based optimization (stateless)
        return None

    def close(self):
        pass

    def render(self):
        pass

    def sample_params(self):
        import random
        params = {}
        for k, v in self.param_space.items():
            if isinstance(v[0], int) and isinstance(v[1], int):
                params[k] = random.randint(v[0], v[1])
            else:
                params[k] = random.uniform(v[0], v[1])
        return params

if __name__ == "__main__":
    env = AlgorithmConstructionEnv(symbol="BTC-USDT", chunks=1, interval="5min", age_days=10)
    print(env.params)

    params = {
        'fast_period': 9,
        'slow_period': 26,
        'adx_threshold': 25,
        'momentum_period': 10,
        'momentum_threshold': 0.75,
        'wick_threshold': 0.5
    }
    print(env.step(params))