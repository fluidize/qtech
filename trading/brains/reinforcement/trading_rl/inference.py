import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import sys
sys.path.append("trading")
sys.path.append("trading/backtesting")

from ppo_agent import PPOAgent, PPOPolicy
from trading_env import TradingEnvironment
from backtesting import VectorizedBacktesting
from rich import print

class PPOInference:
    """
    Inference class for trained PPO trading models.
    Can generate signals for backtesting or live trading.
    """
    
    def __init__(self, model_path: str, state_dim: int, hidden_dim: int = 64):
        """Initialize with trained model"""
        self.model_path = model_path
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Load trained policy
        self.policy = PPOPolicy(state_dim, hidden_dim)
        self.policy.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.policy.eval()
        
        print(f"[green]Loaded PPO model from {model_path}[/green]")
    
    def generate_signals(self, 
                        symbol: str = "BTC-USDT",
                        chunks: int = 10,
                        interval: str = "5m",
                        age_days: int = 0,
                        context_length: int = 20,
                        data_source: str = "binance") -> pd.Series:
        """
        Generate trading signals for given data.
        
        Returns:
            pd.Series: Trading signals (1=short, 2=flat, 3=long)
        """
        
        # Create environment for feature calculation
        env = TradingEnvironment(
            symbol=symbol,
            chunks=chunks,
            interval=interval,
            age_days=age_days,
            context_length=context_length,
            data_source=data_source
        )
        
        # Initialize signals series
        signals = pd.Series(2, index=env.data.index)  # Start with flat
        
        # Generate signals starting from context_length
        for step in range(env.max_steps):
            # Get state
            state_idx = env.start_step + step
            if state_idx >= len(env.features):
                break
                
            state = env.features[state_idx]
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # Get action from policy (deterministic)
            with torch.no_grad():
                logits, _ = self.policy(state_tensor)
                action = torch.argmax(logits, dim=-1).item()
                
                # Convert to trading signal
                trading_signal = action + 1  # 0->1, 1->2, 2->3
                signals.iloc[state_idx] = trading_signal
        
        return signals
    
    def backtest_signals(self, 
                        symbol: str = "BTC-USDT",
                        chunks: int = 10,
                        interval: str = "5m",
                        age_days: int = 0,
                        context_length: int = 20,
                        data_source: str = "binance",
                        initial_capital: float = 10000,
                        slippage_pct: float = 0.001,
                        commission_fixed: float = 0.0,
                        show_plot: bool = True) -> Dict[str, Any]:
        """
        Generate signals and run backtest.
        
        Returns:
            Dict containing signals and performance metrics
        """
        
        # Generate signals
        signals = self.generate_signals(
            symbol=symbol,
            chunks=chunks,
            interval=interval,
            age_days=age_days,
            context_length=context_length,
            data_source=data_source
        )
        
        # Setup backtesting
        vb = VectorizedBacktesting(
            instance_name="PPO_Inference",
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            commission_fixed=commission_fixed,
            reinvest=False,
            leverage=1.0
        )
        
        # Load data
        vb.fetch_data(
            symbol=symbol,
            chunks=chunks,
            interval=interval,
            age_days=age_days,
            data_source=data_source
        )
        
        # Create strategy function from signals
        def ppo_strategy(data):
            return signals
        
        # Run backtest
        vb.run_strategy(ppo_strategy, verbose=True)
        
        # Get metrics
        metrics = vb.get_performance_metrics()
        
        # Show plot if requested
        if show_plot:
            vb.plot_performance(extended=False)
        
        return {
            'signals': signals,
            'metrics': metrics,
            'backtest_data': vb.data
        }
    
    def predict_next_action(self, state: np.ndarray) -> int:
        """
        Predict next action given current state.
        Useful for live trading.
        
        Args:
            state: Current market state features
            
        Returns:
            int: Trading action (1=short, 2=flat, 3=long)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = self.policy(state_tensor)
            action = torch.argmax(logits, dim=-1).item()
            
            # Convert to trading signal
            trading_signal = action + 1  # 0->1, 1->2, 2->3
            
        return trading_signal
    
    def get_action_probabilities(self, state: np.ndarray) -> Dict[str, float]:
        """
        Get action probabilities for interpretability.
        
        Args:
            state: Current market state features
            
        Returns:
            Dict with action probabilities
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = self.policy(state_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze()
            
        return {
            'short_prob': probs[0].item(),
            'flat_prob': probs[1].item(),
            'long_prob': probs[2].item(),
            'state_value': value.item()
        }

def load_and_test_model(model_path: str, 
                       symbol: str = "BTC-USDT",
                       chunks: int = 10,
                       interval: str = "5m",
                       age_days: int = 0):
    """
    Convenience function to load model and run backtest.
    """
    
    # Determine state dimension (should match training)
    # This assumes standard feature set from TradingEnvironment
    STATE_DIM = 19  # Based on feature_cols in TradingEnvironment
    
    # Load model
    inference = PPOInference(model_path, STATE_DIM)
    
    # Run backtest
    results = inference.backtest_signals(
        symbol=symbol,
        chunks=chunks,
        interval=interval,
        age_days=age_days,
        show_plot=True
    )
    
    # Print results
    metrics = results['metrics']
    print(f"\n[bold green]PPO Model Performance:[/bold green]")
    print(f"Total Return: {metrics['Total_Return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.3f}")
    print(f"Win Rate: {metrics['Win_Rate']*100:.1f}%")
    print(f"Max Drawdown: {metrics['Max_Drawdown']*100:.2f}%")
    print(f"Total Trades: {metrics['Total_Trades']}")
    
    return results

if __name__ == "__main__":
    # Example usage
    model_path = "trading_rl_models/ppo_model_final.pth"
    
    # Test the model
    results = load_and_test_model(
        model_path=model_path,
        symbol="BTC-USDT",
        chunks=20,
        interval="5m",
        age_days=7
    ) 