import torch
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table
import time
import os
from typing import Dict, List

from ppo_agent import PPOAgent
from trading_env import TradingEnvironment

class PPOTrainer:
    def __init__(self, 
                 symbol: str = "BTC-USDT",
                 chunks: int = 10,
                 interval: str = "5m",
                 age_days: int = 7,
                 context_length: int = 20,
                 data_source: str = "binance",
                 initial_capital: float = 10000,
                 slippage_pct: float = 0.001,
                 commission_fixed: float = 0.0):
        
        # Initialize environment
        self.env = TradingEnvironment(
            symbol=symbol,
            chunks=chunks,
            interval=interval,
            age_days=age_days,
            context_length=context_length,
            data_source=data_source,
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            commission_fixed=commission_fixed
        )
        
        # Initialize agent
        self.agent = PPOAgent(
            state_dim=self.env.state_dim,
            lr=3e-4,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=4,
            hidden_dim=64
        )
        
        self.console = Console()
        
        # Training metrics
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_sharpe = []
        self.episode_win_rates = []
        
    def train(self, 
              episodes: int = 100,
              update_frequency: int = 50,
              save_frequency: int = 25,
              model_dir: str = "models"):
        """Train the PPO agent"""
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"[bold green]Starting PPO Training[/bold green]")
        print(f"Environment: {self.env.symbol} | Interval: {self.env.interval}")
        print(f"Episodes: {episodes} | State Dim: {self.env.state_dim}")
        print(f"Max Steps per Episode: {self.env.max_steps}")
        
        start_time = time.time()
        
        for episode in range(episodes):
            episode_start = time.time()
            
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            step_count = 0
            
            # Run episode
            while True:
                # Get action from agent
                action, log_prob, value = self.agent.policy.act(state.unsqueeze(0))
                action = action.item()
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(state, torch.tensor(action), reward, 
                                          log_prob, value, done)
                
                episode_reward += reward
                step_count += 1
                state = next_state
                
                if done:
                    break
            
            # Update agent periodically
            if (episode + 1) % update_frequency == 0:
                self.agent.update()
            
            # Get final metrics
            final_metrics = self.env.get_final_metrics()
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_returns.append(final_metrics.get('Total_Return', 0.0))
            self.episode_sharpe.append(final_metrics.get('Sharpe_Ratio', 0.0))
            self.episode_win_rates.append(final_metrics.get('Win_Rate', 0.0))
            
            # Logging
            episode_time = time.time() - episode_start
            
            if (episode + 1) % 10 == 0:
                self._log_progress(episode + 1, episode_reward, final_metrics, 
                                 step_count, episode_time)
            
            # Save model periodically
            if (episode + 1) % save_frequency == 0:
                model_path = os.path.join(model_dir, f"ppo_model_episode_{episode + 1}.pth")
                self.agent.save_model(model_path)
                print(f"[blue]Model saved to {model_path}[/blue]")
        
        # Final save
        final_model_path = os.path.join(model_dir, "ppo_model_final.pth")
        self.agent.save_model(final_model_path)
        
        total_time = time.time() - start_time
        print(f"\n[bold green]Training completed in {total_time:.2f} seconds[/bold green]")
        
        # Final summary
        self._print_final_summary()
        
        return self.agent
    
    def _log_progress(self, episode: int, episode_reward: float, 
                     metrics: Dict, step_count: int, episode_time: float):
        """Log training progress"""
        
        # Calculate recent averages
        recent_episodes = min(10, len(self.episode_rewards))
        avg_reward = np.mean(self.episode_rewards[-recent_episodes:])
        avg_return = np.mean(self.episode_returns[-recent_episodes:])
        avg_sharpe = np.mean(self.episode_sharpe[-recent_episodes:])
        avg_win_rate = np.mean(self.episode_win_rates[-recent_episodes:])
        
        table = Table(title=f"Episode {episode} Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="green")
        table.add_column("Avg (10ep)", style="yellow")
        
        table.add_row("Episode Reward", f"{episode_reward:.4f}", f"{avg_reward:.4f}")
        table.add_row("Total Return", f"{metrics.get('Total_Return', 0)*100:.2f}%", 
                     f"{avg_return*100:.2f}%")
        table.add_row("Sharpe Ratio", f"{metrics.get('Sharpe_Ratio', 0):.3f}", 
                     f"{avg_sharpe:.3f}")
        table.add_row("Win Rate", f"{metrics.get('Win_Rate', 0)*100:.1f}%", 
                     f"{avg_win_rate*100:.1f}%")
        table.add_row("Steps", f"{step_count}", "-")
        table.add_row("Time (s)", f"{episode_time:.2f}", "-")
        
        self.console.print(table)
    
    def _print_final_summary(self):
        """Print final training summary"""
        
        if not self.episode_rewards:
            return
            
        table = Table(title="Training Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Best", style="green")
        table.add_column("Worst", style="red")
        table.add_column("Average", style="yellow")
        
        table.add_row("Episode Reward", 
                     f"{max(self.episode_rewards):.4f}",
                     f"{min(self.episode_rewards):.4f}",
                     f"{np.mean(self.episode_rewards):.4f}")
        
        table.add_row("Total Return", 
                     f"{max(self.episode_returns)*100:.2f}%",
                     f"{min(self.episode_returns)*100:.2f}%",
                     f"{np.mean(self.episode_returns)*100:.2f}%")
        
        if self.episode_sharpe:
            valid_sharpe = [s for s in self.episode_sharpe if not np.isnan(s)]
            if valid_sharpe:
                table.add_row("Sharpe Ratio", 
                             f"{max(valid_sharpe):.3f}",
                             f"{min(valid_sharpe):.3f}",
                             f"{np.mean(valid_sharpe):.3f}")
        
        table.add_row("Win Rate", 
                     f"{max(self.episode_win_rates)*100:.1f}%",
                     f"{min(self.episode_win_rates)*100:.1f}%",
                     f"{np.mean(self.episode_win_rates)*100:.1f}%")
        
        self.console.print(table)

def main():
    """Main training function"""
    
    # Training configuration
    config = {
        'symbol': 'BTC-USDT',
        'chunks': 20,
        'interval': '5m',
        'age_days': 7,
        'context_length': 20,
        'data_source': 'binance',
        'initial_capital': 10000,
        'slippage_pct': 0.001,
        'commission_fixed': 0.0
    }
    
    trainer = PPOTrainer(**config)
    
    # Train the agent
    trained_agent = trainer.train(
        episodes=200,
        update_frequency=25,
        save_frequency=50,
        model_dir="trading_rl_models"
    )
    
    print("[bold green]Training completed![/bold green]")

if __name__ == "__main__":
    main() 