import argparse
import torch
import numpy as np
import pandas as pd
import time
import os
import sys
from datetime import datetime
import plotly.graph_objects as go

from spot_trader import SpotTradingEnv
from agent import DQNAgent, PPOAgent

def parse_arguments():
    """Parse command line arguments for inference"""
    parser = argparse.ArgumentParser(description='Run inference with trained RL agents')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'ppo'],
                        help='RL algorithm (default: dqn)')
    parser.add_argument('--symbol', type=str, default='BTC-USDT',
                        help='Trading pair symbol (default: BTC-USDT)')
    parser.add_argument('--interval', type=str, default='5min',
                        help='Trading interval (default: 5min)')
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                        help='Initial capital (default: 10000.0)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run (default: 1)')
    parser.add_argument('--chunks', type=int, default=5,
                        help='Number of data chunks (default: 5)')
    parser.add_argument('--age-days', type=int, default=0,
                        help='Age of data in days (default: 0, most recent)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate performance plot')
    parser.add_argument('--save-plot', action='store_true',
                        help='Save performance plot to file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    
    return parser.parse_args()

def run_inference(env, agent, args):
    """
    Run inference with a trained agent
    
    Args:
        env: Trading environment
        agent: Trained agent (DQN or PPO)
        args: Command line arguments
    
    Returns:
        Dictionary of performance metrics
    """
    print(f"Running inference with {args.algorithm.upper()} on {args.symbol}...")
    
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        step = 0
        
        # Track episode metrics
        episode_reward = 0
        trade_count = 0
        
        while not done:
            # Select action using trained policy
            action = agent.select_action(state, evaluate=True)
            
            # Track if we make a trade
            position_before = env.position_history[-1] if env.position_history else 0
            
            # Execute action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update state
            state = next_state
            episode_reward += reward
            step += 1
            
            # Check if we made a trade
            position_after = env.position_history[-1]
            if position_before != position_after:
                trade_count += 1
            
            # Print progress if verbose
            if args.verbose and step % 10 == 0:
                print(f"Episode {episode}/{args.episodes}, Step {step}, " +
                      f"Reward: {episode_reward:.2f}, " +
                      f"Portfolio Value: ${info['portfolio_value']:.2f}, " +
                      f"Return: {info['portfolio_return']*100:.2f}%, " +
                      f"Position: {'Long' if info['position'] > 0 else 'None'}")
        
        # Print episode summary
        print(f"Episode {episode} completed - " +
              f"Steps: {step}, " +
              f"Total Reward: {episode_reward:.2f}, " +
              f"Trades: {trade_count}, " +
              f"Final Portfolio Value: ${env.trading_env.portfolio.total_value:.2f}, " +
              f"Return: {env.trading_env.portfolio.total_profit_loss_pct:.2f}%")
    
    # Get performance metrics
    metrics = env.get_performance_metrics()
    
    # Print detailed metrics
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Generate and show plot if requested
    if args.plot or args.save_plot:
        plot_performance(env, args)
    
    return metrics

def plot_performance(env, args):
    """
    Generate and optionally save performance plot
    
    Args:
        env: Trading environment after running inference
        args: Command line arguments
    """
    fig = env.plot_performance(show_graph=args.plot)
    
    if args.save_plot:
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.algorithm}_{args.symbol.replace('-', '_')}_{timestamp}.html"
        
        # Save plot
        fig.write_html(filename)
        print(f"Performance plot saved to {filename}")

def load_agent(algorithm, model_path, env):
    """
    Load a trained agent from file
    
    Args:
        algorithm: Algorithm type ('dqn' or 'ppo')
        model_path: Path to model file
        env: Trading environment to get state/action dimensions
        
    Returns:
        Loaded agent
    """
    # Get state and action dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if algorithm == 'dqn':
        # Create DQN agent with placeholder parameters
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[128, 64],  # Will be overwritten by loaded model
            epsilon_start=0.0,      # Use deterministic policy for inference
            epsilon_end=0.0,
            epsilon_decay=1.0
        )
    else:  # PPO
        # Create PPO agent with placeholder parameters
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 128],  # Will be overwritten by loaded model
            continuous_actions=False
        )
    
    # Load model weights
    print(f"Loading model from {model_path}")
    agent.load(model_path)
    
    return agent

def main():
    """Main function for inference"""
    args = parse_arguments()
    
    # Create environment
    env = SpotTradingEnv(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        chunks=args.chunks,
        interval=args.interval,
        age_days=args.age_days,
        window_size=20,  # Fixed for now
        commission_rate=0.001,  # Fixed for now
        reward_scaling=1.0,     # Fixed for now
        include_position=True,  # Fixed for now
        use_kucoin=True         # Fixed for now
    )
    
    # Load trained agent
    agent = load_agent(args.algorithm, args.model_path, env)
    
    # Run inference
    start_time = time.time()
    metrics = run_inference(env, agent, args)
    end_time = time.time()
    
    print(f"Inference completed in {end_time - start_time:.2f} seconds")
    
if __name__ == "__main__":
    main() 