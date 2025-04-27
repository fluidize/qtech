import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tqdm import tqdm
import sys

from spot_trader import SpotTradingEnv
from agent import DQNAgent, PPOAgent
from models import DQNNetwork, PPONetwork

def train_dqn(
    env,
    agent,
    num_episodes=100,
    max_steps=1000,
    eval_interval=10,
    save_interval=25,
    log_interval=1,
    model_dir='models',
    model_name='dqn_trader'
):
    """
    Train a DQN agent on the trading environment
    
    Args:
        env: Trading environment
        agent: DQN agent
        num_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        eval_interval: Interval between evaluations
        save_interval: Interval between model saves
        log_interval: Interval between logging
        model_dir: Directory to save models
        model_name: Base name for model files
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Training stats
    episode_rewards = []
    episode_losses = []
    eval_returns = []
    
    best_eval_return = -float('inf')
    
    print(f"Starting training for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        
        # Progress bar for steps
        with tqdm(total=max_steps, desc=f"Episode {episode}/{num_episodes}", leave=False) as pbar:
            for step in range(max_steps):
                # Select action
                action = agent.select_action(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, next_state, reward, done)
                
                # Update agent
                loss = agent.update()
                if loss is not None:
                    episode_loss.append(loss)
                
                # Update epsilon
                agent.update_epsilon()
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'reward': f"{episode_reward:.2f}",
                    'epsilon': f"{agent.epsilon:.2f}",
                    'loss': f"{np.mean(episode_loss) if episode_loss else 0:.4f}"
                })
                
                if done:
                    break
        
        # Record episode stats
        episode_rewards.append(episode_reward)
        episode_losses.append(np.mean(episode_loss) if episode_loss else 0)
        
        # Logging
        if episode % log_interval == 0:
            print(f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f}, "
                 f"Avg Loss: {np.mean(episode_loss) if episode_loss else 0:.4f}, "
                 f"Epsilon: {agent.epsilon:.2f}")
            
        # Evaluation
        if episode % eval_interval == 0:
            eval_return = evaluate_agent(env, agent)
            eval_returns.append(eval_return)
            
            print(f"Evaluation after episode {episode}: Return = {eval_return:.2f}")
            
            # Save best model
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                agent.save(os.path.join(model_dir, f"{model_name}_best.pth"))
                print(f"New best model saved with return {best_eval_return:.2f}")
        
        # Regular model saving
        if episode % save_interval == 0:
            agent.save(os.path.join(model_dir, f"{model_name}_{episode}.pth"))
    
    # Save final model
    agent.save(os.path.join(model_dir, f"{model_name}_final.pth"))
    
    # Plot training results
    plot_results(episode_rewards, episode_losses, eval_returns, eval_interval, model_name)
    
    return agent

def train_ppo(
    env,
    agent,
    num_episodes=100,
    max_steps=1000,
    update_interval=128,
    eval_interval=10,
    save_interval=25,
    log_interval=1,
    model_dir='models',
    model_name='ppo_trader'
):
    """
    Train a PPO agent on the trading environment
    
    Args:
        env: Trading environment
        agent: PPO agent
        num_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        update_interval: Number of steps before policy update
        eval_interval: Interval between evaluations
        save_interval: Interval between model saves
        log_interval: Interval between logging
        model_dir: Directory to save models
        model_name: Base name for model files
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Training stats
    episode_rewards = []
    episode_losses = []
    eval_returns = []
    steps_done = 0
    
    best_eval_return = -float('inf')
    
    print(f"Starting training for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_loss = {"total": []}
        
        # Progress bar for steps
        with tqdm(total=max_steps, desc=f"Episode {episode}/{num_episodes}", leave=False) as pbar:
            for step in range(max_steps):
                # Select action
                action = agent.select_action(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store outcome for this action
                agent.store_outcome(reward, done)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                steps_done += 1
                
                # Update policy if needed
                if steps_done % update_interval == 0:
                    loss_dict = agent.update(next_state)
                    for k, v in loss_dict.items():
                        if k not in episode_loss:
                            episode_loss[k] = []
                        episode_loss[k].append(v)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'reward': f"{episode_reward:.2f}",
                    'loss': f"{np.mean(episode_loss['total']) if episode_loss['total'] else 0:.4f}"
                })
                
                if done:
                    break
            
            # Final update at end of episode
            if len(agent.states) > 0:
                loss_dict = agent.update()
                for k, v in loss_dict.items():
                    if k not in episode_loss:
                        episode_loss[k] = []
                    episode_loss[k].append(v)
        
        # Record episode stats
        episode_rewards.append(episode_reward)
        episode_losses.append(np.mean(episode_loss["total"]) if episode_loss["total"] else 0)
        
        # Logging
        if episode % log_interval == 0:
            loss_msg = ", ".join([f"{k}: {np.mean(v):.4f}" for k, v in episode_loss.items() if v])
            print(f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f}, "
                 f"Losses: {loss_msg}")
            
        # Evaluation
        if episode % eval_interval == 0:
            eval_return = evaluate_agent(env, agent)
            eval_returns.append(eval_return)
            
            print(f"Evaluation after episode {episode}: Return = {eval_return:.2f}")
            
            # Save best model
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                agent.save(os.path.join(model_dir, f"{model_name}_best.pth"))
                print(f"New best model saved with return {best_eval_return:.2f}")
        
        # Regular model saving
        if episode % save_interval == 0:
            agent.save(os.path.join(model_dir, f"{model_name}_{episode}.pth"))
    
    # Save final model
    agent.save(os.path.join(model_dir, f"{model_name}_final.pth"))
    
    # Plot training results
    plot_results(episode_rewards, episode_losses, eval_returns, eval_interval, model_name)
    
    return agent

def evaluate_agent(env, agent, num_episodes=5):
    """
    Evaluate agent performance on several episodes
    
    Args:
        env: Trading environment
        agent: Agent to evaluate
        num_episodes: Number of evaluation episodes
        
    Returns:
        Average return across episodes
    """
    returns = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Use greedy/deterministic action selection
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            episode_reward += reward
            
        returns.append(episode_reward)
    
    # Show performance metrics
    metrics = env.get_performance_metrics()
    print(f"Evaluation metrics: {metrics}")
    
    return np.mean(returns)

def plot_results(rewards, losses, eval_returns, eval_interval, model_name):
    """
    Plot training results
    
    Args:
        rewards: List of episode rewards
        losses: List of episode losses
        eval_returns: List of evaluation returns
        eval_interval: Interval between evaluations
        model_name: Base name for plot file
    """
    plt.figure(figsize=(12, 10))
    
    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    
    # Plot losses
    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    # Plot evaluation returns
    plt.subplot(3, 1, 3)
    eval_episodes = np.arange(1, len(eval_returns) + 1) * eval_interval
    plt.plot(eval_episodes, eval_returns)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title('Evaluation Returns')
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_results.png")
    plt.close()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train RL agents for trading')
    
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['dqn', 'ppo'],
                        help='RL algorithm to use (default: ppo)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes (default: 100)')
    parser.add_argument('--symbol', type=str, default='BTC-USDT',
                        help='Trading pair symbol (default: BTC-USDT)')
    parser.add_argument('--interval', type=str, default='5min',
                        help='Trading interval (default: 5min)')
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                        help='Initial capital (default: 10000.0)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer dimensions (default: [128, 64])')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--buffer-capacity', type=int, default=100000,
                        help='Replay buffer capacity (default: 100000)')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='Use prioritized experience replay (default: False)')
    
    dqn_group = parser.add_argument_group('DQN')
    dqn_group.add_argument('--epsilon-start', type=float, default=1.0,
                           help='Initial exploration rate (default: 1.0)')
    dqn_group.add_argument('--epsilon-end', type=float, default=0.05,
                           help='Final exploration rate (default: 0.05)')
    dqn_group.add_argument('--epsilon-decay', type=float, default=0.999,
                           help='Exploration decay rate (default: 0.999)')
    dqn_group.add_argument('--target-update-freq', type=int, default=10,
                           help='Target network update frequency (default: 10)')
    
    ppo_group = parser.add_argument_group('PPO')
    ppo_group.add_argument('--gae-lambda', type=float, default=0.95,
                           help='GAE lambda parameter (default: 0.95)')
    ppo_group.add_argument('--clip-epsilon', type=float, default=0.2,
                           help='PPO clipping parameter (default: 0.2)')
    ppo_group.add_argument('--critic-coef', type=float, default=0.5,
                           help='Critic loss coefficient (default: 0.5)')
    ppo_group.add_argument('--entropy-coef', type=float, default=0.01,
                           help='Entropy bonus coefficient (default: 0.01)')
    ppo_group.add_argument('--update-interval', type=int, default=128,
                           help='Steps before policy update (default: 128)')
    ppo_group.add_argument('--optimization-epochs', type=int, default=10,
                           help='Optimization epochs per update (default: 10)')
    
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Interval between evaluations (default: 10)')
    parser.add_argument('--save-interval', type=int, default=25,
                        help='Interval between model saves (default: 25)')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save models (default: models)')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Base name for model files (default: algorithm_symbol_timestamp)')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to model file to load (default: None)')
    
    return parser.parse_args()

class Config:
    def __init__(self):
        self.algorithm = 'ppo'  # RL algorithm to use
        self.episodes = 50  # Number of training episodes
        self.symbol = 'BTC-USDT'  # Trading pair symbol
        self.interval = '1min'  # Trading interval
        self.initial_capital = 10000.0  # Initial capital
        self.hidden_dims = [128, 64]  # Hidden layer dimensions
        self.learning_rate = 0.0001  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.batch_size = 128  # Batch size
        self.buffer_capacity = 100000  # Replay buffer capacity
        self.prioritized_replay = False  # Use prioritized experience replay
        
        # DQN specific parameters
        self.epsilon_start = 1.0  # Initial exploration rate
        self.epsilon_end = 0.05  # Final exploration rate
        self.epsilon_decay = 0.999  # Exploration decay rate
        self.target_update_freq = 10  # Target network update frequency
        
        # PPO specific parameters
        self.gae_lambda = 0.95  # GAE lambda parameter
        self.clip_epsilon = 0.2  # PPO clipping parameter
        self.critic_coef = 0.5  # Critic loss coefficient
        self.entropy_coef = 0.01  # Entropy bonus coefficient
        self.update_interval = 128  # Steps before policy update
        self.optimization_epochs = 10  # Optimization epochs per update
         
        # Training parameters
        self.max_steps = 1000  # Maximum steps per episode
        self.eval_interval = 10  # Interval between evaluations
        self.save_interval = 25  # Interval between model saves
        self.model_dir = 'models'  # Directory to save models
        self.model_name = None  # Base name for model files
        self.load_model = None  # Path to model file to load

def main():
    """Main function for training RL agents"""
    # args = parse_arguments()
    args = Config()
    
    # Generate model name if not provided
    if args.model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_name = f"{args.algorithm}_{args.symbol.replace('-', '_')}_{timestamp}"
    
    # Create environment
    env = SpotTradingEnv(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        interval=args.interval,
        window_size=20,  # Fixed for now
        commission_rate=0.001,  # Fixed for now
        reward_scaling=1.0,  # Fixed for now
        include_position=True,  # Fixed for now
        use_kucoin=True  # Fixed for now
    )
    
    # Get observation dimension from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent based on selected algorithm
    if args.algorithm == 'dqn':
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=args.hidden_dims,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_update_freq=args.target_update_freq,
            batch_size=args.batch_size,
            buffer_capacity=args.buffer_capacity,
            prioritized_replay=args.prioritized_replay
        )
        
        # Load model if specified
        if args.load_model:
            print(f"Loading model from {args.load_model}")
            agent.load(args.load_model)
        
        # Train agent
        train_dqn(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            model_dir=args.model_dir,
            model_name=args.model_name
        )
    
    elif args.algorithm == 'ppo':
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=args.hidden_dims,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            critic_coef=args.critic_coef,
            entropy_coef=args.entropy_coef,
            batch_size=args.batch_size,
            optimization_epochs=args.optimization_epochs,
            continuous_actions=False  # Fixed for now
        )
        
        # Load model if specified
        if args.load_model:
            print(f"Loading model from {args.load_model}")
            agent.load(args.load_model)
        
        # Train agent
        train_ppo(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            update_interval=args.update_interval,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            model_dir=args.model_dir,
            model_name=args.model_name
        )

        evaluate_agent(env, agent)

if __name__ == "__main__":
    main()