import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from simple_trading_env import TradingEnv
from simple_dqn_agent import DQNAgent

def train_agent(env, agent, episodes=100, max_steps=None, eval_frequency=10):
    """Train the DQN agent on the trading environment."""
    
    # Track performance for visualization
    episode_rewards = []
    portfolio_values = []
    evaluation_returns = []
    
    for episode in range(1, episodes + 1):
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Training loop for this episode
        with tqdm(desc=f"Episode {episode}/{episodes}", unit="step") as pbar:
            while not done:
                # Select action
                action = agent.select_action(state)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Handle any NaN rewards
                if np.isnan(reward):
                    reward = 0.0
                
                # Store transition in replay buffer
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                # Update agent
                agent.update()
                
                # Update state and metrics
                state = next_state
                total_reward += reward
                steps += 1
                
                # Check if we've reached max steps
                if max_steps and steps >= max_steps:
                    done = True
                
                pbar.update(1)
                pbar.set_postfix({
                    "reward": f"{total_reward:.2f}", 
                    "portfolio": f"${info['portfolio_value']:.2f}",
                    "epsilon": f"{agent.epsilon:.2f}"
                })
        
        # Track metrics
        episode_rewards.append(total_reward)
        portfolio_values.append(env.portfolio_values[-1] if env.portfolio_values else env.initial_balance)
        
        # Periodically evaluate the agent (without exploration)
        if episode % eval_frequency == 0:
            eval_return = evaluate_agent(env, agent)
            evaluation_returns.append(eval_return)
            print(f"\nEvaluation after episode {episode}: Return = {eval_return:.2f}")
        
        # Save the model periodically
        if episode % 20 == 0:
            agent.save(f"trading_dqn_model_ep{episode}.pt")
    
    # Save final model
    agent.save("trading_dqn_model_final.pt")
    
    # Plot training results
    plot_results(episode_rewards, portfolio_values, evaluation_returns, eval_frequency)
    
    return agent

def evaluate_agent(env, agent, episodes=5):
    """Evaluate the agent without exploration."""
    returns = []
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Select action (without exploration)
            action = agent.select_action(state, training=False)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Handle any NaN rewards
            if np.isnan(reward):
                reward = 0.0
            
            # Update state and metrics
            state = next_state
            total_reward += reward
        
        returns.append(total_reward)
    
    return np.mean(returns)

def plot_results(rewards, portfolio_values, eval_returns, eval_frequency):
    """Plot training results."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot episode rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Plot portfolio values
    ax2.plot(portfolio_values)
    ax2.set_title('Final Portfolio Value')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Portfolio Value ($)')
    
    # Plot evaluation returns
    eval_episodes = [i * eval_frequency for i in range(len(eval_returns))]
    ax3.plot(eval_episodes, eval_returns)
    ax3.set_title('Evaluation Returns')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Return')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    # Create environment
    env = TradingEnv(symbol="BTC-USDT", initial_balance=10000.0, window_size=10)
    
    # Download and prepare data - now returns price_data and feature_data
    price_data, feature_data = env.fetch_data(chunks=1, age_days=0, interval="1min")
    
    # Initialize the agent
    # Calculate input dimensions based on market data and position info
    state = env.reset()
    market_data_shape = state['market_data'].shape
    position_shape = state['position'].shape
    
    # Calculate total state dimension - flatten all dimensions
    market_data_size = np.prod(market_data_shape)
    position_size = np.prod(position_shape)
    state_dim = int(market_data_size + position_size)
    
    print(f"State dimension: {state_dim}")
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=3,  # hold, buy, sell
        learning_rate=0.0001,  # Lower learning rate for unscaled data
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=50000,
        batch_size=32,
        target_update_freq=10
    )
    
    # Train the agent
    trained_agent = train_agent(
        env=env,
        agent=agent,
        episodes=50,
        max_steps=None,  # Set to None to run until end of data
        eval_frequency=5
    )
    
    # Get final performance summary
    summary = env.get_performance_summary()
    print("\nTraining Complete!")
    print(f"Initial Portfolio Value: ${summary['initial_balance']:.2f}")
    print(f"Final Portfolio Value: ${summary['final_balance']:.2f}")
    print(f"Profit: ${summary['profit']:.2f} ({summary['profit_percent']:.2f}%)")
    print(f"Total Trades: {summary['trade_count']} (Buys: {summary['buys']}, Sells: {summary['sells']})") 