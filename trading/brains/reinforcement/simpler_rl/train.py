import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from simple_trading_env import TradingEnv
from simple_dqn_agent import DQNAgent

def train_agent(env, agent, episodes=100, max_steps=None, eval_frequency=10, save_frequency=None):
    """Train the DQN agent on the trading environment."""
    
    episode_rewards = []
    portfolio_values = []
    evaluation_returns = []
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        total_buys = 0
        total_sells = 0
        steps = 0
        
        with tqdm(desc=f"Episode {episode}/{episodes}", unit="step") as progress_bar:
            while not done:
                action = agent.select_action(state)
                
                next_state, reward, done, info = env.step(action) # Take action in environment
                
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                agent.update()
                    
                state = next_state
                total_reward += reward
                steps += 1
                total_buys += 1 if action == 1 else 0
                total_sells += 1 if action == 2 else 0

                if max_steps and steps >= max_steps:
                    done = True
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "reward": f"{total_reward:.2f}", 
                    "portfolio": f"${info['portfolio_value']:.2f}",
                    "epsilon": f"{agent.epsilon:.2f}",
                    "buy_signals": f"{total_buys}",
                    "sell_signals": f"{total_sells}"
                })
        
        episode_rewards.append(total_reward)
        portfolio_values.append(env.portfolio_values[-1] if env.portfolio_values else env.initial_balance)
        
        # Periodically evaluate the agent (without exploration)
        if episode % eval_frequency == 0:
            eval_return = evaluate_agent(env, agent)
            evaluation_returns.append(eval_return)
            print(f"\nEvaluation after episode {episode}: Return = {eval_return:.2f} | Trades = {env.get_performance_summary()['trade_count']}")
        
        if (save_frequency) and (episode % save_frequency == 0):
            agent.save(f"trading_dqn_model_ep{episode}.pt")
    
    if save_frequency:
        agent.save(f"trading_dqn_model_final.pt")
    
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
            action = agent.select_action(state, training=False)
            
            next_state, reward, done, info = env.step(action)
            
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
    # plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    env = TradingEnv(symbol="BTC-USDT", 
                     initial_balance=10000.0, 
                     window_size=10,
                     commission=0.001
    )
    
    price_data, feature_data = env.fetch_data(chunks=1, age_days=0, interval="5min")
    
    # Initial state to determine dimensions
    state = env.reset()
    market_data_shape = state['market_data'].shape
    position_shape = state['position'].shape
    
    market_data_size = np.prod(market_data_shape)
    position_size = np.prod(position_shape)
    state_dim = int(market_data_size + position_size)
    
    if len(feature_data.columns) * env.window_size != market_data_size:
        print(f"WARNING: Feature dimensions mismatch. Expected {len(feature_data.columns) * env.window_size}, got {market_data_size}")
    
    print(f"Market data shape: {market_data_shape}")
    print(f"Position shape: {position_shape}")
    print(f"Feature count: {len(feature_data.columns)}")
    print(f"Window size: {env.window_size}")
    print(f"Total state dimension: {state_dim}")
    
    market_data_flat = state['market_data'].flatten()
    position_flat = state['position'].flatten()
    combined = np.concatenate([market_data_flat, position_flat])
    
    print(f"Combined state vector length: {len(combined)}")
    print(f"State vector and calculated dimensions match: {len(combined) == state_dim}")
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=3,  # hold, buy, sell
        learning_rate=0.0005,  # Adjusted learning rate for better stability
        gamma=0.97,  # Slightly lower discount factor for more immediate rewards
        epsilon_start=1.0,
        epsilon_end=0.05,  # Lower end epsilon for more exploitation in later stages
        epsilon_decay=1-(1/2000),  # Slower decay for better exploration
        buffer_size=100000,  # Larger buffer size
        batch_size=128,  # Larger batch size for better gradient estimates
        target_update_freq=5  # More frequent target updates
    )
    
    trained_agent = train_agent(
        env=env,
        agent=agent,
        episodes=100,  # More episodes for better learning
        max_steps=None,  # Set to None to run until end of data
        eval_frequency=5,
    )
    
    summary = env.get_performance_summary()
    print("\nTraining Complete!")
    print(f"Initial Portfolio Value: ${summary['initial_balance']:.2f}")
    print(f"Final Portfolio Value: ${summary['final_balance']:.2f}")
    print(f"Profit: ${summary['profit']:.2f} ({summary['profit_percent']:.2f}%)")
    print(f"Total Trades: {summary['trade_count']} (Buys: {summary['buys']}, Sells: {summary['sells']})") 