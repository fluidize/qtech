import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler

from simple_trading_env import TradingEnv
from simple_agent import PPOAgent

class DataNormalizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fit = False
        
    def fit_transform(self, data):
        if not self.is_fit:
            self.scaler.fit(data)
            self.is_fit = True
        return self.scaler.transform(data)
    
    def transform(self, data):
        return self.scaler.transform(data)

def train_agent(env, agent, episodes=100, max_steps=None, eval_frequency=10, save_frequency=None):
    """Train the PPO agent on the trading environment."""
    episode_rewards = []
    portfolio_values = []
    evaluation_returns = []
    best_eval_return = float('-inf')
    
    # Initialize data normalizer
    normalizer = DataNormalizer()
    
    # Early stopping parameters
    patience = 20
    no_improvement_count = 0
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        # Normalize market data
        if 'market_data' in state:
            market_data_shape = state['market_data'].shape
            flat_data = state['market_data'].reshape(-1, market_data_shape[-1])
            normalized_data = normalizer.fit_transform(flat_data)
            state['market_data'] = normalized_data.reshape(market_data_shape)
            
        done = False
        total_reward = 0
        total_buys = 0
        total_sells = 0
        steps = 0
        episode_portfolio_values = []
        
        with tqdm(desc=f"Episode {episode}/{episodes}", unit="step") as progress_bar:
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Normalize next state market data
                if 'market_data' in next_state:
                    market_data_shape = next_state['market_data'].shape
                    flat_data = next_state['market_data'].reshape(-1, market_data_shape[-1])
                    normalized_data = normalizer.transform(flat_data)
                    next_state['market_data'] = normalized_data.reshape(market_data_shape)
                
                agent.store_outcome(reward, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                # Track actions and portfolio
                total_buys += 1 if action == 2 else 0  # Long
                total_sells += 1 if action == 0 else 0  # Short
                episode_portfolio_values.append(info['portfolio_value'])
                
                if max_steps and steps >= max_steps:
                    done = True
                    
                # Update progress bar with more metrics
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "reward": f"{total_reward:.2f}",
                    "portfolio": f"${info['portfolio_value']:.2f}",
                    "buys": total_buys,
                    "sells": total_sells,
                    "avg_trade": f"${np.mean(episode_portfolio_values):.2f}"
                })
        
        episode_rewards.append(total_reward)
        portfolio_values.append(env.portfolio_values[-1] if env.portfolio_values else env.initial_balance)
        
        # Update agent with normalized states
        agent.update(next_state)
        
        # Evaluation phase
        if episode % eval_frequency == 0:
            eval_return = evaluate_agent(env, agent, normalizer)
            evaluation_returns.append(eval_return)
            
            # Save best model
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                agent.save("best_trading_model.pt")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            print(f"\nEvaluation after episode {episode}:")
            print(f"Return = {eval_return:.2f}")
            print(f"Best Return = {best_eval_return:.2f}")
            print(f"Trade Summary: {env.get_performance_summary()}")
            
            # Early stopping check
            if no_improvement_count >= patience:
                print(f"\nStopping early due to no improvement for {patience} evaluations")
                break
        
        # Regular checkpoints
        if save_frequency and episode % save_frequency == 0:
            agent.save(f"trading_ppo_model_ep{episode}.pt")
    
    # Save final model
    if save_frequency:
        agent.save("trading_ppo_model_final.pt")
    
    plot_results(episode_rewards, portfolio_values, evaluation_returns, eval_frequency)
    return agent

def evaluate_agent(env, agent, normalizer, episodes=5):
    """Evaluate the agent without exploration."""
    returns = []
    for _ in range(episodes):
        state = env.reset()
        if 'market_data' in state:
            market_data_shape = state['market_data'].shape
            flat_data = state['market_data'].reshape(-1, market_data_shape[-1])
            normalized_data = normalizer.transform(flat_data)
            state['market_data'] = normalized_data.reshape(market_data_shape)
        
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            if 'market_data' in next_state:
                market_data_shape = next_state['market_data'].shape
                flat_data = next_state['market_data'].reshape(-1, market_data_shape[-1])
                normalized_data = normalizer.transform(flat_data)
                next_state['market_data'] = normalized_data.reshape(market_data_shape)
            
            state = next_state
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns)

def plot_results(rewards, portfolio_values, eval_returns, eval_frequency):
    """Plot training results with improved visualization."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot episode rewards with rolling mean
    ax1.plot(rewards, alpha=0.6, label='Raw')
    window = min(len(rewards) // 10, 20)
    rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(range(window-1, len(rewards)), rolling_mean, label=f'{window}-Episode Moving Avg')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    ax1.legend()
    
    # Plot portfolio values with percentage gains
    initial_value = portfolio_values[0]
    portfolio_returns = [(v - initial_value) / initial_value * 100 for v in portfolio_values]
    ax2.plot(portfolio_returns)
    ax2.set_title('Portfolio Returns (%)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True)
    
    # Plot evaluation returns
    eval_episodes = [i * eval_frequency for i in range(len(eval_returns))]
    ax3.plot(eval_episodes, eval_returns, marker='o')
    ax3.set_title('Evaluation Returns')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Return')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    # Initialize environment with more conservative settings
    env = TradingEnv(
        initial_balance=10000.0, 
        window_size=20,  # Increased window size
        commission=0.001,
        max_position_size=0.2  # More conservative position sizing
    )
    
    # Fetch data with more history for better learning
    price_data, feature_data = env.fetch_data(
        symbol="BTC-USDT",
        chunks=60,  # More historical data
        age_days=0,
        interval="15min"
    )
    
    state = env.reset()
    market_data_shape = state['market_data'].shape
    position_shape = state['position'].shape
    
    market_data_size = np.prod(market_data_shape)
    position_size = np.prod(position_shape)
    state_dim = int(market_data_size + position_size)
    
    if len(feature_data.columns) * env.window_size != market_data_size:
        print(f"WARNING: Feature dimensions mismatch. Expected {len(feature_data.columns) * env.window_size}, got {market_data_size}")
    
    # Initialize agent with tuned hyperparameters
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=3,
        learning_rate=1e-4,  # Lower learning rate
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        critic_coef=1.0,  # Increased critic importance
        entropy_coef=0.02,  # Slightly increased exploration
        batch_size=128,  # Larger batch size
        optimization_epochs=5  # Reduced to prevent overfitting
    )
    
    print("Starting training...")
    
    trained_agent = train_agent(
        env=env,
        agent=agent,
        episodes=200,
        max_steps=None,
        eval_frequency=5,  # More frequent evaluation
        save_frequency=25
    )
    
    # Print final performance summary
    summary = env.get_performance_summary()
    print("\nTraining Complete!")
    print(f"Initial Portfolio Value: ${summary['initial_balance']:.2f}")
    print(f"Final Portfolio Value: ${summary['final_balance']:.2f}")
    print(f"Profit: ${summary['profit']:.2f} ({summary['profit_percent']:.2f}%)")
    print(f"Total Trades: {summary['trade_count']}")
    print(f"Buy/Sell Ratio: {summary['buys']}/{summary['sells']}") 