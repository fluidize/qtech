## Components

- `spot_trader.py`: Trading environment compatible with OpenAI Gym interface
- `models.py`: Neural network models for RL agents
- `memory.py`: Memory buffers for experience replay
- `agent.py`: Agent implementations (DQN, PPO)
- `train.py`: Training script with command-line interface
- `inference.py`: Inference script for running trained models

### Training an Agent

```bash
# Train a DQN agent on BTC-USDT with 5-minute intervals
python trading/brains/reinforcement/train.py --algorithm dqn --symbol BTC-USDT --interval 5min --episodes 100

# Train a PPO agent with customized parameters
python trading/brains/reinforcement/train.py --algorithm ppo --symbol ETH-USDT --interval 1min --episodes 200 --learning-rate 0.0003 --clip-epsilon 0.3 --entropy-coef 0.02
```

### Running Inference

```bash
# Run inference with a trained DQN model
python trading/brains/reinforcement/inference.py --model-path models/dqn_BTC-USDT_best.pth --algorithm dqn --symbol BTC-USDT --interval 5min --plot --save-plot

# Run inference with a trained PPO model on multiple episodes
python trading/brains/reinforcement/inference.py --model-path models/ppo_ETH-USDT_best.pth --algorithm ppo --symbol ETH-USDT --interval 1min --episodes 5 --verbose
```

## Environment Details

The trading environment (`SpotTradingEnv`) provides:

- Observation space: Historical OHLCV data, technical indicators, and account information
- Action space: 3 discrete actions (0 = Hold, 1 = Buy, 2 = Sell)
- Reward: Percentage change in portfolio value between steps
- Episode termination: When the data is exhausted or when specified steps are reached

## Agent Architecture

### DQN Agent

The DQN agent includes:
- Policy and target networks
- Experience replay (regular or prioritized)
- Epsilon-greedy exploration
- Target network updates

### PPO Agent

The PPO agent includes:
- Actor-critic network architecture
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Entropy bonus for exploration

## Hyperparameters

Important hyperparameters for each algorithm:

### DQN
- `learning_rate`: Learning rate for optimizer
- `gamma`: Discount factor
- `epsilon_start`, `epsilon_end`, `epsilon_decay`: Exploration parameters
- `target_update_freq`: Target network update frequency
- `batch_size`: Batch size for training
- `buffer_capacity`: Replay buffer capacity

### PPO
- `learning_rate`: Learning rate for optimizer
- `gamma`: Discount factor
- `gae_lambda`: GAE lambda parameter
- `clip_epsilon`: PPO clipping parameter
- `critic_coef`: Value loss coefficient
- `entropy_coef`: Entropy bonus coefficient
- `update_interval`: Steps before policy update
- `optimization_epochs`: Optimization epochs per update