# Simple RL Trading Agent

This is a simplified Reinforcement Learning (RL) model for trading cryptocurrency or stocks. It consists of a DQN (Deep Q-Network) agent that learns to make trading decisions in a simulated environment.

## Structure

The implementation consists of just three files:

1. `simple_trading_env.py` - The trading environment
2. `simple_dqn_agent.py` - The DQN agent implementation
3. `train.py` - Training script 

## Mathematics Behind the Model

### 1. RL Framework

This model uses the standard RL framework with:
- **States (S)**: Market data and current position
- **Actions (A)**: Hold (0), Buy (1), Sell (2)
- **Rewards (R)**: Based on profit/loss and portfolio growth
- **Transitions**: Market movements + agent actions → new states

### 2. DQN (Deep Q-Network) Algorithm

The DQN algorithm is based on Q-learning with a neural network approximator:

#### Q-Learning Update Rule

The core equation in Q-learning is:

Q(s, a) ← Q(s, a) + α[r + γ · max<sub>a'</sub> Q(s', a') - Q(s, a)]

Where:
- Q(s, a) is the value of taking action a in state s
- α is the learning rate
- r is the reward
- γ (gamma) is the discount factor
- s' is the next state
- max<sub>a'</sub> Q(s', a') is the maximum Q-value for the next state

#### Neural Network Training

The DQN approximates the Q-function with a neural network by minimizing the loss:

L = (r + γ · max<sub>a'</sub> Q(s', a', θ<sup>-</sup>) - Q(s, a, θ))<sup>2</sup>

Where:
- θ are the parameters of the Q-network
- θ<sup>-</sup> are the parameters of the target network

### 3. Exploration Strategy

- **Epsilon-Greedy**: With probability ε, take a random action (explore)
- **Epsilon Decay**: Gradually reduce ε from 1.0 to 0.01 to transition from exploration to exploitation

### 4. Experience Replay

- Store (state, action, reward, next_state, done) tuples in a replay buffer
- Sample random batches to break correlation between consecutive samples
- Improves learning stability and efficiency

### 5. Reward Design

Rewards are designed to encourage profit-seeking behavior:
- Profit on sale: Proportional to profit percentage
- Position value change: Small reward for price increases while holding
- Trading fee penalty: Small negative reward for excessive trading
- Inactivity penalty: Small negative reward for extended periods without trading

## Trading Environment Features

- Realistic market simulation with price data from Yahoo Finance
- Technical indicators: Moving averages, RSI, volatility
- Transaction fees/commissions
- Portfolio tracking

## How to Use

1. **Install Dependencies**:
   ```
   pip install numpy pandas torch matplotlib tqdm yfinance sklearn
   ```

2. **Run Training**:
   ```
   python train.py
   ```

3. **Modify Parameters**:
   - Change the trading symbol in `train.py`
   - Adjust RL hyperparameters in the agent initialization
   - Modify reward function in `simple_trading_env.py`

## Improvement Ideas

1. **Better Feature Engineering**: Add more technical indicators or alternative data
2. **Advanced Reward Functions**: Design rewards that better reflect risk-adjusted returns
3. **Advanced Architectures**: Try LSTM or transformer networks for time series processing
4. **More Action Space**: Allow partial buys/sells or leverage
5. **Risk Management**: Add stop-loss or position sizing mechanisms 