import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Dict, List, Tuple
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.LayerNorm(out_features)
        self.lin2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.LayerNorm(out_features)
        
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        residual = x
        
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.lin2(x)
        x = self.bn2(x)
        
        x += self.shortcut(residual)
        x = F.leaky_relu(x, 0.2)
        
        return x

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for the trading agent with GRU layer.
    Maps state observations to Q-values for each action.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 3, lstm_layers: int = 1):
        super().__init__()
        
        # GRU layer for processing sequential data
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Residual blocks for further processing
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """Forward pass through the network with GRU processing."""
        # Reshape input for GRU (batch_size, sequence_length, features)
        # For a single input, add batch and sequence dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        x, _ = self.lstm(x)
        
        # Get the output from the last time step
        x = x[:, -1, :]
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.output_layer(x)
        
        return x

class ReplayBuffer:
    """
    Experience replay buffer to store and sample transitions.
    Helps break correlation between consecutive samples.
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """Sample a batch of transitions randomly."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def sample_all(self) -> List:
        """Sample all transitions."""
        return list(self.buffer)
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

class DQNAgent:
    """
    DQN Agent that learns to trade through Q-learning.
    
    Uses epsilon-greedy exploration and target networks for stability.
    The underlying neural network uses GRU (Gated Recurrent Unit) layers
    to better capture sequential patterns and time dependencies in the
    market data, which can improve prediction performance for time series data.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int = 3,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,  # Discount factor
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 10):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Initialize Q-networks (main and target)
        self.q_network = DQNNetwork(input_dim=state_dim, hidden_dim=64, output_dim=action_dim)
        self.target_network = DQNNetwork(input_dim=state_dim, hidden_dim=64, output_dim=action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=learning_rate,
            amsgrad=True,
            weight_decay=1e-6
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Track training metrics
        self.losses = []
    
    def select_action(self, state, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy strategy.
        
        During training, we explore with probability epsilon.
        During evaluation, we always exploit.
        """
        if training and random.random() < self.epsilon: #randomly select an action / explore
            return random.randint(0, self.action_dim - 1)
        
        if isinstance(state, Dict):
            market_data = state['market_data']
            position = state['position']
            
            # Handle 2D market data properly
            if len(market_data.shape) == 2:
                market_data_flat = market_data.flatten().astype(np.float32)
            else:
                market_data_flat = market_data.astype(np.float32)
                
            position_flat = position.flatten().astype(np.float32)
            
            combined = np.concatenate([market_data_flat, position_flat])
            
            # Ensure state dimensions match what the network expects
            if len(combined) != self.state_dim:
                if len(combined) < self.state_dim:
                    # Pad if too short
                    combined = np.pad(combined, (0, self.state_dim - len(combined)), 'constant')
                else:
                    # Truncate if too long
                    combined = combined[:self.state_dim]
                    
            state_tensor = torch.FloatTensor(combined)
        else:
            state_array = np.array(state, dtype=np.float32)
            if len(state_array) != self.state_dim:
                if len(state_array) < self.state_dim:
                    state_array = np.pad(state_array, (0, self.state_dim - len(state_array)))
                else:
                    state_array = state_array[:self.state_dim]
            
            state_tensor = torch.FloatTensor(state_array)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def update(self):
        """Update the Q-network using a batch of experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch of transitions
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch for training
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for transition in batch:
            state, action, reward, next_state, done = transition
            
            # Process dictionary observation from environment
            if isinstance(state, Dict):
                market_data = state['market_data']
                position = state['position']
                
                # Handle 2D market data properly
                if len(market_data.shape) == 2:
                    market_data_flat = market_data.flatten().astype(np.float32)
                else:
                    market_data_flat = market_data.astype(np.float32)
                    
                position_flat = position.flatten().astype(np.float32)
                
                state_array = np.concatenate([market_data_flat, position_flat])
                # Ensure the state has the correct dimension
                if len(state_array) != self.state_dim:
                    if len(state_array) < self.state_dim:
                        state_array = np.pad(state_array, (0, self.state_dim - len(state_array)), 'constant')
                    else:
                        state_array = state_array[:self.state_dim]
                
                # Do the same for next_state
                next_market_data = next_state['market_data']
                next_position = next_state['position']
                
                # Handle 2D market data properly
                if len(next_market_data.shape) == 2:
                    next_market_data_flat = next_market_data.flatten().astype(np.float32)
                else:
                    next_market_data_flat = next_market_data.astype(np.float32)
                
                next_position_flat = next_position.flatten().astype(np.float32)
                
                next_state_array = np.concatenate([next_market_data_flat, next_position_flat])
                # Ensure the next_state has the correct dimension
                if len(next_state_array) != self.state_dim:
                    if len(next_state_array) < self.state_dim:
                        next_state_array = np.pad(next_state_array, (0, self.state_dim - len(next_state_array)), 'constant')
                    else:
                        next_state_array = next_state_array[:self.state_dim]
                        
                state = state_array
                next_state = next_state_array
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to tensors, ensuring all data is float type
        states = torch.FloatTensor(np.array(states, dtype=np.float32))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.losses.append(loss.item())
        
        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str = "trading_dqn_model.pt"):
        """Save the model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str = "trading_dqn_model.pt"):
        """Load the model weights."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}") 