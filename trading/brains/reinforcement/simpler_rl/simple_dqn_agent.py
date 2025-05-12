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
    Simple perceptron architecture for the trading agent.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 3):
        super().__init__()
        
        # Simple feedforward network
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)

class ReplayBuffer:
    """
    Experience replay buffer to store and sample transitions.
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """Sample a batch of transitions randomly with priority to recent experiences."""
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def sample_all(self) -> List:
        """Sample all transitions."""
        return list(self.buffer)
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

class DQNAgent:
    """
    Simple DQN Agent that learns to trade through Q-learning.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int = 3,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 50000,
                 batch_size: int = 64,
                 target_update_freq: int = 5):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Initialize the Q-network and target network
        self.q_network = DQNNetwork(input_dim=state_dim, hidden_dim=64, output_dim=action_dim)
        self.target_network = DQNNetwork(input_dim=state_dim, hidden_dim=64, output_dim=action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with simpler settings
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.losses = []

    def state_to_tensor(self, state: Dict) -> torch.Tensor:
        """Convert state dictionary to tensor."""
        market_data = state['market_data']
        position = state['position']
        
        if isinstance(market_data, np.ndarray):
            market_data_flat = market_data.flatten().astype(np.float32)
        else:
            market_data_flat = market_data.flatten().astype(np.float32)
            
        if isinstance(position, np.ndarray):
            position_flat = position.flatten().astype(np.float32)
        else:
            position_flat = position.flatten().astype(np.float32)
        
        combined = np.concatenate([market_data_flat, position_flat])
        
        # Ensure the state has the correct dimension
        if len(combined) != self.state_dim:
            if len(combined) < self.state_dim:
                combined = np.pad(combined, (0, self.state_dim - len(combined)), 'constant')
            else:
                combined = combined[:self.state_dim]
                
        return torch.FloatTensor(combined)
    
    def select_action(self, state: Dict, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy strategy.
        
        During training, we explore with probability epsilon.
        During evaluation, we always exploit.
        """
        if training and random.random() < self.epsilon: #randomly select an action / explore
            return random.randint(0, self.action_dim - 1)
        
        if isinstance(state, Dict):
            state_tensor = self.state_to_tensor(state)
        else:
            raise ValueError("State must be a dictionary.")

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def update(self):
        """Update the Q-network using a batch of experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for transition in batch:
            state, action, reward, next_state, done = transition
            
            state = self.state_to_tensor(state)
            next_state = self.state_to_tensor(next_state)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        states = torch.FloatTensor(np.array(states, dtype=np.float32))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        current_q_values = self.q_network(states).gather(1, actions) #q-values for chosen actions
        
        # Get next Q-values (standard Q-learning)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate loss
        loss = F.mse_loss(current_q_values, target_q_values)
        self.losses.append(loss.item())
        
        # Update network
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