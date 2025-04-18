import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict

class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Deep Q-Network for trading decisions
        
        Args:
            input_dim: Dimension of state input
            output_dim: Number of possible actions
            hidden_dims: List of hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        self.feature_layers = nn.ModuleList()
        
        # Input layer
        self.feature_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.feature_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Layer normalization for better training stability
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in hidden_dims])
        
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: Current state tensor
            
        Returns:
            Q-values for each action
        """
        x = state
        
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            
        return self.output_layer(x)

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Actor-Critic Network for trading decisions
        
        Args:
            input_dim: Dimension of state input
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer dimensions
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction
        self.feature_layers = nn.ModuleList()
        self.feature_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            self.feature_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in hidden_dims])
        
        # Actor (policy) head
        self.actor = nn.Linear(hidden_dims[-1], action_dim)
        
        # Critic (value) head
        self.critic = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: Current state tensor
            
        Returns:
            action_probs: Action probability distribution
            value: Value estimate of the state
        """
        x = state
        
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        
        return action_probs, value
    
class PPONetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, continuous_actions: bool = False, 
                 hidden_dims: List[int] = [256, 128]):
        """
        Proximal Policy Optimization Network
        
        Args:
            input_dim: Dimension of state input
            action_dim: Number of possible actions
            continuous_actions: Whether actions are continuous (True) or discrete (False)
            hidden_dims: List of hidden layer dimensions
        """
        super(PPONetwork, self).__init__()
        
        # Feature extraction shared between actor and critic
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU()
        )
        
        # Actor (policy) network
        if continuous_actions:
            self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Sequential(
                nn.Linear(hidden_dims[-1], action_dim),
            )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1)
        )
        
        self.continuous_actions = continuous_actions
        
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: Current state tensor
            
        Returns:
            If continuous actions:
                mean: Mean of action distribution
                log_std: Log standard deviation of action distribution
                value: Value estimate of the state
            If discrete actions:
                action_probs: Action probability distribution
                value: Value estimate of the state
        """
        features = self.shared_layers(state)
        
        if self.continuous_actions:
            mean = self.actor_mean(features)
            value = self.critic(features)
            return mean, self.actor_log_std, value
        else:
            action_logits = self.actor(features)
            action_probs = F.softmax(action_logits, dim=-1)
            value = self.critic(features)
            return action_probs, value 