import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for value-based reinforcement learning.
    Maps states to Q-values for each possible action.
    Uses layer normalization for more stable training.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64]):
        super(DQNNetwork, self).__init__()
        
        self.feature_layers = nn.ModuleList()
        
        # Build neural network layers dynamically based on hidden_dims
        self.feature_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            self.feature_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
        # Final layer outputs Q-values for each action
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Layer normalization helps stabilize training by normalizing activations
        # This is especially useful in RL where input distributions can shift
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in hidden_dims])
        
    def forward(self, state):
        """Forward pass maps state to Q-values for each action"""
        x = state
        
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            x = self.layer_norms[i](x)
            x = F.relu(x)  # ReLU activation for non-linearity
            
        return self.output_layer(x)

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network architecture with shared feature extraction.
    
    This network has two heads:
    1. Actor: Outputs action probability distribution (policy)
    2. Critic: Outputs value estimate of the state
    
    Shared layers extract common features before branching.
    """
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction layers
        self.feature_layers = nn.ModuleList()
        self.feature_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            self.feature_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in hidden_dims])
        
        # Actor head: outputs action probabilities (policy distribution)
        self.actor = nn.Linear(hidden_dims[-1], action_dim)
        
        # Critic head: outputs scalar state value estimate V(s)
        self.critic = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, state):
        """
        Forward pass through the network.
        
        Returns:
            action_probs: Categorical probability distribution over actions (Actor output)
            value: Estimated state value (Critic output)
        """
        # Shared feature extraction
        x = state
        
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            
        # Actor head: apply softmax to get valid probability distribution
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic head: output scalar state value estimate
        value = self.critic(x)
        
        return action_probs, value
    
class PPONetwork(nn.Module):
    """
    Proximal Policy Optimization (PPO) Network with actor-critic architecture.
    
    This network has two main components:
    1. Actor: Outputs policy (action distribution)
       - For discrete actions: outputs categorical distribution
       - For continuous actions: outputs mean and log_std of a Normal distribution
       
    2. Critic: Outputs value estimate of state
    
    Both components share initial feature extraction layers for efficiency.
    """
    
    def __init__(self, input_dim: int, action_dim: int, continuous_actions: bool = False, 
                 hidden_dims: List[int] = [256, 128]):
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor - processes states before actor/critic heads
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU()
        )
        
        # Actor network - determines the policy
        if continuous_actions:
            # For continuous actions, we model a multivariate Gaussian (Normal) distribution
            # We output the mean vector directly
            self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
            
            # For log standard deviation, we use a parameter vector instead of a network
            # This is a common practice that works well and is more stable
            # We initialize it to zeros, meaning std=1 initially (since exp(0)=1)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # For discrete actions, we output logits that are converted to probabilities
            self.actor = nn.Sequential(
                nn.Linear(hidden_dims[-1], action_dim),
            )
        
        # Critic network - estimates state value function V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1)
        )
        
        self.continuous_actions = continuous_actions
        
    def forward(self, state):
        """
        Forward pass through the network.
        
        For continuous actions:
        - Returns mean and log_std of Gaussian distribution, plus value estimate
        - Actions are sampled from N(mean, exp(log_std)^2)
        
        For discrete actions:
        - Returns categorical probabilities over action space, plus value estimate
        - Actions are sampled from Categorical(action_probs)
        
        The critic's value estimate helps compute advantages for policy updates.
        """
        # Extract features from the state
        features = self.shared_layers(state)
        
        if self.continuous_actions:
            # For continuous action spaces
            mean = self.actor_mean(features)
            value = self.critic(features)
            return mean, self.actor_log_std, value
        else:
            # For discrete action spaces
            action_logits = self.actor(features)
            
            # Apply softmax to get valid probability distribution
            # This transforms raw logits into probabilities that sum to 1
            action_probs = F.softmax(action_logits, dim=-1)
            
            value = self.critic(features)
            return action_probs, value 