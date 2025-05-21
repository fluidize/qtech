import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models import ActorNetwork, CriticNetwork

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.actor_net = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic_net = CriticNetwork(state_dim, hidden_dim)
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            action = self.actor_net(state)
            return action.argmax().item()
