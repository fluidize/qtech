from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Output in [-1, 1]

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def list_to_param_dict(param_space: Dict[str], values: List[float]) -> Dict[str, float]:
    """
    Convert a list of floats to a parameter dictionary using param_space.
    """
    param_dict = {}
    for i, (k, v) in enumerate(param_space.items()):
        low, high = v
        if isinstance(low, int) and isinstance(high, int):
            param_dict[k] = int(round(values[i] * (high - low) + low))
        else:
            param_dict[k] = values[i] * (high - low) + low
    return param_dict

class ActorCriticAgent:
    def __init__(self, param_space, hidden_dim=128, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.param_space = param_space
        self.action_dim = len(param_space)
        self.state_dim = 1  # Dummy state
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.actor_net = ActorNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.critic_net = CriticNetwork(self.state_dim, hidden_dim)
        self.optimizer = optim.Adam(list(self.actor_net.parameters()) + list(self.critic_net.parameters()), lr=lr)

        self.best_params = None
        self.best_reward = -np.inf

    def select_action(self):
        # Stateless: dummy input
        dummy_state = torch.zeros(1, 1)
        if np.random.rand() < self.epsilon or self.best_params is None:
            # Random action in [0, 1]
            values = np.random.rand(self.action_dim)
        else:
            with torch.no_grad():
                values = self.actor_net(dummy_state).squeeze(0).numpy()
                # Map from [-1, 1] to [0, 1]
                values = (values + 1) / 2
        params = list_to_param_dict(self.param_space, values)
        return params

    def update(self, params, reward):
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = params
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

class OptimizerAgent:
    def __init__(self, param_space, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.param_space = param_space
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.best_params = None
        self.best_reward = -np.inf

    def select_action(self):
        # Epsilon-greedy: random sample or best known
        if np.random.rand() < self.epsilon or self.best_params is None:
            params = {}
            for k, v in self.param_space.items():
                if isinstance(v[0], int) and isinstance(v[1], int):
                    params[k] = np.random.randint(v[0], v[1]+1)
                else:
                    params[k] = np.random.uniform(v[0], v[1])
            return params
        else:
            return self.best_params

    def update(self, params, reward):
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = params
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
