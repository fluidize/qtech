import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Dict, List, Tuple
import torch.nn.functional as F

class SimpleActorCriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Separate actor and critic networks for better specialization
        self.actor_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # 3 actions: short, flat, long
        )
        
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        shared_features = self.shared(x)
        action_logits = self.actor_net(shared_features)
        value = self.critic_net(shared_features)
        return action_logits, value

class PPOAgent:
    """
    Simple PPO Agent for discrete action spaces (hold, buy, sell).
    """
    def __init__(self, state_dim, action_dim=3, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, critic_coef=0.5, entropy_coef=0.01, batch_size=64, 
                 optimization_epochs=10, max_grad_norm=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.optimization_epochs = optimization_epochs
        self.max_grad_norm = max_grad_norm

        self.network = SimpleActorCriticNetwork(state_dim)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        # On-policy buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.losses = {'total': [], 'actor': [], 'critic': [], 'entropy': []}

    def state_to_tensor(self, state: Dict) -> torch.Tensor:
        """Convert state dict to tensor with proper normalization."""
        market_data = state['market_data']
        position = state['position']
        
        # Ensure proper shapes and types
        market_data_flat = market_data.flatten().astype(np.float32)
        position_flat = position.flatten().astype(np.float32)
        
        # Combine and handle dimension mismatch
        combined = np.concatenate([market_data_flat, position_flat])
        if len(combined) != self.state_dim:
            if len(combined) < self.state_dim:
                combined = np.pad(combined, (0, self.state_dim - len(combined)), 'constant')
            else:
                combined = combined[:self.state_dim]
        
        return torch.FloatTensor(combined)

    def select_action(self, state: Dict, training: bool = True) -> int:
        """Select action using the current policy."""
        state_tensor = self.state_to_tensor(state)
        
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if training:
                # During training, sample from the probability distribution
                m = torch.distributions.Categorical(probs)
                action = m.sample().item()
                log_prob = m.log_prob(torch.tensor(action)).item()
            else:
                # During evaluation, take the most probable action
                action = torch.argmax(probs).item()
                log_prob = None
        
        if training:
            self.states.append(state_tensor.numpy())
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value.item())
        
        return action

    def store_outcome(self, reward: float, done: bool):
        """Store the outcome of an action."""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_advantages(self, last_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        # Calculate deltas and advantages
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Compute returns and normalize advantages
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return torch.FloatTensor(returns), torch.FloatTensor(advantages)

    def update(self, next_state: Dict = None) -> Dict[str, float]:
        """Update the policy using PPO."""
        if len(self.states) == 0:
            return
            
        # Get last value for advantage computation
        if next_state is not None:
            with torch.no_grad():
                last_value = self.network(self.state_to_tensor(next_state))[1].item()
        else:
            last_value = 0.0
            
        # Compute advantages and returns
        returns, advantages = self.compute_advantages(last_value)
        
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # PPO update loop
        for _ in range(self.optimization_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                logits, values = self.network(batch_states)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                # Calculate losses
                new_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO policy loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus for exploration
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store losses
                self.losses['total'].append(loss.item())
                self.losses['actor'].append(actor_loss.item())
                self.losses['critic'].append(critic_loss.item())
                self.losses['entropy'].append(entropy.item())
        
        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        
        # Return average losses
        return {k: np.mean(v[-self.optimization_epochs:]) for k, v in self.losses.items()}

    def save(self, path: str = "trading_ppo_model.pt"):
        """Save model and optimizer state."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'critic_coef': self.critic_coef,
                'entropy_coef': self.entropy_coef,
            }
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str = "trading_ppo_model.pt"):
        """Load model and optimizer state."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Load and verify hyperparameters
        saved_params = checkpoint['hyperparameters']
        for key, value in saved_params.items():
            if hasattr(self, key) and getattr(self, key) != value:
                print(f"Warning: Loaded {key}={value} differs from current {key}={getattr(self, key)}")
        
        print(f"Model loaded from {path}") 