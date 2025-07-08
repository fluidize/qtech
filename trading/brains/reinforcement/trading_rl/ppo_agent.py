import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(PPOPolicy, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head - outputs probabilities for 3 actions (short=1, flat=2, long=3)
        self.actor = nn.Linear(hidden_dim, 3)
        
        # Critic head - outputs state value
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        shared = self.shared_layers(x)
        action_logits = self.actor(shared)
        value = self.critic(shared)
        return action_logits, value
    
    def act(self, state):
        """Sample action from policy"""
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Convert to trading signals: 0->1 (short), 1->2 (flat), 2->3 (long)
        trading_action = action + 1
        
        return trading_action, log_prob, value
    
    def evaluate(self, state, action):
        """Evaluate action probabilities and values for PPO update"""
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        
        # Convert trading signals back to model actions: 1->0, 2->1, 3->2
        model_action = action - 1
        
        log_prob = dist.log_prob(model_action)
        entropy = dist.entropy()
        
        return log_prob, value, entropy

class PPOAgent:
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, hidden_dim=64):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = PPOPolicy(state_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self):
        """Compute discounted returns"""
        returns = []
        discounted_sum = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            
        return torch.tensor(returns, dtype=torch.float32)
    
    def update(self):
        """PPO update step"""
        if len(self.states) == 0:
            return
            
        # Convert lists to tensors and detach from computation graph
        states = torch.stack(self.states).detach()
        actions = torch.stack(self.actions).detach()
        old_log_probs = torch.stack(self.log_probs).detach()
        old_values = torch.stack(self.values).detach()
        
        # Compute returns and advantages
        returns = self.compute_returns()
        advantages = returns - old_values.squeeze()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy outputs
            log_probs, values, entropy = self.policy.evaluate(states, actions)
            
            # Compute ratio
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            entropy_loss = -entropy.mean()
            
            total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        self.clear_memory()
    
    def clear_memory(self):
        """Clear stored transitions"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def save_model(self, path):
        """Save model"""
        torch.save(self.policy.state_dict(), path)
    
    def load_model(self, path):
        """Load model"""
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval() 