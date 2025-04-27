import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List, Dict, Optional
import os
import time
from collections import namedtuple

from memory import ReplayBuffer, PrioritizedReplayBuffer, Transition
from models import DQNNetwork, ActorCriticNetwork, PPONetwork

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for trading decisions.
    Implements key DQN features: target network, experience replay, epsilon-greedy exploration.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = [128, 64],
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.999,
                 target_update_freq: int = 10,
                 batch_size: int = 64,
                 buffer_capacity: int = 100000,
                 prioritized_replay: bool = False,
                 device: str = 'auto'):
        """Initialize the DQN agent"""
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu' if device == 'auto' else device)
        print(f"Using device: {self.device}")
            
        # Initialize policy network (active) and target network (stable reference)
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is used for inference only
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Select the appropriate replay buffer based on whether we're using prioritized replay
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_capacity)
            self.prioritized_replay = True
        else:
            self.memory = ReplayBuffer(buffer_capacity)
            self.prioritized_replay = False
            
        # Store hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon_start  # Current exploration rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Training metrics
        self.losses = []
        self.rewards = []
        self.steps = 0
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.
        During evaluation, always selects the best action.
        During training, randomly explores with probability epsilon.
        """
        if evaluate or random.random() > self.epsilon:
            # Greedy action selection (exploit)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            # Random action selection (explore)
            return random.randrange(self.action_dim)
        
    def update_epsilon(self):
        """Decay exploration rate over time to reduce exploration"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def store_transition(self, state, action, next_state, reward, done):
        """Store experience tuple in replay buffer for later training"""
        self.memory.push(state, action, next_state, reward, done)
        
    def update(self) -> Optional[float]:
        """
        Update policy network using a batch of experiences.
        
        Implements the core DQN learning algorithm:
        1. Sample transitions from replay buffer
        2. Compute Q-values for current states/actions
        3. Compute target Q-values using Bellman equation
        4. Update policy network to minimize the difference
        
        Returns:
            Loss value if update performed, None if insufficient samples
        """
        if not self.memory.can_sample(self.batch_size):
            return None
        
        # Sample from replay buffer
        if self.prioritized_replay:
            transitions, indices, weights = self.memory.sample(self.batch_size)
            weights_tensor = torch.FloatTensor(weights).to(self.device)
        else:
            transitions = self.memory.sample(self.batch_size)
            weights_tensor = torch.ones(self.batch_size).to(self.device)
            
        # Prepare batch data
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).view(-1, 1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        # Current Q-values: Q(s,a) for the actions that were actually taken
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute target Q-values using Bellman equation: r + γ * max_a' Q(s',a')
        with torch.no_grad():  # No need to compute gradients for target computation
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            
        # Target Q = reward + (discount * next Q-value), unless the episode is done
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss - either weighted MSE (for prioritized replay) or regular MSE
        if self.prioritized_replay:
            # For prioritized replay, we need per-element losses to update priorities
            element_wise_loss = F.smooth_l1_loss(q_values.squeeze(), 
                                              expected_q_values.unsqueeze(1).squeeze(),
                                              reduction='none')
            
            # Apply importance sampling weights to correct for sampling bias
            loss = torch.mean(element_wise_loss * weights_tensor)
            
            # Update priorities based on TD error
            priorities = element_wise_loss.detach().cpu().numpy() + 1e-6  # Avoid zero priorities
            self.memory.update_priorities(indices, priorities)
        else:
            loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values.unsqueeze(1).squeeze())
        
        # Update policy network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients problem
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        # Periodically update target network with policy network weights
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss_value
    
    def save(self, path: str):
        """Save agent model to file"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': [layer.out_features for layer in self.policy_net.feature_layers],
        }, path)
        
    def load(self, path: str):
        """Load agent model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model if dimensions don't match
        if checkpoint['state_dim'] != self.state_dim or checkpoint['action_dim'] != self.action_dim:
            self.state_dim = checkpoint['state_dim']
            self.action_dim = checkpoint['action_dim']
            hidden_dims = checkpoint['hidden_dims']
            
            self.policy_net = DQNNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.target_net = DQNNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.optimizer = optim.Adam(self.policy_net.parameters())
            
        # Load state dicts
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
        
class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for trading decisions.
    
    PPO is an on-policy algorithm that optimizes a "surrogate" objective function
    with constraints to prevent too large policy updates. It achieves this by clipping 
    the probability ratio between old and new policies.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 128],
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 critic_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 batch_size: int = 64,
                 optimization_epochs: int = 10,
                 continuous_actions: bool = False,
                 device: str = 'auto'):
        """Initialize the PPO agent"""
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu' if device == 'auto' else device)
        print(f"Using device: {self.device}")
        
        # Initialize the actor-critic network
        self.network = PPONetwork(state_dim, action_dim, continuous_actions, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # PPO hyperparameters
        self.gamma = gamma  # Discount factor for future rewards
        self.gae_lambda = gae_lambda  # Controls the bias-variance tradeoff in advantage estimation
        self.clip_epsilon = clip_epsilon  # Clip parameter for PPO's trust region constraint
        self.critic_coef = critic_coef  # Weight for value function loss
        self.entropy_coef = entropy_coef  # Weight for entropy loss (encourages exploration)
        self.batch_size = batch_size  # Mini-batch size for optimization
        self.optimization_epochs = optimization_epochs  # Number of passes through the experience buffer
        self.continuous_actions = continuous_actions  # Whether the action space is continuous
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Experience buffer for on-policy learning
        # Unlike DQN, PPO collects and uses experience immediately, then discards it
        self.states = []  # States encountered
        self.actions = []  # Actions taken
        self.log_probs = []  # Log probabilities of actions under the policy
        self.rewards = []  # Rewards received
        self.values = []  # Value estimates from critic
        self.dones = []  # Terminal state flags
        
        # Training metrics
        self.steps = 0
        self.losses = {"total": [], "actor": [], "critic": [], "entropy": []}
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action based on current policy.
        
        In PPO:
        - During training, actions are sampled stochastically from the policy distribution
        - During evaluation, actions are selected deterministically (highest probability)
        
        For continuous actions, this uses a Normal distribution.
        For discrete actions, this uses a Categorical distribution.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if self.continuous_actions:
                # For continuous actions, the network outputs mean and log_std of a Normal distribution
                mean, log_std, value = self.network(state_tensor)
                
                if evaluate:
                    # During evaluation, just use the mean (most likely action)
                    action = mean
                    log_prob = None
                else:
                    # During training, sample from the distribution to explore
                    std = torch.exp(log_std)  # Convert log_std to std
                    normal = torch.distributions.Normal(mean, std)
                    action = normal.sample()  # Sample an action
                    log_prob = normal.log_prob(action)  # Compute log probability of the sampled action
                    
                action = action.cpu().numpy().flatten()
            else:
                # For discrete actions, the network outputs action probabilities
                action_probs, value = self.network(state_tensor)
                
                if evaluate:
                    # During evaluation, choose the action with highest probability
                    action = torch.argmax(action_probs, dim=1).item()
                    log_prob = None
                else:
                    # During training, sample from the categorical distribution
                    m = torch.distributions.Categorical(action_probs)
                    action = m.sample().item()  # Sample an action
                    log_prob = m.log_prob(torch.tensor([action]).to(self.device)).item()  # Compute log probability
            
            # Store experience for training (if not in evaluation mode)
            if not evaluate:
                self.states.append(state)
                self.actions.append(action)
                self.log_probs.append(log_prob)
                self.values.append(value.item())
                
            return action
        
    def store_outcome(self, reward: float, done: bool):
        """
        Store reward and done signal for the last action taken.
        Called after each environment step to track outcomes.
        """
        self.rewards.append(reward)
        self.dones.append(done)
        
    def compute_advantages(self, last_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).
        
        GAE provides a balance between bias and variance in advantage estimation:
        - λ=0 corresponds to a TD(0) estimate (low variance, high bias)
        - λ=1 corresponds to Monte Carlo estimate (high variance, low bias)
        
        The advantage formula is:
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error
        
        Args:
            last_value: Value estimate for the final state
            
        Returns:
            Tuple of (returns, advantages)
        """
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        
        values = self.values + [last_value]
        
        # Compute advantages using GAE (Generalized Advantage Estimation)
        gae = 0
        for t in reversed(range(len(self.rewards))):
            # TD error: reward + discounted next value - current value
            delta = self.rewards[t] + self.gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            
            # Recursive formula for GAE advantage: δ_t + γλA_{t+1}, with initialization A_T = 0
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae
            
            # Returns are advantages + value estimates (R_t = A_t + V(s_t))
            returns[t] = advantages[t] + values[t]
            
        return torch.FloatTensor(returns).to(self.device), torch.FloatTensor(advantages).to(self.device)
        
    def update(self, next_state: np.ndarray = None) -> Dict[str, float]:
        """
        Update policy using the collected experiences.
        
        This implements the PPO algorithm:
        1. Compute advantages using GAE
        2. Normalize advantages (to stabilize training)
        3. Perform multiple optimization epochs over mini-batches
        4. For each batch:
           a. Compute probability ratio between new and old policies
           b. Compute clipped surrogate objective
           c. Compute value loss and entropy bonus
           d. Update the policy and value networks
        
        Args:
            next_state: Next state value to use for the last experience
            
        Returns:
            Dictionary of loss metrics
        """
        # Skip update if no experiences collected
        if len(self.states) == 0:
            return {k: 0.0 for k in self.losses.keys()}
        
        # Get value of final state to complete the trajectory
        if next_state is not None:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                if self.continuous_actions:
                    _, _, last_value = self.network(next_state_tensor)
                else:
                    _, last_value = self.network(next_state_tensor)
                last_value = last_value.item()
        else:
            last_value = 0.0  # Assume final state has zero value if not provided
            
        # Compute advantages and returns
        returns, advantages = self.compute_advantages(last_value)
        
        # Convert experience lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        
        if self.continuous_actions:
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
            # For continuous actions, sum log probs across action dimensions
            old_log_probs = torch.FloatTensor(np.array(self.log_probs)).sum(dim=1).to(self.device)
        else:
            actions = torch.LongTensor(np.array(self.actions)).to(self.device)
            old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        
        # Normalize advantages (reduces variance and stabilizes learning)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        # Loss tracking
        total_losses = []
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        
        # Perform multiple optimization epochs
        for _ in range(self.optimization_epochs):
            # Randomly shuffle indices for mini-batch sampling
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            # Process mini-batches
            for start_idx in range(0, len(self.states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.states))
                batch_indices = indices[start_idx:end_idx]
                
                # Extract mini-batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass through the network
                if self.continuous_actions:
                    mean, log_std, values = self.network(batch_states)
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    log_probs = dist.log_prob(batch_actions).sum(dim=1)
                    entropy = dist.entropy().mean()
                else:
                    action_probs, values = self.network(batch_states)
                    dist = torch.distributions.Categorical(action_probs)
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                
                # Compute policy ratio: π_new(a|s) / π_old(a|s)
                # Using log probabilities: exp(log(π_new) - log(π_old))
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate objectives
                # surr1: standard policy gradient objective multiplied by advantages
                surr1 = ratios * batch_advantages
                # surr2: clipped ratio times advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                
                # Compute losses
                # Actor loss: negative of minimum surrogate objective (we minimize the negative)
                actor_loss = -torch.min(surr1, surr2).mean()
                # Critic loss: MSE between predicted values and target returns
                critic_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                # Entropy loss: Encourages exploration by maximizing entropy
                entropy_loss = -entropy * self.entropy_coef
                
                # Total loss: combination of actor, critic, and entropy losses
                loss = actor_loss + self.critic_coef * critic_loss + entropy_loss
                
                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradient norm to stabilize training
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                # Record losses
                total_losses.append(loss.item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Update training metrics
        if total_losses:
            self.losses["total"].append(np.mean(total_losses))
            self.losses["actor"].append(np.mean(actor_losses))
            self.losses["critic"].append(np.mean(critic_losses))
            self.losses["entropy"].append(np.mean(entropy_losses))
            
        # Clear experience buffer after update (PPO is on-policy)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Return mean loss values
        return {
            "total": np.mean(total_losses) if total_losses else 0.0,
            "actor": np.mean(actor_losses) if actor_losses else 0.0,
            "critic": np.mean(critic_losses) if critic_losses else 0.0,
            "entropy": np.mean(entropy_losses) if entropy_losses else 0.0
        }
    
    def save(self, path: str):
        """Save agent model to file"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'continuous_actions': self.continuous_actions,
            'hidden_dims': [256, 128],  # Hardcoded for simplicity
            'steps': self.steps,
        }, path)
        
    def load(self, path: str):
        """Load agent model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model if necessary
        if (checkpoint['state_dim'] != self.state_dim or 
            checkpoint['action_dim'] != self.action_dim or
            checkpoint['continuous_actions'] != self.continuous_actions):
            
            self.state_dim = checkpoint['state_dim']
            self.action_dim = checkpoint['action_dim']
            self.continuous_actions = checkpoint['continuous_actions']
            hidden_dims = checkpoint['hidden_dims']
            
            self.network = PPONetwork(
                self.state_dim, 
                self.action_dim,
                self.continuous_actions,
                hidden_dims
            ).to(self.device)
            
            self.optimizer = optim.Adam(self.network.parameters())
            
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.steps = checkpoint['steps'] 