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
        """
        Initialize the DQN agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of epsilon decay
            target_update_freq: Frequency of target network updates
            batch_size: Batch size for training
            buffer_capacity: Maximum capacity of replay buffer
            prioritized_replay: Whether to use prioritized experience replay
            device: Device to use for tensor operations ('cpu', 'cuda', or 'auto')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
            
        # Initialize networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_capacity)
            self.prioritized_replay = True
        else:
            self.memory = ReplayBuffer(buffer_capacity)
            self.prioritized_replay = False
            
        # Set hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
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
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state
            evaluate: Whether to use greedy policy (True) or epsilon-greedy (False)
            
        Returns:
            Selected action
        """
        if evaluate or random.random() > self.epsilon:
            # Greedy action selection
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            # Random action selection
            return random.randrange(self.action_dim)
        
    def update_epsilon(self):
        """
        Update exploration rate using epsilon decay
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def store_transition(self, state, action, next_state, reward, done):
        """
        Store transition in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether episode is done
        """
        self.memory.push(state, action, next_state, reward, done)
        
    def update(self) -> Optional[float]:
        """
        Update policy network using a batch of transitions
        
        Returns:
            Loss value if update performed, None otherwise
        """
        # Check if enough samples are available
        if not self.memory.can_sample(self.batch_size):
            return None
        
        # Sample from replay buffer
        if self.prioritized_replay:
            transitions, indices, weights = self.memory.sample(self.batch_size)
            weights_tensor = torch.FloatTensor(weights).to(self.device)
        else:
            transitions = self.memory.sample(self.batch_size)
            weights_tensor = torch.ones(self.batch_size).to(self.device)
            
        # Convert batch of transitions to transition of batches
        batch = Transition(*zip(*transitions))
        
        # Create tensors for batch elements
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).view(-1, 1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        # Compute Q(s_t, a) for actions taken
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute max Q(s_{t+1}, a) for all next states
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            
        # Compute expected Q values
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        if self.prioritized_replay:
            # Compute element-wise loss for priority update
            element_wise_loss = F.smooth_l1_loss(q_values.squeeze(), 
                                              expected_q_values.unsqueeze(1).squeeze(),
                                              reduction='none')
            
            # Apply importance sampling weights
            loss = torch.mean(element_wise_loss * weights_tensor)
            
            # Update priorities in buffer
            priorities = element_wise_loss.detach().cpu().numpy() + 1e-6  # Add small constant to avoid zero priorities
            self.memory.update_priorities(indices, priorities)
        else:
            loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values.unsqueeze(1).squeeze())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        # Record loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss_value
    
    def save(self, path: str):
        """
        Save agent model to file
        
        Args:
            path: Path to save model
        """
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
        """
        Load agent model from file
        
        Args:
            path: Path to load model from
        """
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
        
        # Load training state
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
        
class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for trading decisions.
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
        """
        Initialize the PPO agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter for advantage estimation
            clip_epsilon: PPO clipping parameter
            critic_coef: Coefficient for critic loss
            entropy_coef: Coefficient for entropy bonus
            batch_size: Batch size for training
            optimization_epochs: Number of optimization epochs per update
            continuous_actions: Whether actions are continuous
            device: Device to use for tensor operations
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Initialize network
        self.network = PPONetwork(state_dim, action_dim, continuous_actions, hidden_dims).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Set hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.optimization_epochs = optimization_epochs
        self.continuous_actions = continuous_actions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Training metrics
        self.steps = 0
        self.losses = {"total": [], "actor": [], "critic": [], "entropy": []}
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action based on current policy
        
        Args:
            state: Current state
            evaluate: Whether to select deterministically for evaluation
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if self.continuous_actions:
                mean, log_std, value = self.network(state_tensor)
                
                if evaluate:
                    # Deterministic action for evaluation
                    action = mean
                    log_prob = None
                else:
                    # Sample from normal distribution
                    std = torch.exp(log_std)
                    normal = torch.distributions.Normal(mean, std)
                    action = normal.sample()
                    log_prob = normal.log_prob(action)
                    
                action = action.cpu().numpy().flatten()
            else:
                action_probs, value = self.network(state_tensor)
                
                if evaluate:
                    # Deterministic action for evaluation
                    action = torch.argmax(action_probs, dim=1).item()
                    log_prob = None
                else:
                    # Sample from categorical distribution
                    m = torch.distributions.Categorical(action_probs)
                    action = m.sample().item()
                    log_prob = m.log_prob(torch.tensor([action]).to(self.device)).item()
            
            # Store experience if not evaluating
            if not evaluate:
                self.states.append(state)
                self.actions.append(action)
                self.log_probs.append(log_prob)
                self.values.append(value.item())
                
            return action
        
    def store_outcome(self, reward: float, done: bool):
        """
        Store reward and done signal for the last action
        
        Args:
            reward: Reward received
            done: Whether the episode is done
        """
        self.rewards.append(reward)
        self.dones.append(done)
        
    def compute_advantages(self, last_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using GAE
        
        Args:
            last_value: Value estimate for the final state
            
        Returns:
            Tuple of (returns, advantages)
        """
        # Initialize return and advantage arrays
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        
        values = self.values + [last_value]
        
        gae = 0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        return torch.FloatTensor(returns).to(self.device), torch.FloatTensor(advantages).to(self.device)
        
    def update(self, next_state: np.ndarray = None) -> Dict[str, float]:
        """
        Update policy using the collected experiences
        
        Args:
            next_state: Next state value to use for the last experience
            
        Returns:
            Dictionary of loss metrics
        """
        # Ensure we have enough experience
        if len(self.states) == 0:
            return {k: 0.0 for k in self.losses.keys()}
        
        # Get value of final state
        if next_state is not None:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                if self.continuous_actions:
                    _, _, last_value = self.network(next_state_tensor)
                else:
                    _, last_value = self.network(next_state_tensor)
                last_value = last_value.item()
        else:
            last_value = 0.0
            
        # Compute returns and advantages
        returns, advantages = self.compute_advantages(last_value)
        
        # Convert list to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        
        if self.continuous_actions:
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
            old_log_probs = torch.FloatTensor(np.array(self.log_probs)).sum(dim=1).to(self.device)
        else:
            actions = torch.LongTensor(np.array(self.actions)).to(self.device)
            old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        # Optimization loop
        total_losses = []
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        
        for _ in range(self.optimization_epochs):
            # Create data loader for mini-batches
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(self.states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
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
                
                # Compute ratios and surrogate objectives
                ratios = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                
                # Compute losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                entropy_loss = -entropy * self.entropy_coef
                
                # Total loss
                loss = actor_loss + self.critic_coef * critic_loss + entropy_loss
                
                # Optimize
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
            
        # Clear experience buffer
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
        """
        Save agent model to file
        
        Args:
            path: Path to save model
        """
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
        """
        Load agent model from file
        
        Args:
            path: Path to load model from
        """
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
            
        # Load state dicts
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load training state
        self.steps = checkpoint['steps'] 