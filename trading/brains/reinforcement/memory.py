import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple

# Define transition type
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """
    A simple replay buffer to store and sample experiences for RL training.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state, action, next_state, reward, done):
        """
        Add a transition to the buffer
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            reward: Reward received
            done: Whether the episode ended
        """
        self.memory.append(Transition(state, action, next_state, reward, done))
        
    def sample(self, batch_size: int) -> List[Transition]:
        """
        Randomly sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        """
        Get current size of buffer
        
        Returns:
            Number of transitions in buffer
        """
        return len(self.memory)
    
    def can_sample(self, batch_size: int) -> bool:
        """
        Check if enough transitions are available to sample
        
        Args:
            batch_size: Desired sample size
            
        Returns:
            True if enough transitions are available, False otherwise
        """
        return len(self) >= batch_size

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for more efficient learning
    by sampling important transitions more frequently.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent (0 = uniform sampling, 1 = greedy prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta each time we sample
        """
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
    def push(self, state, action, next_state, reward, done):
        """
        Add a transition to the buffer with maximum priority
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            reward: Reward received
            done: Whether the episode ended
        """
        # Create transition
        transition = Transition(state, action, next_state, reward, done)
        
        # Add with max priority to ensure it gets sampled at least once
        if self.size < self.capacity:
            self.memory.append(transition)
            self.size += 1
        else:
            self.memory[self.position] = transition
            
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions based on priorities
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (transitions, indices, importance weights)
        """
        if self.size < batch_size:
            return self.memory, np.arange(len(self.memory)), np.ones(len(self.memory))
            
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Compute importance sampling weights
        weights = (self.size * probabilities[indices]) ** -self.beta
        weights /= weights.max()  # Normalize weights
        
        # Increase beta for next time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Gather sampled transitions
        transitions = [self.memory[idx] for idx in indices]
        
        return transitions, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for transitions at given indices
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """
        Get current size of buffer
        
        Returns:
            Number of transitions in buffer
        """
        return self.size
    
    def can_sample(self, batch_size: int) -> bool:
        """
        Check if enough transitions are available to sample
        
        Args:
            batch_size: Desired sample size
            
        Returns:
            True if enough transitions are available, False otherwise
        """
        return len(self) >= batch_size 