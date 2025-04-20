import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple

# Define transition type for storing experience tuples
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """
    Standard experience replay buffer with uniform sampling.
    
    Stores agent's experience as (state, action, next_state, reward, done) tuples
    and allows random sampling from these experiences for training.
    This breaks correlation between consecutive experiences and improves stability.
    """
    
    def __init__(self, capacity: int):
        """Initialize buffer with fixed capacity (FIFO when full)"""
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state, action, next_state, reward, done):
        """Store a new transition in the buffer"""
        self.memory.append(Transition(state, action, next_state, reward, done))
        
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random batch of transitions for training"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.memory)
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer contains enough samples for a batch"""
        return len(self) >= batch_size

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.
    
    Enhances learning efficiency by sampling transitions with higher TD errors
    more frequently, focusing training on the most informative experiences.
    
    Key concepts:
    - Prioritized sampling: Samples transitions with probability proportional to priority^α
    - Importance sampling: Corrects bias introduced by non-uniform sampling using weights
    - Annealing: Gradually increases β from initial value to 1.0 during training
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum transitions to store
            alpha: Controls how much prioritization is used (0=uniform, 1=full prioritization)
            beta: Controls importance sampling weight correction (0=no correction, 1=full correction)
            beta_increment: Amount to increase beta each sampling towards 1.0
        """
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.capacity = capacity
        self.position = 0  # Current position for circular buffer insert
        self.size = 0      # Current number of stored transitions
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0  # Initial max priority for new transitions
        
    def push(self, state, action, next_state, reward, done):
        """
        Add a transition to the buffer with maximum priority to ensure it gets sampled.
        
        New transitions get maximum priority so they're guaranteed to be sampled at
        least once before their priority is updated based on TD error.
        """
        transition = Transition(state, action, next_state, reward, done)
        
        # If buffer not yet full, append; otherwise, overwrite at current position
        if self.size < self.capacity:
            self.memory.append(transition)
            self.size += 1
        else:
            self.memory[self.position] = transition
            
        # Assign max priority to new transition
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity  # Circular buffer
        
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions based on priorities using the formula:
        P(i) = (priority_i ^ alpha) / sum(priority_j ^ alpha)
        
        Also computes importance sampling weights using the formula:
        w_i = (N * P(i))^(-beta)
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (transitions, indices, importance sampling weights)
        """
        if self.size < batch_size:
            return self.memory, np.arange(len(self.memory)), np.ones(len(self.memory))
            
        # Step 1: Compute sampling probabilities based on priorities
        priorities = self.priorities[:self.size]
        # Apply alpha exponent to convert priorities to probabilities
        # Higher alpha -> more emphasis on high priority samples
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()  # Normalize to valid probability distribution
        
        # Step 2: Sample indices based on these probabilities
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Step 3: Compute importance sampling weights to correct for sampling bias
        # The bias comes from non-uniform sampling of experiences
        # We need to correct Q-learning loss since some transitions are sampled more often
        weights = (self.size * probabilities[indices]) ** -self.beta  # Formula: (N*P(i))^(-β)
        weights /= weights.max()  # Normalize weights for stability
        
        # Step 4: Increase beta (annealing) to decrease the amount of correction over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Step 5: Retrieve the sampled transitions
        transitions = [self.memory[idx] for idx in indices]
        
        return transitions, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for transitions based on TD errors.
        
        Higher TD errors indicate transitions from which the agent can learn more,
        so these transitions should be sampled more frequently.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically absolute TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)  # Track max priority for new samples
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return self.size
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer contains enough samples for a batch"""
        return len(self) >= batch_size 