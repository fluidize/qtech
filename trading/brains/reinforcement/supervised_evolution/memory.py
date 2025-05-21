import numpy as np
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.9):
        self.buffer = []
        self.capacity = capacity
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def random_sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def priority_sample(self, batch_size):
        weight_function = lambda i: (1/(self.alpha ** i)) #basic exponential function
        weights = np.array([weight_function(i) for i, x in enumerate(self.buffer)])
        weights /= weights.sum()
        return random.choices(self.buffer, weights=weights, k=batch_size)
    
    def reset(self):
        self.buffer = []

if __name__ == "__main__":
    buffer = PrioritizedReplayBuffer(capacity=1000)
    for i in range(1000):
        buffer.add(i, i, i, i, i)
    print(buffer.random_sample(10))
    print(buffer.priority_sample(10))