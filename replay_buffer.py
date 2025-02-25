#Here we make a replay_buffer class, with methods for sampling and pushing

from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity = 10000):
        self.maxcapacity = capacity
        self.buffer = deque(maxlen = capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size = 100):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states), 
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.stack(next_states),
                torch.tensor(dones, dtype=torch.float32))
    def len(self):
        return len(self.buffer)


