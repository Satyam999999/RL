import random
import numpy as np
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity, batch_size, device):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences"""
        transitions = random.sample(self.memory, self.batch_size)
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043)
        # Converts batch of [(s,a,r,s',d), ...] into [s_batch, a_batch, ...]
        batch = list(zip(*transitions))

        # Convert to PyTorch Tensors
        states      = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(self.device)
        actions     = torch.tensor(np.array(batch[1]), dtype=torch.long).unsqueeze(1).to(self.device)
        rewards     = torch.tensor(np.array(batch[2]), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(self.device)
        dones       = torch.tensor(np.array(batch[4]), dtype=torch.float32).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)