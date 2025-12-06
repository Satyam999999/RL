import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 1. The Neural Network Architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# 2. The Agent Class
class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.gamma = config['agent']['gamma']
        self.lr = config['agent']['learning_rate']
        self.batch_size = config['agent']['batch_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available(): self.device = torch.device("mps") # Mac optimization

        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        
        # Copy weights to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is never in training mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon):
        """Epsilon-Greedy Action Selection"""
        if random.random() < epsilon:
            return random.randint(0, 1) # Explore
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item() # Exploit

    def optimize_model(self, memory):
        """Train the neural network on a batch of experiences"""
        if len(memory) < self.batch_size:
            return None # Not enough data yet

        states, actions, rewards, next_states, dones = memory.sample()

        # 1. Compute Q(s, a) - The "Guess"
        # gather() picks the Q-value for the specific action we took
        q_values = self.policy_net(states).gather(1, actions)

        # 2. Compute V(s') = max Q(s', a') - The "Target"
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Bellman Equation: R + gamma * max_Q(next_state) * (1 - done)
            expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 3. Compute Loss
        loss = self.loss_fn(q_values, expected_q_values)

        # 4. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (prevents exploding gradients in RL)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from Policy Net to Target Net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())