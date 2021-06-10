import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, namedtuple
import random


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_unit=64, fc2_unit=64,
                 fc3_unit=128):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)

        self.A_fc1 = nn.Linear(fc2_unit, fc3_unit)
        self.V_fc1 = nn.Linear(fc2_unit, fc3_unit)

        self.A_fc2 = nn.Linear(fc3_unit, action_dim)
        self.V_fc2 = nn.Linear(fc3_unit, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        A = F.relu(self.A_fc1(x))
        V = F.relu(self.V_fc1(x))

        A = self.A_fc2(A)
        V = self.V_fc2(V)

        Q = V + (A - A.mean())

        return Q


class ReplayMemory():
    def __init__(self, memory_size, batch_size, device):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.experiences = namedtuple("Experience",
                                      field_names=["state", "action", "reward",
                                                   "next_state", "done"])

        self.device = device

    def push(self, state, action, reward, next_state, done):
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.Tensor([e.state for e in experiences]).reshape(1, -1)
        actions = torch.Tensor([e.action for e in experiences]).reshape(1, -1)
        rewards = torch.Tensor([e.reward for e in experiences]).reshape(1, -1)
        next_states = torch.Tensor(
            [e.next_state for e in experiences]).reshape(1, -1)
        dones = torch.Tensor([e.done for e in experiences]).reshape(1, -1)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
