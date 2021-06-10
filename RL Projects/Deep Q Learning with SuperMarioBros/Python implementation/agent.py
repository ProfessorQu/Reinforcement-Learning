import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from dqn_memory import DQN, ReplayMemory


class Agent():
    def __init__(self, state_dim, action_dim, lr,
                 epsilon, min_epsilon, epsilon_decay,
                 gamma, tau, update_target=4,
                 memory_size=int(1e5), batch_size=64):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.qnetwork_local = DQN(state_dim, action_dim,
                                  fc1_unit=128, fc2_unit=128,
                                  fc3_unit=256).to(self.device)
        self.qnetwork_target = DQN(state_dim, action_dim,
                                   fc1_unit=128, fc2_unit=128,
                                   fc3_unit=256).to(self.device)

        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayMemory(memory_size, batch_size)
        self.batch_size = batch_size

        self.t_step = 0
        self.update_target = update_target
        self.episode = 0

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.gamma = gamma
        self.tau = tau

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_target

        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()

                self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        criterion = nn.MSELoss()

        self.qnetwork_local.train()
        self.qnetwork_target.eval()

        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.qnetwork_target(
                next_states).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (self.gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(),
                                             self.qnetwork_local.parameters()):
            target_param.data.copy(self.tau * local_param.data +
                                   (1 - self.tau) * target_param.data)

    def act(self, state, episode):
        state = torch.tensor(state).to(self.device)

        self.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        self.qnetwork_local.train()

        if self.episode < episode:
            self.epsilon *= self.epsilon_decay
            self.episode = episode

        self.epsilon = max(self.epsilon, self.min_epsilon)

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))
