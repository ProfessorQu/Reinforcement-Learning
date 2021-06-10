from gym.wrappers import FrameStack

import torch

from collections import deque
import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from agent import Agent

env = gym_super_mario_bros.make('SuperMarioBros-v0')

env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = FrameStack(env, num_stack=4)

print("Setting up agent...")
state_dim = np.prod(env.observation_space.shape)
agent = Agent(state_dim=state_dim, action_dim=env.action_space.n,
              epsilon=1.0, min_epsilon=0.001, epsilon_decay=0.995)

print("Starting training...")
printed_episode = 0
scores = []
scores_window = deque(maxlen=100)
for episode in range(20):
    state = env.reset()
    score = 0
    for t in range(50):
        state = (np.array(state, dtype=np.float32) / 255).flatten()

        action = agent.act(state, episode)
        next_state, reward, done, _ = env.step(action)

        next_state = (np.array(next_state, dtype=np.float32) / 255).flatten()

        agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward
        if done:
            break

        scores_window.append(score)
        scores.append(score)
        print(f"\rEpisode {episode}"
              f"\tAverage Score: {np.mean(scores_window)}", end="")

        if episode // 10 == printed_episode:
            print(f"\rEpisode {episode}"
                  f"\tAverage Score: {np.mean(scores_window)}")
            printed_episode = episode // 10 + 1

torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
