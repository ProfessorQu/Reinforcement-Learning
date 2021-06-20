from gym.wrappers import FrameStack

import matplotlib.pyplot as plt
import numpy as np
import torch

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from agent import Agent

env = gym_super_mario_bros.make('SuperMarioBros-v0')

env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = FrameStack(env, num_stack=16)

print("Setting up agent...")
state_dim = np.prod(env.observation_space.shape)
agent = Agent(state_dim=state_dim, action_dim=env.action_space.n,
              epsilon=1.0, min_epsilon=0.001, epsilon_decay=0.995,
              lr=0.0005, gamma=0.99, tau=0.001, load=True)

print("Starting training...")
printed_episode = 0
scores = []

n_episodes = 30
n_steps = 500

for e in range(n_episodes):
    state = env.reset()
    state = (np.array(state, dtype=np.float32) / 255).flatten()

    score, done = 0, False

    eps_score = []
    t = 0
    for t in range(n_steps):
        if t % 4 == 0:
            action = agent.act(state, e)

        next_state, reward, done, _ = env.step(action)

        next_state = (np.array(next_state, dtype=np.float32) / 255).flatten()

        agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward

        eps_score.append(score)

        print(f"\rEpisode {e}"
              f"\tTime Step: {t}    "
              f"\tAverage Episode Score: {np.mean(eps_score):.2f}   "
              f"\tAverage Total Score: {np.mean(scores):.2f}   ", end="")

        if e // 10 == printed_episode and e >= 10:
            print(f"\rEpisode {e}"
                  f"\tTime Step: {t}    "
                  f"\tAverage Episode Score: {np.mean(eps_score):.2f}    "
                  f"\tAverage Total Score: {np.mean(scores):.2f}   ", end="")
            printed_episode = e // 10 + 1

        if done:
            break

    torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
    scores.append(score)

plt.plot([i for i in range(n_episodes)], scores)
plt.show()

torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
