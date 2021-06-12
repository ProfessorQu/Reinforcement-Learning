
from gym.wrappers import FrameStack

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
              epsilon=1.0, min_epsilon=0.001, epsilon_decay=0.995,
              lr=0.0005, gamma=0.99, tau=0.001, load=True)

print("Starting playing...")
scores = []

actions = ["Do Nothing", "Right", "Jump Right", "Run Right",
           "Run Jump Right", "Jump", "Left"]

for episode in range(3):
    state = env.reset()
    score, t, done = 0, 0, False
    eps_score = []
    while not done:
        state = (np.array(state, dtype=np.float32) / 255).flatten()

        if t % 2 == 0:
            action = agent.act(state, episode)

        next_state, reward, done, _ = env.step(action)

        env.render()

        state = next_state
        score += reward

        scores.append(score)
        eps_score.append(score)

        print(f"\rEpisode {episode}"
              f"\tAverage Score: {round(np.mean(eps_score), 2)}"
              f"\tAction: {actions[action]}            ", end="")

        t += 1
