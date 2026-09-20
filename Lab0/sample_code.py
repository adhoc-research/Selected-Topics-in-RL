import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Enduro-v5", render_mode="human")

observation, info = env.reset()

total_reward = 0
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    if terminated or truncated:
        observation = env.reset()

print("Total reward: {}".format(total_reward))
env.close()