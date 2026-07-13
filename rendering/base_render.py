from stable_baselines3 import SAC, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym

import pandas as pd
import torch
import numpy as np

import os
import imageio

env = gym.make("Ant-v5", render_mode="rgb_array")

model = SAC.load("checkpointBest.pt", device="cuda")

obs, _ = env.reset()
frames = []

FREQ = 1
for t in range(400):
    if t % FREQ == 0:
      action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, _ = env.step(action)
    frames.append(env.render())

env.close()

imageio.mimsave(
    f"rollout.gif",
    frames,
    fps=30,
)