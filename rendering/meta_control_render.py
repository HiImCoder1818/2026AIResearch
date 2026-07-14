from stable_baselines3 import SAC, PPO
import gymnasium as gym
import numpy as np

import imageio
import os
import time

os.environ["MUJOCO_GL"] = "egl"

LATENT_SPACE = 50
HORIZON = 100

env = gym.make("Ant-v5")

meta_controller = PPO.load("checkpointBest.pt", device="cuda")
skills_model = SAC.load("checkpointFinal.pt", device="cuda")

obs, _ = env.reset()
frames = []

for t in range(1000):
    if t % HORIZON == 0:
      skill_idx, _ = meta_controller.predict(obs, deterministic=True)
      print(skill_idx)
      skill_one_hot = np.zeros(LATENT_SPACE)
      skill_one_hot[skill_idx] = 1.0

    state_with_skill = np.concatenate([obs, skill_one_hot])
    action, _ = skills_model.predict(state_with_skill, deterministic=True)

    frames.append(env.render())

    obs, reward, terminated, truncated, _ = env.step(action)

imageio.mimsave(
    f"data/renders/rolloutMeta.gif",
    frames,
    fps=30,
)