from stable_baselines3 import SAC, PPO
import gymnasium as gym

import imageio
import os
import time

os.environ["MUJOCO_GL"] = "egl"

env = gym.make("Ant-v5", render_mode="rgb_array")

model = SAC.load("data/models/checkpointTest.pt", device="cuda")

obs, _ = env.reset()
frames = []

start = time.time()

FREQ = 1
for t in range(200):
    if t % FREQ == 0:
      action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, _ = env.step(action)
    frames.append(env.render())

print(time.time()-start)

env.close()

imageio.mimsave(
    f"data/renders/rollout.gif",
    frames,
    fps=30,
)