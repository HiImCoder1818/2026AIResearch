from stable_baselines3 import SAC
import gymnasium as gym
import numpy as np
import imageio

import math
import subprocess
from pathlib import Path
import os

os.environ["MUJOCO_GL"] = "egl"

LATENT_SPACE = 50
ENV_NAME = "Ant-v5"

def make_grid_video(
    inputs,
    output="data/renders/skills.mp4",
    tile_width=320,
    tile_height=240,
    fps=30,
):
    """
    Create a grid video from any number of GIFs/videos.

    Parameters
    ----------
    inputs : list[str]
        List of input gif/video filenames.
    output : str
        Output mp4 filename.
    tile_width : int
        Width of each grid cell.
    tile_height : int
        Height of each grid cell.
    fps : int
        Output FPS.
    """

    n = len(inputs)
    if n == 0:
        raise ValueError("No input files provided.")

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cmd = ["ffmpeg", "-y"]

    # Add inputs
    for file in inputs:
        cmd += ["-i", str(file)]

    filter_parts = []

    # Resize every input
    for i in range(n):
        filter_parts.append(
            f"[{i}:v]fps={fps},scale={tile_width}:{tile_height},setsar=1[v{i}]"
        )

    # Build layout string
    layout = []
    for i in range(n):
        x = (i % cols) * tile_width
        y = (i // cols) * tile_height
        layout.append(f"{x}_{y}")

    layout = "|".join(layout)

    # xstack input labels
    labels = "".join(f"[v{i}]" for i in range(n))

    filter_parts.append(
        f"{labels}xstack=inputs={n}:layout={layout}:fill=black[v]"
    )

    filter_complex = ";".join(filter_parts)

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output,
    ]

    subprocess.run(cmd, check=True)

def render_skills():
    env = gym.make(ENV_NAME)
    model = SAC.load("checkpointBest.pt", device="cuda")

    for skill in range(LATENT_SPACE):
        obs, _ = env.reset()
        frames = []

        for t in range(200):
            skill_one_hot = np.zeros(LATENT_SPACE)
            skill_one_hot[skill] = 1.0

            state = np.concatenate([obs, skill_one_hot])
            action, _ = model.predict(state, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)

        imageio.mimsave(
            f"rollout{skill}.gif",
            frames,
            fps=30
        )

    env.close()

if __name__ == "__main__":
    render_skills()

    gifs = [Path(f"rollout{i}.gif") for i in range(LATENT_SPACE)]

    make_grid_video(
        gifs,
        tile_width=320,
        tile_height=240,
    )

    for gif in gifs:
        gif.unlink()