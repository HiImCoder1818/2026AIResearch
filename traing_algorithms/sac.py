from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.sac.policies import SACPolicy

from gymnasium import spaces

import pandas as pd
import numpy as np
import torch

from dataclasses import dataclass

@dataclass
class Args:
    # Environment
    env_name: str = "HalfCheetah-v5"
    n_envs: int = 8
    state_shape: int = None
    action_shape: int = None

    # Training budget
    total_timesteps: int = 1_000_000

    # Optimizer
    learning_rate: float = 3e-4
    batch_size: int = 1024

    # RL discounting
    gamma: float = 0.99

    # Replay buffer
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000

    # Target critic
    tau: float = 0.005

    # SAC entropy
    ent_coef: str = "auto"
    target_entropy: str = "auto"

    # Training schedule
    train_freq: int = 100
    gradient_steps: int = 200

    # Network
    device="cpu"

    # Logging
    model_path: str = ""
    csv_path: str = ""
    eval_freq: int = 2016
    eval_timesteps: int = 100

class Environment:
  def __init__(self, args, env_kwargs):
    self.env_name = args.env_name
    self.n_envs = args.n_envs

    self.env = make_vec_env(
        self.env_name,
        n_envs=self.n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv
    )

    if args.state_shape is not None:
        self.env.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(args.state_shape,),
            dtype=np.float32
        )

    self.state_shape = self.env.observation_space.shape[0]

  def new_episode(self):
    return self.env.reset()

  def step(self, action):
    next_obs, reward, done, info = self.env.step(action)
    return next_obs, reward, done, info

class PolicyBuilder(SACPolicy):
  def __init__(self, *args, architecture=None, **kwargs):
    self.architecture = architecture
    super().__init__(*args, **kwargs)

  def _build_mlp_extractor(self):
    self.mlp_extractor = self.architecture(
        self.features_dim
    )

class Model:
  def __init__(self, env, policy, args):
    model_arch = policy

    if not isinstance(policy, str):
        model_arch = PolicyBuilder
        policy_kwargs = {
            "architecture": policy
        }

    else:
        model_arch = policy
        policy_kwargs = {}

    self.model = SAC(
        policy=model_arch,
        env=env.env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        tau=args.tau,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
        device=args.device,
        policy_kwargs=policy_kwargs,
        verbose=0
    )

    self.env = env
    self.model._setup_model()
    self.model.set_logger(
        configure(
            folder=None,
            format_strings=[]
        )
    )

    self.args = args
    self.timesteps = 0
    self.max_reward = float("-inf")
    self.total_rewards = np.zeros(self.args.n_envs)
    self.metrics = {
        "reward": [],
        "timestep": [],
    }

  def take_action(self, state):
    obs_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.model.device)
    with torch.no_grad():
        actions = self.model.policy.actor(obs_tensor)

    return actions.cpu().numpy()

  def load(self, checkpoint):
    self.model = SAC.load(checkpoint, env=self.env.env, device=self.args.device)
    self.model.set_logger(
        configure(
            folder=None,
            format_strings=[]
        )
    )

  def load_metrics(self, file):
    df = pd.read_csv(file)
    for row in df:
      self.metrics[row] = list(df[row])

  def collect_rollout(self, obs, next_obs, action, reward, done, info):
    self.model.replay_buffer.add(obs, next_obs, action, reward, done, info)
    self.total_rewards += reward

  def evaluate(self):
    total_rewards = np.zeros(self.args.n_envs)

    obs = self.env.new_episode()
    for _ in range(self.args.eval_timesteps):
        action = self.take_action(obs)
        obs, rewards, _, _ = self.env.step(action)
        total_rewards += rewards

    return total_rewards

  def update(self, done, next_obs, verbose=True, save=True, eval_reward=True):
    updating = False
    new_best = False
    if self.timesteps > self.model.learning_starts and self.timesteps % self.args.train_freq == 0:
      self.model.train(
          batch_size=self.args.batch_size,
          gradient_steps=self.args.gradient_steps
      )
      updating = True

    self.timesteps += self.args.n_envs

    eval_condition = self.timesteps%self.args.eval_freq == 0 and eval_reward
    train_condition = done.any() and not eval_reward

    if eval_condition or train_condition:
        if eval_reward:
          finished_rewards = self.evaluate()
        else:
          finished_rewards = self.total_rewards.copy()

        avg_reward = finished_rewards.mean()

        if self.max_reward < avg_reward:
          new_best = True
          if verbose:
            print("New Best Reward, Saving Model")

          self.model.save(self.args.model_path)
          self.max_reward = avg_reward

        obs = self.env.new_episode()

        self.metrics["reward"].append(avg_reward)
        self.metrics["timestep"].append(self.timesteps)

        if verbose:
          print(f"Timestep: {self.timesteps} | Reward: {avg_reward}")

        if save:
          pd.DataFrame(self.metrics).to_csv(self.args.csv_path, index=False)

        self.total_rewards[done] = 0.0

        return obs, updating, new_best, finished_rewards

    return next_obs, updating, new_best, None