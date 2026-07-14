from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

from dataclasses import dataclass

import numpy as np
import torch
import pandas as pd

@dataclass
class Args:
    # Environment
    env_name: str = "HalfCheetah-v5"
    n_envs: int = 8
    state_shape: int = None
    action_shape: int = None

    # Training budget
    total_timesteps: int = 1_000_000
    n_steps: int = 256

    # PPO Training Updates
    learning_rate: float = 3e-4
    batch_size: int = 1024
    n_epochs: int = 10

    # RL discounting & GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO Specific Coefficients
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5

    # Network
    device: str = "cpu"

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

    if args.action_shape is not None:
      self.env.action_space = spaces.Discrete(args.action_shape)

    self.state_shape = self.env.observation_space.shape[0]

  def new_episode(self):
    return self.env.reset()

  def step(self, action):
    next_obs, reward, done, info = self.env.step(action)
    return next_obs, reward, done, info

class PolicyBuilder(ActorCriticPolicy):
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

    if isinstance(policy, str):
        model_arch = policy
        policy_kwargs = None
    else:
        model_arch = PolicyBuilder
        policy_kwargs = {
            "architecture": policy
        }

    self.model = PPO(
      policy=model_arch,
      env=env.env,
      learning_rate=args.learning_rate,
      n_steps=args.n_steps,
      batch_size=args.batch_size,
      n_epochs=args.n_epochs,
      gamma=args.gamma,
      gae_lambda=args.gae_lambda,
      clip_range=args.clip_range,
      ent_coef=args.ent_coef,
      vf_coef=args.vf_coef,
      device=args.device,
      policy_kwargs=policy_kwargs,
      verbose=0
    )

    self.args = args
    self.env = env
    self.model.set_logger(
        configure(
            folder=None,
            format_strings=[]
        )
    )

    self.metrics = {
      "timestep": [],
      "reward": [],
    }

    self.buffer = self.model.rollout_buffer
    self.episode_starts = np.ones(args.n_envs)
    self.total_rewards = np.zeros(self.args.n_envs)
    self.timesteps = 0
    self.max_reward = float("-inf")

  def load(self, checkpoint):
    self.model = PPO.load(
        checkpoint,
        env=self.env.env,
        device=self.args.device,
    )

    self.model.set_logger(
        configure(
            folder=None,
            format_strings=[]
        )
    )

    self.buffer = self.model.rollout_buffer

  def load_metrics(self, file):
    df = pd.read_csv(file)
    for row in df:
      self.metrics[row] = list(df[row])

  def take_action(self, state):
    obs_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.model.device)
    with torch.no_grad():
        actions, values, log_probs = self.model.policy(obs_tensor)

    return actions.cpu().numpy(), values, log_probs

  def collect_rollout(self, obs, action, reward, value, log_prob):
    self.buffer.add(obs, action, reward, self.episode_starts, value, log_prob)
    self.total_rewards += reward

  def evaluate(self):
    total_rewards = np.zeros(self.args.n_envs)

    obs = self.env.new_episode()
    for _ in range(self.args.eval_timesteps):
        action, _, _ = self.take_action(obs)
        obs, rewards, _, _ = self.env.step(action)
        total_rewards += rewards

    return total_rewards

  def update(self, done, next_obs, verbose=True, eval_reward=True):
    self.timesteps += self.args.n_envs
    
    eval_condition = self.timesteps%self.args.eval_freq == 0 and eval_reward
    train_condition = done.any() and not eval_reward

    if eval_condition or train_condition:
        if eval_reward:
          finished_rewards = self.evaluate()
        else:
          finished_rewards = self.total_rewards.copy()

        avg_reward = finished_rewards.mean()

        self.metrics["timestep"].append(self.timesteps)
        self.metrics["reward"].append(avg_reward)

        if self.max_reward < avg_reward:
            if verbose:
                print(f"New Best Reward: {avg_reward:.2f} Saving Model...")
            self.model.save(self.args.model_path)
            self.max_reward = avg_reward

        if verbose:
            print(f"Timestep: {self.timesteps} | Reward: {avg_reward}")

        self.total_rewards[done] = 0.0

        pd.DataFrame(self.metrics).to_csv(self.args.csv_path, index=False)

    if self.buffer.full:
      with torch.no_grad():
          next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.args.device)
          _, next_values, _ = self.model.policy(next_obs_tensor)

      self.buffer.compute_returns_and_advantage(last_values=next_values, dones=done)

      self.model.train()
      self.buffer.reset()

    self.episode_starts = done

    return next_obs