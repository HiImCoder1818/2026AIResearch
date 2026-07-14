from traing_algorithms.sac import Args, Model, Environment

from torch import nn
import torch

import pandas as pd
import numpy as np

from collections import deque
import random

args = Args()

LATENT_SPACE = 50

class Discriminator(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim=256):
    super().__init__()

    self.net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )

  def forward(self, state):
    return self.net(state)

  def sample_z(self):
    z = np.random.randint(0, LATENT_SPACE, (args.n_envs,))
    z_one_hot = np.eye(LATENT_SPACE)[z]
    return z_one_hot, z

args.model_path = "/content/drive/MyDrive/2026 AI Research/Code & Data/"
args.csv_path = "/content/drive/MyDrive/2026 AI Research/Code & Data/"
args.device = "cpu"
args.env_name = "Ant-v5"
args.state_shape = 105 + LATENT_SPACE

args.total_timesteps = 800_000
# args.learning_starts = 1
# args.train_freq = 1000
# args.gradient_steps = 1

save_times = 8
interval = args.total_timesteps // save_times
interval = (interval // args.n_envs) * args.n_envs

env = Environment(args, False)
model = Model(env, "MlpPolicy", args)
discriminator = Discriminator(args.state_shape-LATENT_SPACE, LATENT_SPACE).to(args.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

obs = env.new_episode()

skills_buffer = deque(maxlen=100_000)

metrics = {
    "timestep": [],
    "loss": [],
    "acc": []
}

skill_rewards = {
    "timestep": [],
    **{f"skill{k}": [] for k in range(LATENT_SPACE)}
}

############## SAMPLE Z #################
current_z, z = discriminator.sample_z()
#########################################

while model.timesteps <= args.total_timesteps:
    ############## GET ACTION WITH Z ########
    state = np.column_stack([obs, current_z])
    action = model.take_action(state)
    #########################################

    ############## STEP ENV ##############
    next_obs, _, done, info = env.step(action)
    next_state = np.column_stack([next_obs, current_z])
    #########################################

    ############## COMPUTE DISCRIMINATOR ####
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=args.device)
    logits = discriminator(obs_tensor) #(n_envs, LATENT_SPACE)

    probs = torch.log_softmax(logits, dim=-1)
    reward = probs[torch.arange(args.n_envs, device=args.device), z].detach().cpu().numpy() - np.log(1/LATENT_SPACE) #(n_envs, 1)

    #########################################

    model.collect_rollout(state, next_state, action, reward, done, info)
    for i in range(args.n_envs):
        skills_buffer.append(
            (
                next_obs[i].copy(),
                z[i]
            )
        )

    ############## UPDATE MODELS ##############
    obs, updating, new_best, finished_reward = model.update(done, next_obs)
    if done.any():
      skill_rewards["timestep"].append(model.timesteps)
      for i in range(args.n_envs):
        if done[i]:
            skill = int(z[i])
            skill_rewards[f"skill{skill}"].append(finished_reward[i])

      new_current_z, new_z = discriminator.sample_z()
      current_z[done] = new_current_z[done]
      z[done] = new_z[done]

    if model.timesteps % interval == 0 and model.timesteps > args.total_timesteps/4:
        print("Saving Discriminator")
        torch.save(
          discriminator.state_dict(),
          args.model_path + f"discriminator{model.timesteps}.pt"
        )

    if updating:
      if len(skills_buffer) < 1024:
        continue

      avg_loss = 0
      for _ in range(3):
          # minibatching with skills replay buffer
          batch = random.sample(skills_buffer, args.batch_size)

          states = torch.tensor(
              np.stack([s for s, _ in batch]),
              dtype=torch.float32,
              device=args.device
          )

          skills = torch.tensor(
              [z for _, z in batch],
              dtype=torch.long,
              device=args.device
          )

          optimizer.zero_grad()

          logits = discriminator(states)
          loss = criterion(logits, skills)
          avg_loss += loss.item()/3

          loss.backward()
          optimizer.step()

      with torch.no_grad():
        batch = random.sample(skills_buffer, args.batch_size)
        states = torch.tensor(
            np.stack([s for s, _ in batch]),
            dtype=torch.float32,
            device=args.device
        )

        skills = torch.tensor(
          [z for _, z in batch],
          dtype=torch.long,
          device=args.device
        )

        logits = discriminator(states)
        acc = (logits.argmax(-1) == skills).float().mean().item()

      print(f"Discrimnator Loss: {avg_loss} | Discrimnator Acc: {acc}")

      metrics["loss"].append(avg_loss)
      metrics["acc"].append(acc)
      metrics["timestep"].append(model.timesteps)

      pd.DataFrame(metrics).to_csv(args.csv_path + "discrim.csv", index=False)
      pd.DataFrame({
          key: pd.Series(value)
          for key, value in skill_rewards.items()
      }).to_csv(
          args.csv_path + "skillsReward.csv",
          index=False
      )

    ###########################################

torch.save(
  discriminator.state_dict(),
  args.model_path + f"discriminator{model.timesteps}.pt"
)