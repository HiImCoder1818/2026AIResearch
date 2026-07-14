from traing_algorithms.ppo import Args, Model, Environment
from stable_baselines3 import SAC

import numpy as np
from torch import nn

class Policy(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh()
        )

        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh()
        )

        self.latent_dim_pi = 1024
        self.latent_dim_vf = 1024

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)

def main():
    LATENT_SPACE = 50
    HORIZON = 100

    args = Args()
    args.model_path = "/content/drive/MyDrive/2026 AI Research/Code & Data/task/"
    args.csv_path = "/content/drive/MyDrive/2026 AI Research/Code & Data/task/"
    args.device = "cpu"
    args.action_shape = LATENT_SPACE

    args.env_name = "Ant-v5"
    args.n_envs = 8
    args.total_timesteps = 5_000_000

    args.batch_size = 256
    args.n_epochs = 50
    args.n_steps = 64
    args.learning_rate = 1e-4

    args.gamma = 0.995
    args.gae_lambda = 0.90
    args.clip_range = 0.2
    args.ent_coef = 0.001

    env = Environment(args, dict(
    terminate_when_unhealthy=False,
    forward_reward_weight=25,
    ctrl_cost_weight=0.3,
    ))
    meta_controller = Model(env, Policy, args)
    # meta_controller.load(args.model_path + "checkpointBest.pt")
    # meta_controller.load_metrics(args.csv_path + "data.csv")
    skills_model = SAC.load("checkpointFinal.pt", device=args.device)

    obs = env.new_episode()

    meta_controller.timesteps = 0

    while meta_controller.timesteps <= args.total_timesteps/HORIZON:
        option_reward = 0

        skill_idx, value, log_prob = meta_controller.take_action(obs)

        skill_one_hot = np.zeros((args.n_envs, LATENT_SPACE))
        skill_one_hot[np.arange(args.n_envs), skill_idx] = 1.0

        obs_start = obs.copy()

        for i in range(HORIZON):
            state_with_skill = np.concatenate([obs, skill_one_hot], axis=1)
            action, _ = skills_model.predict(state_with_skill, deterministic=False)

            next_obs, reward, done, info = env.step(action)
            obs = next_obs

            option_reward += reward

            # print(done)
            meta_controller.collect_rollout(obs_start, skill_idx, option_reward/HORIZON, value.cpu(), log_prob.cpu())
            obs = meta_controller.update(done, obs, verbose=True)

if __name__ == '__main__':
    main()