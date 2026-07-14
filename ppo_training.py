from traing_algorithms.ppo import Args, Model, Environment

def main():
    args = Args()
    args.model_path = "/content/drive/MyDrive/2026 AI Research/Code & Data/"
    args.csv_path = "/content/drive/MyDrive/2026 AI Research/Code & Data/"
    args.env_name = "Ant-v5"
    args.device = "cuda"

    args.total_timesteps = 2_000_000

    env = Environment(args, dict(terminate_when_unhealthy=False))
    model = Model(env, "MlpPolicy", args)

    obs = env.new_episode()

    while model.timesteps <= args.total_timesteps:
        action, val, log_prob = model.take_action(obs)

        next_obs, rewards, dones, infos = env.step(action)

        model.collect_rollout(obs, action, rewards, val, log_prob)

        obs = model.update(dones, next_obs)

if __name__ == '__main__':
    main()