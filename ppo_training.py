from traing_algorithms.ppo import Args, Model, Environment

def main(has_saves=False):
    args = Args()
    args.model_path = f"data/models/checkpointPPO.pt"
    args.csv_path = f"data/training_curves/dataPPO.csv"
    args.env_name = "Ant-v5"
    args.device = "cuda"

    args.total_timesteps = 1_000_000
    args.n_envs = 24

    env = Environment(args, dict(terminate_when_unhealthy=False))
    model = Model(env, "MlpPolicy", args)

    if has_saves:
        model.load(args.model_path)
        model.load_metrics(args.csv_path)

    obs = env.new_episode()

    while model.timesteps <= args.total_timesteps:
        action, val, log_prob = model.take_action(obs)

        next_obs, rewards, dones, infos = env.step(action)

        model.collect_rollout(obs, action, rewards, val, log_prob)

        obs = model.update(dones, next_obs, eval_reward=False)

if __name__ == '__main__':
    main(True)