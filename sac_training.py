from traing_algorithms.sac import Args, Environment, Model
import time

def main(has_saves=False):
    args = Args()
    args.env_name = "Ant-v5"
    args.total_timesteps = 1_000_000
    args.device = "cuda"
    args.n_envs = 24
    args.model_path = f"data/models/checkpointTest.pt"
    args.csv_path = f"data/training_curves/testData.csv"

    env = Environment(args, dict(terminate_when_unhealthy=False))
    model = Model(env, "MlpPolicy", args)

    if has_saves:
        model.load(args.model_path)
        model.load_metrics(args.csv_path)

    obs = env.new_episode()

    start = time.time()
    while model.timesteps <= args.total_timesteps:
        action = model.take_action(obs)

        next_obs, reward, done, info = env.step(action)

        model.collect_rollout(obs, next_obs, action, reward, done, info)

        obs, updating, _, _ = model.update(done, next_obs, eval_reward=False)
        if updating:
            print(f"{time.time()-start}s")

    print(f"{time.time()-start}s to finish training")

if __name__ == '__main__':
    main(True)