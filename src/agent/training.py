from stable_baselines3 import PPO
from gymnasium_env import InferenceSchedulerEnv

def main():
    env = InferenceSchedulerEnv(
        address="localhost:50051",
        num_models=3,
        num_gpus=2,
        scheduler_slots=3
    )

    # Initialise model
    model = PPO("MlpPolicy", env, verbose=1, device='cpu', n_steps=2)

    # Train the agent
    model.learn(total_timesteps=2)

    # save model
    model.save("testing_grpc")

    env.close()

if __name__ == "__main__":
    main()