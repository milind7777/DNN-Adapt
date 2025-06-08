import torch
import numpy as np
from stable_baselines3 import PPO
from gymnasium_env import InferenceSchedulerEnv

def evaluate_policy(model, env, render=False):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        print(f"Step {step}: Action={action}, Reward={reward:.3f}")
        for i in range(3):
            ind = i * 4
            print(f"Model id: {i}, Request Rate: {obs[ind+0]}, Queue Size: {obs[ind+1]}, SLO: {obs[ind+2]}, SLO rate: {obs[ind+3]}")
        id_to_model = [
            "efficientbetb0",
            "resnet18",
            "vit16",
            "EMPTY"
        ]
        ind = 3 * 4
        for i in range(6):
            model_id = -1
            ind = 3 * 4 + i * 6
            for j in range(4):
                if obs[ind + j] == 1:
                    model_id = j
            
            print(f"SLOT: {i}, model: {id_to_model[model_id]}, batch: {obs[ind + 4]}, paralle: {obs[ind + 5]}")
        
        if render:
            print(env.render())

    print(f"\nEvaluation complete: Total reward = {total_reward:.3f}, Steps = {step}")

def main():
    # Path to unzipped model checkpoint
    checkpoint_path = "./checkpoints/ppo_scheduler_model_9000_steps.zip"  # Update if needed

    # Recreate environment
    env = InferenceSchedulerEnv(
        address="localhost:50051",
        num_models=3,
        num_gpus=2,
        scheduler_slots=3
    )

    # Load model
    model = PPO.load(checkpoint_path, env=env, device='cpu')  # Change device if needed

    # Run evaluation
    evaluate_policy(model, env)

    # Cleanup
    env.unwrapped.grpc_close()

if __name__ == "__main__":
    main()
