import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_scheduler_training(log_file="scheduler_episodes.json"):
    with open(log_file, 'r') as f:
        episodes = json.load(f)
    
    # Plot episode rewards
    rewards = [ep['total_reward'] for ep in episodes]
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot slo reward
    plt.subplot(3, 2, 2)
    slo_rewards = [ep['slo_reward'] for ep in episodes]
    plt.plot(slo_rewards)
    plt.title('SLO Reward Cumulative')
    plt.xlabel('Episode')
    plt.ylabel('Reward Normalized')

    # Plot gpu reward
    plt.subplot(3, 2, 3)
    gpu_rewards = [ep['gpu_reward'] for ep in episodes]
    plt.plot(gpu_rewards)
    plt.title('GPU Reward Cumulative')
    plt.xlabel('Episode')
    plt.ylabel('Reward Normalized')

    # Plot batch fill reward
    plt.subplot(3, 2, 4)
    bf_rewards = [ep['batch_fill_reward'] for ep in episodes]
    plt.plot(bf_rewards)
    plt.title('Batch Fill Reward Cumulative')
    plt.xlabel('Episode')
    plt.ylabel('Reward Normalized')

    # Plot slot switch reward
    plt.subplot(3, 2, 5)
    ss_rewards = [ep['slot_switch_reward'] for ep in episodes]
    plt.plot(ss_rewards)
    plt.title('Slot Switch Reward Cumulative')
    plt.xlabel('Episode')
    plt.ylabel('Reward Normalized')

    # # Plot schedule diversity
    # plt.subplot(2, 2, 2)
    # diversity = [ep['schedule_diversity'] for ep in episodes]
    # plt.plot(diversity)
    # plt.title('Schedule Diversity')
    # plt.xlabel('Episode')
    # plt.ylabel('Diversity Score')
    
    # # Plot episode lengths
    # plt.subplot(2, 2, 3)
    # lengths = [ep['episode_length'] for ep in episodes]
    # plt.plot(lengths)
    # plt.title('Episode Lengths')
    # plt.xlabel('Episode')
    # plt.ylabel('Steps')
    
    # # Plot reward distribution
    # plt.subplot(2, 2, 4)
    # plt.hist(rewards, bins=20)
    # plt.title('Reward Distribution')
    # plt.xlabel('Total Reward')
    # plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('scheduler_training_analysis.png')
    plt.show()

if __name__ == "__main__":
    analyze_scheduler_training()