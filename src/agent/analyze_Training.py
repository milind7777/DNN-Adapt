import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_scheduler_training(log_file="scheduler_episodes.json"):
    with open(log_file, 'r') as f:
        episodes = json.load(f)
    
    # Plot episode rewards
    rewards = [ep['total_reward'] for ep in episodes]
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot schedule diversity
    plt.subplot(2, 2, 2)
    diversity = [ep['schedule_diversity'] for ep in episodes]
    plt.plot(diversity)
    plt.title('Schedule Diversity')
    plt.xlabel('Episode')
    plt.ylabel('Diversity Score')
    
    # Plot episode lengths
    plt.subplot(2, 2, 3)
    lengths = [ep['episode_length'] for ep in episodes]
    plt.plot(lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot reward distribution
    plt.subplot(2, 2, 4)
    plt.hist(rewards, bins=20)
    plt.title('Reward Distribution')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('scheduler_training_analysis.png')
    plt.show()

if __name__ == "__main__":
    analyze_scheduler_training()