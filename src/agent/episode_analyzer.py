#!/usr/bin/env python3
"""
Interactive Episode Analyzer for Scheduler Training
Usage:
    python episode_analyzer.py                    # Interactive mode
    python episode_analyzer.py plot <episode_num> # Plot specific episode
    python episode_analyzer.py summary            # Print summary of all episodes
    python episode_analyzer.py compare <ep1> <ep2> <ep3>  # Compare episodes
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import os

class EpisodeAnalyzer:
    def __init__(self, log_file="scheduler_episodes.json", plots_dir="./plots"):
        self.log_file = log_file
        self.plots_dir = plots_dir
        self.episode_data = []
        self.model_names = ["efficientnetb0", "resnet18", "vit16"]
        self.model_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        
        os.makedirs(plots_dir, exist_ok=True)
        self.load_data()
    
    def load_data(self):
        """Load episode data from JSON file"""
        try:
            with open(self.log_file, 'r') as f:
                self.episode_data = json.load(f)
            print(f"Loaded {len(self.episode_data)} episodes from {self.log_file}")
        except FileNotFoundError:
            print(f"Warning: Log file {self.log_file} not found!")
            self.episode_data = []
    
    def print_summary(self):
        """Print summary statistics of all episodes"""
        if not self.episode_data:
            print("No episode data available!")
            return
        
        print("\n" + "="*80)
        print("EPISODE SUMMARY")
        print("="*80)
        
        total_rewards = [ep['total_reward'] for ep in self.episode_data]
        episode_lengths = [ep['episode_length'] for ep in self.episode_data]
        diversities = [ep['schedule_diversity'] for ep in self.episode_data]
        
        print(f"Total Episodes: {len(self.episode_data)}")
        print(f"Episode Range: {min(ep['episode'] for ep in self.episode_data)} - {max(ep['episode'] for ep in self.episode_data)}")
        print()
        
        print("REWARD STATISTICS:")
        print(f"  Mean Total Reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
        print(f"  Best Episode: {max(total_rewards):.3f} (Episode {self.episode_data[np.argmax(total_rewards)]['episode']})")
        print(f"  Worst Episode: {min(total_rewards):.3f} (Episode {self.episode_data[np.argmin(total_rewards)]['episode']})")
        print()
        
        print("EPISODE CHARACTERISTICS:")
        print(f"  Average Length: {np.mean(episode_lengths):.1f} steps")
        print(f"  Average Diversity: {np.mean(diversities):.3f}")
        print()
        
        # Reward component analysis
        if self.episode_data[0].get('slo_reward') is not None:
            slo_rewards = [ep.get('slo_reward', 0) for ep in self.episode_data]
            gpu_rewards = [ep.get('gpu_reward', 0) for ep in self.episode_data]
            batch_rewards = [ep.get('batch_fill_reward', 0) for ep in self.episode_data]
            switch_rewards = [ep.get('slot_switch_reward', 0) for ep in self.episode_data]
            
            print("REWARD COMPONENTS (Average):")
            print(f"  SLO Reward: {np.mean(slo_rewards):.3f}")
            print(f"  GPU Reward: {np.mean(gpu_rewards):.3f}")
            print(f"  Batch Fill Reward: {np.mean(batch_rewards):.3f}")
            print(f"  Slot Switch Reward: {np.mean(switch_rewards):.3f}")
        
        print("="*80)
    
    def plot_episode(self, episode_num):
        """Plot detailed analysis for a specific episode"""
        target_episode = None
        for episode in self.episode_data:
            if episode['episode'] == episode_num:
                target_episode = episode
                break
        
        if target_episode is None:
            print(f"Error: Episode {episode_num} not found!")
            available = [ep['episode'] for ep in self.episode_data]
            print(f"Available episodes: {sorted(available)}")
            return
        
        self._create_detailed_plot(target_episode)
        print(f"Plot saved to {self.plots_dir}/episode_{episode_num}_detailed.png")
    
    def compare_episodes(self, episode_nums):
        """Compare multiple episodes side by side"""
        episodes = []
        for num in episode_nums:
            episode = next((ep for ep in self.episode_data if ep['episode'] == num), None)
            if episode:
                episodes.append(episode)
            else:
                print(f"Warning: Episode {num} not found, skipping...")
        
        if not episodes:
            print("No valid episodes to compare!")
            return
        
        self._create_comparison_plot(episodes)
        episode_str = "_".join(str(ep['episode']) for ep in episodes)
        print(f"Comparison plot saved to {self.plots_dir}/episodes_comparison_{episode_str}.png")
    
    def plot_training_progress(self):
        """Plot overall training progress"""
        if not self.episode_data:
            print("No episode data available!")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        episodes = [ep['episode'] for ep in self.episode_data]
        total_rewards = [ep['total_reward'] for ep in self.episode_data]
        
        # 1. Training progress
        ax1.plot(episodes, total_rewards, 'b-', linewidth=2, alpha=0.7, label='Episode Reward')
        
        # Add moving average
        if len(total_rewards) > 10:
            window = min(20, len(total_rewards) // 3)
            moving_avg = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=3, label=f'Moving Avg ({window})')
        
        ax1.set_title('Training Progress - Total Reward', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward distribution
        ax2.hist(total_rewards, bins=min(20, len(total_rewards)//2), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(total_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(total_rewards):.3f}')
        ax2.set_title('Reward Distribution')
        ax2.set_xlabel('Total Reward')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Reward components over time
        if self.episode_data[0].get('slo_reward') is not None:
            components = ['slo_reward', 'gpu_reward', 'batch_fill_reward', 'slot_switch_reward']
            labels = ['SLO', 'GPU', 'Batch Fill', 'Slot Switch']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for component, label, color in zip(components, labels, colors):
                values = [ep.get(component, 0) for ep in self.episode_data]
                ax3.plot(episodes, values, color=color, linewidth=2, label=label, alpha=0.8)
            
            ax3.set_title('Reward Components Over Time')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Component Reward')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Episode characteristics
        lengths = [ep['episode_length'] for ep in self.episode_data]
        diversities = [ep['schedule_diversity'] for ep in self.episode_data]
        
        ax4_twin = ax4.twinx()
        
        ax4.plot(episodes, lengths, 'g-', linewidth=2, label='Episode Length')
        ax4_twin.plot(episodes, diversities, 'orange', linewidth=2, label='Schedule Diversity')
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Episode Length', color='green')
        ax4_twin.set_ylabel('Schedule Diversity', color='orange')
        ax4.set_title('Episode Length vs Schedule Diversity')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training progress plot saved to {self.plots_dir}/training_progress.png")
    
    def _create_detailed_plot(self, episode):
        """Create detailed plot for a single episode"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        steps = list(range(len(episode['steps'])))
        rewards = [step['reward'] for step in episode['steps']]
        
        # 1. Reward timeline (spanning top row)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(steps, rewards, 'b-', linewidth=2, alpha=0.7)
        ax1.fill_between(steps, rewards, alpha=0.3)
        ax1.set_title(f'Episode {episode["episode"]} - Reward Timeline', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Add annotations for significant events
        max_reward_step = np.argmax(rewards)
        min_reward_step = np.argmin(rewards)
        ax1.annotate(f'Max: {rewards[max_reward_step]:.3f}', 
                    xy=(max_reward_step, rewards[max_reward_step]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 2. Individual reward components
        ax2 = fig.add_subplot(gs[1, 0])
        individual_rewards = np.array([step['individual_reward'] for step in episode['steps']])
        if individual_rewards.size > 0 and individual_rewards.shape[1] >= 4:
            reward_types = ['SLO', 'GPU', 'Batch Fill', 'Slot Switch']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, (reward_type, color) in enumerate(zip(reward_types, colors)):
                ax2.plot(steps, individual_rewards[:, i], color=color, linewidth=2, label=reward_type)
            
            ax2.set_title('Reward Components')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Component Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Action heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        actions = [step['reduced_action'] for step in episode['steps']]
        action_matrix = np.array(actions).T
        
        im = ax3.imshow(action_matrix, aspect='auto', cmap='viridis')
        ax3.set_title('Action Pattern')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Action Component')
        ax3.set_yticks(range(4))
        ax3.set_yticklabels(['Slot ID', 'Model ID', 'Batch Delta', 'Parallel'])
        plt.colorbar(im, ax=ax3, shrink=0.8)
        
        # 4. Model usage pie chart
        ax4 = fig.add_subplot(gs[1, 2])
        model_actions = [action[1] for action in actions if action[1] < len(self.model_names)]
        if model_actions:
            unique_models, counts = np.unique(model_actions, return_counts=True)
            labels = [self.model_names[i] for i in unique_models]
            colors_pie = [self.model_colors[i] for i in unique_models]
            
            ax4.pie(counts, labels=labels, colors=colors_pie, autopct='%1.1f%%')
            ax4.set_title('Model Usage Distribution')
        
        # 5. GPU utilization timeline (spanning bottom two rows)
        ax5 = fig.add_subplot(gs[2:, :])
        self._plot_detailed_gpu_timeline(ax5, episode['steps'])
        
        plt.suptitle(f'Episode {episode["episode"]} Detailed Analysis\n'
                    f'Total Reward: {episode["total_reward"]:.3f} | '
                    f'Length: {episode["episode_length"]} steps | '
                    f'Diversity: {episode["schedule_diversity"]:.3f}', 
                    fontsize=18, fontweight='bold')
        
        plt.savefig(f'{self.plots_dir}/episode_{episode["episode"]}_detailed.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detailed_gpu_timeline(self, ax, steps):
        """Plot detailed GPU utilization timeline with action annotations"""
        num_slots = 2 * 3  # 2 GPUs, 3 slots each
        slot_height = 0.8
        
        # Create timeline
        for step_idx, step in enumerate(steps):
            action = step['reduced_action']
            action_slot_id = action[0] if action[0] < num_slots else -1
            action_model_id = action[1] if action[1] < len(self.model_names) else -1
            
            # Highlight action slot
            if action_slot_id >= 0:
                if action_model_id >= 0:
                    color = self.model_colors[action_model_id]
                    alpha = 0.8
                    label = self.model_names[action_model_id]
                else:
                    color = 'gray'
                    alpha = 0.5
                    label = 'EMPTY'
                
                # Draw action block
                rect = patches.Rectangle((step_idx, action_slot_id - slot_height/2), 
                                       1, slot_height, 
                                       linewidth=1, edgecolor='black',
                                       facecolor=color, alpha=alpha)
                ax.add_patch(rect)
                
                # Add text annotation for significant actions
                if step_idx % 5 == 0:  # Annotate every 5th step to avoid clutter
                    ax.text(step_idx + 0.5, action_slot_id, label, 
                           ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Formatting
        ax.set_xlim(0, len(steps))
        ax.set_ylim(-0.5, num_slots - 0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('GPU Slot')
        ax.set_title('GPU Slot Utilization Timeline (Actions Highlighted)')
        
        # Create y-axis labels
        slot_labels = []
        for gpu_id in range(2):
            for slot_id in range(3):
                slot_labels.append(f'GPU{gpu_id}-S{slot_id}')
        
        ax.set_yticks(range(num_slots))
        ax.set_yticklabels(slot_labels)
        
        # Add legend
        legend_elements = [patches.Patch(color=self.model_colors[i], label=self.model_names[i]) 
                          for i in range(len(self.model_names))]
        legend_elements.append(patches.Patch(color='gray', label='EMPTY'))
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    def _create_comparison_plot(self, episodes):
        """Create comparison plot for multiple episodes"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7DC6F', '#BB8FCE']
        
        # 1. Reward timelines
        ax = axes[0]
        for i, episode in enumerate(episodes):
            steps = list(range(len(episode['steps'])))
            rewards = [step['reward'] for step in episode['steps']]
            color = colors[i % len(colors)]
            ax.plot(steps, rewards, color=color, linewidth=2, 
                   label=f"Episode {episode['episode']}", alpha=0.8)
        
        ax.set_title('Reward Timelines Comparison')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Total rewards bar chart
        ax = axes[1]
        episode_nums = [ep['episode'] for ep in episodes]
        total_rewards = [ep['total_reward'] for ep in episodes]
        bars = ax.bar(episode_nums, total_rewards, color=colors[:len(episodes)], alpha=0.7)
        
        for bar, reward in zip(bars, total_rewards):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{reward:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Total Rewards Comparison')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        
        # 3. Episode statistics
        ax = axes[2]
        stats = ['mean', 'std', 'min', 'max']
        x = np.arange(len(episodes))
        width = 0.2
        
        for i, stat in enumerate(stats):
            values = [ep['reward_stats'][stat] for ep in episodes]
            ax.bar(x + i * width, values, width, label=stat.capitalize(), alpha=0.7)
        
        ax.set_title('Reward Statistics Comparison')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(episode_nums)
        ax.legend()
        
        # 4. Action diversity
        ax = axes[3]
        for i, episode in enumerate(episodes):
            actions = [tuple(step['reduced_action']) for step in episode['steps']]
            unique_actions = len(set(actions))
            total_actions = len(actions)
            diversity = unique_actions / total_actions if total_actions > 0 else 0
            
            ax.bar(episode['episode'], diversity, color=colors[i % len(colors)], alpha=0.7)
        
        ax.set_title('Action Diversity Comparison')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Diversity Score')
        
        # 5. Episode length comparison
        ax = axes[4]
        lengths = [ep['episode_length'] for ep in episodes]
        ax.bar(episode_nums, lengths, color=colors[:len(episodes)], alpha=0.7)
        ax.set_title('Episode Length Comparison')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Steps')
        
        # 6. Reward component breakdown
        ax = axes[5]
        if episodes[0].get('slo_reward') is not None:
            components = ['slo_reward', 'gpu_reward', 'batch_fill_reward', 'slot_switch_reward']
            component_names = ['SLO', 'GPU', 'Batch Fill', 'Slot Switch']
            
            x = np.arange(len(episodes))
            width = 0.2
            
            for i, (component, name) in enumerate(zip(components, component_names)):
                values = [ep.get(component, 0) for ep in episodes]
                ax.bar(x + i * width, values, width, label=name, alpha=0.7)
            
            ax.set_title('Reward Components Comparison')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Component Value')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(episode_nums)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/episodes_comparison_{"_".join(map(str, episode_nums))}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def interactive_mode(self):
        """Run interactive analysis mode"""
        while True:
            print("\n" + "="*60)
            print("SCHEDULER EPISODE ANALYZER")
            print("="*60)
            print("Available commands:")
            print("  1. summary - Show episode summary")
            print("  2. plot <episode_num> - Plot specific episode")
            print("  3. compare <ep1> <ep2> [ep3] - Compare episodes")
            print("  4. progress - Show training progress")
            print("  5. list - List available episodes")
            print("  6. reload - Reload data from file")
            print("  7. quit - Exit")
            print("-" * 60)
            
            try:
                command = input("Enter command: ").strip().split()
                
                if not command:
                    continue
                
                if command[0] == "quit":
                    break
                elif command[0] == "summary":
                    self.print_summary()
                elif command[0] == "plot" and len(command) > 1:
                    episode_num = int(command[1])
                    self.plot_episode(episode_num)
                elif command[0] == "compare" and len(command) > 2:
                    episode_nums = [int(x) for x in command[1:]]
                    self.compare_episodes(episode_nums)
                elif command[0] == "progress":
                    self.plot_training_progress()
                elif command[0] == "list":
                    if self.episode_data:
                        episodes = [ep['episode'] for ep in self.episode_data]
                        print(f"Available episodes: {sorted(episodes)}")
                    else:
                        print("No episodes available.")
                elif command[0] == "reload":
                    self.load_data()
                else:
                    print("Unknown command. Try again.")
            
            except (ValueError, IndexError) as e:
                print(f"Error: {e}. Please check your command syntax.")
            except KeyboardInterrupt:
                print("\nExiting...")
                break

def main():
    parser = argparse.ArgumentParser(description='Analyze scheduler training episodes')
    parser.add_argument('--log-file', default='scheduler_episodes.json', 
                       help='Path to episode log file')
    parser.add_argument('--plots-dir', default='./plots', 
                       help='Directory to save plots')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Plot specific episode')
    plot_parser.add_argument('episode', type=int, help='Episode number to plot')
    
    # Summary command
    subparsers.add_parser('summary', help='Print episode summary')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple episodes')
    compare_parser.add_argument('episodes', type=int, nargs='+', 
                               help='Episode numbers to compare')
    
    # Progress command
    subparsers.add_parser('progress', help='Show training progress')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    analyzer = EpisodeAnalyzer(args.log_file, args.plots_dir)
    
    if args.command == 'plot':
        analyzer.plot_episode(args.episode)
    elif args.command == 'summary':
        analyzer.print_summary()
    elif args.command == 'compare':
        analyzer.compare_episodes(args.episodes)
    elif args.command == 'progress':
        analyzer.plot_training_progress()
    elif args.command == 'interactive' or args.command is None:
        analyzer.interactive_mode()

if __name__ == "__main__":
    main()
