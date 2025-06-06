import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium_env import InferenceSchedulerEnv
from stable_baselines3.common.callbacks import CheckpointCallback

class EntropyDecayCallback(BaseCallback):
    def __init__(self, initial_ent_coef=0.01, final_ent_coef=0.0, total_timesteps=1e5, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        new_ent_coef = self.initial_ent_coef * (1 - progress) + self.final_ent_coef * progress
        self.model.ent_coef = new_ent_coef
        return True

class SchedulerEpisodeCallback(BaseCallback):
    def __init__(self, num_gpus=2, num_models=3, scheduler_slots=3, 
                 log_file="scheduler_episodes.json", 
                 pretty_log_file="scheduler_pretty.log", 
                 plots_dir="./plots",
                 verbose=0):
        super().__init__(verbose)
        self.num_gpus = num_gpus
        self.num_models = num_models
        self.scheduler_slots = scheduler_slots
        self.log_file = log_file
        self.pretty_log_file = pretty_log_file
        self.plots_dir = plots_dir
        self.episode_data = []
        self.current_episode = {
            'steps': [],
            'reduced_actions': [],
            'observations': [],
            'rewards': []
        }
        self.episode_count = 0
        
        # Model names for pretty printing
        self.model_names = ["efficientnetb0", "resnet18", "vit16"]
        self.model_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        
        # Create plots directory
        import os
        os.makedirs(plots_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Extract scheduler-specific data
        action = self.locals['actions'][0]
        obs = self.locals['obs_tensor'][0].cpu().numpy()
        reward = float(self.locals['rewards'][0])
        done = bool(self.locals['dones'][0])
        
        # Check for NaN values
        if np.isnan(reward):
            print(f"WARNING: NaN reward detected at step {len(self.current_episode['steps'])}")
            reward = 0.0
        
        if np.any(np.isnan(obs)):
            print(f"WARNING: NaN observation detected at step {len(self.current_episode['steps'])}")
            obs = np.nan_to_num(obs, nan=0.0)
        
        # Parse reduced action format: [slot_id, model_id, batch_delta, in_parallel]
        reduced_action = action.tolist()
        
        # Print observation space info every 10 steps or on first step of episode
        step_num = len(self.current_episode['steps'])
        if step_num == 0 or step_num % 10 == 0:
            self._print_observation_space(obs, step_num)
        
        # Create pretty printed schedule
        pretty_schedule = self._create_pretty_schedule_reduced(reduced_action, obs)
        
        # Store step data
        step_data = {
            'step': len(self.current_episode['steps']),
            'reward': reward,
            'reduced_action': reduced_action,
            'observation_summary': {
                'mean': float(np.mean(obs)),
                'std': float(np.std(obs)),
                'min': float(np.min(obs)),
                'max': float(np.max(obs))
            },
            'pretty_schedule': pretty_schedule,
            'individual_reward': [float(r) for r in self.training_env.envs[0].unwrapped.last_info]
        }
        
        self.current_episode['steps'].append(step_data)
        self.current_episode['reduced_actions'].append(reduced_action)
        self.current_episode['observations'].append(obs.tolist())
        self.current_episode['rewards'].append(reward)
        
        if done:
            print("Is done")
            print(f"Episode {self.episode_count} terminated after {len(self.current_episode['steps'])} steps")
            self._process_completed_episode()
        return True

    # after _on_step and before _create_pretty_schedule
    def _process_completed_episode(self):
        """Process completed episode immediately when done=True"""
        if len(self.current_episode['steps']) > 0:
            rewards = self.current_episode['rewards']
            valid_rewards = [r for r in rewards if not np.isnan(r)]
            
            try:
                individual_rewards = np.sum([step['individual_reward'] for step in self.current_episode['steps']], axis=0)
                if len(individual_rewards) >= 4:
                    slo_reward = float(individual_rewards[0])
                    gpu_reward = float(individual_rewards[1])
                    batch_fill_reward = float(individual_rewards[2])
                    slot_switch_reward = float(individual_rewards[3])
                else:
                    slo_reward = gpu_reward = batch_fill_reward = slot_switch_reward = 0.0
            except (IndexError, ValueError, TypeError) as e:
                print(f"WARNING: Error processing individual rewards: {e}")
                slo_reward = gpu_reward = batch_fill_reward = slot_switch_reward = 0.0

            episode_summary = {
                'episode': self.episode_count,
                'timestamp': datetime.now().isoformat(),
                'total_reward': sum(valid_rewards),
                'slo_reward': slo_reward,
                'gpu_reward': gpu_reward,
                'batch_fill_reward': batch_fill_reward,
                'slot_switch_reward': slot_switch_reward,
                'episode_length': len(self.current_episode['steps']),
                'reward_stats': {
                    'mean': float(np.mean(valid_rewards)) if valid_rewards else 0.0,
                    'std': float(np.std(valid_rewards)) if len(valid_rewards) > 1 else 0.0,
                    'min': float(np.min(valid_rewards)) if valid_rewards else 0.0,
                    'max': float(np.max(valid_rewards)) if valid_rewards else 0.0
                },
                'schedule_diversity': self._calculate_schedule_diversity(),
                'nan_reward_count': len(rewards) - len(valid_rewards),
                'steps': self.current_episode['steps']
            }
            
            self.episode_data.append(episode_summary)
            
            # Save after each episode
            try:
                with open(self.log_file, 'w') as f:
                    json.dump(self.episode_data, f, indent=2)
            except Exception as e:
                print(f"WARNING: Failed to save episode data: {e}")
            
            self._write_pretty_log(episode_summary)
            
            # Generate plots for this episode
            self._plot_episode_analysis(episode_summary)
            
            # Generate reward trend for last 3 episodes
            if len(self.episode_data) >= 3:
                self._plot_recent_reward_trends()
            
            if self.verbose >= 1:
                print(f"Episode {self.episode_count} COMPLETED: "
                    f"Reward={episode_summary['total_reward']:.3f}, "
                    f"Length={episode_summary['episode_length']}, "
                    f"Avg Reward={episode_summary['reward_stats']['mean']:.3f}")
            
            # Reset for next episode
            self.current_episode = {
                'steps': [], 'reduced_actions': [], 'observations': [], 'rewards': []
            }
            self.episode_count += 1

    def _create_pretty_schedule_reduced(self, reduced_action, observation):
        """Create pretty schedule for reduced action format - correctly interpreting actions"""
        pretty_lines = []
        
        # Parse reduced action: [slot_id, model_id, batch_delta, in_parallel]
        action_slot_id = int(reduced_action[0]) if reduced_action[0] < self.num_gpus * self.scheduler_slots else -1
        action_model_id = int(reduced_action[1]) if reduced_action[1] < self.num_models else -1
        batch_delta = int(reduced_action[2]) - 4  # Convert back to actual delta
        in_parallel = bool(reduced_action[3])
        
        # Extract model data from new observation structure with bounds checking
        model_data = []
        for m_id in range(self.num_models):
            base_idx = m_id * 4
            if base_idx + 3 < len(observation):
                model_info = {
                    'request_rate': observation[base_idx] * 100.0,  # Denormalize
                    'queue_size': observation[base_idx + 1] * 500.0,  # Denormalize
                    'slo_latency': observation[base_idx + 2] * 2000.0,  # Denormalize
                    'slo_satisfaction': observation[base_idx + 3] * 100.0  # Denormalize
                }
            else:
                # Fallback values
                model_info = {
                    'request_rate': 0.0, 'queue_size': 0.0, 'slo_latency': 1000.0, 'slo_satisfaction': 0.0
                }
            model_data.append(model_info)
        
        # Extract current slot configurations from observation
        slot_start_idx = 4 * self.num_models
        features_per_slot = self.num_models + 1 + 2  # one-hot (including empty) + batch_size + parallel
        
        current_slots = []
        for s_id in range(self.num_gpus * self.scheduler_slots):
            base_idx = slot_start_idx + s_id * features_per_slot
            
            # Find deployed model from one-hot encoding (including empty slot at index num_models)
            deployed_model = -1
            for m in range(self.num_models + 1):  # +1 for empty slot
                if base_idx + m < len(observation) and observation[base_idx + m] > 0.5:
                    deployed_model = m if m < self.num_models else -1  # -1 for empty
                    break
            
            # Get batch size and parallel flag with correct indexing
            batch_size_idx = base_idx + self.num_models + 1  # After one-hot encoding including empty
            parallel_idx = base_idx + self.num_models + 2    # After batch size
            
            batch_size = int(observation[batch_size_idx] * 512.0) if batch_size_idx < len(observation) else 0
            is_parallel = observation[parallel_idx] > 0.5 if parallel_idx < len(observation) else False
            
            current_slots.append({
                'model_id': deployed_model,
                'batch_size': batch_size,
                'in_parallel': is_parallel
            })
        
        # Create schedule for each GPU (matching old format)
        for gpu_id in range(self.num_gpus):
            pretty_lines.append("****************************************************************************")
            pretty_lines.append(f"    GPU {gpu_id}: A6000  |  GPU MEMORY: 48GB | STEP: {len(self.current_episode['steps'])}")
            pretty_lines.append("    Session List: {model_name, SLO, request_rate, queue_size, observed_batch, batch_delta, deployed_batch, execution_mode}")
            
            # Show action being taken if it affects this GPU
            action_gpu_id = action_slot_id // self.scheduler_slots if action_slot_id >= 0 else -1
            action_slot_idx = action_slot_id % self.scheduler_slots if action_slot_id >= 0 else -1
            
            if action_gpu_id == gpu_id and action_slot_id >= 0:
                action_model_name = self.model_names[action_model_id] if 0 <= action_model_id < len(self.model_names) else "EMPTY"
                pretty_lines.append(f"    >>> ACTION: Slot {action_slot_idx} -> {action_model_name} (ID:{action_model_id}), Delta {batch_delta}, Parallel {in_parallel}")
            
            # Process each slot for this GPU
            for slot_idx in range(self.scheduler_slots):
                global_slot_id = gpu_id * self.scheduler_slots + slot_idx
                
                if global_slot_id < len(current_slots):
                    slot_info = current_slots[global_slot_id]
                    
                    # Determine what to display based on current state and action
                    if global_slot_id == action_slot_id:
                        # This slot is being modified by the action
                        if action_model_id >= 0:
                            # Deploying a model (either new or replacing existing)
                            model_name = self.model_names[action_model_id]
                            model_info = model_data[action_model_id]
                            execution_mode = "parallel" if in_parallel else "sequential"
                            
                            if slot_info['model_id'] == action_model_id:
                                # Same model, just changing batch/parallel settings
                                observed_batch = slot_info['batch_size']
                                deployed_batch = max(0, observed_batch + batch_delta)
                            else:
                                # New model deployment or replacement
                                observed_batch = 0 if slot_info['model_id'] == -1 else slot_info['batch_size']
                                deployed_batch = max(0, batch_delta)  # Start with delta as base
                            
                            pretty_lines.append(f"      Slot {slot_idx}: {model_name}, {model_info['slo_latency']:.1f}ms, {model_info['request_rate']:.2f}req/s, "
                                              f"queue={model_info['queue_size']:.0f}, observed_batch={observed_batch}, batch_delta={batch_delta:+d}, "
                                              f"deployed_batch={deployed_batch}, execution_mode:{execution_mode}")
                        else:
                            # Emptying the slot
                            if slot_info['model_id'] >= 0:
                                old_model_name = self.model_names[slot_info['model_id']]
                                pretty_lines.append(f"      Slot {slot_idx}: EMPTY (removing {old_model_name})")
                            else:
                                pretty_lines.append(f"      Slot {slot_idx}: EMPTY")
                    else:
                        # This slot is not being modified - show current state
                        if slot_info['model_id'] >= 0:
                            model_name = self.model_names[slot_info['model_id']]
                            model_info = model_data[slot_info['model_id']]
                            execution_mode = "parallel" if slot_info['in_parallel'] else "sequential"
                            current_batch = slot_info['batch_size']
                            
                            pretty_lines.append(f"      Slot {slot_idx}: {model_name}, {model_info['slo_latency']:.1f}ms, {model_info['request_rate']:.2f}req/s, "
                                              f"queue={model_info['queue_size']:.0f}, observed_batch={current_batch}, batch_delta=+0, "
                                              f"deployed_batch={current_batch}, execution_mode:{execution_mode}")
                        else:
                            pretty_lines.append(f"      Slot {slot_idx}: EMPTY")
                else:
                    pretty_lines.append(f"      Slot {slot_idx}: ERROR - Slot index out of range")
            
            pretty_lines.append("****************************************************************************")
            pretty_lines.append("")  
        
        return "\n".join(pretty_lines)

    def _write_pretty_log(self, episode_summary):
        with open(self.pretty_log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"EPISODE {episode_summary['episode']} - {episode_summary['timestamp']}\n")
            f.write(f"Total Reward: {episode_summary['total_reward']:.3f} | Episode Length: {episode_summary['episode_length']}\n")
            f.write(f"{'='*80}\n\n")
            
            for step_idx, step in enumerate(episode_summary['steps']):
                f.write(f"STEP {step_idx} - Reward: {step['reward']:.3f}\n")
                f.write(step['pretty_schedule'])
                f.write(f"\n{'-'*40}\n\n")

    def _plot_episode_analysis(self, episode_summary):
        """Create comprehensive plots for a single episode"""
        episode_num = episode_summary['episode']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Reward over time
        ax1 = fig.add_subplot(gs[0, :])
        steps = list(range(len(episode_summary['steps'])))
        rewards = [step['reward'] for step in episode_summary['steps']]
        
        ax1.plot(steps, rewards, 'b-', linewidth=2, alpha=0.7)
        ax1.fill_between(steps, rewards, alpha=0.3)
        ax1.set_title(f'Episode {episode_num} - Reward per Step', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Add moving average
        if len(rewards) > 5:
            window = min(10, len(rewards) // 2)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(steps[window-1:], moving_avg, 'r--', linewidth=2, label=f'Moving Avg ({window})')
            ax1.legend()
        
        # 2. Individual reward components
        ax2 = fig.add_subplot(gs[1, 0])
        individual_rewards = np.array([step['individual_reward'] for step in episode_summary['steps']])
        if individual_rewards.size > 0 and individual_rewards.shape[1] >= 4:
            reward_types = ['SLO', 'GPU', 'Batch Fill', 'Slot Switch']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, (reward_type, color) in enumerate(zip(reward_types, colors)):
                ax2.plot(steps, individual_rewards[:, i], color=color, linewidth=2, label=reward_type)
            
            ax2.set_title('Individual Reward Components')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Reward')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Action distribution
        ax3 = fig.add_subplot(gs[1, 1])
        actions = [step['reduced_action'] for step in episode_summary['steps']]
        slot_actions = [action[0] for action in actions if action[0] < self.num_gpus * self.scheduler_slots]
        model_actions = [action[1] for action in actions if action[1] < self.num_models]
        
        if slot_actions:
            unique_slots, slot_counts = np.unique(slot_actions, return_counts=True)
            ax3.bar(unique_slots, slot_counts, alpha=0.7, color='skyblue')
            ax3.set_title('Slot Usage Distribution')
            ax3.set_xlabel('Slot ID')
            ax3.set_ylabel('Usage Count')
            ax3.set_xticks(range(self.num_gpus * self.scheduler_slots))
        
        # 4. Model deployment frequency
        ax4 = fig.add_subplot(gs[1, 2])
        if model_actions:
            unique_models, model_counts = np.unique(model_actions, return_counts=True)
            bars = ax4.bar(unique_models, model_counts, alpha=0.7, 
                          color=[self.model_colors[i] for i in unique_models])
            ax4.set_title('Model Deployment Frequency')
            ax4.set_xlabel('Model ID')
            ax4.set_ylabel('Deployment Count')
            ax4.set_xticks(range(self.num_models))
            ax4.set_xticklabels([self.model_names[i] for i in range(self.num_models)], rotation=45)
        
        # 5. GPU utilization timeline
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_gpu_timeline(ax5, episode_summary['steps'])
        
        plt.suptitle(f'Episode {episode_num} Analysis - Total Reward: {episode_summary["total_reward"]:.3f}', 
                     fontsize=16, fontweight='bold')
        
        # Save plot
        plt.savefig(f'{self.plots_dir}/episode_{episode_num}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gpu_timeline(self, ax, steps):
        """Plot GPU slot utilization timeline"""
        step_count = len(steps)
        y_positions = []
        labels = []
        
        # Create y-axis positions for each GPU slot
        for gpu_id in range(self.num_gpus):
            for slot_id in range(self.scheduler_slots):
                y_positions.append(gpu_id * self.scheduler_slots + slot_id)
                labels.append(f'GPU{gpu_id}-S{slot_id}')
        
        # Extract deployment info from observations
        for step_idx, step in enumerate(steps):
            obs = step.get('observation_summary', {})
            action = step['reduced_action']
            
            # Parse action to understand what happened
            action_slot_id = action[0] if action[0] < self.num_gpus * self.scheduler_slots else -1
            action_model_id = action[1] if action[1] < self.num_models else -1
            
            # Simulate slot states (simplified version)
            for slot_global_id in range(self.num_gpus * self.scheduler_slots):
                y_pos = slot_global_id
                
                # Color based on deployed model (simplified assumption)
                if slot_global_id == action_slot_id and action_model_id >= 0:
                    color = self.model_colors[action_model_id]
                    ax.barh(y_pos, 1, left=step_idx, height=0.8, color=color, alpha=0.7)
                elif step_idx > 0:  # Assume previous state continues
                    continue
        
        ax.set_xlim(0, step_count)
        ax.set_ylim(-0.5, len(y_positions) - 0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Step')
        ax.set_title('GPU Slot Utilization Timeline')
        
        # Add legend for models
        legend_elements = [patches.Patch(color=self.model_colors[i], label=self.model_names[i]) 
                          for i in range(self.num_models)]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _plot_recent_reward_trends(self):
        """Plot reward trends for the last 3 episodes"""
        recent_episodes = self.episode_data[-3:]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Total rewards comparison
        episode_nums = [ep['episode'] for ep in recent_episodes]
        total_rewards = [ep['total_reward'] for ep in recent_episodes]
        
        bars = ax1.bar(episode_nums, total_rewards, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Total Rewards - Last 3 Episodes')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        
        # Add value labels on bars
        for bar, reward in zip(bars, total_rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{reward:.3f}', ha='center', va='bottom')
        
        # 2. Individual reward components
        reward_components = ['slo_reward', 'gpu_reward', 'batch_fill_reward', 'slot_switch_reward']
        component_names = ['SLO', 'GPU', 'Batch Fill', 'Slot Switch']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        x = np.arange(len(episode_nums))
        width = 0.2
        
        for i, (component, name, color) in enumerate(zip(reward_components, component_names, colors)):
            values = [ep.get(component, 0) for ep in recent_episodes]
            ax2.bar(x + i * width, values, width, label=name, color=color, alpha=0.7)
        
        ax2.set_title('Reward Components - Last 3 Episodes')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Component Reward')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(episode_nums)
        ax2.legend()
        
        # 3. Episode statistics
        stats = ['mean', 'std', 'min', 'max']
        stat_names = ['Mean', 'Std Dev', 'Min', 'Max']
        
        for i, (stat, name) in enumerate(zip(stats, stat_names)):
            values = [ep['reward_stats'][stat] for ep in recent_episodes]
            ax3.plot(episode_nums, values, marker='o', linewidth=2, label=name)
        
        ax3.set_title('Reward Statistics - Last 3 Episodes')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Episode length and diversity
        episode_lengths = [ep['episode_length'] for ep in recent_episodes]
        diversities = [ep['schedule_diversity'] for ep in recent_episodes]
        
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar([x - 0.2 for x in episode_nums], episode_lengths, 0.4, 
                       label='Episode Length', color='lightblue', alpha=0.7)
        bars2 = ax4_twin.bar([x + 0.2 for x in episode_nums], diversities, 0.4, 
                            label='Schedule Diversity', color='lightcoral', alpha=0.7)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Episode Length', color='blue')
        ax4_twin.set_ylabel('Schedule Diversity', color='red')
        ax4.set_title('Episode Length vs Schedule Diversity')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/recent_episodes_trend.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_schedule_diversity(self):
        if not self.current_episode['reduced_actions']:
            return 0.0
        
        unique_actions = len(set(map(tuple, self.current_episode['reduced_actions'])))
        total_actions = len(self.current_episode['reduced_actions'])
        return unique_actions / total_actions if total_actions > 0 else 0.0

    def _print_observation_space(self, observation, step_num):
        """Print detailed observation space information"""
        print(f"\n{'='*60}")
        print(f"OBSERVATION SPACE - Episode {self.episode_count}, Step {step_num}")
        print(f"{'='*60}")
        print(f"Observation length: {len(observation)}")
        print(f"Observation shape: {observation.shape}")
        print(f"Observation range: [{np.min(observation):.4f}, {np.max(observation):.4f}]")
        print(f"Mean: {np.mean(observation):.4f}, Std: {np.std(observation):.4f}")
        
        # Print model features section
        print(f"\n--- MODEL FEATURES (indices 0-{4*self.num_models-1}) ---")
        for m_id in range(self.num_models):
            base_idx = m_id * 4
            if base_idx + 3 < len(observation):
                print(f"Model {m_id} ({self.model_names[m_id]}):")
                print(f"  Request Rate (norm): {observation[base_idx]:.4f}")
                print(f"  Queue Size (norm):   {observation[base_idx+1]:.4f}")
                print(f"  SLO Latency (norm):  {observation[base_idx+2]:.4f}")
                print(f"  SLO Satisfaction:    {observation[base_idx+3]:.4f}")
        
        # Print slot features section
        slot_start_idx = 4 * self.num_models
        features_per_slot = self.num_models + 1 + 2
        print(f"\n--- SLOT FEATURES (indices {slot_start_idx}-{len(observation)-1}) ---")
        
        for s_id in range(self.num_gpus * self.scheduler_slots):
            base_idx = slot_start_idx + s_id * features_per_slot
            gpu_id = s_id // self.scheduler_slots
            slot_idx = s_id % self.scheduler_slots
            
            if base_idx + features_per_slot <= len(observation):
                # Find deployed model from one-hot encoding
                deployed_model = -1
                one_hot_vals = []
                for m in range(self.num_models+1) :
                    val = observation[base_idx + m ]
                    one_hot_vals.append(val)
                    if val > 0.5:
                        deployed_model = m
                
                batch_size_norm = observation[base_idx + self.num_models + 1] if base_idx + self.num_models < len(observation) else 0
                parallel_flag = observation[base_idx + self.num_models + 2] if base_idx + self.num_models + 1 < len(observation) else 0
                
                print(f"GPU {gpu_id}, Slot {slot_idx} (global slot {s_id}):")
                print(f"  One-hot encoding: {[f'{v:.3f}' for v in one_hot_vals]} -> Model {deployed_model}")
                print(f"  Batch size (norm): {batch_size_norm:.4f}")
                print(f"  Parallel flag: {parallel_flag:.4f}")
        
        print(f"{'='*60}\n")

def main():
    # Environment parameters
    num_models = 3
    num_gpus = 2
    scheduler_slots = 3
    
    base_env = InferenceSchedulerEnv(
        address="localhost:50051",
        num_models=num_models,
        num_gpus=num_gpus,
        scheduler_slots=scheduler_slots
    )

    env = RecordEpisodeStatistics(base_env)

    # Configure logger
    _logger = configure("./logs", ["stdout", "csv", "tensorboard"])
    
    # Initialize model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device='cpu', 
        n_steps=100,
        n_epochs=5,
        batch_size=50,
        learning_rate=3e-4, 
        tensorboard_log="./tb_logs"
    )

    model.set_logger(_logger)

    # Create callback with pretty logging
    callback = SchedulerEpisodeCallback(
        num_gpus=num_gpus,
        num_models=num_models,
        scheduler_slots=scheduler_slots,
        log_file="scheduler_episodes.json",
        pretty_log_file="scheduler_pretty.log",
        verbose=1
    )    
    
    entropy_callback = EntropyDecayCallback(
        initial_ent_coef=0.01, 
        final_ent_coef=0.0, 
        total_timesteps=14400
    )

    # Clear pretty log file at starts
    with open("scheduler_pretty.log", 'w') as f:
        f.write(f"SCHEDULER TRAINING LOG - {datetime.now().isoformat()}\n")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=1800,                     # Save every 1000 environment steps
        save_path="./checkpoints/",         # Directory to save checkpoints
        name_prefix="ppo_scheduler_model"   # Prefix for checkpoint files
    )

    print("Starting training...")
    model.learn(total_timesteps=14400, callback=[callback, checkpoint_callback, entropy_callback])

    # Print final statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if hasattr(env, 'episode_returns') and len(env.return_queue) > 0:
        print(f"Episodes completed: {len(env.return_queue)}")
        print(f"Average episode return: {np.mean(env.return_queue):.3f}")
        print(f"Best episode return: {np.max(env.return_queue):.3f}")
        print(f"Average episode length: {np.mean(env.return_queue):.1f}")

    model.save("testing_grpc")
    env.unwrapped.grpc_close()
    print(f"\nDetailed logs saved to: scheduler_episodes.json")
    print(f"Pretty schedule logs saved to: scheduler_pretty.log")
    print(f"TensorBoard logs: tensorboard --logdir ./tb_logs/")


if __name__ == "__main__":
    main()
