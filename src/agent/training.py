import json
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium_env import InferenceSchedulerEnv
from stable_baselines3.common.callbacks import CheckpointCallback

class SchedulerEpisodeCallback(BaseCallback):
    def __init__(self, num_gpus=2, num_models=3, scheduler_slots=3, 
                 log_file="scheduler_episodes.json", 
                 pretty_log_file="scheduler_pretty.log", 
                 verbose=0):
        super().__init__(verbose)
        self.num_gpus = num_gpus
        self.num_models = num_models
        self.scheduler_slots = scheduler_slots
        self.log_file = log_file
        self.pretty_log_file = pretty_log_file
        self.episode_data = []
        self.current_episode = {
            'steps': [],
            'reduced_actions': [],  # Changed from schedule_actions/batch_actions
            'observations': [],
            'rewards': []
        }
        self.episode_count = 0
        
        # Model names for pretty printing
        self.model_names = ["efficientnetb0", "resnet18", "vit16"]
        
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
            
            individual_rewards = np.sum([step['individual_reward'] for step in self.current_episode['steps']], axis=0)
            # print("Type:", type(individual_rewards))
            # print("Dtype:", individual_rewards.dtype)
            # print("Length:", len(individual_rewards))
            # print("Shape of arr[0]:", individual_rewards[0].shape)
            # print("Type of arr[0]:", type(individual_rewards[0]))

            episode_summary = {
                'episode': self.episode_count,
                'timestamp': datetime.now().isoformat(),
                'total_reward': sum(valid_rewards),
                'slo_reward': float(individual_rewards[0]),
                'gpu_reward': float(individual_rewards[1]),
                'batch_fill_reward': float(individual_rewards[2]),
                'slot_switch_reward': float(individual_rewards[3]),
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
            with open(self.log_file, 'w') as f:
                json.dump(self.episode_data, f, indent=2)
            
            self._write_pretty_log(episode_summary)
            
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
        """Create pretty schedule for reduced action format """
        pretty_lines = []
        
        # Parse reduced action: [slot_id, model_id, batch_delta, in_parallel]
        action_slot_id = reduced_action[0] if reduced_action[0] < self.num_gpus * self.scheduler_slots else -1
        action_model_id = reduced_action[1] if reduced_action[1] < self.num_models else -1
        batch_delta = reduced_action[2] - 4  # Convert back to actual delta
        in_parallel = bool(reduced_action[3])
        
        # Extract model data from new observation structure
        model_data = []
        for m_id in range(self.num_models):
            base_idx = m_id * 4
            model_info = {
                'request_rate': observation[base_idx] * 100.0,  # Denormalize
                'queue_size': observation[base_idx + 1] * 500.0,  # Denormalize
                'slo_latency': observation[base_idx + 2] * 2000.0,  # Denormalize
                'slo_satisfaction': observation[base_idx + 3] * 100.0  # Denormalize
            }
            model_data.append(model_info)
        
        # Extract current slot configurations from observation
        slot_start_idx = 4 * self.num_models
        features_per_slot = self.num_models + 1 + 2
        
        current_slots = []
        for s_id in range(self.num_gpus * self.scheduler_slots):
            base_idx = slot_start_idx + s_id * features_per_slot
            
            # Find deployed model from one-hot encoding
            deployed_model = -1
            for m in range(self.num_models):
                if base_idx + m < len(observation) and observation[base_idx + m] > 0.5:
                    deployed_model = m
                    break
            
            # Get batch size and parallel flag
            batch_size = int(observation[base_idx + self.num_models] * 512.0) if base_idx + self.num_models < len(observation) else 0
            is_parallel = observation[base_idx + self.num_models + 1] > 0.5 if base_idx + self.num_models + 1 < len(observation) else False
            
            current_slots.append({
                'model_id': deployed_model,
                'batch_size': batch_size,
                'in_parallel': is_parallel
            })
        
        # Create schedule for each GPU (similar to old format)
        for gpu_id in range(self.num_gpus):
            pretty_lines.append("****************************************************************************")
            pretty_lines.append(f"    GPU {gpu_id}: A6000  |  GPU MEMORY: 48GB | STEP: {len(self.current_episode['steps'])}")
            pretty_lines.append("    Session List: {model_name, SLO, request_rate, queue_size, current_batch, batch_delta, new_batch, execution_mode}")
            
            # Show action being taken if it affects this GPU
            action_gpu_id = action_slot_id // self.scheduler_slots if action_slot_id >= 0 else -1
            action_slot_idx = action_slot_id % self.scheduler_slots if action_slot_id >= 0 else -1
            
            if action_gpu_id == gpu_id and action_slot_id >= 0:
                pretty_lines.append(f"    >>> ACTION: Modifying Slot {action_slot_idx} -> Model {action_model_id}, Delta {batch_delta}, Parallel {in_parallel}")
            
            # Process each slot for this GPU
            for slot_idx in range(self.scheduler_slots):
                global_slot_id = gpu_id * self.scheduler_slots + slot_idx
                slot_info = current_slots[global_slot_id]
                
                if slot_info['model_id'] >= 0:  # Valid model deployed
                    model_name = self.model_names[slot_info['model_id']]
                    model_info = model_data[slot_info['model_id']]
                    execution_mode = "parallel" if slot_info['in_parallel'] else "sequential"
                    
                    # get new batch size if this slot is being modified
                    current_batch_size = slot_info['batch_size']
                    new_batch_size = current_batch_size
                    
                    if global_slot_id == action_slot_id and action_model_id == slot_info['model_id']:
                        new_batch_size = max(1, current_batch_size + batch_delta)
                        pretty_lines.append(f"      Slot {slot_idx}: {model_name}, {model_info['slo_latency']:.1f}ms, {model_info['request_rate']:.2f}req/s, "
                                          f"queue={model_info['queue_size']:.0f}, current_batch={current_batch_size}, batch_delta={batch_delta:+d}, new_batch={new_batch_size}, exe_mode:{execution_mode}")
                    else:
                        pretty_lines.append(f"      Slot {slot_idx}: {model_name}, {model_info['slo_latency']:.1f}ms, {model_info['request_rate']:.2f}req/s, "
                                          f"queue={model_info['queue_size']:.0f}, current_batch={current_batch_size}, batch_delta=+0, new_batch={new_batch_size}, exe_mode:{execution_mode}")
                else:
                    # Check if this empty slot is being filled by the action
                    if global_slot_id == action_slot_id and action_model_id >= 0:
                        model_name = self.model_names[action_model_id]
                        model_info = model_data[action_model_id]
                        execution_mode = "parallel" if in_parallel else "sequential"
                        new_batch_size = max(1, batch_delta)  # Starting from 0 + delta
                        pretty_lines.append(f"      Slot {slot_idx}: {model_name}, {model_info['slo_latency']:.1f}ms, {model_info['request_rate']:.2f}req/s, "
                                          f"queue={model_info['queue_size']:.0f}, current_batch=0, batch_delta={batch_delta:+d}, new_batch={new_batch_size}, exe_mode:{execution_mode}")
                    else:
                        pretty_lines.append(f"      Slot {slot_idx}: EMPTY")
            
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

    # def _on_rollout_end(self) -> None:
    #     if len(self.current_episode['steps']) > 0:
    #         print(f"Processing incomplete episode with {len(self.current_episode['steps'])} steps at rollout end")
    #         self._process_completed_episode()
        
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
                for m in range(self.num_models):
                    val = observation[base_idx + m]
                    one_hot_vals.append(val)
                    if val > 0.5:
                        deployed_model = m
                
                batch_size_norm = observation[base_idx + self.num_models] if base_idx + self.num_models < len(observation) else 0
                parallel_flag = observation[base_idx + self.num_models + 1] if base_idx + self.num_models + 1 < len(observation) else 0
                
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
        n_steps=60, 
        batch_size=20,
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
    
    # Clear pretty log file at starts
    with open("scheduler_pretty.log", 'w') as f:
        f.write(f"SCHEDULER TRAINING LOG - {datetime.now().isoformat()}\n")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=1800,                     # Save every 1000 environment steps
        save_path="./checkpoints/",         # Directory to save checkpoints
        name_prefix="ppo_scheduler_model"   # Prefix for checkpoint files
    )

    print("Starting training...")
    model.learn(total_timesteps=14400, callback=[callback, checkpoint_callback])

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
