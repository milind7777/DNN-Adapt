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
            'schedule_actions': [],
            'batch_actions': [],
            'observations': [],
            'rewards': []
        }
        self.episode_count = 0
        
        # Model names for pretty printing
        self.model_names = ["efficientnetb0" ,"resnet18", "vit16"]
        self.model_feature_dim = 2 + 3 * self.num_gpus  # 2 + 3 * num_gpus
        
    def _on_step(self) -> bool:
        # Extract scheduler-specific data
        action = self.locals['actions'][0]
        obs = self.locals['obs_tensor'][0].cpu().numpy()
        reward = float(self.locals['rewards'][0])
        #terminated = self.locals['']
        done = bool(self.locals['dones'][0])
        # If done, finalize the current episode
        # print("Done:", done)
        
        # Check for NaN values
        if np.isnan(reward):
            print(f"WARNING: NaN reward detected at step {len(self.current_episode['steps'])}")
            reward = 0.0
        
        if np.any(np.isnan(obs)):
            print(f"WARNING: NaN observation detected at step {len(self.current_episode['steps'])}")
            obs = np.nan_to_num(obs, nan=0.0)
        
        # Parse action into schedule and batch components
        schedule_fields = 2 * self.num_gpus * self.scheduler_slots
        schedule_action = action[:schedule_fields].tolist()
        batch_action = action[schedule_fields:].tolist()
        
        # Create  printed schedule
        pretty_schedule = self._create_pretty_schedule(schedule_action, batch_action, obs)
        
        # Store step data
        step_data = {
            'step': len(self.current_episode['steps']),
            'reward': reward,
            'schedule_action': schedule_action,
            'batch_action': batch_action,
            'observation_summary': {
                'mean': float(np.mean(obs)),
                'std': float(np.std(obs)),
                'min': float(np.min(obs)),
                'max': float(np.max(obs))
            },
            'pretty_schedule': pretty_schedule
        }
        
        self.current_episode['steps'].append(step_data)
        self.current_episode['schedule_actions'].append(schedule_action)
        self.current_episode['batch_actions'].append(batch_action)
        self.current_episode['observations'].append(obs.tolist())
        self.current_episode['rewards'].append(reward)
        
        if done:
            print("Is done")
            print(f"Episode {self.episode_count} terminated after {len(self.current_episode['steps'])} steps")
            self._process_completed_episode()
        return True

    # ADD THIS ENTIRE METHOD after _on_step and before _create_pretty_schedule
    def _process_completed_episode(self):
        """Process completed episode immediately when done=True"""
        if len(self.current_episode['steps']) > 0:
            rewards = self.current_episode['rewards']
            valid_rewards = [r for r in rewards if not np.isnan(r)]
            
            episode_summary = {
                'episode': self.episode_count,
                'timestamp': datetime.now().isoformat(),
                'total_reward': sum(valid_rewards),
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
                'steps': [], 'schedule_actions': [], 'batch_actions': [], 
                'observations': [], 'rewards': []
            }
            self.episode_count += 1


    def _create_pretty_schedule(self, schedule_action, batch_action, observation):
        """Create a pretty-printed schedule similar to C++ Node::pretty_print()"""
        pretty_lines = []
        
        # Extract observation data based on environment 
        obs_idx = 0
        model_data = []
        
        for model_id in range(self.num_models):
            model_info = {
                'request_rate': observation[obs_idx] * 200.0,  # Denormalize
                'slo_latency': observation[obs_idx + 1] * 5000.0,  # Denormalize
                'slo_satisfaction': [],
                'gpu_locations': [],
                'batch_sizes': []
            }
            
            # SLO satisfaction rates (3 values per model for num_gpus)
            offset = 2
            for j in range(self.num_gpus):
                slo_sat = observation[obs_idx + offset + j] * 100.0  # Denormalize to percentage
                model_info['slo_satisfaction'].append(slo_sat)
            
            # GPU locations (num_gpus values)
            offset = 2 + self.num_gpus
            for j in range(self.num_gpus):
                gpu_loc = int(observation[obs_idx + offset + j])
                model_info['gpu_locations'].append(gpu_loc)
            
            # Current batch sizes (num_gpus values) - these are the current batch sizes
            offset = 2 + 2 * self.num_gpus
            for j in range(self.num_gpus):
                current_batch = observation[obs_idx + offset + j] * 1024.0  # Denormalize
                model_info['batch_sizes'].append(int(current_batch))
            
            model_data.append(model_info)
            obs_idx += self.model_feature_dim  # Move to next model (2 + 3 * num_gpus)
        
        # Create schedule for each GPU
        for gpu_id in range(self.num_gpus):
            pretty_lines.append("****************************************************************************")
            pretty_lines.append(f"    GPU {gpu_id}: A6000  |  GPU MEMORY: 48GB | STEP: {len(self.current_episode['steps'])}")
            pretty_lines.append("    Session List: {model_name, SLO, request_rate, current_batch, batch_delta, new_batch, execution_mode}")
            
            # Process each slot for this GPU
            for slot_id in range(self.scheduler_slots):
                slot_idx = (gpu_id * self.scheduler_slots + slot_id) * 2
                model_id = schedule_action[slot_idx]
                in_parallel = schedule_action[slot_idx + 1]
                
                if model_id < self.num_models:  # Valid model
                    # Get batch delta for this model on this GPU
                    batch_idx = gpu_id * self.num_models + model_id
                    batch_delta = batch_action[batch_idx] - 4  # Apply _get_batch_delta conversion
                    
                    # Get current batch size from observation for this model on this GPU
                    current_batch_size = model_data[model_id]['batch_sizes'][gpu_id]
                    
                    # Calculate new batch size after applying delta
                    new_batch_size = max(1, current_batch_size + batch_delta)  # Ensure positive
                    
                    model_name = self.model_names[model_id]
                    request_rate = model_data[model_id]['request_rate']
                    slo_latency = model_data[model_id]['slo_latency']
                    execution_mode = "parallel" if in_parallel else "sequential"
                    
                    pretty_lines.append(f"      Slot {slot_id}: {model_name}, {slo_latency:.1f}ms, {request_rate:.2f}req/s, "
                                    f"current_batch={current_batch_size}, batch_delta={batch_delta:+d}, "
                                    f"new_batch={new_batch_size}, {execution_mode}")
                else:
                    pretty_lines.append(f"      Slot {slot_id}: EMPTY")
            
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

    def _on_rollout_end(self) -> None:
        if len(self.current_episode['steps']) > 0:
            print(f"Processing incomplete episode with {len(self.current_episode['steps'])} steps at rollout end")
            self._process_completed_episode()
        
    def _calculate_schedule_diversity(self):
        if not self.current_episode['schedule_actions']:
            return 0.0
        
        unique_schedules = len(set(map(tuple, self.current_episode['schedule_actions'])))
        total_schedules = len(self.current_episode['schedule_actions'])
        return unique_schedules / total_schedules if total_schedules > 0 else 0.0


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
        #batch_size=5,
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
        save_freq=1800/8,                     # Save every 1000 environment steps
        save_path="./checkpoints/",         # Directory to save checkpoints
        name_prefix="ppo_scheduler_model"   # Prefix for checkpoint files
    )

    print("Starting training...")
    model.learn(total_timesteps=14400/8, callback=[callback, checkpoint_callback])

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
