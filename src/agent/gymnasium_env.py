import scheduler_pb2
import scheduler_pb2_grpc

import grpc
import numpy as np
import gymnasium as gym
from gymnasium import spaces

NUM_MODELS = 3
NUM_GPUS = 2
NUM_SLOTS_PER_GPU = 3

class InferenceSchedulerEnv(gym.Env):
    def __init__(self, address="localhost:50051", num_models=NUM_MODELS, num_gpus=NUM_GPUS, scheduler_slots=NUM_SLOTS_PER_GPU):
        super().__init__()

        self.num_gpus = num_gpus
        self.num_models = num_models
        self.scheduler_slots = scheduler_slots

        self.channel = grpc.insecure_channel(address)
        self.stub = scheduler_pb2_grpc.SchedulerSimStub(self.channel)

        # # OBSERVATION SPACE # # 
        # Observation space and tracking for setup
        # For each model:
        #     ?   1. Request rate - float (normalized)
        #     ?   2. SLO latency in ms - float (normalized)
        #     ?   3. SLO latency satisfaction % - float (normalized)
        #     ?   4. GPU locations - array of [int] of size num_gpus
        #     ?   5. Batch size - array of [int] of size GPU

        # Model Space Vector Dimension = (3 + 2 * gpus) * (models) = 21
        self.model_feature_dim = 3 + 2 * self.num_gpus

        self.max_request_rate = 200.0
        self.max_slo_latency  = 5000.0
        self.max_slo_rate = 100.0
        self.max_batch_size = 1024.0

        # For each GPU:
        #     ?   1. Peak memory in MB per schedule - float
        #     ?   2. SLO satisfaction % per schedule - float
        #     ?   3. % GPU utilized per schedule - float (Hard to do skipping for now)

        # GPU Space Vector Dimension = (2) * (gpus) = 4
        self.gpu_feature_dim = 2

        self.max_memory_mb = 48 * 1024.0

        obs_dim = self.num_models * self.model_feature_dim + self.num_gpus * self.gpu_feature_dim
        self.obs_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # # Action Space # #
        # 1. Slot-level per gpu
        #   a. Model-id -> [0, num_models] -> last one represents empty slot
        #   b. In parallel -> 0 or 1, whether to run this model in parallel or sequential
        schedule_entry_spec = [
            self.num_models + 1,
            2
        ] * self.num_gpus * self.scheduler_slots

        # 2. Batch size per model per gpu
        #   a. Batch size for each model on that GPU -> array of [int] of size num_models
        batch_delta_spec = [
            10
        ] * self.num_models * self.num_gpus

        self.action_space = spaces.MultiDiscrete(schedule_entry_spec + batch_delta_spec)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        grpc_req = scheduler_pb2.ResetRequest()
        response = self.stub.Reset(grpc_req)

        raw_observation = np.array(response.observation, dtype=np.float32)
        observation = self._process_observation(raw_observation)
        info = None

        return observation, info

    def _get_batch_delta(self, delta: int) -> int:
        return delta - 4
    
    def step(self, action):
        schedule_fields = 2 * self.num_gpus * self.scheduler_slots
        schedule_entry_actions = action[:schedule_fields]
        batch_delta_actions    = action[schedule_fields:]

        grpc_req = scheduler_pb2.StepRequest()

        for i in range(0, len(schedule_fields), 2):
            entry = scheduler_pb2.ScheduleEntry()
            entry.model_id = action[i]
            entry.in_parallel = action[i+1]
            grpc_req.entries.append(entry)

        for i in range(0, len(batch_delta_actions)):
            grpc_req.batch_deltas.append(self._get_batch_delta(batch_delta_actions[i]))

        # take the step in the cpp scheduler system
        response = self.stub.Step(grpc_req)

        raw_observation = np.array(reponse.observation, dtype=np.float32)
        observation = self._process_observation(raw_observation)
        reward = response.reward
        terminated = response.done
        truncated = false
        info = {}

        return observation, reward, terminated, truncated, info
    
    def _process_observation(self, observation):
        for i in range(0, self.num_models, self.model_feature_dim):
            observation[i]   = observation[i] / self.max_request_rate
            observation[i+1] = observation[i+1] / self.max_slo_latency
            observation[i+2] = observation[i+2] / self.max_slo_rate
            
            offset = 3 + self.num_gpus
            for j in range(0, self.num_gpus):
                observation[i+offset+j] = observation[i+offset+j] / self.max_batch_size

        offset = self.model_feature_dim * self.num_models
        for i in range(offset, len(observation), 2):
            observation[i]   = observation[i] / self.max_memory_mb
            observation[i+1] = observation[i+1] / self.max_slo_rate

        return observation 

    def grpc_close(self):
        self.channel.close()
        


