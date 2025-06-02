import agent_scheduler_pb2
import agent_scheduler_pb2_grpc

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
        self.last_info = {}

        self.channel = grpc.insecure_channel(address)
        self.stub = agent_scheduler_pb2_grpc.SchedulerSimStub(self.channel)

        # # OBSERVATION SPACE # # 
        # Observation space and tracking for setup
        # For each model:
        #     ?   1. Request rate - float (normalized)
        #     ?   2. SLO latency in ms - float (normalized)
        #     ?   3. SLO latency satisfaction % - array of [float] of size num_gpus (normalized)
        #     ?   4. GPU locations - array of [int] of size num_gpus
        #     ?   5. Batch size - array of [int] of size GPU

        # # OBSERVATION SPACE (PER SLOT) # # 
        # Observation space and tracking for setup
        # For each model:
        #     ?   1. Request rate - float (normalized)
        #     ?   2. Queue size   - float (normalized)
        #     ?   3. SLO latency in ms - float (normalized)
        #     ?   3. SLO latency satisfaction % (normalized)

        # For each slot:
        #     ?   1. Model id deployed - one hot encoding
        #     ?   2. Batch size - float (normalized)
        #     ?   3. In parallel - bool

        # Model Space Vector Dimension = (4 * models) + (slots * gpus * (models + 1 + 2)) = 12 + 36 = 48
        self.model_feature_dim = 4 * self.num_models + (self.scheduler_slots * self.num_gpus * (self.num_models + 1 + 2))

        self.max_request_rate = 100.0
        self.max_queue_size = 500.0
        self.max_slo_latency  = 2000.0
        self.max_slo_rate = 100.0
        self.max_batch_size = 512.0

        # For each GPU:
        #     ?   1. Peak memory in MB per schedule - float
        #     ?   2. % GPU utilized per schedule - float (Hard to do skipping for now)

        # GPU Space Vector Dimension = (1) * (gpus) = 4
        # self.gpu_feature_dim = 1

        # self.max_memory_mb = 48 * 1024.0

        # obs_dim = self.num_models * self.model_feature_dim + self.num_gpus * self.gpu_feature_dim
        obs_dim = self.model_feature_dim
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # # # Action Space # #
        # # 1. Slot-level per gpu
        # #   a. Model-id -> [0, num_models] -> last one represents empty slot
        # #   b. In parallel -> 0 or 1, whether to run this model in parallel or sequential
        # schedule_entry_spec = [
        #     self.num_models + 1,
        #     2
        # ] * self.num_gpus * self.scheduler_slots

        # # 2. Batch size per model per gpu
        # #   a. Batch size for each model on that GPU -> array of [int] of size num_models
        # batch_delta_spec = [
        #     10
        # ] * self.num_models * self.num_gpus

        # # Action Space (REDUCED)
        #   1. Slot id - [0, gpus * slots] (including no-op at the end)
        #   2. Model id - [0, models] (including empty model at the end)
        #   3. Batch delta - [-4, +5]  
        #   4. In paralle - 0 or 1, whether to run this model in parallel or sequential
        
        schedule_entry_spec = [
            self.num_gpus * self.scheduler_slots + 1,
            self.num_models + 1,
            10,
            2
        ]

        self.action_space = spaces.MultiDiscrete(schedule_entry_spec)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        grpc_req = agent_scheduler_pb2.ResetRequest()
        grpc_req.seed = 1
        response = self.stub.Reset(grpc_req)

        raw_observation = np.array(response.observation, dtype=np.float32)
        observation = self._process_observation_per_slot(raw_observation)
        info = None

        return observation, info

    def _get_batch_delta(self, delta: int) -> int:
        return delta - 4
    
    def step(self, action):
        # schedule_fields = 2 * self.num_gpus * self.scheduler_slots
        # schedule_entry_actions = action[:schedule_fields]
        # batch_delta_actions    = action[schedule_fields:]

        grpc_req = agent_scheduler_pb2.StepRequestReduced()
        
        entry = agent_scheduler_pb2.SlotEntry()
        entry.slot_id = action[0]
        entry.model_id = action[1]
        entry.batch_delta = self._get_batch_delta(action[2])
        entry.in_parallel = action[3]

        grpc_req.slot_entry = entry

        # for i in range(0, schedule_fields, 2):
        #     entry = agent_scheduler_pb2.ScheduleEntry()
        #     entry.model_id = action[i]
        #     entry.in_parallel = action[i+1]
        #     grpc_req.entries.append(entry)

        # for i in range(0, len(batch_delta_actions)):
        #     grpc_req.batch_deltas.append(self._get_batch_delta(batch_delta_actions[i]))

        # take the step in the cpp scheduler system
        response = self.stub.Step(grpc_req)

        raw_observation = np.array(response.observation, dtype=np.float32)
        observation = self._process_observation_per_slot(raw_observation)
        reward = response.reward
        terminated = response.done
        truncated = False
        info = {}

        # track contribution from each aspect of the reward
        self.last_info = np.array(response.info, dtype=np.float32)

        return observation, reward, terminated, truncated, info
    
    def _process_observation(self, observation):
        for i in range(0, self.num_models * self.model_feature_dim, self.model_feature_dim):
            # 1. Request rate - float (normalized)
            observation[i]   = observation[i] / self.max_request_rate

            # 2. SLO latency in ms - float (normalized)
            observation[i+1] = observation[i+1] / self.max_slo_latency

            # 3. SLO latency satisfaction % - array of [float] of size num_gpus (normalized)
            offset = 2
            for j in range(0, self.num_gpus):
                observation[i+offset+j] = observation[i+offset+j] / self.max_slo_rate
            
            # 4. GPU locations - array of [int] of size num_gpus
            # No normalization required

            # 5. Batch size - array of [int] of size 
            offset = 2 + 2 * self.num_gpus
            for j in range(0, self.num_gpus):
                observation[i+offset+j] = observation[i+offset+j] / self.max_batch_size

        offset = self.model_feature_dim * self.num_models
        for i in range(offset, len(observation)):
            # 1. Peak memory in MB per schedule - float
            observation[i]   = observation[i] / self.max_memory_mb

            # 2. % GPU utilized per schedule - float (Hard to do skipping for now)
            # This feature is currently skipped

        return observation 

    def _process_observation_per_slot(self, observation):
        for i in range(0, 4 * self.num_models, 4):
            #  1. Request rate - float (normalized)
            observation[i] /= self.max_request_rate
            
            #  2. Queue size   - float (normalized)
            observation[i+1] /= self.max_queue_size

            #  3. SLO latency in ms - float (normalized)
            observation[i+2] /= self.max_slo_latency

            #  4. SLO latency satisfaction % (normalized)
            observation[i+3] /= self.max_slo_rate
        
        for i in range(4 * self.num_models, self.model_feature_dim, self.num_models + 1 + 2):
            #  1. Model id deployed - one hot encoding
            #  2. Batch size - float (normalized)
            observation[i+self.num_models+1] /= self.max_batch_size
            #  3. In parallel - bool

        return observation


    def grpc_close(self):
        self.channel.close()
        


