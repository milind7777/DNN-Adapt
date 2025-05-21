### IN Progress ###


import os
import sys
import grpc
import time
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime
from concurrent import futures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from com import gpu_scheduler_pb2
from com import gpu_scheduler_pb2_grpc


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchedulerRLServicer(gpu_scheduler_pb2_grpc.SchedulerRLServicer):
    def __init__(self):
        """Initialize the RL-based scheduler service"""
        # Constants for the RL model
        self.state_dim = 49  # 2 GPUs * 3 features + 10 sessions * 4 features + 3 global features
        self.action_dim = 100  # Simplified action space dimension
        
        # Initialize the RL agent define the RL agent
        #self.agent = none
        #
       
        # Set default model profiles
        self._init_model_profiles()
        
        # Try to load a saved model
        model_files = [f for f in os.listdir(self.agent.model_dir) 
                     if f.endswith('.pt')]
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getmtime(
                os.path.join(self.agent.model_dir, x)))
            self.agent.load_model(os.path.join(self.agent.model_dir, latest_model))
        
        # Periodically save the model
        self.update_count = 0
        self.save_frequency = 50
        
        logger.info("RL Scheduler Service initialized")
    
    def _init_model_profiles(self):
        """Initialize default model profiles"""
        # VIT16 latency profile (batch_size -> latency in ms)
       
    
    def GetSchedule(self, request, context):
        """Handle a request for a schedule"""
        logger.info(f"Received schedule request with {len(request.sessions)} sessions")
        
        try:
            # Get schedule from the RL agent
            response = self.agent.get_action(request)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating schedule: {str(e)}", exc_info=True)
            response = gpu_scheduler_pb2.ScheduleResponse()
            response.success = False
            response.error_message = f"Internal server error: {str(e)}"
            return response
    
    def UpdateExperience(self, request, context):
        """Handle an experience update"""
        logger.info(f"Received experience update with reward {request.reward}")
        
        try:
            # Process the experience update
            loss = self.agent.process_experience_update(request)
            
            # Periodic model saving
            self.update_count += 1
            if self.update_count % self.save_frequency == 0:
                self.agent.save_model()
            
            response = gpu_scheduler_pb2.UpdateResponse()
            response.success = True
            if loss is not None:
                response.message = f"Update successful, loss: {loss}"
            else:
                response.message = "Added to replay buffer"
            
            return response
            
        except Exception as e:
            logger.error(f"Error updating experience: {str(e)}", exc_info=True)
            response = gpu_scheduler_pb2.UpdateResponse()
            response.success = False
            response.message = f"Error updating experience: {str(e)}"
            return response

def serve():
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gpu_scheduler_pb2_grpc.add_SchedulerRLServicer_to_server(SchedulerRLServicer(), server)
    server_address = '[::]:50051'  # Bind to all addresses on port 50051
    server.add_insecure_port(server_address)
    server.start()
    logger.info(f"RL Scheduler server started on {server_address}")
    
    try:


        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        server.stop(0)
        logger.info("Server stopped")

if __name__ == '__main__':
    serve()