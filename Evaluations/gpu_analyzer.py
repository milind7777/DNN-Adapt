import re
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict

class GPUUtilizationAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.systems = ['nexus', 'batch8', 'dnn']
        self.gpu_pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] \[Executor\] \[debug\] GPU (\d+)')
        self.slot_pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] \[Executor\] \[debug\] (\d+)\. (\w+), (\d+)')
        
    def parse_log_file(self, system_name):
        """Parse GPU utilization data from log file"""
        log_path = os.path.join(self.base_path, system_name, 'cpp_log', 'dnn_adapt.log')
        
        if not os.path.exists(log_path):
            print(f"Warning: Log file not found for {system_name}: {log_path}")
            return pd.DataFrame()
        
        gpu_data = []
        current_timestamp = None
        current_gpu = None
        gpu_state = {}
        
        try:
            with open(log_path, 'r') as file:
                for line in file:
                    # Check for GPU header
                    gpu_match = self.gpu_pattern.search(line)
                    if gpu_match:
                        current_timestamp = gpu_match.group(1)
                        current_gpu = int(gpu_match.group(2))
                        continue
                    
                    # Check for slot information
                    slot_match = self.slot_pattern.search(line)
                    if slot_match and current_gpu is not None:
                        timestamp = slot_match.group(1)
                        slot_id = int(slot_match.group(2))
                        model_name = slot_match.group(3)
                        batch_size = int(slot_match.group(4))
                        
                        # Store the slot state
                        if current_timestamp not in gpu_state:
                            gpu_state[current_timestamp] = {}
                        if current_gpu not in gpu_state[current_timestamp]:
                            gpu_state[current_timestamp][current_gpu] = {}
                        
                        gpu_state[current_timestamp][current_gpu][slot_id] = {
                            'model': model_name,
                            'batch_size': batch_size
                        }
                        
                        # If we have collected all 3 slots for this GPU, process the state
                        if len(gpu_state[current_timestamp][current_gpu]) == 3:
                            gpu_data.append({
                                'timestamp': current_timestamp,
                                'system': system_name.upper(),
                                'gpu_id': current_gpu,
                                'slot_0_model': gpu_state[current_timestamp][current_gpu].get(0, {}).get('model', 'EMPTY'),
                                'slot_1_model': gpu_state[current_timestamp][current_gpu].get(1, {}).get('model', 'EMPTY'),
                                'slot_2_model': gpu_state[current_timestamp][current_gpu].get(2, {}).get('model', 'EMPTY'),
                                'slot_0_batch': gpu_state[current_timestamp][current_gpu].get(0, {}).get('batch_size', 0),
                                'slot_1_batch': gpu_state[current_timestamp][current_gpu].get(1, {}).get('batch_size', 0),
                                'slot_2_batch': gpu_state[current_timestamp][current_gpu].get(2, {}).get('batch_size', 0),
                            })
        
        except Exception as e:
            print(f"Error parsing log file for {system_name}: {e}")
            return pd.DataFrame()
        
        if gpu_data:
            df = pd.DataFrame(gpu_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        return pd.DataFrame()
    
    def load_all_gpu_data(self):
        """Load GPU data from all systems"""
        all_data = []
        for system in self.systems:
            data = self.parse_log_file(system)
            if not data.empty:
                all_data.append(data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return self.normalize_timestamps(combined_df)
        return pd.DataFrame()
    
    def normalize_timestamps(self, df):
        """Convert timestamps to seconds from start for each system"""
        if df.empty:
            return df
        
        normalized_data = []
        for system in df['system'].unique():
            system_data = df[df['system'] == system].copy()
            start_time = system_data['timestamp'].min()
            system_data['seconds'] = (system_data['timestamp'] - start_time).dt.total_seconds()
            normalized_data.append(system_data)
        
        return pd.concat(normalized_data, ignore_index=True)
    
    def calculate_gpu_utilization(self, df):
        """Calculate GPU utilization percentages"""
        if df.empty:
            return pd.DataFrame()
        
        utilization_data = []
        
        for system in df['system'].unique():
            system_data = df[df['system'] == system]
            
            # Get time intervals
            system_data = system_data.sort_values('seconds')
            time_intervals = []
            
            for i in range(len(system_data) - 1):
                current_time = system_data.iloc[i]['seconds']
                next_time = system_data.iloc[i + 1]['seconds']
                duration = next_time - current_time
                
                for gpu_id in [0, 1]:
                    gpu_data = system_data[system_data['gpu_id'] == gpu_id].iloc[i] if i < len(system_data[system_data['gpu_id'] == gpu_id]) else None
                    
                    if gpu_data is not None:
                        # Count occupied slots by model
                        models_in_slots = [
                            gpu_data['slot_0_model'],
                            gpu_data['slot_1_model'], 
                            gpu_data['slot_2_model']
                        ]
                        
                        # Calculate utilization for each model
                        model_counts = defaultdict(int)
                        total_slots = 3
                        occupied_slots = 0
                        
                        for model in models_in_slots:
                            if model != 'EMPTY':
                                model_counts[model] += 1
                                occupied_slots += 1
                        
                        utilization_data.append({
                            'system': system,
                            'gpu_id': gpu_id,
                            'time_start': current_time,
                            'duration': duration,
                            'total_utilization': (occupied_slots / total_slots) * 100,
                            'efficientnetb0_utilization': (model_counts['efficientnetb0'] / total_slots) * 100,
                            'resnet18_utilization': (model_counts['resnet18'] / total_slots) * 100,
                            'vit16_utilization': (model_counts['vit16'] / total_slots) * 100,
                        })
        
        return pd.DataFrame(utilization_data)
    
    def calculate_time_weighted_utilization(self, utilization_df):
        """Calculate time-weighted average utilization"""
        if utilization_df.empty:
            return pd.DataFrame()
        
        summary_data = []
        
        for system in utilization_df['system'].unique():
            system_data = utilization_df[utilization_df['system'] == system]
            
            for gpu_id in [0, 1]:
                gpu_data = system_data[system_data['gpu_id'] == gpu_id]
                
                if len(gpu_data) == 0:
                    continue
                
                total_time = gpu_data['duration'].sum()
                
                # Calculate weighted averages
                total_util = (gpu_data['total_utilization'] * gpu_data['duration']).sum() / total_time
                eff_util = (gpu_data['efficientnetb0_utilization'] * gpu_data['duration']).sum() / total_time
                res_util = (gpu_data['resnet18_utilization'] * gpu_data['duration']).sum() / total_time
                vit_util = (gpu_data['vit16_utilization'] * gpu_data['duration']).sum() / total_time
                
                summary_data.append({
                    'system': system,
                    'gpu_id': gpu_id,
                    'gpu_name': f'GPU {gpu_id}',
                    'total_utilization': total_util,
                    'efficientnetb0_utilization': eff_util,
                    'resnet18_utilization': res_util,
                    'vit16_utilization': vit_util,
                })
        
        return pd.DataFrame(summary_data)
