import pandas as pd
import os
import re
from datetime import datetime

class GPULogProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.systems = ['Nexus', 'Batch8', 'DNN']
        
    def parse_gpu_occupation_from_log(self, system_name, log_filename='dnn_adapt.log'):
        """Parse GPU occupation data from log file"""
        log_path = os.path.join(self.base_path, system_name.lower(), 'cpp_log', log_filename)
        
        try:
            with open(log_path, 'r') as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Warning: Log file not found for {system_name}: {log_path}")
            return pd.DataFrame()
        
        # Extract GPU occupation patterns
        gpu_data = []
        
        # Pattern to match GPU debug lines
        gpu_pattern = r'\[([\d-]+\s[\d:\.]+)\].*\[Executor\].*GPU (\d+)'
        slot_pattern = r'\[([\d-]+\s[\d:\.]+)\].*\[Executor\].*(\d+)\.\s+(\w+),\s+(\d+)'
        
        lines = content.split('\n')
        current_timestamp = None
        current_gpu = None
        
        for i, line in enumerate(lines):
            # Check for GPU line
            gpu_match = re.search(gpu_pattern, line)
            if gpu_match:
                current_timestamp = gpu_match.group(1)
                current_gpu = int(gpu_match.group(2))
                continue
            
            # Check for slot lines (next 3 lines after GPU line)
            slot_match = re.search(slot_pattern, line)
            if slot_match and current_timestamp and current_gpu is not None:
                timestamp = slot_match.group(1)
                slot_id = int(slot_match.group(2))
                model_name = slot_match.group(3)
                batch_size = int(slot_match.group(4))
                
                gpu_data.append({
                    'timestamp': timestamp,
                    'system': system_name,
                    'gpu_id': current_gpu,
                    'slot_id': slot_id,
                    'model_name': model_name,
                    'batch_size': batch_size
                })
        
        if gpu_data:
            df = pd.DataFrame(gpu_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        return pd.DataFrame()
    
    def load_all_gpu_data(self):
        """Load GPU occupation data from all systems"""
        all_gpu_data = []
        
        for system in self.systems:
            gpu_data = self.parse_gpu_occupation_from_log(system)
            if not gpu_data.empty:
                all_gpu_data.append(gpu_data)
        
        if all_gpu_data:
            combined_df = pd.concat(all_gpu_data, ignore_index=True)
            return self.normalize_gpu_timestamps(combined_df)
        
        return pd.DataFrame()
    
    def normalize_gpu_timestamps(self, df):
        """Convert timestamps to seconds from start for each system"""
        if df.empty:
            return df
        
        normalized_data = []
        for system in df['system'].unique():
            system_data = df[df['system'] == system].copy()
            if not system_data.empty:
                start_time = system_data['timestamp'].min()
                system_data['seconds'] = (system_data['timestamp'] - start_time).dt.total_seconds()
                normalized_data.append(system_data)
        
        if normalized_data:
            return pd.concat(normalized_data, ignore_index=True)
        return pd.DataFrame()
    
    def calculate_gpu_utilization(self, df):
        """Calculate GPU utilization percentages per second"""
        if df.empty:
            return pd.DataFrame()
        
        # For each timestamp, calculate what models are occupying each GPU
        utilization_data = []
        
        for system in df['system'].unique():
            system_data = df[df['system'] == system]
            
            # Group by seconds to get utilization per second
            for seconds, group in system_data.groupby('seconds'):
                gpu_util = {}
                
                for gpu_id in [0, 1]:
                    gpu_data = group[group['gpu_id'] == gpu_id]
                    
                    # Count occupied slots per model
                    model_counts = {}
                    total_slots = 3  # 3 slots per GPU
                    
                    for _, row in gpu_data.iterrows():
                        if row['model_name'] != 'EMPTY':
                            model_name = row['model_name']
                            model_counts[model_name] = model_counts.get(model_name, 0) + 1
                    
                    # Calculate utilization percentage for each model
                    for model_name, slot_count in model_counts.items():
                        utilization_data.append({
                            'system': system,
                            'seconds': seconds,
                            'gpu_id': gpu_id,
                            'model_name': model_name,
                            'slots_occupied': slot_count,
                            'utilization_percent': (slot_count / total_slots) * 100
                        })
                    
                    # Add empty slots
                    empty_slots = total_slots - sum(model_counts.values())
                    if empty_slots > 0:
                        utilization_data.append({
                            'system': system,
                            'seconds': seconds,
                            'gpu_id': gpu_id,
                            'model_name': 'EMPTY',
                            'slots_occupied': empty_slots,
                            'utilization_percent': (empty_slots / total_slots) * 100
                        })
        
        return pd.DataFrame(utilization_data)
