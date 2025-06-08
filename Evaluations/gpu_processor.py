import pandas as pd
import os
import re
from datetime import datetime

class GPUDataProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.systems = ['Nexus', 'Batch8', 'DNN']
        
    def parse_gpu_log(self, system_name, log_filename='dnn_adapt.log'):
        """Parse GPU occupation data from log file"""
        log_path = os.path.join(self.base_path, system_name.lower(), 'cpp_log', log_filename)
        
        try:
            with open(log_path, 'r') as file:
                log_content = file.read()
        except FileNotFoundError:
            print(f"Warning: Log file not found for {system_name}: {log_path}")
            return pd.DataFrame()
        
        gpu_data = []
        lines = log_content.split('\n')
        
        current_timestamp = None
        current_gpu = None
        gpu_state = {}
        
        for line in lines:
            # Extract timestamp
            timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]', line)
            if timestamp_match:
                current_timestamp = timestamp_match.group(1)
            
            # Check for GPU debug lines
            if '[Executor] [debug] GPU' in line:
                gpu_match = re.search(r'GPU (\d+)', line)
                if gpu_match:
                    current_gpu = int(gpu_match.group(1))
                    gpu_state[current_gpu] = {}
            
            # Check for slot occupation lines
            if '[Executor] [debug]' in line and current_gpu is not None and current_timestamp:
                slot_match = re.search(r'(\d+)\.\s+(\w+),\s+(\d+)', line)
                if slot_match:
                    slot_id = int(slot_match.group(1))
                    model_name = slot_match.group(2)
                    batch_size = int(slot_match.group(3))
                    
                    gpu_state[current_gpu][slot_id] = {
                        'model': model_name if model_name != 'EMPTY' else None,
                        'batch_size': batch_size
                    }
                    
                    # If we have complete state for both GPUs, record it
                    if len(gpu_state) == 2 and all(len(slots) == 3 for slots in gpu_state.values()):
                        try:
                            timestamp_dt = datetime.strptime(current_timestamp, '%Y-%m-%d %H:%M:%S.%f')
                            
                            for gpu_id in [0, 1]:
                                for slot_id in [0, 1, 2]:
                                    slot_info = gpu_state[gpu_id].get(slot_id, {'model': None, 'batch_size': 0})
                                    gpu_data.append({
                                        'timestamp': timestamp_dt,
                                        'system': system_name,
                                        'gpu_id': gpu_id,
                                        'slot_id': slot_id,
                                        'model': slot_info['model'],
                                        'batch_size': slot_info['batch_size'],
                                        'occupied': slot_info['model'] is not None
                                    })
                        except ValueError:
                            continue
                        
                        # Reset for next reading
                        gpu_state = {}
                        current_gpu = None
        
        if gpu_data:
            df = pd.DataFrame(gpu_data)
            return self.normalize_gpu_timestamps(df)
        return pd.DataFrame()
    
    def normalize_gpu_timestamps(self, df):
        """Convert timestamps to seconds from start"""
        if df.empty:
            return df
        
        start_time = df['timestamp'].min()
        df['seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
        return df
    
    def load_all_gpu_data(self):
        """Load GPU data from all systems"""
        all_gpu_data = []
        for system in self.systems:
            data = self.parse_gpu_log(system)
            if not data.empty:
                all_gpu_data.append(data)
        
        if all_gpu_data:
            return pd.concat(all_gpu_data, ignore_index=True)
        return pd.DataFrame()
    
    def calculate_gpu_utilization(self, df, time_window=1.0):
        """Calculate GPU utilization percentage per model per system"""
        if df.empty:
            return pd.DataFrame()
        
        utilization_data = []
        
        for system in df['system'].unique():
            system_data = df[df['system'] == system]
            max_time = system_data['seconds'].max()
            
            # Create time bins
            time_bins = pd.cut(system_data['seconds'], 
                             bins=int(max_time / time_window) + 1, 
                             labels=False)
            system_data = system_data.copy()
            system_data['time_bin'] = time_bins
            
            for time_bin in system_data['time_bin'].dropna().unique():
                bin_data = system_data[system_data['time_bin'] == time_bin]
                bin_start_time = time_bin * time_window
                
                for gpu_id in [0, 1]:
                    gpu_data = bin_data[bin_data['gpu_id'] == gpu_id]
                    total_slots = len(gpu_data)
                    
                    if total_slots > 0:
                        # Calculate utilization per model
                        model_counts = gpu_data[gpu_data['occupied']]['model'].value_counts()
                        
                        for model, count in model_counts.items():
                            utilization_data.append({
                                'system': system,
                                'seconds': bin_start_time,
                                'gpu_id': gpu_id,
                                'model': model,
                                'utilization_percent': (count / total_slots) * 100,
                                'slot_count': count
                            })
                        
                        # Add empty slots
                        empty_count = total_slots - gpu_data['occupied'].sum()
                        if empty_count > 0:
                            utilization_data.append({
                                'system': system,
                                'seconds': bin_start_time,
                                'gpu_id': gpu_id,
                                'model': 'EMPTY',
                                'utilization_percent': (empty_count / total_slots) * 100,
                                'slot_count': empty_count
                            })
        
        return pd.DataFrame(utilization_data)
