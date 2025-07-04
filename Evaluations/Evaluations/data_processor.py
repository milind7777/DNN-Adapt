import pandas as pd
import os
from datetime import datetime, timedelta

class ViolationDataProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.systems = ['Nexus', 'Batch8', 'DNN']
        self.models = ['efficientnetb0', 'resnet18', 'vit16']
        
    def load_system_data(self, system_name, filename='second_violations.csv'):
        """Load data for a specific system"""
        file_path = os.path.join(self.base_path, system_name, f'csv-data-{system_name.lower()}', filename)
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['system'] = system_name
            return df
        except FileNotFoundError:
            print(f"Warning: File not found for {system_name}: {file_path}")
            return pd.DataFrame()
    
    def load_all_data(self):
        """Load data from all systems"""
        all_data = []
        for system in self.systems:
            data = self.load_system_data(system)
            if not data.empty:
                all_data.append(data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return self.normalize_timestamps(combined_df)
        return pd.DataFrame()
    
    def normalize_timestamps(self, df):
        """Convert timestamps to seconds from start"""
        if df.empty:
            return df
        
        # Group by system and normalize timestamps within each system
        normalized_data = []
        for system in df['system'].unique():
            system_data = df[df['system'] == system].copy()
            start_time = system_data['timestamp'].min()
            system_data['seconds'] = (system_data['timestamp'] - start_time).dt.total_seconds()
            normalized_data.append(system_data)
        
        return pd.concat(normalized_data, ignore_index=True)
    
    def get_total_violations_per_second(self, df):
        """Calculate total violations per second for each system"""
        return df.groupby(['system', 'seconds'])['violation_count'].sum().reset_index()
    
    def get_model_violations_per_second(self, df):
        """Get violations per model per second for each system"""
        return df.groupby(['system', 'model_name', 'seconds'])['violation_count'].sum().reset_index()
