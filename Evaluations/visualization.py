import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class ViolationVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        # Updated color mapping to match actual system names
        self.colors = {'Nexus': '#1f77b4', 'Batch8': '#ff7f0e', 'DNN': '#2ca02c'}
        self.model_colors = {'efficientnetb0': '#d62728', 'resnet18': '#9467bd', 'vit16': '#8c564b'}
        self.gpu_colors = {'GPU 0': '#1f77b4', 'GPU 1': '#ff7f0e'}
        # Default colors for any missing systems
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def get_color(self, system_name, index=0):
        """Get color for a system, with fallback to default colors"""
        return self.colors.get(system_name, self.default_colors[index % len(self.default_colors)])
        
    def plot_total_violations_line(self, total_data, title="Total SLO Violations Over Time"):
        """Create line graph for total violations across all systems"""
        plt.figure(figsize=self.figsize)
        
        for system in total_data['system'].unique():
            system_data = total_data[total_data['system'] == system]
            plt.plot(system_data['seconds'], system_data['violation_count'], 
                    label=system, color=self.get_color(system), linewidth=2, marker='o', markersize=4)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Total Violation Count')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_total_violations_bar(self, total_data, title="Total SLO Violations - Bar Chart"):
        """Create bar chart for total violations across all systems"""
        # Calculate total violations per system
        system_totals = total_data.groupby('system')['violation_count'].sum().reset_index()
        
        plt.figure(figsize=(8, 6))
        # Use get_color method with enumeration for fallback
        colors = [self.get_color(sys, idx) for idx, sys in enumerate(system_totals['system'])]
        bars = plt.bar(system_totals['system'], system_totals['violation_count'], color=colors)
        
        plt.xlabel('System')
        plt.ylabel('Total Violation Count')
        plt.title(title)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_model_violations_line(self, model_data, title="SLO Violations by Model Over Time"):
        """Create line graph for violations per model across all systems"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        systems = sorted(model_data['system'].unique())
        
        for idx, system in enumerate(systems):
            ax = axes[idx]
            system_data = model_data[model_data['system'] == system]
            
            for model in system_data['model_name'].unique():
                model_system_data = system_data[system_data['model_name'] == model]
                ax.plot(model_system_data['seconds'], model_system_data['violation_count'],
                       label=model, color=self.model_colors.get(model, self.default_colors[0]), 
                       linewidth=2, marker='o', markersize=3)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Violation Count')
            ax.set_title(f'{system} - Model Violations')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_model_violations_bar(self, model_data, title="Total Violations by Model and System"):
        """Create grouped bar chart for violations per model across systems"""
        # Calculate total violations per model per system
        model_totals = model_data.groupby(['system', 'model_name'])['violation_count'].sum().reset_index()
        model_pivot = model_totals.pivot(index='model_name', columns='system', values='violation_count').fillna(0)
        
        plt.figure(figsize=self.figsize)
        # Use get_color method for each column
        colors = [self.get_color(col, idx) for idx, col in enumerate(model_pivot.columns)]
        model_pivot.plot(kind='bar', color=colors)
        
        plt.xlabel('Model')
        plt.ylabel('Total Violation Count')
        plt.title(title)
        plt.legend(title='System')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_violations_heatmap(self, model_data, title="Violations Heatmap - Model vs System"):
        """Create heatmap showing violations across models and systems"""
        model_totals = model_data.groupby(['system', 'model_name'])['violation_count'].sum().reset_index()
        model_pivot = model_totals.pivot(index='model_name', columns='system', values='violation_count').fillna(0)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(model_pivot, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Total Violations'})
        plt.xlabel('System')
        plt.ylabel('Model')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_gpu_utilization_over_time(self, utilization_df, title="GPU Utilization Over Time"):
        """Create line plot showing GPU utilization over time for each system"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        systems = sorted(utilization_df['system'].unique())
        
        for idx, system in enumerate(systems):
            ax = axes[idx]
            system_data = utilization_df[utilization_df['system'] == system]
            
            for gpu_id in [0, 1]:
                gpu_data = system_data[system_data['gpu_id'] == gpu_id].sort_values('time_start')
                if len(gpu_data) > 0:
                    ax.plot(gpu_data['time_start'], gpu_data['total_utilization'], 
                           label=f'GPU {gpu_id}', color=self.gpu_colors[f'GPU {gpu_id}'], 
                           linewidth=2.5, alpha=0.8)
            
            ax.set_ylabel('Utilization (%)')
            ax.set_title(f'{system} - GPU Utilization')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        axes[-1].set_xlabel('Time (seconds)')
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_model_gpu_utilization(self, utilization_df, title="Model GPU Utilization by System"):
        """Create stacked bar chart showing model utilization across GPUs"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        systems = sorted(utilization_df['system'].unique())
        models = ['efficientnetb0', 'resnet18', 'vit16']
        
        for idx, system in enumerate(systems):
            ax = axes[idx]
            system_data = utilization_df[utilization_df['system'] == system]
            
            gpu_names = []
            model_data = {model: [] for model in models}
            
            for gpu_id in [0, 1]:
                gpu_data = system_data[system_data['gpu_id'] == gpu_id]
                if len(gpu_data) > 0:
                    gpu_names.append(f'GPU {gpu_id}')
                    total_time = gpu_data['duration'].sum()
                    
                    for model in models:
                        util_col = f'{model}_utilization'
                        weighted_util = (gpu_data[util_col] * gpu_data['duration']).sum() / total_time if total_time > 0 else 0
                        model_data[model].append(weighted_util)
            
            if gpu_names:
                bottom = np.zeros(len(gpu_names))
                for model in models:
                    color = self.model_colors.get(model, self.default_colors[0])
                    ax.bar(gpu_names, model_data[model], bottom=bottom, 
                          label=model, color=color, alpha=0.8)
                    bottom += model_data[model]
            
            ax.set_ylabel('Utilization (%)')
            ax.set_title(f'{system}')
            ax.legend()
            ax.set_ylim(0, 100)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_gpu_utilization_summary(self, summary_df, title="Average GPU Utilization Summary"):
        """Create summary bar chart of GPU utilization across systems"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Total utilization by system and GPU
        systems = sorted(summary_df['system'].unique())
        gpu0_util = []
        gpu1_util = []
        
        for system in systems:
            system_data = summary_df[summary_df['system'] == system]
            gpu0_data = system_data[system_data['gpu_id'] == 0]
            gpu1_data = system_data[system_data['gpu_id'] == 1]
            
            gpu0_util.append(gpu0_data['total_utilization'].iloc[0] if len(gpu0_data) > 0 else 0)
            gpu1_util.append(gpu1_data['total_utilization'].iloc[0] if len(gpu1_data) > 0 else 0)
        
        x = np.arange(len(systems))
        width = 0.35
        
        ax1.bar(x - width/2, gpu0_util, width, label='GPU 0', color=self.gpu_colors.get('GPU 0', self.default_colors[0]))
        ax1.bar(x + width/2, gpu1_util, width, label='GPU 1', color=self.gpu_colors.get('GPU 1', self.default_colors[1]))
        
        ax1.set_xlabel('System')
        ax1.set_ylabel('Average Utilization (%)')
        ax1.set_title('Total GPU Utilization by System')
        ax1.set_xticks(x)
        ax1.set_xticklabels(systems)
        ax1.legend()
        ax1.set_ylim(0, 100)
        
        # Model distribution across all systems
        models = ['efficientnetb0', 'resnet18', 'vit16']
        model_utils = []
        
        for model in models:
            util_col = f'{model}_utilization'
            total_util = summary_df[util_col].mean()
            model_utils.append(total_util)
        
        colors = [self.model_colors.get(model, self.default_colors[0]) for model in models]
        ax2.bar(models, model_utils, color=colors)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Average Utilization (%)')
        ax2.set_title('Average Model Utilization Across All GPUs')
        ax2.tick_params(axis='x', rotation=45)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
