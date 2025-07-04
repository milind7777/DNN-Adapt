import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ViolationVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = {'Nexus': '#1f77b4', 'Batch8': '#ff7f0e', 'DNN': '#2ca02c'}
        self.model_colors = {'efficientnetb0': '#d62728', 'resnet18': '#9467bd', 'vit16': '#8c564b'}
        
    def plot_total_violations_line(self, total_data, title="Total SLO Violations Over Time"):
        """Create line graph for total violations across all systems"""
        plt.figure(figsize=self.figsize)
        
        for system in total_data['system'].unique():
            system_data = total_data[total_data['system'] == system]
            plt.plot(system_data['seconds'], system_data['violation_count'], 
                    label=system, color=self.colors.get(system), linewidth=2, marker='o', markersize=4)
        
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
        bars = plt.bar(system_totals['system'], system_totals['violation_count'], 
                      color=[self.colors.get(sys) for sys in system_totals['system']])
        
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
                       label=model, color=self.model_colors.get(model), linewidth=2, marker='o', markersize=3)
            
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
        model_pivot.plot(kind='bar', color=[self.colors.get(col) for col in model_pivot.columns])
        
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
