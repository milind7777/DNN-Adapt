import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional

class ViolationPlotter:
    """A modular class for plotting violation data from multiple systems."""
    
    def __init__(self, base_path: str = "/Users/sai/Downloads/MS-Project-Results/Evaluations"):
        self.base_path = Path(base_path)
        self.systems = {
            'Nexus': self.base_path / 'Nexus' / 'csv-data-nexus' / 'second_violations.csv',
            'Batch8': self.base_path / 'Batch8' / 'csv-data-batch8' / 'second_violations.csv',
            'DNN': self.base_path / 'DNN' / 'csv-data-dnn' / 'second_violations.csv'
        }
        self.data = {}
        
    def load_data(self, time_col: str = 'time', violations_col: str = 'violations', 
                  model_col: Optional[str] = 'model') -> None:
        """Load CSV data from all systems."""
        for system_name, file_path in self.systems.items():
            try:
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    df['system'] = system_name
                    self.data[system_name] = df
                    print(f"✓ Loaded data for {system_name}: {len(df)} records")
                else:
                    print(f"✗ File not found: {file_path}")
            except Exception as e:
                print(f"✗ Error loading {system_name}: {e}")
    
    def plot_violations_per_second(self, time_col: str = 'time', 
                                 violations_col: str = 'violations',
                                 plot_type: str = 'line') -> None:
        """Plot total violations per second for all systems."""
        plt.figure(figsize=(12, 6))
        
        for system_name, df in self.data.items():
            if plot_type == 'line':
                plt.plot(df[time_col], df[violations_col], label=system_name, marker='o')
            elif plot_type == 'bar':
                plt.bar(df[time_col], df[violations_col], label=system_name, alpha=0.7)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Violations Count')
        plt.title(f'Violations per Second - All Systems ({plot_type.capitalize()} Plot)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_violations_per_model(self, model_col: str = 'model', 
                                violations_col: str = 'violations',
                                plot_type: str = 'bar') -> None:
        """Plot total violations per model for all systems."""
        plt.figure(figsize=(14, 6))
        
        # Aggregate data by model and system
        aggregated_data = []
        for system_name, df in self.data.items():
            if model_col in df.columns:
                model_violations = df.groupby(model_col)[violations_col].sum().reset_index()
                model_violations['system'] = system_name
                aggregated_data.append(model_violations)
        
        if aggregated_data:
            combined_df = pd.concat(aggregated_data, ignore_index=True)
            
            if plot_type == 'bar':
                # Create grouped bar chart
                models = combined_df[model_col].unique()
                systems = combined_df['system'].unique()
                x = np.arange(len(models))
                width = 0.25
                
                for i, system in enumerate(systems):
                    system_data = combined_df[combined_df['system'] == system]
                    violations = [system_data[system_data[model_col] == model][violations_col].iloc[0] 
                                if len(system_data[system_data[model_col] == model]) > 0 else 0 
                                for model in models]
                    plt.bar(x + i * width, violations, width, label=system)
                
                plt.xlabel('Models')
                plt.ylabel('Total Violations')
                plt.title(f'Total Violations per Model - All Systems ({plot_type.capitalize()} Plot)')
                plt.xticks(x + width, models, rotation=45)
                plt.legend()
                
            elif plot_type == 'line':
                for system in combined_df['system'].unique():
                    system_data = combined_df[combined_df['system'] == system]
                    plt.plot(system_data[model_col], system_data[violations_col], 
                           marker='o', label=system)
                plt.xlabel('Models')
                plt.ylabel('Total Violations')
                plt.title(f'Total Violations per Model - All Systems ({plot_type.capitalize()} Plot)')
                plt.xticks(rotation=45)
                plt.legend()
        else:
            print("No model data found in the CSV files.")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_comparison_dashboard(self, time_col: str = 'time', 
                                  violations_col: str = 'violations',
                                  model_col: Optional[str] = 'model') -> None:
        """Create a comprehensive dashboard with multiple plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Line plot - Violations per second
        for system_name, df in self.data.items():
            axes[0, 0].plot(df[time_col], df[violations_col], label=system_name, marker='o')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Violations Count')
        axes[0, 0].set_title('Violations per Second (Line Plot)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Bar plot - Violations per second (sample)
        if self.data:
            sample_size = min(20, min(len(df) for df in self.data.values()))
            for system_name, df in self.data.items():
                sample_df = df.head(sample_size)
                axes[0, 1].bar(sample_df[time_col], sample_df[violations_col], 
                             alpha=0.7, label=system_name)
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Violations Count')
        axes[0, 1].set_title('Violations per Second (Bar Plot - Sample)')
        axes[0, 1].legend()
        
        # Plot 3: Model violations (if available)
        if model_col and any(model_col in df.columns for df in self.data.values()):
            aggregated_data = []
            for system_name, df in self.data.items():
                if model_col in df.columns:
                    model_violations = df.groupby(model_col)[violations_col].sum().reset_index()
                    model_violations['system'] = system_name
                    aggregated_data.append(model_violations)
            
            if aggregated_data:
                combined_df = pd.concat(aggregated_data, ignore_index=True)
                models = combined_df[model_col].unique()
                systems = combined_df['system'].unique()
                x = np.arange(len(models))
                width = 0.25
                
                for i, system in enumerate(systems):
                    system_data = combined_df[combined_df['system'] == system]
                    violations = [system_data[system_data[model_col] == model][violations_col].iloc[0] 
                                if len(system_data[system_data[model_col] == model]) > 0 else 0 
                                for model in models]
                    axes[1, 0].bar(x + i * width, violations, width, label=system)
                
                axes[1, 0].set_xlabel('Models')
                axes[1, 0].set_ylabel('Total Violations')
                axes[1, 0].set_title('Total Violations per Model')
                axes[1, 0].set_xticks(x + width)
                axes[1, 0].set_xticklabels(models, rotation=45)
                axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'Model data not available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Model Violations (No Data)')
        
        # Plot 4: Summary statistics
        axes[1, 1].axis('off')
        summary_text = "System Statistics:\n\n"
        for system_name, df in self.data.items():
            total_violations = df[violations_col].sum()
            avg_violations = df[violations_col].mean()
            max_violations = df[violations_col].max()
            summary_text += f"{system_name}:\n"
            summary_text += f"  Total: {total_violations}\n"
            summary_text += f"  Average: {avg_violations:.2f}\n"
            summary_text += f"  Max: {max_violations}\n\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.show()
    
    def get_summary_statistics(self, violations_col: str = 'violations') -> Dict:
        """Get summary statistics for all systems."""
        stats = {}
        for system_name, df in self.data.items():
            stats[system_name] = {
                'total_violations': df[violations_col].sum(),
                'average_violations': df[violations_col].mean(),
                'max_violations': df[violations_col].max(),
                'min_violations': df[violations_col].min(),
                'std_violations': df[violations_col].std(),
                'record_count': len(df)
            }
        return stats

def main():
    """Main function to demonstrate the ViolationPlotter usage."""
    # Initialize the plotter
    plotter = ViolationPlotter()
    
    # Load data (you may need to adjust column names based on your CSV structure)
    print("Loading violation data...")
    plotter.load_data(time_col='time', violations_col='violations', model_col='model')
    
    if not plotter.data:
        print("No data loaded. Please check file paths and CSV structure.")
        return
    
    # Create individual plots
    print("\nCreating violations per second plots...")
    plotter.plot_violations_per_second(plot_type='line')
    plotter.plot_violations_per_second(plot_type='bar')
    
    print("\nCreating violations per model plots...")
    plotter.plot_violations_per_model(plot_type='bar')
    plotter.plot_violations_per_model(plot_type='line')
    
    # Create comprehensive dashboard
    print("\nCreating comparison dashboard...")
    plotter.create_comparison_dashboard()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    stats = plotter.get_summary_statistics()
    for system, system_stats in stats.items():
        print(f"\n{system}:")
        for metric, value in system_stats.items():
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()