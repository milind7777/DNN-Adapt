import os
import matplotlib.pyplot as plt
from gpu_analyzer import GPUUtilizationAnalyzer
from visualization import ViolationVisualizer

def main():
    # Configuration
    base_path = "/Users/sai/Downloads/MS-Project-Results/results-ramp-constant"
    output_dir = os.path.join(base_path, "generated_graphs")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer and visualizer
    analyzer = GPUUtilizationAnalyzer(base_path)
    visualizer = ViolationVisualizer()
    
    print("Loading GPU utilization data from all systems...")
    gpu_data = analyzer.load_all_gpu_data()
    
    if gpu_data.empty:
        print("No GPU data found. Please check log files and paths.")
        return
    
    print(f"Loaded GPU data for {len(gpu_data)} records across {gpu_data['system'].nunique()} systems")
    print(f"Time range: {gpu_data['seconds'].min():.0f} - {gpu_data['seconds'].max():.0f} seconds")
    
    # Calculate utilization metrics
    print("Calculating GPU utilization metrics...")
    utilization_df = analyzer.calculate_gpu_utilization(gpu_data)
    summary_df = analyzer.calculate_time_weighted_utilization(utilization_df)
    
    # Generate GPU utilization graphs
    print("Generating GPU utilization over time graph...")
    fig1 = visualizer.plot_gpu_utilization_over_time(utilization_df)
    fig1.savefig(os.path.join(output_dir, "gpu_utilization_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("Generating model GPU utilization graph...")
    fig2 = visualizer.plot_model_gpu_utilization(utilization_df)
    fig2.savefig(os.path.join(output_dir, "model_gpu_utilization.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("Generating GPU utilization summary graph...")
    fig3 = visualizer.plot_gpu_utilization_summary(summary_df)
    fig3.savefig(os.path.join(output_dir, "gpu_utilization_summary.png"), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("GPU UTILIZATION SUMMARY STATISTICS")
    print("="*60)
    
    print("\nAverage GPU Utilization by System:")
    for system in summary_df['system'].unique():
        system_data = summary_df[summary_df['system'] == system]
        print(f"\n  {system}:")
        for _, row in system_data.iterrows():
            print(f"    GPU {row['gpu_id']}: {row['total_utilization']:.1f}%")
    
    print("\nModel Utilization Across All Systems:")
    models = ['efficientnetb0', 'resnet18', 'vit16']
    for model in models:
        util_col = f'{model}_utilization'
        avg_util = summary_df[util_col].mean()
        print(f"  {model}: {avg_util:.1f}%")
    
    print(f"\nGPU utilization graphs saved to: {output_dir}")
    print("Generated files:")
    for filename in ["gpu_utilization_over_time.png", "model_gpu_utilization.png", "gpu_utilization_summary.png"]:
        print(f"  - {filename}")

if __name__ == "__main__":
    main()
