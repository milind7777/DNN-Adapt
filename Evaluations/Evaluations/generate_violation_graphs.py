import os
import matplotlib.pyplot as plt
from data_processor import ViolationDataProcessor
from visualization import ViolationVisualizer

def main():
    # Configuration
    base_path = "/Users/sai/Downloads/MS-Project-Results/results-ramp-constant"
    output_dir = os.path.join(base_path, "generated_graphs")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor and visualizer
    processor = ViolationDataProcessor(base_path)
    visualizer = ViolationVisualizer()
    
    print("Loading data from all systems...")
    all_data = processor.load_all_data()
    
    if all_data.empty:
        print("No data found. Please check file paths and ensure CSV files exist.")
        return
    
    print(f"Loaded data for {len(all_data)} records across {all_data['system'].nunique()} systems")
    print(f"Time range: {all_data['seconds'].min():.0f} - {all_data['seconds'].max():.0f} seconds")
    
    # Process data
    total_violations = processor.get_total_violations_per_second(all_data)
    model_violations = processor.get_model_violations_per_second(all_data)
    
    # Generate total violations graphs
    print("\nGenerating total violations line graph...")
    fig1 = visualizer.plot_total_violations_line(total_violations)
    fig1.savefig(os.path.join(output_dir, "total_violations_line.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("Generating total violations bar chart...")
    fig2 = visualizer.plot_total_violations_bar(total_violations)
    fig2.savefig(os.path.join(output_dir, "total_violations_bar.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Generate model-specific graphs
    print("Generating model violations line graphs...")
    fig3 = visualizer.plot_model_violations_line(model_violations)
    fig3.savefig(os.path.join(output_dir, "model_violations_line.png"), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("Generating model violations bar chart...")
    fig4 = visualizer.plot_model_violations_bar(model_violations)
    fig4.savefig(os.path.join(output_dir, "model_violations_bar.png"), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print("Generating violations heatmap...")
    fig5 = visualizer.plot_violations_heatmap(model_violations)
    fig5.savefig(os.path.join(output_dir, "violations_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print("\nTotal Violations by System:")
    system_summary = total_violations.groupby('system')['violation_count'].sum().sort_values(ascending=False)
    for system, violations in system_summary.items():
        print(f"  {system}: {violations:,} violations")
    
    print("\nViolations by Model (across all systems):")
    model_summary = model_violations.groupby('model_name')['violation_count'].sum().sort_values(ascending=False)
    for model, violations in model_summary.items():
        print(f"  {model}: {violations:,} violations")
    
    print(f"\nGraphs saved to: {output_dir}")
    print("Generated files:")
    for filename in ["total_violations_line.png", "total_violations_bar.png", 
                     "model_violations_line.png", "model_violations_bar.png", "violations_heatmap.png"]:
        print(f"  - {filename}")

if __name__ == "__main__":
    main()
