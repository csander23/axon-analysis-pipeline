"""
Analysis utilities for slide scanning pipeline
"""

import os
import yaml
import importlib.util
from config_manager import AnalysisConfig, ConfigManager

def load_and_configure_analysis(input_dir, output_dir, active_config_dir):
    """
    Load and configure the analysis based on directory structure and config file.
    
    Args:
        input_dir: Path to input data directory
        output_dir: Path to output directory
        active_config_dir: Path to active configuration directory
        
    Returns:
        AnalysisConfig object ready for analysis
    """
    # Load base configuration from active config directory
    config_path = os.path.join(active_config_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Auto-detect groups and conditions from directory structure
    groups = []
    conditions = []
    
    if os.path.exists(input_dir):
        # First level directories are bioreplicates (no name requirements)
        groups = sorted([d for d in os.listdir(input_dir) 
                        if os.path.isdir(os.path.join(input_dir, d))])
        
        if groups:
            # Get conditions from first group (assuming all groups have same conditions)
            first_group_path = os.path.join(input_dir, groups[0])
            # Second level directories are conditions (no name requirements)
            conditions = sorted([d for d in os.listdir(first_group_path) 
                               if os.path.isdir(os.path.join(first_group_path, d))])
    
    print(f"Auto-detected groups: {groups}")
    print(f"Auto-detected conditions: {conditions}")
    
    # Generate colors for conditions
    colors = ConfigManager.generate_colors(conditions)
    
    # Create configuration dictionary with overrides
    config_dict = base_config.copy()
    config_dict.update({
        "input_dir": input_dir,
        "output_dir": output_dir,
        "use_hierarchical_structure": len(groups) > 0,
        "auto_detect_groups": True,
        "auto_detect_conditions": True,
        "groups": groups,
        "conditions": conditions,
        "colors": colors,
        "save_summary_plots": True
    })
    
    # Handle legacy data_output_dir parameter
    if "data_output_dir" in config_dict:
        config_dict["output_dir"] = config_dict.pop("data_output_dir")
    
    # Create and validate config
    config = AnalysisConfig(**config_dict)
    config.ensure_output_dirs()
    
    return config

def run_analysis(config):
    """
    Import and run the main analysis script.
    
    Args:
        config: AnalysisConfig object
    """
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Import the main analysis module
    spec = importlib.util.spec_from_file_location(
        "slide_scanning_analysis", 
        os.path.join(current_dir, "slide_scanning_analysis.py")
    )
    analysis_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis_module)
    
    # Run analysis
    print("\nStarting analysis...\n")
    analysis_module.run_analysis(config)
    
    # Print results location
    print("\nAnalysis complete! Results are in:")
    print(f"  Images: {config.get_output_dir('images')}")
    print(f"  Plots: {config.get_output_dir('plots')}")
    print(f"  Results: {config.get_output_dir('results')}")

def handle_new_regression_model(regression_dir, active_config_dir):
    """
    Handle the creation of a new regression model.
    
    Args:
        regression_dir: Directory containing regression models
        active_config_dir: Directory for active configuration
    """
    import glob
    import shutil
    
    # Find the most recently created model
    json_files = glob.glob(os.path.join(regression_dir, "*_regression_model_*.json"))
    if json_files:
        latest_model = max(json_files, key=os.path.getctime)
        model_name = os.path.basename(latest_model)
        
        print(f"\nNew regression model created: {model_name}")
        print("\nTo activate this model:")
        print(f"1. Copy {model_name} to {active_config_dir}/active_model.json")
        print("2. Set use_regression_model: true in config.yaml")
        print("3. Re-run the analysis")
        
        # Find associated files
        base_name = model_name.replace('.json', '')
        txt_file = os.path.join(regression_dir, base_name + '.txt')
        plot_file = os.path.join(regression_dir, base_name + '_regression_plot.png')
        
        print(f"\nAssociated files:")
        if os.path.exists(txt_file):
            print(f"  - Documentation: {os.path.basename(txt_file)}")
        if os.path.exists(plot_file):
            print(f"  - Plot: {os.path.basename(plot_file)}")
        
        return latest_model
    else:
        print("\nNo new regression model was created.")
        return None
