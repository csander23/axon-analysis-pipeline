"""
Utility functions for the axon analysis pipeline.
Handles configuration loading, path management, and analysis execution.
"""

import os
import yaml
import glob
import shutil
from typing import List, Dict, Optional
from config_manager import AnalysisConfig
import importlib.util

def load_and_configure_analysis(
    input_dir: str,
    output_dir: str,
    active_config_dir: str
) -> AnalysisConfig:
    """
    Load and configure the analysis based on directory structure and config file.
    
    Args:
        input_dir: Path to input data directory
        output_dir: Path to output directory
        active_config_dir: Path to active configuration directory
        
    Returns:
        AnalysisConfig object ready for analysis
    """
    # Get groups and conditions from directory structure
    # First level directories are bioreplicates (no name requirements)
    groups = sorted([d for d in os.listdir(input_dir) 
                    if os.path.isdir(os.path.join(input_dir, d))])
    
    conditions = []
    if groups:
        # Get conditions from first group (assuming all groups have same conditions)
        group_path = os.path.join(input_dir, groups[0])
        # Second level directories are conditions (no name requirements)
        conditions = sorted([d for d in os.listdir(group_path) 
                           if os.path.isdir(os.path.join(group_path, d))])
    
    print(f"Found {len(groups)} groups: {groups}")
    print(f"Found {len(conditions)} conditions: {conditions}")
    
    # Load configuration
    config_path = os.path.join(active_config_dir, "config.yaml")
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        print(f"\nLoaded configuration from: {config_path}")
    
    # Convert data_output_dir to output_dir if needed
    config_dict = base_config.copy()
    if 'data_output_dir' in config_dict:
        config_dict['output_dir'] = config_dict.pop('data_output_dir')
    
    # Override with specific settings
    config_dict.update({
        "input_dir": input_dir,
        "output_dir": output_dir,
        "use_hierarchical_structure": True,
        "auto_detect_groups": False,
        "auto_detect_conditions": False,
        "groups": groups,
        "conditions": conditions,
        "colors": {
            "Condition_A": "red",    # KO
            "Condition_B": "blue",   # WT
            "Condition_C": "green",  # E2
            "Condition_D": "orange"  # E4
        },
        "replicate_offsets": {
            "Group_A": -35,  # B114
            "Group_B": 5,    # B117
            "Group_C": 20,   # B115/B116
            "Group_D": -35   # B114
        },
        "regression_model_path": os.path.join(active_config_dir, "active_model.json"),
        "save_summary_plots": True
    })
    
    # Create and validate config
    config = AnalysisConfig(**config_dict)
    config.ensure_output_dirs()
    
    return config

def run_analysis(config: AnalysisConfig) -> None:
    """
    Run the analysis using the provided configuration.
    
    Args:
        config: AnalysisConfig object with analysis parameters
    """
    # Import analysis module
    spec = importlib.util.spec_from_file_location(
        "branch_based_snakes",
        os.path.join(os.path.dirname(__file__), "branch-based-snakes.py")
    )
    analysis_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis_module)
    
    # Run analysis
    print("\nStarting analysis...\n")
    analysis_module.run_analysis(config)
    
    # Print results location
    print("\nAnalysis complete! Results are in:")
    print(f"1. Main CSVs: {config.output_dir}")
    print(f"2. Image summaries: {os.path.join(config.output_dir, 'Images/processed')}")
    print(f"3. Debug visualizations: {os.path.join(config.output_dir, 'Debug')}")
    print(f"4. Overall plots: {os.path.join(config.output_dir, 'Plots')}")

def handle_new_regression_model(
    regression_dir: str,
    active_config_dir: str
) -> None:
    """
    Find the most recently created regression model and print instructions.
    Does NOT automatically copy the model - user must do this manually.
    
    Args:
        regression_dir: Directory containing regression models
        active_config_dir: Directory for active configuration and model
    """
    json_files = glob.glob(os.path.join(regression_dir, "*_regression_model_*.json"))
    if not json_files:
        print("\nNo regression model was created. Please try again.")
        return
        
    latest_model = max(json_files, key=os.path.getctime)
    latest_txt = latest_model.replace('.json', '.txt')
    latest_plot = latest_model.replace('.json', '_regression_plot.png')
    
    print("\nNew regression model created successfully!")
    print(f"\nModel files are in: {regression_dir}")
    print(f"• Model: {os.path.basename(latest_model)}")
    print(f"• Documentation: {os.path.basename(latest_txt)}")
    print(f"• Plot: {os.path.basename(latest_plot)}")
    print("\nTo use this model:")
    print(f"1. Copy these files to {active_config_dir} as:")
    print("   • active_model.json")
    print("   • active_model.txt")
    print("   • active_model_regression_plot.png")
    print("\n2. Make sure config.yaml is also in this directory")
    print("3. Run the analysis cell again to use the new model")
