#!/usr/bin/env python3
"""
Runner script for branch-based-snakes.py
This script demonstrates how to programmatically configure and run the L1CAM analysis.
"""

import os
import sys
import importlib.util
from config_manager import AnalysisConfig, ConfigManager

# Import the module with hyphens in the name
spec = importlib.util.spec_from_file_location("branch_based_snakes", 
                                            os.path.join(os.path.dirname(__file__), "branch-based-snakes.py"))
branch_based_snakes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(branch_based_snakes)
run_analysis = branch_based_snakes.run_analysis


def create_custom_config():
    """
    Create a custom configuration programmatically.
    This demonstrates how to override default parameters.
    """
    
    # Start with default configuration
    config = AnalysisConfig(
        input_dir="/Users/charlessander/Desktop/patzke lab computing/Sudhof Lab/L1CAM Analysis/making normalized cdf"
    )
    
    # Override specific parameters as needed
    config.min_sizes = [30, 50]  # Analyze multiple min sizes
    config.conditions = ["WT", "KO", "E2", "E4"]
    config.save_summary_plots = True  # Enable summary plot generation
    config.gaussian_sigma = 1.5  # Slightly more smoothing
    config.pink_density_threshold = 0.06  # Slightly higher threshold
    
    # Custom colors
    config.colors = {
        "WT": "blue",
        "KO": "red", 
        "E2": "green",
        "E4": "orange"
    }
    
    return config


def run_with_yaml_config(config_path: str):
    """
    Run analysis using a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    print(f"Loading configuration from: {config_path}")
    
    try:
        config = ConfigManager.load_from_yaml(config_path)
        print("Configuration loaded successfully!")
        
        # Print some key parameters
        print(f"Input directory: {config.input_dir}")
        print(f"Output directory: {config.output_dir}")
        print(f"Min sizes: {config.min_sizes}")
        print(f"Conditions: {config.conditions}")
        
        # Run the analysis
        run_analysis(config)
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        sys.exit(1)


def run_with_custom_config():
    """
    Run analysis using programmatically created configuration.
    """
    print("Creating custom configuration...")
    
    try:
        config = create_custom_config()
        config.validate()
        
        print("Custom configuration created successfully!")
        print(f"Input directory: {config.input_dir}")
        print(f"Output directory: {config.output_dir}")
        print(f"Min sizes: {config.min_sizes}")
        print(f"Conditions: {config.conditions}")
        
        # Optionally save the configuration for future use
        config_save_path = os.path.join(
            os.path.dirname(__file__), 
            "custom_config.yaml"
        )
        ConfigManager.save_to_yaml(config, config_save_path)
        print(f"Configuration saved to: {config_save_path}")
        
        # Run the analysis
        run_analysis(config)
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        sys.exit(1)


def run_with_parameter_overrides():
    """
    Demonstrate how to load a config file and override specific parameters.
    """
    print("Loading base configuration and applying overrides...")
    
    try:
        # Load base configuration from file
        base_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        config = ConfigManager.load_from_yaml(base_config_path)
        
        # Override specific parameters
        config.min_sizes = [25, 40, 60]  # Different min sizes
        config.gaussian_sigma = 2.0  # More smoothing
        config.save_summary_plots = True  # Enable plots
        config.video_fps = 10  # Faster video playback
        
        # Override input directory if needed
        # config.input_dir = "/path/to/different/input/directory"
        
        print("Configuration overrides applied!")
        print(f"New min sizes: {config.min_sizes}")
        print(f"New gaussian sigma: {config.gaussian_sigma}")
        
        # Run the analysis
        run_analysis(config)
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        sys.exit(1)


def main():
    """
    Main function demonstrating different ways to run the analysis.
    """
    print("=== L1CAM Analysis Runner ===")
    print("This script demonstrates different ways to configure and run the analysis.")
    print()
    
    # Check if config file exists
    config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "yaml" and os.path.exists(config_file):
            print("Running with YAML configuration...")
            run_with_yaml_config(config_file)
            
        elif mode == "custom":
            print("Running with custom programmatic configuration...")
            run_with_custom_config()
            
        elif mode == "override":
            print("Running with parameter overrides...")
            run_with_parameter_overrides()
            
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: yaml, custom, override")
            sys.exit(1)
    else:
        # Default behavior - try YAML config first, then custom
        if os.path.exists(config_file):
            print("No mode specified. Running with YAML configuration...")
            run_with_yaml_config(config_file)
        else:
            print("No config file found. Running with custom configuration...")
            run_with_custom_config()


if __name__ == "__main__":
    main()
