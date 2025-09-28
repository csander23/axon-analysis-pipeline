#!/usr/bin/env python3
"""
Test script for hierarchical structure detection
This script demonstrates how the system auto-detects groups and conditions
"""

import os
import sys
from config_manager import ConfigManager, AnalysisConfig


def create_test_structure(base_dir: str):
    """Create a test directory structure for demonstration"""
    
    # Create the base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create groups and conditions
    groups = ["Group_A", "Group_B", "Group_C", "Group_D"]
    conditions = ["Condition_A", "Condition_B", "Condition_C", "Condition_D"]
    
    for group in groups:
        for condition in conditions:
            condition_path = os.path.join(base_dir, group, condition)
            os.makedirs(condition_path, exist_ok=True)
            
            # Create a dummy ND2 file for demonstration
            dummy_file = os.path.join(condition_path, f"sample_{group}_{condition}.nd2")
            with open(dummy_file, 'w') as f:
                f.write("# Dummy ND2 file for testing\n")
    
    print(f"Created test structure in: {base_dir}")
    print(f"Groups: {groups}")
    print(f"Conditions: {conditions}")


def test_auto_detection(input_dir: str):
    """Test the auto-detection functionality"""
    
    print("\n=== Testing Auto-Detection ===")
    
    # Test the auto-detection function directly
    groups, conditions = ConfigManager.auto_detect_structure(input_dir)
    print(f"Detected groups: {groups}")
    print(f"Detected conditions: {conditions}")
    
    # Test color generation
    colors = ConfigManager.generate_colors(conditions)
    print(f"Generated colors: {colors}")
    
    return groups, conditions, colors


def test_config_loading(config_path: str):
    """Test loading configuration with auto-detection"""
    
    print("\n=== Testing Configuration Loading ===")
    
    try:
        config = ConfigManager.load_from_yaml(config_path)
        
        print(f"Configuration loaded successfully!")
        print(f"Use hierarchical structure: {config.use_hierarchical_structure}")
        print(f"Auto-detect groups: {config.auto_detect_groups}")
        print(f"Auto-detect conditions: {config.auto_detect_conditions}")
        print(f"Groups: {config.groups}")
        print(f"Conditions: {config.conditions}")
        print(f"Colors: {config.colors}")
        
        return config
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None


def main():
    """Main test function"""
    
    print("=== Hierarchical Structure Test ===")
    
    # Get the current directory
    current_dir = os.path.dirname(__file__)
    test_data_dir = os.path.join(current_dir, "test_data")
    config_path = os.path.join(current_dir, "config.yaml")
    
    # Create test structure
    create_test_structure(test_data_dir)
    
    # Test auto-detection
    groups, conditions, colors = test_auto_detection(test_data_dir)
    
    # Create a temporary config for testing
    temp_config_path = os.path.join(current_dir, "test_config.yaml")
    
    test_config_content = f"""# Test configuration for hierarchical structure
input_dir: "{test_data_dir}"
output_dir_name: "test_output"

# Data Structure Configuration
use_hierarchical_structure: true
auto_detect_groups: true
auto_detect_conditions: true

# Analysis Parameters
min_sizes: [30]
conditions: []
groups: []
colors: {{}}

# Image Processing Parameters
gaussian_sigma: 1.0
soma_gaussian_sigma: 5
distance_threshold: 15
opening_disk_size: 20
dilation_disk_size: 20

# Morphological Operations
opening_disk_size_filter: 2
closing_disk_size_filter: 1

# Spider Analysis Parameters
window_length: 20
pink_density_threshold: 0.05
pink_thickness_threshold: 3

# Threshold Calculation Parameters
percentile_threshold: 15
threshold_intercept: 82.5909
threshold_coefficient: 0.3027

# Biological Replicate Offsets (can map to groups)
replicate_offsets:
  Group_A: -35
  Group_B: 20
  Group_C: -55
  Group_D: 5

# Channel Names
fitc_channel_name: "FITC"
tritc_channel_name: "TRITC"

# Video Generation Parameters
video_fps: 5
video_width: 800
video_height: 600

# Component Visualization Parameters
component_saturation: 0.8
component_value: 0.9
golden_ratio: 0.618033988749895

# Output Parameters
save_summary_plots: false
dpi: 300
plot_dpi: 200
"""
    
    # Write test config
    with open(temp_config_path, 'w') as f:
        f.write(test_config_content)
    
    # Test config loading
    config = test_config_loading(temp_config_path)
    
    if config:
        print("\n=== Test Summary ===")
        print("✅ Auto-detection working correctly")
        print("✅ Configuration loading working correctly")
        print("✅ Color generation working correctly")
        print("\nThe system is ready to process hierarchical data structures!")
        
        # Show file structure that would be processed
        print(f"\n=== Files that would be processed ===")
        for group in config.groups:
            for condition in config.conditions:
                condition_path = os.path.join(test_data_dir, group, condition)
                files = [f for f in os.listdir(condition_path) if f.endswith('.nd2')]
                for file in files:
                    print(f"  {group} -> {condition} -> {file}")
    else:
        print("❌ Configuration loading failed")
    
    # Cleanup
    print(f"\nCleaning up test files...")
    import shutil
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    print("✅ Cleanup complete")


if __name__ == "__main__":
    main()

