"""
Configuration Manager for slide-scanning-analysis.py
Handles loading and validation of configuration parameters from YAML files.
"""

import yaml
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AnalysisConfig:
    """Configuration class for slide scanning analysis"""
    
    # Input/Output
    input_dir: str
    output_dir: str  # Primary output directory
    output_subdirs: Dict[str, str] = field(default_factory=lambda: {
        "images": "Images",
        "plots": "Plots",
        "results": "Results"
    })
    
    # Data Structure Configuration
    use_hierarchical_structure: bool = False
    auto_detect_groups: bool = False
    auto_detect_conditions: bool = False
    
    # Regression Model Configuration
    use_regression_model: bool = False
    regression_model_path: str = ""
    
    # Raw Threshold Configuration
    use_raw_threshold: bool = True
    raw_threshold_value: float = 120.0
    
    # Analysis Parameters
    min_sizes: List[int] = field(default_factory=lambda: [30])
    conditions: List[str] = field(default_factory=lambda: ["Control", "KO", "ApoE2", "ApoE4"])
    groups: List[str] = field(default_factory=lambda: [])
    colors: Dict[str, str] = field(default_factory=lambda: {
        "Control": "#FF7F00", "KO": "#E31A1C", "ApoE2": "#33A02C", "ApoE4": "#1F78B4"
    })
    
    # Image Processing Parameters
    gaussian_sigma: float = 1.0
    soma_gaussian_sigma: float = 5.0
    distance_threshold: int = 15
    opening_disk_size: int = 20
    dilation_disk_size: int = 20
    
    # Morphological Operations
    opening_disk_size_filter: int = 2
    closing_disk_size_filter: int = 1
    
    # Spider Analysis Parameters
    window_length: int = 20
    pink_density_threshold: float = 0.05
    pink_thickness_threshold: float = 3.0
    
    # Threshold Calculation Parameters
    percentile_threshold: int = 15
    threshold_intercept: float = 82.5909
    threshold_coefficient: float = 0.3027
    
    # Biological Replicate Offsets
    replicate_offsets: Dict[str, int] = field(default_factory=lambda: {
        "Group_A": 0, "Group_B": 0, "Group_C": 0, "Group_D": 0
    })
    
    # Channel Names
    fitc_channel_name: str = "FITC"
    tritc_channel_name: str = "TRITC"
    
    # Component Visualization Parameters
    component_saturation: float = 0.8
    component_value: float = 0.9
    golden_ratio: float = 0.618033988749895
    
    # Output Parameters
    save_summary_plots: bool = True
    dpi: int = 300
    plot_dpi: int = 200
    
    # Thick vs Thin Analysis Parameters (slide scanning specific)
    enable_thick_thin_analysis: bool = True
    width_threshold: float = 3.7  # Different from confocal (3.7 vs 6)
    branch_distance_threshold: int = 12
    min_wide_region_size: int = 10
    branch_count_threshold: int = 2
    normalize_to_wt: bool = True
    
    # Slide Scanning Specific Parameters
    parallel_processing: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "max_workers": 12,
        "chunk_size": 3,
        "progress_update_interval": 5,
        "memory_optimization": True
    })
    
    # Image format and loading parameters
    image_formats: List[str] = field(default_factory=lambda: [".tif", ".tiff", ".png", ".jpg", ".jpeg"])
    load_as_grayscale: bool = False
    
    def get_output_dir(self, output_type: str) -> str:
        """
        Get the full path for a specific output type directory.
        
        Args:
            output_type: Type of output ("images", "plots", "results")
            
        Returns:
            str: Full path to the output directory
        """
        if output_type not in self.output_subdirs:
            raise ValueError(f"Unknown output type: {output_type}. Valid types: {list(self.output_subdirs.keys())}")
        
        return os.path.join(self.output_dir, self.output_subdirs[output_type])
    
    def get_output_path(self, output_type: str, filename: str) -> str:
        """
        Get the full path for an output file.
        
        Args:
            output_type: Type of output ("images", "plots", "results")
            filename: Name of the file
            
        Returns:
            str: Full path to the output file
        """
        return os.path.join(self.get_output_dir(output_type), filename)
    
    def ensure_output_dirs(self) -> None:
        """Create all output directories if they don't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        for output_type in self.output_subdirs:
            os.makedirs(self.get_output_dir(output_type), exist_ok=True)
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        # Validate input directory
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        # Validate output subdirectories
        required_subdirs = {"images", "plots", "results"}
        missing_subdirs = required_subdirs - set(self.output_subdirs.keys())
        if missing_subdirs:
            raise ValueError(f"Missing required output subdirectories: {missing_subdirs}")
        
        # Validate threshold configuration - only one method should be enabled
        threshold_methods = [self.use_regression_model, self.use_raw_threshold]
        if sum(threshold_methods) > 1:
            raise ValueError("Only one threshold method can be enabled: use_regression_model, use_raw_threshold, or manual parameters")
        
        # Validate raw threshold configuration
        if self.use_raw_threshold:
            if self.raw_threshold_value <= 0:
                raise ValueError("raw_threshold_value must be positive")
        
        if not self.min_sizes:
            raise ValueError("min_sizes cannot be empty")
        
        if not self.conditions:
            raise ValueError("conditions cannot be empty")
        
        # Validate that all conditions have colors
        missing_colors = set(self.conditions) - set(self.colors.keys())
        if missing_colors:
            raise ValueError(f"Missing colors for conditions: {missing_colors}")
        
        # Validate numeric parameters
        if self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        
        if self.window_length <= 0:
            raise ValueError("window_length must be positive")
        
        if self.pink_density_threshold < 0:
            raise ValueError("pink_density_threshold must be non-negative")
        
        # Validate parallel processing parameters
        if self.parallel_processing["enabled"]:
            if self.parallel_processing["max_workers"] <= 0:
                raise ValueError("max_workers must be positive")
            if self.parallel_processing["chunk_size"] <= 0:
                raise ValueError("chunk_size must be positive")


class ConfigManager:
    """Manager class for loading and handling configuration"""
    
    @staticmethod
    def auto_detect_structure(input_dir: str) -> tuple[List[str], List[str]]:
        """
        Auto-detect groups and conditions from directory structure.
        
        Args:
            input_dir: Path to input directory
            
        Returns:
            Tuple of (groups, conditions)
        """
        groups = []
        conditions = set()
        
        if not os.path.exists(input_dir):
            return groups, list(conditions)
        
        # Look for Group_X directories
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path) and item.startswith('Group_'):
                groups.append(item)
                
                # Look for Condition_Y directories within each group
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path) and subitem.startswith('Condition_'):
                        conditions.add(subitem)
        
        groups.sort()  # Sort for consistent ordering
        conditions = sorted(list(conditions))  # Sort for consistent ordering
        
        return groups, conditions
    
    @staticmethod
    def generate_colors(conditions: List[str]) -> Dict[str, str]:
        """
        Generate distinct colors for conditions using slide scanning color scheme.
        
        Args:
            conditions: List of condition names
            
        Returns:
            Dictionary mapping condition names to colors
        """
        # Use the slide scanning color scheme from legacy model
        slide_scanning_colors = {
            'Control': '#FF7F00',     # Orange
            'KO': '#E31A1C',          # Bright red
            'ApoE4': '#1F78B4',       # Blue  
            'ApoE2': '#33A02C',       # Green
            'ApoE2-NTD': '#B2DF8A',   # Light green
            'ApoE4-NTD': '#A6CEE3',   # Light blue
            'ApoE-CTD': '#DDA0DD'     # Plum
        }
        
        colors = {}
        for i, condition in enumerate(conditions):
            if condition in slide_scanning_colors:
                colors[condition] = slide_scanning_colors[condition]
            else:
                # Generate additional colors if needed
                import colorsys
                hue = i / len(conditions)
                saturation = 0.8
                value = 0.9
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255),
                    int(rgb[1] * 255),
                    int(rgb[2] * 255)
                )
                colors[condition] = hex_color
        
        return colors
    
    @staticmethod
    def load_from_yaml(config_path: str) -> AnalysisConfig:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        # Create AnalysisConfig instance from dictionary
        config = AnalysisConfig(**config_dict)
        
        # Auto-detect structure if enabled
        if config.use_hierarchical_structure:
            if config.auto_detect_groups or config.auto_detect_conditions:
                detected_groups, detected_conditions = ConfigManager.auto_detect_structure(config.input_dir)
                
                if config.auto_detect_groups:
                    config.groups = detected_groups
                    print(f"Auto-detected groups: {config.groups}")
                
                if config.auto_detect_conditions:
                    config.conditions = detected_conditions
                    print(f"Auto-detected conditions: {config.conditions}")
                
                # Generate colors if not provided or if conditions were auto-detected
                if not config.colors or config.auto_detect_conditions:
                    config.colors = ConfigManager.generate_colors(config.conditions)
                    print(f"Generated colors: {config.colors}")
        
        config.validate()
        return config
    
    @staticmethod
    def load_from_dict(config_dict: Dict[str, Any]) -> AnalysisConfig:
        """Load configuration from dictionary"""
        config = AnalysisConfig(**config_dict)
        config.validate()
        return config
    
    @staticmethod
    def get_default_config() -> AnalysisConfig:
        """Get default configuration"""
        config = AnalysisConfig(
            input_dir="/Users/charlessander/Desktop/patzke lab computing/Sudhof Lab/Axon Analysis Pipeline/Slide_Scanning_Axon_Analysis_Pipeline/Slide_Scanning_Dataset",
            output_dir="/Users/charlessander/Desktop/patzke lab computing/Sudhof Lab/Axon Analysis Pipeline/Slide_Scanning_Axon_Analysis_Pipeline/Data_Output"
        )
        config.validate()
        return config


# Convenience function for easy import
def load_config(config_path: Optional[str] = None) -> AnalysisConfig:
    """
    Load configuration from file or return default configuration
    
    Args:
        config_path: Path to YAML configuration file. If None, uses default config.
    
    Returns:
        AnalysisConfig instance
    """
    if config_path is None:
        return ConfigManager.get_default_config()
    else:
        return ConfigManager.load_from_yaml(config_path)
