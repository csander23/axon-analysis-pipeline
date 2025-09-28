"""
Slide Scanning L1CAM Analysis - Main Analysis Script
Based on the legacy model from L1CAM P2 comprehensive analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters, morphology, measure, color
from skimage.filters import threshold_otsu
import os
from skimage.morphology import skeletonize, remove_small_objects, label, closing, opening, disk, dilation, reconstruction
from skimage.measure import regionprops
from skimage.color import label2rgb
from scipy.ndimage import gaussian_filter, convolve, distance_transform_edt
import pandas as pd
import re
from collections import defaultdict
from scipy.stats import ecdf, linregress, f_oneway, ttest_ind
import glob
import itertools
from PIL import Image
import cv2
import csv
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import sys
import argparse
from config_manager import load_config, AnalysisConfig

# Filter parameters (from legacy slide scanning model)
FILTER_PARAMS = {
    # Component size and skeleton length filters
    "MIN_SKELETON_LENGTH": 25,        # Minimum skeleton length (pixels)
    "MAX_SKELETON_LENGTH": 2000000,   # Maximum skeleton length (pixels)
    
    # Thickness filters (using radius values to match original script)
    "MIN_THICKNESS": 0,               # Minimum thickness (radius in pixels)
    "MAX_THICKNESS": 7.0,             # Maximum thickness (radius in pixels)
    "MAX_COMPONENT_THICKNESS": 7.5,   # Maximum component thickness (radius in pixels)
    "MIN_AVG_THICKNESS": 0,           # Minimum average thickness (radius in pixels)
    "MAX_AVG_THICKNESS": 4,           # Maximum average thickness (radius in pixels)
    
    # Morphological filters
    "MIN_OBJECT_SIZE": 25,            # Minimum object size for morphological operations
    "OPENING_DISK_SIZE": 2,           # Disk size for opening operation
    "CLOSING_DISK_SIZE": 1,           # Disk size for closing operation
    
    # Spider analysis parameters
    "SPIDER_WINDOW_LENGTH": 20,       # Window length for spider analysis (pixels)
    "BRANCH_DENSITY_THRESHOLD": 0.09, # Branch density threshold (legacy value)
    "MIN_AVG_THICKNESS_PINK": 3,      # Minimum average thickness for pink regions (radius in pixels)
}

def load_image(image_path, config):
    """Load image from various formats supported by slide scanning."""
    try:
        # Try PIL first for common formats
        if any(image_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']):
            img = Image.open(image_path)
            if config.load_as_grayscale:
                img = img.convert('L')
                return np.array(img)
            else:
                img = img.convert('RGB')
                return np.array(img)
        else:
            # Fallback to OpenCV
            if config.load_as_grayscale:
                return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def extract_grayscale_legacy(image, config):
    """Convert to grayscale using legacy method (mean of all channels) and normalize."""
    if len(image.shape) == 3:
        # RGB image - convert to grayscale using mean of all channels (legacy method)
        gray = np.mean(image, axis=2)
    else:
        # Already grayscale
        gray = image
    
    # Normalize to 0-1 range (legacy approach)
    gray = gray.astype(np.float32) / 255.0
    return gray

def calculate_threshold(image, config):
    """Calculate threshold based on configuration."""
    if config.use_raw_threshold:
        return config.raw_threshold_value
    elif config.use_regression_model:
        # Implement regression model logic if needed
        # For now, fallback to manual calculation
        vals = image.ravel()
        thr15 = np.percentile(vals, config.percentile_threshold)
        above15 = vals[vals > thr15]
        mean_above15 = above15.mean() if above15.size > 0 else 0.0
        return config.threshold_intercept + config.threshold_coefficient * mean_above15
    else:
        # Manual threshold calculation
        vals = image.ravel()
        thr15 = np.percentile(vals, config.percentile_threshold)
        above15 = vals[vals > thr15]
        mean_above15 = above15.mean() if above15.size > 0 else 0.0
        return config.threshold_intercept + config.threshold_coefficient * mean_above15

def find_branch_points(skeleton):
    """Find branch points in skeleton."""
    neighbor_kernel = np.array([[1,1,1], [1,0,1], [1,1,1]])
    neighbor_count = convolve(skeleton.astype(int), neighbor_kernel, mode="constant", cval=0)
    branch_points = (skeleton & (neighbor_count >= 3))
    return branch_points

def calculate_thickness(skeleton, distance_map):
    """Calculate thickness at skeleton points."""
    thickness = np.zeros_like(skeleton, dtype=float)
    thickness[skeleton] = distance_map[skeleton]
    return thickness

def spider_analysis(skeleton, branch_points, thickness):
    """Perform spider analysis to identify high branch density regions."""
    pink_mask = np.zeros_like(skeleton, dtype=bool)
    
    # Get skeleton coordinates
    skeleton_coords = np.column_stack(np.where(skeleton))
    
    for coord in skeleton_coords:
        y, x = coord
        if not skeleton[y, x]:
            continue
            
        # Create spider window around this point
        spider_coords = get_spider_coords(skeleton, (y, x), FILTER_PARAMS["SPIDER_WINDOW_LENGTH"])
        
        if len(spider_coords) == 0:
            continue
            
        # Calculate branch density in spider
        branch_count = 0
        thickness_values = []
        
        for sy, sx in spider_coords:
            if branch_points[sy, sx]:
                branch_count += 1
            thickness_values.append(thickness[sy, sx])
        
        density = branch_count / len(spider_coords) if spider_coords else 0
        avg_thickness = np.mean(thickness_values) if thickness_values else 0
        
        # Apply pink criteria (legacy values)
        if (density > FILTER_PARAMS["BRANCH_DENSITY_THRESHOLD"] and 
            avg_thickness >= FILTER_PARAMS["MIN_AVG_THICKNESS_PINK"]):
            for sy, sx in spider_coords:
                pink_mask[sy, sx] = True
    
    return pink_mask

def get_spider_coords(skeleton, start_coord, window_length):
    """Get coordinates within spider window from starting point."""
    y, x = start_coord
    visited = set()
    coords = []
    queue = [(y, x, 0)]  # (y, x, distance)
    
    while queue:
        cy, cx, dist = queue.pop(0)
        
        if (cy, cx) in visited or dist > window_length:
            continue
            
        if (0 <= cy < skeleton.shape[0] and 0 <= cx < skeleton.shape[1] and 
            skeleton[cy, cx]):
            visited.add((cy, cx))
            coords.append((cy, cx))
            
            # Add neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (ny, nx) not in visited:
                        queue.append((ny, nx, dist + 1))
    
    return coords

def calculate_thick_thin_analysis(skeleton, distance_map, config):
    """Calculate thick vs thin analysis based on width threshold."""
    if not config.enable_thick_thin_analysis:
        return None, None, np.nan
    
    # Calculate radius at each skeleton point
    radius_map = np.zeros_like(skeleton, dtype=float)
    radius_map[skeleton] = distance_map[skeleton]
    
    # Classify as thick or thin based on width threshold
    thick_mask = (radius_map >= config.width_threshold) & skeleton
    thin_mask = skeleton & ~thick_mask
    
    # Count pixels
    thick_count = np.count_nonzero(thick_mask)
    thin_count = np.count_nonzero(thin_mask)
    
    # Calculate ratio
    ratio_thick_thin = thick_count / thin_count if thin_count > 0 else np.nan
    
    return thick_mask, thin_mask, ratio_thick_thin

def process_single_image(image_path, config, condition, biological_replicate):
    """Process a single image and return analysis results."""
    print(f"  Processing: {os.path.basename(image_path)}")
    
    # Load image
    image = load_image(image_path, config)
    if image is None:
        return None
    
    # Convert to grayscale using legacy method (mean of all channels) and normalize
    gray = extract_grayscale_legacy(image, config)
    print(f"    Image shape: {gray.shape}, dtype: {gray.dtype}")
    print(f"    Image intensity range: {gray.min():.1f} - {gray.max():.1f}")
    print(f"    Image mean: {gray.mean():.1f}, std: {gray.std():.1f}")
    print(f"    Image percentiles - 50th: {np.percentile(gray, 50):.1f}, 90th: {np.percentile(gray, 90):.1f}, 95th: {np.percentile(gray, 95):.1f}, 99th: {np.percentile(gray, 99):.1f}")
    
    # Apply Gaussian smoothing
    smooth_gray = gaussian_filter(gray, sigma=config.gaussian_sigma)
    
    # Calculate threshold
    threshold = calculate_threshold(smooth_gray, config)
    print(f"    Threshold: {threshold:.2f}")
    
    # Apply threshold
    thresholded = smooth_gray > threshold
    print(f"    Thresholded pixels: {np.sum(thresholded)}")
    
    # Morphological operations
    filtered_mask = remove_small_objects(thresholded, min_size=FILTER_PARAMS["MIN_OBJECT_SIZE"])
    print(f"    After small object removal: {np.sum(filtered_mask)}")
    filtered_mask = opening(filtered_mask, disk(FILTER_PARAMS["OPENING_DISK_SIZE"]))
    print(f"    After opening: {np.sum(filtered_mask)}")
    filtered_mask = closing(filtered_mask, disk(FILTER_PARAMS["CLOSING_DISK_SIZE"]))
    print(f"    After closing: {np.sum(filtered_mask)}")
    
    # Skeletonize
    skeleton = skeletonize(filtered_mask)
    
    # Find branch points
    branch_points = find_branch_points(skeleton)
    
    # Calculate thickness
    distance_map = distance_transform_edt(filtered_mask)
    thickness = calculate_thickness(skeleton, distance_map)
    
    # Filter skeleton by component size (legacy approach)
    print(f"    Filtering skeleton components by size...")
    skeleton_labeled = label(skeleton)
    skeleton_components = regionprops(skeleton_labeled)
    
    # Create mask for valid skeleton components (≥ MIN_SKELETON_LENGTH pixels)
    valid_components_mask = np.zeros_like(skeleton, dtype=bool)
    valid_component_count = 0
    for prop in skeleton_components:
        if len(prop.coords) >= FILTER_PARAMS["MIN_SKELETON_LENGTH"]:
            coords = prop.coords
            for y, x in coords:
                valid_components_mask[y, x] = True
            valid_component_count += 1
    
    print(f"    Valid skeleton components (≥{FILTER_PARAMS['MIN_SKELETON_LENGTH']} pixels): {valid_component_count}")
    
    # Apply thickness filter to skeleton (legacy approach)
    thickness_mask = ((thickness >= FILTER_PARAMS["MIN_THICKNESS"]) & 
                     (thickness <= FILTER_PARAMS["MAX_THICKNESS"]))
    print(f"    Pixels passing thickness filter: {np.sum(thickness_mask & valid_components_mask)} / {np.sum(valid_components_mask)}")
    
    # Apply thickness filter to skeleton
    filtered_skeleton = valid_components_mask & thickness_mask
    
    # Perform spider analysis on the remaining skeleton (after size and thickness filtering)
    pink_mask = spider_analysis(filtered_skeleton, branch_points, thickness)
    
    # Classify skeleton pixels (matching original approach)
    # Blue region: normal branch density
    # Pink region: high branch density (assigned by spider analysis)
    blue_mask = filtered_skeleton & ~pink_mask
    
    print(f"    All skeleton pixels: {np.sum(skeleton > 0)}")
    print(f"    Valid skeleton pixels (≥{FILTER_PARAMS['MIN_SKELETON_LENGTH']} pixels): {np.sum(skeleton & valid_components_mask)}")
    print(f"    Pixels after thickness filter: {np.sum(filtered_skeleton > 0)}")
    print(f"    Blue region pixels: {np.sum(blue_mask)}")
    print(f"    Pink region pixels: {np.sum(pink_mask)}")
    
    # Thick vs thin analysis
    thick_mask, thin_mask, ratio_thick_thin = calculate_thick_thin_analysis(
        filtered_skeleton, distance_map, config)
    
    # Collect thickness data
    blue_thickness = thickness[blue_mask]
    pink_thickness = thickness[pink_mask]
    all_thickness = thickness[filtered_skeleton]
    
    # Create summary plot
    if config.save_summary_plots:
        save_summary_plot(image_path, gray, filtered_mask, skeleton, 
                         blue_mask, pink_mask, thick_mask, thin_mask, 
                         threshold, config, distance_map)
    
    return {
        'image': os.path.basename(image_path),
        'condition': condition,
        'biological_replicate': biological_replicate,
        'threshold': threshold,
        'all_thickness': all_thickness,
        'blue_thickness': blue_thickness,
        'pink_thickness': pink_thickness,
        'thick_count': np.count_nonzero(thick_mask) if thick_mask is not None else 0,
        'thin_count': np.count_nonzero(thin_mask) if thin_mask is not None else 0,
        'ratio_thick_thin': ratio_thick_thin,
        'total_skeleton_length': np.count_nonzero(filtered_skeleton),
        'blue_skeleton_length': np.count_nonzero(blue_mask),
        'pink_skeleton_length': np.count_nonzero(pink_mask)
    }

def save_summary_plot(image_path, gray, filtered_mask, skeleton, blue_mask, pink_mask, 
                     thick_mask, thin_mask, threshold, config, distance_map=None):
    """Save summary plot for the image."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create 2x2 figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Brighten original image
    from skimage import exposure
    gray_rescaled = exposure.rescale_intensity(np.asarray(gray, dtype=np.float32), 
                                              in_range="image", out_range='float').astype(np.float32)
    gray_brighter = np.clip(gray_rescaled ** 0.5, 0, 1)
    
    # Original image (brightened)
    axs[0,0].imshow(gray_brighter, cmap='gray')
    axs[0,0].set_title('Original Image (Brightened)')
    axs[0,0].axis('off')
    
    # Filtered mask
    axs[0,1].imshow(filtered_mask, cmap='gray')
    axs[0,1].set_title('Filtered Mask')
    axs[0,1].axis('off')
    
    # Debug: Print mask statistics
    print(f"    Debug - Filtered mask pixels: {np.sum(filtered_mask)}")
    print(f"    Debug - Skeleton pixels: {np.sum(skeleton)}")
    print(f"    Debug - Blue mask pixels: {np.sum(blue_mask)}")
    print(f"    Debug - Pink mask pixels: {np.sum(pink_mask)}")
    
    # Thickness filtering visualization - show excluded thick regions in yellow
    thickness_filtered_colored = np.zeros((*skeleton.shape, 3))
    
    # Calculate thickness mask for visualization
    if distance_map is not None and np.sum(skeleton) > 0:
        thickness_values = np.zeros_like(skeleton, dtype=float)
        thickness_values[skeleton] = distance_map[skeleton]
        
        # Show different thickness categories
        valid_thickness = (thickness_values >= FILTER_PARAMS["MIN_THICKNESS"]) & (thickness_values <= FILTER_PARAMS["MAX_THICKNESS"]) & skeleton
        excluded_thick = (thickness_values > FILTER_PARAMS["MAX_THICKNESS"]) & skeleton
        excluded_thin = (thickness_values < FILTER_PARAMS["MIN_THICKNESS"]) & skeleton
        
        # Color coding: Blue for normal, Pink for high branch density, Yellow for excluded thick
        thickness_filtered_colored[blue_mask] = [0, 0, 1]  # Blue for normal regions
        thickness_filtered_colored[pink_mask] = [1, 0, 1]  # Pink for high branch density
        thickness_filtered_colored[excluded_thick] = [1, 1, 0]   # Yellow for excluded (too thick)
        thickness_filtered_colored[excluded_thin] = [0.5, 0.5, 0.5]  # Gray for excluded (too thin)
        
        print(f"    Thickness filtering - Valid: {np.sum(valid_thickness)}, Too thick (yellow): {np.sum(excluded_thick)}, Too thin (gray): {np.sum(excluded_thin)}")
    else:
        # Fallback if no distance map - show blue/pink regions
        thickness_filtered_colored[blue_mask] = [0, 0, 1]  # Blue
        thickness_filtered_colored[pink_mask] = [1, 0, 1]  # Pink
        if np.sum(blue_mask) == 0 and np.sum(pink_mask) == 0:
            thickness_filtered_colored[skeleton] = [0.5, 0.5, 0.5]  # Gray for all skeleton
    
    axs[1,0].imshow(thickness_filtered_colored)
    axs[1,0].set_title('Skeleton: Blue=Normal, Pink=High Density, Yellow=Too Thick')
    axs[1,0].axis('off')
    
    # Thick vs thin (if enabled) or empty panel
    if config.enable_thick_thin_analysis and thick_mask is not None:
        thick_thin_colored = np.zeros((*skeleton.shape, 3))
        thick_thin_colored[thick_mask] = [1, 0, 0]  # Red for thick
        thick_thin_colored[thin_mask] = [0, 0, 1]   # Blue for thin
        axs[1,1].imshow(thick_thin_colored)
        axs[1,1].set_title('Thick vs Thin Analysis')
        axs[1,1].axis('off')
    else:
        # Empty panel
        axs[1,1].axis('off')
        axs[1,1].set_title('Thick vs Thin Analysis (Disabled)')
    
    # Add title with threshold
    fig.suptitle(f'{base_name} - Threshold: {threshold:.4f}', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(config.get_output_dir("images"), f"{base_name}_summary.png")
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close()

def run_analysis(config: AnalysisConfig):
    """Run the slide scanning L1CAM analysis."""
    
    print("=== Slide Scanning L1CAM Analysis ===")
    print(f"Input directory: {config.input_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Conditions: {config.conditions}")
    print(f"Groups: {config.groups}")
    
    # Collect image files
    image_files_info = []
    
    if config.use_hierarchical_structure:
        print(f"\nUsing hierarchical structure: Groups -> Conditions -> Images")
        
        for group in config.groups:
            group_path = os.path.join(config.input_dir, group)
            if not os.path.exists(group_path):
                print(f"Warning: Group directory not found: {group_path}")
                continue
                
            for condition in config.conditions:
                condition_path = os.path.join(group_path, condition)
                if not os.path.exists(condition_path):
                    print(f"Warning: Condition directory not found: {condition_path}")
                    continue
                
                # Find image files
                for file in os.listdir(condition_path):
                    if any(file.lower().endswith(ext) for ext in config.image_formats):
                        file_path = os.path.join(condition_path, file)
                        image_files_info.append({
                            'path': file_path,
                            'filename': file,
                            'group': group,
                            'condition': condition,
                            'biological_replicate': group
                        })
        
        print(f"\nFound {len(image_files_info)} images across {len(config.groups)} groups and {len(config.conditions)} conditions")
    else:
        # Flat structure
        for file in os.listdir(config.input_dir):
            if any(file.lower().endswith(ext) for ext in config.image_formats):
                file_path = os.path.join(config.input_dir, file)
                image_files_info.append({
                    'path': file_path,
                    'filename': file,
                    'group': None,
                    'condition': None,
                    'biological_replicate': None
                })
        print(f"\nFound {len(image_files_info)} images to process")
    
    # Process images
    results = []
    thickness_data = {condition: [] for condition in config.conditions}
    blue_thickness_data = {condition: [] for condition in config.conditions}
    pink_thickness_data = {condition: [] for condition in config.conditions}
    
    print(f"\n=== Processing Images ===")
    for i, file_info in enumerate(image_files_info, 1):
        print(f"\nProcessing image {i}/{len(image_files_info)}: {file_info['filename']}")
        
        result = process_single_image(
            file_info['path'], 
            config, 
            file_info['condition'], 
            file_info['biological_replicate']
        )
        
        if result:
            results.append(result)
            
            # Collect thickness data for CDFs
            if len(result['all_thickness']) > 0:
                thickness_data[result['condition']].append(result['all_thickness'])
            if len(result['blue_thickness']) > 0:
                blue_thickness_data[result['condition']].append(result['blue_thickness'])
            if len(result['pink_thickness']) > 0:
                pink_thickness_data[result['condition']].append(result['pink_thickness'])
    
    # Generate summary statistics
    print(f"\n=== Generating Results ===")
    
    # Save individual results
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(config.get_output_dir("results"), "individual_image_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Saved individual results: {results_csv}")
    
    # Generate CDF plots
    generate_cdf_plots(thickness_data, "All Skeleton", config)
    generate_cdf_plots(blue_thickness_data, "Blue Regions", config)
    generate_cdf_plots(pink_thickness_data, "Pink Regions", config)
    
    # Generate thick vs thin analysis if enabled
    if config.enable_thick_thin_analysis:
        generate_thick_thin_analysis(results_df, config)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {config.output_dir}")

def generate_cdf_plots(thickness_data, region_name, config):
    """Generate CDF plots for thickness data."""
    plt.figure(figsize=(10, 6))
    
    for condition in config.conditions:
        if condition not in thickness_data or len(thickness_data[condition]) == 0:
            continue
        
        # Combine all thickness values for this condition
        all_values = np.concatenate(thickness_data[condition])
        if len(all_values) == 0:
            continue
        
        # Calculate CDF
        sorted_values = np.sort(all_values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        
        # Plot
        plt.plot(sorted_values, cdf, label=f"{condition} (n={len(all_values)})", 
                color=config.colors.get(condition, 'gray'), linewidth=2)
    
    plt.xlabel('Thickness (pixels)')
    plt.ylabel('Cumulative Probability')
    plt.title(f'{region_name} Thickness CDF by Condition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f"{region_name.lower().replace(' ', '_')}_thickness_cdf.png"
    output_path = os.path.join(config.get_output_dir("plots"), filename)
    plt.savefig(output_path, dpi=config.plot_dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved CDF plot: {output_path}")

def generate_thick_thin_analysis(results_df, config):
    """Generate thick vs thin analysis results."""
    if 'ratio_thick_thin' not in results_df.columns:
        return
    
    # Filter out invalid ratios
    valid_results = results_df[results_df['ratio_thick_thin'].notna()]
    
    if len(valid_results) == 0:
        print("No valid thick vs thin data found")
        return
    
    # Save thick vs thin data
    thick_thin_csv = os.path.join(config.get_output_dir("results"), "thick_thin_analysis.csv")
    valid_results.to_csv(thick_thin_csv, index=False)
    print(f"Saved thick vs thin analysis: {thick_thin_csv}")
    
    # Generate summary plot
    plt.figure(figsize=(10, 6))
    
    for condition in config.conditions:
        condition_data = valid_results[valid_results['condition'] == condition]
        if len(condition_data) > 0:
            plt.scatter([condition] * len(condition_data), condition_data['ratio_thick_thin'],
                       color=config.colors.get(condition, 'gray'), alpha=0.7, s=50)
    
    plt.ylabel('Thick:Thin Ratio')
    plt.xlabel('Condition')
    plt.title('Thick vs Thin Ratio by Condition')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(config.get_output_dir("plots"), "thick_thin_ratio_by_condition.png")
    plt.savefig(output_path, dpi=config.plot_dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved thick vs thin plot: {output_path}")

def main():
    """Main function to handle command line arguments and run analysis"""
    parser = argparse.ArgumentParser(description='Slide Scanning L1CAM Analysis')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to configuration YAML file. If not provided, uses default configuration.')
    parser.add_argument('--input-dir', '-i', type=str, default=None,
                       help='Override input directory from config')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Override output directory from config')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config if args.config else 'default settings'}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Override config with command line arguments if provided
    if args.input_dir:
        config.input_dir = args.input_dir
        print(f"Overriding input directory: {config.input_dir}")
    
    if args.output_dir:
        config.output_dir = args.output_dir
        print(f"Overriding output directory: {config.output_dir}")
    
    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"Configuration validation error: {e}")
        sys.exit(1)
    
    # Ensure output directories exist
    config.ensure_output_dirs()
    
    # Run analysis
    run_analysis(config)

if __name__ == "__main__":
    main()
