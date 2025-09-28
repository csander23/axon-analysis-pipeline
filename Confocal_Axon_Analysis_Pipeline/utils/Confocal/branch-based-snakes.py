from aicsimageio import AICSImage
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.filters import threshold_otsu
import os
from skimage.morphology import skeletonize, remove_small_objects, label, closing, opening, disk, dilation, reconstruction
from skimage.measure import regionprops
from skimage.color import label2rgb
from scipy.ndimage import gaussian_filter, convolve, distance_transform_edt
import pandas as pd
import re
from collections import defaultdict
from scipy.stats import ecdf
import glob
import scipy.ndimage as ndimage
import sys
import argparse
from config_manager import load_config, AnalysisConfig


def run_analysis(config: AnalysisConfig):
    """Run the L1CAM analysis with the provided configuration"""
    
    # ---- CONFIGURATION ----
    input_dir = config.input_dir
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    min_sizes = config.min_sizes
    conditions = config.conditions
    colors = config.colors
    
    print("=== L1CAM Analysis - Sliding Window Processing ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Min sizes to analyze: {min_sizes}")
    print(f"Conditions: {conditions}")
    
    # Load threshold parameters
    print("\n=== Threshold Parameters ===")
    if config.use_raw_threshold:
        print("Using raw threshold:")
        print(f"• Fixed threshold value: {config.raw_threshold_value:.1f}")
    elif config.use_regression_model:
        import json
        print("Using active regression model:")
        with open(config.regression_model_path, 'r') as f:
            model_data = json.load(f)
        print(f"• Intercept: {model_data['intercept']:.4f}")
        print(f"• Coefficient: {model_data['metric_coefficient']:.4f}")
        print(f"• Percentile: {config.percentile_threshold}")
        print(f"• Replicate offsets: {config.replicate_offsets}")
    else:
        print("Using manual regression parameters:")
        print(f"• Intercept: {config.threshold_intercept:.4f}")
        print(f"• Coefficient: {config.threshold_coefficient:.4f}")
        print(f"• Percentile: {config.percentile_threshold}")
        print(f"• Replicate offsets: {config.replicate_offsets}")
    print("=" * 30 + "\n")

    # Collect ND2 files based on structure type
    nd2_files_info = []
    
    if config.use_hierarchical_structure:
        print(f"\nUsing hierarchical structure: Groups -> Conditions -> ND2 files")
        print(f"Groups: {config.groups}")
        print(f"Conditions: {config.conditions}")
        
        # Scan hierarchical structure
        for group in config.groups:
            group_path = os.path.join(input_dir, group)
            if not os.path.exists(group_path):
                print(f"Warning: Group directory not found: {group_path}")
                continue
                
            for condition in config.conditions:
                condition_path = os.path.join(group_path, condition)
                if not os.path.exists(condition_path):
                    print(f"Warning: Condition directory not found: {condition_path}")
                    continue
                
                # Find ND2 files in this condition directory
                for file in os.listdir(condition_path):
                    if file.endswith(".nd2"):
                        file_path = os.path.join(condition_path, file)
                        nd2_files_info.append({
                            'path': file_path,
                            'filename': file,
                            'group': group,
                            'condition': condition,
                            'biological_replicate': group  # Group serves as biological replicate
                        })
        
        print(f"\nFound {len(nd2_files_info)} ND2 files across {len(config.groups)} groups and {len(config.conditions)} conditions")
    else:
        # Original flat structure
        nd2_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".nd2")]
        for nd2_path in nd2_paths:
            nd2_files_info.append({
                'path': nd2_path,
                'filename': os.path.basename(nd2_path),
                'group': None,
                'condition': None,
                'biological_replicate': None
            })
        print(f"\nFound {len(nd2_files_info)} ND2 files to process")

    # Store results
    thickness_data = {min_size: defaultdict(list) for min_size in min_sizes}
    blue_thickness_data = {min_size: defaultdict(list) for min_size in min_sizes}
    pink_thickness_data = {min_size: defaultdict(list) for min_size in min_sizes}
    avg_thickness_per_image = {min_size: {} for min_size in min_sizes}
    image_conditions = {}
    
    # Store individual component data
    component_data = {min_size: [] for min_size in min_sizes}

    for file_idx, file_info in enumerate(nd2_files_info, 1):
        nd2_path = file_info['path']
        base_name = file_info['filename']
        print(f"\n--- Processing file {file_idx}/{len(nd2_files_info)}: {base_name} ---")
        
        if config.use_hierarchical_structure:
            # Use structure-based information
            cond = file_info['condition']
            biological_replicate = file_info['group']  # Group is the biological replicate
            print(f"  Group (Biological Replicate): {biological_replicate}")
            print(f"  Condition: {cond}")
        else:
            # Extract condition from filename (legacy method)
            cond_match = re.search(r'(KO|WT|E2|E4)', base_name, flags=re.IGNORECASE)
            cond = cond_match.group(1).upper() if cond_match else 'UNK'
            
            # Extract biological replicate from filename (legacy method)
            rep_match = re.search(r'(B114|B115|B116|B117)', base_name)
            biological_replicate = rep_match.group(1) if rep_match else 'UNK'
            print(f"  Detected condition: {cond}")
            print(f"  Detected biological replicate: {biological_replicate}")
        
        image_conditions[base_name] = cond
        
        print("  Loading image data...")
        img = AICSImage(nd2_path)
        try:
            green_index = img.channel_names.index(config.fitc_channel_name)
            print(f"  Found {config.fitc_channel_name} channel at index {green_index}")
        except ValueError:
            raise ValueError(f"Could not find '{config.fitc_channel_name}' channel in: " + str(img.channel_names))
        
        print("  Extracting green channel and creating max projection...")
        green_channel = img.get_image_data("CZYX")[green_index]
        max_proj = np.max(green_channel, axis=0)
        smooth_proj = gaussian_filter(max_proj, sigma=config.gaussian_sigma)
        print(f"  Max projection shape: {max_proj.shape}")
        
        # Thresholding
        if config.use_raw_threshold:
            print("  Using raw threshold...")
            threshold = config.raw_threshold_value
            print(f"  Fixed threshold: {threshold:.2f}")
        else:
            print("  Calculating adaptive threshold...")
            vals = smooth_proj.ravel()
            thr15 = np.percentile(vals, config.percentile_threshold)
            above15 = vals[vals > thr15]
            mean_above15 = above15.mean() if above15.size > 0 else 0.0
            
            if config.use_hierarchical_structure:
                # Use group (biological replicate) for threshold offset
                rep_offset = config.replicate_offsets.get(biological_replicate, 0.0)
                print(f"  Group {biological_replicate}, threshold offset: {rep_offset}")
            else:
                # Legacy: extract from filename
                rep_match = re.search(r'(B114|B115|B116|B117)', base_name)
                if rep_match:
                    rep = rep_match.group(1).upper()
                    rep_offset = config.replicate_offsets.get(rep, 0.0)
                    print(f"  Batch {rep}, threshold offset: {rep_offset}")
                else:
                    rep_offset = 0.0
                    print("  No replicate offset found, using 0.0")
            
            # Calculate threshold
            if config.use_regression_model:
                threshold = model_data['intercept'] + model_data['metric_coefficient'] * mean_above15 + rep_offset
            else:
                threshold = config.threshold_intercept + config.threshold_coefficient * mean_above15 + rep_offset
            print(f"  Calculated threshold: {threshold:.2f}")
        
        thresholded = smooth_proj > threshold
        print(f"  Thresholded pixels: {np.sum(thresholded)}")
        
        # Remove soma regions if TRITC channel exists
        print(f"  Checking for {config.tritc_channel_name} channel (soma removal)...")
        try:
            red_index = img.channel_names.index(config.tritc_channel_name)
            print(f"  Found {config.tritc_channel_name} channel at index {red_index}")
        except ValueError:
            red_index = None
            print(f"  No {config.tritc_channel_name} channel found, skipping soma removal")
        
        if red_index is not None:
            print(f"  Processing {config.tritc_channel_name} channel for soma removal...")
            red_channel = img.get_image_data("CZYX")[red_index]
            red_proj_original = np.max(red_channel, axis=0)
            
            # Apply Gaussian blur to the red projection
            red_blur = gaussian_filter(red_proj_original, sigma=config.soma_gaussian_sigma)
            
            # Compute Otsu on the blurred projection
            blur_otsu = threshold_otsu(red_blur)
            red_mask = red_blur > blur_otsu
            print(f"  Initial {config.tritc_channel_name} mask pixels: {np.sum(red_mask)}")
            
            # Distance-transform based soma extraction
            dist = distance_transform_edt(red_mask)
            seeds = dist >= config.distance_threshold
            soma_mask = reconstruction(seeds.astype(np.uint8), red_mask.astype(np.uint8), method='dilation').astype(bool)
            soma_mask = opening(soma_mask, disk(config.opening_disk_size))
            soma_mask = dilation(soma_mask, disk(config.dilation_disk_size))
            
            print(f"  Final soma mask pixels: {np.sum(soma_mask)}")
        else:
            soma_mask = None
        
        for min_size in min_sizes:
            print(f"    Processing min_size={min_size}...")
            
            print("      Removing small objects...")
            filtered_mask = remove_small_objects(thresholded, min_size=min_size)
            print(f"      Pixels after small object removal: {np.sum(filtered_mask)}")
            
            print("      Applying morphological operations...")
            filtered_mask = opening(filtered_mask, disk(config.opening_disk_size_filter))
            filtered_mask = closing(filtered_mask, disk(config.closing_disk_size_filter))
            
            # Store the mask before soma removal for visualization
            l1cam_binary_before_soma = filtered_mask.copy()
            
            if soma_mask is not None:
                filtered_mask = filtered_mask & (~soma_mask)
                print("      Applied soma removal")
            
            print("      Creating skeleton...")
            skeleton = skeletonize(filtered_mask)
            print(f"      Skeleton pixels: {np.sum(skeleton)}")
            
            # Compute branch points
            print("      Computing branch points...")
            neighbor_kernel = np.array([[1,1,1], [1,0,1], [1,1,1]])
            neighbor_count = convolve(skeleton.astype(int), neighbor_kernel, mode="constant", cval=0)
            branch_points = (skeleton & (neighbor_count >= 3))
            branch_count = np.count_nonzero(branch_points)
            total_length = np.count_nonzero(skeleton)
            connectivity = branch_count / total_length if total_length else 0
            print(f"      Branch points: {branch_count}, Connectivity: {connectivity:.4f}")
            
            print("      Calculating distance transform...")
            dist_map = distance_transform_edt(filtered_mask)
            radius_map = np.zeros_like(dist_map)
            radius_map[skeleton] = dist_map[skeleton]
            
            # Thick vs Thin Analysis (if enabled)
            if config.enable_thick_thin_analysis:
                print("      Performing thick vs thin analysis...")
                # Count how many branch points fall within the specified radius of each pixel
                kernel = disk(config.branch_distance_threshold).astype(int)
                branch_neighbor_count = convolve(branch_points.astype(int), kernel, mode='constant', cval=0)

                # Initial wide regions based on radius threshold
                initial_wide_mask = (radius_map >= config.width_threshold) & skeleton

                # Re‑label as thin if the pixel is within range of too many branch points
                near_many_branches = branch_neighbor_count >= config.branch_count_threshold
                wide_far_from_branch = initial_wide_mask & ~near_many_branches
                labeled_wide = label(wide_far_from_branch)
                filtered_wide_mask = np.zeros_like(initial_wide_mask, dtype=bool)

                # Filter small wide regions
                for region in regionprops(labeled_wide):
                    if region.area >= config.min_wide_region_size:
                        for y, x in region.coords:
                            filtered_wide_mask[y, x] = True

                wide_mask = filtered_wide_mask
                thin_mask = skeleton & ~wide_mask

                # Compute wide vs thin statistics
                wide_count = np.count_nonzero(wide_mask)
                thin_count = np.count_nonzero(thin_mask)
                ratio_wide_thin = wide_count / thin_count if thin_count > 0 else np.nan

                # Create color-coded thick vs thin visualization
                h_img, w_img = max_proj.shape
                thick_thin_img = np.zeros((h_img, w_img, 3), dtype=float)
                thick_thin_img[wide_mask, 0] = 1.0  # red for thick
                thick_thin_img[thin_mask, 2] = 1.0  # blue for thin

                # Thick vs thin visualization is now part of the summary plot only

                # Add thick vs thin data to component data
                component_data[min_size].append({
                    'image': base_name,
                    'component_id': 'thick_thin',
                    'wide_count': wide_count,
                    'thin_count': thin_count,
                    'ratio_wide_thin': ratio_wide_thin,
                    'condition': cond,
                    'biological_replicate': biological_replicate
                })

            # Create summary plot components
            print("      Creating summary plot...")
            max_proj_rescaled = exposure.rescale_intensity(np.asarray(max_proj, dtype=np.float32), in_range="image", out_range='float').astype(np.float32)
            max_proj_brighter = np.clip(max_proj_rescaled ** 0.5, 0, 1)
            
            # Brighten radius map more aggressively for better visibility
            radius_map_rescaled = exposure.rescale_intensity(np.asarray(radius_map, dtype=np.float32), in_range="image", out_range='float').astype(np.float32)
            radius_map_brighter = np.clip(radius_map_rescaled ** 0.3, 0, 1)  # More aggressive brightening (0.3 instead of 0.5)
            
            # Label skeleton components
            labeled_skel, num = label(skeleton, return_num=True, connectivity=2)
            print(f"      Found {num} skeleton components")
            
            # Create mask of remaining components (after min_size filtering)
            remaining_skeleton = np.zeros_like(skeleton, dtype=bool)
            for i in range(1, num+1):
                comp = (labeled_skel == i)
                if np.sum(comp) >= min_size:
                    remaining_skeleton = remaining_skeleton | comp
            
            print(f"      Remaining skeleton pixels after min_size={min_size} filtering: {np.sum(remaining_skeleton)}")
            
            # Spider analysis on remaining skeleton
            pink_mask = np.zeros(skeleton.shape, dtype=bool)
            
            # Helper function for spider analysis
            def spider_pixels(start_pix):
                """Return all skeleton coords reachable within window_length steps."""
                window_len = config.window_length
                reached, frontier = {start_pix}, {start_pix}
                for _ in range(window_len):
                    new_frontier = set()
                    for y, x in frontier:
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dy == dx == 0:
                                    continue
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]
                                        and remaining_skeleton[ny, nx] and (ny, nx) not in reached):
                                    reached.add((ny, nx))
                                    new_frontier.add((ny, nx))
                    if not new_frontier:
                        break
                    frontier = new_frontier
                return reached

            # Iterate over branch pixels in the remaining skeleton
            branch_coords = np.column_stack(np.where(branch_points & remaining_skeleton))
            print(f"      Analyzing {len(branch_coords)} branch points with spiders...")
            
            for by, bx in branch_coords:
                spider = spider_pixels((by, bx))
                spider_list = list(spider)
                # Calculate metrics inside this spider
                branch_cnt = 0
                thick_vals = []
                for sy, sx in spider_list:
                    if neighbor_count[sy, sx] >= 3:
                        branch_cnt += 1
                    thick_vals.append(radius_map[sy, sx])
                density = branch_cnt / len(spider_list) if spider_list else 0.0
                avg_thick = np.mean(thick_vals) if thick_vals else 0.0

                # Pink criterion
                if density > config.pink_density_threshold and avg_thick >= config.pink_thickness_threshold:
                    for sy, sx in spider_list:
                        pink_mask[sy, sx] = True

            # Color the skeleton
            colored_skel = np.zeros((*skeleton.shape, 3), dtype=np.float32)
            colored_skel[skeleton & (~remaining_skeleton)] = [1, 1, 1]  # White for excluded
            colored_skel[remaining_skeleton] = [0.3, 0.7, 1.0]  # Light blue for remaining
            colored_skel[pink_mask] = [1, 0, 1]  # Pink for high branch density
            
            # Process first file with extra debug info
            sliding_window_debug = (file_idx == 1)

            # Process individual components
            total_windows = 0
            for i in range(1, num+1):
                comp = (labeled_skel == i)
                comp = np.asarray(comp, dtype=bool)
                coords = np.column_stack(np.where(comp))
                if len(coords) == 0:
                    continue
                
                # Skip components that are too small
                if np.sum(comp) < min_size:
                    continue
                
                # Progress indicator
                if num > 50 and i % 50 == 0:
                    print(f"        Processing component {i}/{num} ({len(coords)} pixels)... [Progress: {i/num*100:.1f}%]")
                elif num <= 50:
                    print(f"        Processing component {i}/{num} ({len(coords)} pixels)...")
                
                # Branch-based spider snakes
                window_len = config.window_length
                pink_spider_mask = np.zeros_like(skeleton, dtype=bool)

                # Helper function for component-specific spider analysis
                def spider_pixels_comp(start_pix):
                    """Return all skeleton coords reachable within window_len steps."""
                    reached, frontier = {start_pix}, {start_pix}
                    for _ in range(window_len):
                        new_frontier = set()
                        for y, x in frontier:
                            for dy in (-1, 0, 1):
                                for dx in (-1, 0, 1):
                                    if dy == dx == 0:
                                        continue
                                    ny, nx = y + dy, x + dx
                                    if (0 <= ny < comp.shape[0] and 0 <= nx < comp.shape[1]
                                            and comp[ny, nx] and (ny, nx) not in reached):
                                        reached.add((ny, nx))
                                        new_frontier.add((ny, nx))
                        if not new_frontier:
                            break
                        frontier = new_frontier
                    return reached

                # Iterate over branch pixels in this component
                branch_coords = np.column_stack(np.where(branch_points & comp))
                for by, bx in branch_coords:
                    spider = spider_pixels_comp((by, bx))
                    spider_list = list(spider)
                    # Calculate metrics inside this spider
                    branch_cnt = 0
                    thick_vals = []
                    for sy, sx in spider_list:
                        if neighbor_count[sy, sx] >= 3:
                            branch_cnt += 1
                        thick_vals.append(radius_map[sy, sx])
                    density = branch_cnt / len(spider_list) if spider_list else 0.0
                    avg_thick = np.mean(thick_vals) if thick_vals else 0.0

                    # Pink criterion
                    if density > config.pink_density_threshold and avg_thick >= config.pink_thickness_threshold:
                        for sy, sx in spider_list:
                            pink_spider_mask[sy, sx] = True


                # Merge new pink spiders with global pink_mask
                pink_mask |= pink_spider_mask

            print(f"      Total sliding windows created: {total_windows}")
            print(f"      Component processing complete for min_size={min_size}")
            colored_skel[pink_mask] = [1, 0, 1]
            
            # Always save summary plots
            print("      Saving summary plot...")
            
            # l1cam_binary_before_soma is already created during processing
            # filtered_mask is the L1CAM binary after soma removal
            l1cam_minus_soma = filtered_mask
            
            if config.enable_thick_thin_analysis:
                # 3x4 layout like legacy version with thick vs thin
                fig, axs = plt.subplots(3, 4, figsize=(32, 18))
                
                # Row 1: Initial processing steps
                axs[0,0].imshow(max_proj_brighter, cmap='gray')
                axs[0,0].set_title("L1CAM Max Intensity Projection")
                
                if red_index is not None:
                    red_proj_rescaled = exposure.rescale_intensity(red_proj_original, in_range='image', out_range=(0, 1))
                    red_proj_brighter = np.clip(red_proj_rescaled ** 0.5, 0, 1)
                    axs[0,1].imshow(red_proj_brighter, cmap='gray')
                else:
                    axs[0,1].imshow(np.zeros_like(max_proj), cmap='gray')
                axs[0,1].set_title("MAP2 Max Intensity Projection")
                
                axs[0,2].imshow(np.asarray(l1cam_binary_before_soma, dtype=np.uint8), cmap='gray', vmin=0, vmax=1)
                axs[0,2].set_title("L1CAM Binary (before soma removal)")
                
                if soma_mask is not None:
                    axs[0,3].imshow(np.asarray(soma_mask, dtype=np.uint8), cmap='gray', vmin=0, vmax=1)
                else:
                    axs[0,3].imshow(np.zeros_like(max_proj, dtype=bool).astype(np.uint8), cmap='gray', vmin=0, vmax=1)
                axs[0,3].set_title("Soma Binary")
                
                # Row 2: Processing results
                axs[1,0].imshow(np.asarray(l1cam_minus_soma, dtype=np.uint8), cmap='gray', vmin=0, vmax=1)
                axs[1,0].set_title("L1CAM Binary Minus Soma")
                
                axs[1,1].imshow(colored_skel)
                axs[1,1].set_title("Skeleton: Blue=Included, Pink=High Branch Density")
                
                axs[1,2].imshow(radius_map_brighter, cmap="inferno")
                axs[1,2].set_title("Skeleton Radius Map")
                
                axs[1,3].imshow(thick_thin_img)
                axs[1,3].set_title(f"Thick vs Thin (ratio={ratio_wide_thin:.2f})")
                
                # Row 3: Empty for now, can be used for additional analysis
                for j in range(4):
                    axs[2,j].axis('off')
                    
            else:
                # 2x4 layout without thick vs thin
                fig, axs = plt.subplots(2, 4, figsize=(32, 12))
                
                # Row 1: Initial processing steps
                axs[0,0].imshow(max_proj_brighter, cmap='gray')
                axs[0,0].set_title("L1CAM Max Intensity Projection")
                
                if red_index is not None:
                    red_proj_rescaled = exposure.rescale_intensity(red_proj_original, in_range='image', out_range=(0, 1))
                    red_proj_brighter = np.clip(red_proj_rescaled ** 0.5, 0, 1)
                    axs[0,1].imshow(red_proj_brighter, cmap='gray')
                else:
                    axs[0,1].imshow(np.zeros_like(max_proj), cmap='gray')
                axs[0,1].set_title("MAP2 Max Intensity Projection")
                
                axs[0,2].imshow(np.asarray(l1cam_binary_before_soma, dtype=np.uint8), cmap='gray', vmin=0, vmax=1)
                axs[0,2].set_title("L1CAM Binary (before soma removal)")
                
                if soma_mask is not None:
                    axs[0,3].imshow(np.asarray(soma_mask, dtype=np.uint8), cmap='gray', vmin=0, vmax=1)
                else:
                    axs[0,3].imshow(np.zeros_like(max_proj, dtype=bool).astype(np.uint8), cmap='gray', vmin=0, vmax=1)
                axs[0,3].set_title("Soma Binary")
                
                # Row 2: Processing results
                axs[1,0].imshow(np.asarray(l1cam_minus_soma, dtype=np.uint8), cmap='gray', vmin=0, vmax=1)
                axs[1,0].set_title("L1CAM Binary Minus Soma")
                
                axs[1,1].imshow(colored_skel)
                axs[1,1].set_title("Skeleton: Blue=Included, Pink=High Branch Density")
                
                axs[1,2].imshow(radius_map_brighter, cmap="inferno")
                axs[1,2].set_title("Skeleton Radius Map")
                
                axs[1,3].axis('off')  # Empty panel
            
            # Turn off axes for all panels
            for i in range(axs.shape[0]):
                for j in range(axs.shape[1]):
                    axs[i,j].axis("off")
            
            # Add title with threshold information first, then adjust layout
            if config.use_raw_threshold:
                fig.suptitle(f"Fixed Threshold: {threshold:.1f} (Raw Threshold)", fontsize=18, y=0.95)
            elif config.use_regression_model:
                fig.suptitle(f"Calculated Threshold: {threshold:.1f} (Regression Model)", fontsize=18, y=0.95)
            else:
                fig.suptitle(f"Calculated Threshold: {threshold:.1f} (Manual Parameters)", fontsize=18, y=0.95)
            
            # Adjust layout to make room for title and ensure visibility
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)  # Leave more space at top for title
            
            # Save to Images directory
            image_dir = config.get_output_dir("images")
            os.makedirs(image_dir, exist_ok=True)
            plt.savefig(os.path.join(image_dir, f"{base_name}_summary.png"), dpi=config.dpi)
            plt.close(fig)
            
            # Calculate thickness statistics
            print("      Calculating thickness statistics...")

            # All-skeleton thickness for legacy plots (only remaining skeleton)
            thickness_vals = radius_map[remaining_skeleton]
            thickness_data[min_size][cond].append(thickness_vals)
            avg_thick = thickness_vals.mean() if thickness_vals.size else 0.0
            avg_thickness_per_image[min_size][base_name] = avg_thick
            print(f"      Avg thickness (remaining skeleton): {avg_thick:.2f} px")

            # Blue-only thickness (exclude pink regions from remaining skeleton)
            blue_mask = remaining_skeleton & (~pink_mask)
            blue_vals = radius_map[blue_mask]
            blue_vals = blue_vals[blue_vals > 0]

            if blue_vals.size:
                blue_thickness_data[min_size][cond].append(blue_vals)
                avg_blue = blue_vals.mean()
                print(f"      Avg blue-region thickness:   {avg_blue:.2f} px  (n={blue_vals.size})")
            else:
                print("      No blue pixels with non-zero thickness")

            # Pink-only thickness (high branch density regions from remaining skeleton)
            pink_vals = radius_map[pink_mask]
            pink_vals = pink_vals[pink_vals > 0]

            if pink_vals.size:
                pink_thickness_data[min_size][cond].append(pink_vals)
                avg_pink = pink_vals.mean()
                print(f"      Avg pink-region thickness:   {avg_pink:.2f} px  (n={pink_vals.size})")
            else:
                print("      No pink pixels with non-zero thickness")
            

            # Create component visualization with different colors for each blue component
            if sliding_window_debug and min_size != 0:
                print(f"      Creating component visualization...")
                
                component_img = np.zeros((*skeleton.shape, 3), dtype=np.float32)
                
                # Get all remaining skeleton components (blue components) AFTER removing pink regions
                blue_only_skeleton = remaining_skeleton & (~pink_mask)
                labeled_components = label(blue_only_skeleton)
                if isinstance(labeled_components, tuple):
                    labeled_components = labeled_components[0]
                num_components = int(labeled_components.max()) if labeled_components.max() > 0 else 0
                
                print(f"        Found {num_components} blue components (after removing pink regions)")
                
                # Generate distinct colors for each component
                import colorsys
                colors_list = []
                for i in range(num_components):
                    golden_ratio = config.golden_ratio
                    hue = (i * golden_ratio) % 1.0
                    saturation = config.component_saturation
                    value = config.component_value
                    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                    colors_list.append(rgb)
                
                # Color each component with a different color
                for component_id in range(1, num_components + 1):
                    component_mask = (labeled_components == component_id)
                    component_img[component_mask] = colors_list[component_id - 1]
                
                # Make pink regions gray
                component_img[pink_mask] = [0.5, 0.5, 0.5]
                
                # Component visualization removed - no longer needed

            # Collect individual component data for ALL images
            print(f"      Collecting component data...")
            blue_only_skeleton = remaining_skeleton & (~pink_mask)
            labeled_components = label(blue_only_skeleton)
            if isinstance(labeled_components, tuple):
                labeled_components = labeled_components[0]
            num_components = int(labeled_components.max()) if labeled_components.max() > 0 else 0
            
            print(f"        Found {num_components} blue components (after removing pink regions)")
            
            # Collect individual component data
            for component_id in range(1, num_components + 1):
                component_mask = (labeled_components == component_id)
                component_thickness_vals = radius_map[component_mask]
                if component_thickness_vals.size > 0:
                    avg_component_thickness = np.mean(component_thickness_vals)
                    component_data[min_size].append({
                        'image': base_name,
                        'component_id': component_id,
                        'avg_thickness': avg_component_thickness,
                        'condition': cond,
                        'biological_replicate': biological_replicate
                    })

    print("\n=== Creating Summary Plots ===")
    # Overlay CDFs for all conditions for each min_size (ALL SKELETON)
    for min_size in min_sizes:
        print(f"  Creating overlay CDF for min_size={min_size}...")
        plt.figure(figsize=(8,6))
        
        for cond in conditions:
            if len(thickness_data[min_size][cond]) == 0:
                continue
                
            print(f"    Processing condition {cond} with {len(thickness_data[min_size][cond])} images...")
            
            # Calculate individual image CDFs
            image_cdfs = []
            for thickness_vals in thickness_data[min_size][cond]:
                if len(thickness_vals) == 0:
                    continue
                sorted_vals = np.sort(thickness_vals)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                image_cdfs.append((sorted_vals, cdf))
            
            if len(image_cdfs) == 0:
                continue
                
            # Find the range of thickness values across all images
            all_thicknesses = []
            for thickness_vals in thickness_data[min_size][cond]:
                all_thicknesses.extend(thickness_vals)
            
            min_thickness = min(all_thicknesses)
            max_thickness = max(all_thicknesses)
            
            # Create a common x-axis
            common_x = np.linspace(min_thickness, max_thickness, 1000)
            
            # Interpolate each image's CDF
            from scipy.interpolate import interp1d
            interpolated_cdfs = []
            
            for sorted_vals, cdf in image_cdfs:
                interp_func = interp1d(sorted_vals, cdf, 
                                      bounds_error=False, 
                                      fill_value=(0, 1))
                interpolated_cdf = interp_func(common_x)
                interpolated_cdfs.append(interpolated_cdf)
            
            # Average the CDFs
            average_cdf = np.mean(interpolated_cdfs, axis=0)
            
            # Plot the averaged CDF
            plt.plot(common_x, average_cdf, color=colors[cond], label=f'{cond} (n={len(image_cdfs)} images)', linewidth=2)
        
        plt.xlabel('Thickness (pixels)')
        plt.ylabel('Cumulative Probability')
        plt.title(f'Overlay CDFs by Condition (min_size={min_size}) - Equal Image Weighting')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.get_output_dir("plots"), f'overall_condition_overlay.png'), dpi=config.plot_dpi)
        plt.close()

    # Blue-only overlay CDFs
    print("\n=== Creating BLUE-ONLY Summary Plots ===")
    for min_size in min_sizes:
        print(f"  Creating BLUE-ONLY overlay CDF for min_size={min_size}...")
        plt.figure(figsize=(8,6))
        
        for cond in conditions:
            if len(blue_thickness_data[min_size][cond]) == 0:
                continue
                
            print(f"    Processing BLUE condition {cond} with {len(blue_thickness_data[min_size][cond])} images...")
            
            # Calculate individual image CDFs
            image_cdfs = []
            for blue_vals in blue_thickness_data[min_size][cond]:
                if len(blue_vals) == 0:
                    continue
                sorted_vals = np.sort(blue_vals)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                image_cdfs.append((sorted_vals, cdf))
            
            if len(image_cdfs) == 0:
                continue
                
            # Find the range of thickness values across all images
            all_thicknesses = []
            for blue_vals in blue_thickness_data[min_size][cond]:
                all_thicknesses.extend(blue_vals)
            
            min_thickness = min(all_thicknesses)
            max_thickness = max(all_thicknesses)
            
            # Create a common x-axis
            common_x = np.linspace(min_thickness, max_thickness, 1000)
            
            # Interpolate each image's CDF
            from scipy.interpolate import interp1d
            interpolated_cdfs = []
            
            for sorted_vals, cdf in image_cdfs:
                interp_func = interp1d(sorted_vals, cdf, 
                                    bounds_error=False, 
                                    fill_value=(0, 1))
                interpolated_cdf = interp_func(common_x)
                interpolated_cdfs.append(interpolated_cdf)
            
            # Average the CDFs
            average_cdf = np.mean(interpolated_cdfs, axis=0)
            
            # Plot the averaged CDF
            plt.plot(common_x, average_cdf, color=colors[cond], label=f'{cond} BLUE (n={len(image_cdfs)} images)', linewidth=2)
        
        plt.xlabel('Thickness (pixels)')
        plt.ylabel('Cumulative Probability')
        plt.title(f'BLUE-ONLY Overlay CDFs by Condition (min_size={min_size}) - Equal Image Weighting')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.get_output_dir("plots"), f'blue_only_overall_condition_overlay.png'), dpi=config.plot_dpi)
        plt.close()

    # Pink-only overlay CDFs
    print("\n=== Creating PINK-ONLY Summary Plots ===")
    for min_size in min_sizes:
        print(f"  Creating PINK-ONLY overlay CDF for min_size={min_size}...")
        plt.figure(figsize=(8,6))
        
        for cond in conditions:
            if len(pink_thickness_data[min_size][cond]) == 0:
                continue
                
            print(f"    Processing PINK condition {cond} with {len(pink_thickness_data[min_size][cond])} images...")
            
            # Calculate individual image CDFs
            image_cdfs = []
            for pink_vals in pink_thickness_data[min_size][cond]:
                if len(pink_vals) == 0:
                    continue
                sorted_vals = np.sort(pink_vals)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                image_cdfs.append((sorted_vals, cdf))
            
            if len(image_cdfs) == 0:
                continue
                
            # Find the range of thickness values across all images
            all_thicknesses = []
            for pink_vals in pink_thickness_data[min_size][cond]:
                all_thicknesses.extend(pink_vals)
            
            min_thickness = min(all_thicknesses)
            max_thickness = max(all_thicknesses)
            
            # Create a common x-axis
            common_x = np.linspace(min_thickness, max_thickness, 1000)
            
            # Interpolate each image's CDF
            from scipy.interpolate import interp1d
            interpolated_cdfs = []
            
            for sorted_vals, cdf in image_cdfs:
                interp_func = interp1d(sorted_vals, cdf, 
                                    bounds_error=False, 
                                    fill_value=(0, 1))
                interpolated_cdf = interp_func(common_x)
                interpolated_cdfs.append(interpolated_cdf)
            
            # Average the CDFs
            average_cdf = np.mean(interpolated_cdfs, axis=0)
            
            # Plot the averaged CDF
            plt.plot(common_x, average_cdf, color=colors[cond], label=f'{cond} PINK (n={len(image_cdfs)} images)', linewidth=2)
        
        plt.xlabel('Thickness (pixels)')
        plt.ylabel('Cumulative Probability')
        plt.title(f'PINK-ONLY Overlay CDFs by Condition (min_size={min_size}) - Equal Image Weighting')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.get_output_dir("plots"), f'pink_only_overall_condition_overlay.png'), dpi=config.plot_dpi)
        plt.close()

    # Normalized component CDFs
    print("\n=== Creating Normalized Component CDFs ===")
    for min_size in min_sizes:
        print(f"  Creating normalized component data for min_size={min_size}...")
        if component_data[min_size]:
            df_components = pd.DataFrame(component_data[min_size])
            
            # Calculate WT average for each biological replicate
            wt_means = {}
            for rep in df_components['biological_replicate'].unique():
                rep_wt_data = df_components[(df_components['biological_replicate'] == rep) & 
                                        (df_components['condition'] == 'WT')]
                if len(rep_wt_data) > 0:
                    wt_means[rep] = rep_wt_data['avg_thickness'].mean()
                else:
                    print(f"    Warning: No WT data found for replicate {rep}")
            
            # Normalize all components by their biological replicate's WT average
            df_normalized = df_components.copy()
            df_normalized['normalized_thickness'] = df_normalized.apply(
                lambda row: row['avg_thickness'] / wt_means.get(row['biological_replicate'], 1.0), 
                axis=1
            )
            
            # Save normalized data
            csv_path = os.path.join(config.get_output_dir("results"), f'normalized_component_data_min{min_size}.csv')
            df_normalized.to_csv(csv_path, index=False)
            print(f"    Saved normalized component data: {csv_path}")
            
            # Create CDF plot for normalized data
            print(f"    Creating normalized CDF plot...")
            plt.figure(figsize=(10, 6))
            
            for cond in conditions:
                cond_data = df_normalized[df_normalized['condition'] == cond]
                if len(cond_data) > 0:
                    # Calculate CDF for this condition
                    sorted_vals = np.sort(cond_data['normalized_thickness'])
                    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                    
                    plt.plot(sorted_vals, cdf, color=colors[cond], 
                            label=f'{cond} (n={len(cond_data)} components)', linewidth=2)
            
            plt.xlabel('Normalized Thickness (relative to WT average)')
            plt.ylabel('Cumulative Probability')
            plt.title(f'Normalized Component Thickness CDFs (min_size={min_size})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plot_path = os.path.join(config.get_output_dir("plots"), f'normalized_component_cdf.png')
            plt.savefig(plot_path, dpi=config.plot_dpi, bbox_inches='tight')
            plt.close()
            print(f"    Saved normalized CDF plot: {plot_path}")
        else:
            print(f"    No component data found for min_size={min_size}")

    print("\n=== Generating CSV Reports ===")
    # CSVs for average thickness per image and per condition
    for min_size in min_sizes:
        print(f"  Creating CSV reports for min_size={min_size}...")
        # Per image
        df_img = pd.DataFrame.from_records(list(avg_thickness_per_image[min_size].items()), columns=['image', 'avg_thickness'])
        df_img['condition'] = df_img['image'].apply(lambda x: image_conditions.get(x, 'UNK'))
        csv_path = os.path.join(config.get_output_dir("results"), f'avg_thickness_per_image_min{min_size}.csv')
        df_img.to_csv(csv_path, index=False)
        print(f"    Saved per-image data: {csv_path}")
        
        # Per condition
        df_cond = df_img.groupby('condition')['avg_thickness'].mean().reset_index()
        csv_path = os.path.join(config.get_output_dir("results"), f'avg_thickness_per_condition_min{min_size}.csv')
        df_cond.to_csv(csv_path, index=False)
        print(f"    Saved per-condition data: {csv_path}")

    # Generate component-level CSV with biological replicate
    print("\n=== Generating Component-Level CSV Reports ===")
    for min_size in min_sizes:
        print(f"  Creating component-level CSV for min_size={min_size}...")
        if component_data[min_size]:
            df_components = pd.DataFrame(component_data[min_size])
            
            # Split thick vs thin data into separate CSV if enabled
            if config.enable_thick_thin_analysis:
                df_thick_thin = df_components[df_components['component_id'] == 'thick_thin'].copy()
                df_components = df_components[df_components['component_id'] != 'thick_thin'].copy()
                
                # Save thick vs thin data
                thick_thin_csv = os.path.join(config.get_output_dir("results"), f'thick_thin_data_min{min_size}.csv')
                df_thick_thin.to_csv(thick_thin_csv, index=False)
                print(f"    Saved thick vs thin data: {thick_thin_csv}")
                
                if config.normalize_to_wt:
                    # Normalize to WT per replicate
                    wt_ratios = df_thick_thin[df_thick_thin['condition'] == 'WT'].groupby('biological_replicate')['ratio_wide_thin'].mean()
                    df_thick_thin['normalized_ratio'] = df_thick_thin.apply(
                        lambda r: (r['ratio_wide_thin'] / wt_ratios[r['biological_replicate']]) * 100 
                        if r['biological_replicate'] in wt_ratios.index else np.nan, 
                        axis=1
                    )
                    
                    # Save normalized data
                    norm_csv = os.path.join(config.get_output_dir("results"), f'normalized_thick_thin_data_min{min_size}.csv')
                    df_thick_thin.to_csv(norm_csv, index=False)
                    print(f"    Saved normalized thick vs thin data: {norm_csv}")
                    
                    # Create summary statistics by condition
                    summary = df_thick_thin.groupby('condition').agg({
                        'ratio_wide_thin': ['mean', 'std', 'count'],
                        'normalized_ratio': ['mean', 'std']
                    }).round(3)
                    summary.columns = ['ratio_mean', 'ratio_std', 'n', 'norm_ratio_mean', 'norm_ratio_std']
                    summary = summary.reset_index()
                    
                    # Save summary statistics
                    summary_csv = os.path.join(config.get_output_dir("results"), f'thick_thin_summary_min{min_size}.csv')
                    summary.to_csv(summary_csv, index=False)
                    print(f"    Saved thick vs thin summary: {summary_csv}")
                    
                    # Perform statistical analysis
                    from scipy.stats import f_oneway
                    conditions = ['WT', 'KO', 'E2', 'E4']
                    samples = {c: df_thick_thin[df_thick_thin['condition'] == c]['normalized_ratio'].dropna().values 
                             for c in conditions}
                    valid_samples = [s for s in samples.values() if len(s) > 0]
                    
                    if len(valid_samples) >= 2:
                        f_stat, p_val = f_oneway(*valid_samples)
                        stats_txt = os.path.join(config.get_output_dir("results"), f'thick_thin_stats_min{min_size}.txt')
                        with open(stats_txt, 'w') as f:
                            f.write(f"One-way ANOVA results:\n")
                            f.write(f"F-statistic: {f_stat:.4f}\n")
                            f.write(f"p-value: {p_val:.4f}\n")
                        print(f"    Saved statistical analysis: {stats_txt}")
            
            # Save regular component data
            csv_path = os.path.join(config.get_output_dir("results"), f'component_level_data_min{min_size}.csv')
            df_components.to_csv(csv_path, index=False)
            print(f"    Saved component-level data: {csv_path}")
            print(f"    Total components: {len(df_components)}")
        else:
            print(f"    No component data found for min_size={min_size}")

    print("\n=== Analysis Complete! ===")
    print(f"All results saved to: {output_dir}")
    print("\nGenerated files in Images/:")
    print("  - Summary plots for each analyzed image")
    print("\nGenerated files in Plots/:")
    print("  - Overall condition overlay CDFs")
    print("  - Blue-only and pink-only overlay CDFs")
    print("  - Normalized component CDFs")
    print("\nGenerated files in Results/:")
    print("  - Component-level data and statistics")
    print("  - Normalized thickness data")
    if config.enable_thick_thin_analysis:
        print("  - Thick vs thin analysis data")
        print("  - Normalized thick vs thin ratios")
        print("  - Statistical analysis results")


def main():
    """Main function to handle command line arguments and run analysis"""
    parser = argparse.ArgumentParser(description='L1CAM Analysis - Branch-based Snakes')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to configuration YAML file. If not provided, uses default configuration.')
    parser.add_argument('--input-dir', '-i', type=str, default=None,
                       help='Override input directory from config')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Override output directory name from config')
    
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
        config.output_dir_name = args.output_dir
        print(f"Overriding output directory name: {config.output_dir_name}")
    
    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"Configuration validation error: {e}")
        sys.exit(1)
    
    # Run analysis
    run_analysis(config)


if __name__ == "__main__":
    main()
