#!/usr/bin/env python3
"""
Standalone script to generate condition-based CDF plots.
This is the inverse of biological replicate plots - each condition gets its own subplot,
with lines representing different biological replicates.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def safe_convert_to_numeric(data_list):
    """Convert a list of values to numeric, removing non-numeric values."""
    numeric_data = []
    for value in data_list:
        try:
            numeric_value = float(value)
            if not np.isnan(numeric_value) and not np.isinf(numeric_value):
                numeric_data.append(numeric_value)
        except (ValueError, TypeError):
            continue
    return np.array(numeric_data)

def generate_condition_based_cdf_plots(output_dir, design_name, threshold_type="individual"):
    """
    Generate CDF plots showing individual lines for every biological replicate within each condition.
    Each condition gets its own subplot, with lines representing different biological replicates.
    This is the inverse of the biological replicate plots.
    """
    # Create design-specific output directory
    design_output_dir = os.path.join(output_dir, design_name)
    os.makedirs(design_output_dir, exist_ok=True)
    
    # Load the comprehensive file mapping to get biological replicate info
    mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
    design_mapping = mapping_df[mapping_df['experimental_design'] == design_name]
    
    # Base colors for each biological replicate (consistent across conditions)
    replicate_colors = {
        'B8': '#FF6B6B',    # Red
        'B10': '#4ECDC4',   # Teal
        'B11': '#45B7D1',   # Blue
        'B12': '#96CEB4',   # Green
        'B1': '#FFEAA7',    # Yellow
        'B2': '#DDA0DD',    # Plum
        'B3': '#F0A500',    # Orange
        'B27': '#C0392B'    # Dark red for combined groups
    }
    
    # Organize data by condition and biological replicate from CSV files
    condition_replicate_data = {}
    
    # Process CSV files to reconstruct condition-replicate data
    design_output_dir_csv = os.path.join("comprehensive_cdf_analysis_results", design_name)
    
    print(f"  Processing CSV files for {design_name} with {threshold_type} thresholding...")
    print(f"  Looking in directory: {design_output_dir_csv}")
    
    files_processed = 0
    files_found = 0
        
    for _, row in design_mapping.iterrows():
        filename = row['filename']
        condition = row['condition']
        biological_replicate = row['biological_replicate']
        
        # Extract base filename (remove .jpg extension)
        base_filename = os.path.splitext(filename)[0]
        
        # Construct path to CSV file based on threshold type
        csv_dir = "csv_files_individual" if threshold_type == "individual" else "csv_files_batch"
        csv_path = os.path.join(design_output_dir_csv, condition, csv_dir, f"{base_filename}_blue_components.csv")
        
        # Check if CSV file exists
        if os.path.exists(csv_path):
            files_found += 1
            try:
                # Read CSV data
                df = pd.read_csv(csv_path)
                if not df.empty and 'average_thickness' in df.columns:
                    thickness_values = df['average_thickness'].tolist()
                    
                    # Initialize condition in condition_replicate_data if needed
                    if condition not in condition_replicate_data:
                        condition_replicate_data[condition] = {}
                    if biological_replicate not in condition_replicate_data[condition]:
                        condition_replicate_data[condition][biological_replicate] = []
                    
                    # Add thickness values for this image
                    condition_replicate_data[condition][biological_replicate].extend(thickness_values)
                    files_processed += 1
                    
                    if files_processed <= 3:  # Debug first few files
                        print(f"    Processed {base_filename}: {len(thickness_values)} thickness values")
                    
            except Exception as e:
                print(f"    Warning: Error processing {base_filename}: {e}")
                continue
        else:
            if files_found <= 3:  # Debug first few missing files
                print(f"    File not found: {csv_path}")
    
    print(f"  Found {files_found} CSV files, processed {files_processed} files successfully")
    print(f"  Condition replicate data structure: {list(condition_replicate_data.keys()) if condition_replicate_data else 'Empty'}")
    
    if not condition_replicate_data:
        print(f"  No valid condition replicate data found for {design_name}")
        return
        
    # Get all conditions and biological replicates
    conditions = sorted(list(condition_replicate_data.keys()))
    all_replicates = set()
    for condition_data in condition_replicate_data.values():
        all_replicates.update(condition_data.keys())
    biological_replicates = sorted(list(all_replicates))
    
    # Create the plot - 2x2 layout for 4 conditions (genotypes design)
    if design_name == "genotypes" and len(conditions) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    elif len(conditions) <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    else:
        # For more conditions, create a larger grid
        n_cols = 3
        n_rows = (len(conditions) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
    
    # Individual condition plots
    for i, condition in enumerate(conditions):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        print(f"  Plotting condition {condition}")
        
        # Plot CDF for each biological replicate within this condition
        if condition in condition_replicate_data:
            bio_reps = condition_replicate_data[condition]
            
            for bio_rep, thickness_data in bio_reps.items():
                if thickness_data:
                    # Convert to numeric and remove invalid values
                    numeric_data = safe_convert_to_numeric(thickness_data)
                    
                    if len(numeric_data) > 0:
                        # Sort data for CDF
                        sorted_data = np.sort(numeric_data)
                        
                        # Calculate CDF
                        n = len(sorted_data)
                        y_values = np.arange(1, n + 1) / n
                        
                        # Get color for this biological replicate
                        rep_color = replicate_colors.get(bio_rep, '#808080')  # Default gray
                        
                        print(f"    Plotting replicate {bio_rep}: {len(numeric_data)} data points")
                        
                        # Plot CDF line for this biological replicate
                        ax.plot(sorted_data, y_values, 
                               color=rep_color, 
                               linewidth=2.5, 
                               alpha=0.8,
                               label=f'{bio_rep} (n={len(numeric_data)})')
                    else:
                        print(f"    Warning: No valid data for replicate {bio_rep} in condition {condition}")
                else:
                    print(f"    Warning: No thickness data for replicate {bio_rep} in condition {condition}")
        
        ax.set_xlabel('Average Thickness (pixels)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title(f'{condition}\n({threshold_type.title()} Thresholding)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Only add legend if there are any lines plotted
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=10, loc='best')
    
    # Remove empty subplots
    for i in range(len(conditions), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'{design_name}_condition_based_cdf_{threshold_type}.png'
    plot_path = os.path.join(design_output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Condition-based CDF plot saved: {plot_path}")
    
    # Save replicate data to CSV
    csv_data = []
    for condition in conditions:
        if condition in condition_replicate_data:
            for bio_rep, thickness_data in condition_replicate_data[condition].items():
                numeric_data = safe_convert_to_numeric(thickness_data)
                for thickness in numeric_data:
                    csv_data.append({
                        'condition': condition,
                        'biological_replicate': bio_rep,
                        'thickness': thickness,
                        'threshold_type': threshold_type
                    })
    
    csv_filename = f'{design_name}_condition_based_cdf_data_{threshold_type}.csv'
    csv_path = os.path.join(design_output_dir, csv_filename)
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_path, index=False)
    
    print(f"Condition-based CDF data saved: {csv_path}")
    print(f"Total data points: {len(csv_data)}")


if __name__ == "__main__":
    # Set working directory
    os.chdir("/Users/charlessander/Desktop/patzke lab computing/Sudhof Lab/L1CAM P2")
    
    # Generate condition-based CDF plots for genotypes with both threshold types
    output_dir = "comprehensive_cdf_analysis_results"
    
    print("Generating condition-based CDF plots for genotypes...")
    generate_condition_based_cdf_plots(output_dir, "genotypes", "individual")
    generate_condition_based_cdf_plots(output_dir, "genotypes", "batch")
    
    print("\nGeneration complete!") 