#!/usr/bin/env python3
"""
Blue Component Filter Script

This script applies preset filter combinations to blue component data from the CSV output
of final_analyzer_cdf_branch_snakes.py. Each component must pass ALL filters in a combination
to be included in the analysis.

Usage:
    python blue_component_filter.py

Configuration:
    - Edit the input_dir and output_dir variables in the main() function
    - Modify the filter_combinations list to add/remove/change filter combinations
    - Run the script to apply all preset filter combinations automatically

Available Filter Types:
    - max_thickness: Filter by maximum thickness <= filter_value
    - min_thickness: Filter by minimum thickness >= filter_value  
    - avg_thickness: Filter by average thickness <= filter_value
    - skeleton_length: Filter by skeleton length >= filter_value
    - skeleton_length_max: Filter by skeleton length <= filter_value
    - mother_length: Filter by mother component length >= filter_value
    - mother_length_max: Filter by mother component length <= filter_value

Color Scheme:
    - Control groups use lighter colors
    - Simvastatin groups use darker colors
    - Each genotype has distinct colors for easy comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import re
from collections import defaultdict
from scipy.stats import ks_2samp
import itertools

def load_blue_component_data(input_dir):
    """
    Load all blue component CSV files from the input directory.
    Returns a dictionary mapping group names to lists of DataFrames.
    """
    print(f"Loading blue component data from: {input_dir}")
    
    # Define the 8 groups
    groups = {
        'KO_Control': 'KO_Control',
        'KO_Simvastatin': 'KO_Simvastatin', 
        'ApoE4_Control': 'ApoE4_Control',
        'ApoE4_Simvastatin': 'ApoE4_Simvastatin',
        'Control_Control': 'Control_Control',
        'Control_Simvastatin': 'Control_Simvastatin',
        'ApoE2_Control': 'ApoE2_Control',
        'ApoE2_Simvastatin': 'ApoE2_Simvastatin'
    }
    
    group_data = defaultdict(list)
    
    # Search for blue component CSV files in each group's csv_files directory
    for group_key, group_name in groups.items():
        group_csv_dir = os.path.join(input_dir, group_name, "csv_files")
        if os.path.exists(group_csv_dir):
            # Find all blue component CSV files
            blue_csv_pattern = os.path.join(group_csv_dir, "*_blue_components.csv")
            blue_csv_files = glob.glob(blue_csv_pattern)
            
            for csv_file in blue_csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    # Add image name column for tracking
                    image_name = os.path.basename(csv_file).replace('_blue_components.csv', '')
                    df['image_name'] = image_name
                    df['group'] = group_name
                    group_data[group_name].append(df)
                    print(f"  Loaded: {image_name} ({len(df)} components)")
                except Exception as e:
                    print(f"  Error loading {csv_file}: {e}")
        else:
            print(f"  Warning: Directory not found: {group_csv_dir}")
    
    return group_data

def apply_combined_filters(data, filters):
    """
    Apply all filters together to the blue component data.
    Each component must pass ALL filters to be included.
    
    Filter types:
    - 'max_thickness': Filter by maximum thickness <= filter_value
    - 'min_thickness': Filter by minimum thickness >= filter_value
    - 'avg_thickness': Filter by average thickness within range
    - 'skeleton_length': Filter by skeleton length >= filter_value
    - 'skeleton_length_max': Filter by skeleton length <= filter_value
    - 'mother_length': Filter by mother component length >= filter_value
    - 'mother_length_max': Filter by mother component length <= filter_value
    """
    print(f"\nApplying combined filters:")
    for filter_type, filter_value, description in filters:
        print(f"  - {description}")
    
    # Hard-coded image exclusion
    excluded_images = [
        "ApoE_Statins_B10_Simvastatin_GroupB_Red-2",
        "ApoE_Statins_B7_Control_GroupC_Red-4"
    ]
    print(f"  - Excluding images: {', '.join(excluded_images)}")
    
    filtered_data = defaultdict(list)
    
    for group_name, group_dfs in data.items():
        for df in group_dfs:
            # Check if this dataframe belongs to any excluded image
            if 'image_name' in df.columns:
                should_skip = False
                for excluded_image in excluded_images:
                    if excluded_image in df['image_name'].values:
                        print(f"  Skipping excluded image: {excluded_image}")
                        should_skip = True
                        break
                if should_skip:
                    continue
            
            filtered_df = df.copy()
            
            # Apply each filter sequentially
            for filter_type, filter_value, description in filters:
                if filter_type == 'max_thickness':
                    filtered_df = filtered_df[filtered_df['max_thickness'] <= filter_value]
                elif filter_type == 'min_thickness':
                    filtered_df = filtered_df[filtered_df['min_thickness'] >= filter_value]
                elif filter_type == 'avg_thickness':
                    if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                        min_val, max_val = filter_value
                        filtered_df = filtered_df[(filtered_df['average_thickness'] >= min_val) & 
                                               (filtered_df['average_thickness'] <= max_val)]
                    else:
                        filtered_df = filtered_df[filtered_df['average_thickness'] <= filter_value]
                elif filter_type == 'skeleton_length':
                    filtered_df = filtered_df[filtered_df['skeleton_length'] >= filter_value]
                elif filter_type == 'skeleton_length_max':
                    filtered_df = filtered_df[filtered_df['skeleton_length'] <= filter_value]
                elif filter_type == 'mother_length':
                    filtered_df = filtered_df[filtered_df['mother_component_length'] >= filter_value]
                elif filter_type == 'mother_length_max':
                    filtered_df = filtered_df[filtered_df['mother_component_length'] <= filter_value]
                else:
                    print(f"  Warning: Unknown filter type '{filter_type}', skipping filter")
            
            if len(filtered_df) > 0:
                filtered_data[group_name].append(filtered_df)
                print(f"  {group_name}: {len(df)} -> {len(filtered_df)} components")
            else:
                print(f"  {group_name}: {len(df)} -> 0 components (all filtered out)")
    
    return filtered_data

def generate_cdf_plots(filtered_data, output_dir, filter_type, filter_value):
    """Generate CDF plots for the filtered blue component data."""
    print(f"\nGenerating CDF plots...")
    
    # Define color scheme with different colors for Control vs Simvastatin
    # Reorganized in logical order: WT, ApoE4, ApoE2, KO
    color_scheme = {
        'Control_Control': '#66B3FF',  # Light blue
        'Control_Simvastatin': '#0066CC', # Dark blue
        'ApoE4_Control': '#FFB366',   # Light orange
        'ApoE4_Simvastatin': '#FF8000', # Dark orange
        'ApoE2_Control': '#90EE90',   # Light green
        'ApoE2_Simvastatin': '#228B22', # Dark green
        'KO_Control': '#FF6B6B',      # Light red
        'KO_Simvastatin': '#CC0000',  # Dark red
    }
    
    # Define color scheme for Control-only plot (reorganized)
    control_color_scheme = {
        'Control_Control': '#66B3FF',  # Blue
        'ApoE4_Control': '#FFB366',   # Orange
        'ApoE2_Control': '#90EE90',   # Green
        'KO_Control': '#FF6B6B',      # Red
    }
    
    # Define logical order for plotting
    plot_order = ['Control_Control', 'Control_Simvastatin', 'ApoE4_Control', 'ApoE4_Simvastatin', 'ApoE2_Control', 'ApoE2_Simvastatin', 'KO_Control', 'KO_Simvastatin']
    control_order = ['Control_Control', 'ApoE4_Control', 'ApoE2_Control', 'KO_Control']
    
    # Collect all average thickness values by group
    group_thicknesses = defaultdict(list)
    
    for group_name, group_dfs in filtered_data.items():
        for df in group_dfs:
            group_thicknesses[group_name].extend(df['average_thickness'].tolist())
    
    # Create CDF plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    filter_label = f"{filter_type}={filter_value}"
    ax.set_title(f'Blue Component CDFs: Average Thickness\nFilter: {filter_label}', fontsize=14)
    
    # Plot in logical order to control legend order
    for group_name in plot_order:
        if group_name in group_thicknesses and group_thicknesses[group_name]:
            thicknesses = group_thicknesses[group_name]
            color = color_scheme.get(group_name, '#808080')  # Default gray if not found
            sorted_thicknesses = np.sort(thicknesses)
            cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
            
            ax.plot(sorted_thicknesses, cdf, color=color, linewidth=3, 
                   label=f'{group_name} (n={len(thicknesses)} components)')
    
    ax.set_xlabel('Component Average Thickness (pixels)')
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    # Save plot
    safe_filter_label = filter_label.replace('=', '_').replace(' ', '_')
    plot_filename = f'blue_components_filtered_{safe_filter_label}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  CDF plot saved: {plot_path}")
    
    # Create Control-only plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    control_filter_label = f"{filter_type}={filter_value} (Control Only)"
    ax.set_title(f'Blue Component CDFs: Average Thickness\nFilter: {control_filter_label}', fontsize=14)
    
    # Plot Control conditions in logical order
    for group_name in control_order:
        if group_name in group_thicknesses and group_thicknesses[group_name] and group_name in control_color_scheme:
            thicknesses = group_thicknesses[group_name]
            color = control_color_scheme[group_name]
            sorted_thicknesses = np.sort(thicknesses)
            cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
            
            ax.plot(sorted_thicknesses, cdf, color=color, linewidth=3, 
                   label=f'{group_name} (n={len(thicknesses)} components)')
    
    ax.set_xlabel('Component Average Thickness (pixels)')
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    # Save Control-only plot
    control_plot_filename = f'blue_components_control_only_{safe_filter_label}.png'
    control_plot_path = os.path.join(output_dir, control_plot_filename)
    plt.savefig(control_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Control-only CDF plot saved: {control_plot_path}")
    
    return plot_path

def perform_statistical_analysis(filtered_data, output_dir, filter_type, filter_value):
    """Perform Kolmogorov-Smirnov tests between groups."""
    print(f"\nPerforming statistical analysis...")
    
    # Collect all average thickness values by group
    group_thicknesses = defaultdict(list)
    
    for group_name, group_dfs in filtered_data.items():
        for df in group_dfs:
            group_thicknesses[group_name].extend(df['average_thickness'].tolist())
    
    # Perform KS tests between all pairs of groups
    ks_results = []
    group_pairs = list(itertools.combinations(group_thicknesses.keys(), 2))
    
    for group1, group2 in group_pairs:
        if group_thicknesses[group1] and group_thicknesses[group2]:
            thicknesses1 = np.array(group_thicknesses[group1])
            thicknesses2 = np.array(group_thicknesses[group2])
            
            ks_statistic, p_value = ks_2samp(thicknesses1, thicknesses2)
            
            ks_results.append({
                'condition1': group1,
                'condition2': group2,
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'n1': len(thicknesses1),
                'n2': len(thicknesses2),
                'significant': p_value < 0.05
            })
            
            print(f"  {group1} vs {group2}: KS = {ks_statistic:.4f}, p = {p_value:.6f} {'*' if p_value < 0.05 else ''}")
    
    # Save KS test results
    if ks_results:
        ks_df = pd.DataFrame(ks_results)
        safe_filter_label = f"{filter_type}_{filter_value}".replace('=', '_').replace(' ', '_')
        csv_filename = f'ks_test_results_filtered_{safe_filter_label}.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        ks_df.to_csv(csv_path, index=False)
        print(f"  KS test results saved: {csv_path}")
    
    return ks_results

def save_filtered_data(filtered_data, output_dir, filter_type, filter_value):
    """Save the filtered data to CSV files."""
    print(f"\nSaving filtered data...")
    
    safe_filter_label = f"{filter_type}_{filter_value}".replace('=', '_').replace(' ', '_')
    
    for group_name, group_dfs in filtered_data.items():
        if group_dfs:
            # Combine all DataFrames for this group
            combined_df = pd.concat(group_dfs, ignore_index=True)
            
            # Save to CSV
            csv_filename = f'{group_name}_blue_components_filtered_{safe_filter_label}.csv'
            csv_path = os.path.join(output_dir, csv_filename)
            combined_df.to_csv(csv_path, index=False)
            print(f"  {group_name}: {len(combined_df)} components saved to {csv_filename}")

def generate_component_metric_cdfs(data, output_dir, filter_combination=None):
    """
    Generate CDFs for all component metrics (skeleton_length, mother_component_length, 
    average_thickness, max_thickness, min_thickness) and show where filters apply.
    """
    print(f"\nGenerating component metric CDFs...")
    
    # Define color scheme (reorganized in logical order)
    color_scheme = {
        'Control_Control': '#66B3FF',  # Light blue
        'Control_Simvastatin': '#0066CC', # Dark blue
        'ApoE4_Control': '#FFB366',   # Light orange
        'ApoE4_Simvastatin': '#FF8000', # Dark orange
        'ApoE2_Control': '#90EE90',   # Light green
        'ApoE2_Simvastatin': '#228B22', # Dark green
        'KO_Control': '#FF6B6B',      # Light red
        'KO_Simvastatin': '#CC0000',  # Dark red
    }
    
    # Define logical order for plotting
    plot_order = ['Control_Control', 'Control_Simvastatin', 'ApoE4_Control', 'ApoE4_Simvastatin', 'ApoE2_Control', 'ApoE2_Simvastatin', 'KO_Control', 'KO_Simvastatin']
    
    # Define metrics to analyze
    metrics = {
        'skeleton_length': 'Skeleton Length (pixels)',
        'mother_component_length': 'Mother Component Length (pixels)',
        'average_thickness': 'Average Thickness (pixels)',
        'max_thickness': 'Maximum Thickness (pixels)',
        'min_thickness': 'Minimum Thickness (pixels)'
    }
    
    # Collect all data by group and metric (optimized)
    group_metrics = defaultdict(lambda: defaultdict(list))
    
    for group_name, group_dfs in data.items():
        for df in group_dfs:
            if len(df) > 0:  # Skip empty dataframes
                for metric in metrics.keys():
                    if metric in df.columns:
                        # Use numpy array for better performance
                        values = df[metric].values
                        group_metrics[group_name][metric].extend(values.tolist())
    
    # Generate CDF for each metric
    for metric, metric_label in metrics.items():
        print(f"  Generating CDF for {metric}...")
        
        # Count total components for this metric
        total_components = sum(len(group_data[metric]) for group_data in group_metrics.values() if metric in group_data and group_data[metric])
        print(f"    Total components for {metric}: {total_components}")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot CDFs for each group in logical order
        for group_name in plot_order:
            if group_name in group_metrics and metric in group_metrics[group_name] and group_metrics[group_name][metric]:
                group_data = group_metrics[group_name]
                color = color_scheme.get(group_name, '#808080')
                values = np.array(group_data[metric])
                
                print(f"    Processing {group_name}: {len(values)} components")
                
                # Limit to first 10000 components for very large datasets to prevent memory issues
                if len(values) > 10000:
                    print(f"    Limiting {group_name} to first 10000 components (total: {len(values)})")
                    values = values[:10000]
                
                sorted_values = np.sort(values)
                cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
                
                ax.plot(sorted_values, cdf, color=color, linewidth=3, 
                       label=f'{group_name} (n={len(values)} components)')
        
        # Add filter lines if filters are provided
        if filter_combination:
            for filter_type, filter_value, description in filter_combination:
                if filter_type == metric or filter_type == f"{metric}_max":
                    if filter_type == metric:  # Lower bound
                        ax.axvline(x=filter_value, color='red', linestyle='--', alpha=0.7, 
                                 label=f'Filter: {description}')
                    elif filter_type == f"{metric}_max":  # Upper bound
                        ax.axvline(x=filter_value, color='red', linestyle='--', alpha=0.7, 
                                 label=f'Filter: {description}')
                elif filter_type == 'avg_thickness' and metric == 'average_thickness':
                    if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                        min_val, max_val = filter_value
                        ax.axvline(x=min_val, color='red', linestyle='--', alpha=0.7, 
                                 label=f'Filter: Min {description}')
                        ax.axvline(x=max_val, color='red', linestyle='--', alpha=0.7, 
                                 label=f'Filter: Max {description}')
                    else:
                        ax.axvline(x=filter_value, color='red', linestyle='--', alpha=0.7, 
                                 label=f'Filter: {description}')
        
        ax.set_xlabel(metric_label)
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Component {metric_label} CDFs\nBy Experimental Group')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        # Save plot
        safe_metric = metric.replace('_', '')
        plot_filename = f'component_{safe_metric}_cdfs.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    CDF plot saved: {plot_path}")
        print(f"    Completed {metric} CDF")

def generate_filter_application_plots(data, output_dir, filter_combination):
    """
    Generate plots showing how many components pass each filter and where filters apply.
    """
    print(f"\nGenerating filter application analysis...")
    
    # Define color scheme (reorganized in logical order)
    color_scheme = {
        'Control_Control': '#66B3FF',  # Light blue
        'Control_Simvastatin': '#0066CC', # Dark blue
        'ApoE4_Control': '#FFB366',   # Light orange
        'ApoE4_Simvastatin': '#FF8000', # Dark orange
        'ApoE2_Control': '#90EE90',   # Light green
        'ApoE2_Simvastatin': '#228B22', # Dark green
        'KO_Control': '#FF6B6B',      # Light red
        'KO_Simvastatin': '#CC0000',  # Dark red
    }
    
    # Define logical order for plotting
    plot_order = ['Control_Control', 'Control_Simvastatin', 'ApoE4_Control', 'ApoE4_Simvastatin', 'ApoE2_Control', 'ApoE2_Simvastatin', 'KO_Control', 'KO_Simvastatin']
    
    # Collect filter statistics
    filter_stats = defaultdict(lambda: defaultdict(dict))
    
    for group_name, group_dfs in data.items():
        total_components = 0
        for df in group_dfs:
            total_components += len(df)
        
        filter_stats[group_name]['total'] = total_components
        filter_stats[group_name]['filters'] = {}
        
        # Apply each filter individually and count passing components
        for filter_type, filter_value, description in filter_combination:
            passing_components = 0
            for df in group_dfs:
                if filter_type == 'max_thickness':
                    passing_components += len(df[df['max_thickness'] <= filter_value])
                elif filter_type == 'min_thickness':
                    passing_components += len(df[df['min_thickness'] >= filter_value])
                elif filter_type == 'avg_thickness':
                    if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                        min_val, max_val = filter_value
                        passing_components += len(df[(df['average_thickness'] >= min_val) & 
                                                   (df['average_thickness'] <= max_val)])
                    else:
                        passing_components += len(df[df['average_thickness'] <= filter_value])
                elif filter_type == 'skeleton_length':
                    passing_components += len(df[df['skeleton_length'] >= filter_value])
                elif filter_type == 'skeleton_length_max':
                    passing_components += len(df[df['skeleton_length'] <= filter_value])
                elif filter_type == 'mother_length':
                    passing_components += len(df[df['mother_component_length'] >= filter_value])
                elif filter_type == 'mother_length_max':
                    passing_components += len(df[df['mother_component_length'] <= filter_value])
            
            filter_stats[group_name]['filters'][description] = passing_components
    
    # Create filter application plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare data for plotting
    groups = list(filter_stats.keys())
    filter_descriptions = [desc for _, _, desc in filter_combination]
    
    # Create stacked bar chart
    x = np.arange(len(groups))
    width = 0.8
    
    # Calculate cumulative percentages
    cumulative_data = np.zeros((len(filter_descriptions), len(groups)))
    
    for i, desc in enumerate(filter_descriptions):
        for j, group in enumerate(groups):
            if i == 0:
                # First filter: percentage of total
                total = filter_stats[group]['total']
                passing = filter_stats[group]['filters'][desc]
                cumulative_data[i, j] = (passing / total) * 100 if total > 0 else 0
            else:
                # Subsequent filters: percentage of previous filter
                prev_desc = filter_descriptions[i-1]
                prev_passing = filter_stats[group]['filters'][prev_desc]
                current_passing = filter_stats[group]['filters'][desc]
                cumulative_data[i, j] = (current_passing / prev_passing) * 100 if prev_passing > 0 else 0
    
    # Plot stacked bars
    bottom = np.zeros(len(groups))
    colors = plt.cm.Set3(np.linspace(0, 1, len(filter_descriptions)))
    
    for i, desc in enumerate(filter_descriptions):
        ax.bar(x, cumulative_data[i], width, bottom=bottom, 
               label=desc, color=colors[i], alpha=0.8)
        bottom += cumulative_data[i]
    
    ax.set_xlabel('Experimental Groups')
    ax.set_ylabel('Percentage of Components Passing Filters')
    ax.set_title('Filter Application Analysis\nPercentage of Components Passing Each Filter')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'filter_application_analysis.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Filter application plot saved: {plot_path}")
    
    # Create summary table
    print(f"\nFilter Application Summary:")
    print("=" * 120)
    print(f"{'Group':<20} {'Total':<8}", end="")
    for desc in filter_descriptions:
        print(f" {'Passing':<8}", end="")
    print(f" {'Final':<8}")
    print("-" * 120)
    
    for group in groups:
        total = filter_stats[group]['total']
        print(f"{group:<20} {total:<8}", end="")
        
        prev_passing = total
        for desc in filter_descriptions:
            passing = filter_stats[group]['filters'][desc]
            print(f" {passing:<8}", end="")
            prev_passing = passing
        
        print(f" {prev_passing:<8}")
    
    print("=" * 120)

def generate_individual_filter_cdfs(data, output_dir):
    """
    Generate CDFs showing the impact of each individual filter type.
    """
    print(f"\nGenerating individual filter impact CDFs...")
    
    # Define color scheme (reorganized in logical order)
    color_scheme = {
        'Control_Control': '#66B3FF',  # Light blue
        'Control_Simvastatin': '#0066CC', # Dark blue
        'ApoE4_Control': '#FFB366',   # Light orange
        'ApoE4_Simvastatin': '#FF8000', # Dark orange
        'ApoE2_Control': '#90EE90',   # Light green
        'ApoE2_Simvastatin': '#228B22', # Dark green
        'KO_Control': '#FF6B6B',      # Light red
        'KO_Simvastatin': '#CC0000',  # Dark red
    }
    
    # Define logical order for plotting
    plot_order = ['Control_Control', 'Control_Simvastatin', 'ApoE4_Control', 'ApoE4_Simvastatin', 'ApoE2_Control', 'ApoE2_Simvastatin', 'KO_Control', 'KO_Simvastatin']
    
    # Define filter examples for each metric
    filter_examples = [
        ('max_thickness', 5, 'Max thickness <= 5'),
        ('min_thickness', 2, 'Min thickness >= 2'),
        ('avg_thickness', 4, 'Average thickness <= 4'),
        ('skeleton_length', 50, 'Skeleton length >= 50'),
        ('skeleton_length_max', 500, 'Skeleton length <= 500'),
        ('mother_length', 100, 'Mother length >= 100'),
        ('mother_length_max', 2000, 'Mother length <= 2000'),
    ]
    
    for filter_type, filter_value, description in filter_examples:
        print(f"  Generating CDF for {filter_type} filter...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Collect data for this filter type
        group_data = defaultdict(list)
        
        for group_name, group_dfs in data.items():
            for df in group_dfs:
                if filter_type in df.columns:
                    # Apply this specific filter
                    if filter_type == 'max_thickness':
                        filtered_df = df[df['max_thickness'] <= filter_value]
                    elif filter_type == 'min_thickness':
                        filtered_df = df[df['min_thickness'] >= filter_value]
                    elif filter_type == 'avg_thickness':
                        filtered_df = df[df['average_thickness'] <= filter_value]
                    elif filter_type == 'skeleton_length':
                        filtered_df = df[df['skeleton_length'] >= filter_value]
                    elif filter_type == 'skeleton_length_max':
                        filtered_df = df[df['skeleton_length'] <= filter_value]
                    elif filter_type == 'mother_length':
                        filtered_df = df[df['mother_component_length'] >= filter_value]
                    elif filter_type == 'mother_length_max':
                        filtered_df = df[df['mother_component_length'] <= filter_value]
                    else:
                        filtered_df = df
                    
                    if len(filtered_df) > 0:
                        group_data[group_name].extend(filtered_df['average_thickness'].tolist())
        
        # Plot CDFs in logical order
        for group_name in plot_order:
            if group_name in group_data and group_data[group_name]:
                thicknesses = group_data[group_name]
                color = color_scheme.get(group_name, '#808080')
                sorted_thicknesses = np.sort(thicknesses)
                cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                
                ax.plot(sorted_thicknesses, cdf, color=color, linewidth=3, 
                       label=f'{group_name} (n={len(thicknesses)} components)')
        
        # Add filter line
        if filter_type == 'avg_thickness':
            ax.axvline(x=filter_value, color='red', linestyle='--', alpha=0.7, 
                      label=f'Filter: {description}')
        
        ax.set_xlabel('Component Average Thickness (pixels)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Component CDFs After {description} Filter')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        # Save plot
        safe_filter = filter_type.replace('_', '')
        plot_filename = f'individual_filter_{safe_filter}_cdfs.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    CDF plot saved: {plot_path}")

def generate_component_count_cdfs(filtered_data, output_dir, filter_combination):
    """
    Generate CDFs showing the distribution of blue component counts per image for each condition.
    """
    print(f"\nGenerating component count per image CDFs...")
    
    # Define color scheme (reorganized in logical order)
    color_scheme = {
        'Control_Control': '#66B3FF',  # Light blue
        'Control_Simvastatin': '#0066CC', # Dark blue
        'ApoE4_Control': '#FFB366',   # Light orange
        'ApoE4_Simvastatin': '#FF8000', # Dark orange
        'ApoE2_Control': '#90EE90',   # Light green
        'ApoE2_Simvastatin': '#228B22', # Dark green
        'KO_Control': '#FF6B6B',      # Light red
        'KO_Simvastatin': '#CC0000',  # Dark red
    }
    
    # Define logical order for plotting
    plot_order = ['Control_Control', 'Control_Simvastatin', 'ApoE4_Control', 'ApoE4_Simvastatin', 'ApoE2_Control', 'ApoE2_Simvastatin', 'KO_Control', 'KO_Simvastatin']
    
    # Collect component counts per image by group
    group_component_counts = defaultdict(list)
    
    for group_name, group_dfs in filtered_data.items():
        for df in group_dfs:
            # Count components in this image
            component_count = len(df)
            group_component_counts[group_name].append(component_count)
    
    # Create CDF plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot CDFs for each group
    for group_name in plot_order:
        if group_name in group_component_counts and group_component_counts[group_name]:
            color = color_scheme.get(group_name, '#808080')
            sorted_counts = np.sort(group_component_counts[group_name])
            cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
            
            ax.plot(sorted_counts, cdf, color=color, linewidth=3, 
                   label=f'{group_name} (n={len(group_component_counts[group_name])} images)')
    
    # Add statistics
    print(f"  Component count statistics per image:")
    for group_name, counts in group_component_counts.items():
        if counts:
            mean_count = np.mean(counts)
            median_count = np.median(counts)
            min_count = np.min(counts)
            max_count = np.max(counts)
            print(f"    {group_name}: mean={mean_count:.1f}, median={median_count:.1f}, range=[{min_count}, {max_count}]")
    
    ax.set_xlabel('Number of Blue Components per Image')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Distribution of Blue Component Counts per Image\nAfter Filtering')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'component_count_per_image_cdfs.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Component count CDF plot saved: {plot_path}")
    
    # Also create a box plot for comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    for group_name, counts in group_component_counts.items():
        if counts:
            box_data.append(counts)
            box_labels.append(group_name)
    
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        for i, patch in enumerate(bp['boxes']):
            group_name = box_labels[i]
            color = color_scheme.get(group_name, '#808080')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_ylabel('Number of Blue Components per Image')
    ax.set_title('Blue Component Counts per Image by Condition\nAfter Filtering')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save box plot
    box_plot_filename = 'component_count_per_image_boxplot.png'
    box_plot_path = os.path.join(output_dir, box_plot_filename)
    plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Component count box plot saved: {box_plot_path}")

def generate_replicate_cdfs(filtered_data, output_dir, filter_type, filter_value):
    """
    Generate CDF plots showing each biological replicate separately.
    Each condition (genotype/treatment) will have multiple lines - one for each biological replicate.
    """
    print(f"\nGenerating replicate-specific CDFs...")
    
    # Define color scheme for conditions (reorganized in logical order)
    condition_colors = {
        'Control_Control': '#66B3FF',  # Light blue
        'Control_Simvastatin': '#0066CC', # Dark blue
        'ApoE4_Control': '#FFB366',   # Light orange
        'ApoE4_Simvastatin': '#FF8000', # Dark orange
        'ApoE2_Control': '#90EE90',   # Light green
        'ApoE2_Simvastatin': '#228B22', # Dark green
        'KO_Control': '#FF6B6B',      # Light red
        'KO_Simvastatin': '#CC0000',  # Dark red
    }
    
    # Define logical order for plotting
    plot_order = ['Control_Control', 'Control_Simvastatin', 'ApoE4_Control', 'ApoE4_Simvastatin', 'ApoE2_Control', 'ApoE2_Simvastatin', 'KO_Control', 'KO_Simvastatin']
    
    # Collect data by condition and replicate
    condition_replicate_data = defaultdict(lambda: defaultdict(list))
    
    for group_name, group_dfs in filtered_data.items():
        for df in group_dfs:
            if 'image_name' in df.columns and len(df) > 0:
                # Extract replicate from image name (e.g., "B10" from "ApoE_Statins_B10_Control_GroupC_Red")
                image_name = df['image_name'].iloc[0]
                replicate_match = re.search(r'B(\d+)', image_name)
                if replicate_match:
                    replicate = f"B{replicate_match.group(1)}"
                    # Use numpy array for better performance
                    thicknesses = df['average_thickness'].values
                    condition_replicate_data[group_name][replicate].extend(thicknesses.tolist())
    
    # Create CDF plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Track which conditions we've plotted for legend
    plotted_conditions = set()
    
    for condition, replicate_data_dict in condition_replicate_data.items():
        color = condition_colors.get(condition, '#808080')
        
        for replicate, thicknesses in replicate_data_dict.items():
            if thicknesses:
                # Use different line styles for different replicates within same condition
                replicate_num = int(replicate.replace('B', ''))
                line_style_idx = (replicate_num - 7) % len(line_styles)  # Adjust based on your B-numbering
                line_style = line_styles[line_style_idx]
                
                sorted_thicknesses = np.sort(thicknesses)
                cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                
                # Plot with condition color but different line style
                ax.plot(sorted_thicknesses, cdf, color=color, linestyle=line_style, linewidth=2, 
                       label=f'{condition}-{replicate} (n={len(thicknesses)} components)')
                
                plotted_conditions.add(condition)
    
    ax.set_xlabel('Component Average Thickness (pixels)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Blue Component CDFs by Biological Replicate\nFilter: {filter_type}={filter_value}')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'blue_components_replicate_cdfs_{filter_type}_{filter_value}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Replicate CDF plot saved: {plot_path}")
    
    # Only create the simplified plot if we have data
    if len(condition_replicate_data) > 0:
        # Also create a version with just condition colors (no line style variation)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for condition, replicate_data_dict in condition_replicate_data.items():
            color = condition_colors.get(condition, '#808080')
            
            for replicate, thicknesses in replicate_data_dict.items():
                if thicknesses:
                    sorted_thicknesses = np.sort(thicknesses)
                    cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                    
                    # Plot with condition color
                    ax.plot(sorted_thicknesses, cdf, color=color, linewidth=2, alpha=0.7)
        
        # Add legend for conditions only
        for condition, color in condition_colors.items():
            ax.plot([], [], color=color, linewidth=3, label=condition)
        
        ax.set_xlabel('Component Average Thickness (pixels)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Blue Component CDFs by Condition (All Replicates)\nFilter: {filter_type}={filter_value}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        # Save simplified plot
        simple_plot_filename = f'blue_components_condition_cdfs_{filter_type}_{filter_value}.png'
        simple_plot_path = os.path.join(output_dir, simple_plot_filename)
        plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Condition CDF plot saved: {simple_plot_path}")

def main():
    # === CONFIGURATION ===
    # Set your input and output directories here
    input_dir = "./cdf_analysis_branch_snakes"  # Directory containing the organized group folders
    output_dir = "./filtered_blue_components"   # Directory where filtered results will be saved
    
    # === PRESET FILTER COMBINATIONS ===
    # Define different combinations of filters to apply
    # Each combination is a list of filters that will be applied together
    # Now using true skeleton_length instead of area for more accurate branch/snake analysis
    filter_combinations = [
        # Single comprehensive filter combination with all filters
        [
            ("max_thickness", 15, "Max thickness <= 1000"),
            ("min_thickness", 0, "Min thickness >= 0"),
            ("avg_thickness", (0, 8), "Average thickness between 2.5 and 5.5"),  # Range filter
            ("skeleton_length", 15, "Skeleton length >= 50 pixels"),
            ("skeleton_length_max", 2000000, "Skeleton length <= 1000 pixels"),
            ("mother_length", 50, "Mother component length >= 100 pixels"),
            ("mother_length_max", 2000000, "Mother component length <= 2000 pixels"),
        ]
    ]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Blue Component Filter Script ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of filter combinations to apply: {len(filter_combinations)}")
    
    # Load data
    data = load_blue_component_data(input_dir)
    
    if not data:
        print("No blue component data found!")
        return
    
    # Generate overall component metric CDFs (before any filtering)
    print(f"\n{'='*60}")
    print(f"Generating Overall Component Metric CDFs (No Filters)")
    print(f"{'='*60}")
    overall_output_dir = os.path.join(output_dir, "overall_component_metrics")
    os.makedirs(overall_output_dir, exist_ok=True)
    generate_component_metric_cdfs(data, overall_output_dir)
    generate_individual_filter_cdfs(data, overall_output_dir)
    
    # Apply each filter combination
    for i, filter_combination in enumerate(filter_combinations, 1):
        print(f"\n{'='*60}")
        print(f"Applying Filter Combination {i}/{len(filter_combinations)}")
        print(f"Number of filters in combination: {len(filter_combination)}")
        print(f"{'='*60}")
        
        # Apply combined filters
        filtered_data = apply_combined_filters(data, filter_combination)
        
        if not filtered_data:
            print("No data remaining after filtering!")
            continue
        
        # Create subdirectory for this filter combination
        filter_names = [f"{filter_type}_{filter_value}" for filter_type, filter_value, _ in filter_combination]
        safe_filter_label = "_".join(filter_names).replace('=', '_').replace(' ', '_')
        filter_output_dir = os.path.join(output_dir, f"combination_{i}_{safe_filter_label}")
        os.makedirs(filter_output_dir, exist_ok=True)
        
        # Generate CDF plots
        generate_cdf_plots(filtered_data, filter_output_dir, "combined", f"combination_{i}")
        
        # Generate component metric CDFs (using filtered data for performance)
        generate_component_metric_cdfs(filtered_data, filter_output_dir, filter_combination)

        # Generate filter application plots
        generate_filter_application_plots(data, filter_output_dir, filter_combination)
        
        # Perform statistical analysis
        perform_statistical_analysis(filtered_data, filter_output_dir, "combined", f"combination_{i}")
        
        # Save filtered data
        save_filtered_data(filtered_data, filter_output_dir, "combined", f"combination_{i}")
        
        # Generate individual filter CDFs
        generate_individual_filter_cdfs(data, filter_output_dir)

        # Generate component count CDFs
        generate_component_count_cdfs(filtered_data, filter_output_dir, filter_combination)
        
        # Generate replicate-specific CDFs (only for the first combination to avoid performance issues)
        if i == 1:
            generate_replicate_cdfs(filtered_data, filter_output_dir, "combined", f"combination_{i}")
        
        print(f"✓ Filter Combination {i} complete")
    
    print(f"\n{'='*60}")
    print(f"=== ALL FILTER COMBINATIONS COMPLETE ===")
    print(f"Results saved to: {output_dir}")
    print(f"Generated files:")
    print(f"  - Overall component metrics CDFs (no filters applied)")
    print(f"  - Individual filter impact CDFs")
    print(f"  - Filter application analysis plots")
    print(f"  - Each combination has its own subdirectory with:")
    print(f"    ├── CDF plots for filtered data")
    print(f"    ├── Component metric CDFs with filter lines")
    print(f"    ├── Filter application analysis")
    print(f"    ├── KS test results")
    print(f"    └── Filtered CSV files")
    print(f"  - Component count per image CDFs")
    print(f"  - Component count per image box plots")
    print(f"  - Replicate-specific CDFs")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 