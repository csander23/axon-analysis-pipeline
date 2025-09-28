#!/bin/bash

# Setup script for Axon Analysis Pipeline

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Setting up Axon Analysis Pipeline ==="
echo "Project root: $PROJECT_ROOT"

# Create conda environment
echo -e "\n1. Creating conda environment..."
conda env create -f "$SCRIPT_DIR/environment.yml"

# Activate environment
echo -e "\n2. Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate axon_analysis

# Install package
echo -e "\n3. Installing package..."
pip install -e "$SCRIPT_DIR"

echo -e "\n=== Setup Complete! ==="
echo "To use the pipeline:"
echo "1. Activate the environment: conda activate axon_analysis"
echo "2. Open Jupyter Notebook: jupyter notebook"
echo "3. Navigate to and open: Axon_Analysis_Pipeline.ipynb"
