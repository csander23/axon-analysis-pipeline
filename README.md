# Axon Analysis Pipeline

Built by Charles Sander and Silvia Natale in the Südhof Lab at Stanford University.

A comprehensive, automated tool for analyzing axon morphology in microscopy images. This pipeline supports both **confocal microscopy** (with MAP2 and axonal staining) and **slide scanning** (fluorescent protein expression) data.

## What This Pipeline Does

- **Channel Handling**: Processes Nikon ND2 files containing both MAP2 and L1CAM channels
- **Soma Exclusion**: Uses MAP2 channel to identify and remove cell bodies from L1CAM analysis
- **Thresholding**: Creates binary masks of L1CAM-positive axons using a regression model
- **Morphological Analysis**: Analyzes axon structure through:
  - Skeletonization for centerline extraction
  - Distance transform for thickness measurement
  - Branch point detection
  - Thick vs thin classification

## Data Structure Expected

Your data should be organized in a two-level hierarchy:

For Confocal Data:
```
Confocal_Dataset/
├── Bioreplicate1/          # First level: Biological replicates
│   ├── Control/            # Second level: Experimental conditions  
│   │   ├── image1.nd2     # Nikon ND2 files with MAP2 and L1CAM channels
│   │   ├── image2.nd2
│   │   └── ...
│   ├── Treatment/
│   │   ├── image1.nd2
│   │   └── ...
├── Bioreplicate2/
│   ├── Control/
│   └── Treatment/
└── ...
```

For Slide Scanning Data:
```
Slide_Scanning_Dataset/
├── Bioreplicate1/
│   ├── Control/
│   │   ├── image1.jpg         # Image files
│   │   ├── image2.jpg
│   │   └── ...
│   └── Treatment/
├── Bioreplicate2/
└── ...
```

**Note**: 
- Confocal pipeline expects Nikon ND2 files with both MAP2 and L1CAM channels
- Slide scanning pipeline processes image files in common formats (JPG, PNG, TIF)
- The actual bioreplicate and condition folder names don't matter - the pipeline uses directory levels to determine the structure

## Getting Started

### Step 1: Download the Code

```bash
# Clone the repository
git clone https://github.com/csander23/axon-analysis-pipeline.git
cd axon-analysis-pipeline
```

### Step 2: Set Up the Environment

This pipeline uses conda to manage all dependencies automatically:

```bash
# Create the conda environment (this installs all required packages)
conda env create -f environment.yml

# Activate the environment
conda activate axon-analysis
```

The environment file handles all the complex dependencies including Python 3.8, image processing libraries (scikit-image, opencv, aicsimageio), scientific computing (numpy, pandas, scipy), machine learning (scikit-learn), visualization (matplotlib, ipywidgets), and Jupyter notebook support.

### Step 3: Choose Your Analysis Type

The pipeline has two separate workflows:

## Confocal Microscopy Analysis

**Use this for**: Images with MAP2 and L1CAM staining from confocal microscopes

### Quick Start - Confocal

1. **Open the notebook**:
   ```bash
   cd Confocal_Axon_Analysis_Pipeline
   jupyter notebook Axon_Analysis_Pipeline.ipynb
   ```

2. **Set your paths** (Cell 1):
   ```python
   # Edit these paths to point to your data
   INPUT_DIR = "/path/to/your/confocal/dataset"
   OUTPUT_DIR = "/path/to/save/results"
   ```

3. **Run the analysis** (Cell 2):
   - Just run this cell - it will automatically process all your images
   - Results appear in your OUTPUT_DIR

4. **Optional: Create custom regression model** (Cell 3):
   - Set `CREATE_NEW_MODEL = True`
   - Run the cell to open an interactive GUI
   - Adjust thresholds for sample images using the slider
   - Click "Finish Now" when done

### Understanding Confocal Configuration

The pipeline uses `Active_Model_and_Configuration_Directory/config.yaml` to control analysis:

```yaml
# Threshold Method (choose one)
use_regression_model: true     # Use your custom .json model
use_raw_threshold: false       # Use a fixed threshold value
raw_threshold_value: 15.0      # Fixed threshold (if use_raw_threshold: true)

# Manual parameters (if use_regression_model: false)
percentile_threshold: 15       # Use 15th percentile for threshold calculation
threshold_intercept: 82.59     # Manual regression intercept
threshold_coefficient: 0.30    # Manual regression slope

# Analysis options
enable_thick_thin_analysis: true    # Analyze thick vs thin segments
width_threshold: 3.7               # Pixels - threshold for thick classification
```

### Confocal Regression Models

When you create a regression model, it saves three files:
- `confocal_regression_model_TIMESTAMP.json` - The actual model
- `confocal_regression_model_TIMESTAMP.txt` - readable documentation  
- `confocal_regression_plot_TIMESTAMP.png` - Visualization of the model

**To use your model**:
1. Copy the `.json` file to `Active_Model_and_Configuration_Directory/`
2. Set `use_regression_model: true` in `config.yaml`

## Slide Scanning Analysis

**Use this for**: Fluorescent protein images from slide scanners

### Quick Start - Slide Scanning

1. **Open the notebook**:
   ```bash
   cd Slide_Scanning_Axon_Analysis_Pipeline  
   jupyter notebook Slide_Scanning_Analysis_Pipeline.ipynb
   ```

2. **Set your paths** (Cell 1):
   ```python
   # Edit these paths to point to your data
   INPUT_DIR = "/path/to/your/slide/scanning/dataset"
   OUTPUT_DIR = "/path/to/save/results"
   ```

3. **Run the analysis** (Cell 2):
   - Processes all images automatically
   - Uses legacy-compatible parameters optimized for slide scanning

4. **Optional: Create custom regression model** (Cell 3):
   - Set `CREATE_NEW_MODEL = True`  
   - Interactive GUI with threshold range 0.010-0.050 (normalized images)
   - Optimized for slide scanning image characteristics

### Understanding Slide Scanning Configuration

Slide scanning uses different parameters optimized for this imaging modality:

```yaml
# Threshold method
use_regression_model: false        # Use manual parameters (legacy-compatible)
use_raw_threshold: false          # Use calculated thresholds

# Legacy-compatible parameters (normalized 0-1 range)
percentile_threshold: 0.012       # Absolute threshold in 0-1 range
threshold_intercept: -0.003583    # Legacy intercept
threshold_coefficient: 1.4865     # Legacy slope

# Slide scanning specific
width_threshold: 3.7              # Same thick/thin threshold as confocal
```

## Understanding Your Results

Both pipelines create organized output folders:

```
Data_Output/
├── Images/                    # Summary visualizations for each image
│   ├── image1_summary.png     # 2x2 panel showing processing steps
│   └── ...
├── Plots/                     # Final analysis plots
│   ├── thickness_cdf.png      # Cumulative distribution of thickness
│   └── thick_thin_analysis.png
└── Results/                   # Data files
    ├── individual_results.csv # Per-image quantification
    ├── summary_statistics.csv # Condition summaries
    └── thick_thin_ratios.csv  # Thick vs thin analysis
```

### Summary Image Panels

Each `*_summary.png` shows a 2x2 grid:
- **Top Left**: Original image (brightened for visibility)
- **Top Right**: Binary mask after thresholding and filtering
- **Bottom Left**: Skeleton analysis (Blue=normal, Pink=high branching, Yellow=excluded thick regions)
- **Bottom Right**: Thick vs thin classification (Red=thick, Blue=thin) - if enabled

## Advanced Configuration

### Regression Models Explained

**Confocal Models**: Use mean intensity above 15th percentile as predictor
- Threshold range: typically 0-500 intensity units
- Includes biological replicate offsets
- Accounts for staining variability between experiments

**Slide Scanning Models**: Use whole image mean intensity as predictor  
- Threshold range: 0.010-0.050 (normalized 0-1 range)
- Legacy-compatible with previous analysis pipelines
- Optimized for fluorescent protein expression

### Creating High-Quality Models

1. **Sample Size**: Use 10-30 images representing all conditions
2. **Distribution**: Ensure images from all biological replicates
3. **Threshold Setting**: Look for thresholds that capture axon structure without noise
4. **Validation**: Check that binary masks accurately represent axon morphology

### Thick vs Thin Analysis

This analysis classifies axon segments based on:
- **Width**: Segments ≥3.7 pixels are "thick"
- **Branching Context**: Segments near many branch points are reclassified as "thin"
- **Filtering**: Excludes artificially thick regions (soma, overlapping axons)

## Troubleshooting

### Common Issues

**"No images found"**:
- Check that your directory structure matches the expected hierarchy
- For confocal: Ensure files are Nikon ND2 format (.nd2)
- For slide scanning: Ensure files are in supported formats (JPG, PNG, TIF)

**"Threshold too high/low"**:
- For confocal: Try adjusting `percentile_threshold` in config.yaml
- For slide scanning: Check that images are properly normalized (0-1 range)

**"Binary mask is blank"**:
- Lower the threshold values in your regression model
- Check image intensity ranges - may need different normalization

**Interactive GUI not working**:
- Restart Jupyter kernel and try again
- Ensure `%matplotlib widget` is working (may need `pip install ipywidgets`)

### Getting Help

1. Check that all paths in the notebook are correct
2. Verify your data structure matches the expected hierarchy
3. Look at the summary images to see if thresholding is working
4. Try the manual threshold parameters before creating custom models

## Updating the Pipeline

To get the latest version:

```bash
cd axon-analysis-pipeline
git pull origin main
conda env update -f environment.yml
```

This ensures you have the latest features and bug fixes while maintaining compatibility with your existing data and models.
