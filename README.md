# Axon Analysis Pipeline

A comprehensive tool for analyzing axon morphology in both confocal microscopy and slide scanning images. This pipeline provides automated analysis of axon thickness, branching patterns, and morphological features.

## Features

- **Dual Pipeline Support**: 
  - Confocal microscopy analysis with MAP2/L1CAM staining
  - Slide scanning analysis for fluorescent protein expression
- **Automated Analysis**:
  - Objective thresholding using regression models
  - Soma exclusion
  - Axon skeletonization
  - Thickness quantification
  - Branching analysis
  - Thick vs. thin ratio calculation
- **Flexible Data Structure**:
  - Supports hierarchical organization (Bioreplicate/Condition)
  - Automatic detection of experimental groups
- **Interactive Tools**:
  - GUI for regression model creation
  - Threshold adjustment interface
  - Real-time visualization

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/axon-analysis-pipeline.git
   cd axon-analysis-pipeline
   ```

2. **Create Conda Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate axon-analysis
   ```

## Directory Structure

```
Axon Analysis Pipeline/
├── Confocal_Axon_Analysis_Pipeline/
│   ├── Active_Model_and_Configuration_Directory/
│   │   ├── config.yaml
│   │   └── active_model.json (optional)
│   ├── Confocal_Dataset/
│   │   ├── Bioreplicate1/
│   │   │   ├── Condition1/
│   │   │   └── Condition2/
│   │   └── Bioreplicate2/
│   └── Axon_Analysis_Pipeline.ipynb
└── Slide_Scanning_Axon_Analysis_Pipeline/
    ├── Active_Model_and_Configuration_Directory/
    │   ├── config.yaml
    │   └── active_model.json (optional)
    ├── Slide_Scanning_Dataset/
    │   ├── Bioreplicate1/
    │   │   ├── Condition1/
    │   │   └── Condition2/
    │   └── Bioreplicate2/
    └── Slide_Scanning_Analysis_Pipeline.ipynb
```

## Usage

### Confocal Analysis

1. **Setup Your Data**:
   - Place your confocal images in `Confocal_Axon_Analysis_Pipeline/Confocal_Dataset/`
   - Organize in Bioreplicate/Condition structure
   - Ensure paired MAP2/L1CAM images

2. **Run Analysis**:
   ```bash
   cd Confocal_Axon_Analysis_Pipeline
   jupyter notebook Axon_Analysis_Pipeline.ipynb
   ```

3. **Follow Notebook Steps**:
   - Set input/output paths
   - Create regression model (optional)
   - Run analysis
   - View results

### Slide Scanning Analysis

1. **Setup Your Data**:
   - Place your slide scanner images in `Slide_Scanning_Axon_Analysis_Pipeline/Slide_Scanning_Dataset/`
   - Organize in Bioreplicate/Condition structure

2. **Run Analysis**:
   ```bash
   cd Slide_Scanning_Axon_Analysis_Pipeline
   jupyter notebook Slide_Scanning_Analysis_Pipeline.ipynb
   ```

3. **Follow Notebook Steps**:
   - Set input/output paths
   - Create regression model (optional)
   - Run analysis
   - View results

## Configuration

Both pipelines use `config.yaml` files for parameter configuration:

- Thresholding parameters
- Morphological operation settings
- Analysis thresholds
- Output organization
- Visualization settings

See example config files in respective `Active_Model_and_Configuration_Directory` folders.

## Output Structure

Results are organized in:
- `Images/`: Summary visualizations for each image
- `Plots/`: Final analysis plots (CDFs, distributions)
- `Results/`: CSV files with quantification data

## Dependencies

Key dependencies (automatically installed via environment.yml):
- Python 3.8+
- numpy
- pandas
- scikit-image
- matplotlib
- opencv-python
- ipywidgets
- jupyter

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:
[Your citation information here]

## Contact

For questions or support, please contact [Your contact information]
