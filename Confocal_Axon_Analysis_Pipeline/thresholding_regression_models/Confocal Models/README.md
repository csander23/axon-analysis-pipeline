# Confocal Thresholding Regression Models

This directory contains regression models used for determining thresholds in confocal image analysis. Models are saved as pickle files (`.pkl`) with corresponding documentation files (`.txt`) that include timestamps and model details. The models combine image metrics with biological replicate information to produce consistent thresholds across different experimental conditions.

## Model Files

Each model is saved in two formats:

1. **Pickle File** (`.pkl`):
   - Contains the serialized model object
   - Includes all coefficients and parameters
   - Named with timestamp: `confocal_threshold_model_TYPE_YYYY-MM-DD_HH-MM-SS.pkl`

2. **Documentation File** (`.txt`):
   - Human-readable model documentation
   - Contains model parameters, formulas, and usage examples
   - Named with same timestamp: `confocal_threshold_model_TYPE_YYYY-MM-DD_HH-MM-SS.txt`

## Models Available

### 1. Default Model
```python
Coefficients:
- Intercept: 82.5909
- Metric Coefficient: 0.3027

Replicate Offsets:
- B114: -35
- B115: 20
- B116: -55
- B117: 5
```

### 2. Alternative Model
```python
Coefficients:
- Intercept: 67.8832
- Metric Coefficient: 0.4825

Replicate Offsets:
- B114: -8.3905
- B115: 37.7299
- B116: -39.3154
- B117: 9.9760
```

## Usage

### Loading and Using Models
```python
from confocal_threshold_regression import RegressionModel

# Load the latest model
model = RegressionModel.load("confocal_threshold_model_default_2025-09-27_14-30-00.pkl")

# Compute threshold
metric_value = compute_metric(image)  # Your image metric
replicate = "B114"                    # Your replicate ID

threshold = model.predict_threshold(metric_value, replicate)
```

### Creating and Saving New Models
```python
from confocal_threshold_regression import create_model, save_all_models

# Save all model variants
model_paths = save_all_models("path/to/models/dir")

# Or create and save individual model
model = create_model("default")
pkl_path, txt_path = model.save("path/to/models/dir")
```

## Model Details

The models use a linear regression approach with two components:

1. **Image Metric**: `mean_above_val15`
   - Computed as the mean value of pixels above the 15th percentile
   - Captures overall image intensity characteristics

2. **Replicate Offset**:
   - Biological replicate-specific adjustment
   - Helps maintain consistency within experimental groups

### Threshold Formula
```
threshold = intercept + (metric_coefficient × metric_value) + replicate_offset
```

## Model Selection

- **Default Model**: More conservative thresholds, better for high-contrast images
- **Alternative Model**: More aggressive thresholds, better for low-contrast images

## Implementation

The models are implemented in `confocal_threshold_regression.py` with the following features:

- Object-oriented design with `RegressionModel` base class
- Easy extension for new model variants
- Built-in utility functions for metric computation
- Automatic replicate detection from filenames
- Model persistence with versioning and documentation
- Timestamp-based model versioning