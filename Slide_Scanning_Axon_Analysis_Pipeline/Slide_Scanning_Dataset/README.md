# Slide Scanning Dataset Directory

Place your slide scanning microscopy data files here.

## Expected File Format
- Image files in common formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

## Directory Structure
Organize your data in the following hierarchical structure:

```
Slide_Scanning_Dataset/
├── Bioreplicate_1/
│   ├── Condition_A/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── Condition_B/
│       ├── image1.jpg
│       └── ...
├── Bioreplicate_2/
│   ├── Condition_A/
│   └── Condition_B/
└── ...
```

The pipeline will automatically detect:
- **Biological replicates** from the first-level directories
- **Experimental conditions** from the second-level directories

Directory names can be anything - the pipeline only cares about the structure, not the naming convention.
