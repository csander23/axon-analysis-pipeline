# Confocal Dataset Directory

Place your confocal microscopy data files here.

## Expected File Format
- `.nd2` files (Nikon confocal microscopy format)

## Directory Structure
Organize your data in the following hierarchical structure:

```
Confocal_Dataset/
├── Bioreplicate_1/
│   ├── Condition_A/
│   │   ├── image1.nd2
│   │   ├── image2.nd2
│   │   └── ...
│   └── Condition_B/
│       ├── image1.nd2
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
