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
import numpy as np
# Add parallel processing imports
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Import genotypes mappings
from genotypes_hardcoded_mappings import GENOTYPES_MAPPING, GENOTYPES_STATINS_MAPPING

# ---- CONFIGURATION ----
input_dir = "./Images to analyze SNT"
output_dir = "./comprehensive_cdf_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Parallel processing configuration
PARALLEL_CONFIG = {
    "enabled": True,                    # Enable/disable parallel processing
    "max_workers": 12,                  # Increased from 4 to 12 workers for speed
    "chunk_size": 3,                    # Smaller chunk size to reduce memory pressure
    "progress_update_interval": 5,      # More frequent progress updates
    "memory_optimization": True,        # Enable memory optimization
    "preload_libraries": True,          # Preload common libraries in workers
}

# Thick-thin analysis configuration
THICK_THIN_CONFIG = {
    "width_threshold": 3.7,             # Threshold for thick vs thin classification
}

# Color scheme configuration for consistent plotting across all functions
COLOR_SCHEMES = {
    "genotypes": {
        'KO': '#E31A1C',         # Bright red
        'ApoE4': '#1F78B4',      # Blue  
        'ApoE2': '#33A02C',      # Green
        'Control': '#FF7F00'     # Orange
    },
    "genotypes_statins": {
        'KO_Control': '#E31A1C',         # Bright red
        'ApoE4_Control': '#1F78B4',      # Blue
        'ApoE2_Control': '#33A02C',      # Green
        'Control_Control': '#FF7F00',    # Orange
        'KO_Simvastatin': '#A50F15',     # Dark red
        'ApoE4_Simvastatin': '#08519C',  # Dark blue
        'ApoE2_Simvastatin': '#006837',  # Dark green
        'Control_Simvastatin': '#D94801' # Dark orange
    },
    "domain_analysis": {
        'Control': '#FF7F00',        # Orange
        'ApoE2-NTD': '#B2DF8A',     # Light green
        'ApoE4-NTD': '#A6CEE3',     # Light blue
        'ApoE2': '#33A02C',         # Green
        'ApoE4': '#1F78B4',         # Blue
        'ApoE-CTD': '#DDA0DD'       # Plum (for the 6th group)
    }
}

def get_color_for_condition(design_name, condition):
    """Get the appropriate color for a given condition and experimental design."""
    if design_name in COLOR_SCHEMES:
        return COLOR_SCHEMES[design_name].get(condition, '#808080')  # Default gray
    else:
        # Fallback to Set3 colormap for unknown designs
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        # Use a hash of the condition name to get consistent colors
        import hashlib
        color_idx = int(hashlib.md5(condition.encode()).hexdigest(), 16) % len(colors)
        return colors[color_idx]

# Experimental designs configuration
EXPERIMENTAL_DESIGNS = {
    "genotypes": {
        "description": "Genotype Analysis (Control Only)",
        "batches": ["B8", "B10", "B11", "B12"],
        "genotype_mapping": {
            "GroupA": "KO",
            "GroupB": "ApoE4", 
            "GroupC": "Control",
            "GroupD": "ApoE2"
        }
    },
    "genotypes_statins": {
        "description": "Genotype + Statin Treatment Analysis",
        "batches": ["B8", "B10"],
        "genotype_mapping": {
            "GroupA": "KO",
            "GroupB": "ApoE4", 
            "GroupC": "Control",
            "GroupD": "ApoE2"
        }
    },
    "domain_analysis": {
        "description": "Domain Analysis (3 Biological Replicates)",
        "batches": ["B1", "B2", "B3"],
        "genotype_mapping": {
            "H": "ApoE4",
            "L": "ApoE2",
            "N": "ApoE4-NTD", 
            "P": "ApoE2-NTD",
            "R": "Control",
            "B": "Control",
            "X": "ApoE2-NTD",
            "Z": "ApoE2",
            "J": "ApoE4",
            "A": "ApoE4",
            "Y": "ApoE-CTD",
            "O": "ApoE-CTD"
        }
    }
}

# Hardcoded mapping for domain analysis reorganization into 3 batches
# Based on Excel file: SummaryData_SparseLabelingApoEFragments_Viruses (1).xlsx
DOMAIN_ANALYSIS_MAPPING = {
    "Batch16_GroupH-1.jpg": {
        "batch": "B1",
        "genotype": "ApoE4"
    },
    "Batch16_GroupH_2-1.jpg": {
        "batch": "B1",
        "genotype": "ApoE4"
    },
    "Batch16_GroupH_2-2.jpg": {
        "batch": "B1",
        "genotype": "ApoE4"
    },
    "Batch16_GroupL_2-1.jpg": {
        "batch": "B1",
        "genotype": "ApoE2"
    },
    "Batch16_GroupL_2-2.jpg": {
        "batch": "B1",
        "genotype": "ApoE2"
    },
    "Batch16_GroupL_2-3.jpg": {
        "batch": "B1",
        "genotype": "ApoE2"
    },
    "Batch16_GroupL_2-4.jpg": {
        "batch": "B1",
        "genotype": "ApoE2"
    },
    "Batch16_GroupN_2-1.jpg": {
        "batch": "B1",
        "genotype": "ApoE4-NTD"
    },
    "Batch16_GroupN_2-2.jpg": {
        "batch": "B1",
        "genotype": "ApoE4-NTD"
    },
    "Batch16_GroupR-1.jpg": {
        "batch": "B1",
        "genotype": "Control"
    },
    "Batch16_GroupR_2-1.jpg": {
        "batch": "B1",
        "genotype": "Control"
    },
    "Batch17_GroupH-1.jpg": {
        "batch": "B1",
        "genotype": "ApoE4"
    },
    "Batch17_GroupH_2-1.jpg": {
        "batch": "B1",
        "genotype": "ApoE4"
    },
    "Batch17_GroupH_2-2.jpg": {
        "batch": "B1",
        "genotype": "ApoE4"
    },
    "Batch17_GroupL_2-1.jpg": {
        "batch": "B1",
        "genotype": "ApoE2"
    },
    "Batch17_GroupL_2-2.jpg": {
        "batch": "B1",
        "genotype": "ApoE2"
    },
    "Batch17_GroupL_2-3.jpg": {
        "batch": "B1",
        "genotype": "ApoE2"
    },
    "Batch17_GroupN-5.jpg": {
        "batch": "B1",
        "genotype": "ApoE4-NTD"
    },
    "Batch17_GroupN-6.jpg": {
        "batch": "B1",
        "genotype": "ApoE4-NTD"
    },
    "Batch17_GroupN_2-1.jpg": {
        "batch": "B1",
        "genotype": "ApoE4-NTD"
    },
    "Batch17_GroupN_2-2.jpg": {
        "batch": "B1",
        "genotype": "ApoE4-NTD"
    },
    "Batch17_GroupN_2-3.jpg": {
        "batch": "B1",
        "genotype": "ApoE4-NTD"
    },
    "Batch17_GroupP-1.jpg": {
        "batch": "B1",
        "genotype": "ApoE2-NTD"
    },
    "Batch17_GroupP-2.jpg": {
        "batch": "B1",
        "genotype": "ApoE2-NTD"
    },
    "Batch17_GroupP_2-1.jpg": {
        "batch": "B1",

        "genotype": "ApoE2-NTD"
    },
    "Batch17_GroupP_2-2.jpg": {
        "batch": "B1",

        "genotype": "ApoE2-NTD"
    },
    "Batch17_GroupR-1.jpg": {
        "batch": "B1",

        "genotype": "Control"
    },
    "Batch17_GroupR-2.jpg": {
        "batch": "B1",

        "genotype": "Control"
    },
    "Batch17_GroupR-3.jpg": {
        "batch": "B1",

        "genotype": "Control"
    },
    "Batch17_GroupR_2-1.jpg": {
        "batch": "B1",

        "genotype": "Control"
    },
    "Batch17_GroupR_2-2.jpg": {
        "batch": "B1",

        "genotype": "Control"
    },
    "Batch17_GroupR_2-3.jpg": {
        "batch": "B1",

        "genotype": "Control"
    },
    "Batch17_GroupR_2-4.jpg": {
        "batch": "B1",

        "genotype": "Control"
    },
    "Batch18_GroupH_2-1.jpg": {
        "batch": "B1",

        "genotype": "ApoE4"
    },
    "Batch18_GroupH_2-2.jpg": {
        "batch": "B1",

        "genotype": "ApoE4"
    },
    "Batch18_GroupH_2-3.jpg": {
        "batch": "B1",

        "genotype": "ApoE4"
    },
    "Batch18_GroupL-1.jpg": {
        "batch": "B1",

        "genotype": "ApoE2"
    },
    "Batch18_GroupL_2-1.jpg": {
        "batch": "B1",

        "genotype": "ApoE2"
    },
    "Batch18_GroupP-1.jpg": {
        "batch": "B1",

        "genotype": "ApoE2-NTD"
    },
    "Batch18_GroupP_2-1.jpg": {
        "batch": "B1",

        "genotype": "ApoE2-NTD"
    },
    "Batch18_GroupR_2-1.jpg": {
        "batch": "B1",

        "genotype": "Control"
    },
    "Batch19_GroupL-1.jpg": {
        "batch": "B1",

        "genotype": "ApoE2"
    },
    "Batch19_GroupN-1.jpg": {
        "batch": "B1",

        "genotype": "ApoE4-NTD"
    },
    "Batch19_GroupN-2.jpg": {
        "batch": "B1",

        "genotype": "ApoE4-NTD"
    },
    "Batch19_GroupN_2-1.jpg": {
        "batch": "B1",

        "genotype": "ApoE4-NTD"
    },
    "Batch19_GroupP-1.jpg": {
        "batch": "B1",

        "genotype": "ApoE2-NTD"
    },
    "Batch19_GroupP-2.jpg": {
        "batch": "B1",

        "genotype": "ApoE2-NTD"
    },
    "Batch19_GroupP_2-1.jpg": {
        "batch": "B1",

        "genotype": "ApoE2-NTD"
    },
    "Batch19_GroupP_2-2.jpg": {
        "batch": "B1",

        "genotype": "ApoE2-NTD"
    },
    "Batch23_GroupB-1.jpg": {
        "batch": "B2",

        "genotype": "Control"
    },
    "Batch23_GroupB-2.jpg": {
        "batch": "B2",

        "genotype": "Control"
    },
    "Batch23_GroupB_2-1.jpg": {
        "batch": "B2",

        "genotype": "Control"
    },
    "Batch23_GroupB_2-2.jpg": {
        "batch": "B2",

        "genotype": "Control"
    },
    "Batch23_GroupB_2-3.jpg": {
        "batch": "B2",

        "genotype": "Control"
    },
    "Batch23_GroupB_2-4.jpg": {
        "batch": "B2",

        "genotype": "Control"
    },
    "Batch23_GroupB_2-5.jpg": {
        "batch": "B2",

        "genotype": "Control"
    },
    "Batch23_GroupB_2-6.jpg": {
        "batch": "B2",

        "genotype": "Control"
    },
    "Batch23_GroupB_2-7.jpg": {
        "batch": "B2",

        "genotype": "Control"
    },
    "Batch23_GroupB_2-8.jpg": {
        "batch": "B2",

        "genotype": "Control"
    },
    "Batch23_GroupJ-1.jpg": {
        "batch": "B2",

        "genotype": "ApoE4"
    },
    "Batch23_GroupJ-2.jpg": {
        "batch": "B2",

        "genotype": "ApoE4"
    },
    "Batch23_GroupJ-3.jpg": {
        "batch": "B2",

        "genotype": "ApoE4"
    },
    "Batch23_GroupJ-4.jpg": {
        "batch": "B2",

        "genotype": "ApoE4"
    },
    "Batch23_GroupJ-5.jpg": {
        "batch": "B2",

        "genotype": "ApoE4"
    },
    "Batch23_GroupJ-6.jpg": {
        "batch": "B2",

        "genotype": "ApoE4"
    },
    "Batch23_GroupJ-7.jpg": {
        "batch": "B2",

        "genotype": "ApoE4"
    },
    "Batch23_GroupJ_2-1.jpg": {
        "batch": "B2",

        "genotype": "ApoE4"
    },
    "Batch23_GroupJ_2-2.jpg": {
        "batch": "B2",

        "genotype": "ApoE4"
    },
    "Batch23_GroupJ_2-3.jpg": {
        "batch": "B2",

        "genotype": "ApoE4"
    },
    "Batch23_GroupL-1.jpg": {
        "batch": "B2",

        "genotype": "ApoE4-NTD"
    },
    "Batch23_GroupL-2.jpg": {
        "batch": "B2",

        "genotype": "ApoE4-NTD"
    },
    "Batch23_GroupL-3.jpg": {
        "batch": "B2",

        "genotype": "ApoE4-NTD"
    },
    "Batch23_GroupL-4.jpg": {
        "batch": "B2",

        "genotype": "ApoE4-NTD"
    },
    "Batch23_GroupL-5.jpg": {
        "batch": "B2",

        "genotype": "ApoE4-NTD"
    },
    "Batch23_GroupL_2-1.jpg": {
        "batch": "B2",

        "genotype": "ApoE4-NTD"
    },
    "Batch23_GroupL_2-2.jpg": {
        "batch": "B2",

        "genotype": "ApoE4-NTD"
    },
    "Batch23_GroupX-1.jpg": {
        "batch": "B2",

        "genotype": "ApoE2-NTD"
    },
    "Batch23_GroupX-2.jpg": {
        "batch": "B2",

        "genotype": "ApoE2-NTD"
    },
    "Batch23_GroupX-3.jpg": {
        "batch": "B2",

        "genotype": "ApoE2-NTD"
    },
    "Batch23_GroupX_2-1.jpg": {
        "batch": "B2",

        "genotype": "ApoE2-NTD"
    },
    "Batch23_GroupX_2-2.jpg": {
        "batch": "B2",

        "genotype": "ApoE2-NTD"
    },
    "Batch23_GroupX_2-3.jpg": {
        "batch": "B2",

        "genotype": "ApoE2-NTD"
    },
    "Batch23_GroupX_2-4.jpg": {
        "batch": "B2",

        "genotype": "ApoE2-NTD"
    },
    "Batch23_GroupZ-3.jpg": {
        "batch": "B2",

        "genotype": "ApoE2"
    },
    "Batch23_GroupZ-4.jpg": {
        "batch": "B2",

        "genotype": "ApoE2"
    },
    "Batch23_GroupZ-5.jpg": {
        "batch": "B2",

        "genotype": "ApoE2"
    },
    "Batch23_GroupZ-6.jpg": {
        "batch": "B2",

        "genotype": "ApoE2"
    },
    "Batch23_GroupZ_2-3.jpg": {
        "batch": "B2",

        "genotype": "ApoE2"
    },
    "Batch23_GroupZ_2-4.jpg": {
        "batch": "B2",

        "genotype": "ApoE2"
    },
    "Batch23_GroupZ_2-5.jpg": {
        "batch": "B2",

        "genotype": "ApoE2"
    },
    "Batch24_GroupB_2-1.jpg": {
        "batch": "B3",

        "genotype": "Control"
    },
    "Batch24_GroupB_2-2.jpg": {
        "batch": "B3",

        "genotype": "Control"
    },
    "Batch24_GroupB_2-3.jpg": {
        "batch": "B3",

        "genotype": "Control"
    },
    "Batch24_GroupB_2-4.jpg": {
        "batch": "B3",

        "genotype": "Control"
    },
    "Batch24_GroupB_2-5.jpg": {
        "batch": "B3",

        "genotype": "Control"
    },
    "Batch24_GroupB_2-6.jpg": {
        "batch": "B3",

        "genotype": "Control"
    },
    "Batch24_GroupB_2-7.jpg": {
        "batch": "B3",

        "genotype": "Control"
    },
    "Batch24_GroupB_2-8.jpg": {
        "batch": "B3",

        "genotype": "Control"
    },
    "Batch24_GroupX_2-1.jpg": {
        "batch": "B2",

        "genotype": "ApoE2-NTD"
    },
    "Batch24_GroupX_2-2.jpg": {
        "batch": "B2",

        "genotype": "ApoE2-NTD"
    },
    "Batch24_GroupZ-1.jpg": {
        "batch": "B2",

        "genotype": "ApoE2"
    },
    "Batch24_GroupZ-2.jpg": {
        "batch": "B2",

        "genotype": "ApoE2"
    },
    "Batch24_GroupZ-3.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch24_GroupZ-4.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch24_GroupZ_2-1.jpg": {
        "batch": "B2",

        "genotype": "ApoE2"
    },
    "Batch25_GroupJ-1.jpg": {
        "batch": "B3",

        "genotype": "ApoE4"
    },
    "Batch25_GroupJ-2.jpg": {
        "batch": "B3",

        "genotype": "ApoE4"
    },
    "Batch25_GroupJ-3.jpg": {
        "batch": "B3",

        "genotype": "ApoE4"
    },
    "Batch25_GroupJ-4.jpg": {
        "batch": "B3",

        "genotype": "ApoE4"
    },
    "Batch25_GroupJ-5.jpg": {
        "batch": "B3",

        "genotype": "ApoE4"
    },
    "Batch25_GroupJ_2-1.jpg": {
        "batch": "B3",

        "genotype": "ApoE4"
    },
    "Batch25_GroupJ_2-2.jpg": {
        "batch": "B3",

        "genotype": "ApoE4"
    },
    "Batch25_GroupJ_2-3.jpg": {
        "batch": "B3",

        "genotype": "ApoE4"
    },
    "Batch25_GroupJ_2-4.jpg": {
        "batch": "B3",

        "genotype": "ApoE4"
    },
    "Batch25_GroupJ_2-5.jpg": {
        "batch": "B3",

        "genotype": "ApoE4"
    },
    "Batch25_GroupX-1.jpg": {
        "batch": "B3",

        "genotype": "ApoE2-NTD"
    },
    "Batch25_GroupX-10.jpg": {
        "batch": "B3",

        "genotype": "ApoE2-NTD"
    },
    "Batch25_GroupX-2.jpg": {
        "batch": "B3",

        "genotype": "ApoE2-NTD"
    },
    "Batch25_GroupX-3.jpg": {
        "batch": "B3",

        "genotype": "ApoE2-NTD"
    },
    "Batch25_GroupX-4.jpg": {
        "batch": "B3",

        "genotype": "ApoE2-NTD"
    },
    "Batch25_GroupX_2-1.jpg": {
        "batch": "B2",

        "genotype": "ApoE2-NTD"
    },
    "Batch25_GroupX_2-2.jpg": {
        "batch": "B3",

        "genotype": "ApoE2-NTD"
    },
    "Batch25_GroupX_2-3.jpg": {
        "batch": "B3",

        "genotype": "ApoE2-NTD"
    },
    "Batch26_GroupB-1.jpg": {
        "batch": "B3",

        "genotype": "Control"
    },
    "Batch26_GroupB-2.jpg": {
        "batch": "B3",

        "genotype": "Control"
    },
    "Batch26_GroupL-1.jpg": {
        "batch": "B3",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupL-2.jpg": {
        "batch": "B3",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupL-3.jpg": {
        "batch": "B3",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupL-4.jpg": {
        "batch": "B3",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupL_2-1.jpg": {
        "batch": "B2",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupL_2-2.jpg": {
        "batch": "B2",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupL_2-3.jpg": {
        "batch": "B2",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupL_2-4.jpg": {
        "batch": "B3",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupL_2-5.jpg": {
        "batch": "B3",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupL_2-6.jpg": {
        "batch": "B3",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupL_2-7.jpg": {
        "batch": "B3",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupL_2-8.jpg": {
        "batch": "B3",

        "genotype": "ApoE4-NTD"
    },
    "Batch26_GroupX_2-1.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch26_GroupX_2-2.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch26_GroupX_2-3.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch26_GroupZ-1.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch26_GroupZ-2.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch26_GroupZ-3.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch26_GroupZ-4.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch26_GroupZ-5.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch26_GroupZ-6.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch26_GroupZ_2-1.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
    "Batch26_GroupZ_2-2.jpg": {
        "batch": "B3",

        "genotype": "ApoE2"
    },
}

def get_batch_from_directory_structure(filepath):
    """
    Determine batch assignment based on directory structure rather than filename.
    This avoids the issue of inconsistent batch numbers in filenames.
    """
    # Normalize path separators
    filepath = filepath.replace('\\', '/')
    
    # Extract batch from directory structure
    if "/Control/" in filepath:
        # Control directory structure
        if "/Batch 8/" in filepath:
            return "B8"
        elif "/Batch 10/" in filepath:
            return "B10"
        elif "/Batch 27/" in filepath:
            return "B27"
        elif "/B8/" in filepath:
            return "B8"
        elif "/B10/" in filepath:
            return "B10"
        elif "/B11/" in filepath:
            return "B11"
        elif "/B12/" in filepath:
            return "B12"
    elif "/Statin/" in filepath:
        # Statin directory structure
        if "/B8/" in filepath:
            return "B8"
        elif "/B10/" in filepath:
            return "B10"
        elif "/B27/" in filepath:
            return "B27"
    elif "/Fragments/" in filepath:
        # Domain analysis directory structure - reorganize into B1, B2, B3
        if "/Batch_16/" in filepath or "/Batch16/" in filepath:
            return "B16"  # Will be reorganized to B1, B2, or B3 by hardcoded mapping
        elif "/Batch_17/" in filepath or "/Batch17/" in filepath:
            return "B17"  # Will be reorganized to B1, B2, or B3 by hardcoded mapping
        elif "/Batch_18/" in filepath or "/Batch18/" in filepath:
            return "B18"  # Will be reorganized to B1, B2, or B3 by hardcoded mapping
        elif "/Batch_19/" in filepath or "/Batch19/" in filepath:
            return "B19"  # Will be reorganized to B1, B2, or B3 by hardcoded mapping
        elif "/Batch_23/" in filepath or "/Batch23/" in filepath:
            return "B23"  # Will be reorganized to B1, B2, or B3 by hardcoded mapping
        elif "/Batch_24/" in filepath or "/Batch24/" in filepath:
            return "B24"  # Will be reorganized to B1, B2, or B3 by hardcoded mapping
        elif "/Batch_25/" in filepath or "/Batch25/" in filepath:
            return "B25"  # Will be reorganized to B1, B2, or B3 by hardcoded mapping
        elif "/Batch_26/" in filepath or "/Batch26/" in filepath:
            return "B26"  # Will be reorganized to B1, B2, or B3 by hardcoded mapping
    
    return None

def get_domain_analysis_mapping(filename):
    """Get the hardcoded mapping for domain analysis files."""
    return DOMAIN_ANALYSIS_MAPPING.get(filename, None)

def create_comprehensive_file_mapping():
    """
    Create a comprehensive CSV that maps every input file to its exact biological replicate and condition.
    This is the first thing that happens when the program runs.
    """
    print("=== Creating Comprehensive File Mapping ===")
    
    # Initialize mapping data
    file_mappings = []
    
    # Find all JPG files in the directory structure
    jpg_files = []
    
    # Search in Control directory (B8, B10, B11, B12)
    control_dir = os.path.join(input_dir, "Control")
    if os.path.exists(control_dir):
        for batch in ["B8", "B10", "B11", "B12"]:
            batch_dir = os.path.join(control_dir, batch)
            if os.path.exists(batch_dir):
                batch_files = glob.glob(os.path.join(batch_dir, "*.jpg"))
                jpg_files.extend(batch_files)
    
    # Search in Statins Experiment directory (much simpler organization)
    statins_experiment_dir = os.path.join(input_dir, "Statins Experiment")
    if os.path.exists(statins_experiment_dir):
        # Control batches
        control_dir = os.path.join(statins_experiment_dir, "Control")
        if os.path.exists(control_dir):
            for batch in ["Batch 8", "Batch 10", "Batch 27"]:
                batch_dir = os.path.join(control_dir, batch)
                if os.path.exists(batch_dir):
                    batch_files = glob.glob(os.path.join(batch_dir, "*.jpg"))
                    jpg_files.extend(batch_files)
        
        # Statin batches
        statin_dir = os.path.join(statins_experiment_dir, "Statin")
        if os.path.exists(statin_dir):
            for batch in ["B8", "B10", "B27"]:
                batch_dir = os.path.join(statin_dir, batch)
                if os.path.exists(batch_dir):
                    batch_files = glob.glob(os.path.join(batch_dir, "*.jpg"))
                    jpg_files.extend(batch_files)
    
    # Search in domain analysis directories (B16-B19 and B23-B26 batches)
    domain_dir = os.path.join(input_dir, "Fragments")
    if os.path.exists(domain_dir):
        for batch in ["Batch_16", "Batch_17", "Batch_18", "Batch_19", "Batch_23", "Batch_24", "Batch_25", "Batch_26"]:
            batch_dir = os.path.join(domain_dir, batch)
            if os.path.exists(batch_dir):
                batch_files = glob.glob(os.path.join(batch_dir, "*.jpg"))
                jpg_files.extend(batch_files)
    
    print(f"Found {len(jpg_files)} JPG files to map")
    
    # Process each file to create mapping
    for jpg_file in jpg_files:
        filename = os.path.basename(jpg_file)
        directory = os.path.dirname(jpg_file)
        
        # Get experimental design info using the existing function
        design_info = get_experimental_design_info(jpg_file)
        
        if design_info:
            # Handle both dictionary and list formats
            if isinstance(design_info, dict):
                # New format (genotypes mappings)
                if design_info['experimental_design'] == 'genotypes_statins':
                    # For genotypes_statins, combine genotype and treatment
                    genotype = design_info.get('genotype', 'unknown')
                    treatment = design_info.get('biological_condition', 'unknown')
                    condition = f"{genotype}_{treatment}"
                else:
                    # For other experiments, use genotype or biological_condition
                    condition = design_info.get('genotype', design_info.get('biological_condition', 'unknown'))
                
                mapping = {
                    'filename': filename,
                    'directory': directory,
                    'full_path': jpg_file,
                    'experimental_design': design_info['experimental_design'],
                    'biological_replicate': design_info['biological_replicate'],
                    'condition': condition
                }
                file_mappings.append(mapping)
            else:
                # Old format (list of dictionaries)
                for info in design_info:
                    mapping = {
                        'filename': filename,
                        'directory': directory,
                        'full_path': jpg_file,
                        'experimental_design': info['design'],
                        'biological_replicate': info['batch'],
                        'condition': info['condition']  # Just the condition
                    }
                    file_mappings.append(mapping)
        else:
            # If no design info found, still include in mapping with unknown values
            mapping = {
                'filename': filename,
                'directory': directory,
                'full_path': jpg_file,
                'experimental_design': 'unknown',
                'biological_replicate': 'unknown',
                'condition': 'unknown'
            }
            file_mappings.append(mapping)
    
    # Create DataFrame and save to CSV
    mapping_df = pd.DataFrame(file_mappings)
    
    # Save comprehensive mapping CSV
    mapping_csv_path = os.path.join(output_dir, "comprehensive_file_mapping.csv")
    mapping_df.to_csv(mapping_csv_path, index=False)
    
    print(f"Comprehensive file mapping saved to: {mapping_csv_path}")
    print(f"Total files mapped: {len(file_mappings)}")
    
    # Print summary statistics
    print("\n=== File Mapping Summary ===")
    
    # Summary by experimental design
    design_counts = mapping_df['experimental_design'].value_counts()
    print("Files by experimental design:")
    for design, count in design_counts.items():
        print(f"  {design}: {count} files")
    
    # Summary by biological replicate
    replicate_counts = mapping_df['biological_replicate'].value_counts()
    print("\nFiles by biological replicate:")
    for replicate, count in replicate_counts.items():
        print(f"  {replicate}: {count} files")
    
    # Summary by condition
    condition_counts = mapping_df['condition'].value_counts()
    print("\nFiles by condition:")
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} files")
    
    # Fix the mapping issues as described in the text
    print("\n=== Fixing Mapping Issues ===")
    
    # Domain analysis: 1 replicate per B folder (B16-B19, B23-B26)
    domain_replicates = ['B16', 'B17', 'B18', 'B19', 'B23', 'B24', 'B25', 'B26']
    domain_files = mapping_df[mapping_df['experimental_design'] == 'domain_analysis']
    print(f"Domain analysis replicates: {len(domain_replicates)} (1 per B folder)")
    print(f"Domain analysis files: {len(domain_files)}")
    
    # Genotype analysis: 4 replicates (B8, B10, B11, B12)
    genotype_replicates = ['B8', 'B10', 'B11', 'B12']
    genotype_files = mapping_df[mapping_df['experimental_design'] == 'genotypes']
    print(f"Genotype analysis replicates: {len(genotype_replicates)} (B8, B10, B11, B12)")
    print(f"Genotype analysis files: {len(genotype_files)}")
    
    # Genotype + Statin analysis: 3 replicates (B8, B10, B27)
    genotype_statin_replicates = ['B8', 'B10', 'B27']
    genotype_statin_files = mapping_df[mapping_df['experimental_design'] == 'genotypes_statins']
    print(f"Genotype + Statin analysis replicates: {len(genotype_statin_replicates)} (B8, B10, B27)")
    print(f"Genotype + Statin analysis files: {len(genotype_statin_files)}")
    
    # The mapping is already correct, so we'll use the original mapping_df
    # No need for a separate "corrected" file since the mapping is already correct
    corrected_df = mapping_df
    
    # Create experimental design summary
    print("\n=== Creating Experimental Design Summary ===")
    create_experimental_design_summary(corrected_df)
    
    # Return the mapping for use in the rest of the program
    return corrected_df

def create_experimental_design_summary(mapping_df):
    """
    Create a comprehensive summary of each experimental design with detailed breakdowns
    of biological replicates and conditions.
    """
    print("Creating experimental design summary...")
    
    # Filter out unknown entries and excluded batches
    valid_mapping = mapping_df[mapping_df['experimental_design'] != 'unknown'].copy()
    
    # Exclude B19 and B25 from domain analysis (no control conditions for normalization)
    excluded_batches = ['B19', 'B25']
    excluded_reason = "No control conditions available for normalization"
    
    # Add exclusion information to the summary
    excluded_data = []
    for batch in excluded_batches:
        batch_files = mapping_df[
            (mapping_df['experimental_design'] == 'domain_analysis') & 
            (mapping_df['biological_replicate'] == batch)
        ]
        if not batch_files.empty:
            excluded_data.append({
                'experimental_design': 'domain_analysis',
                'biological_replicate': batch,
                'exclusion_status': 'EXCLUDED',
                'exclusion_reason': excluded_reason,
                'file_count': len(batch_files),
                'conditions': ', '.join(batch_files['condition'].unique())
            })
    
    # Remove excluded batches from valid mapping
    valid_mapping = valid_mapping[
        ~((valid_mapping['experimental_design'] == 'domain_analysis') & 
          (valid_mapping['biological_replicate'].isin(excluded_batches)))
    ]
    
    # Create summary data
    summary_data = []
    
    # Process each experimental design
    for design in valid_mapping['experimental_design'].unique():
        design_data = valid_mapping[valid_mapping['experimental_design'] == design]
        
        print(f"\n{design.upper()} EXPERIMENTAL DESIGN:")
        print("=" * 50)
        
        # Get unique biological replicates for this design
        replicates = design_data['biological_replicate'].unique()
        print(f"Biological Replicates: {len(replicates)} ({', '.join(sorted(replicates))})")
        
        # Get unique conditions for this design
        conditions = design_data['condition'].unique()
        print(f"Conditions: {len(conditions)} ({', '.join(sorted(conditions))})")
        

        
        # Total files in this design
        total_files = len(design_data)
        print(f"Total Files: {total_files}")
        
        # Detailed breakdown by replicate and condition
        print("\nDetailed Breakdown:")
        print("-" * 30)
        
        # Create pivot table for replicate vs condition
        # For all designs, group by replicate and condition
        pivot_data = design_data.groupby(['biological_replicate', 'condition']).size().reset_index(name='file_count')
        
        for _, row in pivot_data.iterrows():
            replicate = row['biological_replicate']
            condition = row['condition']
            count = row['file_count']
            
            print(f"  {replicate} - {condition}: {count} files")
            
            # Add to summary data - simplified structure
            summary_data.append({
                'experimental_design': design,
                'biological_replicate': replicate,
                'condition': condition,
                'file_count': count,
                'total_design_files': total_files,
                'total_replicates': len(replicates),
                'total_conditions': len(conditions)
            })
        
        # Summary by replicate
        print(f"\nSummary by Biological Replicate:")
        replicate_summary = design_data.groupby('biological_replicate').size()
        for replicate, count in replicate_summary.items():
            print(f"  {replicate}: {count} files")
        
        # Summary by condition
        print(f"\nSummary by Biological Condition:")
        condition_summary = design_data.groupby('condition').size()
        for condition, count in condition_summary.items():
            print(f"  {condition}: {count} files")
        
        print("\n" + "=" * 50)
    
    # Create comprehensive summary DataFrame with both detailed breakdown and statistics
    # Only keep essential columns
    summary_df = pd.DataFrame(summary_data)[['experimental_design', 'biological_replicate', 'condition', 'file_count', 'total_design_files', 'total_replicates', 'total_conditions']]
    
    # Add detailed statistics to the same file
    detailed_summary = []
    
    # Calculate statistics from the corrected summary_data instead of original mapping
    summary_df_temp = pd.DataFrame(summary_data)
    
    for design in valid_mapping['experimental_design'].unique():
        design_summary_data = summary_df_temp[summary_df_temp['experimental_design'] == design]
        
        if len(design_summary_data) > 0:
            # Calculate statistics from corrected summary data
            replicates = design_summary_data['biological_replicate'].unique()
            conditions = design_summary_data['condition'].unique()
            total_files = design_summary_data['file_count'].sum()
            
            # Calculate average files per replicate
            avg_files_per_replicate = total_files / len(replicates) if len(replicates) > 0 else 0
            
            # Calculate average files per condition
            avg_files_per_condition = total_files / len(conditions) if len(conditions) > 0 else 0
            
            detailed_summary.append({
                'experimental_design': design,
                'total_files': int(total_files),
                'biological_replicates': len(replicates),
                'conditions': len(conditions),
                'avg_files_per_replicate': round(avg_files_per_replicate, 2),
                'avg_files_per_condition': round(avg_files_per_condition, 2),
                'replicate_list': ', '.join(sorted(replicates)),
                'condition_list': ', '.join(sorted(conditions))
            })
        else:
            # Fallback to original calculation if no summary data available
            design_data = valid_mapping[valid_mapping['experimental_design'] == design]
            replicates = design_data['biological_replicate'].unique()
            conditions = design_data['condition'].unique()
            total_files = len(design_data)
            
            avg_files_per_replicate = total_files / len(replicates) if len(replicates) > 0 else 0
            avg_files_per_condition = total_files / len(conditions) if len(conditions) > 0 else 0
            
            detailed_summary.append({
                'experimental_design': design,
                'total_files': total_files,
                'biological_replicates': len(replicates),
                'conditions': len(conditions),
                'avg_files_per_replicate': round(avg_files_per_replicate, 2),
                'avg_files_per_condition': round(avg_files_per_condition, 2),
                'replicate_list': ', '.join(sorted(replicates)),
                'condition_list': ', '.join(sorted(conditions))
            })
    
    # Create detailed summary DataFrame
    detailed_df = pd.DataFrame(detailed_summary)
    
    # Add exclusion information to the summary
    if excluded_data:
        print(f"\n=== EXCLUDED BATCHES ===")
        print(f"Note: The following batches are excluded from analysis due to missing control conditions:")
        for excluded in excluded_data:
            print(f"  {excluded['biological_replicate']}: {excluded['file_count']} files ({excluded['conditions']}) - {excluded['exclusion_reason']}")
        
        # Add excluded data to summary
        excluded_df = pd.DataFrame(excluded_data)
        combined_summary = pd.concat([summary_df, detailed_df, excluded_df], ignore_index=True)
    else:
        # Combine both summaries into one comprehensive file
        # First add the detailed breakdown, then add the statistics
        combined_summary = pd.concat([summary_df, detailed_df], ignore_index=True)
    
    # Save combined summary CSV
    summary_csv_path = os.path.join(output_dir, "experimental_design_summary.csv")
    combined_summary.to_csv(summary_csv_path, index=False)
    
    print(f"\nExperimental design summary saved to: {summary_csv_path}")
    
    # Print overall summary
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Total files processed: {len(valid_mapping)}")
    print(f"Experimental designs: {len(valid_mapping['experimental_design'].unique())}")
    print(f"Biological replicates: {len(valid_mapping['biological_replicate'].unique())}")
    print(f"Conditions: {len(valid_mapping['condition'].unique())}")
    
    return summary_df, detailed_df

# Create output directories for each experimental design using hardcoded mappings
for design_name, design_config in EXPERIMENTAL_DESIGNS.items():
    design_output_dir = os.path.join(output_dir, design_name)
    os.makedirs(design_output_dir, exist_ok=True)
    
    # Create directories based on the hardcoded mappings
    if design_name == "genotypes":
        for genotype in ["KO", "ApoE4", "Control", "ApoE2"]:
            condition_dir = os.path.join(design_output_dir, genotype)
            os.makedirs(condition_dir, exist_ok=True)
            individual_dir_batch = os.path.join(condition_dir, "individual_images_batch")
            individual_dir_individual = os.path.join(condition_dir, "individual_images_individual")
            csv_dir_batch = os.path.join(condition_dir, "csv_files_batch")
            csv_dir_individual = os.path.join(condition_dir, "csv_files_individual")
            os.makedirs(individual_dir_batch, exist_ok=True)
            os.makedirs(individual_dir_individual, exist_ok=True)
            os.makedirs(csv_dir_batch, exist_ok=True)
            os.makedirs(csv_dir_individual, exist_ok=True)
    
    elif design_name == "genotypes_statins":
        for genotype in ["KO", "ApoE4", "Control", "ApoE2"]:
            for treatment in ["Control", "Simvastatin"]:
                condition_dir = os.path.join(design_output_dir, f"{genotype}_{treatment}")
                os.makedirs(condition_dir, exist_ok=True)
                individual_dir_batch = os.path.join(condition_dir, "individual_images_batch")
                individual_dir_individual = os.path.join(condition_dir, "individual_images_individual")
                csv_dir_batch = os.path.join(condition_dir, "csv_files_batch")
                csv_dir_individual = os.path.join(condition_dir, "csv_files_individual")
                os.makedirs(individual_dir_batch, exist_ok=True)
                os.makedirs(individual_dir_individual, exist_ok=True)
                os.makedirs(csv_dir_batch, exist_ok=True)
                os.makedirs(csv_dir_individual, exist_ok=True)
    
    elif design_name == "domain_analysis":
        # Use the actual genotypes from the corrected mapping
        for genotype in ["Control", "ApoE2", "ApoE4", "ApoE2-NTD", "ApoE4-NTD"]:
            condition_dir = os.path.join(design_output_dir, genotype)
            os.makedirs(condition_dir, exist_ok=True)
            individual_dir_batch = os.path.join(condition_dir, "individual_images_batch")
            individual_dir_individual = os.path.join(condition_dir, "individual_images_individual")
            csv_dir_batch = os.path.join(condition_dir, "csv_files_batch")
            csv_dir_individual = os.path.join(condition_dir, "csv_files_individual")
            os.makedirs(individual_dir_batch, exist_ok=True)
            os.makedirs(individual_dir_individual, exist_ok=True)
            os.makedirs(csv_dir_batch, exist_ok=True)
            os.makedirs(csv_dir_individual, exist_ok=True)

# Hard-coded image exclusion
excluded_images = [
    "ApoE_Statins_B10_Simvastatin_GroupB_Red-2",
    "ApoE_Statins_B7_Control_GroupC_Red-4"
]

# ---- FILTERING PARAMETERS ----
# All filtering thresholds in one place for easy configuration
FILTER_PARAMS = {
    # Component size and skeleton length filters
    "MIN_SKELETON_LENGTH": 25,        # Minimum skeleton length (pixels)
    "MAX_SKELETON_LENGTH": 2000000,   # Maximum skeleton length (pixels)
    
    # Thickness filters (now using radius values to match original script)
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
    "SPIDER_WINDOW_LENGTH": 20,       # Window length for spider analysis (pixels) - matches original script
    "BRANCH_DENSITY_THRESHOLD": 0.09, # Branch density threshold (not percentile)
    "MIN_AVG_THICKNESS_PINK": 3,      # Minimum average thickness for pink regions (radius in pixels)
}

def get_experimental_design_info(filepath):
    """Determine which experimental design(s) this file belongs to and its condition."""
    filename = os.path.basename(filepath)
    filename_upper = filename.upper()
    
    # Check all hardcoded mappings first - this ensures consistent behavior
    
    # For files in Statins Experiment directory, prioritize genotypes_statins mapping
    if "/Statins Experiment/" in filepath:
        # Check genotypes_statins mappings first for statins experiment files
        if filename in GENOTYPES_STATINS_MAPPING:
            info = GENOTYPES_STATINS_MAPPING[filename]
            return {
                "experimental_design": info["experimental_design"],
                "biological_replicate": info["batch"],
                "biological_condition": info["biological_condition"],
                "genotype": info["genotype"]
            }
    
    # Check genotypes mappings first
    if filename in GENOTYPES_MAPPING:
        info = GENOTYPES_MAPPING[filename]
        return {
            "experimental_design": info["experimental_design"],
            "biological_replicate": info["batch"],
            "biological_condition": info["biological_condition"],
            "genotype": info["genotype"]
        }
    
    # Check genotypes_statins mappings (for files not in Statins Experiment directory)
    if filename in GENOTYPES_STATINS_MAPPING:
        info = GENOTYPES_STATINS_MAPPING[filename]
        return {
            "experimental_design": info["experimental_design"],
            "biological_replicate": info["batch"],
            "biological_condition": info["biological_condition"],
            "genotype": info["genotype"]
        }
    
    # Check domain analysis mappings
    if filename in DOMAIN_ANALYSIS_MAPPING:
        info = DOMAIN_ANALYSIS_MAPPING[filename]

        return {
            "experimental_design": "domain_analysis",
            "biological_replicate": info["batch"],
            "biological_condition": info["genotype"],  # Use genotype as biological condition for domain analysis
            "genotype": info["genotype"]
        }
    
    # If no hardcoded mapping found, try to assign based on directory structure and filename patterns
    if "/Statins Experiment/" in filepath:
        # Files in Statins Experiment directory should be genotypes_statins
        # Extract genotype from filename: GroupA=KO, GroupB=ApoE4, GroupC=Control, GroupD=ApoE2
        # For B27, genotypes are directly in filename: WT, KO, apoE2, apoE4
        genotype = "Control"  # Default
        
        if "B27" in filepath or "Batch 27" in filepath:
            # B27 uses different naming patterns in Control vs Statin directories
            if "/Control/" in filepath:
                # Control directory uses Group6 and different genotype names
                if "ApoE2" in filename:
                    genotype = "ApoE2"
                elif "ApoE4" in filename:
                    genotype = "ApoE4"
                elif "KO" in filename:
                    genotype = "KO"
                elif "WT" in filename:
                    genotype = "Control"
            else:
                # Statin directory uses Group3 and different genotype names
                if "WT" in filename:
                    genotype = "Control"
                elif "KO" in filename:
                    genotype = "KO"
                elif "apoE2" in filename:
                    genotype = "ApoE2"
                elif "apoE4" in filename:
                    genotype = "ApoE4"
        else:
            # Other batches use GroupA, GroupB, etc.
            if "GroupA" in filename:
                genotype = "KO"
            elif "GroupB" in filename:
                genotype = "ApoE4"
            elif "GroupC" in filename:
                genotype = "Control"
            elif "GroupD" in filename:
                genotype = "ApoE2"
        
        if "/Control/" in filepath:
            return {
                "experimental_design": "genotypes_statins",
                "biological_replicate": "B8" if "B8" in filepath else "B10" if "B10" in filepath else "B27",
                "biological_condition": "Control",
                "genotype": genotype
            }
        elif "/Statin/" in filepath:
            # For B27, all files in /Statin/ directory are Simvastatin
            # Control files are in separate /Control/ directory
            treatment = "Simvastatin"
            
            return {
                "experimental_design": "genotypes_statins",
                "biological_replicate": "B8" if "B8" in filepath else "B10" if "B10" in filepath else "B27",
                "biological_condition": treatment,
                "genotype": genotype
            }
    elif "/Control/" in filepath and "/Statins Experiment/" not in filepath:
        # Files in Control directory (not Statins Experiment) should be genotypes
        # Extract genotype from filename: GroupA=KO, GroupB=ApoE4, GroupC=Control, GroupD=ApoE2
        # For B27, genotypes are directly in filename: WT, KO, apoE2, apoE4
        genotype = "Control"  # Default
        
        if "B27" in filepath or "Batch 27" in filepath:
            # B27 uses different naming patterns in Control vs Statin directories
            if "/Control/" in filepath:
                # Control directory uses Group6 and different genotype names
                if "ApoE2" in filename:
                    genotype = "ApoE2"
                elif "ApoE4" in filename:
                    genotype = "ApoE4"
                elif "KO" in filename:
                    genotype = "KO"
                elif "WT" in filename:
                    genotype = "Control"
            else:
                # Statin directory uses Group3 and different genotype names
                if "WT" in filename:
                    genotype = "Control"
                elif "KO" in filename:
                    genotype = "KO"
                elif "apoE2" in filename:
                    genotype = "ApoE2"
                elif "apoE4" in filename:
                    genotype = "ApoE4"
        else:
            # Other batches use GroupA, GroupB, etc.
            if "GroupA" in filename:
                genotype = "KO"
            elif "GroupB" in filename:
                genotype = "ApoE4"
            elif "GroupC" in filename:
                genotype = "Control"
            elif "GroupD" in filename:
                genotype = "ApoE2"
        
        return {
            "experimental_design": "genotypes",
            "biological_replicate": "B8" if "B8" in filepath else "B10" if "B10" in filepath else "B11" if "B11" in filepath else "B12",
            "biological_condition": "Control",
            "genotype": genotype
        }
    
    # If no hardcoded mapping found, return None to indicate unknown file
    return None

def generate_cdf_plots(data_dict, output_dir, design_name):
    """Generate CDF plots for the collected data using functions from blue filter."""
    print(f"\nGenerating CDF plots for {design_name}...")
    
    # Convert defaultdict to regular dict if needed
    if hasattr(data_dict, 'default_factory'):
        data_dict = dict(data_dict)
    
    # Use centralized color scheme
    # colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 12 distinct colors - REPLACED
    
    # Define logical order for plotting (this will determine legend order)
    plot_order = {
        'genotypes': ['KO', 'ApoE4', 'Control', 'ApoE2'],
        'genotypes_statins': ['KO_Control', 'KO_Simvastatin', 'ApoE4_Control', 'ApoE4_Simvastatin', 'Control_Control', 'Control_Simvastatin', 'ApoE2_Control', 'ApoE2_Simvastatin'],
        'domain_analysis': ['Control', 'ApoE2-NTD', 'ApoE4-NTD', 'ApoE2', 'ApoE4']
    }

    # Collect all average thickness values by group
    group_thicknesses = defaultdict(list)
    
    for condition, thickness_values in data_dict.items():
        if thickness_values:
            # Handle different data types and flatten the data
            flat_thicknesses = []
            if isinstance(thickness_values, (list, np.ndarray)):
                for item in thickness_values:
                    if isinstance(item, (list, np.ndarray)):
                        flat_thicknesses.extend(item)
                    else:
                        flat_thicknesses.append(item)
            else:
                flat_thicknesses = [thickness_values]
            
            # Convert to numpy array and filter
            if flat_thicknesses:
                thicknesses = np.array(flat_thicknesses)
                thicknesses = thicknesses[np.isfinite(thicknesses)]
                thicknesses = thicknesses[thicknesses > 0]
                if len(thicknesses) > 0:
                    group_thicknesses[condition].extend(thicknesses.tolist())
    
    # Create CDF plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.set_title(f'CDF Analysis: {EXPERIMENTAL_DESIGNS[design_name]["description"]}', fontsize=14)
    
    # Plot in logical order to control legend order
    current_order = plot_order.get(design_name, list(group_thicknesses.keys()))
    
    for i, condition in enumerate(current_order):
        if condition in group_thicknesses and group_thicknesses[condition]:
            thicknesses = group_thicknesses[condition]
            color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
            thicknesses = safe_convert_to_numeric(thicknesses)
            
            if len(thicknesses) > 0:
                sorted_thicknesses = np.sort(thicknesses)
                cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                
                ax.plot(sorted_thicknesses, cdf, color=color, linewidth=3, 
                       label=f'{condition} (n={len(thicknesses)} components)')
    
    ax.set_xlabel('Component Average Thickness (pixels)')
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'{design_name}_cdf.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  CDF plot saved: {plot_path}")
    
    # Save the processed CDF data
    cdf_csv_filename = f'{design_name}_cdf_data.csv'
    cdf_csv_path = os.path.join(output_dir, cdf_csv_filename)
    
    # Create a DataFrame with the processed CDF data
    cdf_data = []
    for condition in current_order:
        if condition in group_thicknesses and group_thicknesses[condition]:
            thicknesses = group_thicknesses[condition]
            thicknesses = safe_convert_to_numeric(thicknesses)
            
            if len(thicknesses) > 0:
                sorted_thicknesses = np.sort(thicknesses)
                cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                
                for i, (thickness, cdf_val) in enumerate(zip(sorted_thicknesses, cdf)):
                    cdf_data.append({
                        'condition': condition,
                        'thickness': thickness,
                        'cdf_value': cdf_val
                    })
    
    cdf_df = pd.DataFrame(cdf_data)
    cdf_df.to_csv(cdf_csv_path, index=False)
    print(f"  CDF data saved: {cdf_csv_path}")
    
    # Create Control-only plot if applicable
    if design_name == "genotypes":
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        ax.set_title(f'CDF Analysis: {EXPERIMENTAL_DESIGNS[design_name]["description"]} (Control Only)', fontsize=14)
        
        # Plot Control conditions in logical order
        control_order = plot_order['genotypes']
        
        for i, condition in enumerate(control_order):
            if condition in group_thicknesses and group_thicknesses[condition]:
                thicknesses = group_thicknesses[condition]
                color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
                thicknesses = safe_convert_to_numeric(thicknesses)
                
                if len(thicknesses) > 0:
                    sorted_thicknesses = np.sort(thicknesses)
                    cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                    
                    ax.plot(sorted_thicknesses, cdf, color=color, linewidth=3, 
                           label=f'{condition} (n={len(thicknesses)} components)')
        
        ax.set_xlabel('Component Average Thickness (pixels)')
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        # Save Control-only plot
        control_plot_filename = f'{design_name}_control_only_cdf.png'
        control_plot_path = os.path.join(output_dir, control_plot_filename)
        plt.savefig(control_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Control-only CDF plot saved: {control_plot_path}")
    
    return plot_path

def safe_convert_to_numeric(data_list):
    """
    Safely convert a list of values to numeric values, filtering out non-numeric data.
    Handles nested lists, defaultdict objects, and inhomogeneous data structures.
    
    Args:
        data_list: List of values (can be mixed types, nested lists, or defaultdict)
    
    Returns:
        numpy.ndarray: Array of numeric values > 0
    """
    numeric_data = []
    
    # Handle defaultdict objects
    if hasattr(data_list, 'default_factory'):
        data_list = list(data_list.values())
    
    # Handle different data types
    if isinstance(data_list, (list, np.ndarray)):
        for item in data_list:
            if isinstance(item, (list, np.ndarray)):
                # Recursively handle nested lists
                nested_data = safe_convert_to_numeric(item)
                numeric_data.extend(nested_data)
            else:
                # Handle single values
                try:
                    numeric_value = float(item)
                    if numeric_value > 0 and np.isfinite(numeric_value):
                        numeric_data.append(numeric_value)
                except (ValueError, TypeError):
                    continue  # Skip non-numeric values
    else:
        # Handle single values
        try:
            numeric_value = float(data_list)
            if numeric_value > 0 and np.isfinite(numeric_value):
                numeric_data.append(numeric_value)
        except (ValueError, TypeError):
            pass  # Skip non-numeric values
    
    return np.array(numeric_data)

def generate_normalized_cdf_plots(data_dict, batch_data, output_dir, design_name):
    """Generate CDF plots with thickness values normalized to WT control within each biological replicate."""
    print(f"\nGenerating normalized CDF plots for {design_name}...")
    
    # Convert defaultdict to regular dict if needed
    if hasattr(data_dict, 'default_factory'):
        data_dict = dict(data_dict)
    if hasattr(batch_data, 'default_factory'):
        batch_data = dict(batch_data)
    
    # Use centralized color scheme
    # colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 12 distinct colors - REPLACED
    
        # Define logical order for plotting (this will determine legend order)
    plot_order = {
        'genotypes': ['KO', 'ApoE4', 'Control', 'ApoE2'],
        'genotypes_statins': ['KO_Control', 'KO_Simvastatin', 'ApoE4_Control', 'ApoE4_Simvastatin', 'Control_Control', 'Control_Simvastatin', 'ApoE2_Control', 'ApoE2_Simvastatin'],
        'domain_analysis': ['Control', 'ApoE2-NTD', 'ApoE4-NTD', 'ApoE2', 'ApoE4']
    }

    # Collect normalized thickness data
    normalized_data = defaultdict(list)
    
    if design_name == "genotypes":
        # Check if we have batch_data structure or component-level data
        if isinstance(batch_data, dict) and any(isinstance(v, dict) for v in batch_data.values()):
            # This is batch-level data structure
            print("  Using batch-level data structure for normalization")
            for batch in ["B8", "B10", "B11", "B12"]:
                if batch in batch_data:
                    print(f"  Processing biological replicate: {batch}")
                    
                    # Find Control data for this batch
                    control_data = batch_data[batch].get("Control", [])
                    
                    if control_data:
                        control_data = safe_convert_to_numeric(control_data)
                        control_data = control_data[control_data > 0]  # Remove zeros
                        
                        if len(control_data) > 0:
                            normalization_factor = np.mean(control_data)
                            print(f"    Control mean for {batch}: {normalization_factor:.3f}")
                            
                            # Normalize all data for this batch
                            for condition, thickness_values in batch_data[batch].items():
                                if thickness_values:
                                    thickness_values = safe_convert_to_numeric(thickness_values)
                                    thickness_values = thickness_values[thickness_values > 0]  # Remove zeros
                                    
                                    if len(thickness_values) > 0:
                                        normalized_thicknesses = thickness_values / normalization_factor
                                        normalized_data[condition].extend(normalized_thicknesses.tolist())
                    else:
                        print(f"    Warning: No valid Control data for {batch}")
                else:
                    print(f"    Warning: No Control data found for {batch}")
        else:
            # This is component-level data - normalize using overall WT control mean
            print("  Using component-level data structure for normalization")
            
            # Get Control data from the main data_dict
            control_data = data_dict.get("Control", [])
            
            if control_data:
                control_data = safe_convert_to_numeric(control_data)
                control_data = control_data[control_data > 0]  # Remove zeros
                
                if len(control_data) > 0:
                    normalization_factor = np.mean(control_data)
                    print(f"    Control mean for normalization: {normalization_factor:.3f}")
                    
                    # Normalize all data using the WT control mean
                    for condition, thickness_values in data_dict.items():
                        if thickness_values:
                            thickness_values = safe_convert_to_numeric(thickness_values)
                            thickness_values = thickness_values[thickness_values > 0]  # Remove zeros
                            
                            if len(thickness_values) > 0:
                                normalized_thicknesses = thickness_values / normalization_factor
                                normalized_data[condition].extend(normalized_thicknesses.tolist())
                                print(f"    Normalized {condition}: {len(normalized_thicknesses)} components")
                else:
                    print("    Warning: No valid Control data found")
            else:
                print("    Warning: No Control data found in data_dict")
    
    elif design_name == "genotypes_statins":
        # Handle B8 and B10 separately, Group3+Group6 together
        for batch in ["B8", "B10"]:
            if batch in batch_data:
                print(f"  Processing biological replicate: {batch}")
                
                # Find WT control data for this batch (ONLY Control, not Statin)
                wt_control_data = batch_data[batch].get("Control", [])
                
                if wt_control_data:
                    wt_control_data = safe_convert_to_numeric(wt_control_data)
                    wt_control_data = wt_control_data[wt_control_data > 0]  # Remove zeros
                    
                    if len(wt_control_data) > 0:
                        normalization_factor = np.mean(wt_control_data)
                        print(f"    Control mean for {batch}: {normalization_factor:.3f}")
                        
                        # Normalize all data for this batch
                        for condition, thickness_values in batch_data[batch].items():
                            if thickness_values:
                                thickness_values = safe_convert_to_numeric(thickness_values)
                                thickness_values = thickness_values[thickness_values > 0]  # Remove zeros
                                
                                if len(thickness_values) > 0:
                                    normalized_thicknesses = thickness_values / normalization_factor
                                    normalized_data[condition].extend(normalized_thicknesses.tolist())
                else:
                    print(f"    Warning: No valid Control data for {batch}")
            else:
                print(f"    Warning: No Control data found for {batch}")
        
        # Handle Group3+Group6 together as one biological replicate
        group3_data = batch_data.get("B27", {})  # Group3 and Group6 data
        if group3_data:
            print(f"  Processing biological replicate: B27 (Group3+Group6)")
            
            # Find WT control data for Group3+Group6 (ONLY Control, not Statin)
            wt_control_data = group3_data.get("Control", [])
            
            if wt_control_data:
                wt_control_data = safe_convert_to_numeric(wt_control_data)
                wt_control_data = wt_control_data[wt_control_data > 0]  # Remove zeros
                
                if len(wt_control_data) > 0:
                    normalization_factor = np.mean(wt_control_data)
                    print(f"    Control mean for B27: {normalization_factor:.3f}")
                    
                    # Normalize all data for Group3+Group6
                    for condition, thickness_values in group3_data.items():
                        if thickness_values:
                            thickness_values = safe_convert_to_numeric(thickness_values)
                            thickness_values = thickness_values[thickness_values > 0]  # Remove zeros
                            
                            if len(thickness_values) > 0:
                                normalized_thicknesses = thickness_values / normalization_factor
                                normalized_data[condition].extend(normalized_thicknesses.tolist())
                else:
                    print(f"    Warning: No valid Control data for B27")
            else:
                print(f"    Warning: No Control data found for B27")
    
    elif design_name == "domain_analysis":
        # Each batch (B1, B2, B3) is a separate biological replicate
        for batch in ["B1", "B2", "B3"]:
            if batch in batch_data:
                print(f"  Processing biological replicate: {batch}")
                
                # Find Control data for this batch (normalize to Control, not WT_Control)
                control_data = batch_data[batch].get("Control", [])
                
                if control_data:
                    control_data = safe_convert_to_numeric(control_data)
                    control_data = control_data[control_data > 0]  # Remove zeros
                    
                    if len(control_data) > 0:
                        normalization_factor = np.mean(control_data)
                        print(f"    Control mean for {batch}: {normalization_factor:.3f}")
                        
                        # Normalize all data for this batch
                        for condition, thickness_values in batch_data[batch].items():
                            if thickness_values:
                                thickness_values = safe_convert_to_numeric(thickness_values)
                                thickness_values = thickness_values[thickness_values > 0]  # Remove zeros
                                
                                if len(thickness_values) > 0:
                                    normalized_thicknesses = thickness_values / normalization_factor
                                    normalized_data[condition].extend(normalized_thicknesses.tolist())
                else:
                    print(f"    Warning: No valid Control data for {batch}")
            else:
                print(f"    Warning: No Control data found for {batch}")
    
    # Create normalized CDF plot
    if normalized_data:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Adjust title based on design
        if design_name == "domain_analysis":
            title = f'Component-Level Normalized CDF Analysis: {EXPERIMENTAL_DESIGNS[design_name]["description"]}\n(Normalized to Control baseline within each biological replicate)'
        else:
            title = f'Component-Level Normalized CDF Analysis: {EXPERIMENTAL_DESIGNS[design_name]["description"]}\n(Normalized to Control baseline within each biological replicate)'
        
        ax.set_title(title, fontsize=14)
        
        # Plot in logical order to control legend order
        current_order = plot_order.get(design_name, list(normalized_data.keys()))
        
        for condition in current_order:
            if condition in normalized_data and normalized_data[condition]:
                thicknesses = normalized_data[condition]
                color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
                thicknesses = np.array(thicknesses)
                
                if len(thicknesses) > 0:
                    sorted_thicknesses = np.sort(thicknesses)
                    cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                    
                    ax.plot(sorted_thicknesses, cdf, color=color, linewidth=3, 
                           label=f'{condition} (n={len(thicknesses)} components)')
        
        # Adjust x-axis label based on design
        if design_name == "domain_analysis":
            ax.set_xlabel('Normalized Average Blue Component Thickness per Component (Control = 1.0 per biological replicate)')
        else:
            ax.set_xlabel('Normalized Average Blue Component Thickness per Component (Control = 1.0 per biological replicate)')
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        # Save normalized plot
        normalized_plot_filename = f'{design_name}_component_level_normalized_cdf.png'
        normalized_plot_path = os.path.join(output_dir, normalized_plot_filename)
        plt.savefig(normalized_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Component-level normalized CDF plot saved: {normalized_plot_path}")
        
        # Create Control-only normalized plot if applicable
        if design_name == "genotypes":
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            ax.set_title(f'Component-Level Normalized CDF Analysis: {EXPERIMENTAL_DESIGNS[design_name]["description"]}\n(Control Only, Normalized to Control)', fontsize=14)
            
            # Plot Control conditions in logical order  
            control_order = plot_order['genotypes']
            
            for condition in control_order:
                if condition in normalized_data and normalized_data[condition]:
                    thicknesses = normalized_data[condition]
                    color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
                    thicknesses = np.array(thicknesses)
                    
                    if len(thicknesses) > 0:
                        sorted_thicknesses = np.sort(thicknesses)
                        cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                        
                        ax.plot(sorted_thicknesses, cdf, color=color, linewidth=3, 
                               label=f'{condition} (n={len(thicknesses)} components)')
            
            # Adjust x-axis label based on design
            if design_name == "domain_analysis":
                ax.set_xlabel('Normalized Average Blue Component Thickness per Component (Control = 1.0 per biological replicate)')
            else:
                ax.set_xlabel('Normalized Average Blue Component Thickness per Component (Control = 1.0)')
            ax.set_ylabel('Cumulative Probability')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            
            # Save Control-only normalized plot
            control_normalized_plot_filename = f'{design_name}_component_level_control_only_normalized_cdf.png'
            control_normalized_plot_path = os.path.join(output_dir, control_normalized_plot_filename)
            plt.savefig(control_normalized_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Component-level Control-only normalized CDF plot saved: {control_normalized_plot_path}")
    else:
        print("  Warning: No normalized data available for plotting")
    
    return normalized_data
    
def save_cdf_data_csv(data_dict, output_dir, design_name, plot_type="component_level"):
    """Save CDF data as CSV files with each column as a different biological condition."""
    print(f"\nSaving {plot_type} CDF data CSV for {design_name}...")
    
    # Define expected conditions based on design
    if design_name == "genotypes":
        expected_conditions = ['KO', 'ApoE4', 'Control', 'ApoE2']
    elif design_name == "genotypes_statins":
        expected_conditions = ['KO_Control', 'KO_Simvastatin', 'ApoE4_Control', 'ApoE4_Simvastatin', 'Control_Control', 'Control_Simvastatin', 'ApoE2_Control', 'ApoE2_Simvastatin']
    elif design_name == "domain_analysis":
        expected_conditions = ['Control', 'ApoE2-NTD', 'ApoE4-NTD', 'ApoE2', 'ApoE4']
    else:
        expected_conditions = list(data_dict.keys())
    
    # Find the maximum length of any condition
    max_length = 0
    for condition in expected_conditions:
        if condition in data_dict and data_dict[condition]:
            thicknesses = np.array(data_dict[condition])
            thicknesses = thicknesses[thicknesses > 0]  # Remove zeros
            max_length = max(max_length, len(thicknesses))
    
    # Create DataFrame with each condition as a column
    csv_data = {}
    for condition in expected_conditions:
        if condition in data_dict and data_dict[condition]:
            thicknesses = np.array(data_dict[condition])
            thicknesses = thicknesses[thicknesses > 0]  # Remove zeros
            # Pad with NaN to make all columns same length
            data = thicknesses.tolist() + [np.nan] * (max_length - len(thicknesses))
            csv_data[condition] = data
        else:
            csv_data[condition] = [np.nan] * max_length  # Empty column with NaN
    
    if csv_data:
        cdf_df = pd.DataFrame(csv_data)
        csv_filename = f'{design_name}_{plot_type}_cdf_data.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        cdf_df.to_csv(csv_path, index=False)
        print(f"  {plot_type} CDF data saved: {csv_path}")
        print(f"  Columns: {list(cdf_df.columns)}")
        print(f"  Data points per condition:")
        for col in cdf_df.columns:
            non_null_count = cdf_df[col].notna().sum()
            print(f"    {col}: {non_null_count} data points")
        return csv_path
    else:
        print(f"  No {plot_type} CDF data to save")
        return None

def save_normalized_data_csvs(normalized_data, output_dir, design_name):
    """Save normalized thickness data as CSV files for statistical analysis."""
    print(f"\nSaving normalized data CSV for {design_name}...")
    
    if design_name == "genotypes":
        # Create DataFrame with 4 columns for genotypes
        csv_data = {}
        
        # Define expected columns in logical order: KO, ApoE4, Control, ApoE2
        expected_conditions = ['KO', 'ApoE4', 'Control', 'ApoE2']
        
        # Find the maximum length of any condition
        max_length = 0
        for condition in expected_conditions:
            if condition in normalized_data and normalized_data[condition]:
                max_length = max(max_length, len(normalized_data[condition]))
        
        for condition in expected_conditions:
            if condition in normalized_data and normalized_data[condition]:
                # Pad with NaN to make all columns same length
                data = normalized_data[condition] + [np.nan] * (max_length - len(normalized_data[condition]))
                csv_data[condition] = data
            else:
                csv_data[condition] = [np.nan] * max_length  # Empty column with NaN
        
        # Create DataFrame
        df = pd.DataFrame(csv_data)
        
        # Save CSV
        csv_filename = f'{design_name}_normalized_data.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data points per condition:")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            print(f"    {col}: {non_null_count} components")
    
    elif design_name == "genotypes_statins":
        # Create DataFrame with 8 columns for genotypes_statins
        csv_data = {}
        
        # Define expected columns in logical order: KO (both), ApoE4 (both), Control (both), ApoE2 (both)
        expected_conditions = [
            'KO_Control', 'KO_Simvastatin',
            'ApoE4_Control', 'ApoE4_Simvastatin', 
            'Control_Control', 'Control_Simvastatin',
            'ApoE2_Control', 'ApoE2_Simvastatin'
        ]
        
        # Find the maximum length of any condition
        max_length = 0
        for condition in expected_conditions:
            if condition in normalized_data and normalized_data[condition]:
                max_length = max(max_length, len(normalized_data[condition]))
        
        for condition in expected_conditions:
            if condition in normalized_data and normalized_data[condition]:
                # Pad with NaN to make all columns same length
                data = normalized_data[condition] + [np.nan] * (max_length - len(normalized_data[condition]))
                csv_data[condition] = data
            else:
                csv_data[condition] = [np.nan] * max_length  # Empty column with NaN
        
        # Create DataFrame
        df = pd.DataFrame(csv_data)
        
        # Save CSV
        csv_filename = f'{design_name}_normalized_data.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data points per condition:")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            print(f"    {col}: {non_null_count} components")
    
    elif design_name == "domain_analysis":
        # Create DataFrame with 6 columns for domain_analysis
        csv_data = {}
        
        # Define expected columns in logical order: Control, ApoE2-NTD, ApoE4-NTD, ApoE2, ApoE4
        expected_conditions = ['Control', 'ApoE2-NTD', 'ApoE4-NTD', 'ApoE2', 'ApoE4']
        
        # Find the maximum length of any condition
        max_length = 0
        for condition in expected_conditions:
            if condition in normalized_data and normalized_data[condition]:
                max_length = max(max_length, len(normalized_data[condition]))
        
        for condition in expected_conditions:
            if condition in normalized_data and normalized_data[condition]:
                # Pad with NaN to make all columns same length
                data = normalized_data[condition] + [np.nan] * (max_length - len(normalized_data[condition]))
                csv_data[condition] = data
            else:
                csv_data[condition] = [np.nan] * max_length  # Empty column with NaN
        
        # Create DataFrame
        df = pd.DataFrame(csv_data)
        
        # Save CSV
        csv_filename = f'{design_name}_normalized_data.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data points per condition:")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            print(f"    {col}: {non_null_count} components")
    
    else:
        # Default case for other design names (like "genotypes", "genotypes_statins")
        csv_data = {}
        
        # Use all conditions found in normalized_data
        expected_conditions = list(normalized_data.keys())
        
        if expected_conditions:
            # Find the maximum length of any condition
            max_length = 0
            for condition in expected_conditions:
                if normalized_data[condition]:
                    max_length = max(max_length, len(normalized_data[condition]))
            
            for condition in expected_conditions:
                if normalized_data[condition]:
                    # Pad with NaN to make all columns same length
                    data = normalized_data[condition] + [np.nan] * (max_length - len(normalized_data[condition]))
                    csv_data[condition] = data
                else:
                    csv_data[condition] = [np.nan] * max_length  # Empty column with NaN
            
            # Create DataFrame
            df = pd.DataFrame(csv_data)
            
            # Save CSV
            csv_filename = f'{design_name}_normalized_data.csv'
            csv_path = os.path.join(output_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Data points per condition:")
            for col in df.columns:
                non_null_count = df[col].notna().sum()
                print(f"    {col}: {non_null_count} components")
        else:
            print(f"  No normalized data found for {design_name}")
            csv_path = None
    
    return csv_path

def generate_replicate_normalized_cdf_plots(output_dir, design_name):
    """
    Generate normalized CDF plots where thicknesses are normalized to control condition 
    within each biological replicate. Creates separate plots for image-level and batch-level thresholds.
    
    Args:
        output_dir: Output directory for plots
        design_name: Name of experimental design
    """
    print(f"\nGenerating replicate-normalized CDF plots for {design_name}...")
    
    # Define control condition names for each experimental design
    control_conditions = {
        "genotypes": "Control",
        "genotypes_statins": "Control_Control", 
        "domain_analysis": "Control"
    }
    
    if design_name not in control_conditions:
        print(f"  Unknown experimental design: {design_name}")
        return
        
    control_condition = control_conditions[design_name]
    
    # Load the comprehensive file mapping
    mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
    design_mapping = mapping_df[mapping_df['experimental_design'] == design_name]
    
    if design_mapping.empty:
        print(f"  No files found for experimental design: {design_name}")
        return
    
    # Get all biological replicates for this design
    biological_replicates = sorted(design_mapping['biological_replicate'].unique())
    design_output_dir = os.path.join("comprehensive_cdf_analysis_results", design_name)
    
    # Process both threshold types
    for threshold_type in ["individual", "batch"]:
        print(f"  Processing {threshold_type} threshold data...")
        
        # Collect normalized data for all replicates
        all_normalized_data = defaultdict(list)  # condition -> list of normalized thicknesses
        
        # Process each biological replicate separately
        for biological_replicate in biological_replicates:
            print(f"    Processing biological replicate: {biological_replicate}")
            
            # Get files for this replicate
            replicate_files = design_mapping[design_mapping['biological_replicate'] == biological_replicate]
            
            # Collect all thickness data for this replicate
            replicate_data = defaultdict(list)  # condition -> list of thicknesses
            
            for _, row in replicate_files.iterrows():
                filename = row['filename']
                condition = row['condition']
                base_filename = os.path.splitext(filename)[0]
                
                # Construct CSV path
                csv_path = os.path.join(design_output_dir, condition, f"csv_files_{threshold_type}", f"{base_filename}_blue_components.csv")
                
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty and 'average_thickness' in df.columns:
                            thicknesses = df['average_thickness'].tolist()
                            replicate_data[condition].extend(thicknesses)
                    except Exception as e:
                        print(f"      Warning: Error reading {csv_path}: {e}")
            
            # Calculate control average for this replicate
            if control_condition in replicate_data and replicate_data[control_condition]:
                control_avg = np.mean(replicate_data[control_condition])
                print(f"      Control average thickness: {control_avg:.3f}")
                
                # Normalize all conditions by the control average
                for condition, thicknesses in replicate_data.items():
                    if thicknesses:  # Only process if we have data
                        normalized_thicknesses = [t / control_avg for t in thicknesses]
                        all_normalized_data[condition].extend(normalized_thicknesses)
                        print(f"        {condition}: {len(thicknesses)} components normalized")
            else:
                print(f"      Warning: No control data found for replicate {biological_replicate}")
        
        if not all_normalized_data:
            print(f"    No normalized data available for {threshold_type} threshold")
            continue
        
        # Generate CDF plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get all conditions and sort them
        conditions = sorted(all_normalized_data.keys())
        
        # Plot CDFs for each condition
        for condition in conditions:
            if condition in all_normalized_data and all_normalized_data[condition]:
                thickness_values = all_normalized_data[condition]
                
                # Calculate CDF using scipy
                res = ecdf(thickness_values)
                x_values = res.cdf.quantiles
                y_values = res.cdf.probabilities
                
                # Get color for this condition
                color = get_color_for_condition(design_name, condition)
                
                # Plot CDF
                ax.plot(x_values, y_values, label=f'{condition} (n={len(thickness_values)})', 
                       color=color, linewidth=2, alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Normalized Thickness (relative to control)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'{design_name} - Normalized Thickness CDF ({threshold_type.title()} Thresholds)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits to start where data actually begins
        if all_normalized_data:
            all_values = []
            for values in all_normalized_data.values():
                all_values.extend(values)
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                x_margin = (max_val - min_val) * 0.05  # 5% margin
                ax.set_xlim(left=max(0, min_val - x_margin), right=max_val + x_margin)
        
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'{design_name}_normalized_thickness_cdf_{threshold_type}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Normalized CDF plot saved: {plot_path}")
        
        # Save data as CSV
        csv_filename = f'{design_name}_normalized_thickness_data_{threshold_type}.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Create CSV data
        csv_data = []
        for condition, thickness_values in all_normalized_data.items():
            for thickness in thickness_values:
                csv_data.append({
                    'condition': condition,
                    'normalized_thickness': thickness,
                    'threshold_type': threshold_type
                })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(csv_path, index=False)
        print(f"    Normalized thickness data saved: {csv_path}")
        print(f"    Total data points: {len(csv_data)}")

def generate_thick_thin_ratio_graphs(output_dir, design_name):
    """
    Generate thick-thin ratio graphs with image-level and replicate-level analysis.
    Creates 4 graphs per experimental design: 2 threshold types × 2 analysis levels.
    Includes one-way ANOVA with significance markers.
    
    Args:
        output_dir: Output directory for plots
        design_name: Name of experimental design
    """
    print(f"\nGenerating thick-thin ratio graphs for {design_name}...")
    
    # Define control condition names for each experimental design
    control_conditions = {
        "genotypes": "Control",
        "genotypes_statins": "Control_Control", 
        "domain_analysis": "Control"
    }
    
    if design_name not in control_conditions:
        print(f"  Unknown experimental design: {design_name}")
        return
        
    control_condition = control_conditions[design_name]
    
    # Load the comprehensive file mapping
    mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
    design_mapping = mapping_df[mapping_df['experimental_design'] == design_name]
    
    if design_mapping.empty:
        print(f"  No files found for experimental design: {design_name}")
        return
    
    # Get all biological replicates for this design
    biological_replicates = sorted(design_mapping['biological_replicate'].unique())
    design_output_dir = os.path.join("comprehensive_cdf_analysis_results", design_name)
    
    def add_significance_markers(ax, data_by_condition, x_positions, y_max):
        """Add ANOVA significance markers and pairwise comparison bars to the plot."""
        try:
            # Prepare data for ANOVA
            condition_names = [condition for condition, data in data_by_condition.items() if len(data) > 0]
            condition_data = [data for condition, data in data_by_condition.items() if len(data) > 0]
            
            if len(condition_data) < 2:
                return
                
            # Perform one-way ANOVA
            f_stat, p_value = f_oneway(*condition_data)
            
            # Add overall ANOVA result
            if p_value < 0.001:
                sig_text = "***"
            elif p_value < 0.01:
                sig_text = "**"
            elif p_value < 0.05:
                sig_text = "*"
            else:
                sig_text = "ns"
            
            # Add overall ANOVA text - make it more prominent
            ax.text(0.02, 0.98, f'ANOVA: {sig_text} (p={p_value:.3f})', 
                   transform=ax.transAxes, fontsize=12, ha='left', va='top', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black', alpha=0.9))
            
            # Perform pairwise comparisons if ANOVA is significant
            if p_value < 0.05:
                # Create mapping from condition name to x position using the actual x_positions passed to function
                # The conditions should be in the same order as they appear in the plot
                condition_to_x = {}
                conditions_with_data = [condition for condition, data in data_by_condition.items() if len(data) > 0]
                
                # Sort conditions the same way as in the main plotting code
                sorted_conditions = sorted(conditions_with_data)
                
                # Map each condition to its actual x position from the bar plot
                for i, condition in enumerate(sorted_conditions):
                    if i < len(x_positions):
                        condition_to_x[condition] = x_positions[i]
                    else:
                        condition_to_x[condition] = i
                
                # Perform pairwise t-tests with Bonferroni correction
                pairs_to_test = []
                for i in range(len(condition_names)):
                    for j in range(i + 1, len(condition_names)):
                        pairs_to_test.append((condition_names[i], condition_names[j]))
                
                # Bonferroni correction
                alpha = 0.05 / len(pairs_to_test) if len(pairs_to_test) > 0 else 0.05
                
                significant_pairs = []
                for cond1, cond2 in pairs_to_test:
                    if cond1 in data_by_condition and cond2 in data_by_condition:
                        data1 = data_by_condition[cond1]
                        data2 = data_by_condition[cond2]
                        if len(data1) > 0 and len(data2) > 0:
                            _, p_val = ttest_ind(data1, data2)
                            if p_val < alpha:
                                significant_pairs.append((cond1, cond2, p_val))
                
                # Draw significance bars - make them more prominent
                bar_height = y_max * 0.08  # Increased from 0.05
                current_height = y_max * 1.05  # Start a bit lower
                
                for i, (cond1, cond2, p_val) in enumerate(significant_pairs[:6]):  # Allow up to 6 bars
                    if cond1 in condition_to_x and cond2 in condition_to_x:
                        x1 = condition_to_x[cond1]
                        x2 = condition_to_x[cond2]
                        
                        print(f"        Drawing comparison line: {cond1} (x={x1}) vs {cond2} (x={x2})")
                        
                        # Determine significance level
                        if p_val < 0.001:
                            sig_symbol = "***"
                        elif p_val < 0.01:
                            sig_symbol = "**"
                        elif p_val < 0.05:
                            sig_symbol = "*"
                        else:
                            continue  # Skip non-significant
                        
                        # Calculate line position - start higher to avoid overlapping with bars
                        line_y = current_height + (i * bar_height * 1.3)  # More spacing between lines
                        
                        # Draw horizontal line connecting the two bars - make it thicker and more visible
                        ax.plot([x1, x2], [line_y, line_y], 'k-', linewidth=3, alpha=1.0, zorder=10)
                        
                        # Draw vertical ticks down to the bars - make them taller and thicker
                        tick_height = bar_height * 0.3
                        ax.plot([x1, x1], [line_y - tick_height, line_y + tick_height], 'k-', linewidth=3, alpha=1.0, zorder=10)
                        ax.plot([x2, x2], [line_y - tick_height, line_y + tick_height], 'k-', linewidth=3, alpha=1.0, zorder=10)
                        
                        # Add significance symbol - make it larger and more prominent
                        ax.text((x1 + x2) / 2, line_y + tick_height + bar_height*0.1, sig_symbol, 
                               ha='center', va='bottom', fontsize=16, weight='bold', 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', edgecolor='black', alpha=0.95),
                               zorder=11)
                
                # Adjust y-axis limit to accommodate significance bars
                if significant_pairs:
                    new_y_max = current_height + len(significant_pairs[:6]) * bar_height * 1.6 + bar_height
                    ax.set_ylim(top=new_y_max)
                   
        except Exception as e:
            print(f"    Warning: Could not perform statistical analysis: {e}")
    
    # Process both threshold types
    for threshold_type in ["individual", "batch"]:
        print(f"  Processing {threshold_type} threshold data...")
        
        # Create figure with 1 row and 2 columns (image-level + replicate-level)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # === IMAGE-LEVEL ANALYSIS (Left subplot) ===
        ax1 = axes[0]
        
        # Collect image-level data
        all_image_data = defaultdict(list)  # condition -> list of normalized thick percentages
        
        # Process each biological replicate separately for normalization
        for biological_replicate in biological_replicates:
            print(f"    Processing biological replicate: {biological_replicate}")
            
            # Get files for this replicate
            replicate_files = design_mapping[design_mapping['biological_replicate'] == biological_replicate]
            
            # Collect all thick percentage data for this replicate
            replicate_data = defaultdict(list)  # condition -> list of thick percentages
            
            for _, row in replicate_files.iterrows():
                filename = row['filename']
                condition = row['condition']
                base_filename = os.path.splitext(filename)[0]
                
                # Construct CSV path
                csv_path = os.path.join(design_output_dir, condition, f"csv_files_{threshold_type}", f"{base_filename}_blue_components.csv")
                
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty and 'thick_percentage' in df.columns:
                            # Calculate image-level average thick percentage
                            image_thick_pct = df['thick_percentage'].mean()
                            replicate_data[condition].append(image_thick_pct)
                    except Exception as e:
                        print(f"      Warning: Error reading {csv_path}: {e}")
            
            # Calculate control average for this replicate
            if control_condition in replicate_data and replicate_data[control_condition]:
                control_avg = np.mean(replicate_data[control_condition])
                print(f"      Control average thick %: {control_avg:.1f}")
                
                # Normalize all conditions by the control average
                for condition, thick_percentages in replicate_data.items():
                    if thick_percentages:  # Only process if we have data
                        normalized_values = [tp / control_avg for tp in thick_percentages]
                        all_image_data[condition].extend(normalized_values)
                        print(f"        {condition}: {len(thick_percentages)} images normalized")
            else:
                print(f"      Warning: No control data found for replicate {biological_replicate}")
        
        # Plot image-level data
        if all_image_data:
            conditions = sorted(all_image_data.keys())
            x_positions = np.arange(len(conditions))
            bar_width = 0.6
            
            for j, condition in enumerate(conditions):
                if condition in all_image_data and all_image_data[condition]:
                    values = all_image_data[condition]
                    color = get_color_for_condition(design_name, condition)
                    
                    # Plot bar with error bar
                    bar = ax1.bar(x_positions[j], np.mean(values), bar_width, 
                                 color=color, alpha=0.4, 
                                 label=condition, yerr=np.std(values) if len(values) > 1 else 0)
                    
                    # Add individual data points as dots
                    for value in values:
                        ax1.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                                   color=color, alpha=0.9, s=25, zorder=5, 
                                   edgecolors='black', linewidth=0.5)
            
            # Add reference line at 1.0
            ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add significance markers
            y_max = max([max(values) for values in all_image_data.values()]) * 1.1
            add_significance_markers(ax1, all_image_data, x_positions, y_max)
            
            ax1.set_xlabel('Biological Condition')
            ax1.set_ylabel('Normalized Thick Percentage\n(relative to control)')
            ax1.set_title(f'Image-Level Analysis\n({threshold_type.title()} Thresholding)')
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels(conditions, rotation=45, ha='right')
            ax1.set_ylim(bottom=0, top=y_max)
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # === REPLICATE-LEVEL ANALYSIS (Right subplot) ===
        ax2 = axes[1]
        
        # Collect replicate-level data
        all_replicate_data = defaultdict(list)  # condition -> list of normalized replicate averages
        
        # Process each biological replicate
        for biological_replicate in biological_replicates:
            # Get files for this replicate
            replicate_files = design_mapping[design_mapping['biological_replicate'] == biological_replicate]
            
            # Collect all thick percentage data for this replicate
            replicate_data = defaultdict(list)  # condition -> list of thick percentages
            
            for _, row in replicate_files.iterrows():
                filename = row['filename']
                condition = row['condition']
                base_filename = os.path.splitext(filename)[0]
                
                # Construct CSV path
                csv_path = os.path.join(design_output_dir, condition, f"csv_files_{threshold_type}", f"{base_filename}_blue_components.csv")
                
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty and 'thick_percentage' in df.columns:
                            # Calculate image-level average thick percentage
                            image_thick_pct = df['thick_percentage'].mean()
                            replicate_data[condition].append(image_thick_pct)
                    except Exception as e:
                        continue
            
            # Calculate condition averages for this replicate
            replicate_averages = {}
            for condition, thick_percentages in replicate_data.items():
                if thick_percentages:
                    replicate_averages[condition] = np.mean(thick_percentages)
            
            # Normalize by control within this replicate
            if control_condition in replicate_averages:
                control_avg = replicate_averages[control_condition]
                for condition, avg_value in replicate_averages.items():
                    normalized_value = avg_value / control_avg
                    all_replicate_data[condition].append(normalized_value)
        
        # Plot replicate-level data
        if all_replicate_data:
            conditions = sorted(all_replicate_data.keys())
            x_positions = np.arange(len(conditions))
            bar_width = 0.6
            
            for j, condition in enumerate(conditions):
                if condition in all_replicate_data and all_replicate_data[condition]:
                    values = all_replicate_data[condition]
                    color = get_color_for_condition(design_name, condition)
                    
                    # Plot bar with error bar
                    bar = ax2.bar(x_positions[j], np.mean(values), bar_width, 
                                 color=color, alpha=0.4, 
                                 label=condition, yerr=np.std(values) if len(values) > 1 else 0)
                    
                    # Add individual replicate means as dots
                    for value in values:
                        ax2.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                                   color=color, alpha=0.9, s=25, zorder=5, 
                                   edgecolors='black', linewidth=0.5)
            
            # Add reference line at 1.0
            ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add significance markers
            y_max = max([max(values) for values in all_replicate_data.values()]) * 1.1
            add_significance_markers(ax2, all_replicate_data, x_positions, y_max)
            
            ax2.set_xlabel('Biological Condition')
            ax2.set_ylabel('Normalized Thick Percentage\n(relative to control)')
            ax2.set_title(f'Replicate-Level Analysis\n({threshold_type.title()} Thresholding)')
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels(conditions, rotation=45, ha='right')
            ax2.set_ylim(bottom=0, top=y_max)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'{design_name}_thick_thin_ratio_analysis_{threshold_type}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Thick-thin ratio analysis plot saved: {plot_path}")
        
        # Save data as CSV
        csv_filename = f'{design_name}_thick_thin_ratio_data_{threshold_type}.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Create CSV data
        csv_data = []
        # Image-level data
        for condition, values in all_image_data.items():
            for value in values:
                csv_data.append({
                    'analysis_type': 'image_level',
                    'threshold_type': threshold_type,
                    'condition': condition,
                    'normalized_thick_percentage': value
                })
        
        # Replicate-level data
        for condition, values in all_replicate_data.items():
            for value in values:
                csv_data.append({
                    'analysis_type': 'replicate_level',
                    'threshold_type': threshold_type,
                    'condition': condition,
                    'normalized_thick_percentage': value
                })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(csv_path, index=False)
        print(f"    Thick-thin ratio data saved: {csv_path}")
        print(f"    Total data points: {len(csv_data)}")

def perform_statistical_analysis(data_dict, output_dir, design_name):
    """Perform statistical analysis and create summary."""
    print(f"\nPerforming statistical analysis for {design_name}...")
    
    results = []
    
    # Calculate basic statistics for each condition
    for condition, thickness_values in data_dict.items():
        if thickness_values:
            thickness_values = safe_convert_to_numeric(thickness_values)
            thickness_values = thickness_values[thickness_values > 0]
            
            if len(thickness_values) > 0:
                stats = {
                    'Condition': condition,
                    'N': len(thickness_values),
                    'Mean': np.mean(thickness_values),
                    'Median': np.median(thickness_values),
                    'Std': np.std(thickness_values),
                    'Min': np.min(thickness_values),
                    'Max': np.max(thickness_values)
                }
                results.append(stats)
    
    # Create summary DataFrame
    if results:
        summary_df = pd.DataFrame(results)
        stats_filename = f'{design_name}_statistics.csv'
        stats_path = os.path.join(output_dir, stats_filename)
        summary_df.to_csv(stats_path, index=False)
        print(f"  Statistical summary saved: {stats_path}")
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))
    else:
        print("  No data available for statistical analysis")

def find_branch_points(skeleton):
    """Find branch points in the skeleton."""
    # Create a kernel to detect branch points
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Convolve the skeleton with the kernel
    convolved = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    
    # Branch points are where the convolution result is >= 13
    # (center pixel + at least 3 neighbors)
    branch_points = (convolved >= 13) & (skeleton > 0)
    
    return branch_points

def spider_analysis(skeleton, branch_points, thickness):
    """Perform spider analysis from branch points (matching original approach)."""
    from collections import deque
    
    height, width = skeleton.shape
    pink_mask = np.zeros_like(skeleton, dtype=bool)
    
    # Get coordinates of branch points in the remaining skeleton only
    branch_coords = np.where(branch_points & skeleton)
    branch_coords = list(zip(branch_coords[0], branch_coords[1]))
    
    print(f"  Analyzing {len(branch_coords)} branch points with spiders...")
    
    # Helper: perform BFS along the skeleton up to `window_len` steps
    def spider_pixels(start_pix):
        """Return all skeleton coords reachable within `window_len` steps."""
        window_len = FILTER_PARAMS["SPIDER_WINDOW_LENGTH"]  # Use 20 like original
        reached, frontier = {start_pix}, {start_pix}
        for _ in range(window_len):
            new_frontier = set()
            for y, x in frontier:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < height and 0 <= nx < width and 
                            skeleton[ny, nx] and (ny, nx) not in reached):
                            reached.add((ny, nx))
                            new_frontier.add((ny, nx))
            if not new_frontier:
                break
            frontier = new_frontier
        return reached
    
    # Iterate over branch pixels in the remaining skeleton only
    for i, (by, bx) in enumerate(branch_coords):
        if i % 100 == 0:
            progress = i / len(branch_coords) * 100
            print(f"  Progress: {i}/{len(branch_coords)} branch points processed ({progress:.1f}%)")
        
        spider = spider_pixels((by, bx))
        spider_list = list(spider)
        
        # --- metrics inside this spider ---
        branch_cnt = 0
        thick_vals = []
        for sy, sx in spider_list:
            # Count branch points (pixels with >= 3 neighbors)
            neighbor_count = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == dx == 0:
                        continue
                    ny, nx = sy + dy, sx + dx
                    if (0 <= ny < height and 0 <= nx < width and skeleton[ny, nx]):
                        neighbor_count += 1
            if neighbor_count >= 3:
                branch_cnt += 1
            thick_vals.append(thickness[sy, sx])
        
        density = branch_cnt / len(spider_list) if spider_list else 0.0
        avg_thick = np.mean(thick_vals) if thick_vals else 0.0
        
        # ---- pink criterion (same as original) ----
        if density > FILTER_PARAMS["BRANCH_DENSITY_THRESHOLD"] and avg_thick >= FILTER_PARAMS["MIN_AVG_THICKNESS_PINK"]:
            for sy, sx in spider_list:
                pink_mask[sy, sx] = True
    
    return pink_mask

def calculate_thickness(skeleton, distance_map):
    """Calculate thickness at each skeleton pixel using distance transform."""
    thickness = np.zeros_like(skeleton, dtype=float)
    
    # The distance_map gives the distance from each pixel to the nearest edge
    # For skeleton pixels, this distance represents the radius
    # Use radius values directly (not diameter) to match original script
    skeleton_coords = np.where(skeleton > 0)
    thickness[skeleton_coords] = distance_map[skeleton_coords]
    
    return thickness

def calculate_mother_component_lengths(components, region_mask):
    """
    Calculate mother component lengths for each component.
    Mother component = the larger connected component that contains this smaller component.
    """
    # Label the entire region to find connected components
    region_labeled = label(region_mask)
    region_components = regionprops(region_labeled)
    
    # Sort components by size (largest first) to identify mother components
    component_sizes = [(i, len(comp.coords)) for i, comp in enumerate(components)]
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # For each component, find which larger component it belongs to
    mother_lengths = []
    
    for i, component in enumerate(components):
        coords = component.coords
        component_size = len(coords)
        
        # Find the largest component that contains this component
        mother_length = component_size  # Default to own size
        
        # Check if this component is contained within any larger component
        for larger_idx, larger_size in component_sizes:
            if larger_idx == i:  # Skip self
                continue
            if larger_size <= component_size:  # Skip smaller or equal components
                continue
                
            larger_component = components[larger_idx]
            larger_coords = set((y, x) for y, x in larger_component.coords)
            
            # Check if this component is mostly contained within the larger component
            overlap_count = 0
            for y, x in coords:
                if (y, x) in larger_coords:
                    overlap_count += 1
            
            # If more than 50% overlap, consider it a mother component
            if overlap_count > len(coords) * 0.5:
                mother_length = larger_size
                break
        
        mother_lengths.append(mother_length)
    
    return mother_lengths

def calculate_batch_threshold(batch_images, batch_name):
    """Calculate a single threshold for an entire biological replicate using mean intensity above 0.012 from all images."""
    print(f"\nCalculating single threshold for batch {batch_name}...")
    
    # Collect mean intensities above 0.012 from all images in this batch
    batch_mean_intensities = []
    image_names = []
    
    for image_path in batch_images:
        try:
            # Load image with retry logic for timeout errors and corrupted files
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Add timeout to prevent hanging on corrupted files
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Image loading timed out - file may be corrupted")
                    
                    # Set timeout for image loading (10 seconds)
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(10)
                    
                    try:
                        img = Image.open(image_path)
                        img_array = np.array(img)
                        signal.alarm(0)  # Cancel timeout
                        break
                    except (TimeoutError, OSError, ValueError, MemoryError) as e:
                        signal.alarm(0)  # Cancel timeout
                        if attempt < max_retries - 1:
                            print(f"  Attempt {attempt + 1} failed for {os.path.basename(image_path)}: {e}")
                            print(f"  Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            # If all retries failed, try alternative loading method
                            try:
                                print(f"  Trying alternative loading method for {os.path.basename(image_path)}...")
                                # Try with different PIL settings for corrupted files
                                from PIL import ImageFile
                                ImageFile.LOAD_TRUNCATED_IMAGES = True
                                with Image.open(image_path) as img:
                                    img.load()  # Force load
                                    img_array = np.array(img)
                                break
                            except Exception as e2:
                                raise ValueError(f"Failed to load corrupted image {os.path.basename(image_path)}: {e2}")
                                
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  Attempt {attempt + 1} failed for {os.path.basename(image_path)}: {e}")
                        print(f"  Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise e
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Normalize
            gray = gray.astype(np.float32) / 255.0
            
            # Calculate mean of pixels above 0.012 threshold (same as in analyze_image)
            vals = gray.ravel()
            thr_012 = 0.012
            above_012 = vals[vals > thr_012]
            mean_above_012 = above_012.mean() if above_012.size > 0 else 0.0
            
            batch_mean_intensities.append(mean_above_012)
            image_names.append(image_path)  # Store full path instead of just filename
            
            print(f"  {os.path.basename(image_path)}: mean_above_0.012 = {mean_above_012:.4f}")
            
        except Exception as e:
            print(f"  Warning: Error processing {image_path}: {e}")
            continue
    
    if not batch_mean_intensities:
        print(f"  Error: No valid images found for batch {batch_name}")
        return None, None, None
    
    # Calculate the overall mean intensity above 0.012 for this batch
    batch_overall_mean = np.mean(batch_mean_intensities)
    print(f"  Batch {batch_name} overall mean_above_0.012: {batch_overall_mean:.4f}")
    
    # Use the correct regression equation from the regression analysis
    # Best predictor: mean_above_0.012 with slope=1.4865603768263806, intercept=-0.003583494788075847
    slope = 1.4865603768263806
    intercept = -0.003583494788075847
    threshold = slope * batch_overall_mean + intercept
    
    print(f"  Batch {batch_name} calculated threshold: {threshold:.4f}")
    
    return threshold, batch_mean_intensities, image_names

def generate_threshold_comparison_plots(batch_thresholds, output_dir, design_name):
    """Generate new bar graph visualizations for threshold comparisons as requested."""
    print(f"\nGenerating new threshold comparison plots for {design_name}...")
    
    # Use centralized color scheme
    # condition_colors will be determined by get_color_for_condition() function
    
    # Collect all data for deviation analysis
    all_deviations = defaultdict(list)  # condition -> list of deviations
    all_replicate_deviations = defaultdict(lambda: defaultdict(list))  # condition -> replicate -> list of deviations
    
    # Create plots for each batch
    for batch_name, batch_data in batch_thresholds.items():
        if batch_data is None:
            continue
            
        threshold, mean_intensities, image_names = batch_data
        
        if not mean_intensities:
            continue
            
        # Calculate individual thresholds for each image and extract condition info
        individual_thresholds = []
        conditions = []
        valid_indices = []
        biological_replicates = []
        
        for i, (mean_intensity, image_name) in enumerate(zip(mean_intensities, image_names)):
            # Use the correct regression equation from the regression analysis
            slope = 1.4865603768263806
            intercept = -0.003583494788075847
            individual_threshold = slope * mean_intensity + intercept
            individual_thresholds.append(individual_threshold)
            
            # Extract condition and biological replicate from the comprehensive mapping
            condition = None
            biological_replicate = None
            try:
                # Load the comprehensive mapping to get condition info
                mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                filename = os.path.basename(image_name)
                # Filter by experimental design first to avoid mixing conditions from different designs
                design_filtered = mapping_df[mapping_df['experimental_design'] == design_name]
                file_info = design_filtered[design_filtered['filename'] == filename]
                if not file_info.empty:
                    condition = file_info.iloc[0]['condition']
                    biological_replicate = file_info.iloc[0]['biological_replicate']
            except Exception as e:
                print(f"    Warning: Could not get condition info for {image_name}: {e}")
                continue
            
            # Skip excluded images
            skip_image = False
            for excluded_image in excluded_images:
                if excluded_image in image_name:
                    skip_image = True
                    break
            
            if not skip_image and condition and biological_replicate:
                conditions.append(condition)
                biological_replicates.append(biological_replicate)
                valid_indices.append(i)
        
        # Create bar graph for this batch
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Organize data by condition
        condition_data = defaultdict(list)
        for i, condition in enumerate(conditions):
            condition_data[condition].append(individual_thresholds[valid_indices[i]])
        
        # Create bar positions using actual conditions from data
        conditions_list = sorted(list(condition_data.keys()))
        x_positions = np.arange(len(conditions_list))
        bar_width = 0.6
        
        # Create bars
        bars = []
        for i, condition in enumerate(conditions_list):
            if condition in condition_data:
                values = condition_data[condition]
                color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
                bar = ax.bar(x_positions[i], np.mean(values), bar_width, 
                           color=color, alpha=0.4, 
                           label=condition, yerr=np.std(values) if len(values) > 1 else 0)
                bars.append(bar)
                
                # Add individual data points as small dots
                for j, value in enumerate(values):
                    ax.scatter(x_positions[i] + np.random.normal(0, 0.02), value, 
                             color=color, alpha=0.9, s=25, zorder=5, edgecolors='black', linewidth=0.5)
                
                # Calculate deviations for this condition
                deviations = [value - threshold for value in values]
                all_deviations[condition].extend(deviations)
                
                # Use the biological replicate information for each individual image
                for j, value in enumerate(values):
                    if j < len(biological_replicates):
                        biological_replicate = biological_replicates[j]
                        deviation = value - threshold
                        all_replicate_deviations[condition][biological_replicate].append(deviation)
        
        # Plot the batch threshold as a horizontal line
        ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2, 
                  label=f'Batch Threshold: {threshold:.4f}')
        
        # Customize the plot
        ax.set_xlabel('Biological Condition')
        ax.set_ylabel('Threshold Value')
        ax.set_title(f'Per-Image Thresholds by Biological Condition - {batch_name} {design_name}')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions_list, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'{design_name}_{batch_name}_threshold_comparison.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Threshold comparison plot saved: {plot_path}")
        
        # Save the data as CSV
        csv_filename = f'{design_name}_{batch_name}_threshold_data.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Create condition list for all images (including excluded ones)
        all_conditions = []
        for image_name in image_names:
            condition = None
            
            # Use get_experimental_design_info to get the correct condition mapping
            design_info = get_experimental_design_info(image_name)
            if design_info:
                # Extract condition from the design info dictionary
                if 'biological_condition' in design_info and 'genotype' in design_info:
                    if design_name == "genotypes_statins":
                        # For genotypes_statins, combine genotype and treatment
                        condition = f"{design_info['genotype']}_{design_info['biological_condition']}"
                    else:
                        # For other designs, use the condition directly
                        condition = design_info['biological_condition']
                elif 'biological_condition' in design_info:
                    condition = design_info['biological_condition']
                elif 'genotype' in design_info:
                    condition = design_info['genotype']
            
            all_conditions.append(condition)
        
        # Save CSV with all data
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image Name', 'Condition', 'Biological Replicate', 'Mean Intensity Above 0.012', 'Individual Threshold', 'Batch Threshold', 'Deviation'])
            
            for i, (image_name, mean_intensity) in enumerate(zip(image_names, mean_intensities)):
                condition = all_conditions[i] if i < len(all_conditions) else 'Unknown'
                biological_replicate = biological_replicates[i] if i < len(biological_replicates) else 'Unknown'
                individual_threshold = slope * mean_intensity + intercept
                deviation = individual_threshold - threshold
                writer.writerow([image_name, condition, biological_replicate, f"{mean_intensity:.6f}", f"{individual_threshold:.6f}", f"{threshold:.6f}", f"{deviation:.6f}"])
        
        print(f"  Threshold data saved: {csv_path}")
    
    # Create deviation plots across all batches
    if all_deviations:
        # Plot 1: All individual deviations combined
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        conditions_list = sorted(list(all_deviations.keys()))
        x_positions = np.arange(len(conditions_list))
        bar_width = 0.6
        
        for i, condition in enumerate(conditions_list):
            if condition in all_deviations:
                values = all_deviations[condition]
                color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
                bar = ax.bar(x_positions[i], np.mean(values), bar_width, 
                           color=color, alpha=0.4, 
                           label=condition, yerr=np.std(values) if len(values) > 1 else 0)
                
                # Add individual data points as small dots
                for j, value in enumerate(values):
                    ax.scatter(x_positions[i] + np.random.normal(0, 0.02), value, 
                             color=color, alpha=0.9, s=25, zorder=5, edgecolors='black', linewidth=0.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Biological Condition')
        ax.set_ylabel('Deviation from Batch Threshold')
        ax.set_title(f'Threshold Deviations by Biological Condition - {design_name} (All Replicates)')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions_list, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'{design_name}_all_replicates_deviation_comparison.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  All replicates deviation plot saved: {plot_path}")
        
        # Plot 2: Average deviations by replicate
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for i, condition in enumerate(conditions_list):
            if condition in all_replicate_deviations:
                replicate_means = []
                for biological_replicate in all_replicate_deviations[condition]:
                    replicate_means.append(np.mean(all_replicate_deviations[condition][biological_replicate]))
                
                if replicate_means:
                    color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
                    bar = ax.bar(x_positions[i], np.mean(replicate_means), bar_width, 
                               color=color, alpha=0.4, 
                               label=condition, yerr=np.std(replicate_means) if len(replicate_means) > 1 else 0)
                    
                    # Add individual replicate means as small dots
                    for j, value in enumerate(replicate_means):
                        ax.scatter(x_positions[i] + np.random.normal(0, 0.02), value, 
                                 color=color, alpha=0.9, s=25, zorder=5, edgecolors='black', linewidth=0.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Biological Condition')
        ax.set_ylabel('Average Deviation from Batch Threshold')
        ax.set_title(f'Average Threshold Deviations by Biological Condition - {design_name} (Replicate Level)')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions_list, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'{design_name}_replicate_level_deviation_comparison.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Replicate level deviation plot saved: {plot_path}")
        
        # Save deviation data as CSV
        csv_filename = f'{design_name}_deviation_data.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['batch', 'biological_condition', 'deviation'])
            
            for condition in conditions_list:
                if condition in all_replicate_deviations:
                    for biological_replicate in all_replicate_deviations[condition]:
                        deviations = all_replicate_deviations[condition][biological_replicate]
                        
                        for deviation in deviations:
                            writer.writerow([biological_replicate, condition, f"{deviation:.6f}"])
        
        print(f"  Deviation data saved: {csv_path}")

def generate_combined_deviation_plots(batch_thresholds, output_dir, design_name):
    """Generate combined deviation plots showing replicate-level vs individual biological replicate deviations."""
    print(f"\nGenerating combined deviation plots for {design_name}...")
    
    # Use centralized color scheme
    # condition_colors will be determined by get_color_for_condition() function
    
    # Collect all deviation data organized by biological replicate
    replicate_data = defaultdict(lambda: defaultdict(list))  # biological_replicate -> genotype -> deviations
    all_deviations = []
    
    # Process each batch to get deviation data
    for batch_name, batch_data in batch_thresholds.items():
        if batch_data is None:
            continue
            
        threshold, mean_intensities, image_names = batch_data
        
        if not mean_intensities:
            continue
            
        # Calculate individual thresholds and deviations for each image
        for i, (mean_intensity, image_name) in enumerate(zip(mean_intensities, image_names)):
            # Use the correct regression equation
            slope = 1.4865603768263806
            intercept = -0.003583494788075847
            individual_threshold = slope * mean_intensity + intercept
            
            # Extract condition and biological replicate from the comprehensive mapping
            condition = None
            biological_replicate = None
            try:
                # Load the comprehensive mapping to get condition info
                mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                filename = os.path.basename(image_name)
                # Filter by experimental design first to avoid mixing conditions from different designs
                design_filtered = mapping_df[mapping_df['experimental_design'] == design_name]
                file_info = design_filtered[design_filtered['filename'] == filename]
                if not file_info.empty:
                    condition = file_info.iloc[0]['condition']
                    biological_replicate = file_info.iloc[0]['biological_replicate']
            except Exception as e:
                print(f"    Warning: Could not get condition info for {image_name}: {e}")
                continue
            
            # Skip excluded images
            skip_image = False
            for excluded_image in excluded_images:
                if excluded_image in image_name:
                    skip_image = True
                    break
            
            if not skip_image and condition and biological_replicate:
                # Use the condition name directly
                deviation = individual_threshold - threshold
                replicate_data[biological_replicate][condition].append(deviation)
                all_deviations.append(deviation)
    
    if not replicate_data:
        print(f"  No valid data found for {design_name}")
        return
    
    # Get all unique biological replicates and genotypes
    # Filter biological replicates to only include those that belong to this experimental design
    try:
        mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
        design_replicates = mapping_df[mapping_df['experimental_design'] == design_name]['biological_replicate'].unique()
        biological_replicates = sorted([rep for rep in replicate_data.keys() if rep in design_replicates])
    except Exception as e:
        print(f"    Warning: Could not filter biological replicates for {design_name}: {e}")
        biological_replicates = sorted(list(replicate_data.keys()))
    
    # Get all unique conditions from the data
    all_conditions = set()
    for replicate_data_dict in replicate_data.values():
        all_conditions.update(replicate_data_dict.keys())
    conditions = sorted(list(all_conditions))
    
    # Create figure with 1 row and num_replicates + 1 columns (individual replicates + replicate level)
    num_replicates = len(biological_replicates)
    fig, axes = plt.subplots(1, num_replicates + 1, figsize=(6 * (num_replicates + 1), 8))
    
    # Find global y-axis limits
    y_min = min(all_deviations) - 0.01
    y_max = max(all_deviations) + 0.01
    
    # Plot individual biological replicate deviations
    for i, biological_replicate in enumerate(biological_replicates):
        ax = axes[i]
        
        x_positions = np.arange(len(conditions))
        bar_width = 0.6
        
        for j, condition in enumerate(conditions):
            if condition in replicate_data[biological_replicate]:
                values = replicate_data[biological_replicate][condition]
                if values:  # Only plot if there are values
                    color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
                    bar = ax.bar(x_positions[j], np.mean(values), bar_width, 
                               color=color, alpha=0.4, 
                               label=condition, yerr=np.std(values) if len(values) > 1 else 0)
                    
                    # Add individual data points as small dots
                    for k, value in enumerate(values):
                        ax.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                                 color=color, alpha=0.9, s=25, zorder=5, 
                                 edgecolors='black', linewidth=0.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Biological Condition')
        ax.set_ylabel('Deviation from Batch Threshold')
        ax.set_title(f'{biological_replicate} - Individual Deviations')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        
        # Only show legend on first plot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot replicate-level deviations (last subplot) - same as the replicate_level_deviation_comparison.png
    ax = axes[-1]
    
    for j, condition in enumerate(conditions):
        replicate_means = []
        for biological_replicate in biological_replicates:
            if condition in replicate_data[biological_replicate] and replicate_data[biological_replicate][condition]:
                replicate_means.append(np.mean(replicate_data[biological_replicate][condition]))
        
        if replicate_means:
            color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
            bar = ax.bar(x_positions[j], np.mean(replicate_means), bar_width, 
                       color=color, alpha=0.4, 
                       label=condition, yerr=np.std(replicate_means) if len(replicate_means) > 1 else 0)
            
            # Add individual replicate means as small dots
            for k, value in enumerate(replicate_means):
                ax.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                         color=color, alpha=0.9, s=25, zorder=5, 
                         edgecolors='black', linewidth=0.5)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Biological Condition')
    ax.set_ylabel('Average Deviation from Batch Threshold')
    ax.set_title('Replicate-Level Deviations')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'{design_name}_combined_deviation_comparison.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Combined deviation plot saved: {plot_path}")


def generate_combined_threshold_plots(batch_thresholds, output_dir, design_name):
    """Generate combined threshold plots showing actual threshold values (not deviations) used in each biological replicate."""
    print(f"\nGenerating combined threshold plots for {design_name}...")
    
    # Collect all threshold data organized by biological replicate
    replicate_data = defaultdict(lambda: defaultdict(list))  # biological_replicate -> condition -> thresholds
    batch_threshold_values = {}  # batch_name -> batch_threshold
    
    # Process each batch to get threshold data
    for batch_name, batch_data in batch_thresholds.items():
        if batch_data is None:
            continue
            
        threshold, mean_intensities, image_names = batch_data
        
        if not mean_intensities:
            continue
            
        # Store the batch threshold for this batch
        batch_threshold_values[batch_name] = threshold
        
        # Calculate individual thresholds for each image
        for i, (mean_intensity, image_name) in enumerate(zip(mean_intensities, image_names)):
            # Use the correct regression equation
            slope = 1.4865603768263806
            intercept = -0.003583494788075847
            individual_threshold = slope * mean_intensity + intercept
            
            # Extract condition and biological replicate from the comprehensive mapping
            condition = None
            biological_replicate = None
            try:
                # Load the comprehensive mapping to get condition info
                mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                filename = os.path.basename(image_name)
                # Filter by experimental design first to avoid mixing conditions from different designs
                design_filtered = mapping_df[mapping_df['experimental_design'] == design_name]
                file_info = design_filtered[design_filtered['filename'] == filename]
                if not file_info.empty:
                    condition = file_info.iloc[0]['condition']
                    biological_replicate = file_info.iloc[0]['biological_replicate']
            except Exception as e:
                print(f"    Warning: Could not get condition info for {image_name}: {e}")
                continue
            
            # Skip excluded images
            skip_image = False
            for excluded_image in excluded_images:
                if excluded_image in image_name:
                    skip_image = True
                    break
            
            if not skip_image and condition and biological_replicate:
                # Store the individual threshold (not the deviation)
                replicate_data[biological_replicate][condition].append(individual_threshold)
    
    if not replicate_data:
        print(f"  No valid data found for {design_name}")
        return
    
    # Get all unique biological replicates and conditions
    # Filter biological replicates to only include those that belong to this experimental design
    try:
        mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
        design_replicates = mapping_df[mapping_df['experimental_design'] == design_name]['biological_replicate'].unique()
        biological_replicates = sorted([rep for rep in replicate_data.keys() if rep in design_replicates])
    except Exception as e:
        print(f"    Warning: Could not filter biological replicates for {design_name}: {e}")
        biological_replicates = sorted(list(replicate_data.keys()))
    
    # Get all unique conditions from the data
    all_conditions = set()
    for replicate_data_dict in replicate_data.values():
        all_conditions.update(replicate_data_dict.keys())
    conditions = sorted(list(all_conditions))
    
    # Create figure with 1 row and num_replicates + 1 columns (individual replicates + replicate level)
    num_replicates = len(biological_replicates)
    fig, axes = plt.subplots(1, num_replicates + 1, figsize=(6 * (num_replicates + 1), 8))
    
    # Handle case where there's only one replicate (axes won't be an array)
    if num_replicates == 0:
        print(f"  No biological replicates found for {design_name}")
        return
    elif num_replicates == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    
    # Find global y-axis limits for consistency
    all_thresholds = []
    for replicate_data_dict in replicate_data.values():
        for thresholds in replicate_data_dict.values():
            all_thresholds.extend(thresholds)
    
    # Add batch thresholds to the range for context
    all_thresholds.extend(batch_threshold_values.values())
    
    if all_thresholds:
        y_min = min(all_thresholds) - 0.05
        y_max = max(all_thresholds) + 0.05
    else:
        y_min, y_max = 0, 1
    
    # Plot individual biological replicate thresholds
    for i, biological_replicate in enumerate(biological_replicates):
        ax = axes[i]
        
        x_positions = np.arange(len(conditions))
        bar_width = 0.6
        
        for j, condition in enumerate(conditions):
            if condition in replicate_data[biological_replicate]:
                values = replicate_data[biological_replicate][condition]
                if values:  # Only plot if there are values
                    color = get_color_for_condition(design_name, condition)
                    bar = ax.bar(x_positions[j], np.mean(values), bar_width, 
                               color=color, alpha=0.4, 
                               label=condition, yerr=np.std(values) if len(values) > 1 else 0)
                    
                    # Add individual data points as small dots
                    for k, value in enumerate(values):
                        ax.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                                 color=color, alpha=0.9, s=25, zorder=5, 
                                 edgecolors='black', linewidth=0.5)
        
        # Add horizontal line for batch threshold for this replicate (if available)
        # Find the batch threshold for this biological replicate
        replicate_batch_threshold = None
        for batch_name, threshold_value in batch_threshold_values.items():
            if biological_replicate in batch_name or batch_name in biological_replicate:
                replicate_batch_threshold = threshold_value
                break
        
        if replicate_batch_threshold is not None:
            ax.axhline(y=replicate_batch_threshold, color='black', 
                      linestyle='-', linewidth=2, alpha=0.7, 
                      label=f'Batch Threshold ({replicate_batch_threshold:.3f})')
        
        ax.set_xlabel('Biological Condition')
        ax.set_ylabel('Threshold Value')
        ax.set_title(f'{biological_replicate} - Individual Thresholds')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        
        # Only show legend on first plot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot replicate-level thresholds (last subplot)
    ax = axes[-1]
    
    x_positions = np.arange(len(conditions))
    bar_width = 0.6
    
    for j, condition in enumerate(conditions):
        replicate_means = []
        for biological_replicate in biological_replicates:
            if condition in replicate_data[biological_replicate] and replicate_data[biological_replicate][condition]:
                replicate_means.append(np.mean(replicate_data[biological_replicate][condition]))
        
        if replicate_means:
            color = get_color_for_condition(design_name, condition)
            bar = ax.bar(x_positions[j], np.mean(replicate_means), bar_width, 
                       color=color, alpha=0.4, 
                       label=condition, yerr=np.std(replicate_means) if len(replicate_means) > 1 else 0)
            
            # Add individual replicate means as small dots
            for k, value in enumerate(replicate_means):
                ax.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                         color=color, alpha=0.9, s=25, zorder=5, 
                         edgecolors='black', linewidth=0.5)
    

    
    ax.set_xlabel('Biological Condition')
    ax.set_ylabel('Average Threshold Value')
    ax.set_title('Replicate-Level Thresholds')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'{design_name}_combined_threshold_comparison.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Combined threshold plot saved: {plot_path}")


def generate_group_specific_deviation_plots(batch_thresholds, output_dir, design_name):
    """Generate individual deviation plots for each condition in the experimental design."""
    print(f"\nGenerating group-specific deviation plots for {design_name}...")
    
    # Define condition colors for all experimental designs
    condition_colors = {
        # genotype_only conditions
        'WT_Control': '#66B3FF',      # Light blue
        'ApoE4_Control': '#FFB366',   # Light orange
        'ApoE2_Control': '#90EE90',   # Light green
        'KO_Control': '#FF6B6B',      # Light red
        # genotype_statin conditions
        'WT_Statin': '#0066CC',       # Darker blue
        'ApoE4_Statin': '#FF8000',    # Darker orange
        'ApoE2_Statin': '#228B22',    # Darker green
        'KO_Statin': '#CC0000',       # Darker red
        # domain_analysis conditions
        'Control': '#87CEEB',         # Sky blue
        'ApoE2-NTD': '#98FB98',       # Pale green
        'ApoE4-NTD': '#FFA07A',       # Light salmon
        'ApoE-CTD': '#DDA0DD',        # Plum
        'ApoE2': '#32CD32',           # Lime green
        'ApoE4': '#FF6347',           # Tomato red
        # genotypes conditions
        'KO': 'red',
        'ApoE4': 'blue', 
        'ApoE2': 'green',
        'Control': 'orange',
        # genotypes_statins conditions
        'KO_Control': 'red',
        'KO_Simvastatin': 'darkred',
        'ApoE4_Control': 'blue',
        'ApoE4_Simvastatin': 'darkblue',
        'ApoE2_Control': 'green',
        'ApoE2_Simvastatin': 'darkgreen',
        'Control_Control': 'orange',
        'Control_Simvastatin': 'darkorange'
    }
    
    
    # Collect all deviation data
    all_batch_deviations = {}
    all_replicate_deviations = defaultdict(lambda: defaultdict(list))
    
    # Process each batch to get deviation data
    for batch_name, batch_data in batch_thresholds.items():
        if batch_data is None:
            continue
            
        threshold, mean_intensities, image_names = batch_data
        
        if not mean_intensities:
            continue
            
        # Calculate individual thresholds and deviations for each image
        batch_deviations = defaultdict(list)
        
        for i, (mean_intensity, image_name) in enumerate(zip(mean_intensities, image_names)):
            # Use the correct regression equation
            slope = 1.4865603768263806
            intercept = -0.003583494788075847
            individual_threshold = slope * mean_intensity + intercept
            
            # Extract condition and biological replicate from the comprehensive mapping
            condition = None
            biological_replicate = None
            try:
                # Load the comprehensive mapping to get condition info
                mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                filename = os.path.basename(image_name)
                # Filter by experimental design first to avoid mixing conditions from different designs
                design_filtered = mapping_df[mapping_df['experimental_design'] == design_name]
                file_info = design_filtered[design_filtered['filename'] == filename]
                if not file_info.empty:
                    condition = file_info.iloc[0]['condition']
                    biological_replicate = file_info.iloc[0]['biological_replicate']
            except Exception as e:
                print(f"    Warning: Could not get condition info for {image_name}: {e}")
                continue
            
            # Skip excluded images
            skip_image = False
            for excluded_image in excluded_images:
                if excluded_image in image_name:
                    skip_image = True
                    break
            
            if not skip_image and condition and biological_replicate:
                deviation = individual_threshold - threshold
                batch_deviations[condition].append(deviation)
                all_replicate_deviations[condition][biological_replicate].append(deviation)
        
        all_batch_deviations[batch_name] = batch_deviations
    
    if not all_batch_deviations:
        print(f"  No valid data found for {design_name}")
        return
    
    # Get all unique conditions from the data
    all_conditions = set()
    for batch_deviations in all_batch_deviations.values():
        all_conditions.update(batch_deviations.keys())
    conditions = sorted(list(all_conditions))
    
    # Create individual plots for each condition
    for condition in conditions:
        # Check if this condition has data
        condition_has_data = False
        for batch_deviations in all_batch_deviations.values():
            if condition in batch_deviations and batch_deviations[condition]:
                condition_has_data = True
                break
        
        if not condition_has_data:
            print(f"  No data found for condition {condition}, skipping...")
            continue
        
        print(f"  Generating plot for condition: {condition}")
        
        # Create figure with 1 row and num_batches + 1 columns (individual batches + replicate level)
        num_batches = len(all_batch_deviations)
        fig, axes = plt.subplots(1, num_batches + 1, figsize=(6 * (num_batches + 1), 8))
        
        # Find global y-axis limits for this condition
        all_deviations = []
        for batch_deviations in all_batch_deviations.values():
            if condition in batch_deviations:
                all_deviations.extend(batch_deviations[condition])
        
        # Also include replicate-level deviations for this condition
        if condition in all_replicate_deviations:
            for batch_name in batch_thresholds.keys():
                if batch_name in all_replicate_deviations[condition]:
                    replicate_means = [np.mean(all_replicate_deviations[condition][batch_name])]
                    all_deviations.extend(replicate_means)
        
        if all_deviations:
            y_min = min(all_deviations) - 0.01
            y_max = max(all_deviations) + 0.01
        else:
            y_min, y_max = -0.1, 0.1
        
        # Plot individual batch deviations for this condition
        for i, (batch_name, batch_deviations) in enumerate(all_batch_deviations.items()):
            ax = axes[i]
            
            if condition in batch_deviations and batch_deviations[condition]:
                values = batch_deviations[condition]
                color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
                bar_width = 0.6
                
                bar = ax.bar(0, np.mean(values), bar_width, 
                           color=color, alpha=0.4, 
                           label=condition, yerr=np.std(values) if len(values) > 1 else 0)
                
                # Add individual data points as small dots
                for k, value in enumerate(values):
                    ax.scatter(0 + np.random.normal(0, 0.02), value, 
                             color=color, alpha=0.9, s=25, zorder=5, 
                             edgecolors='black', linewidth=0.5)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('Batch')
            ax.set_ylabel('Deviation from Batch Threshold')
            ax.set_title(f'{batch_name} - {condition}')
            ax.set_xticks([0])
            ax.set_xticklabels([condition], rotation=45, ha='right')
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)
            
            # Only show legend on first plot
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot replicate-level deviations for this condition (last subplot)
        ax = axes[-1]
        
        if condition in all_replicate_deviations:
            replicate_means = []
            for batch_name in batch_thresholds.keys():
                if batch_name in all_replicate_deviations[condition]:
                    replicate_means.append(np.mean(all_replicate_deviations[condition][batch_name]))
            
            if replicate_means:
                color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
                bar_width = 0.6
                
                bar = ax.bar(0, np.mean(replicate_means), bar_width, 
                           color=color, alpha=0.4, 
                           label=condition, yerr=np.std(replicate_means) if len(replicate_means) > 1 else 0)
                
                # Add individual replicate means as small dots
                for k, value in enumerate(replicate_means):
                    ax.scatter(0 + np.random.normal(0, 0.02), value, 
                             color=color, alpha=0.9, s=25, zorder=5, 
                             edgecolors='black', linewidth=0.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Condition')
        ax.set_ylabel('Average Deviation from Batch Threshold')
        ax.set_title(f'All Replicates - {condition}')
        ax.set_xticks([0])
        ax.set_xticklabels([condition], rotation=45, ha='right')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the condition-specific plot
        plot_filename = f'{design_name}_{condition}_deviation_comparison.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Group-specific deviation plot saved: {plot_path}")
    
    # Save deviation data to CSV for all groups
    deviation_data = []
    for batch_name, batch_deviations in all_batch_deviations.items():
        for condition, deviations in batch_deviations.items():
            for deviation in deviations:
                deviation_data.append({
                    'batch': batch_name,
                    'biological_condition': condition,
                    'deviation': deviation
                })
    
    if deviation_data:
        csv_filename = f'{design_name}_deviation_data.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        
        df = pd.DataFrame(deviation_data)
        df.to_csv(csv_path, index=False)
        print(f"  Deviation data saved: {csv_path}")

def analyze_image_with_threshold(image_path, threshold_value=None):
    """Analyze a single image with branch-based snakes approach, optionally using a pre-calculated threshold."""
    print(f"Processing {os.path.basename(image_path)}...")
    
    # Load image with retry logic for timeout errors and corrupted files
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Add timeout for file operations
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("File operation timed out")
            
            # Set timeout for file operations (30 seconds)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            try:
                # First check if file is accessible
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"File does not exist: {image_path}")
                
                if not os.access(image_path, os.R_OK):
                    raise PermissionError(f"File not readable: {image_path}")
                
                # Try to open with PIL
                img = Image.open(image_path)
                
                # Force load the image data
                img.load()
                
                # Convert to array
                img_array = np.array(img)
                
                # Cancel timeout
                signal.alarm(0)
                break
                
            except (TimeoutError, OSError, ValueError, MemoryError) as e:
                signal.alarm(0)  # Cancel timeout
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed for {os.path.basename(image_path)}: {e}")
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise e
                    
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1} failed for {os.path.basename(image_path)}: {e}")
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise e
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Normalize
    gray = gray.astype(np.float32) / 255.0
    
    # Apply adaptive thresholding using regression-based prediction
    # Calculate mean of pixels above 0.012 threshold
    vals = gray.ravel()
    thr_012 = 0.012
    above_012 = vals[vals > thr_012]
    mean_above_012 = above_012.mean() if above_012.size > 0 else 0.0
    
    # Use provided threshold or calculate new one
    if threshold_value is not None:
        predicted_threshold = threshold_value
        print(f"  Using provided threshold: {predicted_threshold:.4f} (mean_above_0.012: {mean_above_012:.4f})")
    else:
        # Use the correct regression equation from the regression analysis
        # Best predictor: mean_above_0.012 with slope=1.4865603768263806, intercept=-0.003583494788075847
        slope = 1.4865603768263806
        intercept = -0.003583494788075847
        predicted_threshold = slope * mean_above_012 + intercept
        print(f"  Predicted threshold: {predicted_threshold:.4f} (mean_above_0.012: {mean_above_012:.4f})")
    
    return mean_above_012, predicted_threshold, gray

def process_image_core(image_path, threshold_value=None):
    """Core image processing function used by both regular and batch threshold analyses."""
    # Get threshold (either provided or calculated)
    mean_above_012, predicted_threshold, gray = analyze_image_with_threshold(image_path, threshold_value)
    
    # Apply threshold
    thresholded = gray > predicted_threshold
    
    # Morphological operations
    filtered_mask = remove_small_objects(thresholded, min_size=FILTER_PARAMS["MIN_OBJECT_SIZE"])
    filtered_mask = opening(filtered_mask, disk(FILTER_PARAMS["OPENING_DISK_SIZE"]))
    filtered_mask = closing(filtered_mask, disk(FILTER_PARAMS["CLOSING_DISK_SIZE"]))
    
    # Skeletonize
    skeleton = skeletonize(filtered_mask)
    
    # Find branch points
    branch_points = find_branch_points(skeleton)
    
    # Calculate thickness
    distance_map = distance_transform_edt(filtered_mask)
    thickness = calculate_thickness(skeleton, distance_map)
    
    # Filter skeleton components by size (same threshold as skeleton length filter)
    print(f"  Filtering skeleton components by size...")
    skeleton_labeled = label(skeleton)
    skeleton_components = regionprops(skeleton_labeled)
    
    # Create mask for valid skeleton components (≥ MIN_SKELETON_LENGTH pixels)
    valid_components_mask = np.zeros_like(skeleton, dtype=bool)
    for prop in skeleton_components:
        if len(prop.coords) >= FILTER_PARAMS["MIN_SKELETON_LENGTH"]:
            coords = prop.coords
            for y, x in coords:
                valid_components_mask[y, x] = True
    
    print(f"  Valid skeleton components (≥{FILTER_PARAMS['MIN_SKELETON_LENGTH']} pixels): {len([p for p in skeleton_components if len(p.coords) >= FILTER_PARAMS['MIN_SKELETON_LENGTH']])}")
    
    # Calculate mother component lengths from original skeleton components
    print(f"  Calculating mother component lengths from original skeleton...")
    original_skeleton_components = [p for p in skeleton_components if len(p.coords) >= FILTER_PARAMS["MIN_SKELETON_LENGTH"]]
    original_mother_lengths = calculate_mother_component_lengths(original_skeleton_components, skeleton)
    
    # Create mapping from skeleton coordinates to mother component length
    skeleton_to_mother_length = np.zeros_like(skeleton, dtype=float)
    for i, prop in enumerate(original_skeleton_components):
        coords = prop.coords
        mother_length = original_mother_lengths[i]
        for y, x in coords:
            skeleton_to_mother_length[y, x] = mother_length
    
    # Apply thickness filter to skeleton
    thickness_mask = (thickness >= FILTER_PARAMS["MIN_THICKNESS"]) & (thickness <= FILTER_PARAMS["MAX_THICKNESS"])
    print(f"  Pixels passing thickness filter: {np.sum(thickness_mask & valid_components_mask)} / {np.sum(valid_components_mask)}")
    
    # Apply thickness filter to skeleton
    filtered_skeleton = valid_components_mask & thickness_mask
    
    # Perform spider analysis on the remaining skeleton (after size and thickness filtering)
    pink_mask = spider_analysis(filtered_skeleton, branch_points, thickness)
    
    # Classify skeleton pixels (matching original approach)
    # Blue region: normal branch density
    # Pink region: high branch density (assigned by spider analysis)
    blue_region = filtered_skeleton & ~pink_mask
    pink_region = pink_mask
    
    print(f"  All skeleton pixels: {np.sum(skeleton > 0)}")
    print(f"  Valid skeleton pixels (≥{FILTER_PARAMS['MIN_SKELETON_LENGTH']} pixels): {np.sum(skeleton & valid_components_mask)}")
    print(f"  Pixels after thickness filter: {np.sum(filtered_skeleton > 0)}")
    print(f"  Blue region pixels: {np.sum(blue_region)}")
    print(f"  Pink region pixels: {np.sum(pink_region)}")
    
    return {
        'gray': gray,
        'thresholded': thresholded,
        'filtered_mask': filtered_mask,
        'skeleton': skeleton,
        'thickness': thickness,
        'filtered_skeleton': filtered_skeleton,
        'blue_region': blue_region,
        'pink_region': pink_region,
        'skeleton_to_mother_length': skeleton_to_mother_length,
        'predicted_threshold': predicted_threshold,
        'branch_points': branch_points
    }

def analyze_image_with_batch_threshold(image_path, threshold_value, output_dir=None):
    """Analyze a single image using a pre-calculated batch threshold."""
    # Use the shared core processing function
    processed_data = process_image_core(image_path, threshold_value)
    
    # Use the shared component processing and visualization function
    # If output_dir is provided, use it; otherwise use the image directory
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    results = process_components_and_visualize(processed_data, image_path, output_dir)
    
    return results

def analyze_image(image_path, output_dir=None):
    """Analyze a single image with branch-based snakes approach."""
    # Use the shared core processing function (no threshold provided = calculate per image)
    processed_data = process_image_core(image_path)
    
    # Use the shared component processing and visualization function
    # If output_dir is provided, use it; otherwise use the global output_dir
    if output_dir is None:
        output_dir = globals().get('output_dir', os.path.dirname(image_path))
    
    results = process_components_and_visualize(processed_data, image_path, output_dir)
    
    return results

def process_single_image_parallel(args):
    """
    Wrapper function for parallel processing of a single image.
    This function needs to be at module level for multiprocessing.
    
    Args:
        args: Tuple of (image_path, output_dir, design_name, condition, individual_images_dir)
    
    Returns:
        dict: Results dictionary with success status and data
    """
    image_path, output_dir, design_name, condition, individual_images_dir = args
    
    try:
        base_name = os.path.basename(image_path)
        
        # Check if this file should be excluded
        for excluded_image in excluded_images:
            if excluded_image in base_name:
                return {
                    'success': False,
                    'image_path': image_path,
                    'reason': 'excluded',
                    'data': None
                }
        
        # Check if this image has already been processed (image-thresholded)
        if individual_images_dir is None:
            individual_dir = os.path.join(output_dir, "individual_images_individual")
        else:
            individual_dir = individual_images_dir
        csv_dir = output_dir  # output_dir is already the correct csv directory
        
        # Ensure directories exist
        os.makedirs(individual_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        
        summary_plot_path = os.path.join(individual_dir, f"{base_name.replace('.jpg', '')}_summary_plot.png")
        main_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pixel_thickness.csv")
        blue_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_blue_components.csv")
        pink_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pink_components.csv")
        
        # Check if all expected output files exist
        files_exist = (
            os.path.exists(summary_plot_path) and
            os.path.exists(main_csv_path) and
            os.path.exists(blue_csv_path) and
            os.path.exists(pink_csv_path)
        )
        
        if files_exist:
            # Load existing data for CDF plotting
            try:
                blue_df = pd.read_csv(blue_csv_path)
                if 'average_thickness' in blue_df.columns:
                    thickness_values = blue_df['average_thickness'].tolist()
                    return {
                        'success': True,
                        'image_path': image_path,
                        'reason': 'already_processed',
                        'data': {
                            'thickness_values': thickness_values,
                            'image_avg_thickness': np.mean(thickness_values) if thickness_values else 0
                        }
                    }
            except Exception as e:
                return {
                    'success': False,
                    'image_path': image_path,
                    'reason': f'error_loading_existing: {str(e)}',
                    'data': None
                }
        
        # Process the image
        results = analyze_image(image_path, csv_dir)
        
        # Save pixel-level data
        pixel_df = pd.DataFrame(results['pixel_data'])
        pixel_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pixel_thickness.csv")
        pixel_df.to_csv(pixel_csv_path, index=False)
        
        # Save blue component-level data
        blue_component_df = pd.DataFrame(results['blue_component_data'])
        blue_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_blue_components.csv")
        blue_component_df.to_csv(blue_csv_path, index=False)
        
        # Save pink component-level data
        pink_component_df = pd.DataFrame(results['pink_component_data'])
        pink_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pink_components.csv")
        pink_component_df.to_csv(pink_csv_path, index=False)
        
        # Move summary plot to condition directory
        new_summary_path = os.path.join(individual_dir, f"{base_name.replace('.jpg', '')}_summary_plot.png")
        if os.path.exists(results['summary_plot_path']):
            os.rename(results['summary_plot_path'], new_summary_path)
        
        # Extract thickness data
        thickness_values = []
        image_avg_thickness = 0
        if 'average_thickness' in blue_component_df.columns:
            thickness_values = blue_component_df['average_thickness'].tolist()
            if len(thickness_values) > 0:
                image_avg_thickness = np.mean(thickness_values)
        
        return {
            'success': True,
            'image_path': image_path,
            'reason': 'processed',
            'data': {
                'thickness_values': thickness_values,
                'image_avg_thickness': image_avg_thickness,
                'pixel_count': len(results['pixel_data']),
                'blue_component_count': len(results['blue_component_data']),
                'pink_component_count': len(results['pink_component_data'])
            }
        }
        
    except Exception as e:
        # Include more detailed error information for debugging
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return {
            'success': False,
            'image_path': image_path,
            'reason': f'processing_error: {error_msg}',
            'data': None
        }

def process_components_and_visualize(processed_data, image_path, output_dir):
    """Shared function for component processing and visualization used by both analyses."""
    # Extract data from processed_data
    gray = processed_data['gray']
    filtered_mask = processed_data['filtered_mask']
    skeleton = processed_data['skeleton']
    thickness = processed_data['thickness']
    filtered_skeleton = processed_data['filtered_skeleton']
    blue_region = processed_data['blue_region']
    pink_region = processed_data['pink_region']
    skeleton_to_mother_length = processed_data['skeleton_to_mother_length']
    branch_points = processed_data['branch_points']
    
    # Process blue components (same as original analyze_image)
    print(f"  Processing blue components...")
    blue_labeled = label(blue_region)
    blue_components = regionprops(blue_labeled)
    blue_components_original = blue_components.copy()  # Save original for pre-filter visualization
    
    # Process pink components
    print(f"  Processing pink components...")
    pink_labeled = label(pink_region)
    pink_components = regionprops(pink_labeled)
    
    # Apply component filters ONLY to blue components (same as blue_component_filter.py)
    print(f"  Applying filters to blue components only:")
    print(f"    - max_thickness <= {FILTER_PARAMS['MAX_COMPONENT_THICKNESS']}")
    print(f"    - min_thickness >= {FILTER_PARAMS['MIN_THICKNESS']}")
    print(f"    - avg_thickness between {FILTER_PARAMS['MIN_AVG_THICKNESS']} and {FILTER_PARAMS['MAX_AVG_THICKNESS']}")
    print(f"    - skeleton_length >= {FILTER_PARAMS['MIN_SKELETON_LENGTH']} pixels")
    print(f"    - skeleton_length <= {FILTER_PARAMS['MAX_SKELETON_LENGTH']} pixels")
    print(f"    - mother_component_length >= {FILTER_PARAMS['MIN_SKELETON_LENGTH']} pixels")
    print(f"    - mother_component_length <= {FILTER_PARAMS['MAX_SKELETON_LENGTH']} pixels")
    print(f"  Pink components (high density regions) are not filtered")
    
    def apply_component_filters(components, thickness_map, region_mask, mother_length_map):
        """Apply filters to components based on thickness, size, and mother component length criteria."""
        filtered_components = []
        
        for prop in components:
            coords = prop.coords
            if len(coords) == 0:
                continue
                
            # Get thickness values for this component
            component_thicknesses = []
            for y, x in coords:
                if thickness_map[y, x] > 0:
                    component_thicknesses.append(thickness_map[y, x])
            
            if not component_thicknesses:
                continue
                
            avg_thickness = np.mean(component_thicknesses)
            max_thickness = np.max(component_thicknesses)
            min_thickness = np.min(component_thicknesses)
            skeleton_length = len(coords)
            
            # Get mother component length for this component
            mother_lengths = []
            for y, x in coords:
                if mother_length_map[y, x] > 0:
                    mother_lengths.append(mother_length_map[y, x])
            mother_component_length = np.max(mother_lengths) if mother_lengths else skeleton_length
            
            # Apply filters
            if (max_thickness <= FILTER_PARAMS["MAX_COMPONENT_THICKNESS"] and
                min_thickness >= FILTER_PARAMS["MIN_THICKNESS"] and
                FILTER_PARAMS["MIN_AVG_THICKNESS"] <= avg_thickness <= FILTER_PARAMS["MAX_AVG_THICKNESS"] and
                FILTER_PARAMS["MIN_SKELETON_LENGTH"] <= skeleton_length <= FILTER_PARAMS["MAX_SKELETON_LENGTH"] and
                FILTER_PARAMS["MIN_SKELETON_LENGTH"] <= mother_component_length <= FILTER_PARAMS["MAX_SKELETON_LENGTH"]):
                filtered_components.append(prop)
        
        return filtered_components
    
    # Apply filters to blue components
    filtered_blue_components = apply_component_filters(blue_components, thickness, blue_region, skeleton_to_mother_length)
    
    print(f"  Blue components: {len(blue_components)} -> {len(filtered_blue_components)} (after filtering)")
    print(f"  Pink components: {len(pink_components)} (no filtering applied - high density regions)")
    
    # Update blue region to only include filtered components
    blue_region_filtered = np.zeros_like(blue_region, dtype=bool)
    
    for prop in filtered_blue_components:
        coords = prop.coords
        for y, x in coords:
            blue_region_filtered[y, x] = True
    
    # Update the blue region for visualization (pink region stays unfiltered)
    blue_region = blue_region_filtered
    blue_components = filtered_blue_components
    
    # Label blue components
    print(f"  Labeling blue components...")
    blue_component_map = np.zeros_like(skeleton, dtype=int)
    for i, prop in enumerate(blue_components):
        coords = prop.coords
        for y, x in coords:
            blue_component_map[y, x] = i + 1
    
    print(f"  Found {len(blue_components)} blue components")
    
    # Create summary plot (3x3 grid to accommodate pre/post filter panels + thick pixels panel)
    fig, axes = plt.subplots(3, 3, figsize=(27, 18))
    
    # Original image (brightened)
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image (Brightened)')
    axes[0, 0].axis('off')
    
    # Filtered mask
    axes[0, 1].imshow(filtered_mask, cmap='gray')
    axes[0, 1].set_title('Filtered Mask')
    axes[0, 1].axis('off')
    
    # Skeleton radius map
    skeleton_radius = np.zeros_like(skeleton, dtype=float)
    skeleton_radius[filtered_skeleton > 0] = thickness[filtered_skeleton > 0]
    skeleton_radius[skeleton_radius == 0] = np.nan
    im1 = axes[0, 2].imshow(skeleton_radius, cmap='viridis')
    axes[0, 2].set_title('Skeleton Radius Map')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2])
    
    # Excluded pixels (top right)
    excluded_pixels = np.zeros_like(skeleton, dtype=bool)
    # Pixels that were in original skeleton but got filtered out
    excluded_pixels = (skeleton > 0) & ~filtered_skeleton
    axes[0, 2].imshow(excluded_pixels, cmap='Reds')
    axes[0, 2].set_title(f'Excluded Pixels (n={np.sum(excluded_pixels)})')
    axes[0, 2].axis('off')
    
    # Branch-based snakes: Blue=Normal, Pink=High Branch Density
    region_vis = np.zeros((*skeleton.shape, 3))
    region_vis[blue_region] = [0, 0, 1]  # Blue
    region_vis[pink_region] = [1, 0, 1]  # Pink
    axes[1, 0].imshow(region_vis)
    axes[1, 0].set_title('Branch-Based Snakes: Blue=Normal, Pink=High Branch Density')
    axes[1, 0].axis('off')
    
    # Blue components (individual colored components)
    blue_component_colored = np.zeros((*skeleton.shape, 3))
    for i, prop in enumerate(blue_components):
        coords = prop.coords
        # Generate a unique color for each component
        color = plt.cm.tab20(i % 20)
        for y, x in coords:
            blue_component_colored[y, x] = color[:3]
    axes[1, 1].imshow(blue_component_colored)
    axes[1, 1].set_title(f'Blue Components Post-Filter (n={len(blue_components)})')
    axes[1, 1].axis('off')
    
    # Pink components (individual colored components)
    pink_component_colored = np.zeros((*skeleton.shape, 3))
    for i, prop in enumerate(pink_components):
        coords = prop.coords
        # Generate a unique color for each component
        color = plt.cm.tab20(i % 20)
        for y, x in coords:
            pink_component_colored[y, x] = color[:3]
    axes[1, 2].imshow(pink_component_colored)
    axes[1, 2].set_title(f'Pink Components (n={len(pink_components)})')
    axes[1, 2].axis('off')
    
    # Blue components pre-filter (individual colored components)
    blue_pre_filter_colored = np.zeros((*skeleton.shape, 3))
    for i, prop in enumerate(blue_components_original):  # Use original unfiltered components
        coords = prop.coords
        # Generate a unique color for each component
        color = plt.cm.tab20(i % 20)
        for y, x in coords:
            blue_pre_filter_colored[y, x] = color[:3]
    axes[2, 0].imshow(blue_pre_filter_colored)
    axes[2, 0].set_title(f'Blue Components Pre-Filter (n={len(blue_components_original)})')
    axes[2, 0].axis('off')
    
    # Thick pixels visualization (bright orange)
    thick_pixels_mask = np.zeros_like(skeleton, dtype=bool)

    # Create mask of all thick pixels across all blue components
    # Use the global thickness values that were already calculated
    for i, prop in enumerate(blue_components):
        coords = prop.coords
        
        # Get skeleton pixels within this blue component
        component_skeleton = np.zeros_like(skeleton, dtype=bool)
        for y, x in coords:
            if skeleton[y, x]:
                component_skeleton[y, x] = True
        
        if not np.any(component_skeleton):
            continue
        
        # Use the global thickness values for this component
        wide_mask = (thickness >= THICK_THIN_CONFIG["width_threshold"]) & component_skeleton
        
        # Add thick pixels to the overall mask
        thick_pixels_mask |= wide_mask
    
    # Filter out isolated thick pixels - require at least 8 connected thick pixels
    if np.any(thick_pixels_mask):
        # Label connected components of thick pixels
        thick_components = label(thick_pixels_mask)
        thick_props = regionprops(thick_components)
        
        # Create new mask with only components that have at least 8 pixels
        filtered_thick_mask = np.zeros_like(thick_pixels_mask, dtype=bool)
        for prop in thick_props:
            if prop.area >= 8:  # Require at least 8 connected thick pixels
                coords = prop.coords
                for y, x in coords:
                    filtered_thick_mask[y, x] = True
        
        thick_pixels_mask = filtered_thick_mask
    
    # Create visualization with bright orange thick pixels overlaid on original image
    thick_pixels_vis = gray.copy()
    thick_pixels_vis = np.stack([thick_pixels_vis] * 3, axis=-1)  # Convert to RGB
    
    # Add bright orange for thick pixels
    bright_orange = np.array([1, 0.5, 0])  # Bright orange color
    for y in range(thick_pixels_vis.shape[0]):
        for x in range(thick_pixels_vis.shape[1]):
            if thick_pixels_mask[y, x]:
                # Blend with original image (70% orange, 30% original)
                thick_pixels_vis[y, x] = 0.7 * bright_orange + 0.3 * thick_pixels_vis[y, x]
    
    axes[2, 1].imshow(thick_pixels_vis)
    axes[2, 1].set_title(f'Thick Pixels (Bright Orange, n={np.sum(thick_pixels_mask)})')
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    # Save summary plot
    base_name = os.path.basename(image_path).replace('.jpg', '')
    summary_plot_path = os.path.join(output_dir, f"{base_name}_summary_plot.png")
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Summary plot created for {os.path.basename(image_path)}")
    
    # Prepare data for CSV output (same as original analyze_image)
    # Pixel-level data
    pixel_data = []
    for y in range(skeleton.shape[0]):
        for x in range(skeleton.shape[1]):
            if skeleton[y, x] > 0:
                region_type = 'blue' if blue_region[y, x] else 'pink'
                pixel_data.append({
                    'y_coordinate': y,
                    'x_coordinate': x,
                    'thickness': thickness[y, x],
                    'region_type': region_type
                })
    
    # Calculate mother component lengths for blue components from original skeleton mapping
    blue_mother_lengths = []
    for prop in blue_components:
        coords = prop.coords
        mother_lengths = [skeleton_to_mother_length[y, x] for y, x in coords]
        blue_mother_lengths.append(np.max(mother_lengths) if mother_lengths else len(coords))
    
    # Blue component-level data
    blue_component_data = []
    for i, prop in enumerate(blue_components):
        coords = prop.coords
        component_thicknesses = [thickness[y, x] for y, x in coords if thickness[y, x] > 0]
        
        if component_thicknesses:
            blue_component_data.append({
                'component_id': i + 1,
                'area': prop.area,
                'perimeter': prop.perimeter,
                'average_thickness': np.mean(component_thicknesses),
                'max_thickness': np.max(component_thicknesses),
                'min_thickness': np.min(component_thicknesses),
                'centroid_y': prop.centroid[0],
                'centroid_x': prop.centroid[1],
                'mother_component_length': blue_mother_lengths[i],
                'region_type': 'blue'
            })
    
    # Pink component-level data
    pink_component_data = []
    for i, prop in enumerate(pink_components):
        coords = prop.coords
        component_thicknesses = [thickness[y, x] for y, x in coords if thickness[y, x] > 0]
        
        if component_thicknesses:
            pink_component_data.append({
                'component_id': i + 1,
                'area': prop.area,
                'perimeter': prop.perimeter,
                'average_thickness': np.mean(component_thicknesses),
                'max_thickness': np.max(component_thicknesses),
                'min_thickness': np.min(component_thicknesses),
                'centroid_y': prop.centroid[0],
                'centroid_x': prop.centroid[1],
                'region_type': 'pink'
            })
    
    # Perform thick-thin analysis on blue components
    thick_thin_data = analyze_thick_thin_blue_components(blue_components, skeleton, thickness, branch_points, width_threshold=THICK_THIN_CONFIG["width_threshold"])
    
    # Merge thick-thin data with blue component data
    for i, thick_thin_result in enumerate(thick_thin_data):
        if i < len(blue_component_data):
            blue_component_data[i].update(thick_thin_result)
    
    return {
        'pixel_data': pixel_data,
        'blue_component_data': blue_component_data,
        'pink_component_data': pink_component_data,
        'thick_thin_data': thick_thin_data,
        'summary_plot_path': summary_plot_path
    }

def analyze_thick_thin_blue_components(blue_components, skeleton, thickness, branch_points, width_threshold=THICK_THIN_CONFIG["width_threshold"]):
    """Analyze thick vs thin blue components and return thick-thin data."""
    thick_thin_data = []
    
    for i, prop in enumerate(blue_components):
        coords = prop.coords
        
        # Get skeleton pixels within this blue component
        component_skeleton = np.zeros_like(skeleton, dtype=bool)
        for y, x in coords:
            if skeleton[y, x]:
                component_skeleton[y, x] = True
        
        if not np.any(component_skeleton):
            # If no skeleton pixels, skip this component
            thick_thin_data.append({
                'component_id': i + 1,
                'thick_pixels': 0,
                'thin_pixels': 0,
                'ratio_thick_thin': 0.0,
                'thick_percentage': 0.0,
                'thin_percentage': 0.0
            })
            continue
        
        # Use the global thickness values for this component
        wide_mask = (thickness >= width_threshold) & component_skeleton
        
        # Count thick and thin pixels
        thick_pixels = np.sum(wide_mask)
        thin_pixels = np.sum(component_skeleton) - thick_pixels
        
        # Calculate ratios and percentages
        ratio_thick_thin = thick_pixels / thin_pixels if thin_pixels > 0 else 0.0
        total_pixels = thick_pixels + thin_pixels
        thick_percentage = (thick_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
        thin_percentage = (thin_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
        
        thick_thin_data.append({
            'component_id': i + 1,
            'thick_pixels': int(thick_pixels),
            'thin_pixels': int(thin_pixels),
            'ratio_thick_thin': round(ratio_thick_thin, 3),
            'thick_percentage': round(thick_percentage, 1),
            'thin_percentage': round(thin_percentage, 1)
        })
    
    return thick_thin_data

def process_single_threshold_comparison(args):
    """
    Process a single image for threshold comparison analysis.
    
    Args:
        args: Tuple containing (image_path, mean_intensity, batch_threshold, batch_name, analysis_dir, design_name)
    
    Returns:
        dict: Results of the comparison analysis
    """
    image_path, mean_intensity, batch_threshold, batch_name, analysis_dir, design_name = args
    
    try:
        # Calculate individual threshold for this image
        slope = 1.4865603768263806
        intercept = -0.003583494788075847
        individual_threshold = slope * mean_intensity + intercept
        
        # Calculate deviation from batch threshold
        deviation = individual_threshold - batch_threshold
        
        # Verify the image file exists
        if not os.path.exists(image_path):
            return {
                'success': False,
                'reason': f"Could not find image file: {image_path}"
            }
        
        # Get condition from the comprehensive mapping
        condition = None
        try:
            # Load the comprehensive mapping to get condition info
            mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
            filename = os.path.basename(image_path)
            file_info = mapping_df[mapping_df['filename'] == filename]
            if not file_info.empty:
                condition = file_info.iloc[0]['condition']
        except Exception as e:
            return {
                'success': False,
                'reason': f"Could not get condition info for {os.path.basename(image_path)}: {e}"
            }
        
        if not condition:
            return {
                'success': False,
                'reason': f"Could not find condition for {os.path.basename(image_path)} in {design_name}"
            }
        
        # Create temporary output directories for this comparison
        temp_output_dir = os.path.join(analysis_dir, "temp_analysis")
        batch_temp_dir = os.path.join(temp_output_dir, "batch_threshold")
        individual_temp_dir = os.path.join(temp_output_dir, "individual_threshold")
        os.makedirs(batch_temp_dir, exist_ok=True)
        os.makedirs(individual_temp_dir, exist_ok=True)
        
        # Analyze with batch threshold
        batch_results = analyze_image_with_batch_threshold(image_path, batch_threshold, batch_temp_dir)
        
        # Analyze with individual threshold
        individual_results = analyze_image_with_batch_threshold(image_path, individual_threshold, individual_temp_dir)
        
        # Extract component data
        batch_blue_components = batch_results['blue_component_data']
        individual_blue_components = individual_results['blue_component_data']
        
        # Calculate component count change (batch - individual)
        batch_count = len(batch_blue_components)
        individual_count = len(individual_blue_components)
        component_change = batch_count - individual_count
        
        # Calculate average thickness ratio (batch / individual)
        if len(batch_blue_components) > 0 and len(individual_blue_components) > 0:
            batch_avg_thickness = np.mean([comp['average_thickness'] for comp in batch_blue_components])
            individual_avg_thickness = np.mean([comp['average_thickness'] for comp in individual_blue_components])
            thickness_ratio = batch_avg_thickness / individual_avg_thickness if individual_avg_thickness > 0 else 1.0
        else:
            thickness_ratio = 1.0  # Default if no components
        
        return {
            'success': True,
            'deviation': deviation,
            'component_change': component_change,
            'thickness_ratio': thickness_ratio,
            'condition': condition,
            'image_name': os.path.basename(image_path)
        }
        
    except Exception as e:
        return {
            'success': False,
            'reason': f"Error processing {os.path.basename(image_path)}: {str(e)}"
        }

def process_single_image_parallel_batch_threshold(args):
    """
    Wrapper function for parallel processing of a single image with batch threshold.
    This function needs to be at module level for multiprocessing.
    
    Args:
        args: Tuple of (image_path, output_dir, design_name, condition, threshold_value, individual_images_dir)
    
    Returns:
        dict: Results dictionary with success status and data
    """
    image_path, output_dir, design_name, condition, threshold_value, individual_images_dir = args
    
    # Create csv_files_batch directory at the same level as individual_images
    if individual_images_dir is None:
        individual_dir = os.path.join(output_dir, "individual_images_batch")
    else:
        individual_dir = individual_images_dir
    csv_dir = individual_dir.replace("individual_images_batch", "csv_files_batch")
    
    try:
        base_name = os.path.basename(image_path)
        
        # Check if this file should be excluded
        for excluded_image in excluded_images:
            if excluded_image in base_name:
                return {
                    'success': False,
                    'image_path': image_path,
                    'reason': 'excluded',
                    'data': None
                }
        
        # Check if this image has already been processed
        summary_plot_path = os.path.join(individual_dir, f"{base_name.replace('.jpg', '')}_summary_plot.png")
        main_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pixel_thickness.csv")
        blue_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_blue_components.csv")
        pink_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pink_components.csv")
        
        # Check if all expected output files exist
        files_exist = (
            os.path.exists(summary_plot_path) and
            os.path.exists(main_csv_path) and
            os.path.exists(blue_csv_path) and
            os.path.exists(pink_csv_path)
        )
        
        if files_exist:
            # Load existing data for CDF plotting
            try:
                blue_df = pd.read_csv(blue_csv_path)
                if 'average_thickness' in blue_df.columns:
                    thickness_values = blue_df['average_thickness'].tolist()
                    return {
                        'success': True,
                        'image_path': image_path,
                        'reason': 'already_processed',
                        'data': {
                            'thickness_values': thickness_values,
                            'image_avg_thickness': np.mean(thickness_values) if thickness_values else 0
                        }
                    }
            except Exception as e:
                return {
                    'success': False,
                    'image_path': image_path,
                    'reason': f'error_loading_existing: {str(e)}',
                    'data': None
                }
        
        # Process the image with batch threshold
        results = analyze_image_with_batch_threshold(image_path, threshold_value, individual_dir)
        
        # Save pixel-level data to csv_files directory
        pixel_df = pd.DataFrame(results['pixel_data'])
        pixel_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pixel_thickness.csv")
        pixel_df.to_csv(pixel_csv_path, index=False)
        
        # Save blue component-level data to csv_files directory
        blue_component_df = pd.DataFrame(results['blue_component_data'])
        blue_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_blue_components.csv")
        blue_component_df.to_csv(blue_csv_path, index=False)
        
        # Save pink component-level data to csv_files directory
        pink_component_df = pd.DataFrame(results['pink_component_data'])
        pink_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pink_components.csv")
        pink_component_df.to_csv(pink_csv_path, index=False)
        
        # Move summary plot to individual_images directory
        new_summary_path = os.path.join(individual_dir, f"{base_name.replace('.jpg', '')}_summary_plot.png")
        if os.path.exists(results['summary_plot_path']):
            os.rename(results['summary_plot_path'], new_summary_path)
        
        # Extract thickness data
        thickness_values = []
        image_avg_thickness = 0
        if 'average_thickness' in blue_component_df.columns:
            thickness_values = blue_component_df['average_thickness'].tolist()
            if len(thickness_values) > 0:
                image_avg_thickness = np.mean(thickness_values)
        
        return {
            'success': True,
            'image_path': image_path,
            'reason': 'processed',
            'data': {
                'thickness_values': thickness_values,
                'image_avg_thickness': image_avg_thickness,
                'pixel_count': len(results['pixel_data']),
                'blue_component_count': len(results['blue_component_data']),
                'pink_component_count': len(results['pink_component_data'])
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'image_path': image_path,
            'reason': f'error: {str(e)}',
            'data': None
        }

def process_images_parallel_batch_threshold(condition_files, output_dir, design_name, condition, threshold_value, individual_images_dir=None, max_workers=None):
    """
    Process multiple images in parallel for a given condition with batch threshold.
    
    Args:
        condition_files: List of image file paths
        output_dir: Output directory
        design_name: Name of the experimental design
        condition: Condition name
        threshold_value: Batch threshold value to use
        max_workers: Maximum number of worker processes (default: CPU count)
    
    Returns:
        tuple: (all_thickness_data, batch_data, image_level_data, image_level_batch_data, files_with_errors)
    """
    # Use configuration or auto-detect optimal number of workers
    if max_workers is None:
        if PARALLEL_CONFIG["max_workers"] is not None:
            max_workers = PARALLEL_CONFIG["max_workers"]
        else:
            # Auto-detect optimal number of workers
            cpu_count = mp.cpu_count()
            # Use 75% of CPU cores to avoid overwhelming the system
            max_workers = max(1, min(int(cpu_count * 0.75), len(condition_files)))
    
    print(f"  Processing {len(condition_files)} images with {max_workers} parallel workers...")
    print(f"  CPU cores available: {mp.cpu_count()}")
    
    # Memory optimization: pre-allocate data structures
    if PARALLEL_CONFIG["memory_optimization"]:
        estimated_components = len(condition_files) * 50  # Estimate 50 components per image
        all_thickness_data = []
        # Note: Python lists don't have reserve() method, but we can pre-allocate if needed
    else:
        all_thickness_data = []
    
    batch_data = defaultdict(lambda: defaultdict(list))
    image_level_data = []
    image_level_batch_data = defaultdict(lambda: defaultdict(list))
    files_with_errors = []
    
    # Prepare arguments for parallel processing
    args_list = [(image_path, output_dir, design_name, condition, threshold_value, individual_images_dir) for image_path in condition_files]
    
    # Process images in parallel
    start_time = time.time()
    processed_count = 0
    progress_interval = PARALLEL_CONFIG["progress_update_interval"]
    
    # Use context manager for better resource management
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_image_parallel_batch_threshold, args): args for args in args_list}
        
        # Process completed tasks
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            image_path = args[0]
            base_name = os.path.basename(image_path)
            
            try:
                result = future.result()
                processed_count += 1
                
                if result['success']:
                    if result['reason'] == 'excluded':
                        print(f"    🚫 Excluded: {base_name} (hard-coded exclusion)")
                    elif result['reason'] == 'already_processed':
                        print(f"    ✓ Already processed: {base_name}")
                        # Collect data from existing files
                        if result['data']:
                            all_thickness_data.extend(result['data']['thickness_values'])
                            image_level_data.append(result['data']['image_avg_thickness'])
                            
                            # Track batch information from comprehensive mapping
                            try:
                                mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                                filename = os.path.basename(image_path)
                                file_info = mapping_df[mapping_df['filename'] == filename]
                                if not file_info.empty:
                                    batch = file_info.iloc[0]['biological_replicate']
                                    batch_data[batch][condition].extend(result['data']['thickness_values'])
                                    image_level_batch_data[batch][condition].append(result['data']['image_avg_thickness'])
                            except Exception as e:
                                print(f"    Warning: Could not get batch info for {filename}: {e}")
                    elif result['reason'] == 'processed':
                        print(f"    ✓ Completed: {base_name}")
                        if result['data']:
                            all_thickness_data.extend(result['data']['thickness_values'])
                            image_level_data.append(result['data']['image_avg_thickness'])
                            
                            # Track batch information from comprehensive mapping
                            try:
                                mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                                filename = os.path.basename(image_path)
                                file_info = mapping_df[mapping_df['filename'] == filename]
                                if not file_info.empty:
                                    batch = file_info.iloc[0]['biological_replicate']
                                    batch_data[batch][condition].extend(result['data']['thickness_values'])
                                    image_level_batch_data[batch][condition].append(result['data']['image_avg_thickness'])
                            except Exception as e:
                                print(f"    Warning: Could not get batch info for {filename}: {e}")
                        
                        print(f"      - {result['data']['pixel_count']} pixels analyzed")
                        print(f"      - {result['data']['blue_component_count']} blue components found")
                        print(f"      - {result['data']['pink_component_count']} pink components found")
                else:
                    print(f"    ❌ Error processing {base_name}: {result['reason']}")
                    files_with_errors.append(image_path)
                
                # Progress update
                if processed_count % progress_interval == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_image = elapsed_time / processed_count
                    remaining_images = len(condition_files) - processed_count
                    estimated_remaining_time = remaining_images * avg_time_per_image
                    print(f"    Progress: {processed_count}/{len(condition_files)} images processed ({processed_count/len(condition_files)*100:.1f}%)")
                    print(f"    Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")
                
            except Exception as e:
                print(f"    ❌ Exception processing {base_name}: {str(e)}")
                files_with_errors.append(image_path)
                processed_count += 1
    
    # Final progress update
    total_time = time.time() - start_time
    print(f"  Parallel processing completed in {total_time/60:.1f} minutes")
    print(f"  Successfully processed: {len(condition_files) - len(files_with_errors)}/{len(condition_files)} images")
    if files_with_errors:
        print(f"  Files with errors: {len(files_with_errors)}")
    
    return all_thickness_data, batch_data, image_level_data, image_level_batch_data, files_with_errors

def process_images_parallel(condition_files, output_dir, design_name, condition, individual_images_dir=None, max_workers=None):
    """
    Process multiple images in parallel for a given condition.
    
    Args:
        condition_files: List of image file paths
        output_dir: Output directory
        design_name: Name of the experimental design
        condition: Condition name
        max_workers: Maximum number of worker processes (default: CPU count)
    
    Returns:
        tuple: (all_thickness_data, batch_data, image_level_data, image_level_batch_data, files_with_errors)
    """
    # Use configuration or auto-detect optimal number of workers
    if max_workers is None:
        if PARALLEL_CONFIG["max_workers"] is not None:
            max_workers = PARALLEL_CONFIG["max_workers"]
        else:
            # Auto-detect optimal number of workers
            cpu_count = mp.cpu_count()
            # Use 75% of CPU cores to avoid overwhelming the system
            max_workers = max(1, min(int(cpu_count * 0.75), len(condition_files)))
    
    print(f"  Processing {len(condition_files)} images with {max_workers} parallel workers...")
    print(f"  CPU cores available: {mp.cpu_count()}")
    
    # Memory optimization: pre-allocate data structures
    if PARALLEL_CONFIG["memory_optimization"]:
        estimated_components = len(condition_files) * 50  # Estimate 50 components per image
        all_thickness_data = []
        # Note: Python lists don't have reserve() method, but we can pre-allocate if needed
    else:
        all_thickness_data = []
    
    batch_data = defaultdict(lambda: defaultdict(list))
    image_level_data = []
    image_level_batch_data = defaultdict(lambda: defaultdict(list))
    files_with_errors = []
    
    # Prepare arguments for parallel processing
    args_list = [(image_path, output_dir, design_name, condition, individual_images_dir) for image_path in condition_files]
    
    # Process images in parallel
    start_time = time.time()
    processed_count = 0
    progress_interval = PARALLEL_CONFIG["progress_update_interval"]
    
    # Use context manager for better resource management
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_image_parallel, args): args for args in args_list}
        
        # Process completed tasks
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            image_path = args[0]
            base_name = os.path.basename(image_path)
            
            try:
                result = future.result()
                processed_count += 1
                
                if result['success']:
                    if result['reason'] == 'excluded':
                        print(f"    🚫 Excluded: {base_name} (hard-coded exclusion)")
                    elif result['reason'] == 'already_processed':
                        print(f"    ✓ Already processed: {base_name}")
                        # Collect data from existing files
                        if result['data']:
                            all_thickness_data.extend(result['data']['thickness_values'])
                            image_level_data.append(result['data']['image_avg_thickness'])
                            
                            # Track batch information from comprehensive mapping
                            try:
                                mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                                filename = os.path.basename(image_path)
                                file_info = mapping_df[mapping_df['filename'] == filename]
                                if not file_info.empty:
                                    batch = file_info.iloc[0]['biological_replicate']
                                    batch_data[batch][condition].extend(result['data']['thickness_values'])
                                    image_level_batch_data[batch][condition].append(result['data']['image_avg_thickness'])
                            except Exception as e:
                                print(f"    Warning: Could not get batch info for {filename}: {e}")
                    elif result['reason'] == 'processed':
                        print(f"    ✓ Completed: {base_name}")
                        if result['data']:
                            all_thickness_data.extend(result['data']['thickness_values'])
                            image_level_data.append(result['data']['image_avg_thickness'])
                            
                            # Track batch information from comprehensive mapping
                            try:
                                mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                                filename = os.path.basename(image_path)
                                file_info = mapping_df[mapping_df['filename'] == filename]
                                if not file_info.empty:
                                    batch = file_info.iloc[0]['biological_replicate']
                                    batch_data[batch][condition].extend(result['data']['thickness_values'])
                                    image_level_batch_data[batch][condition].append(result['data']['image_avg_thickness'])
                            except Exception as e:
                                print(f"    Warning: Could not get batch info for {filename}: {e}")
                        
                        print(f"      - {result['data']['pixel_count']} pixels analyzed")
                        print(f"      - {result['data']['blue_component_count']} blue components found")
                        print(f"      - {result['data']['pink_component_count']} pink components found")
                else:
                    print(f"    ❌ Error processing {base_name}: {result['reason']}")
                    files_with_errors.append(image_path)
                
                # Progress update
                if processed_count % progress_interval == 0 or processed_count == len(condition_files):
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    remaining = (len(condition_files) - processed_count) / rate if rate > 0 else 0
                    print(f"    Progress: {processed_count}/{len(condition_files)} ({processed_count/len(condition_files)*100:.1f}%) - "
                          f"Rate: {rate:.1f} images/sec - ETA: {remaining/60:.1f} min")
                
            except Exception as e:
                print(f"    ❌ Unexpected error processing {base_name}: {str(e)}")
                files_with_errors.append(image_path)
                processed_count += 1
    
    total_time = time.time() - start_time
    print(f"  ✓ Parallel processing completed in {total_time:.1f} seconds")
    print(f"  ✓ Processed {processed_count} images at {processed_count/total_time:.1f} images/sec")
    
    return all_thickness_data, batch_data, image_level_data, image_level_batch_data, files_with_errors

def process_images_sequential(condition_files, output_dir, design_name, condition):
    """
    Process multiple images sequentially for a given condition (fallback method).
    
    Args:
        condition_files: List of image file paths
        output_dir: Output directory
        design_name: Name of the experimental design
        condition: Condition name
    
    Returns:
        tuple: (all_thickness_data, batch_data, image_level_data, image_level_batch_data, files_with_errors)
    """
    print(f"  Processing {len(condition_files)} images sequentially...")
    
    # Initialize data collection structures
    all_thickness_data = []
    batch_data = defaultdict(lambda: defaultdict(list))
    image_level_data = []
    image_level_batch_data = defaultdict(lambda: defaultdict(list))
    files_with_errors = []
    
    # Process images sequentially
    start_time = time.time()
    processed_count = 0
    
    for jpg_file in condition_files:
        base_name = os.path.basename(jpg_file)
        
        # Check if this file should be excluded
        skip_file = False
        for excluded_image in excluded_images:
            if excluded_image in base_name:
                print(f"    🚫 Excluded: {base_name} (hard-coded exclusion)")
                skip_file = True
                break
        
        if skip_file:
            continue
        
        # Get the condition directory
        condition_dir = output_dir
        individual_dir = os.path.join(condition_dir, "individual_images_individual")
        csv_dir = os.path.join(condition_dir, "csv_files_individual")
        
        # Check if this image has already been processed
        summary_plot_path = os.path.join(individual_dir, f"{base_name.replace('.jpg', '')}_summary_plot.png")
        main_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pixel_thickness.csv")
        blue_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_blue_components.csv")
        pink_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pink_components.csv")
        
        # Check if all expected output files exist
        files_exist = (
            os.path.exists(summary_plot_path) and
            os.path.exists(main_csv_path) and
            os.path.exists(blue_csv_path) and
            os.path.exists(pink_csv_path)
        )
        
        if files_exist:
            print(f"    ✓ Already processed: {base_name}")
            # Load existing data for CDF plotting (component-level)
            try:
                blue_df = pd.read_csv(blue_csv_path)
                if 'average_thickness' in blue_df.columns:
                    thickness_values = pd.to_numeric(blue_df['average_thickness'], errors='coerce').dropna().tolist()
                    all_thickness_data.extend(thickness_values)
                    
                    # Track batch information for normalization (component-level)
                    try:
                        mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                        filename = os.path.basename(jpg_file)
                        file_info = mapping_df[mapping_df['filename'] == filename]
                        if not file_info.empty:
                            batch = file_info.iloc[0]['biological_replicate']
                            batch_data[batch][condition].extend(thickness_values)
                    except Exception as e:
                        print(f"      Warning: Could not get batch info for {filename}: {e}")
                    
                    # Also collect image-level data (average of all components in this image)
                    if len(thickness_values) > 0:
                        image_avg_thickness = np.mean(thickness_values)
                        image_level_data.append(image_avg_thickness)
                        
                        # Track batch information for normalization (image-level)
                        try:
                            mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                            filename = os.path.basename(jpg_file)
                            file_info = mapping_df[mapping_df['filename'] == filename]
                            if not file_info.empty:
                                batch = file_info.iloc[0]['biological_replicate']
                                image_level_batch_data[batch][condition].append(image_avg_thickness)
                        except Exception as e:
                            print(f"      Warning: Could not get batch info for {filename}: {e}")
            except Exception as e:
                print(f"      Warning: Could not load existing data: {e}")
            continue
        
        try:
            # Analyze the image
            results = analyze_image(jpg_file, csv_dir)
            
            # Save pixel-level data
            pixel_df = pd.DataFrame(results['pixel_data'])
            pixel_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pixel_thickness.csv")
            pixel_df.to_csv(pixel_csv_path, index=False)
            
            # Save blue component-level data
            blue_component_df = pd.DataFrame(results['blue_component_data'])
            blue_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_blue_components.csv")
            blue_component_df.to_csv(blue_csv_path, index=False)
            
            # Save pink component-level data
            pink_component_df = pd.DataFrame(results['pink_component_data'])
            pink_csv_path = os.path.join(csv_dir, f"{base_name.replace('.jpg', '')}_pink_components.csv")
            pink_component_df.to_csv(pink_csv_path, index=False)
            
            # Move summary plot to condition directory
            new_summary_path = os.path.join(individual_dir, f"{base_name.replace('.jpg', '')}_summary_plot.png")
            if os.path.exists(results['summary_plot_path']):
                os.rename(results['summary_plot_path'], new_summary_path)
            
            # Collect thickness data for CDF plotting (component-level)
            if 'average_thickness' in blue_component_df.columns:
                thickness_values = pd.to_numeric(blue_component_df['average_thickness'], errors='coerce').dropna().tolist()
                all_thickness_data.extend(thickness_values)
                
                # Track batch information for normalization (component-level)
                try:
                    mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                    filename = os.path.basename(jpg_file)
                    file_info = mapping_df[mapping_df['filename'] == filename]
                    if not file_info.empty:
                        batch = file_info.iloc[0]['biological_replicate']
                        batch_data[batch][condition].extend(thickness_values)
                except Exception as e:
                    print(f"      Warning: Could not get batch info for {filename}: {e}")
                
                # Also collect image-level data (average of all components in this image)
                if len(thickness_values) > 0:
                    image_avg_thickness = np.mean(thickness_values)
                    image_level_data.append(image_avg_thickness)
                    
                    # Track batch information for normalization (image-level)
                    try:
                        mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                        filename = os.path.basename(jpg_file)
                        file_info = mapping_df[mapping_df['filename'] == filename]
                        if not file_info.empty:
                            batch = file_info.iloc[0]['biological_replicate']
                            image_level_batch_data[batch][condition].append(image_avg_thickness)
                    except Exception as e:
                        print(f"      Warning: Could not get batch info for {filename}: {e}")
            
            print(f"    ✓ Completed: {base_name}")
            print(f"      - {len(results['pixel_data'])} pixels analyzed")
            print(f"      - {len(results['blue_component_data'])} blue components found")
            print(f"      - {len(results['pink_component_data'])} pink components found")
            
        except Exception as e:
            print(f"    ❌ Error processing {base_name}: {str(e)}")
            files_with_errors.append(jpg_file)
            continue
        
        processed_count += 1
        
        # Progress update
        if processed_count % PARALLEL_CONFIG["progress_update_interval"] == 0 or processed_count == len(condition_files):
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            remaining = (len(condition_files) - processed_count) / rate if rate > 0 else 0
            print(f"    Progress: {processed_count}/{len(condition_files)} ({processed_count/len(condition_files)*100:.1f}%) - "
                  f"Rate: {rate:.1f} images/sec - ETA: {remaining/60:.1f} min")
    
    total_time = time.time() - start_time
    print(f"  ✓ Sequential processing completed in {total_time:.1f} seconds")
    print(f"  ✓ Processed {processed_count} images at {processed_count/total_time:.1f} images/sec")
    
    return all_thickness_data, batch_data, image_level_data, image_level_batch_data, files_with_errors

def generate_image_level_data(component_data_dict, batch_data, output_dir, design_name):
    """Generate image-level data by averaging blue component thicknesses per image."""
    print(f"\nGenerating image-level data for {design_name}...")
    
    # Dictionary to collect image-level thickness data
    image_level_data = defaultdict(list)
    image_level_batch_data = defaultdict(lambda: defaultdict(list))
    
    # Dictionary to collect orange pixel percentage data
    orange_pixel_data = defaultdict(list)
    orange_pixel_batch_data = defaultdict(lambda: defaultdict(list))
    
    # Process each condition
    for condition, component_thicknesses in component_data_dict.items():
        # Find all CSV files for this condition (check both batch and individual directories)
        condition_dir_batch = os.path.join(output_dir, design_name, condition, "csv_files_batch")
        condition_dir_individual = os.path.join(output_dir, design_name, condition, "csv_files_individual")
        
        blue_csv_files = []
        if os.path.exists(condition_dir_batch):
            blue_csv_files.extend(glob.glob(os.path.join(condition_dir_batch, "*_blue_components.csv")))
        if os.path.exists(condition_dir_individual):
            blue_csv_files.extend(glob.glob(os.path.join(condition_dir_individual, "*_blue_components.csv")))
            
            print(f"  Processing {condition}: {len(blue_csv_files)} image files")
            
            for csv_file in blue_csv_files:
                try:
                    # Extract image name from CSV filename
                    base_name = os.path.basename(csv_file).replace('_blue_components.csv', '')
                    
                    # Read the CSV file
                    df = pd.read_csv(csv_file)
                    
                    if 'average_thickness' in df.columns and len(df) > 0:
                        # Calculate image-level average thickness (average of all components in this image)
                        image_avg_thickness = df['average_thickness'].mean()
                        
                        # Add to image-level data
                        image_level_data[condition].append(image_avg_thickness)
                        
                        # Calculate orange pixel percentage for this image
                        if 'thick_percentage' in df.columns:
                            # Calculate weighted average of thick percentages based on component sizes
                            if 'area' in df.columns:
                                # Weight by component area
                                total_area = df['area'].sum()
                                if total_area > 0:
                                    weighted_thick_percentage = (df['thick_percentage'] * df['area']).sum() / total_area
                                else:
                                    weighted_thick_percentage = df['thick_percentage'].mean()
                            else:
                                # Simple average if no area data
                                weighted_thick_percentage = df['thick_percentage'].mean()
                            
                            orange_pixel_data[condition].append(weighted_thick_percentage)
                        
                        # Track batch information for normalization
                        # Extract batch info from the comprehensive mapping
                        try:
                            # Load the comprehensive mapping to get batch info
                            mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
                            file_info = mapping_df[mapping_df['filename'] == f"{base_name}.jpg"]
                            if not file_info.empty:
                                batch = file_info.iloc[0]['biological_replicate']
                                image_level_batch_data[batch][condition].append(image_avg_thickness)
                                if 'thick_percentage' in df.columns:
                                    orange_pixel_batch_data[batch][condition].append(weighted_thick_percentage)
                        except Exception as e:
                            print(f"    Warning: Could not get batch info for {base_name}: {e}")
                            continue
                        
                except Exception as e:
                    print(f"    Warning: Could not process {csv_file}: {e}")
                    continue
    
    # Print summary of image-level data collected
    print(f"\nImage-level data summary:")
    for condition, thicknesses in image_level_data.items():
        print(f"  {condition}: {len(thicknesses)} images")
    
    # Print summary of orange pixel data collected
    print(f"\nOrange pixel percentage data summary:")
    for condition, percentages in orange_pixel_data.items():
        print(f"  {condition}: {len(percentages)} images")
    
    return image_level_data, image_level_batch_data, orange_pixel_data, orange_pixel_batch_data

def generate_image_level_cdf_plots(image_data_dict, output_dir, design_name):
    """Generate CDF plots for image-level data."""
    print(f"\nGenerating image-level CDF plots for {design_name}...")
    
    # Convert defaultdict to regular dict if needed
    if hasattr(image_data_dict, 'default_factory'):
        image_data_dict = dict(image_data_dict)
    
    # Use centralized color scheme
    # Old color_scheme definition removed - now using get_color_for_condition()
    
    # Define logical order for plotting
    plot_order = {
        'genotypes': ['KO', 'ApoE4', 'Control', 'ApoE2'],
        'genotypes_statins': ['KO_Control', 'KO_Simvastatin', 'ApoE4_Control', 'ApoE4_Simvastatin', 'Control_Control', 'Control_Simvastatin', 'ApoE2_Control', 'ApoE2_Simvastatin'],
        'domain_analysis': ['Control', 'ApoE2-NTD', 'ApoE4-NTD', 'ApoE2', 'ApoE4']
    }
    
    # Extract base design name (remove _batch or _image suffix)
    base_design_name = design_name.replace('_batch', '').replace('_image', '')
    
    # Create CDF plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.set_title(f'Component-Level CDF Analysis: {EXPERIMENTAL_DESIGNS[base_design_name]["description"]}\n(Average Blue Component Thickness per Component)', fontsize=14)
    
    # Plot in logical order to control legend order
    current_order = plot_order.get(base_design_name, list(image_data_dict.keys()))
    
    for condition in current_order:
        if condition in image_data_dict and image_data_dict[condition]:
            thicknesses = image_data_dict[condition]
            color = get_color_for_condition(base_design_name, condition)  # Use centralized color scheme
            
            # Use the robust safe_convert_to_numeric function to handle all data types
            thicknesses = safe_convert_to_numeric(thicknesses)
            
            if len(thicknesses) > 0:
                sorted_thicknesses = np.sort(thicknesses)
                cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                
                ax.plot(sorted_thicknesses, cdf, color=color, linewidth=3, 
                       label=f'{condition} (n={len(thicknesses)} components)')
    
    ax.set_xlabel('Average Blue Component Thickness per Component (pixels)')
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'{design_name}_component_level_cdf.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Component-level CDF plot saved: {plot_path}")
    
    # Create Control-only plot if applicable
    if design_name == "genotypes":
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        ax.set_title(f'Component-Level CDF Analysis: {EXPERIMENTAL_DESIGNS[design_name]["description"]}\n(Control Only, Average Blue Component Thickness per Component)', fontsize=14)
        
        # Plot Control conditions in logical order
        control_order = plot_order['genotypes']
        
        for condition in control_order:
            if condition in image_data_dict and image_data_dict[condition]:
                thicknesses = image_data_dict[condition]
                color = get_color_for_condition(base_design_name, condition)  # Use centralized color scheme
                thicknesses = safe_convert_to_numeric(thicknesses)
                
                if len(thicknesses) > 0:
                    sorted_thicknesses = np.sort(thicknesses)
                    cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                    
                    ax.plot(sorted_thicknesses, cdf, color=color, linewidth=3, 
                           label=f'{condition} (n={len(thicknesses)} components)')
        
        ax.set_xlabel('Average Blue Component Thickness per Component (pixels)')
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        # Save Control-only plot
        control_plot_filename = f'{design_name}_component_level_control_only_cdf.png'
        control_plot_path = os.path.join(output_dir, control_plot_filename)
        plt.savefig(control_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Image-level Control-only CDF plot saved: {control_plot_path}")
    
    return plot_path

def generate_image_level_normalized_cdf_plots(image_data_dict, batch_data, output_dir, design_name):
    """Generate normalized CDF plots for image-level data."""
    print(f"\nGenerating image-level normalized CDF plots for {design_name}...")
    
    # Convert defaultdict to regular dict if needed
    if hasattr(image_data_dict, 'default_factory'):
        image_data_dict = dict(image_data_dict)
    
    # Define color scheme (reorganized in logical order)
    color_scheme = {
        'WT_Control': '#66B3FF',      # Light blue
        'WT_Statin': '#0066CC',       # Darker blue
        'ApoE4_Control': '#FFB366',   # Light orange
        'ApoE4_Statin': '#FF8000',    # Darker orange
        'ApoE2_Control': '#90EE90',   # Light green
        'ApoE2_Statin': '#228B22',    # Darker green
        'KO_Control': '#FF6B6B',      # Light red
        'KO_Statin': '#CC0000',       # Darker red
        # Domain analysis colors
        'Control': '#87CEEB',         # Sky blue
        'ApoE2-NTD': '#98FB98',       # Pale green
        'ApoE4-NTD': '#FFA07A',       # Light salmon
        'ApoE-CTD': '#DDA0DD',        # Plum
        'ApoE2': '#32CD32',           # Lime green
        'ApoE4': '#FF6347',           # Tomato red
        # Fragments groups
        'GroupB_Control': '#FF69B4',  # Hot pink
        'GroupJ_Control': '#9370DB',  # Medium purple
        'GroupL_Control': '#20B2AA',  # Light sea green
        'GroupX_Control': '#FFD700',  # Gold
        'GroupZ_Control': '#FF6347',  # Tomato
    }
    
    # Define logical order for plotting
    plot_order = {
        'genotypes': ['KO', 'ApoE4', 'Control', 'ApoE2'],
        'genotypes_statins': ['KO_Control', 'KO_Simvastatin', 'ApoE4_Control', 'ApoE4_Simvastatin', 'Control_Control', 'Control_Simvastatin', 'ApoE2_Control', 'ApoE2_Simvastatin'],
        'domain_analysis': ['Control', 'ApoE2-NTD', 'ApoE4-NTD', 'ApoE2', 'ApoE4']
    }
    
    # Collect normalized thickness data
    normalized_data = defaultdict(list)
    
    if design_name == "genotypes":
        # Each batch (B8, B10, B11, B12) is a separate biological replicate
        for batch in ["B8", "B10", "B11", "B12"]:
            if batch in batch_data:
                print(f"  Processing biological replicate: {batch}")
                
                # Find Control data for this batch (normalize to Control, not WT_Control)
                control_data = batch_data[batch].get("Control", [])
                
                if control_data:
                    control_data = safe_convert_to_numeric(control_data)
                    control_data = control_data[control_data > 0]  # Remove zeros
                    
                    if len(control_data) > 0:
                        normalization_factor = np.mean(control_data)
                        print(f"    Control mean for {batch}: {normalization_factor:.3f}")
                        
                        # Normalize all data for this batch
                        for condition, thickness_values in batch_data[batch].items():
                            if thickness_values:
                                thickness_values = safe_convert_to_numeric(thickness_values)
                                thickness_values = thickness_values[thickness_values > 0]  # Remove zeros
                                
                                if len(thickness_values) > 0:
                                    normalized_thicknesses = thickness_values / normalization_factor
                                    normalized_data[condition].extend(normalized_thicknesses.tolist())
                else:
                    print(f"    Warning: No valid Control data for {batch}")
            else:
                print(f"    Warning: No Control data found for {batch}")
    
    elif design_name == "genotypes_statins":
        # Handle B8 and B10 separately, Group3+Group6 together
        for batch in ["B8", "B10"]:
            if batch in batch_data:
                print(f"  Processing biological replicate: {batch}")
                
                # Find Control data for this batch (ONLY Control, not Statin)
                control_data = batch_data[batch].get("Control", [])
                
                if control_data:
                    control_data = safe_convert_to_numeric(control_data)
                    control_data = control_data[control_data > 0]  # Remove zeros
                    
                    if len(control_data) > 0:
                        normalization_factor = np.mean(control_data)
                        print(f"    Control mean for {batch}: {normalization_factor:.3f}")
                        
                        # Normalize all data for this batch
                        for condition, thickness_values in batch_data[batch].items():
                            if thickness_values:
                                thickness_values = safe_convert_to_numeric(thickness_values)
                                thickness_values = thickness_values[thickness_values > 0]  # Remove zeros
                                
                                if len(thickness_values) > 0:
                                    normalized_thicknesses = thickness_values / normalization_factor
                                    normalized_data[condition].extend(normalized_thicknesses.tolist())
                else:
                    print(f"    Warning: No valid Control data for {batch}")
            else:
                print(f"    Warning: No Control data found for {batch}")
        
        # Handle Group3+Group6 together as one biological replicate
        group3_data = batch_data.get("B27", {})  # Group3 and Group6 data
        if group3_data:
            print(f"  Processing biological replicate: B27 (Group3+Group6)")
            
            # Find Control data for Group3+Group6 (ONLY Control, not Statin)
            control_data = group3_data.get("Control", [])
            
            if control_data:
                control_data = safe_convert_to_numeric(control_data)
                control_data = control_data[control_data > 0]  # Remove zeros
                
                if len(control_data) > 0:
                    normalization_factor = np.mean(control_data)
                    print(f"    Control mean for B27: {normalization_factor:.3f}")
                    
                    # Normalize all data for Group3+Group6
                    for condition, thickness_values in group3_data.items():
                        if thickness_values:
                            thickness_values = safe_convert_to_numeric(thickness_values)
                            thickness_values = thickness_values[thickness_values > 0]  # Remove zeros
                            
                            if len(thickness_values) > 0:
                                normalized_thicknesses = thickness_values / normalization_factor
                                normalized_data[condition].extend(normalized_thicknesses.tolist())
                else:
                    print(f"    Warning: No valid Control data for B27")
            else:
                print(f"    Warning: No Control data found for B27")
    
    elif design_name == "domain_analysis":
        # Each batch (B1, B2, B3) is a separate biological replicate
        for batch in ["B1", "B2", "B3"]:
            if batch in batch_data:
                print(f"  Processing biological replicate: {batch}")
                
                # Find Control data for this batch (normalize to Control, not WT_Control)
                control_data = batch_data[batch].get("Control", [])
                
                if control_data:
                    control_data = safe_convert_to_numeric(control_data)
                    control_data = control_data[control_data > 0]  # Remove zeros
                    
                    if len(control_data) > 0:
                        normalization_factor = np.mean(control_data)
                        print(f"    Control mean for {batch}: {normalization_factor:.3f}")
                        
                        # Normalize all data for this batch
                        for condition, thickness_values in batch_data[batch].items():
                            if thickness_values:
                                thickness_values = safe_convert_to_numeric(thickness_values)
                                thickness_values = thickness_values[thickness_values > 0]  # Remove zeros
                                
                                if len(thickness_values) > 0:
                                    normalized_thicknesses = thickness_values / normalization_factor
                                    normalized_data[condition].extend(normalized_thicknesses.tolist())
                    else:
                        print(f"    Warning: No Control data found for {batch}")
    
    # Create normalized CDF plot
    if normalized_data:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Extract base design name (remove _batch or _image suffix)
        base_design_name = design_name.replace('_batch', '').replace('_image', '')
        
        # Adjust title based on design
        if base_design_name == "domain_analysis":
            title = f'Component-Level Normalized CDF Analysis: {EXPERIMENTAL_DESIGNS[base_design_name]["description"]}\n(Normalized to Control baseline within each biological replicate)'
        else:
            title = f'Component-Level Normalized CDF Analysis: {EXPERIMENTAL_DESIGNS[base_design_name]["description"]}\n(Normalized to Control baseline within each biological replicate)'
        
        ax.set_title(title, fontsize=14)
        
        # Plot in logical order to control legend order
        current_order = plot_order.get(base_design_name, list(normalized_data.keys()))
        
        for condition in current_order:
            if condition in normalized_data and normalized_data[condition]:
                thicknesses = normalized_data[condition]
                color = color_scheme.get(condition, '#808080')  # Default gray if not found
                thicknesses = np.array(thicknesses)
                
                if len(thicknesses) > 0:
                    sorted_thicknesses = np.sort(thicknesses)
                    cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                    
                    ax.plot(sorted_thicknesses, cdf, color=color, linewidth=3, 
                           label=f'{condition} (n={len(thicknesses)} components)')
        
        # Adjust x-axis label based on design
        if base_design_name == "domain_analysis":
            ax.set_xlabel('Normalized Average Blue Component Thickness per Component (Control = 1.0 per biological replicate)')
        else:
            ax.set_xlabel('Normalized Average Blue Component Thickness per Component (Control = 1.0 per biological replicate)')
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits to start where data actually begins
        if normalized_data:
            all_values = []
            for values in normalized_data.values():
                all_values.extend(values)
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                x_margin = (max_val - min_val) * 0.05  # 5% margin
                ax.set_xlim(left=max(0, min_val - x_margin), right=max_val + x_margin)
        
        ax.legend()
        plt.tight_layout()
        
        # Save normalized plot
        normalized_plot_filename = f'{design_name}_component_level_normalized_cdf.png'
        normalized_plot_path = os.path.join(output_dir, normalized_plot_filename)
        plt.savefig(normalized_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Component-level normalized CDF plot saved: {normalized_plot_path}")
        
        # Create Control-only normalized plot if applicable
        if base_design_name == "genotypes":
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            ax.set_title(f'Component-Level Normalized CDF Analysis: {EXPERIMENTAL_DESIGNS[base_design_name]["description"]}\n(Control Only, Normalized to Control)', fontsize=14)
            
            control_color_scheme = {
                'KO': '#FF6B6B',      # Red
                'ApoE4': '#FFB366',   # Orange
                'ApoE2': '#90EE90',   # Green
                'Control': '#66B3FF',      # Blue
            }
            
            # Plot Control conditions in logical order
            control_order = plot_order['genotypes']
            
            for condition in control_order:
                if condition in normalized_data and normalized_data[condition] and condition in control_color_scheme:
                    thicknesses = normalized_data[condition]
                    color = control_color_scheme[condition]
                    thicknesses = np.array(thicknesses)
                    
                    if len(thicknesses) > 0:
                        sorted_thicknesses = np.sort(thicknesses)
                        cdf = np.arange(1, len(sorted_thicknesses) + 1) / len(sorted_thicknesses)
                        
                        ax.plot(sorted_thicknesses, cdf, color=color, linewidth=3, 
                               label=f'{condition} (n={len(thicknesses)} components)')
            
            # Adjust x-axis label based on design
            if base_design_name == "domain_analysis":
                ax.set_xlabel('Normalized Average Blue Component Thickness per Component (Control = 1.0 per biological replicate)')
            else:
                ax.set_xlabel('Normalized Average Blue Component Thickness per Component (Control = 1.0)')
            ax.set_ylabel('Cumulative Probability')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits to start where data actually begins for control plot
            control_values = []
            for condition in control_order:
                if condition in normalized_data and normalized_data[condition]:
                    control_values.extend(normalized_data[condition])
            if control_values:
                min_val = min(control_values)
                max_val = max(control_values)
                x_margin = (max_val - min_val) * 0.05  # 5% margin
                ax.set_xlim(left=max(0, min_val - x_margin), right=max_val + x_margin)
            
            ax.legend()
            plt.tight_layout()
            
            # Save Control-only normalized plot
            control_normalized_plot_filename = f'{design_name}_component_level_control_only_normalized_cdf.png'
            control_normalized_plot_path = os.path.join(output_dir, control_normalized_plot_filename)
            plt.savefig(control_normalized_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Component-level Control-only normalized CDF plot saved: {control_normalized_plot_path}")
    else:
        print("  Warning: No normalized data available for plotting")
    
    return normalized_data

def generate_image_vs_batch_threshold_comparison(component_data_dict_image, component_data_dict_batch, output_dir, design_name):
    """Generate comparison plots showing deviations between image-thresholded and batch-thresholded analysis."""
    print(f"\nGenerating image vs batch threshold comparison plots for {design_name}...")
    
    # Define condition colors for plotting
    condition_colors = {
        # genotype conditions
        'KO': 'red',
        'ApoE4': 'blue', 
        'ApoE2': 'green',
        'Control': 'orange',
        # genotypes_statins conditions
        'KO_Control': 'red',
        'KO_Simvastatin': 'darkred',
        'ApoE4_Control': 'blue',
        'ApoE4_Simvastatin': 'darkblue',
        'ApoE2_Control': 'green',
        'ApoE2_Simvastatin': 'darkgreen',
        'Control_Control': 'orange',
        'Control_Simvastatin': 'darkorange',
        # domain_analysis conditions (5 total: ApoE2, ApoE2-NTD, ApoE4, ApoE4-NTD, Control)
        'ApoE2-NTD': 'lightgreen',
        'ApoE4-NTD': 'lightblue',
        # Note: ApoE2, ApoE4, Control are already defined above for genotypes
    }
    
    # Get all unique conditions
    all_conditions = set(component_data_dict_image.keys()) | set(component_data_dict_batch.keys())
    conditions = sorted(list(all_conditions))
    
    if not conditions:
        print(f"  No data found for {design_name}")
        return
    
    # Load the comprehensive mapping to get biological replicate info
    try:
        mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
        design_files = mapping_df[mapping_df['experimental_design'] == design_name]
        
        # Get all unique biological replicates for this design
        biological_replicates = sorted(design_files['biological_replicate'].unique())
    except Exception as e:
        print(f"  Warning: Could not load comprehensive mapping: {e}")
        return
    
    if not biological_replicates:
        print(f"  No biological replicates found for {design_name}")
        return
    
    # Collect data organized by biological replicate and condition - calculate deviations
    replicate_data = defaultdict(lambda: defaultdict(list))  # biological_replicate -> condition -> deviations
    all_deviations = []
    
    # Process each condition to calculate image vs batch deviations per image
    for condition in conditions:
        # Find all CSV files for this condition (check both batch and individual directories)
        # For genotypes, directory names have "_Control" suffix, but for genotypes_statins and domain_analysis they don't
        if design_name == 'genotypes':
            condition_dir_batch = os.path.join(output_dir, design_name, condition, "csv_files_batch")
            condition_dir_individual = os.path.join(output_dir, design_name, condition, "csv_files_individual")
        else:
            condition_dir_batch = os.path.join(output_dir, design_name, condition, "csv_files_batch")
            condition_dir_individual = os.path.join(output_dir, design_name, condition, "csv_files_individual")
        
        # Collect image-level data
        image_data = {}  # base_name -> thickness
        batch_data = {}  # base_name -> thickness
        
        # Process image-thresholded data
        if os.path.exists(condition_dir_individual):
            blue_csv_files = glob.glob(os.path.join(condition_dir_individual, "*_blue_components.csv"))
            
            for csv_file in blue_csv_files:
                try:
                    # Extract image name from CSV filename
                    base_name = os.path.basename(csv_file).replace('_blue_components.csv', '')
                    
                    # Read the CSV file
                    df = pd.read_csv(csv_file)
                    
                    if 'average_thickness' in df.columns and len(df) > 0:
                        # Calculate image-level average thickness (average of all components in this image)
                        image_avg_thickness = df['average_thickness'].mean()
                        image_data[base_name] = image_avg_thickness
                        
                except Exception as e:
                    print(f"    Warning: Could not process {csv_file}: {e}")
                    continue
        
        # Process batch-thresholded data
        if os.path.exists(condition_dir_batch):
            blue_csv_files = glob.glob(os.path.join(condition_dir_batch, "*_blue_components.csv"))
            
            for csv_file in blue_csv_files:
                try:
                    # Extract image name from CSV filename
                    base_name = os.path.basename(csv_file).replace('_blue_components.csv', '')
                    
                    # Read the CSV file
                    df = pd.read_csv(csv_file)
                    
                    if 'average_thickness' in df.columns and len(df) > 0:
                        # Calculate image-level average thickness (average of all components in this image)
                        batch_avg_thickness = df['average_thickness'].mean()
                        batch_data[base_name] = batch_avg_thickness
                        
                except Exception as e:
                    print(f"    Warning: Could not process {csv_file}: {e}")
                    continue
        
        # Calculate deviations for matching images (batch - image)
        for base_name in image_data:
            if base_name in batch_data:
                deviation = batch_data[base_name] - image_data[base_name]
                
                # Get biological replicate info
                file_info = mapping_df[mapping_df['filename'] == f"{base_name}.jpg"]
                if file_info.empty:
                    file_info = mapping_df[mapping_df['filename'] == base_name]
                
                if not file_info.empty:
                    batch = file_info.iloc[0]['biological_replicate']
                    replicate_data[batch][condition].append(deviation)
                    all_deviations.append(deviation)
    
    if not replicate_data:
        print(f"  No valid deviation data found for {design_name}")
        return
    
    if not all_deviations:
        print(f"  No deviations calculated for {design_name}")
        return
    
    # Create figure with horizontal layout: individual replicates + overall comparison
    num_replicates = len(biological_replicates)
    fig, axes = plt.subplots(1, num_replicates + 1, figsize=(6 * (num_replicates + 1), 8))
    
    # Ensure axes is always a list
    if num_replicates == 0:
        print(f"  No biological replicates data found")
        return
    elif num_replicates == 1:
        axes = [axes] if not isinstance(axes, list) else axes
    
    # Find global y-axis limits for consistent scaling
    y_min = min(all_deviations) - 0.01
    y_max = max(all_deviations) + 0.01
    
    # Plot individual biological replicate deviations
    for i, biological_replicate in enumerate(biological_replicates):
        ax = axes[i]
        
        x_positions = np.arange(len(conditions))
        bar_width = 0.6
        
        for j, condition in enumerate(conditions):
            if condition in replicate_data[biological_replicate]:
                values = replicate_data[biological_replicate][condition]
                if values:  # Only plot if there are values
                    color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
                    bar = ax.bar(x_positions[j], np.mean(values), bar_width, 
                               color=color, alpha=0.4, 
                               label=condition, yerr=np.std(values) if len(values) > 1 else 0)
                    
                    # Add individual data points as small dots
                    for k, value in enumerate(values):
                        ax.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                                 color=color, alpha=0.9, s=25, zorder=5, 
                                 edgecolors='black', linewidth=0.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Biological Condition')
        ax.set_ylabel('Deviation (Batch - Image Thickness)')
        ax.set_title(f'{biological_replicate} - Batch vs Image Deviations')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        
        # Only show legend on first plot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot overall deviations (last subplot)
    ax = axes[-1]
    
    for j, condition in enumerate(conditions):
        replicate_means = []
        for biological_replicate in biological_replicates:
            if condition in replicate_data[biological_replicate] and replicate_data[biological_replicate][condition]:
                replicate_means.append(np.mean(replicate_data[biological_replicate][condition]))
        
        if replicate_means:
            color = get_color_for_condition(design_name, condition)  # Use centralized color scheme
            bar = ax.bar(x_positions[j], np.mean(replicate_means), bar_width, 
                       color=color, alpha=0.4, 
                       label=condition, yerr=np.std(replicate_means) if len(replicate_means) > 1 else 0)
            
            # Add individual replicate means as small dots
            for k, value in enumerate(replicate_means):
                ax.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                         color=color, alpha=0.9, s=25, zorder=5, 
                         edgecolors='black', linewidth=0.5)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Biological Condition')
    ax.set_ylabel('Average Deviation (Batch - Image Thickness)')
    ax.set_title('Overall Batch vs Image Deviations')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comparison plot in the correct directory
    design_output_dir = os.path.join(output_dir, design_name)
    plot_filename = f'{design_name}_image_vs_batch_threshold_comparison.png'
    plot_path = os.path.join(design_output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Image vs Batch threshold comparison plot saved: {plot_path}")
    
    # Save comparison data to CSV in the correct directory
    comparison_data = []
    for condition in conditions:
        row = {'condition': condition}
        
        if condition in component_data_dict_image and component_data_dict_image[condition]:
            image_data = component_data_dict_image[condition]
            row.update({
                'image_thresholded_mean': np.mean(image_data),
                'image_thresholded_std': np.std(image_data),
                'image_thresholded_count': len(image_data)
            })
        else:
            row.update({
                'image_thresholded_mean': None,
                'image_thresholded_std': None,
                'image_thresholded_count': 0
            })
        
        if condition in component_data_dict_batch and component_data_dict_batch[condition]:
            batch_data = component_data_dict_batch[condition]
            row.update({
                'batch_thresholded_mean': np.mean(batch_data),
                'batch_thresholded_std': np.std(batch_data),
                'batch_thresholded_count': len(batch_data)
            })
        else:
            row.update({
                'batch_thresholded_mean': None,
                'batch_thresholded_std': None,
                'batch_thresholded_count': 0
            })
        
        if row['image_thresholded_mean'] is not None and row['batch_thresholded_mean'] is not None:
            row['difference'] = row['batch_thresholded_mean'] - row['image_thresholded_mean']
            row['percent_difference'] = (row['difference'] / row['image_thresholded_mean']) * 100
        else:
            row['difference'] = None
            row['percent_difference'] = None
        
        comparison_data.append(row)
    
    # Save to CSV in the correct directory
    csv_filename = f'{design_name}_image_vs_batch_threshold_comparison.csv'
    csv_path = os.path.join(design_output_dir, csv_filename)
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(csv_path, index=False)
    print(f"  Image vs Batch threshold comparison data saved: {csv_path}")


def generate_biological_replicate_cdf_plots(data_dict, output_dir, design_name, threshold_type="individual"):
    """
    Generate CDF plots showing individual lines for every biological replicate within each condition.
    Colors are similar within conditions but with slight variations between replicates.
    Reconstructs data from CSV files since component-level data is provided.
    
    Args:
        data_dict: Dictionary containing component data organized by condition
        output_dir: Output directory for plots
        design_name: Name of experimental design (genotypes, genotypes_statins, domain_analysis)
        threshold_type: Either "individual" or "batch" for labeling
    """
    # Create design-specific output directory
    design_output_dir = os.path.join(output_dir, design_name)
    os.makedirs(design_output_dir, exist_ok=True)
    
    # Load the comprehensive file mapping to get biological replicate info
    mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
    design_mapping = mapping_df[mapping_df['experimental_design'] == design_name]
    
    # Base colors for each condition
    base_condition_colors = {
        'genotypes': {
            'KO': [1.0, 0.42, 0.42],      # Red base
            'ApoE4': [0.31, 0.80, 0.77],     # Teal base  
            'Control': [0.27, 0.72, 0.82],   # Blue base
            'ApoE2': [0.59, 0.81, 0.71]         # Green base
        },
        'genotypes_statins': {
            'KO_Control': [1.0, 0.42, 0.42],
            'KO_Simvastatin': [1.0, 0.56, 0.56],
            'ApoE4_Control': [0.31, 0.80, 0.77],
            'ApoE4_Simvastatin': [0.49, 0.83, 0.83],
            'Control_Control': [0.27, 0.72, 0.82],
            'Control_Simvastatin': [0.42, 0.77, 0.88],
            'ApoE2_Control': [0.59, 0.81, 0.71],
            'ApoE2_Simvastatin': [0.70, 0.85, 0.78]
        },
        'domain_analysis': {
            'Control': [1.0, 0.42, 0.42],
            'ApoE2-NTD': [1.0, 0.56, 0.56],
            'ApoE4-NTD': [0.31, 0.80, 0.77],
            'ApoE4': [0.49, 0.83, 0.83],
            'ApoE2': [0.27, 0.72, 0.82]
        }
    }
    
    colors = base_condition_colors.get(design_name, {})
    
    # Organize data by biological replicate from CSV files
    bio_replicate_data = {}
    
        # Process CSV files to reconstruct biological replicate data
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
                    
                    # Initialize condition and replicate in bio_replicate_data if needed
                    if condition not in bio_replicate_data:
                        bio_replicate_data[condition] = {}
                    if biological_replicate not in bio_replicate_data[condition]:
                        bio_replicate_data[condition][biological_replicate] = []
                    
                    # Add thickness values for this image
                    bio_replicate_data[condition][biological_replicate].extend(thickness_values)
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
    print(f"  Bio replicate data structure: {list(bio_replicate_data.keys()) if bio_replicate_data else 'Empty'}")
    
    if not bio_replicate_data:
        print(f"  No valid biological replicate data found for {design_name}")
        return
        
    # Get all conditions and biological replicates
    conditions = sorted(list(bio_replicate_data.keys()))
    all_replicates = set()
    for condition_data in bio_replicate_data.values():
        all_replicates.update(condition_data.keys())
    biological_replicates = sorted(list(all_replicates))
    
    # Create the plot - 2x2 layout for 4 biological replicates
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Function to generate color variations for biological replicates
    def generate_replicate_colors(base_color, num_replicates):
        """Generate slight color variations for biological replicates"""
        replicate_colors = []
        for i in range(num_replicates):
            # Create variations by adjusting brightness and saturation
            variation_factor = 0.15  # Amount of variation
            color_variation = [
                max(0, min(1, base_color[0] + (i - num_replicates/2) * variation_factor / num_replicates)),
                max(0, min(1, base_color[1] + (i - num_replicates/2) * variation_factor / num_replicates)),  
                max(0, min(1, base_color[2] + (i - num_replicates/2) * variation_factor / num_replicates))
            ]
            replicate_colors.append(color_variation)
        return replicate_colors
    
    # Individual biological replicate plots
    for i, bio_rep in enumerate(biological_replicates[:4]):
        ax = axes[i]
        
        print(f"  Plotting biological replicate {bio_rep}")
        
        # Plot CDF for each condition within this biological replicate
        for condition in conditions:
            if condition in bio_replicate_data and bio_rep in bio_replicate_data[condition]:
                thickness_data = bio_replicate_data[condition][bio_rep]
                
                if thickness_data:
                    # Convert to numeric and remove invalid values
                    numeric_data = safe_convert_to_numeric(thickness_data)
                    
                    if len(numeric_data) > 0:
                        # Sort data for CDF
                        sorted_data = np.sort(numeric_data)
                        
                        # Calculate CDF
                        n = len(sorted_data)
                        y_values = np.arange(1, n + 1) / n
                        
                        # Get color for this condition
                        base_color = colors.get(condition, [0.4, 0.4, 0.4])
                        
                        print(f"    Plotting condition {condition}: {len(numeric_data)} data points")
                        
                        # Plot CDF line for this condition
                        ax.plot(sorted_data, y_values, 
                               color=base_color, 
                               linewidth=2, 
                               alpha=0.8,
                               label=f'{condition}')
                    else:
                        print(f"    Warning: No valid data for condition {condition} in bio replicate {bio_rep}")
                else:
                    print(f"    Warning: No thickness data for condition {condition} in bio replicate {bio_rep}")
        
        ax.set_xlabel('Average Thickness (pixels)', fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_title(f'Biological Replicate {bio_rep}\n({threshold_type.title()} Thresholding)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Only add legend if there are any lines plotted
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc='best')
    
    # No combined plot needed in 2x2 layout - all 4 subplots used for individual replicates
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'{design_name}_biological_replicate_cdf_{threshold_type}.png'
    plot_path = os.path.join(design_output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Biological replicate CDF plot saved: {plot_path}")
    
    # Save replicate data to CSV
    csv_data = []
    for condition in conditions:
        if condition in bio_replicate_data:
            for bio_rep, thickness_data in bio_replicate_data[condition].items():
                numeric_data = safe_convert_to_numeric(thickness_data)
                for thickness in numeric_data:
                    csv_data.append({
                        'condition': condition,
                        'biological_replicate': bio_rep,
                        'thickness': thickness,
                        'threshold_type': threshold_type
                    })
    
    if csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv_filename = f'{design_name}_biological_replicate_cdf_{threshold_type}_data.csv'
        csv_path = os.path.join(design_output_dir, csv_filename)
        csv_df.to_csv(csv_path, index=False)
        print(f"Biological replicate CDF data saved: {csv_path}")

def generate_thickness_deviation_plots(component_data_dict_image, component_data_dict_batch, output_dir, design_name):
    """
    Generate thickness deviation plots showing the difference between image-level and batch-level average thicknesses.
    Structure exactly matches combined_deviation_comparison.png but for thickness data.
    Reconstructs image-level data from existing CSV files.
    
    Args:
        component_data_dict_image: Image-level thresholded component data (not used, reconstructed from CSVs)
        component_data_dict_batch: Batch-level thresholded component data (not used, reconstructed from CSVs)
        output_dir: Output directory for plots
        design_name: Name of experimental design
    """
    print(f"\nGenerating thickness deviation plots for {design_name}...")
    
    # Load the comprehensive file mapping to get biological replicate info
    mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
    design_mapping = mapping_df[mapping_df['experimental_design'] == design_name]
    
    # Collect all deviation data organized by biological replicate
    replicate_data = defaultdict(lambda: defaultdict(list))  # biological_replicate -> condition -> deviations
    all_deviations = []
    
    # Process CSV files to reconstruct image-level thickness data
    design_output_dir = os.path.join("comprehensive_cdf_analysis_results", design_name)
    
    for _, row in design_mapping.iterrows():
        filename = row['filename']
        condition = row['condition']
        biological_replicate = row['biological_replicate']
        
        # Extract base filename (remove .jpg extension)
        base_filename = os.path.splitext(filename)[0]
        
        # Construct paths to batch and individual CSV files
        batch_csv_path = os.path.join(design_output_dir, condition, "csv_files_batch", f"{base_filename}_blue_components.csv")
        individual_csv_path = os.path.join(design_output_dir, condition, "csv_files_individual", f"{base_filename}_blue_components.csv")
        
        # Check if both CSV files exist
        if os.path.exists(batch_csv_path) and os.path.exists(individual_csv_path):
            try:
                # Read batch-thresholded data
                batch_df = pd.read_csv(batch_csv_path)
                if not batch_df.empty and 'average_thickness' in batch_df.columns:
                    batch_avg_thickness = batch_df['average_thickness'].mean()
                else:
                    batch_avg_thickness = 0
                
                # Read individual-thresholded data
                individual_df = pd.read_csv(individual_csv_path)
                if not individual_df.empty and 'average_thickness' in individual_df.columns:
                    individual_avg_thickness = individual_df['average_thickness'].mean()
                else:
                    individual_avg_thickness = 0
                
                # Calculate thickness deviation
                thickness_deviation = individual_avg_thickness - batch_avg_thickness
                
                # Store the deviation
                replicate_data[biological_replicate][condition].append(thickness_deviation)
                all_deviations.append(thickness_deviation)
                
            except Exception as e:
                print(f"    Warning: Error processing {base_filename}: {e}")
                continue
    
    if not replicate_data:
        print(f"  No valid thickness deviation data found for {design_name}")
        return
    
    # Get all unique biological replicates and conditions
    biological_replicates = sorted(list(replicate_data.keys()))
    all_conditions = set()
    for replicate_data_dict in replicate_data.values():
        all_conditions.update(replicate_data_dict.keys())
    conditions = sorted(list(all_conditions))
    
    # Create figure with 1 row and num_replicates + 1 columns (individual replicates + replicate level)
    num_replicates = len(biological_replicates)
    fig, axes = plt.subplots(1, num_replicates + 1, figsize=(6 * (num_replicates + 1), 8))
    
    # Handle case where there's only one replicate (axes won't be an array)
    if num_replicates == 0:
        print(f"  No biological replicates found for {design_name}")
        return
    elif num_replicates == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    
    # Find global y-axis limits
    if all_deviations:
        y_min = min(all_deviations) - 0.1
        y_max = max(all_deviations) + 0.1
    else:
        y_min, y_max = -1, 1
    
    # Plot individual biological replicate deviations
    for i, biological_replicate in enumerate(biological_replicates):
        ax = axes[i]
        
        x_positions = np.arange(len(conditions))
        bar_width = 0.6
        
        for j, condition in enumerate(conditions):
            if condition in replicate_data[biological_replicate]:
                values = replicate_data[biological_replicate][condition]
                if values:  # Only plot if there are values
                    color = get_color_for_condition(design_name, condition)
                    bar = ax.bar(x_positions[j], np.mean(values), bar_width, 
                               color=color, alpha=0.4, 
                               label=condition, yerr=np.std(values) if len(values) > 1 else 0)
                    
                    # Add individual data points as small dots
                    for k, value in enumerate(values):
                        ax.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                                 color=color, alpha=0.9, s=25, zorder=5, 
                                 edgecolors='black', linewidth=0.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Biological Condition')
        ax.set_ylabel('Thickness Deviation (Individual - Batch)')
        ax.set_title(f'{biological_replicate} - Individual Deviations')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        
        # Only show legend on first plot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot replicate-level deviations (last subplot) - same structure as combined deviation
    ax = axes[-1]
    
    x_positions = np.arange(len(conditions))
    bar_width = 0.6
    
    for j, condition in enumerate(conditions):
        replicate_means = []
        for biological_replicate in biological_replicates:
            if condition in replicate_data[biological_replicate] and replicate_data[biological_replicate][condition]:
                replicate_means.append(np.mean(replicate_data[biological_replicate][condition]))
        
        if replicate_means:
            color = get_color_for_condition(design_name, condition)
            bar = ax.bar(x_positions[j], np.mean(replicate_means), bar_width, 
                       color=color, alpha=0.4, 
                       label=condition, yerr=np.std(replicate_means) if len(replicate_means) > 1 else 0)
            
            # Add individual replicate means as small dots
            for k, value in enumerate(replicate_means):
                ax.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                         color=color, alpha=0.9, s=25, zorder=5, 
                         edgecolors='black', linewidth=0.5)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Biological Condition')
    ax.set_ylabel('Average Thickness Deviation (Individual - Batch)')
    ax.set_title('Replicate-Level Deviations')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'{design_name}_thickness_deviation_comparison.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Thickness deviation plot saved: {plot_path}")
    
    # Save the data as CSV
    csv_filename = f'{design_name}_thickness_deviation_data.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Create CSV data
    csv_data = []
    for biological_replicate in biological_replicates:
        for condition in conditions:
            if condition in replicate_data[biological_replicate]:
                values = replicate_data[biological_replicate][condition]
                for value in values:
                    csv_data.append({
                        'biological_replicate': biological_replicate,
                        'condition': condition,
                        'thickness_deviation': value
                    })
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_path, index=False)
    print(f"  Thickness deviation data saved: {csv_path}")
    print(f"  Successfully reconstructed thickness deviation data from {len(csv_data)} images")


def generate_component_count_deviation_plots(component_data_dict_image, component_data_dict_batch, output_dir, design_name):
    """
    Generate component count deviation plots showing the difference between image-level and batch-level component counts.
    Structure exactly matches thickness_deviation_comparison.png but for component count data.
    
    Args:
        component_data_dict_image: Image-level thresholded component data
        component_data_dict_batch: Batch-level thresholded component data  
        output_dir: Output directory for plots
        design_name: Name of experimental design
    """
    print(f"\nGenerating component count deviation plots for {design_name}...")
    
    # Load the comprehensive file mapping to get biological replicate info
    mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
    design_mapping = mapping_df[mapping_df['experimental_design'] == design_name]
    
    # Collect all deviation data organized by biological replicate
    replicate_data = defaultdict(lambda: defaultdict(list))  # biological_replicate -> condition -> deviations
    all_deviations = []
    
    # Process CSV files to reconstruct component count data
    design_output_dir_csv = os.path.join("comprehensive_cdf_analysis_results", design_name)
    
    for _, row in design_mapping.iterrows():
        filename = row['filename']
        condition = row['condition']
        biological_replicate = row['biological_replicate']
        
        # Extract base filename (remove .jpg extension)
        base_filename = os.path.splitext(filename)[0]
        
        # Check both individual and batch CSV files
        individual_csv_path = os.path.join(design_output_dir_csv, condition, "csv_files_individual", f"{base_filename}_blue_components.csv")
        batch_csv_path = os.path.join(design_output_dir_csv, condition, "csv_files_batch", f"{base_filename}_blue_components.csv")
        
        individual_count = 0
        batch_count = 0
        
        # Count components from individual threshold CSV
        if os.path.exists(individual_csv_path):
            try:
                df_individual = pd.read_csv(individual_csv_path)
                individual_count = len(df_individual) if not df_individual.empty else 0
            except Exception as e:
                print(f"    Warning: Error reading individual CSV {base_filename}: {e}")
        
        # Count components from batch threshold CSV
        if os.path.exists(batch_csv_path):
            try:
                df_batch = pd.read_csv(batch_csv_path)
                batch_count = len(df_batch) if not df_batch.empty else 0
            except Exception as e:
                print(f"    Warning: Error reading batch CSV {base_filename}: {e}")
        
        # Calculate component count deviation
        if individual_count > 0 or batch_count > 0:
            count_deviation = individual_count - batch_count
            
            # Store the deviation
            replicate_data[biological_replicate][condition].append(count_deviation)
            all_deviations.append(count_deviation)
    
    if not replicate_data:
        print(f"  No valid component count deviation data found for {design_name}")
        return
    
    # Get all unique biological replicates and conditions
    biological_replicates = sorted(list(replicate_data.keys()))
    all_conditions = set()
    for replicate_data_dict in replicate_data.values():
        all_conditions.update(replicate_data_dict.keys())
    conditions = sorted(list(all_conditions))
    
    if not conditions:
        print(f"  No conditions found for {design_name}")
        return
    
    # Create figure with 1 row and num_replicates + 1 columns (individual replicates + replicate level)
    num_replicates = len(biological_replicates)
    fig, axes = plt.subplots(1, num_replicates + 1, figsize=(6 * (num_replicates + 1), 8))
    
    # Handle case where there's only one replicate (axes won't be an array)
    if num_replicates == 0:
        print(f"  No biological replicates found for {design_name}")
        return
    elif num_replicates == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    
    # Find global y-axis limits
    if all_deviations:
        y_min = min(all_deviations) - 0.1
        y_max = max(all_deviations) + 0.1
    else:
        y_min, y_max = -1, 1
    
    # Plot individual biological replicate deviations
    for i, biological_replicate in enumerate(biological_replicates):
        ax = axes[i]
        
        x_positions = np.arange(len(conditions))
        bar_width = 0.6
        
        for j, condition in enumerate(conditions):
            if condition in replicate_data[biological_replicate]:
                values = replicate_data[biological_replicate][condition]
                if values:  # Only plot if there are values
                    color = get_color_for_condition(design_name, condition)
                    bar = ax.bar(x_positions[j], np.mean(values), bar_width, 
                               color=color, alpha=0.4, 
                               label=condition, yerr=np.std(values) if len(values) > 1 else 0)
                    
                    # Add individual data points as small dots
                    for k, value in enumerate(values):
                        ax.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                                 color=color, alpha=0.9, s=25, zorder=5, 
                                 edgecolors='black', linewidth=0.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Biological Condition')
        ax.set_ylabel('Component Count Deviation (Individual - Batch)')
        ax.set_title(f'{biological_replicate} - Individual Deviations')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        
        # Only show legend on first plot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot replicate-level deviations (last subplot) - same structure as thickness deviation
    ax = axes[-1]
    
    x_positions = np.arange(len(conditions))
    bar_width = 0.6
    
    for j, condition in enumerate(conditions):
        replicate_means = []
        for biological_replicate in biological_replicates:
            if condition in replicate_data[biological_replicate] and replicate_data[biological_replicate][condition]:
                replicate_means.append(np.mean(replicate_data[biological_replicate][condition]))
        
        if replicate_means:
            color = get_color_for_condition(design_name, condition)
            bar = ax.bar(x_positions[j], np.mean(replicate_means), bar_width, 
                       color=color, alpha=0.4, 
                       label=condition, yerr=np.std(replicate_means) if len(replicate_means) > 1 else 0)
            
            # Add individual replicate means as small dots
            for k, value in enumerate(replicate_means):
                ax.scatter(x_positions[j] + np.random.normal(0, 0.02), value, 
                         color=color, alpha=0.9, s=25, zorder=5, 
                         edgecolors='black', linewidth=0.5)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Biological Condition')
    ax.set_ylabel('Average Component Count Deviation (Individual - Batch)')
    ax.set_title('Replicate-Level Deviations')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'{design_name}_component_count_deviation_comparison.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Component count deviation plot saved: {plot_path}")
    
    # Save the data as CSV
    csv_filename = f'{design_name}_component_count_deviation_data.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Create CSV data
    csv_data = []
    for biological_replicate in biological_replicates:
        for condition in conditions:
            if condition in replicate_data[biological_replicate]:
                values = replicate_data[biological_replicate][condition]
                for value in values:
                    csv_data.append({
                        'biological_replicate': biological_replicate,
                        'condition': condition,
                        'component_count_deviation': value
                    })
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_path, index=False)
    print(f"  Component count deviation data saved: {csv_path}")
    print(f"  Successfully reconstructed component count deviation data from {len(csv_data)} images")


def generate_thickness_vs_deviation_scatter_plots(component_data_dict_image, component_data_dict_batch, output_dir, design_name):
    """
    Generate scatter plots showing thickness change vs threshold deviation for each image.
    Each image is a data point, color-coded by condition.
    
    Args:
        component_data_dict_image: Image-level thresholded component data
        component_data_dict_batch: Batch-level thresholded component data
        output_dir: Output directory for plots
        design_name: Name of experimental design
    """
    print(f"\nGenerating thickness vs deviation scatter plots for {design_name}...")
    
    # Load the comprehensive file mapping to get biological replicate info
    mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
    design_mapping = mapping_df[mapping_df['experimental_design'] == design_name]
    
    # Collect data for scatter plot
    scatter_data = []
    
    # Process CSV files to reconstruct image-level data
    design_output_dir = os.path.join("comprehensive_cdf_analysis_results", design_name)
    
    for _, row in design_mapping.iterrows():
        filename = row['filename']
        condition = row['condition']
        biological_replicate = row['biological_replicate']
        
        # Extract base filename (remove .jpg extension)
        base_filename = os.path.splitext(filename)[0]
        
        # Construct paths to batch and individual CSV files
        batch_csv_path = os.path.join(design_output_dir, condition, "csv_files_batch", f"{base_filename}_blue_components.csv")
        individual_csv_path = os.path.join(design_output_dir, condition, "csv_files_individual", f"{base_filename}_blue_components.csv")
        
        # Check if both CSV files exist
        if os.path.exists(batch_csv_path) and os.path.exists(individual_csv_path):
            try:
                # Read batch-thresholded data
                batch_df = pd.read_csv(batch_csv_path)
                if not batch_df.empty and 'average_thickness' in batch_df.columns:
                    batch_avg_thickness = batch_df['average_thickness'].mean()
                else:
                    batch_avg_thickness = 0
                
                # Read individual-thresholded data
                individual_df = pd.read_csv(individual_csv_path)
                if not individual_df.empty and 'average_thickness' in individual_df.columns:
                    individual_avg_thickness = individual_df['average_thickness'].mean()
                else:
                    individual_avg_thickness = 0
                
                # Calculate thickness change (individual - batch)
                thickness_change = individual_avg_thickness - batch_avg_thickness
                
                # Calculate component count change (individual - batch)
                individual_count = len(individual_df) if not individual_df.empty else 0
                batch_count = len(batch_df) if not batch_df.empty else 0
                component_count_change = individual_count - batch_count
                
                # Use component count change as threshold deviation proxy
                # This makes sense because different thresholds detect different numbers of components
                threshold_deviation = component_count_change
                
                # Store the data point
                scatter_data.append({
                    'filename': filename,
                    'condition': condition,
                    'biological_replicate': biological_replicate,
                    'threshold_deviation': threshold_deviation,
                    'thickness_change': thickness_change
                })
                
            except Exception as e:
                print(f"    Warning: Error processing {base_filename}: {e}")
                continue
    
    if not scatter_data:
        print(f"  No valid scatter plot data found for {design_name}")
        return
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group data by condition for color coding
    conditions = sorted(list(set([d['condition'] for d in scatter_data])))
    
    # Plot scatter points for each condition (different colors)
    for condition in conditions:
        condition_data = [d for d in scatter_data if d['condition'] == condition]
        x_values = [d['threshold_deviation'] for d in condition_data]
        y_values = [d['thickness_change'] for d in condition_data]
        
        color = get_color_for_condition(design_name, condition)
        
        # Plot scatter points
        ax.scatter(x_values, y_values, c=color, alpha=0.7, s=50, label=condition)
    
    # Add single linear regression line for ALL data points
    all_x_values = [d['threshold_deviation'] for d in scatter_data]
    all_y_values = [d['thickness_change'] for d in scatter_data]
    
    if len(all_x_values) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(all_x_values, all_y_values)
        line_x = np.array([min(all_x_values), max(all_x_values)])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color='black', alpha=0.8, linestyle='-', linewidth=2, label='Overall Fit')
        
        # Add overall R-squared value
        ax.text(0.02, 0.02, f'Overall R² = {r_value**2:.3f}', 
               transform=ax.transAxes, fontsize=12, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    ax.set_xlabel('Threshold Deviation')
    ax.set_ylabel('Thickness Change (Individual - Batch)')
    ax.set_title(f'{design_name} - Thickness Change vs Threshold Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_filename = f'{design_name}_thickness_vs_deviation_scatter.png'
    plot_path = os.path.join(output_dir, design_name, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data to CSV
    csv_filename = f'{design_name}_thickness_vs_deviation_data.csv'
    csv_path = os.path.join(output_dir, design_name, csv_filename)
    scatter_df = pd.DataFrame(scatter_data)
    scatter_df.to_csv(csv_path, index=False)
    
    print(f"  Thickness vs deviation scatter plot saved: {plot_filename}")
    print(f"  Thickness vs deviation data saved: {csv_filename}")


def generate_component_count_vs_deviation_scatter_plots(component_data_dict_image, component_data_dict_batch, output_dir, design_name):
    """
    Generate scatter plots showing component count change vs threshold deviation for each image.
    Each image is a data point, color-coded by condition.
    
    Args:
        component_data_dict_image: Image-level thresholded component data
        component_data_dict_batch: Batch-level thresholded component data
        output_dir: Output directory for plots
        design_name: Name of experimental design
    """
    print(f"\nGenerating component count vs deviation scatter plots for {design_name}...")
    
    # Load the comprehensive file mapping to get biological replicate info
    mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
    design_mapping = mapping_df[mapping_df['experimental_design'] == design_name]
    
    # Collect data for scatter plot
    scatter_data = []
    
    # Process CSV files to reconstruct image-level data
    design_output_dir = os.path.join("comprehensive_cdf_analysis_results", design_name)
    
    for _, row in design_mapping.iterrows():
        filename = row['filename']
        condition = row['condition']
        biological_replicate = row['biological_replicate']
        
        # Extract base filename (remove .jpg extension)
        base_filename = os.path.splitext(filename)[0]
        
        # Construct paths to batch and individual CSV files
        batch_csv_path = os.path.join(design_output_dir, condition, "csv_files_batch", f"{base_filename}_blue_components.csv")
        individual_csv_path = os.path.join(design_output_dir, condition, "csv_files_individual", f"{base_filename}_blue_components.csv")
        
        # Check if both CSV files exist
        if os.path.exists(batch_csv_path) and os.path.exists(individual_csv_path):
            try:
                # Count components from batch threshold CSV
                batch_df = pd.read_csv(batch_csv_path)
                batch_count = len(batch_df) if not batch_df.empty else 0
                
                # Count components from individual threshold CSV
                individual_df = pd.read_csv(individual_csv_path)
                individual_count = len(individual_df) if not individual_df.empty else 0
                
                # Calculate component count change (individual - batch)
                count_change = individual_count - batch_count
                
                # Calculate thickness change as the x-axis variable
                if not batch_df.empty and 'average_thickness' in batch_df.columns:
                    batch_avg_thickness = batch_df['average_thickness'].mean()
                else:
                    batch_avg_thickness = 0
                    
                if not individual_df.empty and 'average_thickness' in individual_df.columns:
                    individual_avg_thickness = individual_df['average_thickness'].mean()
                else:
                    individual_avg_thickness = 0
                    
                thickness_change = individual_avg_thickness - batch_avg_thickness
                
                # Use thickness change as the threshold deviation proxy for this plot
                threshold_deviation = thickness_change
                
                # Store the data point
                scatter_data.append({
                    'filename': filename,
                    'condition': condition,
                    'biological_replicate': biological_replicate,
                    'threshold_deviation': threshold_deviation,
                    'count_change': count_change,
                    'thickness_change': thickness_change
                })
                
            except Exception as e:
                print(f"    Warning: Error processing {base_filename}: {e}")
                continue
    
    if not scatter_data:
        print(f"  No valid scatter plot data found for {design_name}")
        return
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group data by condition for color coding
    conditions = sorted(list(set([d['condition'] for d in scatter_data])))
    
    # Plot scatter points for each condition (different colors)
    for condition in conditions:
        condition_data = [d for d in scatter_data if d['condition'] == condition]
        x_values = [d['threshold_deviation'] for d in condition_data]
        y_values = [d['count_change'] for d in condition_data]
        
        color = get_color_for_condition(design_name, condition)
        
        # Plot scatter points
        ax.scatter(x_values, y_values, c=color, alpha=0.7, s=50, label=condition)
    
    # Add single linear regression line for ALL data points
    all_x_values = [d['threshold_deviation'] for d in scatter_data]
    all_y_values = [d['count_change'] for d in scatter_data]
    
    if len(all_x_values) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(all_x_values, all_y_values)
        line_x = np.array([min(all_x_values), max(all_x_values)])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color='black', alpha=0.8, linestyle='-', linewidth=2, label='Overall Fit')
        
        # Add overall R-squared value
        ax.text(0.02, 0.02, f'Overall R² = {r_value**2:.3f}', 
               transform=ax.transAxes, fontsize=12, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    ax.set_xlabel('Threshold Deviation')
    ax.set_ylabel('Component Count Change (Individual - Batch)')
    ax.set_title(f'{design_name} - Component Count Change vs Threshold Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_filename = f'{design_name}_component_count_vs_deviation_scatter.png'
    plot_path = os.path.join(output_dir, design_name, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data to CSV
    csv_filename = f'{design_name}_component_count_vs_deviation_data.csv'
    csv_path = os.path.join(output_dir, design_name, csv_filename)
    scatter_df = pd.DataFrame(scatter_data)
    scatter_df.to_csv(csv_path, index=False)
    
    print(f"  Component count vs deviation scatter plot saved: {plot_filename}")
    print(f"  Component count vs deviation data saved: {csv_filename}")


def generate_condition_based_cdf_plots(data_dict, output_dir, design_name, threshold_type="individual"):
    """
    Generate CDF plots showing individual lines for every biological replicate within each condition.
    Each condition gets its own subplot, with lines representing different biological replicates.
    This is the inverse of the biological replicate plots.
    
    Args:
        data_dict: Dictionary containing component data organized by condition
        output_dir: Output directory for plots
        design_name: Name of experimental design (genotypes, genotypes_statins, domain_analysis)
        threshold_type: Either "individual" or "batch" for labeling
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


def perform_ks_tests_normalized_cdfs(design_name, output_dir):
    """
    Perform KS tests comparing batch vs individual thresholding for normalized CDFs.
    Returns results for table generation.
    """
    from scipy import stats
    
    print(f"  Performing KS tests on normalized CDFs for {design_name}...")
    
    # Load the comprehensive file mapping
    mapping_df = pd.read_csv('comprehensive_cdf_analysis_results/comprehensive_file_mapping.csv')
    design_mapping = mapping_df[mapping_df['experimental_design'] == design_name]
    
    # Function to load and normalize data
    def load_normalized_data(threshold_type):
        condition_replicate_data = defaultdict(lambda: defaultdict(list))
        design_output_dir_csv = os.path.join("comprehensive_cdf_analysis_results", design_name)
        
        for _, row in design_mapping.iterrows():
            filename = row['filename']
            condition = row['condition']
            biological_replicate = row['biological_replicate']
            base_filename = os.path.splitext(filename)[0]
            
            csv_dir = "csv_files_individual" if threshold_type == "individual" else "csv_files_batch"
            csv_path = os.path.join(design_output_dir_csv, condition, csv_dir, f"{base_filename}_blue_components.csv")
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty and 'average_thickness' in df.columns:
                        thickness_values = df['average_thickness'].tolist()
                        condition_replicate_data[condition][biological_replicate].extend(thickness_values)
                except Exception:
                    continue
        
        # Normalize data within each biological replicate using Control as reference
        normalized_data = defaultdict(list)
        for condition in condition_replicate_data:
            for bio_rep in condition_replicate_data[condition]:
                if 'Control' in condition_replicate_data and bio_rep in condition_replicate_data['Control']:
                    control_data = safe_convert_to_numeric(condition_replicate_data['Control'][bio_rep])
                    if len(control_data) > 0:
                        control_mean = np.mean(control_data)
                        condition_data = safe_convert_to_numeric(condition_replicate_data[condition][bio_rep])
                        if len(condition_data) > 0:
                            normalized_values = condition_data / control_mean
                            normalized_data[condition].extend(normalized_values)
        
        return normalized_data
    
    # Load data for both threshold types
    batch_data = load_normalized_data("batch")
    individual_data = load_normalized_data("individual")
    
    # Get common conditions
    common_conditions = set(batch_data.keys()) & set(individual_data.keys())
    
    results = []
    for condition in sorted(common_conditions):
        batch_values = safe_convert_to_numeric(batch_data[condition])
        individual_values = safe_convert_to_numeric(individual_data[condition])
        
        if len(batch_values) > 0 and len(individual_values) > 0:
            # Perform two-sample KS test
            ks_stat, p_value = stats.ks_2samp(batch_values, individual_values)
            
            # Determine significance level
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = "ns"
            
            results.append({
                'Design': design_name,
                'Condition': condition,
                'N_Batch': len(batch_values),
                'N_Individual': len(individual_values),
                'KS_Statistic': round(ks_stat, 4),
                'P_Value': p_value,
                'Significance': significance
            })
    
    return results

def generate_ks_test_summary_table(all_ks_results, output_dir):
    """
    Generate a formatted summary table of KS test results and save to file.
    """
    if not all_ks_results:
        print("  No KS test results to summarize.")
        return
    
    print(f"\n{'='*80}")
    print("KS TEST RESULTS: NORMALIZED CDFS (BATCH vs INDIVIDUAL THRESHOLDING)")
    print(f"{'='*80}")
    
    # Create formatted table
    results_df = pd.DataFrame(all_ks_results)
    
    # Format p-values
    results_df['P_Value_Formatted'] = results_df['P_Value'].apply(
        lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}"
    )
    
    # Create display table
    display_df = results_df[['Design', 'Condition', 'N_Batch', 'N_Individual', 'KS_Statistic', 'P_Value_Formatted', 'Significance']].copy()
    display_df.columns = ['Design', 'Condition', 'N (Batch)', 'N (Individual)', 'KS Stat', 'P-Value', 'Sig.']
    
    # Print formatted table
    print(display_df.to_string(index=False))
    print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
    
    # Save to CSV
    ks_results_file = os.path.join(output_dir, "ks_test_results_normalized_cdfs.csv")
    results_df.to_csv(ks_results_file, index=False)
    print(f"KS test results saved to: {ks_results_file}")
    
    # Summary statistics
    total = len(results_df)
    sig_001 = sum(results_df['P_Value'] < 0.001)
    sig_01 = sum(results_df['P_Value'] < 0.01)
    sig_05 = sum(results_df['P_Value'] < 0.05)
    
    print(f"\nSUMMARY: {total} comparisons, {sig_001} p<0.001, {sig_01} p<0.01, {sig_05} p<0.05")

def main_full_analysis():
    """
    Orchestrates the full analysis pipeline:
    1. Image-level analysis
    2. Batch-level analysis
    3. Deviation analysis (comparison between image- and batch-level)
    for all experimental designs.
    """
    import os
    print("\n===== FULL ANALYSIS PIPELINE STARTED =====\n")
    # Define your experimental designs (update as needed)
    experimental_designs = [
        "genotypes",
        "genotypes_statins",
        "domain_analysis"
    ]
    # You may want to update these paths as needed
    output_dir = "comprehensive_cdf_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    for design_name in experimental_designs:
        print(f"\n--- Running analysis for design: {design_name} ---\n")
        
        # 1. Create or load the file mapping for this design
        mapping_df = create_comprehensive_file_mapping()
        create_experimental_design_summary(mapping_df)
        
        # Filter files for this specific experimental design
        design_files = mapping_df[mapping_df['experimental_design'] == design_name]
        
        if len(design_files) == 0:
            print(f"No files found for design: {design_name}")
            continue
            
        print(f"Processing {len(design_files)} files for {design_name}")
        
        # 2. Run batch-level analysis (calculate batch thresholds by biological replicate)
        print(f"[Batch-level analysis for {design_name}]")
        batch_threshold_data = {}  # Separate variable for threshold data
        batch_data = {}  # Keep batch_data for thickness data
        
        # Group files by biological replicate instead of condition
        unique_batches = design_files['biological_replicate'].unique()
        print(f"  Found {len(unique_batches)} biological replicates: {unique_batches}")
        
        for batch in unique_batches:
            batch_files = design_files[design_files['biological_replicate'] == batch]['full_path'].tolist()
            if len(batch_files) > 0:
                print(f"  Processing {len(batch_files)} files for biological replicate: {batch}")
                batch_threshold_result = calculate_batch_threshold(batch_files, batch)
                if batch_threshold_result[0] is not None:  # Check if threshold was calculated successfully
                    batch_threshold_data[batch] = batch_threshold_result  # Store the full tuple (threshold, mean_intensities, image_names)
                else:
                    print(f"  Warning: Could not calculate threshold for biological replicate: {batch}")
        
        # 3. Run both image-thresholded and batch-thresholded analysis
        print(f"[Image-level analysis for {design_name}]")
        component_data_dict_batch = {}  # Batch-thresholded data
        component_data_dict_image = {}  # Image-thresholded data
        
        # Get unique conditions for directory creation
        unique_conditions = design_files['condition'].unique()
        
        for condition in unique_conditions:
            condition_files = design_files[design_files['condition'] == condition]['full_path'].tolist()
            if len(condition_files) > 0:
                print(f"  Processing images for condition: {condition}")
                
                # Create organized output directory structure
                design_output_dir = os.path.join(output_dir, design_name)
                
                # Create condition-specific output directory
                if design_name == "genotypes":
                    # Use genotype as condition (e.g., "KO", "ApoE4", "Control", "ApoE2") - no _Control suffix needed
                    condition_output_dir = os.path.join(design_output_dir, condition)
                elif design_name == "genotypes_statins":
                    # Use combined condition (e.g., "KO_Control", "ApoE4_Simvastatin")
                    condition_output_dir = os.path.join(design_output_dir, condition)
                else:
                    # For domain_analysis and others, use condition as is
                    condition_output_dir = os.path.join(design_output_dir, condition)
                
                # Create separate directories for batch and image thresholded data
                individual_dir_batch = os.path.join(condition_output_dir, "individual_images_batch")
                individual_dir_individual = os.path.join(condition_output_dir, "individual_images_individual")
                csv_dir_batch = os.path.join(condition_output_dir, "csv_files_batch")
                csv_dir_individual = os.path.join(condition_output_dir, "csv_files_individual")
                
                # Ensure directories exist
                os.makedirs(individual_dir_batch, exist_ok=True)
                os.makedirs(individual_dir_individual, exist_ok=True)
                os.makedirs(csv_dir_batch, exist_ok=True)
                os.makedirs(csv_dir_individual, exist_ok=True)
                
                # Process images with image-level thresholding (no batch threshold)
                print(f"    Running image-thresholded analysis...")
                condition_data_image = process_images_parallel(
                    condition_files, csv_dir_individual, design_name, condition, individual_dir_individual
                )
                # Extract just the thickness data from the tuple
                all_thickness_data_image, batch_data_image, image_level_data_image, image_level_batch_data_image, files_with_errors_image = condition_data_image
                component_data_dict_image[condition] = all_thickness_data_image
                
                # Process images with batch thresholding (group by biological replicate)
                print(f"    Running batch-thresholded analysis...")
                all_thickness_data_batch = []
                
                # Group files by biological replicate for batch thresholding
                for batch in unique_batches:
                    batch_files = design_files[
                        (design_files['condition'] == condition) & 
                        (design_files['biological_replicate'] == batch)
                    ]['full_path'].tolist()
                    
                    if len(batch_files) > 0 and batch in batch_threshold_data:
                        print(f"      Processing {len(batch_files)} files for batch {batch} in condition {condition}")
                        threshold_value = batch_threshold_data[batch][0]  # Extract just the threshold value from the tuple
                        batch_data_result = process_images_parallel_batch_threshold(
                            batch_files, csv_dir_batch, design_name, condition, threshold_value, individual_dir_batch
                        )
                        # Extract just the thickness data from the tuple
                        batch_thickness_data, _, _, _, _ = batch_data_result
                        all_thickness_data_batch.extend(batch_thickness_data)
                    elif len(batch_files) > 0:
                        print(f"      Warning: No batch threshold available for batch {batch} in condition {condition}")
                
                component_data_dict_batch[condition] = all_thickness_data_batch
        
        # Use batch-thresholded data as the main component_data_dict for compatibility
        component_data_dict = component_data_dict_batch
        
        # 4. Generate image-level CDF plots and data
        print(f"[Generating image-level analysis for {design_name}]")
        if component_data_dict_batch and isinstance(component_data_dict_batch, dict):
            design_output_dir = os.path.join(output_dir, design_name)
            # Generate batch-thresholded plots
            generate_image_level_data(component_data_dict_batch, batch_data, design_output_dir, design_name + "_batch")
            generate_image_level_cdf_plots(component_data_dict_batch, design_output_dir, design_name + "_batch")
            generate_image_level_normalized_cdf_plots(component_data_dict_batch, batch_data, design_output_dir, design_name + "_batch")
            # Generate biological replicate CDF plots for batch thresholding
            generate_biological_replicate_cdf_plots(component_data_dict_batch, design_output_dir, design_name, "batch")
            # Generate condition-based CDF plots for batch thresholding
            generate_condition_based_cdf_plots(component_data_dict_batch, design_output_dir, design_name, "batch")
        else:
            print(f"  Warning: component_data_dict_batch is empty or not a dictionary for {design_name}: {type(component_data_dict_batch)}")
        
        if component_data_dict_image and isinstance(component_data_dict_image, dict):
            design_output_dir = os.path.join(output_dir, design_name)
            # Generate image-thresholded plots
            generate_image_level_data(component_data_dict_image, batch_data, design_output_dir, design_name + "_image")
            generate_image_level_cdf_plots(component_data_dict_image, design_output_dir, design_name + "_image")
            generate_image_level_normalized_cdf_plots(component_data_dict_image, batch_data, design_output_dir, design_name + "_image")
            # Generate biological replicate CDF plots for individual thresholding
            generate_biological_replicate_cdf_plots(component_data_dict_image, design_output_dir, design_name, "individual")
            # Generate condition-based CDF plots for individual thresholding
            generate_condition_based_cdf_plots(component_data_dict_image, design_output_dir, design_name, "individual")
        else:
            print(f"  Warning: component_data_dict_image is empty or not a dictionary for {design_name}: {type(component_data_dict_image)}")
        
        # 5. Run deviation analysis (compare image- and batch-level)
        print(f"[Deviation analysis for {design_name}]")
        if batch_threshold_data:
            design_output_dir = os.path.join(output_dir, design_name)
            # generate_threshold_comparison_plots(batch_threshold_data, design_output_dir, design_name)  # Disabled - creates unnecessary individual threshold comparison plots
            generate_combined_deviation_plots(batch_threshold_data, design_output_dir, design_name)
            generate_combined_threshold_plots(batch_threshold_data, design_output_dir, design_name)
            # generate_group_specific_deviation_plots(batch_threshold_data, design_output_dir, design_name)  # Disabled - creates unnecessary individual single-bar graphs
        
        # 6. Generate image vs batch threshold comparison plots
        print(f"[Image vs Batch threshold comparison for {design_name}]")
        if component_data_dict_image and component_data_dict_batch:
            design_output_dir = os.path.join(output_dir, design_name)
            generate_image_vs_batch_threshold_comparison(component_data_dict_image, component_data_dict_batch, design_output_dir, design_name)
            generate_thickness_deviation_plots(component_data_dict_image, component_data_dict_batch, design_output_dir, design_name)
            generate_component_count_deviation_plots(component_data_dict_image, component_data_dict_batch, design_output_dir, design_name)
            generate_thickness_vs_deviation_scatter_plots(component_data_dict_image, component_data_dict_batch, design_output_dir, design_name)
            generate_component_count_vs_deviation_scatter_plots(component_data_dict_image, component_data_dict_batch, design_output_dir, design_name)
        
        # 6. Generate normalized CDFs (biological replicate-based normalization)
        print(f"[Generating normalized CDFs for {design_name}]")
        design_output_dir = os.path.join(output_dir, design_name)
        generate_replicate_normalized_cdf_plots(design_output_dir, design_name)
        
        # 6.5. Generate thick-thin ratio graphs
        print(f"[Generating thick-thin ratio graphs for {design_name}]")
        generate_thick_thin_ratio_graphs(design_output_dir, design_name)
        
        # 7. Generate summary plots and save results
        print(f"[Generating summary plots and saving results for {design_name}]")
        if component_data_dict_batch:
            design_output_dir = os.path.join(output_dir, design_name)
            # Save batch-thresholded data
            save_cdf_data_csv(component_data_dict_batch, design_output_dir, design_name, "component_level_batch")
            save_normalized_data_csvs(component_data_dict_batch, design_output_dir, design_name + "_batch")
        
        if component_data_dict_image:
            design_output_dir = os.path.join(output_dir, design_name)
            # Save image-thresholded data
            save_cdf_data_csv(component_data_dict_image, design_output_dir, design_name, "component_level_image")
            save_normalized_data_csvs(component_data_dict_image, design_output_dir, design_name + "_image")
        
        print(f"Completed analysis for {design_name}")

    # 8. Perform KS tests on normalized CDFs for all designs
    print(f"\n[Kolmogorov-Smirnov Tests on Normalized CDFs]")
    all_ks_results = []
    for design_name in experimental_designs:
        try:
            design_ks_results = perform_ks_tests_normalized_cdfs(design_name, output_dir)
            all_ks_results.extend(design_ks_results)
        except Exception as e:
            print(f"  Error performing KS tests for {design_name}: {e}")
    
    # Generate summary table
    generate_ks_test_summary_table(all_ks_results, output_dir)

    print("\n===== FULL ANALYSIS PIPELINE COMPLETE =====\n")

if __name__ == "__main__":
    main_full_analysis()

    