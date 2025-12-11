"""
Shared constants and default configurations for myotube segmentation.
"""

__all__ = [
    'DEFAULT_POST_PROCESSING_CONFIG',
    'DEFAULT_GUI_CONFIG',
    'IMAGE_EXTENSIONS',
    'STATUS_SUCCESS',
    'STATUS_ERROR',
]


# Default post-processing configuration
DEFAULT_POST_PROCESSING_CONFIG = {
    'min_area': 100,                    # Minimum myotube area (pixels)
    'max_area': 50000,                  # Maximum myotube area (pixels)
    'min_aspect_ratio': 1.5,            # Minimum length/width ratio for myotubes
    'confidence_threshold': 0.25,       # Minimum detection confidence
    'merge_threshold': 0.8,             # IoU threshold for merging overlapping instances
    'fill_holes': True,                 # Fill holes in segmentation masks
    'smooth_boundaries': True,          # Smooth mask boundaries
    'remove_edge_instances': False,     # Remove instances touching image edges
    'final_min_area': 1000,             # Final minimum area filter
}


# Default GUI configuration
DEFAULT_GUI_CONFIG = {
    'input_path': '',
    'output_dir': '',  # Will be set to Desktop/myotube_results by GUI
    'config': '',
    'weights': '',
    'mask2former_path': '',
    'confidence': 0.25,
    'min_area': 100,
    'max_area': 50000,
    'final_min_area': 1000,
    'cpu': False,
    'max_image_size': '',
    'force_1024': False,
    'use_tiling': True,
    'grid_size': 2,
    'tile_overlap': 0.20,
    'skip_merged_masks': True,
    'save_measurements': False,
}


# Image file extensions
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']


# Status file names for Fiji integration
STATUS_SUCCESS = 'BATCH_SUCCESS'
STATUS_ERROR = 'ERROR'
