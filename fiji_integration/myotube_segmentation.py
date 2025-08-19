#!/usr/bin/env python3
"""
Myotube Instance Segmentation for Fiji Integration

This script provides myotube instance segmentation with modular post-processing
pipeline and seamless Fiji integration via composite ROIs.

Usage:
    python myotube_segmentation.py input_image output_dir [--config CONFIG] [--weights WEIGHTS]

Features:
    - Modular post-processing pipeline (easy to extend)
    - Composite ROI generation for multi-segment instances
    - Automatic Fiji-compatible output formats
    - Configurable confidence thresholds and filtering
"""

import os
import sys
import argparse
import json
import zipfile
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torch

# Find Mask2Former project directory
# Priority: 1) Environment variable, 2) Config file, 3) Auto-detection, 4) Current directory

def find_mask2former_project():
    """Find Mask2Former project directory using multiple methods."""
    
    # Method 1: Environment variable (easiest to change)
    project_path = os.environ.get('MASK2FORMER_PATH')
    if project_path and os.path.exists(os.path.join(project_path, 'demo', 'predictor.py')):
        print(f"📁 Using Mask2Former from environment: {project_path}")
        return project_path
    
    # Method 2: Config file in same directory as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'mask2former_config.txt')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            project_path = f.read().strip()
        if project_path and os.path.exists(os.path.join(project_path, 'demo', 'predictor.py')):
            print(f"📁 Using Mask2Former from config file: {project_path}")
            return project_path
    
    # Method 3: Auto-detection in common locations
    possible_paths = [
        os.path.dirname(script_dir),  # Parent of script directory
        '/fs04/scratch2/tf41/ben/Mask2Former',  # Your current path
        '/Users/wangbingsheng/PycharmProjects/CSIRO-UROP/Mask2Former',
        '/home/bwang/ar85_scratch2/ben/download/Mask2Former',
        os.path.expanduser('~/Mask2Former'),
        os.path.expanduser('~/CSIRO-UROP/Mask2Former'),
        os.getcwd()  # Current working directory
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'demo', 'predictor.py')):
            print(f"📁 Auto-detected Mask2Former at: {path}")
            return path
    
    # Method 4: Error with helpful instructions
    print("❌ Could not find Mask2Former project!")
    print("🔧 To fix this, choose one of these options:")
    print("   1. Set environment variable: export MASK2FORMER_PATH='/fs04/scratch2/tf41/ben/Mask2Former'")
    print(f"   2. Create config file: echo '/fs04/scratch2/tf41/ben/Mask2Former' > {config_file}")
    print("   3. Make sure the project is in one of these locations:")
    for path in possible_paths:
        print(f"      - {path}")
    
    raise ImportError("Mask2Former project not found. See instructions above.")

# Add project to Python path
project_dir = find_mask2former_project()
sys.path.insert(0, project_dir)

from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from mask2former import add_maskformer2_config


class PostProcessingPipeline:
    """
    Modular post-processing pipeline for instance segmentation results.
    Easy to extend with new post-processing steps.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize post-processing pipeline.
        
        Args:
            config: Configuration dictionary for post-processing parameters
        """
        self.config = config or self.get_default_config()
        self.steps = []
        self._setup_default_pipeline()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default post-processing configuration."""
        return {
            'min_area': 100,           # Minimum myotube area (pixels)
            'max_area': 50000,         # Maximum myotube area (pixels)
            'min_aspect_ratio': 1.5,   # Minimum length/width ratio for myotubes
            'confidence_threshold': 0.5, # Minimum detection confidence
            'merge_threshold': 0.8,    # IoU threshold for merging overlapping instances
            'fill_holes': True,        # Fill holes in segmentation masks
            'smooth_boundaries': True,  # Smooth mask boundaries
            'remove_edge_instances': False, # Remove instances touching image edges
        }
    
    def _setup_default_pipeline(self):
        """Setup default post-processing steps."""
        self.add_step('filter_by_confidence', self._filter_by_confidence)
        self.add_step('filter_by_area', self._filter_by_area)
        self.add_step('filter_by_aspect_ratio', self._filter_by_aspect_ratio)
        self.add_step('fill_holes', self._fill_holes)
        self.add_step('smooth_boundaries', self._smooth_boundaries)
        self.add_step('remove_edge_instances', self._remove_edge_instances)
        self.add_step('merge_overlapping', self._merge_overlapping_instances)
    
    def add_step(self, name: str, function: callable, position: int = -1):
        """
        Add a post-processing step to the pipeline.
        
        Args:
            name: Name of the processing step
            function: Function to execute (should take and return instances dict)
            position: Position in pipeline (-1 for end)
        """
        step = {'name': name, 'function': function}
        if position == -1:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
    
    def remove_step(self, name: str):
        """Remove a processing step by name."""
        self.steps = [step for step in self.steps if step['name'] != name]
    
    def process(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """
        Run the complete post-processing pipeline.
        
        Args:
            instances: Raw instances from segmentation model
            image: Original input image
            
        Returns:
            Processed instances dictionary
        """
        print(f"🔄 Running post-processing pipeline with {len(self.steps)} steps...")
        
        # Convert to our internal format
        processed_instances = self._convert_to_internal_format(instances, image)
        
        # Run each post-processing step
        for step in self.steps:
            try:
                print(f"   ➤ {step['name']}: {len(processed_instances['masks'])} instances")
                result = step['function'](processed_instances, image)
                if result is not None:
                    processed_instances = result
                else:
                    print(f"   ⚠️  Warning: Step '{step['name']}' returned None, keeping original")
            except Exception as e:
                print(f"   ⚠️  Warning: Step '{step['name']}' failed: {e}")
                import traceback
                print(f"   🔍 DEBUG: Full traceback: {traceback.format_exc()}")
                # Keep the original processed_instances on error
                continue
        
        print(f"✅ Post-processing complete: {len(processed_instances['masks'])} final instances")
        return processed_instances
    
    def _convert_to_internal_format(self, instances, image: np.ndarray) -> Dict[str, Any]:
        """Convert model output to internal processing format."""
        if hasattr(instances, 'pred_masks'):
            # Detectron2 Instances format
            masks = instances.pred_masks.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            
            # Get boxes from model, but validate them
            pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
            
            # Check if predicted boxes are valid
            valid_boxes = []
            for i, (mask, box) in enumerate(zip(masks, pred_boxes)):
                width = box[2] - box[0]
                height = box[3] - box[1]
                
                # If model box is invalid, calculate from mask
                if width <= 0 or height <= 0:
                    # Calculate bounding box from mask
                    coords = np.where(mask)
                    if len(coords[0]) > 0:
                        y_min, y_max = coords[0].min(), coords[0].max()
                        x_min, x_max = coords[1].min(), coords[1].max()
                        calculated_box = np.array([x_min, y_min, x_max, y_max])
                        valid_boxes.append(calculated_box)
                        if i < 3:  # Only print for first few
                            print(f"      🔧 Fixed box {i}: [0,0,0,0] → [{x_min},{y_min},{x_max},{y_max}]")
                    else:
                        # Empty mask, use original (probably will be filtered out anyway)
                        valid_boxes.append(box)
                else:
                    # Use original valid box
                    valid_boxes.append(box)
            
            boxes = np.array(valid_boxes)
        else:
            # Already in our format
            return instances
            
        return {
            'masks': masks,
            'scores': scores,
            'boxes': boxes,
            'image_shape': image.shape[:2]
        }
    
    # Post-processing step implementations
    def _filter_by_confidence(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Filter instances by confidence score."""
        if not self.config.get('confidence_threshold'):
            return instances
            
        threshold = self.config['confidence_threshold']
        keep = instances['scores'] >= threshold
        
        return {
            'masks': instances['masks'][keep],
            'scores': instances['scores'][keep],
            'boxes': instances['boxes'][keep],
            'image_shape': instances['image_shape']
        }
    
    def _filter_by_area(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Filter instances by area."""
        min_area = self.config.get('min_area', 0)
        max_area = self.config.get('max_area', float('inf'))
        
        areas = np.array([mask.sum() for mask in instances['masks']])
        keep = (areas >= min_area) & (areas <= max_area)
        
        return {
            'masks': instances['masks'][keep],
            'scores': instances['scores'][keep],
            'boxes': instances['boxes'][keep],
            'image_shape': instances['image_shape']
        }
    
    def _filter_by_aspect_ratio(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Filter instances by aspect ratio (length/width)."""
        min_ratio = self.config.get('min_aspect_ratio', 0)
        if min_ratio <= 0:
            return instances
        
        keep_indices = []
        for i, mask in enumerate(instances['masks']):
            # Calculate bounding box aspect ratio
            box = instances['boxes'][i]
            width = box[2] - box[0]
            height = box[3] - box[1]
            
            # Skip invalid boxes (zero width or height)
            if width <= 0 or height <= 0:
                print(f"      ⚠️  Skipping instance {i} with invalid box: width={width}, height={height}")
                continue
                
            aspect_ratio = max(width, height) / min(width, height)
            
            if aspect_ratio >= min_ratio:
                keep_indices.append(i)
        
        keep = np.array(keep_indices)
        if len(keep) == 0:
            # Return empty instances
            return {
                'masks': np.array([]),
                'scores': np.array([]),
                'boxes': np.array([]),
                'image_shape': instances['image_shape']
            }
        
        return {
            'masks': instances['masks'][keep],
            'scores': instances['scores'][keep],
            'boxes': instances['boxes'][keep],
            'image_shape': instances['image_shape']
        }
    
    def _fill_holes(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Fill holes in segmentation masks."""
        if not self.config.get('fill_holes', True):
            return instances
        
        # Check if we have any masks to process
        if len(instances['masks']) == 0:
            return instances
        
        try:
            from scipy import ndimage
            filled_masks = []
            
            for mask in instances['masks']:
                # Ensure mask is boolean for fill_holes
                bool_mask = mask.astype(bool)
                # Fill holes using binary fill_holes
                filled_mask = ndimage.binary_fill_holes(bool_mask)
                # Convert back to original dtype
                filled_masks.append(filled_mask.astype(mask.dtype))
            
            # Preserve array structure
            if len(filled_masks) > 0:
                instances['masks'] = np.array(filled_masks)
            
        except Exception as e:
            print(f"      ⚠️  Warning: Fill holes failed ({e}), keeping original masks")
            
        return instances
    
    def _smooth_boundaries(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Smooth mask boundaries."""
        if not self.config.get('smooth_boundaries', True):
            return instances
        
        smoothed_masks = []
        
        for mask in instances['masks']:
            # Apply morphological opening and closing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            smoothed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
            smoothed_masks.append(smoothed.astype(mask.dtype))
        
        instances['masks'] = np.array(smoothed_masks)
        return instances
    
    def _remove_edge_instances(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Remove instances that touch image edges."""
        if not self.config.get('remove_edge_instances', False):
            return instances
        
        h, w = instances['image_shape']
        keep_indices = []
        
        for i, mask in enumerate(instances['masks']):
            # Check if mask touches any edge
            touches_edge = (
                mask[0, :].any() or  # Top edge
                mask[-1, :].any() or  # Bottom edge
                mask[:, 0].any() or  # Left edge
                mask[:, -1].any()    # Right edge
            )
            
            if not touches_edge:
                keep_indices.append(i)
        
        keep = np.array(keep_indices)
        if len(keep) == 0:
            return {
                'masks': np.array([]),
                'scores': np.array([]),
                'boxes': np.array([]),
                'image_shape': instances['image_shape']
            }
        
        return {
            'masks': instances['masks'][keep],
            'scores': instances['scores'][keep],
            'boxes': instances['boxes'][keep],
            'image_shape': instances['image_shape']
        }
    
    def _merge_overlapping_instances(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Merge overlapping instances based on IoU threshold."""
        merge_threshold = self.config.get('merge_threshold', 0.8)
        if merge_threshold >= 1.0 or len(instances['masks']) <= 1:
            return instances
        
        # Calculate IoU matrix
        masks = instances['masks']
        n = len(masks)
        iou_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                intersection = np.logical_and(masks[i], masks[j]).sum()
                union = np.logical_or(masks[i], masks[j]).sum()
                iou = intersection / union if union > 0 else 0
                iou_matrix[i, j] = iou_matrix[j, i] = iou
        
        # Find groups to merge
        merged_groups = []
        used = set()
        
        for i in range(n):
            if i in used:
                continue
            
            group = [i]
            used.add(i)
            
            # Find all instances that should be merged with this one
            for j in range(i + 1, n):
                if j not in used and iou_matrix[i, j] >= merge_threshold:
                    group.append(j)
                    used.add(j)
            
            merged_groups.append(group)
        
        # Create merged instances
        merged_masks = []
        merged_scores = []
        merged_boxes = []
        
        for group in merged_groups:
            if len(group) == 1:
                # Single instance, keep as is
                idx = group[0]
                merged_masks.append(instances['masks'][idx])
                merged_scores.append(instances['scores'][idx])
                merged_boxes.append(instances['boxes'][idx])
            else:
                # Merge multiple instances
                combined_mask = np.logical_or.reduce([instances['masks'][idx] for idx in group])
                max_score = max([instances['scores'][idx] for idx in group])
                
                # Calculate bounding box for merged mask
                coords = np.where(combined_mask)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    merged_box = np.array([x_min, y_min, x_max, y_max])
                else:
                    merged_box = instances['boxes'][group[0]]
                
                merged_masks.append(combined_mask)
                merged_scores.append(max_score)
                merged_boxes.append(merged_box)
        
        return {
            'masks': np.array(merged_masks),
            'scores': np.array(merged_scores),
            'boxes': np.array(merged_boxes),
            'image_shape': instances['image_shape']
        }


class ImageJROIGenerator:
    """
    Generator for ImageJ-compatible ROI files from binary masks.
    Creates proper ROI files that can be loaded into ImageJ/Fiji ROI Manager.
    """
    
    # ImageJ ROI file format constants
    MAGIC = b'Iout'
    VERSION = 227
    
    # ROI types
    POLYGON = 0
    RECT = 1
    OVAL = 2
    LINE = 3
    FREELINE = 4
    POLYLINE = 5
    NOROI = 6
    FREEHAND = 7
    TRACED = 8
    ANGLE = 9
    POINT = 10
    
    def __init__(self):
        """Initialize ROI generator."""
        pass
    
    def mask_to_roi_file(self, mask: np.ndarray, name: str = "") -> bytes:
        """
        Convert a binary mask to ImageJ ROI file format.
        
        Args:
            mask: Binary mask array (2D numpy array)
            name: Name for the ROI
            
        Returns:
            Bytes content of the ROI file
        """
        from skimage import measure
        
        # Find contours in the mask
        contours = measure.find_contours(mask, 0.5)
        
        if len(contours) == 0:
            # Create empty ROI if no contours found
            return self._create_empty_roi(name)
        
        # For composite ROIs (multi-segment instances), we need to handle multiple contours
        if len(contours) == 1:
            # Single contour - create simple polygon ROI
            return self._create_polygon_roi(contours[0], name)
        else:
            # Multiple contours - create composite ROI using largest contour
            # In the future, this could be enhanced to create true composite ROIs
            largest_contour = max(contours, key=len)
            return self._create_polygon_roi(largest_contour, name)
    
    def _create_polygon_roi(self, contour: np.ndarray, name: str = "") -> bytes:
        """Create a polygon ROI from contour coordinates."""
        import struct
        
        # Convert contour coordinates (y, x) to (x, y) and ensure integers
        coords = [(int(point[1]), int(point[0])) for point in contour]
        
        if len(coords) < 3:
            return self._create_empty_roi(name)
        
        # Calculate bounding box
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        left = min(x_coords)
        top = min(y_coords)
        right = max(x_coords)
        bottom = max(y_coords)
        
        # Convert to relative coordinates
        rel_coords = [(x - left, y - top) for x, y in coords]
        
        # Create ROI header
        roi_header = self._create_roi_header(
            roi_type=self.POLYGON,
            top=top,
            left=left,
            bottom=bottom,
            right=right,
            n_coordinates=len(coords),
            name=name
        )
        
        # Pack coordinate data
        coord_data = b''
        for x, y in rel_coords:
            coord_data += struct.pack('>hh', x, y)  # signed 16-bit coordinates
        
        return roi_header + coord_data
    
    def _create_empty_roi(self, name: str = "") -> bytes:
        """Create an empty ROI file."""
        return self._create_roi_header(
            roi_type=self.NOROI,
            top=0, left=0, bottom=0, right=0,
            n_coordinates=0,
            name=name
        )
    
    def _create_roi_header(self, roi_type: int, top: int, left: int, bottom: int, right: int,
                          n_coordinates: int, name: str = "") -> bytes:
        """Create ROI file header in ImageJ format."""
        import struct
        
        # Basic header
        header = struct.pack('>4sH', self.MAGIC, self.VERSION)
        
        # ROI header (64 bytes total)
        header_data = struct.pack(
            '>BBHhhhhHHHHhhhhHHH',
            roi_type,           # ROI type
            0,                  # Subtype
            top,                # Top
            left,               # Left  
            bottom,             # Bottom
            right,              # Right
            n_coordinates,      # N coordinates
            0,                  # X1 (line start)
            0,                  # Y1 (line start)
            0,                  # X2 (line end)
            0,                  # Y2 (line end)
            0,                  # Reserved
            0,                  # Reserved
            0,                  # Reserved
            0,                  # Reserved
            0,                  # Stroke width
            0,                  # Shape ROI size
            0                   # Stroke color
        )
        
        # Pad header to 64 bytes
        header_size = len(header) + len(header_data)
        padding_needed = 64 - header_size
        if padding_needed > 0:
            header_data += b'\x00' * padding_needed
        
        # Add name if provided (as null-terminated string)
        name_data = b''
        if name:
            name_bytes = name.encode('utf-8')[:60]  # Limit name length
            name_data = name_bytes + b'\x00'
        
        return header + header_data + name_data


class MyotubeFijiIntegration:
    """
    Main class for Fiji integration of myotube instance segmentation.
    """
    
    def __init__(self, config_file: str = None, model_weights: str = None):
        """
        Initialize the Fiji integration.
        
        Args:
            config_file: Path to model config file
            model_weights: Path to model weights
        """
        self.config_file = config_file
        self.model_weights = model_weights
        self.predictor = None
        self.post_processor = PostProcessingPipeline()
        
        # Setup paths
        self.setup_paths()
        
    def setup_paths(self):
        """Setup default paths if not provided."""
        # Use the detected project directory instead of script location
        base_dir = Path(project_dir)
        
        if not self.config_file:
            # Try to find the best available config
            config_options = [
                base_dir / "stage2_config.yaml",
                base_dir / "stage1_config.yaml",
                base_dir / "stage2_panoptic_config.yaml",
                base_dir / "stage1_panoptic_config.yaml",
                base_dir / "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
            ]
            
            print(f"🔍 Looking for config files in: {base_dir}")
            for config_path in config_options:
                print(f"   Checking: {config_path.name} - {'✅' if config_path.exists() else '❌'}")
                if config_path.exists():
                    self.config_file = str(config_path)
                    break
        
        if not self.model_weights:
            # Try to find the best available weights
            weight_options = [
                base_dir / "output_stage2_manual/model_final.pth",
                base_dir / "output_stage2_manual/model_best.pth",
                base_dir / "output_stage2_panoptic_manual/model_final.pth",
                base_dir / "output_stage2_panoptic_manual/model_best.pth",
                base_dir / "output_stage1_algorithmic/model_final.pth",
                base_dir / "output_stage1_algorithmic/model_best.pth",
                base_dir / "output_stage1_panoptic_algorithmic/model_final.pth",
                base_dir / "output_stage1_panoptic_algorithmic/model_best.pth",
            ]
            
            print(f"🔍 Looking for model weights in: {base_dir}")
            for weight_path in weight_options:
                if weight_path.exists():
                    print(f"   Found: {weight_path.name}")
                    self.model_weights = str(weight_path)
                    break
                    
        if not self.config_file:
            print("❌ No config file found! Available options:")
            print("   1. Specify with --config argument")
            print("   2. Place config files in project directory")
            print("   3. Use default COCO config")
            # Use default COCO config as fallback
            default_config = base_dir / "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
            if default_config.exists():
                self.config_file = str(default_config)
                print(f"   ✅ Using fallback: {default_config.name}")
            else:
                raise FileNotFoundError(
                    f"No config files found in {base_dir}. "
                    "Please check your Mask2Former installation or specify --config path."
                )
            
        if not self.model_weights:
            print("❌ No model weights found! Available options:")
            print("   1. Specify with --weights argument")
            print("   2. Train model and place weights in output directories")
            print("   3. Use COCO pre-trained weights")
            # Use COCO pre-trained as fallback
            self.model_weights = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl"
            print("   Using COCO pre-trained weights (will download)")
        
        print(f"📁 Config file: {self.config_file}")
        print(f"🔮 Model weights: {self.model_weights}")
    
    def initialize_predictor(self, force_cpu=False):
        """Initialize the segmentation predictor."""
        if self.predictor is not None:
            return
        
        self.force_cpu = force_cpu
        
        print("🚀 Initializing Mask2Former predictor...")
        
        # Clear GPU cache before initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB total")
            print(f"   🔥 GPU Memory: {torch.cuda.memory_allocated() // 1e6:.0f}MB allocated before init")
        
        # Validate files exist
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        # Setup configuration in correct order (like demo.py)
        cfg = get_cfg()
        # CRITICAL: Add configs in same order as demo.py
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        
        try:
            # Temporarily allow unknown keys to accommodate minor version diffs
            if hasattr(cfg, 'set_new_allowed'):
                cfg.set_new_allowed(True)
            cfg.merge_from_file(self.config_file)
        except Exception as e:
            print(f"❌ Error loading config file: {self.config_file}")
            print(f"   Error: {e}")
            
            # Try with a known working config as fallback
            fallback_config = os.path.join(project_dir, "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
            if os.path.exists(fallback_config):
                print(f"   🔄 Trying fallback config: {fallback_config}")
                try:
                    if hasattr(cfg, 'set_new_allowed'):
                        cfg.set_new_allowed(True)
                    cfg.merge_from_file(fallback_config)
                except Exception as e2:
                    print(f"   ❌ Fallback config also failed: {e2}")
                    print("   🔧 Creating minimal working config...")
                    self._setup_minimal_config(cfg)
            else:
                print("   🔧 Creating minimal working config...")
                self._setup_minimal_config(cfg)
        finally:
            # Disallow unknown keys after merging to avoid silent errors later
            if hasattr(cfg, 'set_new_allowed'):
                cfg.set_new_allowed(False)
        
        cfg.MODEL.WEIGHTS = self.model_weights
        
        # Memory optimization: preserve training resolution by default
        original_size = getattr(cfg.INPUT, 'IMAGE_SIZE', 1024)
        
        # Only reduce if extremely large (>2048) - otherwise preserve training resolution
        if cfg.INPUT.IMAGE_SIZE > 2048:
            cfg.INPUT.IMAGE_SIZE = 1500  # Use your training size as reasonable max
            print(f"   🔧 Reduced input size: {original_size} → {cfg.INPUT.IMAGE_SIZE} (extreme size limit)")
        else:
            print(f"   ✅ Using training resolution: {cfg.INPUT.IMAGE_SIZE}px (matching training config)")
        
        # Store training size for later use
        self.training_image_size = cfg.INPUT.IMAGE_SIZE
        
        # Memory optimization: ensure batch size is 1 for inference
        if hasattr(cfg.SOLVER, 'IMS_PER_BATCH'):
            cfg.SOLVER.IMS_PER_BATCH = 1
        
        # Only set threshold if this is a mask2former config
        if hasattr(cfg.MODEL, 'MASK_FORMER'):
            cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.25  # Lower threshold for better detection
        
        # Force CPU if requested
        if self.force_cpu or not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
            print("   🖥️  Using CPU inference")
        
        # Freeze config before creating predictor (like demo.py does)
        cfg.freeze()
        
        try:
            self.predictor = DefaultPredictor(cfg)
            device = "CPU" if cfg.MODEL.DEVICE == "cpu" else "GPU"
            print(f"✅ Predictor initialized successfully on {device}!")
            
            if torch.cuda.is_available() and cfg.MODEL.DEVICE != "cpu":
                print(f"   🔥 GPU Memory: {torch.cuda.memory_allocated() // 1e6:.0f}MB allocated after init")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ GPU out of memory during initialization")
                if not force_cpu:  # Only try CPU fallback if not already using CPU
                    print(f"   💡 Trying CPU fallback...")
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Create new config for CPU (like AsyncPredictor does)
                    cpu_cfg = cfg.clone()
                    cpu_cfg.defrost()
                    cpu_cfg.MODEL.DEVICE = "cpu"
                    cpu_cfg.freeze()
                    
                    self.predictor = DefaultPredictor(cpu_cfg)
                    print("✅ Successfully switched to CPU inference!")
                else:
                    print("❌ Out of memory even on CPU - try reducing image size")
                    raise e
            else:
                raise e
    
    def _setup_minimal_config(self, cfg):
        """Setup minimal working Mask2Former config when file configs fail."""
        print("   Setting up minimal Mask2Former configuration...")
        
        # Basic model setup for instance segmentation
        cfg.MODEL.META_ARCHITECTURE = "MaskFormer"
        cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        
        # SEM_SEG_HEAD config
        cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1  # Just myotubes
        cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
        cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
        cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
        cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
        cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6
        
        # MASK_FORMER config
        cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "StandardTransformerDecoder"
        cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
        cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
        cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
        cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
        cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
        cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
        cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
        cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
        cfg.MODEL.MASK_FORMER.NHEADS = 8
        cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
        cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
        cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
        cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
        cfg.MODEL.MASK_FORMER.PRE_NORM = False
        cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False
        cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32
        cfg.MODEL.MASK_FORMER.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE = True
        
        # Test config
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.25
        cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.8
        
        # Input config
        cfg.INPUT.IMAGE_SIZE = 1024
        cfg.INPUT.MIN_SCALE = 0.1
        cfg.INPUT.MAX_SCALE = 2.0
        cfg.INPUT.FORMAT = "RGB"
        
        # Dataset config
        cfg.DATASETS.TEST = ("myotube_test",)  # Dummy dataset name
        
        print("   ✅ Minimal config created")
    
    def segment_image(self, image_path: str, output_dir: str, 
                     custom_config: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Segment myotubes in an image and save Fiji-compatible outputs.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            custom_config: Custom post-processing configuration
            
        Returns:
            Dictionary with paths to generated files
        """
        print(f"🔬 Segmenting myotubes in: {os.path.basename(image_path)}")
        
        # Initialize predictor if needed
        force_cpu = custom_config.get('force_cpu', False) if custom_config else False
        self.initialize_predictor(force_cpu=force_cpu)
        
        # Update post-processing config if provided
        if custom_config:
            self.post_processor.config.update(custom_config)
        
        # Load and process image
        image = read_image(image_path, format="BGR")
        original_image = cv2.imread(image_path)
        
        # Smart image resizing: respect training resolution unless explicitly overridden
        h, w = image.shape[:2]
        training_size = getattr(self, 'training_image_size', 1500)
        max_size = custom_config.get('max_image_size', None) if custom_config else None
        
        # Determine target size
        if max_size and max_size < training_size:
            # User explicitly wants smaller images for memory
            target_size = max_size
            reason = "user-requested memory optimization"
        elif max(h, w) > training_size * 1.5:  # Only resize if much larger than training
            target_size = training_size
            reason = "matching training resolution"
        elif max_size and max(h, w) > max_size:
            target_size = max_size  
            reason = "size limit"
        else:
            target_size = None  # No resizing needed
        
        if target_size and max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            print(f"   🔧 Resized image: {w}×{h} → {new_w}×{new_h} ({reason})")
            # Store scaling info for mask resizing later
            self._scale_factor = scale
            self._original_size = (h, w)
            self._inference_size = (new_h, new_w)
        else:
            print(f"   ✅ Keeping original size: {w}×{h} (within training resolution range)")
            # No scaling needed
            self._scale_factor = 1.0
            self._original_size = (h, w)
            self._inference_size = (h, w)
        
        # Clear GPU cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   🔥 GPU Memory before inference: {torch.cuda.memory_allocated() // 1e6:.0f}MB")
        
        # Run segmentation
        print("   🔄 Running inference...")
        try:
            predictions = self.predictor(image)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("   ❌ GPU out of memory during inference")
                print("   💡 Try reducing image size or using CPU mode")
                raise RuntimeError("GPU out of memory. Try: --cpu or resize image to <1024px") from e
            else:
                raise e
        instances = predictions["instances"]
        
        if len(instances) == 0:
            print("   ⚠️  No myotubes detected!")
            return self._create_empty_outputs(image_path, output_dir)
        
        print(f"   🎯 Detected {len(instances)} potential myotubes")
        
        # Apply post-processing
        processed_instances = self.post_processor.process(instances, original_image)
        
        # Generate outputs
        output_files = self._generate_fiji_outputs(
            processed_instances, original_image, image_path, output_dir
        )
        
        return output_files
    
    def _create_empty_outputs(self, image_path: str, output_dir: str) -> Dict[str, str]:
        """Create empty output files when no instances are detected."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        
        # Create empty ROI file
        roi_path = os.path.join(output_dir, f"{base_name}_rois.zip")
        with zipfile.ZipFile(roi_path, 'w') as zf:
            pass  # Empty zip file
        
        # Create empty overlay (just copy original)
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.tif")
        original = cv2.imread(image_path)
        cv2.imwrite(overlay_path, original)
        
        # Create empty measurements
        measurements_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
        with open(measurements_path, 'w') as f:
            f.write("Instance,Area,Perimeter,AspectRatio,Confidence\n")
        
        return {
            'rois': roi_path,
            'overlay': overlay_path,
            'measurements': measurements_path,
            'count': 0
        }
    
    def _generate_fiji_outputs(self, instances: Dict[str, Any], original_image: np.ndarray,
                              image_path: str, output_dir: str) -> Dict[str, str]:
        """Generate all Fiji-compatible output files."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        
        # Generate composite ROIs
        roi_path = os.path.join(output_dir, f"{base_name}_rois.zip")
        self._save_composite_rois(instances, roi_path)
        
        # Generate colored overlay
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.tif")
        self._save_colored_overlay(instances, original_image, overlay_path)
        
        # Generate measurements CSV
        measurements_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
        self._save_measurements(instances, measurements_path)
        
        # Generate summary info
        info_path = os.path.join(output_dir, f"{base_name}_info.json")
        self._save_info(instances, image_path, info_path)
        
        print(f"✅ Generated outputs for {len(instances['masks'])} myotubes")
        
        return {
            'rois': roi_path,
            'overlay': overlay_path,
            'measurements': measurements_path,
            'info': info_path,
            'count': len(instances['masks'])
        }
    
    def _save_composite_rois(self, instances: Dict[str, Any], output_path: str):
        """Save instances as composite ROIs for Fiji ROI Manager."""
        roi_generator = ImageJROIGenerator()
        
        print(f"   💾 Generating ROI file: {output_path}")
        print(f"   📊 Processing {len(instances['masks'])} instances for ROI generation")
        
        with zipfile.ZipFile(output_path, 'w') as zf:
            for i, mask in enumerate(instances['masks']):
                roi_name = f"Myotube_{i+1}.roi"
                
                # Resize mask to original image size for ROI generation
                if hasattr(self, '_scale_factor') and self._scale_factor != 1.0:
                    original_h, original_w = self._original_size
                    resized_mask = cv2.resize(
                        mask.astype(np.uint8), 
                        (original_w, original_h), 
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    print(f"      🔧 Resized mask {i+1}: {mask.shape} → {resized_mask.shape}")
                else:
                    resized_mask = mask.astype(bool)
                
                # Check if mask has any pixels
                pixel_count = resized_mask.sum()
                if pixel_count == 0:
                    print(f"      ⚠️  Warning: Mask {i+1} is empty (0 pixels)")
                    continue
                
                # Generate proper ImageJ ROI file
                roi_content = roi_generator.mask_to_roi_file(resized_mask, f"Myotube_{i+1}")
                
                if len(roi_content) == 0:
                    print(f"      ⚠️  Warning: ROI content for mask {i+1} is empty")
                    continue
                    
                zf.writestr(roi_name, roi_content)
                print(f"      ✅ Generated ROI {i+1}: {len(roi_content)} bytes, {pixel_count} pixels")
        
        # Check final zip file
        try:
            with zipfile.ZipFile(output_path, 'r') as zf:
                roi_count = len(zf.namelist())
                print(f"   📦 ROI zip contains {roi_count} files: {zf.namelist()[:3]}{'...' if roi_count > 3 else ''}")
        except Exception as e:
            print(f"   ❌ Error reading ROI zip: {e}")
    

    
    def _save_colored_overlay(self, instances: Dict[str, Any], original_image: np.ndarray, 
                             output_path: str):
        """Save colored overlay for visualization."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Create colored overlay
        overlay = original_image.copy()
        
        # Generate distinct colors for each instance
        colors = plt.cm.Set3(np.linspace(0, 1, len(instances['masks'])))
        
        for i, (mask, score) in enumerate(zip(instances['masks'], instances['scores'])):
            # Create colored mask
            color = (colors[i][:3] * 255).astype(np.uint8)
            
            # Resize mask to match original image size
            if hasattr(self, '_scale_factor') and self._scale_factor != 1.0:
                # Resize mask to original image size
                original_h, original_w = self._original_size
                resized_mask = cv2.resize(
                    mask.astype(np.uint8), 
                    (original_w, original_h), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                resized_mask = mask.astype(bool)
            
            # Apply color to mask region
            colored_mask = np.zeros_like(original_image)
            colored_mask[resized_mask] = color
            
            # Blend with original image
            alpha = 0.6
            overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
            
            # Add confidence score text
            coords = np.where(mask)
            if len(coords[0]) > 0:
                center_y, center_x = coords[0].mean(), coords[1].mean()
                cv2.putText(overlay, f"{score:.2f}", 
                           (int(center_x), int(center_y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save overlay
        cv2.imwrite(output_path, overlay)
    
    def _save_measurements(self, instances: Dict[str, Any], output_path: str):
        """Save measurements CSV for analysis."""
        import pandas as pd
        from skimage import measure
        
        measurements = []
        
        for i, (mask, score, box) in enumerate(zip(instances['masks'], instances['scores'], instances['boxes'])):
            # Resize mask to original size for accurate measurements
            if hasattr(self, '_scale_factor') and self._scale_factor != 1.0:
                original_h, original_w = self._original_size
                resized_mask = cv2.resize(
                    mask.astype(np.uint8), 
                    (original_w, original_h), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                
                # Scale bounding box back to original coordinates
                scale_factor = 1.0 / self._scale_factor
                scaled_box = box * scale_factor
            else:
                resized_mask = mask.astype(bool)
                scaled_box = box
            
            # Calculate basic measurements (using resized mask for accuracy)
            area = resized_mask.sum()
            
            # Calculate perimeter
            contours = measure.find_contours(resized_mask, 0.5)
            perimeter = sum(len(contour) for contour in contours)
            
            # Calculate aspect ratio from scaled bounding box
            width = scaled_box[2] - scaled_box[0]
            height = scaled_box[3] - scaled_box[1]
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            
            measurements.append({
                'Instance': f'Myotube_{i+1}',
                'Area': area,
                'Perimeter': perimeter,
                'AspectRatio': aspect_ratio,
                'Confidence': score,
                'BoundingBox_X': box[0],
                'BoundingBox_Y': box[1],
                'BoundingBox_Width': width,
                'BoundingBox_Height': height
            })
        
        # Save to CSV
        df = pd.DataFrame(measurements)
        df.to_csv(output_path, index=False)
    
    def _save_info(self, instances: Dict[str, Any], image_path: str, output_path: str):
        """Save processing information."""
        info = {
            'input_image': os.path.basename(image_path),
            'num_instances': len(instances['masks']),
            'image_shape': instances['image_shape'],
            'config_file': self.config_file,
            'model_weights': self.model_weights,
            'post_processing_config': self.post_processor.config,
            'processing_steps': [step['name'] for step in self.post_processor.steps]
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)


def main():
    """Main function for command-line usage."""
    
    # Setup environment like demo.py does
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    setup_logger()
    
    # Register datasets like demo.py does (if register_two_stage_datasets exists)
    try:
        from register_two_stage_datasets import register_two_stage_datasets
        register_two_stage_datasets(
            dataset_root="./myotube_batch_output", 
            register_instance=True, 
            register_panoptic=True
        )
    except ImportError:
        # Dataset registration not available, continue without it
        pass
    
    parser = argparse.ArgumentParser(description="Myotube Instance Segmentation for Fiji")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--config", help="Path to model config file")
    parser.add_argument("--weights", help="Path to model weights")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="Confidence threshold for detection")
    parser.add_argument("--min-area", type=int, default=100,
                       help="Minimum myotube area in pixels")
    parser.add_argument("--max-area", type=int, default=50000,
                       help="Maximum myotube area in pixels")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU inference (slower but uses less memory)")
    parser.add_argument("--max-image-size", type=int, default=None,
                       help="Maximum image dimension (larger images will be resized). Default: respect training resolution")
    parser.add_argument("--force-1024", action="store_true",
                       help="Force 1024px input resolution for memory optimization (may reduce accuracy)")
    
    args = parser.parse_args()
    
    # Custom post-processing config
    max_image_size = 1024 if args.force_1024 else args.max_image_size
    custom_config = {
        'confidence_threshold': args.confidence,
        'min_area': args.min_area,
        'max_area': args.max_area,
        'max_image_size': max_image_size,
        'force_cpu': args.cpu
    }
    
    # Initialize integration
    integration = MyotubeFijiIntegration(
        config_file=args.config,
        model_weights=args.weights
    )
    
    # Apply memory optimization settings
    if args.cpu:
        print("🖥️  CPU inference mode enabled")
    if args.force_1024:
        print("📏 Forced 1024px input resolution (memory optimization - may reduce accuracy)")
    elif args.max_image_size:
        print(f"📏 Max image size set to: {args.max_image_size}px")
    else:
        print("📏 Using training resolution (1500px) for best accuracy")
    
    # Process image
    try:
        output_files = integration.segment_image(
            args.input_image, 
            args.output_dir, 
            custom_config
        )
        
        print("\n" + "="*60)
        print("🎉 MYOTUBE SEGMENTATION COMPLETED!")
        print("="*60)
        print(f"📊 Results: {output_files['count']} myotubes detected")
        print(f"📁 Output files:")
        for key, path in output_files.items():
            if key != 'count':
                print(f"   {key}: {os.path.basename(path)}")
        print("="*60)
        
        # Signal success to ImageJ macro (format expected by Fiji macro)
        success_file = os.path.join(args.output_dir, "SUCCESS")
        
        # Use ultra-short format for ImageJ line length limits (5-char chunks!)
        base_dir = os.path.dirname(output_files['rois'])
        
        # Debug: Print what we're about to write
        print(f"📝 Writing SUCCESS file: {success_file}")
        print(f"   DIR:{base_dir}")
        print(f"   COUNT:{output_files['count']}")
        print(f"   Using SIMPLE format: just the number!")
        
        # BYPASS broken File.openAsString() - just write the count!
        # Fiji will search for ROI files in the known output directory
        with open(success_file, 'w', encoding='utf-8') as f:
            f.write(str(output_files['count']))  # Just the number, nothing else!
        
        # Debug: Verify what was written
        try:
            with open(success_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"📄 SUCCESS file content ({len(content)} chars):")
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    print(f"     Line {i}: '{line}'")
        except Exception as e:
            print(f"❌ Error reading SUCCESS file: {e}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Signal failure to ImageJ macro
        error_file = os.path.join(args.output_dir, "ERROR")
        with open(error_file, 'w') as f:
            f.write(f"ERROR: {str(e)}\n")
        
        sys.exit(1)


if __name__ == "__main__":
    main()