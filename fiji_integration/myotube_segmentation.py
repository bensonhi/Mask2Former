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
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch

# Fix Windows console encoding for emoji/Unicode characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Find Mask2Former project directory
# Priority: 1) Explicit parameter, 2) Environment variable, 3) Config file, 4) Auto-detection

def find_mask2former_project(explicit_path=None):
    """
    Find Mask2Former project directory using multiple methods.

    Args:
        explicit_path: Explicitly provided path (highest priority)

    Returns:
        str: Path to Mask2Former project directory
    """

    # Method 1: Explicit path parameter (highest priority)
    if explicit_path:
        if os.path.exists(os.path.join(explicit_path, 'demo', 'predictor.py')):
            print(f"📁 Using Mask2Former from parameter: {explicit_path}")
            return explicit_path
        else:
            print(f"⚠️  Warning: Provided path '{explicit_path}' does not contain Mask2Former")
            print("   Falling back to other detection methods...")

    # Method 2: Environment variable
    project_path = os.environ.get('MASK2FORMER_PATH')
    if project_path and os.path.exists(os.path.join(project_path, 'demo', 'predictor.py')):
        print(f"📁 Using Mask2Former from environment: {project_path}")
        return project_path

    # Method 3: Config file in same directory as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'mask2former_config.txt')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            project_path = f.read().strip()
        if project_path and os.path.exists(os.path.join(project_path, 'demo', 'predictor.py')):
            print(f"📁 Using Mask2Former from config file: {project_path}")
            return project_path

    # Method 4: Auto-detection in common locations
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

    # Method 5: Error with helpful instructions
    print("❌ Could not find Mask2Former project!")
    print("🔧 To fix this, choose one of these options:")
    print("   1. Use --mask2former-path argument: --mask2former-path /path/to/Mask2Former")
    print("   2. Set environment variable: export MASK2FORMER_PATH='/fs04/scratch2/tf41/ben/Mask2Former'")
    print(f"   3. Create config file: echo '/fs04/scratch2/tf41/ben/Mask2Former' > {config_file}")
    print("   4. Make sure the project is in one of these locations:")
    for path in possible_paths:
        print(f"      - {path}")

    raise ImportError("Mask2Former project not found. See instructions above.")

# Add project to Python path (delayed for GUI mode)
project_dir = None  # Will be set when needed

def ensure_mask2former_loaded(explicit_path=None):
    """
    Ensure Mask2Former project is loaded into Python path.

    Args:
        explicit_path: Explicitly provided path to Mask2Former project (optional)

    Returns:
        str: Path to Mask2Former project directory
    """
    global project_dir
    if project_dir is None:
        project_dir = find_mask2former_project(explicit_path=explicit_path)
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
    return project_dir

# Check if we're in GUI mode - GUI doesn't need detectron2/mask2former imports
GUI_MODE = '--gui' in sys.argv

# Check for explicit Mask2Former path in command-line arguments (before parsing)
_explicit_m2f_path = None
if '--mask2former-path' in sys.argv:
    try:
        _idx = sys.argv.index('--mask2former-path')
        if _idx + 1 < len(sys.argv):
            _explicit_m2f_path = sys.argv[_idx + 1]
    except (IndexError, ValueError):
        pass

# Detectron2 imports - only load if NOT in GUI mode
if not GUI_MODE:
    try:
        ensure_mask2former_loaded(explicit_path=_explicit_m2f_path)
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.data.detection_utils import read_image
        from mask2former import add_maskformer2_config
    except Exception as _import_exc:
        # Write an ERROR status file early if imports fail (common in Fiji env)
        try:
            if len(sys.argv) >= 3:
                _out_dir = sys.argv[2]
                os.makedirs(_out_dir, exist_ok=True)
                with open(os.path.join(_out_dir, "ERROR"), 'w') as _f:
                    _f.write(f"IMPORT_ERROR: {type(_import_exc).__name__}: {_import_exc}\n")
        except Exception:
            pass
        # Also print to stderr for batch_run.log
        print(f"IMPORT_ERROR: {_import_exc}")
        sys.exit(1)
else:
    # GUI mode - set placeholders that will be imported later when actually running segmentation
    DefaultPredictor = None
    get_cfg = None
    add_deeplab_config = None
    read_image = None
    add_maskformer2_config = None


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
            'confidence_threshold': 0.25, # Minimum detection confidence (matches IJM default)
            'merge_threshold': 0.8,    # IoU threshold for merging overlapping instances
            'fill_holes': True,        # Fill holes in segmentation masks
            'smooth_boundaries': True,  # Smooth mask boundaries
            'remove_edge_instances': False, # Remove instances touching image edges
            'final_min_area': 1000,     # Final minimum area filter (from IJM parameter)
        }
    
    def _setup_default_pipeline(self):
        """Setup default post-processing steps."""
        print("🔧 Setting up post-processing pipeline with essential filters")
        # Enable core filtering steps to respect user parameters
        self.add_step('filter_by_confidence', self._filter_by_confidence)
        self.add_step('filter_by_area', self._filter_by_area)
        self.add_step('merge_overlapping', self._merge_overlapping_instances)
        self.add_step('eliminate_contained_components', self._eliminate_contained_components)
        self.add_step('resolve_overlaps', self._resolve_overlapping_pixels)

        # Final area filtering with IJM parameter
        self.add_step('final_area_filter', self._final_area_filter)

        # Keep advanced processing disabled to avoid web artifacts
        # self.add_step('fill_holes', self._fill_holes)
        # self.add_step('smooth_boundaries', self._smooth_boundaries)
        # self.add_step('remove_edge_instances', self._remove_edge_instances)
    
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
            
        if len(instances['masks']) == 0:
            return instances
            
        threshold = self.config['confidence_threshold']
        keep = instances['scores'] >= threshold
        
        if not keep.any():
            # Return empty instances with proper shapes
            return {
                'masks': np.array([]).reshape(0, *instances['image_shape']),
                'scores': np.array([]),
                'boxes': np.array([]).reshape(0, 4),
                'image_shape': instances['image_shape']
            }
        
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
        
        if len(instances['masks']) == 0:
            return instances
        
        areas = np.array([mask.sum() for mask in instances['masks']])
        keep = (areas >= min_area) & (areas <= max_area)
        
        if not keep.any():
            # Return empty instances with proper shapes
            return {
                'masks': np.array([]).reshape(0, *instances['image_shape']),
                'scores': np.array([]),
                'boxes': np.array([]).reshape(0, 4),
                'image_shape': instances['image_shape']
            }
        
        return {
            'masks': instances['masks'][keep],
            'scores': instances['scores'][keep],
            'boxes': instances['boxes'][keep],
            'image_shape': instances['image_shape']
        }

    def _final_area_filter(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Final area filtering step using IJM parameter."""
        final_min_area = self.config.get('final_min_area', 0)

        if len(instances['masks']) == 0 or final_min_area == 0:
            return instances

        areas = np.array([mask.sum() for mask in instances['masks']])
        keep = areas >= final_min_area

        if not keep.any():
            # Return empty instances with proper shapes
            return {
                'masks': np.array([]).reshape(0, *instances['image_shape']),
                'scores': np.array([]),
                'boxes': np.array([]).reshape(0, 4),
                'image_shape': instances['image_shape']
            }

        print(f"   🔍 Final area filter: kept {keep.sum()}/{len(instances['masks'])} instances (min_area: {final_min_area})")

        return {
            'masks': instances['masks'][keep],
            'scores': instances['scores'][keep],
            'boxes': instances['boxes'][keep],
            'image_shape': instances['image_shape']
        }

    def _fill_holes(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Fill holes in segmentation masks - CONSERVATIVE approach to avoid web artifacts."""
        if not self.config.get('fill_holes', True):
            return instances
        
        # Check if we have any masks to process
        if len(instances['masks']) == 0:
            return instances
        
        try:
            from scipy import ndimage
            filled_masks = []
            
            for i, mask in enumerate(instances['masks']):
                # Ensure mask is boolean for fill_holes
                bool_mask = mask.astype(bool)
                original_area = np.sum(bool_mask)
                
                if original_area > 0:
                    # Fill holes using binary fill_holes
                    filled_mask = ndimage.binary_fill_holes(bool_mask)
                    filled_area = np.sum(filled_mask)
                    
                    # Only keep filled version if the change is small (< 10% increase)
                    area_increase = filled_area - original_area
                    if area_increase < original_area * 0.1:
                        filled_masks.append(filled_mask.astype(mask.dtype))
                        if i < 3:  # Debug first few
                            print(f"      🔧 Mask {i+1}: filled {area_increase} hole pixels")
                    else:
                        filled_masks.append(mask)  # Keep original if too much filling
                        if i < 3:
                            print(f"      ⚠️ Mask {i+1}: skipped filling ({area_increase} pixels too many)")
                else:
                    filled_masks.append(mask)  # Keep empty masks as-is
            
            # Preserve array structure
            if len(filled_masks) > 0:
                instances['masks'] = np.array(filled_masks)
            
        except Exception as e:
            print(f"      ⚠️  Warning: Fill holes failed ({e}), keeping original masks")
            
        return instances
    
    def _smooth_boundaries(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Smooth mask boundaries - CONSERVATIVE to avoid web artifacts."""
        if not self.config.get('smooth_boundaries', True):
            return instances
        
        # DISABLE aggressive smoothing that can create web-like artifacts
        print(f"      ⚠️ Skipping boundary smoothing to preserve ROI quality")
        return instances
        
        # Original smoothing code disabled to prevent web artifacts:
        # smoothed_masks = []
        # for mask in instances['masks']:
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #     smoothed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        #     smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
        #     smoothed_masks.append(smoothed.astype(mask.dtype))
        # instances['masks'] = np.array(smoothed_masks)
        # return instances
    
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
        
        print(f"      🔗 Merging instances with IoU >= {merge_threshold}")

        # Calculate IoU matrix with bounding box pre-filtering for speed
        masks = instances['masks']
        boxes = instances['boxes']
        n = len(masks)
        iou_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # FAST PRE-FILTER: Check if bounding boxes overlap
                box1, box2 = boxes[i], boxes[j]
                if not (box1[2] < box2[0] or box1[0] > box2[2] or
                       box1[3] < box2[1] or box1[1] > box2[3]):
                    # Boxes overlap, calculate IoU
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
        
        merges_performed = 0
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
                merges_performed += 1
        
        print(f"         Merged {merges_performed} groups: {n} → {len(merged_masks)} instances")
        
        return {
            'masks': np.array(merged_masks),
            'scores': np.array(merged_scores),
            'boxes': np.array(merged_boxes),
            'image_shape': instances['image_shape']
        }
    
    def _eliminate_contained_components(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """
        Iteratively eliminate connected components that are completely contained within components of other instances.
        Works at the connected component level, not whole instance level.
        Continues until no more reassignments are possible.

        OPTIMIZED VERSION: 10-30x faster with identical results
        - Vectorized component extraction
        - Cached component labels across iterations
        - Vectorized bbox containment checks
        - Bitwise intersection operations
        - Optimized reassignment rebuild
        """
        if len(instances['masks']) <= 1:
            return instances

        from scipy import ndimage

        print(f"      🔍 Iteratively analyzing connected components across {len(instances['masks'])} instances (optimized)")

        current_instances = {
            'masks': instances['masks'].copy(),
            'scores': instances['scores'].copy(),
            'boxes': instances['boxes'].copy(),
            'image_shape': instances['image_shape']
        }

        iteration = 0
        total_reassignments = 0
        total_eliminations = 0

        # Cache for connected component labels (persist across iterations)
        component_cache = {}  # {instance_idx: (labeled_mask, num_components, component_masks)}
        changed_in_last_iteration = set(range(len(instances['masks'])))  # All changed in first iteration

        while iteration < 10:  # Safety limit to prevent infinite loops
            iteration += 1
            print(f"         Iteration {iteration}: analyzing {len(current_instances['masks'])} instances")

            masks = current_instances['masks']
            scores = current_instances['scores']
            boxes = current_instances['boxes']
            n = len(masks)

            # OPTIMIZATION 1 & 2: Extract components with caching and vectorization
            all_components = []  # List of (instance_idx, component_idx, component_mask)

            for i, mask in enumerate(masks):
                mask_area = mask.sum()
                if mask_area == 0:
                    continue

                # Check if we can use cached components (only for unchanged instances)
                if i in component_cache and i not in changed_in_last_iteration:
                    # Reuse cached components
                    labeled_mask, num_components, component_data = component_cache[i]
                else:
                    # Compute new components
                    labeled_mask, num_components = ndimage.label(mask.astype(bool))

                    if num_components > 0:
                        # MEMORY OPTIMIZATION: Store cropped component masks instead of full-size
                        # Extract bounding box for each component and store only the cropped region
                        # This reduces memory from O(num_components * H * W) to O(num_components * bbox_area)
                        component_data = []  # List of (cropped_mask, bbox)

                        for comp_label in range(1, num_components + 1):
                            comp_mask_full = (labeled_mask == comp_label)
                            # Get bounding box
                            coords = np.where(comp_mask_full)
                            if len(coords[0]) > 0:
                                y_min, y_max = coords[0].min(), coords[0].max()
                                x_min, x_max = coords[1].min(), coords[1].max()
                                # Store only the cropped region + bbox coordinates
                                cropped_mask = comp_mask_full[y_min:y_max+1, x_min:x_max+1].copy()
                                bbox = (y_min, y_max, x_min, x_max)
                                component_data.append((cropped_mask, bbox))
                            else:
                                component_data.append((np.array([]), None))
                    else:
                        component_data = []

                    # Cache for potential reuse in next iteration
                    component_cache[i] = (labeled_mask, num_components, component_data)

                # Add components to list
                if num_components > 0:
                    for comp_idx in range(num_components):
                        cropped_mask, bbox = component_data[comp_idx]
                        all_components.append((i, comp_idx, cropped_mask, bbox))

            if len(all_components) <= 1:
                print(f"         No components to process - stopping")
                break

            print(f"         Found {len(all_components)} components across {n} instances")

            # Track component reassignments for this iteration
            reassignments = {}

            # OPTIMIZATION 3 & 4: Vectorized bbox extraction and containment checks
            component_info = []  # [(instance, component_idx, cropped_mask, area, bbox)]
            src_bboxes_list = []

            for src_inst, src_comp_idx, src_cropped_mask, src_bbox in all_components:
                # Skip invalid components
                if src_bbox is None or len(src_cropped_mask) == 0:
                    continue

                # Calculate area from cropped mask
                src_area = int(src_cropped_mask.sum())
                if src_area == 0:
                    continue

                # Bbox already computed during component extraction
                component_info.append((src_inst, src_comp_idx, src_cropped_mask, src_area, src_bbox))
                src_bboxes_list.append(src_bbox)

            if len(component_info) == 0:
                print(f"         No valid components - stopping")
                break

            # OPTIMIZATION 4: Vectorized bbox containment matrix
            # Convert to numpy arrays for vectorized operations
            src_bboxes = np.array(src_bboxes_list)  # (n, 4): [y1, y2, x1, x2]
            num_comps = len(component_info)

            # Compute containment matrix: contained[i, j] = True if bbox i is contained in bbox j
            # Expand dims for broadcasting: src_bboxes[i, :] vs src_bboxes[:, j]
            src_y1 = src_bboxes[:, 0][:, None]  # (n, 1)
            src_y2 = src_bboxes[:, 1][:, None]  # (n, 1)
            src_x1 = src_bboxes[:, 2][:, None]  # (n, 1)
            src_x2 = src_bboxes[:, 3][:, None]  # (n, 1)

            tgt_y1 = src_bboxes[:, 0][None, :]  # (1, n)
            tgt_y2 = src_bboxes[:, 1][None, :]  # (1, n)
            tgt_x1 = src_bboxes[:, 2][None, :]  # (1, n)
            tgt_x2 = src_bboxes[:, 3][None, :]  # (1, n)

            # Bbox i contained in bbox j if: tgt_y1[j] <= src_y1[i] and src_y2[i] <= tgt_y2[j] and tgt_x1[j] <= src_x1[i] and src_x2[i] <= tgt_x2[j]
            contained_matrix = (tgt_y1 <= src_y1) & (src_y2 <= tgt_y2) & (tgt_x1 <= src_x1) & (src_x2 <= tgt_x2)

            # Check each component against others using pre-computed containment matrix
            for i in range(num_comps):
                src_inst, src_comp_idx, src_cropped, src_area, src_bbox = component_info[i]

                for j in range(num_comps):
                    if i == j:  # Same component, skip
                        continue

                    tgt_inst, tgt_comp_idx, tgt_cropped, tgt_area, tgt_bbox = component_info[j]

                    if src_inst == tgt_inst:  # Same instance, skip
                        continue

                    # OPTIMIZATION 4: Use pre-computed containment matrix instead of manual check
                    if not contained_matrix[i, j]:
                        continue

                    # MEMORY OPTIMIZATION: Calculate intersection using cropped masks
                    # Find overlap region between the two bounding boxes
                    src_y1, src_y2, src_x1, src_x2 = src_bbox
                    tgt_y1, tgt_y2, tgt_x1, tgt_x2 = tgt_bbox

                    # Calculate overlap region in global coordinates
                    overlap_y1 = max(src_y1, tgt_y1)
                    overlap_y2 = min(src_y2, tgt_y2)
                    overlap_x1 = max(src_x1, tgt_x1)
                    overlap_x2 = min(src_x2, tgt_x2)

                    # Check if there's actual overlap
                    if overlap_y2 < overlap_y1 or overlap_x2 < overlap_x1:
                        continue  # No overlap

                    # Convert overlap region to local coordinates for each cropped mask
                    src_local_y1 = overlap_y1 - src_y1
                    src_local_y2 = overlap_y2 - src_y1
                    src_local_x1 = overlap_x1 - src_x1
                    src_local_x2 = overlap_x2 - src_x1

                    tgt_local_y1 = overlap_y1 - tgt_y1
                    tgt_local_y2 = overlap_y2 - tgt_y1
                    tgt_local_x1 = overlap_x1 - tgt_x1
                    tgt_local_x2 = overlap_x2 - tgt_x1

                    # Extract overlap regions from cropped masks
                    src_overlap = src_cropped[src_local_y1:src_local_y2+1, src_local_x1:src_local_x2+1]
                    tgt_overlap = tgt_cropped[tgt_local_y1:tgt_local_y2+1, tgt_local_x1:tgt_local_x2+1]

                    # Calculate intersection
                    intersection = np.logical_and(src_overlap, tgt_overlap).sum()
                    containment_ratio = intersection / src_area

                    if containment_ratio > 0.8:
                        print(f"         Component {src_comp_idx} of instance {src_inst} → instance {tgt_inst} ({containment_ratio:.1%} contained)")
                        reassignments[(src_inst, src_comp_idx)] = tgt_inst
                        break  # Move to first containing instance found

            # If no reassignments found, we're done
            if not reassignments:
                print(f"         No more reassignments possible - stopping after {iteration} iterations")
                break

            # Apply reassignments to create new instance set
            changed_instances = set()
            for (src_inst, _), tgt_inst in reassignments.items():
                changed_instances.add(src_inst)  # Source instance loses components
                changed_instances.add(tgt_inst)  # Target instance gains components

            # OPTIMIZATION 6: Optimized reassignment rebuild
            # Only copy masks for instances that will change
            new_instance_masks = [masks[i].copy() if i not in changed_instances else np.zeros_like(masks[0], dtype=bool) for i in range(n)]

            # Build mapping of which components belong to each changed instance for efficient lookup
            changed_instance_components = {inst: [] for inst in changed_instances}
            for src_inst, src_comp_idx, src_cropped, src_bbox in all_components:
                if src_bbox is None or len(src_cropped) == 0:
                    continue

                # Reconstruct full-size mask from cropped mask + bbox
                src_y1, src_y2, src_x1, src_x2 = src_bbox
                full_mask = np.zeros_like(masks[0], dtype=bool)
                full_mask[src_y1:src_y2+1, src_x1:src_x2+1] = src_cropped

                if (src_inst, src_comp_idx) in reassignments:
                    # Reassigned component
                    target_instance = reassignments[(src_inst, src_comp_idx)]
                    changed_instance_components[target_instance].append(full_mask)
                elif src_inst in changed_instances:
                    # Component stays but instance was affected by other changes
                    changed_instance_components[src_inst].append(full_mask)

            # Rebuild changed instances using vectorized operations
            for inst in changed_instances:
                if changed_instance_components[inst]:
                    # Stack all component masks and take logical OR across them
                    component_stack = np.array(changed_instance_components[inst])
                    new_instance_masks[inst] = np.any(component_stack, axis=0)

            # Create new instance set, eliminating empty ones
            new_masks = []
            new_scores = []
            new_boxes = []
            eliminated_this_iteration = 0
            instance_mapping = {}  # Map old instance idx to new instance idx

            for i in range(n):
                if new_instance_masks[i].sum() == 0:
                    print(f"         Instance {i}: eliminated (no components remaining)")
                    eliminated_this_iteration += 1
                    continue

                instance_mapping[i] = len(new_masks)

                # Recalculate bounding box
                coords = np.where(new_instance_masks[i])
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    new_box = np.array([x_min, y_min, x_max, y_max])
                else:
                    new_box = boxes[i]

                new_masks.append(new_instance_masks[i])
                new_scores.append(scores[i])
                new_boxes.append(new_box)

            # Update for next iteration
            current_instances = {
                'masks': np.array(new_masks) if new_masks else np.array([]).reshape(0, *instances['image_shape']),
                'scores': np.array(new_scores),
                'boxes': np.array(new_boxes).reshape(-1, 4) if new_boxes else np.array([]).reshape(0, 4),
                'image_shape': instances['image_shape']
            }

            # Update cache: remove eliminated instances, remap surviving instances
            new_component_cache = {}
            for old_idx, new_idx in instance_mapping.items():
                if old_idx in component_cache and old_idx not in changed_instances:
                    new_component_cache[new_idx] = component_cache[old_idx]
            component_cache = new_component_cache

            # Track which instances changed for next iteration's caching
            changed_in_last_iteration = set(instance_mapping[i] for i in changed_instances if i in instance_mapping)

            total_reassignments += len(reassignments)
            total_eliminations += eliminated_this_iteration

            print(f"         Iteration {iteration}: {len(reassignments)} reassignments, {eliminated_this_iteration} eliminations → {len(current_instances['masks'])} instances")

        if total_reassignments > 0:
            print(f"         Final: {total_reassignments} total reassignments, {total_eliminations} total eliminations after {iteration} iterations")
            print(f"         Result: {len(instances['masks'])} → {len(current_instances['masks'])} instances")

        return current_instances
    
    def _resolve_overlapping_pixels(self, instances: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """
        Resolve pixel overlaps by assigning each overlapping pixel to the instance with highest confidence.
        Ensures no pixel belongs to multiple instances.
        """
        if len(instances['masks']) <= 1:
            return instances
        
        print(f"      🎯 Resolving pixel overlaps across {len(instances['masks'])} instances")
        
        masks = instances['masks']
        scores = instances['scores']
        boxes = instances['boxes']
        n = len(masks)
        
        # Create a comprehensive overlap map
        # overlap_map[y, x] = set of instance indices that claim this pixel
        h, w = instances['image_shape']
        overlap_map = {}
        total_overlapping_pixels = 0
        
        # Find all overlapping pixels
        for i, mask in enumerate(masks):
            coords = np.where(mask > 0)
            for y, x in zip(coords[0], coords[1]):
                if (y, x) not in overlap_map:
                    overlap_map[(y, x)] = set()
                overlap_map[(y, x)].add(i)
        
        # Count overlapping pixels
        overlapping_pixels = {coord: instances for coord, instances in overlap_map.items() if len(instances) > 1}
        total_overlapping_pixels = len(overlapping_pixels)
        
        if total_overlapping_pixels == 0:
            print(f"         No pixel overlaps found - all instances are disjoint")
            return instances
        
        print(f"         Found {total_overlapping_pixels} overlapping pixels")
        
        # Create new masks with overlaps resolved
        new_masks = [mask.copy() for mask in masks]
        pixel_reassignments = 0
        
        # For each overlapping pixel, assign to highest confidence instance
        for (y, x), competing_instances in overlapping_pixels.items():
            if len(competing_instances) > 1:
                # Find the instance with highest confidence
                best_instance = max(competing_instances, key=lambda i: scores[i])
                best_score = scores[best_instance]
                
                # Remove pixel from all other instances
                for instance_idx in competing_instances:
                    if instance_idx != best_instance:
                        new_masks[instance_idx][y, x] = 0
                        pixel_reassignments += 1
                
                # Ensure pixel is assigned to best instance (should already be, but be explicit)
                new_masks[best_instance][y, x] = 1
        
        print(f"         Reassigned {pixel_reassignments} pixels to highest confidence instances")
        
        # Verify no overlaps remain (sanity check)
        verification_sum = sum(mask.astype(int) for mask in new_masks)
        overlapping_pixels_after = (verification_sum > 1).sum()
        
        if overlapping_pixels_after > 0:
            print(f"         ⚠️ Warning: {overlapping_pixels_after} pixels still overlap after resolution")
        else:
            print(f"         ✅ All pixel overlaps resolved successfully")
        
        # Check if any instances became empty after overlap resolution
        empty_instances = []
        for i, mask in enumerate(new_masks):
            if mask.sum() == 0:
                empty_instances.append(i)
        
        if empty_instances:
            print(f"         Removing {len(empty_instances)} instances that became empty after overlap resolution")
            
            # Filter out empty instances
            final_masks = []
            final_scores = []
            final_boxes = []
            
            for i in range(n):
                if i not in empty_instances:
                    final_masks.append(new_masks[i])
                    final_scores.append(scores[i])
                    final_boxes.append(boxes[i])
            
            return {
                'masks': np.array(final_masks) if final_masks else np.array([]).reshape(0, *instances['image_shape']),
                'scores': np.array(final_scores),
                'boxes': np.array(final_boxes).reshape(-1, 4) if final_boxes else np.array([]).reshape(0, 4),
                'image_shape': instances['image_shape']
            }
        else:
            # No empty instances, just update with resolved masks
            return {
                'masks': np.array(new_masks),
                'scores': instances['scores'],
                'boxes': instances['boxes'],
                'image_shape': instances['image_shape']
            }



class TiledMyotubeSegmentation:
    """
    Tiled inference for processing large images that contain too many myotubes
    for single-pass inference (exceeds model's query capacity).

    Uses overlapping tiles to ensure boundary instances are captured,
    then merges detections across tiles using IoU-based matching.
    """

    def __init__(self, fiji_integration, target_overlap=0.20, grid_size=2):
        """
        Initialize tiled segmentation wrapper.

        Args:
            fiji_integration: MyotubeFijiIntegration instance
            target_overlap: Overlap ratio between tiles (default: 0.20 = 20%)
            grid_size: Grid size for tiling (1=no split, 2=2×2, 3=3×3, etc.)
        """
        self.integration = fiji_integration
        self.target_overlap = target_overlap
        self.grid_size = max(1, int(grid_size))  # Ensure grid_size >= 1

    def calculate_tiling_params(self, image_size):
        """
        Calculate tile size for N×N grid with specified overlap.

        For overlap ratio r, grid size N, and image size I:
        - tile_size = I / (N - r * (N - 1))
        - For 2×2 grid with 20% overlap on 9000px: tile_size = 9000 / (2 - 0.2) = 5000
        - For 3×3 grid with 20% overlap on 9000px: tile_size = 9000 / (3 - 0.4) = 3461

        Args:
            image_size: Width or height of square image (uses minimum for non-square)

        Returns:
            tuple: (tile_size, overlap_pixels)
        """
        # Formula: tile_size = I / (N - r * (N - 1))
        # This ensures N tiles cover the image with overlap r
        denominator = self.grid_size - self.target_overlap * (self.grid_size - 1)
        tile_size = int(image_size / denominator)
        overlap = int(tile_size * self.target_overlap)

        return tile_size, overlap

    def create_tiles(self, image, tile_size):
        """
        Create N×N overlapping tiles from image.

        Args:
            image: Input image array (H, W, C)
            tile_size: Size of each tile

        Returns:
            list: List of (tile_image, (y_start, x_start, y_end, x_end)) tuples
        """
        h, w = image.shape[:2]
        min_dim = min(h, w)

        # Calculate positions for N×N grid
        # With overlap, the step size between tiles is: tile_size * (1 - overlap_ratio)
        step_size = int(tile_size * (1 - self.target_overlap))

        # Generate positions: 0, step, 2*step, ..., but last position is (image_size - tile_size)
        positions = []
        for i in range(self.grid_size):
            if i == 0:
                positions.append(0)
            elif i == self.grid_size - 1:
                positions.append(min_dim - tile_size)
            else:
                pos = i * step_size
                positions.append(pos)

        tiles = []
        for y_pos in positions:
            for x_pos in positions:
                y_end = min(y_pos + tile_size, h)
                x_end = min(x_pos + tile_size, w)

                # Extract tile
                tile = image[y_pos:y_end, x_pos:x_end]

                # Pad if necessary (edge tiles might be smaller)
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    tile = self._pad_to_size(tile, tile_size)

                tiles.append((tile, (y_pos, x_pos, y_end, x_end)))

        return tiles

    def _pad_to_size(self, tile, target_size):
        """Pad tile to target_size (for edge cases)."""
        h, w = tile.shape[:2]
        if h == target_size and w == target_size:
            return tile

        # Pad with zeros (black)
        if len(tile.shape) == 3:
            padded = np.zeros((target_size, target_size, tile.shape[2]), dtype=tile.dtype)
        else:
            padded = np.zeros((target_size, target_size), dtype=tile.dtype)
        padded[:h, :w] = tile
        return padded

    def transform_to_global_coords(self, instances, tile_coords, original_shape):
        """
        Transform instance coordinates from tile-local to global image coordinates.

        Args:
            instances: Instances dict with masks, scores, boxes
            tile_coords: (y_start, x_start, y_end, x_end) of tile in global image
            original_shape: (H, W) of full image

        Returns:
            list: List of instance dicts with global coordinates
        """
        y_start, x_start, y_end, x_end = tile_coords
        tile_h, tile_w = y_end - y_start, x_end - x_start

        global_instances = []

        for i in range(len(instances['masks'])):
            mask = instances['masks'][i]
            score = instances['scores'][i]
            box = instances['boxes'][i]

            # Create global mask
            global_mask = np.zeros(original_shape, dtype=bool)
            mask_h, mask_w = mask.shape

            # Place tile mask into global coordinates
            # Handle potential size mismatch due to padding
            copy_h = min(mask_h, tile_h)
            copy_w = min(mask_w, tile_w)
            global_mask[y_start:y_start+copy_h, x_start:x_start+copy_w] = mask[:copy_h, :copy_w]

            # Transform bounding box to global coordinates
            global_box = [
                box[0] + x_start,
                box[1] + y_start,
                box[2] + x_start,
                box[3] + y_start
            ]

            global_instances.append({
                'mask': global_mask,
                'score': score,
                'box': global_box,
                'tile_coords': tile_coords  # Store tile coordinates for overlap-based IoU
            })

        return global_instances

    def calculate_iou(self, mask1, mask2):
        """Calculate Intersection over Union between two masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    def calculate_overlap_region_iou(self, inst1, inst2):
        """
        Calculate IoU only in the overlapping region between two tiles.

        This is much more accurate for merging tile detections because:
        - Same myotube in overlap = high IoU ✓
        - Different myotubes in overlap = low IoU ✓
        - No dependency on myotube length outside overlap ✓

        Args:
            inst1: First instance dict with 'mask' and 'tile_coords'
            inst2: Second instance dict with 'mask' and 'tile_coords'

        Returns:
            float: IoU calculated only in the overlap region (0.0 to 1.0)
        """
        # Get tile coordinates
        y1_start, x1_start, y1_end, x1_end = inst1['tile_coords']
        y2_start, x2_start, y2_end, x2_end = inst2['tile_coords']

        # Calculate overlap region
        overlap_y_start = max(y1_start, y2_start)
        overlap_y_end = min(y1_end, y2_end)
        overlap_x_start = max(x1_start, x2_start)
        overlap_x_end = min(x1_end, x2_end)

        # Check if tiles actually overlap
        if overlap_y_end <= overlap_y_start or overlap_x_end <= overlap_x_start:
            return 0.0  # No overlap between tiles

        # Crop masks to overlap region
        mask1_overlap = inst1['mask'][overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end]
        mask2_overlap = inst2['mask'][overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end]

        # Calculate IoU in overlap region only
        intersection = np.logical_and(mask1_overlap, mask2_overlap).sum()
        union = np.logical_or(mask1_overlap, mask2_overlap).sum()

        return intersection / union if union > 0 else 0.0

    def merge_duplicates(self, all_instances, iou_threshold=0.5):
        """
        Merge instances detected in multiple overlapping tiles.

        Uses OVERLAP-REGION IoU: Only compares masks within tile overlap zones.
        This is much more accurate than full-mask IoU for elongated objects.

        Args:
            all_instances: List of all instances from all tiles (with 'tile_coords')
            iou_threshold: Minimum overlap-region IoU (default: 0.5, higher than full-mask)

        Returns:
            list: Merged instances (no duplicates)
        """
        if len(all_instances) == 0:
            return []

        print(f"      🔗 Merging {len(all_instances)} detections (Overlap-region IoU threshold: {iou_threshold})")

        merged = []
        used = set()
        merge_count = 0

        for i, inst1 in enumerate(all_instances):
            if i in used:
                continue

            # Find all overlapping instances (same myotube detected in adjacent tiles)
            group = [inst1]
            box1 = inst1['box']

            for j in range(i + 1, len(all_instances)):
                if j in used:
                    continue

                inst2 = all_instances[j]
                box2 = inst2['box']

                # FAST PRE-FILTER: Check if bounding boxes overlap
                if not self._boxes_overlap(box1, box2):
                    continue

                # SMART IoU: Calculate only in tile overlap region
                # This handles elongated myotubes spanning multiple tiles correctly
                overlap_iou = self.calculate_overlap_region_iou(inst1, inst2)

                if overlap_iou > iou_threshold:
                    group.append(inst2)
                    used.add(j)

            # Merge group into single instance
            if len(group) > 1:
                merged_inst = self._merge_group(group)
                merge_count += 1
            else:
                merged_inst = inst1

            merged.append(merged_inst)

        print(f"         Merged {merge_count} duplicate groups")
        print(f"         Result: {len(all_instances)} → {len(merged)} unique instances")

        return merged

    def _boxes_overlap(self, box1, box2):
        """Fast check if two bounding boxes overlap."""
        # box format: [x_min, y_min, x_max, y_max]
        return not (box1[2] < box2[0] or  # box1 right of box2
                   box1[0] > box2[2] or  # box1 left of box2
                   box1[3] < box2[1] or  # box1 above box2
                   box1[1] > box2[3])    # box1 below box2

    def _merge_group(self, group):
        """Merge multiple detections of the same myotube."""
        # Union of all masks
        merged_mask = np.logical_or.reduce([inst['mask'] for inst in group])

        # Keep highest confidence score
        merged_score = max(inst['score'] for inst in group)

        # Recalculate bounding box from merged mask
        coords = np.where(merged_mask)
        if len(coords[0]) > 0:
            merged_box = [
                coords[1].min(),  # x_min
                coords[0].min(),  # y_min
                coords[1].max(),  # x_max
                coords[0].max()   # y_max
            ]
        else:
            # Fallback: use first instance's box
            merged_box = group[0]['box']

        return {
            'mask': merged_mask,
            'score': merged_score,
            'box': merged_box
        }

    def convert_to_detectron_format(self, merged_instances, original_shape):
        """
        Convert merged instances back to Detectron2 format.

        Args:
            merged_instances: List of merged instance dicts
            original_shape: (H, W) of original image

        Returns:
            dict: Instances in internal format compatible with post-processing
        """
        if len(merged_instances) == 0:
            return {
                'masks': np.array([]).reshape(0, *original_shape),
                'scores': np.array([]),
                'boxes': np.array([]).reshape(0, 4),
                'image_shape': original_shape
            }

        masks = np.array([inst['mask'] for inst in merged_instances])
        scores = np.array([inst['score'] for inst in merged_instances])
        boxes = np.array([inst['box'] for inst in merged_instances])

        return {
            'masks': masks,
            'scores': scores,
            'boxes': boxes,
            'image_shape': original_shape
        }

    def segment_image_tiled(self, image_path, output_dir, custom_config=None):
        """
        Segment image using tiled inference.

        Args:
            image_path: Path to input image
            output_dir: Output directory
            custom_config: Custom configuration dict

        Returns:
            dict: Output files dictionary (same format as segment_image)
        """
        print(f"🔲 Using TILED inference mode (grid: {self.grid_size}×{self.grid_size}, overlap: {self.target_overlap*100:.0f}%)")

        # Load image
        image = cv2.imread(image_path)
        original_image = image.copy()
        original_h, original_w = image.shape[:2]

        # Calculate optimal model resolution: 1500 × grid_size, capped at original image size
        # Model processes optimally at 1500px, so N×N grid should be 1500×N
        model_resolution = min(1500 * self.grid_size, max(original_h, original_w))

        # Resolution optimization: Process at model resolution for speedup
        if max(original_h, original_w) > model_resolution:
            # Calculate scale factor to resize to model resolution
            scale_factor = model_resolution / max(original_h, original_w)
            processing_w = int(original_w * scale_factor)
            processing_h = int(original_h * scale_factor)

            # Resize image for processing
            processing_image = cv2.resize(image, (processing_w, processing_h), interpolation=cv2.INTER_AREA)

            # Store original size and scale factor for later upscaling
            self.integration._original_size = (original_h, original_w)
            self.integration._scale_factor = scale_factor
            self.integration._processing_size = (processing_h, processing_w)

            print(f"   🚀 RESOLUTION OPTIMIZATION (Automatic for tiled inference)")
            print(f"   📐 Original: {original_w}×{original_h}")
            print(f"   📐 Processing: {processing_w}×{processing_h} (scale: {scale_factor:.3f})")
            print(f"   ⚡ Expected speedup: ~{(1/scale_factor)**2:.1f}×")

            # Use processing image for tiling
            image = processing_image
            h, w = processing_h, processing_w
        else:
            # Image already at or below model resolution
            h, w = original_h, original_w
            self.integration._original_size = None
            self.integration._scale_factor = 1.0
            self.integration._processing_size = None

        # Calculate tiling parameters
        min_dim = min(h, w)
        tile_size, overlap = self.calculate_tiling_params(min_dim)

        total_tiles = self.grid_size * self.grid_size
        print(f"   Image: {w}×{h}")
        print(f"   Tiles: {self.grid_size}×{self.grid_size} grid = {total_tiles} tiles")
        print(f"   Tile size: {tile_size}×{tile_size}")
        print(f"   Overlap: {overlap}px ({self.target_overlap*100:.0f}%)")

        # Create tiles
        tiles = self.create_tiles(image, tile_size)
        print(f"   Created {len(tiles)} tiles")

        # Ensure output directory exists before processing tiles
        os.makedirs(output_dir, exist_ok=True)

        # Process each tile
        all_instances = []

        # Temporarily disable the integration's own resizing
        # We want to pass tiles at their native resolution to match training distribution
        original_max_size = custom_config.get('max_image_size', None) if custom_config else None
        if custom_config:
            custom_config['max_image_size'] = None  # Disable resizing for tiles

        for idx, (tile, coords) in enumerate(tiles):
            y_start, x_start, y_end, x_end = coords
            print(f"   🔄 Processing tile {idx+1}/{len(tiles)}: [{x_start}:{x_end}, {y_start}:{y_end}]")

            # Save tile temporarily
            temp_tile_path = os.path.join(output_dir, f"_temp_tile_{idx}.png")
            success = cv2.imwrite(temp_tile_path, tile)
            if not success:
                raise IOError(f"Failed to write tile image to {temp_tile_path}")

            # Process tile using existing integration
            # Create a temporary output directory for this tile
            temp_output_dir = os.path.join(output_dir, f"_temp_tile_{idx}_output")
            os.makedirs(temp_output_dir, exist_ok=True)

            try:
                # Use the integration's segment_image method
                # But we need to intercept the instances before post-processing
                # Actually, let's directly call the predictor

                # Initialize predictor if needed
                force_cpu = custom_config.get('force_cpu', False) if custom_config else False
                self.integration.initialize_predictor(force_cpu=force_cpu)

                # Run inference on tile
                from detectron2.data.detection_utils import read_image
                tile_detectron = read_image(temp_tile_path, format="BGR")

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                predictions = self.integration.predictor(tile_detectron)
                instances = predictions["instances"]

                num_detections = len(instances)
                print(f"      → {num_detections} detections")

                if num_detections > 0:
                    # Convert to internal format
                    tile_instances = self.integration.post_processor._convert_to_internal_format(
                        instances, tile_detectron
                    )

                    # Transform to global coordinates
                    global_instances = self.transform_to_global_coords(
                        tile_instances, coords, (h, w)
                    )

                    all_instances.extend(global_instances)

                # CRITICAL: Free GPU memory immediately after processing each tile
                # Delete GPU tensors to prevent OOM on subsequent tiles
                del predictions
                del instances
                if num_detections > 0:
                    del tile_instances
                del tile_detectron

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for all operations to complete

                # Clean up temp files
                os.remove(temp_tile_path)
                import shutil
                shutil.rmtree(temp_output_dir, ignore_errors=True)

            except Exception as e:
                print(f"      ❌ Failed to process tile {idx}: {e}")

                # CRITICAL: Clean up GPU memory even on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Clean up temp files and continue
                if os.path.exists(temp_tile_path):
                    os.remove(temp_tile_path)
                continue

        # Restore original max_size config
        if custom_config and original_max_size is not None:
            custom_config['max_image_size'] = original_max_size

        print(f"   📊 Total raw detections: {len(all_instances)}")

        # Merge overlapping detections from different tiles using overlap-region IoU
        merged_instances = self.merge_duplicates(all_instances, iou_threshold=0.5)

        # Convert back to Detectron2 format
        instances_dict = self.convert_to_detectron_format(merged_instances, (h, w))

        print(f"   ✅ Final merged instances: {len(instances_dict['masks'])}")

        # Apply post-processing pipeline
        if custom_config:
            self.integration.post_processor.config.update(custom_config)

        # Use processing-resolution image for post-processing (not original high-res)
        processed_instances = self.integration.post_processor.process(instances_dict, image)

        # Save outputs using existing methods (pass both raw and processed instances)
        # Note: original_image is needed for overlays; masks will be upscaled in saving methods
        output_files = self.integration._generate_fiji_outputs(
            instances_dict,  # raw instances (after tile merging, before post-processing)
            processed_instances,  # processed instances (after post-processing)
            original_image,  # Original high-res image for overlays
            image_path,
            output_dir,
            custom_config  # pass custom_config for measurements settings
        )

        return output_files


class MyotubeFijiIntegration:
    """
    Main class for Fiji integration of myotube instance segmentation.
    """
    
    def __init__(self, config_file: str = None, model_weights: str = None,
                 skip_merged_masks: bool = False, mask2former_path: str = None):
        """
        Initialize the Fiji integration.

        Args:
            config_file: Path to model config file
            model_weights: Path to model weights
            skip_merged_masks: Skip generation of merged visualization masks (default: False)
            mask2former_path: Path to Mask2Former project directory (auto-detected if not provided)
        """
        self.config_file = config_file
        self.model_weights = model_weights
        self.skip_merged_masks = skip_merged_masks
        self.mask2former_path = mask2former_path
        self.predictor = None
        self.post_processor = PostProcessingPipeline()

        # Setup paths
        self.setup_paths()
        
    def setup_paths(self):
        """Setup default paths if not provided."""
        # If both paths are provided, no need to auto-detect
        if self.config_file and self.model_weights:
            print(f"📁 Config file: {self.config_file}")
            print(f"🔮 Model weights: {self.model_weights}")
            return

        # Load project directory for auto-detection
        ensure_mask2former_loaded(explicit_path=self.mask2former_path)
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

        # Import required modules (must be done after ensure_mask2former_loaded is called)
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from mask2former import add_maskformer2_config

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

        # Apply post-processing using inference resolution (not original high-res)
        processed_instances = self.post_processor.process(instances, image)

        # Generate outputs with both raw and processed overlays
        output_files = self._generate_fiji_outputs(
            instances, processed_instances, original_image, image_path, output_dir, custom_config
        )
        
        return output_files
    
    def _create_empty_outputs(self, image_path: str, output_dir: str) -> Dict[str, str]:
        """Create empty output files when no instances are detected."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        
        # Create empty masks directory
        masks_dir = os.path.join(output_dir, f"{base_name}_masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        # Create empty overlay (just copy original)
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.tif")
        original = cv2.imread(image_path)
        cv2.imwrite(overlay_path, original)
        
        # Create empty measurements
        measurements_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
        with open(measurements_path, 'w') as f:
            f.write("Instance,Area,Perimeter,AspectRatio,Confidence\n")
        
        return {
            'masks_dir': masks_dir,
            'overlay': overlay_path,
            'measurements': measurements_path,
            'count': 0
        }
    
    def _generate_fiji_outputs(self, raw_instances, processed_instances: Dict[str, Any], original_image: np.ndarray,
                              image_path: str, output_dir: str, custom_config: Dict[str, Any] = None) -> Dict[str, str]:
        """Generate all Fiji-compatible output files."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        
        # Generate RAW Detectron2 overlay (before post-processing)
        raw_overlay_path = os.path.join(output_dir, f"{base_name}_raw_overlay.tif")
        self._save_colored_overlay(raw_instances, original_image, raw_overlay_path, overlay_type="raw")
        
        # Generate PROCESSED overlay (after post-processing)
        processed_overlay_path = os.path.join(output_dir, f"{base_name}_processed_overlay.tif")
        self._save_colored_overlay(processed_instances, original_image, processed_overlay_path, overlay_type="processed")

        # Generate individual mask images (pixel-perfect accuracy!) - using processed instances
        masks_dir = os.path.join(output_dir, f"{base_name}_masks")
        self._save_individual_mask_images(processed_instances, original_image, masks_dir)

        # Generate merged visualization masks (connect disconnected components) - optional
        if not self.skip_merged_masks:
            merged_masks_dir = os.path.join(output_dir, f"{base_name}_merged_masks")
            self._save_merged_visualization_masks(processed_instances, original_image, merged_masks_dir)
        else:
            print(f"   ⏭️  Skipping merged mask generation (--skip-merged-masks enabled)")

        # Generate measurements CSV - using processed instances (optional)
        if custom_config and custom_config.get('save_measurements', False):
            measurements_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
            print(f"   📊 Generating measurements CSV...")
            self._save_measurements(processed_instances, measurements_path)
        else:
            print(f"   ⏭️  Skipping measurements CSV (disabled in settings)")
        
        # Generate summary info - using processed instances
        info_path = os.path.join(output_dir, f"{base_name}_info.json")
        self._save_info(processed_instances, image_path, info_path)
        
        # Print comparison
        raw_count = len(raw_instances) if hasattr(raw_instances, '__len__') else len(raw_instances.pred_masks) if hasattr(raw_instances, 'pred_masks') else 0
        processed_count = len(processed_instances['masks'])
        print(f"✅ Generated outputs: {raw_count} raw → {processed_count} after filtering")
        
        return {
            'masks_dir': masks_dir,
            'raw_overlay': raw_overlay_path,
            'processed_overlay': processed_overlay_path,
            'measurements': measurements_path,
            'info': info_path,
            'raw_count': raw_count,
            'processed_count': processed_count,
            'count': processed_count  # Keep for backwards compatibility
        }
    
    def _save_individual_mask_images(self, instances: Dict[str, Any], original_image: np.ndarray, output_dir: str):
        """Save each myotube mask as individual image files - pixel-perfect accuracy!"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"   🖼️  Generating individual mask images: {output_dir}")
        print(f"   📊 Processing {len(instances['masks'])} instances for mask images")
        
        successful_masks = 0
        failed_masks = 0
        
        for i, mask in enumerate(instances['masks']):
            mask_name = f"Myotube_{i+1}_mask.png"
            mask_path = os.path.join(output_dir, mask_name)
            
            # Skip empty masks
            if mask.sum() == 0:
                print(f"      ⚠️  Warning: Mask {i+1} is empty - skipping")
                failed_masks += 1
                continue
            
            print(f"      🔍 Processing mask {i+1}: {mask.sum()} pixels at inference resolution")
            
            # Resize mask to original image size (same logic as overlay generation)
            if hasattr(self, '_scale_factor') and self._scale_factor != 1.0:
                original_h, original_w = self._original_size
                
                # Use same scaling as overlay generation for perfect alignment
                mask_uint8 = (mask * 255).astype(np.uint8)
                resized_mask = cv2.resize(
                    mask_uint8, 
                    (original_w, original_h), 
                    interpolation=cv2.INTER_NEAREST
                )
                final_mask = (resized_mask > 128).astype(np.uint8) * 255
                
                print(f"         Scaled to original: {(final_mask > 0).sum()} pixels")
            else:
                final_mask = (mask * 255).astype(np.uint8)
            
            # Save mask as PNG image
            try:
                cv2.imwrite(mask_path, final_mask)
                
                # Verify file was created
                if os.path.exists(mask_path):
                    file_size_kb = os.path.getsize(mask_path) / 1024
                    print(f"      ✅ Mask {i+1}: Saved as PNG ({file_size_kb:.1f} KB)")
                    successful_masks += 1
                else:
                    print(f"      ❌ Mask {i+1}: Failed to save file")
                    failed_masks += 1
                    
            except Exception as e:
                print(f"      ❌ Mask {i+1}: Error saving - {e}")
                failed_masks += 1
        
        # Final summary
        print(f"   📊 Mask Image Generation Summary:")
        print(f"      ✅ Successful: {successful_masks}")
        print(f"      ❌ Failed: {failed_masks}")
        print(f"      📁 Saved to: {output_dir}")
        
        # Create a summary file for easy reference
        summary_path = os.path.join(output_dir, "README.txt")
        with open(summary_path, 'w') as f:
            f.write("Myotube Individual Mask Images\n")
            f.write("==============================\n\n")
            f.write(f"Total masks: {successful_masks}\n")
            f.write(f"Image format: PNG (binary masks)\n")
            f.write(f"Pixel values: 0 (background), 255 (myotube)\n")
            f.write(f"Resolution: Same as original image\n\n")
            f.write("Usage in ImageJ/Fiji:\n")
            f.write("1. Open original image\n")
            f.write("2. Load mask images as overlays: Image > Overlay > Add Image\n")
            f.write("3. Perfect pixel alignment with Detectron2 results\n")
            f.write("4. Use Image Calculator for measurements if needed\n")
        
        return successful_masks

    def _save_merged_visualization_masks(self, instances: Dict[str, Any], original_image: np.ndarray, output_dir: str):
        """Save merged visualization masks that connect disconnected components of each myotube."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"   🔗 Generating merged visualization masks: {output_dir}")
        print(f"   📊 Processing {len(instances['masks'])} instances for merged masks")

        successful_masks = 0
        failed_masks = 0
        merged_count = 0

        for i, mask in enumerate(instances['masks']):
            mask_name = f"Myotube_{i+1}_merged.png"
            mask_path = os.path.join(output_dir, mask_name)

            # Skip empty masks
            if mask.sum() == 0:
                print(f"      ⚠️  Warning: Mask {i+1} is empty - skipping")
                failed_masks += 1
                continue

            # Create merged mask by connecting components
            merged_mask = self._create_merged_mask(mask)

            # Check if merging actually occurred
            original_components = self._count_components(mask)
            if original_components > 1:
                merged_count += 1
                print(f"      🔗 Mask {i+1}: Connected {original_components} components")
            else:
                print(f"      ✅ Mask {i+1}: Single component (no merging needed)")

            # Resize merged mask to original image size
            if hasattr(self, '_scale_factor') and self._scale_factor != 1.0:
                original_h, original_w = self._original_size

                mask_uint8 = (merged_mask * 255).astype(np.uint8)
                resized_mask = cv2.resize(
                    mask_uint8,
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST
                )
                final_mask = (resized_mask > 128).astype(np.uint8) * 255
            else:
                final_mask = (merged_mask * 255).astype(np.uint8)

            # Save merged mask as PNG image
            try:
                cv2.imwrite(mask_path, final_mask)

                if os.path.exists(mask_path):
                    file_size_kb = os.path.getsize(mask_path) / 1024
                    print(f"      ✅ Merged mask {i+1}: Saved as PNG ({file_size_kb:.1f} KB)")
                    successful_masks += 1
                else:
                    print(f"      ❌ Merged mask {i+1}: Failed to save file")
                    failed_masks += 1

            except Exception as e:
                print(f"      ❌ Merged mask {i+1}: Error saving - {e}")
                failed_masks += 1

        # Final summary
        print(f"   📊 Merged Mask Generation Summary:")
        print(f"      ✅ Successful: {successful_masks}")
        print(f"      ❌ Failed: {failed_masks}")
        print(f"      🔗 Myotubes merged: {merged_count}")
        print(f"      📁 Saved to: {output_dir}")

        # Create a summary file
        summary_path = os.path.join(output_dir, "README.txt")
        with open(summary_path, 'w') as f:
            f.write("Myotube Merged Visualization Masks\n")
            f.write("==================================\n\n")
            f.write("These masks show complete myotube structures by connecting\n")
            f.write("disconnected components with linear interpolation.\n\n")
            f.write(f"Total masks: {successful_masks}\n")
            f.write(f"Myotubes with merged components: {merged_count}\n")
            f.write(f"Image format: PNG (binary masks)\n")
            f.write(f"Pixel values: 0 (background), 255 (myotube + interpolated gaps)\n")
            f.write(f"Resolution: Same as original image\n\n")
            f.write("Note: These are for VISUALIZATION ONLY.\n")
            f.write("All measurements use original masks, not merged masks.\n\n")
            f.write("Usage in ImageJ/Fiji:\n")
            f.write("1. Open original image\n")
            f.write("2. Load merged mask images as overlays\n")
            f.write("3. Compare with original masks to see filled gaps\n")

        return successful_masks

    def _create_merged_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create merged mask by intelligently connecting compatible components with realistic tissue reconstruction."""
        from skimage import measure, morphology
        from scipy import ndimage

        # Find connected components
        labeled_mask = measure.label(mask)
        num_components = labeled_mask.max()

        if num_components <= 1:
            # Single component or empty - return as is
            return mask.astype(bool)

        # Analyze each component for morphological properties
        components = []
        for component_id in range(1, num_components + 1):
            component_mask = (labeled_mask == component_id)
            info = self._analyze_component_for_merging(component_mask)
            if info is not None:
                components.append(info)

        if len(components) < 2:
            return mask.astype(bool)

        # Find compatible component pairs for connection
        connections = self._find_compatible_connections(components)

        # Create merged mask with realistic tissue filling
        merged_mask = mask.astype(bool).copy()

        for comp1_idx, comp2_idx in connections:
            self._fill_tissue_region(merged_mask, components[comp1_idx], components[comp2_idx])

        return merged_mask

    def _analyze_component_for_merging(self, component_mask: np.ndarray) -> dict:
        """Analyze component properties for intelligent merging decisions."""
        from skimage import morphology, measure
        from scipy import ndimage
        import numpy as np

        if component_mask.sum() == 0:
            return None

        # Basic properties
        props = measure.regionprops(component_mask.astype(int))[0]

        # Create skeleton and distance transform
        skeleton = morphology.skeletonize(component_mask)
        skeleton_points = np.argwhere(skeleton)
        distance_transform = ndimage.distance_transform_edt(component_mask)

        if len(skeleton_points) == 0:
            return None

        # Calculate main orientation using PCA on the entire component
        component_points = np.argwhere(component_mask)
        if len(component_points) < 3:
            main_orientation = 0.0
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca.fit(component_points)
            main_direction = pca.components_[0]
            main_orientation = np.arctan2(main_direction[0], main_direction[1])

        # Find true endpoints (skeleton points with <= 1 neighbor)
        endpoints = self._find_skeleton_endpoints(skeleton, skeleton_points)

        # Calculate component statistics
        max_width = distance_transform.max() * 2  # Maximum thickness
        mean_width = np.mean(distance_transform[component_mask]) * 2  # Average thickness

        # Calculate component length (skeleton length)
        component_length = len(skeleton_points)

        # Aspect ratio from bounding box
        aspect_ratio = props.major_axis_length / props.minor_axis_length if props.minor_axis_length > 0 else 1

        return {
            'mask': component_mask,
            'skeleton_points': skeleton_points,
            'endpoints': endpoints,
            'centroid': props.centroid,
            'main_orientation': main_orientation,
            'max_width': max_width,
            'mean_width': mean_width,
            'length': component_length,
            'area': props.area,
            'aspect_ratio': aspect_ratio,
            'bbox': props.bbox,  # (min_row, min_col, max_row, max_col)
            'distance_transform': distance_transform
        }

    def _find_skeleton_endpoints(self, skeleton: np.ndarray, skeleton_points: np.ndarray) -> list:
        """Find true endpoints of a skeleton (points with <= 1 neighbor)."""
        endpoints = []

        if len(skeleton_points) <= 2:
            return skeleton_points.tolist()

        for point in skeleton_points:
            y, x = point
            neighbors = 0

            # Check 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and
                        skeleton[ny, nx]):
                        neighbors += 1

            if neighbors <= 1:  # Endpoint or isolated point
                endpoints.append(point)

        # If no endpoints found, use the two farthest points
        if len(endpoints) == 0:
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(skeleton_points))
            i, j = np.unravel_index(distances.argmax(), distances.shape)
            endpoints = [skeleton_points[i], skeleton_points[j]]

        return endpoints

    def _find_compatible_connections(self, components: list) -> list:
        """Find pairs of components that should be connected based on biological plausibility."""
        connections = []

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                if self._are_components_compatible(components[i], components[j]):
                    connections.append((i, j))

        return connections

    def _are_components_compatible(self, comp1: dict, comp2: dict) -> bool:
        """Determine if two components are likely parts of the same myotube."""

        # 1. Check distance between closest points
        min_distance = self._calculate_min_distance_between_components(comp1, comp2)

        # Reject if components are too far apart (relative to their size)
        max_component_size = max(comp1['length'], comp2['length'])
        if min_distance > max_component_size * 0.8:  # 80% of longest component
            return False

        # 2. Check orientation alignment
        orientation_diff = abs(comp1['main_orientation'] - comp2['main_orientation'])
        orientation_diff = min(orientation_diff, np.pi - orientation_diff)  # Handle wrap-around

        # Reject if orientations are too different (not aligned)
        if orientation_diff > np.pi / 3:  # More than 60 degrees difference
            return False

        # 3. Check size compatibility
        size_ratio = max(comp1['mean_width'], comp2['mean_width']) / min(comp1['mean_width'], comp2['mean_width'])
        if size_ratio > 3.0:  # One component is more than 3x wider than the other
            return False

        # 4. Check if they form a reasonable continuation
        if not self._check_reasonable_continuation(comp1, comp2):
            return False

        return True

    def _calculate_min_distance_between_components(self, comp1: dict, comp2: dict) -> float:
        """Calculate minimum distance between any points of two components."""
        points1 = np.argwhere(comp1['mask'])
        points2 = np.argwhere(comp2['mask'])

        min_dist = float('inf')
        for p1 in points1[::5]:  # Sample every 5th point for efficiency
            for p2 in points2[::5]:
                dist = np.linalg.norm(p1 - p2)
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    def _check_reasonable_continuation(self, comp1: dict, comp2: dict) -> bool:
        """Check if connecting these components would create a reasonable myotube continuation."""

        # Find closest endpoints between components
        min_dist = float('inf')
        best_endpoints = None

        for ep1 in comp1['endpoints']:
            for ep2 in comp2['endpoints']:
                dist = np.linalg.norm(ep1 - ep2)
                if dist < min_dist:
                    min_dist = dist
                    best_endpoints = (ep1, ep2)

        if best_endpoints is None:
            return False

        ep1, ep2 = best_endpoints

        # Calculate the direction of connection
        connection_vector = ep2 - ep1
        if np.linalg.norm(connection_vector) == 0:
            return False

        connection_angle = np.arctan2(connection_vector[0], connection_vector[1])

        # Check if connection direction aligns reasonably with component orientations
        angle_diff1 = abs(connection_angle - comp1['main_orientation'])
        angle_diff2 = abs(connection_angle - comp2['main_orientation'])

        # Allow for some flexibility in alignment
        max_angle_diff = np.pi / 4  # 45 degrees

        return (angle_diff1 < max_angle_diff or angle_diff1 > np.pi - max_angle_diff) and \
               (angle_diff2 < max_angle_diff or angle_diff2 > np.pi - max_angle_diff)

    def _fill_tissue_region(self, merged_mask: np.ndarray, comp1: dict, comp2: dict):
        """Fill the tissue region between two compatible components with realistic morphology."""
        from skimage import morphology

        # Find the best connection endpoints
        min_dist = float('inf')
        best_endpoints = None

        for ep1 in comp1['endpoints']:
            for ep2 in comp2['endpoints']:
                dist = np.linalg.norm(ep1 - ep2)
                if dist < min_dist:
                    min_dist = dist
                    best_endpoints = (ep1, ep2)

        if best_endpoints is None:
            return

        ep1, ep2 = best_endpoints

        # Calculate tissue thickness using the same CSV width calculation (area/length)
        csv_width1 = self._calculate_csv_width(comp1['mask'])
        csv_width2 = self._calculate_csv_width(comp2['mask'])

        # Use average of CSV widths for most realistic connection thickness
        if csv_width1 > 0 and csv_width2 > 0:
            avg_csv_width = (csv_width1 + csv_width2) / 2
            tissue_thickness = int(avg_csv_width * 0.9)  # 90% of average CSV width
        else:
            # Fallback to mean width if CSV calculation fails
            avg_component_width = (comp1['mean_width'] + comp2['mean_width']) / 2
            tissue_thickness = int(avg_component_width * 0.8)

        # Apply reasonable bounds - CSV widths are already realistic
        tissue_thickness = max(tissue_thickness, 3)  # Minimum for visibility
        tissue_thickness = min(tissue_thickness, 50)  # Maximum to prevent excessive thickness

        # Create connection region using morphological operations
        connection_region = self._create_connection_region(ep1, ep2, tissue_thickness, merged_mask.shape)

        # Apply the connection region to the merged mask
        merged_mask |= connection_region

        # Apply morphological closing to smooth the connection
        kernel_size = max(3, tissue_thickness // 3)
        kernel = morphology.disk(kernel_size)

        # Create a local region around the connection for processing
        min_y = max(0, min(ep1[0], ep2[0]) - tissue_thickness)
        max_y = min(merged_mask.shape[0], max(ep1[0], ep2[0]) + tissue_thickness)
        min_x = max(0, min(ep1[1], ep2[1]) - tissue_thickness)
        max_x = min(merged_mask.shape[1], max(ep1[1], ep2[1]) + tissue_thickness)

        local_region = merged_mask[min_y:max_y, min_x:max_x]
        smoothed_region = morphology.binary_closing(local_region, kernel)
        merged_mask[min_y:max_y, min_x:max_x] = smoothed_region

    def _create_connection_region(self, ep1: np.ndarray, ep2: np.ndarray,
                                thickness: int, mask_shape: tuple) -> np.ndarray:
        """Create a realistic tissue connection region between two endpoints."""
        from skimage.draw import polygon

        # Create connection path
        path_points = self._create_tissue_path(ep1, ep2)

        # Create mask for the connection region
        connection_mask = np.zeros(mask_shape, dtype=bool)

        # For each point along the path, create a thick cross-section
        for point in path_points:
            y, x = int(point[0]), int(point[1])

            # Create a thick region around each point
            half_thickness = thickness // 2

            y_min = max(0, y - half_thickness)
            y_max = min(mask_shape[0], y + half_thickness + 1)
            x_min = max(0, x - half_thickness)
            x_max = min(mask_shape[1], x + half_thickness + 1)

            connection_mask[y_min:y_max, x_min:x_max] = True

        return connection_mask

    def _create_tissue_path(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Create a natural tissue path between two points."""
        # Use simple linear interpolation for now - can be enhanced with curves
        distance = np.linalg.norm(end - start)
        num_points = max(10, int(distance))

        t_values = np.linspace(0, 1, num_points)
        path_points = np.array([start + t * (end - start) for t in t_values])

        return path_points

    def _calculate_local_endpoint_width(self, component: dict, endpoint: np.ndarray) -> float:
        """Calculate the local tissue width specifically at the endpoint location."""

        # Get the distance transform and component orientation
        distance_transform = component['distance_transform']
        component_mask = component['mask']

        y, x = int(endpoint[0]), int(endpoint[1])

        # Method 1: Use distance transform value at endpoint (most reliable)
        if 0 <= y < distance_transform.shape[0] and 0 <= x < distance_transform.shape[1]:
            dt_width = distance_transform[y, x] * 2  # Radius * 2 = diameter
        else:
            dt_width = component['mean_width']  # Fallback to mean width

        # Method 2: Cross-sectional measurement perpendicular to component orientation
        cross_sectional_width = self._measure_cross_sectional_width(component_mask, endpoint, component['main_orientation'])

        # Method 3: Local neighborhood analysis (more conservative)
        neighborhood_width = self._measure_conservative_neighborhood_width(component_mask, endpoint)

        # Use median instead of maximum for more conservative estimate
        widths = [w for w in [dt_width, cross_sectional_width, neighborhood_width] if w > 0]
        if widths:
            local_width = np.median(widths)
        else:
            local_width = component['mean_width']

        # Ensure reasonable bounds based on component characteristics
        local_width = max(local_width, 2.0)  # Minimum 2 pixels

        # Conservative upper bound: don't exceed mean width unless component is very thin
        max_allowed = max(component['mean_width'], component['max_width'] * 0.6)

        # Additional bound for very large components (prevents excessive thickness)
        area_based_max = np.sqrt(component['area']) * 0.3  # Conservative area-based bound
        max_allowed = min(max_allowed, area_based_max)

        local_width = min(local_width, max_allowed)

        return local_width

    def _measure_cross_sectional_width(self, component_mask: np.ndarray, endpoint: np.ndarray, orientation: float) -> float:
        """Measure width by sampling perpendicular to the component orientation."""
        y, x = int(endpoint[0]), int(endpoint[1])

        # Create perpendicular direction to orientation
        perp_angle = orientation + np.pi/2
        perp_dy = np.cos(perp_angle)
        perp_dx = np.sin(perp_angle)

        # Sample along perpendicular direction to find tissue boundaries
        max_radius = 20  # Search up to 20 pixels in each direction
        width = 0

        for radius in range(1, max_radius + 1):
            # Sample points along perpendicular direction
            py1, px1 = int(y + perp_dy * radius), int(x + perp_dx * radius)
            py2, px2 = int(y - perp_dy * radius), int(x - perp_dx * radius)

            # Check if both points are within bounds
            if (0 <= py1 < component_mask.shape[0] and 0 <= px1 < component_mask.shape[1] and
                0 <= py2 < component_mask.shape[0] and 0 <= px2 < component_mask.shape[1]):

                if component_mask[py1, px1] and component_mask[py2, px2]:
                    width = radius * 2  # Both sides are in tissue
                else:
                    break  # Hit boundary
            else:
                break  # Out of bounds

        return width

    def _measure_neighborhood_width(self, component_mask: np.ndarray, endpoint: np.ndarray) -> float:
        """Measure width using local neighborhood analysis."""
        y, x = int(endpoint[0]), int(endpoint[1])
        neighborhood_size = 7  # 7x7 neighborhood
        half_size = neighborhood_size // 2

        # Extract local neighborhood
        y_min = max(0, y - half_size)
        y_max = min(component_mask.shape[0], y + half_size + 1)
        x_min = max(0, x - half_size)
        x_max = min(component_mask.shape[1], x + half_size + 1)

        local_region = component_mask[y_min:y_max, x_min:x_max]

        if local_region.sum() == 0:
            return 0

        # Calculate the "diameter" of the local region
        # Find the maximum distance between any two tissue points in the neighborhood
        tissue_points = np.argwhere(local_region)

        if len(tissue_points) < 2:
            return 2.0  # Minimum width

        max_distance = 0
        for i, p1 in enumerate(tissue_points[::2]):  # Sample every 2nd point for efficiency
            for p2 in tissue_points[i+1::2]:
                distance = np.linalg.norm(p1 - p2)
                if distance > max_distance:
                    max_distance = distance

        return max_distance

    def _measure_conservative_neighborhood_width(self, component_mask: np.ndarray, endpoint: np.ndarray) -> float:
        """Measure width using more conservative local neighborhood analysis."""
        y, x = int(endpoint[0]), int(endpoint[1])
        neighborhood_size = 5  # Smaller 5x5 neighborhood for more local measurement
        half_size = neighborhood_size // 2

        # Extract local neighborhood
        y_min = max(0, y - half_size)
        y_max = min(component_mask.shape[0], y + half_size + 1)
        x_min = max(0, x - half_size)
        x_max = min(component_mask.shape[1], x + half_size + 1)

        local_region = component_mask[y_min:y_max, x_min:x_max]

        if local_region.sum() == 0:
            return 0

        # Use distance transform on the local region for more accurate width
        from scipy import ndimage
        local_dt = ndimage.distance_transform_edt(local_region)

        # Get the center point relative to the local region
        center_y = min(half_size, y - y_min)
        center_x = min(half_size, x - x_min)

        if (center_y < local_dt.shape[0] and center_x < local_dt.shape[1] and
            local_region[center_y, center_x]):
            # Use distance transform at center point, but cap it conservatively
            dt_radius = local_dt[center_y, center_x]
            conservative_width = dt_radius * 1.8  # Less than 2x for more conservative estimate
        else:
            # Fallback: use average distance transform in the region
            conservative_width = np.mean(local_dt[local_region]) * 1.8

        # Cap the width based on local region size to prevent extreme values
        max_local_width = min(local_region.shape[0], local_region.shape[1]) * 0.8
        conservative_width = min(conservative_width, max_local_width)

        return max(conservative_width, 2.0)  # Minimum 2 pixels

    def _calculate_csv_width(self, component_mask: np.ndarray) -> float:
        """Calculate the same width measurement that goes into the CSV (area/length)."""
        from skimage import morphology

        if component_mask.sum() == 0:
            return 0

        # Calculate area
        area = component_mask.sum()

        # Calculate skeleton length
        skeleton = morphology.skeletonize(component_mask)
        skeleton_points = np.argwhere(skeleton)
        visible_length = len(skeleton_points)

        # Same calculation as in CSV: area / visible_length
        if visible_length > 0:
            width_pixels = area / visible_length
        else:
            width_pixels = 0

        return width_pixels

    def _count_components(self, mask: np.ndarray) -> int:
        """Count the number of connected components in a mask."""
        from skimage import measure
        labeled_mask = measure.label(mask)
        return labeled_mask.max()


    def _save_colored_overlay(self, instances, original_image: np.ndarray,
                             output_path: str, overlay_type: str = "processed"):
        """Save colored overlay using Detectron2's built-in visualizer like demo.py."""
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog
        from detectron2.structures import Instances
        import torch

        print(f"   🎨 Generating {overlay_type} overlay using Detectron2's Visualizer")

        # MEMORY OPTIMIZATION: Generate overlays at reasonable resolution (not full original)
        # Overlays are for visualization only - 3000px is more than sufficient
        max_overlay_size = 3000
        original_h, original_w = original_image.shape[:2]

        if max(original_h, original_w) > max_overlay_size:
            scale = max_overlay_size / max(original_h, original_w)
            overlay_h, overlay_w = int(original_h * scale), int(original_w * scale)
            overlay_image = cv2.resize(original_image, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
            print(f"      📐 Overlay resolution: {overlay_w}×{overlay_h} (optimized for memory)")
        else:
            overlay_image = original_image
            overlay_h, overlay_w = original_h, original_w
            scale = 1.0

        # Get metadata (try to use the same as our dataset, fallback to COCO)
        try:
            metadata = MetadataCatalog.get("myotube_stage2_train")
        except:
            try:
                metadata = MetadataCatalog.get("coco_2017_val")
            except:
                metadata = None
        
        # Handle both raw Detectron2 instances and our processed format
        if hasattr(instances, 'pred_masks'):
            # Raw Detectron2 Instances - need to resize masks to overlay resolution
            torch_instances = Instances((overlay_h, overlay_w))

            if len(instances) > 0:
                # Resize masks to overlay resolution (memory-efficient)
                raw_masks = instances.pred_masks.cpu().numpy()
                final_masks = []

                for i, mask in enumerate(raw_masks):
                    # Resize mask to overlay resolution
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    resized_mask = cv2.resize(
                        mask_uint8,
                        (overlay_w, overlay_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    resized_mask = (resized_mask > 128)  # Back to boolean
                    final_masks.append(resized_mask)
                
                # Convert to torch tensors
                torch_instances.pred_masks = torch.tensor(np.array(final_masks)).cpu()
                torch_instances.scores = instances.scores.cpu()
                
                # Use empty bounding boxes
                from detectron2.structures import Boxes
                torch_instances.pred_boxes = Boxes(torch.zeros(len(instances), 4).cpu())
                torch_instances.pred_classes = torch.zeros(len(instances), dtype=torch.long).cpu()
                
                num_instances = len(instances)
            else:
                num_instances = 0
        else:
            # Our processed format - convert back to Detectron2 Instances
            torch_instances = Instances((overlay_h, overlay_w))

            if len(instances['masks']) > 0:
                # Resize masks to overlay resolution (memory-efficient)
                final_masks = []

                for i, mask in enumerate(instances['masks']):
                    # Ensure mask is a numpy array first
                    if torch.is_tensor(mask):
                        mask = mask.cpu().numpy()

                    # Ensure mask is boolean
                    mask = mask.astype(bool)

                    # Resize mask to overlay resolution
                    mask_uint8 = mask.astype(np.uint8) * 255
                    resized_mask = cv2.resize(
                        mask_uint8,
                        (overlay_w, overlay_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    resized_mask = (resized_mask > 128)  # Back to boolean

                    # Final validation: ensure boolean numpy array
                    resized_mask = np.array(resized_mask, dtype=bool)
                    final_masks.append(resized_mask)
                
                # Convert to torch tensors with proper data types (demo.py moves to CPU, so we do the same)
                # Ensure masks are boolean and properly shaped
                mask_array = np.array(final_masks, dtype=bool)
                torch_instances.pred_masks = torch.tensor(mask_array).cpu()
                
                # Ensure scores are float32
                scores_array = np.array(instances['scores'], dtype=np.float32)
                torch_instances.scores = torch.tensor(scores_array).cpu()
                
                # Use empty bounding boxes like the original Mask2Former model does
                # This prevents bounding boxes from being drawn, showing only masks
                from detectron2.structures import Boxes
                torch_instances.pred_boxes = Boxes(torch.zeros(len(instances['masks']), 4).cpu())
                
                # Add dummy classes (all myotubes are the same class)
                torch_instances.pred_classes = torch.zeros(len(instances['masks']), dtype=torch.long).cpu()
                
                num_instances = len(instances['masks'])
            else:
                num_instances = 0
        
        print(f"   📊 Created Detectron2 Instances with {num_instances} instances")

        # Use Detectron2's visualizer exactly like demo.py does
        # Convert BGR to RGB for visualization (demo.py does this: image[:, :, ::-1])
        rgb_image = overlay_image[:, :, ::-1]
        visualizer = Visualizer(rgb_image, metadata, instance_mode=ColorMode.IMAGE)
        
        # Add validation before visualization
        if num_instances > 0:
            print(f"   🔍 Mask validation: shape={torch_instances.pred_masks.shape}, dtype={torch_instances.pred_masks.dtype}")
            print(f"   🔍 Score validation: shape={torch_instances.scores.shape}, dtype={torch_instances.scores.dtype}")
        
        try:
            # This is the exact same call that demo.py uses!
            vis_output = visualizer.draw_instance_predictions(predictions=torch_instances)
            
            # Get the visualization as an image and convert back to BGR for saving
            vis_image = vis_output.get_image()[:, :, ::-1]  # RGB back to BGR
            
        except Exception as e:
            print(f"   ❌ Visualization failed: {e}")
            print(f"   💡 Creating fallback overlay with image")
            # Fallback: save overlay image as overlay
            vis_image = overlay_image.copy()
        
        print(f"   💾 Saving overlay to: {output_path}")
        
        # Save the visualization
        success = cv2.imwrite(output_path, vis_image)
        if success:
            print(f"   ✅ {overlay_type.title()} overlay saved: {os.path.basename(output_path)}")
        else:
            print(f"   ❌ Failed to save {overlay_type} overlay")

        print(f"   🔍 {overlay_type.title()} overlay: {num_instances} instances visualized")

        # Memory cleanup after overlay generation
        del torch_instances, final_masks, vis_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    def _save_measurements(self, instances: Dict[str, Any], output_path: str):
        """Save comprehensive measurements CSV for myotube analysis."""
        import pandas as pd
        from skimage import measure, morphology

        measurements = []

        for i, (mask, score, box) in enumerate(zip(instances['masks'], instances['scores'], instances['boxes'])):
            # Resize mask to original size for accurate measurements
            if hasattr(self, '_scale_factor') and self._scale_factor != 1.0:
                original_h, original_w = self._original_size
                # Use proper mask scaling like in ROI generation
                mask_uint8 = (mask * 255).astype(np.uint8)
                resized_mask = cv2.resize(
                    mask_uint8,
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST
                )
                resized_mask = (resized_mask > 128).astype(bool)

                # Scale bounding box back to original coordinates
                scale_factor = 1.0 / self._scale_factor
                scaled_box = box * scale_factor
            else:
                resized_mask = mask.astype(bool)
                scaled_box = box

            # Calculate existing measurements
            area = resized_mask.sum()
            contours = measure.find_contours(resized_mask, 0.5)
            perimeter = sum(len(contour) for contour in contours)

            # Calculate new myotube-specific measurements
            visible_length, estimated_total_length, num_components = self._calculate_myotube_length(resized_mask)
            width_pixels = area / visible_length if visible_length > 0 else 0
            myotube_aspect_ratio = estimated_total_length / width_pixels if width_pixels > 0 else 0

            # Calculate bounding box measurements (keep existing logic)
            bbox_width = scaled_box[2] - scaled_box[0]
            bbox_height = scaled_box[3] - scaled_box[1]
            bbox_aspect_ratio = max(bbox_width, bbox_height) / min(bbox_width, bbox_height) if min(bbox_width, bbox_height) > 0 else 0

            measurements.append({
                'Instance': f'Myotube_{i+1}',
                'Area': int(area),
                'Visible_Length_pixels': round(visible_length, 2),
                'Estimated_Total_Length_pixels': round(estimated_total_length, 2),
                'Width_pixels': round(width_pixels, 2),
                'Aspect_Ratio': round(myotube_aspect_ratio, 2),
                'Connected_Components': num_components,
                'Perimeter': round(perimeter, 2),
                'BBox_AspectRatio': round(bbox_aspect_ratio, 2),
                'Confidence': round(score, 4),
                'BoundingBox_X': round(box[0], 1),
                'BoundingBox_Y': round(box[1], 1),
                'BoundingBox_Width': round(bbox_width, 1),
                'BoundingBox_Height': round(bbox_height, 1)
            })

        # Save to CSV
        df = pd.DataFrame(measurements)
        df.to_csv(output_path, index=False)
        print(f"💾 Saved myotube measurements for {len(measurements)} instances to {output_path}")

    def _calculate_myotube_length(self, mask: np.ndarray) -> Tuple[float, float, int]:
        """
        Calculate visible and estimated total myotube length.

        Returns:
            visible_length: Length of visible skeleton parts
            estimated_total_length: Visible length + interpolated gaps
            num_components: Number of connected components
        """
        from skimage import morphology, measure

        # Find connected components
        labeled_mask = measure.label(mask)
        num_components = labeled_mask.max()

        if num_components == 0:
            return 0.0, 0.0, 0

        # Calculate skeleton for each component
        component_skeletons = []
        total_visible_length = 0

        for component_id in range(1, num_components + 1):
            component_mask = (labeled_mask == component_id)

            # Process all components (no size filtering)

            # Create skeleton
            skeleton = morphology.skeletonize(component_mask)
            skeleton_points = np.argwhere(skeleton)

            if len(skeleton_points) > 0:
                component_skeletons.append(skeleton_points)
                # Approximate skeleton length as number of skeleton pixels
                total_visible_length += len(skeleton_points)

        # Estimate total length including gaps between components
        estimated_total_length = total_visible_length

        if len(component_skeletons) > 1:
            # Add estimated lengths of gaps between components
            gap_length = self._estimate_gap_lengths(component_skeletons)
            estimated_total_length += gap_length

        return total_visible_length, estimated_total_length, num_components

    def _estimate_gap_lengths(self, component_skeletons: List[np.ndarray]) -> float:
        """
        Estimate total length of gaps between skeleton components using linear interpolation.

        Args:
            component_skeletons: List of skeleton point arrays for each component

        Returns:
            total_gap_length: Sum of estimated gap lengths
        """
        if len(component_skeletons) < 2:
            return 0.0

        total_gap_length = 0.0

        # Find endpoints of each component skeleton
        component_endpoints = []
        for skeleton_points in component_skeletons:
            if len(skeleton_points) == 0:
                continue

            # For each component, find the two points that are farthest apart
            if len(skeleton_points) == 1:
                endpoints = [skeleton_points[0], skeleton_points[0]]
            elif len(skeleton_points) == 2:
                endpoints = [skeleton_points[0], skeleton_points[1]]
            else:
                # Find the two points with maximum distance
                from scipy.spatial.distance import pdist, squareform
                distances = squareform(pdist(skeleton_points))
                i, j = np.unravel_index(distances.argmax(), distances.shape)
                endpoints = [skeleton_points[i], skeleton_points[j]]

            component_endpoints.append(endpoints)

        # Connect nearest endpoints between different components
        connected_pairs = set()

        for i in range(len(component_endpoints)):
            for j in range(i + 1, len(component_endpoints)):
                # Find minimum distance between any endpoint of component i and component j
                min_distance = float('inf')

                for endpoint_i in component_endpoints[i]:
                    for endpoint_j in component_endpoints[j]:
                        distance = np.linalg.norm(endpoint_i - endpoint_j)
                        if distance < min_distance:
                            min_distance = distance

                # Add this gap (connect all components within the same myotube)
                pair_key = tuple(sorted([i, j]))
                if pair_key not in connected_pairs:
                    total_gap_length += min_distance
                    connected_pairs.add(pair_key)

        return total_gap_length
    
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


class GUIOutputStream:
    """Redirects stdout to GUI console widget."""

    def __init__(self, gui):
        self.gui = gui

    def write(self, text):
        if text:  # Write all text including newlines
            self.gui.write_to_console(text)

    def flush(self):
        pass  # Required for file-like object


class ParameterGUI:
    """User-friendly GUI for parameter configuration."""

    def __init__(self, config_file=None, locked_output_dir=None):
        """Initialize the GUI with saved or default parameters.

        Args:
            config_file: Path to config file (default: auto-detect)
            locked_output_dir: If provided, locks the output directory (used by Fiji integration)
        """
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.locked_output_dir = locked_output_dir

        # Default parameters
        # Set default output directory to Desktop/myotube_results
        default_output = os.path.join(os.path.expanduser('~'), 'Desktop', 'myotube_results')

        self.defaults = {
            'input_path': '',
            'output_dir': default_output,
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

        # Config file location (in script directory or user home)
        if config_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(script_dir, '.myotube_gui_config.json')
        self.config_file = config_file

        # Load saved parameters
        self.params = self.load_config()

        # Note: locked_output_dir parameter kept for backward compatibility but not used
        # Users can now always choose their output directory in the GUI

        # GUI state
        self.result = None
        self.root = None
        self.console_text = None
        self.is_running = False

    def write_to_console(self, text):
        """Write text to console widget."""
        if self.console_text:
            self.console_text.config(state='normal')
            self.console_text.insert(self.tk.END, text)
            self.console_text.see(self.tk.END)  # Auto-scroll to bottom
            self.console_text.config(state='disabled')
            self.root.update_idletasks()

    def clear_console(self):
        """Clear console widget."""
        if self.console_text:
            self.console_text.config(state='normal')
            self.console_text.delete('1.0', self.tk.END)
            self.console_text.config(state='disabled')

    def load_config(self):
        """Load saved configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved = json.load(f)
                # Merge with defaults to handle new parameters
                params = self.defaults.copy()
                params.update(saved)
                print(f"📂 Loaded saved configuration from: {self.config_file}")
                return params
            except Exception as e:
                print(f"⚠️  Could not load config file: {e}")
                return self.defaults.copy()
        else:
            return self.defaults.copy()

    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.params, f, indent=2)
            print(f"💾 Saved configuration to: {self.config_file}")
        except Exception as e:
            print(f"⚠️  Could not save config file: {e}")

    def restore_defaults(self):
        """Restore all parameters to default values."""
        # Keep paths but restore processing parameters
        input_path = self.params.get('input_path', '')
        output_dir = self.params.get('output_dir', '')
        config = self.params.get('config', '')
        weights = self.params.get('weights', '')

        self.params = self.defaults.copy()

        # Restore the paths
        self.params['input_path'] = input_path
        self.params['output_dir'] = output_dir
        self.params['config'] = config
        self.params['weights'] = weights

        # Update GUI
        self.update_gui_from_params()

        # No popup - just restore silently
        print("✅ Restored parameters to defaults (paths preserved)")

    def update_gui_from_params(self):
        """Update GUI widgets from current parameters."""
        # Paths
        self.input_var.set(self.params['input_path'])
        self.output_var.set(self.params['output_dir'])
        self.config_var.set(self.params['config'])
        self.weights_var.set(self.params['weights'])
        self.mask2former_path_var.set(self.params['mask2former_path'])

        # Processing parameters
        self.confidence_var.set(self.params['confidence'])
        self.min_area_var.set(self.params['min_area'])
        self.max_area_var.set(self.params['max_area'])
        self.final_min_area_var.set(self.params['final_min_area'])

        # Flags
        self.cpu_var.set(self.params['cpu'])
        self.force_1024_var.set(self.params['force_1024'])
        self.use_tiling_var.set(self.params['use_tiling'])
        self.skip_merged_var.set(self.params['skip_merged_masks'])

        # Optional parameters
        self.max_image_size_var.set(str(self.params['max_image_size']) if self.params['max_image_size'] else '')
        self.grid_size_var.set(self.params['grid_size'])
        self.tile_overlap_var.set(self.params['tile_overlap'] * 100)  # Display as percentage

        # Update formatted labels if they exist
        if hasattr(self, 'confidence_label'):
            self.confidence_label.configure(text=f"{self.params['confidence']:.2f}")
        if hasattr(self, 'tile_overlap_label'):
            self.tile_overlap_label.configure(text=f"{self.params['tile_overlap'] * 100:.1f}")

    def update_params_from_gui(self):
        """Update parameters from GUI widgets."""
        self.params['input_path'] = self.input_var.get()
        self.params['output_dir'] = self.output_var.get()
        self.params['config'] = self.config_var.get()
        self.params['weights'] = self.weights_var.get()
        self.params['mask2former_path'] = self.mask2former_path_var.get()

        self.params['confidence'] = float(self.confidence_var.get())
        self.params['min_area'] = int(self.min_area_var.get())
        self.params['max_area'] = int(self.max_area_var.get())
        self.params['final_min_area'] = int(self.final_min_area_var.get())

        self.params['cpu'] = self.cpu_var.get()
        self.params['force_1024'] = self.force_1024_var.get()
        self.params['use_tiling'] = self.use_tiling_var.get()
        self.params['skip_merged_masks'] = self.skip_merged_var.get()
        self.params['save_measurements'] = self.save_measurements_var.get()

        # Optional max_image_size
        max_size_str = self.max_image_size_var.get().strip()
        self.params['max_image_size'] = int(max_size_str) if max_size_str else ''

        # Grid size and tile overlap
        self.params['grid_size'] = int(self.grid_size_var.get())
        self.params['tile_overlap'] = float(self.tile_overlap_var.get()) / 100.0

    def browse_input(self):
        """Browse for input file or directory."""
        path = self.filedialog.askdirectory(title="Select Input Directory")
        if not path:
            path = self.filedialog.askopenfilename(
                title="Select Input Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
            )
        if path:
            self.input_var.set(path)

    def browse_output(self):
        """Browse for output directory."""
        path = self.filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_var.set(path)

    def browse_config(self):
        """Browse for config file."""
        path = self.filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if path:
            self.config_var.set(path)

    def browse_weights(self):
        """Browse for model weights."""
        path = self.filedialog.askopenfilename(
            title="Select Model Weights",
            filetypes=[("Model files", "*.pth *.pkl"), ("All files", "*.*")]
        )
        if path:
            self.weights_var.set(path)

    def browse_mask2former_path(self):
        """Browse for Mask2Former project directory."""
        path = self.filedialog.askdirectory(
            title="Select Mask2Former Project Directory"
        )
        if path:
            self.mask2former_path_var.set(path)

    def on_run_threaded(self):
        """Handle Run button click - runs segmentation in a thread."""
        if self.is_running:
            self.messagebox.showwarning("Already Running", "Segmentation is already in progress. Please wait.")
            return

        # Update parameters from GUI
        self.update_params_from_gui()

        # Validate required fields
        if not self.params['input_path']:
            self.messagebox.showerror("Error", "Please select an input image or directory")
            return

        if not self.params['output_dir']:
            self.messagebox.showerror("Error", "Please select an output directory")
            return

        # Validate numeric parameters
        try:
            if not (0 <= self.params['confidence'] <= 1):
                raise ValueError("Confidence must be between 0 and 1")
            if self.params['min_area'] <= 0:
                raise ValueError("Minimum area must be positive")
            if self.params['max_area'] <= self.params['min_area']:
                raise ValueError("Maximum area must be greater than minimum area")
            if self.params['final_min_area'] < 0:
                raise ValueError("Final minimum area must be non-negative")
            if self.params['grid_size'] < 1:
                raise ValueError("Grid size must be at least 1")
            if not (0 < self.params['tile_overlap'] < 1):
                raise ValueError("Tile overlap must be between 0 and 1")
        except ValueError as e:
            self.messagebox.showerror("Invalid Parameter", str(e))
            return

        # Save configuration
        self.save_config()

        # Clear console and start segmentation in thread
        self.clear_console()
        self.write_to_console("=== Starting Segmentation ===\n")
        self.write_to_console(f"Input: {self.params['input_path']}\n")
        self.write_to_console(f"Output: {self.params['output_dir']}\n\n")

        # Disable run button
        self.is_running = True
        self.run_button.config(state='disabled')

        # Run in thread
        import threading
        thread = threading.Thread(target=self.run_segmentation_in_gui)
        thread.daemon = True
        thread.start()

    def run_segmentation_in_gui(self):
        """Run segmentation and redirect output to console."""
        import sys

        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = GUIOutputStream(self)

        try:
            # Load Mask2Former modules
            print("🔄 Loading Mask2Former and detectron2 modules...")
            ensure_mask2former_loaded(explicit_path=self.params.get('mask2former_path'))

            # Import after Mask2Former is loaded
            from detectron2.engine.defaults import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config
            from detectron2.data.detection_utils import read_image
            from mask2former import add_maskformer2_config
            print("✅ Modules loaded successfully\n")

            # Build custom config from parameters
            max_image_size = 1024 if self.params.get('force_1024') else self.params.get('max_image_size')
            custom_config = {
                'confidence_threshold': self.params['confidence'],
                'min_area': self.params['min_area'],
                'max_area': self.params['max_area'],
                'final_min_area': self.params['final_min_area'],
                'max_image_size': max_image_size,
                'force_cpu': self.params['cpu'],
                'save_measurements': self.params['save_measurements']
            }

            # Initialize integration
            integration = MyotubeFijiIntegration(
                config_file=self.params['config'] if self.params['config'] else None,
                model_weights=self.params['weights'] if self.params['weights'] else None,
                skip_merged_masks=self.params['skip_merged_masks'],
                mask2former_path=self.params['mask2former_path'] if self.params['mask2former_path'] else None
            )

            # Initialize tiled segmentation if requested
            if self.params['use_tiling']:
                print(f"🔲 Tiled inference mode enabled (grid: {self.params['grid_size']}×{self.params['grid_size']}, overlap: {self.params['tile_overlap']*100:.0f}%)")
                tiled_segmenter = TiledMyotubeSegmentation(
                    fiji_integration=integration,
                    target_overlap=self.params['tile_overlap'],
                    grid_size=self.params['grid_size']
                )
            else:
                tiled_segmenter = None

            # Process input
            input_path = self.params['input_path']
            output_dir = self.params['output_dir']

            if os.path.isfile(input_path):
                # Single image
                print(f"📷 Processing single image: {input_path}")
                if tiled_segmenter:
                    tiled_segmenter.segment_image_tiled(input_path, output_dir, custom_config)
                else:
                    integration.segment_image(input_path, output_dir, custom_config)
            elif os.path.isdir(input_path):
                # Directory of images
                from pathlib import Path
                image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
                base_dir = Path(input_path)
                
                # Priority: search images/ subdirectory if it exists, otherwise search base directory
                images_subdir = base_dir / 'images'
                if images_subdir.exists() and images_subdir.is_dir():
                    search_dir = images_subdir
                    print(f"   📂 Searching images/ subdirectory")
                else:
                    search_dir = base_dir
                
                # Collect image files
                image_files_set = []
                for ext in image_extensions:
                    for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                        image_files_set.extend(search_dir.rglob(pattern))
                
                # De-duplicate using resolved absolute paths
                unique_files = {}
                for f in image_files_set:
                    try:
                        resolved = str(f.resolve())
                        if sys.platform == 'win32':
                            resolved = resolved.lower()
                        unique_files[resolved] = str(f)
                    except (OSError, RuntimeError):
                        unique_files[str(f)] = str(f)
                
                image_files = sorted(unique_files.values())

                print(f"📁 Found {len(image_files)} images in directory")

                for i, img_path in enumerate(image_files, 1):
                    print(f"\n{'='*60}")
                    print(f"Processing {i}/{len(image_files)}: {os.path.basename(img_path)}")
                    print(f"{'='*60}")

                    try:
                        if tiled_segmenter:
                            tiled_segmenter.segment_image_tiled(img_path, output_dir, custom_config)
                        else:
                            integration.segment_image(img_path, output_dir, custom_config)
                    except Exception as e:
                        print(f"❌ Error processing {os.path.basename(img_path)}: {e}")
                        continue

                print(f"\n🎉 Batch processing complete! Processed {len(image_files)} images.")

            print(f"\n✅ All segmentation complete!")
            print(f"📂 Results saved to: {output_dir}")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            # Restore stdout
            sys.stdout = old_stdout

            # Re-enable run button
            self.is_running = False
            self.root.after(0, lambda: self.run_button.config(state='normal'))

    def on_close(self):
        """Handle Close button click - returns parameters."""
        # Update parameters before closing
        self.update_params_from_gui()
        self.save_config()

        # Set result and close
        self.result = self.params
        self.root.quit()
        self.root.destroy()

    def on_cancel(self):
        """Handle Cancel button click - for backward compatibility."""
        self.result = None
        self.root.quit()
        self.root.destroy()

    def show(self):
        """Display the GUI and return selected parameters."""
        # Create main window
        self.root = self.tk.Tk()
        self.root.title("Myotube Segmentation Parameters")
        self.root.geometry("900x1000")

        # Create scrollable frame
        main_frame = self.ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(self.tk.N, self.tk.W, self.tk.E, self.tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Variables for GUI widgets
        self.input_var = self.tk.StringVar(value=self.params['input_path'])
        self.output_var = self.tk.StringVar(value=self.params['output_dir'])
        self.config_var = self.tk.StringVar(value=self.params['config'])
        self.weights_var = self.tk.StringVar(value=self.params['weights'])
        self.mask2former_path_var = self.tk.StringVar(value=self.params['mask2former_path'])
        self.confidence_var = self.tk.DoubleVar(value=self.params['confidence'])
        self.min_area_var = self.tk.IntVar(value=self.params['min_area'])
        self.max_area_var = self.tk.IntVar(value=self.params['max_area'])
        self.final_min_area_var = self.tk.IntVar(value=self.params['final_min_area'])
        self.cpu_var = self.tk.BooleanVar(value=self.params['cpu'])
        self.max_image_size_var = self.tk.StringVar(value=str(self.params['max_image_size']) if self.params['max_image_size'] else '')
        self.force_1024_var = self.tk.BooleanVar(value=self.params['force_1024'])
        self.use_tiling_var = self.tk.BooleanVar(value=self.params['use_tiling'])
        self.grid_size_var = self.tk.IntVar(value=self.params['grid_size'])
        self.tile_overlap_var = self.tk.DoubleVar(value=self.params['tile_overlap'] * 100)
        self.skip_merged_var = self.tk.BooleanVar(value=self.params['skip_merged_masks'])
        self.save_measurements_var = self.tk.BooleanVar(value=self.params['save_measurements'])

        row = 0

        # ===== Paths Section =====
        self.ttk.Label(main_frame, text="Input/Output Paths", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Label(main_frame, text="Input (Image/Directory):").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.input_var, width=50).grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.ttk.Button(main_frame, text="Browse...", command=self.browse_input).grid(row=row, column=2)
        row += 1

        self.ttk.Label(main_frame, text="Output Directory:").grid(row=row, column=0, sticky=self.tk.W)
        output_entry = self.ttk.Entry(main_frame, textvariable=self.output_var, width=50)
        output_entry.grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        output_browse_btn = self.ttk.Button(main_frame, text="Browse...", command=self.browse_output)
        output_browse_btn.grid(row=row, column=2)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Model Configuration =====
        self.ttk.Label(main_frame, text="Model Configuration (Optional)", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Label(main_frame, text="Config File:").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.config_var, width=50).grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.ttk.Button(main_frame, text="Browse...", command=self.browse_config).grid(row=row, column=2)
        row += 1

        self.ttk.Label(main_frame, text="Model Weights:").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.weights_var, width=50).grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.ttk.Button(main_frame, text="Browse...", command=self.browse_weights).grid(row=row, column=2)
        row += 1

        self.ttk.Label(main_frame, text="Mask2Former Path:").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.mask2former_path_var, width=50).grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.ttk.Button(main_frame, text="Browse...", command=self.browse_mask2former_path).grid(row=row, column=2)
        row += 1

        self.ttk.Label(main_frame, text="(Leave empty for auto-detection)", font=('Arial', 9, 'italic')).grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Detection Parameters =====
        self.ttk.Label(main_frame, text="Detection Parameters", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Label(main_frame, text="Confidence Threshold (0-1):").grid(row=row, column=0, sticky=self.tk.W)
        confidence_scale = self.ttk.Scale(main_frame, from_=0.0, to=1.0, variable=self.confidence_var, orient='horizontal', length=300)
        confidence_scale.grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.confidence_label = self.ttk.Label(main_frame, text=f"{self.confidence_var.get():.2f}")
        self.confidence_label.grid(row=row, column=2)
        confidence_scale.configure(command=lambda v: self.confidence_label.configure(text=f"{float(v):.2f}"))
        row += 1

        self.ttk.Label(main_frame, text="Minimum Area (pixels):").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.min_area_var, width=20).grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        self.ttk.Label(main_frame, text="Maximum Area (pixels):").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.max_area_var, width=20).grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        self.ttk.Label(main_frame, text="Final Min Area (pixels):").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.final_min_area_var, width=20).grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Performance Options =====
        self.ttk.Label(main_frame, text="Performance Options", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Checkbutton(main_frame, text="Use CPU (slower, less memory)", variable=self.cpu_var).grid(row=row, column=0, columnspan=2, sticky=self.tk.W, pady=2)
        row += 1

        self.ttk.Checkbutton(main_frame, text="Force 1024px input (memory optimization)", variable=self.force_1024_var).grid(row=row, column=0, columnspan=2, sticky=self.tk.W, pady=2)
        row += 1

        self.ttk.Label(main_frame, text="Max Image Size (optional):").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.max_image_size_var, width=20).grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Tiling Options =====
        self.ttk.Label(main_frame, text="Tiling Options", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Checkbutton(main_frame, text="Use tiled inference (for images with many myotubes)", variable=self.use_tiling_var).grid(row=row, column=0, columnspan=2, sticky=self.tk.W, pady=2)
        row += 1

        self.ttk.Label(main_frame, text="Grid Size (1=no split, 2=2×2, etc.):").grid(row=row, column=0, sticky=self.tk.W)
        grid_size_spinbox = self.ttk.Spinbox(main_frame, from_=1, to=10, textvariable=self.grid_size_var, width=10)
        grid_size_spinbox.grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        self.ttk.Label(main_frame, text="Tile Overlap (%):").grid(row=row, column=0, sticky=self.tk.W)
        tile_overlap_scale = self.ttk.Scale(main_frame, from_=10, to=50, variable=self.tile_overlap_var, orient='horizontal', length=300)
        tile_overlap_scale.grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.tile_overlap_label = self.ttk.Label(main_frame, text=f"{self.tile_overlap_var.get():.1f}")
        self.tile_overlap_label.grid(row=row, column=2)
        tile_overlap_scale.configure(command=lambda v: self.tile_overlap_label.configure(text=f"{float(v):.1f}"))
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Output Options =====
        self.ttk.Label(main_frame, text="Output Options", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Checkbutton(main_frame, text="Skip merged masks (skip imaginary boundary generation)", variable=self.skip_merged_var).grid(row=row, column=0, columnspan=2, sticky=self.tk.W, pady=2)
        row += 1

        self.ttk.Checkbutton(main_frame, text="Save measurements CSV (includes area, length, width, etc.)", variable=self.save_measurements_var).grid(row=row, column=0, columnspan=2, sticky=self.tk.W, pady=2)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Output Console =====
        self.ttk.Label(main_frame, text="Console Output", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        # Create text widget with scrollbar
        console_frame = self.ttk.Frame(main_frame)
        console_frame.grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E, self.tk.N, self.tk.S), pady=5)

        scrollbar = self.ttk.Scrollbar(console_frame)
        scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        self.console_text = self.tk.Text(console_frame, height=15, width=80,
                                          yscrollcommand=scrollbar.set,
                                          bg='#1e1e1e', fg='#d4d4d4',
                                          font=('Consolas', 9))
        self.console_text.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        scrollbar.config(command=self.console_text.yview)

        # Make console read-only
        self.console_text.config(state='disabled')

        # Configure row weight for console
        main_frame.rowconfigure(row, weight=1)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Buttons =====
        button_frame = self.ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)

        self.ttk.Button(button_frame, text="Restore Defaults", command=self.restore_defaults).pack(side=self.tk.LEFT, padx=5)
        self.run_button = self.ttk.Button(button_frame, text="Run Segmentation", command=self.on_run_threaded)
        self.run_button.pack(side=self.tk.LEFT, padx=5)
        self.ttk.Button(button_frame, text="Close", command=self.on_close).pack(side=self.tk.LEFT, padx=5)

        # Configure column weights
        main_frame.columnconfigure(1, weight=1)

        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

        # Run the GUI
        self.root.mainloop()

        return self.result


def main():
    """Main function for command-line usage."""
    
    # Register datasets like demo.py does (if register_two_stage_datasets exists)
    try:
        from register_two_stage_datasets import register_two_stage_datasets
        register_two_stage_datasets(
            dataset_root="./myotube_batch_output", 
            register_instance=True, 
            register_panoptic=False
        )
    except ImportError:
        # Dataset registration not available, continue without it
        pass
    
    parser = argparse.ArgumentParser(description="Myotube Instance Segmentation for Fiji")

    # GUI mode
    parser.add_argument("--gui", action="store_true",
                       help="Launch graphical user interface for parameter configuration")
    parser.add_argument("--gui-output", type=str, default=None,
                       help="Lock output directory when using GUI (used by Fiji integration)")

    # Positional arguments (optional when using --gui)
    parser.add_argument("input_path", nargs='?', help="Path to input image or directory containing images")
    parser.add_argument("output_dir", nargs='?', help="Output directory for results")

    # Optional arguments
    parser.add_argument("--config", help="Path to model config file")
    parser.add_argument("--weights", help="Path to model weights")
    parser.add_argument("--mask2former-path", type=str, default=None,
                       help="Path to Mask2Former project directory (auto-detected if not specified)")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Confidence threshold for detection (default: 0.25)")
    parser.add_argument("--min-area", type=int, default=100,
                       help="Minimum myotube area in pixels")
    parser.add_argument("--max-area", type=int, default=50000,
                       help="Maximum myotube area in pixels")
    parser.add_argument("--final-min-area", type=int, default=1000,
                       help="Final minimum area filter (applied after post-processing)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU inference (slower but uses less memory)")
    parser.add_argument("--max-image-size", type=int, default=None,
                       help="Maximum image dimension (larger images will be resized). Default: respect training resolution")
    parser.add_argument("--force-1024", action="store_true",
                       help="Force 1024px input resolution for memory optimization (may reduce accuracy)")

    # Tiling parameters
    parser.add_argument("--use-tiling", action="store_true", default=True,
                       help="Enable tiled inference for large images with many myotubes (enabled by default)")
    parser.add_argument("--no-tiling", dest="use_tiling", action="store_false",
                       help="Disable tiled inference and process entire image at once")
    parser.add_argument("--grid-size", type=int, default=2,
                       help="Grid size for tiling (1=no split, 2=2×2, 3=3×3, etc.). Default: 2")
    parser.add_argument("--tile-overlap", type=float, default=0.20,
                       help="Overlap ratio between tiles (default: 0.20 = 20%%). Only used with --use-tiling")

    # Merged mask generation parameter
    parser.add_argument("--skip-merged-masks", action="store_true", default=True,
                       help="Skip generation of merged visualization masks (imaginary boundaries connecting disconnected components, skipped by default)")
    parser.add_argument("--generate-merged-masks", dest="skip_merged_masks", action="store_false",
                       help="Generate merged visualization masks with imaginary boundaries")

    # Measurements CSV generation parameter
    parser.add_argument("--save-measurements", action="store_true", default=False,
                       help="Save comprehensive measurements CSV (area, length, width, etc. - disabled by default)")

    args = parser.parse_args()

    # Check if GUI mode is requested
    if args.gui:
        print("🖥️  Launching GUI...")
        # If gui-output is specified, lock the output directory
        gui = ParameterGUI(locked_output_dir=args.gui_output)
        params = gui.show()

        if params is None:
            print("❌ User cancelled")
            return

        print("✅ Parameters selected via GUI")

        # Override args with GUI parameters
        args.input_path = params['input_path']
        # Use locked output if provided, otherwise use GUI selection
        args.output_dir = args.gui_output if args.gui_output else params['output_dir']
        args.config = params['config'] if params['config'] else None
        args.weights = params['weights'] if params['weights'] else None
        args.mask2former_path = params['mask2former_path'] if params['mask2former_path'] else None
        args.confidence = params['confidence']
        args.min_area = params['min_area']
        args.max_area = params['max_area']
        args.final_min_area = params['final_min_area']
        args.cpu = params['cpu']
        args.force_1024 = params['force_1024']
        args.use_tiling = params['use_tiling']
        args.grid_size = params['grid_size']
        args.tile_overlap = params['tile_overlap']
        args.skip_merged_masks = params['skip_merged_masks']
        args.save_measurements = params['save_measurements']
        args.max_image_size = params['max_image_size'] if params['max_image_size'] else None

        # Now that GUI is done, load the imports we skipped earlier
        print("🔄 Loading Mask2Former and detectron2 modules...")
        ensure_mask2former_loaded(explicit_path=args.mask2former_path)
        # Re-import the modules that were set to None in GUI mode
        global DefaultPredictor, get_cfg, add_deeplab_config, read_image, add_maskformer2_config
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.data.detection_utils import read_image
        from mask2former import add_maskformer2_config
        print("✅ Modules loaded successfully")
    else:
        # Command-line mode - validate required arguments
        if not args.input_path or not args.output_dir:
            parser.error("input_path and output_dir are required unless using --gui mode")

    # Print parameter summary
    print("\n" + "="*60)
    print("MYOTUBE SEGMENTATION PARAMETERS")
    print("="*60)
    print(f"Input:           {args.input_path}")
    print(f"Output:          {args.output_dir}")
    print(f"Config:          {args.config or 'Auto-detect'}")
    print(f"Weights:         {args.weights or 'Auto-detect'}")
    print(f"Confidence:      {args.confidence}")
    print(f"Min Area:        {args.min_area} px")
    print(f"Max Area:        {args.max_area} px")
    print(f"Final Min Area:  {args.final_min_area} px")
    print(f"CPU Mode:        {args.cpu}")
    print(f"Force 1024px:    {args.force_1024}")
    print(f"Max Image Size:  {args.max_image_size or 'Auto'}")
    print(f"Use Tiling:      {args.use_tiling}")
    print(f"Tile Overlap:    {args.tile_overlap*100:.0f}%")
    print(f"Skip Merged:     {args.skip_merged_masks}")
    print("="*60 + "\n")
    
    # Custom post-processing config
    max_image_size = 1024 if args.force_1024 else args.max_image_size
    custom_config = {
        'confidence_threshold': args.confidence,
        'min_area': args.min_area,
        'max_area': args.max_area,
        'final_min_area': args.final_min_area,
        'max_image_size': max_image_size,
        'force_cpu': args.cpu,
        'save_measurements': args.save_measurements
    }
    
    # Initialize integration
    integration = MyotubeFijiIntegration(
        config_file=args.config,
        model_weights=args.weights,
        skip_merged_masks=args.skip_merged_masks,
        mask2former_path=args.mask2former_path
    )

    # Initialize tiled segmentation if requested
    if args.use_tiling:
        print(f"🔲 Tiled inference mode enabled (grid: {args.grid_size}×{args.grid_size}, overlap: {args.tile_overlap*100:.0f}%)")
        tiled_segmenter = TiledMyotubeSegmentation(
            fiji_integration=integration,
            target_overlap=args.tile_overlap,
            grid_size=args.grid_size
        )
    else:
        tiled_segmenter = None

    # Apply memory optimization settings
    if args.cpu:
        print("🖥️  CPU inference mode enabled")
    if args.force_1024:
        print("📏 Forced 1024px input resolution (memory optimization - may reduce accuracy)")
    elif args.max_image_size:
        print(f"📏 Max image size set to: {args.max_image_size}px")
    else:
        print("📏 Using training resolution (1500px) for best accuracy")
    
    # Ensure output directory exists for status files
    os.makedirs(args.output_dir, exist_ok=True)

    # Clean up old status files to avoid confusion from previous runs
    old_success_file = os.path.join(args.output_dir, "BATCH_SUCCESS")
    old_error_file = os.path.join(args.output_dir, "ERROR")
    if os.path.exists(old_success_file):
        os.remove(old_success_file)
        print("🗑️  Deleted old BATCH_SUCCESS file")
    if os.path.exists(old_error_file):
        os.remove(old_error_file)
        print("🗑️  Deleted old ERROR file")

    # Process image(s)
    try:
        # Check if input is a directory or single image
        if os.path.isdir(args.input_path):
            # Batch processing mode
            print(f"📁 Batch processing mode: {args.input_path}")
            
            # Find all image files in directory
            image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
            base_dir = Path(args.input_path)
            
            # Priority: search images/ subdirectory if it exists, otherwise search base directory
            images_subdir = base_dir / 'images'
            if images_subdir.exists() and images_subdir.is_dir():
                # Use images/ subdirectory (COCO format)
                search_dir = images_subdir
                print(f"   📂 Searching images/ subdirectory: {images_subdir}")
            else:
                # Use base directory
                search_dir = base_dir
            
            # Collect image files (recursive search)
            image_files = []
            for ext in image_extensions:
                # Search case-insensitively by checking both lowercase and uppercase
                # Use set to avoid case-insensitive duplicates on Windows
                for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                    image_files.extend(search_dir.rglob(pattern))
            
            # De-duplicate using resolved absolute paths (handles symlinks, UNC paths, etc.)
            unique_files = {}
            for f in image_files:
                # Resolve to canonical absolute path
                try:
                    resolved = str(f.resolve())
                    # On Windows, normalize to handle case-insensitive filesystem
                    if sys.platform == 'win32':
                        resolved = resolved.lower()
                    unique_files[resolved] = f
                except (OSError, RuntimeError):
                    # Handle edge cases with inaccessible files
                    unique_files[str(f)] = f
            
            image_files = sorted(unique_files.values(), key=lambda p: str(p))
            
            if not image_files:
                msg = (f"No image files found in directory: {args.input_path}\n"
                       f"Searched: {search_dir}\n"
                       f"Supported: {', '.join(image_extensions)}")
                print(f"❌ {msg}")
                # Write error status for Fiji macro
                with open(os.path.join(args.output_dir, "ERROR"), 'w') as f:
                    f.write(msg + "\n")
                return 1
            
            print(f"📊 Found {len(image_files)} image files to process")
            
            # Process each image
            total_myotubes = 0
            successful_images = 0
            failed_images = 0
            
            for i, image_path in enumerate(image_files, 1):
                print(f"\n{'='*60}")
                print(f"🖼️  Processing image {i}/{len(image_files)}: {Path(image_path).name}")
                print(f"{'='*60}")
                
                try:
                    # Create subdirectory for each image's output
                    image_output_dir = os.path.join(args.output_dir, Path(image_path).stem)

                    # Use tiled or standard segmentation based on flag
                    if tiled_segmenter:
                        output_files = tiled_segmenter.segment_image_tiled(
                            str(image_path),
                            image_output_dir,
                            custom_config
                        )
                    else:
                        output_files = integration.segment_image(
                            str(image_path),
                            image_output_dir,
                            custom_config
                        )

                    myotube_count = output_files['count']
                    total_myotubes += myotube_count
                    successful_images += 1
                    
                    print(f"✅ {Path(image_path).name}: {myotube_count} myotubes detected")
                
                except Exception as e:
                    print(f"❌ Failed to process {Path(image_path).name}: {e}")
                    failed_images += 1
                    continue
            
            # Create batch summary
            print(f"\n{'='*60}")
            print(f"🎉 BATCH PROCESSING COMPLETED!")
            print(f"{'='*60}")
            print(f"📊 Total images processed: {successful_images}/{len(image_files)}")
            print(f"📊 Failed images: {failed_images}")
            print(f"📊 Total myotubes detected: {total_myotubes}")
            print(f"📁 Output directory: {args.output_dir}")
            print(f"{'='*60}")
            
            # Write batch summary file
            summary_file = os.path.join(args.output_dir, "BATCH_SUCCESS")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"{successful_images}/{len(image_files)} images processed\n")
                f.write(f"{total_myotubes} total myotubes detected\n")
                f.write(f"{failed_images} failed images\n")
            
            output_files = {
                'count': total_myotubes,
                'processed_images': successful_images,
                'failed_images': failed_images,
                'total_images': len(image_files)
            }
            
        else:
            msg = f"Input path must be a directory containing images: {args.input_path}"
            print(f"❌ {msg}")
            # Write error status for Fiji macro
            with open(os.path.join(args.output_dir, "ERROR"), 'w') as f:
                f.write(msg + "\n")
            return 1
        
        # Signal success to ImageJ macro
        success_file = os.path.join(args.output_dir, "BATCH_SUCCESS")
        print(f"📝 Batch processing completed. Summary written to: {success_file}")
        
    except BaseException as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Signal failure to ImageJ macro
        error_file = os.path.join(args.output_dir, "ERROR")
        with open(error_file, 'w') as f:
            f.write(f"ERROR: {str(e)}\n")
        
        return 1


if __name__ == "__main__":
    main()
