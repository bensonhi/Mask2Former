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
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
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

try:
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
        """
        if len(instances['masks']) <= 1:
            return instances
        
        from scipy import ndimage
        
        print(f"      🔍 Iteratively analyzing connected components across {len(instances['masks'])} instances")
        
        current_instances = {
            'masks': instances['masks'].copy(),
            'scores': instances['scores'].copy(),
            'boxes': instances['boxes'].copy(),
            'image_shape': instances['image_shape']
        }
        
        iteration = 0
        total_reassignments = 0
        total_eliminations = 0
        
        while iteration < 10:  # Safety limit to prevent infinite loops
            iteration += 1
            print(f"         Iteration {iteration}: analyzing {len(current_instances['masks'])} instances")
            
            masks = current_instances['masks']
            scores = current_instances['scores']
            boxes = current_instances['boxes']
            n = len(masks)
            
            # Extract ALL connected components from each instance (no early exit optimization)
            all_components = []  # List of (instance_idx, component_idx, component_mask)
            
            for i, mask in enumerate(masks):
                mask_area = mask.sum()
                if mask_area == 0:
                    continue
                    
                # Process all masks regardless of size
                
                # Find connected components in this instance
                labeled_mask, num_components = ndimage.label(mask.astype(bool))
                
                for comp_idx in range(1, num_components + 1):
                    component_mask = (labeled_mask == comp_idx)
                    all_components.append((i, comp_idx - 1, component_mask))
            
            if len(all_components) <= 1:
                print(f"         No components to process - stopping")
                break
                
            print(f"         Found {len(all_components)} components across {n} instances")
            
            # Track component reassignments for this iteration
            reassignments = {}
            
            # OPTIMIZATION: Pre-filter using bounding box overlap
            component_info = []  # [(instance, component_idx, mask, area, bbox)]
            for src_inst, src_comp_idx, src_mask in all_components:
                src_area = src_mask.sum()
                if src_area == 0:
                    continue
                
                # Calculate bounding box for quick filtering
                coords = np.where(src_mask)
                if len(coords[0]) > 0:
                    bbox = (coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max())
                    component_info.append((src_inst, src_comp_idx, src_mask, src_area, bbox))
            
            # Check each component against others with bounding box pre-filtering
            for i, (src_inst, src_comp_idx, src_mask, src_area, src_bbox) in enumerate(component_info):
                for j, (tgt_inst, tgt_comp_idx, tgt_mask, tgt_area, tgt_bbox) in enumerate(component_info):
                    if src_inst == tgt_inst or i == j:  # Same instance or same component, skip
                        continue
                    
                    # OPTIMIZATION: Quick bounding box containment check
                    src_y1, src_y2, src_x1, src_x2 = src_bbox
                    tgt_y1, tgt_y2, tgt_x1, tgt_x2 = tgt_bbox
                    
                    # If source bbox is not contained in target bbox, skip expensive computation
                    if not (tgt_y1 <= src_y1 and src_y2 <= tgt_y2 and tgt_x1 <= src_x1 and src_x2 <= tgt_x2):
                        continue
                    
                    # OPTIMIZATION: Only do expensive pixel-wise intersection for bbox-contained cases
                    intersection = np.logical_and(src_mask, tgt_mask).sum()
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
            
            # Start with original masks for unchanged instances
            new_instance_masks = [masks[i].copy() for i in range(n)]
            
            # Only rebuild changed instances
            for inst in changed_instances:
                new_instance_masks[inst] = np.zeros_like(masks[0], dtype=bool)
            
            # Process components with reassignments
            for src_inst, src_comp_idx, src_mask in all_components:
                if (src_inst, src_comp_idx) in reassignments:
                    # Reassigned component
                    target_instance = reassignments[(src_inst, src_comp_idx)]
                    new_instance_masks[target_instance] = np.logical_or(new_instance_masks[target_instance], src_mask)
                elif src_inst in changed_instances:
                    # Component stays but instance was affected by other changes
                    new_instance_masks[src_inst] = np.logical_or(new_instance_masks[src_inst], src_mask)
            
            # Create new instance set, eliminating empty ones
            new_masks = []
            new_scores = []
            new_boxes = []
            eliminated_this_iteration = 0
            
            for i in range(n):
                if new_instance_masks[i].sum() == 0:
                    print(f"         Instance {i}: eliminated (no components remaining)")
                    eliminated_this_iteration += 1
                    continue
                    
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
        
        # Generate outputs with both raw and processed overlays
        output_files = self._generate_fiji_outputs(
            instances, processed_instances, original_image, image_path, output_dir
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
                              image_path: str, output_dir: str) -> Dict[str, str]:
        """Generate all Fiji-compatible output files."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        
        # Generate individual mask images (pixel-perfect accuracy!) - using processed instances
        masks_dir = os.path.join(output_dir, f"{base_name}_masks")
        self._save_individual_mask_images(processed_instances, original_image, masks_dir)
        
        # Generate RAW Detectron2 overlay (before post-processing)
        raw_overlay_path = os.path.join(output_dir, f"{base_name}_raw_overlay.tif")
        self._save_colored_overlay(raw_instances, original_image, raw_overlay_path, overlay_type="raw")
        
        # Generate PROCESSED overlay (after post-processing)
        processed_overlay_path = os.path.join(output_dir, f"{base_name}_processed_overlay.tif")
        self._save_colored_overlay(processed_instances, original_image, processed_overlay_path, overlay_type="processed")
        
        # Generate measurements CSV - using processed instances
        measurements_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
        self._save_measurements(processed_instances, measurements_path)
        
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
    
    

    
    def _save_colored_overlay(self, instances, original_image: np.ndarray, 
                             output_path: str, overlay_type: str = "processed"):
        """Save colored overlay using Detectron2's built-in visualizer like demo.py."""
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog
        from detectron2.structures import Instances
        import torch
        
        print(f"   🎨 Generating {overlay_type} overlay using Detectron2's Visualizer")
        
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
            # Raw Detectron2 Instances - need to resize masks to original image size
            torch_instances = Instances(original_image.shape[:2])
            
            if len(instances) > 0:
                # Resize masks to original image size 
                raw_masks = instances.pred_masks.cpu().numpy()
                final_masks = []
                
                for i, mask in enumerate(raw_masks):
                    # Resize mask to original image size if needed
                    if hasattr(self, '_scale_factor') and self._scale_factor != 1.0:
                        original_h, original_w = self._original_size
                        # Convert boolean mask to uint8 for resizing
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        resized_mask = cv2.resize(
                            mask_uint8, 
                            (original_w, original_h), 
                            interpolation=cv2.INTER_NEAREST
                        )
                        resized_mask = (resized_mask > 128)  # Back to boolean
                    else:
                        resized_mask = mask > 0  # Ensure boolean
                    
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
            torch_instances = Instances(original_image.shape[:2])
            
            if len(instances['masks']) > 0:
                # Ensure masks are at original image resolution for visualization
                final_masks = []
                
                for i, mask in enumerate(instances['masks']):
                    # Ensure mask is a numpy array first
                    if torch.is_tensor(mask):
                        mask = mask.cpu().numpy()
                    
                    # Ensure mask is boolean
                    mask = mask.astype(bool)
                    
                    # Resize mask to original image size if needed
                    if hasattr(self, '_scale_factor') and self._scale_factor != 1.0:
                        original_h, original_w = self._original_size
                        # Convert to uint8 for resizing
                        mask_uint8 = mask.astype(np.uint8) * 255
                        resized_mask = cv2.resize(
                            mask_uint8, 
                            (original_w, original_h), 
                            interpolation=cv2.INTER_NEAREST
                        )
                        resized_mask = (resized_mask > 128)  # Back to boolean
                    else:
                        resized_mask = mask
                    
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
        rgb_image = original_image[:, :, ::-1]
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
            print(f"   💡 Creating fallback overlay with original image")
            # Fallback: save original image as overlay
            vis_image = original_image.copy()
        
        print(f"   💾 Saving overlay to: {output_path}")
        
        # Save the visualization
        success = cv2.imwrite(output_path, vis_image)
        if success:
            print(f"   ✅ {overlay_type.title()} overlay saved: {os.path.basename(output_path)}")
        else:
            print(f"   ❌ Failed to save {overlay_type} overlay")
            
        print(f"   🔍 {overlay_type.title()} overlay: {num_instances} instances visualized")
    
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
    parser.add_argument("input_path", help="Path to input image or directory containing images")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--config", help="Path to model config file")
    parser.add_argument("--weights", help="Path to model weights")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="Confidence threshold for detection")
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
    
    args = parser.parse_args()
    
    # Custom post-processing config
    max_image_size = 1024 if args.force_1024 else args.max_image_size
    custom_config = {
        'confidence_threshold': args.confidence,
        'min_area': args.min_area,
        'max_area': args.max_area,
        'final_min_area': args.final_min_area,
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
    
    # Ensure output directory exists for status files
    os.makedirs(args.output_dir, exist_ok=True)

    # Process image(s)
    try:
        # Check if input is a directory or single image
        if os.path.isdir(args.input_path):
            # Batch processing mode
            print(f"📁 Batch processing mode: {args.input_path}")
            
            # Find all image files in directory (robust: check images/ subdir and recurse)
            image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
            search_dirs = []
            base_dir = Path(args.input_path)
            images_subdir = base_dir / 'images'
            if images_subdir.exists():
                search_dirs.append(images_subdir)
            search_dirs.append(base_dir)

            image_files = []
            for sd in search_dirs:
                for ext in image_extensions:
                    image_files.extend(sd.glob(f"*{ext}"))
                    image_files.extend(sd.glob(f"*{ext.upper()}"))
                    # also recursive
                    image_files.extend(sd.rglob(f"*{ext}"))
                    image_files.extend(sd.rglob(f"*{ext.upper()}"))
            
            # De-duplicate
            image_files = sorted({str(p) for p in image_files})
            
            if not image_files:
                msg = (f"No image files found in directory: {args.input_path}\n"
                       f"Searched: {', '.join(str(d) for d in search_dirs)}\n"
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
