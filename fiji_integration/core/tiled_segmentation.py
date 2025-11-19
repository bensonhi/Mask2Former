"""
Tiled inference for processing large images.

This module provides tiled segmentation for images that contain too many
instances for single-pass inference.
"""

import os
import numpy as np
import cv2
import torch
from typing import Dict, Any, List, Tuple, Optional

from fiji_integration.core.interfaces import SegmentationInterface


__all__ = ['TiledMyotubeSegmentation']


class TiledMyotubeSegmentation:
    """
    Tiled inference for processing large images that contain too many myotubes
    for single-pass inference (exceeds model's query capacity).

    Uses overlapping tiles to ensure boundary instances are captured,
    then merges detections across tiles using IoU-based matching.
    """

    def __init__(self, segmentation_backend: SegmentationInterface,
                 target_overlap: float = 0.20, grid_size: int = 2):
        """
        Initialize tiled segmentation wrapper.

        Args:
            segmentation_backend: SegmentationInterface implementation
            target_overlap: Overlap ratio between tiles (default: 0.20 = 20%)
            grid_size: Grid size for tiling (1=no split, 2=2×2, 3=3×3, etc.)
        """
        self.backend = segmentation_backend
        self.target_overlap = target_overlap
        self.grid_size = max(1, int(grid_size))  # Ensure grid_size >= 1

    def calculate_tiling_params(self, image_size: int) -> Tuple[int, int]:
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

    def create_tiles(self, image: np.ndarray, tile_size: int) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
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

    def _pad_to_size(self, tile: np.ndarray, target_size: int) -> np.ndarray:
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

    def transform_to_global_coords(self, instances: Dict[str, Any],
                                   tile_coords: Tuple[int, int, int, int],
                                   original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
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

    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Intersection over Union between two masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    def calculate_overlap_region_iou(self, inst1: Dict[str, Any], inst2: Dict[str, Any]) -> float:
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

    def merge_duplicates(self, all_instances: List[Dict[str, Any]],
                        iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
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

    def _boxes_overlap(self, box1: List[float], box2: List[float]) -> bool:
        """Fast check if two bounding boxes overlap."""
        # box format: [x_min, y_min, x_max, y_max]
        return not (box1[2] < box2[0] or  # box1 right of box2
                   box1[0] > box2[2] or  # box1 left of box2
                   box1[3] < box2[1] or  # box1 above box2
                   box1[1] > box2[3])    # box1 below box2

    def _merge_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    def convert_to_detectron_format(self, merged_instances: List[Dict[str, Any]],
                                    original_shape: Tuple[int, int]) -> Dict[str, Any]:
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

    def segment_image_tiled(self, image_path: str, output_dir: str,
                           custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
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
            self.backend._original_size = (original_h, original_w)
            self.backend._scale_factor = scale_factor
            self.backend._processing_size = (processing_h, processing_w)

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
            self.backend._original_size = None
            self.backend._scale_factor = 1.0
            self.backend._processing_size = None

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

            # Create a temporary output directory for this tile
            temp_output_dir = os.path.join(output_dir, f"_temp_tile_{idx}_output")
            os.makedirs(temp_output_dir, exist_ok=True)

            try:
                # Initialize predictor if needed
                force_cpu = custom_config.get('force_cpu', False) if custom_config else False
                self.backend.initialize_predictor(force_cpu=force_cpu)

                # Run inference on tile
                from detectron2.data.detection_utils import read_image
                tile_detectron = read_image(temp_tile_path, format="BGR")

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                predictions = self.backend.predictor(tile_detectron)
                instances = predictions["instances"]

                num_detections = len(instances)
                print(f"      → {num_detections} detections")

                if num_detections > 0:
                    # Convert to internal format
                    tile_instances = self.backend.post_processor._convert_to_internal_format(
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
            self.backend.post_processor.config.update(custom_config)

        # Use processing-resolution image for post-processing (not original high-res)
        processed_instances = self.backend.post_processor.process(instances_dict, image)

        # Save outputs using existing methods (pass both raw and processed instances)
        # Note: original_image is needed for overlays; masks will be upscaled in saving methods
        output_files = self.backend._generate_fiji_outputs(
            instances_dict,  # raw instances (after tile merging, before post-processing)
            processed_instances,  # processed instances (after post-processing)
            original_image,  # Original high-res image for overlays
            image_path,
            output_dir,
            custom_config  # pass custom_config for measurements settings
        )

        return output_files
