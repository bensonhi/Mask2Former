"""
Analysis tab for nuclei-myotube relationship analysis.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import threading
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import cv2
from scipy import ndimage
from skimage import measure
import pandas as pd

from fiji_integration.gui.base_tab import TabInterface


class NucleiMyotubeAnalyzer:
    """Analyzes spatial relationships between nuclei and myotubes."""

    def __init__(self, myotube_dir: str, nuclei_dir: str, output_dir: str,
                 overlap_threshold: float = 0.60,
                 periphery_overlap_threshold: float = 0.80,
                 min_nucleus_area: int = 400,
                 max_nucleus_area: int = 2000,
                 max_eccentricity: float = 0.9,
                 full_image_mode: bool = False,
                 skip_alignment_resize: bool = False,
                 progress_callback=None):
        """
        Initialize the analyzer.

        Args:
            myotube_dir: Directory containing myotube segmentation results
            nuclei_dir: Directory containing nuclei binary images
            output_dir: Directory where analysis results will be saved
            overlap_threshold: Minimum overlap ratio for nuclei-myotube assignment (default: 0.60)
            periphery_overlap_threshold: Threshold to distinguish central vs peripheral nuclei (default: 0.80)
            min_nucleus_area: Minimum nucleus area in pixels (default: 400)
            max_nucleus_area: Maximum nucleus area in pixels (default: 2000)
            max_eccentricity: Maximum eccentricity (default: 0.9, where 0=circle, 1=line)
            full_image_mode: If True, process full images without quadrant cropping (default: False)
            skip_alignment_resize: If True, skip resizing nuclei to match processed dimensions (default: False)
            progress_callback: Callback function to report progress
        """
        self.myotube_dir = Path(myotube_dir)
        self.nuclei_dir = Path(nuclei_dir)
        self.output_dir = Path(output_dir)
        self.overlap_threshold = overlap_threshold
        self.periphery_overlap_threshold = periphery_overlap_threshold
        self.min_nucleus_area = min_nucleus_area
        self.max_nucleus_area = max_nucleus_area
        self.max_eccentricity = max_eccentricity
        self.full_image_mode = full_image_mode
        self.skip_alignment_resize = skip_alignment_resize
        self.progress_callback = progress_callback

        # Results storage
        self.myotube_results = []  # For myotube-centric CSV
        self.nuclei_results = []   # For nuclei-centric CSV

        # Filter statistics
        self.filter_stats = {
            'total_detected': 0,
            'filtered_by_size': 0,
            'filtered_by_eccentricity': 0,
            'filtered_by_overlap': 0,
            'passed_all_filters': 0
        }

    def log(self, message: str):
        """Log a message via callback or print."""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message, flush=True)

    def find_nuclei_image(self, myotube_folder_name: str) -> Optional[Path]:
        """
        Find the corresponding nuclei binary image for a myotube folder.

        Args:
            myotube_folder_name: Name of the myotube segmentation folder

        Returns:
            Path to nuclei image or None if not found
        """
        # Get base name (remove position suffix if not in full_image_mode)
        base_name = myotube_folder_name
        if not self.full_image_mode:
            # Remove the position suffix (_bl, _br, _tl, _tr) to get base name
            for suffix in ['_bl', '_br', '_tl', '_tr']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break

        # Create list of base names to try (original + channel substitutions)
        base_names_to_try = [base_name]

        # Handle channel suffix substitution (_grey -> _blue, etc.)
        channel_substitutions = [
            ('_grey', '_blue'),
            ('_gray', '_blue'),
            ('_red', '_blue'),
            ('_green', '_blue'),
            ('_Merged_grey', '_Merged_blue'),
            ('_Merged_gray', '_Merged_blue'),
        ]

        for old_suffix, new_suffix in channel_substitutions:
            if old_suffix in base_name:
                alternative_name = base_name.replace(old_suffix, new_suffix)
                base_names_to_try.append(alternative_name)

        # Try each base name variant
        for base_name_variant in base_names_to_try:
            # First check for _seg.npy (from CellPose) - search recursively
            npy_pattern = f"{base_name_variant}_seg.npy"
            # Search in subdirectory first (exact match)
            subdir_path = self.nuclei_dir / base_name_variant
            if subdir_path.is_dir():
                npy_path = subdir_path / npy_pattern
                if npy_path.exists():
                    return npy_path

            # Then search recursively in all subdirectories
            for npy_path in self.nuclei_dir.rglob(npy_pattern):
                return npy_path

            # Then check for standard image formats - search recursively
            nuclei_patterns = [
                f"{base_name_variant}_nuclei.png",
                f"{base_name_variant}_nuclei.tif",
                f"{base_name_variant}_dapi.png",
                f"{base_name_variant}_dapi.tif",
                f"{base_name_variant}_blue.png",
                f"{base_name_variant}_blue.tif",
                f"{base_name_variant}_405.png",
                f"{base_name_variant}_405.tif",
                f"{base_name_variant}_binary.png",
                f"{base_name_variant}_binary.tif"
            ]

            # Search recursively for other image formats
            for pattern in nuclei_patterns:
                # First check in main directory
                nuclei_path = self.nuclei_dir / pattern
                if nuclei_path.exists():
                    return nuclei_path

                # Then search recursively in subdirectories
                for nuclei_path in self.nuclei_dir.rglob(pattern):
                    return nuclei_path

        return None

    def load_nuclei_from_cellpose(self, npy_path: Path) -> np.ndarray:
        """
        Load nuclei segmentation from CellPose _seg.npy file.

        Args:
            npy_path: Path to _seg.npy file

        Returns:
            Binary nuclei image (0 for background, 1 for nuclei)
        """
        # Load the segmentation masks
        data = np.load(str(npy_path), allow_pickle=True)

        # CellPose saves masks in different formats depending on version
        # Try to extract masks array
        if data.ndim == 0:
            # It's a 0-d array (object), extract the item
            masks = data.item()
            if isinstance(masks, dict):
                masks_array = masks.get('masks', None)
                if masks_array is None:
                    raise ValueError(f"No 'masks' key found in {npy_path}")
            else:
                masks_array = masks
        else:
            # It's already an array
            masks_array = data

        # Convert to binary (any non-zero value is nuclei)
        binary_nuclei = (masks_array > 0).astype(np.uint8)

        return binary_nuclei

    def get_crop_coordinates(self, folder_name: str, full_image_shape: Tuple[int, int],
                           crop_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate crop coordinates based on position suffix or return full image coordinates.

        Args:
            folder_name: Myotube folder name with position suffix (or full image name)
            full_image_shape: (height, width) of full nuclei image
            crop_shape: (height, width) of cropped myotube region

        Returns:
            (y1, y2, x1, x2) crop coordinates
        """
        h_full, w_full = full_image_shape
        h_crop, w_crop = crop_shape

        # If full_image_mode, return coordinates for the entire image
        if self.full_image_mode:
            return 0, h_full, 0, w_full

        # Extract position from folder name
        position = folder_name[-2:]  # Last two characters

        if position == 'tl':  # Top-left
            y1, x1 = 0, 0
        elif position == 'tr':  # Top-right
            y1, x1 = 0, w_full - w_crop
        elif position == 'bl':  # Bottom-left
            y1, x1 = h_full - h_crop, 0
        elif position == 'br':  # Bottom-right
            y1, x1 = h_full - h_crop, w_full - w_crop
        else:
            raise ValueError(f"Unknown position suffix: {position}")

        y2 = y1 + h_crop
        x2 = x1 + w_crop

        return y1, y2, x1, x2

    def load_myotube_masks(self, myotube_folder: Path) -> Dict[int, np.ndarray]:
        """
        Load individual myotube masks from the masks directory.

        Args:
            myotube_folder: Path to myotube segmentation result folder

        Returns:
            Dictionary mapping myotube_id to binary mask
        """
        masks_dir = myotube_folder / f"{myotube_folder.name}_masks"
        myotube_masks = {}

        if not masks_dir.exists():
            self.log(f"Warning: Masks directory not found: {masks_dir}")
            return myotube_masks

        self.log(f"Loading myotube masks...")
        mask_files = list(masks_dir.glob("Myotube_*_mask.png"))
        self.log(f"Found {len(mask_files)} mask files")

        for mask_file in mask_files:
            # Extract myotube ID from filename
            myotube_id = int(mask_file.stem.split('_')[1])

            # Load mask as binary
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            if mask is not None:
                myotube_masks[myotube_id] = (mask > 0).astype(np.uint8)

        self.log(f"Loaded {len(myotube_masks)} myotube masks")
        return myotube_masks

    def find_nuclei_components(self, nuclei_binary: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Find connected components (individual nuclei) in binary image.

        Args:
            nuclei_binary: Binary nuclei image

        Returns:
            Tuple of (labeled_nuclei_array, nuclei_list)
        """
        self.log(f"Labeling nuclei components...")
        # Label connected components
        labeled_nuclei = measure.label(nuclei_binary, connectivity=2)
        self.log(f"Computing region properties...")
        props = measure.regionprops(labeled_nuclei)

        nuclei_list = []
        for i, prop in enumerate(props):
            # Calculate circularity (4π × Area / Perimeter²)
            perimeter = prop.perimeter
            circularity = 4 * np.pi * prop.area / (perimeter ** 2) if perimeter > 0 else 0.0

            nuclei_list.append({
                'nucleus_id': i + 1,
                'label': prop.label,
                'area': prop.area,
                'centroid': prop.centroid,
                'bbox': prop.bbox,
                'circularity': circularity,
                'eccentricity': prop.eccentricity,
                'solidity': prop.solidity
            })

        self.log(f"Found {len(nuclei_list)} nuclei")
        return labeled_nuclei, nuclei_list

    def calculate_overlap(self, nucleus_mask: np.ndarray, myotube_mask: np.ndarray) -> Dict:
        """
        Calculate overlap metrics between nucleus and myotube.

        Args:
            nucleus_mask: Binary nucleus mask
            myotube_mask: Binary myotube mask

        Returns:
            Dictionary with overlap metrics
        """
        intersection = np.logical_and(nucleus_mask, myotube_mask)
        intersection_area = np.sum(intersection)
        nucleus_area = np.sum(nucleus_mask)
        myotube_area = np.sum(myotube_mask)

        if nucleus_area == 0:
            return {'overlap_pixels': 0, 'overlap_ratio': 0.0, 'iou': 0.0}

        overlap_ratio = intersection_area / nucleus_area
        union_area = nucleus_area + myotube_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return {
            'overlap_pixels': int(intersection_area),
            'overlap_ratio': float(overlap_ratio),
            'iou': float(iou)
        }

    def assign_myotube_pixels_to_nuclei(self, myotube_masks: Dict[int, np.ndarray],
                                      labeled_nuclei: np.ndarray,
                                      nuclei_list: List[Dict]) -> Dict[int, Dict]:
        """
        Assign each myotube pixel to the closest nucleus using distance transform.

        Args:
            myotube_masks: Dictionary of myotube masks
            labeled_nuclei: Labeled nuclei image
            nuclei_list: List of nucleus dictionaries

        Returns:
            Dictionary mapping myotube_id to nucleus assignment info
        """
        if not nuclei_list:
            return {}

        # Compute distance transform and find nearest nucleus label for each pixel
        distance, nearest_label = ndimage.distance_transform_edt(
            labeled_nuclei == 0,
            return_distances=True,
            return_indices=True
        )

        # Create a map where each pixel has the label of its nearest nucleus
        nearest_nucleus_map = labeled_nuclei.copy()
        mask = labeled_nuclei == 0
        nearest_nucleus_map[mask] = labeled_nuclei[tuple(nearest_label[:, mask])]

        myotube_assignments = {}

        for myotube_id, myotube_mask in myotube_masks.items():
            if np.sum(myotube_mask) == 0:
                continue

            # Get the nucleus labels for all myotube pixels
            myotube_pixel_labels = nearest_nucleus_map[myotube_mask > 0]

            # Count assignments to each nucleus
            nucleus_pixel_counts = {}
            unique_labels, counts = np.unique(myotube_pixel_labels, return_counts=True)

            for label, count in zip(unique_labels, counts):
                if label > 0:  # Skip background
                    nucleus_pixel_counts[int(label)] = int(count)

            # Find the nucleus with the most assigned pixels
            if nucleus_pixel_counts:
                closest_nucleus_id = max(nucleus_pixel_counts.keys(),
                                       key=lambda x: nucleus_pixel_counts[x])
                closest_nucleus_pixels = nucleus_pixel_counts[closest_nucleus_id]
            else:
                closest_nucleus_id = None
                closest_nucleus_pixels = 0

            myotube_assignments[myotube_id] = {
                'closest_nucleus_id': closest_nucleus_id,
                'closest_nucleus_pixels': closest_nucleus_pixels,
                'nucleus_pixel_counts': nucleus_pixel_counts
            }

        return myotube_assignments

    def analyze_sample(self, myotube_folder: Path) -> bool:
        """
        Analyze a single myotube segmentation sample.

        Args:
            myotube_folder: Path to myotube segmentation result folder

        Returns:
            True if analysis successful, False otherwise
        """
        folder_name = myotube_folder.name
        self.log(f"\nAnalyzing: {folder_name}")

        # Load myotube info
        self.log(f"Loading info file...")
        info_file = myotube_folder / f"{folder_name}_info.json"
        if not info_file.exists():
            self.log(f"Warning: Info file not found: {info_file}")
            return False

        with open(info_file, 'r') as f:
            info = json.load(f)

        image_shape = tuple(info['image_shape'])
        num_myotubes = info['num_instances']
        self.log(f"Image shape: {image_shape}, myotubes: {num_myotubes}")

        # Find corresponding nuclei image
        self.log(f"Finding nuclei image...")
        nuclei_image_path = self.find_nuclei_image(folder_name)
        if nuclei_image_path is None:
            base_name = folder_name
            for suffix in ['_bl', '_br', '_tl', '_tr']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            self.log(f"SKIPPED: No nuclei image found for {folder_name}")
            self.log(f"    Searched for patterns like: {base_name}_seg.npy, {base_name}_nuclei.png, etc.")
            return False

        # Load nuclei image
        self.log(f"Loading nuclei image: {nuclei_image_path.name}")

        # Check if it's a CellPose NPY file or regular image
        if nuclei_image_path.suffix == '.npy':
            try:
                nuclei_full = self.load_nuclei_from_cellpose(nuclei_image_path)
                self.log(f"Loaded CellPose segmentation: {nuclei_full.shape}")
            except Exception as e:
                self.log(f"Warning: Could not load CellPose NPY file: {e}")
                return False
        else:
            nuclei_full = cv2.imread(str(nuclei_image_path), cv2.IMREAD_GRAYSCALE)
            if nuclei_full is None:
                self.log(f"Warning: Could not load nuclei image: {nuclei_image_path}")
                return False
            self.log(f"Nuclei image loaded: {nuclei_full.shape}")

        # Handle alignment resize (if needed and not skipped)
        if not self.skip_alignment_resize and not self.full_image_mode:
            h_full, w_full = nuclei_full.shape
            expected_processed_h = image_shape[0] * 2
            expected_processed_w = image_shape[1] * 2

            if h_full != expected_processed_h or w_full != expected_processed_w:
                self.log(f"Resizing nuclei from {nuclei_full.shape} to ({expected_processed_h}, {expected_processed_w})")
                nuclei_full = cv2.resize(nuclei_full, (expected_processed_w, expected_processed_h), interpolation=cv2.INTER_NEAREST)
                self.log(f"Resized nuclei to: {nuclei_full.shape}")
        elif self.skip_alignment_resize:
            self.log(f"Skipping alignment resize (skip_alignment_resize=True)")

        # Crop nuclei image to match myotube region
        try:
            y1, y2, x1, x2 = self.get_crop_coordinates(folder_name, nuclei_full.shape, image_shape)
            nuclei_cropped = nuclei_full[y1:y2, x1:x2]

            if self.full_image_mode:
                self.log(f"Full image mode - using entire nuclei image: {nuclei_cropped.shape}")
            else:
                self.log(f"Cropped nuclei to: {nuclei_cropped.shape}")

            # Resize to match myotube image dimensions if needed
            if nuclei_cropped.shape != tuple(image_shape):
                self.log(f"Resizing nuclei from {nuclei_cropped.shape} to {image_shape}")
                nuclei_cropped = cv2.resize(nuclei_cropped, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
                self.log(f"Resized nuclei to: {nuclei_cropped.shape}")
        except ValueError as e:
            self.log(f"Warning: {e}")
            return False

        # Ensure nuclei image is binary
        if nuclei_cropped.max() > 1:
            nuclei_cropped = (nuclei_cropped > 127).astype(np.uint8)
        self.log(f"Binary conversion done")

        # Load myotube masks
        myotube_masks = self.load_myotube_masks(myotube_folder)
        if not myotube_masks:
            self.log(f"Warning: No myotube masks found for {folder_name}")
            return False

        # Get actual image shape from the mask files (not from info.json which may be processing resolution)
        first_mask = next(iter(myotube_masks.values()))
        actual_image_shape = first_mask.shape
        self.log(f"Actual mask dimensions: {actual_image_shape}")

        # Update image_shape to match actual mask dimensions
        if actual_image_shape != tuple(image_shape):
            self.log(f"Info file reports {image_shape}, but masks are at {actual_image_shape}")
            self.log(f"Using actual mask dimensions: {actual_image_shape}")
            image_shape = actual_image_shape

        # Find nuclei components
        labeled_nuclei, nuclei_list = self.find_nuclei_components(nuclei_cropped)
        self.log(f"Found {len(nuclei_list)} nuclei and {len(myotube_masks)} myotubes")

        # Assign myotube pixels to nuclei
        self.log(f"Computing myotube-nucleus assignments...")
        myotube_assignments = self.assign_myotube_pixels_to_nuclei(myotube_masks, labeled_nuclei, nuclei_list)
        self.log(f"Assignments computed.")

        # Create labeled myotube image
        self.log(f"Creating labeled myotube image...")
        labeled_myotubes = np.zeros(image_shape, dtype=np.int32)
        for myotube_id, myotube_mask in myotube_masks.items():
            labeled_myotubes[myotube_mask > 0] = myotube_id

        # Pre-compute myotube areas
        myotube_areas = {myotube_id: np.sum(mask) for myotube_id, mask in myotube_masks.items()}
        self.log(f"Labeled myotube image created")

        # Analyze each nucleus with sequential filtering
        self.log(f"Analyzing nucleus-myotube overlaps with filtering...")
        for nucleus in nuclei_list:
            nucleus_id = nucleus['nucleus_id']
            nucleus_label = nucleus['label']
            nucleus_area = nucleus['area']
            nucleus_eccentricity = nucleus['eccentricity']

            # Track total detected nuclei
            self.filter_stats['total_detected'] += 1

            # FILTER 1: Size filtering
            if nucleus_area < self.min_nucleus_area or nucleus_area > self.max_nucleus_area:
                self.filter_stats['filtered_by_size'] += 1
                self.nuclei_results.append({
                    'nucleus_id': nucleus_id,
                    'nucleus_area': nucleus_area,
                    'circularity': nucleus['circularity'],
                    'eccentricity': nucleus_eccentricity,
                    'solidity': nucleus['solidity'],
                    'assigned_myotube_id': None,
                    'overlap_pixels': 0,
                    'overlap_percentage': 0.0,
                    'myotube_pixels_assigned_to_nucleus': 0,
                    'filter_status': 'filtered_size',
                    'filter_reason': f'Area {nucleus_area} outside range [{self.min_nucleus_area}, {self.max_nucleus_area}]'
                })
                continue

            # FILTER 2: Eccentricity filtering
            if nucleus_eccentricity > self.max_eccentricity:
                self.filter_stats['filtered_by_eccentricity'] += 1
                self.nuclei_results.append({
                    'nucleus_id': nucleus_id,
                    'nucleus_area': nucleus_area,
                    'circularity': nucleus['circularity'],
                    'eccentricity': nucleus_eccentricity,
                    'solidity': nucleus['solidity'],
                    'assigned_myotube_id': None,
                    'overlap_pixels': 0,
                    'overlap_percentage': 0.0,
                    'myotube_pixels_assigned_to_nucleus': 0,
                    'filter_status': 'filtered_eccentricity',
                    'filter_reason': f'Eccentricity {nucleus_eccentricity:.3f} > {self.max_eccentricity}'
                })
                continue

            # Nucleus passed size and eccentricity filters - check overlap
            nucleus_pixels = (labeled_nuclei == nucleus_label)

            # Get myotube labels that overlap with this nucleus
            overlapping_myotube_labels = labeled_myotubes[nucleus_pixels]
            overlapping_myotube_labels = overlapping_myotube_labels[overlapping_myotube_labels > 0]

            if len(overlapping_myotube_labels) == 0:
                assigned_myotube = None
                best_overlap_pixels = 0
                best_overlap = 0.0
                myotube_pixels_for_nucleus = 0
            else:
                # Count overlap pixels for each myotube
                unique_labels, counts = np.unique(overlapping_myotube_labels, return_counts=True)

                # Find myotube with highest overlap ratio
                best_myotube_id = None
                best_overlap = 0.0
                best_overlap_pixels = 0

                for myotube_id, overlap_pixels in zip(unique_labels, counts):
                    overlap_ratio = overlap_pixels / nucleus_area
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_overlap_pixels = int(overlap_pixels)
                        best_myotube_id = int(myotube_id)

                # Only assign if overlap meets threshold
                assigned_myotube = best_myotube_id if best_overlap >= self.overlap_threshold else None

                # Get myotube pixels assigned to this nucleus
                myotube_pixels_for_nucleus = 0
                if assigned_myotube and assigned_myotube in myotube_assignments:
                    assignment_info = myotube_assignments[assigned_myotube]
                    if nucleus_id in assignment_info['nucleus_pixel_counts']:
                        myotube_pixels_for_nucleus = assignment_info['nucleus_pixel_counts'][nucleus_id]

            # FILTER 3: Overlap threshold
            if len(overlapping_myotube_labels) > 0 and assigned_myotube is None:
                self.filter_stats['filtered_by_overlap'] += 1
                filter_status = 'filtered_overlap'
                filter_reason = f'Best overlap {best_overlap*100:.1f}% < threshold {self.overlap_threshold*100:.1f}%'
            elif assigned_myotube is not None:
                self.filter_stats['passed_all_filters'] += 1
                filter_status = 'passed'
                filter_reason = 'Passed all filters'
            else:
                self.filter_stats['filtered_by_overlap'] += 1
                filter_status = 'filtered_overlap'
                filter_reason = 'No overlap with any myotube'

            # Store nuclei-centric result
            self.nuclei_results.append({
                'nucleus_id': nucleus_id,
                'nucleus_area': nucleus_area,
                'circularity': nucleus['circularity'],
                'eccentricity': nucleus_eccentricity,
                'solidity': nucleus['solidity'],
                'assigned_myotube_id': assigned_myotube,
                'overlap_pixels': best_overlap_pixels,
                'overlap_percentage': best_overlap * 100,
                'myotube_pixels_assigned_to_nucleus': myotube_pixels_for_nucleus,
                'filter_status': filter_status,
                'filter_reason': filter_reason
            })

        # Count nuclei per myotube
        sample_myotube_results = []
        for myotube_id in myotube_masks.keys():
            count = sum(1 for result in self.nuclei_results
                       if result['assigned_myotube_id'] == myotube_id)

            sample_myotube_results.append({
                'myotube_id': myotube_id,
                'myotube_area': myotube_areas[myotube_id],
                'nuclei_count': count
            })

        # Get relative path to preserve folder structure
        try:
            rel_path = myotube_folder.relative_to(self.myotube_dir)
        except ValueError:
            rel_path = Path(myotube_folder.name)

        # Create output folder for this sample
        sample_output_dir = self.output_dir / rel_path
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        # Save cropped nuclei image to output folder
        nuclei_output_path = sample_output_dir / f"{folder_name}_nuclei_cropped.png"
        cv2.imwrite(str(nuclei_output_path), nuclei_cropped * 255)

        # Create nuclei overlay visualization (all nuclei with filter status)
        self.create_nuclei_overlay(myotube_folder, sample_output_dir, folder_name, labeled_nuclei, nuclei_list)

        # Create periphery overlay visualization (only assigned nuclei, colored by overlap)
        self.create_periphery_overlay(myotube_folder, sample_output_dir, folder_name, labeled_nuclei, nuclei_list)

        # Save CSVs for this sample
        self.save_sample_results(sample_output_dir, folder_name, sample_myotube_results)

        # Clear results for next sample
        self.nuclei_results = []

        return True

    def create_nuclei_overlay(self, myotube_sample_folder: Path, output_sample_folder: Path, sample_name: str,
                             labeled_nuclei: np.ndarray, nuclei_list: List[Dict]):
        """
        Create visualization overlay showing filtered and assigned nuclei.

        Args:
            myotube_sample_folder: Path to the myotube sample folder (where overlay exists)
            output_sample_folder: Path to output folder for this sample
            sample_name: Name of the sample
            labeled_nuclei: Labeled nuclei image
            nuclei_list: List of nucleus dictionaries
        """
        overlay_path = myotube_sample_folder / f"{sample_name}_processed_overlay.tif"
        if not overlay_path.exists():
            self.log(f"  Warning: Processed overlay not found: {overlay_path.name}")
            return

        overlay = cv2.imread(str(overlay_path))
        if overlay is None:
            self.log(f"  Warning: Could not load overlay image")
            return

        # Check if overlay and nuclei dimensions match - resize nuclei if needed
        overlay_h, overlay_w = overlay.shape[:2]
        nuclei_h, nuclei_w = labeled_nuclei.shape

        if (overlay_h, overlay_w) != (nuclei_h, nuclei_w):
            self.log(f"  Note: Overlay is {overlay_w}×{overlay_h}, nuclei are {nuclei_w}×{nuclei_h}")
            self.log(f"  Resizing nuclei to match overlay dimensions for visualization")
            labeled_nuclei_resized = cv2.resize(labeled_nuclei, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
            # Also need to adjust nuclei_list centroids
            scale_y = overlay_h / nuclei_h
            scale_x = overlay_w / nuclei_w
            nuclei_list_scaled = []
            for nucleus in nuclei_list:
                nucleus_scaled = nucleus.copy()
                nucleus_scaled['centroid'] = (nucleus['centroid'][0] * scale_y, nucleus['centroid'][1] * scale_x)
                nuclei_list_scaled.append(nucleus_scaled)
            labeled_nuclei = labeled_nuclei_resized
            nuclei_list = nuclei_list_scaled

        # Create nucleus filter status mapping
        nucleus_filter_map = {}
        for result in self.nuclei_results:
            nucleus_filter_map[result['nucleus_id']] = result['filter_status']

        # Draw nuclei contours
        for nucleus in nuclei_list:
            nucleus_id = nucleus['nucleus_id']
            nucleus_label = nucleus['label']
            centroid = nucleus['centroid']

            nucleus_mask = (labeled_nuclei == nucleus_label).astype(np.uint8)
            contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Color based on filter status
            filter_status = nucleus_filter_map.get(nucleus_id, 'unknown')
            if filter_status == 'passed':
                color = (0, 255, 0)  # Green
            elif filter_status == 'filtered_size':
                color = (0, 0, 255)  # Red
            elif filter_status == 'filtered_eccentricity':
                color = (0, 255, 255)  # Yellow
            elif filter_status == 'filtered_overlap':
                color = (255, 0, 0)  # Blue
            else:
                color = (128, 128, 128)  # Gray

            cv2.drawContours(overlay, contours, -1, color, 2)

            # Draw nucleus ID label
            label_text = f"{nucleus_id}"
            centroid_pos = (int(centroid[1]), int(centroid[0]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            text_pos = (centroid_pos[0] - text_w // 2, centroid_pos[1] + text_h // 2)
            cv2.putText(overlay, label_text, text_pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Save overlay to output folder
        output_path = output_sample_folder / f"{sample_name}_nuclei_overlay.tif"
        cv2.imwrite(str(output_path), overlay)
        self.log(f"  Saved: {output_path.name}")

    def create_periphery_overlay(self, myotube_sample_folder: Path, output_sample_folder: Path, sample_name: str,
                                 labeled_nuclei: np.ndarray, nuclei_list: List[Dict]):
        """
        Create visualization overlay showing only assigned nuclei, colored by central vs peripheral.

        Args:
            myotube_sample_folder: Path to the myotube sample folder (where overlay exists)
            output_sample_folder: Path to output folder for this sample
            sample_name: Name of the sample
            labeled_nuclei: Labeled nuclei image
            nuclei_list: List of nucleus dictionaries
        """
        overlay_path = myotube_sample_folder / f"{sample_name}_processed_overlay.tif"
        if not overlay_path.exists():
            self.log(f"  Warning: Processed overlay not found for periphery overlay: {overlay_path.name}")
            return

        overlay = cv2.imread(str(overlay_path))
        if overlay is None:
            self.log(f"  Warning: Could not load overlay image for periphery overlay")
            return

        # Check if overlay and nuclei dimensions match - resize nuclei if needed
        overlay_h, overlay_w = overlay.shape[:2]
        nuclei_h, nuclei_w = labeled_nuclei.shape

        if (overlay_h, overlay_w) != (nuclei_h, nuclei_w):
            labeled_nuclei_resized = cv2.resize(labeled_nuclei, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
            # Also need to adjust nuclei_list centroids
            scale_y = overlay_h / nuclei_h
            scale_x = overlay_w / nuclei_w
            nuclei_list_scaled = []
            for nucleus in nuclei_list:
                nucleus_scaled = nucleus.copy()
                nucleus_scaled['centroid'] = (nucleus['centroid'][0] * scale_y, nucleus['centroid'][1] * scale_x)
                nuclei_list_scaled.append(nucleus_scaled)
            labeled_nuclei = labeled_nuclei_resized
            nuclei_list = nuclei_list_scaled

        # Create nucleus data mapping (only for passed nuclei)
        nucleus_data_map = {}
        for result in self.nuclei_results:
            if result['filter_status'] == 'passed' and result['assigned_myotube_id'] is not None:
                nucleus_data_map[result['nucleus_id']] = {
                    'overlap_percentage': result['overlap_percentage']
                }

        # Draw only assigned nuclei contours
        for nucleus in nuclei_list:
            nucleus_id = nucleus['nucleus_id']

            # Skip if not in the assigned nuclei map
            if nucleus_id not in nucleus_data_map:
                continue

            nucleus_label = nucleus['label']
            centroid = nucleus['centroid']
            overlap_pct = nucleus_data_map[nucleus_id]['overlap_percentage']

            nucleus_mask = (labeled_nuclei == nucleus_label).astype(np.uint8)
            contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Color based on overlap threshold
            if overlap_pct >= self.periphery_overlap_threshold * 100:
                color = (0, 255, 0)  # Green - central nuclei
            else:
                color = (0, 255, 255)  # Yellow - peripheral nuclei

            cv2.drawContours(overlay, contours, -1, color, 2)

            # Draw nucleus ID label
            label_text = f"{nucleus_id}"
            centroid_pos = (int(centroid[1]), int(centroid[0]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            text_pos = (centroid_pos[0] - text_w // 2, centroid_pos[1] + text_h // 2)
            cv2.putText(overlay, label_text, text_pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Save overlay to output folder
        output_path = output_sample_folder / f"{sample_name}_periphery_overlay.tif"
        cv2.imwrite(str(output_path), overlay)
        self.log(f"  Saved: {output_path.name}")

    def save_sample_results(self, sample_folder: Path, sample_name: str, myotube_results: List[Dict]):
        """Save CSV results and summary for a single sample."""
        # Save myotube-centric CSV
        myotube_csv_path = sample_folder / f"{sample_name}_myotube_nuclei_counts.csv"
        myotube_df = pd.DataFrame(myotube_results)
        myotube_df.to_csv(myotube_csv_path, index=False)
        self.log(f"  Saved: {myotube_csv_path.name}")

        # Save nuclei-centric CSV
        nuclei_csv_path = sample_folder / f"{sample_name}_nuclei_myotube_assignments.csv"
        nuclei_df = pd.DataFrame(self.nuclei_results)
        nuclei_df.to_csv(nuclei_csv_path, index=False)
        self.log(f"  Saved: {nuclei_csv_path.name}")

        # Save summary
        summary_path = sample_folder / f"{sample_name}_analysis_summary.txt"
        self._write_summary(summary_path, sample_name, myotube_df, nuclei_df)
        self.log(f"  Saved: {summary_path.name}")

    def _write_summary(self, summary_path: Path, sample_name: str, myotube_df: pd.DataFrame, nuclei_df: pd.DataFrame):
        """Write analysis summary file."""
        total_myotubes = len(myotube_df)
        total_nuclei = len(nuclei_df)

        passed_nuclei_df = nuclei_df[nuclei_df['filter_status'] == 'passed']
        filtered_size_df = nuclei_df[nuclei_df['filter_status'] == 'filtered_size']
        filtered_ecc_df = nuclei_df[nuclei_df['filter_status'] == 'filtered_eccentricity']
        filtered_overlap_df = nuclei_df[nuclei_df['filter_status'] == 'filtered_overlap']

        num_passed = len(passed_nuclei_df)
        num_filtered_size = len(filtered_size_df)
        num_filtered_ecc = len(filtered_ecc_df)
        num_filtered_overlap = len(filtered_overlap_df)

        assigned_nuclei_df = nuclei_df[nuclei_df['assigned_myotube_id'].notna()]
        num_assigned_nuclei = len(assigned_nuclei_df)

        myotubes_with_nuclei = myotube_df[myotube_df['nuclei_count'] > 0]
        num_myotubes_with_nuclei = len(myotubes_with_nuclei)
        num_myotubes_without_nuclei = total_myotubes - num_myotubes_with_nuclei

        avg_nuclei_per_myotube = myotube_df['nuclei_count'].mean()
        total_myotube_area = myotube_df['myotube_area'].sum()
        avg_myotube_area = myotube_df['myotube_area'].mean()

        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"NUCLEI-MYOTUBE ANALYSIS SUMMARY\n")
            f.write(f"Sample: {sample_name}\n")
            f.write("=" * 80 + "\n\n")

            f.write("OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total myotubes analyzed:        {total_myotubes}\n")
            f.write(f"Total nuclei detected:           {total_nuclei}\n\n")

            f.write("FILTER SETTINGS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Size range:                      {self.min_nucleus_area} - {self.max_nucleus_area} pixels\n")
            f.write(f"Max eccentricity:                {self.max_eccentricity}\n")
            f.write(f"Overlap threshold:               {self.overlap_threshold * 100:.1f}%\n\n")

            f.write("SEQUENTIAL FILTER RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total nuclei detected:           {total_nuclei}\n")
            if total_nuclei > 0:
                f.write(f"  Filtered by size:              {num_filtered_size} ({num_filtered_size/total_nuclei*100:.1f}%)\n")
                f.write(f"  Filtered by eccentricity:      {num_filtered_ecc} ({num_filtered_ecc/total_nuclei*100:.1f}%)\n")
                f.write(f"  Filtered by overlap:           {num_filtered_overlap} ({num_filtered_overlap/total_nuclei*100:.1f}%)\n")
                f.write(f"  Passed all filters:            {num_passed} ({num_passed/total_nuclei*100:.1f}%)\n\n")

            f.write("MYOTUBE STATISTICS\n")
            f.write("-" * 80 + "\n")
            if total_myotubes > 0:
                f.write(f"Myotubes with nuclei:            {num_myotubes_with_nuclei} ({num_myotubes_with_nuclei/total_myotubes*100:.1f}%)\n")
                f.write(f"Myotubes without nuclei:         {num_myotubes_without_nuclei} ({num_myotubes_without_nuclei/total_myotubes*100:.1f}%)\n")
            f.write(f"Average nuclei per myotube:      {avg_nuclei_per_myotube:.2f}\n")
            f.write(f"Total myotube area (pixels):     {total_myotube_area:,.0f}\n")
            f.write(f"Average myotube area (pixels):   {avg_myotube_area:,.0f}\n\n")

            f.write("=" * 80 + "\n")

    def analyze_all_samples(self):
        """Analyze all myotube segmentation samples."""
        # Recursively find all folders containing _info.json files
        myotube_folders = []
        for info_file in self.myotube_dir.rglob('*_info.json'):
            folder = info_file.parent
            if not folder.name.startswith('.'):
                myotube_folders.append(folder)

        self.log(f"Found {len(myotube_folders)} myotube samples to analyze")
        self.log(f"Looking for nuclei images in: {self.nuclei_dir}")
        self.log(f"Overlap threshold: {self.overlap_threshold}")
        self.log("-" * 80)

        successful_analyses = 0
        skipped_no_nuclei = 0
        failed_analyses = 0

        for i, folder in enumerate(myotube_folders):
            self.log(f"[{i+1}/{len(myotube_folders)}]")
            result = self.analyze_sample(folder)
            if result:
                successful_analyses += 1
            else:
                nuclei_path = self.find_nuclei_image(folder.name)
                if nuclei_path is None:
                    skipped_no_nuclei += 1
                else:
                    failed_analyses += 1

        self.log("-" * 80)
        self.log(f"ANALYSIS SUMMARY:")
        self.log(f"   Successfully analyzed: {successful_analyses}")
        self.log(f"   Skipped (no nuclei): {skipped_no_nuclei}")
        self.log(f"   Failed (other error): {failed_analyses}")
        self.log(f"   Total samples: {len(myotube_folders)}")


class AnalysisTab(TabInterface):
    """Tab for nuclei-myotube relationship analysis."""

    def __init__(self, config_file=None):
        """
        Initialize the analysis tab.

        Args:
            config_file: Path to config file (default: auto-detect)
        """
        super().__init__()

        # Config file location
        if config_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            fiji_integration_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            config_file = os.path.join(fiji_integration_dir, '.analysis_gui_config.json')
        self.config_file = config_file

        # Default parameters
        home_dir = os.path.expanduser('~')
        workflow_base = os.path.join(home_dir, 'fiji_workflow')

        self.default_params = {
            'myotube_folder': os.path.join(workflow_base, '2_myotube_segmentation'),
            'nuclei_folder': os.path.join(workflow_base, '3_cellpose_segmentation'),
            'output_folder': os.path.join(workflow_base, '4_nuclei_myotube_analysis'),
            'min_area': 400,
            'max_area': 6000,
            'max_eccentricity': 0.95,
            'overlap_threshold': 0.1,
            'periphery_overlap_threshold': 0.95,
            'full_image_mode': True,
        }

        # Load saved parameters or use defaults
        self.params = self.load_config()

        # GUI widgets (will be created in build_ui)
        self.myotube_folder_var = None
        self.nuclei_folder_var = None
        self.output_folder_var = None
        self.min_area_var = None
        self.max_area_var = None
        self.max_eccentricity_var = None
        self.overlap_threshold_var = None
        self.periphery_overlap_threshold_var = None
        self.full_image_mode_var = None
        self.run_button = None
        self.stop_button = None

    def get_tab_name(self) -> str:
        return "Nuclei-Myotube Analysis"

    def load_config(self):
        """Load saved configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved = json.load(f)
                # Merge with defaults to handle new parameters
                params = self.default_params.copy()
                params.update(saved)
                print(f"[LOADED] Loaded saved Analysis configuration from: {self.config_file}")
                return params
            except Exception as e:
                print(f"[WARNING]  Could not load config file: {e}")
                return self.default_params.copy()
        else:
            return self.default_params.copy()

    def save_config(self, config=None):
        """Save configuration to file."""
        if config is None:
            config = self.params
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[SAVED] Saved configuration to: {self.config_file}")
        except Exception as e:
            print(f"[WARNING]  Could not save config file: {e}")

    def validate_parameters(self):
        """Validate current parameters."""
        myotube_folder = self.params['myotube_folder'].strip()
        nuclei_folder = self.params['nuclei_folder'].strip()

        if not myotube_folder:
            return False, "Please select a myotube results folder"

        if not nuclei_folder:
            return False, "Please select a nuclei results folder"

        if not os.path.exists(myotube_folder):
            return False, f"Myotube folder does not exist: {myotube_folder}"

        if not os.path.exists(nuclei_folder):
            return False, f"Nuclei folder does not exist: {nuclei_folder}"

        min_area = self.params['min_area']
        max_area = self.params['max_area']
        if min_area >= max_area:
            return False, "Min area must be less than max area"

        max_eccentricity = self.params['max_eccentricity']
        if max_eccentricity < 0 or max_eccentricity > 1:
            return False, "Max eccentricity must be between 0 and 1"

        overlap_threshold = self.params['overlap_threshold']
        if overlap_threshold < 0 or overlap_threshold > 1:
            return False, "Overlap threshold must be between 0 and 1"

        return True, None

    def build_ui(self, parent_frame, console_text):
        """Build the UI for this tab."""
        self.console_text = console_text

        # Create tkinter variables from saved params
        self.myotube_folder_var = tk.StringVar(value=self.params['myotube_folder'])
        self.nuclei_folder_var = tk.StringVar(value=self.params['nuclei_folder'])
        self.output_folder_var = tk.StringVar(value=self.params['output_folder'])
        self.min_area_var = tk.StringVar(value=str(self.params['min_area']))
        self.max_area_var = tk.StringVar(value=str(self.params['max_area']))
        self.max_eccentricity_var = tk.StringVar(value=str(self.params['max_eccentricity']))
        self.overlap_threshold_var = tk.StringVar(value=str(self.params['overlap_threshold']))
        self.periphery_overlap_threshold_var = tk.StringVar(value=str(self.params['periphery_overlap_threshold']))
        self.full_image_mode_var = tk.BooleanVar(value=self.params['full_image_mode'])

        # Input/Output Section
        io_frame = ttk.LabelFrame(parent_frame, text="Input/Output", padding=10)
        io_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Myotube folder
        ttk.Label(io_frame, text="Myotube Results Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        myotube_entry = ttk.Entry(io_frame, textvariable=self.myotube_folder_var, width=50)
        myotube_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse", command=self.browse_myotube_folder).grid(row=0, column=2, padx=5, pady=2)

        # Nuclei folder
        ttk.Label(io_frame, text="Nuclei Results Folder:").grid(row=1, column=0, sticky=tk.W, pady=2)
        nuclei_entry = ttk.Entry(io_frame, textvariable=self.nuclei_folder_var, width=50)
        nuclei_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse", command=self.browse_nuclei_folder).grid(row=1, column=2, padx=5, pady=2)

        # Output folder
        ttk.Label(io_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, pady=2)
        output_entry = ttk.Entry(io_frame, textvariable=self.output_folder_var, width=50)
        output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse", command=self.browse_output_folder).grid(row=2, column=2, padx=5, pady=2)

        io_frame.columnconfigure(1, weight=1)

        # Filter Parameters Section
        param_frame = ttk.LabelFrame(parent_frame, text="Filter Parameters", padding=10)
        param_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Size range
        ttk.Label(param_frame, text="Nucleus Size Range (pixels):").grid(row=0, column=0, sticky=tk.W, pady=2)
        size_frame = ttk.Frame(param_frame)
        size_frame.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(size_frame, text="Min:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(size_frame, textvariable=self.min_area_var, width=10).pack(side=tk.LEFT, padx=2)

        ttk.Label(size_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(size_frame, textvariable=self.max_area_var, width=10).pack(side=tk.LEFT, padx=2)

        # Eccentricity
        ttk.Label(param_frame, text="Max Eccentricity (0-1):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(param_frame, textvariable=self.max_eccentricity_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)

        # Overlap threshold
        ttk.Label(param_frame, text="Overlap Threshold (0-1):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(param_frame, textvariable=self.overlap_threshold_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)

        # Periphery overlap threshold
        ttk.Label(param_frame, text="Periphery Overlap Threshold (0-1):").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(param_frame, textvariable=self.periphery_overlap_threshold_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)

        # Processing Options Section
        options_frame = ttk.LabelFrame(parent_frame, text="Processing Options", padding=10)
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        ttk.Checkbutton(options_frame, text="Full Image Mode (process complete images without quadrant cropping)",
                       variable=self.full_image_mode_var).grid(row=0, column=0, sticky=tk.W, pady=2)

        # Create buttons in button frame
        self.run_button = ttk.Button(self.button_frame, text="Run Analysis", command=self.on_run_threaded)
        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.on_stop, state='disabled')
        self.restore_button = ttk.Button(self.button_frame, text="Restore Defaults", command=self.restore_defaults)

    def get_button_frame_widgets(self):
        """Return list of (button, side) tuples for button frame."""
        return [
            (self.restore_button, tk.LEFT),
            (self.run_button, tk.LEFT),
            (self.stop_button, tk.LEFT)
        ]

    def browse_myotube_folder(self):
        """Browse for myotube results folder."""
        folder = filedialog.askdirectory(title="Select Myotube Results Folder")
        if folder:
            self.myotube_folder_var.set(folder)

    def browse_nuclei_folder(self):
        """Browse for nuclei results folder."""
        folder = filedialog.askdirectory(title="Select Nuclei Results Folder")
        if folder:
            self.nuclei_folder_var.set(folder)

    def browse_output_folder(self):
        """Browse for output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)

    def update_params_from_gui(self):
        """Update parameters from GUI widgets."""
        self.params['myotube_folder'] = self.myotube_folder_var.get()
        self.params['nuclei_folder'] = self.nuclei_folder_var.get()
        self.params['output_folder'] = self.output_folder_var.get()
        self.params['min_area'] = int(self.min_area_var.get())
        self.params['max_area'] = int(self.max_area_var.get())
        self.params['max_eccentricity'] = float(self.max_eccentricity_var.get())
        self.params['overlap_threshold'] = float(self.overlap_threshold_var.get())
        self.params['periphery_overlap_threshold'] = float(self.periphery_overlap_threshold_var.get())
        self.params['full_image_mode'] = self.full_image_mode_var.get()

    def update_gui_from_params(self):
        """Update GUI widgets from current parameters."""
        self.myotube_folder_var.set(self.params['myotube_folder'])
        self.nuclei_folder_var.set(self.params['nuclei_folder'])
        self.output_folder_var.set(self.params['output_folder'])
        self.min_area_var.set(str(self.params['min_area']))
        self.max_area_var.set(str(self.params['max_area']))
        self.max_eccentricity_var.set(str(self.params['max_eccentricity']))
        self.overlap_threshold_var.set(str(self.params['overlap_threshold']))
        self.periphery_overlap_threshold_var.set(str(self.params['periphery_overlap_threshold']))
        self.full_image_mode_var.set(self.params['full_image_mode'])

    def restore_defaults(self):
        """Restore all parameters to default values."""
        self.params = self.default_params.copy()
        self.update_gui_from_params()
        self.log("[OK] Restored parameters to defaults")

    def on_run_threaded(self):
        """Run analysis in a separate thread."""
        if self.is_running:
            return

        # Update params from GUI
        try:
            self.update_params_from_gui()
        except ValueError as e:
            self.log(f"Error: Invalid parameter value: {e}")
            return

        # Validate inputs
        valid, error_msg = self.validate_parameters()
        if not valid:
            self.log(f"Error: {error_msg}")
            return

        # Save configuration
        self.save_config()

        # Update UI
        self.is_running = True
        self.stop_requested = False
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')

        # Start analysis thread
        thread = threading.Thread(target=self.run_analysis, args=(
            self.params['myotube_folder'],
            self.params['nuclei_folder'],
            self.params['output_folder'],
            self.params['min_area'],
            self.params['max_area'],
            self.params['max_eccentricity'],
            self.params['overlap_threshold'],
            self.params['periphery_overlap_threshold']
        ), daemon=True)
        thread.start()

    def run_analysis(self, myotube_folder, nuclei_folder, output_folder, min_area, max_area, max_eccentricity, overlap_threshold, periphery_overlap_threshold):
        """Run the analysis."""
        try:
            self.log("=" * 80)
            self.log("NUCLEI-MYOTUBE RELATIONSHIP ANALYSIS")
            self.log("=" * 80)
            self.log(f"Myotube folder: {myotube_folder}")
            self.log(f"Nuclei folder: {nuclei_folder}")
            self.log(f"Output folder: {output_folder}")
            self.log(f"Filter settings:")
            self.log(f"  Size range: {min_area} - {max_area} pixels")
            self.log(f"  Max eccentricity: {max_eccentricity}")
            self.log(f"  Overlap threshold: {overlap_threshold * 100:.1f}%")
            self.log(f"  Periphery overlap threshold: {periphery_overlap_threshold * 100:.1f}%")
            self.log(f"  Full image mode: {self.full_image_mode_var.get()}")
            self.log("=" * 80)

            # Create analyzer (always skip alignment resize)
            analyzer = NucleiMyotubeAnalyzer(
                myotube_dir=myotube_folder,
                nuclei_dir=nuclei_folder,
                output_dir=output_folder,
                overlap_threshold=overlap_threshold,
                periphery_overlap_threshold=periphery_overlap_threshold,
                min_nucleus_area=min_area,
                max_nucleus_area=max_area,
                max_eccentricity=max_eccentricity,
                full_image_mode=self.full_image_mode_var.get(),
                skip_alignment_resize=True,
                progress_callback=self.log
            )

            # Run analysis
            analyzer.analyze_all_samples()

            self.log("\n" + "=" * 80)
            self.log("Analysis complete!")
            self.log(f"Results saved in: {output_folder}")
            self.log("For each sample, the following files are created:")
            self.log("  - {sample_name}_myotube_nuclei_counts.csv")
            self.log("  - {sample_name}_nuclei_myotube_assignments.csv")
            self.log("  - {sample_name}_analysis_summary.txt")
            self.log("  - {sample_name}_nuclei_cropped.png")
            self.log("  - {sample_name}_nuclei_overlay.tif")
            self.log("\nVisualization Color Coding:")
            self.log("  GREEN:  Passed all filters and assigned to myotube")
            self.log("  RED:    Filtered by size (too small/large)")
            self.log("  YELLOW: Filtered by eccentricity (too elongated)")
            self.log("  BLUE:   Filtered by overlap (insufficient overlap)")
            self.log("=" * 80)

        except Exception as e:
            self.log(f"Error during analysis: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            # Reset UI state
            self.is_running = False
            self.run_button.config(state='normal')
            self.stop_button.config(state='disabled')

    def on_stop(self):
        """Stop the analysis."""
        if self.is_running:
            self.stop_requested = True
            self.log("\nStop requested...")

    def log(self, message: str):
        """Log a message to the console."""
        self.write_to_console(message + "\n")

    def restore_defaults(self):
        """Restore default parameter values."""
        self.min_area_var.set("400")
        self.max_area_var.set("2000")
        self.max_eccentricity_var.set("0.9")
        self.overlap_threshold_var.set("0.6")
        self.full_image_mode_var.set(True)
        self.log("Parameters restored to defaults")
