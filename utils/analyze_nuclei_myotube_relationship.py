#!/usr/bin/env python3
"""
Nuclei-Myotube Relationship Analysis

This script analyzes the spatial relationship between segmented myotube masks and nuclei binary images.
It calculates overlap ratios, assigns nuclei to myotubes, and generates comprehensive CSV reports.

Usage:
    python analyze_nuclei_myotube_relationship.py --nuclei_dir /path/to/nuclei/images [options]
    python analyze_nuclei_myotube_relationship.py --myotube_dir /custom/myotube/path --nuclei_dir /path/to/nuclei/images [options]

Features:
    - Crops nuclei images to match myotube segmentation regions
    - Calculates nuclei-myotube overlap with configurable threshold
    - Assigns myotube pixels to closest nuclei
    - Computes nuclei shape metrics: circularity, eccentricity, solidity
    - Generates two CSV reports: myotube-centric and nuclei-centric
    - Saves cropped nuclei binary images in myotube result folders
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage
from skimage import measure
import pandas as pd
from tqdm import tqdm


class NucleiMyotubeAnalyzer:
    """Analyzes spatial relationships between nuclei and myotubes."""

    def __init__(self, myotube_dir: str, nuclei_dir: str, overlap_threshold: float = 0.60):
        """
        Initialize the analyzer.

        Args:
            myotube_dir: Directory containing myotube segmentation results
            nuclei_dir: Directory containing nuclei binary images
            overlap_threshold: Minimum overlap ratio for nuclei-myotube assignment (default: 0.60)
        """
        self.myotube_dir = Path(myotube_dir)
        self.nuclei_dir = Path(nuclei_dir)
        self.overlap_threshold = overlap_threshold

        # Results storage
        self.myotube_results = []  # For myotube-centric CSV
        self.nuclei_results = []   # For nuclei-centric CSV

    def find_nuclei_image(self, myotube_folder_name: str) -> Optional[Path]:
        """
        Find the corresponding nuclei binary image for a myotube folder.

        Args:
            myotube_folder_name: Name of the myotube segmentation folder

        Returns:
            Path to nuclei image or None if not found
        """
        # Remove the position suffix (_bl, _br, _tl, _tr) to get base name
        base_name = myotube_folder_name
        for suffix in ['_bl', '_br', '_tl', '_tr']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break

        # Try different nuclei image naming patterns
        nuclei_patterns = [
            f"{base_name}_nuclei.png",
            f"{base_name}_nuclei.tif",
            f"{base_name}_dapi.png",
            f"{base_name}_dapi.tif",
            f"{base_name}_blue.png",
            f"{base_name}_blue.tif",
            f"{base_name}_405.png",
            f"{base_name}_405.tif",
            f"{base_name}_binary.png",
            f"{base_name}_binary.tif"
        ]

        for pattern in nuclei_patterns:
            nuclei_path = self.nuclei_dir / pattern
            if nuclei_path.exists():
                return nuclei_path

        return None

    def get_crop_coordinates(self, folder_name: str, full_image_shape: Tuple[int, int],
                           crop_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate crop coordinates based on position suffix.

        Args:
            folder_name: Myotube folder name with position suffix
            full_image_shape: (height, width) of full nuclei image
            crop_shape: (height, width) of cropped myotube region

        Returns:
            (y1, y2, x1, x2) crop coordinates
        """
        h_full, w_full = full_image_shape
        h_crop, w_crop = crop_shape

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
        Load individual myotube masks from the masks directory with optimizations.

        Args:
            myotube_folder: Path to myotube segmentation result folder

        Returns:
            Dictionary mapping myotube_id to binary mask
        """
        masks_dir = myotube_folder / f"{myotube_folder.name}_masks"
        myotube_masks = {}

        if not masks_dir.exists():
            print(f"Warning: Masks directory not found: {masks_dir}", flush=True)
            return myotube_masks

        print(f"Loading myotube masks...", flush=True)
        mask_files = list(masks_dir.glob("Myotube_*_mask.png"))
        print(f"Found {len(mask_files)} mask files", flush=True)

        for mask_file in mask_files:
            # Extract myotube ID from filename
            myotube_id = int(mask_file.stem.split('_')[1])

            # Load mask as binary - use IMREAD_UNCHANGED for faster loading
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            if mask is not None:
                myotube_masks[myotube_id] = (mask > 0).astype(np.uint8)

        print(f"Loaded {len(myotube_masks)} myotube masks", flush=True)
        return myotube_masks

    def find_nuclei_components(self, nuclei_binary: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Find connected components (individual nuclei) in binary image.
        Returns labeled image and basic nucleus info (without individual masks for efficiency).

        Args:
            nuclei_binary: Binary nuclei image

        Returns:
            Tuple of (labeled_nuclei_array, nuclei_list)
        """
        print(f"Labeling nuclei components...", flush=True)
        # Label connected components
        labeled_nuclei = measure.label(nuclei_binary, connectivity=2)
        print(f"Computing region properties...", flush=True)
        props = measure.regionprops(labeled_nuclei)

        nuclei_list = []
        for i, prop in enumerate(props):
            # Calculate circularity (4π × Area / Perimeter²)
            # Range: 0 to 1, where 1 = perfect circle
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
                # Note: Don't store individual masks here - too memory intensive
                # We'll extract them on-demand from labeled_nuclei
            })

        print(f"Found {len(nuclei_list)} nuclei", flush=True)
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
        This is much faster than computing distances to all pixels individually.

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
        # Use nearest neighbor propagation via distance transform
        distance, nearest_label = ndimage.distance_transform_edt(
            labeled_nuclei == 0,
            return_distances=True,
            return_indices=True
        )

        # Create a map where each pixel has the label of its nearest nucleus
        nearest_nucleus_map = labeled_nuclei.copy()
        mask = labeled_nuclei == 0  # Where there are no nuclei
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
        print(f"\nAnalyzing: {folder_name}", flush=True)

        # Load myotube info
        print(f"Loading info file...", flush=True)
        info_file = myotube_folder / f"{folder_name}_info.json"
        if not info_file.exists():
            print(f"Warning: Info file not found: {info_file}", flush=True)
            return False

        with open(info_file, 'r') as f:
            info = json.load(f)

        image_shape = tuple(info['image_shape'])  # [height, width]
        num_myotubes = info['num_instances']
        print(f"Image shape: {image_shape}, myotubes: {num_myotubes}", flush=True)

        # Find corresponding nuclei image
        print(f"Finding nuclei image...", flush=True)
        nuclei_image_path = self.find_nuclei_image(folder_name)
        if nuclei_image_path is None:
            # Get base name without position suffix for display
            base_name = folder_name
            for suffix in ['_bl', '_br', '_tl', '_tr']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            print(f"⚠️  SKIPPED: No nuclei image found for {folder_name}")
            print(f"    Searched for patterns like: {base_name}_nuclei.png, {base_name}_dapi.png, etc.")
            return False

        # Load nuclei image
        print(f"Loading nuclei image: {nuclei_image_path.name}")
        nuclei_full = cv2.imread(str(nuclei_image_path), cv2.IMREAD_GRAYSCALE)
        if nuclei_full is None:
            print(f"Warning: Could not load nuclei image: {nuclei_image_path}")
            return False
        print(f"Nuclei image loaded: {nuclei_full.shape}")

        # Resize nuclei to match the processed image dimensions used for myotube segmentation
        # The myotube quadrants are from a resized processed image (e.g., 9450×9438 → 9000×8988)
        h_full, w_full = nuclei_full.shape
        expected_processed_h = image_shape[0] * 2  # Quadrant height * 2
        expected_processed_w = image_shape[1] * 2  # Quadrant width * 2

        if h_full != expected_processed_h or w_full != expected_processed_w:
            # Need to resize to match processed image dimensions
            print(f"Resizing nuclei from {nuclei_full.shape} to ({expected_processed_h}, {expected_processed_w})")
            nuclei_full = cv2.resize(nuclei_full, (expected_processed_w, expected_processed_h), interpolation=cv2.INTER_NEAREST)
            print(f"Resized nuclei to: {nuclei_full.shape}")

        # Crop nuclei image to match myotube region
        try:
            y1, y2, x1, x2 = self.get_crop_coordinates(folder_name, nuclei_full.shape, image_shape)
            nuclei_cropped = nuclei_full[y1:y2, x1:x2]
            print(f"Cropped nuclei to: {nuclei_cropped.shape}")

            # Resize to match myotube image dimensions if needed
            if nuclei_cropped.shape != tuple(image_shape):
                print(f"Resizing nuclei from {nuclei_cropped.shape} to {image_shape}")
                nuclei_cropped = cv2.resize(nuclei_cropped, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
                print(f"Resized nuclei to: {nuclei_cropped.shape}")
        except ValueError as e:
            print(f"Warning: {e}")
            return False

        # Ensure nuclei image is binary
        if nuclei_cropped.max() > 1:
            nuclei_cropped = (nuclei_cropped > 127).astype(np.uint8)
        print(f"Binary conversion done")

        # Save cropped nuclei image
        nuclei_output_path = myotube_folder / f"{folder_name}_nuclei_cropped.png"
        cv2.imwrite(str(nuclei_output_path), nuclei_cropped * 255)

        # Load myotube masks
        myotube_masks = self.load_myotube_masks(myotube_folder)
        if not myotube_masks:
            print(f"Warning: No myotube masks found for {folder_name}")
            return False

        # Find nuclei components
        labeled_nuclei, nuclei_list = self.find_nuclei_components(nuclei_cropped)
        print(f"Found {len(nuclei_list)} nuclei and {len(myotube_masks)} myotubes")

        # Assign myotube pixels to nuclei (distance-based)
        print(f"Computing myotube-nucleus assignments...", flush=True)
        myotube_assignments = self.assign_myotube_pixels_to_nuclei(myotube_masks, labeled_nuclei, nuclei_list)
        print(f"Assignments computed.", flush=True)

        # Create labeled myotube image (encode myotube_id in pixels)
        print(f"Creating labeled myotube image...", flush=True)
        labeled_myotubes = np.zeros(image_shape, dtype=np.int32)
        for myotube_id, myotube_mask in myotube_masks.items():
            labeled_myotubes[myotube_mask > 0] = myotube_id

        # Pre-compute myotube areas for quick access
        myotube_areas = {myotube_id: np.sum(mask) for myotube_id, mask in myotube_masks.items()}
        print(f"Labeled myotube image created", flush=True)

        # Analyze each nucleus - MUCH faster with labeled image
        print(f"Analyzing nucleus-myotube overlaps...", flush=True)
        for nucleus in nuclei_list:
            nucleus_id = nucleus['nucleus_id']
            nucleus_label = nucleus['label']
            nucleus_area = nucleus['area']

            # Extract individual nucleus mask on-demand from labeled image
            nucleus_pixels = (labeled_nuclei == nucleus_label)

            # Get myotube labels that overlap with this nucleus
            overlapping_myotube_labels = labeled_myotubes[nucleus_pixels]
            # Remove zeros (background)
            overlapping_myotube_labels = overlapping_myotube_labels[overlapping_myotube_labels > 0]

            if len(overlapping_myotube_labels) == 0:
                # No overlap with any myotube
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

                # Get myotube pixels assigned to this nucleus (from distance-based assignment)
                myotube_pixels_for_nucleus = 0
                if assigned_myotube and assigned_myotube in myotube_assignments:
                    # Find how many pixels of the assigned myotube are closest to this nucleus
                    assignment_info = myotube_assignments[assigned_myotube]
                    if nucleus_id in assignment_info['nucleus_pixel_counts']:
                        myotube_pixels_for_nucleus = assignment_info['nucleus_pixel_counts'][nucleus_id]

            # Store nuclei-centric result for this sample
            self.nuclei_results.append({
                'nucleus_id': nucleus_id,
                'nucleus_area': nucleus_area,
                'circularity': nucleus['circularity'],
                'eccentricity': nucleus['eccentricity'],
                'solidity': nucleus['solidity'],
                'assigned_myotube_id': assigned_myotube,
                'overlap_pixels': best_overlap_pixels,
                'overlap_percentage': best_overlap * 100,
                'myotube_pixels_assigned_to_nucleus': myotube_pixels_for_nucleus
            })

        # Count nuclei per myotube for this sample
        sample_myotube_results = []
        for myotube_id in myotube_masks.keys():
            count = sum(1 for result in self.nuclei_results
                       if result['assigned_myotube_id'] == myotube_id)

            sample_myotube_results.append({
                'myotube_id': myotube_id,
                'myotube_area': myotube_areas[myotube_id],
                'nuclei_count': count
            })

        # Create nuclei overlay visualization
        self.create_nuclei_overlay(myotube_folder, folder_name, labeled_nuclei, nuclei_list)

        # Save CSVs for this sample in its folder
        self.save_sample_results(myotube_folder, folder_name, sample_myotube_results)

        # Clear results for next sample
        self.nuclei_results = []

        return True

    def create_nuclei_overlay(self, sample_folder: Path, sample_name: str,
                             labeled_nuclei: np.ndarray, nuclei_list: List[Dict]):
        """
        Create visualization overlay showing assigned (green) and unassigned (red) nuclei.

        Args:
            sample_folder: Path to the sample's myotube segmentation folder
            sample_name: Name of the sample
            labeled_nuclei: Labeled nuclei image
            nuclei_list: List of nucleus dictionaries
        """
        # Load the processed overlay image
        overlay_path = sample_folder / f"{sample_name}_processed_overlay.tif"
        if not overlay_path.exists():
            print(f"  Warning: Processed overlay not found: {overlay_path.name}", flush=True)
            return

        # Read overlay image
        overlay = cv2.imread(str(overlay_path))
        if overlay is None:
            print(f"  Warning: Could not load overlay image", flush=True)
            return

        # Create a mapping of nucleus_id to assignment status
        assigned_nucleus_ids = set()
        for result in self.nuclei_results:
            if result['assigned_myotube_id'] is not None:
                assigned_nucleus_ids.add(result['nucleus_id'])

        # Draw nuclei contours and labels on overlay
        for nucleus in nuclei_list:
            nucleus_id = nucleus['nucleus_id']
            nucleus_label = nucleus['label']
            centroid = nucleus['centroid']

            # Extract nucleus mask
            nucleus_mask = (labeled_nuclei == nucleus_label).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Choose color based on assignment
            if nucleus_id in assigned_nucleus_ids:
                color = (0, 255, 0)  # Green for assigned
            else:
                color = (0, 0, 255)  # Red for unassigned

            # Draw contour with thick line for visibility
            cv2.drawContours(overlay, contours, -1, color, 2)

            # Draw nucleus ID label at centroid (white text only, no background)
            label_text = f"{nucleus_id}"
            centroid_pos = (int(centroid[1]), int(centroid[0]))  # (x, y) from (row, col)

            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2  # Thicker for better visibility
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            # Draw white text centered at centroid
            text_pos = (centroid_pos[0] - text_w // 2, centroid_pos[1] + text_h // 2)
            cv2.putText(overlay, label_text, text_pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Save overlay with nuclei
        output_path = sample_folder / f"{sample_name}_nuclei_overlay.tif"
        cv2.imwrite(str(output_path), overlay)
        print(f"  Saved: {output_path.name}", flush=True)

    def save_sample_results(self, sample_folder: Path, sample_name: str, myotube_results: List[Dict]):
        """
        Save CSV results and summary for a single sample in its folder.

        Args:
            sample_folder: Path to the sample's myotube segmentation folder
            sample_name: Name of the sample
            myotube_results: List of myotube result dictionaries
        """
        # Save myotube-centric CSV
        myotube_csv_path = sample_folder / f"{sample_name}_myotube_nuclei_counts.csv"
        myotube_df = pd.DataFrame(myotube_results)
        myotube_df.to_csv(myotube_csv_path, index=False)
        print(f"  Saved: {myotube_csv_path.name}", flush=True)

        # Save nuclei-centric CSV
        nuclei_csv_path = sample_folder / f"{sample_name}_nuclei_myotube_assignments.csv"
        nuclei_df = pd.DataFrame(self.nuclei_results)
        nuclei_df.to_csv(nuclei_csv_path, index=False)
        print(f"  Saved: {nuclei_csv_path.name}", flush=True)

        # Generate and save summary file
        summary_path = sample_folder / f"{sample_name}_analysis_summary.txt"

        # Calculate statistics
        total_myotubes = len(myotube_df)
        total_nuclei = len(nuclei_df)
        assigned_nuclei_df = nuclei_df[nuclei_df['assigned_myotube_id'].notna()]
        num_assigned_nuclei = len(assigned_nuclei_df)
        num_unassigned_nuclei = total_nuclei - num_assigned_nuclei

        myotubes_with_nuclei = myotube_df[myotube_df['nuclei_count'] > 0]
        num_myotubes_with_nuclei = len(myotubes_with_nuclei)
        num_myotubes_without_nuclei = total_myotubes - num_myotubes_with_nuclei

        avg_nuclei_per_myotube = myotube_df['nuclei_count'].mean()
        total_myotube_area = myotube_df['myotube_area'].sum()
        avg_myotube_area = myotube_df['myotube_area'].mean()

        avg_nucleus_area = nuclei_df['nucleus_area'].mean()
        avg_circularity = nuclei_df['circularity'].mean()
        avg_eccentricity = nuclei_df['eccentricity'].mean()
        avg_solidity = nuclei_df['solidity'].mean()

        if num_assigned_nuclei > 0:
            avg_overlap_pct = assigned_nuclei_df['overlap_percentage'].mean()
            min_overlap_pct = assigned_nuclei_df['overlap_percentage'].min()
            max_overlap_pct = assigned_nuclei_df['overlap_percentage'].max()
        else:
            avg_overlap_pct = 0.0
            min_overlap_pct = 0.0
            max_overlap_pct = 0.0

        # Write summary
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"NUCLEI-MYOTUBE ANALYSIS SUMMARY\n")
            f.write(f"Sample: {sample_name}\n")
            f.write("=" * 80 + "\n\n")

            f.write("OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total myotubes analyzed:        {total_myotubes}\n")
            f.write(f"Total nuclei detected:           {total_nuclei}\n")
            f.write(f"Overlap threshold:               {self.overlap_threshold * 100:.1f}%\n")
            f.write("\n")

            f.write("NUCLEI ASSIGNMENT\n")
            f.write("-" * 80 + "\n")
            f.write(f"Nuclei assigned to myotubes:     {num_assigned_nuclei} ({num_assigned_nuclei/total_nuclei*100:.1f}%)\n")
            f.write(f"Nuclei not assigned:             {num_unassigned_nuclei} ({num_unassigned_nuclei/total_nuclei*100:.1f}%)\n")
            f.write("\n")

            f.write("MYOTUBE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Myotubes with nuclei:            {num_myotubes_with_nuclei} ({num_myotubes_with_nuclei/total_myotubes*100:.1f}%)\n")
            f.write(f"Myotubes without nuclei:         {num_myotubes_without_nuclei} ({num_myotubes_without_nuclei/total_myotubes*100:.1f}%)\n")
            f.write(f"Average nuclei per myotube:      {avg_nuclei_per_myotube:.2f}\n")
            f.write(f"Total myotube area (pixels):     {total_myotube_area:,.0f}\n")
            f.write(f"Average myotube area (pixels):   {avg_myotube_area:,.0f}\n")
            f.write("\n")

            f.write("NUCLEI STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average nucleus area (pixels):   {avg_nucleus_area:,.0f}\n")
            f.write(f"Average circularity:             {avg_circularity:.3f} (0=irregular, 1=perfect circle)\n")
            f.write(f"Average eccentricity:            {avg_eccentricity:.3f} (0=circle, 1=line)\n")
            f.write(f"Average solidity:                {avg_solidity:.3f} (1=convex, <1=concave/irregular)\n")
            if num_assigned_nuclei > 0:
                f.write(f"Average overlap percentage:      {avg_overlap_pct:.2f}%\n")
                f.write(f"Min overlap percentage:          {min_overlap_pct:.2f}%\n")
                f.write(f"Max overlap percentage:          {max_overlap_pct:.2f}%\n")
            f.write("\n")

            f.write("NUCLEI DISTRIBUTION PER MYOTUBE\n")
            f.write("-" * 80 + "\n")
            nuclei_count_dist = myotube_df['nuclei_count'].value_counts().sort_index()
            for count, freq in nuclei_count_dist.items():
                f.write(f"  {int(count)} nuclei: {freq} myotubes\n")
            f.write("\n")

            f.write("=" * 80 + "\n")

        print(f"  Saved: {summary_path.name}", flush=True)

    def analyze_all_samples(self):
        """Analyze all myotube segmentation samples."""
        myotube_folders = [f for f in self.myotube_dir.iterdir()
                          if f.is_dir() and not f.name.startswith('.')]

        print(f"Found {len(myotube_folders)} myotube samples to analyze")
        print(f"Looking for nuclei images in: {self.nuclei_dir}")
        print(f"Overlap threshold: {self.overlap_threshold}")
        print("-" * 80)

        successful_analyses = 0
        skipped_no_nuclei = 0
        failed_analyses = 0

        for i, folder in enumerate(myotube_folders):
            print(f"[{i+1}/{len(myotube_folders)}]", end=" ")
            result = self.analyze_sample(folder)
            if result:
                successful_analyses += 1
            else:
                # Check if it was skipped due to missing nuclei or other failure
                nuclei_path = self.find_nuclei_image(folder.name)
                if nuclei_path is None:
                    skipped_no_nuclei += 1
                else:
                    failed_analyses += 1

        print("-" * 80)
        print(f"📊 ANALYSIS SUMMARY:")
        print(f"   ✅ Successfully analyzed: {successful_analyses}")
        print(f"   ⚠️  Skipped (no nuclei): {skipped_no_nuclei}")
        print(f"   ❌ Failed (other error): {failed_analyses}")
        print(f"   📁 Total samples: {len(myotube_folders)}")

        if skipped_no_nuclei > 0:
            print(f"\n💡 TIP: To reduce skipped samples, ensure nuclei images follow naming patterns:")
            print(f"    - {{base_name}}_nuclei.png/tif")
            print(f"    - {{base_name}}_dapi.png/tif")
            print(f"    - {{base_name}}_blue.png/tif")
            print(f"    - {{base_name}}_405.png/tif")
            print(f"    - {{base_name}}_binary.png/tif")

    def save_results(self, output_dir: str):
        """
        Print final summary. Results are already saved per-sample in their folders.

        Args:
            output_dir: Not used (kept for compatibility)
        """
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("Results saved in each sample's folder:")
        print("  - {sample_name}_myotube_nuclei_counts.csv")
        print("  - {sample_name}_nuclei_myotube_assignments.csv")
        print("  - {sample_name}_analysis_summary.txt")
        print("  - {sample_name}_nuclei_cropped.png")
        print("  - {sample_name}_nuclei_overlay.tif (GREEN=assigned, RED=unassigned)")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze nuclei-myotube spatial relationships")
    parser.add_argument("--myotube_dir", default="/home/bwang/tmp/myotube_segmentation/output",
                       help="Directory containing myotube segmentation results (default: /home/bwang/tmp/myotube_segmentation/output)")
    parser.add_argument("--nuclei_dir", default="/home/bwang/tmp/myotube_segmentation/nuclei",
                       help="Directory containing nuclei binary images (default: /home/bwang/tmp/myotube_segmentation/nuclei)")
    parser.add_argument("--output_dir", default="./nuclei_myotube_analysis_results",
                       help="Output directory for CSV files")
    parser.add_argument("--overlap_threshold", type=float, default=0.60,
                       help="Minimum overlap ratio for nuclei-myotube assignment (default: 0.60)")

    args = parser.parse_args()

    # Validate input directories
    if not Path(args.myotube_dir).exists():
        print(f"Error: Myotube directory does not exist: {args.myotube_dir}")
        sys.exit(1)

    if not Path(args.nuclei_dir).exists():
        print(f"Error: Nuclei directory does not exist: {args.nuclei_dir}")
        sys.exit(1)

    # Initialize analyzer
    analyzer = NucleiMyotubeAnalyzer(
        myotube_dir=args.myotube_dir,
        nuclei_dir=args.nuclei_dir,
        overlap_threshold=args.overlap_threshold
    )

    # Run analysis
    print("Starting nuclei-myotube relationship analysis...", flush=True)
    analyzer.analyze_all_samples()

    # Save results
    analyzer.save_results(args.output_dir)
    print("Analysis complete!")


if __name__ == "__main__":
    main()