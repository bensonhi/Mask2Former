#!/usr/bin/env python3
"""
Myotube Instance Segmentation Script

This script performs automatic segmentation of overlapping myotubes in fluorescence 
microscopy images using traditional image processing methods.

Date: 2025-01-07
"""

import os
import warnings
import json
import datetime
from typing import List, Tuple, Dict, Union, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, exposure
from skimage.filters import frangi
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage.segmentation import watershed
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class MyotubeSegmenter:
    """
    A comprehensive class for segmenting individual myotubes from fluorescence 
    microscopy images using advanced image processing techniques.
    
    The segmentation pipeline includes:
    1. Image loading and resizing
    2. Preprocessing (denoising, contrast enhancement)
    3. Tubular structure enhancement (Frangi filter)
    4. Binarization with morphological opening
    5. Skeletonization and segment breaking
    6. Intelligent merging based on geometric criteria
    """
    
    def __init__(self, image_path: str):
        """
        Initialize the segmenter with an image path.
        
        Args:
            image_path: Path to the input fluorescence microscopy image
        """
        self.image_path = image_path
        self.original_image: Optional[np.ndarray] = None
        self.preprocessed_image: Optional[np.ndarray] = None
        self.enhanced_image: Optional[np.ndarray] = None
        self.binary_image: Optional[np.ndarray] = None
        self.skeleton: Optional[np.ndarray] = None
        self.instance_masks: List[np.ndarray] = []
        self.total_myotubes: int = 0
        self.original_height: Optional[int] = None
        self.original_width: Optional[int] = None
        
    def load_image(self, target_size: int = 2000) -> np.ndarray:
        """
        Load and resize the fluorescence microscopy image with high-quality interpolation.
        
        Args:
            target_size: Target size for the longer dimension (default: 2000)
        
        Returns:
            Loaded and resized image as numpy array
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        print(f"Loading image: {self.image_path}")
        
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
            
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        
        # Get original dimensions and print info
        original_height, original_width = image.shape
        self.original_height = original_height
        self.original_width = original_width
        print(f"Original image shape: {image.shape}")
        
        # Calculate new dimensions while maintaining aspect ratio
        if original_height > original_width:
            new_height = target_size
            new_width = int(original_width * (target_size / original_height))
        else:
            new_width = target_size
            new_height = int(original_height * (target_size / original_width))
        
        # Resize using high-quality interpolation if needed
        if original_height > target_size or original_width > target_size:
            print(f"Resizing image from {original_height}x{original_width} to {new_height}x{new_width}")
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            print("Image is already smaller than target size, keeping original dimensions")
            
        self.original_image = image.astype(np.float32) / 255.0
        print(f"Final image shape: {self.original_image.shape}")
        return self.original_image
    
    def preprocess_image(self) -> np.ndarray:
        """
        Apply comprehensive preprocessing including denoising, histogram matching,
        contrast enhancement, and morphological operations.
        
        Returns:
            Preprocessed image
        """
        print("Applying preprocessing (denoising, histogram matching, contrast enhancement, morphology)...")
        
        image = self.original_image.copy()
        
        # Gaussian denoising
        denoised = filters.gaussian(image, sigma=1.0)
        
        # Histogram matching to reference image
        reference_image_path = "./C4-MAX_20250306_T34_48hp.lif - Region 4_Merged.lif - Region 4_Merged.png"
        try:
            if os.path.exists(reference_image_path):
                print(f"Applying histogram matching to reference: {reference_image_path}")
                reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
                if reference_image is not None:
                    # Convert reference to same format as current image
                    reference_image = reference_image.astype(np.float32) / 255.0
                    # Apply histogram matching
                    denoised = exposure.match_histograms(denoised, reference_image)
                    print("Histogram matching completed")
                else:
                    print(f"Warning: Could not load reference image {reference_image_path}")
            else:
                print(f"Warning: Reference image not found: {reference_image_path}")
        except Exception as e:
            print(f"Warning: Histogram matching failed: {str(e)}")
        
        # Contrast enhancement using CLAHE
        image_uint8 = (denoised * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image_uint8)
        
        # Convert back to float and rescale intensity
        enhanced = enhanced.astype(np.float32) / 255.0
        enhanced = exposure.rescale_intensity(enhanced, in_range='image', out_range=(0, 1))
        
        # Apply morphological operations (erosion followed by dilation)
        enhanced = morphology.erosion(enhanced, disk(2))
        enhanced = morphology.dilation(enhanced, disk(1))
        
        self.preprocessed_image = enhanced
        print("Preprocessing completed")
        return self.preprocessed_image

    def enhance_tubular_structures(self) -> np.ndarray:
        """
        Use Frangi filter to enhance elongated tubular structures while suppressing noise.
        
        Returns:
            Enhanced image with tubular structures highlighted
        """
        print("Enhancing tubular structures using Frangi filter...")
        
        if self.preprocessed_image is None:
            raise ValueError("Preprocessed image not available. Run preprocess_image() first.")
        
        # Frangi filter optimized for myotube detection
        enhanced = frangi(
            self.preprocessed_image, 
                        sigmas=range(2, 12, 2),  # Detect myotubes of width 4-24 pixels
                        alpha=0.5,               # Tubular structure sensitivity
                        beta=0.5,                # Noise suppression
                        gamma=None,              # Auto-calculated
            black_ridges=False       # Bright structures
        )
        
        # Normalize the response
        self.enhanced_image = exposure.rescale_intensity(enhanced)
        print("Tubular structure enhancement completed")
        return self.enhanced_image
    
    def binarize_and_skeletonize(self, threshold_method: Union[str, float] = 0.02,
                                opening_kernel_size: int = 3) -> np.ndarray:
        """
        Binarize the enhanced image with morphological opening and compute skeleton.
        
        Args:
            threshold_method: Thresholding method ('otsu', 'adaptive', or numeric value)
            opening_kernel_size: Size of morphological opening kernel (default: 3)

        Returns:
            Skeletonized binary image
        """
        print("Binarizing and skeletonizing image...")

        if self.enhanced_image is None:
            raise ValueError("Enhanced image not available. Run enhance_tubular_structures() first.")

        # Binarization with multiple method support
        if threshold_method == 'otsu':
            threshold = filters.threshold_otsu(self.enhanced_image)
            binary = self.enhanced_image > threshold
        elif threshold_method == 'adaptive':
            threshold = filters.threshold_local(self.enhanced_image, block_size=35, offset=0)
            binary = self.enhanced_image > threshold
        elif isinstance(threshold_method, (int, float)):
            binary = self.enhanced_image > threshold_method
        else:
            threshold = filters.threshold_otsu(self.enhanced_image)
            binary = self.enhanced_image > threshold

        # Sequential morphological operations for cleaning
        binary = morphology.remove_small_objects(binary, min_size=500)
        binary = morphology.binary_opening(binary, disk(opening_kernel_size))
        binary = morphology.binary_closing(binary, disk(2))
        
        self.binary_image = binary

        # Skeletonization
        self.skeleton = skeletonize(binary)
        print("Binarization and skeletonization completed")
        return self.skeleton

    def detect_branch_points(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Detect branch points in the skeleton using convolution.

        Args:
            skeleton: Binary skeleton image

        Returns:
            Binary image with branch points marked
        """
        kernel = np.array([
            [1, 1, 1],
                          [1, 10, 1],
            [1, 1, 1]
        ], dtype=np.uint8)

        convolved = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        return (convolved >= 13) & skeleton

    def detect_turning_points(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Detect straight segments in the skeleton using morphological operations.
        
        Args:
            skeleton: Binary skeleton image
            
        Returns:
            Binary image with straight segments marked
        """
        if np.sum(skeleton) == 0:
            return np.zeros_like(skeleton, dtype=bool)
        
        # Find boundary points around skeleton
        dilated = morphology.binary_dilation(skeleton, disk(1))
        boundary = dilated & ~skeleton
        
        # Count boundary neighbors - straight segments have more neighbors
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = cv2.filter2D(boundary.astype(np.uint8), -1, kernel)
        
        # Return straight segments (removes actual turning points)
        return skeleton & (neighbor_count >= 4)

    def split_skeleton_segments(self) -> List[np.ndarray]:
        """
        Split the skeleton into disconnected segments by removing branch points.

        Returns:
            List of individual skeleton segments
        """
        print("Splitting skeleton into segments...")

        if self.skeleton is None:
            raise ValueError("Skeleton not available. Run binarize_and_skeletonize() first.")

        skeleton = self.skeleton.copy()
        
        # Remove branch points to disconnect overlapping structures
        branch_points = self.detect_branch_points(skeleton)
        skeleton_no_branches = skeleton & ~branch_points

        # Find connected components
        labeled_skeleton = measure.label(skeleton_no_branches)

        # Extract segments with minimum length threshold
        segments = []
        for label_id in range(1, labeled_skeleton.max() + 1):
            segment = (labeled_skeleton == label_id)
            if np.sum(segment) > 20:  # Minimum skeleton length
                segments.append(segment)

        print(f"Found {len(segments)} skeleton segments")
        return segments

    def break_skeleton_segments(self, skeleton_segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Break skeleton segments at turning points to separate individual myotubes.
        
        Args:
            skeleton_segments: List of connected skeleton segments
            
        Returns:
            List of individual myotube skeletons after breaking
        """
        print("Breaking skeleton segments at turning points...")
        
        individual_myotubes = []
        
        for segment in skeleton_segments:
            # Keep only straight segments (removes turning points)
            straight_segments = self.detect_turning_points(segment)
            broken_segment = segment & straight_segments
            
            # Find connected components after breaking
            labeled_broken = measure.label(broken_segment)
            
            # Extract individual components with quality checks
            for label_id in range(1, labeled_broken.max() + 1):
                individual_segment = (labeled_broken == label_id)
                if np.sum(individual_segment) > 15:  # Minimum length threshold
                    individual_myotubes.append(individual_segment)
        
        print(f"Broke segments into {len(individual_myotubes)} individual myotubes")
        return individual_myotubes

    def reconstruct_myotube_masks(self, skeleton_segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Reconstruct full myotube masks from skeleton segments using watershed segmentation.
        
        Args:
            skeleton_segments: List of individual skeleton segments
            
        Returns:
            List of reconstructed myotube masks
        """
        print("Reconstructing myotube masks from skeleton segments...")
        
        if not skeleton_segments:
            self.instance_masks = []
            self.total_myotubes = 0
            return []
        
        # Create markers for watershed
        markers = np.zeros_like(self.binary_image, dtype=int)
        valid_segments = []
        
        for i, segment in tqdm(enumerate(skeleton_segments), 
                              total=len(skeleton_segments), 
                              desc="Creating watershed markers"):
            seed = binary_dilation(segment, disk(3))
            if np.sum(seed) > 0:
                markers[seed] = i + 1
                valid_segments.append((i + 1, segment))
        
        if not valid_segments:
            print("No valid segments found")
            self.instance_masks = []
            self.total_myotubes = 0
            return []
        
        # Perform watershed segmentation
        distance = ndimage.distance_transform_edt(self.binary_image)
        watershed_result = watershed(-distance, markers, mask=self.binary_image)
        
        # Extract and clean individual masks
        masks = []
        for label_id, _ in tqdm(valid_segments, desc="Extracting and cleaning masks"):
            myotube_mask = (watershed_result == label_id)
            
            # Clean up the mask
            myotube_mask = morphology.remove_small_holes(myotube_mask, area_threshold=100)
            myotube_mask = morphology.binary_closing(myotube_mask, disk(2))
            
            # Keep masks above minimum area threshold
            if np.sum(myotube_mask) > 100:
                masks.append(myotube_mask)
        
        self.instance_masks = masks
        self.total_myotubes = len(masks)
        print(f"Reconstructed {self.total_myotubes} unique myotube masks")
        return masks

    def get_farthest_points(self, mask: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Find the two extreme points (endpoints) of a myotube mask along the PCA direction.
        
        Uses Principal Component Analysis to find the main direction of the elongated 
        myotube and then finds the two points that are furthest apart along this direction.

        Args:
            mask: Binary mask of myotube

        Returns:
            Tuple of two endpoint coordinates: ((y1, x1), (y2, x2)) or (None, None) if invalid
        """
        y_coords, x_coords = np.where(mask)
        if len(y_coords) < 2:
            return None, None
        
        # Stack coordinates as [y, x] points for PCA
        points = np.column_stack((y_coords, x_coords))
        
        if len(points) < 2:
            return None, None
        
        try:
            # Apply PCA to find the principal direction
            pca = PCA(n_components=1)
            pca.fit(points)

            # Get the principal component (direction vector)
            principal_direction = pca.components_[0]  # Shape: (2,) for [y, x]
            
            # Project all points onto the principal direction
            # Center the data first
            mean_point = np.mean(points, axis=0)
            centered_points = points - mean_point
            
            # Project points onto principal direction
            projections = np.dot(centered_points, principal_direction)

            # Find the points with minimum and maximum projections (extreme points)
            min_idx = np.argmin(projections)
            max_idx = np.argmax(projections)
            
            endpoint1 = tuple(points[min_idx])
            endpoint2 = tuple(points[max_idx])
            
            return endpoint1, endpoint2
            
        except Exception as e:
            # Fallback to original method if PCA fails
            print(f"PCA failed for mask, using fallback method: {e}")
            distances = cdist(points, points)
            max_idx = np.unravel_index(np.argmax(distances), distances.shape)
            return tuple(points[max_idx[0]]), tuple(points[max_idx[1]])
        
    def get_direction_vector(self, point1: Optional[Tuple[int, int]], 
                           point2: Optional[Tuple[int, int]]) -> np.ndarray:
        """
        Calculate normalized direction vector between two points.
        
        Args:
            point1: First point (y1, x1)
            point2: Second point (y2, x2)
            
        Returns:
            Normalized direction vector or zero vector if points are invalid
        """
        if point1 is None or point2 is None:
            return np.array([0.0, 0.0])
        
        direction = np.array([point2[0] - point1[0], point2[1] - point1[1]], dtype=float)
        norm = np.linalg.norm(direction)
        return direction / norm if norm > 0 else np.array([0.0, 0.0])
    
    def get_pca_direction_vector(self, mask: np.ndarray) -> np.ndarray:
        """
        Calculate the principal direction vector of a myotube mask using PCA.
        
        Args:
            mask: Binary mask of myotube
            
        Returns:
            Normalized PCA direction vector or zero vector if calculation fails
        """
        y_coords, x_coords = np.where(mask)
        if len(y_coords) < 2:
            return np.array([0.0, 0.0])
        
        # Stack coordinates as [y, x] points for PCA
        points = np.column_stack((y_coords, x_coords))
        
        if len(points) < 2:
            return np.array([0.0, 0.0])
        
        try:
            # Apply PCA to find the principal direction
            pca = PCA(n_components=1)
            pca.fit(points)
            
            # Get the principal component (direction vector)
            principal_direction = pca.components_[0]  # Shape: (2,) for [y, x]
            
            # Ensure it's normalized
            norm = np.linalg.norm(principal_direction)
            if norm > 0:
                return principal_direction / norm
            else:
                return np.array([0.0, 0.0])
                
        except Exception as e:
            print(f"PCA direction calculation failed for mask, returning zero vector: {e}")
            return np.array([0.0, 0.0])

    def calculate_merge_score(self, mask1: np.ndarray, mask2: np.ndarray, 
                             direction1: np.ndarray, direction2: np.ndarray,
                             farthest1: Tuple, farthest2: Tuple) -> Union[float, str]:
        """
        Calculate geometric merge score for two myotube segments.

        Args:
            mask1, mask2: Binary masks (for consistency, not used in calculation)
            direction1, direction2: Pre-calculated direction vectors
            farthest1, farthest2: Pre-calculated farthest point pairs

        Returns:
            Numeric merge score (lower is better) or rejection reason string
        """
        if farthest1[0] is None or farthest2[0] is None:
            return "REJECTED: Invalid farthest points"
        
        # Find closest pair of endpoints
        endpoints1 = [farthest1[0], farthest1[1]]
        endpoints2 = [farthest2[0], farthest2[1]]
        
        min_distance = float('inf')
        closest_pair = None
        
        for i, end1 in enumerate(endpoints1):
            for j, end2 in enumerate(endpoints2):
                distance = np.sqrt((end1[0] - end2[0])**2 + (end1[1] - end2[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j, end1, end2)
        
        # Distance threshold check
        max_distance_threshold = 60.0
        if min_distance > max_distance_threshold:
            return f"REJECTED: Too far apart ({min_distance:.1f} > {max_distance_threshold})"
        
        # Geometric analysis
        i, j, closest1, closest2 = closest_pair
        distance_vector = np.array([closest2[0] - closest1[0], closest2[1] - closest1[1]])

        # Decompose distance into parallel and perpendicular components
        avg_direction = (direction1 + direction2) / 2
        avg_direction_norm = np.linalg.norm(avg_direction)
        if avg_direction_norm > 0:
            avg_direction = avg_direction / avg_direction_norm
        else:
            return "REJECTED: Invalid direction vectors"

        parallel_distance = abs(np.dot(distance_vector, avg_direction))
        projection = np.dot(distance_vector, avg_direction) * avg_direction
        perpendicular_distance = np.linalg.norm(distance_vector - projection)

        # Apply distance thresholds
        if parallel_distance > 60.0:
            return f"REJECTED: Parallel distance too large ({parallel_distance:.1f})"
        if perpendicular_distance > 30.0:
            return f"REJECTED: Perpendicular distance too large ({perpendicular_distance:.1f})"

        # Check end-to-end connection pattern
        far1 = endpoints1[1-i]
        far2 = endpoints2[1-j]
        
        vector1_to_far = np.array([far1[0] - closest1[0], far1[1] - closest1[1]])
        vector2_to_far = np.array([far2[0] - closest2[0], far2[1] - closest2[1]])
        
        norm1, norm2 = np.linalg.norm(vector1_to_far), np.linalg.norm(vector2_to_far)
        
        if norm1 > 0 and norm2 > 0:
            connection_dot = np.dot(vector1_to_far / norm1, vector2_to_far / norm2)
            if connection_dot > -0.3:
                return f"REJECTED: Not end-to-end connected (dot: {connection_dot:.3f})"
        else:
            return "REJECTED: Invalid vector norms"
        
        # Check direction alignment
        dot_product = np.abs(np.dot(direction1, direction2))
        direction_score = 1.0 - dot_product
        if direction_score > 0.15:
            return f"REJECTED: Poor direction alignment ({direction_score:.3f})"

        # Calculate final merge score
        parallel_score = parallel_distance / 60.0
        perpendicular_score = perpendicular_distance / 30.0
        return 2.0 * direction_score + 0.3 * parallel_score + 0.7 * perpendicular_score
    
    def visualize_merge_pair(self, seg1_data: dict, seg2_data: dict, score: Union[float, str]) -> None:
        """
        Visualize a pair of segments about to be merged for debugging purposes.
        
        Args:
            seg1_data: First segment data dictionary
            seg2_data: Second segment data dictionary
            score: Merge score for this pair
        """
        mask1, mask2 = seg1_data['mask'], seg2_data['mask']
        height, width = mask1.shape
        
        # Create colored visualization
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        vis_image[mask1, 0] = 255  # Red for segment 1
        vis_image[mask2, 1] = 255  # Green for segment 2
        
        # Mark farthest points
        for farthest_points, color in [(seg1_data['farthest_points'], (255, 255, 0)), 
                                      (seg2_data['farthest_points'], (0, 255, 255))]:
            if farthest_points[0] is not None and farthest_points[1] is not None:
                for point in farthest_points:
                    cv2.circle(vis_image, (point[1], point[0]), 3, color, -1)
        
        # Add information text
        cv2.putText(vis_image, f"Merge Score: {score} | Seg1: {seg1_data['id']} | Seg2: {seg2_data['id']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, "Press any key to continue...", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Merge Preview', vis_image)
        cv2.waitKey(0)
    
    def merge_broken_segments(self, masks: List[np.ndarray], 
                            max_iterations: int = 1000,
                            score_threshold: float = 1,
                            show_merge_preview: bool = False,
                            min_size_threshold: int = 4000) -> List[np.ndarray]:
        """
        Iteratively merge broken myotube segments based on geometric criteria.
        Uses optimized merge score caching to avoid redundant calculations.
        
        Args:
            masks: List of myotube masks
            max_iterations: Maximum number of merge iterations
            score_threshold: Maximum score to consider for merging
            show_merge_preview: Whether to show merge preview before each merge
            min_size_threshold: Minimum number of pixels for a myotube to be preserved
            
        Returns:
            List of merged myotube masks that meet the size threshold
        """
        print("Merging broken myotube segments...")
        
        if len(masks) <= 1:
            return masks
        
        # Initialize segment data with cached calculations
        segments = []
        for i, mask in enumerate(masks):
            farthest_points = self.get_farthest_points(mask)
            # Use PCA-based direction vector instead of point-to-point calculation
            direction_vector = self.get_pca_direction_vector(mask)
            segments.append({
                'id': i,
                'mask': mask,
                'farthest_points': farthest_points,
                'direction_vector': direction_vector
            })
        
        # Pre-calculate all initial merge scores (optimization)
        print("Pre-calculating merge scores...")
        merge_scores = {}
        total_pairs = len(segments) * (len(segments) - 1) // 2
        current_pair = 0

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                current_pair += 1

                # Progress indicator for large datasets
                if total_pairs > 1000 and current_pair % (total_pairs // 10) == 0:
                    progress = (current_pair / total_pairs) * 100
                    print(f"  Progress: {progress:.0f}% ({current_pair}/{total_pairs} pairs)")

                seg1, seg2 = segments[i], segments[j]
                score = self.calculate_merge_score(
                        seg1['mask'], seg2['mask'],
                        seg1['direction_vector'], seg2['direction_vector'],
                        seg1['farthest_points'], seg2['farthest_points']
                    )

                # Store score with segment IDs as key (order-independent)
                key = (min(seg1['id'], seg2['id']), max(seg1['id'], seg2['id']))
                merge_scores[key] = score
        
        print(f"Pre-calculated {len(merge_scores)} merge scores")
        
        iteration = 0
        
        while iteration < max_iterations and len(segments) > 1:
            print(f"Merge iteration {iteration + 1}: {len(segments)} segments")
            
            # Find best merge candidate from cached scores
            best_score = float('inf')
            best_pair_idx = None
            best_key = None
            
            # Create mapping from segment ID to current index
            id_to_idx = {seg['id']: idx for idx, seg in enumerate(segments)}
            
            for key, score in merge_scores.items():
                seg1_id, seg2_id = key
                
                # Check if both segments still exist
                if seg1_id in id_to_idx and seg2_id in id_to_idx:
                    if isinstance(score, (int, float)) and score < best_score and score <= score_threshold:
                        best_score = score
                        best_pair_idx = (id_to_idx[seg1_id], id_to_idx[seg2_id])
                        best_key = key
            
            # Stop if no good merge found
            if best_pair_idx is None:
                print("No more merges possible")
                break
            
            # Perform the best merge
            i, j = best_pair_idx
            seg1_id, seg2_id = best_key
            print(f"Merging segments {seg1_id} and {seg2_id} (score: {best_score:.3f})")
            
            # Show merge preview if requested
            if show_merge_preview:
                self.visualize_merge_pair(segments[i], segments[j], best_score)
            
            # Merge masks and update data
            merged_mask = segments[i]['mask'] | segments[j]['mask']
            new_farthest_points = self.get_farthest_points(merged_mask)
            # Use PCA-based direction vector for merged segment
            new_direction_vector = self.get_pca_direction_vector(merged_mask)
            
            new_id = max(seg1_id, seg2_id)
            merged_segment = {
                'id': new_id,
                'mask': merged_mask,
                'farthest_points': new_farthest_points,
                'direction_vector': new_direction_vector
            }
            
            # Update merge scores cache: remove old scores and calculate new ones
            keys_to_remove = set()
            new_scores = {}
            
            # Remove scores involving the merged segments
            for key in merge_scores.keys():
                if seg1_id in key or seg2_id in key:
                    keys_to_remove.add(key)
            
            for key in keys_to_remove:
                del merge_scores[key]
            
            # Calculate new scores for the merged segment with all remaining segments
            remaining_segments = [seg for idx, seg in enumerate(segments) if idx not in (i, j)]
            
            for seg in remaining_segments:
                score = self.calculate_merge_score(
                    merged_segment['mask'], seg['mask'],
                    merged_segment['direction_vector'], seg['direction_vector'],
                    merged_segment['farthest_points'], seg['farthest_points']
                )
                
                key = (min(new_id, seg['id']), max(new_id, seg['id']))
                new_scores[key] = score
            
            # Update data structures
            merge_scores.update(new_scores)
            segments = remaining_segments + [merged_segment]
            iteration += 1
        
        final_masks = [seg['mask'] for seg in segments]
        print(f"Merging completed: {len(masks)} → {len(final_masks)} segments")
        
        # Filter out myotubes smaller than the minimum size threshold
        if min_size_threshold > 0:
            print(f"Filtering myotubes by minimum size threshold: {min_size_threshold} pixels")
            size_filtered_masks = []
            for mask in final_masks:
                pixel_count = np.sum(mask)
                if pixel_count >= min_size_threshold:
                    size_filtered_masks.append(mask)
                else:
                    print(f"  Removed myotube with {pixel_count} pixels (below threshold)")
            
            print(f"Size filtering completed: {len(final_masks)} → {len(size_filtered_masks)} segments")
            return size_filtered_masks
        
        return final_masks
    
    def calculate_bounding_boxes(self) -> List[Tuple[int, int, int, int]]:
        """
        Calculate bounding boxes for each myotube instance.

        Returns:
            List of bounding boxes as (min_row, min_col, max_row, max_col)
        """
        bboxes = []
        for mask in self.instance_masks:
            props = measure.regionprops(mask.astype(int))
            bbox = props[0].bbox if props else (0, 0, 0, 0)
            bboxes.append(bbox)
        return bboxes

    def _generate_distinct_colors(self, n: int) -> List[Tuple[float, float, float]]:
        """Generate n distinct colors using HSV color space."""
        import colorsys
        colors = []
        for i in range(n):
            hue = (i * 137.5) % 360
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.8 + (i % 2) * 0.2
            rgb = colorsys.hsv_to_rgb(hue/360.0, saturation, value)
            colors.append(rgb)
        return colors

    def visualize_results(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive visualization of segmentation results.
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure object
        """
        print("Creating visualization...")

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()

        # Original image
        axes[0].imshow(self.original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Preprocessed image
        axes[1].imshow(self.preprocessed_image, cmap='gray')
        axes[1].set_title('Preprocessed Image')
        axes[1].axis('off')

        # Enhanced tubular structures
        axes[2].imshow(self.enhanced_image, cmap='hot')
        axes[2].set_title('Enhanced Tubular Structures\n(Frangi Filter)')
        axes[2].axis('off')
        
        # Binary image
        axes[3].imshow(self.binary_image, cmap='gray')
        axes[3].set_title('Binary Image')
        axes[3].axis('off')

        # Skeleton
        axes[4].imshow(self.skeleton, cmap='gray')
        axes[4].set_title('Skeleton')
        axes[4].axis('off')

        # Skeleton segments
        initial_segments = self.split_skeleton_segments()
        initial_combined = np.zeros_like(self.skeleton)
        for segment in initial_segments:
            initial_combined |= segment
        axes[5].imshow(initial_combined, cmap='gray')
        axes[5].set_title(f'Skeleton Segments\n({len(initial_segments)} segments)')
        axes[5].axis('off')

        # Turning points visualization
        all_break_points = np.zeros_like(self.skeleton, dtype=bool)
        for segment in initial_segments:
            turning_points = self.detect_turning_points(segment)
            all_break_points |= turning_points

        break_viz = np.zeros((*self.skeleton.shape, 3))
        break_viz[self.skeleton, 1] = 1.0  # Skeleton in green
        break_viz[all_break_points, 0] = 1.0  # Straight segments in red
        break_viz[all_break_points, 1] = 0.0
        axes[6].imshow(break_viz)
        axes[6].set_title('Straight Segments Detection\n(Red: straight, Green: corners)')
        axes[6].axis('off')

        # Individual myotubes and final segmentation
        colors = self._generate_distinct_colors(len(self.instance_masks))
        alpha = 0.6

        for idx, title_suffix in enumerate(['Individual Myotubes', 'Final Segmentation']):
            colored_masks = np.zeros((*self.original_image.shape, 3))
            
            for i, mask in enumerate(self.instance_masks):
                color = colors[i][:3]
                for c in range(3):
                    colored_masks[mask, c] = color[c]
            
            overlay_rgb = np.stack([self.original_image] * 3, axis=-1)
            final_overlay = alpha * colored_masks + (1 - alpha) * overlay_rgb
            
            axes[7 + idx].imshow(final_overlay)
            axes[7 + idx].set_title(f'{title_suffix}\n({len(self.instance_masks)} myotubes)')
            
            # Add ID numbers
            for i, mask in enumerate(self.instance_masks):
                props = measure.regionprops(mask.astype(int))
                if props:
                    centroid = props[0].centroid
                    axes[7 + idx].text(centroid[1], centroid[0], str(i + 1),
                                      color='white', fontsize=5, fontweight='bold',
                                      ha='center', va='center',
                                      bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            axes[7 + idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        return fig
    
    def create_merge_scores_dataframe(self):
        """
        Create a DataFrame showing merge scores for all pairs in final segmentation.
        Optimized with pre-calculated geometric features for better performance.
        
        Returns:
            DataFrame with columns: myotube_1, myotube_2, score_or_reason
        """
        try:
            import pandas as pd
        except ImportError:
            print("Warning: pandas not available, skipping DataFrame creation")
            return None
        
        print("Creating merge scores DataFrame...")
        
        if len(self.instance_masks) < 2:
            return pd.DataFrame(columns=['myotube_1', 'myotube_2', 'score_or_reason'])
        
        # Calculate total number of pairs
        total_pairs = len(self.instance_masks) * (len(self.instance_masks) - 1) // 2
        print(f"Calculating merge scores for {total_pairs} pairs...")
        
        from tqdm import tqdm
        
        # OPTIMIZATION: Pre-calculate farthest points and direction vectors for all masks
        print("Pre-calculating geometric features for all myotubes...")
        myotube_features = []
        
        for i, mask in enumerate(tqdm(self.instance_masks, desc="Pre-calculating features", unit="masks")):
            farthest_points = self.get_farthest_points(mask)
            # Use PCA-based direction vector instead of point-to-point calculation
            direction_vector = self.get_pca_direction_vector(mask)
            
            myotube_features.append({
                'mask': mask,
                'farthest_points': farthest_points,
                'direction_vector': direction_vector
            })
        
        print(f"Pre-calculated features for {len(myotube_features)} myotubes")
        
        # Calculate merge scores using pre-calculated features
        pairs_data = []
        pairs_iter = tqdm(total=total_pairs, desc="Calculating merge scores", unit="pairs")
        
        for i in range(len(self.instance_masks)):
            for j in range(i + 1, len(self.instance_masks)):
                # Use pre-calculated features (major performance improvement)
                features1 = myotube_features[i]
                features2 = myotube_features[j]
                
                score = self.calculate_merge_score(
                    features1['mask'], features2['mask'],
                    features1['direction_vector'], features2['direction_vector'],
                    features1['farthest_points'], features2['farthest_points']
                )
                
                pairs_data.append({
                    'myotube_1': i + 1,
                    'myotube_2': j + 1,
                    'score_or_reason': score
                })
                
                pairs_iter.update(1)
        
        pairs_iter.close()
        
        df = pd.DataFrame(pairs_data)
        print(f"Created DataFrame with {len(df)} pairs")
        return df
    
    def save_merge_scores_csv(self, output_path: str = 'myotube_merge_scores.csv'):
        """
        Save merge scores DataFrame to CSV file.
        
        Args:
            output_path: Path to save the CSV file
        """
        df = self.create_merge_scores_dataframe()
        if df is not None:
            df.to_csv(output_path, index=False)
        print(f"Merge scores saved to: {output_path}")
        return df

    def run_segmentation(self, 
                        threshold_method: Union[str, float] = 0.01,
                        show_merge_preview: bool = False, 
                        target_size: int = 2000,
                        opening_kernel_size: int = 2,
                        min_size_threshold: int = 1000) -> Dict:
        """
        Run the complete segmentation pipeline with comprehensive error handling.

        Args:
            threshold_method: Thresholding method ('otsu', 'adaptive', or numeric value)
            show_merge_preview: Whether to show merge preview before each merge
            target_size: Target size for image resizing (default: 2000)
            opening_kernel_size: Size of morphological opening kernel (default: 3)
            min_size_threshold: Minimum number of pixels for a myotube to be preserved (default: 500)

        Returns:
            Dictionary with segmentation results and statistics
        """
        print("Starting myotube segmentation pipeline...")
        print("=" * 50)

        try:
            # Run pipeline steps
            self.load_image(target_size=target_size)
            self.preprocess_image()
            self.enhance_tubular_structures()
            self.binarize_and_skeletonize(threshold_method=threshold_method, 
                                        opening_kernel_size=opening_kernel_size)
            
            # Split and break skeleton segments
            skeleton_segments = self.split_skeleton_segments()
            individual_myotubes = self.break_skeleton_segments(skeleton_segments)
            
            # Reconstruct and merge masks
            self.reconstruct_myotube_masks(individual_myotubes)
            merged_masks = self.merge_broken_segments(self.instance_masks, 
                                                    show_merge_preview=show_merge_preview,
                                                    min_size_threshold=min_size_threshold)
            self.instance_masks = merged_masks
            self.total_myotubes = len(merged_masks)
            
            # Calculate statistics
            areas = [np.sum(mask) for mask in self.instance_masks]
            bboxes = self.calculate_bounding_boxes()
            
            results = {
                'total_myotubes': self.total_myotubes,
                'instance_masks': self.instance_masks,
                'areas': areas,
                'bounding_boxes': bboxes,
                'mean_area': np.mean(areas) if areas else 0,
                'std_area': np.std(areas) if areas else 0
            }
            
            print("=" * 50)
            print("Segmentation completed successfully!")
            print(f"Total myotubes detected: {self.total_myotubes}")
            print(f"Mean myotube area: {results['mean_area']:.1f} ± {results['std_area']:.1f} pixels")
            
            return results
            
        except Exception as e:
            print(f"Error during segmentation: {str(e)}")
            raise


def main():
    """
    Main function to run the myotube segmentation pipeline with proper error handling.
    """
    image_path = "../C4-MAX_20250306_T34_48hp.lif - Region 4_Merged.lif - Region 4_Merged.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Please make sure the image file exists in the current directory.")
        return
    
    # Initialize and run segmentation
    segmenter = MyotubeSegmenter(image_path)
    
    try:
        results = segmenter.run_segmentation(
            show_merge_preview=False,
            target_size=2000,
            min_size_threshold=1000  # Only keep myotubes with at least 500 pixels
        )
        
        # Create visualization and save results
        fig = segmenter.visualize_results(save_path='myotube_segmentation_result.png')
        plt.show()
        
        segmenter.save_merge_scores_csv('myotube_merge_scores.csv')
        
        # Print summary
        print("\nSegmentation Summary:")
        print(f"- Total myotubes detected: {results['total_myotubes']}")
        print(f"- Average myotube area: {results['mean_area']:.1f} pixels")
        print(f"- Results saved to: myotube_segmentation_result.png")
        print(f"- Merge scores saved to: myotube_merge_scores.csv")
        
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 