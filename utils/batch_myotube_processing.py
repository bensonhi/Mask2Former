#!/usr/bin/env python3
"""
Batch Myotube Segmentation and COCO Dataset Creation

This script processes multiple myotube images in a folder and creates a combined
COCO format dataset with processed images at a specified resolution.

Features:
- Recursively process all image files in directory and subdirectories
- Create a single COCO format JSON file combining all images
- Save processed images at specified resolution (default: 1500px) in flat structure
- Save annotations at the same resolution as processed images
- Support for multiple image formats (PNG, JPG, TIFF, etc.)
- Uses all_contours mode for complete myotube structure preservation
- Flatten directory structure in output (all images in single folder)

Usage:
    python batch_myotube_processing.py --input_dir /path/to/images --output_dir /path/to/output
"""

import os
import glob
import json
import datetime
import argparse
from typing import List, Dict, Optional, Tuple
import shutil
import random

import cv2
import numpy as np
from tqdm import tqdm

# Import the MyotubeSegmenter class
from myotube_segmentation import MyotubeSegmenter


class BatchMyotubeProcessor:
    """
    Batch processor for myotube segmentation and COCO dataset creation.
    Always uses all_contours mode for complete structural preservation.
    """
    
    def __init__(self, input_dir: str, output_dir: str, target_resolution: int = 1500, 
                 segmentation_resolution: int = 2000):
        """
        Initialize the batch processor.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed results (flat structure)
            target_resolution: Target resolution for output images and annotations (default: 1500)
            segmentation_resolution: Resolution for segmentation processing (default: 2000, optimally tuned)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_resolution = target_resolution
        self.segmentation_resolution = segmentation_resolution
        
        # Create output directories for images and annotations
        self.images_dir = os.path.join(self.output_dir, "images")
        self.annotations_dir = os.path.join(self.output_dir, "annotations")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        # Update COCO file path to annotations directory
        self.coco_file_path = os.path.join(self.annotations_dir, "annotations.json")
        self.train_coco_file_path = os.path.join(self.annotations_dir, "algorithmic_train_annotations.json")
        self.test_coco_file_path = os.path.join(self.annotations_dir, "algorithmic_test_annotations.json")
        
        # Supported image formats
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
        
        # Initialize COCO data structure
        self.coco_data = {
            "info": {
                "description": "Myotube Instance Segmentation Dataset",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "BatchMyotubeProcessor",
                "date_created": datetime.datetime.now().isoformat(),
                "output_resolution": target_resolution,
                "segmentation_resolution": segmentation_resolution,
                "polygon_mode": "all_contours"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "myotube",
                    "supercategory": "cell"
                }
            ]
        }
        
        self.next_image_id = 1
        self.next_annotation_id = 1
    
    def find_image_files(self) -> List[str]:
        """
        Find all supported image files in the input directory and its subdirectories.
        
        Returns:
            List of image file paths
        """
        image_files = []
        
        for ext in self.supported_formats:
            # Recursive search pattern to include subdirectories
            pattern = os.path.join(self.input_dir, f"**/*{ext}")
            image_files.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(self.input_dir, f"**/*{ext.upper()}")
            image_files.extend(glob.glob(pattern, recursive=True))
        
        image_files = sorted(list(set(image_files)))  # Remove duplicates and sort
        print(f"Found {len(image_files)} image files (including subdirectories)")
        return image_files
    
    def mask_to_polygons(self, mask: np.ndarray) -> List[List[float]]:
        """
        Convert a binary mask to polygon format for COCO annotations.
        Always uses all_contours mode for complete structural preservation.
        
        Args:
            mask: Binary mask as numpy array
            
        Returns:
            List of polygons, where each polygon is a list of [x1,y1,x2,y2,...] coordinates
        """
        # Find contours in the mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Include all contours for complete structural preservation
        polygons = []
        for contour in contours:
            # Skip very small contours
            if len(contour) < 3:
                continue
                
            # Flatten contour coordinates to [x1,y1,x2,y2,...] format
            polygon = contour.flatten().tolist()
            
            # COCO requires at least 6 coordinates (3 points)
            if len(polygon) >= 6:
                polygons.append(polygon)
        
        return polygons
    
    def resize_image_to_target(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize image to target resolution while maintaining aspect ratio.
        
        Args:
            image: Input image as numpy array
            target_size: Target size for the longer dimension
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        # Calculate new dimensions while maintaining aspect ratio
        if height > width:
            new_height = target_size
            new_width = int(width * (target_size / height))
        else:
            new_width = target_size
            new_height = int(height * (target_size / width))
        
        # Only resize if image is larger than target
        if height > target_size or width > target_size:
            if len(image.shape) == 3:
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            else:
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            return resized
        else:
            return image
    
    def process_single_image(self, image_path: str) -> Optional[Dict]:
        """
        Process a single image and return segmentation results.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing processed image info and segmentation results
        """
        print(f"Processing: {os.path.basename(image_path)}")
        
        try:
            # Initialize segmenter and run segmentation at optimal resolution (2000px)
            segmenter = MyotubeSegmenter(image_path)
            results = segmenter.run_segmentation(target_size=self.segmentation_resolution)
            
            # Load the original image for output scaling
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Warning: Could not load image {image_path}")
                return None
            
            # Scale original image directly to target output resolution (avoid double-scaling)
            processed_image = self.resize_image_to_target(original_image, self.target_resolution)
            
            # Generate unique output filename (include subdirectory to avoid conflicts)
            relative_path = os.path.relpath(image_path, self.input_dir)
            relative_dir = os.path.dirname(relative_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Create unique filename that includes subdirectory info
            if relative_dir and relative_dir != '.':
                # Replace path separators with underscores to create flat filename
                safe_dir = relative_dir.replace(os.sep, '_').replace(' ', '_')
                output_filename = f"{safe_dir}_{base_name}_processed.png"
            else:
                output_filename = f"{base_name}_processed.png"
            
            output_path = os.path.join(self.images_dir, output_filename)
            
            # Save processed image
            cv2.imwrite(output_path, processed_image)
            
            # Get processed image dimensions
            processed_height, processed_width = processed_image.shape[:2]
            
            # Calculate scale factor from segmentation resolution to target output resolution
            # segmenter.original_image is the image used for segmentation (at segmentation_resolution)
            if segmenter.original_image is None:
                print(f"Warning: Segmenter original image is None for {image_path}")
                return None
            
            seg_height, seg_width = segmenter.original_image.shape[:2]
            scale_x = processed_width / seg_width
            scale_y = processed_height / seg_height
            
            # Process masks and create annotations
            annotations = []
            for mask_id, mask in enumerate(segmenter.instance_masks):
                # Scale mask from segmentation resolution to target output resolution
                scaled_mask = cv2.resize(
                    mask.astype(np.uint8), 
                    (processed_width, processed_height), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                
                # Convert mask to polygons (always using all_contours mode)
                polygons = self.mask_to_polygons(scaled_mask)
                
                if not polygons:
                    print(f"Warning: No polygons found for mask {mask_id + 1} in {output_filename}")
                    continue
                
                # Calculate area and bounding box
                area = float(np.sum(scaled_mask))
                y_indices, x_indices = np.where(scaled_mask)
                
                if len(x_indices) == 0 or len(y_indices) == 0:
                    continue
                
                x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
                y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
                bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
                
                # Create annotation
                annotation = {
                    "id": self.next_annotation_id,
                    "image_id": self.next_image_id,
                    "category_id": 1,
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                
                annotations.append(annotation)
                self.next_annotation_id += 1
            
            return {
                "image_info": {
                    "id": self.next_image_id,
                    "width": processed_width,
                    "height": processed_height,
                    "file_name": output_filename,
                    "license": 1,
                    "date_captured": datetime.datetime.now().isoformat(),
                    "original_file": os.path.basename(image_path),
                    "myotube_count": len(segmenter.instance_masks)
                },
                "annotations": annotations,
                "stats": {
                    "total_myotubes": len(segmenter.instance_masks),
                    "valid_annotations": len(annotations),
                    "total_polygon_parts": sum(len(ann["segmentation"]) for ann in annotations),
                    "segmentation_resolution": f"{seg_width}x{seg_height}",
                    "output_resolution": f"{processed_width}x{processed_height}"
                }
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def process_batch(self) -> None:
        """
        Process all images in the input directory and create COCO dataset.
        Always uses all_contours mode for complete structural preservation.
        """
        print("Starting batch processing...")
        print(f"Input directory: {self.input_dir} (including subdirectories)")
        print(f"Output directory: {self.output_dir} (flat structure)")
        print(f"Segmentation resolution: {self.segmentation_resolution}px (optimal tuned parameters)")
        print(f"Output resolution: {self.target_resolution}px (training/inference resolution)")
        print(f"Polygon mode: all_contours (preserves complete myotube structure)")
        print("="*60)
        
        # Find all image files
        image_files = self.find_image_files()
        
        if not image_files:
            print("No image files found in the input directory!")
            return
        
        # Process each image
        total_myotubes = 0
        total_annotations = 0
        total_polygon_parts = 0
        successful_images = 0
        
        for image_path in tqdm(image_files, desc="Processing images"):
            result = self.process_single_image(image_path)
            
            if result is not None:
                # Add image info to COCO data
                self.coco_data["images"].append(result["image_info"])
                
                # Add annotations to COCO data
                self.coco_data["annotations"].extend(result["annotations"])
                
                # Update statistics
                total_myotubes += result["stats"]["total_myotubes"]
                total_annotations += result["stats"]["valid_annotations"]
                total_polygon_parts += result["stats"]["total_polygon_parts"]
                successful_images += 1
                
                # Move to next image ID
                self.next_image_id += 1
                
                print(f"  ✓ {result['stats']['total_myotubes']} myotubes, "
                      f"{result['stats']['valid_annotations']} annotations, "
                      f"{result['stats']['total_polygon_parts']} polygon parts, "
                      f"seg: {result['stats']['segmentation_resolution']} → out: {result['stats']['output_resolution']}")
        
        # Save COCO format file
        print(f"\nSaving COCO dataset to: {self.coco_file_path}")
        with open(self.coco_file_path, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        
        # Create train/test split and save separate annotation files
        if successful_images > 0:
            self.create_train_test_split(train_ratio=0.9, random_seed=42)
        
        # Print final summary
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Processed images: {successful_images}/{len(image_files)}")
        print(f"Total myotubes detected: {total_myotubes}")
        print(f"Total annotations created: {total_annotations}")
        print(f"Total polygon parts: {total_polygon_parts}")
        print(f"Average polygon parts per myotube: {total_polygon_parts/total_myotubes:.1f}")
        print(f"Images saved to: {self.images_dir}")
        print(f"COCO annotations saved to: {self.coco_file_path}")
        print(f"Segmentation resolution: {self.segmentation_resolution}px (processing)")
        print(f"Output resolution: {self.target_resolution}px (images & annotations)")
        
        print("\n✅ Complete structural preservation mode")
        print("   • All contour parts included for each myotube")
        print("   • Preserves complex shapes, holes, and branching")
        print("   • Optimal for training deep learning models")
        print("   • May create multiple polygon parts per myotube")


    def create_train_test_split(self, train_ratio: float = 0.9, random_seed: int = 42) -> None:
        """
        Create train/test split of the dataset and save separate annotation files.
        
        Args:
            train_ratio: Ratio of images to use for training (default: 0.9 for 90/10 split)
            random_seed: Random seed for reproducible splits
        """
        print(f"\nCreating train/test split ({train_ratio:.0%}/{1-train_ratio:.0%})...")
        
        # Set random seed for reproducible splits
        random.seed(random_seed)
        
        # Get all image IDs
        all_image_ids = [img["id"] for img in self.coco_data["images"]]
        
        # Shuffle and split
        random.shuffle(all_image_ids)
        split_idx = int(len(all_image_ids) * train_ratio)
        train_image_ids = set(all_image_ids[:split_idx])
        test_image_ids = set(all_image_ids[split_idx:])
        
        # Create train dataset
        train_data = {
            "info": self.coco_data["info"].copy(),
            "licenses": self.coco_data["licenses"].copy(),
            "categories": self.coco_data["categories"].copy(),
            "images": [img for img in self.coco_data["images"] if img["id"] in train_image_ids],
            "annotations": [ann for ann in self.coco_data["annotations"] if ann["image_id"] in train_image_ids]
        }
        train_data["info"]["description"] = "Myotube Instance Segmentation Dataset - Training Set"
        
        # Create test dataset
        test_data = {
            "info": self.coco_data["info"].copy(),
            "licenses": self.coco_data["licenses"].copy(),
            "categories": self.coco_data["categories"].copy(),
            "images": [img for img in self.coco_data["images"] if img["id"] in test_image_ids],
            "annotations": [ann for ann in self.coco_data["annotations"] if ann["image_id"] in test_image_ids]
        }
        test_data["info"]["description"] = "Myotube Instance Segmentation Dataset - Test Set"
        
        # Save train annotations
        with open(self.train_coco_file_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        # Save test annotations
        with open(self.test_coco_file_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Print split statistics
        train_myotubes = len(train_data["annotations"])
        test_myotubes = len(test_data["annotations"])
        total_myotubes = train_myotubes + test_myotubes
        
        print(f"Train set: {len(train_data['images'])} images, {train_myotubes} myotubes ({train_myotubes/total_myotubes:.1%})")
        print(f"Test set: {len(test_data['images'])} images, {test_myotubes} myotubes ({test_myotubes/total_myotubes:.1%})")
        print(f"Train annotations saved to: {self.train_coco_file_path}")
        print(f"Test annotations saved to: {self.test_coco_file_path}")


def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(description="Batch process myotube images and create COCO dataset")
    parser.add_argument("--input_dir", "-i", default="max_projected_images",
                       help="Directory containing input images (default: max_projected_images)")
    parser.add_argument("--output_dir", "-o", default="./myotube_batch_output",
                       help="Directory to save processed results (default: ./myotube_batch_output)")
    parser.add_argument("--resolution", "-r", type=int, default=9000,
                       help="Output resolution for processed images and annotations (default: 9000)")
    parser.add_argument("--seg_resolution", "-s", type=int, default=2000,
                       help="Segmentation processing resolution with tuned parameters (default: 2000)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory!")
        return
    
    # Create processor and run batch processing
    processor = BatchMyotubeProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_resolution=args.resolution,
        segmentation_resolution=args.seg_resolution
    )
    
    processor.process_batch()


if __name__ == "__main__":
    main() 