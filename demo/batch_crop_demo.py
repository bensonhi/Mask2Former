#!/usr/bin/env python3
"""
Batch Crop Demo Script for Myotube Test Images

Automatically crops each test image into 4 regions and runs demo.py on each crop.
This allows comprehensive analysis of large myotube images by processing different regions.

Usage:
    cd demo
    python batch_crop_demo.py --stage 2                                  # Test Stage 2 instance on algorithmic test set
    python batch_crop_demo.py --stage 2 --mode panoptic                  # Test Stage 2 panoptic on algorithmic test set
    python batch_crop_demo.py --stage 2 --test-manual                    # Test Stage 2 instance on manual test set
    python batch_crop_demo.py --stage 2 --mode panoptic --test-manual    # Test Stage 2 panoptic on manual test set
    python batch_crop_demo.py --confidence-threshold 0.3                 # Custom confidence threshold
    python batch_crop_demo.py --crop-overlap 0.1                         # 10% overlap between crops
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np

def load_test_images(annotations_file):
    """Load image filenames from COCO annotations file."""
    try:
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        images = [(img['file_name'], img['id']) for img in data['images']]
        print(f"Found {len(images)} test images in {annotations_file}")
        return images
    
    except FileNotFoundError:
        print(f"❌ Annotations file not found: {annotations_file}")
        print("   Make sure to create algorithmic_test_annotations.json first")
        return []
    except Exception as e:
        print(f"❌ Error reading annotations: {e}")
        return []

def create_image_crops(image_path, overlap_ratio=0.0):
    """
    Create 4 crops from an image (2x2 grid).
    
    Args:
        image_path: Path to input image
        overlap_ratio: Overlap between adjacent crops (0.0 to 0.5)
    
    Returns:
        List of (crop_array, crop_name) tuples
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return []
        
        h, w = image.shape[:2]
        
        # Calculate crop size with overlap
        crop_h = h // 2
        crop_w = w // 2
        
        # Calculate overlap in pixels
        overlap_h = int(crop_h * overlap_ratio)
        overlap_w = int(crop_w * overlap_ratio)
        
        crops = []
        crop_positions = [
            ("top_left", 0, 0),
            ("top_right", 0, crop_w - overlap_w),
            ("bottom_left", crop_h - overlap_h, 0),
            ("bottom_right", crop_h - overlap_h, crop_w - overlap_w)
        ]
        
        for crop_name, start_y, start_x in crop_positions:
            # Calculate end coordinates
            end_y = min(start_y + crop_h + overlap_h, h)
            end_x = min(start_x + crop_w + overlap_w, w)
            
            # Adjust start coordinates if we hit the boundary
            start_y = max(0, end_y - crop_h - overlap_h)
            start_x = max(0, end_x - crop_w - overlap_w)
            
            # Extract crop
            crop = image[start_y:end_y, start_x:end_x]
            crops.append((crop, crop_name, (start_y, start_x, end_y, end_x)))
        
        print(f"  📐 Created 4 crops from {w}×{h} image (overlap: {overlap_ratio*100:.1f}%)")
        return crops
        
    except Exception as e:
        print(f"❌ Error creating crops: {e}")
        return []

def save_crop_to_temp(crop_array, temp_dir, original_name, crop_name):
    """Save a crop array to a temporary file."""
    # Create filename for crop
    base_name = Path(original_name).stem
    ext = Path(original_name).suffix
    crop_filename = f"{base_name}_{crop_name}{ext}"
    crop_path = os.path.join(temp_dir, crop_filename)
    
    # Save crop
    cv2.imwrite(crop_path, crop_array)
    return crop_path, crop_filename

def run_demo_on_crop(crop_path, crop_filename, output_dir, config_file, model_weights, confidence_threshold=0.5):
    """Run demo.py on a single crop."""
    
    # Check if input crop exists
    if not os.path.exists(crop_path):
        print(f"  ⚠️  Crop not found: {crop_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct demo command
    cmd = [
        sys.executable, "demo.py",
        "--config-file", config_file,
        "--input", crop_path,
        "--output", output_dir,
        "--confidence-threshold", str(confidence_threshold),
        "--opts", "MODEL.WEIGHTS", model_weights
    ]
    
    print(f"    🔄 Processing crop: {crop_filename}")
    
    try:
        # Run demo command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print(f"    ✅ Success: {crop_filename}")
            return True
        else:
            print(f"    ❌ Failed: {crop_filename}")
            print(f"       Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"    ❌ Exception: {e}")
        return False

def process_image_with_crops(image_name, image_id, images_dir, output_dir, config_file, model_weights, confidence_threshold=0.5, overlap_ratio=0.0):
    """Process a single image by cropping it into 4 regions and running demo on each."""
    
    # Construct paths
    input_path = os.path.join(images_dir, image_name)
    
    # Check if input image exists
    if not os.path.exists(input_path):
        print(f"  ⚠️  Image not found: {input_path}")
        return 0, 4
    
    print(f"  🖼️  Processing: {image_name}")
    
    # Create crops
    crops = create_image_crops(input_path, overlap_ratio)
    if not crops:
        return 0, 4
    
    # Create temporary directory for crops
    with tempfile.TemporaryDirectory() as temp_dir:
        successful_crops = 0
        total_crops = len(crops)
        
        for crop_array, crop_name, crop_coords in crops:
            # Save crop to temporary file
            crop_path, crop_filename = save_crop_to_temp(crop_array, temp_dir, image_name, crop_name)
            
            # Run demo on crop
            success = run_demo_on_crop(
                crop_path, crop_filename, output_dir, 
                config_file, model_weights, confidence_threshold
            )
            
            if success:
                successful_crops += 1
    
    failed_crops = total_crops - successful_crops
    print(f"  📊 Image summary: {successful_crops}/{total_crops} crops successful")
    
    return successful_crops, failed_crops

def get_stage_config(stage, test_manual=False, mode="instance"):
    """Get configuration for specified stage and mode."""
    if stage == 1:
        annotations_file = "../myotube_batch_output/annotations/manual_test_annotations.json" if test_manual else "../myotube_batch_output/annotations/algorithmic_test_annotations.json"
        output_suffix = "_manual" if test_manual else "_algorithmic"
        test_type = "Manual Test" if test_manual else "Algorithmic Test"
        
        if mode == "panoptic":
            config_file = "../stage1_panoptic_config.yaml"
            model_weights = "../output_stage1_panoptic_algorithmic/model_final.pth"
            output_dir = f"./batch_crop_demo_output_stage1_panoptic{output_suffix}"
            description = f"Stage 1 Panoptic (Algorithmic Model) on {test_type}"
        else:  # instance
            config_file = "../stage1_config.yaml"
            model_weights = "../output_stage1_algorithmic/model_final.pth"
            output_dir = f"./batch_crop_demo_output_stage1{output_suffix}"
            description = f"Stage 1 Instance (Algorithmic Model) on {test_type}"
        
        return {
            "annotations_file": annotations_file,
            "config_file": config_file,
            "model_weights": model_weights,
            "output_dir": output_dir,
            "description": description
        }
    elif stage == 2:
        annotations_file = "../myotube_batch_output/annotations/manual_test_annotations.json" if test_manual else "../myotube_batch_output/annotations/algorithmic_test_annotations.json"
        output_suffix = "_manual" if test_manual else "_algorithmic"
        test_type = "Manual Test" if test_manual else "Algorithmic Test"
        
        if mode == "panoptic":
            config_file = "../stage2_panoptic_config.yaml"
            model_weights = "../output_stage2_panoptic_manual/model_final.pth"
            output_dir = f"./batch_crop_demo_output_stage2_panoptic{output_suffix}"
            description = f"Stage 2 Panoptic (Manual Fine-tuned Model) on {test_type}"
        else:  # instance
            config_file = "../stage2_config.yaml"
            model_weights = "../output_stage2_manual/model_final.pth"
            output_dir = f"./batch_crop_demo_output_stage2{output_suffix}"
            description = f"Stage 2 Instance (Manual Fine-tuned Model) on {test_type}"
        
        return {
            "annotations_file": annotations_file,
            "config_file": config_file,
            "model_weights": model_weights,
            "output_dir": output_dir,
            "description": description
        }
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")

def main():
    """Main batch crop demo function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Batch crop demo for myotube segmentation")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=2,
                       help="Stage to test (1: algorithmic model, 2: manual fine-tuned model)")
    parser.add_argument("--mode", choices=["instance", "panoptic"], default="instance",
                       help="Segmentation mode (instance or panoptic)")
    parser.add_argument("--test-manual", action="store_true",
                       help="Use manual test annotations instead of algorithmic test annotations")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Confidence threshold for predictions")
    parser.add_argument("--crop-overlap", type=float, default=0.0,
                       help="Overlap ratio between adjacent crops (0.0 to 0.5)")
    
    args = parser.parse_args()
    
    # Validate overlap ratio
    if not (0.0 <= args.crop_overlap <= 0.5):
        print("❌ Crop overlap must be between 0.0 and 0.5")
        return 1
    
    # Get stage-specific configuration
    stage_config = get_stage_config(args.stage, args.test_manual, args.mode)
    
    annotations_file = stage_config["annotations_file"]
    images_dir = "../myotube_batch_output/images"
    output_dir = stage_config["output_dir"]
    config_file = stage_config["config_file"]
    model_weights = stage_config["model_weights"]
    
    print(f"🎭 Batch Crop Demo Processing - {stage_config['description']}")
    print("="*70)
    print(f"Stage:        {args.stage}")
    print(f"Mode:         {args.mode}")
    print(f"Test Set:     {'Manual' if args.test_manual else 'Algorithmic'}")
    print(f"Crop Strategy: 4 crops per image (2×2 grid)")
    print(f"Crop Overlap: {args.crop_overlap*100:.1f}%")
    print(f"Annotations:  {annotations_file}")
    print(f"Images dir:   {images_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Config:       {config_file}")
    print(f"Model:        {model_weights}")
    print(f"Confidence:   {args.confidence_threshold}")
    print("="*70)
    
    # Verify required files exist
    required_files = [config_file, model_weights]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file missing: {file_path}")
            return 1
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory missing: {images_dir}")
        return 1
    
    # Load test images
    test_images = load_test_images(annotations_file)
    if not test_images:
        return 1
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    total_successful_crops = 0
    total_failed_crops = 0
    successful_images = 0
    failed_images = 0
    
    for image_name, image_id in test_images:
        successful_crops, failed_crops = process_image_with_crops(
            image_name, image_id, images_dir, output_dir, 
            config_file, model_weights, args.confidence_threshold, args.crop_overlap
        )
        
        total_successful_crops += successful_crops
        total_failed_crops += failed_crops
        
        if successful_crops > 0:
            successful_images += 1
        if failed_crops > 0:
            failed_images += 1
    
    # Calculate statistics
    total_crops = total_successful_crops + total_failed_crops
    total_images = len(test_images)
    success_rate = (total_successful_crops / total_crops * 100) if total_crops > 0 else 0
    
    # Summary
    print("\n" + "="*70)
    print(f"BATCH CROP DEMO COMPLETE - {stage_config['description']}")
    print("="*70)
    print(f"📊 CROP STATISTICS:")
    print(f"   ✅ Successful crops: {total_successful_crops}")
    print(f"   ❌ Failed crops:     {total_failed_crops}")
    print(f"   📈 Success rate:     {success_rate:.1f}%")
    print(f"")
    print(f"📊 IMAGE STATISTICS:")
    print(f"   🖼️  Total images:     {total_images}")
    print(f"   ✅ Images with success: {successful_images}")
    print(f"   ❌ Images with failures: {failed_images}")
    print(f"   📈 Expected crops:   {total_images * 4}")
    print(f"")
    print(f"📁 Output:     {output_dir}")
    print(f"🎯 Stage:      {args.stage}")
    print(f"🎯 Mode:       {args.mode}")
    print(f"📐 Crops:      4 per image (2×2 grid)")
    
    if total_successful_crops > 0:
        print(f"\n🎯 Results saved to: {output_dir}")
        print(f"   Each crop saved with suffix: _top_left, _top_right, _bottom_left, _bottom_right")
        print(f"   Model used: {stage_config['description']}")
        
        # Give guidance on finding results
        print(f"\n💡 To find results for a specific image:")
        print(f"   Look for files named: [image_name]_[crop_position].[ext]")
        print(f"   Example: image001_top_left.jpg, image001_top_right.jpg, etc.")
    
    return 0 if total_failed_crops == 0 else 1

if __name__ == "__main__":
    sys.exit(main())