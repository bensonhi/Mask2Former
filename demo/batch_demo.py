#!/usr/bin/env python3
"""
Batch Demo Script for Myotube Test Images (Instance Only)

Automatically runs demo.py on test images for specified stage.

Usage:
    cd demo
    python batch_demo.py --stage 1                       # Test Stage 1 instance on algorithmic test set
    python batch_demo.py --stage 2                       # Test Stage 2 instance on algorithmic or manual test set (default stage 2)
    python batch_demo.py --stage 2 --test-manual         # Test Stage 2 instance on manual test set
    python batch_demo.py --confidence-threshold 0.3      # Custom confidence threshold
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

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

def run_demo_on_image(image_name, image_id, images_dir, output_dir, config_file, model_weights, confidence_threshold=0.5):
    """Run demo.py on a single image."""
    
    # Construct paths
    input_path = os.path.join(images_dir, image_name)
    
    # Check if input image exists
    if not os.path.exists(input_path):
        print(f"  ⚠️  Image not found: {input_path}")
        return False
    
    # Create output directory (all images will use the same directory)
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct demo command
    cmd = [
        sys.executable, "demo.py",
        "--config-file", config_file,
        "--input", input_path,
        "--output", output_dir,
        "--confidence-threshold", str(confidence_threshold),
        "--opts", "MODEL.WEIGHTS", model_weights
    ]
    
    print(f"  🔄 Processing: {image_name}")
    
    try:
        # Run demo command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print(f"  ✅ Success: {input_path}")
            return True
        else:
            print(f"  ❌ Failed: {image_name}")
            print(f"     Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

def get_stage_config(stage, test_manual=False):
    """Get configuration for specified stage (instance only)."""
    if stage == 1:
        annotations_file = "../myotube_batch_output/annotations/manual_test_annotations.json" if test_manual else "../myotube_batch_output/annotations/algorithmic_test_annotations.json"
        output_suffix = "_manual" if test_manual else "_algorithmic"
        test_type = "Manual Test" if test_manual else "Algorithmic Test"
        config_file = "../stage1_config.yaml"
        model_weights = "../output_stage1_algorithmic/model_final.pth"
        output_dir = f"./batch_demo_output_stage1{output_suffix}"
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
        config_file = "../stage2_config.yaml"
        model_weights = "../output_stage2_manual/model_final.pth"
        output_dir = f"./batch_demo_output_stage2{output_suffix}"
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
    """Main batch demo function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Batch demo for myotube segmentation (instance only)")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=2,
                       help="Stage to test (1: algorithmic model, 2: manual fine-tuned model)")
    parser.add_argument("--test-manual", action="store_true",
                       help="Use manual test annotations instead of algorithmic test annotations")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Confidence threshold for predictions")
    
    args = parser.parse_args()
    
    # Get stage-specific configuration
    stage_config = get_stage_config(args.stage, args.test_manual)
    
    annotations_file = stage_config["annotations_file"]
    images_dir = "../myotube_batch_output/images"
    output_dir = stage_config["output_dir"]
    config_file = stage_config["config_file"]
    model_weights = stage_config["model_weights"]
    
    print(f"🎭 Batch Myotube Demo Processing - {stage_config['description']}")
    print("="*60)
    print(f"Stage:       {args.stage}")
    # Instance-only
    print(f"Test Set:    {'Manual' if args.test_manual else 'Algorithmic'}")
    print(f"Annotations: {annotations_file}")
    print(f"Images dir:  {images_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Config:      {config_file}")
    print(f"Model:       {model_weights}")
    print(f"Confidence:  {args.confidence_threshold}")
    print("="*60)
    
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
    successful = 0
    failed = 0
    
    for image_name, image_id in test_images:
        success = run_demo_on_image(
            image_name, image_id, images_dir, output_dir, 
            config_file, model_weights, args.confidence_threshold
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"BATCH DEMO COMPLETE - {stage_config['description']}")
    print("="*60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed:     {failed}")
    print(f"📁 Output:     {output_dir}")
    print(f"🎯 Stage:      {args.stage}")
    print(f"🎯 Mode:       {args.mode}")
    
    if successful > 0:
        print(f"\n🎯 Results saved to: {output_dir}")
        print(f"   All images in the same folder with unique filenames")
        print(f"   Model used: {stage_config['description']}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 
