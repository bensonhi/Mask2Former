#!/usr/bin/env python3
"""
Scale COCO Instance Annotations to Match Actual Image Resolutions

This script:
1. Reads a COCO annotation JSON file
2. For each image, gets the actual image dimensions from the file
3. Scales all annotations (bboxes, segmentations, areas) to match actual resolution
4. Saves the scaled annotation file

Usage:
    python scale_coco_annotations.py --input annotations.json --image_dir images/ --output scaled_annotations.json
"""

import os
import json
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

def find_image_file(image_dir, base_filename):
    """
    Find image file with various extensions.
    
    Args:
        image_dir: Directory to search in
        base_filename: Base filename from annotation
        
    Returns:
        Full path to found image file or None
    """
    # Try the exact filename first
    exact_path = os.path.join(image_dir, base_filename)
    if os.path.exists(exact_path):
        return exact_path
    
    # Try different extensions
    base_name = os.path.splitext(base_filename)[0]
    extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    
    for ext in extensions:
        test_path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(test_path):
            return test_path
    
    return None

def get_image_dimensions(image_path):
    """
    Get actual image dimensions from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        (width, height) tuple or None if image can't be loaded
    """
    try:
        # Try PIL first (faster for just getting dimensions)
        with Image.open(image_path) as img:
            return img.size  # PIL returns (width, height)
    except Exception:
        try:
            # Fallback to OpenCV
            img = cv2.imread(image_path)
            if img is not None:
                height, width = img.shape[:2]
                return (width, height)
        except Exception:
            pass
    
    return None

def scale_bbox(bbox, scale_x, scale_y):
    """
    Scale a COCO bbox [x, y, width, height].
    
    Args:
        bbox: [x, y, width, height] in original coordinates
        scale_x: X scaling factor
        scale_y: Y scaling factor
        
    Returns:
        Scaled bbox [x, y, width, height]
    """
    x, y, w, h = bbox
    return [
        x * scale_x,
        y * scale_y, 
        w * scale_x,
        h * scale_y
    ]

def scale_segmentation(segmentation, scale_x, scale_y):
    """
    Scale COCO segmentation polygons.
    
    Args:
        segmentation: List of polygons, each polygon is [x1,y1,x2,y2,...]
        scale_x: X scaling factor
        scale_y: Y scaling factor
        
    Returns:
        Scaled segmentation polygons
    """
    scaled_segmentation = []
    
    for polygon in segmentation:
        if len(polygon) < 6:  # Need at least 3 points (6 coordinates)
            continue
            
        scaled_polygon = []
        for i in range(0, len(polygon), 2):
            if i + 1 < len(polygon):
                x = polygon[i] * scale_x
                y = polygon[i + 1] * scale_y
                scaled_polygon.extend([x, y])
        
        if len(scaled_polygon) >= 6:
            scaled_segmentation.append(scaled_polygon)
    
    return scaled_segmentation

def scale_coco_annotations(input_json, image_dir, output_json):
    """
    Scale COCO annotations to match actual image resolutions.
    
    Args:
        input_json: Path to input COCO annotation file
        image_dir: Directory containing the images
        output_json: Path to save scaled annotation file
    """
    
    print(f"📖 Loading COCO annotations from: {input_json}")
    
    # Load COCO annotation file
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
    
    print(f"📊 Found {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    
    # Create lookup for annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Process each image
    scaled_images = []
    scaled_annotations = []
    skipped_images = 0
    total_scaled_annotations = 0
    
    print(f"🔍 Processing images and scaling annotations...")
    
    for img_info in tqdm(coco_data['images'], desc="Scaling annotations"):
        image_id = img_info['id']
        filename = img_info['file_name']
        
        # Get original dimensions from JSON
        orig_width = img_info['width']
        orig_height = img_info['height']
        
        # Find actual image file (try different extensions)
        image_path = find_image_file(image_dir, filename)
        
        if image_path is None:
            print(f"⚠️  Warning: Image not found: {filename} (tried various extensions)")
            # Show available files for debugging
            available_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'))][:5]
            print(f"   Available image files (first 5): {available_files}")
            skipped_images += 1
            continue
        
        # Get actual image dimensions
        actual_dims = get_image_dimensions(image_path)
        if actual_dims is None:
            print(f"⚠️  Warning: Could not read image dimensions: {image_path}")
            skipped_images += 1
            continue
        
        actual_width, actual_height = actual_dims
        
        # Calculate scaling factors
        scale_x = actual_width / orig_width
        scale_y = actual_height / orig_height
        
        # Update image info with actual dimensions
        scaled_img_info = img_info.copy()
        scaled_img_info['width'] = actual_width
        scaled_img_info['height'] = actual_height
        scaled_images.append(scaled_img_info)
        
        # Scale annotations for this image
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                scaled_ann = ann.copy()
                
                # Scale bbox
                if 'bbox' in ann:
                    scaled_ann['bbox'] = scale_bbox(ann['bbox'], scale_x, scale_y)
                
                # Scale segmentation
                if 'segmentation' in ann and ann['segmentation']:
                    scaled_ann['segmentation'] = scale_segmentation(ann['segmentation'], scale_x, scale_y)
                
                # Scale area (area scales by scale_x * scale_y)
                if 'area' in ann:
                    scaled_ann['area'] = ann['area'] * scale_x * scale_y
                
                scaled_annotations.append(scaled_ann)
                total_scaled_annotations += 1
        
        # Print progress for significant scaling
        if abs(scale_x - 1.0) > 0.1 or abs(scale_y - 1.0) > 0.1:
            print(f"   📏 {filename}: {orig_width}×{orig_height} → {actual_width}×{actual_height} "
                  f"(scale: {scale_x:.2f}×{scale_y:.2f})")
    
    # Create scaled COCO data
    scaled_coco_data = {
        "info": coco_data.get("info", {}).copy(),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data.get("categories", []),
        "images": scaled_images,
        "annotations": scaled_annotations
    }
    
    # Update info section
    if "info" not in scaled_coco_data:
        scaled_coco_data["info"] = {}
    
    scaled_coco_data["info"]["description"] = scaled_coco_data["info"].get("description", "") + " [SCALED TO ACTUAL IMAGE RESOLUTIONS]"
    scaled_coco_data["info"]["date_created"] = scaled_coco_data["info"].get("date_created", "")
    
    # Save scaled annotations (skip if dry run)
    if output_json is not None:
        print(f"💾 Saving scaled annotations to: {output_json}")
        
        with open(output_json, 'w') as f:
            json.dump(scaled_coco_data, f, indent=2)
    else:
        print(f"🔍 DRY RUN: Would save scaled annotations")
    
    # Print summary
    print(f"\n✅ Scaling complete!")
    print(f"   📊 Processed images: {len(scaled_images)}")
    print(f"   📊 Skipped images: {skipped_images}")
    print(f"   📊 Scaled annotations: {total_scaled_annotations}")
    print(f"   📁 Output saved to: {output_json}")
    
    return scaled_coco_data

def main():
    parser = argparse.ArgumentParser(description="Scale COCO annotations to match actual image resolutions")
    parser.add_argument("--input", "-i", required=True, help="Input COCO annotation JSON file")
    parser.add_argument("--image_dir", "-d", required=True, help="Directory containing the images")
    parser.add_argument("--output", "-o", required=True, help="Output path for scaled annotation JSON file")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be scaled without saving")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"❌ Error: Input annotation file not found: {args.input}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"❌ Error: Image directory not found: {args.image_dir}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    
    # Run scaling
    try:
        scaled_data = scale_coco_annotations(args.input, args.image_dir, args.output if not args.dry_run else None)
        
        if args.dry_run:
            print(f"\n🔍 Dry run complete. Would have saved {len(scaled_data['annotations'])} scaled annotations.")
        
    except Exception as e:
        print(f"❌ Error during scaling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 