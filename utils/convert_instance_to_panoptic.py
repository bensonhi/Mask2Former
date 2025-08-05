#!/usr/bin/env python3
"""
Convert COCO instance segmentation dataset (train and test/val) to COCO panoptic format for myotube segmentation.
- For each image, generates a PNG mask with unique segment IDs for each myotube.
- Produces panoptic JSONs for both train and test/val splits.

Usage:
    python convert_instance_to_panoptic.py --input_dir myotube_batch_output/annotations --image_dir myotube_batch_output/images --output_dir myotube_batch_output/panoptic
"""
import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools import mask as maskUtils
import argparse
from collections import defaultdict

def convert_instance_to_panoptic(instance_json, image_dir, pan_mask_dir, panoptic_json_path):
    # Validate input files
    if not os.path.exists(instance_json):
        print(f"Error: Instance JSON not found: {instance_json}")
        return
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return
        
    os.makedirs(pan_mask_dir, exist_ok=True)
    
    try:
        with open(instance_json, "r") as f:
            coco = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file {instance_json}: {e}")
        return
    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]
    anns_per_image = defaultdict(list)
    for ann in annotations:
        anns_per_image[ann["image_id"]].append(ann)
    panoptic_json = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "myotube",
                "supercategory": "cell",
                "isthing": 1,
                "color": [255, 0, 0]
            },
            {
                "id": 0,
                "name": "background",
                "supercategory": "background", 
                "isthing": 0,
                "color": [0, 0, 0]
            }
        ]
    }
    for img_id, img in tqdm(images.items(), desc=f"Processing {os.path.basename(instance_json)}"):
        height, width = img["height"], img["width"]
        pan_mask = np.zeros((height, width), dtype=np.uint16)
        segments_info = []
        ann_list = anns_per_image[img_id]
        segm_id = 1  # Start from 1, 0 is background
        for ann in ann_list:
            if "segmentation" in ann:
                rle = maskUtils.frPyObjects(ann["segmentation"], height, width)
                mask = maskUtils.decode(rle)
                if mask.ndim == 3:
                    mask = np.any(mask, axis=2)
                mask = mask.astype(np.bool_)
            else:
                continue
            # Check for overlaps with existing segments
            overlap_region = (pan_mask > 0) & mask
            if np.any(overlap_region):
                print(f"Warning: Segment {segm_id} overlaps with existing segments in image {img_id}")
                # Only assign to non-overlapping regions
                mask = mask & (pan_mask == 0)
            # Skip if mask becomes empty after overlap removal
            if not np.any(mask):
                print(f"  Skipping segment {segm_id}: empty after overlap removal")
                continue
                
            pan_mask[mask] = segm_id
            area = int(np.sum(mask))
            y_indices, x_indices = np.where(mask)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue
            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
            bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
            segments_info.append({
                "id": segm_id,
                "category_id": 1,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            segm_id += 1
        
        # Add background segment (category 0) for stuff evaluation
        background_area = int(np.sum(pan_mask == 0))
        if background_area > 0:
            segments_info.append({
                "id": 0,  # Background segment ID
                "category_id": 0,  # Background category
                "area": background_area,
                "bbox": [0, 0, width, height],  # Full image bbox for background
                "iscrowd": 0
            })
        # Generate panoptic mask filename (same base name as image, different extension)
        # Detectron2 expects image.jpg and mask.png to have the same base filename
        pan_mask_path = os.path.join(
            pan_mask_dir,
            f"{os.path.splitext(img['file_name'])[0]}.png"
        )
        # Convert panoptic mask to RGB format for panopticapi compatibility
        from panopticapi.utils import id2rgb
        
        # DEBUG: Test the id2rgb function
        test_rgb = id2rgb(1)
        print(f"DEBUG: id2rgb(1) returns {test_rgb}")
        
        # Create RGB mask (H, W, 3) uint8 format
        pan_mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        
        # First, mark all pixels as background (category 0)
        background_mask = pan_mask == 0
        pan_mask_rgb[background_mask] = id2rgb(0)  # Background = black (0,0,0)
        
        # Then, assign myotube segments
        for segm_id in np.unique(pan_mask):
            if segm_id == 0:
                continue  # background already handled
            mask = pan_mask == segm_id
            pan_mask_rgb[mask] = id2rgb(segm_id)
        
        # DEBUG: Check RGB values before saving
        print(f"DEBUG: Before save - RGB at [100,100]: {pan_mask_rgb[100,100]}")
        print(f"DEBUG: Before save - unique R values: {np.unique(pan_mask_rgb[:,:,0])[:10]}")
        print(f"DEBUG: Before save - unique G values: {np.unique(pan_mask_rgb[:,:,1])[:10]}")
        print(f"DEBUG: Before save - unique B values: {np.unique(pan_mask_rgb[:,:,2])[:10]}")
        
        # Save panoptic mask as RGB PNG
        success = cv2.imwrite(pan_mask_path, pan_mask_rgb)
        if not success:
            print(f"Warning: Failed to save panoptic mask: {pan_mask_path}")
        else:
            print(f"  ✓ Saved RGB panoptic mask: {os.path.basename(pan_mask_path)} ({np.unique(pan_mask).max()} segments)")
            
            # DEBUG: Check what was actually saved
            loaded_check = cv2.imread(pan_mask_path)
            print(f"DEBUG: After load - RGB at [100,100]: {loaded_check[100,100]}")
            print(f"DEBUG: After load - unique R: {np.unique(loaded_check[:,:,0])[:10]}")
            print(f"DEBUG: After load - unique G: {np.unique(loaded_check[:,:,1])[:10]}")
            print(f"DEBUG: After load - unique B: {np.unique(loaded_check[:,:,2])[:10]}")
        # Convert PNG file extension to JPG for Detectron2 compatibility
        img_file_name = img["file_name"]
        if img_file_name.endswith('.png'):
            img_file_name = img_file_name.replace('.png', '.jpg')
        
        panoptic_json["images"].append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": img_file_name
        })
        panoptic_json["annotations"].append({
            "image_id": img_id,
            "file_name": os.path.basename(pan_mask_path),
            "segments_info": segments_info
        })
    with open(panoptic_json_path, "w") as f:
        json.dump(panoptic_json, f, indent=2)
    
    # Print conversion statistics
    total_segments = sum(len(ann["segments_info"]) for ann in panoptic_json["annotations"])
    print(f"Panoptic conversion complete: {panoptic_json_path}")
    print(f"  Images: {len(panoptic_json['images'])}")
    print(f"  Total segments: {total_segments}")
    print(f"  Avg segments per image: {total_segments/len(panoptic_json['images']):.1f}")
    print(f"  Panoptic masks saved to: {pan_mask_dir}")


def update_instance_json_extensions(input_dir):
    """
    Update instance JSON files to reference JPG instead of PNG files.
    
    Args:
        input_dir: Directory containing instance annotation JSON files
    """
    json_files = [
        "algorithmic_train_annotations.json", "algorithmic_test_annotations.json",
        "manual_train_annotations.json", "manual_test_annotations.json"
    ]
    
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        if not os.path.exists(json_path):
            continue
            
        print(f"   📝 Updating {json_file} to reference JPG files...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Update image file_name fields from .png to .jpg
        updated_count = 0
        for image in data.get('images', []):
            if image['file_name'].endswith('.png'):
                image['file_name'] = image['file_name'].replace('.png', '.jpg')
                updated_count += 1
        
        # Save the updated JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"      ✅ Updated {updated_count} image references")


def rename_png_to_jpg(image_dir):
    """
    Rename all PNG image files to JPG for Detectron2 panoptic compatibility.
    
    Args:
        image_dir: Directory containing the images
    """
    import shutil
    
    png_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    if not png_files:
        print(f"No PNG files found in {image_dir}")
        return
        
    print(f"🔄 Renaming {len(png_files)} PNG files to JPG in {image_dir}")
    
    for png_file in png_files:
        png_path = os.path.join(image_dir, png_file)
        jpg_file = png_file.replace('.png', '.jpg')
        jpg_path = os.path.join(image_dir, jpg_file)
        
        # Simply rename the file (don't convert format)
        shutil.move(png_path, jpg_path)
        print(f"   📁 {png_file} → {jpg_file}")
    
    print(f"✅ Renamed {len(png_files)} files from PNG to JPG")


def main():
    parser = argparse.ArgumentParser(description="Convert COCO instance to panoptic format for myotube task.")
    parser.add_argument("--input_dir", default="myotube_batch_output/annotations", help="Directory with instance annotation JSONs (default: myotube_batch_output/annotations)")
    parser.add_argument("--image_dir", default="myotube_batch_output/images", help="Directory with images (default: myotube_batch_output/images)")
    parser.add_argument("--output_dir", default="myotube_batch_output/panoptic", help="Directory to save panoptic masks and JSONs (default: myotube_batch_output/panoptic)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Process all relevant splits for two-stage training
    splits_processed = 0
    for split in [
        "algorithmic_train_annotations.json", "algorithmic_test_annotations.json",
        "manual_train_annotations.json", "manual_test_annotations.json"
    ]:
        instance_json = os.path.join(args.input_dir, split)
        if not os.path.exists(instance_json):
            print(f"Skipping {split}: file not found")
            continue
            
        print(f"\nProcessing {split}...")
        pan_mask_dir = os.path.join(args.output_dir, split.replace("_annotations.json", "_panoptic_masks"))
        panoptic_json_path = os.path.join(args.output_dir, split.replace("_annotations.json", "_panoptic.json"))
        convert_instance_to_panoptic(instance_json, args.image_dir, pan_mask_dir, panoptic_json_path)
        splits_processed += 1
    
    # Update instance JSON files and rename images for Detectron2 compatibility
    if splits_processed > 0:
        print(f"\n🔄 Converting image file extensions for Detectron2 compatibility...")
        update_instance_json_extensions(args.input_dir)
        rename_png_to_jpg(args.image_dir)
    
    print(f"\n✅ Conversion complete! Processed {splits_processed} annotation files.")
    print(f"Panoptic data saved to: {args.output_dir}")
    print(f"✅ All JSON files updated and images renamed for Detectron2 compatibility")
    print(f"✅ Both instance and panoptic training will now work with JPG files")

if __name__ == "__main__":
    main() 