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
        # Generate panoptic mask filename (no prefix)
        pan_mask_path = os.path.join(
            pan_mask_dir,
            f"{os.path.splitext(img['file_name'])[0]}_panoptic.png"
        )
        # Save panoptic mask as uint16 PNG
        success = cv2.imwrite(pan_mask_path, pan_mask.astype(np.uint16))
        if not success:
            print(f"Warning: Failed to save panoptic mask: {pan_mask_path}")
        panoptic_json["images"].append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": img["file_name"]
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
    
    print(f"\n✅ Conversion complete! Processed {splits_processed} annotation files.")
    print(f"Panoptic data saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 