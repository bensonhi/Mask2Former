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
    os.makedirs(pan_mask_dir, exist_ok=True)
    with open(instance_json, "r") as f:
        coco = json.load(f)
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
        # Add prefix to mask file name based on split type
        if "manual" in os.path.basename(instance_json):
            prefix = "manual_"
        elif "algorithmic" in os.path.basename(instance_json):
            prefix = "algorithmic_"
        else:
            prefix = ""
        pan_mask_path = os.path.join(
            pan_mask_dir,
            f"{prefix}{os.path.splitext(img['file_name'])[0]}_panoptic.png"
        )
        cv2.imwrite(pan_mask_path, pan_mask.astype(np.uint16))
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
    print(f"Panoptic conversion complete: {panoptic_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert COCO instance to panoptic format for myotube task.")
    parser.add_argument("--input_dir", default="myotube_batch_output/annotations", help="Directory with instance annotation JSONs (default: myotube_batch_output/annotations)")
    parser.add_argument("--image_dir", default="myotube_batch_output/images", help="Directory with images (default: myotube_batch_output/images)")
    parser.add_argument("--output_dir", default="myotube_batch_output/panoptic", help="Directory to save panoptic masks and JSONs (default: myotube_batch_output/panoptic)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Process all relevant splits for two-stage training
    for split in [
        "algorithmic_train_annotations.json", "algorithmic_test_annotations.json",
        "manual_train_annotations.json", "manual_test_annotations.json"
    ]:
        instance_json = os.path.join(args.input_dir, split)
        if not os.path.exists(instance_json):
            continue
        pan_mask_dir = os.path.join(args.output_dir, split.replace("_annotations.json", "_panoptic_masks"))
        panoptic_json_path = os.path.join(args.output_dir, split.replace("_annotations.json", "_panoptic.json"))
        convert_instance_to_panoptic(instance_json, args.image_dir, pan_mask_dir, panoptic_json_path)

if __name__ == "__main__":
    main() 