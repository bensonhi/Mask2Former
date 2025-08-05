#!/usr/bin/env python3

import os
import json
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt

def check_panoptic_annotations():
    """Check the quality of panoptic ground truth annotations."""
    
    print("🔍 Checking panoptic ground truth quality...")
    
    # Check panoptic annotations
    panoptic_dir = "myotube_batch_output/panoptic"
    
    # Stage 1 (algorithmic) annotations
    stage1_json = os.path.join(panoptic_dir, "algorithmic_train_panoptic.json")
    stage1_masks_dir = os.path.join(panoptic_dir, "algorithmic_train_panoptic_masks")
    
    if not os.path.exists(stage1_json):
        print(f"❌ Stage 1 JSON not found: {stage1_json}")
        return
    
    if not os.path.exists(stage1_masks_dir):
        print(f"❌ Stage 1 masks dir not found: {stage1_masks_dir}")
        return
    
    print(f"✅ Found panoptic annotations")
    
    # Load JSON
    with open(stage1_json, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Panoptic JSON stats:")
    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    print(f"  Categories: {len(data['categories'])}")
    
    # Check categories
    print(f"\n📋 Categories:")
    for cat in data['categories']:
        print(f"  ID {cat['id']}: {cat['name']} (isthing: {cat.get('isthing', 'N/A')})")
    
    # Analyze annotations
    print(f"\n🔍 Analyzing annotations...")
    
    total_segments = 0
    myotube_segments = 0
    background_segments = 0
    empty_annotations = 0
    
    # Check first few annotations in detail
    for i, ann in enumerate(data['annotations'][:5]):
        image_id = ann['image_id']
        mask_file = ann['file_name']
        segments_info = ann['segments_info']
        
        print(f"\n📄 Annotation {i+1} (Image ID: {image_id}):")
        print(f"  Mask file: {mask_file}")
        print(f"  Segments: {len(segments_info)}")
        
        if len(segments_info) == 0:
            empty_annotations += 1
            print("  ⚠️  No segments in this annotation!")
            continue
        
        # Load the actual mask
        mask_path = os.path.join(stage1_masks_dir, mask_file)
        if os.path.exists(mask_path):
            # Try both PNG and RGB formats
            try:
                mask = np.array(Image.open(mask_path))
                if len(mask.shape) == 3:
                    # RGB format - convert to ID
                    from panopticapi.utils import rgb2id
                    mask = rgb2id(mask)
                
                unique_vals = np.unique(mask)
                print(f"  🎭 Mask shape: {mask.shape}")
                print(f"  🎭 Unique values in mask: {unique_vals}")
                print(f"  🎭 Value counts: {[(val, np.sum(mask == val)) for val in unique_vals]}")
                
                # Check if segments match mask
                for seg in segments_info:
                    seg_id = seg['id']
                    cat_id = seg['category_id']
                    area = seg['area']
                    
                    if seg_id in unique_vals:
                        actual_area = np.sum(mask == seg_id)
                        print(f"    Segment {seg_id}: cat={cat_id}, area={area}, actual_area={actual_area}")
                        
                        if cat_id == 1:  # myotube
                            myotube_segments += 1
                        elif cat_id == 0:  # background
                            background_segments += 1
                    else:
                        print(f"    ⚠️  Segment {seg_id} not found in mask!")
                
                total_segments += len(segments_info)
                
            except Exception as e:
                print(f"  ❌ Error loading mask: {e}")
        else:
            print(f"  ❌ Mask file not found: {mask_path}")
    
    # Overall statistics
    print(f"\n📊 Overall Statistics:")
    print(f"  Total annotations checked: {min(5, len(data['annotations']))}")
    print(f"  Empty annotations: {empty_annotations}")
    print(f"  Total segments: {total_segments}")
    print(f"  Myotube segments (cat_id=1): {myotube_segments}")
    print(f"  Background segments (cat_id=0): {background_segments}")
    
    # Check class balance across all annotations
    print(f"\n🔍 Checking class balance across all annotations...")
    
    all_myotube_pixels = 0
    all_background_pixels = 0
    processed_masks = 0
    
    for ann in data['annotations'][:10]:  # Check first 10
        mask_file = ann['file_name']
        mask_path = os.path.join(stage1_masks_dir, mask_file)
        
        if os.path.exists(mask_path):
            try:
                mask = np.array(Image.open(mask_path))
                if len(mask.shape) == 3:
                    from panopticapi.utils import rgb2id
                    mask = rgb2id(mask)
                
                # Count pixels for each segment
                for seg in ann['segments_info']:
                    seg_id = seg['id']
                    cat_id = seg['category_id']
                    
                    pixel_count = np.sum(mask == seg_id)
                    
                    if cat_id == 1:  # myotube
                        all_myotube_pixels += pixel_count
                    elif cat_id == 0:  # background
                        all_background_pixels += pixel_count
                
                processed_masks += 1
                
            except Exception as e:
                print(f"Error processing {mask_file}: {e}")
    
    total_pixels = all_myotube_pixels + all_background_pixels
    
    print(f"\n📈 Class Balance (first {processed_masks} masks):")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Myotube pixels: {all_myotube_pixels:,} ({100*all_myotube_pixels/total_pixels:.2f}%)")
    print(f"  Background pixels: {all_background_pixels:,} ({100*all_background_pixels/total_pixels:.2f}%)")
    
    # Check if class imbalance is severe
    if total_pixels > 0:
        myotube_ratio = all_myotube_pixels / total_pixels
        if myotube_ratio < 0.01:
            print(f"⚠️  SEVERE CLASS IMBALANCE: Only {myotube_ratio*100:.3f}% myotube pixels!")
            print(f"   This could cause the model to predict only background.")
        elif myotube_ratio < 0.05:
            print(f"⚠️  MODERATE CLASS IMBALANCE: {myotube_ratio*100:.2f}% myotube pixels")
        else:
            print(f"✅ Reasonable class balance: {myotube_ratio*100:.2f}% myotube pixels")

if __name__ == "__main__":
    check_panoptic_annotations()