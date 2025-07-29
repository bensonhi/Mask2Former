#!/usr/bin/env python3
"""
Debug script to verify panoptic data format and compatibility.
Checks that masks and JSON files match the expected format for evaluation.
"""

import os
import json
import cv2
import numpy as np
from panopticapi.utils import rgb2id

def check_panoptic_data(panoptic_dir, split_name="algorithmic_train"):
    """Check panoptic data format for a specific split."""
    
    print(f"🔍 Checking panoptic data for {split_name}...")
    
    # Check JSON file
    json_path = os.path.join(panoptic_dir, f"{split_name}_panoptic.json")
    mask_dir = os.path.join(panoptic_dir, f"{split_name}_panoptic_masks")
    
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return False
        
    if not os.path.exists(mask_dir):
        print(f"❌ Mask directory not found: {mask_dir}")
        return False
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"📊 JSON Statistics:")
    print(f"   Images: {len(data['images'])}")
    print(f"   Annotations: {len(data['annotations'])}")
    print(f"   Categories: {len(data['categories'])}")
    
    # Check categories
    print(f"📂 Categories:")
    for cat in data['categories']:
        print(f"   ID {cat['id']}: {cat['name']} (isthing: {cat.get('isthing', 'N/A')})")
    
    # Check first few masks
    total_segments = 0
    valid_masks = 0
    
    for i, ann in enumerate(data['annotations'][:3]):  # Check first 3
        mask_file = ann['file_name']
        mask_path = os.path.join(mask_dir, mask_file)
        
        print(f"\n🖼️ Checking mask {i+1}: {mask_file}")
        
        if not os.path.exists(mask_path):
            print(f"   ❌ Mask file not found: {mask_path}")
            continue
        
        # Load mask
        mask_img = cv2.imread(mask_path)
        if mask_img is None:
            print(f"   ❌ Could not load mask: {mask_path}")
            continue
            
        print(f"   📏 Mask shape: {mask_img.shape}")
        print(f"   📊 Mask dtype: {mask_img.dtype}")
        
        # Check if it's RGB format
        if len(mask_img.shape) == 3 and mask_img.shape[2] == 3:
            print(f"   ✅ RGB format detected")
            
            # Convert RGB to IDs and check segments
            try:
                mask_ids = rgb2id(mask_img)
                unique_ids = np.unique(mask_ids)
                print(f"   🎯 Unique segment IDs: {unique_ids}")
                
                # Check segments_info
                segments_info = ann['segments_info']
                print(f"   📋 segments_info count: {len(segments_info)}")
                
                json_ids = set(seg['id'] for seg in segments_info)
                mask_nonzero_ids = set(unique_ids[unique_ids > 0])  # Exclude background
                
                print(f"   📝 JSON segment IDs: {json_ids}")
                print(f"   🖼️ Mask segment IDs: {mask_nonzero_ids}")
                
                if json_ids == mask_nonzero_ids:
                    print(f"   ✅ Segment IDs match between JSON and mask")
                    valid_masks += 1
                else:
                    print(f"   ❌ Segment ID mismatch!")
                    print(f"      Missing in mask: {json_ids - mask_nonzero_ids}")
                    print(f"      Extra in mask: {mask_nonzero_ids - json_ids}")
                
                total_segments += len(mask_nonzero_ids)
                
            except Exception as e:
                print(f"   ❌ Error processing mask: {e}")
        else:
            print(f"   ❌ Not RGB format!")
    
    print(f"\n📊 Summary for {split_name}:")
    print(f"   Valid masks: {valid_masks}/{min(3, len(data['annotations']))}")
    print(f"   Total segments found: {total_segments}")
    
    return valid_masks > 0 and total_segments > 0

def main():
    panoptic_dir = "myotube_batch_output/panoptic"
    
    print("🔍 Debugging Panoptic Data Format")
    print("=" * 50)
    
    if not os.path.exists(panoptic_dir):
        print(f"❌ Panoptic directory not found: {panoptic_dir}")
        print("💡 Run utils/convert_instance_to_panoptic.py first")
        return
    
    # Check both train and test splits
    splits = ["algorithmic_train", "algorithmic_test"]
    
    all_valid = True
    for split in splits:
        valid = check_panoptic_data(panoptic_dir, split)
        all_valid = all_valid and valid
        print()
    
    if all_valid:
        print("✅ All panoptic data appears to be correctly formatted!")
    else:
        print("❌ Issues found in panoptic data format.")
        print("💡 Regenerate panoptic data with the fixed conversion script.")

if __name__ == "__main__":
    main() 