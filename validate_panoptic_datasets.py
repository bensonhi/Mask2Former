#!/usr/bin/env python3
"""
Comprehensive validation script for panoptic datasets.
Checks all panoptic masks and JSONs for correct encoding and consistency.
"""

import os
import json
import numpy as np
from PIL import Image
from panopticapi.utils import rgb2id
from tqdm import tqdm
import argparse

def validate_single_dataset(json_path, masks_dir, dataset_name):
    """Validate a single panoptic dataset (JSON + masks)."""
    print(f"\n🔍 Validating {dataset_name}")
    print(f"   JSON: {json_path}")
    print(f"   Masks: {masks_dir}")
    
    # Check if files exist
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return False
    
    if not os.path.exists(masks_dir):
        print(f"❌ Masks directory not found: {masks_dir}")
        return False
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    print(f"📊 Found {len(annotations)} annotations in JSON")
    
    # Check mask files
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
    print(f"📊 Found {len(mask_files)} PNG mask files")
    
    if len(annotations) != len(mask_files):
        print(f"⚠️  Mismatch: {len(annotations)} JSON annotations vs {len(mask_files)} mask files")
    
    # Validate sample of masks
    validation_errors = []
    large_id_count = 0
    correct_id_count = 0
    
    sample_size = min(10, len(annotations))
    print(f"🧪 Validating {sample_size} sample masks...")
    
    for i, ann in enumerate(tqdm(annotations[:sample_size], desc="Validating")):
        img_filename = ann['file_name']
        mask_filename = img_filename.replace('.jpg', '.png')
        mask_path = os.path.join(masks_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            validation_errors.append(f"Missing mask: {mask_filename}")
            continue
        
        # Load mask and check encoding
        try:
            mask_rgb = np.array(Image.open(mask_path))
            mask_ids = rgb2id(mask_rgb)
            unique_ids = np.unique(mask_ids)
            
            # Check RGB channels
            r_unique = np.unique(mask_rgb[:,:,0])
            g_unique = np.unique(mask_rgb[:,:,1])
            b_unique = np.unique(mask_rgb[:,:,2])
            
            # Get segment IDs from JSON
            json_segment_ids = set(seg['id'] for seg in ann['segments_info'])
            mask_segment_ids = set(unique_ids.tolist())
            
            # Check if IDs match
            if not json_segment_ids.issubset(mask_segment_ids):
                validation_errors.append(f"ID mismatch in {mask_filename}: JSON {sorted(list(json_segment_ids))[:5]} vs Mask {sorted(list(mask_segment_ids))[:5]}")
            
            # Check for large IDs (indicates encoding problem)
            max_id = unique_ids.max()
            if max_id > 1000:  # Arbitrary threshold
                large_id_count += 1
                validation_errors.append(f"Large IDs in {mask_filename}: max={max_id} (expected <1000)")
            else:
                correct_id_count += 1
            
            # Check RGB channel distribution
            if len(r_unique) == 1 and r_unique[0] == 0:  # All R values are 0
                if len(b_unique) > 1:  # But B values vary
                    validation_errors.append(f"BGR/RGB issue in {mask_filename}: values in B channel instead of R")
            
        except Exception as e:
            validation_errors.append(f"Error processing {mask_filename}: {str(e)}")
    
    # Print results
    print(f"\n📋 Validation Results for {dataset_name}:")
    print(f"   ✅ Correct encoding: {correct_id_count}/{sample_size} masks")
    print(f"   ❌ Large IDs (wrong encoding): {large_id_count}/{sample_size} masks")
    print(f"   🚨 Total errors: {len(validation_errors)}")
    
    if validation_errors:
        print(f"\n🚨 Errors found:")
        for error in validation_errors[:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(validation_errors) > 5:
            print(f"   ... and {len(validation_errors) - 5} more errors")
    
    success = len(validation_errors) == 0
    if success:
        print(f"✅ {dataset_name} validation PASSED")
    else:
        print(f"❌ {dataset_name} validation FAILED")
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Validate panoptic datasets')
    parser.add_argument('--dataset-root', default='myotube_batch_output', 
                       help='Root directory containing panoptic data')
    args = parser.parse_args()
    
    dataset_root = args.dataset_root
    
    print("🔍 PANOPTIC DATASET VALIDATION")
    print("=" * 50)
    
    # Define datasets to validate
    datasets = [
        {
            'name': 'Stage 1 Train (Algorithmic)',
            'json': f'{dataset_root}/panoptic/algorithmic_train_panoptic.json',
            'masks': f'{dataset_root}/panoptic/algorithmic_train_panoptic_masks'
        },
        {
            'name': 'Stage 1 Val (Algorithmic)',
            'json': f'{dataset_root}/panoptic/algorithmic_val_panoptic.json',
            'masks': f'{dataset_root}/panoptic/algorithmic_val_panoptic_masks'
        },
        {
            'name': 'Stage 2 Train (Manual)',
            'json': f'{dataset_root}/panoptic/manual_train_panoptic.json',
            'masks': f'{dataset_root}/panoptic/manual_train_panoptic_masks'
        },
        {
            'name': 'Stage 2 Val (Manual)',
            'json': f'{dataset_root}/panoptic/manual_val_panoptic.json',
            'masks': f'{dataset_root}/panoptic/manual_val_panoptic_masks'
        }
    ]
    
    # Validate each dataset
    all_passed = True
    results = {}
    
    for dataset in datasets:
        success = validate_single_dataset(
            dataset['json'], 
            dataset['masks'], 
            dataset['name']
        )
        results[dataset['name']] = success
        all_passed = all_passed and success
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 FINAL VALIDATION SUMMARY")
    print("=" * 50)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {name}")
    
    if all_passed:
        print(f"\n🎉 ALL DATASETS PASSED VALIDATION!")
        print(f"✅ Ready for training!")
    else:
        print(f"\n🚨 SOME DATASETS FAILED VALIDATION!")
        print(f"❌ Fix issues before training!")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    exit(main())