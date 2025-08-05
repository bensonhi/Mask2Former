#!/usr/bin/env python3

"""
Debug script to check what the training pipeline actually feeds to the model
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from mask2former import add_maskformer2_config
from detectron2.data import build_detection_train_loader
from register_two_stage_datasets import register_two_stage_datasets

def main():
    print("🔍 Debugging training data pipeline...")
    
    # Register datasets
    register_two_stage_datasets(register_panoptic=True)
    
    # Load config exactly like train_net.py
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_panoptic_deeplab_config(cfg) 
    add_maskformer2_config(cfg)
    cfg.merge_from_file('stage1_panoptic_config.yaml')
    
    print(f"✅ Config loaded:")
    print(f"   Backbone: {cfg.MODEL.BACKBONE.NAME}")
    print(f"   Dataset mapper: {cfg.INPUT.DATASET_MAPPER_NAME}")
    print(f"   Crop enabled: {cfg.INPUT.CROP.ENABLED}")
    print(f"   Crop size: {cfg.INPUT.CROP.SIZE}")
    print(f"   Single category max area: {cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA}")
    
    # Build data loader exactly like train_net.py
    print("\n🔄 Building training data loader...")
    try:
        from mask2former.data.dataset_mappers.mask_former_panoptic_dataset_mapper import MaskFormerPanopticDatasetMapper
        
        # Use the same logic as train_net.py
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            train_loader = build_detection_train_loader(cfg, mapper=mapper)
            print("✅ Data loader built with MaskFormerPanopticDatasetMapper")
        else:
            train_loader = build_detection_train_loader(cfg)
            print("✅ Data loader built with default mapper")
    except Exception as e:
        print(f"❌ Error building data loader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("📊 Analyzing training samples...")
    data_iter = iter(train_loader)
    
    for sample_idx in range(3):  # Check first 3 samples
        print(f"\n--- Sample {sample_idx + 1} ---")
        try:
            batch = next(data_iter)
            sample = batch[0]  # First item in batch
            
            print(f"Sample keys: {list(sample.keys())}")
            
            if 'image' in sample:
                image = sample['image']
                print(f"Image: shape={image.shape}, dtype={image.dtype}")
            
            if 'pan_seg' in sample:
                pan_seg = sample['pan_seg']
                print(f"✅ Found pan_seg: shape={pan_seg.shape}, dtype={pan_seg.dtype}")
                
                # Analyze pixel distribution
                pan_np = pan_seg.cpu().numpy() if torch.is_tensor(pan_seg) else pan_seg
                unique_vals, counts = np.unique(pan_np, return_counts=True)
                total_pixels = pan_np.size
                
                print(f"Pixel value distribution:")
                for val, count in zip(unique_vals, counts):
                    percentage = count / total_pixels * 100
                    print(f"  Value {val}: {count:,} pixels ({percentage:.1f}%)")
                
                # Check if this is a problematic sample (all background)
                if len(unique_vals) == 1 and unique_vals[0] == 0:
                    print("⚠️  WARNING: This sample is 100% background!")
                elif np.sum(pan_np == 0) / total_pixels > 0.95:
                    print("⚠️  WARNING: This sample is >95% background!")
                else:
                    print("✅ This sample has good foreground content")
            else:
                print("❌ CRITICAL: No 'pan_seg' found in sample!")
                print("   This means panoptic ground truth is missing from training!")
                
                # Check what panoptic info we do have
                if 'pan_seg_file_name' in sample:
                    print(f"   ℹ️  pan_seg_file_name: {sample['pan_seg_file_name']}")
                if 'segments_info' in sample:
                    print(f"   ℹ️  segments_info: {len(sample['segments_info'])} segments")
            
            if 'instances' in sample:
                instances = sample['instances']
                print(f"Instances: {len(instances)} objects")
                if len(instances) > 0:
                    print(f"  Classes: {instances.gt_classes}")
                    print(f"  Has masks: {instances.has('gt_masks')}")
                    if instances.has('gt_masks'):
                        masks = instances.gt_masks
                        print(f"  Mask shapes: {[mask.shape for mask in masks.tensor[:3]]}")  # First 3 masks
                        
        except Exception as e:
            print(f"❌ Error processing sample {sample_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n🏁 Analysis complete!")

if __name__ == "__main__":
    main()