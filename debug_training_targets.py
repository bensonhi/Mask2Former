#!/usr/bin/env python3

"""
Debug script to check training targets and see if they match model expectations
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
from mask2former.data.dataset_mappers.mask_former_panoptic_dataset_mapper import MaskFormerPanopticDatasetMapper

def main():
    print("🔍 Debugging training targets...")
    
    # Register datasets
    register_two_stage_datasets(register_panoptic=True)
    
    # Load config exactly like training
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_panoptic_deeplab_config(cfg) 
    add_maskformer2_config(cfg)
    cfg.merge_from_file('stage1_panoptic_config.yaml')
    
    print(f"✅ Config loaded:")
    print(f"   Num classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"   Dataset mapper: {cfg.INPUT.DATASET_MAPPER_NAME}")
    
    # Build the EXACT same data loader as training
    mapper = MaskFormerPanopticDatasetMapper(cfg, True)
    train_loader = build_detection_train_loader(cfg, mapper=mapper)
    
    print("✅ Training data loader built")
    
    # Get one training batch
    data_iter = iter(train_loader)
    batch = next(data_iter)
    sample = batch[0]  # First item in batch
    
    print(f"\n🔍 Training sample analysis:")
    print(f"   Sample keys: {list(sample.keys())}")
    
    # Check image
    if 'image' in sample:
        image = sample['image']
        print(f"   Image: shape={image.shape}, dtype={image.dtype}")
    
    # Check panoptic segmentation target
    if 'pan_seg' in sample:
        pan_seg = sample['pan_seg']
        print(f"   ✅ pan_seg: shape={pan_seg.shape}, dtype={pan_seg.dtype}")
        
        # Analyze target values
        unique_vals, counts = torch.unique(pan_seg, return_counts=True)
        total_pixels = pan_seg.numel()
        
        print(f"   Target pixel distribution:")
        for val, count in zip(unique_vals, counts):
            percentage = count.item() / total_pixels * 100
            print(f"     Value {val.item()}: {count.item():,} pixels ({percentage:.1f}%)")
            
        # Check if we have myotube pixels (should be class 1)
        myotube_pixels = (pan_seg == 1).sum().item()
        background_pixels = (pan_seg == 0).sum().item()
        
        print(f"   Class distribution:")
        print(f"     Background (0): {background_pixels:,} pixels")
        print(f"     Myotube (1): {myotube_pixels:,} pixels")
        
        if myotube_pixels > 0:
            print(f"   ✅ Training target HAS myotube pixels!")
        else:
            print(f"   ❌ Training target has NO myotube pixels!")
    else:
        print(f"   ❌ No 'pan_seg' in training sample!")
    
    # Check instances
    if 'instances' in sample:
        instances = sample['instances']
        print(f"   Instances: {len(instances)} objects")
        if len(instances) > 0:
            classes = instances.gt_classes
            print(f"     Classes: {classes}")
            print(f"     Unique classes: {torch.unique(classes)}")
            
            # Check if classes are correctly mapped
            myotube_instances = (classes == 1).sum().item()
            background_instances = (classes == 0).sum().item()
            
            print(f"     Background instances (0): {background_instances}")
            print(f"     Myotube instances (1): {myotube_instances}")
            
            if myotube_instances > 0:
                print(f"   ✅ Instance targets HAVE myotube class!")
            else:
                print(f"   ❌ Instance targets have NO myotube class!")
    
    # Check a few more samples
    print(f"\n🔍 Checking more samples...")
    for i in range(2, 5):  # Check samples 2-4
        try:
            batch = next(data_iter)
            sample = batch[0]
            
            pan_seg = sample.get('pan_seg', None)
            instances = sample.get('instances', None)
            
            if pan_seg is not None:
                myotube_pixels = (pan_seg == 1).sum().item()
                print(f"   Sample {i}: {myotube_pixels:,} myotube pixels")
            
            if instances is not None:
                classes = instances.gt_classes
                myotube_instances = (classes == 1).sum().item()
                print(f"   Sample {i}: {myotube_instances} myotube instances")
                
        except StopIteration:
            break
        except Exception as e:
            print(f"   Sample {i}: Error - {e}")

if __name__ == "__main__":
    main()