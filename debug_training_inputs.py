#!/usr/bin/env python3

"""
Debug Training Inputs - Check what data is actually reaching the model during training

This script loads the training data pipeline and inspects the exact format
of data being fed to the model to understand why mask/dice losses are zero.
"""

import torch
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
import cv2
from register_two_stage_datasets import register_two_stage_datasets

def debug_training_data():
    """Debug the training data pipeline to see what the model receives."""
    
    # Register datasets
    print("🔄 Registering panoptic datasets...")
    register_two_stage_datasets("myotube_batch_output", register_instance=True, register_panoptic=True)
    
    # Load config
    cfg = get_cfg()
    cfg.merge_from_file("stage1_panoptic_config_fixed.yaml")
    cfg.MODEL.DEVICE = "cpu"  # Use CPU for debugging
    
    print("🔧 Building training data loader...")
    train_loader = build_detection_train_loader(cfg)
    
    print("📊 Inspecting training samples...")
    
    for i, batch in enumerate(train_loader):
        if i >= 3:  # Check first 3 samples
            break
            
        print(f"\n🔍 BATCH {i+1}:")
        print(f"   Batch size: {len(batch)}")
        
        for j, sample in enumerate(batch):
            print(f"\n   📋 SAMPLE {j+1}:")
            print(f"      Keys: {list(sample.keys())}")
            
            # Check image
            if 'image' in sample:
                image = sample['image']
                print(f"      Image shape: {image.shape}")
                print(f"      Image dtype: {image.dtype}")
            
            # Check instances (if present)
            if 'instances' in sample:
                instances = sample['instances']
                print(f"      Instances: {len(instances)} objects")
                if len(instances) > 0:
                    print(f"      Instance classes: {instances.gt_classes}")
                    print(f"      Instance masks shape: {instances.gt_masks.tensor.shape}")
            
            # Check panoptic ground truth
            if 'pan_seg' in sample:
                pan_seg = sample['pan_seg']
                print(f"      ✅ pan_seg found!")
                print(f"      pan_seg shape: {pan_seg.shape}")
                print(f"      pan_seg dtype: {pan_seg.dtype}")
                print(f"      pan_seg unique values: {torch.unique(pan_seg)}")
                print(f"      pan_seg value counts:")
                unique_vals, counts = torch.unique(pan_seg, return_counts=True)
                for val, count in zip(unique_vals, counts):
                    print(f"        Class {val}: {count} pixels")
            else:
                print(f"      ❌ NO pan_seg found!")
            
            # Check segments_info
            if 'segments_info' in sample:
                segments_info = sample['segments_info']
                print(f"      ✅ segments_info found: {len(segments_info)} segments")
                for seg in segments_info[:3]:  # Show first 3
                    print(f"        Segment: id={seg.get('id', 'N/A')}, category_id={seg.get('category_id', 'N/A')}, area={seg.get('area', 'N/A')}")
            else:
                print(f"      ❌ NO segments_info found!")
            
            # Check if we have the required fields for panoptic training
            required_fields = ['pan_seg', 'segments_info']
            missing_fields = [field for field in required_fields if field not in sample]
            if missing_fields:
                print(f"      🚨 MISSING REQUIRED FIELDS: {missing_fields}")
            else:
                print(f"      ✅ All required panoptic fields present")
    
    print(f"\n📋 SUMMARY:")
    print(f"   If 'pan_seg' is missing, the model can't compute mask/dice losses!")
    print(f"   If 'pan_seg' has only background (class 0), losses will be near zero!")

if __name__ == "__main__":
    debug_training_data()