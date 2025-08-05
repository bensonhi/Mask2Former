#!/usr/bin/env python3

"""
Debug script to check if the trained model is actually predicting myotubes
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
from detectron2.engine import DefaultPredictor
import cv2

def main():
    print("🔍 Debugging trained model predictions...")
    
    # Register datasets
    register_two_stage_datasets(register_panoptic=True)
    
    # Load config exactly like demo.py
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_panoptic_deeplab_config(cfg) 
    add_maskformer2_config(cfg)
    cfg.merge_from_file('stage1_panoptic_config.yaml')
    
    # Use the trained model
    cfg.MODEL.WEIGHTS = "output_stage1_panoptic_algorithmic/model_0000999.pth"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"✅ Config loaded:")
    print(f"   Model weights: {cfg.MODEL.WEIGHTS}")
    print(f"   Device: {cfg.MODEL.DEVICE}")
    print(f"   Num classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    print("✅ Predictor created")
    
    # Test on a training image to see if model learned anything
    from detectron2.data import DatasetCatalog
    dataset = DatasetCatalog.get("myotube_stage1_panoptic_train")
    
    # Get first training image
    sample = dataset[0]
    print(f"\n🔍 Testing on training image: {sample['file_name']}")
    
    # Load and predict
    image = cv2.imread(sample['file_name'])
    if image is None:
        print(f"❌ Could not load image: {sample['file_name']}")
        return
        
    print(f"📷 Image shape: {image.shape}")
    
    # Run prediction
    predictions = predictor(image)
    
    print(f"\n🎯 Model predictions:")
    for key, value in predictions.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}")
        elif key == 'panoptic_seg' and isinstance(value, tuple):
            seg, info = value
            print(f"  panoptic_seg:")
            print(f"    seg: shape={seg.shape}, unique_values={torch.unique(seg)}")
            print(f"    segments_info: {len(info)} segments")
            
            # Analyze what the model predicted
            seg_np = seg.cpu().numpy()
            unique_vals, counts = np.unique(seg_np, return_counts=True)
            print(f"    Prediction statistics:")
            for val, count in zip(unique_vals, counts):
                percentage = count / seg_np.size * 100
                print(f"      Value {val}: {count:,} pixels ({percentage:.1f}%)")
        else:
            print(f"  {key}: {type(value)}")
    
    # Compare with ground truth
    print(f"\n📊 Ground truth comparison:")
    print(f"   GT segments: {len(sample['segments_info'])}")
    
    # Load ground truth panoptic mask
    if 'pan_seg_file_name' in sample:
        gt_mask_path = sample['pan_seg_file_name']
        print(f"   GT mask: {gt_mask_path}")
        
        # Try to load and analyze GT mask
        try:
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_COLOR)
            if gt_mask is not None:
                from panopticapi.utils import rgb2id
                gt_pan = rgb2id(gt_mask)
                gt_unique = np.unique(gt_pan)
                print(f"   GT unique values: {gt_unique}")
                print(f"   GT has {len(gt_unique)} different segments")
            else:
                print(f"   ❌ Could not load GT mask")
        except Exception as e:
            print(f"   ❌ Error loading GT mask: {e}")

if __name__ == "__main__":
    main()