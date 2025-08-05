#!/usr/bin/env python3

"""
Simple debug to check model predictions vs thresholds
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
from register_two_stage_datasets import register_two_stage_datasets
from detectron2.engine import DefaultPredictor
import cv2

def main():
    print("🔍 Simple debug: checking thresholds...")
    
    # Register datasets
    register_two_stage_datasets(register_panoptic=True)
    
    # Load config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_panoptic_deeplab_config(cfg) 
    add_maskformer2_config(cfg)
    cfg.merge_from_file('stage1_panoptic_config.yaml')
    
    # Use the trained model
    cfg.MODEL.WEIGHTS = "output_stage1_panoptic_algorithmic/model_0000099.pth"
    
    print(f"🎯 Current thresholds:")
    print(f"   OBJECT_MASK_THRESHOLD: {cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD}")
    print(f"   OVERLAP_THRESHOLD: {cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD}")
    
    # Try with VERY low threshold
    print(f"\n🔧 Testing with very low threshold (0.01)...")
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.01
    
    predictor = DefaultPredictor(cfg)
    
    # Get a training sample
    from detectron2.data import DatasetCatalog
    dataset = DatasetCatalog.get("myotube_stage1_panoptic_train")
    sample = dataset[0]
    
    print(f"\n🔍 Testing on: {os.path.basename(sample['file_name'])}")
    
    # Load and predict
    image = cv2.imread(sample['file_name'])
    predictions = predictor(image)
    
    print(f"\n🎯 Results with threshold 0.01:")
    if 'panoptic_seg' in predictions:
        panoptic_seg, segments_info = predictions['panoptic_seg']
        print(f"  Found {len(segments_info)} segments")
        
        seg_np = panoptic_seg.cpu().numpy()
        unique_vals = np.unique(seg_np)
        print(f"  Unique values: {unique_vals}")
        
        if len(segments_info) > 0:
            print("  ✅ SUCCESS! Model is making predictions!")
            for i, segment in enumerate(segments_info[:3]):  # Show first 3
                print(f"    Segment {i}: {segment}")
        else:
            print("  ❌ Still no segments detected")
    else:
        print("  ❌ No panoptic_seg found")
    
    # Also try with threshold 0.0
    print(f"\n🔧 Testing with threshold 0.0...")
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    predictor = DefaultPredictor(cfg)
    
    predictions = predictor(image)
    
    if 'panoptic_seg' in predictions:
        panoptic_seg, segments_info = predictions['panoptic_seg']
        print(f"  Found {len(segments_info)} segments with threshold 0.0")
        
        if len(segments_info) > 0:
            print("  ✅ SUCCESS! Model works with threshold 0.0!")
        else:
            print("  ❌ Still no segments even with threshold 0.0")
            print("  This suggests the model isn't learning to predict myotubes")

if __name__ == "__main__":
    main()