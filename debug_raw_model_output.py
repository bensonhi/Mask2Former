#!/usr/bin/env python3

"""
Debug script to check RAW model outputs before post-processing
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
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import cv2

def main():
    print("🔍 Debugging RAW model outputs...")
    
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
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"✅ Config loaded:")
    print(f"   Model weights: {cfg.MODEL.WEIGHTS}")
    print(f"   Device: {cfg.MODEL.DEVICE}")
    print(f"   Num classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"   Object mask threshold: {cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD}")
    
    # Build model
    model = build_model(cfg)
    model.eval()
    
    # Load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    print("✅ Model loaded")
    
    # Get a training sample
    from detectron2.data import DatasetCatalog
    dataset = DatasetCatalog.get("myotube_stage1_panoptic_train")
    sample = dataset[0]
    
    print(f"\n🔍 Testing on: {sample['file_name']}")
    
    # Load and preprocess image like detectron2 does
    image = cv2.imread(sample['file_name'])
    print(f"📷 Original image shape: {image.shape}")
    
    # Use DefaultPredictor's preprocessing
    from detectron2.engine import DefaultPredictor
    
    # Create predictor to get preprocessed input
    predictor = DefaultPredictor(cfg)
    
    # Get the same preprocessing transforms that DefaultPredictor uses
    transform_gen = predictor.transform_gen
    original_image = image[:, :, ::-1]  # BGR to RGB
    
    # Apply transforms
    height, width = original_image.shape[:2]
    image_transformed = transform_gen.get_transform(original_image).apply_image(original_image)
    image_tensor = torch.as_tensor(np.ascontiguousarray(image_transformed).transpose(2, 0, 1))
    
    print(f"📷 Transformed image shape: {image_tensor.shape}")
    
    # Prepare input batch exactly like DefaultPredictor
    inputs = {
        "image": image_tensor,
        "height": height,
        "width": width
    }
    
    with torch.no_grad():
        # Run model
        outputs = model([inputs])
        output = outputs[0]
        
        print(f"\n🎯 RAW Model outputs:")
        for key, value in output.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                
                # Check classification predictions
                if key == 'pred_logits':
                    # Apply softmax to see class probabilities
                    probs = torch.softmax(value, dim=-1)
                    print(f"    Class probabilities (first few queries):")
                    for i in range(min(5, probs.shape[0])):
                        bg_prob = probs[i, 0].item()  # Background
                        myotube_prob = probs[i, 1].item()  # Myotube
                        print(f"      Query {i}: bg={bg_prob:.4f}, myotube={myotube_prob:.4f}")
                
                # Check mask predictions
                elif key == 'pred_masks':
                    print(f"    Mask stats: min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f}")
                    
                    # Check if any masks have high confidence
                    high_conf_masks = (value > 0.5).sum(dim=(1,2))
                    print(f"    Masks with >0.5 confidence: {(high_conf_masks > 0).sum()} / {len(high_conf_masks)}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Check post-processing
        if 'panoptic_seg' in output:
            panoptic_seg, segments_info = output['panoptic_seg']
            print(f"\n🔄 Post-processed results:")
            print(f"  Panoptic seg: {panoptic_seg.shape}, unique: {torch.unique(panoptic_seg)}")
            print(f"  Segments found: {len(segments_info)}")
        else:
            print(f"\n❌ No 'panoptic_seg' in output!")
    
    print(f"\n📊 Config thresholds:")
    print(f"  OBJECT_MASK_THRESHOLD: {cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD}")
    print(f"  OVERLAP_THRESHOLD: {cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD}")

if __name__ == "__main__":
    main()