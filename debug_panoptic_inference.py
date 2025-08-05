#!/usr/bin/env python3

import os
import sys
import numpy as np
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

def main():
    print("🔍 Debugging panoptic inference...")
    
    # Setup config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("stage1_panoptic_config_clean.yaml")
    
    # Use latest checkpoint
    checkpoint_dir = "./output_stage1_panoptic_algorithmic"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if checkpoint_files:
            # Look for model_final.pth first, otherwise use the last one alphabetically
            if "model_final.pth" in checkpoint_files:
                latest_checkpoint = "model_final.pth"
            else:
                latest_checkpoint = sorted(checkpoint_files)[-1]
            cfg.MODEL.WEIGHTS = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"✅ Using checkpoint: {latest_checkpoint}")
        else:
            print("❌ No checkpoint found!")
            return
    else:
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # Very low threshold
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0  # Very low threshold
    
    predictor = DefaultPredictor(cfg)
    
    # Load a test image
    images_dir = "myotube_batch_output/images"
    
    # Debug: check current directory
    print(f"🔍 Current directory: {os.getcwd()}")
    print(f"🔍 Looking for images in: {os.path.abspath(images_dir)}")
    print(f"🔍 Directory exists: {os.path.exists(images_dir)}")
    
    if os.path.exists(images_dir):
        all_files = os.listdir(images_dir)
        print(f"🔍 All files in directory: {all_files[:5]}...")  # Show first 5
        image_files = [f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"🔍 Image files found: {len(image_files)}")
    else:
        print(f"❌ Directory not found: {images_dir}")
        return
    
    if not image_files:
        print(f"❌ No image files found in: {images_dir}")
        return
    
    test_image_path = os.path.join(images_dir, image_files[0])
    print(f"📸 Using test image: {image_files[0]}")
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    image = cv2.imread(test_image_path)
    print(f"📸 Loaded image: {image.shape}")
    
    # Resize image to manageable size for debugging
    if image.shape[0] > 1000 or image.shape[1] > 1000:
        scale = min(1000 / image.shape[0], 1000 / image.shape[1])
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        image = cv2.resize(image, (new_width, new_height))
        print(f"📸 Resized image to: {image.shape}")
    
    # Run inference
    print("🚀 Running inference...")
    with torch.no_grad():
        predictions = predictor(image)
    
    print(f"🎯 Prediction keys: {predictions.keys()}")
    
    # Check panoptic predictions
    if 'panoptic_seg' in predictions:
        panoptic_seg, segments_info = predictions['panoptic_seg']
        print(f"📊 Panoptic seg shape: {panoptic_seg.shape}")
        print(f"📊 Number of segments: {len(segments_info)}")
        
        # Check unique values in panoptic mask
        seg_np = panoptic_seg.cpu().numpy()
        unique_vals = np.unique(seg_np)
        print(f"📊 Unique values in panoptic mask: {unique_vals}")
        print(f"📊 Value counts: {[(val, np.sum(seg_np == val)) for val in unique_vals]}")
        
        if len(segments_info) > 0:
            print("✅ SUCCESS! Model is making predictions!")
            for i, segment in enumerate(segments_info):
                print(f"  Segment {i}: {segment}")
        else:
            print("❌ No segments detected even with threshold 0.0")
            
        # Check raw model outputs
        print("\n🔧 Checking raw model outputs...")
        if hasattr(predictor.model, 'panoptic_inference'):
            # Get the raw mask_cls and mask_pred from the model
            print("  Model has panoptic_inference method")
        
    else:
        print("❌ No panoptic_seg in predictions")
    
    # Check if model outputs any masks at all
    if 'instances' in predictions:
        instances = predictions['instances']
        print(f"📊 Instance predictions: {len(instances)}")
        if len(instances) > 0:
            print(f"  Scores: {instances.scores}")
            print(f"  Classes: {instances.pred_classes}")
    
    # Check metadata
    try:
        meta = MetadataCatalog.get("myotube_stage1_panoptic_val")
        print(f"🔍 Dataset metadata:")
        print(f"  thing_classes: {getattr(meta, 'thing_classes', 'Not found')}")
        print(f"  stuff_classes: {getattr(meta, 'stuff_classes', 'Not found')}")
        print(f"  thing_dataset_id_to_contiguous_id: {getattr(meta, 'thing_dataset_id_to_contiguous_id', 'Not found')}")
    except:
        print("❌ Could not load dataset metadata")

if __name__ == "__main__":
    main()