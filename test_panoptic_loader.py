#!/usr/bin/env python3
"""
Test script to debug panoptic data loading and see what the evaluator is receiving.
"""

import os
import sys
sys.path.append('.')

from register_two_stage_datasets import load_myotube_panoptic_json

def test_panoptic_loader():
    """Test the custom panoptic loader to see what data it returns."""
    
    # Test with Stage 1 data
    panoptic_dir = "myotube_batch_output/panoptic"
    images_dir = "myotube_batch_output/images"
    
    stage1_train_json = os.path.join(panoptic_dir, "algorithmic_train_panoptic.json")
    stage1_train_masks = os.path.join(panoptic_dir, "algorithmic_train_panoptic_masks")
    
    # Metadata that should match our registration
    panoptic_metadata = {
        "thing_classes": ["myotube"],
        "stuff_classes": [],
        "thing_dataset_id_to_contiguous_id": {1: 0},
        "stuff_dataset_id_to_contiguous_id": {},
    }
    
    print("🔍 Testing panoptic data loader...")
    print(f"JSON file: {stage1_train_json}")
    print(f"Masks dir: {stage1_train_masks}")
    print(f"Images dir: {images_dir}")
    
    # Check if files exist
    if not os.path.exists(stage1_train_json):
        print(f"❌ JSON file not found: {stage1_train_json}")
        return
    
    if not os.path.exists(stage1_train_masks):
        print(f"❌ Masks directory not found: {stage1_train_masks}")
        return
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    try:
        # Load data using our custom loader
        dataset_dicts = load_myotube_panoptic_json(
            stage1_train_json, images_dir, stage1_train_masks, panoptic_metadata
        )
        
        print(f"📊 Loaded {len(dataset_dicts)} dataset entries")
        
        if len(dataset_dicts) == 0:
            print("❌ No dataset entries loaded!")
            return
        
        # Examine first few entries
        for i, entry in enumerate(dataset_dicts[:3]):
            print(f"\n🖼️ Entry {i+1}:")
            print(f"   file_name: {entry.get('file_name', 'MISSING')}")
            print(f"   image_id: {entry.get('image_id', 'MISSING')}")
            print(f"   pan_seg_file_name: {entry.get('pan_seg_file_name', 'MISSING')}")
            
            # Check if files exist
            if 'file_name' in entry:
                exists = os.path.exists(entry['file_name'])
                print(f"   image exists: {exists}")
            
            if 'pan_seg_file_name' in entry:
                exists = os.path.exists(entry['pan_seg_file_name'])
                print(f"   mask exists: {exists}")
            
            # Check segments_info
            segments_info = entry.get('segments_info', [])
            print(f"   segments_info: {len(segments_info)} segments")
            
            for j, seg in enumerate(segments_info[:3]):  # First 3 segments
                print(f"     Segment {j+1}: ID={seg.get('id', 'MISSING')}, "
                      f"category_id={seg.get('category_id', 'MISSING')}, "
                      f"isthing={seg.get('isthing', 'MISSING')}")
        
        print(f"\n✅ Panoptic loader appears to be working correctly!")
        print(f"   Total entries: {len(dataset_dicts)}")
        total_segments = sum(len(entry.get('segments_info', [])) for entry in dataset_dicts)
        print(f"   Total segments: {total_segments}")
        
        if total_segments == 0:
            print("❌ WARNING: No segments found in any entries!")
        
    except Exception as e:
        print(f"❌ Error in panoptic loader: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🧪 Testing Panoptic Data Loader")
    print("=" * 50)
    test_panoptic_loader()

if __name__ == "__main__":
    main() 