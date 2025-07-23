#!/usr/bin/env python3

"""
Test script to verify dataset registration with proper metadata
"""

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import os

def test_registration():
    print("🧪 Testing dataset registration with panoptic metadata...")
    
    # Define metadata
    metadata = {
        "thing_classes": ["myotube"],
        "thing_colors": [[255, 0, 0]],
        "thing_dataset_id_to_contiguous_id": {1: 0},
        "stuff_classes": ["background"],
        "stuff_colors": [[0, 0, 0]],
        "stuff_dataset_id_to_contiguous_id": {0: 1},
    }
    
    print(f"📋 Metadata to register: {list(metadata.keys())}")
    
    # Check if annotation file exists
    ann_file = "myotube_batch_output/annotations/algorithmic_train_annotations.json"
    img_dir = "myotube_batch_output/images"
    
    if not os.path.exists(ann_file):
        print(f"❌ Annotation file not found: {ann_file}")
        print("   Please ensure your dataset is in the correct location")
        return False
    
    # Register with metadata
    dataset_name = "test_myotube_panoptic"
    try:
        register_coco_instances(
            dataset_name,
            metadata,
            ann_file,
            img_dir
        )
        print(f"✅ Successfully registered dataset: {dataset_name}")
    except Exception as e:
        print(f"❌ Failed to register dataset: {e}")
        return False
    
    # Verify metadata
    try:
        meta = MetadataCatalog.get(dataset_name)
        print(f"📋 Available metadata keys: {list(meta._metadata.keys())}")
        
        if hasattr(meta, 'thing_dataset_id_to_contiguous_id'):
            print(f"✅ thing_dataset_id_to_contiguous_id: {meta.thing_dataset_id_to_contiguous_id}")
            print(f"✅ thing_classes: {meta.thing_classes}")
            return True
        else:
            print("❌ thing_dataset_id_to_contiguous_id missing!")
            print(f"   Available attributes: {dir(meta)}")
            return False
    except Exception as e:
        print(f"❌ Error verifying metadata: {e}")
        return False

if __name__ == "__main__":
    success = test_registration()
    if success:
        print("\n🎉 Test successful! Metadata registration works correctly.")
        print("   You can now register your datasets for panoptic mode.")
    else:
        print("\n❌ Test failed. There's an issue with metadata registration.")
        print("   Check the error messages above for details.") 