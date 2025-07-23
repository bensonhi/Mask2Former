#!/usr/bin/env python3

"""
Two-Stage Dataset Registration for Myotube Segmentation

This script registers datasets for two-stage training:
- Stage 1: Algorithmic annotations (~100 images from batch processing)
- Stage 2: Manual annotations (~5 images with high-quality annotations)

Expected directory structure:
- algorithmic_dataset/
  ├── images/
  └── annotations/
      ├── train_annotations.json
      └── test_annotations.json  (or val_annotations.json)

- manual_dataset/
  ├── images/
  └── annotations/
      ├── train_annotations.json
      └── test_annotations.json  (or val_annotations.json)
"""

import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_myotube_metadata():
    """
    Get metadata for myotube dataset compatible with panoptic segmentation.
    
    Returns:
        dict: Metadata with thing/stuff mappings for single myotube class
    """
    # For myotube segmentation with single class:
    # - Category ID 1 in COCO annotations = myotube (thing)
    # - Contiguous ID 0 = first (and only) class in model
    
    meta = {
        # Thing classes (countable objects like myotubes)
        "thing_classes": ["myotube"],
        "thing_colors": [[255, 0, 0]],  # Red color for visualization
        "thing_dataset_id_to_contiguous_id": {1: 0},  # COCO category_id=1 -> model class=0
        
        # Stuff classes (background regions)
        "stuff_classes": ["background"],
        "stuff_colors": [[0, 0, 0]],  # Black background
        "stuff_dataset_id_to_contiguous_id": {0: 1},  # Background -> class=1
    }
    return meta

def register_two_stage_datasets(
    dataset_root: str = "myotube_batch_output"
):
    """
    Register datasets for two-stage myotube training from unified directory.
    
    Expected structure:
    dataset_root/
    ├── images/
    └── annotations/
        ├── algorithmic_train_annotations.json
        ├── algorithmic_test_annotations.json (or algorithmic_val_annotations.json)
        ├── manual_train_annotations.json
        └── manual_test_annotations.json (or manual_val_annotations.json)
    
    Args:
        dataset_root: Path to unified dataset directory
    """
    
    print("🔄 Registering two-stage myotube datasets from unified directory...")
    print(f"   Dataset root: {dataset_root}")
    print("   Note: Re-registering will overwrite any existing registrations")
    
    # Check if dataset root exists
    if not os.path.exists(dataset_root):
        print(f"⚠️  Warning: Dataset not found at {dataset_root}")
        print(f"   Please create the unified dataset structure")
        return
    
    # Common images directory for both stages
    images_dir = os.path.join(dataset_root, "images")
    annotations_dir = os.path.join(dataset_root, "annotations")
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(annotations_dir):
        print(f"❌ Annotations directory not found: {annotations_dir}")
        return

    # ===== STAGE 1: ALGORITHMIC ANNOTATIONS =====
    print(f"\n📊 Stage 1: Algorithmic Annotations")
    
    # Get metadata for panoptic compatibility
    myotube_metadata = get_myotube_metadata()
    print(f"   📋 Using metadata: {list(myotube_metadata.keys())}")
    
    # Look for algorithmic annotation files
    stage1_train_ann = os.path.join(annotations_dir, "algorithmic_train_annotations.json")
    stage1_val_ann = os.path.join(annotations_dir, "manual_train_annotations.json")
    
    # Fallback names
    if not os.path.exists(stage1_val_ann):
        stage1_val_ann = os.path.join(annotations_dir, "algorithmic_val_annotations.json")
    if not os.path.exists(stage1_val_ann):
        # Use train annotations for validation if no separate validation set
        stage1_val_ann = stage1_train_ann
        print("   ℹ️  Using training annotations for validation (no separate val set)")
    
    if os.path.exists(stage1_train_ann):
        register_coco_instances(
            "myotube_stage1_train",
            myotube_metadata,
            stage1_train_ann,
            images_dir
        )
        
        register_coco_instances(
            "myotube_stage1_val",
            myotube_metadata,
            stage1_val_ann,
            images_dir
        )
        
        print(f"   ✅ Registered myotube_stage1_train")
        print(f"   ✅ Registered myotube_stage1_val")
        
        # Count images in Stage 1
        import json
        with open(stage1_train_ann, 'r') as f:
            stage1_data = json.load(f)
        print(f"   📈 Training images: {len(stage1_data['images'])}")
        print(f"   📈 Training annotations: {len(stage1_data['annotations'])}")
    else:
        print(f"   ❌ Algorithmic train annotations not found: {stage1_train_ann}")

    # ===== STAGE 2: MANUAL ANNOTATIONS =====
    print(f"\n🎯 Stage 2: Manual Annotations")
    
    # Look for manual annotation files
    stage2_train_ann = os.path.join(annotations_dir, "manual_train_annotations.json")
    stage2_val_ann = os.path.join(annotations_dir, "manual_test_annotations.json")
    
    # Fallback names
    if not os.path.exists(stage2_val_ann):
        stage2_val_ann = os.path.join(annotations_dir, "manual_val_annotations.json")
    if not os.path.exists(stage2_val_ann):
        # Use train annotations for validation if no separate validation set
        stage2_val_ann = stage2_train_ann
        print("   ℹ️  Using training annotations for validation (no separate val set)")
    
    if os.path.exists(stage2_train_ann):
        register_coco_instances(
            "myotube_stage2_train",
            myotube_metadata,
            stage2_train_ann,
            images_dir
        )
        
        register_coco_instances(
            "myotube_stage2_val",
            myotube_metadata,
            stage2_val_ann,
            images_dir
        )
        
        print(f"   ✅ Registered myotube_stage2_train")
        print(f"   ✅ Registered myotube_stage2_val")
        
        # Count images in Stage 2
        import json
        with open(stage2_train_ann, 'r') as f:
            stage2_data = json.load(f)
        print(f"   📈 Training images: {len(stage2_data['images'])}")
        print(f"   📈 Training annotations: {len(stage2_data['annotations'])}")
    else:
        print(f"   ❌ Manual train annotations not found: {stage2_train_ann}")
    
    # ===== VERIFICATION =====
    print(f"\n🔍 Verifying metadata registration...")
    for dataset_name in ["myotube_stage1_train", "myotube_stage2_train"]:
        try:
            meta = MetadataCatalog.get(dataset_name)
            if hasattr(meta, 'thing_dataset_id_to_contiguous_id'):
                print(f"   ✅ {dataset_name}: metadata correctly set")
                print(f"      thing_dataset_id_to_contiguous_id: {meta.thing_dataset_id_to_contiguous_id}")
            else:
                print(f"   ❌ {dataset_name}: missing thing_dataset_id_to_contiguous_id")
        except Exception as e:
            print(f"   ❌ {dataset_name}: error accessing metadata - {e}")
    
    print(f"\n🎉 Dataset registration complete!")
    print(f"   You can now use panoptic mode with OVERLAP_THRESHOLD: 0.0")

def check_dataset_structure(dataset_root):
    """Check unified dataset structure and files."""
    print("🔍 Checking unified dataset structure...")
    
    # Common paths where unified dataset might be located
    potential_paths = [
        "myotube_batch_output",
        "myotube_dataset",
        "dataset", 
        "../myotube_batch_output",
        "../myotube_dataset",
        "../dataset",
        "data/myotube_batch_output"
    ]
    
    found_paths = []
    for path in potential_paths:
        if os.path.exists(path):
            found_paths.append(path)
            print(f"   ✅ Found directory: {path}")
    
    if dataset_root and os.path.exists(dataset_root):
        print(f"   📁 Checking structure in: {dataset_root}")
        
        # Check for required files
        annotations_dir = os.path.join(dataset_root, "annotations")
        images_dir = os.path.join(dataset_root, "images")
        
        if os.path.exists(annotations_dir):
            print(f"   ✅ Annotations directory found")
            
            # Check for 4 annotation files
            expected_files = [
                "algorithmic_train_annotations.json",
                "algorithmic_test_annotations.json", 
                "manual_train_annotations.json",
                "manual_test_annotations.json"
            ]
            
            for file in expected_files:
                file_path = os.path.join(annotations_dir, file)
                if os.path.exists(file_path):
                    print(f"   ✅ Found: {file}")
                else:
                    print(f"   ❌ Missing: {file}")
        else:
            print(f"   ❌ Annotations directory not found")
        
        if os.path.exists(images_dir):
            print(f"   ✅ Images directory found")
        else:
            print(f"   ❌ Images directory not found")
    
    return found_paths

if __name__ == "__main__":
    # Check for existing datasets
    found_paths = check_dataset_structure("myotube_dataset")
    
    # Determine dataset path
    dataset_path = "myotube_batch_output"
    
    # Use found path if available
    for path in found_paths:
        if "myotube_batch_output" in path:
            dataset_path = path
            print(f"   🔄 Using {path} as unified dataset")
            break
        elif "myotube_dataset" in path:
            dataset_path = path
            print(f"   🔄 Using {path} as unified dataset")
            break
        elif "dataset" in path:
            dataset_path = path
            print(f"   🔄 Using {path} as unified dataset")
            break
    
    # Register datasets
    register_two_stage_datasets(dataset_path) 