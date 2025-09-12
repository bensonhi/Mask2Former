#!/usr/bin/env python3

"""
Two-Stage Dataset Registration for Myotube Segmentation (Instance Only)

This script registers datasets for two-stage instance segmentation training:
- Stage 1: Algorithmic annotations (~100 images from batch processing)
- Stage 2: Manual annotations (~5 images with high-quality annotations)

Panoptic registration has been removed/disabled per project decision.

Expected unified directory structure:
dataset_root/
  ├── images/
  └── annotations/
      ├── algorithmic_train_annotations.json
      ├── algorithmic_test_annotations.json (or val)
      ├── manual_train_annotations.json
      └── manual_test_annotations.json (or val)
"""

import os
import json
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.file_io import PathManager  # unused now; kept for compatibility
import detectron2.utils.comm as comm  # unused in this module; kept for compatibility

def _register_instance_datasets(annotations_dir, images_dir):
    """Register instance segmentation datasets."""
    registered = []
    
    # Stage 1: Algorithmic annotations
    print(f"   📊 Stage 1: Algorithmic Annotations")
    stage1_train_ann = os.path.join(annotations_dir, "algorithmic_train_annotations.json")
    stage1_val_ann = os.path.join(annotations_dir, "algorithmic_test_annotations.json")
    
    if os.path.exists(stage1_train_ann):
        register_coco_instances("myotube_stage1_train", {}, stage1_train_ann, images_dir)
        registered.append("myotube_stage1_train")
        print(f"      ✅ Registered myotube_stage1_train")
        
        # Count images and annotations
        with open(stage1_train_ann, 'r') as f:
            data = json.load(f)
        print(f"      📈 Training images: {len(data['images'])}")
        print(f"      📈 Training annotations: {len(data['annotations'])}")
    else:
        print(f"      ❌ Algorithmic train annotations not found: {stage1_train_ann}")
    
    if os.path.exists(stage1_val_ann):
        register_coco_instances("myotube_stage1_val", {}, stage1_val_ann, images_dir)
        registered.append("myotube_stage1_val")
        print(f"      ✅ Registered myotube_stage1_val")
    else:
        print(f"      ⚠️  Algorithmic val annotations not found, using train for validation")
        if stage1_train_ann in [r.split('/')[-1] for r in registered]:
            register_coco_instances("myotube_stage1_val", {}, stage1_train_ann, images_dir)
            registered.append("myotube_stage1_val")
    
    # Stage 2: Manual annotations
    print(f"   🎯 Stage 2: Manual Annotations")
    stage2_train_ann = os.path.join(annotations_dir, "manual_train_annotations.json")
    stage2_val_ann = os.path.join(annotations_dir, "manual_test_annotations.json")
    
    if os.path.exists(stage2_train_ann):
        register_coco_instances("myotube_stage2_train", {}, stage2_train_ann, images_dir)
        registered.append("myotube_stage2_train")
        print(f"      ✅ Registered myotube_stage2_train")
        
        # Count images and annotations
        with open(stage2_train_ann, 'r') as f:
            data = json.load(f)
        print(f"      📈 Training images: {len(data['images'])}")
        print(f"      📈 Training annotations: {len(data['annotations'])}")
    else:
        print(f"      ❌ Manual train annotations not found: {stage2_train_ann}")
    
    if os.path.exists(stage2_val_ann):
        register_coco_instances("myotube_stage2_val", {}, stage2_val_ann, images_dir)
        registered.append("myotube_stage2_val")
        print(f"      ✅ Registered myotube_stage2_val")
    else:
        print(f"      ⚠️  Manual val annotations not found, using train for validation")
        if stage2_train_ann in [r.split('/')[-1] for r in registered]:
            register_coco_instances("myotube_stage2_val", {}, stage2_train_ann, images_dir)
            registered.append("myotube_stage2_val")
    
    return registered

def _register_panoptic_datasets(panoptic_dir, images_dir):
    """Panoptic registration disabled. This function is kept for compatibility and returns empty list."""
    print("   ⚠️ Panoptic registration is disabled in this project.")
    return []

def register_two_stage_datasets(
    dataset_root: str = "myotube_batch_output",
    register_instance: bool = True,
    register_panoptic: bool = False
):
    """
    Register datasets for two-stage myotube training from unified directory.
    
    Expected structure:
    dataset_root/
    ├── images/
    ├── annotations/  (instance annotations)
    │   ├── algorithmic_train_annotations.json
    │   ├── algorithmic_test_annotations.json
    │   ├── manual_train_annotations.json
    │   └── manual_test_annotations.json
    └── panoptic/    (panoptic annotations, if register_panoptic=True)
        ├── algorithmic_train_panoptic.json
        ├── algorithmic_test_panoptic.json  
        ├── manual_train_panoptic.json
        ├── manual_test_panoptic.json
        ├── algorithmic_train_panoptic_masks/
        ├── algorithmic_test_panoptic_masks/
        ├── manual_train_panoptic_masks/
        └── manual_test_panoptic_masks/
    
    Args:
        dataset_root: Path to unified dataset directory
        register_instance: Whether to register instance segmentation datasets (default: True)
        register_panoptic: Whether to register panoptic segmentation datasets (default: False)
    """
    
    print("🔄 Registering two-stage myotube datasets from unified directory...")
    print(f"   Dataset root: {dataset_root}")
    print(f"   Instance segmentation: {register_instance}")
    print(f"   Panoptic segmentation: {register_panoptic}")
    
    # Check if dataset root exists
    if not os.path.exists(dataset_root):
        print(f"⚠️  Warning: Dataset not found at {dataset_root}")
        print(f"   Please create the unified dataset structure")
        return
    
    # Common images directory for both stages
    images_dir = os.path.join(dataset_root, "images")
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    registered_datasets = []

    # ===== INSTANCE SEGMENTATION DATASETS =====
    if register_instance:
        print(f"\n📊 Registering Instance Segmentation Datasets")
        
        annotations_dir = os.path.join(dataset_root, "annotations")
        if not os.path.exists(annotations_dir):
            print(f"   ❌ Instance annotations directory not found: {annotations_dir}")
        else:
            registered_datasets.extend(_register_instance_datasets(annotations_dir, images_dir))
    
    # ===== PANOPTIC SEGMENTATION DATASETS =====
    if register_panoptic:
        print(f"\n🎭 Registering Panoptic Segmentation Datasets")
        print(f"   ⚠️ Skipped: Panoptic mode is disabled. Instance-only project.")
    
    # Metadata is now set directly during dataset registration
    
    print(f"\n✅ Two-stage dataset registration completed (Instance Only)!")
    print(f"   Stage 1: Algorithmic annotations for robust feature learning")
    print(f"   Stage 2: Manual annotations for precise fine-tuning")
    if register_instance:
        print(f"   📊 Instance datasets registered: {len(registered_datasets)}")
    if register_panoptic:
        print(f"   🎭 Panoptic segmentation requested but disabled.")
    print(f"   Classes: ['myotube']")


def register_all_myotube(_root="myotube_batch_output"):
    """Main registration function to be called from other scripts (instance only)."""
    register_two_stage_datasets(dataset_root=_root, register_instance=True, register_panoptic=False)


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
    import argparse
    
    parser = argparse.ArgumentParser(description="Register two-stage myotube datasets (instance only)")
    parser.add_argument("--dataset_root", default="myotube_batch_output", 
                       help="Root directory of unified dataset")
    parser.add_argument("--instance", action="store_true", default=True,
                       help="Register instance segmentation datasets (default: True)")
    # Keep panoptic flag for compatibility; it will be ignored with a notice
    parser.add_argument("--panoptic", action="store_true", default=False,
                       help="(Ignored) Panoptic registration is disabled")
    parser.add_argument("--both", action="store_true", 
                       help="(Ignored) Panoptic registration is disabled")
    
    args = parser.parse_args()
    
    # Handle --both flag
    if args.both or args.panoptic:
        print("⚠️  Panoptic registration flags detected but panoptic is disabled. Proceeding with instance only.")
    register_instance = True
    register_panoptic = False
    
    # Check for existing datasets
    found_paths = check_dataset_structure(args.dataset_root)
    
    # Determine dataset path
    dataset_path = args.dataset_root
    
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
    register_two_stage_datasets(
        dataset_root=dataset_path, 
        register_instance=register_instance,
        register_panoptic=register_panoptic
    ) 
