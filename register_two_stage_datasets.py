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
import json
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.file_io import PathManager

def load_myotube_panoptic_json(json_file, image_dir, gt_dir, meta):
    """
    Load myotube panoptic JSON and convert to Detectron2 format.
    """
    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            # For myotube, we only have things, no stuff
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]
        # Find corresponding image info
        image_info = None
        for img in json_info["images"]:
            if img["id"] == image_id:
                image_info = img
                break
        
        if image_info is None:
            continue
            
        image_file = os.path.join(image_dir, image_info["file_name"])
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    
    return ret

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
    """Register panoptic segmentation datasets."""
    registered = []
    
    # Define metadata for myotube panoptic segmentation
    # Category ID 1 = myotube (thing), Category ID 0 = background (stuff)
    panoptic_metadata = {
        "thing_classes": ["myotube"],
        "stuff_classes": ["background"],
        "thing_dataset_id_to_contiguous_id": {1: 0},  # Map category 1 -> 0 
        "stuff_dataset_id_to_contiguous_id": {0: 0},  # Map category 0 -> 0
    }
    
    # Note: We pass the corresponding instance JSON files as the 6th parameter
    # This is required for COCOEvaluator during training evaluation
    
    # Stage 1: Algorithmic panoptic annotations
    print(f"   📊 Stage 1: Algorithmic Panoptic Annotations")
    stage1_train_json = os.path.join(panoptic_dir, "algorithmic_train_panoptic.json")
    stage1_train_masks = os.path.join(panoptic_dir, "algorithmic_train_panoptic_masks")
    stage1_val_json = os.path.join(panoptic_dir, "algorithmic_test_panoptic.json")
    stage1_val_masks = os.path.join(panoptic_dir, "algorithmic_test_panoptic_masks")
    
    if os.path.exists(stage1_train_json) and os.path.exists(stage1_train_masks):
        # Get corresponding instance JSON for evaluation
        instance_train_json = os.path.join(os.path.dirname(panoptic_dir), "annotations", "algorithmic_train_annotations.json")
        
        # Register dataset manually
        DatasetCatalog.register(
            "myotube_stage1_panoptic_train",
            lambda: load_myotube_panoptic_json(stage1_train_json, images_dir, stage1_train_masks, panoptic_metadata)
        )
        MetadataCatalog.get("myotube_stage1_panoptic_train").set(
            panoptic_root=stage1_train_masks,
            image_root=images_dir,
            panoptic_json=stage1_train_json,
            json_file=instance_train_json,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **panoptic_metadata,
        )
        
        registered.append("myotube_stage1_panoptic_train")
        print(f"      ✅ Registered myotube_stage1_panoptic_train")
        
        # Count images and annotations
        with open(stage1_train_json, 'r') as f:
            data = json.load(f)
        print(f"      📈 Training images: {len(data['images'])}")
        print(f"      📈 Training annotations: {len(data['annotations'])}")
    else:
        print(f"      ❌ Algorithmic panoptic train files not found")
    
    if os.path.exists(stage1_val_json) and os.path.exists(stage1_val_masks):
        instance_val_json = os.path.join(os.path.dirname(panoptic_dir), "annotations", "algorithmic_test_annotations.json")
        
        # Register validation dataset manually
        DatasetCatalog.register(
            "myotube_stage1_panoptic_val",
            lambda: load_myotube_panoptic_json(stage1_val_json, images_dir, stage1_val_masks, panoptic_metadata)
        )
        MetadataCatalog.get("myotube_stage1_panoptic_val").set(
            panoptic_root=stage1_val_masks,
            image_root=images_dir,
            panoptic_json=stage1_val_json,
            json_file=instance_val_json,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **panoptic_metadata,
        )
        
        registered.append("myotube_stage1_panoptic_val")
        print(f"      ✅ Registered myotube_stage1_panoptic_val")
    else:
        print(f"      ⚠️  Algorithmic panoptic val files not found, using train for validation")
        if os.path.exists(stage1_train_json) and os.path.exists(stage1_train_masks):
            instance_train_json = os.path.join(os.path.dirname(panoptic_dir), "annotations", "algorithmic_train_annotations.json")
            
            # Register validation dataset manually (using train data)
            DatasetCatalog.register(
                "myotube_stage1_panoptic_val",
                lambda: load_myotube_panoptic_json(stage1_train_json, images_dir, stage1_train_masks, panoptic_metadata)
            )
            MetadataCatalog.get("myotube_stage1_panoptic_val").set(
                panoptic_root=stage1_train_masks,
                image_root=images_dir,
                panoptic_json=stage1_train_json,
                json_file=instance_train_json,
                evaluator_type="coco_panoptic_seg",
                ignore_label=255,
                label_divisor=1000,
                **panoptic_metadata,
            )
            
            registered.append("myotube_stage1_panoptic_val")
    
    # Stage 2: Manual panoptic annotations
    print(f"   🎯 Stage 2: Manual Panoptic Annotations")
    stage2_train_json = os.path.join(panoptic_dir, "manual_train_panoptic.json")
    stage2_train_masks = os.path.join(panoptic_dir, "manual_train_panoptic_masks")
    stage2_val_json = os.path.join(panoptic_dir, "manual_test_panoptic.json")
    stage2_val_masks = os.path.join(panoptic_dir, "manual_test_panoptic_masks")
    
    if os.path.exists(stage2_train_json) and os.path.exists(stage2_train_masks):
        instance_train_json = os.path.join(os.path.dirname(panoptic_dir), "annotations", "manual_train_annotations.json")
        
        # Register stage 2 train dataset manually
        DatasetCatalog.register(
            "myotube_stage2_panoptic_train",
            lambda: load_myotube_panoptic_json(stage2_train_json, images_dir, stage2_train_masks, panoptic_metadata)
        )
        MetadataCatalog.get("myotube_stage2_panoptic_train").set(
            panoptic_root=stage2_train_masks,
            image_root=images_dir,
            panoptic_json=stage2_train_json,
            json_file=instance_train_json,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **panoptic_metadata,
        )
        
        registered.append("myotube_stage2_panoptic_train")
        print(f"      ✅ Registered myotube_stage2_panoptic_train")
        
        # Count images and annotations
        with open(stage2_train_json, 'r') as f:
            data = json.load(f)
        print(f"      📈 Training images: {len(data['images'])}")
        print(f"      📈 Training annotations: {len(data['annotations'])}")
    else:
        print(f"      ❌ Manual panoptic train files not found")
    
    if os.path.exists(stage2_val_json) and os.path.exists(stage2_val_masks):
        instance_val_json = os.path.join(os.path.dirname(panoptic_dir), "annotations", "manual_test_annotations.json")
        
        # Register stage 2 val dataset manually
        DatasetCatalog.register(
            "myotube_stage2_panoptic_val",
            lambda: load_myotube_panoptic_json(stage2_val_json, images_dir, stage2_val_masks, panoptic_metadata)
        )
        MetadataCatalog.get("myotube_stage2_panoptic_val").set(
            panoptic_root=stage2_val_masks,
            image_root=images_dir,
            panoptic_json=stage2_val_json,
            json_file=instance_val_json,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **panoptic_metadata,
        )
        
        registered.append("myotube_stage2_panoptic_val")
        print(f"      ✅ Registered myotube_stage2_panoptic_val")
    else:
        print(f"      ⚠️  Manual panoptic val files not found, using train for validation")
        if os.path.exists(stage2_train_json) and os.path.exists(stage2_train_masks):
            instance_train_json = os.path.join(os.path.dirname(panoptic_dir), "annotations", "manual_train_annotations.json")
            
            # Register stage 2 val dataset manually (using train data)
            DatasetCatalog.register(
                "myotube_stage2_panoptic_val",
                lambda: load_myotube_panoptic_json(stage2_train_json, images_dir, stage2_train_masks, panoptic_metadata)
            )
            MetadataCatalog.get("myotube_stage2_panoptic_val").set(
                panoptic_root=stage2_train_masks,
                image_root=images_dir,
                panoptic_json=stage2_train_json,
                json_file=instance_train_json,
                evaluator_type="coco_panoptic_seg",
                ignore_label=255,
                label_divisor=1000,
                **panoptic_metadata,
            )
            
            registered.append("myotube_stage2_panoptic_val")
    
    return registered

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
        
        panoptic_dir = os.path.join(dataset_root, "panoptic")
        if not os.path.exists(panoptic_dir):
            print(f"   ❌ Panoptic directory not found: {panoptic_dir}")
            print(f"   💡 Run utils/convert_instance_to_panoptic.py to create panoptic annotations")
        else:
            registered_datasets.extend(_register_panoptic_datasets(panoptic_dir, images_dir))
    
    # Metadata is now set directly during dataset registration
    
    print(f"\n✅ Two-stage dataset registration completed!")
    print(f"   Stage 1: Algorithmic annotations for robust feature learning")
    print(f"   Stage 2: Manual annotations for precise fine-tuning")
    if register_instance:
        print(f"   📊 Instance segmentation: {len([d for d in registered_datasets if 'panoptic' not in d])} datasets")
    if register_panoptic:
        print(f"   🎭 Panoptic segmentation: {len([d for d in registered_datasets if 'panoptic' in d])} datasets")
    print(f"   Classes: ['myotube'] + ['background'] for panoptic")

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
    
    parser = argparse.ArgumentParser(description="Register two-stage myotube datasets")
    parser.add_argument("--dataset_root", default="myotube_batch_output", 
                       help="Root directory of unified dataset")
    parser.add_argument("--instance", action="store_true", default=True,
                       help="Register instance segmentation datasets (default: True)")
    parser.add_argument("--panoptic", action="store_true", default=False,
                       help="Register panoptic segmentation datasets (default: False)")
    parser.add_argument("--both", action="store_true", 
                       help="Register both instance and panoptic datasets")
    
    args = parser.parse_args()
    
    # Handle --both flag
    if args.both:
        register_instance = True
        register_panoptic = True
    else:
        register_instance = args.instance
        register_panoptic = args.panoptic
    
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