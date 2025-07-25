#!/usr/bin/env python3

"""
Two-Stage Myotube Training Script

This script implements two-stage training for myotube segmentation with support for both
instance and panoptic segmentation modes:
1. Stage 1: Train on algorithmic annotations (~100 images) for robust feature learning
2. Stage 2: Fine-tune on manual annotations (~2-5 images) for precise segmentation

Usage:
    # Instance segmentation (default)
    python train_two_stage.py --mode instance
    
    # Panoptic segmentation
    python train_two_stage.py --mode panoptic
    
    # Run only Stage 1
    python train_two_stage.py --stage 1 --mode panoptic
    
    # Run only Stage 2 (requires Stage 1 checkpoint)
    python train_two_stage.py --stage 2 --mode instance
    
    # Custom dataset paths
    python train_two_stage.py --dataset /path/to/unified/dataset --mode panoptic
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Import dataset registration
from register_two_stage_datasets import register_two_stage_datasets

# Import Mask2Former training
from train_net import main as train_main


def find_latest_checkpoint(output_dir: str) -> str:
    """
    Find the latest checkpoint in an output directory.
    
    Args:
        output_dir: Path to training output directory
        
    Returns:
        Path to latest checkpoint file
    """
    if not os.path.exists(output_dir):
        return ""
    
    # Look for model_final.pth first (completed training)
    final_model = os.path.join(output_dir, "model_final.pth")
    if os.path.exists(final_model):
        return final_model
    
    # Look for latest checkpoint
    checkpoint_pattern = os.path.join(output_dir, "model_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return ""
    
    # Sort by modification time and return the latest
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def count_dataset_images(dataset_root, stage, mode="instance"):
    """Count images in a dataset stage for specified mode."""
    import json
    
    if mode == "panoptic":
        # Panoptic annotations
        panoptic_dir = os.path.join(dataset_root, "panoptic")
        if stage == 1:
            ann_file = os.path.join(panoptic_dir, "algorithmic_train_panoptic.json")
        else:
            ann_file = os.path.join(panoptic_dir, "manual_train_panoptic.json")
    else:
        # Instance annotations
        annotations_dir = os.path.join(dataset_root, "annotations")
        if stage == 1:
            ann_file = os.path.join(annotations_dir, "algorithmic_train_annotations.json")
        else:
            ann_file = os.path.join(annotations_dir, "manual_train_annotations.json")
    
    if os.path.exists(ann_file):
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            if mode == "panoptic":
                # Count segments in panoptic format
                total_segments = sum(len(ann['segments_info']) for ann in data['annotations'])
                return len(data['images']), total_segments
            else:
                # Count annotations in instance format
                return len(data['images']), len(data['annotations'])
        except Exception:
            return 0, 0
    return 0, 0

def stage1_training(args, dataset_root, mode="instance"):
    """Execute Stage 1 training on algorithmic annotations."""
    num_images, num_annotations = count_dataset_images(dataset_root, 1, mode)
    
    # Determine config file and output directory based on mode
    if mode == "panoptic":
        config_file = "stage1_panoptic_config.yaml"
        output_dir = "./output_stage1_panoptic_algorithmic/"
    else:
        config_file = "stage1_config.yaml"
        output_dir = "./output_stage1_algorithmic/"
    
    print("🚀 STAGE 1: Training on Algorithmic Annotations")
    print("="*60)
    print(f"   Mode: {mode.title()} Segmentation")
    print(f"   Dataset: {num_images} images with {num_annotations} algorithmic annotations")
    print(f"   Purpose: Robust feature learning from large dataset")
    print(f"   Config: {config_file}")
    print(f"   Output: {output_dir}")
    print("="*60)
    
    # Prepare arguments for Stage 1
    stage1_args = argparse.Namespace(
        config_file=config_file,
        num_gpus=args.num_gpus,
        resume=args.resume,
        eval_only=False,
        opts=[]
    )
    
    # Start Stage 1 training
    print("🔄 Starting Stage 1 training...")
    try:
        train_main(stage1_args)
        print("✅ Stage 1 training completed successfully!")
        
        # Check if checkpoint was created
        stage1_checkpoint = find_latest_checkpoint(output_dir.rstrip('/'))
        if stage1_checkpoint:
            print(f"📄 Stage 1 checkpoint: {stage1_checkpoint}")
            return stage1_checkpoint
        else:
            print("⚠️  Warning: No Stage 1 checkpoint found")
            return ""
            
    except Exception as e:
        print(f"❌ Stage 1 training failed: {str(e)}")
        raise


def stage2_training(args, dataset_root, stage1_checkpoint: str = "", mode="instance"):
    """Execute Stage 2 fine-tuning on manual annotations."""
    num_images, num_annotations = count_dataset_images(dataset_root, 2, mode)
    
    # Determine config file and output directory based on mode
    if mode == "panoptic":
        config_file = "stage2_panoptic_config.yaml"
        output_dir = "./output_stage2_panoptic_manual/"
        stage1_default_dir = "./output_stage1_panoptic_algorithmic"
    else:
        config_file = "stage2_config.yaml"
        output_dir = "./output_stage2_manual/"
        stage1_default_dir = "./output_stage1_algorithmic"
    
    print("\n🎯 STAGE 2: Fine-tuning on Manual Annotations")
    print("="*60)
    print(f"   Mode: {mode.title()} Segmentation")
    print(f"   Dataset: {num_images} images with {num_annotations} manual annotations")
    print(f"   Purpose: Precise fine-tuning for high-quality segmentation")
    print(f"   Config: {config_file}")
    print(f"   Output: {output_dir}")
    
    # Determine checkpoint to use for Stage 2
    if not stage1_checkpoint:
        # Look for existing Stage 1 checkpoint
        stage1_checkpoint = find_latest_checkpoint(stage1_default_dir)
        
    if stage1_checkpoint:
        print(f"   Checkpoint: {stage1_checkpoint}")
    else:
        print("   ⚠️  No Stage 1 checkpoint found - using COCO pre-trained weights")
        stage1_checkpoint = "model_final_83d103.pkl"  # Updated to match configs
    
    print("="*60)
    
    # Update Stage 2 config with checkpoint path
    stage2_opts = [
        "MODEL.WEIGHTS", stage1_checkpoint
    ]
    
    # Prepare arguments for Stage 2
    stage2_args = argparse.Namespace(
        config_file=config_file,
        num_gpus=args.num_gpus,
        resume=args.resume,
        eval_only=False,
        opts=stage2_opts
    )
    
    # Start Stage 2 training
    print("🔄 Starting Stage 2 fine-tuning...")
    try:
        train_main(stage2_args)
        print("✅ Stage 2 fine-tuning completed successfully!")
        
        # Check final checkpoint
        stage2_checkpoint = find_latest_checkpoint(output_dir.rstrip('/'))
        if stage2_checkpoint:
            print(f"📄 Final model: {stage2_checkpoint}")
            return stage2_checkpoint
        else:
            print("⚠️  Warning: No Stage 2 checkpoint found")
            return ""
            
    except Exception as e:
        print(f"❌ Stage 2 training failed: {str(e)}")
        raise


def verify_datasets(dataset_root: str, mode="instance"):
    """Verify that unified dataset structure exists for specified mode."""
    print(f"🔍 Verifying unified dataset for {mode} mode...")
    
    issues = []
    
    # Check unified dataset structure
    if not os.path.exists(dataset_root):
        issues.append(f"Dataset root not found: {dataset_root}")
        
    images_dir = os.path.join(dataset_root, "images")
    if not os.path.exists(images_dir):
        issues.append(f"Missing images directory: {images_dir}")
    
    if mode == "panoptic":
        # Check panoptic structure
        panoptic_dir = os.path.join(dataset_root, "panoptic")
        if not os.path.exists(panoptic_dir):
            issues.append(f"Missing panoptic directory: {panoptic_dir}")
            issues.append("Run utils/convert_instance_to_panoptic.py to create panoptic annotations")
        else:
            # Check required panoptic files
            required_files = [
                "algorithmic_train_panoptic.json",
                "manual_train_panoptic.json"
            ]
            
            for file in required_files:
                file_path = os.path.join(panoptic_dir, file)
                if not os.path.exists(file_path):
                    issues.append(f"Missing panoptic file: {file}")
    else:
        # Check instance structure
        annotations_dir = os.path.join(dataset_root, "annotations")
        if not os.path.exists(annotations_dir):
            issues.append(f"Missing annotations directory: {annotations_dir}")
        else:
            # Check required annotation files
            required_files = [
                "algorithmic_train_annotations.json",
                "manual_train_annotations.json"
            ]
            
            for file in required_files:
                file_path = os.path.join(annotations_dir, file)
                if not os.path.exists(file_path):
                    issues.append(f"Missing annotation file: {file}")
    
    if issues:
        print("❌ Dataset verification failed:")
        for issue in issues:
            print(f"   • {issue}")
        print(f"\n💡 Expected unified structure for {mode} mode:")
        print(f"   {dataset_root}/")
        print(f"   ├── images/")
        if mode == "panoptic":
            print(f"   └── panoptic/")
            print(f"       ├── algorithmic_train_panoptic.json")
            print(f"       ├── algorithmic_test_panoptic.json")
            print(f"       ├── manual_train_panoptic.json")
            print(f"       └── manual_test_panoptic.json")
        else:
            print(f"   └── annotations/")
            print(f"       ├── algorithmic_train_annotations.json")
            print(f"       ├── algorithmic_test_annotations.json")
            print(f"       ├── manual_train_annotations.json")
            print(f"       └── manual_test_annotations.json")
        return False
    
    print(f"✅ Unified dataset verified for {mode} mode!")
    return True


def main():
    """Main function for two-stage training."""
    parser = argparse.ArgumentParser(description="Two-stage myotube training with instance and panoptic support")
    
    # Training control
    parser.add_argument("--stage", type=int, choices=[1, 2], 
                       help="Run specific stage only (default: run both)")
    parser.add_argument("--mode", choices=["instance", "panoptic"], default="instance",
                       help="Segmentation mode: instance or panoptic (default: instance)")
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="Number of GPUs")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true",
                       help="Perform evaluation only")
    
    # Dataset paths
    parser.add_argument("--dataset", default="myotube_batch_output",
                       help="Path to unified dataset directory")
    
    args = parser.parse_args()
    
    # Count datasets for display
    stage1_images, stage1_annotations = count_dataset_images(args.dataset, 1, args.mode)
    stage2_images, stage2_annotations = count_dataset_images(args.dataset, 2, args.mode)
    
    print("🎭 Mask2Former Two-Stage Myotube Training")
    print("="*60)
    print(f"   Unified dataset: {args.dataset}")
    print(f"   Segmentation mode: {args.mode.title()}")
    print(f"   Stage 1: {stage1_images} images, {stage1_annotations} algorithmic annotations")
    print(f"   Stage 2: {stage2_images} images, {stage2_annotations} manual annotations")
    print(f"   GPUs: {args.num_gpus}")
    
    if args.eval_only:
        print("   Mode: Evaluation only")
    else:
        if args.stage:
            print(f"   Training: Stage {args.stage} only")
        else:
            print("   Training: Full two-stage training")
    print("="*60)
    
    # Register datasets based on mode
    print(f"\n🔄 Registering {args.mode} datasets...")
    if args.mode == "panoptic":
        register_two_stage_datasets(args.dataset, register_instance=False, register_panoptic=True)
    else:
        register_two_stage_datasets(args.dataset, register_instance=True, register_panoptic=False)
    
    # Verify datasets exist
    if not args.eval_only:
        if not verify_datasets(args.dataset, args.mode):
            return 1
    
    try:
        stage1_checkpoint = ""
        
        # Execute training stages
        if args.eval_only:
            print("\n📊 Evaluation mode - please run specific evaluation commands")
            if args.mode == "panoptic":
                print("   Stage 1: python train_net.py --config-file stage1_panoptic_config.yaml --eval-only")
                print("   Stage 2: python train_net.py --config-file stage2_panoptic_config.yaml --eval-only")
            else:
                print("   Stage 1: python train_net.py --config-file stage1_config.yaml --eval-only")
                print("   Stage 2: python train_net.py --config-file stage2_config.yaml --eval-only")
            return 0
        
        if args.stage is None or args.stage == 1:
            # Run Stage 1
            stage1_checkpoint = stage1_training(args, args.dataset, args.mode)
        
        if args.stage is None or args.stage == 2:
            # Run Stage 2
            stage2_checkpoint = stage2_training(args, args.dataset, stage1_checkpoint, args.mode)
        
        # Training summary
        print("\n" + "="*60)
        print(f"🎉 TWO-STAGE {args.mode.upper()} TRAINING COMPLETED!")
        print("="*60)
        
        # Determine output directories based on mode
        if args.mode == "panoptic":
            stage1_output = './output_stage1_panoptic_algorithmic'
            stage2_output = './output_stage2_panoptic_manual'
        else:
            stage1_output = './output_stage1_algorithmic'
            stage2_output = './output_stage2_manual'
        
        if args.stage is None:
            print("✅ Both stages completed successfully")
            print(f"   Stage 1 checkpoint: {find_latest_checkpoint(stage1_output)}")
            print(f"   Stage 2 checkpoint: {find_latest_checkpoint(stage2_output)}")
            print(f"\n🎯 Final model ready for inference!")
            print(f"   Best model: {stage2_output}/model_final.pth")
        elif args.stage == 1:
            print("✅ Stage 1 completed - ready for Stage 2")
            print(f"   Checkpoint: {find_latest_checkpoint(stage1_output)}")
            print(f"\n➡️  Next: python train_two_stage.py --stage 2 --mode {args.mode}")
        elif args.stage == 2:
            print("✅ Stage 2 completed - model ready for inference")
            print(f"   Final model: {find_latest_checkpoint(stage2_output)}")
        
        print("\n📋 Evaluation commands:")
        if args.mode == "panoptic":
            print("   Stage 1: python train_net.py --config-file stage1_panoptic_config.yaml --eval-only")
            print("   Stage 2: python train_net.py --config-file stage2_panoptic_config.yaml --eval-only")
        else:
            print("   Stage 1: python train_net.py --config-file stage1_config.yaml --eval-only")
            print("   Stage 2: python train_net.py --config-file stage2_config.yaml --eval-only")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 