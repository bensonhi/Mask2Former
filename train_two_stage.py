#!/usr/bin/env python3

"""
Two-Stage Myotube Training Script (Instance Only)

This script implements two-stage training for myotube instance segmentation:
1. Stage 1: Train on algorithmic annotations (~100 images) for robust feature learning
2. Stage 2: Fine-tune on manual annotations (~2-5 images) for precise segmentation

Usage:
    # Full two-stage training
    python train_two_stage.py
    
    # Run only Stage 1
    python train_two_stage.py --stage 1
    
    # Run only Stage 2 (requires Stage 1 checkpoint)
    python train_two_stage.py --stage 2
    
    # Custom dataset paths
    python train_two_stage.py --dataset /path/to/unified/dataset
    
    # Custom config files
    python train_two_stage.py --stage1-config custom_stage1.yaml --stage2-config custom_stage2.yaml
"""

import os
import sys
import argparse
import glob

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


def find_best_checkpoint(output_dir: str) -> str:
    """
    Find the best checkpoint based on evaluation metrics.
    
    Args:
        output_dir: Path to training output directory
        
    Returns:
        Path to best checkpoint file, or latest if best not found
    """
    if not os.path.exists(output_dir):
        return ""
    
    # Look for best checkpoint files first
    best_patterns = [
        "model_best.pth",  # Common best checkpoint name
        "best_model.pth",
        "model_best_*.pth"
    ]
    
    for pattern in best_patterns:
        best_files = glob.glob(os.path.join(output_dir, pattern))
        if best_files:
            # Sort by modification time and return the latest best
            best_files.sort(key=os.path.getmtime, reverse=True)
            print(f"   ✅ Found best checkpoint: {best_files[0]}")
            return best_files[0]
    
    # Fallback to latest checkpoint
    latest = find_latest_checkpoint(output_dir)
    if latest:
        print(f"   ⚠️  No best checkpoint found, using latest: {latest}")
    return latest


def count_dataset_images(dataset_root, stage):
    """Count images and annotations for a dataset stage (instance only)."""
    import json
    annotations_dir = os.path.join(dataset_root, "annotations")
    if stage == 1:
        ann_file = os.path.join(annotations_dir, "algorithmic_train_annotations.json")
    else:
        ann_file = os.path.join(annotations_dir, "manual_train_annotations.json")

    if os.path.exists(ann_file):
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
            return len(data['images']), len(data['annotations'])
        except Exception:
            return 0, 0
    return 0, 0

def stage1_training(args, dataset_root):
    """Execute Stage 1 training on algorithmic annotations (instance only)."""
    num_images, num_annotations = count_dataset_images(dataset_root, 1)
    
    config_file = args.stage1_config if args.stage1_config else "stage1_config.yaml"
    output_dir = "./output_stage1_algorithmic/"
    
    print("🚀 STAGE 1: Training on Algorithmic Annotations (Instance)")
    print("="*60)
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


def stage2_training(args, dataset_root, stage1_checkpoint: str = ""):
    """Execute Stage 2 fine-tuning on manual annotations (instance only)."""
    num_images, num_annotations = count_dataset_images(dataset_root, 2)
    
    config_file = args.stage2_config if args.stage2_config else "stage2_config.yaml"
    output_dir = "./output_stage2_manual/"
    stage1_default_dir = "./output_stage1_algorithmic"
    
    print("\n🎯 STAGE 2: Fine-tuning on Manual Annotations (Instance)")
    print("="*60)
    print(f"   Dataset: {num_images} images with {num_annotations} manual annotations")
    print(f"   Purpose: Precise fine-tuning for high-quality segmentation")
    print(f"   Config: {config_file}")
    print(f"   Output: {output_dir}")
    
    # Determine checkpoint to use for Stage 2
    if not stage1_checkpoint:
        # Look for best Stage 1 checkpoint first, fallback to latest
        stage1_checkpoint = find_best_checkpoint(stage1_default_dir)
        
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


def verify_datasets(dataset_root: str):
    """Verify that unified dataset structure exists (instance only)."""
    print(f"🔍 Verifying unified dataset (instance mode)...")
    
    issues = []
    
    # Check unified dataset structure
    if not os.path.exists(dataset_root):
        issues.append(f"Dataset root not found: {dataset_root}")
        
    images_dir = os.path.join(dataset_root, "images")
    if not os.path.exists(images_dir):
        issues.append(f"Missing images directory: {images_dir}")
    
    # Instance structure only
    annotations_dir = os.path.join(dataset_root, "annotations")
    if not os.path.exists(annotations_dir):
        issues.append(f"Missing annotations directory: {annotations_dir}")
    else:
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
        print(f"\n💡 Expected unified structure (instance mode):")
        print(f"   {dataset_root}/")
        print(f"   ├── images/")
        print(f"   └── annotations/")
        print(f"       ├── algorithmic_train_annotations.json")
        print(f"       ├── algorithmic_test_annotations.json")
        print(f"       ├── manual_train_annotations.json")
        print(f"       └── manual_test_annotations.json")
        return False
    
    print(f"✅ Unified dataset verified (instance)!")
    return True


def main():
    """Main function for two-stage training."""
    parser = argparse.ArgumentParser(description="Two-stage myotube training (instance only)")
    
    # Training control
    parser.add_argument("--stage", type=int, choices=[1, 2], 
                       help="Run specific stage only (default: run both)")
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="Number of GPUs")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true",
                       help="Perform evaluation only")
    
    # Dataset paths
    parser.add_argument("--dataset", default="myotube_batch_output",
                       help="Path to unified dataset directory")
    
    # Config file overrides
    parser.add_argument("--stage1-config", default=None,
                       help="Custom config file for Stage 1 (overrides default)")
    parser.add_argument("--stage2-config", default=None,
                       help="Custom config file for Stage 2 (overrides default)")
    
    args = parser.parse_args()
    
    # Count datasets for display
    stage1_images, stage1_annotations = count_dataset_images(args.dataset, 1)
    stage2_images, stage2_annotations = count_dataset_images(args.dataset, 2)
    
    print("🎭 Mask2Former Two-Stage Myotube Training (Instance)")
    print("="*60)
    print(f"   Unified dataset: {args.dataset}")
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
    print(f"\n🔄 Registering instance datasets...")
    register_two_stage_datasets(args.dataset, register_instance=True, register_panoptic=False)
    
    # Verify datasets exist
    if not args.eval_only:
        if not verify_datasets(args.dataset):
            return 1
    
    try:
        stage1_checkpoint = ""
        
        # Execute training stages
        if args.eval_only:
            print("\n📊 Evaluation mode - run:")
            stage1_config = args.stage1_config if args.stage1_config else "stage1_config.yaml"
            stage2_config = args.stage2_config if args.stage2_config else "stage2_config.yaml"
            print(f"   Stage 1: python train_net.py --config-file {stage1_config} --eval-only")
            print(f"   Stage 2: python train_net.py --config-file {stage2_config} --eval-only")
            return 0
        
        if args.stage is None or args.stage == 1:
            # Run Stage 1
            stage1_checkpoint = stage1_training(args, args.dataset)
        
        if args.stage is None or args.stage == 2:
            # Run Stage 2
            stage2_checkpoint = stage2_training(args, args.dataset, stage1_checkpoint)
        
        # Training summary
        print("\n" + "="*60)
        print(f"🎉 TWO-STAGE INSTANCE TRAINING COMPLETED!")
        print("="*60)
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
            print(f"\n➡️  Next: python train_two_stage.py --stage 2")
        elif args.stage == 2:
            print("✅ Stage 2 completed - model ready for inference")
            print(f"   Final model: {find_latest_checkpoint(stage2_output)}")
        
        print("\n📋 Evaluation commands:")
        stage1_config = args.stage1_config if args.stage1_config else "stage1_config.yaml"
        stage2_config = args.stage2_config if args.stage2_config else "stage2_config.yaml"
        print(f"   Stage 1: python train_net.py --config-file {stage1_config} --eval-only")
        print(f"   Stage 2: python train_net.py --config-file {stage2_config} --eval-only")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
