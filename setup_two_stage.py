#!/usr/bin/env python3

"""
Setup Verification Script for Two-Stage Myotube Training

This script verifies that all necessary components are in place for 
two-stage myotube training with Mask2Former.

Features:
- Checks for unified dataset with 4 annotation files
- Automatically counts images and annotations in each stage
- Verifies configuration files
- Tests dataset registration
- Validates dependencies
- Provides setup guidance
"""

import os
import sys
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def count_dataset_images(annotation_file):
    """Count images and annotations in a COCO annotation file."""
    if not os.path.exists(annotation_file):
        return 0, 0
    
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        return len(data['images']), len(data['annotations'])
    except Exception:
        return 0, 0

def check_unified_dataset_structure(dataset_root):
    """Verify unified dataset has proper structure with 4 annotation files"""
    print(f"\n🔍 Checking unified dataset structure...")
    
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset root not found: {dataset_root}")
        return False, {}, {}
    
    # Check required directories
    images_dir = os.path.join(dataset_root, "images")
    annotations_dir = os.path.join(dataset_root, "annotations")
    
    structure_ok = True
    structure_ok &= check_file_exists(images_dir, "Images directory")
    structure_ok &= check_file_exists(annotations_dir, "Annotations directory")
    
    # Check for 4 annotation files and count their contents
    annotation_files = {
        "algorithmic_train": "algorithmic_train_annotations.json",
        "algorithmic_test": "algorithmic_test_annotations.json", 
        "manual_train": "manual_train_annotations.json",
        "manual_test": "manual_test_annotations.json"
    }
    
    algo_stats = {}
    manual_stats = {}
    
    for file_type, filename in annotation_files.items():
        file_path = os.path.join(annotations_dir, filename)
        exists = check_file_exists(file_path, f"{file_type.replace('_', ' ').title()} annotations")
        
        if file_type == "algorithmic_train" and not exists:
            structure_ok = False
        if file_type == "manual_train" and not exists:
            structure_ok = False
        
        # Count images and annotations for all existing files
        if exists:
            num_images, num_annotations = count_dataset_images(file_path)
            
            if "algorithmic" in file_type:
                if "train" in file_type:
                    algo_stats['train_images'] = num_images
                    algo_stats['train_annotations'] = num_annotations
                else:
                    algo_stats['test_images'] = num_images
                    algo_stats['test_annotations'] = num_annotations
            else:
                if "train" in file_type:
                    manual_stats['train_images'] = num_images
                    manual_stats['train_annotations'] = num_annotations
                else:
                    manual_stats['test_images'] = num_images
                    manual_stats['test_annotations'] = num_annotations
            
            print(f"      📊 {num_images} images, {num_annotations} annotations")
    
    # Print dataset summaries
    if algo_stats:
        train_imgs = algo_stats.get('train_images', 0)
        train_anns = algo_stats.get('train_annotations', 0)
        test_imgs = algo_stats.get('test_images', 0)
        test_anns = algo_stats.get('test_annotations', 0)
        
        print(f"\n   📊 Algorithmic Dataset Summary:")
        print(f"      Training: {train_imgs} images, {train_anns} annotations")
        if test_imgs > 0:
            print(f"      Test: {test_imgs} images, {test_anns} annotations")
        else:
            print(f"      Test: Using training data for validation")
        
        avg_anns = train_anns / train_imgs if train_imgs > 0 else 0
        print(f"      Average: {avg_anns:.1f} annotations per image")
    
    if manual_stats:
        train_imgs = manual_stats.get('train_images', 0)
        train_anns = manual_stats.get('train_annotations', 0)
        test_imgs = manual_stats.get('test_images', 0)
        test_anns = manual_stats.get('test_annotations', 0)
        
        print(f"\n   🎯 Manual Dataset Summary:")
        print(f"      Training: {train_imgs} images, {train_anns} annotations")
        if test_imgs > 0:
            print(f"      Test: {test_imgs} images, {test_anns} annotations")
        else:
            print(f"      Test: Using training data for validation")
        
        avg_anns = train_anns / train_imgs if train_imgs > 0 else 0
        print(f"      Average: {avg_anns:.1f} annotations per image")
    
    # Count total images in images directory
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        print(f"\n   📁 Total images in directory: {len(image_files)}")
    
    return structure_ok, algo_stats, manual_stats

def check_config_files():
    """Check two-stage configuration files"""
    print("\n⚙️  Checking configuration files...")
    
    config_ok = True
    config_ok &= check_file_exists("stage1_config.yaml", "Stage 1 config")
    config_ok &= check_file_exists("stage2_config.yaml", "Stage 2 config")
    config_ok &= check_file_exists("register_two_stage_datasets.py", "Dataset registration script")
    config_ok &= check_file_exists("train_two_stage.py", "Two-stage training script")
    
    return config_ok

def check_mask2former_setup():
    """Verify Mask2Former components are available"""
    print("\n🎭 Checking Mask2Former setup...")
    
    m2f_ok = True
    m2f_ok &= check_file_exists("train_net.py", "Base training script")
    m2f_ok &= check_file_exists("configs", "Configs directory")
    m2f_ok &= check_file_exists("configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml", "Base Swin config")
    m2f_ok &= check_file_exists("model_final_54b88a.pkl", "Pre-trained model")
    
    if m2f_ok:
        # Check model file size
        model_size = os.path.getsize("model_final_54b88a.pkl") / (1024*1024)
        print(f"   📊 Pre-trained model size: {model_size:.1f} MB")
    
    return m2f_ok

def check_python_environment():
    """Check Python dependencies"""
    print("\n🐍 Checking Python environment...")
    
    print(f"   Python version: {sys.version}")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("detectron2", "Detectron2"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("tqdm", "TQDM")
    ]
    
    missing_deps = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT INSTALLED")
            missing_deps.append(name)
    
    return len(missing_deps) == 0, missing_deps

def test_dataset_registration(dataset_root):
    """Test dataset registration functionality"""
    print("\n🧪 Testing dataset registration...")
    
    try:
        from register_two_stage_datasets import register_two_stage_datasets
        register_two_stage_datasets(dataset_root)
        print("   ✅ Dataset registration successful")
        return True
    except Exception as e:
        print(f"   ❌ Dataset registration failed: {str(e)}")
        return False

def suggest_dataset_creation():
    """Provide guidance for unified dataset creation"""
    print("\n💡 Unified Dataset Creation Guide:")
    print("="*50)
    
    print("\n📁 Expected Directory Structure:")
    print("   myotube_dataset/")
    print("   ├── images/                              # All images (algorithmic + manual)")
    print("   └── annotations/")
    print("       ├── algorithmic_train_annotations.json  # ~100 images with automatic annotations")
    print("       ├── algorithmic_test_annotations.json   # Optional test split")
    print("       ├── manual_train_annotations.json       # ~5 images with manual annotations")
    print("       └── manual_test_annotations.json        # Optional test split")
    
    print("\n📊 Step 1: Create Algorithmic Annotations")
    print("   1. Prepare your raw microscopy images")
    print("   2. Run batch processing:")
    print("      cd utils")
    print("      python batch_myotube_processing.py \\")
    print("        --input_dir /path/to/raw/images \\")
    print("        --output_dir ../temp_algorithmic \\")
    print("        --resolution 1500")
    print("   3. Organize output files:")
    print("      mkdir -p myotube_dataset/{images,annotations}")
    print("      mv temp_algorithmic/annotations/train_annotations.json myotube_dataset/annotations/algorithmic_train_annotations.json")
    print("      mv temp_algorithmic/annotations/test_annotations.json myotube_dataset/annotations/algorithmic_test_annotations.json")
    print("      cp temp_algorithmic/images/* myotube_dataset/images/")
    
    print("\n🎯 Step 2: Create Manual Annotations")
    print("   1. Select 5 representative images from myotube_dataset/images/")
    print("   2. Create high-quality manual annotations using:")
    print("      - CVAT (Computer Vision Annotation Tool)")
    print("      - LabelMe")
    print("      - VGG Image Annotator (VIA)")
    print("   3. Export in COCO format as:")
    print("      - manual_train_annotations.json")
    print("      - manual_test_annotations.json (optional)")
    print("   4. Place in myotube_dataset/annotations/")

def print_next_steps(all_ok, algo_stats, manual_stats, missing_deps):
    """Print next steps based on setup status"""
    print("\n" + "="*60)
    
    if all_ok:
        print("🎉 TWO-STAGE SETUP COMPLETE!")
        print("="*60)
        print("\n📋 Ready for training:")
        print("   # Run both stages")
        print("   python train_two_stage.py")
        print()
        print("   # Run Stage 1 only")
        print("   python train_two_stage.py --stage 1")
        print()
        print("   # Run Stage 2 only (after Stage 1)")
        print("   python train_two_stage.py --stage 2")
        
        print(f"\n📊 Dataset Summary:")
        if algo_stats:
            train_imgs = algo_stats.get('train_images', 0)
            train_anns = algo_stats.get('train_annotations', 0)
            print(f"   Stage 1: {train_imgs} images, {train_anns} annotations")
        if manual_stats:
            train_imgs = manual_stats.get('train_images', 0)
            train_anns = manual_stats.get('train_annotations', 0)
            print(f"   Stage 2: {train_imgs} images, {train_anns} annotations")
        
        print(f"\n🎯 Training Strategy:")
        print(f"   1. Stage 1 trains for 4000 iterations on algorithmic data")
        print(f"   2. Stage 2 fine-tunes for 1500 iterations on manual data")
        print(f"   3. Final model combines robust features + precise annotations")
        
    else:
        print("⚠️  SETUP INCOMPLETE - Please fix the issues above")
        
        if missing_deps:
            print(f"\n📦 Install missing dependencies:")
            print(f"   pip install {' '.join(dep.lower() for dep in missing_deps)}")
        
        if not algo_stats:
            print(f"\n📊 Create algorithmic dataset:")
            print(f"   cd utils && python batch_myotube_processing.py --input_dir /path/to/images --output_dir ../temp_algorithmic --resolution 1500")
        
        if not manual_stats:
            print(f"\n🎯 Create manual dataset:")
            print(f"   Use annotation tools to create manual annotations in COCO format")

def main():
    """Main setup verification function"""
    print("🔍 Two-Stage Myotube Training Setup Verification")
    print("="*60)
    
    # Define unified dataset path
    dataset_root = "myotube_dataset"
    
    # Check if myotube_batch_output exists (legacy dataset)
    if os.path.exists("myotube_batch_output") and not os.path.exists(dataset_root):
        print("   ℹ️  Found myotube_batch_output - can convert to unified structure")
        dataset_root = "myotube_batch_output"
    
    # Run all checks
    dataset_ok, algo_stats, manual_stats = check_unified_dataset_structure(dataset_root)
    config_ok = check_config_files()
    m2f_ok = check_mask2former_setup()
    env_ok, missing_deps = check_python_environment()
    
    # Test dataset registration if everything looks good
    registration_ok = False
    if config_ok and env_ok:
        registration_ok = test_dataset_registration(dataset_root)
    
    # Overall status
    all_ok = dataset_ok and config_ok and m2f_ok and env_ok and registration_ok
    
    # Provide guidance
    if not all_ok:
        suggest_dataset_creation()
    
    print_next_steps(all_ok, algo_stats, manual_stats, missing_deps)
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 