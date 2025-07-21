#!/usr/bin/env python3

"""
Setup script to verify environment and dataset for Mask2Former finetuning
"""

import os
import sys
import json

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def check_dataset():
    """Verify dataset structure"""
    print("\n📁 Checking dataset structure...")
    
    # Check main directories
    dataset_ok = True
    dataset_ok &= check_file_exists("myotube_batch_output", "Dataset directory")
    dataset_ok &= check_file_exists("myotube_batch_output/images", "Images directory") 
    dataset_ok &= check_file_exists("myotube_batch_output/annotations", "Annotations directory")
    dataset_ok &= check_file_exists("myotube_batch_output/annotations/instances_train.json", "Training annotations")
    
    if dataset_ok:
        # Count images and annotations
        try:
            with open("myotube_batch_output/annotations/instances_train.json", 'r') as f:
                data = json.load(f)
            print(f"   📊 Images: {len(data['images'])}")
            print(f"   📊 Annotations: {len(data['annotations'])}")
            print(f"   📊 Categories: {len(data['categories'])}")
            for cat in data['categories']:
                print(f"      - ID: {cat['id']}, Name: {cat['name']}")
        except Exception as e:
            print(f"   ⚠️  Could not parse annotation file: {e}")
    
    return dataset_ok

def check_mask2former():
    """Verify Mask2Former setup"""
    print("\n🎭 Checking Mask2Former setup...")
    
    # Check Mask2Former directory and files (we're now inside Mask2Former)
    m2f_ok = True
    m2f_ok &= check_file_exists("train_net.py", "Training script")
    m2f_ok &= check_file_exists("configs", "Config directory")
    m2f_ok &= check_file_exists("configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml", "Base config")
    m2f_ok &= check_file_exists("model_final_54b88a.pkl", "Pre-trained model")
    
    if m2f_ok:
        # Check model file size
        model_size = os.path.getsize("model_final_54b88a.pkl") / (1024*1024)
        print(f"   📊 Model size: {model_size:.1f} MB")
    
    return m2f_ok

def check_config_files():
    """Check if our custom config files are created"""
    print("\n⚙️  Checking custom configuration files...")
    
    config_ok = True
    config_ok &= check_file_exists("myotube_config.yaml", "Custom config file")
    config_ok &= check_file_exists("register_myotube_dataset.py", "Dataset registration script")
    config_ok &= check_file_exists("train_myotube.py", "Training script")
    
    return config_ok

def check_python_environment():
    """Check Python and basic dependencies"""
    print("\n🐍 Checking Python environment...")
    
    print(f"   Python version: {sys.version}")
    
    # Try importing key dependencies
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"), 
        ("detectron2", "Detectron2"),
        ("cv2", "OpenCV")
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

def print_next_steps(all_ok, missing_deps):
    """Print next steps based on setup status"""
    print("\n" + "="*60)
    
    if all_ok:
        print("🎉 SETUP COMPLETE! Ready for training.")
        print("\n📋 Next steps:")
        print("1. Run training:")
        print("   python train_myotube.py")
        print("\n2. Monitor training:")
        print("   - Check ./output_myotube/ for logs and checkpoints")
        print("   - Training will run for 5000 iterations")
        print("\n3. Evaluate model:")
        print("   python train_myotube.py --eval-only")
        
    else:
        print("⚠️  SETUP INCOMPLETE. Please fix the issues above.")
        
        if missing_deps:
            print(f"\n📦 Install missing dependencies:")
            print(f"   pip install {' '.join(dep.lower() for dep in missing_deps)}")
            
        print("\n🔧 Additional setup may be required:")
        print("   - Follow Mask2Former installation instructions")
        print("   - Ensure GPU drivers and CUDA are properly installed")

def main():
    """Main setup verification function"""
    print("🔍 Mask2Former Finetuning Setup Verification")
    print("="*60)
    
    # Run all checks
    dataset_ok = check_dataset()
    m2f_ok = check_mask2former()
    config_ok = check_config_files() 
    env_ok, missing_deps = check_python_environment()
    
    # Overall status
    all_ok = dataset_ok and m2f_ok and config_ok and env_ok
    
    print_next_steps(all_ok, missing_deps)
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 