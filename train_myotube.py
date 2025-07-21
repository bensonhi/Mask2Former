#!/usr/bin/env python3

"""
Training script for myotube segmentation using Mask2Former
"""

import os
import sys

# Import dataset registration
from register_myotube_dataset import register_myotube_dataset

# Import Mask2Former training script
from train_net import main as train_main
import argparse

def main():
    """Main training function"""
    
    # Register the dataset first
    print("🔄 Registering myotube dataset...")
    register_myotube_dataset()
    
    # Set up arguments for training
    parser = argparse.ArgumentParser(description="Train Mask2Former on myotube dataset")
    parser.add_argument("--config-file", 
                       default="myotube_config.yaml",
                       help="path to config file")
    parser.add_argument("--num-gpus", 
                       type=int, 
                       default=1,
                       help="number of gpus")
    parser.add_argument("--resume", 
                       action="store_true",
                       help="resume from checkpoint")
    parser.add_argument("--eval-only", 
                       action="store_true",
                       help="perform evaluation only")
    parser.add_argument("opts",
                       help="Modify config options using the command-line",
                       default=None,
                       nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # Print configuration info
    print("🚀 Starting Mask2Former training...")
    print(f"   Config file: {args.config_file}")
    print(f"   Number of GPUs: {args.num_gpus}")
    print(f"   Dataset: myotube (1 class)")
    
    # Start training
    train_main(args)

if __name__ == "__main__":
    main() 