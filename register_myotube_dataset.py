#!/usr/bin/env python3

import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def register_myotube_dataset():
    """
    Register the myotube dataset for training and validation.
    Since we only have one annotation file, we'll use it for both train and val.
    """
    
    # Path to your dataset
    dataset_root = "myotube_batch_output"
    
    # Register training dataset
    register_coco_instances(
        "myotube_train", 
        {}, 
        os.path.join(dataset_root, "annotations", "instances_train.json"), 
        os.path.join(dataset_root, "images")
    )
    
    # Since you only have training annotations, we'll use the same for validation
    # In practice, you should split your data or create separate validation annotations
    register_coco_instances(
        "myotube_val", 
        {}, 
        os.path.join(dataset_root, "annotations", "instances_train.json"), 
        os.path.join(dataset_root, "images")
    )
    
    # Set metadata for the datasets
    for dataset_name in ["myotube_train", "myotube_val"]:
        MetadataCatalog.get(dataset_name).set(
            thing_classes=["myotube"],  # Your class name
            evaluator_type="coco",      # Use COCO evaluator for instance segmentation
        )
    
    print("✅ Myotube dataset registered successfully!")
    print("  - myotube_train: Training dataset")
    print("  - myotube_val: Validation dataset")
    print("  - Classes: ['myotube']")

if __name__ == "__main__":
    register_myotube_dataset() 