#!/usr/bin/env python3
"""
Quick debug script to check categories and identify the division by zero cause.
"""

import os
import json
import sys
sys.path.append('.')

def check_categories():
    """Check categories in panoptic JSON and identify potential mismatches."""
    
    panoptic_json = "myotube_batch_output/panoptic/algorithmic_train_panoptic.json"
    
    if not os.path.exists(panoptic_json):
        print(f"❌ Panoptic JSON not found: {panoptic_json}")
        return
    
    with open(panoptic_json, 'r') as f:
        data = json.load(f)
    
    print("🔍 Analyzing panoptic JSON categories...")
    print(f"   File: {panoptic_json}")
    
    categories = data.get('categories', [])
    print(f"   📊 Categories ({len(categories)}):")
    
    thing_categories = []
    stuff_categories = []
    
    for cat in categories:
        print(f"      ID: {cat.get('id', 'MISSING')}")
        print(f"      Name: {cat.get('name', 'MISSING')}")
        print(f"      isthing: {cat.get('isthing', 'MISSING')}")
        print(f"      supercategory: {cat.get('supercategory', 'MISSING')}")
        print(f"      color: {cat.get('color', 'MISSING')}")
        print()
        
        if cat.get('isthing') == 1:
            thing_categories.append(cat)
        elif cat.get('isthing') == 0:
            stuff_categories.append(cat)
    
    print(f"   🎯 Thing categories: {len(thing_categories)}")
    print(f"   🧱 Stuff categories: {len(stuff_categories)}")
    
    # Check annotations
    annotations = data.get('annotations', [])
    print(f"   📝 Annotations: {len(annotations)}")
    
    # Sample a few annotations to check segments_info
    for i, ann in enumerate(annotations[:3]):
        segments_info = ann.get('segments_info', [])
        print(f"   📋 Annotation {i+1}: {len(segments_info)} segments")
        
        category_ids_used = set()
        for seg in segments_info:
            category_ids_used.add(seg.get('category_id'))
        
        print(f"      Category IDs used: {category_ids_used}")
        
        # Check if category IDs exist in categories
        defined_category_ids = set(cat.get('id') for cat in categories)
        missing = category_ids_used - defined_category_ids
        if missing:
            print(f"      ❌ Missing category definitions: {missing}")
        else:
            print(f"      ✅ All category IDs are defined")
    
    # Check what the evaluation will see
    print(f"\n🔍 Evaluation perspective:")
    print(f"   Categories with isthing=True: {[cat for cat in categories if cat.get('isthing') == 1]}")
    print(f"   Categories with isthing=False: {[cat for cat in categories if cat.get('isthing') == 0]}")
    
    # The critical issue: when pq_average is called with isthing=True or isthing=False,
    # it filters categories by isthing value. If no categories match, N=0 -> division by zero
    
    return categories

def check_metadata_consistency():
    """Check if dataset metadata matches the JSON categories."""
    
    try:
        from register_two_stage_datasets import register_two_stage_datasets
        from detectron2.data import MetadataCatalog
        
        # Register datasets
        register_two_stage_datasets(
            dataset_root="myotube_batch_output",
            register_instance=False,
            register_panoptic=True
        )
        
        metadata = MetadataCatalog.get("myotube_stage1_panoptic_val")
        
        print(f"\n🔍 Dataset metadata:")
        print(f"   thing_classes: {getattr(metadata, 'thing_classes', 'MISSING')}")
        print(f"   stuff_classes: {getattr(metadata, 'stuff_classes', 'MISSING')}")
        print(f"   thing_dataset_id_to_contiguous_id: {getattr(metadata, 'thing_dataset_id_to_contiguous_id', 'MISSING')}")
        print(f"   stuff_dataset_id_to_contiguous_id: {getattr(metadata, 'stuff_dataset_id_to_contiguous_id', 'MISSING')}")
        
        # The issue might be:
        # 1. thing_classes = ["myotube"] but no categories with isthing=1 in JSON
        # 2. stuff_classes = [] but evaluation expects stuff categories
        # 3. ID mapping mismatch
        
    except Exception as e:
        print(f"❌ Error checking metadata: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚀 Quick Category Debug")
    print("=" * 40)
    
    categories = check_categories()
    check_metadata_consistency()
    
    print(f"\n💡 Potential fixes:")
    print(f"   1. Ensure categories have correct 'isthing' values")
    print(f"   2. Verify thing/stuff category counts match metadata")
    print(f"   3. Check category ID mappings")

if __name__ == "__main__":
    main() 