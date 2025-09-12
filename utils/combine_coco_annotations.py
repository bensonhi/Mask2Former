#!/usr/bin/env python3
"""
Combine Two COCO Instance Annotation Files

This script combines two COCO annotation files that have the same categories
but different images and annotations. Perfect for combining algorithmic and 
manual annotations for the myotube project.

Features:
- Merges images from both files
- Merges annotations from both files  
- Handles ID conflicts by reassigning IDs
- Preserves all metadata
- Validates that categories match

Usage:
    # With sensible defaults (run from utils/):
    # - file1: ./instances_default.json
    # - file2: ./instances_default1.json
    # - output: ./instances_combined.json
    python combine_coco_annotations.py

    # Or specify files explicitly
    python combine_coco_annotations.py --file1 algo_train.json --file2 manual_train.json --output combined_train.json
"""

import os
import json
import argparse
from collections import defaultdict

def load_coco_file(file_path):
    """Load and validate COCO annotation file."""
    print(f"📖 Loading: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Validate required fields
    required_fields = ['images', 'annotations', 'categories']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in {file_path}")
    
    print(f"   📊 {len(data['images'])} images, {len(data['annotations'])} annotations, {len(data['categories'])} categories")
    
    return data

def validate_categories_match(data1, data2, file1_name, file2_name):
    """Validate that both files have the same categories."""
    print(f"🔍 Validating categories match between files...")
    
    # Sort categories by ID for comparison
    cats1 = sorted(data1['categories'], key=lambda x: x.get('id', 0))
    cats2 = sorted(data2['categories'], key=lambda x: x.get('id', 0))
    
    if len(cats1) != len(cats2):
        raise ValueError(f"Different number of categories: {file1_name} has {len(cats1)}, {file2_name} has {len(cats2)}")
    
    for cat1, cat2 in zip(cats1, cats2):
        # Check essential fields
        essential_fields = ['id', 'name']
        for field in essential_fields:
            if cat1.get(field) != cat2.get(field):
                raise ValueError(f"Category mismatch in field '{field}': {cat1.get(field)} vs {cat2.get(field)}")
    
    print(f"   ✅ Categories match!")
    return cats1

def get_id_mappings(data1, data2):
    """
    Create ID mappings to resolve conflicts between the two files.
    
    Returns:
        tuple: (image_id_mapping_file2, annotation_id_mapping_file2)
    """
    print(f"🔧 Creating ID mappings to resolve conflicts...")
    
    # Get existing IDs from file1
    existing_image_ids = set(img['id'] for img in data1['images'])
    existing_annotation_ids = set(ann['id'] for ann in data1['annotations'])
    
    # Find max IDs to start new assignments from
    max_image_id = max(existing_image_ids) if existing_image_ids else 0
    max_annotation_id = max(existing_annotation_ids) if existing_annotation_ids else 0
    
    # Create mappings for file2
    image_id_mapping = {}
    annotation_id_mapping = {}
    
    # Map image IDs
    next_image_id = max_image_id + 1
    for img in data2['images']:
        old_id = img['id']
        if old_id in existing_image_ids:
            image_id_mapping[old_id] = next_image_id
            next_image_id += 1
        else:
            image_id_mapping[old_id] = old_id  # Keep original if no conflict
    
    # Map annotation IDs  
    next_annotation_id = max_annotation_id + 1
    for ann in data2['annotations']:
        old_id = ann['id']
        if old_id in existing_annotation_ids:
            annotation_id_mapping[old_id] = next_annotation_id
            next_annotation_id += 1
        else:
            annotation_id_mapping[old_id] = old_id  # Keep original if no conflict
    
    print(f"   📋 Image ID remappings needed: {sum(1 for old, new in image_id_mapping.items() if old != new)}")
    print(f"   📋 Annotation ID remappings needed: {sum(1 for old, new in annotation_id_mapping.items() if old != new)}")
    
    return image_id_mapping, annotation_id_mapping

def apply_id_mappings(data, image_id_mapping, annotation_id_mapping):
    """Apply ID mappings to data from file2."""
    
    # Update image IDs
    for img in data['images']:
        old_id = img['id']
        img['id'] = image_id_mapping[old_id]
    
    # Update annotation IDs and image_id references, skip annotations without matching images
    valid_annotations = []
    for ann in data['annotations']:
        # Update annotation ID
        old_ann_id = ann['id']
        ann['id'] = annotation_id_mapping[old_ann_id]
        
        # Update image_id reference - skip if no matching image
        old_img_id = ann['image_id']
        if old_img_id in image_id_mapping:
            ann['image_id'] = image_id_mapping[old_img_id]
            valid_annotations.append(ann)
        else:
            print(f"   ⚠️  Skipping annotation {old_ann_id} - no matching image ID {old_img_id}")
    
    data['annotations'] = valid_annotations
    
    return data

def combine_coco_files(file1_path, file2_path, output_path):
    """
    Combine two COCO annotation files.
    
    Args:
        file1_path: Path to first COCO file
        file2_path: Path to second COCO file  
        output_path: Path to save combined file
    """
    
    print(f"🔄 Combining COCO annotation files...")
    print(f"   File 1: {file1_path}")
    print(f"   File 2: {file2_path}")
    print(f"   Output: {output_path}")
    print("=" * 60)
    
    # Load both files
    data1 = load_coco_file(file1_path)
    data2 = load_coco_file(file2_path)
    
    # Validate categories match
    categories = validate_categories_match(data1, data2, 
                                         os.path.basename(file1_path), 
                                         os.path.basename(file2_path))
    
    # Create ID mappings to resolve conflicts
    image_id_mapping, annotation_id_mapping = get_id_mappings(data1, data2)
    
    # Apply mappings to data2
    print(f"🔧 Applying ID mappings to second file...")
    data2_mapped = apply_id_mappings(data2.copy(), image_id_mapping, annotation_id_mapping)
    
    # Combine the data
    print(f"🔗 Merging data...")
    combined_data = {
        "info": data1.get("info", {}).copy(),
        "licenses": data1.get("licenses", []).copy(),
        "categories": categories,
        "images": data1["images"] + data2_mapped["images"],
        "annotations": data1["annotations"] + data2_mapped["annotations"]
    }
    
    # Update info section
    if "info" not in combined_data:
        combined_data["info"] = {}
    
    original_desc = combined_data["info"].get("description", "")
    combined_data["info"]["description"] = f"{original_desc} [COMBINED FROM {os.path.basename(file1_path)} + {os.path.basename(file2_path)}]"
    
    # Add combination statistics to info
    combined_data["info"]["combination_stats"] = {
        "file1": {
            "name": os.path.basename(file1_path),
            "images": len(data1["images"]),
            "annotations": len(data1["annotations"])
        },
        "file2": {
            "name": os.path.basename(file2_path), 
            "images": len(data2["images"]),
            "annotations": len(data2["annotations"])
        },
        "combined": {
            "images": len(combined_data["images"]),
            "annotations": len(combined_data["annotations"])
        }
    }
    
    # Save combined file
    print(f"💾 Saving combined annotations...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    # Print summary
    print(f"\n✅ Combination complete!")
    print(f"   📊 Total images: {len(combined_data['images'])} ({len(data1['images'])} + {len(data2['images'])})")
    print(f"   📊 Total annotations: {len(combined_data['annotations'])} ({len(data1['annotations'])} + {len(data2['annotations'])})")
    print(f"   📊 Categories: {len(combined_data['categories'])}")
    print(f"   📁 Saved to: {output_path}")
    
    # Check for potential issues
    image_filenames = [img['file_name'] for img in combined_data['images']]
    duplicate_filenames = [name for name in set(image_filenames) if image_filenames.count(name) > 1]
    
    if duplicate_filenames:
        print(f"\n⚠️  Warning: Duplicate image filenames found:")
        for dup in duplicate_filenames[:5]:  # Show first 5
            print(f"      {dup}")
        if len(duplicate_filenames) > 5:
            print(f"      ... and {len(duplicate_filenames) - 5} more")
        print(f"   This might cause issues if both files reference the same image files.")
    else:
        print(f"\n✅ No duplicate image filenames found.")
    
    return combined_data

def main():
    parser = argparse.ArgumentParser(description="Combine two COCO instance annotation files")
    parser.add_argument(
        "--file1", "-f1",
        default="instances_default.json",
        help="First COCO annotation file (default: instances_default.json)"
    )
    parser.add_argument(
        "--file2", "-f2",
        default="instances_default1.json",
        help="Second COCO annotation file (default: instances_default1.json)"
    ) 
    parser.add_argument(
        "--output", "-o",
        default="./instances_combined.json",
        help="Output path for combined annotation file (default: ./instances_combined.json)"
    )
    parser.add_argument("--dry_run", action="store_true", help="Show what would be combined without saving")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.file1):
        print(f"❌ Error: File 1 not found: {args.file1}")
        print("   Tip: Place instances_default.json in the current directory or pass --file1")
        return
    
    if not os.path.exists(args.file2):
        print(f"❌ Error: File 2 not found: {args.file2}")
        print("   Tip: Place instances_default1.json in the current directory or pass --file2")
        return
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be saved")
    
    # Run combination
    try:
        combined_data = combine_coco_files(
            args.file1, 
            args.file2, 
            args.output if not args.dry_run else "/tmp/dummy_output.json"
        )
        
        if args.dry_run:
            print(f"\n🔍 Dry run complete. Would have saved combined file with:")
            print(f"   📊 {len(combined_data['images'])} images")
            print(f"   📊 {len(combined_data['annotations'])} annotations")
            # Clean up dummy file
            dummy_path = "/tmp/dummy_output.json"
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
        
    except Exception as e:
        print(f"❌ Error during combination: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
