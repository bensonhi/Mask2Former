#!/usr/bin/env python3
"""
Duplicate CVAT Annotations for Grey Channel Images

This script processes CVAT annotation files that contain only green channel images
and creates a new annotation file with both green and grey channel images by:

1. Reading the input CVAT annotation file (COCO format)
2. Finding corresponding grey images in the dataset
3. Duplicating annotations from green to grey images
4. Saving the extended annotation file with both channels

Usage:
    # With sensible defaults (run from utils/):
    # - input:      ./instances_combined_scaled.json
    # - images_dir: ./myotube_batch_output/images
    # - output:     ./myotube_batch_output/annotations
    python duplicate_cvat_annotations_for_grey.py

    # Or specify inputs explicitly
    python duplicate_cvat_annotations_for_grey.py --input instances_combined_scaled.json --images_dir ./myotube_batch_output/images
"""

import json
import copy
import os
import argparse
from typing import List, Dict, Tuple


def find_grey_images(images_dir: str) -> List[str]:
    """Find all grey images in the dataset directory."""
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    grey_images = [f for f in os.listdir(images_dir) 
                   if 'grey' in f.lower() and f.endswith('.png')]
    return grey_images


def find_green_image_in_dataset(cvat_filename: str, images_dir: str) -> str:
    """Find the actual green image file that corresponds to CVAT annotation filename."""
    if not os.path.exists(images_dir):
        return cvat_filename
    
    actual_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    
    # If CVAT filename already has _green_processed.png, check if it exists
    if '_green_processed.png' in cvat_filename and cvat_filename in actual_files:
        return cvat_filename
    
    # If CVAT filename has _processed.png (old naming), try to find matching green image
    if '_processed.png' in cvat_filename:
        expected_green = cvat_filename.replace('_processed.png', '_green_processed.png')
        if expected_green in actual_files:
            return expected_green
    
    # If no pattern matches, return original (might be already correct)
    return cvat_filename


def find_grey_match(green_filename: str, grey_images: List[str]) -> str:
    """Find corresponding grey image for a green image."""
    if '_green_processed.png' in green_filename:
        # Replace _green_processed.png with _grey_processed.png
        expected_grey = green_filename.replace('_green_processed.png', '_grey_processed.png')
        
        if expected_grey in grey_images:
            return expected_grey
    elif '_processed.png' in green_filename:
        # Handle old naming - replace _processed.png with _grey_processed.png
        expected_grey = green_filename.replace('_processed.png', '_grey_processed.png')
        
        if expected_grey in grey_images:
            return expected_grey
    
    return None


def duplicate_annotations_for_grey(input_file: str, output_dir: str, images_dir: str) -> Dict[str, int]:
    """
    Process CVAT annotations to include grey channel images.
    
    Args:
        input_file: Path to input CVAT annotation file
        output_dir: Directory to save output annotation files
        images_dir: Directory containing images
        
    Returns:
        Dictionary with processing statistics
    """
    # Create output filenames
    manual_train_file = os.path.join(output_dir, 'manual_train_annotations.json')
    manual_test_file = os.path.join(output_dir, 'manual_test_annotations.json')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print('=== Processing CVAT Annotations to Add Grey Images ===')
    print(f'Input: {input_file}')
    print(f'Output dir: {output_dir}')
    print(f'Train file: {manual_train_file}')
    print(f'Test file: {manual_test_file}')
    print(f'Images dir: {images_dir}')
    
    # Load CVAT annotations
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input annotation file not found: {input_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in input file: {e}")
    
    original_images = len(data['images'])
    original_annotations = len(data['annotations'])
    
    print(f'Original: {original_images} images, {original_annotations} annotations')
    
    # Find existing grey images in dataset
    grey_images = find_grey_images(images_dir)
    print(f'Found {len(grey_images)} grey images in dataset')
    
    # Create green-grey pairs
    new_images = []
    new_annotations = []
    
    # Get max IDs to avoid conflicts
    max_img_id = max([img['id'] for img in data['images']]) if data['images'] else 0
    max_ann_id = max([ann['id'] for ann in data['annotations']]) if data['annotations'] else 0
    
    next_img_id = max_img_id + 1
    next_ann_id = max_ann_id + 1
    
    pairs_created = 0
    total_grey_annotations = 0
    
    for img_info in data['images']:
        cvat_filename = img_info['file_name']
        
        # Find the actual green image filename that exists in the dataset
        actual_green_filename = find_green_image_in_dataset(cvat_filename, images_dir)
        
        # Update the green image filename in the annotation if it was different
        if actual_green_filename != cvat_filename:
            img_info['file_name'] = actual_green_filename
            print(f'  🔄 Updated green image filename: {cvat_filename} → {actual_green_filename}')
        
        # Try to find corresponding grey image
        grey_match = find_grey_match(actual_green_filename, grey_images)
        
        if grey_match:
            # Create new grey image entry
            grey_img = copy.deepcopy(img_info)
            grey_img['id'] = next_img_id
            grey_img['file_name'] = grey_match
            
            # Add channel type metadata if not present
            if 'channel_type' not in grey_img:
                grey_img['channel_type'] = 'grey'
            
            new_images.append(grey_img)
            
            # Duplicate all annotations for this image
            green_annotations = [ann for ann in data['annotations'] 
                               if ann['image_id'] == img_info['id']]
            
            for ann in green_annotations:
                grey_ann = copy.deepcopy(ann)
                grey_ann['id'] = next_ann_id
                grey_ann['image_id'] = next_img_id
                new_annotations.append(grey_ann)
                next_ann_id += 1
            
            print(f'  ✅ Paired: {actual_green_filename} ↔ {grey_match} ({len(green_annotations)} annotations)')
            pairs_created += 1
            total_grey_annotations += len(green_annotations)
            next_img_id += 1
        else:
            print(f'  ⚠️  No grey match for: {actual_green_filename}')
    
    print(f'\nCreated {pairs_created} green-grey pairs')
    
    # Statistics to return
    stats = {
        'original_images': original_images,
        'original_annotations': original_annotations,
        'pairs_created': pairs_created,
        'grey_images_added': len(new_images),
        'grey_annotations_added': len(new_annotations),
        'final_images': 0,
        'final_annotations': 0
    }
    
    if pairs_created > 0:
        # Add new data to the dataset
        data['images'].extend(new_images)
        data['annotations'].extend(new_annotations)
        
        # Update info section if present
        if 'info' in data:
            original_desc = data['info'].get('description', '')
            data['info']['description'] = original_desc + ' - Extended with grey channel images'
            data['info']['grey_channels_added'] = True
            data['info']['processing_date'] = __import__('datetime').datetime.now().isoformat()
        
        # Update final statistics
        stats['final_images'] = len(data['images'])
        stats['final_annotations'] = len(data['annotations'])
        
        # Save two identical annotation files
        try:
            # Save manual_train_annotations.json
            with open(manual_train_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Save manual_test_annotations.json (identical copy)
            with open(manual_test_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f'\n✅ Saved train annotations to: {manual_train_file}')
            print(f'✅ Saved test annotations to: {manual_test_file}')
            print(f'Final: {stats["final_images"]} images, {stats["final_annotations"]} annotations')
            print(f'Added: {stats["grey_images_added"]} grey images, {stats["grey_annotations_added"]} annotations')
            print('\nYour CVAT annotations now include both green and grey channel images!')
            
        except IOError as e:
            raise IOError(f"Failed to save output files: {e}")
            
    else:
        print('\n⚠️  No pairs found - no output file created')
        print('This could mean:')
        print('  • No corresponding grey images exist in the dataset')
        print('  • Image naming patterns don\'t match expected format')
        print('  • Images directory path is incorrect')
    
    return stats


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Duplicate CVAT annotations for grey channel images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python duplicate_cvat_annotations_for_grey.py --input instances_combined_scaled.json
  
  python duplicate_cvat_annotations_for_grey.py \\
    --input my_cvat_annotations.json \\
    --output ./custom_output_dir \\
    --images_dir ./custom_images_folder
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default='./instances_combined_scaled.json',
        help='Input CVAT annotation file (COCO format JSON) (default: ./instances_combined_scaled.json)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./myotube_batch_output/annotations',
        help='Output directory for annotation files (default: ./myotube_batch_output/annotations)'
    )
    
    parser.add_argument(
        '--images_dir', '-d',
        default='./myotube_batch_output/images',
        help='Directory containing images (default: ./myotube_batch_output/images)'
    )
    
    args = parser.parse_args()
    
    try:
        # Process the annotations
        stats = duplicate_annotations_for_grey(
            input_file=args.input,
            output_dir=args.output,
            images_dir=args.images_dir
        )
        
        # Print final summary
        print('\n' + '='*60)
        print('CVAT ANNOTATION PROCESSING COMPLETE')
        print('='*60)
        
        if stats['pairs_created'] > 0:
            print(f"✅ Successfully processed {stats['pairs_created']} green-grey pairs")
            print(f"📊 Original: {stats['original_images']} images, {stats['original_annotations']} annotations")
            print(f"📊 Final: {stats['final_images']} images, {stats['final_annotations']} annotations")
            print(f"📊 Added: {stats['grey_images_added']} grey images, {stats['grey_annotations_added']} annotations")
            print(f"📁 Train annotations saved to: {os.path.join(args.output, 'manual_train_annotations.json')}")
            print(f"📁 Test annotations saved to: {os.path.join(args.output, 'manual_test_annotations.json')}")
            print("\n🎉 Your CVAT annotations are now ready for multi-channel training!")
            return 0
        else:
            print("⚠️  No green-grey pairs could be created")
            print("Please check that:")
            print(f"   • Grey images exist in: {args.images_dir}")
            print(f"   • Image naming follows expected pattern")
            print(f"   • Input file contains valid annotations: {args.input}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
