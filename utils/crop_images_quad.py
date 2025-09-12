#!/usr/bin/env python3
"""
Crop Images to Quadrants (Annotation-Filtered)

This script crops images into 4 quadrants, but only processes images that are present
in the annotations file (myotube_batch_output/annotations/algorithmic_test_annotations.json).
The cropped images are saved to myotube_batch_output/cropped_images.

Usage:
    python utils/crop_images_quad.py [--output-suffix SUFFIX] [--overlap PIXELS] [--annotations FILE]

Features:
- Only processes images listed in the annotations JSON file
- Crops each image into 4 quadrants (top-left, top-right, bottom-left, bottom-right)
- Processes all common image formats (PNG, JPG, TIFF, etc.)
- Creates organized output structure with quadrant naming
- Maintains original image quality
- Optional overlap between quadrants
- Progress tracking and error handling
- Preserves original filename structure

Output structure:
myotube_batch_output/
├── images/                     # Original images
│   ├── image1.tif
│   └── image2.png
├── annotations/
│   └── algorithmic_test_annotations.json  # Annotation file with image list
└── cropped_images/            # Cropped quadrants (or custom suffix)
    ├── image1_tl.tif          # Top-left
    ├── image1_tr.tif          # Top-right  
    ├── image1_bl.tif          # Bottom-left
    ├── image1_br.tif          # Bottom-right
    ├── image2_tl.png
    └── ...
"""

import os
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_supported_extensions():
    """Get list of supported image file extensions."""
    return {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}


def load_annotations(annotation_file):
    """
    Load annotations file and extract list of image filenames.
    
    Args:
        annotation_file: Path to the annotations JSON file
        
    Returns:
        Set of image filenames from the annotations
    """
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Extract filenames from the images array
        image_filenames = set()
        if 'images' in annotations:
            for image_info in annotations['images']:
                if 'file_name' in image_info:
                    image_filenames.add(image_info['file_name'])
        
        print(f"📋 Loaded annotations: {len(image_filenames)} images found")
        return image_filenames
        
    except FileNotFoundError:
        print(f"❌ Annotations file not found: {annotation_file}")
        return set()
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing annotations JSON: {e}")
        return set()
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return set()


def crop_image_to_quadrants(image_path, output_dir, overlap=0):
    """
    Crop a single image into 4 quadrants and save them.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save cropped images
        overlap: Number of pixels to overlap between quadrants (default: 0)
    
    Returns:
        List of paths to the created cropped images
    """
    try:
        # Load image using PIL to handle various formats
        img = Image.open(image_path)
        img_array = np.array(img)
        
        height, width = img_array.shape[:2]
        
        # Calculate crop coordinates with overlap
        mid_x = width // 2
        mid_y = height // 2
        
        # Define quadrant boundaries with overlap
        quadrants = {
            'tl': (0, 0, mid_x + overlap, mid_y + overlap),                    # Top-left
            'tr': (mid_x - overlap, 0, width, mid_y + overlap),                # Top-right
            'bl': (0, mid_y - overlap, mid_x + overlap, height),               # Bottom-left
            'br': (mid_x - overlap, mid_y - overlap, width, height)            # Bottom-right
        }
        
        # Get base filename without extension
        base_name = Path(image_path).stem
        extension = Path(image_path).suffix
        
        created_files = []
        
        # Create and save each quadrant
        for quad_name, (x1, y1, x2, y2) in quadrants.items():
            # Ensure coordinates are within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Crop the quadrant
            if len(img_array.shape) == 3:  # Color image
                cropped = img_array[y1:y2, x1:x2, :]
            else:  # Grayscale image
                cropped = img_array[y1:y2, x1:x2]
            
            # Convert back to PIL Image
            cropped_img = Image.fromarray(cropped)
            
            # Create output filename
            output_filename = f"{base_name}_{quad_name}{extension}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the cropped image
            cropped_img.save(output_path)
            created_files.append(output_path)
            
            # Print details for first few images
            if len(created_files) <= 4:
                crop_size = f"{x2-x1}×{y2-y1}"
                print(f"      ✅ {quad_name.upper()}: {crop_size} → {output_filename}")
        
        return created_files
        
    except Exception as e:
        print(f"      ❌ Error cropping {os.path.basename(image_path)}: {e}")
        return []


def process_images(input_dir, output_dir, overlap=0, annotation_filter=None):
    """
    Process images in the input directory, optionally filtered by annotations.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save cropped images
        overlap: Overlap in pixels between quadrants
        annotation_filter: Set of filenames to process (from annotations), or None for all
    
    Returns:
        Dictionary with processing statistics
    """
    
    # Get all supported image files
    supported_exts = get_supported_extensions()
    all_image_files = []
    
    for file_path in Path(input_dir).glob('*'):
        if file_path.suffix.lower() in supported_exts:
            all_image_files.append(file_path)
    
    if not all_image_files:
        print(f"❌ No supported image files found in {input_dir}")
        print(f"   Supported formats: {', '.join(supported_exts)}")
        return {'processed': 0, 'failed': 0, 'total_crops': 0, 'filtered': 0, 'total_available': 0}
    
    print(f"📊 Found {len(all_image_files)} total images in directory")
    
    # Filter by annotations if provided
    if annotation_filter:
        image_files = []
        skipped_files = []
        
        for file_path in all_image_files:
            if file_path.name in annotation_filter:
                image_files.append(file_path)
            else:
                skipped_files.append(file_path.name)
        
        print(f"🔍 Annotation filtering:")
        print(f"   ✅ Images to process: {len(image_files)}")
        print(f"   ⏭️  Images skipped: {len(skipped_files)}")
        
        if len(skipped_files) > 0 and len(skipped_files) <= 10:
            print(f"   Skipped files: {', '.join(skipped_files[:10])}")
        elif len(skipped_files) > 10:
            print(f"   Skipped files: {', '.join(skipped_files[:5])} ... and {len(skipped_files)-5} more")
        
        # Check for missing annotated images
        missing_in_directory = annotation_filter - {f.name for f in all_image_files}
        if missing_in_directory:
            print(f"   ⚠️  Annotated images not found in directory: {len(missing_in_directory)}")
            if len(missing_in_directory) <= 5:
                print(f"      Missing: {', '.join(list(missing_in_directory)[:5])}")
            else:
                print(f"      Missing: {', '.join(list(missing_in_directory)[:3])} ... and {len(missing_in_directory)-3} more")
    else:
        image_files = all_image_files
        print(f"📊 Processing all {len(image_files)} images (no annotation filter)")
    
    if not image_files:
        print(f"❌ No images to process after filtering")
        return {'processed': 0, 'failed': 0, 'total_crops': 0, 'filtered': len(all_image_files), 'total_available': len(all_image_files)}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    processed = 0
    failed = 0
    total_crops = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        print(f"   🔄 Processing: {img_path.name}")
        
        try:
            # Load image to get dimensions
            img = Image.open(img_path)
            width, height = img.size
            print(f"      📏 Original size: {width}×{height}")
            
            # Crop to quadrants
            created_files = crop_image_to_quadrants(img_path, output_dir, overlap)
            
            if created_files:
                processed += 1
                total_crops += len(created_files)
                print(f"      ✅ Created {len(created_files)} quadrants")
            else:
                failed += 1
                print(f"      ❌ Failed to create quadrants")
                
        except Exception as e:
            failed += 1
            print(f"      ❌ Error processing {img_path.name}: {e}")
    
    return {
        'processed': processed,
        'failed': failed,
        'total_crops': total_crops,
        'total_images': len(image_files),
        'total_available': len(all_image_files),
        'filtered': len(all_image_files) - len(image_files)
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Crop images in myotube_batch_output/images into 4 quadrants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--output-suffix', 
        default='cropped_images',
        help='Suffix for output directory name (default: cropped_images)'
    )
    
    parser.add_argument(
        '--overlap', 
        type=int,
        default=0,
        help='Overlap in pixels between quadrants (default: 0)'
    )
    
    parser.add_argument(
        '--input-dir',
        default='myotube_batch_output/images',
        help='Input directory containing images (default: myotube_batch_output/images)'
    )
    
    parser.add_argument(
        '--annotations',
        default='myotube_batch_output/annotations/algorithmic_test_annotations.json',
        help='Path to annotations JSON file to filter images (default: myotube_batch_output/annotations/algorithmic_test_annotations.json)'
    )
    
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Process all images, ignore annotations file'
    )
    
    args = parser.parse_args()
    
    # Define paths
    input_dir = args.input_dir
    output_dir = f"myotube_batch_output/{args.output_suffix}"
    annotations_file = args.annotations
    
    print("🖼️  IMAGE QUADRANT CROPPING (ANNOTATION-FILTERED)")
    print("="*60)
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📏 Overlap: {args.overlap} pixels")
    if not args.no_filter:
        print(f"📋 Annotations file: {annotations_file}")
    else:
        print(f"📋 Filtering: Disabled (processing all images)")
    print("="*60)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        print("\n💡 Make sure you're running this from the project root directory:")
        print("   python utils/crop_images_quad.py")
        return 1
    
    # Check if input directory has images
    supported_exts = get_supported_extensions()
    has_images = any(
        Path(input_dir).glob(f'*{ext}') 
        for ext in supported_exts
    )
    
    if not has_images:
        print(f"⚠️  No supported image files found in {input_dir}")
        print(f"   Supported formats: {', '.join(supported_exts)}")
        return 1
    
    # Load annotations if filtering is enabled
    annotation_filter = None
    if not args.no_filter:
        if os.path.exists(annotations_file):
            annotation_filter = load_annotations(annotations_file)
            if not annotation_filter:
                print("❌ No images found in annotations file or failed to load")
                return 1
        else:
            print(f"❌ Annotations file not found: {annotations_file}")
            print("💡 Use --no-filter to process all images without filtering")
            return 1
    
    # Process images
    try:
        stats = process_images(input_dir, output_dir, args.overlap, annotation_filter)
        
        # Print results
        print("\n" + "="*60)
        print("🎉 CROPPING COMPLETED!")
        print("="*60)
        print(f"📊 Results:")
        if annotation_filter:
            print(f"   📁 Total images available: {stats['total_available']}")
            print(f"   🔍 Images filtered by annotations: {stats['total_images']}")
            print(f"   ⏭️  Images skipped: {stats['filtered']}")
        print(f"   ✅ Images processed successfully: {stats['processed']}")
        print(f"   ❌ Images failed: {stats['failed']}")
        print(f"   🖼️  Total quadrants created: {stats['total_crops']}")
        print(f"   📁 Output directory: {output_dir}")
        print("="*60)
        
        if stats['failed'] > 0:
            print(f"⚠️  {stats['failed']} images failed to process - check error messages above")
            
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())