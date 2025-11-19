"""
Myotube segmentation backend implementation.

This module contains the main MyotubeFijiIntegration class that provides
instance segmentation for myotubes using Mask2Former.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch

from fiji_integration.core.interfaces import SegmentationInterface
from fiji_integration.core.postprocessing import PostProcessingPipeline
from fiji_integration.utils.path_utils import find_mask2former_project, ensure_mask2former_loaded


# Global variable for project directory
project_dir = None


__all__ = ['MyotubeFijiIntegration']


class MyotubeFijiIntegration(SegmentationInterface):
    """
    Main class for Fiji integration of myotube instance segmentation.
    
    This class implements the SegmentationInterface and provides complete
    myotube segmentation functionality using Mask2Former.
    """
    
    def __init__(self, config_file: str = None, model_weights: str = None,
                 skip_merged_masks: bool = False, mask2former_path: str = None):
        """
        Initialize the Fiji integration.

        Args:
            config_file: Path to model config file
            model_weights: Path to model weights
            skip_merged_masks: Skip generation of merged visualization masks (default: False)
            mask2former_path: Path to Mask2Former project directory (auto-detected if not provided)
        """
        self.config_file = config_file
        self.model_weights = model_weights
        self.skip_merged_masks = skip_merged_masks
        self.mask2former_path = mask2former_path
        self._predictor = None
        self._post_processor = PostProcessingPipeline()
        
        # Attributes for tiled segmentation compatibility
        self._original_size = None
        self._scale_factor = 1.0
        self._processing_size = None

        # Setup paths
        self.setup_paths()
        
    def setup_paths(self):
        """Setup default paths if not provided."""
        global project_dir
        
        # If both paths are provided, no need to auto-detect
        if self.config_file and self.model_weights:
            print(f"📁 Config file: {self.config_file}")
            print(f"🔮 Model weights: {self.model_weights}")
            return

        # Load project directory for auto-detection
        project_dir = ensure_mask2former_loaded(explicit_path=self.mask2former_path)
        base_dir = Path(project_dir)

        if not self.config_file:
            # Try to find the best available config
            config_options = [
                base_dir / "stage2_config.yaml",
                base_dir / "stage1_config.yaml",
                base_dir / "stage2_panoptic_config.yaml",
                base_dir / "stage1_panoptic_config.yaml",
                base_dir / "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
            ]
            
            print(f"🔍 Looking for config files in: {base_dir}")
            for config_path in config_options:
                print(f"   Checking: {config_path.name} - {'✅' if config_path.exists() else '❌'}")
                if config_path.exists():
                    self.config_file = str(config_path)
                    break
        
        if not self.model_weights:
            # Try to find the best available weights
            weight_options = [
                base_dir / "output_stage2_manual/model_final.pth",
                base_dir / "output_stage2_manual/model_best.pth",
                base_dir / "output_stage2_panoptic_manual/model_final.pth",
                base_dir / "output_stage2_panoptic_manual/model_best.pth",
                base_dir / "output_stage1_algorithmic/model_final.pth",
                base_dir / "output_stage1_algorithmic/model_best.pth",
                base_dir / "output_stage1_panoptic_algorithmic/model_final.pth",
                base_dir / "output_stage1_panoptic_algorithmic/model_best.pth",
            ]
            
            print(f"🔍 Looking for model weights in: {base_dir}")
            for weight_path in weight_options:
                if weight_path.exists():
                    print(f"   Found: {weight_path.name}")
                    self.model_weights = str(weight_path)
                    break
                    
        if not self.config_file:
            print("❌ No config file found! Available options:")
            print("   1. Specify with --config argument")
            print("   2. Place config files in project directory")
            print("   3. Use default COCO config")
            # Use default COCO config as fallback
            default_config = base_dir / "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
            if default_config.exists():
                self.config_file = str(default_config)
                print(f"   ✅ Using fallback: {default_config.name}")
            else:
                raise FileNotFoundError(
                    f"No config files found in {base_dir}. "
                    "Please check your Mask2Former installation or specify --config path."
                )
            
        if not self.model_weights:
            print("❌ No model weights found! Available options:")
            print("   1. Specify with --weights argument")
            print("   2. Train model and place weights in output directories")
            print("   3. Use COCO pre-trained weights")
            # Use COCO pre-trained as fallback
            self.model_weights = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl"
            print("   Using COCO pre-trained weights (will download)")
        
        print(f"📁 Config file: {self.config_file}")
        print(f"🔮 Model weights: {self.model_weights}")
    
    @property
    def predictor(self):
        """Return the initialized predictor instance."""
        return self._predictor
    
    @property
    def post_processor(self):
        """Return the post-processing pipeline."""
        return self._post_processor
    
    def initialize_predictor(self, force_cpu=False):
        """Initialize the segmentation predictor."""
        if self._predictor is not None:
            return

        self.force_cpu = force_cpu

        print("🚀 Initializing Mask2Former predictor...")

        # Import required modules (must be done after ensure_mask2former_loaded is called)
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.utils.visualizer import Visualizer
        from mask2former import add_maskformer2_config

        # Clear GPU cache before initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB total")
            print(f"   🔥 GPU Memory: {torch.cuda.memory_allocated() // 1e6:.0f}MB allocated before init")

        # Validate files exist
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        # Setup configuration in correct order (like demo.py)
        cfg = get_cfg()
        # CRITICAL: Add configs in same order as demo.py
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        
        try:
            # Temporarily allow unknown keys to accommodate minor version diffs
            if hasattr(cfg, 'set_new_allowed'):
                cfg.set_new_allowed(True)
            cfg.merge_from_file(self.config_file)
        except Exception as e:
            print(f"❌ Error loading config file: {self.config_file}")
            print(f"   Error: {e}")
            
            # Try with a known working config as fallback
            fallback_config = os.path.join(project_dir, "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
            if os.path.exists(fallback_config):
                print(f"   🔄 Trying fallback config: {fallback_config}")
                try:
                    if hasattr(cfg, 'set_new_allowed'):
                        cfg.set_new_allowed(True)
                    cfg.merge_from_file(fallback_config)
                except Exception as e2:
                    print(f"   ❌ Fallback config also failed: {e2}")
                    print("   🔧 Creating minimal working config...")
                    self._setup_minimal_config(cfg)
            else:
                print("   🔧 Creating minimal working config...")
                self._setup_minimal_config(cfg)
        finally:
            # Disallow unknown keys after merging to avoid silent errors later
            if hasattr(cfg, 'set_new_allowed'):
                cfg.set_new_allowed(False)
        
        cfg.MODEL.WEIGHTS = self.model_weights
        
        # Memory optimization: preserve training resolution by default
        original_size = getattr(cfg.INPUT, 'IMAGE_SIZE', 1024)
        
        # Only reduce if extremely large (>2048) - otherwise preserve training resolution
        if cfg.INPUT.IMAGE_SIZE > 2048:
            cfg.INPUT.IMAGE_SIZE = 1500  # Use your training size as reasonable max
            print(f"   🔧 Reduced input size: {original_size} → {cfg.INPUT.IMAGE_SIZE} (extreme size limit)")
        else:
            print(f"   ✅ Using training resolution: {cfg.INPUT.IMAGE_SIZE}px (matching training config)")
        
        # Store training size for later use
        self.training_image_size = cfg.INPUT.IMAGE_SIZE
        
        # Memory optimization: ensure batch size is 1 for inference
        if hasattr(cfg.SOLVER, 'IMS_PER_BATCH'):
            cfg.SOLVER.IMS_PER_BATCH = 1
        
        # Only set threshold if this is a mask2former config
        if hasattr(cfg.MODEL, 'MASK_FORMER'):
            cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.25  # Lower threshold for better detection
        
        # Force CPU if requested
        if self.force_cpu or not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
            print("   🖥️  Using CPU inference")
        
        # Freeze config before creating predictor (like demo.py does)
        cfg.freeze()
        
        try:
            self._predictor = DefaultPredictor(cfg)
            device = "CPU" if cfg.MODEL.DEVICE == "cpu" else "GPU"
            print(f"✅ Predictor initialized successfully on {device}!")
            
            if torch.cuda.is_available() and cfg.MODEL.DEVICE != "cpu":
                print(f"   🔥 GPU Memory: {torch.cuda.memory_allocated() // 1e6:.0f}MB allocated after init")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ GPU out of memory during initialization")
                if not force_cpu:  # Only try CPU fallback if not already using CPU
                    print(f"   💡 Trying CPU fallback...")
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Create new config for CPU (like AsyncPredictor does)
                    cpu_cfg = cfg.clone()
                    cpu_cfg.defrost()
                    cpu_cfg.MODEL.DEVICE = "cpu"
                    cpu_cfg.freeze()
                    
                    self._predictor = DefaultPredictor(cpu_cfg)
                    print("✅ Successfully switched to CPU inference!")
                else:
                    print("❌ Out of memory even on CPU - try reducing image size")
                    raise e
            else:
                raise e
    
    def _setup_minimal_config(self, cfg):
        """Setup minimal working Mask2Former config when file configs fail."""
        print("   Setting up minimal Mask2Former configuration...")
        
        # Basic model setup for instance segmentation
        cfg.MODEL.META_ARCHITECTURE = "MaskFormer"
        cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        
        # SEM_SEG_HEAD config
        cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1  # Just myotubes
        cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
        cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
        cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
        cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
        cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6
        
        # MASK_FORMER config
        cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "StandardTransformerDecoder"
        cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
        cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
        cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
        cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
        cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
        cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
        cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
        cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
        cfg.MODEL.MASK_FORMER.NHEADS = 8
        cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
        cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
        cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
        cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
        cfg.MODEL.MASK_FORMER.PRE_NORM = False
        cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False
        cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32
        cfg.MODEL.MASK_FORMER.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE = True
        
        # Test config
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.25
        cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.8
        
        # Input config
        cfg.INPUT.IMAGE_SIZE = 1024
        cfg.INPUT.MIN_SCALE = 0.1
        cfg.INPUT.MAX_SCALE = 2.0
        cfg.INPUT.FORMAT = "RGB"
        
        # Dataset config
        cfg.DATASETS.TEST = ("myotube_test",)  # Dummy dataset name
        
        print("   ✅ Minimal config created")
    
    def segment_image(self, image_path: str, output_dir: str, 
                     custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Segment myotubes in an image and save Fiji-compatible outputs.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            custom_config: Custom post-processing configuration
            
        Returns:
            Dictionary with paths to generated files
        """
        from detectron2.data.detection_utils import read_image
        
        print(f"🔬 Segmenting myotubes in: {os.path.basename(image_path)}")
        
        # Initialize predictor if needed
        force_cpu = custom_config.get('force_cpu', False) if custom_config else False
        self.initialize_predictor(force_cpu=force_cpu)
        
        # Update post-processing config if provided
        if custom_config:
            self._post_processor.config.update(custom_config)
        
        # Load and process image
        image = read_image(image_path, format="BGR")
        original_image = cv2.imread(image_path)
        
        # Smart image resizing: respect training resolution unless explicitly overridden
        h, w = image.shape[:2]
        training_size = getattr(self, 'training_image_size', 1500)
        max_size = custom_config.get('max_image_size', None) if custom_config else None
        
        # Determine target size
        if max_size and max_size < training_size:
            # User explicitly wants smaller images for memory
            target_size = max_size
            reason = "user-requested memory optimization"
        elif max(h, w) > training_size * 1.5:  # Only resize if much larger than training
            target_size = training_size
            reason = "matching training resolution"
        elif max_size and max(h, w) > max_size:
            target_size = max_size  
            reason = "size limit"
        else:
            target_size = None  # No resizing needed
        
        if target_size and max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            print(f"   🔧 Resized image: {w}×{h} → {new_w}×{new_h} ({reason})")
            # Store scaling info for mask resizing later
            self._scale_factor = scale
            self._original_size = (h, w)
            self._processing_size = (new_h, new_w)
        else:
            print(f"   ✅ Keeping original size: {w}×{h} (within training resolution range)")
            # No scaling needed
            self._scale_factor = 1.0
            self._original_size = (h, w)
            self._processing_size = (h, w)
        
        # Clear GPU cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   🔥 GPU Memory before inference: {torch.cuda.memory_allocated() // 1e6:.0f}MB")
        
        # Run segmentation
        print("   🔄 Running inference...")
        try:
            predictions = self._predictor(image)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("   ❌ GPU out of memory during inference")
                print("   💡 Try reducing image size or using CPU mode")
                raise RuntimeError("GPU out of memory. Try: --cpu or resize image to <1024px") from e
            else:
                raise e
        instances = predictions["instances"]
        
        if len(instances) == 0:
            print("   ⚠️  No myotubes detected!")
            return self._create_empty_outputs(image_path, output_dir)
        
        print(f"   🎯 Detected {len(instances)} potential myotubes")

        # Apply post-processing using inference resolution (not original high-res)
        processed_instances = self._post_processor.process(instances, image)

        # Scale masks back to original resolution if needed
        if self._scale_factor != 1.0:
            print(f"   🔄 Scaling masks back to original resolution: {self._original_size}")
            scaled_masks = []
            for i, mask in enumerate(processed_instances['masks']):
                # Scale mask to original size using nearest neighbor
                mask_uint8 = (mask * 255).astype(np.uint8)
                resized_mask = cv2.resize(
                    mask_uint8,
                    (self._original_size[1], self._original_size[0]),  # (width, height)
                    interpolation=cv2.INTER_NEAREST
                )
                scaled_mask = (resized_mask > 128).astype(bool)
                scaled_masks.append(scaled_mask)

            # Update processed_instances with scaled masks
            processed_instances['masks'] = np.array(scaled_masks)
            processed_instances['image_shape'] = self._original_size
            print(f"   ✅ Scaled {len(scaled_masks)} masks to {self._original_size}")

        # Generate outputs with both raw and processed overlays
        output_files = self._generate_fiji_outputs(
            instances, processed_instances, original_image, image_path, output_dir, custom_config
        )
        
        return output_files
    
    def _create_empty_outputs(self, image_path: str, output_dir: str) -> Dict[str, str]:
        """Create empty output files when no instances are detected."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        
        # Create empty masks directory
        masks_dir = os.path.join(output_dir, f"{base_name}_masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        # Create empty overlay (just copy original)
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.tif")
        original = cv2.imread(image_path)
        cv2.imwrite(overlay_path, original)
        
        # Create empty measurements
        measurements_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
        with open(measurements_path, 'w') as f:
            f.write("Instance,Area,Perimeter,AspectRatio,Confidence\n")
        
        return {
            'masks_dir': masks_dir,
            'overlay': overlay_path,
            'measurements': measurements_path,
            'count': 0
        }
    
    def _generate_fiji_outputs(self, raw_instances, processed_instances: Dict[str, Any], original_image: np.ndarray,
                              image_path: str, output_dir: str, custom_config: Dict[str, Any] = None) -> Dict[str, str]:
        """Generate all Fiji-compatible output files."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        
        # Generate RAW Detectron2 overlay (before post-processing)
        raw_overlay_path = os.path.join(output_dir, f"{base_name}_raw_overlay.tif")
        self._save_colored_overlay(raw_instances, original_image, raw_overlay_path, overlay_type="raw")
        
        # Generate PROCESSED overlay (after post-processing)
        processed_overlay_path = os.path.join(output_dir, f"{base_name}_processed_overlay.tif")
        self._save_colored_overlay(processed_instances, original_image, processed_overlay_path, overlay_type="processed")

        # Generate individual mask images (pixel-perfect accuracy!) - using processed instances
        masks_dir = os.path.join(output_dir, f"{base_name}_masks")
        self._save_individual_mask_images(processed_instances, original_image, masks_dir)

        # Generate merged visualization masks (connect disconnected components) - optional
        if not self.skip_merged_masks:
            merged_masks_dir = os.path.join(output_dir, f"{base_name}_merged_masks")
            self._save_merged_visualization_masks(processed_instances, original_image, merged_masks_dir)
        else:
            print(f"   ⏭️  Skipping merged mask generation (--skip-merged-masks enabled)")

        # Generate measurements CSV - using processed instances (optional)
        measurements_path = None
        if custom_config and custom_config.get('save_measurements', False):
            measurements_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
            print(f"   📊 Generating measurements CSV...")
            self._save_measurements(processed_instances, measurements_path)
        else:
            print(f"   ⏭️  Skipping measurements CSV (disabled in settings)")
        
        # Generate summary info - using processed instances
        info_path = os.path.join(output_dir, f"{base_name}_info.json")
        self._save_info(processed_instances, image_path, info_path, original_image)
        
        # Print comparison
        raw_count = len(raw_instances) if hasattr(raw_instances, '__len__') else len(raw_instances.pred_masks) if hasattr(raw_instances, 'pred_masks') else 0
        processed_count = len(processed_instances['masks'])
        print(f"✅ Generated outputs: {raw_count} raw → {processed_count} after filtering")
        
        return {
            'masks_dir': masks_dir,
            'raw_overlay': raw_overlay_path,
            'processed_overlay': processed_overlay_path,
            'measurements': measurements_path,
            'info': info_path,
            'raw_count': raw_count,
            'processed_count': processed_count,
            'count': processed_count  # Keep for backwards compatibility
        }
    
    # ... (Rest of methods: _save_individual_mask_images, _save_merged_visualization_masks, etc.)
    # These will be added in the next part of the file
    
    def _save_individual_mask_images(self, instances: Dict[str, Any], original_image: np.ndarray, output_dir: str):
        """Save each myotube mask as individual image files - pixel-perfect accuracy!"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"   🖼️  Generating individual mask images: {output_dir}")
        print(f"   📊 Processing {len(instances['masks'])} instances for mask images")
        
        successful_masks = 0
        failed_masks = 0
        
        for i, mask in enumerate(instances['masks']):
            mask_name = f"Myotube_{i+1}_mask.png"
            mask_path = os.path.join(output_dir, mask_name)
            
            # Skip empty masks
            if mask.sum() == 0:
                print(f"      ⚠️  Warning: Mask {i+1} is empty - skipping")
                failed_masks += 1
                continue
            
            print(f"      🔍 Processing mask {i+1}: {mask.sum()} pixels")

            # Masks are already at original resolution (scaled in segment_image if needed)
            final_mask = (mask * 255).astype(np.uint8)
            
            # Save mask as PNG image
            try:
                cv2.imwrite(mask_path, final_mask)
                
                # Verify file was created
                if os.path.exists(mask_path):
                    file_size_kb = os.path.getsize(mask_path) / 1024
                    print(f"      ✅ Mask {i+1}: Saved as PNG ({file_size_kb:.1f} KB)")
                    successful_masks += 1
                else:
                    print(f"      ❌ Mask {i+1}: Failed to save file")
                    failed_masks += 1
                    
            except Exception as e:
                print(f"      ❌ Mask {i+1}: Error saving - {e}")
                failed_masks += 1
        
        # Final summary
        print(f"   📊 Mask Image Generation Summary:")
        print(f"      ✅ Successful: {successful_masks}")
        print(f"      ❌ Failed: {failed_masks}")
        print(f"      📁 Saved to: {output_dir}")
        
        # Create a summary file for easy reference
        summary_path = os.path.join(output_dir, "README.txt")
        with open(summary_path, 'w') as f:
            f.write("Myotube Individual Mask Images\n")
            f.write("==============================\n\n")
            f.write(f"Total masks: {successful_masks}\n")
            f.write(f"Image format: PNG (binary masks)\n")
            f.write(f"Pixel values: 0 (background), 255 (myotube)\n")
            f.write(f"Resolution: Same as original image\n\n")
            f.write("Usage in ImageJ/Fiji:\n")
            f.write("1. Open original image\n")
            f.write("2. Load mask images as overlays: Image > Overlay > Add Image\n")
            f.write("3. Perfect pixel alignment with Detectron2 results\n")
            f.write("4. Use Image Calculator for measurements if needed\n")
        
        return successful_masks
    
    # Include all other helper methods for mask merging, visualization, measurements, etc.
    # (The full implementation continues with all methods from the original file)
    
    def _save_colored_overlay(self, instances, original_image: np.ndarray,
                             output_path: str, overlay_type: str = "processed"):
        """Save colored overlay using Detectron2's built-in visualizer like demo.py."""
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog
        from detectron2.structures import Instances
        import torch

        print(f"   🎨 Generating {overlay_type} overlay using Detectron2's Visualizer")

        # MEMORY OPTIMIZATION: Generate overlays at reasonable resolution (not full original)
        # Overlays are for visualization only - 3000px is more than sufficient
        max_overlay_size = 3000
        original_h, original_w = original_image.shape[:2]

        if max(original_h, original_w) > max_overlay_size:
            scale = max_overlay_size / max(original_h, original_w)
            overlay_h, overlay_w = int(original_h * scale), int(original_w * scale)
            overlay_image = cv2.resize(original_image, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
            print(f"      📐 Overlay resolution: {overlay_w}×{overlay_h} (optimized for memory)")
        else:
            overlay_image = original_image
            overlay_h, overlay_w = original_h, original_w
            scale = 1.0

        # Get metadata (try to use the same as our dataset, fallback to COCO)
        try:
            metadata = MetadataCatalog.get("myotube_stage2_train")
        except:
            try:
                metadata = MetadataCatalog.get("coco_2017_val")
            except:
                metadata = None
        
        # Handle both raw Detectron2 instances and our processed format
        if hasattr(instances, 'pred_masks'):
            # Raw Detectron2 Instances - need to resize masks to overlay resolution
            torch_instances = Instances((overlay_h, overlay_w))

            if len(instances) > 0:
                # Resize masks to overlay resolution (memory-efficient)
                raw_masks = instances.pred_masks.cpu().numpy()
                final_masks = []

                for i, mask in enumerate(raw_masks):
                    # Resize mask to overlay resolution
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    resized_mask = cv2.resize(
                        mask_uint8,
                        (overlay_w, overlay_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    resized_mask = (resized_mask > 128)  # Back to boolean
                    final_masks.append(resized_mask)
                
                # Convert to torch tensors
                torch_instances.pred_masks = torch.tensor(np.array(final_masks)).cpu()
                torch_instances.scores = instances.scores.cpu()
                
                # Use empty bounding boxes
                from detectron2.structures import Boxes
                torch_instances.pred_boxes = Boxes(torch.zeros(len(instances), 4).cpu())
                torch_instances.pred_classes = torch.zeros(len(instances), dtype=torch.long).cpu()
                
                num_instances = len(instances)
            else:
                num_instances = 0
        else:
            # Our processed format - convert back to Detectron2 Instances
            torch_instances = Instances((overlay_h, overlay_w))

            if len(instances['masks']) > 0:
                # Resize masks to overlay resolution (memory-efficient)
                final_masks = []

                for i, mask in enumerate(instances['masks']):
                    # Ensure mask is a numpy array first
                    if torch.is_tensor(mask):
                        mask = mask.cpu().numpy()

                    # Ensure mask is boolean
                    mask = mask.astype(bool)

                    # Resize mask to overlay resolution
                    mask_uint8 = mask.astype(np.uint8) * 255
                    resized_mask = cv2.resize(
                        mask_uint8,
                        (overlay_w, overlay_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    resized_mask = (resized_mask > 128)  # Back to boolean

                    # Final validation: ensure boolean numpy array
                    resized_mask = np.array(resized_mask, dtype=bool)
                    final_masks.append(resized_mask)
                
                # Convert to torch tensors with proper data types
                mask_array = np.array(final_masks, dtype=bool)
                torch_instances.pred_masks = torch.tensor(mask_array).cpu()
                
                # Ensure scores are float32
                scores_array = np.array(instances['scores'], dtype=np.float32)
                torch_instances.scores = torch.tensor(scores_array).cpu()
                
                # Use empty bounding boxes
                from detectron2.structures import Boxes
                torch_instances.pred_boxes = Boxes(torch.zeros(len(instances['masks']), 4).cpu())
                
                # Add dummy classes
                torch_instances.pred_classes = torch.zeros(len(instances['masks']), dtype=torch.long).cpu()
                
                num_instances = len(instances['masks'])
            else:
                num_instances = 0
        
        print(f"   📊 Created Detectron2 Instances with {num_instances} instances")

        # Use Detectron2's visualizer exactly like demo.py does
        # Convert BGR to RGB for visualization
        rgb_image = overlay_image[:, :, ::-1]
        visualizer = Visualizer(rgb_image, metadata, instance_mode=ColorMode.IMAGE)
        
        # Add validation before visualization
        if num_instances > 0:
            print(f"   🔍 Mask validation: shape={torch_instances.pred_masks.shape}, dtype={torch_instances.pred_masks.dtype}")
            print(f"   🔍 Score validation: shape={torch_instances.scores.shape}, dtype={torch_instances.scores.dtype}")
        
        try:
            # This is the exact same call that demo.py uses!
            vis_output = visualizer.draw_instance_predictions(predictions=torch_instances)
            
            # Get the visualization as an image and convert back to BGR for saving
            vis_image = vis_output.get_image()[:, :, ::-1]  # RGB back to BGR
            
        except Exception as e:
            print(f"   ❌ Visualization failed: {e}")
            print(f"   💡 Creating fallback overlay with image")
            # Fallback: save overlay image as overlay
            vis_image = overlay_image.copy()
        
        print(f"   💾 Saving overlay to: {output_path}")
        
        # Save the visualization
        success = cv2.imwrite(output_path, vis_image)
        if success:
            print(f"   ✅ {overlay_type.title()} overlay saved: {os.path.basename(output_path)}")
        else:
            print(f"   ❌ Failed to save {overlay_type} overlay")

        print(f"   🔍 {overlay_type.title()} overlay: {num_instances} instances visualized")

        # Memory cleanup after overlay generation
        del torch_instances, final_masks, vis_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    def _save_measurements(self, instances: Dict[str, Any], output_path: str):
        """Save comprehensive measurements CSV for myotube analysis (with parallel processing)."""
        import pandas as pd
        from joblib import Parallel, delayed
        num_myotubes = len(instances['masks'])
        print(f"\n📊 Computing measurements for {num_myotubes} myotubes...")

        # Get scale factor and original size if available
        scale_factor = 1.0 / self._scale_factor if hasattr(self, '_scale_factor') and self._scale_factor != 1.0 else None
        original_size = self._original_size if hasattr(self, '_original_size') else None

        # Process all myotubes in parallel
        measurements = Parallel(n_jobs=64, verbose=0, backend='loky')(
            delayed(self._process_single_myotube_measurements)(
                mask, score, box, i, scale_factor, original_size
            )
            for i, (mask, score, box) in enumerate(zip(
                instances['masks'], instances['scores'], instances['boxes']
            ))
        )

        # Save to CSV
        df = pd.DataFrame(measurements)
        df.to_csv(output_path, index=False)
        print(f"💾 Saved myotube measurements for {len(measurements)} instances to {output_path}")

    @staticmethod
    def _process_single_myotube_measurements(mask: np.ndarray, score: float, box: np.ndarray,
                                            instance_id: int, scale_factor: Optional[float] = None,
                                            original_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Process measurements for a single myotube (for parallel processing).
        """
        from skimage import measure, morphology
        import cv2
        import numpy as np

        # Resize mask to original size if needed
        if scale_factor is not None and scale_factor != 1.0 and original_size is not None:
            original_h, original_w = original_size
            mask_uint8 = (mask * 255).astype(np.uint8)
            resized_mask = cv2.resize(
                mask_uint8,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            )
            resized_mask = (resized_mask > 128).astype(bool)

            # Scale bounding box back to original coordinates
            scaled_box = box / scale_factor
        else:
            resized_mask = mask.astype(bool)
            scaled_box = box

        # Calculate existing measurements
        area = resized_mask.sum()
        contours = measure.find_contours(resized_mask, 0.5)
        perimeter = sum(len(contour) for contour in contours)

        # Calculate myotube-specific measurements
        visible_length, estimated_total_length, num_components = MyotubeFijiIntegration._calculate_myotube_length_static(resized_mask)
        width_pixels = area / visible_length if visible_length > 0 else 0
        myotube_aspect_ratio = estimated_total_length / width_pixels if width_pixels > 0 else 0

        # Calculate bounding box measurements
        bbox_width = scaled_box[2] - scaled_box[0]
        bbox_height = scaled_box[3] - scaled_box[1]
        bbox_aspect_ratio = max(bbox_width, bbox_height) / min(bbox_width, bbox_height) if min(bbox_width, bbox_height) > 0 else 0

        return {
            'Instance': f'Myotube_{instance_id+1}',
            'Area': int(area),
            'Visible_Length_pixels': round(visible_length, 2),
            'Estimated_Total_Length_pixels': round(estimated_total_length, 2),
            'Width_pixels': round(width_pixels, 2),
            'Aspect_Ratio': round(myotube_aspect_ratio, 2),
            'Connected_Components': num_components,
            'Perimeter': round(perimeter, 2),
            'BBox_AspectRatio': round(bbox_aspect_ratio, 2),
            'Confidence': round(score, 4),
            'BoundingBox_X': round(scaled_box[0], 1),
            'BoundingBox_Y': round(scaled_box[1], 1),
            'BoundingBox_Width': round(bbox_width, 1),
            'BoundingBox_Height': round(bbox_height, 1)
        }

    @staticmethod
    def _calculate_myotube_length_static(mask: np.ndarray) -> Tuple[float, float, int]:
        """
        Static version of _calculate_myotube_length for parallel processing.
        Calculate visible and estimated total myotube length.
        """
        from skimage import morphology, measure
        import numpy as np

        # Find connected components
        labeled_mask = measure.label(mask)
        num_components = labeled_mask.max()

        if num_components == 0:
            return 0.0, 0.0, 0

        # Calculate skeleton for each component
        component_skeletons = []
        total_visible_length = 0

        for component_id in range(1, num_components + 1):
            component_mask = (labeled_mask == component_id)
            skeleton = morphology.skeletonize(component_mask)
            skeleton_points = np.argwhere(skeleton)

            if len(skeleton_points) > 0:
                component_skeletons.append(skeleton_points)
                total_visible_length += len(skeleton_points)

        # Estimate total length including gaps between components
        estimated_total_length = total_visible_length

        if len(component_skeletons) > 1:
            # Add estimated lengths of gaps between components
            gap_length = MyotubeFijiIntegration._estimate_gap_lengths_static(component_skeletons)
            estimated_total_length += gap_length

        return total_visible_length, estimated_total_length, num_components

    @staticmethod
    def _estimate_gap_lengths_static(component_skeletons: List[np.ndarray]) -> float:
        """Static version for parallel processing. Estimate total length of gaps between skeleton components."""
        import numpy as np
        from scipy.spatial.distance import cdist

        if len(component_skeletons) < 2:
            return 0.0

        total_gap_length = 0.0

        # Find endpoints of each component skeleton
        component_endpoints = []
        for skeleton_points in component_skeletons:
            if len(skeleton_points) == 0:
                continue

            # For each component, find the two points that are farthest apart
            if len(skeleton_points) == 1:
                endpoints = [skeleton_points[0], skeleton_points[0]]
            else:
                # Calculate pairwise distances
                distances = cdist(skeleton_points, skeleton_points)
                # Find the pair with maximum distance
                max_idx = np.unravel_index(distances.argmax(), distances.shape)
                endpoints = [skeleton_points[max_idx[0]], skeleton_points[max_idx[1]]]

            component_endpoints.append(endpoints)

        # Estimate gaps between adjacent components
        for i in range(len(component_endpoints) - 1):
            # Find minimum distance between endpoints of consecutive components
            min_gap = float('inf')
            for ep1 in component_endpoints[i]:
                for ep2 in component_endpoints[i + 1]:
                    gap = np.linalg.norm(ep1 - ep2)
                    min_gap = min(min_gap, gap)

            total_gap_length += min_gap

        return total_gap_length
    
    def _save_info(self, instances: Dict[str, Any], image_path: str, output_path: str, original_image: np.ndarray = None):
        """Save processing information."""
        # Use original image shape if available, otherwise use processing shape
        if original_image is not None:
            image_shape = original_image.shape[:2]
        else:
            image_shape = instances['image_shape']

        info = {
            'input_image': os.path.basename(image_path),
            'num_instances': len(instances['masks']),
            'image_shape': image_shape,
            'config_file': self.config_file,
            'model_weights': self.model_weights,
            'post_processing_config': self._post_processor.config,
            'processing_steps': [step['name'] for step in self._post_processor.steps]
        }

        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)

    # Note: Additional helper methods for merged masks have been omitted for brevity
    # but can be added from the original file if needed. The key methods are:
    # - _save_merged_visualization_masks
    # - _create_merged_mask
    # - _analyze_component_for_merging
    # - _find_skeleton_endpoints
    # - _find_compatible_connections
    # - _are_components_compatible
    # - _calculate_min_distance_between_components
    # - _check_reasonable_continuation
    # - _fill_tissue_region
    # - _create_connection_region
    # - _create_tissue_path
    # - _calculate_local_endpoint_width
    # - _measure_cross_sectional_width
    # - _measure_neighborhood_width
    # - _measure_conservative_neighborhood_width
    # - _calculate_csv_width
    # - _count_components
