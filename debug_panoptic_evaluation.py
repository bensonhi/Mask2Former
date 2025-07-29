#!/usr/bin/env python3
"""
Debug script to instrument panoptic evaluation and identify why n=0 in pq_average.
This script patches the panopticapi evaluation to add detailed logging.
"""

import os
import sys
import json
import tempfile
import shutil
sys.path.append('.')

# Patch panopticapi to add logging
def patch_panoptic_evaluation():
    """Add detailed logging to panopticapi evaluation functions."""
    
    # Import after path setup
    from panopticapi import evaluation
    
    # Store original functions
    original_pq_average = evaluation.PQStat.pq_average
    original_pq_compute = evaluation.pq_compute
    original_pq_compute_single_core = evaluation.pq_compute_single_core
    
    def debug_pq_average(self, categories, isthing=None):
        """Instrumented version of pq_average with detailed logging."""
        print(f"\n🔍 DEBUG pq_average called:")
        print(f"   Categories: {len(categories)} categories")
        for i, cat in enumerate(categories):
            print(f"      {i}: {cat}")
        print(f"   isthing filter: {isthing}")
        
        if isthing is not None:
            categories = [cat for cat in categories if cat['isthing'] == isthing]
            print(f"   After isthing filter: {len(categories)} categories")
        
        N = len(categories)
        print(f"   N (final category count): {N}")
        
        if N == 0:
            print("   ❌ CRITICAL: N=0 - This will cause division by zero!")
            print("   Available PQStat data:")
            print(f"      self.pq: {getattr(self, 'pq', 'MISSING')}")
            print(f"      self.sq: {getattr(self, 'sq', 'MISSING')}")  
            print(f"      self.rq: {getattr(self, 'rq', 'MISSING')}")
            return original_pq_average(self, categories, isthing)
            
        # Call original with instrumentation
        result = original_pq_average(self, categories, isthing)
        print(f"   ✅ pq_average completed successfully")
        return result
    
    def debug_pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
        """Instrumented version of pq_compute_single_core."""
        print(f"\n🔍 DEBUG pq_compute_single_core proc_id={proc_id}:")
        print(f"   annotation_set: {len(annotation_set)} annotations")
        print(f"   gt_folder: {gt_folder}")
        print(f"   pred_folder: {pred_folder}")
        print(f"   categories: {len(categories)} categories")
        
        # Check if folders exist and have files
        if os.path.exists(gt_folder):
            gt_files = os.listdir(gt_folder)
            print(f"   GT folder has {len(gt_files)} files")
        else:
            print(f"   ❌ GT folder doesn't exist!")
            
        if os.path.exists(pred_folder):
            pred_files = os.listdir(pred_folder)
            print(f"   Pred folder has {len(pred_files)} files")
        else:
            print(f"   ❌ Pred folder doesn't exist!")
        
        # Call original
        result = original_pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories)
        print(f"   ✅ pq_compute_single_core completed")
        return result
    
    def debug_pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):
        """Instrumented version of pq_compute."""
        print(f"\n🔍 DEBUG pq_compute called:")
        print(f"   gt_json_file: {gt_json_file}")
        print(f"   pred_json_file: {pred_json_file}")
        print(f"   gt_folder: {gt_folder}")
        print(f"   pred_folder: {pred_folder}")
        
        # Load and examine JSON files
        if os.path.exists(gt_json_file):
            with open(gt_json_file, 'r') as f:
                gt_data = json.load(f)
            print(f"   GT JSON: {len(gt_data.get('annotations', []))} annotations, {len(gt_data.get('categories', []))} categories")
            for cat in gt_data.get('categories', []):
                print(f"      GT Category: {cat}")
        else:
            print(f"   ❌ GT JSON doesn't exist!")
            
        if os.path.exists(pred_json_file):
            with open(pred_json_file, 'r') as f:
                pred_data = json.load(f)
            print(f"   Pred JSON: {len(pred_data.get('annotations', []))} annotations, {len(pred_data.get('categories', []))} categories")
            for cat in pred_data.get('categories', []):
                print(f"      Pred Category: {cat}")
        else:
            print(f"   ❌ Pred JSON doesn't exist!")
        
        # Call original
        result = original_pq_compute(gt_json_file, pred_json_file, gt_folder, pred_folder)
        print(f"   ✅ pq_compute completed")
        return result
    
    # Apply patches
    evaluation.PQStat.pq_average = debug_pq_average
    evaluation.pq_compute = debug_pq_compute
    evaluation.pq_compute_single_core = debug_pq_compute_single_core
    
    print("✅ Panoptic evaluation instrumentation applied!")

def test_evaluation_with_instrumentation():
    """Test the evaluation with instrumentation to see where it fails."""
    
    # Apply patches first
    patch_panoptic_evaluation()
    
    # Now import detectron2 components (after patching)
    from register_two_stage_datasets import register_two_stage_datasets
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.evaluation import COCOPanopticEvaluator
    from detectron2.config import get_cfg
    from mask2former import add_maskformer2_config
    
    print("🧪 Testing panoptic evaluation with instrumentation...")
    
    # Register datasets
    register_two_stage_datasets(
        dataset_root="myotube_batch_output",
        register_instance=False,
        register_panoptic=True
    )
    
    # Setup minimal config
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    
    # Create evaluator
    with tempfile.TemporaryDirectory() as output_dir:
        print(f"Using temp output dir: {output_dir}")
        
        try:
            evaluator = COCOPanopticEvaluator("myotube_stage1_panoptic_val", output_dir)
            print("✅ COCOPanopticEvaluator created successfully")
            
            # Check metadata
            metadata = MetadataCatalog.get("myotube_stage1_panoptic_val")
            print(f"📊 Dataset metadata:")
            print(f"   thing_classes: {getattr(metadata, 'thing_classes', 'MISSING')}")
            print(f"   stuff_classes: {getattr(metadata, 'stuff_classes', 'MISSING')}")
            print(f"   thing_dataset_id_to_contiguous_id: {getattr(metadata, 'thing_dataset_id_to_contiguous_id', 'MISSING')}")
            print(f"   stuff_dataset_id_to_contiguous_id: {getattr(metadata, 'stuff_dataset_id_to_contiguous_id', 'MISSING')}")
            
        except Exception as e:
            print(f"❌ Failed to create evaluator: {e}")
            import traceback
            traceback.print_exc()

def main():
    print("🔍 Panoptic Evaluation Debug with Instrumentation")
    print("=" * 60)
    test_evaluation_with_instrumentation()

if __name__ == "__main__":
    main() 