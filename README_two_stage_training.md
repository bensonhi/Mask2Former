# Two-Stage Myotube Training with Mask2Former

This system implements a sophisticated two-stage training approach for myotube segmentation, combining the strengths of algorithmic annotations and manual annotations for optimal performance.

## 🎯 **Two-Stage Training Strategy**

### **Stage 1: Robust Feature Learning** 
- **Dataset**: ~100 images with algorithmic annotations
- **Purpose**: Learn general myotube features from large, diverse dataset
- **Annotations**: Generated automatically using traditional CV methods
- **Training**: 3000 iterations with conservative learning rate

### **Stage 2: Precision Fine-tuning**
- **Dataset**: ~5 images with high-quality manual annotations  
- **Purpose**: Fine-tune for precise, high-quality segmentation
- **Annotations**: Carefully crafted manual annotations
- **Training**: 1000 iterations starting from Stage 1 checkpoint

## 📁 **Project Structure**

```
Mask2Former/
├── 🔧 Two-Stage Training Files
│   ├── stage1_config.yaml              # Stage 1 configuration
│   ├── stage2_config.yaml              # Stage 2 configuration
│   ├── register_two_stage_datasets.py  # Dataset registration
│   ├── train_two_stage.py              # Main training script
│   ├── setup_two_stage.py              # Setup verification
│   ├── organize_unified_dataset.py     # Dataset organization helper
│   └── README_two_stage_training.md    # This file
│
├── 📊 Unified Dataset
│   └── myotube_dataset/                 # Combined dataset
│       ├── images/                      # All images
│       └── annotations/
│           ├── algorithmic_train_annotations.json    # Stage 1 training
│           ├── algorithmic_test_annotations.json     # Stage 1 validation  
│           ├── manual_train_annotations.json         # Stage 2 training
│           └── manual_test_annotations.json          # Stage 2 validation
│
└── 📈 Training Outputs
    ├── output_stage1_algorithmic/       # Stage 1 results
    └── output_stage2_manual/            # Stage 2 results (final model)
```

## 🚀 **Quick Start**

### **1. Setup Verification**
```bash
python setup_two_stage.py
```
This automatically counts your dataset images and annotations, checks all requirements, and provides setup guidance.

### **2. Create Datasets**

#### **Create Unified Dataset Structure**
```bash
# Create directory structure
mkdir -p myotube_dataset/{images,annotations}

# Generate algorithmic annotations from raw images
cd utils
python batch_myotube_processing.py \
    --input_dir /path/to/raw/images \
    --output_dir ../temp_algorithmic \
    --resolution 1500

# Move to unified structure
mv temp_algorithmic/annotations/train_annotations.json ../myotube_dataset/annotations/algorithmic_train_annotations.json
mv temp_algorithmic/annotations/test_annotations.json ../myotube_dataset/annotations/algorithmic_test_annotations.json
cp temp_algorithmic/images/* ../myotube_dataset/images/

# Create manual annotations for ~5 selected images
# Use CVAT, LabelMe, or VIA to create:
# - manual_train_annotations.json
# - manual_test_annotations.json (optional)
# Place in myotube_dataset/annotations/
```

#### **Converting Existing Separate Datasets**
If you already have separate algorithmic and manual datasets:
```bash
# Organize existing datasets into unified structure
python organize_unified_dataset.py \
    --algorithmic_dataset myotube_batch_output \
    --manual_dataset manual_dataset \
    --output_dataset myotube_dataset

# Or convert just algorithmic dataset
python organize_unified_dataset.py \
    --algorithmic_dataset myotube_batch_output \
    --output_dataset myotube_dataset
```

### **3. Run Training**

#### **Full Two-Stage Training**
```bash
python train_two_stage.py
```

#### **Individual Stages**
```bash
# Stage 1 only (algorithmic annotations)
python train_two_stage.py --stage 1

# Stage 2 only (manual fine-tuning)
python train_two_stage.py --stage 2
```

#### **Custom Dataset Path**
```bash
python train_two_stage.py --dataset /path/to/myotube_dataset
```

## ⚙️ **Configuration Details**

### **Stage 1 Configuration** (`stage1_config.yaml`)
- **Architecture**: Mask2Former + Swin-Base backbone
- **Pre-trained weights**: COCO instance segmentation model
- **Image size**: 1500×1500 (matches original microscopy resolution)
- **Batch size**: 2 (optimized for high-resolution images)
- **Learning rate**: 0.00003 (conservative for dense structures)
- **Iterations**: 4000 (extended for complex overlapping myotubes)
- **Object queries**: 200 (increased for dense structures)
- **Augmentations**: Conservative scaling to preserve thin structures

### **Stage 2 Configuration** (`stage2_config.yaml`)
- **Architecture**: Same as Stage 1
- **Pre-trained weights**: Stage 1 checkpoint
- **Image size**: 1500×1500 (preserves manual annotation precision)
- **Batch size**: 1 (intensive processing for high-resolution fine-tuning)
- **Learning rate**: 0.00008 (optimized for precise tuning)
- **Iterations**: 1500 (extended for complex manual annotations)
- **Object queries**: 200 (consistency with Stage 1)
- **Augmentations**: Careful scaling to preserve annotation boundaries

## 📊 **Training Monitoring**

### **Automatic Dataset Counting**
The system automatically counts images and annotations from your JSON files:
- No need to manually specify dataset sizes
- Real-time display of training data statistics
- Automatic validation of dataset completeness

### **Progress Tracking**
- **Stage 1 logs**: `./output_stage1_algorithmic/log.txt`
- **Stage 2 logs**: `./output_stage2_manual/log.txt`
- **Metrics**: JSON files in respective output directories

### **Checkpoints**
- **Stage 1 final**: `./output_stage1_algorithmic/model_final.pth`
- **Stage 2 final**: `./output_stage2_manual/model_final.pth` *(use for inference)*

### **Evaluation**
```bash
# Evaluate Stage 1
python train_net.py --config-file stage1_config.yaml --eval-only

# Evaluate Stage 2  
python train_net.py --config-file stage2_config.yaml --eval-only
```

## 🔬 **Scientific Rationale**

### **Optimized for Dense Myotube Structures**

Based on analysis of your microscopy images (1500×1500 resolution), the configuration has been optimized for:
- **High-density overlapping myotubes** with extensive crossing and branching
- **Variable morphology** from thin elongated to curved and branched structures  
- **Bright punctate signals** (nuclei/organelles) requiring careful discrimination
- **Complex connectivity** requiring preservation of full myotube lengths

### **Why Two-Stage Training?**

1. **Large-Scale Feature Learning**: Stage 1 leverages ~100 images to learn robust myotube features, handling variations in morphology, orientation, and density.

2. **Quality Refinement**: Stage 2 uses carefully annotated examples to fine-tune the model for precise boundary detection and complex shape handling.

3. **Noise Robustness**: Starting with algorithmic annotations builds resistance to annotation noise, while manual fine-tuning ensures high precision.

4. **Data Efficiency**: Maximizes the value of limited manual annotation effort by building on a strong foundation.

### **Expected Improvements**
- **Better generalization** across different imaging conditions
- **Improved boundary precision** from manual annotations
- **Reduced overfitting** on small manual datasets
- **More robust feature representations**

## 🎯 **Performance Optimization**

### **Memory Management**
- Adjust `IMS_PER_BATCH` in configs if experiencing OOM errors
- Stage 1: Can use larger batches (4-8)
- Stage 2: Use smaller batches (1-2) for precision

### **Training Speed**
- **GPU**: Recommended for reasonable training times
- **Multiple GPUs**: Use `--num-gpus 2` for faster training
- **CPU**: Not recommended (very slow)

### **Hyperparameter Tuning**
```bash
# Adjust learning rates
python train_two_stage.py --stage 1 --opts SOLVER.BASE_LR 0.0001

# Modify training length
python train_two_stage.py --stage 2 --opts SOLVER.MAX_ITER 1500

# Change batch sizes
python train_two_stage.py --opts SOLVER.IMS_PER_BATCH 6
```

## 🔧 **Advanced Usage**

### **Resume Training**
```bash
# Resume Stage 1
python train_two_stage.py --stage 1 --resume

# Resume Stage 2
python train_two_stage.py --stage 2 --resume
```

### **Custom Preprocessing**
Modify `utils/batch_myotube_processing.py` for:
- Different image resolutions
- Custom segmentation parameters
- Alternative annotation formats

### **Multi-GPU Training**
```bash
python train_two_stage.py --num-gpus 4
```

## 📈 **Expected Training Times**

| Stage | Dataset Size | GPU (RTX 4090) | GPU (RTX 3080) | 
|-------|-------------|----------------|----------------|
| Stage 1 | ~100 images (1500×1500) | 4-5 hours | 6-7 hours |
| Stage 2 | ~5 images (1500×1500) | 1-1.5 hours | 1.5-2 hours |
| **Total** | | **5-6.5 hours** | **7.5-9 hours** |

*Note: Times increased due to higher resolution (1500×1500) processing*

## 🆘 **Troubleshooting**

### **Common Issues**

#### **Dataset Not Found**
```bash
# Check unified dataset structure
python setup_two_stage.py

# Regenerate algorithmic annotations
cd utils && python batch_myotube_processing.py --input_dir /path/to/images --output_dir ../temp_algorithmic --resolution 1500

# Organize into unified structure
mkdir -p myotube_dataset/{images,annotations}
mv temp_algorithmic/annotations/train_annotations.json myotube_dataset/annotations/algorithmic_train_annotations.json
cp temp_algorithmic/images/* myotube_dataset/images/
```

#### **CUDA Out of Memory**
```bash
# Reduce batch size
python train_two_stage.py --opts SOLVER.IMS_PER_BATCH 2
```

#### **Stage 1 Checkpoint Missing**
```bash
# Check if Stage 1 completed successfully
ls output_stage1_algorithmic/

# Run Stage 1 separately
python train_two_stage.py --stage 1
```

#### **Low Performance**
- Verify annotation quality in manual dataset
- Check if images are properly preprocessed
- Ensure sufficient training iterations

### **Getting Help**
1. Run `python setup_two_stage.py` for diagnostic information
2. Check log files in output directories
3. Verify dataset formats with COCO validation tools

## 📚 **References**

- **Mask2Former**: [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)
- **Two-stage training**: Common approach in computer vision for leveraging both large and high-quality datasets
- **Instance segmentation**: [Detectron2 documentation](https://detectron2.readthedocs.io/)

---

**🎯 Ready to achieve state-of-the-art myotube segmentation with two-stage training!** 