# Mask2Former Myotube Finetuning Setup

This directory contains all the necessary files to finetune Mask2Former on your custom myotube dataset.

## 📁 Files Overview

- **`myotube_config.yaml`** - Custom configuration file for training
- **`register_myotube_dataset.py`** - Script to register your dataset with Detectron2
- **`train_myotube.py`** - Main training script
- **`setup_finetuning.py`** - Setup verification script
- **`README_myotube_finetuning.md`** - This file

## 🗂️ Dataset Structure

Your dataset is now organized inside the Mask2Former directory:
```
Mask2Former/
├── myotube_batch_output/           # ← Your dataset
│   ├── annotations/
│   │   └── instances_train.json    # COCO format annotations
│   └── images/                     # Training images
│       ├── T33_merged_*.png
│       ├── T34_merged_*.png
│       └── T35_merged_*.png
├── myotube_config.yaml             # ← Custom config
├── train_myotube.py                # ← Training script
└── [other Mask2Former files]
```

**Dataset Statistics:**
- Images: 23
- Annotations: 920
- Classes: 1 (myotube)

## 🚀 Quick Start

**Important**: Run all commands from the `Mask2Former/` directory:
```bash
cd Mask2Former
```

### 1. Verify Setup
```bash
python setup_finetuning.py
```
This will check if all required files and dependencies are in place.

### 2. Start Training
```bash
python train_myotube.py
```

### 3. Monitor Training
- Logs and checkpoints will be saved to `./output_myotube/`
- Training runs for 5000 iterations with evaluation every 500 iterations

### 4. Evaluate Model
```bash
python train_myotube.py --eval-only
```

## ⚙️ Configuration Details

### Model Configuration
- **Architecture**: Mask2Former with Swin-Base backbone
- **Pre-trained weights**: `Mask2Former/model_final_54b88a.pkl`
- **Classes**: 1 (myotube) - changed from 80 (COCO)
- **Input size**: 1024×1024

### Training Configuration
- **Batch size**: 8 (reduced for single GPU)
- **Learning rate**: 0.00005 (half of original for finetuning)
- **Iterations**: 5000 (suitable for small dataset)
- **Optimizer**: AdamW with gradient clipping
- **LR schedule**: Steps at 3000 and 4500 iterations

### Data Configuration
- **Dataset format**: COCO instance segmentation
- **Augmentation**: Large Scale Jittering (LSJ)
- **Workers**: 2 (for single GPU setup)

## 🎯 Training Tips

### For Better Results:
1. **Data Augmentation**: The config uses LSJ augmentation which is effective for small datasets
2. **Learning Rate**: Start with the configured LR (0.00005) and adjust if needed
3. **Iterations**: 5000 iterations should be sufficient for your dataset size
4. **Evaluation**: Monitor validation metrics every 500 iterations

### Troubleshooting:
- **Out of Memory**: Reduce `IMS_PER_BATCH` in config
- **Slow Convergence**: Increase learning rate slightly
- **Overfitting**: Reduce iterations or add more augmentation

## 📊 Expected Training Time

With your dataset (23 images, 920 annotations):
- **Single GPU (RTX 3080/4090)**: ~30-60 minutes
- **CPU only**: Not recommended (several hours)

## 🔧 Advanced Usage

### Custom Configuration
Edit `myotube_config.yaml` to modify:
- Learning rate: `SOLVER.BASE_LR`
- Batch size: `SOLVER.IMS_PER_BATCH`
- Training iterations: `SOLVER.MAX_ITER`
- Output directory: `OUTPUT_DIR`

### Multi-GPU Training
```bash
python train_myotube.py --num-gpus 2
```

### Resume Training
```bash
python train_myotube.py --resume
```

### Custom Training Arguments
```bash
python train_myotube.py --config-file myotube_config.yaml \
    --opts SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 10000
```

## 📈 Monitoring Training

### TensorBoard (if available)
```bash
tensorboard --logdir output_myotube
```

### Log Files
- Training logs: `output_myotube/log.txt`
- Metrics: `output_myotube/metrics.json`

## 🎯 Model Output

After training, you'll find:
- **`model_final.pth`** - Final trained model
- **`model_best.pth`** - Best model based on validation metrics
- **Evaluation results** - Performance metrics on your dataset

## 🆘 Support

If you encounter issues:

1. **Check Setup**: Run `python setup_finetuning.py` again
2. **Verify Dependencies**: Ensure Detectron2 and PyTorch are properly installed
3. **GPU Memory**: Reduce batch size if you get CUDA out of memory errors
4. **Dataset Format**: Ensure your COCO annotations are valid

## 📚 References

- [Mask2Former Paper](https://arxiv.org/abs/2112.01527)
- [Mask2Former GitHub](https://github.com/facebookresearch/Mask2Former)
- [Detectron2 Documentation](https://detectron2.readthedocs.io/)

---

**Ready to train? Run `python setup_finetuning.py` to get started! 🚀** 