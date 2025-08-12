# Myotube Instance Segmentation for Fiji

This package provides seamless integration of AI-powered myotube instance segmentation into Fiji/ImageJ. Lab members can segment myotubes with a single button click, without needing to interact with code or command lines.

## 🎯 Features

- **One-click segmentation**: Just click a button in Fiji
- **Automatic ROI generation**: Each myotube becomes a ROI in ROI Manager
- **Colored overlay visualization**: See segmentation results overlaid on original image
- **Measurement export**: Automatic CSV export of myotube properties
- **Modular post-processing**: Easy to add custom filtering and refinement steps
- **Multi-segment support**: Handles myotubes that may be split into multiple segments
- **Quality control**: Built-in filtering for size, confidence, and shape

## 📋 Setup Instructions

### 1. Install Python Environment

**Using Conda (Required for Fiji Integration)**
```bash
# Create new environment named 'm2f' (as expected by Fiji macro)
conda create -n m2f python=3.8
conda activate m2f

# Install PyTorch (choose based on your system)
# For CPU only:
pip install torch torchvision

# For GPU (CUDA 11.3):
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# Install other requirements
pip install -r requirements.txt
```

**⚠️ Important:** The Fiji macro is configured to use conda environment named `m2f`. If you want to use a different name, you'll need to edit the `CONDA_ENV` variable in the macro file.

### 2. Install Detectron2

Detectron2 requires special installation:

```bash
# For CPU only
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# For GPU (CUDA 11.3)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# For other CUDA versions, see: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
```

### 3. Setup Fiji Integration

1. **Copy files to Fiji:**
   ```
   # Copy these files to your Fiji installation:
   myotube_segmentation.py  → Fiji.app/macros/ (or plugins/)
   Myotube_Segmentation.ijm → Fiji.app/macros/
   ```

2. **Test conda environment:**
   - Open Terminal/Command Prompt
   - Type: `conda activate m2f`
   - Type: `python --version` (should show Python 3.8+)
   - Type: `python -c "import torch; print('PyTorch OK')"` (should print "PyTorch OK")

3. **Configure environment (if needed):**
   - Open `Myotube_Segmentation.ijm` in text editor
   - Modify `CONDA_ENV` if you used a different environment name
   - Modify `PYTHON_COMMAND` if needed (usually `python` works within conda env)
   - Save and restart Fiji

### 4. Verify Setup

1. Open Fiji
2. Open a test myotube image
3. Go to `Plugins > Macros > Run...` and select `Myotube_Segmentation.ijm`
4. Click "Segment Myotubes" or press 'M'
5. If setup is correct, you should see progress messages and results

## 🚀 Usage

### Basic Workflow

1. **Open image in Fiji**
   - File > Open > Select your myotube image
   - Supported formats: TIFF, PNG, JPEG, etc.

2. **Run segmentation**
   - Method 1: Press 'M' key (shortcut)
   - Method 2: Go to Plugins > Macros > Segment Myotubes
   - Method 3: Use macro toolbar if configured

3. **View results**
   - ROIs automatically appear in ROI Manager
   - Colored overlay opens in new window
   - Check console/log for processing details

4. **Analyze results**
   - Select ROIs in ROI Manager to highlight myotubes
   - Use "Measure" to get area, perimeter, etc.
   - Delete false positives by selecting and pressing Delete
   - Export measurements: ROI Manager > More > Save

### Advanced Usage

**Custom Parameters:**
- Use "Segment Myotubes (Custom Parameters)" for fine-tuning
- Adjust confidence threshold (0-1): higher = fewer, more confident detections
- Adjust size filters: min/max area in pixels
- Modify post-processing steps in Python script

**Batch Processing:**
```ijm
// Example ImageJ macro for batch processing
input_dir = getDirectory("Choose input folder");
output_dir = getDirectory("Choose output folder");

list = getFileList(input_dir);
for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".tif")) {
        open(input_dir + list[i]);
        // Run segmentation macro here
        // Save results
        close();
    }
}
```

## 📊 Output Files

For each processed image, the system generates:

| File | Description | Use |
|------|-------------|-----|
| `*_rois.zip` | ROI Manager compatible file | Load ROIs into Fiji |
| `*_overlay.tif` | Colored visualization | View segmentation results |
| `*_measurements.csv` | Quantitative data | Statistical analysis |
| `*_info.json` | Processing metadata | Reproducibility tracking |

## 🔧 Troubleshooting

### Common Issues

**"Could not execute Python command"**
- Check Python installation: `python --version`
- Try changing `PYTHON_COMMAND` to `python3` or full path
- Ensure Python is in system PATH

**"Could not find myotube_segmentation.py script"**
- Verify script is in Fiji macros or plugins folder
- Check file permissions (should be readable)
- Restart Fiji after copying files

**"No myotubes detected"**
- Check image quality and contrast
- Lower confidence threshold in custom parameters
- Verify image contains myotube-like structures
- Check console log for processing details

**Segmentation runs but no ROIs appear**
- Check if ROI Manager is open (Window > ROI Manager)
- Look for error messages in console
- Try "Load Myotube Results" to reload previous results

**Performance issues**
- Use GPU version of PyTorch for faster processing
- Process smaller image tiles for very large images
- Close other applications to free memory

### Getting Help

1. **Check console output**: Window > Console (in Fiji)
2. **Review log messages**: Processing details and errors are logged
3. **Test components separately**:
   - Test Python: `python -c "import torch; print('OK')"`
   - Test script: `python myotube_segmentation.py test_image.tif output_dir`
   - Test macro: Run individual macro functions

## ⚙️ Customization

### Adding Post-processing Steps

The system uses a modular post-processing pipeline. To add custom steps:

1. **Edit `myotube_segmentation.py`**
2. **Add new processing function**:
   ```python
   def _my_custom_filter(self, instances, image):
       # Your custom processing logic here
       # Must return instances dictionary
       return instances
   ```
3. **Register the step**:
   ```python
   def _setup_default_pipeline(self):
       # ... existing steps ...
       self.add_step('my_custom_filter', self._my_custom_filter)
   ```

### Example Custom Filters

**Filter by elongation (myotube shape):**
```python
def _filter_by_elongation(self, instances, image):
    keep_indices = []
    for i, mask in enumerate(instances['masks']):
        # Calculate major/minor axis ratio
        elongation = calculate_elongation(mask)
        if elongation > 2.0:  # Keep elongated objects
            keep_indices.append(i)
    # Return filtered instances
    return filter_instances(instances, keep_indices)
```

**Filter by fluorescence intensity:**
```python
def _filter_by_intensity(self, instances, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keep_indices = []
    for i, mask in enumerate(instances['masks']):
        mean_intensity = gray_image[mask].mean()
        if mean_intensity > threshold:
            keep_indices.append(i)
    return filter_instances(instances, keep_indices)
```

### Modifying Parameters

**Default parameters** can be changed in the Python script:
```python
def get_default_config(self):
    return {
        'min_area': 200,           # Increase minimum size
        'confidence_threshold': 0.7, # Higher confidence requirement
        'fill_holes': True,        # Fill segmentation holes
        # ... other parameters
    }
```

**Runtime parameters** can be adjusted via the Fiji macro interface.

## 📈 Model Performance

### Expected Performance

- **Accuracy**: Depends on training data quality and image similarity
- **Speed**: ~5-30 seconds per image (depending on size and hardware)
- **Memory**: ~2-8GB RAM (depending on model and image size)

### Optimization Tips

1. **Use GPU**: 3-10x faster processing
2. **Optimize image size**: Resize very large images before processing
3. **Adjust parameters**: Lower confidence for more detections, higher for precision
4. **Batch processing**: Process multiple images in sequence

## 🔄 Updates and Maintenance

### Updating the Model

1. Train new model or get updated weights
2. Copy new weights file to project directory
3. Update path in macro or let auto-detection find it
4. Test with validation images

### Software Updates

1. **Update Python packages**: `pip install --upgrade -r requirements.txt`
2. **Update Detectron2**: Follow official update instructions
3. **Update Fiji**: Use Help > Update Fiji in application

## 📝 Citation and Acknowledgments

If you use this software in your research, please cite:

```bibtex
@software{myotube_segmentation_fiji,
  title={Myotube Instance Segmentation for Fiji},
  author={[Your Lab/Name]},
  year={2024},
  url={[Repository URL]}
}
```

Built using:
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Fiji/ImageJ](https://fiji.sc/)

## 📧 Support

For questions, issues, or contributions:
- Create an issue on the project repository
- Contact: [Your lab contact information]
- Documentation: [Link to detailed docs if available]

---

**Happy segmenting! 🔬✨**