# Fiji Integration Setup Guide for Windows

**Complete beginner's guide to setting up and using automated myotube and nuclei analysis in Fiji**

This guide will help you install and run the myotube segmentation, nuclei segmentation, and analysis tools through Fiji (ImageJ) on Windows. No programming experience required!

---

## Table of Contents
1. [What You'll Need](#what-youll-need)
2. [Step 1: Install Fiji](#step-1-install-fiji)
3. [Step 2: Install Miniconda](#step-2-install-miniconda)
4. [Step 3: Download This Project](#step-3-download-this-project)
5. [Step 4: Download Trained Model](#step-4-download-trained-model)
6. [Step 5: Copy Files to Fiji](#step-5-copy-files-to-fiji)
7. [Step 6: First-Time Setup](#step-6-first-time-setup)
8. [Step 7: Using the Multi-Tab Interface](#step-7-using-the-multi-tab-interface)
9. [Step 8: Complete Workflow Example](#step-8-complete-workflow-example)
10. [Step 9: Understanding the Results](#step-9-understanding-the-results)
11. [Updating the Tool](#updating-the-tool)
12. [Troubleshooting](#troubleshooting)

---

## What You'll Need

- **Windows 10 or 11** (64-bit)
- **At least 8GB of RAM** (16GB recommended)
- **10GB of free disk space**
- **Internet connection** for downloading software
- **Your microscopy images** (TIFF format recommended)
  - Grey channel images for myotube segmentation
  - Blue channel images for nuclei segmentation

**Time required**: 30-60 minutes for first-time setup

---

## Step 1: Install Fiji

Fiji is a distribution of ImageJ (popular microscopy image analysis software) with useful plugins pre-installed.

### 1.1 Download Fiji

1. Go to: https://fiji.sc/
2. Click **"Downloads"** in the menu
3. Download **"fiji-win64.zip"** (for 64-bit Windows)

### 1.2 Install Fiji

1. **Extract the ZIP file**:
   - Right-click on `fiji-win64.zip`
   - Select **"Extract All..."**
   - Choose a location like `C:\Program Files\Fiji` or `C:\Users\YourUsername\Fiji`
   - Click **"Extract"**

2. **Create a desktop shortcut** (optional but recommended):
   - Navigate to the extracted Fiji folder
   - Find `ImageJ-win64.exe`
   - Right-click it and select **"Create shortcut"**
   - Drag the shortcut to your desktop

3. **Test Fiji**:
   - Double-click `ImageJ-win64.exe` (or your desktop shortcut)
   - Fiji should open with a small toolbar window
   - Close Fiji for now

---

## Step 2: Install Miniconda

Miniconda is a Python distribution manager needed to run the segmentation algorithms.

### 2.1 Download Miniconda

1. Go to: https://docs.conda.io/en/latest/miniconda.html
2. Under **"Windows installers"**, download:
   - **"Miniconda3 Windows 64-bit"** (the .exe file)

### 2.2 Install Miniconda

1. **Run the installer**:
   - Double-click the downloaded `Miniconda3-latest-Windows-x86_64.exe`
   - Click **"Next"**

2. **Important installation options**:
   - **License Agreement**: Click "I Agree"
   - **Installation Type**: Choose "Just Me" (recommended)
   - **Destination Folder**: Use the default (usually `C:\Users\YourUsername\miniconda3`)
   - **Advanced Options** - THIS IS CRITICAL:
     - ✅ **CHECK the box "Add Miniconda3 to my PATH environment variable"**
     - ✅ **CHECK the box "Register Miniconda3 as my default Python"**
     - Even if it says "Not recommended", check both boxes! This is essential for Fiji to find Python.

3. **Complete installation**:
   - Click **"Install"**
   - Wait for installation to complete (5-10 minutes)
   - Click **"Finish"**

### 2.3 Verify Installation

1. Open **Command Prompt**:
   - Press `Windows Key + R`
   - Type `cmd` and press Enter

2. Type this command and press Enter:
   ```
   conda --version
   ```

3. You should see something like: `conda 23.x.x`
   - If you see this, installation was successful!
   - If you get an error, see [Troubleshooting](#troubleshooting)

4. Close Command Prompt

---

## Step 3: Download This Project

### Option A: Download as ZIP from GitHub (Easiest - No Git Required)

1. **Go to the GitHub repository**:
   - Open your web browser
   - Go to: **https://github.com/bensonhi/Mask2Former**

2. **Download the ZIP file**:
   - Look for the green **"Code"** button (near the top right of the page)
   - Click the **"Code"** button
   - In the dropdown menu, click **"Download ZIP"**
   - Your browser will download a file named `Mask2Former-main.zip`

3. **Extract the ZIP file**:
   - Go to your **Downloads** folder
   - Find `Mask2Former-main.zip`
   - Right-click on it and select **"Extract All..."**
   - Choose a destination like `C:\Users\YourUsername\`
   - Click **"Extract"**
   - The extracted folder will be named `Mask2Former-main`
   - You can rename it to just `Mask2Former` if you prefer
   - Example paths: `C:\Users\YourUsername\Mask2Former`
   - **Remember this path** - you'll need it when configuring the tools

### Option B: Clone with Git (If you have Git installed)

1. Open Command Prompt
2. Navigate to where you want the project:
   ```
   cd C:\Users\YourUsername
   ```
3. Clone the repository:
   ```
   git clone https://github.com/bensonhi/Mask2Former.git Mask2Former
   ```

---

## Step 4: Download Trained Model

The trained model file is required for myotube segmentation.

### 4.1 Download the Model

1. **Go to the Google Drive link**:
   - Open your web browser
   - Go to: **https://drive.google.com/file/d/1O0fEGpIZrA2I8SbsuO2cPDSGpRWQK38r/view?usp=sharing**

2. **Download the .pth file**:
   - Click the **"Download"** button (usually in the top right)
   - If prompted, click **"Download anyway"** (the file is safe)
   - The file will be named `model_final.pth`
   - Download location: Usually goes to your **Downloads** folder

3. **Move the model file** (optional but recommended):
   - Create a folder for models: `C:\Users\YourUsername\Mask2Former\models`
   - Move the downloaded `.pth` file to this folder
   - **Remember this location** - you'll need it when running segmentation

**Note**: The model file is large (several hundred MB), so the download may take a few minutes.

---

## Step 5: Copy Files to Fiji

This is a **critical step** - you must copy the entire GUI folder structure to Fiji's macros directory.

### 5.1 Locate the Source Files

1. Open **File Explorer**
2. Navigate to your Mask2Former project folder
3. Open the **`fiji_integration`** folder
4. You should see:
   - `Myotube_Segmentation_Windows.ijm` (main macro file)
   - `Myotube_Segmentation.ijm` (Linux/Mac version)
   - `myotube_segmentation.py`
   - `requirements.txt`
   - **`gui/`** folder (contains the multi-tab interface)
   - **`core/`** folder (contains backend processing)
   - **`utils/`** folder (contains utility functions)

### 5.2 Locate Fiji's Macros Folder

1. Open another File Explorer window
2. Navigate to where you installed Fiji (e.g., `C:\Program Files\Fiji`)
3. Open the **`macros`** folder
   - Full path example: `C:\Program Files\Fiji\macros`

### 5.3 Copy the Files and Folders

1. **Select all files and folders** in the `fiji_integration` folder:
   - `Myotube_Segmentation_Windows.ijm`
   - `Myotube_Segmentation.ijm`
   - `myotube_segmentation.py`
   - `requirements.txt`
   - `gui/` folder
   - `core/` folder
   - `utils/` folder

2. **Copy them** (Ctrl+C or right-click → Copy)
3. **Paste them** into the Fiji `macros` folder (Ctrl+V or right-click → Paste)
4. If asked to replace existing files, click **"Replace"** or **"Yes"**

**Visual check**: Your Fiji `macros` folder should now contain:
```
C:\Program Files\Fiji\macros\
├── Myotube_Segmentation_Windows.ijm    ← NEW
├── Myotube_Segmentation.ijm            ← NEW
├── myotube_segmentation.py             ← NEW
├── requirements.txt                    ← NEW
├── gui\                                ← NEW FOLDER
│   ├── __init__.py
│   ├── main_window.py
│   ├── base_tab.py
│   └── tabs\
│       ├── max_projection_tab.py
│       ├── myotube_tab.py
│       ├── cellpose_tab.py
│       └── analysis_tab.py
├── core\                               ← NEW FOLDER
│   ├── __init__.py
│   ├── segmentation.py
│   ├── tiled_segmentation.py
│   └── ...
└── utils\                              ← NEW FOLDER
    ├── __init__.py
    └── ...
```

---

## Step 6: First-Time Setup

**The first time you run the macro, it will automatically install Python dependencies.**

### 6.1 Launch the Macro

1. **Open Fiji** (double-click `ImageJ-win64.exe`)

2. **Run the macro**:
   - Press the **'M'** key (or go to Plugins → Macros → Run...)
   - A list of macros will appear
   - Select **"Myotube_Segmentation_Windows.ijm"**
   - Click **"Open"**

### 6.2 Automatic Installation (First Time Only)

3. **Install Python Dependencies**:
   - A dialog will appear asking if you want to install dependencies
   - Click **"Install Python Dependencies"**
   - Wait for installation (5-15 minutes first time)
   - You'll see progress messages in a console/terminal window
   - Once installation is complete, the multi-tab GUI will appear automatically

---

## Step 7: Using the Multi-Tab Interface

The GUI has **4 tabs** for different processing steps. You can run them independently or in sequence.

### Tab 1: Max Projection & Channel Splitting

**Purpose**: Split multi-channel Z-stack images into separate grey and blue channels with max projection.

**When to use**: If you have multi-channel Z-stack images that need to be separated before segmentation.

**Steps**:
1. Click the **"Max Projection & Channel Splitting"** tab
2. **Input Directory**: Browse to your multi-channel Z-stack images
3. **Output Directory**: Choose where to save the split channels
4. Click **"Run Channel Splitting"**

**Output**:
- `*_grey.tif` - Grey channel (for myotube segmentation)
- `*_blue.tif` - Blue channel (for nuclei segmentation)

---

### Tab 2: Myotube Segmentation

**Purpose**: Detect and segment myotubes in grey channel images using the trained Mask2Former model.

**Steps**:
1. Click the **"Myotube Segmentation"** tab
2. **Input Directory**: Browse to folder containing grey channel images
3. **Output Directory**: Choose where to save segmentation results
4. **Model Configuration** (first time only):
   - *Config File: Browse to `stage2_config.yaml` in your Mask2Former folder
   - *Model Weights: Browse to the `model_final.pth` you downloaded
   - *Mask2Former Path: Browse to your Mask2Former project folder
5. **Post-Processing Settings** (use defaults initially):
   - Confidence Threshold: 0.5
   - Min/Max Area: 1000 - 1000000 pixels
   - Overlap Threshold: 0.5
6. Click **"Run Segmentation"**

**Output** (for each image):
- `[ImageName]_masks/` - Individual myotube mask PNG files
- `[ImageName]_processed_overlay.tif` - Visualization
- `[ImageName]_raw_overlay.tif` - Raw predictions
- `[ImageName]_info.json` - Processing metadata

---

### Tab 3: Nuclei Segmentation (CellPose)

**Purpose**: Segment nuclei in blue channel images using CellPose.

**Steps**:
1. Click the **"Nuclei Segmentation (CellPose)"** tab
2. **Input Directory**: Browse to folder containing blue channel images
3. **Output Directory**: Choose where to save nuclei segmentation
4. **CellPose Settings**:
   - Model Type: `cyto3` (default) or `nuclei`
   - Diameter: 30 pixels (or 0 for auto-detection)
   - Use GPU: Check if you have CUDA-capable GPU
   - Target Resolution: 3000 (scales down large images for speed)
5. **Output Options**:
   - Save NPY: Check (required for analysis)
   - Save Fiji ROIs: Optional
   - Save Visualization: Check
6. Click **"Run CellPose Segmentation"**

**Output** (for each image):
- `[ImageName]_seg.npy` - Nuclei segmentation masks (required for analysis)
- `[ImageName]_RoiSet.zip` - Fiji ROI file (optional)
- `[ImageName]_overlay.png` - Visualization

---

### Tab 4: Nuclei-Myotube Analysis

**Purpose**: Analyze the relationship between nuclei and myotubes - count nuclei per myotube and filter nuclei by quality.

**Steps**:
1. Click the **"Nuclei-Myotube Analysis"** tab
2. **Myotube Folder**: Browse to myotube segmentation results (from Tab 2)
3. **Nuclei Folder**: Browse to nuclei segmentation results (from Tab 3)
4. **Output Folder**: Choose where to save analysis results
5. **Filter Settings**:
   - Nucleus Size Range: 400-2000 pixels
   - Max Eccentricity: 0.9 (filters out elongated non-nuclei)
   - Overlap Threshold: 60% (minimum overlap to assign nucleus to myotube)
6. **Processing Mode**:
   - Full Image Mode: Check if processing full images (not quadrants)
   - Skip Alignment Resize: Check if images are already aligned
7. Click **"Run Analysis"**

**Output** (for each sample):
- `[Sample]_myotube_nuclei_counts.csv` - Nuclei count per myotube
- `[Sample]_nuclei_myotube_assignments.csv` - Detailed nucleus assignments
- `[Sample]_analysis_summary.txt` - Statistics summary
- `[Sample]_nuclei_overlay.tif` - Color-coded visualization:
  - **GREEN**: Nuclei assigned to myotubes (passed all filters)
  - **RED**: Filtered by size
  - **YELLOW**: Filtered by eccentricity
  - **BLUE**: Filtered by overlap

---

## Step 8: Complete Workflow Example

Here's a typical workflow from raw images to final analysis:

### Step-by-Step Example

**Starting with**: Multi-channel Z-stack images with grey (myotube) and blue (nuclei) channels

1. **Tab 1: Split Channels**
   - Input: `C:\MyImages\raw\` (multi-channel Z-stacks)
   - Output: `C:\MyImages\1_channels\`
   - Result: Separate grey and blue TIFF files

2. **Tab 2: Segment Myotubes**
   - Input: `C:\MyImages\1_channels\` (grey channel files)
   - Output: `C:\MyImages\2_myotubes\`
   - Result: Myotube masks and overlays

3. **Tab 3: Segment Nuclei**
   - Input: `C:\MyImages\1_channels\` (blue channel files)
   - Output: `C:\MyImages\3_nuclei\`
   - Result: Nuclei segmentation NPY files

4. **Tab 4: Analyze**
   - Myotube Folder: `C:\MyImages\2_myotubes\`
   - Nuclei Folder: `C:\MyImages\3_nuclei\`
   - Output: `C:\MyImages\4_analysis\`
   - Result: CSV files and overlays showing nuclei-myotube relationships

**Final Output**: CSV files showing number of nuclei per myotube, ready for statistical analysis!

---

## Step 9: Understanding the Results

### Myotube Segmentation Output

```
OutputFolder/
├── ImageName_masks/                    ← Individual myotube masks
│   ├── Myotube_1_mask.png
│   ├── Myotube_2_mask.png
│   └── ...
├── ImageName_processed_overlay.tif     ← Final segmentation visualization
├── ImageName_raw_overlay.tif           ← Raw model output
└── ImageName_info.json                 ← Processing metadata
```

### Nuclei Segmentation Output

```
OutputFolder/
└── ImageName/
    ├── ImageName_seg.npy               ← Nuclei masks (for analysis)
    ├── ImageName_RoiSet.zip            ← Fiji ROIs (optional)
    └── ImageName_overlay.png           ← Visualization
```

### Analysis Output

```
OutputFolder/
└── SampleName/
    ├── SampleName_myotube_nuclei_counts.csv        ← Counts per myotube
    ├── SampleName_nuclei_myotube_assignments.csv   ← Detailed assignments
    ├── SampleName_analysis_summary.txt             ← Statistics
    └── SampleName_nuclei_overlay.tif               ← Color-coded overlay
```

**CSV Files Explained**:

1. **myotube_nuclei_counts.csv**: Summary table
   - Myotube ID
   - Number of assigned nuclei (passed all filters)
   - Total detected nuclei overlapping
   - Filter statistics

2. **nuclei_myotube_assignments.csv**: Detailed table
   - Nucleus ID
   - Assigned myotube ID
   - Overlap percentage
   - Filter status (passed/filtered_size/filtered_eccentricity/filtered_overlap)
   - Morphological measurements (area, circularity, eccentricity)

---

## Updating the Tool

### When to Update

You need to update when:
- You receive a new version of the project
- You pull updates from Git
- Files in `fiji_integration/` folder are modified

### How to Update

**Every time you update the project, repeat Step 5:**

1. Navigate to the project's `fiji_integration` folder
2. Copy all files and folders:
   - Both `.ijm` files
   - `myotube_segmentation.py`
   - `requirements.txt`
   - `gui/` folder
   - `core/` folder
   - `utils/` folder
3. Paste them into Fiji's `macros` folder
4. **Replace/overwrite** all existing files when prompted
5. **Restart Fiji**

**Important**: Always restart Fiji after updating files!

---

## Troubleshooting

### Problem: "conda is not recognized as an internal or external command"

**Cause**: Miniconda wasn't added to PATH during installation

**Solution**:
1. Uninstall Miniconda (Control Panel → Uninstall a program)
2. Reinstall Miniconda
3. **Make sure to check "Add Miniconda3 to my PATH environment variable"**

### Problem: "Could not find Mask2Former directory"

**Cause**: Incorrect path configured

**Solution**:
1. In the Myotube Segmentation tab, click "Browse..." for Mask2Former Path
2. Navigate to the correct location where you extracted the project
3. Make sure the folder contains `mask2former/` subdirectory

### Problem: Macro doesn't appear in Fiji

**Cause**: Files not copied to correct location

**Solution**:
1. Verify files are in Fiji's `macros` folder (not `plugins`)
2. Make sure you copied the `.ijm` files directly to `macros/`, not in a subfolder
3. Restart Fiji completely
4. Press 'M' key to see if macro appears

### Problem: ModuleNotFoundError when running GUI

**Cause**: Folder structure not copied correctly

**Solution**:
1. Make sure you copied the entire `gui/`, `core/`, and `utils/` folders to Fiji's `macros` folder
2. The folder structure should match exactly as shown in Step 5.3
3. Restart Fiji after copying

### Problem: Analysis shows nuclei only in corner of overlay

**Cause**: This was a bug in earlier versions (now fixed)

**Solution**:
1. Update to the latest version (follow "Updating the Tool" section)
2. The overlay will automatically resize nuclei to match myotube overlay dimensions

### Problem: Segmentation produces no results

**Possible causes and solutions**:

1. **Images are not suitable**:
   - Make sure images show myotubes/nuclei clearly
   - Images should have good contrast

2. **Confidence threshold too high**:
   - Try lowering the confidence threshold to 0.3 or 0.2
   - Re-run the segmentation

3. **Wrong channel**:
   - Use grey channel for myotubes
   - Use blue channel for nuclei

---

## Tips for Best Results

1. **Image Quality**:
   - Ensure good contrast between structures and background
   - Avoid overexposed or underexposed images

2. **Workflow Organization**:
   - Create a consistent folder structure for each experiment
   - Use descriptive folder names
   - Example: `Experiment1/1_channels/`, `Experiment1/2_myotubes/`, etc.

3. **Parameter Tuning**:
   - Start with default parameters
   - For myotubes: If too many false positives, increase confidence threshold
   - For nuclei: Adjust diameter if auto-detection doesn't work well
   - For analysis: Adjust overlap threshold based on your biology

4. **Validation**:
   - Always visually inspect the overlay images
   - Check a few samples manually to validate automated counts
   - Use the color-coded analysis overlay to understand filtering

5. **Batch Processing**:
   - Process 10-20 images at a time
   - For large datasets, split into multiple folders
   - The GUI stays open for multiple runs - no need to restart!

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the project documentation**: Look for README files in the project folder

2. **Verify installation**: Make sure all software (Fiji, Miniconda) is properly installed

3. **Check file paths**: Most issues are caused by incorrect file paths

4. **Contact the developer**: Provide details about:
   - Error messages (copy the exact text)
   - What step/tab you were on
   - Your Windows version
   - Screenshots if possible

---

## Quick Reference Card

**To run the tools**:
1. Open Fiji
2. Press **'M'** key
3. Select **"Myotube_Segmentation_Windows.ijm"**
4. Use the appropriate tab for your task
5. Configure settings and click Run

**Complete workflow**:
1. **Tab 1**: Split channels → grey + blue TIFFs
2. **Tab 2**: Segment myotubes → masks + overlays
3. **Tab 3**: Segment nuclei → NPY files
4. **Tab 4**: Analyze → CSV files + overlays

**After updates**:
1. Copy entire `fiji_integration/` contents
2. Paste to Fiji's `macros/` folder
3. Replace all existing files
4. Restart Fiji

---

## Version Information

- **Guide Version**: 2.0
- **Last Updated**: 2025
- **Compatible with**: Windows 10/11, Fiji/ImageJ
- **Features**: Multi-tab interface with max projection, myotube segmentation, nuclei segmentation, and relationship analysis

---

**You're now ready to use the complete automated analysis pipeline! Happy analyzing!**
