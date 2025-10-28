# Fiji Integration Setup Guide for Windows

**Complete beginner's guide to setting up and using automated myotube segmentation in Fiji**

This guide will help you install and run the myotube segmentation tool through Fiji (ImageJ) on Windows. No programming experience required!

---

## Table of Contents
1. [What You'll Need](#what-youll-need)
2. [Step 1: Install Fiji](#step-1-install-fiji)
3. [Step 2: Install Miniconda](#step-2-install-miniconda)
4. [Step 3: Download This Project](#step-3-download-this-project)
5. [Step 4: Download Trained Model](#step-4-download-trained-model)
6. [Step 5: Copy Files to Fiji](#step-5-copy-files-to-fiji)
7. [Step 6: Prepare Your Images](#step-6-prepare-your-images)
8. [Step 7: First-Time Setup & Run Segmentation](#step-7-first-time-setup--run-segmentation)
9. [Step 8: Understanding the Results](#step-8-understanding-the-results)
10. [Updating the Tool](#updating-the-tool)
11. [Troubleshooting](#troubleshooting)

---

## What You'll Need

- **Windows 10 or 11** (64-bit)
- **At least 8GB of RAM** (16GB recommended)
- **10GB of free disk space**
- **Internet connection** for downloading software
- **Your microscopy images** (TIFF format recommended)

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

Miniconda is a Python distribution manager needed to run the segmentation algorithm.

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
   - You can rename it to just `Mask2Former` if you prefer, or keep it as is
   - Example paths: `C:\Users\YourUsername\Mask2Former` or `C:\Users\YourUsername\Mask2Former-main`
   - **Remember this path** - you'll need it in Step 7 when configuring the macro

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
4. Navigate into the project:
   ```
   cd Mask2Former
   ```

---

## Step 4: Download Trained Model

The trained model file is required for segmentation. You need to download it from Google Drive.

### 4.1 Download the Model

1. **Go to the Google Drive link**:
   - Open your web browser
   - Go to: **https://drive.google.com/file/d/1O0fEGpIZrA2I8SbsuO2cPDSGpRWQK38r/view?usp=sharing**

2. **Download the .pth file**:
   - Click the **"Download"** button (usually in the top right)
   - If prompted, click **"Download anyway"** (the file is safe)
   - The file will be named something like `model_final.pth` or similar
   - Download location: Usually goes to your **Downloads** folder

3. **Move the model file** (optional but recommended):
   - Create a folder for models, for example: `C:\Users\YourUsername\Mask2Former\models`
   - Move the downloaded `.pth` file to this folder
   - **Remember this location** - you'll need it in Step 7 when running segmentation

**Note**: The model file is large (several hundred MB), so the download may take a few minutes depending on your internet speed.

---

## Step 5: Copy Files to Fiji

This is a **critical step** - you must copy 4 files from the project to Fiji's macro folder.

### 5.1 Locate the Source Files

1. Open **File Explorer**
2. Navigate to your Mask2Former project folder
3. Open the **`fiji_integration`** folder
4. You should see these 4 files:
   - `Myotube_Segmentation_Windows.ijm`
   - `Myotube_Segmentation.ijm` (Linux/Mac version - copy anyway)
   - `myotube_segmentation.py`
   - `requirements.txt`

### 5.2 Locate Fiji's Macros Folder

1. Open another File Explorer window
2. Navigate to where you installed Fiji (e.g., `C:\Program Files\Fiji`)
3. Open the **`macros`** folder
   - Full path example: `C:\Program Files\Fiji\macros`

### 5.3 Copy the Files

1. **Select all 4 files** in the `fiji_integration` folder
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
└── (other existing .ijm files)
```

---

## Step 6: Prepare Your Images

Before running the segmentation for the first time, prepare your images so you can run segmentation immediately after installation completes.

1. **Create a working folder** for your images, for example:
   ```
   C:\Users\YourUsername\Documents\MyExperiment
   ```

2. **Place your images** in this folder:
   - Supported formats: TIFF, TIF, PNG, JPG
   - Images should show myotubes (muscle fibers)
   - **Important**: Images must be single-channel (gray or green). Multi-channel images are not supported

---

## Step 7: First-Time Setup & Run Segmentation

**The first time you run the macro, it will automatically install Python dependencies, then proceed directly to segmentation.**

### 7.1 Launch the Macro

1. **Open Fiji** (double-click `ImageJ-win64.exe`)

2. **Run the macro**:
   - Press the **'M'** key (or go to Plugins → Macros → Run...)
   - A list of macros will appear
   - Select **"Myotube_Segmentation_Windows.ijm"**
   - Click **"Open"**

### 7.2 Automatic Installation (First Time Only)

3. **Wait for automatic installation**:
   - The macro will automatically detect if Python dependencies are missing
   - It will create the conda environment and install all packages
   - **Wait for installation** (5-15 minutes first time)
   - You'll see progress messages in a console/terminal window
   - Once installation is complete, the Python GUI will appear automatically

### 7.3 Configure and Run Segmentation

4. **Select your image folder** (in Python GUI):
   - The Python GUI window will appear after installation
   - Click **"Browse..."** next to "Input Directory"
   - Navigate to the folder you prepared in Step 6 (e.g., `C:\Users\YourUsername\Documents\MyExperiment`)
   - Click **"Select Folder"**

5. **Select your output folder** (in Python GUI):
   - The "Output Directory" field shows the default location: `Desktop/myotube_results`
   - You can keep this default or click **"Browse..."** to choose a different location
   - Example locations:
     - Default: `C:\Users\YourUsername\Desktop\myotube_results`
     - Custom: `C:\Users\YourUsername\Documents\MyExperiment_Results`
   - The output path will be saved for future runs

6. **Configure Mask2Former path** (first time only):
   - You'll be asked for the Mask2Former directory
   - Browse to where you extracted the project in Step 3
   - Examples: `C:\Users\John\Mask2Former` or `C:\Users\John\Mask2Former-main`
   - The path will be saved for future use

7. **Select the trained model file** (in Python GUI):
   - Click **"Browse..."** next to "Model Weights" or "Model Path"
   - Navigate to where you saved the `.pth` file in Step 4
   - Select the downloaded model file (e.g., `model_final.pth`)
   - Click **"Open"**
   - The model path will be saved for future runs

8. **Configure parameters** (optional):
   - The Python GUI will show settings with these options:

   | Parameter | Default | Description |
   |-----------|---------|-------------|
   | **Confidence Threshold** | 0.5 | Only keep predictions above this confidence (0.0-1.0) |
   | **Min Area (pixels)** | 1000 | Remove myotubes smaller than this |
   | **Max Area (pixels)** | 1000000 | Remove myotubes larger than this |
   | **Overlap Threshold** | 0.5 | For merging overlapping predictions (0.0-1.0) |
   | **Target Resolution** | 9000 | Resize images to this width (pixels) |

   - For first-time users, **use the defaults**
   - Click **"Run"** to start processing

9. **Wait for processing**:
   - A progress bar will show processing status
   - Processing time depends on:
     - Number of images
     - Image size
     - Computer speed
   - Typical: 1-3 minutes per image

10. **Processing complete**:
   - You'll see a message: "Processing complete!"
   - Results will automatically open in your file explorer

---

## Step 8: Understanding the Results

The results will be saved in the output directory you selected in the Python GUI. For each input image, the tool creates:

```
OutputFolder/
├── ImageName_masks/                    ← Folder containing individual myotube masks
├── ImageName_processed_overlay.tif     ← Visualization with colored outlines
├── ImageName_raw_overlay.tif           ← Alternative visualization
├── ImageName_info.json                 ← Processing metadata
└── ImageName_measurements.csv          ← Measurements (if enabled in settings)
```

**Example output structure:**
```
C:\Users\YourUsername\Documents\Results\
├── MyImage_masks\
│   ├── mask_001.png
│   ├── mask_002.png
│   └── ... (one mask per detected myotube)
├── MyImage_processed_overlay.tif
├── MyImage_raw_overlay.tif
├── MyImage_info.json
└── MyImage_measurements.csv (optional)
```

**Files explained**:

- **[ImageName]_masks/**: Folder containing individual mask files
  - One PNG file per detected myotube
  - Named sequentially: `mask_001.png`, `mask_002.png`, etc.
  - Use these for quantification or further processing

- **[ImageName]_processed_overlay.tif**: Final segmentation results after post-processing
  - Shows myotubes after filtering (confidence threshold, size filters, overlap removal)
  - Each myotube gets a different colored outline
  - **Use this file** for presentations and validation

- **[ImageName]_raw_overlay.tif**: Raw model output before post-processing
  - Shows all predictions directly from the neural network
  - Useful for troubleshooting or adjusting parameters

- **[ImageName]_info.json**: Processing metadata and parameters
  - Contains settings used for this segmentation
  - Useful for reproducibility

- **[ImageName]_measurements.csv** (optional): Comprehensive measurements spreadsheet
  - Generated only if "Save measurements CSV" is enabled in the GUI
  - Contains detailed metrics for each myotube:
    - Area, Visible Length, Estimated Total Length, Width
    - Aspect Ratio, Connected Components, Perimeter
    - Bounding Box dimensions and Confidence scores
  - Open in Excel or any spreadsheet program
  - Note: Generation may be slow for images with many myotubes (disabled by default)

---

## Updating the Tool

### When to Update

You need to update the Fiji integration files when:
- You receive a new version of the project
- You pull updates from Git
- Files in `fiji_integration/` folder are modified

### How to Update

**Every time you update or pull the project, repeat Step 5:**

1. Navigate to the project's `fiji_integration` folder
2. Copy all 4 files:
   - `Myotube_Segmentation_Windows.ijm`
   - `Myotube_Segmentation.ijm`
   - `myotube_segmentation.py`
   - `requirements.txt`
3. Paste them into Fiji's `macros` folder
4. **Replace/overwrite** the existing files when prompted

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

**Cause**: Incorrect path configured in the Python GUI

**Solution**:
1. When you run the macro, the Python GUI will ask for the Mask2Former directory
2. Click **"Browse..."** and navigate to the correct location where you extracted the project
3. Examples of correct paths:
   - `C:\Users\YourUsername\Mask2Former`
   - `C:\Users\YourUsername\Mask2Former-main`
   - `D:\Projects\Mask2Former`
4. Make sure the folder contains the `mask2former/` subdirectory and other project files
5. The path will be saved for future runs

### Problem: Macro doesn't appear in Fiji

**Cause**: Files not copied to correct location, or using a different Fiji installation

**Solution**:
1. **Check if you're using the correct Fiji installation**:
   - You may have multiple Fiji installations on your computer
   - Make sure you're opening the Fiji where you copied the macro files
   - Check the window title or Help → About ImageJ to confirm the location
2. Verify files are in the correct Fiji's `macros` folder (not `plugins` or other folders)
3. Make sure you copied the files to the active Fiji installation's macros folder
4. Restart Fiji completely
5. Press 'M' key to see if macro appears

### Problem: Segmentation produces no results

**Possible causes and solutions**:

1. **Images are not suitable**:
   - Make sure images show myotubes clearly
   - Images should have good contrast

2. **Confidence threshold too high**:
   - Try lowering the confidence threshold to 0.3 or 0.2
   - Re-run the segmentation

3. **Model not downloaded**:
   - Make sure the project folder contains the trained model files
   - Check for a folder like `output_stage2_manual` or similar

### Problem: Segmentation is very slow

**Normal processing time**: 1-3 minutes per image

**If much slower**:
1. Check if your computer meets minimum requirements (8GB RAM)
2. Close other programs to free up memory
3. Process fewer images at once
4. Consider using a more powerful computer for large batches

---

## Tips for Best Results

1. **Image Quality**:
   - Ensure good contrast between myotubes and background
   - Avoid overexposed or underexposed images

2. **Batch Processing**:
   - Process 10-20 images at a time
   - For large datasets, split into multiple folders

3. **Parameter Tuning**:
   - Start with default parameters
   - If too many false positives: increase confidence threshold
   - If missing myotubes: decrease confidence threshold
   - If detecting noise: increase minimum area

4. **Validation**:
   - Always visually inspect the overlay images
   - Check both processed and raw overlay files to assess segmentation quality
   - Compare with manual annotations on a few test images

5. **Backing Up**:
   - Always keep a copy of your original images
   - The tool doesn't modify original images, but it's good practice

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the project documentation**: Look for README files or other markdown files in the project folder

2. **Check file paths**: Most issues are caused by incorrect file paths - double-check all paths use double backslashes `\\`

3. **Verify installation**: Make sure all software (Fiji, Miniconda) is properly installed

4. **Contact the developer**: Provide details about:
   - Error messages (copy the exact text)
   - What step you were on
   - Your Windows version
   - Screenshots if possible

---

## Quick Reference Card

**To run segmentation**:
1. Open Fiji
2. Press **'M'** key
3. Select **"Myotube_Segmentation_Windows.ijm"**
4. Choose your image folder
5. Adjust parameters (or use defaults)
6. Click OK and wait

**After updates**:
1. Copy 4 files from `fiji_integration/`
2. Paste to Fiji's `macros/` folder
3. Replace existing files
4. Restart Fiji

**Results location**:
- Look in the output folder you selected in the GUI
- For each image: `[ImageName]_masks/` folder, `_processed_overlay.tif`, `_raw_overlay.tif`, `_info.json`
- Optional: `_measurements.csv` (if enabled in GUI settings)
- Open overlay files to view segmentation results

---

## Version Information

- **Guide Version**: 1.0
- **Last Updated**: 2025
- **Compatible with**: Windows 10/11, Fiji/ImageJ

---

**You're now ready to use automated myotube segmentation! Happy analyzing!**
