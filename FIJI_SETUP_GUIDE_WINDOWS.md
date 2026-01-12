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
  - Myotube/cytoplasm images (in separate folder)
  - Nucleus images (in separate folder)
  - Can be Z-stacks or 2D images

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

### Tab 1: Max Projection

**Purpose**: Apply max intensity projection to Z-stack images in separate myotube and nucleus folders.

**When to use**: If you have Z-stack images (multi-slice 3D images) that need to be converted to 2D via max projection.

**Steps**:
1. Click the **"Max Projection"** tab
2. **Myotube Input**: Browse to your folder containing myotube/cytoplasm images
3. **Nucleus Input**: Browse to your folder containing nucleus images
4. **Output Directory**: Choose where to save the max projected images
5. Click **"Run Max Projection"**

**Note**: You can specify just one folder if you only need to process one channel. Leave the other field empty.

**Output**:
- `myotube_max_projection/MAX_*.tif` - Max projected myotube images
- `nucleus_max_projection/MAX_*.tif` - Max projected nucleus images

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

5. **Detection Parameters**:
   - **Confidence Threshold** (0-1): Default 0.25
     - Higher = fewer detections (stricter)
     - Lower = more detections (may include false positives)
     - *When to adjust*: If missing obvious myotubes, lower to 0.15-0.20

   - **Minimum Area** (pixels): Default 100
     - Filters out tiny detections
     - *When to adjust*: If seeing small artifacts, increase to 200-500

   - **Maximum Area** (pixels): Default 50000
     - Filters out unrealistically large detections
     - *When to adjust*: Rarely needed unless you have very large myotubes

   - **Final Min Area** (pixels): Default 1000
     - Second-stage filter after post-processing
     - Removes small fragments after merging
     - *When to adjust*: If final results still have small fragments, increase

6. **Tiling Options** (for large images or dense myotubes):
   - **Use tiled inference**: Check for images with many myotubes
   - **Grid Size**: Default 2 (creates 2×2=4 tiles)
     - 1 = no tiling
     - 2 = 2×2 grid (4 tiles)
     - 3 = 3×3 grid (9 tiles)
     - *When to adjust*: If many myotubes, use 2-3 for better detection

   - **Tile Overlap** (%): Default 20%
     - Overlap between adjacent tiles
     - Helps detect myotubes at tile boundaries
     - *When to adjust*: Rarely needed

7. **Output Options**:
   - **Skip merged masks**: Default checked
     - Skips generating imaginary boundary files (faster)
     - *Uncheck if*: You need the merged visualization for presentations

   - **Save measurements CSV**: Default unchecked
     - Saves detailed measurements (area, length, width) for each myotube
     - *Check if*: You need quantitative morphology data

8. Click **"Run Segmentation"**

**Output Files** (for each image):
- `[ImageName]_masks/` - Individual myotube mask PNG files (one per myotube)
- `[ImageName]_processed_overlay.tif` - **Main result**: Color-coded overlay on original image
- `[ImageName]_raw_overlay.tif` - Raw model predictions before post-processing
- `[ImageName]_info.json` - Processing metadata (parameters used, image dimensions, etc.)
- `[ImageName]_measurements.csv` - Myotube measurements (if "Save measurements" checked)

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

2. **Input/Output Folders**:
   - **Myotube Folder**: Browse to myotube segmentation results (from Tab 2)
   - **Nuclei Folder**: Browse to nuclei segmentation results (from Tab 3)
   - **Output Folder**: Choose where to save analysis results

3. **Filter Parameters**:
   - **Nucleus Size Range** (pixels): Default 400-6000
     - **Min Area**: Removes small debris/artifacts
     - **Max Area**: Removes large clumps (likely multiple nuclei)
     - *What it does*: Nuclei outside this range are marked as "filtered_size" (shown in RED on overlay)
     - *When to adjust*: Measure some nuclei in Fiji to determine appropriate size range for your images

   - **Max Eccentricity** (0-1): Default 0.95
     - **0** = perfect circle
     - **1** = elongated line
     - Values close to 1 indicate highly elongated shapes (likely artifacts, not nuclei)
     - *What it does*: Nuclei with eccentricity > 0.95 are marked as "filtered_eccentricity" (shown in YELLOW on overlay)
     - *When to adjust*: If losing real nuclei that are slightly elongated, increase to 0.98

   - **Overlap Threshold** (0-1): Default 0.10 (10%)
     - Minimum percentage of nucleus that must overlap with a myotube to be assigned to it
     - Example: 0.10 means ≥10% of the nucleus area must be inside a myotube
     - *What it does*: Nuclei with <10% overlap are marked as "filtered_overlap" (shown in BLUE on overlay)
     - *When to adjust*:
       - Lower (0.05) = include nuclei barely touching myotubes
       - Higher (0.30-0.50) = only include nuclei with substantial overlap

   - **Periphery Overlap Threshold** (0-1): Default 0.95 (95%)
     - Used to distinguish "central" vs "peripheral" nuclei in the periphery overlay
     - Only affects visualization in `*_periphery_overlay.tif`, not the counts
     - *What it does*:
       - Nuclei with overlap ≥ periphery threshold → GREEN (central)
       - Nuclei with overlap < periphery threshold but ≥ regular threshold → YELLOW (peripheral)
     - *When to adjust*:
       - Set to 1.0 if you want only perfectly centered nuclei as green
       - Lower (0.70-0.80) if you want more nuclei classified as central

4. **Processing Options**:
   - **Full Image Mode**: Check if processing complete images (not cropped quadrants)
     - *When to check*: Always check this unless you manually cropped images into quadrants

5. Click **"Run Analysis"**

**Output Files** (for each sample):
- `[Sample]_myotube_nuclei_counts.csv` - **Main result**: Nuclei count per myotube
- `[Sample]_nuclei_myotube_assignments.csv` - Detailed nucleus-by-nucleus data with grid coordinates
- `[Sample]_analysis_summary.txt` - Statistics summary
- `[Sample]_nuclei_overlay.tif` - **Enhanced color-coded visualization** showing all nuclei:
  - **Semi-transparent filled nuclei** (35% opacity) showing myotube structure beneath
  - **Nucleus IDs displayed offset** to the right on dark backgrounds (not covering nuclei)
  - **Grid reference system**: 15×15 grid with column letters (A, B, C...) and row numbers (1-15)
  - Color coding:
    - **GREEN**: Assigned to myotubes (passed all filters)
    - **RED**: Filtered out by size
    - **YELLOW**: Filtered out by eccentricity
    - **BLUE**: Filtered out by overlap
- `[Sample]_periphery_overlay.tif` - Shows only assigned nuclei:
  - **GREEN**: Central nuclei (overlap ≥ periphery threshold)
  - **YELLOW**: Peripheral nuclei (overlap between regular and periphery threshold)

---

## Step 8: Complete Workflow Example

Here's a typical workflow from raw images to final analysis:

### Step-by-Step Example

**Starting with**: Separate folders for myotube and nucleus Z-stack images

1. **Tab 1: Max Projection**
   - Myotube Input: `C:\MyImages\raw_myotube\` (myotube Z-stacks)
   - Nucleus Input: `C:\MyImages\raw_nucleus\` (nucleus Z-stacks)
   - Output: `C:\MyImages\1_max_projection\`
   - Result: Max projected 2D images in separate folders

2. **Tab 2: Segment Myotubes**
   - Input: `C:\MyImages\1_max_projection\myotube_max_projection\`
   - Output: `C:\MyImages\2_myotubes\`
   - Result: Myotube masks and overlays

3. **Tab 3: Segment Nuclei**
   - Input: `C:\MyImages\1_max_projection\nucleus_max_projection\`
   - Output: `C:\MyImages\3_nuclei\`
   - Result: Nuclei segmentation NPY files

4. **Tab 4: Analyze**
   - Myotube Folder: `C:\MyImages\2_myotubes\`
   - Nuclei Folder: `C:\MyImages\3_nuclei\`
   - Output: `C:\MyImages\4_analysis\`
   - Filter Settings: Use defaults (400-6000 pixels, 0.95 eccentricity, 0.10 overlap, 0.95 periphery)
   - Result: CSV files and overlays showing nuclei-myotube relationships

**Final Output**:
- `myotube_nuclei_counts.csv` - Ready for statistical analysis!
- `nuclei_overlay.tif` - Verify filtering worked correctly
- `periphery_overlay.tif` - See central vs peripheral nuclei distribution

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
    ├── SampleName_myotube_nuclei_counts.csv        ← Main result: Counts per myotube
    ├── SampleName_nuclei_myotube_assignments.csv   ← Detailed nucleus data
    ├── SampleName_analysis_summary.txt             ← Statistics summary
    ├── SampleName_nuclei_overlay.tif               ← All nuclei color-coded
    └── SampleName_periphery_overlay.tif            ← Only assigned nuclei (central vs peripheral)
```

**Understanding the CSV Files**:

#### 1. `myotube_nuclei_counts.csv` - **YOUR MAIN RESULTS**

This is the file you'll use for statistical analysis. One row per myotube.

**Columns**:
- `myotube_id`: Unique ID for each myotube (1, 2, 3, ...)
- `nucleus_count`: **Number of nuclei assigned to this myotube** (passed all filters)
- `total_overlapping`: Total nuclei detected overlapping with this myotube (before filtering)
- `filtered_size`: Number filtered out by size criteria
- `filtered_eccentricity`: Number filtered out by eccentricity (too elongated)
- `filtered_overlap`: Number filtered out by overlap threshold (not enough overlap)

**Example row**:
```
myotube_id,nucleus_count,total_overlapping,filtered_size,filtered_eccentricity,filtered_overlap
1,12,15,1,1,1
```
This means: Myotube #1 has **12 assigned nuclei**. It had 15 total overlapping objects, but 1 was too small, 1 was too elongated, and 1 didn't have enough overlap.

#### 2. `nuclei_myotube_assignments.csv` - **DETAILED NUCLEUS DATA**

One row per detected nucleus. Use this to understand why specific nuclei were filtered.

**Columns**:
- `nucleus_id`: Unique ID for each nucleus (1, 2, 3, ...)
- **`grid_ref`**: Grid reference like "C5", "H12" for easy location on overlay (NEW!)
- **`grid_col`**, **`grid_row`**: Numeric grid coordinates (0-14 for 15×15 grid) (NEW!)
- **`centroid_x`**, **`centroid_y`**: Exact pixel coordinates of nucleus center (NEW!)
- `assigned_myotube_id`: Which myotube this nucleus is assigned to (None if filtered out)
- `filter_status`: One of:
  - `passed` - Nucleus assigned to a myotube (counted)
  - `filtered_size` - Outside size range
  - `filtered_eccentricity` - Too elongated
  - `filtered_overlap` - Not enough overlap with any myotube
- `overlap_percentage`: Percentage of nucleus overlapping with its assigned myotube (0-100)
- `nucleus_area`: Nucleus area in pixels
- `eccentricity`: Shape measure (0=circle, 1=line)
- `circularity`: Alternative shape measure (1=perfect circle, lower=irregular)
- `solidity`: Convexity measure (1=convex, <1=concave/irregular)

**Example rows**:
```
nucleus_id,grid_ref,grid_col,grid_row,centroid_x,centroid_y,assigned_myotube_id,filter_status,overlap_percentage,nucleus_area,eccentricity
408,C5,2,4,1234,2156,3,passed,82.5,756,0.65
431,C5,2,4,1289,2198,3,passed,79.3,812,0.58
156,A1,0,0,234,178,None,filtered_size,0,189,0.72
```
- Nucleus 408: In grid cell **C5**, assigned to myotube 3, 82.5% overlap, 756 pixels
- Nucleus 431: In grid cell **C5**, assigned to myotube 3, 79.3% overlap, 812 pixels
- Nucleus 156: In grid cell **A1**, filtered out (too small - only 189 pixels)

**How to use grid coordinates**:
1. Open the CSV in Excel/software and find a nucleus of interest (e.g., nucleus 408)
2. Note its `grid_ref` value (e.g., "C5")
3. Open the `nuclei_overlay.tif` in Fiji/ImageJ
4. Look for column **C** and row **5** in the grid overlay
5. Find nucleus #408 in that grid cell
6. Or use the `centroid_x` and `centroid_y` values to jump directly to the exact pixel location

#### 3. `analysis_summary.txt` - **STATISTICS OVERVIEW**

Text file with summary statistics:
- Total myotubes analyzed
- Total nuclei detected
- Total nuclei assigned (passed filters)
- Filtering breakdown (how many filtered by each criterion)
- Average nuclei per myotube
- Min/max nuclei per myotube

---

**Understanding the Overlay Visualizations**:

#### `nuclei_overlay.tif` - Shows ALL detected nuclei with enhanced visualization

**NEW FEATURES** in this overlay:
- **Semi-transparent filled nuclei** (35% opacity) - You can see both the nucleus color AND the myotube structure beneath
- **Offset nucleus ID labels** - IDs are positioned to the RIGHT of each nucleus (not covering it) with dark backgrounds for readability
- **Grid reference system** - 15×15 grid with labeled axes:
  - Column letters (A, B, C, ..., O) displayed at the top
  - Row numbers (1-15) displayed on the left
  - Use this to cross-reference with the CSV's `grid_ref` column

This overlay helps you understand your filtering:

- **GREEN nuclei**: Passed all filters and assigned to myotubes ✓
  - These are counted in your results
  - Size is within range (400-6000 pixels)
  - Eccentricity < 0.95 (round enough)
  - Overlap ≥ threshold (≥10% overlap with myotube)

- **RED nuclei**: Filtered by size ✗
  - Too small (< min area) or too large (> max area)
  - Likely debris (small) or clumps (large)

- **YELLOW nuclei**: Filtered by eccentricity ✗
  - Too elongated (eccentricity > 0.95)
  - Likely artifacts, not real nuclei

- **BLUE nuclei**: Filtered by overlap ✗
  - Not enough overlap with any myotube (< overlap threshold)
  - May be background nuclei not associated with myotubes

**How to use this**:
1. Visually inspect to verify filtering is working correctly
2. If many real nuclei are being filtered, adjust the parameters
3. To locate specific nuclei from CSV: Look up the `grid_ref` (e.g., "C5"), find that grid cell on the overlay, then look for the nucleus ID number

---

#### `periphery_overlay.tif` - Shows ONLY assigned nuclei (central vs peripheral)

This overlay shows spatial distribution of nuclei within myotubes:

- **GREEN nuclei**: Central nuclei
  - Overlap ≥ periphery overlap threshold (default 95%)
  - Deeply embedded inside the myotube

- **YELLOW nuclei**: Peripheral nuclei
  - Overlap between regular threshold (10%) and periphery threshold (95%)
  - Closer to myotube edges or partially overlapping

**How to use this**:
- If you want to analyze central vs peripheral nuclei separately, you can use the `overlap_percentage` column in the CSV along with your threshold values
- Example: Count nuclei with overlap_percentage ≥ 95 (central) vs 10-95 (peripheral)

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

## Common Questions

### Q: How do I know if my filter settings are correct?

**A**: Check the `nuclei_overlay.tif`:
- If you see many **RED** nuclei that look like real nuclei → Your size range is too restrictive. Increase max area (e.g., from 6000 to 8000) or decrease min area (e.g., from 400 to 300).
- If you see many **YELLOW** nuclei that look round → Your eccentricity threshold is too strict. Increase from 0.95 to 0.98.
- If you see many **BLUE** nuclei clearly inside myotubes → Your overlap threshold is too high. It's already very low at 0.10 (10%), so this is unlikely. If it happens, decrease to 0.05.
- If you see many small **RED** dots everywhere → That's debris being correctly filtered out. No action needed.

### Q: What's the difference between overlap threshold and periphery overlap threshold?

**A**:
- **Overlap threshold** (default 0.10): Determines which nuclei are **counted**. Nuclei with <10% overlap are excluded from analysis.
- **Periphery overlap threshold** (default 0.95): Only affects the **periphery overlay visualization**. Distinguishes central (≥95%) from peripheral (10-95%) nuclei. Both are still counted in your results.

### Q: Should I use tiled inference?

**A**:
- **Use tiling** (Grid Size 2-3) if:
  - Your images have many myotubes (>20)
  - Myotubes are densely packed
  - You're getting incomplete segmentations
- **Don't use tiling** (Grid Size 1) if:
  - Your images have few myotubes (<10)
  - Myotubes are well-separated
  - Processing time is acceptable

### Q: Why are nuclei counts different between `total_overlapping` and `nucleus_count`?

**A**: `total_overlapping` includes ALL nuclei that touch the myotube, even those that failed filters (too small, too elongated, or insufficient overlap). `nucleus_count` only includes nuclei that **passed all filters** and are assigned to the myotube.

### Q: Can I process images at different resolutions together?

**A**: Yes, the tool automatically handles different image sizes. However, for **analysis** (Tab 4), make sure your filter parameters (especially size range) are appropriate for your image resolution. If you have mixed resolutions, you may need to run analysis separately for each resolution.

### Q: What if myotube and nuclei images don't align perfectly?

**A**: The analysis automatically resizes nuclei masks to match myotube overlay dimensions if needed. However, the images should be from the **same field of view**. If they're from different regions or different samples, the analysis won't be meaningful.

### Q: How do I analyze just central nuclei?

**A**:
1. Set periphery overlap threshold to your desired cutoff (default is 0.95 for ≥95% overlap)
2. After analysis, use the `nuclei_myotube_assignments.csv` file
3. Filter for rows where `overlap_percentage ≥ 95`
4. Count these nuclei per myotube in Excel/Python/R

**Example**: If you want nuclei that are mostly inside (≥80% overlap), change periphery overlap threshold to 0.80, then filter the CSV for `overlap_percentage ≥ 80`.

### Q: How do I use the grid reference system to locate specific nuclei?

**A**: The grid system makes it easy to find specific nuclei when you have 1000+ in an image:
1. **Open the CSV** (`nuclei_myotube_assignments.csv`) in Excel or similar
2. **Find the nucleus** you want to examine (e.g., nucleus #347 that was filtered)
3. **Note the `grid_ref` value** (e.g., "E7")
4. **Open the overlay** (`nuclei_overlay.tif`) in Fiji/ImageJ
5. **Locate grid cell E7**: Find column **E** (5th column) and row **7** on the labeled grid
6. **Look for nucleus #347** in that grid cell
7. **Alternative**: Use the `centroid_x` and `centroid_y` values to jump directly to the exact pixel location in Fiji

**Example workflow**: "Why was nucleus #156 filtered? Let me check."
- Look in CSV → nucleus_id=156, grid_ref="A1", filter_status="filtered_size", nucleus_area=189
- Open overlay → Navigate to grid cell A1 → See RED nucleus #156
- Conclusion: It was correctly filtered (too small at 189 pixels vs minimum 400)

### Q: What if some nuclei have the same color as myotubes in the overlay?

**A**: This can happen because myotube colors are randomly generated by the segmentation algorithm. Since there are hundreds of possible colors, occasionally a myotube might be colored green, red, yellow, or blue - the same colors used for nucleus status. This is a visual issue only and doesn't affect the analysis results. When this happens:
- The **nucleus color/fill** still correctly indicates its filter status
- Look at the **nucleus ID number** (offset to the right) to identify specific nuclei
- Use the **grid reference system** and CSV to verify nucleus assignments
- The semi-transparent fill (35% opacity) helps distinguish nuclei from the underlying myotube colors

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
1. **Tab 1**: Max projection → myotube_max_projection/ + nucleus_max_projection/
   - Input: Two separate folders (myotube + nucleus)
2. **Tab 2**: Segment myotubes → masks + overlays
   - Key params: Confidence 0.25, Min area 100, Final min area 1000
3. **Tab 3**: Segment nuclei → NPY files
   - Key params: Model cyto3, Diameter 30, Target res 3000
4. **Tab 4**: Analyze → CSV files + overlays
   - Key params: Size 400-6000, Eccentricity 0.95, Overlap 0.10, Periphery 0.95

**Main result files**:
- **Myotube segmentation**: `*_processed_overlay.tif` (visualization), `*_masks/` (individual masks)
- **Nuclei segmentation**: `*_seg.npy` (required for analysis), `*_overlay.png` (visualization)
- **Analysis**: `*_myotube_nuclei_counts.csv` (main results), `*_nuclei_overlay.tif` (all nuclei), `*_periphery_overlay.tif` (central vs peripheral)

**Overlay color codes**:
- **nuclei_overlay.tif**: GREEN=assigned, RED=size filter, YELLOW=eccentricity filter, BLUE=overlap filter
- **periphery_overlay.tif**: GREEN=central nuclei (≥95% overlap), YELLOW=peripheral nuclei (10-95% overlap)

**After updates**:
1. Copy entire `fiji_integration/` contents (both .ijm files + gui/ + core/ + utils/ folders)
2. Paste to Fiji's `macros/` folder
3. Replace all existing files
4. Restart Fiji

---

## Version Information

- **Guide Version**: 2.3
- **Last Updated**: January 2025
- **Compatible with**: Windows 10/11, Fiji/ImageJ
- **Features**:
  - Multi-tab interface with 4 processing steps
  - Max projection for separate myotube and nucleus folders
  - Myotube segmentation with Mask2Former
  - Nuclei segmentation with CellPose
  - Comprehensive nuclei-myotube relationship analysis
  - Central vs peripheral nuclei classification
  - **Enhanced nuclei overlay visualization with grid reference system**
  - Detailed parameter explanations and CSV result documentation

**Changes in v2.3** (January 2025):
  - **Enhanced nuclei overlay visualization**:
    - Semi-transparent filled nuclei (35% opacity) showing myotube structure beneath
    - Nucleus ID labels offset to the right with dark backgrounds (not obscuring nuclei)
    - Grid reference system (15×15) with column letters (A-O) and row numbers (1-15)
    - Larger, more readable grid labels
  - **Improved CSV output**:
    - Added `grid_ref` column for easy cross-referencing with overlay (e.g., "C5", "H12")
    - Added `grid_col` and `grid_row` numeric coordinates
    - Added `centroid_x` and `centroid_y` exact pixel coordinates
  - **Better cross-referencing workflow**: Scientists can now easily locate specific nuclei from CSV data using grid coordinates

**Changes in v2.2** (December 2024):
  - Simplified Max Projection tab: Now takes two separate input folders (myotube + nucleus) instead of multi-channel images
  - Removed automatic channel detection - users provide pre-separated images
  - Improved conda detection on Windows - now finds conda via PATH automatically

---

**You're now ready to use the complete automated analysis pipeline! Happy analyzing!**
