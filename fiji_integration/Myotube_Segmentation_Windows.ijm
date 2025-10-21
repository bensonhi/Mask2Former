/*
 * Myotube Instance Segmentation for Fiji (Windows Version)
 *
 * This macro provides seamless integration of AI-powered myotube instance segmentation
 * into the Fiji/ImageJ workflow. Users simply click a button to segment myotubes
 * in the current image.
 *
 * Features:
 * - One-click myotube segmentation
 * - Individual mask image loading as overlays
 * - Colored overlay visualization
 * - Measurement export to CSV
 * - Progress feedback and error handling
 *
 * Setup Requirements:
 * 1. Python environment with required packages (see requirements.txt)
 * 2. Trained Mask2Former model for myotubes
 * 3. This macro file in Fiji plugins or macros folder
 *
 * Usage:
 * 1. Run "Segment Myotubes" macro (or press 'M' shortcut)
 * 2. Select folder containing images to process
 * 3. Wait for batch processing (progress shown in status bar)
 * 4. Browse results in automatically opened output directory
 */

// Configuration - Update these paths for your system
var CONDA_ENV = "m2f";  // Conda environment name
var PYTHON_COMMAND = "python";  // Python command within the conda environment
var MASK2FORMER_PATH = "C:\\Users\\YourUsername\\Mask2Former";  // Path to Mask2Former project (Windows style)
var SCRIPT_PATH = "";  // Auto-detected based on macro location
var CONFIG_FILE = "";  // Auto-detected
var MODEL_WEIGHTS = "";  // Auto-detected

// Processing parameters (can be adjusted by users)
var CONFIDENCE_THRESHOLD = 0.25;
var MIN_AREA = 100;
var MAX_AREA = 999999;  // Large number representing infinity
var FINAL_MIN_AREA = 1000;  // Final minimum area filter (applied after post-processing)
var USE_CPU = false;  // Set to true to force CPU inference (slower but less memory)
var FORCE_SMALL_INPUT = false;  // Set to true to force 1024px input (memory optimization, may reduce accuracy)

// Tiling parameters (for large images with too many myotubes)
var USE_TILING = true;  // Set to false to disable tiled inference and process entire image at once
var TILE_OVERLAP = 20;  // Overlap percentage between tiles (default: 20%)

// Merged mask generation parameter
var SKIP_MERGED_MASKS = true;  // Set to false to generate merged visualization masks (imaginary boundaries connecting disconnected components)

// Note: MAX_IMAGE_SIZE parameter removed - automatic resolution matching is more reliable
// For memory optimization, use FORCE_SMALL_INPUT or USE_CPU instead

// UI and workflow state
var TEMP_DIR = "";
var OUTPUT_DIR = "";

/*
 * Main macro function - this is what users click
 */
macro "Segment Myotubes [M]" {
    segmentMyotubes();
}

/*
 * Alternative macro with custom parameters
 */
macro "Segment Myotubes (Custom Parameters)..." {
    if (showParameterDialog()) {
        segmentMyotubes();
    }
}

/*
 * CPU-only macro for memory-constrained systems
 */
macro "Segment Myotubes (CPU Mode) [C]" {
    // Temporarily enable CPU mode
    original_cpu = USE_CPU;
    USE_CPU = true;

    segmentMyotubes();

    // Restore original setting
    USE_CPU = original_cpu;
}

/*
 * Memory-optimized macro with 1024px input resolution
 */
macro "Segment Myotubes (Memory Optimized) [X]" {
    // Temporarily enable memory optimization
    original_force = FORCE_SMALL_INPUT;
    FORCE_SMALL_INPUT = true;

    segmentMyotubes();

    // Restore original setting
    FORCE_SMALL_INPUT = original_force;
}

/*
 * Tiled inference for large images with many myotubes
 * Uses 2×2 grid with 20% overlap to handle images that exceed model query capacity
 */
macro "Segment Myotubes (Tiled High Capacity) [T]" {
    // Temporarily enable tiling
    original_tiling = USE_TILING;
    USE_TILING = true;

    segmentMyotubes();

    // Restore original setting
    USE_TILING = original_tiling;
}

/*
 * GUI mode - User-friendly interface with saved settings
 */
macro "Segment Myotubes (GUI Mode) [G]" {
    segmentMyotubesWithGUI();
}

/*
 * Load previous results (if user wants to reload mask overlays)
 */
macro "Load Myotube Results..." {
    loadPreviousResults();
}

/*
 * Main segmentation function
 */
function segmentMyotubes() {
    // Validate prerequisites
    if (!validateSetup()) {
        return;
    }

    // Ask user to select input directory
    input_dir = getDirectory("Select folder containing images to process");
    if (input_dir == "") {
        print("❌ No input directory selected");
        return;
    }

    print("\\Clear");  // Clear log
    print("=== Batch Myotube Segmentation Started ===");
    print("Input directory: " + input_dir);
    print("Time: " + getTime());

    // Setup directories
    setupDirectories();

    // Show progress
    showProgress(0.1);
    showStatus("Running batch myotube segmentation...");

    // Construct Python command for batch processing
    python_cmd = buildBatchPythonCommand(input_dir);
    print("Command: " + python_cmd);

    // Execute segmentation
    print("Executing batch segmentation...");
    start_time = getTime();

    // Windows-specific execution
    exec("cmd", "/c", python_cmd);

    end_time = getTime();
    processing_time = (end_time - start_time) / 1000;

    // (Cleaned) No extra log tailing. Errors will be reported via ERROR file.

    // Debug: Check what files were created
    print("Debug: Checking output directory...");
    output_files = getFileList(OUTPUT_DIR);
    print("Files found: " + output_files.length);
    for (i = 0; i < output_files.length; i++) {
        print("  - " + output_files[i]);
    }

    showProgress(0.8);
    showStatus("Loading batch results...");

    // Check for success/error (Windows uses backslash separator)
    success_file = OUTPUT_DIR + "\\BATCH_SUCCESS";
    error_file = OUTPUT_DIR + "\\ERROR";

    if (File.exists(success_file)) {
        // Success - load batch results
        loadBatchResults(success_file);
        print("✅ Batch segmentation completed successfully in " + processing_time + " seconds");
        showStatus("Batch myotube segmentation completed successfully!");
    } else if (File.exists(error_file)) {
        // Error occurred
        error_message = File.openAsString(error_file);
        print("❌ Batch segmentation failed:");
        print(error_message);
        showMessage("Batch Segmentation Failed",
                   "An error occurred during batch segmentation:\\n\\n" + error_message);
        showStatus("Batch segmentation failed - check log for details");
    } else {
        // Unknown state
        print("⚠️  No success or error file found - unknown status");
        showMessage("Unknown Status",
                   "Batch segmentation completed but status is unclear.\\n" +
                   "Check the output directory: " + OUTPUT_DIR);
        showStatus("Batch segmentation status unknown");
    }

    showProgress(1.0);

    print("=== Batch Segmentation Complete ===\\n");
}

/*
 * GUI-based segmentation function
 */
function segmentMyotubesWithGUI() {
    // Validate prerequisites
    if (!validateSetup()) {
        return;
    }

    print("\\Clear");  // Clear log
    print("=== Myotube Segmentation (GUI Mode) ===");

    // Setup directories
    setupDirectories();

    // Show progress
    showProgress(0.1);
    showStatus("Launching parameter GUI...");

    // Build Python command with --gui flag (Windows style)
    env_var = "MASK2FORMER_PATH=" + MASK2FORMER_PATH;
    python_script_cmd = PYTHON_COMMAND + " \"" + SCRIPT_PATH + "\" --gui";

    // Windows conda activation - use call conda.bat
    full_cmd = "call conda activate " + CONDA_ENV + " && set " + env_var + " && " + python_script_cmd;

    print("Launching GUI...");
    print("Command: " + full_cmd);

    // Execute segmentation with GUI
    start_time = getTime();

    // Windows-specific execution
    exec("cmd", "/c", full_cmd);

    end_time = getTime();
    processing_time = (end_time - start_time) / 1000;

    showProgress(0.8);
    showStatus("Loading results...");

    // Check for success/error in the standard output directory (Windows uses backslash separator)
    success_file = OUTPUT_DIR + "\\BATCH_SUCCESS";
    error_file = OUTPUT_DIR + "\\ERROR";

    if (File.exists(success_file)) {
        // Success - load batch results
        loadBatchResults(success_file);
        print("✅ Segmentation completed successfully in " + processing_time + " seconds");
        showStatus("Myotube segmentation completed successfully!");
    } else if (File.exists(error_file)) {
        // Error occurred
        error_message = File.openAsString(error_file);
        print("❌ Segmentation failed:");
        print(error_message);
        showMessage("Segmentation Failed",
                   "An error occurred during segmentation:\\n\\n" + error_message);
        showStatus("Segmentation failed - check log for details");
    } else {
        // User may have cancelled or output is in a different directory
        print("ℹ️  GUI session completed");
        print("   If you completed the segmentation, check the output directory you specified");
        showMessage("GUI Session Complete",
                   "If you ran the segmentation, check your specified output directory for results.\\n\\n" +
                   "If you cancelled, you can try again anytime.");
        showStatus("GUI session complete");
    }

    showProgress(1.0);

    print("=== GUI Segmentation Complete ===\\n");
}

/*
 * Validate that all prerequisites are met
 */
function validateSetup() {
    // Auto-detect paths if not set
    if (SCRIPT_PATH == "") {
        // Prefer repo script if MASK2FORMER_PATH is set
        if (MASK2FORMER_PATH != "") {
            // Windows path with backslashes
            repo_script = MASK2FORMER_PATH + "\\fiji_integration\\myotube_segmentation.py";
            if (File.exists(repo_script)) {
                SCRIPT_PATH = repo_script;
            }
        }

        macro_dir = getDirectory("macros");
        plugin_dir = getDirectory("plugins");

        // Try to find the script in common locations
        script_locations = newArray(
            macro_dir + "myotube_segmentation.py",
            plugin_dir + "myotube_segmentation.py",
            getDirectory("startup") + "myotube_segmentation.py",
            repo_script
        );

        if (SCRIPT_PATH == "") {
            for (i = 0; i < script_locations.length; i++) {
                if (File.exists(script_locations[i])) {
                    SCRIPT_PATH = script_locations[i];
                    break;
                }
            }
        }

        if (SCRIPT_PATH == "") {
            showMessage("Setup Error",
                       "Could not find myotube_segmentation.py script.\\n\\n" +
                       "Please ensure the script is in one of these locations:\\n" +
                       "- Fiji macros folder\\n" +
                       "- Fiji plugins folder\\n" +
                       "- Fiji startup folder");
            return false;
        }
    }

    // Test Python availability
    if (!testPythonCommand()) {
        return false;
    }

    print("✅ Setup validation passed");
    print("Script path: " + SCRIPT_PATH);
    return true;
}

/*
 * Test if conda environment and Python command work
 */
function testPythonCommand() {
    // Create a simple test file in Windows temp directory
    test_dir = getDirectory("temp");
    test_file = test_dir + "python_test.txt";

    // Build conda activation command with Python test (Windows style)
    env_var = "MASK2FORMER_PATH=" + MASK2FORMER_PATH;

    // Windows conda activation - use call conda.bat
    test_cmd = "call conda activate " + CONDA_ENV + " && set " + env_var + " && " + PYTHON_COMMAND + " -c \"print('Python test OK')\" > \"" + test_file + "\"";
    exec("cmd", "/c", test_cmd);

    // Check if test file was created
    wait(2000);  // Wait 2 seconds for conda activation

    if (File.exists(test_file)) {
        File.delete(test_file);
        return true;
    } else {
        showMessage("Conda/Python Error",
                   "Could not execute conda environment: " + CONDA_ENV + "\\n\\n" +
                   "Please ensure:\\n" +
                   "1. Conda is installed and in PATH\\n" +
                   "2. Environment '" + CONDA_ENV + "' exists\\n" +
                   "3. Environment has required packages\\n\\n" +
                   "Test manually in Command Prompt:\\n" +
                   "conda activate " + CONDA_ENV + "\\n" +
                   "python -c \"import torch; print('OK')\"");
        return false;
    }
}

/*
 * Setup temporary and output directories
 */
function setupDirectories() {
    // Use Windows-style temp directory
    // Try %USERPROFILE%\AppData\Local\Temp first
    home_dir = getInfo("user.home");
    user_tmp = home_dir + "\\AppData\\Local\\Temp";

    // Fallback to Fiji's temp directory if user temp doesn't exist
    if (!File.exists(user_tmp)) {
        user_tmp = getDirectory("temp");
    }

    // Create myotube_segmentation subdirectory
    TEMP_DIR = user_tmp + "\\myotube_segmentation";
    OUTPUT_DIR = TEMP_DIR + "\\output";

    print("Using temp directory: " + TEMP_DIR);

    // Create directories if they don't exist
    if (!File.exists(TEMP_DIR)) {
        File.makeDirectory(TEMP_DIR);
    }
    if (!File.exists(OUTPUT_DIR)) {
        File.makeDirectory(OUTPUT_DIR);
    }

    print("Temp directory: " + TEMP_DIR);
    print("Output directory: " + OUTPUT_DIR);
}

/*
 * Build the Python command with conda activation and all parameters
 */
function buildPythonCommand(input_image) {
    // Build the Python script command
    python_script_cmd = PYTHON_COMMAND + " \"" + SCRIPT_PATH + "\"";
    python_script_cmd = python_script_cmd + " \"" + input_image + "\"";
    python_script_cmd = python_script_cmd + " \"" + OUTPUT_DIR + "\"";
    python_script_cmd = python_script_cmd + " --confidence " + CONFIDENCE_THRESHOLD;
    python_script_cmd = python_script_cmd + " --min-area " + MIN_AREA;
    python_script_cmd = python_script_cmd + " --max-area " + MAX_AREA;
    python_script_cmd = python_script_cmd + " --final-min-area " + FINAL_MIN_AREA;

    if (FORCE_SMALL_INPUT) {
        python_script_cmd = python_script_cmd + " --force-1024";
    }

    if (SKIP_MERGED_MASKS) {
        python_script_cmd = python_script_cmd + " --skip-merged-masks";
    }

    if (USE_CPU) {
        python_script_cmd = python_script_cmd + " --cpu";
    }

    if (CONFIG_FILE != "") {
        python_script_cmd = python_script_cmd + " --config \"" + CONFIG_FILE + "\"";
    }
    if (MODEL_WEIGHTS != "") {
        python_script_cmd = python_script_cmd + " --weights \"" + MODEL_WEIGHTS + "\"";
    }

    // Wrap with conda activation and environment variable (Windows style)
    env_var = "MASK2FORMER_PATH=" + MASK2FORMER_PATH;

    // Windows conda activation - use call conda.bat
    full_cmd = "call conda activate " + CONDA_ENV + " && set " + env_var + " && " + python_script_cmd;

    return full_cmd;
}

/*
 * Build Python command for batch processing
 */
function buildBatchPythonCommand(input_dir) {
    // Build the Python script command for batch processing
    python_script_cmd = PYTHON_COMMAND + " \"" + SCRIPT_PATH + "\"";
    python_script_cmd = python_script_cmd + " \"" + input_dir + "\"";
    python_script_cmd = python_script_cmd + " \"" + OUTPUT_DIR + "\"";
    python_script_cmd = python_script_cmd + " --confidence " + CONFIDENCE_THRESHOLD;
    python_script_cmd = python_script_cmd + " --min-area " + MIN_AREA;
    python_script_cmd = python_script_cmd + " --max-area " + MAX_AREA;
    python_script_cmd = python_script_cmd + " --final-min-area " + FINAL_MIN_AREA;

    // Tiling parameters
    if (USE_TILING) {
        python_script_cmd = python_script_cmd + " --use-tiling";
        python_script_cmd = python_script_cmd + " --tile-overlap " + (TILE_OVERLAP / 100.0);
    }

    if (FORCE_SMALL_INPUT) {
        python_script_cmd = python_script_cmd + " --force-1024";
    }

    if (SKIP_MERGED_MASKS) {
        python_script_cmd = python_script_cmd + " --skip-merged-masks";
    }

    if (USE_CPU) {
        python_script_cmd = python_script_cmd + " --cpu";
    }

    if (CONFIG_FILE != "") {
        python_script_cmd = python_script_cmd + " --config \"" + CONFIG_FILE + "\"";
    }
    if (MODEL_WEIGHTS != "") {
        python_script_cmd = python_script_cmd + " --weights \"" + MODEL_WEIGHTS + "\"";
    }

    // Wrap with conda activation and environment variable (Windows style)
    env_var = "MASK2FORMER_PATH=" + MASK2FORMER_PATH;

    // Windows conda activation - use call conda.bat
    full_cmd = "call conda activate " + CONDA_ENV + " && set " + env_var + " && " + python_script_cmd;

    return full_cmd;
}

/*
 * Load batch segmentation results
 */
function loadBatchResults(success_file) {
    // Read batch summary
    success_content = File.openAsString(success_file);
    lines = split(success_content, "\\n");

    processed_info = lines[0];  // e.g., "5/10 images processed"
    total_myotubes = lines[1];  // e.g., "150 total myotubes detected"
    failed_info = lines[2];     // e.g., "2 failed images"

    print("📊 Batch Results:");
    print("   " + processed_info);
    print("   " + total_myotubes);
    print("   " + failed_info);

    // Parse numbers for dialog
    processed_parts = split(processed_info, "/");
    successful_count = parseInt(processed_parts[0]);
    total_count = parseInt(split(processed_parts[1], " ")[0]);

    myotube_parts = split(total_myotubes, " ");
    myotube_count = parseInt(myotube_parts[0]);

    // Show batch summary dialog
    showBatchSummaryDialog(successful_count, total_count, myotube_count);

    // Open output directory for user to browse results (Windows Explorer)
    if (File.exists(OUTPUT_DIR)) {
        print("📁 Opening output directory: " + OUTPUT_DIR);
        exec("cmd", "/c", "explorer \"" + OUTPUT_DIR + "\"");
    }
}

/*
 * Load segmentation results into Fiji
 */
function loadResults(success_file) {
    // BYPASS broken File.openAsString() - just read the count and search for files!
    success_content = File.openAsString(success_file);
    num_instances = parseInt(success_content);

    print("🔍 SUCCESS file content: '" + success_content + "'");
    print("🔍 Extracted instances: " + num_instances);

    // We know the output directory - it's the same directory as the SUCCESS file
    base_dir = File.getParent(success_file);
    print("🔍 Looking for files in: " + base_dir);

    roi_file = "";
    raw_overlay_file = "";
    processed_overlay_file = "";

    // Find masks directory and overlay files in the output directory
    masks_dir = "";
    if (base_dir != "" && num_instances > 0) {
        file_list = getFileList(base_dir);
        print("🔍 Found " + file_list.length + " files in output directory");

        for (j = 0; j < file_list.length; j++) {
            filename = file_list[j];
            print("  - " + filename);
            if (endsWith(filename, "_masks") && File.isDirectory(base_dir + "\\" + filename)) {
                masks_dir = base_dir + "\\" + filename;
                print("🔍 Found masks directory: '" + masks_dir + "'");
            } else if (endsWith(filename, "_raw_overlay.tif")) {
                raw_overlay_file = base_dir + "\\" + filename;
                print("🔍 Found raw overlay file: '" + raw_overlay_file + "'");
            } else if (endsWith(filename, "_processed_overlay.tif")) {
                processed_overlay_file = base_dir + "\\" + filename;
                print("🔍 Found processed overlay file: '" + processed_overlay_file + "'");
            }
        }
    }

    print("Loading results:");
    print("  Masks directory: " + masks_dir);
    print("  Raw overlay file: " + raw_overlay_file);
    print("  Processed overlay file: " + processed_overlay_file);
    print("  Instances: " + num_instances);

    // Load individual mask images
    if (File.exists(masks_dir) && File.isDirectory(masks_dir) && num_instances > 0) {
        print("🔍 Masks directory exists: " + masks_dir);

        // Get list of mask files
        mask_files = getFileList(masks_dir);
        print("🔍 Found " + mask_files.length + " files in masks directory");

        // Open both overlay images
        if (File.exists(raw_overlay_file)) {
            open(raw_overlay_file);
            print("✅ Opened raw overlay image: " + raw_overlay_file);
        }

        if (File.exists(processed_overlay_file)) {
            open(processed_overlay_file);
            print("✅ Opened processed overlay image: " + processed_overlay_file);
        }

        // Load new ROIs
        if (File.exists(roi_file)) {
            roiManager("Open", roi_file);
            print("🔍 ROI Manager count after loading: " + roiManager("count"));
            print("✅ Loaded " + roiManager("count") + " ROIs into ROI Manager");

            // ROIs are now loaded - user can manually show them if needed
            if (roiManager("count") > 0) {
                print("✅ " + roiManager("count") + " ROIs loaded into ROI Manager");
                print("   💡 You can now use ROI Manager buttons to show/hide ROIs");
            } else {
                print("⚠️ No ROIs loaded - check ROI file format");
            }
        } else {
            print("❌ ROI file not found: " + roi_file);
        }
    } else {
        print("❌ ROI file missing or no instances:");
        print("   File exists: " + File.exists(roi_file));
        print("   Instances: " + num_instances);
    }

    // Arrange windows nicely if we have overlay images
    if (nImages > 0) {
        arrangeWindows();
    }

    // Show summary
    showSummaryDialog(num_instances);
}

/*
 * Arrange windows for best viewing
 */
function arrangeWindows() {
    // Get screen dimensions
    screen_width = screenWidth;
    screen_height = screenHeight;

    // Arrange windows based on number of open images
    if (nImages == 1) {
        // Single image - center it
        window_width = screen_width - 100;
        window_height = screen_height - 200;
        selectWindow(1);
        setLocation(50, 50, window_width, window_height);
    } else if (nImages == 2) {
        // Two images (raw + processed overlays) - side by side
        window_width = screen_width / 2 - 50;
        window_height = screen_height - 200;

        selectWindow(1);
        rename("Raw Overlay (All Detections)");
        setLocation(10, 10, window_width, window_height);

        selectWindow(2);
        rename("Processed Overlay (Filtered Results)");
        setLocation(window_width + 30, 10, window_width, window_height);
    } else if (nImages >= 3) {
        // Three or more images - arrange in grid
        window_width = screen_width / 3 - 30;
        window_height = screen_height - 200;

        for (i = 1; i <= nImages && i <= 3; i++) {
            selectWindow(i);
            x_pos = (i - 1) * (window_width + 20) + 10;
            setLocation(x_pos, 10, window_width, window_height);
        }
    }

    // Show ROI Manager if it has ROIs
    if (roiManager("count") > 0) {
        roiManager("Show All");
    }
}

/*
 * Show parameter dialog for custom processing
 */
function showParameterDialog() {
    Dialog.create("Myotube Segmentation Parameters");
    Dialog.addMessage("Adjust segmentation parameters:");
    Dialog.addNumber("Confidence Threshold (0-1):", CONFIDENCE_THRESHOLD);
    Dialog.addNumber("Minimum Area (pixels):", MIN_AREA);
    Dialog.addNumber("Maximum Area (pixels):", MAX_AREA);
    Dialog.addNumber("Final Min Area (pixels):", FINAL_MIN_AREA);
    Dialog.addMessage("\\nTiling Options (for images with many myotubes):");
    Dialog.addCheckbox("Use Tiled Inference (2×2 grid with overlap)", USE_TILING);
    Dialog.addNumber("Tile Overlap (%)", TILE_OVERLAP);
    Dialog.addMessage("(Tiling processes image in 4 overlapping tiles to handle high myotube count)");
    Dialog.addMessage("\\nMemory & Performance Options:");
    Dialog.addCheckbox("Use CPU (slower but less memory)", USE_CPU);
    Dialog.addCheckbox("Force 1024px input (memory optimization, may reduce accuracy)", FORCE_SMALL_INPUT);
    Dialog.addMessage("\\nOutput Options:");
    Dialog.addCheckbox("Skip merged masks (skip imaginary boundary generation)", SKIP_MERGED_MASKS);
    Dialog.addMessage("\\nAdvanced Options:");
    Dialog.addString("Conda Environment:", CONDA_ENV, 20);
    Dialog.addString("Python Command:", PYTHON_COMMAND, 20);
    Dialog.addString("Mask2Former Path:", MASK2FORMER_PATH, 50);
    Dialog.addMessage("(Full path to Mask2Former project directory - Windows style with backslashes)");

    Dialog.show();

    // Get values
    CONFIDENCE_THRESHOLD = Dialog.getNumber();
    MIN_AREA = Dialog.getNumber();
    MAX_AREA = Dialog.getNumber();
    FINAL_MIN_AREA = Dialog.getNumber();
    USE_TILING = Dialog.getCheckbox();
    TILE_OVERLAP = Dialog.getNumber();
    USE_CPU = Dialog.getCheckbox();
    FORCE_SMALL_INPUT = Dialog.getCheckbox();
    SKIP_MERGED_MASKS = Dialog.getCheckbox();
    CONDA_ENV = Dialog.getString();
    PYTHON_COMMAND = Dialog.getString();
    MASK2FORMER_PATH = Dialog.getString();

    // Validate parameters
    if (CONFIDENCE_THRESHOLD < 0 || CONFIDENCE_THRESHOLD > 1) {
        showMessage("Invalid Parameter", "Confidence threshold must be between 0 and 1");
        return false;
    }

    if (MIN_AREA <= 0 || MAX_AREA <= MIN_AREA) {
        showMessage("Invalid Parameter", "Area values must be positive and max > min");
        return false;
    }

    if (FINAL_MIN_AREA < 0) {
        showMessage("Invalid Parameter", "Final minimum area must be positive");
        return false;
    }

    return true;
}

/*
 * Load previous segmentation results
 */
function loadPreviousResults() {
    output_dir = getDirectory("Choose output directory with previous results");
    if (output_dir == "") return;

    // Look for success file
    success_file = output_dir + "SUCCESS";
    if (File.exists(success_file)) {
        loadResults(success_file);
    } else {
        showMessage("No Results", "No valid segmentation results found in selected directory.");
    }
}

/*
 * Show summary dialog with results
 */
function showSummaryDialog(num_instances) {
    message = "Segmentation Results:\\n\\n";
    message = message + "🔬 Myotubes detected: " + num_instances + "\\n";
    message = message + "📊 ROIs loaded: " + roiManager("count") + "\\n\\n";

    if (num_instances > 0) {
        message = message + "Next steps:\\n";
        message = message + "• Review ROIs in ROI Manager\\n";
        message = message + "• Delete false positives if needed\\n";
        message = message + "• Use 'Measure' to analyze myotubes\\n";
        message = message + "• Export measurements to CSV\\n";
    } else {
        message = message + "No myotubes detected.\\n";
        message = message + "Try adjusting parameters or check image quality.";
    }

    showMessage("Segmentation Complete", message);
}

/*
 * Show batch processing summary dialog
 */
function showBatchSummaryDialog(successful_count, total_count, myotube_count) {
    message = "Batch Processing Results:\\n\\n";
    message = message + "📁 Images processed: " + successful_count + "/" + total_count + "\\n";
    if (total_count - successful_count > 0) {
        message = message + "❌ Failed images: " + (total_count - successful_count) + "\\n";
    }
    message = message + "🔬 Total myotubes detected: " + myotube_count + "\\n\\n";

    if (myotube_count > 0) {
        message = message + "Results are saved in separate folders for each image.\\n\\n";
        message = message + "Next steps:\\n";
        message = message + "• Browse output directory (opened automatically)\\n";
        message = message + "• Review individual image results\\n";
        message = message + "• Check mask images and overlays\\n";
        message = message + "• Analyze CSV measurements\\n";
    } else {
        message = message + "No myotubes detected in any images.\\n";
        message = message + "Try adjusting parameters or check image quality.";
    }

    showMessage("Batch Processing Complete", message);
}

/*
 * Utility function to show messages with proper formatting
 */
function showMessage(title, message) {
    // Replace \\n with actual newlines for display
    formatted_message = replace(message, "\\n", "\n");
    Dialog.create(title);
    Dialog.addMessage(formatted_message);
    Dialog.show();
}

/*
 * Install Python dependencies from requirements.txt
 */
function installDependencies() {
    print("\\Clear");
    print("=== Installing Python Dependencies ===");

    // Find requirements.txt file
    macro_dir = getDirectory("macros");
    plugin_dir = getDirectory("plugins");

    requirements_locations = newArray(
        macro_dir + "requirements.txt",
        plugin_dir + "requirements.txt",
        MASK2FORMER_PATH + "\\fiji_integration\\requirements.txt"
    );

    requirements_file = "";
    for (i = 0; i < requirements_locations.length; i++) {
        if (File.exists(requirements_locations[i])) {
            requirements_file = requirements_locations[i];
            break;
        }
    }

    if (requirements_file == "") {
        showMessage("Requirements File Not Found",
                   "Could not find requirements.txt\\n\\n" +
                   "Please ensure requirements.txt is in:\\n" +
                   "- Fiji macros folder\\n" +
                   "- Fiji plugins folder\\n" +
                   "- Mask2Former\\fiji_integration folder");
        return;
    }

    print("Found requirements.txt: " + requirements_file);

    // Build pip install command
    pip_cmd = PYTHON_COMMAND + " -m pip install -r \"" + requirements_file + "\"";

    // Windows conda activation - use call conda.bat
    full_cmd = "call conda activate " + CONDA_ENV + " && " + pip_cmd;

    print("Installing dependencies...");
    print("Command: " + full_cmd);
    print("This may take a few seconds to several minutes depending on what needs to be installed.");
    print("");

    showStatus("Installing Python dependencies...");

    // Execute installation
    start_time = getTime();

    // Windows-specific execution
    exec("cmd", "/c", full_cmd);

    end_time = getTime();
    install_time = (end_time - start_time) / 1000;

    print("");
    print("✅ Installation completed in " + install_time + " seconds");
    print("=== Installation Complete ===");

    showMessage("Installation Complete",
               "Python dependencies have been installed/verified.\\n\\n" +
               "Time: " + install_time + " seconds\\n\\n" +
               "Check the log window for details.");

    showStatus("Installation complete");
}

/*
 * Install Python dependencies
 */
macro "Install Python Dependencies" {
    installDependencies();
}

/*
 * Installation and setup instructions (help macro)
 */
macro "Myotube Segmentation Help" {
    help_text = "Myotube Instance Segmentation for Fiji (Windows)\\n\\n";
    help_text = help_text + "SETUP INSTRUCTIONS:\\n";
    help_text = help_text + "1. Create conda environment 'm2f'\\n";
    help_text = help_text + "2. Install required packages in environment\\n";
    help_text = help_text + "3. Place myotube_segmentation.py in Fiji folder\\n";
    help_text = help_text + "4. Ensure trained model weights are available\\n\\n";
    help_text = help_text + "CONDA SETUP (in Command Prompt):\\n";
    help_text = help_text + "conda create -n m2f python=3.8\\n";
    help_text = help_text + "conda activate m2f\\n";
    help_text = help_text + "pip install -r requirements.txt\\n\\n";
    help_text = help_text + "USAGE:\\n";
    help_text = help_text + "1. Open image in Fiji\\n";
    help_text = help_text + "2. Run 'Segment Myotubes' macro or press 'M'\\n";
    help_text = help_text + "3. Review results in ROI Manager\\n\\n";
    help_text = help_text + "TROUBLESHOOTING:\\n";
    help_text = help_text + "• Test in Command Prompt: conda activate m2f && python -c \"import torch\"\\n";
    help_text = help_text + "• Check conda environment exists\\n";
    help_text = help_text + "• Verify all required packages installed\\n";
    help_text = help_text + "• Check model file paths (use Windows backslashes)\\n";
    help_text = help_text + "• See console/log for error details\\n\\n";
    help_text = help_text + "For more information, see README.md";

    showMessage("Myotube Segmentation Help", help_text);
}

// Initialization message
print("Myotube Segmentation macro (Windows version) loaded successfully!");
print("Use 'Segment Myotubes' or press 'M' to start segmentation.");
print("Use 'Myotube Segmentation Help' for setup instructions.");
