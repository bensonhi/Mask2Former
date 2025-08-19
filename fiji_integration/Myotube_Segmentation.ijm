/*
 * Myotube Instance Segmentation for Fiji
 * 
 * This macro provides seamless integration of AI-powered myotube instance segmentation
 * into the Fiji/ImageJ workflow. Users simply click a button to segment myotubes
 * in the current image.
 * 
 * Features:
 * - One-click myotube segmentation
 * - Automatic ROI loading into ROI Manager
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
 * 1. Open an image in Fiji
 * 2. Run "Segment Myotubes" macro (or press 'M' shortcut)
 * 3. Wait for processing (progress shown in status bar)
 * 4. View results in ROI Manager and overlay
 */

// Configuration - Update these paths for your system
var CONDA_ENV = "m2f";  // Conda environment name
var PYTHON_COMMAND = "python";  // Python command within the conda environment
var MASK2FORMER_PATH = "/fs04/scratch2/tf41/ben/Mask2Former";  // Path to Mask2Former project
var SCRIPT_PATH = "";  // Auto-detected based on macro location
var CONFIG_FILE = "";  // Auto-detected
var MODEL_WEIGHTS = "";  // Auto-detected

// Processing parameters (can be adjusted by users)
var CONFIDENCE_THRESHOLD = 0.5;
var MIN_AREA = 100;
var MAX_AREA = 50000;
var USE_CPU = false;  // Set to true to force CPU inference (slower but less memory)
var MAX_IMAGE_SIZE = 2048;  // Maximum image dimension (larger images will be resized)
var FORCE_SMALL_INPUT = false;  // Set to true to force 1024px input (memory optimization, may reduce accuracy)

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
 * Load previous results (if user wants to reload ROIs)
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
    
    // Check if image is open
    if (nImages == 0) {
        showMessage("No Image", "Please open an image first.");
        return;
    }
    
    // Get current image info
    original_title = getTitle();
    original_id = getImageID();
    
    print("\\Clear");  // Clear log
    print("=== Myotube Segmentation Started ===");
    print("Image: " + original_title);
    print("Time: " + getTime());
    
    // Setup directories
    setupDirectories();
    
    // Save current image to temporary location with safe filename
    // Create safe filename (remove spaces and special characters)
    safe_name = replace(original_title, " ", "_");
    safe_name = replace(safe_name, "-", "_");
    safe_name = replace(safe_name, "(", "_");
    safe_name = replace(safe_name, ")", "_");
    safe_name = replace(safe_name, "[", "_");
    safe_name = replace(safe_name, "]", "_");
    
    // Remove any existing "input_" prefix to prevent accumulation
    while (startsWith(safe_name, "input_")) {
        safe_name = substring(safe_name, 6);
    }
    
    temp_input = TEMP_DIR + File.separator + "input_" + safe_name;
    if (endsWith(temp_input, ".tif") == false) {
        temp_input = temp_input + ".tif";
    }
    
    print("Saving input image: " + temp_input);
    saveAs("Tiff", temp_input);
    
    // Show progress
    showProgress(0.1);
    showStatus("Running myotube segmentation...");
    
    // Construct Python command
    python_cmd = buildPythonCommand(temp_input);
    print("Command: " + python_cmd);
    
    // Execute segmentation
    print("Executing segmentation...");
    start_time = getTime();
    
    // Use eval to execute the command (platform independent)
    if (startsWith(getInfo("os.name"), "Windows")) {
        exec("cmd", "/c", python_cmd);
    } else {
        exec("sh", "-c", python_cmd);
    }
    
    end_time = getTime();
    processing_time = (end_time - start_time) / 1000;
    
    // Debug: Check what files were created
    print("Debug: Checking output directory...");
    output_files = getFileList(OUTPUT_DIR);
    print("Files found: " + output_files.length);
    for (i = 0; i < output_files.length; i++) {
        print("  - " + output_files[i]);
    }
    
    showProgress(0.8);
    showStatus("Loading results...");
    
    // Check for success/error
    success_file = OUTPUT_DIR + File.separator + "SUCCESS";
    error_file = OUTPUT_DIR + File.separator + "ERROR";
    
    if (File.exists(success_file)) {
        // Success - load results
        loadResults(success_file);
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
        // Unknown state
        print("⚠️  No success or error file found - unknown status");
        showMessage("Unknown Status", 
                   "Segmentation completed but status is unclear.\\n" +
                   "Check the output directory: " + OUTPUT_DIR);
        showStatus("Segmentation status unknown");
    }
    
    showProgress(1.0);
    
    // Cleanup temporary files
    if (File.exists(temp_input)) {
        File.delete(temp_input);
    }
    
    print("=== Segmentation Complete ===\\n");
}

/*
 * Validate that all prerequisites are met
 */
function validateSetup() {
    // Auto-detect paths if not set
    if (SCRIPT_PATH == "") {
        macro_dir = getDirectory("macros");
        plugin_dir = getDirectory("plugins");
        
        // Try to find the script in common locations
        script_locations = newArray(
            macro_dir + "myotube_segmentation.py",
            plugin_dir + "myotube_segmentation.py",
            getDirectory("startup") + "myotube_segmentation.py"
        );
        
        for (i = 0; i < script_locations.length; i++) {
            if (File.exists(script_locations[i])) {
                SCRIPT_PATH = script_locations[i];
                break;
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
 * Test if conda environment and Python command work`
 */
function testPythonCommand() {
    // Create a simple test file
    test_dir = getDirectory("temp");
    test_file = test_dir + "python_test.txt";
    
    // Build conda activation command with Python test
    env_var = "MASK2FORMER_PATH=" + MASK2FORMER_PATH;
    
    if (startsWith(getInfo("os.name"), "Windows")) {
        // Windows conda activation
        test_cmd = "conda activate " + CONDA_ENV + " && set " + env_var + " && " + PYTHON_COMMAND + " -c \"print('Python test OK')\" > \"" + test_file + "\"";
        exec("cmd", "/c", test_cmd);
    } else {
        // Unix/Mac conda activation
        test_cmd = "source $(conda info --base)/etc/profile.d/conda.sh && conda activate " + CONDA_ENV + " && export " + env_var + " && " + PYTHON_COMMAND + " -c \"print('Python test OK')\" > \"" + test_file + "\"";
        exec("sh", "-c", test_cmd);
    }
    
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
                   "Test manually:\\n" +
                   "conda activate " + CONDA_ENV + "\\n" +
                   "python -c \"import torch; print('OK')\"");
        return false;
    }
}

/*
 * Setup temporary and output directories
 */
function setupDirectories() {
    // Try user's home directory first, fallback to system temp
    home_dir = getInfo("user.home");
    user_tmp = home_dir + File.separator + "tmp";
    
    // Create ~/tmp if it doesn't exist
    if (!File.exists(user_tmp)) {
        File.makeDirectory(user_tmp);
    }
    
    // Try to use ~/tmp, fallback to system temp if it fails
    if (File.exists(user_tmp)) {
        TEMP_DIR = user_tmp + File.separator + "myotube_segmentation";
        print("Using user temp directory: " + TEMP_DIR);
    } else {
        TEMP_DIR = getDirectory("temp") + "myotube_segmentation";
        print("Falling back to system temp directory: " + TEMP_DIR);
    }
    
    OUTPUT_DIR = TEMP_DIR + File.separator + "output";
    
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
    
    if (FORCE_SMALL_INPUT) {
        python_script_cmd = python_script_cmd + " --force-1024";
    } else {
        python_script_cmd = python_script_cmd + " --max-image-size " + MAX_IMAGE_SIZE;
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
    
    // Wrap with conda activation and environment variable
    env_var = "MASK2FORMER_PATH=" + MASK2FORMER_PATH;
    
    if (startsWith(getInfo("os.name"), "Windows")) {
        // Windows conda activation
        full_cmd = "conda activate " + CONDA_ENV + " && set " + env_var + " && " + python_script_cmd;
    } else {
        // Unix/Mac conda activation
        full_cmd = "source $(conda info --base)/etc/profile.d/conda.sh && conda activate " + CONDA_ENV + " && export " + env_var + " && " + python_script_cmd;
    }
    
    return full_cmd;
}

/*
 * Load segmentation results into Fiji
 */
function loadResults(success_file) {
    // Read success file to get result file paths
    success_content = File.openAsString(success_file);
    lines = split(success_content, "\\n");
    
    print("🔍 DEBUG: SUCCESS file contains " + lines.length + " lines:");
    for (j = 0; j < lines.length; j++) {
        print("  Line " + j + ": '" + lines[j] + "'");
    }
    
    base_dir = "";
    dir_chunks = newArray(20); // Support up to 20 chunks
    roi_file = "";
    overlay_file = "";
    num_instances = 0;
    max_chunk_idx = -1;
    
    for (i = 0; i < lines.length; i++) {
        line = lines[i];
        
        // Check for numbered directory chunks (0:, 1:, 2:, etc.)
        if (indexOf(line, ":") > 0) {
            colon_pos = indexOf(line, ":");
            prefix = substring(line, 0, colon_pos);
            content = substring(line, colon_pos + 1);
            
            // Check if prefix is a number (directory chunk)
            chunk_idx = parseInt(prefix);
            if (chunk_idx >= 0 && chunk_idx < 20) {
                dir_chunks[chunk_idx] = content;
                if (chunk_idx > max_chunk_idx) max_chunk_idx = chunk_idx;
                print("🔍 Chunk " + chunk_idx + ": '" + content + "'");
            }
        } else if (startsWith(line, "N:")) {
            count_part = substring(line, 2);
            num_instances = parseInt(count_part);
            print("🔍 Extracted instances: " + num_instances);
        }
    }
    
    // Reconstruct base directory from chunks
    if (max_chunk_idx >= 0) {
        for (j = 0; j <= max_chunk_idx; j++) {
            if (dir_chunks[j] != "") {
                base_dir = base_dir + dir_chunks[j];
            }
        }
        print("🔍 Reconstructed base dir: '" + base_dir + "'");
    }
    
    // Find ROI and overlay files in the directory
    if (base_dir != "" && num_instances > 0) {
        print("🔍 Looking for files in: " + base_dir);
        
        // Look for ROI zip file
        file_list = getFileList(base_dir);
        for (j = 0; j < file_list.length; j++) {
            filename = file_list[j];
            if (endsWith(filename, "_rois.zip")) {
                roi_file = base_dir + File.separator + filename;
                print("🔍 Found ROI file: '" + roi_file + "'");
            } else if (endsWith(filename, "_overlay.tif")) {
                overlay_file = base_dir + File.separator + filename;
                print("🔍 Found overlay file: '" + overlay_file + "'");
            }
        }
    }
    
    print("Loading results:");
    print("  ROI file: " + roi_file);
    print("  Overlay file: " + overlay_file);
    print("  Instances: " + num_instances);
    
    // Load ROIs into ROI Manager
    if (File.exists(roi_file) && num_instances > 0) {
        print("🔍 ROI file exists: " + roi_file);
        print("🔍 File size: " + File.length(roi_file) + " bytes");
        
        // Clear existing ROIs (ask user first)
        if (roiManager("count") > 0) {
            result = getBoolean("Clear existing ROIs in ROI Manager?");
            if (result) {
                roiManager("reset");
            }
        }
        
        print("🔍 ROI Manager count before loading: " + roiManager("count"));
        
        // Load new ROIs
        try {
            roiManager("Open", roi_file);
            print("🔍 ROI Manager count after loading: " + roiManager("count"));
            print("✅ Loaded " + roiManager("count") + " ROIs into ROI Manager");
            
            // Show all ROIs on original image
            if (roiManager("count") > 0) {
                roiManager("Show All");
                roiManager("Show All with labels");
            } else {
                print("⚠️ No ROIs loaded - check ROI file format");
            }
        } catch (error) {
            print("❌ Error loading ROIs: " + error);
        }
    } else {
        print("❌ ROI file missing or no instances:");
        print("   File exists: " + File.exists(roi_file));
        print("   Instances: " + num_instances);
    }
    
    // Open overlay image
    if (File.exists(overlay_file)) {
        open(overlay_file);
        overlay_title = getTitle();
        print("✅ Opened overlay image: " + overlay_title);
        
        // Arrange windows nicely
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
    
    // If we have multiple images, arrange them side by side
    if (nImages >= 2) {
        window_width = screen_width / 2 - 50;
        window_height = screen_height - 200;
        
        // Select and position original image
        selectWindow(1);
        setLocation(10, 10, window_width, window_height);
        
        // Select and position overlay
        selectWindow(2);
        setLocation(window_width + 30, 10, window_width, window_height);
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
    Dialog.addMessage("\\nMemory & Performance Options:");
    Dialog.addCheckbox("Use CPU (slower but less memory)", USE_CPU);
    Dialog.addCheckbox("Force 1024px input (memory optimization, may reduce accuracy)", FORCE_SMALL_INPUT);
    Dialog.addNumber("Max Image Size (pixels):", MAX_IMAGE_SIZE);
    Dialog.addMessage("\\nAdvanced Options:");
    Dialog.addString("Conda Environment:", CONDA_ENV, 20);
    Dialog.addString("Python Command:", PYTHON_COMMAND, 20);
    Dialog.addString("Mask2Former Path:", MASK2FORMER_PATH, 50);
    Dialog.addMessage("(Full path to Mask2Former project directory)");
    
    Dialog.show();
    
    // Get values
    CONFIDENCE_THRESHOLD = Dialog.getNumber();
    MIN_AREA = Dialog.getNumber();
    MAX_AREA = Dialog.getNumber();
    USE_CPU = Dialog.getCheckbox();
    FORCE_SMALL_INPUT = Dialog.getCheckbox();
    MAX_IMAGE_SIZE = Dialog.getNumber();
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
 * Installation and setup instructions (help macro)
 */
macro "Myotube Segmentation Help" {
    help_text = "Myotube Instance Segmentation for Fiji\\n\\n";
    help_text = help_text + "SETUP INSTRUCTIONS:\\n";
    help_text = help_text + "1. Create conda environment 'm2f'\\n";
    help_text = help_text + "2. Install required packages in environment\\n";
    help_text = help_text + "3. Place myotube_segmentation.py in Fiji folder\\n";
    help_text = help_text + "4. Ensure trained model weights are available\\n\\n";
    help_text = help_text + "CONDA SETUP:\\n";
    help_text = help_text + "conda create -n m2f python=3.8\\n";
    help_text = help_text + "conda activate m2f\\n";
    help_text = help_text + "pip install -r requirements.txt\\n\\n";
    help_text = help_text + "USAGE:\\n";
    help_text = help_text + "1. Open image in Fiji\\n";
    help_text = help_text + "2. Run 'Segment Myotubes' macro or press 'M'\\n";
    help_text = help_text + "3. Review results in ROI Manager\\n\\n";
    help_text = help_text + "TROUBLESHOOTING:\\n";
    help_text = help_text + "• Test: conda activate m2f && python -c 'import torch'\\n";
    help_text = help_text + "• Check conda environment exists\\n";
    help_text = help_text + "• Verify all required packages installed\\n";
    help_text = help_text + "• Check model file paths\\n";
    help_text = help_text + "• See console/log for error details\\n\\n";
    help_text = help_text + "For more information, see README.md";
    
    showMessage("Myotube Segmentation Help", help_text);
}

// Initialization message
print("Myotube Segmentation macro loaded successfully!");
print("Use 'Segment Myotubes' or press 'M' to start segmentation.");
print("Use 'Myotube Segmentation Help' for setup instructions.");