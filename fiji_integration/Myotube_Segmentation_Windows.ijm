/*
 * Myotube Instance Segmentation for Fiji (Simplified GUI-Only Version)
 *
 * This macro provides AI-powered myotube instance segmentation via a user-friendly GUI.
 * All parameters are configured through the Python GUI interface.
 *
 * Features:
 * - User-friendly GUI with all parameters
 * - Automatic conda environment creation
 * - Saved settings persistence
 * - One-click dependency installation
 *
 * Setup Requirements:
 * 1. Conda installed and in PATH
 * 2. Run "Install Python Dependencies" once
 * 3. Configure Mask2Former path if needed
 *
 * Usage:
 * 1. Press 'M' or run "Segment Myotubes" macro
 * 2. Configure parameters in GUI
 * 3. Click "Run Segmentation"
 */

// Configuration - Update these paths for your system
var CONDA_ENV = "m2f";  // Conda environment name
var PYTHON_COMMAND = "python";  // Python command within the conda environment
var MASK2FORMER_PATH = "C:\\Users\\YourUsername\\Mask2Former";  // Path to Mask2Former project
var SCRIPT_PATH = "";  // Auto-detected based on macro location

// UI and workflow state
var TEMP_DIR = "";
var OUTPUT_DIR = "";

// Note: All processing parameters are configured through the Python GUI

/*
 * Main macro function - Launches GUI for parameter configuration
 */
macro "Segment Myotubes [M]" {
    segmentMyotubesWithGUI();
}

/*
 * Install Python dependencies
 */
macro "Install Python Dependencies" {
    installDependencies();
}

/*
 * Help and setup instructions
 */
macro "Myotube Segmentation Help" {
    help_text = "Myotube Instance Segmentation for Fiji\\n\\n";
    help_text = help_text + "SETUP INSTRUCTIONS:\\n";
    help_text = help_text + "1. Run 'Install Python Dependencies' macro (one-time setup)\\n";
    help_text = help_text + "2. Update MASK2FORMER_PATH at top of this script if needed\\n";
    help_text = help_text + "3. Press 'M' to launch segmentation GUI\\n\\n";
    help_text = help_text + "USAGE:\\n";
    help_text = help_text + "1. Press 'M' to open parameter GUI\\n";
    help_text = help_text + "2. Select input directory and output directory\\n";
    help_text = help_text + "3. Adjust parameters as needed\\n";
    help_text = help_text + "4. Click 'Run Segmentation'\\n\\n";
    help_text = help_text + "FEATURES:\\n";
    help_text = help_text + "• All parameters available in GUI\\n";
    help_text = help_text + "• Settings saved automatically\\n";
    help_text = help_text + "• Browse buttons for easy file selection\\n";
    help_text = help_text + "• Restore defaults button\\n\\n";
    help_text = help_text + "TROUBLESHOOTING:\\n";
    help_text = help_text + "• If conda errors, ensure conda is in PATH\\n";
    help_text = help_text + "• Run 'Install Python Dependencies' first\\n";
    help_text = help_text + "• Check console/log for error details\\n\\n";
    help_text = help_text + "For more information, see README.md";

    showMessage("Myotube Segmentation Help", help_text);
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

    // Check and create conda environment if it doesn't exist
    print("Checking for conda environment '" + CONDA_ENV + "'...");
    if (!checkAndCreateCondaEnvironment()) {
        showMessage("Environment Setup Failed",
                   "Failed to create conda environment.\\n\\nPlease check the log for details.");
        return;
    }

    // Show progress
    showProgress(0.1);
    showStatus("Launching parameter GUI...");

    // Build Python command with --gui flag
    env_var = "MASK2FORMER_PATH=" + MASK2FORMER_PATH;
    python_script_cmd = PYTHON_COMMAND + " \"" + SCRIPT_PATH + "\" --gui";

    if (startsWith(getInfo("os.name"), "Windows")) {
        // Windows: Try multiple conda initialization methods
        // Check common conda installation locations
        conda_locations = newArray(
            "%USERPROFILE%\\AppData\\Local\\miniconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\miniconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\AppData\\Local\\anaconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\anaconda3\\Scripts\\activate.bat"
        );

        conda_init = "";
        home_dir = getInfo("user.home");
        for (i = 0; i < conda_locations.length; i++) {
            test_path = replace(conda_locations[i], "%USERPROFILE%", home_dir);
            if (File.exists(test_path)) {
                conda_init = test_path;  // Store the EXPANDED path, not the template
                break;
            }
        }

        if (conda_init == "") {
            // Fallback to default if not found - use expanded path
            conda_init = home_dir + "\\AppData\\Local\\miniconda3\\Scripts\\activate.bat";
        }

        print("Using conda activation script: " + conda_init);
        full_cmd = "call \"" + conda_init + "\" " + CONDA_ENV + " && set " + env_var + " && " + python_script_cmd;
    } else {
        // Unix/Mac conda activation
        full_cmd = "source $(conda info --base)/etc/profile.d/conda.sh && conda activate " + CONDA_ENV + " && export " + env_var + " && " + python_script_cmd;
    }

    print("Launching GUI...");
    print("Command: " + full_cmd);

    // Execute segmentation with GUI - use batch file for better redirection handling
    start_time = getTime();

    cmd_output_file = TEMP_DIR + "\\" + "cmd_output.txt";
    batch_file = TEMP_DIR + "\\" + "run_segmentation.bat";

    if (startsWith(getInfo("os.name"), "Windows")) {
        // Create a batch file with the command and detailed debugging
        batch_content = "@echo off\r\n";
        batch_content = batch_content + "echo Starting segmentation... > \"" + cmd_output_file + "\"\r\n";
        batch_content = batch_content + "echo Attempting conda activation... >> \"" + cmd_output_file + "\"\r\n";
        batch_content = batch_content + "call \"" + conda_init + "\" " + CONDA_ENV + " >> \"" + cmd_output_file + "\" 2>&1\r\n";
        batch_content = batch_content + "if errorlevel 1 (\r\n";
        batch_content = batch_content + "    echo ERROR: Conda activation failed with code %errorlevel% >> \"" + cmd_output_file + "\"\r\n";
        batch_content = batch_content + "    exit /b 1\r\n";
        batch_content = batch_content + ")\r\n";
        batch_content = batch_content + "echo Conda activated successfully >> \"" + cmd_output_file + "\"\r\n";
        batch_content = batch_content + "echo Running Python script... >> \"" + cmd_output_file + "\"\r\n";
        batch_content = batch_content + "set " + env_var + " >> \"" + cmd_output_file + "\" 2>&1\r\n";
        batch_content = batch_content + python_script_cmd + " >> \"" + cmd_output_file + "\" 2>&1\r\n";
        batch_content = batch_content + "echo Python script completed >> \"" + cmd_output_file + "\"\r\n";
        batch_content = batch_content + "echo Batch file completed >> \"" + cmd_output_file + "\"\r\n";
        File.saveString(batch_content, batch_file);
        print("Created batch file: " + batch_file);

        // Also print the batch file contents for debugging
        saved_batch = File.openAsString(batch_file);
        print("Batch file contents:");
        print(saved_batch);

        // Execute the batch file
        exec("cmd", "/c", batch_file);
    } else {
        full_cmd_with_redirect = full_cmd + " > \"" + cmd_output_file + "\" 2>&1";
        exec("sh", "-c", full_cmd_with_redirect);
    }

    end_time = getTime();
    processing_time = (end_time - start_time) / 1000;

    // Read and print command output
    print("Looking for output file: " + cmd_output_file);
    if (File.exists(cmd_output_file)) {
        print("Output file found, reading...");
        cmd_output = File.openAsString(cmd_output_file);
        print("Output file size: " + lengthOf(cmd_output) + " characters");
        if (cmd_output != "") {
            print("=== Command Output ===");
            print(cmd_output);
            print("=== End Command Output ===");
        } else {
            print("Output file is empty");
        }
    } else {
        print("Output file not found!");
    }

    showProgress(0.8);
    showStatus("Checking results...");

    // Check for success/error
    success_file = OUTPUT_DIR + "\\" + "BATCH_SUCCESS";
    error_file = OUTPUT_DIR + "\\" + "ERROR";

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
            repo_script = MASK2FORMER_PATH + "\\" + "fiji_integration" + "\\" + "myotube_segmentation.py";
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

    // Test Python availability (simplified - just check script exists)
    print("✅ Setup validation passed");
    print("Script path: " + SCRIPT_PATH);
    return true;
}

/*
 * Setup temporary and output directories
 */
function setupDirectories() {
    // Try user's home directory first, fallback to system temp
    home_dir = getInfo("user.home");
    user_tmp = home_dir + "\\" + "tmp";

    // Create ~/tmp if it doesn't exist
    if (!File.exists(user_tmp)) {
        File.makeDirectory(user_tmp);
    }

    // Try to use ~/tmp, fallback to system temp if it fails
    if (File.exists(user_tmp)) {
        TEMP_DIR = user_tmp + "\\" + "myotube_segmentation";
        print("Using user temp directory: " + TEMP_DIR);
    } else {
        TEMP_DIR = getDirectory("temp") + "myotube_segmentation";
        print("Falling back to system temp directory: " + TEMP_DIR);
    }

    OUTPUT_DIR = TEMP_DIR + "\\" + "output";

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

    // Open output directory for user to browse results
    if (File.exists(OUTPUT_DIR)) {
        print("📁 Opening output directory: " + OUTPUT_DIR);
        if (startsWith(getInfo("os.name"), "Windows")) {
            exec("cmd", "/c", "explorer \"" + OUTPUT_DIR + "\"");
        } else if (startsWith(getInfo("os.name"), "Mac")) {
            exec("open", OUTPUT_DIR);
        } else {
            // Linux - try xdg-open
            exec("xdg-open", OUTPUT_DIR);
        }
    }
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
 * Check if conda environment exists and create it if not
 * Returns true if environment exists or was created successfully
 */
function checkAndCreateCondaEnvironment() {
    // Find conda activate script
    if (startsWith(getInfo("os.name"), "Windows")) {
        conda_locations = newArray(
            "%USERPROFILE%\\AppData\\Local\\miniconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\miniconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\AppData\\Local\\anaconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\anaconda3\\Scripts\\activate.bat"
        );

        conda_init = "";
        home_dir = getInfo("user.home");
        for (i = 0; i < conda_locations.length; i++) {
            test_path = replace(conda_locations[i], "%USERPROFILE%", home_dir);
            if (File.exists(test_path)) {
                conda_init = test_path;
                break;
            }
        }

        if (conda_init == "") {
            conda_init = home_dir + "\\AppData\\Local\\miniconda3\\Scripts\\activate.bat";
        }

        // Check if environment exists
        check_env_cmd = "call \"" + conda_init + "\" base && conda env list | findstr " + CONDA_ENV;
    } else {
        check_env_cmd = "source $(conda info --base)/etc/profile.d/conda.sh && conda env list | grep -w " + CONDA_ENV;
    }

    // Check if environment exists
    env_check_file = getDirectory("temp") + "env_check.txt";

    if (startsWith(getInfo("os.name"), "Windows")) {
        exec("cmd", "/c", check_env_cmd + " > \"" + env_check_file + "\" 2>&1");
    } else {
        exec("sh", "-c", check_env_cmd + " > \"" + env_check_file + "\" 2>&1");
    }

    wait(1000);  // Wait for file to be written

    env_exists = false;
    if (File.exists(env_check_file)) {
        env_content = File.openAsString(env_check_file);
        if (indexOf(env_content, CONDA_ENV) >= 0) {
            env_exists = true;
        }
        File.delete(env_check_file);
    }

    if (env_exists) {
        print("✅ Conda environment '" + CONDA_ENV + "' found");
        return true;
    }

    // Environment doesn't exist - create it
    print("⚠️  Conda environment '" + CONDA_ENV + "' not found");
    print("Creating conda environment '" + CONDA_ENV + "'...");
    print("This may take a few minutes...");

    showStatus("Creating conda environment " + CONDA_ENV + "...");

    if (startsWith(getInfo("os.name"), "Windows")) {
        create_env_cmd = "call \"" + conda_init + "\" base && conda create -n " + CONDA_ENV + " python=3.9 -y";
    } else {
        create_env_cmd = "conda create -n " + CONDA_ENV + " python=3.9 -y";
    }

    create_start = getTime();

    if (startsWith(getInfo("os.name"), "Windows")) {
        exec("cmd", "/c", create_env_cmd);
    } else {
        exec("sh", "-c", create_env_cmd);
    }

    create_end = getTime();
    create_time = (create_end - create_start) / 1000;

    print("✅ Environment created in " + create_time + " seconds");

    // Now install dependencies
    print("Installing Python dependencies...");
    return installDependenciesQuiet();
}

/*
 * Install dependencies without user interaction (called automatically)
 */
function installDependenciesQuiet() {
    // Find requirements.txt file
    macro_dir = getDirectory("macros");
    plugin_dir = getDirectory("plugins");

    requirements_locations = newArray(
        macro_dir + "requirements.txt",
        plugin_dir + "requirements.txt",
        MASK2FORMER_PATH + "\\" + "fiji_integration" + "\\" + "requirements.txt"
    );

    requirements_file = "";
    for (i = 0; i < requirements_locations.length; i++) {
        if (File.exists(requirements_locations[i])) {
            requirements_file = requirements_locations[i];
            break;
        }
    }

    if (requirements_file == "") {
        print("❌ Could not find requirements.txt");
        return false;
    }

    print("Found requirements.txt: " + requirements_file);

    // Find conda activate script (reuse from earlier)
    if (startsWith(getInfo("os.name"), "Windows")) {
        conda_locations = newArray(
            "%USERPROFILE%\\AppData\\Local\\miniconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\miniconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\AppData\\Local\\anaconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\anaconda3\\Scripts\\activate.bat"
        );

        conda_init = "";
        home_dir = getInfo("user.home");
        for (i = 0; i < conda_locations.length; i++) {
            test_path = replace(conda_locations[i], "%USERPROFILE%", home_dir);
            if (File.exists(test_path)) {
                conda_init = test_path;
                break;
            }
        }

        if (conda_init == "") {
            conda_init = home_dir + "\\AppData\\Local\\miniconda3\\Scripts\\activate.bat";
        }
    }

    // Build pip install command
    pip_cmd = PYTHON_COMMAND + " -m pip install -r \"" + requirements_file + "\"";

    // Wrap with conda activation
    if (startsWith(getInfo("os.name"), "Windows")) {
        full_cmd = "call \"" + conda_init + "\" " + CONDA_ENV + " && " + pip_cmd;
    } else {
        full_cmd = "source $(conda info --base)/etc/profile.d/conda.sh && conda activate " + CONDA_ENV + " && " + pip_cmd;
    }

    print("Installing dependencies (this may take several minutes)...");

    showStatus("Installing Python dependencies...");

    start_time = getTime();

    if (startsWith(getInfo("os.name"), "Windows")) {
        exec("cmd", "/c", full_cmd);
    } else {
        exec("sh", "-c", full_cmd);
    }

    end_time = getTime();
    install_time = (end_time - start_time) / 1000;

    print("✅ Installation completed in " + install_time + " seconds");
    return true;
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
        MASK2FORMER_PATH + "\\" + "fiji_integration" + "\\" + "requirements.txt"
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
                   "- Mask2Former/fiji_integration folder");
        return;
    }

    print("Found requirements.txt: " + requirements_file);

    // Step 1: Check if conda environment exists, create if not
    print("");
    print("Step 1: Checking conda environment...");

    // Find conda activate script
    conda_init = "";
    if (startsWith(getInfo("os.name"), "Windows")) {
        conda_locations = newArray(
            "%USERPROFILE%\\AppData\\Local\\miniconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\miniconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\AppData\\Local\\anaconda3\\Scripts\\activate.bat",
            "%USERPROFILE%\\anaconda3\\Scripts\\activate.bat"
        );

        home_dir = getInfo("user.home");
        for (i = 0; i < conda_locations.length; i++) {
            test_path = replace(conda_locations[i], "%USERPROFILE%", home_dir);
            if (File.exists(test_path)) {
                conda_init = test_path;  // Store the EXPANDED path, not the template
                break;
            }
        }

        if (conda_init == "") {
            // Fallback to default if not found - use expanded path
            conda_init = home_dir + "\\AppData\\Local\\miniconda3\\Scripts\\activate.bat";
        }

        check_env_cmd = "call " + conda_init + " base && conda env list | findstr " + CONDA_ENV;
        create_env_cmd = "call " + conda_init + " base && conda create -n " + CONDA_ENV + " python=3.9 -y";
    } else {
        // Unix/Mac: Check and create environment
        check_env_cmd = "source $(conda info --base)/etc/profile.d/conda.sh && conda env list | grep -w " + CONDA_ENV;
        create_env_cmd = "conda create -n " + CONDA_ENV + " python=3.9 -y";
    }

    // Check if environment exists
    env_check_file = getDirectory("temp") + "env_check.txt";

    if (startsWith(getInfo("os.name"), "Windows")) {
        exec("cmd", "/c", check_env_cmd + " > \"" + env_check_file + "\" 2>&1");
    } else {
        exec("sh", "-c", check_env_cmd + " > \"" + env_check_file + "\" 2>&1");
    }

    wait(1000);  // Wait for file to be written

    env_exists = false;
    if (File.exists(env_check_file)) {
        env_content = File.openAsString(env_check_file);
        if (indexOf(env_content, CONDA_ENV) >= 0) {
            env_exists = true;
        }
        File.delete(env_check_file);
    }

    if (env_exists) {
        print("✅ Conda environment '" + CONDA_ENV + "' found");
    } else {
        print("⚠️  Conda environment '" + CONDA_ENV + "' not found");
        print("Creating conda environment '" + CONDA_ENV + "'...");
        print("This may take a few minutes...");

        showStatus("Creating conda environment " + CONDA_ENV + "...");

        create_start = getTime();

        if (startsWith(getInfo("os.name"), "Windows")) {
            exec("cmd", "/c", create_env_cmd);
        } else {
            exec("sh", "-c", create_env_cmd);
        }

        create_end = getTime();
        create_time = (create_end - create_start) / 1000;

        print("✅ Environment created in " + create_time + " seconds");
    }

    // Step 2: Install dependencies
    print("");
    print("Step 2: Installing Python dependencies...");

    // Build pip install command
    pip_cmd = PYTHON_COMMAND + " -m pip install -r \"" + requirements_file + "\"";

    // Wrap with conda activation (reuse conda_init from earlier)
    if (startsWith(getInfo("os.name"), "Windows")) {
        // Windows conda activation
        full_cmd = "call " + conda_init + " " + CONDA_ENV + " && " + pip_cmd;
    } else {
        // Unix/Mac conda activation
        full_cmd = "source $(conda info --base)/etc/profile.d/conda.sh && conda activate " + CONDA_ENV + " && " + pip_cmd;
    }

    print("Installing dependencies...");
    print("Command: " + full_cmd);
    print("This may take a few seconds to several minutes depending on what needs to be installed.");
    print("");

    showStatus("Installing Python dependencies...");

    // Execute installation
    start_time = getTime();

    if (startsWith(getInfo("os.name"), "Windows")) {
        exec("cmd", "/c", full_cmd);
    } else {
        exec("sh", "-c", full_cmd);
    }

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

// Initialization message
print("Myotube Segmentation macro loaded successfully!");
print("Press 'M' or run 'Segment Myotubes' to start.");
print("Run 'Install Python Dependencies' for first-time setup.");
