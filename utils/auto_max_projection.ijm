// Fiji macro to recursively process .tif/.tiff files
// Processes both last channel (green) and second last channel (grey), performs Max Z Projection, and saves results in output folder preserving subfolder structure

inputDir = getDirectory("Choose Input Folder");
outputDir = getDirectory("Choose Output Folder");

processFolder(inputDir);

function createDirectoryRecursively(path) {
    parent = File.getParent(path);
    if (parent != "" && !File.exists(parent)) {
        createDirectoryRecursively(parent);
    }
    if (!File.exists(path)) {
        File.makeDirectory(path);
    }
}

function processFolder(folder) {
    list = getFileList(folder);
    for (i = 0; i < list.length; i++) {
        path = folder + list[i];
        if (File.isDirectory(path)) {
            processFolder(path);
        } else if (endsWith(path, ".tif") || endsWith(path, ".tiff")) {
            processImage(path, inputDir, outputDir);
        }
    }
}

function processImage(path, rootInputDir, rootOutputDir) {
    // Extract file name and relative path first
    slash = File.separator;
    name = substring(path, lastIndexOf(path, slash) + 1);
    relativePath = replace(path, rootInputDir, "");
    
    // Create base name without extension
    baseName = substring(name, 0, lastIndexOf(name, "."));
    extension = substring(name, lastIndexOf(name, "."));
    
    // Construct output paths for both channels
    greyRelativePath = replace(relativePath, name, "MAX_" + baseName + "_grey" + extension);
    greenRelativePath = replace(relativePath, name, "MAX_" + baseName + "_green" + extension);
    greyOutputPath = rootOutputDir + greyRelativePath;
    greenOutputPath = rootOutputDir + greenRelativePath;
    
    // Check if both output files already exist
    if (File.exists(greyOutputPath) && File.exists(greenOutputPath)) {
        print("Skipping (both outputs already exist): " + path);
        return;
    }

    print("Processing: " + path);

    // Open the image
    open(path);

    // Get the title (name) and full path of the current image
    origTitle = getTitle();

    // Get number of channels
    Stack.getDimensions(width, height, channels, slices, frames);
    print("Image has " + channels + " channels");
    
    if (channels == 1) {
        // Single channel image - save as both grey and green
        print("Single channel image - processing as both grey and green");
        
        // Process for grey output
        if (!File.exists(greyOutputPath)) {
            run("Z Project...", "projection=[Max Intensity]");
            projTitle = "MAX_" + origTitle;
            
            // Create output folder and save grey
            greyOutputFolder = File.getParent(greyOutputPath);
            createDirectoryRecursively(greyOutputFolder);
            saveAs("Tiff", greyOutputPath);
            close(projTitle);
        }
        
        // Process for green output
        if (!File.exists(greenOutputPath)) {
            selectWindow(origTitle);
            run("Z Project...", "projection=[Max Intensity]");
            projTitle = "MAX_" + origTitle;
            
            // Create output folder and save green
            greenOutputFolder = File.getParent(greenOutputPath);
            createDirectoryRecursively(greenOutputFolder);
            saveAs("Tiff", greenOutputPath);
            close(projTitle);
        }
        
        // Close the original image
        close(origTitle);
        
    } else {
        // Multi-channel image - split channels
        run("Split Channels");
        
        // Find grey and green channels by analyzing LUT colors
        greyChannel = -1;
        greenChannel = -1;
        
        for (ch = 1; ch <= channels; ch++) {
            selectWindow("C" + ch + "-" + origTitle);
            getLut(reds, greens, blues);
            
            // More precise grey detection: check if it's a proper grayscale LUT
            isGrey = true;
            greyCount = 0;
            for (i = 1; i < 255; i++) { // Skip extremes (0 and 255)
                if (reds[i] == greens[i] && greens[i] == blues[i]) {
                    greyCount++;
                } else {
                    isGrey = false;
                    break;
                }
            }
            // Only consider it grey if most values are truly grayscale
            isGrey = isGrey && (greyCount > 200);
            
            // More precise green detection
            isGreen = false;
            if (!isGrey) {
                // Check if LUT is predominantly green
                greenDominant = 0;
                totalChecked = 0;
                for (i = 50; i < 205; i++) { // Check meaningful range
                    totalChecked++;
                    // Green should be significantly higher than red and blue
                    if (greens[i] > reds[i] + 50 && greens[i] > blues[i] + 50) {
                        greenDominant++;
                    }
                }
                // At least 80% of values should show green dominance
                if (greenDominant > totalChecked * 0.8) {
                    isGreen = true;
                }
            }
            
            print("Channel " + ch + ": Grey=" + isGrey + ", Green=" + isGreen);
            
            if (isGrey && greyChannel == -1) {
                greyChannel = ch;
                print("Identified grey channel: " + ch);
            }
            if (isGreen && greenChannel == -1) {
                greenChannel = ch;
                print("Identified green channel: " + ch);
            }
        }
        
        // Only process channels that were actually identified
        if (greyChannel == -1) {
            print("No grey channel found - skipping grey output");
        }
        if (greenChannel == -1) {
            print("No green channel found - skipping green output");
        }
        
        // Process grey channel
        if (!File.exists(greyOutputPath) && greyChannel != -1) {
            print("Processing channel " + greyChannel + " for grey output");
            print("Grey output path: " + greyOutputPath);
            selectWindow("C" + greyChannel + "-" + origTitle);
            run("Z Project...", "projection=[Max Intensity]");
            greyProjTitle = "MAX_" + origTitle;
            
            // Create output folder and save grey
            greyOutputFolder = File.getParent(greyOutputPath);
            createDirectoryRecursively(greyOutputFolder);
            print("Saving grey to: " + greyOutputPath);
            saveAs("Tiff", greyOutputPath);
            print("Grey saved successfully");
            close(greyProjTitle);
        } else {
            print("Skipping grey: exists=" + File.exists(greyOutputPath) + ", channel=" + greyChannel);
        }
        
        // Process green channel
        if (!File.exists(greenOutputPath) && greenChannel != -1) {
            print("Processing channel " + greenChannel + " for green output");
            print("Green output path: " + greenOutputPath);
            selectWindow("C" + greenChannel + "-" + origTitle);
            run("Z Project...", "projection=[Max Intensity]");
            greenProjTitle = "MAX_" + origTitle;
            
            // Create output folder and save green
            greenOutputFolder = File.getParent(greenOutputPath);
            createDirectoryRecursively(greenOutputFolder);
            print("Saving green to: " + greenOutputPath);
            saveAs("Tiff", greenOutputPath);
            print("Green saved successfully");
            close(greenProjTitle);
        } else {
            print("Skipping green: exists=" + File.exists(greenOutputPath) + ", channel=" + greenChannel);
        }
        
        // Close all remaining split channels
        for (j = 1; j <= channels; j++) {
            chName = "C" + j + "-" + origTitle;
            if (isOpen(chName)) close(chName);
        }
    }
}