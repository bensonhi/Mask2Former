// Fiji macro to recursively process .tif/.tiff files
// Keeps the last channel (e.g., DAPI), performs Max Z Projection, and saves result in output folder preserving subfolder structure

inputDir = getDirectory("Choose Input Folder");
outputDir = getDirectory("Choose Output Folder");

processFolder(inputDir);

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
    relativePath = replace(relativePath, name, "MAX_" + name);

    // Construct final output path
    outputPath = rootOutputDir + relativePath;
    
    // Check if output file already exists
    if (File.exists(outputPath)) {
        print("Skipping (already exists): " + outputPath);
        return;
    }

    print("Processing: " + path);

    // Open the image
    open(path);

    // Get the title (name) and full path of the current image
    origTitle = getTitle();

    // Get number of channels and split channels
    Stack.getDimensions(width, height, channels, slices, frames);
    print("Image has " + channels + " channels, using channel " + channels + " (last channel)");
    run("Split Channels");
    
    // Select the last channel
    lastChannel = channels;
    selectWindow("C" + lastChannel + "-" + origTitle);
    
    // Perform Max Intensity Z Projection
    run("Z Project...", "projection=[Max Intensity]");
    projTitle = "MAX_" + origTitle;

    // Close the original split channel and the rest
    close("C" + lastChannel + "-" + origTitle);
    for (j = 1; j < lastChannel; j++) {
        chName = "C" + j + "-" + origTitle;
        if (isOpen(chName)) close(chName);
    }

    // Create output folder if needed
    outputFolder = File.getParent(outputPath);
    File.makeDirectory(outputFolder);

    // Save result
    saveAs("Tiff", outputPath);

    // Close projection
    close(projTitle);
}