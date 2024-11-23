#feature-id CosmicClarity : SetiAstro > Cosmic Clarity - Satellite Trail Removal
#feature-icon satellite.svg
#feature-info This script works with Seti Astro Cosmic Clarity program to remove satellite trails

/****************************************************************************
 *######################################################################
 *#        ___     __      ___       __                                #
 *#       / __/___/ /__   / _ | ___ / /________                        #
 *#      _\ \/ -_) _ _   / __ |(_-</ __/ __/ _ \                       #
 *#     /___/\__/\//_/  /_/ |_/___/\__/_/  \___/                       #
 *#                                                                    #
 *######################################################################
 *
 * Cosmic Clarity - Satellite Trail Removal
 * Version: V1.0
 * Author: Franklin Marek
 * Website: www.setiastro.com
 *
 * This script works with Seti Astro Cosmic Clarity satellite program to reduce noise in images.
 *
 * This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
 ******************************************************************************/

#include <pjsr/StdButton.jsh>
#include <pjsr/StdIcon.jsh>
#include <pjsr/StdCursor.jsh>
#include <pjsr/Sizer.jsh>
#include <pjsr/FrameStyle.jsh>
#include <pjsr/NumericControl.jsh>
#include <pjsr/FileMode.jsh>
#include <pjsr/DataType.jsh>
#include <pjsr/ImageOp.jsh>
#include <pjsr/SampleType.jsh>
#include <pjsr/UndoFlag.jsh>
#include <pjsr/TextAlign.jsh>
#include <pjsr/FontFamily.jsh>
#include <pjsr/ColorSpace.jsh>

#define VERSION "v1.0"

// Determine platform and appropriate command/shell setup
let CMD_EXEC, SCRIPT_EXT;
if (CoreApplication.platform == "MACOSX" || CoreApplication.platform == "macOS") {
    CMD_EXEC = "/bin/sh";
    SCRIPT_EXT = ".sh";
} else if (CoreApplication.platform == "Linux") {  // Correct platform string for Linux
    CMD_EXEC = "/bin/sh";
    SCRIPT_EXT = ".sh";
} else if (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") {
    CMD_EXEC = "cmd.exe";
    SCRIPT_EXT = ".bat";
} else {
    console.criticalln("Unsupported platform: " + CoreApplication.platform);
}


// Define platform-agnostic folder paths
let pathSeparator = (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") ? "\\" : "/";
let scriptTempDir = File.systemTempDirectory + pathSeparator + "setiastrocosmicclaritySatellite";
let setiastrosatelliteConfigFile = scriptTempDir + pathSeparator + "setiastrocosmicclaritysatellite_config.csv";

// Ensure the temp directory exists
if (!File.directoryExists(scriptTempDir)) {
    File.createDirectory(scriptTempDir);
}

// Define global parameters
var SetiAstrosatelliteParameters = {
    targetView: undefined,
    setiastrosatelliteParentFolderPath: "",
    useGPU: true,
    satelliteMode: "full", // Default mode is 'full'
    processMode: "single", // Default process mode is 'single' (can be 'batch')
    inputDir: "",          // Input directory for batch processing
    outputDir: "",         // Output directory for batch processing

    save: function() {
        Parameters.set("useGPU", this.useGPU);
        Parameters.set("setiastrosatelliteParentFolderPath", this.setiastrosatelliteParentFolderPath);
        Parameters.set("satelliteMode", this.satelliteMode);
        Parameters.set("processMode", this.processMode);
        Parameters.set("inputDir", this.inputDir);
        Parameters.set("outputDir", this.outputDir);
        this.savePathToFile();
    },

    load: function() {
        if (Parameters.has("useGPU"))
            this.useGPU = Parameters.getBoolean("useGPU");
        if (Parameters.has("setiastrosatelliteParentFolderPath"))
            this.setiastrosatelliteParentFolderPath = Parameters.getString("setiastrosatelliteParentFolderPath");
        if (Parameters.has("satelliteMode"))
            this.satelliteMode = Parameters.getString("satelliteMode");
        if (Parameters.has("processMode"))
            this.processMode = Parameters.getString("processMode");
        if (Parameters.has("inputDir"))
            this.inputDir = Parameters.getString("inputDir");
        if (Parameters.has("outputDir"))
            this.outputDir = Parameters.getString("outputDir");
        this.loadPathFromFile();
    },

    savePathToFile: function() {
        try {
            let file = new File;
            file.createForWriting(setiastrosatelliteConfigFile);
            file.outTextLn(this.setiastrosatelliteParentFolderPath);
            file.close();
        } catch (error) {
            console.warningln("Failed to save SetiAstrosatellite parent folder path: " + error.message);
        }
    },

    loadPathFromFile: function() {
        try {
            if (File.exists(setiastrosatelliteConfigFile)) {
                let file = new File;
                file.openForReading(setiastrosatelliteConfigFile);
                let lines = File.readLines(setiastrosatelliteConfigFile);
                if (lines.length > 0) {
                    this.setiastrosatelliteParentFolderPath = lines[0].trim();
                }
                file.close();
            }
        } catch (error) {
            console.warningln("Failed to load SetiAstrosatellite parent folder path: " + error.message);
        }
    }
};

function SetiAstrosatelliteDialog() {
    this.__base__ = Dialog;
    this.__base__();

    console.hide();
    SetiAstrosatelliteParameters.load();

    this.title = new Label(this);
    this.title.text = "SetiAstro Cosmic Clarity - Satellite Trail Removal " + VERSION;
    this.title.textAlignment = TextAlign_Center;

    this.description = new TextBox(this);
    this.description.readOnly = true;
    this.description.text = "This tool integrates with Seti Astro Cosmic Clarity to remove satellite trails from images. " +
                            "You can process a single image or perform batch/live folder monitoring.";
    this.description.setMinWidth(400);

    // Single Image Selection
    this.imageSelectionLabel = new Label(this);
    this.imageSelectionLabel.text = "Select Image:";
    this.imageSelectionLabel.textAlignment = TextAlign_Right | TextAlign_VertCenter;

    this.imageSelectionDropdown = new ComboBox(this);
    this.imageSelectionDropdown.editEnabled = false;

    let windows = ImageWindow.windows;
    let activeWindowId = ImageWindow.activeWindow.mainView.id;
    for (let i = 0; i < windows.length; ++i) {
        this.imageSelectionDropdown.addItem(windows[i].mainView.id);
        if (windows[i].mainView.id === activeWindowId) {
            this.imageSelectionDropdown.currentItem = i;
        }
    }

    this.imageSelectionSizer = new HorizontalSizer;
    this.imageSelectionSizer.spacing = 4;
    this.imageSelectionSizer.add(this.imageSelectionLabel);
    this.imageSelectionSizer.add(this.imageSelectionDropdown, 100);

// Batch Processing - Input Directory
this.inputDirLabel = new Label(this);
this.inputDirLabel.text = "Input Directory:";
this.inputDirLabel.textAlignment = TextAlign_Right | TextAlign_VertCenter;

// TreeBox to Display Selected Input Directory
this.inputDirTreeBox = new TreeBox(this);
this.inputDirTreeBox.numberOfColumns = 1;
this.inputDirTreeBox.headerVisible = false;
this.inputDirTreeBox.setMinSize(300, 20);
this.inputDirTreeBox.setMaxHeight(20);
this.inputDirTreeBox.clear();
if (SetiAstrosatelliteParameters.inputDir) {
    let inputNode = new TreeBoxNode(this.inputDirTreeBox);
    inputNode.setText(0, File.extractName(SetiAstrosatelliteParameters.inputDir));
}

// Button for Selecting Input Directory
this.inputDirButton = new PushButton(this);
this.inputDirButton.icon = this.scaledResource(":/icons/select-view.png");
this.inputDirButton.setScaledFixedSize(20, 20);
this.inputDirButton.toolTip = "Select Input Directory";
this.inputDirButton.onClick = function() {
    let dlg = new GetDirectoryDialog;
    dlg.caption = "Select Input Directory";
    if (dlg.execute()) {
        SetiAstrosatelliteParameters.inputDir = dlg.directory;
        SetiAstrosatelliteParameters.save();
        this.dialog.inputDirTreeBox.clear();
        let inputNode = new TreeBoxNode(this.dialog.inputDirTreeBox);
        inputNode.setText(0, File.extractName(dlg.directory));
    }
}.bind(this);

// Horizontal Sizer for Input Directory
this.inputDirSizer = new HorizontalSizer;
this.inputDirSizer.spacing = 4;
this.inputDirSizer.add(this.inputDirLabel);
this.inputDirSizer.add(this.inputDirTreeBox, 100);
this.inputDirSizer.add(this.inputDirButton);

// Batch Processing - Output Directory
this.outputDirLabel = new Label(this);
this.outputDirLabel.text = "Output Directory:";
this.outputDirLabel.textAlignment = TextAlign_Right | TextAlign_VertCenter;

// TreeBox to Display Selected Output Directory
this.outputDirTreeBox = new TreeBox(this);
this.outputDirTreeBox.numberOfColumns = 1;
this.outputDirTreeBox.headerVisible = false;
this.outputDirTreeBox.setMinSize(300, 20);
this.outputDirTreeBox.setMaxHeight(20);
this.outputDirTreeBox.clear();
if (SetiAstrosatelliteParameters.outputDir) {
    let outputNode = new TreeBoxNode(this.outputDirTreeBox);
    outputNode.setText(0, File.extractName(SetiAstrosatelliteParameters.outputDir));
}

// Button for Selecting Output Directory
this.outputDirButton = new PushButton(this);
this.outputDirButton.icon = this.scaledResource(":/icons/select-view.png");
this.outputDirButton.setScaledFixedSize(20, 20);
this.outputDirButton.toolTip = "Select Output Directory";
this.outputDirButton.onClick = function() {
    let dlg = new GetDirectoryDialog;
    dlg.caption = "Select Output Directory";
    if (dlg.execute()) {
        SetiAstrosatelliteParameters.outputDir = dlg.directory;
        SetiAstrosatelliteParameters.save();
        this.dialog.outputDirTreeBox.clear();
        let outputNode = new TreeBoxNode(this.dialog.outputDirTreeBox);
        outputNode.setText(0, File.extractName(dlg.directory));
    }
}.bind(this);

// Horizontal Sizer for Output Directory
this.outputDirSizer = new HorizontalSizer;
this.outputDirSizer.spacing = 4;
this.outputDirSizer.add(this.outputDirLabel);
this.outputDirSizer.add(this.outputDirTreeBox, 100);
this.outputDirSizer.add(this.outputDirButton);


    // Satellite Mode Radio Buttons
    this.fullsatelliteRadio = new RadioButton(this);
    this.fullsatelliteRadio.text = "Full Mode";
    this.fullsatelliteRadio.checked = SetiAstrosatelliteParameters.satelliteMode === "full";
    this.fullsatelliteRadio.onCheck = function(checked) {
        if (checked) SetiAstrosatelliteParameters.satelliteMode = "full";
    };

    this.luminancesatelliteRadio = new RadioButton(this);
    this.luminancesatelliteRadio.text = "Luminance Only";
    this.luminancesatelliteRadio.checked = SetiAstrosatelliteParameters.satelliteMode === "luminance";
    this.luminancesatelliteRadio.onCheck = function(checked) {
        if (checked) SetiAstrosatelliteParameters.satelliteMode = "luminance";
    };

    this.satelliteModeSizer = new HorizontalSizer;
    this.satelliteModeSizer.spacing = 4;
    this.satelliteModeSizer.add(this.fullsatelliteRadio);
    this.satelliteModeSizer.add(this.luminancesatelliteRadio);

// Single and Batch Checkboxes
this.singleCheckbox = new CheckBox(this);
this.singleCheckbox.text = "Process Single Image from the Dropdown";
this.singleCheckbox.checked = true; // Default to Single Image processing
this.singleCheckbox.onCheck = function(checked) {
    if (checked) {
        this.dialog.batchCheckbox.checked = false; // Uncheck the Batch checkbox
        this.dialog.imageSelectionDropdown.enabled = true; // Enable Single Image dropdown
        this.dialog.inputDirTreeBox.enabled = false; // Disable Input Directory selection
        this.dialog.inputDirButton.enabled = false;
        this.dialog.outputDirTreeBox.enabled = false; // Disable Output Directory selection
        this.dialog.outputDirButton.enabled = false;
    }
}.bind(this);

this.batchCheckbox = new CheckBox(this);
this.batchCheckbox.text = "Batch Process the Input Directory";
this.batchCheckbox.checked = false;
this.batchCheckbox.onCheck = function(checked) {
    if (checked) {
        this.dialog.singleCheckbox.checked = false; // Uncheck the Single checkbox
        this.dialog.imageSelectionDropdown.enabled = false; // Disable Single Image dropdown
        this.dialog.inputDirTreeBox.enabled = true; // Enable Input Directory selection
        this.dialog.inputDirButton.enabled = true;
        this.dialog.outputDirTreeBox.enabled = true; // Enable Output Directory selection
        this.dialog.outputDirButton.enabled = true;
    }
}.bind(this);

    // GPU Acceleration Checkbox
    this.gpuAccelerationCheckbox = new CheckBox(this);
    this.gpuAccelerationCheckbox.text = "Enable GPU Acceleration";
    this.gpuAccelerationCheckbox.checked = SetiAstrosatelliteParameters.useGPU;
    this.gpuAccelerationCheckbox.onCheck = function(checked) {
        SetiAstrosatelliteParameters.useGPU = checked;
    };

    // Wrench Icon Button for Setting the Parent Folder Path
    this.setupButton = new ToolButton(this);
    this.setupButton.icon = this.scaledResource(":/icons/wrench.png");
    this.setupButton.setScaledFixedSize(24, 24);
    this.setupButton.onClick = function() {
        let pathDialog = new GetDirectoryDialog;
        pathDialog.initialPath = SetiAstrosatelliteParameters.setiastrosatelliteParentFolderPath;
        if (pathDialog.execute()) {
            SetiAstrosatelliteParameters.setiastrosatelliteParentFolderPath = pathDialog.directory;
            SetiAstrosatelliteParameters.save();
        }
    };

    // Buttons
    this.okButton = new PushButton(this);
    this.okButton.text = "OK";
    this.okButton.onClick = () => this.ok();

    this.cancelButton = new PushButton(this);
    this.cancelButton.text = "Cancel";
    this.cancelButton.onClick = () => this.cancel();

    this.newInstanceButton = new ToolButton(this);
    this.newInstanceButton.icon = this.scaledResource(":/process-interface/new-instance.png");
    this.newInstanceButton.setScaledFixedSize(24, 24);
    this.newInstanceButton.onMousePress = function() {
        this.dialog.newInstance();
    }.bind(this);

    this.buttonsSizer = new HorizontalSizer;
    this.buttonsSizer.spacing = 6;
    this.buttonsSizer.add(this.newInstanceButton);
    this.buttonsSizer.addStretch();
    this.buttonsSizer.add(this.okButton);
    this.buttonsSizer.add(this.cancelButton);

    // Layout
    this.sizer = new VerticalSizer;
    this.sizer.margin = 6;
    this.sizer.spacing = 6;
    this.sizer.add(this.title);
    this.sizer.add(this.description);
    this.sizer.addSpacing(6);
    this.sizer.add(this.imageSelectionSizer);
    this.sizer.addSpacing(6);
    this.sizer.add(this.inputDirSizer);
    this.sizer.addSpacing(6);
    this.sizer.add(this.outputDirSizer);
    this.sizer.addSpacing(6);
    this.sizer.add(this.satelliteModeSizer);
    this.sizer.addSpacing(6);
    this.sizer.add(this.singleCheckbox);
    this.sizer.add(this.batchCheckbox);
    this.sizer.addSpacing(6);
    this.sizer.add(this.gpuAccelerationCheckbox);
    this.sizer.addSpacing(6);
    this.sizer.add(this.setupButton);
    this.sizer.addStretch();
    this.sizer.add(this.buttonsSizer);

    this.windowTitle = "SetiAstro Cosmic Clarity - Satellite Trail Removal";
    this.adjustToContents();

    // Function to synchronize UI state
this.updateUIState = function() {
    if (this.singleCheckbox.checked) {
        this.imageSelectionDropdown.enabled = true; // Enable Single Image dropdown
        this.inputDirTreeBox.enabled = false; // Disable Input Directory selection
        this.inputDirButton.enabled = false;
        this.outputDirTreeBox.enabled = false; // Disable Output Directory selection
        this.outputDirButton.enabled = false;
    } else if (this.batchCheckbox.checked) {
        this.imageSelectionDropdown.enabled = false; // Disable Single Image dropdown
        this.inputDirTreeBox.enabled = true; // Enable Input Directory selection
        this.inputDirButton.enabled = true;
        this.outputDirTreeBox.enabled = true; // Enable Output Directory selection
        this.outputDirButton.enabled = true;
    }
};

// Synchronize UI state after creation
this.updateUIState();
}
SetiAstrosatelliteDialog.prototype = new Dialog;


function saveImageAsXISF(inputFolderPath, view) {
    // Obtain the ImageWindow object from the view's main window
    let imgWindow = view.isMainView ? view.window : view.mainView.window;

    if (!imgWindow) {
        throw new Error("Image window is undefined for the specified view.");
    }

    let fileName = imgWindow.mainView.id;  // Get the main view's id as the filename
    let filePath = inputFolderPath + pathSeparator + fileName + ".xisf";

    // Set the image format to 32-bit float if not already set
    imgWindow.bitsPerSample = 32;
    imgWindow.ieeefpSampleFormat = true;

    // Save the image in XISF format
    if (!imgWindow.saveAs(filePath, false, false, false, false)) {
        throw new Error("Failed to save image as 32-bit XISF: " + filePath);
    }

    console.writeln("Image saved as 32-bit XISF: " + filePath);
}


function createBatchFile(batchFilePath, exePath, inputDir, outputDir, useGPU, satelliteMode) {
    let batchContent;

    // macOS/Linux shell script
    if (CoreApplication.platform == "MACOSX" || CoreApplication.platform == "macOS") {
        batchContent = "#!/bin/sh\n";
        batchContent += "cd \"" + exePath + "\"\n";
        batchContent += "./setiastrocosmicclarity_satellite " +
            "--input \"" + inputDir + "\" " +
            "--output \"" + outputDir + "\" " +
            "--mode " + satelliteMode + " " +
            "--batch " +
            (useGPU ? "--use-gpu" : "") + "\n"; // Add --use-gpu if enabled
    }
    // Linux shell script
    else if (CoreApplication.platform == "Linux") {
        batchContent = "#!/bin/sh\n";
        batchContent += "cd \"" + exePath + "\"\n";
        batchContent += "./setiastrocosmicclarity_satellite " +
            "--input \"" + inputDir + "\" " +
            "--output \"" + outputDir + "\" " +
            "--mode " + satelliteMode + " " +
            "--batch " +
            (useGPU ? "--use-gpu" : "") + "\n"; // Add --use-gpu if enabled
    }
    // Windows batch file
    else if (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") {
        batchContent = "@echo off\n";
        batchContent += "cd /d \"" + exePath + "\"\n";
        batchContent += "setiastrocosmicclarity_satellite.exe " +
            "--input \"" + inputDir + "\" " +
            "--output \"" + outputDir + "\" " +
            "--mode " + satelliteMode + " " +
            "--batch " +
            (useGPU ? "--use-gpu" : "") + "\n"; // Add --use-gpu if enabled
    } else {
        console.criticalln("Unsupported platform: " + CoreApplication.platform);
        return false;
    }

    // Write the batch file
    try {
        File.writeTextFile(batchFilePath, batchContent);
        console.writeln((CoreApplication.platform == "Linux" || CoreApplication.platform == "macOS") ?
            "Shell script created: " + batchFilePath :
            "Batch file created: " + batchFilePath);
    } catch (error) {
        console.criticalln("Failed to create batch/shell file: " + error.message);
        return false;
    }

    return true;
}




function processSingleImage(targetView) {
    let inputFolderPath = scriptTempDir + pathSeparator + "input";
    let outputFolderPath = scriptTempDir + pathSeparator + "output";

    // Ensure temporary input/output folders exist
    if (!File.directoryExists(inputFolderPath)) File.createDirectory(inputFolderPath);
    if (!File.directoryExists(outputFolderPath)) File.createDirectory(outputFolderPath);

    // Save the selected image to the input directory
    saveImageAsXISF(inputFolderPath, targetView);

    let batchFilePath = SetiAstrosatelliteParameters.setiastrosatelliteParentFolderPath + pathSeparator + "run_setiastrocosmicclaritysatellite" + SCRIPT_EXT;

    // Create and run the batch file (pass folder paths only)
    if (createBatchFile(batchFilePath, SetiAstrosatelliteParameters.setiastrosatelliteParentFolderPath, inputFolderPath, outputFolderPath, SetiAstrosatelliteParameters.useGPU, SetiAstrosatelliteParameters.satelliteMode)) {
        runBatchFile(batchFilePath);
    }
}


function processBatchImages() {
    let inputFolderPath = SetiAstrosatelliteParameters.inputDir;
    let outputFolderPath = SetiAstrosatelliteParameters.outputDir;

    // Check if input/output directories are set
    if (!inputFolderPath || !outputFolderPath) {
        console.criticalln("Input and output directories must be specified for batch processing.");
        return;
    }

    let batchFilePath = SetiAstrosatelliteParameters.setiastrosatelliteParentFolderPath + pathSeparator + "run_setiastrocosmicclaritysatellite" + SCRIPT_EXT;

    // Create and run the batch file
    if (createBatchFile(batchFilePath, SetiAstrosatelliteParameters.setiastrosatelliteParentFolderPath, inputFolderPath, outputFolderPath, SetiAstrosatelliteParameters.useGPU, SetiAstrosatelliteParameters.satelliteMode)) {
        runBatchFile(batchFilePath);
    }
}


// Function to run the batch file
function runBatchFile(batchFilePath, outputFolderPath, targetView = null) {
    let process = new ExternalProcess();
    try {
        // Platform-specific batch file execution
        if (CoreApplication.platform == "MACOSX" || CoreApplication.platform == "macOS" || CoreApplication.platform == "Linux") {
            if (!process.start(CMD_EXEC, [batchFilePath])) {
                console.writeln("SetiAstro Cosmic Clarity process started.");
                console.flush();
            }
        } else if (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") {
            if (!process.start(CMD_EXEC, ["/c", batchFilePath])) {
                console.writeln("SetiAstro Cosmic Clarity process started.");
                console.flush();
            }
        }
    } catch (error) {
        console.criticalln("Error starting process: " + error.message);
    }

    // For single-image mode, append file name to outputFolderPath
    let expectedOutputFile = targetView ? outputFolderPath + pathSeparator + targetView.mainView.id + "_satellited.xisf" : outputFolderPath;

    // Wait for and process the output
    if (waitForFile(expectedOutputFile, 600)) {
        if (targetView) {
            processOutputImage(expectedOutputFile, targetView); // Process single-image mode result
        }
    } else {
        console.criticalln("Output file not found within timeout.");
    }
}


function waitForFile(outputFilePath, timeoutSeconds = 600) {
   console.writeln("Waiting for file: " + outputFilePath);

    if (typeof outputFilePath !== "string" || outputFilePath.trim() === "") {
        console.criticalln("Invalid output file path: " + outputFilePath);
        return false;
    }

    let elapsedTime = 0;
    let pollingInterval = 1000;  // Poll every 1 second
    let postFindDelay = 2000;    // Delay of 2 seconds after finding the file

    while (elapsedTime < timeoutSeconds * 1000) {
        if (File.exists(outputFilePath)) {
            console.writeln("Output file found: " + outputFilePath);

            // Add a delay to ensure the file is fully written
            console.writeln("Waiting for " + (postFindDelay / 1000) + " seconds to ensure the file is completely saved.");
            msleep(postFindDelay);

            return true;
        }
        msleep(pollingInterval);
        elapsedTime += pollingInterval;
    }
    console.criticalln("Timeout waiting for file: " + outputFilePath);
    return false;
}


// Process the satellited image after waiting for it
function processOutputImage(outputFilePath, targetView) {
    if (!File.exists(outputFilePath)) {
        console.criticalln("satellited file not found: " + outputFilePath);
        return;
    }

    let satellitedWindow = ImageWindow.open(outputFilePath)[0];
    if (satellitedWindow) {
        satellitedWindow.show();

        // Now apply PixelMath to replace the original image with the reverted, satellited image
        let pixelMath = new PixelMath;
        pixelMath.expression = "iif(" + satellitedWindow.mainView.id + " == 0, $T, " + satellitedWindow.mainView.id + ")";
        pixelMath.useSingleExpression = true;
        pixelMath.createNewImage = false;
        pixelMath.executeOn(targetView.mainView);  // Replace the target view (main image) with the satellited one

        // Close the satellited image window after PixelMath operation
        satellitedWindow.forceClose();

        // Try deleting the temporary satellited file
        try {
            File.remove(outputFilePath);
            console.writeln("Deleted output file: " + outputFilePath);
        } catch (error) {
            console.warningln("Failed to delete output file: " + outputFilePath);
        }

        // Delete the input file for single mode
        let inputFilePath = scriptTempDir + pathSeparator + "input" + pathSeparator + targetView.mainView.id + ".xisf";
        deleteInputFile(inputFilePath);
    } else {
        console.criticalln("Failed to open satellited image: " + outputFilePath);
    }
}



// Function to delete the input file
function deleteInputFile(inputFilePath) {
    try {
        if (File.exists(inputFilePath)) {
            File.remove(inputFilePath);
            console.writeln("Deleted input file: " + inputFilePath);
        } else {
            console.warningln("Input file not found: " + inputFilePath);
        }
    } catch (error) {
        console.warningln("Failed to delete input file: " + inputFilePath);
    }
}


// Main execution block for running the script
let dialog = new SetiAstrosatelliteDialog();
console.show();
console.writeln("SetiAstro Cosmic Clarity - Satellite Trail Removal process started.");
console.flush();

if (dialog.execute()) {
    if (SetiAstrosatelliteParameters.processMode === "single") {
        let selectedIndex = dialog.imageSelectionDropdown.currentItem;
        let selectedView = ImageWindow.windows[selectedIndex];

        if (!selectedView) {
            console.criticalln("Please select an image.");
        } else {
            let inputFolderPath = scriptTempDir + pathSeparator + "input";
            let outputFolderPath = scriptTempDir + pathSeparator + "output";

            // Ensure temporary folders exist
            if (!File.directoryExists(inputFolderPath)) File.createDirectory(inputFolderPath);
            if (!File.directoryExists(outputFolderPath)) File.createDirectory(outputFolderPath);

            // Save the selected image to the input folder
            saveImageAsXISF(inputFolderPath, selectedView);

            // Generate batch file path
            let batchFilePath = SetiAstrosatelliteParameters.setiastrosatelliteParentFolderPath +
                pathSeparator + "run_setiastrocosmicclaritysatellite" + SCRIPT_EXT;

            // Create and run the batch file
            if (createBatchFile(batchFilePath, SetiAstrosatelliteParameters.setiastrosatelliteParentFolderPath, inputFolderPath, outputFolderPath, SetiAstrosatelliteParameters.useGPU, SetiAstrosatelliteParameters.satelliteMode)) {
                runBatchFile(batchFilePath, outputFolderPath, selectedView); // Pass targetView for single mode
            }
        }
    } else if (SetiAstrosatelliteParameters.processMode === "batch") {
        let inputDir = SetiAstrosatelliteParameters.inputDir;
        let outputDir = SetiAstrosatelliteParameters.outputDir;

        if (!inputDir || !outputDir) {
            console.criticalln("Input and output directories must be specified for batch processing.");
        } else {
            let batchFilePath = SetiAstrosatelliteParameters.setiastrosatelliteParentFolderPath +
                pathSeparator + "run_setiastrocosmicclaritysatellite" + SCRIPT_EXT;

            if (createBatchFile(batchFilePath, SetiAstrosatelliteParameters.setiastrosatelliteParentFolderPath, inputDir, outputDir, SetiAstrosatelliteParameters.useGPU, SetiAstrosatelliteParameters.satelliteMode)) {
                runBatchFile(batchFilePath, outputDir); // No targetView for batch mode
            }
        }
    }
}
