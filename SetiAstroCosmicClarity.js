#feature-id CosmicClarity : SetiAstro > Cosmic Clarity - Sharpen
#feature-icon  cosmicclaritysharpen.svg
#feature-info This script works with Seti Astro Cosmic Clarity program to sharpen images

/******************************************************************************
 *######################################################################
 *#        ___     __      ___       __                                #
 *#       / __/___/ /__   / _ | ___ / /________                        #
 *#      _\ \/ -_) _ _   / __ |(_-</ __/ __/ _ \                       #
 *#     /___/\__/_//_/  /_/ |_/___/\__/_/  \___/                       #
 *#                                                                    #
 *######################################################################
 *
 * Cosmic Clarity
 * Version: V3.2.1
 * Author: Franklin Marek
 * Website: www.setiastro.com
 *
 * This script works with Seti Astro Cosmic Clarity program to sharpen images
 *
 *
 * This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
 * To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
 *
 * You are free to:
 * 1. Share — copy and redistribute the material in any medium or format
 * 2. Adapt — remix, transform, and build upon the material
 *
 * Under the following terms:
 * 1. Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
 * 2. NonCommercial — You may not use the material for commercial purposes.
 *
 * @license CC BY-NC 4.0 (http://creativecommons.org/licenses/by-nc/4.0/)
 *
 * COPYRIGHT © 2024 Franklin Marek. ALL RIGHTS RESERVED.
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

// Platform detection and appropriate shell or command setup
let CMD_EXEC, SCRIPT_EXT;
if (CoreApplication.platform == "MACOSX" || CoreApplication.platform == "macOS" || CoreApplication.platform == "Linux") {
    CMD_EXEC = "/bin/sh"; // For macOS and Linux
    SCRIPT_EXT = ".sh";
} else if (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") {
    CMD_EXEC = "cmd.exe"; // For Windows
    SCRIPT_EXT = ".bat";
} else {
    console.criticalln("Unsupported platform: " + CoreApplication.platform);
}

// Define platform-agnostic folder paths
let pathSeparator = (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") ? "\\" : "/";
let scriptTempDir = File.systemTempDirectory + pathSeparator + "SetiAstroCosmicClarity";
let setiAstroSharpConfigFile = scriptTempDir + pathSeparator + "setiastrocosmicclarity_config.csv";

// Ensure the temp directory exists
if (!File.directoryExists(scriptTempDir)) {
    File.createDirectory(scriptTempDir);
}

#define VERSION "v3.2.1"

// Define global parameters
var SetiAstroSharpParameters = {
    targetView: undefined,
    sharpeningMode: "Both",  // Default mode
    stellarAmount: 0.9,      // Default stellar sharpening amount
    nonStellarStrength: 3.0, // Default non-stellar feature size
    nonStellarAmount: 0.5,   // Default non-stellar sharpening amount
    sharpenChannelsSeparately: false,  // Default RGB sharpening
    setiAstroSharpParentFolderPath: "",
    useGPU: true,
    configFilePath: scriptTempDir + pathSeparator + "setiastrocosmicclarity_config.csv",

    // Save current parameters to script instance
    save: function() {
        Parameters.set("useGPU", this.useGPU);  // Save the GPU acceleration state
        Parameters.set("setiAstroSharpParentFolderPath", this.setiAstroSharpParentFolderPath);
        Parameters.set("sharpeningMode", this.sharpeningMode);
        Parameters.set("stellarAmount", this.stellarAmount);
        Parameters.set("nonStellarStrength", this.nonStellarStrength);
        Parameters.set("nonStellarAmount", this.nonStellarAmount);
        Parameters.set("sharpenChannelsSeparately", this.sharpenChannelsSeparately);
        this.savePathToFile();
    },

    // Load saved parameters from script instance
    load: function() {
        if (Parameters.has("useGPU"))
            this.useGPU = Parameters.getBoolean("useGPU");
        if (Parameters.has("setiAstroSharpParentFolderPath"))
            this.setiAstroSharpParentFolderPath = Parameters.getString("setiAstroSharpParentFolderPath");
        if (Parameters.has("sharpeningMode"))
            this.sharpeningMode = Parameters.getString("sharpeningMode");
        if (Parameters.has("stellarAmount"))
            this.stellarAmount = Parameters.getReal("stellarAmount");
        if (Parameters.has("nonStellarStrength"))
            this.nonStellarStrength = Parameters.getReal("nonStellarStrength");
        if (Parameters.has("nonStellarAmount"))
            this.nonStellarAmount = Parameters.getReal("nonStellarAmount");
        if (Parameters.has("sharpenChannelsSeparately"))
            this.sharpenChannelsSeparately = Parameters.getBoolean("sharpenChannelsSeparately");
        this.loadPathFromFile();
    },

    // Save the SetiAstroSharp parent folder path to a CSV file
    savePathToFile: function() {
        try {
            let file = new File;
            file.createForWriting(this.configFilePath);
            file.outTextLn(this.setiAstroSharpParentFolderPath);
            file.close();
        } catch (error) {
            console.warningln("Failed to save SetiAstroSharp parent folder path: " + error.message);
        }
    },

    // Load the SetiAstroSharp parent folder path from a CSV file
    loadPathFromFile: function() {
        try {
            if (File.exists(this.configFilePath)) {
                let file = new File;
                file.openForReading(this.configFilePath);
                let lines = File.readLines(this.configFilePath);
                if (lines.length > 0) {
                    this.setiAstroSharpParentFolderPath = lines[0].trim();
                }
                file.close();
            }
        } catch (error) {
            console.warningln("Failed to load SetiAstroSharp parent folder path: " + error.message);
        }
    }
};

// Main dialog for SetiAstroCosmicClarity
function SetiAstroSharpDialog() {
    this.__base__ = Dialog;
    this.__base__();

    // Load saved parameters
    SetiAstroSharpParameters.load();

    // Title and description
    this.title = new Label(this);
    this.title.text = "SetiAstroCosmicClarity " + VERSION;
    this.title.textAlignment = TextAlign_Center;

    this.description = new TextBox(this);
    this.description.readOnly = true;
    this.description.text = "This script integrates with SetiAstroCosmicClarity for stellar and non-stellar sharpening.\n" +
                            "It saves the current image, runs the SetiAstroCosmicClarity tool, and replaces " +
                            "the image with the sharpened version.";
    this.description.setMinWidth(400);

    // Image Selection Dropdown
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
            this.imageSelectionDropdown.currentItem = i; // Default to active image
        }
    }

    this.imageSelectionSizer = new HorizontalSizer;
    this.imageSelectionSizer.spacing = 4;
    this.imageSelectionSizer.add(this.imageSelectionLabel);
    this.imageSelectionSizer.add(this.imageSelectionDropdown, 100);

    // Radio buttons for mode selection
    this.sharpeningModeGroup = new RadioButton(this);
    this.sharpeningModeGroup.sizer = new HorizontalSizer;
    this.sharpeningModeGroup.sizer.spacing = 3;

    this.stellarRadioButton = new RadioButton(this);
    this.stellarRadioButton.text = "Stellar";
    this.stellarRadioButton.checked = SetiAstroSharpParameters.sharpeningMode === "Stellar Only";
    this.stellarRadioButton.onClick = () => this.updateVisibility();

    this.nonStellarRadioButton = new RadioButton(this);
    this.nonStellarRadioButton.text = "Non-Stellar";
    this.nonStellarRadioButton.checked = SetiAstroSharpParameters.sharpeningMode === "Non-Stellar Only";
    this.nonStellarRadioButton.onClick = () => this.updateVisibility();

    this.bothRadioButton = new RadioButton(this);
    this.bothRadioButton.text = "Both";
    this.bothRadioButton.checked = SetiAstroSharpParameters.sharpeningMode === "Both";
    this.bothRadioButton.onClick = () => this.updateVisibility();

    this.sharpeningModeGroup.sizer.add(this.stellarRadioButton);
    this.sharpeningModeGroup.sizer.add(this.nonStellarRadioButton);
    this.sharpeningModeGroup.sizer.add(this.bothRadioButton);

    // Sharpen RGB Channels Separately checkbox
    this.sharpenChannelsCheckbox = new CheckBox(this);
    this.sharpenChannelsCheckbox.text = "Sharpen RGB Channels Separately";
    this.sharpenChannelsCheckbox.checked = SetiAstroSharpParameters.sharpenChannelsSeparately;
    this.sharpenChannelsCheckbox.onCheck = function(checked) {
        SetiAstroSharpParameters.sharpenChannelsSeparately = checked;
    };

    // Stellar Amount slider
    this.stellarAmountSlider = new NumericControl(this);
    this.stellarAmountSlider.label.text = "Stellar Amount:";
    this.stellarAmountSlider.setRange(0, 1);
    this.stellarAmountSlider.setValue(SetiAstroSharpParameters.stellarAmount);
    this.stellarAmountSlider.setPrecision(2);
    this.stellarAmountSlider.toolTip = "Applies the amount of sharpening Cosmic Clarity determines for Stars. 0 being no sharpening and 1 being full sharpening as determined by Cosmic Clarity.";
    this.stellarAmountSlider.onValueUpdated = function(value) {
    SetiAstroSharpParameters.stellarAmount = value;
    };

    // Non-Stellar Feature Size slider
    this.nonStellarStrengthSlider = new NumericControl(this);
    this.nonStellarStrengthSlider.label.text = "Non-Stellar Feature Size (PSF):";
    this.nonStellarStrengthSlider.setRange(1, 8);
    this.nonStellarStrengthSlider.setValue(SetiAstroSharpParameters.nonStellarStrength);
    this.nonStellarStrengthSlider.setPrecision(2);
    this.nonStellarStrengthSlider.toolTip = "Sharpening for non-stellar structures (feature size).";
    this.nonStellarStrengthSlider.onValueUpdated = function(value) {
        SetiAstroSharpParameters.nonStellarStrength = value;
    };

    // Non-Stellar Amount slider
    this.nonStellarAmountSlider = new NumericControl(this);
    this.nonStellarAmountSlider.label.text = "Non-Stellar Amount:";
    this.nonStellarAmountSlider.setRange(0, 1);
    this.nonStellarAmountSlider.setValue(SetiAstroSharpParameters.nonStellarAmount);
    this.nonStellarAmountSlider.setPrecision(2);
    this.nonStellarAmountSlider.toolTip = "Applies sharpening for non-stellar structures (0 to 1).";
    this.nonStellarAmountSlider.onValueUpdated = function(value) {
        SetiAstroSharpParameters.nonStellarAmount = value;
    };


    // Wrench Icon Button for setting the SetiAstroSharp parent folder path
    this.setupButton = new ToolButton(this);
    this.setupButton.icon = this.scaledResource(":/icons/wrench.png");
    this.setupButton.setScaledFixedSize(24, 24);
    this.setupButton.onClick = function() {
        let pathDialog = new GetDirectoryDialog;
        pathDialog.initialPath = SetiAstroSharpParameters.setiAstroSharpParentFolderPath;
        if (pathDialog.execute()) {
            SetiAstroSharpParameters.setiAstroSharpParentFolderPath = pathDialog.directory;
            SetiAstroSharpParameters.save();
        }
    };

// OK and Cancel buttons
this.okButton = new PushButton(this);
this.okButton.text = "OK";

// Modified onClick handler for OK button
this.okButton.onClick = () => {
    // Now proceed with the OK functionality after the user dismisses the message
    this.ok();
};


    this.cancelButton = new PushButton(this);
    this.cancelButton.text = "Cancel";
    this.cancelButton.onClick = () => this.cancel();

        // New Instance button
    this.newInstanceButton = new ToolButton(this);
    this.newInstanceButton.icon = this.scaledResource(":/process-interface/new-instance.png");
    this.newInstanceButton.setScaledFixedSize(24, 24);
    this.newInstanceButton.toolTip = "Save a new instance of this script";
    this.newInstanceButton.onMousePress = function() {
    this.dialog.newInstance();
      }.bind(this);

        this.buttonsSizer = new HorizontalSizer;
    this.buttonsSizer.spacing = 6;
    this.buttonsSizer.add(this.newInstanceButton);
    this.buttonsSizer.addStretch();
    this.buttonsSizer.add(this.okButton);
    this.buttonsSizer.add(this.cancelButton);
    this.buttonsSizer.addStretch();

    // Create a horizontal sizer for the checkbox and center it
this.sharpenChannelsSizer = new HorizontalSizer;
this.sharpenChannelsSizer.spacing = 6;  // Optional spacing between items
this.sharpenChannelsSizer.addStretch();  // Add stretch to push the checkbox to the center
this.sharpenChannelsSizer.add(this.sharpenChannelsCheckbox);  // Add the checkbox to the sizer
this.sharpenChannelsSizer.addStretch();  // Add stretch to keep it centered


    // Layout
    this.sizer = new VerticalSizer;
    this.sizer.margin = 6;
    this.sizer.spacing = 6;
    this.sizer.addStretch();
    this.sizer.add(this.title);
    this.sizer.add(this.description);
    this.sizer.addStretch();
    this.sizer.add(this.imageSelectionSizer);
    this.sizer.spacing = 6;
    this.sizer.add(this.sharpeningModeGroup);
    this.sizer.add(this.sharpenChannelsSizer);
    this.sizer.spacing = 6;
    this.sizer.add(this.stellarAmountSlider);
    this.sizer.spacing = 6;
    this.sizer.add(this.nonStellarStrengthSlider);
    this.sizer.add(this.nonStellarAmountSlider);
    this.sizer.addStretch();


    this.gpuAccelerationCheckbox = new CheckBox(this);
    this.gpuAccelerationCheckbox.text = "Enable GPU Acceleration";
    this.gpuAccelerationCheckbox.checked = true;  // Default to enabled
this.gpuAccelerationCheckbox.onCheck = function(checked) {
    SetiAstroSharpParameters.useGPU = checked;  // Change `enableGPU` to `useGPU`
};

    this.sizer.add(this.gpuAccelerationCheckbox);


    this.sizer.add(this.setupButton);
    this.sizer.addSpacing(12);
    this.sizer.add(this.buttonsSizer);

    this.windowTitle = "SetiAstroCosmicClarity Script";
    this.adjustToContents();

    // Initially update visibility based on the selected mode
    this.updateVisibility();
}
SetiAstroSharpDialog.prototype = new Dialog;

SetiAstroSharpDialog.prototype.updateVisibility = function() {
    let stellarMode = this.stellarRadioButton.checked;
    let nonStellarMode = this.nonStellarRadioButton.checked;
    let bothMode = this.bothRadioButton.checked;

    // Stellar Amount slider is visible for Stellar and Both modes
    this.stellarAmountSlider.visible = stellarMode || bothMode;

    // Non-Stellar Strength and Non-Stellar Amount sliders are visible for Non-Stellar and Both modes
    this.nonStellarStrengthSlider.visible = nonStellarMode || bothMode;
    this.nonStellarAmountSlider.visible = nonStellarMode || bothMode;

    // Sharpen RGB Channels Separately checkbox should be visible in all modes
    this.sharpenChannelsCheckbox.visible = true;

    // Update the global parameters based on the selected mode
    SetiAstroSharpParameters.sharpeningMode = stellarMode ? "Stellar Only" :
                                               nonStellarMode ? "Non-Stellar Only" : "Both";
};


// Function to save the image as a 32-bit TIFF file
function saveImageAsTiff(inputFolderPath, view) {
    let fileName = view.mainView.id;
    let filePath = inputFolderPath + pathSeparator + fileName + ".tiff";

    let F = new FileFormat("TIFF", false, true);
    if (F.isNull)
        throw new Error("TIFF format not supported");

    let f = new FileFormatInstance(F);
    if (f.isNull)
        throw new Error("Unable to create FileFormatInstance for TIFF");

    let description = new ImageDescription();
    description.bitsPerSample = 32;
    description.ieeefpSampleFormat = true;

    if (!f.create(filePath, "compression=none")) {
        throw new Error("Unable to create file: " + filePath);
    }

    if (!f.setOptions(description)) {
        throw new Error("Unable to set image options for 32-bit IEEE floating point.");
    }

    let img = view.mainView.image;
    if (!f.writeImage(img)) {
        throw new Error("Failed to write image: " + filePath);
    }

    f.close();
    console.writeln("Image saved as 32-bit TIFF: " + filePath);
    return filePath;
}

// Create batch file to run the sharpen process
function createBatchFile(batchFilePath, exePath, sharpeningMode, stellarAmount, nonStellarStrength, nonStellarAmount, useGPU, sharpenChannelsSeparately) {
    // Ensure that stellarAmount, nonStellarStrength, and nonStellarAmount are valid numbers
    stellarAmount = parseFloat(stellarAmount);
    nonStellarStrength = parseFloat(nonStellarStrength);
    nonStellarAmount = parseFloat(nonStellarAmount);

    let batchContent;

// macOS shell script
if (CoreApplication.platform == "MACOSX" || CoreApplication.platform == "macOS") {
    batchContent = "#!/bin/sh\n";
    batchContent += "cd \"" + exePath + "\"\n";
    batchContent += "osascript -e 'tell application \"Terminal\" to do script \"cd '" + exePath.replace(/"/g, "\\\"") + "'; ./setiastrocosmicclaritymac " +
                    "--sharpening_mode " + (sharpeningMode.replace(/ /g, "\\\\ ")) + " " +  // Escape spaces with double backslashes
                    "--stellar_amount " + stellarAmount.toFixed(2) + " " +
                    "--nonstellar_strength " + nonStellarStrength.toFixed(2) + " " +
                    "--nonstellar_amount " + nonStellarAmount.toFixed(2) + " " +
                    (sharpenChannelsSeparately ? "--sharpen_channels_separately " : "") +
                    (useGPU ? "" : "--disable_gpu") + "; exec bash\"'\n";  // Consistent with denoise approach
} else if (CoreApplication.platform == "Linux") { // Linux shell script
    batchContent = "#!/bin/sh\n";
    batchContent += "cd \"" + exePath + "\"\n";
    batchContent += "gnome-terminal -- bash -c '" +  // Correctly open terminal and run bash
                    "./SetiAstroCosmicClarity " +  // Linux executable
                    "--sharpening_mode \"" + sharpeningMode + "\" " +
                    "--stellar_amount " + stellarAmount.toFixed(2) + " " +
                    "--nonstellar_strength " + nonStellarStrength.toFixed(2) + " " +
                    "--nonstellar_amount " + nonStellarAmount.toFixed(2) + " " +
                    (sharpenChannelsSeparately ? "--sharpen_channels_separately " : "") +  // Add this flag if true
                    (useGPU ? "" : "--disable_gpu") + "; exec bash'\n";  // Add exec bash to keep terminal open
} else if (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") {
    batchContent = "@echo off\n";
    batchContent += "cd /d \"" + exePath + "\"\n";
    batchContent += "start setiastrocosmicclarity.exe " +
                    "--sharpening_mode \"" + sharpeningMode + "\" " +
                    "--stellar_amount " + stellarAmount.toFixed(2) + " " +
                    "--nonstellar_strength " + nonStellarStrength.toFixed(2) + " " +
                    "--nonstellar_amount " + nonStellarAmount.toFixed(2) + " " +
                    (sharpenChannelsSeparately ? "--sharpen_channels_separately " : "") +  // Add this flag if true
                    (useGPU ? "" : "--disable_gpu") + "\n";
} else {
    console.criticalln("Unsupported platform: " + CoreApplication.platform);
    return false;
}

// Write the script to the batch file
try {
    File.writeTextFile(batchFilePath, batchContent);
    console.writeln((CoreApplication.platform == "Linux") ? "Shell script created: " + batchFilePath : "Batch file created: " + batchFilePath);
} catch (error) {
    console.criticalln("Failed to create batch/shell file: " + error.message);
    return false;
}

return true;
}


function processOutputImage(outputFilePath, targetView) {
    if (!File.exists(outputFilePath)) {
        console.criticalln("Sharpened file not found: " + outputFilePath);
        return;
    }

    // Use the existing file path, preferring .tif over .tiff if both exist
    let finalOutputFilePath = File.exists(outputFilePath) ? outputFilePath : outputFilePathTiff;

    let sharpenedWindow = ImageWindow.open(finalOutputFilePath)[0];
    if (sharpenedWindow) {
        sharpenedWindow.show();

        // Now apply the PixelMath expression on the targetView, merging the sharpened image into the original one
        let pixelMath = new PixelMath;
        pixelMath.expression = "iif(" + sharpenedWindow.mainView.id + " == 0, $T, " + sharpenedWindow.mainView.id + ")";
        pixelMath.useSingleExpression = true;
        pixelMath.createNewImage = false;
        pixelMath.executeOn(targetView.mainView); // Apply on the target (original) image

        // Close the sharpened temporary window after processing
        sharpenedWindow.forceClose();

        // Delete the sharpened file after loading and applying
        try {
            File.remove(finalOutputFilePath);
            console.writeln("Deleted sharpened file: " + finalOutputFilePath);
        } catch (error) {
            console.warningln("Failed to delete sharpened file: " + finalOutputFilePath);
        }
    } else {
        console.criticalln("Failed to open sharpened image: " + finalOutputFilePath);
    }
}

function msleep(milliseconds) {
    let start = Date.now();
    while (Date.now() - start < milliseconds) {
        // Busy wait for the specified number of milliseconds
    }
}

function waitForFile(outputFilePath, timeoutSeconds = 60) {
    let elapsedSeconds = 0;
    let pollingInterval = 1000;  // Check every 1 second (1000 ms)
    let timeoutMillis = timeoutSeconds * 1000;

    while (!File.exists(outputFilePath)) {
        if (elapsedSeconds >= timeoutMillis) {
            console.criticalln("Timeout waiting for file: " + outputFilePath);
            return false;
        }

        msleep(pollingInterval);
        elapsedSeconds += pollingInterval;
    }

    console.writeln("File found: " + outputFilePath);
    return true;
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

// Dialog execution
let dialog = new SetiAstroSharpDialog();
                      console.show();
                        Console.criticalln("   ____    __  _   ___       __         \n  / __/__ / /_(_) / _ | ___ / /_______ ");
    Console.warningln(" _\\ \\/ -_) __/ / / __ |(_-</ __/ __/ _ \\ \n/___/\\__/\\__/_/ /_/ |_/__/\\__/__/ \\___/ \n                                         ");

                    console.flush();
// Main execution block for running the script
let dialog = new SetiAstroSharpDialog();

console.writeln("SetiAstroCosmicClarity process started.");
console.flush();

if (dialog.execute()) {
    let selectedIndex = dialog.imageSelectionDropdown.currentItem;
    let selectedView = ImageWindow.windows[selectedIndex];

    if (!selectedView) {
        console.criticalln("Please select an image.");
    } else {
        let inputFolderPath = SetiAstroSharpParameters.setiAstroSharpParentFolderPath + pathSeparator + "input";
        let outputFolderPath = SetiAstroSharpParameters.setiAstroSharpParentFolderPath + pathSeparator + "output";
        let outputFileName = selectedView.mainView.id + "_sharpened.tiff";
        let outputFilePath = outputFolderPath + pathSeparator + outputFileName;

        let inputFilePath = saveImageAsTiff(inputFolderPath, selectedView);
        let batchFilePath = SetiAstroSharpParameters.setiAstroSharpParentFolderPath + pathSeparator + "run_setiastrocosmicclarity" + SCRIPT_EXT;

	if (createBatchFile(
   	 batchFilePath,
 	   SetiAstroSharpParameters.setiAstroSharpParentFolderPath,
 	   SetiAstroSharpParameters.sharpeningMode,
 	   SetiAstroSharpParameters.stellarAmount,
 	   SetiAstroSharpParameters.nonStellarStrength,
	   SetiAstroSharpParameters.nonStellarAmount,
	   SetiAstroSharpParameters.useGPU,
	   SetiAstroSharpParameters.sharpenChannelsSeparately)) {
            let process = new ExternalProcess;
            try {
                if (CoreApplication.platform == "MACOSX" || CoreApplication.platform == "macOS" || CoreApplication.platform == "Linux") {
                    if (!process.start(CMD_EXEC, [batchFilePath])) {
                        console.writeln("SetiAstroCosmicClarity started.");
                        console.flush();
                    }
                } else if (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") {
                    if (!process.start(CMD_EXEC, ["/c", batchFilePath])) {
                        console.writeln("SetiAstroCosmicClarity started.");
                        console.flush();
                    }
                }
            } catch (error) {
                console.criticalln("Error starting process: " + error.message);
            }

            // Wait for the output file and process it
            if (waitForFile(outputFilePath, 600)) {
                processOutputImage(outputFilePath, selectedView);
                deleteInputFile(inputFilePath);
            } else {
                console.criticalln("Output file not found within timeout.");
            }
        }
    }
}
