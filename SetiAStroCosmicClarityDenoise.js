#feature-id CosmicClarity : SetiAstro > Cosmic Clarity - Denoise
#feature-info This script works with Seti Astro Cosmic Clarity program to denoise images

/******************************************************************************
 *######################################################################
 *#        ___     __      ___       __                                #
 *#       / __/___/ /__   / _ | ___ / /________                        #
 *#      _\ \/ -_) _ _   / __ |(_-</ __/ __/ _ \                       #
 *#     /___/\__/_//_/  /_/ |_/___/\__/_/  \___/                       #
 *#                                                                    #
 *######################################################################
 *
 * Cosmic Clarity - Denoise
 * Version: V1.0
 * Author: Franklin Marek
 * Website: www.setiastro.com
 *
 * This script works with Seti Astro Cosmic Clarity Denoise program to reduce noise in images.
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

let CMD_EXEC, SCRIPT_EXT;
if (CoreApplication.platform == "MACOSX" || CoreApplication.platform == "macOS" || CoreApplication.platform == "LINUX") {
    CMD_EXEC = "/bin/sh";
    SCRIPT_EXT = ".sh";
} else if (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") {
    CMD_EXEC = "cmd.exe";
    SCRIPT_EXT = ".bat";
}

let pathSeparator = (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") ? "\\" : "/";
let scriptTempDir = File.systemTempDirectory + pathSeparator + "SetiAstroCosmicClarityDenoise";
let setiAstroDenoiseConfigFile = scriptTempDir + pathSeparator + "setiastrocosmicclaritydenoise_config.csv";

if (!File.directoryExists(scriptTempDir)) {
    File.createDirectory(scriptTempDir);
}

var SetiAstroDenoiseParameters = {
    targetView: undefined,
    isLinear: false,
    denoiseStrength: 4.0,
    setiAstroDenoiseParentFolderPath: "",
    useGPU: true,

    configFilePath: setiAstroDenoiseConfigFile,

    save: function() {
        Parameters.set("isLinear", this.isLinear);
        Parameters.set("useGPU", this.useGPU);
        Parameters.set("setiAstroDenoiseParentFolderPath", this.setiAstroDenoiseParentFolderPath);
        Parameters.set("denoiseStrength", this.denoiseStrength);
        this.savePathToFile();
    },

    load: function() {
        if (Parameters.has("isLinear"))
            this.isLinear = Parameters.getBoolean("isLinear");
        if (Parameters.has("useGPU"))
            this.useGPU = Parameters.getBoolean("useGPU");
        if (Parameters.has("setiAstroDenoiseParentFolderPath"))
            this.setiAstroDenoiseParentFolderPath = Parameters.getString("setiAstroDenoiseParentFolderPath");
        if (Parameters.has("denoiseStrength"))
            this.denoiseStrength = Parameters.getReal("denoiseStrength");
        this.loadPathFromFile();
    },

    savePathToFile: function() {
        try {
            let file = new File;
            file.createForWriting(this.configFilePath);
            file.outTextLn(this.setiAstroDenoiseParentFolderPath);
            file.close();
        } catch (error) {
            console.warningln("Failed to save SetiAstroDenoise parent folder path: " + error.message);
        }
    },

    loadPathFromFile: function() {
        try {
            if (File.exists(this.configFilePath)) {
                let file = new File;
                file.openForReading(this.configFilePath);
                let lines = File.readLines(this.configFilePath);
                if (lines.length > 0) {
                    this.setiAstroDenoiseParentFolderPath = lines[0].trim();
                }
                file.close();
            }
        } catch (error) {
            console.warningln("Failed to load SetiAstroDenoise parent folder path: " + error.message);
        }
    }
};

// Dialog setup, image selection, denoise strength slider, etc.
function SetiAstroDenoiseDialog() {
    this.__base__ = Dialog;
    this.__base__();

    console.hide();
    SetiAstroDenoiseParameters.load();

    this.title = new Label(this);
    this.title.text = "SetiAstroCosmicClarityDenoise " + VERSION;
    this.title.textAlignment = TextAlign_Center;

    this.description = new TextBox(this);
    this.description.readOnly = true;
    this.description.text = "This script integrates with SetiAstroCosmicClarityDenoise for noise reduction.\n" +
                            "It saves the current image, runs the SetiAstroCosmicClarityDenoise tool, and replaces " +
                            "the image with the denoised version.";
    this.description.setMinWidth(400);

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

    // Denoise Strength slider
    this.denoiseStrengthSlider = new NumericControl(this);
    this.denoiseStrengthSlider.label.text = "Noise Reduction Strength:";
    this.denoiseStrengthSlider.setRange(1, 8);
    this.denoiseStrengthSlider.setValue(SetiAstroDenoiseParameters.denoiseStrength);
    this.denoiseStrengthSlider.setPrecision(2);
    this.denoiseStrengthSlider.onValueUpdated = function(value) {
        SetiAstroDenoiseParameters.denoiseStrength = value;
    };

    // Linear state checkbox
    this.linearStateCheckbox = new CheckBox(this);
    this.linearStateCheckbox.text = "Image is in Linear State";
    this.linearStateCheckbox.checked = SetiAstroDenoiseParameters.isLinear;
    this.linearStateCheckbox.onCheck = function(checked) {
        SetiAstroDenoiseParameters.isLinear = checked;
        SetiAstroDenoiseParameters.save();
    };

    // Wrench Icon Button for setting the SetiAstroDenoise parent folder path
    this.setupButton = new ToolButton(this);
    this.setupButton.icon = this.scaledResource(":/icons/wrench.png");
    this.setupButton.setScaledFixedSize(24, 24);
    this.setupButton.onClick = function() {
        let pathDialog = new GetDirectoryDialog;
        pathDialog.initialPath = SetiAstroDenoiseParameters.setiAstroDenoiseParentFolderPath;
        if (pathDialog.execute()) {
            SetiAstroDenoiseParameters.setiAstroDenoiseParentFolderPath = pathDialog.directory;
            SetiAstroDenoiseParameters.save();
        }
    };

    this.okButton = new PushButton(this);
    this.okButton.text = "OK";
    this.okButton.onClick = () => this.ok();

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
    this.sizer.spacing = 6;
    this.sizer.add(this.stellarAmountSlider);
    this.sizer.add(this.nonStellarStrengthSlider);
    this.sizer.add(this.linearStateCheckbox);
    this.sizer.addStretch();

    // GPU Acceleration checkbox (only for Windows)
    if (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") {
        this.gpuAccelerationCheckbox = new CheckBox(this);
        this.gpuAccelerationCheckbox.text = "Enable GPU Acceleration";
        this.gpuAccelerationCheckbox.checked = true;  // Default to enabled
        this.gpuAccelerationCheckbox.onCheck = function(checked) {
            SetiAstroDenoiseParameters.useGPU = checked;
        };

        this.sizer.add(this.gpuAccelerationCheckbox);
    }

    this.sizer.add(this.setupButton);
    this.sizer.addSpacing(12);
    this.sizer.add(this.buttonsSizer);

    this.windowTitle = "SetiAstroCosmicClarity Script";
    this.adjustToContents();

    // Initially update visibility based on the selected mode
    this.updateVisibility();
}
SetiAstroDenoiseDialog.prototype = new Dialog;

// Function to stretch and save image as a 32-bit TIFF
function saveImageAsTiff(inputFolderPath, view) {
    if (SetiAstroDenoiseParameters.isLinear) {
        console.writeln("Image is in linear state, applying full unlinked stretch.");

        let imageMin = view.mainView.image.minimum();
        let imageMedian = view.mainView.image.median();
        SetiAstroDenoiseParameters.originalMin = imageMin;
        SetiAstroDenoiseParameters.originalMedian = imageMedian;

        // Stretching PixelMath
        let pixelMath1 = new PixelMath;
        pixelMath1.expression = "$T - min($T)";
        pixelMath1.useSingleExpression = true;
        pixelMath1.executeOn(view.mainView);
    }

    // Save the image as 32-bit TIFF using FileFormatInstance
    let filePath = inputFolderPath + pathSeparator + view.mainView.id + ".tiff";
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

// Create batch file to run the denoise process
function createBatchFile(batchFilePath, exePath, denoiseStrength, useGPU) {
    let batchContent;

    // macOS/Linux shell script
    if (CoreApplication.platform == "MACOSX" || CoreApplication.platform == "macOS") {
        batchContent = "#!/bin/sh\n";
        batchContent += "cd \"" + exePath + "\"\n";
        batchContent += "./setiastrocosmicclaritydenoise " +
                        "--denoise_strength " + denoiseStrength.toFixed(2) + "\n";

    // Windows batch script with GPU handling
    } else if (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") {
        batchContent = "@echo off\n";
        batchContent += "cd /d \"" + exePath + "\"\n";
        batchContent += "start setiastrocosmicclaritydenoise.exe " +
                        "--denoise_strength " + denoiseStrength.toFixed(2) + " " +
                        (useGPU ? "" : "--disable_gpu") + "\n";
    } else {
        console.criticalln("Unsupported platform: " + CoreApplication.platform);
        return false;
    }

    // Write the script to the specified path
    try {
        File.writeTextFile(batchFilePath, batchContent);
        console.writeln((CoreApplication.platform == "MACOSX" || CoreApplication.platform == "macOS") ?
                        "Shell script created: " + batchFilePath :
                        "Batch file created: " + batchFilePath);
    } catch (error) {
        console.criticalln("Failed to create batch/shell file: " + error.message);
        return false;
    }

    return true;
}

// Process output image after denoising
function processOutputImage(outputFilePath, targetView) {
    if (!File.exists(outputFilePath)) {
        console.criticalln("Denoised file not found: " + outputFilePath);
        return;
    }

    let denoisedWindow = ImageWindow.open(outputFilePath)[0];
    if (denoisedWindow) {
        denoisedWindow.show();
        let pixelMath = new PixelMath;
        pixelMath.expression = "iif(" + denoisedWindow.mainView.id + " == 0, $T, " + denoisedWindow.mainView.id + ")";
        pixelMath.useSingleExpression = true;
        pixelMath.createNewImage = false;
        pixelMath.executeOn(targetView.mainView);
        denoisedWindow.forceClose();

        try {
            File.remove(outputFilePath);
            console.writeln("Deleted denoised file: " + outputFilePath);
        } catch (error) {
            console.warningln("Failed to delete denoised file: " + outputFilePath);
        }
    } else {
        console.criticalln("Failed to open denoised image: " + outputFilePath);
    }
}

// Main execution block for running the script
let dialog = new SetiAstroDenoiseDialog();
console.show();
console.writeln("SetiAstroCosmicClarityDenoise process started.");
console.flush();

if (dialog.execute()) {
    let selectedIndex = dialog.imageSelectionDropdown.currentItem;
    let selectedView = ImageWindow.windows[selectedIndex];

    if (!selectedView) {
        console.criticalln("Please select an image.");
    } else {
        let inputFolderPath = SetiAstroDenoiseParameters.setiAstroDenoiseParentFolderPath + pathSeparator + "input";
        let outputFolderPath = SetiAstroDenoiseParameters.setiAstroDenoiseParentFolderPath + pathSeparator + "output";
        let outputFileName = selectedView.mainView.id + "_denoised.tif";
        let outputFilePath = outputFolderPath + pathSeparator + outputFileName;

        let inputFilePath = saveImageAsTiff(inputFolderPath, selectedView);
        let batchFilePath = SetiAstroDenoiseParameters.setiAstroDenoiseParentFolderPath + pathSeparator + "run_setiastrocosmicclaritydenoise" + SCRIPT_EXT;

        if (createBatchFile(batchFilePath, SetiAstroDenoiseParameters.setiAstroDenoiseParentFolderPath, SetiAstroDenoiseParameters.denoiseStrength, SetiAstroDenoiseParameters.useGPU)) {
            let process = new ExternalProcess;
            try {
                if (CoreApplication.platform == "MACOSX" || CoreApplication.platform == "macOS" || CoreApplication.platform == "LINUX") {
                    if (!process.start(CMD_EXEC, [batchFilePath])) {
                        console.writeln("SetiAstroCosmicClarityDenoise started.");
                        console.flush();
                    }
                } else if (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") {
                    if (!process.start(CMD_EXEC, ["/c", batchFilePath])) {
                        console.writeln("SetiAstroCosmicClarityDenoise started.");
                        console.flush();
                    }
                }
            } catch (error) {
                console.criticalln("Error starting process: " + error.message);
            }

            if (waitForFile(outputFilePath, 120)) {
                processOutputImage(outputFilePath, selectedView);
                deleteInputFile(inputFilePath);
            } else {
                console.criticalln("Output file not found after 120 seconds.");
            }
        }
    }
}
