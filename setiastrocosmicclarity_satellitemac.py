# PyQt5 Imports
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QCheckBox, QFileDialog, QLineEdit, QTextEdit, QStyle, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon

# Standard Library Imports
import os
import sys
import time
import platform
import base64
import ast
import warnings
import argparse

# Third-Party Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
from astropy.io import fits
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import lz4.block
import zstandard

import torch.sparse
import torch._C


# Custom Imports
from xisf import XISF


import argparse  # For command-line argument parsing

GLOBAL_CLIP_TRAIL = False  # Default to False

# Binary Classification Model (Satellite Detection Model)
class BinaryClassificationCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(BinaryClassificationCNN, self).__init__()
        self.features = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features.fc = nn.Linear(self.features.fc.in_features, 1)  # Single output for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x



class SharpeningCNN(nn.Module):
    def __init__(self):
        super(SharpeningCNN, self).__init__()
        
        # Encoder (down-sampling path)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # 1st layer (3 -> 8 feature maps)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 2nd layer (8 -> 16 feature maps)
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Additional layer (16 -> 16)
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2),  # Dilated convolution (16 -> 32)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2),  # Additional dilated layer (32 -> 32)
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),  # Dilated convolution (32 -> 64)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),  # Additional dilated layer (64 -> 64)
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),  # Dilated convolution (64 -> 128)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),  # Additional dilated layer (128 -> 128)
            nn.ReLU()
        )
        
        # Decoder (up-sampling path with skip connections)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),  # Combine with encoder3 output
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Additional layer
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),  # Combine with encoder2 output
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Additional layer
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1),  # Combine with encoder1 output
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),  # Additional layer (16 -> 8 feature maps)
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(8, 3, kernel_size=3, padding=1),  # Output layer (8 -> 3 channels for RGB output)
            nn.Sigmoid()  # Ensure output values are between 0 and 1
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # First encoding block
        e2 = self.encoder2(e1)  # Second encoding block
        e3 = self.encoder3(e2)  # Third encoding block
        e4 = self.encoder4(e3)  # Fourth encoding block (128 feature maps with dilation)

        # Decoder with skip connections
        d4 = self.decoder4(torch.cat([e4, e3], dim=1))  # Combine with encoder3 output
        d3 = self.decoder3(torch.cat([d4, e2], dim=1))  # Combine with encoder2 output
        d2 = self.decoder2(torch.cat([d3, e1], dim=1))  # Combine with encoder1 output
        d1 = self.decoder1(d2)  # Final output layer

        return d1


# Get the directory of the executable or the script location
exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

cached_models = None

def load_models(exe_dir, use_gpu=True):
    global cached_models
    if cached_models:
        return cached_models

    # Use MPS for macOS if GPU support is enabled and MPS is available
    device = torch.device("mps") if use_gpu and torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load detection model
    detection_model = BinaryClassificationCNN(input_channels=3)
    detection_model.load_state_dict(
        torch.load(
            os.path.join(exe_dir, "satellite_trail_detector.pth"),
            map_location=device,
            weights_only=True
        )
    )
    detection_model.eval().to(device)

    # Load removal model
    removal_model = SharpeningCNN()
    removal_model.load_state_dict(
        torch.load(
            os.path.join(exe_dir, "satelliteremoval128featuremaps.pth"),
            map_location=device,
            weights_only=True
        )
    )
    removal_model.eval().to(device)

    # Cache the loaded models and device
    cached_models = {
        "detection_model": detection_model,
        "removal_model": removal_model,
        "device": device
    }
    return cached_models



# Function to extract luminance (Y channel) directly using a matrix for 32-bit float precision
def extract_luminance(image):
    # Ensure the image is a NumPy array in 32-bit float format
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32) / 255.0  # Ensure it's in 32-bit float format

    # Check if the image is grayscale (single channel), and convert it to 3-channel RGB if so
    if image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)  # Duplicate the single channel to create RGB

    # Conversion matrix for RGB to YCbCr (ITU-R BT.601 standard)
    conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                  [-0.168736, -0.331264, 0.5],
                                  [0.5, -0.418688, -0.081312]], dtype=np.float32)

    # Apply the RGB to YCbCr conversion matrix
    ycbcr_image = np.dot(image, conversion_matrix.T)

    # Split the channels: Y is luminance, Cb and Cr are chrominance
    y_channel = ycbcr_image[:, :, 0]
    cb_channel = ycbcr_image[:, :, 1] + 0.5  # Offset back to [0, 1] range
    cr_channel = ycbcr_image[:, :, 2] + 0.5  # Offset back to [0, 1] range

    # Return the Y (luminance) channel in 32-bit float and the Cb, Cr channels for later use
    return y_channel, cb_channel, cr_channel

def merge_luminance(y_channel, cb_channel, cr_channel):
    # Ensure all channels are 32-bit float and in the range [0, 1]
    y_channel = np.clip(y_channel, 0, 1).astype(np.float32)
    cb_channel = np.clip(cb_channel, 0, 1).astype(np.float32) - 0.5  # Adjust Cb to [-0.5, 0.5]
    cr_channel = np.clip(cr_channel, 0, 1).astype(np.float32) - 0.5  # Adjust Cr to [-0.5, 0.5]

    # Convert YCbCr to RGB in 32-bit float format
    rgb_image = ycbcr_to_rgb(y_channel, cb_channel, cr_channel)

    return rgb_image


# Helper function to convert YCbCr (32-bit float) to RGB
def ycbcr_to_rgb(y_channel, cb_channel, cr_channel):
    # Combine Y, Cb, Cr channels into one array for matrix multiplication
    ycbcr_image = np.stack([y_channel, cb_channel, cr_channel], axis=-1)

    # Conversion matrix for YCbCr to RGB (ITU-R BT.601 standard)
    conversion_matrix = np.array([[1.0,  0.0, 1.402],
                                  [1.0, -0.344136, -0.714136],
                                  [1.0,  1.772, 0.0]], dtype=np.float32)

    # Apply the YCbCr to RGB conversion matrix in 32-bit float
    rgb_image = np.dot(ycbcr_image, conversion_matrix.T)

    # Clip to ensure the RGB values stay within [0, 1] range
    rgb_image = np.clip(rgb_image, 0.0, 1.0)

    return rgb_image


# Function to split an image into chunks with overlap
def split_image_into_chunks_with_overlap(image, chunk_size, overlap):
    height, width = image.shape[:2]
    chunks = []
    step_size = chunk_size - overlap  # Define how much to step over (overlapping area)

    for i in range(0, height, step_size):
        for j in range(0, width, step_size):
            end_i = min(i + chunk_size, height)
            end_j = min(j + chunk_size, width)
            chunk = image[i:end_i, j:end_j]
            chunks.append((chunk, i, j))  # Return chunk and its position
    return chunks

# Soft blending weights for the overlap area
def generate_blend_weights(chunk_size, overlap):
    ramp = np.linspace(0, 1, overlap)
    flat = np.ones(chunk_size - 2 * overlap)
    blend_vector = np.concatenate([ramp, flat, ramp[::-1]])  # Create smooth transition
    blend_matrix = np.outer(blend_vector, blend_vector)  # 2D blending weights
    return blend_matrix

# Function to stitch overlapping chunks back together with soft blending while ignoring borders for all chunks
def stitch_chunks_ignore_border(chunks, image_shape, chunk_size, overlap, border_size=16):
    stitched_image = np.zeros(image_shape, dtype=np.float32)
    weight_map = np.zeros(image_shape, dtype=np.float32)  # Track blending weights
    
    for chunk, i, j in chunks:
        actual_chunk_h, actual_chunk_w = chunk.shape[:2]

        # Calculate the boundaries for the current chunk, ignoring the border
        border_h = min(border_size, actual_chunk_h // 2)
        border_w = min(border_size, actual_chunk_w // 2)

        # Always ignore the border for all chunks, including edges
        inner_chunk = chunk[border_h:actual_chunk_h-border_h, border_w:actual_chunk_w-border_w]
        stitched_image[i+border_h:i+actual_chunk_h-border_h, j+border_w:j+actual_chunk_w-border_w] += inner_chunk
        weight_map[i+border_h:i+actual_chunk_h-border_h, j+border_w:j+actual_chunk_w-border_w] += 1

    # Normalize by weight_map to blend overlapping areas
    stitched_image /= np.maximum(weight_map, 1)  # Avoid division by zero
    return stitched_image

# Function to blend two images (before and after)
def blend_images(before, after, amount):
    return (1 - amount) * before + amount * after



class FolderMonitor(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(float)

    def __init__(self, input_dir, output_dir, models, satellite_mode):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.models = models
        self.satellite_mode = satellite_mode
        self.running = True
        self.processed_files = set()  # Initialize processed files list

    def log_message(self, message):
        """Log a message using the provided signal or print to the console."""
        if callable(self.log_signal.emit):  # Ensure the signal is connected and callable
            self.log_signal.emit(message)
        else:
            print(message)  # Fallback to console logging

    def stop(self):
        self.running = False

    def run(self):
        # Initialize processed files with the current files in the folder
        self.processed_files = set([
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(('.tif', '.tiff', '.png', '.fits', '.fit', '.xisf'))
        ])

        while self.running:
            try:
                # List all eligible files in the input directory
                files = [
                    f for f in os.listdir(self.input_dir)
                    if f.lower().endswith(('.tif', '.tiff', '.png', '.fits', '.fit', '.xisf'))
                ]
                # Filter for files that haven't been processed yet
                new_files = [f for f in files if f not in self.processed_files]

                for new_file in new_files:
                    # Skip files with the "_satellited" suffix to avoid reprocessing
                    if "_satellited" in new_file:
                        continue

                    file_path = os.path.join(self.input_dir, new_file)
                    self.log_signal.emit(f"Detected new file: {new_file}")
                    
                    # Add a 5-second delay before processing
                    self.log_signal.emit(f"Waiting 5 seconds to ensure the file is fully saved...")
                    time.sleep(5)

                    try:
                        # Process the new file
                        satellited_image, original_header, bit_depth, file_extension, is_mono, file_meta = satellite_image(
                            file_path,
                            self.models['device'],
                            self.models['detection_model'],
                            self.models['removal_model'],
                            self.satellite_mode,
                            log_signal=self.log_signal,
                            progress_signal=self.progress_signal
                        )
                        if satellited_image is not None:
                            output_image_name = os.path.splitext(new_file)[0] + "_satellited" + file_extension
                            output_file = os.path.join(self.output_dir, output_image_name)
                            save_processed_image(
                                satellited_image,
                                output_file,
                                file_extension,
                                bit_depth,
                                original_header,
                                is_mono,
                                file_meta
                            )
                            self.log_signal.emit(f"Processed and saved: {output_image_name}")

                        else:
                            self.log_signal.emit(f"Failed to process: {new_file}")
                        
                        # Mark both input and output files as processed
                        self.processed_files.add(new_file)
                        self.processed_files.add(output_image_name)
                    except Exception as e:
                        self.log_signal.emit(f"Error processing file {new_file}: {e}")

                time.sleep(2)  # Add a short delay to avoid rapid file checks
            except Exception as e:
                self.log_signal.emit(f"Error monitoring folder: {e}")



class SatelliteToolDialog(QDialog):
    log_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmic Clarity Satellite Removal Tool V1.1.1")

        self.setGeometry(100, 100, 400, 300)

        if hasattr(sys, '_MEIPASS'):
            # PyInstaller path
            icon_path = os.path.join(sys._MEIPASS, 'satellite.png')
        else:
            # Development path
            icon_path = 'satellite.png'

        self.setWindowIcon(QIcon(icon_path))

        # Variables to store user input
        self.use_gpu = True
        self.satellite_mode = "full"
        self.input_dir = ""
        self.output_dir = ""
        self.models = None
        self.live_monitor_thread = None       
        self.log_signal.connect(self.log_message)

                
 

        # Main layout
        layout = QVBoxLayout()

        # Input folder selection
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Directory:")
        self.input_line_edit = QLineEdit()
        self.input_line_edit.setPlaceholderText("Select the input folder")
        input_browse_button = QPushButton("Browse")
        input_browse_button.clicked.connect(self.select_input_folder)
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_line_edit)
        input_layout.addWidget(input_browse_button)
        layout.addLayout(input_layout)

        # Output folder selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        self.output_line_edit = QLineEdit()
        self.output_line_edit.setPlaceholderText("Select the output folder")
        output_browse_button = QPushButton("Browse")
        output_browse_button.clicked.connect(self.select_output_folder)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_line_edit)
        output_layout.addWidget(output_browse_button)
        layout.addLayout(output_layout)

        # GPU selection
        self.gpu_checkbox = QCheckBox("Use GPU Acceleration")
        self.gpu_checkbox.setChecked(True)
        self.gpu_checkbox.stateChanged.connect(self.toggle_gpu)
        layout.addWidget(self.gpu_checkbox)

        # Satellite mode selection
        mode_label = QLabel("Satellite Mode:")
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems(["full", "luminance"])
        self.mode_dropdown.setCurrentText("full")
        self.mode_dropdown.currentTextChanged.connect(self.update_mode)
        layout.addWidget(mode_label)
        layout.addWidget(self.mode_dropdown)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)

        # Progress bar for chunk processing
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)  # Initialize with 0%
        layout.addWidget(self.progress_bar)   

        # Add "Clip Satellite Trail" checkbox
        self.clip_trail_checkbox = QCheckBox("Clip Satellite Trail to 0.000")
        self.clip_trail_checkbox.setChecked(False)  # Default unchecked
        self.clip_trail_checkbox.stateChanged.connect(self.update_clip_trail)
        layout.addWidget(self.clip_trail_checkbox)

        # Control buttons
        self.process_button = QPushButton("Batch Process Input Folder")
        self.process_button.clicked.connect(self.process_input_folder)        

        # Live monitor controls in a horizontal layout
        monitor_layout = QHBoxLayout()

        # Start Live Monitor button with green light icon
        self.live_monitor_button = QPushButton("Start Live Monitor")
        self.live_monitor_button.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.live_monitor_button.clicked.connect(self.start_live_monitor)

        # Stop Live Monitor button with stop icon
        self.stop_monitor_button = QPushButton("Stop Live Monitor")
        self.stop_monitor_button.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogCancelButton))
        self.stop_monitor_button.clicked.connect(self.stop_live_monitor)
        self.stop_monitor_button.setEnabled(False)

        # Add buttons to the horizontal layout
        monitor_layout.addWidget(self.live_monitor_button)
        monitor_layout.addWidget(self.stop_monitor_button)

        # Add the horizontal layout to the main layout
        layout.addWidget(self.process_button)
        layout.addLayout(monitor_layout)

        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(footer_label)

        self.setLayout(layout)

    def update_clip_trail(self, state):
        set_clip_trail(state == Qt.Checked)

    def update_progress_bar(self, progress):
        """Update the progress bar value."""
        self.progress_bar.setValue(int(progress))

    def log_message(self, message):
        """Display a log message in the QTextEdit."""
        self.log_display.append(message)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if folder:
            self.input_dir = folder
            self.input_line_edit.setText(folder)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir = folder
            self.output_line_edit.setText(folder)

    def toggle_gpu(self, state):
        self.use_gpu = state == Qt.Checked

    def update_mode(self, text):
        self.satellite_mode = text

    def start_live_monitor(self):
        if not self.input_dir or not self.output_dir:
            self.log_message("Error: Input and Output directories must be selected.")
            return

        if self.live_monitor_thread and self.live_monitor_thread.isRunning():
            self.log_message("Live monitor is already running.")
            return

        self.log_message("Starting live monitor...")
        use_gpu = self.gpu_checkbox.isChecked()
        self.models = load_models(exe_dir, use_gpu)
        satellite_mode = self.mode_dropdown.currentText()

        self.live_monitor_thread = FolderMonitor(
            self.input_dir,
            self.output_dir,
            self.models,
            satellite_mode
        )
        self.live_monitor_thread.log_signal.connect(self.log_message)
        self.live_monitor_thread.progress_signal.connect(self.update_progress_bar)
        self.live_monitor_thread.start()

        self.live_monitor_button.setEnabled(False)
        self.stop_monitor_button.setEnabled(True)

    def stop_live_monitor(self):
        if self.live_monitor_thread and self.live_monitor_thread.isRunning():
            self.live_monitor_thread.stop()
            self.live_monitor_thread.wait()
            self.live_monitor_button.setEnabled(True)
            self.stop_monitor_button.setEnabled(False)
            self.log_message("Live monitor stopped.")

    def process_input_folder(self):
        """Start batch processing in a separate thread."""
        if not self.input_dir or not self.output_dir:
            self.log_message("Error: Input and Output directories must be selected.")
            return

        self.log_message("Preparing for batch processing...")

        if not self.models:
            self.models = load_models(exe_dir, self.gpu_checkbox.isChecked())

        self.progress_bar.setValue(0)

        # Create and start the batch processing thread
        self.batch_processor = BatchProcessor(
            self.input_dir,
            self.output_dir,
            self.models,
            self.mode_dropdown.currentText()
        )
        self.batch_processor.log_signal.connect(self.log_message)
        self.batch_processor.progress_signal.connect(self.update_progress_bar)
        self.batch_processor.processing_complete.connect(self.on_batch_processing_complete)

        self.log_message("Starting batch processing...")
        self.batch_processor.start()

    def on_batch_processing_complete(self):
        """Handle completion of batch processing."""
        self.log_message("Batch processing complete.")
        self.progress_bar.setValue(100)

    def update_mode(self, text):
        self.satellite_mode = text


def get_user_input():
    app = QApplication([])
    dialog = SatelliteToolDialog()
    if dialog.exec_() == QDialog.Accepted:
        # Ensure input and output folders are valid
        if not dialog.input_dir or not dialog.output_dir:
            raise ValueError("Input and Output directories must be specified.")
        return dialog.input_dir, dialog.output_dir, dialog.use_gpu, dialog.satellite_mode
    else:
        raise RuntimeError("User cancelled input.")

def show_progress(current, total, log_signal=None, progress_signal=None):
    progress_percentage = (current / total) * 100
    message = f"Progress: {progress_percentage:.2f}% ({current}/{total} chunks processed)"
    print(message, end='\r', flush=True)
    if log_signal:
        log_signal.emit(message)
    if progress_signal:
        progress_signal.emit(progress_percentage)  # Emit progress percentage



# Function to replace the 5-pixel border from the original image into the processed image
def replace_border(original_image, processed_image, border_size=16):
    # Ensure the dimensions of both images match
    if original_image.shape != processed_image.shape:
        raise ValueError("Original image and processed image must have the same dimensions.")
    
    # Replace the top border
    processed_image[:border_size, :] = original_image[:border_size, :]
    
    # Replace the bottom border
    processed_image[-border_size:, :] = original_image[-border_size:, :]
    
    # Replace the left border
    processed_image[:, :border_size] = original_image[:, :border_size]
    
    # Replace the right border
    processed_image[:, -border_size:] = original_image[:, -border_size:]
    
    return processed_image


# Function to stretch an image
def stretch_image(image):
    """
    Perform a linear stretch on the image (unlinked for RGB).
    """
    original_min = np.min(image, axis=(0, 1)) if image.ndim == 3 else np.min(image)
    stretched_image = image - original_min  # Shift image so that the min is 0
    target_median = 0.25

    if image.ndim == 3:  # RGB image case
        # Stretch each channel independently
        original_medians = np.median(stretched_image, axis=(0, 1))
        stretched_image = np.array([
            ((median - 1) * target_median * stretched_image[..., c]) / (
                median * (target_median + stretched_image[..., c] - 1) - target_median * stretched_image[..., c]
            ) for c, median in enumerate(original_medians)
        ]).transpose(1, 2, 0)
    else:  # Grayscale image case
        original_medians = np.median(stretched_image)
        stretched_image = ((original_medians - 1) * target_median * stretched_image) / (
            original_medians * (target_median + stretched_image - 1) - target_median * stretched_image
        )

    stretched_image = np.clip(stretched_image, 0, 1)  # Clip to [0, 1] range
    return stretched_image, original_min, original_medians

# Function to unstretch an image with final median adjustment
def unstretch_image(image, original_medians, original_min):
    """
    Undo the stretch to return the image to the original linear state (unlinked for RGB).
    """
    if image.ndim == 3:  # RGB image case
        # Unstretch each channel independently
        unstretched_image = np.array([
            ((median - 1) * original_medians[c] * image[..., c]) / (
                median * (original_medians[c] + image[..., c] - 1) - original_medians[c] * image[..., c]
            ) for c, median in enumerate(np.median(image, axis=(0, 1)))
        ]).transpose(1, 2, 0)
    else:  # Grayscale image case
        median = np.median(image)
        unstretched_image = ((median - 1) * original_medians * image) / (
            median * (original_medians + image - 1) - original_medians * image
        )

    unstretched_image += original_min  # Add back the original minimum
    unstretched_image = np.clip(unstretched_image, 0, 1)  # Clip to [0, 1] range
    return unstretched_image


# Function to add a border of median value around the image
def add_border(image, border_size=5):
    median_value = np.median(image)
    if len(image.shape) == 2:
        return np.pad(image, ((border_size, border_size), (border_size, border_size)), 'constant', constant_values=median_value)
    else:
        return np.pad(image, ((border_size, border_size), (border_size, border_size), (0, 0)), 'constant', constant_values=median_value)

# Function to remove the border added around the image
def remove_border(image, border_size=5):
    if len(image.shape) == 2:
        return image[border_size:-border_size, border_size:-border_size]
    else:
        return image[border_size:-border_size, border_size:-border_size, :]

# Function to load the satellite detection model
def load_detection_model(model_path, device):
    """
    Load the pre-trained satellite detection model.

    Args:
        model_path (str): Path to the detection model.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: The detection model.
    """
    detection_model = BinaryClassificationCNN(input_channels=3)
    detection_model.load_state_dict(torch.load(model_path, map_location=device))
    detection_model.eval()
    detection_model.to(device)
    return detection_model

def set_clip_trail(value):
    global GLOBAL_CLIP_TRAIL
    GLOBAL_CLIP_TRAIL = value




# Function to satellite the image
def satellite_image(image_path, device, detection_model, removal_model, satellite_mode='luminance', progress_signal=None, log_signal=None):
    def log_message(message):
        print(message)  # Always log to console
        if log_signal:
            print(f"log_signal type: {type(log_signal)}")  # Debugging
            if hasattr(log_signal, 'emit') and callable(log_signal.emit):  # PyQt signal
                log_signal.emit(message)
            elif callable(log_signal):  # Callable function
                log_signal(message)

    # Get file extension
    file_extension = os.path.splitext(image_path)[1].lower()
    if file_extension not in ['.png', '.tif', '.tiff', '.fit', '.fits', '.xisf']:
        print(f"Ignoring non-image file: {image_path}")
        return None, None, None, None, None, None, None

    original_header = None
    image = None
    bit_depth = "32-bit float"  # Default bit depth
    is_mono = False
    file_meta, image_meta = None, None
    log_message(f"Loading image: {image_path}")
    try:
        # Load the image based on its extension
        if file_extension in ['.tif', '.tiff']:
            image = tiff.imread(image_path)
            print(f"Loaded TIFF image with dtype: {image.dtype}")
            if image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
                bit_depth = "16-bit"
            elif image.dtype == np.uint32:
                image = image.astype(np.float32) / 4294967295.0
                bit_depth = "32-bit unsigned"
            elif image.dtype == np.float32:
                bit_depth = "32-bit float"
            
            log_message(f"Final bit depth set to: {bit_depth}")

            # Handle alpha channels and grayscale TIFFs
            if image.ndim == 3 and image.shape[-1] == 4:
                print("Detected alpha channel in TIFF. Removing it.")
                image = image[:, :, :3]
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)  # Convert to RGB
                is_mono = True
            else:
                is_mono = False

        elif file_extension in ['.fits', '.fit']:
            # Load the FITS image
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data
                original_header = hdul[0].header  # Capture the FITS header

                # Ensure the image data uses the native byte order
                if image_data.dtype.byteorder not in ('=', '|'):  # Check if byte order is non-native
                    image_data = image_data.astype(image_data.dtype.newbyteorder('='))  # Convert to native byte order


                # Determine the bit depth based on the data type in the FITS file
                if image_data.dtype == np.uint16:
                    bit_depth = "16-bit"
                    print("Identified 16-bit FITS image.")
                    log_message("Identified 16-bit FITS image.")
                elif image_data.dtype == np.float32:
                    bit_depth = "32-bit floating point"
                    print("Identified 32-bit floating point FITS image.")
                    log_message("Identified 32-bit floating point FITS image.")
                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    print("Identified 32-bit unsigned FITS image.")
                    log_message("Identified 32-bit unsigned FITS image.")

                # Handle 3D FITS data (e.g., RGB or multi-layered data)
                if image_data.ndim == 3 and image_data.shape[0] == 3:
                    image = np.transpose(image_data, (1, 2, 0))  # Reorder to (height, width, channels)
                    
                    if bit_depth == "16-bit":
                        image = image.astype(np.float32) / 65535.0  # Normalize to [0, 1] for 16-bit
                    elif bit_depth == "32-bit unsigned":
                        # Apply BSCALE and BZERO if present
                        bzero = original_header.get('BZERO', 0)  # Default to 0 if not present
                        bscale = original_header.get('BSCALE', 1)  # Default to 1 if not present

                        # Convert to float and apply the scaling and offset
                        image = image.astype(np.float32) * bscale + bzero

                        # Normalize based on the actual data range
                        image_min = image.min()  # Get the min value after applying BZERO
                        image_max = image.max()  # Get the max value after applying BZERO

                        # Normalize the image data to the range [0, 1]
                        image = (image - image_min) / (image_max - image_min)
                        print(f"Image range after applying BZERO and BSCALE (3D case): min={image_min}, max={image_max}")

                    is_mono = False  # RGB data

                # Handle 2D FITS data (grayscale)
                elif image_data.ndim == 2:
                    if bit_depth == "16-bit":
                        image = image_data.astype(np.float32) / 65535.0  # Normalize to [0, 1] for 16-bit
                    elif bit_depth == "32-bit unsigned":
                        # Apply BSCALE and BZERO if present
                        bzero = original_header.get('BZERO', 0)  # Default to 0 if not present
                        bscale = original_header.get('BSCALE', 1)  # Default to 1 if not present

                        # Convert to float and apply the scaling and offset
                        image = image_data.astype(np.float32) * bscale + bzero

                        # Normalize based on the actual data range
                        image_min = image.min()  # Get the min value after applying BZERO
                        image_max = image.max()  # Get the max value after applying BZERO

                        # Normalize the image data to the range [0, 1]
                        image = (image - image_min) / (image_max - image_min)
                        print(f"Image range after applying BZERO and BSCALE (2D case): min={image_min}, max={image_max}")

                    elif bit_depth == "32-bit floating point":
                        image = image_data  # No normalization needed for 32-bit float
                    is_mono = True
                    image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel for consistency
                else:
                    raise ValueError("Unsupported FITS format!")

        # Check if file extension is '.xisf'
        elif file_extension == '.xisf':
            # Load XISF file
            xisf = XISF(image_path)
            image = xisf.read_image(0)  # Assuming the image data is in the first image block
            image_meta = xisf.get_images_metadata()  # List of metadata blocks for each image
            file_meta = xisf.get_file_metadata()  # File-level metadata

            # Determine bit depth
            if image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
                bit_depth = "16-bit"
            elif image.dtype == np.uint32:
                image = image.astype(np.float32) / 4294967295.0
                bit_depth = "32-bit unsigned"
            elif image.dtype == np.float32:
                bit_depth = "32-bit floating point"

            # Check if image is mono (2D array) and stack into 3 channels if mono
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                image = np.stack([image.squeeze()] * 3, axis=-1)  # Convert mono to RGB by duplicating channels
                is_mono = True
            else:
                is_mono = False

            print(f"Loaded XISF image with bit depth: {bit_depth}, Mono: {is_mono}")


        else:  # Assume 8-bit PNG
            image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
            bit_depth = "8-bit"
            print(f"Loaded {bit_depth} PNG image.")
            log_message(f"Loaded {bit_depth} PNG image.")

        if image is None:
            raise ValueError(f"Failed to load image from: {image_path}")
        
        original_image=image

        # Add a border around the image with the median value
        image_with_border = image #add_border(image, border_size=16)

        # Check if the image needs stretching based on its median value
        stretch_needed = np.median(image_with_border - np.min(image_with_border)) < 0.08
        original_median = np.median(image_with_border)
        if stretch_needed:
            stretched_image, original_min, original_median = stretch_image(image_with_border)
        else:
            stretched_image = image_with_border
            original_min = None

        assert stretched_image.dtype == np.float32, f"Image dtype is {stretched_image.dtype}, expected np.float32"

        # Process mono or color images
        if is_mono:
            # Convert mono image to 3 channels by stacking
            stretched_image = np.stack([stretched_image[:, :, 0]] * 3, axis=-1)

        if satellite_mode == 'luminance' and is_mono:
            # Extract luminance (single-channel processing for mono)
            satellited_image = satellite_channel(
                stretched_image[:, :, 0], device, removal_model, detection_model, is_mono=True, progress_signal=progress_signal
            )
            satellited_image = satellited_image[:, :, np.newaxis]  # Convert back to single channel

        elif satellite_mode == 'luminance' and not is_mono:
            # Extract luminance and process only that channel
            luminance, cb_channel, cr_channel = extract_luminance(stretched_image)
            satellited_luminance = satellite_channel(
                luminance, device, removal_model, detection_model, is_mono=True, progress_signal=progress_signal
            )
            satellited_image = merge_luminance(satellited_luminance, cb_channel, cr_channel)

        else:  # Full RGB denoising mode
            # Process the 3-channel image directly
            satellited_image = satellite_channel(
                stretched_image, device, removal_model, detection_model, is_mono=False, progress_signal=progress_signal
            )

        # Unstretch if stretched previously
        if stretch_needed:
            satellited_image = unstretch_image(satellited_image, original_median, original_min)

        # If GLOBAL_CLIP_TRAIL is True, set pixels with the minimum value to 0
        if GLOBAL_CLIP_TRAIL:
            min_value = np.min(satellited_image)
            satellited_image = np.where(satellited_image == min_value, 0.000, satellited_image)


        # Remove the border added around the image
        satellited_image = replace_border(original_image, satellited_image, border_size=16)

        assert satellited_image.dtype == np.float32, f"Image dtype is {satellited_image.dtype}, expected np.float32"

        log_message(f"Image {image_path} processed successfully.")
        return satellited_image, original_header, bit_depth, file_extension, is_mono, file_meta, image_meta

    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None, None, None, None, None, None
    
# Function to satellite a single channel or 3-channel image
def satellite_channel(image, device, removal_model, detection_model, is_mono=False, progress_signal=None):
    """
    Detect satellite trails in chunks and process only the detected chunks.

    Args:
        image (numpy.ndarray): Image to process.
        device (torch.device): Device to run the models on.
        removal_model (torch.nn.Module): Pre-trained removal model.
        detection_model (torch.nn.Module): Pre-trained detection model.
        is_mono (bool): Whether the image is monochrome.

    Returns:
        numpy.ndarray: Processed image.
    """
    # Split image into chunks
    chunk_size = 256
    overlap = 64
    chunks = split_image_into_chunks_with_overlap(image, chunk_size=chunk_size, overlap=overlap)
    total_chunks = len(chunks)
    satellited_chunks = []
    
    global GLOBAL_CLIP_TRAIL  # Access the global variable

    # Apply satellite detection and removal models to each chunk
    for idx, (chunk, i, j) in enumerate(chunks):
        original_chunk = chunk.copy()

        # Check for satellite trails in the chunk using the detection model
        if not contains_trail_with_ai(chunk, detection_model, device):
            # If no trail is detected, skip processing and add the original chunk
            satellited_chunks.append((chunk, i, j))
            continue

        # Prepare the chunk tensor
        if chunk.ndim == 2:  # Mono chunk
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            chunk_tensor = chunk_tensor.repeat(1, 3, 1, 1)  # Triplicate mono to 3 channels
        elif chunk.shape[-1] == 1:  # Handle grayscale chunks with explicit channel dim
            chunk_tensor = torch.tensor(chunk.squeeze(-1), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            chunk_tensor = chunk_tensor.repeat(1, 3, 1, 1)  # Triplicate mono to 3 channels
        else:  # RGB chunk
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # Run the removal model
        with torch.no_grad():
            processed_chunk = removal_model(chunk_tensor).squeeze().cpu().numpy()

        # Convert the output back to its original format
        if chunk.ndim == 2 or chunk.shape[-1] == 1:  # Mono chunk
            satellited_chunk = processed_chunk[0]  # Use the first output channel
        else:  # RGB chunk
            satellited_chunk = processed_chunk.transpose(1, 2, 0)  # Convert back to (H, W, C)

        if GLOBAL_CLIP_TRAIL:
            # Compute satellite-only image
            sattrail_only_image = original_chunk - satellited_chunk

            # Clip and scale to isolate the satellite trail
            mean_val = np.mean(sattrail_only_image)
            clipped_image = np.clip((sattrail_only_image - mean_val) * 10, 0, 1)

            # Apply a binary mask: values below 0.5 are set to 0, values >= 0.5 are set to 1
            binary_mask = np.where(clipped_image < 0.1, 0, 1)

            # Subtract the binary mask from the original chunk
            final_chunk = np.clip(original_chunk - binary_mask, 0, 1)
        else:
            final_chunk = satellited_chunk

        satellited_chunks.append((final_chunk, i, j))


        # Update progress
        progress_percentage = (idx + 1) / total_chunks * 100
        if progress_signal:
            progress_signal.emit(progress_percentage)  # Emit as a float, not a string
        QApplication.processEvents()    

    # Finalize by setting the progress bar to 100%
    if progress_signal:
        progress_signal.emit(100.0)        

    # Stitch the chunks back together
    satellited_image = stitch_chunks_ignore_border(
        satellited_chunks, image.shape, chunk_size=chunk_size, overlap=overlap
    )
    return satellited_image



# Function to detect satellite trails in a chunk using the detection AI
def contains_trail_with_ai(chunk, detection_model, device):
    """
    Detect satellite trails in a chunk using the detection model.

    Args:
        chunk (numpy.ndarray): Image chunk to analyze.
        detection_model (torch.nn.Module): Pre-trained detection model.
        device (torch.device): Device to run the model on.

    Returns:
        bool: True if a trail is detected, False otherwise.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # Ensure input size matches the model
    ])
    input_tensor = transform(chunk).unsqueeze(0).to(device)

    # Run the detection model
    with torch.no_grad():
        output = detection_model(input_tensor).item()

    return output > 0.5  # Return True if trail is detected (output > 0.5)

def save_processed_image(image, output_path, file_extension, bit_depth, original_header=None, is_mono=False, file_meta=None, image_meta=None):
    """Save the processed image to the appropriate format."""
    if file_extension in ['.fits', '.fit']:
        # Save as FITS file with header information if the original was FITS
        if original_header is not None:
            if is_mono:  # Grayscale FITS
                if bit_depth == "16-bit":
                    satellited_image_fits = (image[:, :, 0] * 65535).astype(np.uint16)
                elif bit_depth == "32-bit unsigned":
                    bzero = original_header.get('BZERO', 0)
                    bscale = original_header.get('BSCALE', 1)
                    satellited_image_fits = (image[:, :, 0].astype(np.float32) * bscale + bzero).astype(np.uint32)
                else:  # 32-bit float
                    satellited_image_fits = image[:, :, 0].astype(np.float32)

                # Update header for a 2D (grayscale) image
                original_header['NAXIS'] = 2
                original_header['NAXIS1'] = image.shape[1]  # Width
                original_header['NAXIS2'] = image.shape[0]  # Height
                if 'NAXIS3' in original_header:
                    del original_header['NAXIS3']  # Remove if present

                hdu = fits.PrimaryHDU(satellited_image_fits, header=original_header)
            else:  # RGB FITS
                satellited_image_transposed = np.transpose(image, (2, 0, 1))  # Channels, Height, Width
                if bit_depth == "16-bit":
                    satellited_image_fits = (satellited_image_transposed * 65535).astype(np.uint16)
                elif bit_depth == "32-bit unsigned":
                    satellited_image_fits = satellited_image_transposed.astype(np.float32)
                    original_header['BITPIX'] = -32
                else:
                    satellited_image_fits = satellited_image_transposed.astype(np.float32)

                # Update header for a 3D (RGB) image
                original_header['NAXIS'] = 3
                original_header['NAXIS1'] = satellited_image_transposed.shape[2]  # Width
                original_header['NAXIS2'] = satellited_image_transposed.shape[1]  # Height
                original_header['NAXIS3'] = satellited_image_transposed.shape[0]  # Channels

                hdu = fits.PrimaryHDU(satellited_image_fits, header=original_header)

            # Write the FITS file
            hdu.writeto(output_path, overwrite=True)
    elif file_extension in ['.tif', '.tiff']:
        # Save as TIFF based on the original bit depth
        if bit_depth == "16-bit":
            if is_mono:
                tiff.imwrite(output_path, (image[:, :, 0] * 65535).astype(np.uint16))
            else:
                tiff.imwrite(output_path, (image * 65535).astype(np.uint16))
        elif bit_depth == "32-bit unsigned":
            if is_mono:
                tiff.imwrite(output_path, (image[:, :, 0] * 4294967295).astype(np.uint32))
            else:
                tiff.imwrite(output_path, (image * 4294967295).astype(np.uint32))
        else:  # 32-bit float
            if is_mono:
                tiff.imwrite(output_path, image[:, :, 0].astype(np.float32))
            else:
                tiff.imwrite(output_path, image.astype(np.float32))
    elif file_extension == '.xisf':
        try:
            # Debug: Print original image details
            print(f"Original image shape: {image.shape}, dtype: {image.dtype}")
            print(f"Bit depth: {bit_depth}")

            # Adjust bit depth
            if bit_depth == "16-bit":
                processed_image = (image * 65535).astype(np.uint16)
            elif bit_depth == "32-bit unsigned":
                processed_image = (image * 4294967295).astype(np.uint32)
            else:  # Default to 32-bit float
                processed_image = image.astype(np.float32)

            # Adjust for mono images
            if is_mono:
                print("Preparing mono image...")
                processed_image = processed_image[:, :, 0]  # Take the first channel
                processed_image = processed_image[:, :, np.newaxis]  # Add back channel dimension
                image_meta[0]['geometry'] = (processed_image.shape[1], processed_image.shape[0], 1)
                image_meta[0]['colorSpace'] = 'Gray'  # Update metadata for mono


            # Debug: Print processed image details
            print(f"Processed image shape: {processed_image.shape}, dtype: {processed_image.dtype}")

            # Save the image
            XISF.write(
                output_path,
                processed_image,
                creator_app="Seti Astro Cosmic Clarity",
                image_metadata=image_meta[0],  # First block of image metadata
                xisf_metadata=file_meta,       # File-level metadata
                codec='lz4hc',
                shuffle=True
            )
            print(f"Saved {bit_depth} XISF image as RGB with metadata to: {output_path}")

        except Exception as e:
            print(f"Error saving XISF file: {e}")



    else:
        # Save as 8-bit PNG
        satellited_image_8bit = (image * 255).astype(np.uint8)
        satellited_image_pil = Image.fromarray(satellited_image_8bit)
        satellited_image_pil.save(output_path)

    print(f"Saved processed image to: {output_path}")

def log_message(message, log_signal=None):
    """Log a message using the provided signal or callable function."""
    if log_signal:
        if hasattr(log_signal, 'emit') and callable(log_signal.emit):  # PyQt signal
            log_signal.emit(message)
        elif callable(log_signal):  # Regular callable
            log_signal(message)
        else:
            print(f"Warning: log_signal is neither callable nor a signal. Type: {type(log_signal)}")
    else:
        print(message)  # Fallback to console logging

class BatchProcessor(QThread):
    progress_signal = pyqtSignal(float)
    log_signal = pyqtSignal(str)
    processing_complete = pyqtSignal()

    def __init__(self, input_dir, output_dir, models, satellite_mode):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.models = models
        self.satellite_mode = satellite_mode

    def run(self):
        try:
            self.log_signal.emit("Starting batch processing...")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            files = [
                f for f in os.listdir(self.input_dir)
                if f.lower().endswith(('.tif', '.tiff', '.png', '.fits', '.fit', '.xisf'))
            ]
            total_files = len(files)

            if total_files == 0:
                self.log_signal.emit("No valid image files found in the input directory.")
                self.processing_complete.emit()
                return

            processed_files = 0
            for image_name in files:
                image_path = os.path.join(self.input_dir, image_name)
                self.log_signal.emit(f"Processing image: {image_name}")

                try:
                    satellited_image, original_header, bit_depth, file_extension, is_mono, file_meta, image_meta = satellite_image(
                        image_path,
                        self.models['device'],
                        self.models["detection_model"],
                        self.models["removal_model"],
                        self.satellite_mode,
                        log_signal=self.log_signal,
                        progress_signal=self.progress_signal  # Pass chunk-level progress
                    )

                    if satellited_image is not None:
                        output_image_name = os.path.splitext(image_name)[0] + "_satellited" + file_extension
                        output_image_path = os.path.join(self.output_dir, output_image_name)
                        save_processed_image(
                            satellited_image,
                            output_image_path,
                            file_extension,
                            bit_depth,
                            original_header,
                            is_mono,
                            file_meta,
                            image_meta
                        )
                        self.log_signal.emit(f"Processed and saved: {output_image_name}")
                    else:
                        self.log_signal.emit(f"Failed to process: {image_name}")

                except Exception as e:
                    self.log_signal.emit(f"Error processing image {image_name}: {e}")

                # Update progress
                processed_files += 1
                overall_progress = (processed_files / total_files) * 100
                self.progress_signal.emit(overall_progress)

            
        except Exception as e:
            self.log_signal.emit(f"Error during batch processing: {e}")
        finally:
            self.processing_complete.emit()

class ProgressMonitorDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processing Progress")
        self.setGeometry(100, 100, 400, 200)

        if hasattr(sys, '_MEIPASS'):
            # PyInstaller path
            icon_path = os.path.join(sys._MEIPASS, 'satellite.png')
        else:
            # Development path
            icon_path = 'satellite.png'

        self.setWindowIcon(QIcon(icon_path))

        layout = QVBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)

        layout.addWidget(QLabel("Processing..."))
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_display)
        self.setLayout(layout)

    def log_message(self, message):
        self.log_display.append(message)

    def update_progress(self, value):
        self.progress_bar.setValue(int(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cosmic Clarity Satellite Removal Tool")
    parser.add_argument("--input", type=str, help="Path to the input directory containing images.")
    parser.add_argument("--output", type=str, help="Path to the output directory to save processed images.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration if available.")
    parser.add_argument("--mode", type=str, choices=["full", "luminance"], default="full", help="Satellite removal mode: 'full' or 'luminance'.")
    parser.add_argument("--batch", action="store_true", help="Enable batch processing mode for the input directory.")
    parser.add_argument("--monitor", action="store_true", help="Enable live folder monitoring for new files.")
    parser.add_argument("--clip-trail", action="store_true", help="Clip satellite trail to 0.000 after processing.")

    args = parser.parse_args()

    # Declare global before assignment

    GLOBAL_CLIP_TRAIL = args.clip_trail  # Update based on the parsed argument

    app = QApplication([])  # Initialize the Qt application

    if args.input and args.output:
        try:
            input_dir = args.input
            output_dir = args.output
            use_gpu = args.use_gpu
            satellite_mode = args.mode

            print(f"Input Directory: {input_dir}")
            print(f"Output Directory: {output_dir}")
            print(f"Use GPU: {use_gpu}")
            print(f"Satellite Mode: {satellite_mode}")
            print(f"Clip Trail: {GLOBAL_CLIP_TRAIL}")

            # Load models
            models = load_models(exe_dir, use_gpu)

            if args.batch:
                print("Starting batch processing in progress monitor...")

                # Create and show progress monitor
                progress_dialog = ProgressMonitorDialog()
                progress_dialog.show()

                # Define log and progress update functions
                def log_message(msg):
                    progress_dialog.log_message(msg)

                def update_progress(value):
                    progress_dialog.update_progress(value)

                # Create and start batch processing
                batch_processor = BatchProcessor(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    models=models,
                    satellite_mode=satellite_mode
                )
                batch_processor.log_signal.connect(log_message)
                batch_processor.progress_signal.connect(update_progress)
                batch_processor.processing_complete.connect(progress_dialog.close)
                batch_processor.start()

                app.exec_()  # Execute the Qt event loop

            elif args.monitor:
                print("Starting live folder monitoring...")
                folder_monitor = FolderMonitor(input_dir, output_dir, models, satellite_mode)
                folder_monitor.log_signal.connect(lambda msg: print(f"[LOG]: {msg}"))
                folder_monitor.run()
            else:
                print("Error: Please specify --batch or --monitor.")
        except Exception as e:
            print(f"Error during headless execution: {e}")
    else:
        print("No arguments provided. Launching GUI...")
        dialog = SatelliteToolDialog()
        dialog.exec_()
