# PyQt6 Imports
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QCheckBox, QFileDialog, QLineEdit, QTextEdit, QStyle, QProgressBar, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon

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
try:
    import av, logging
    class _AVLogShim:
        DEBUG   = logging.DEBUG
        INFO    = logging.INFO
        WARNING = logging.WARNING
        ERROR   = logging.ERROR

        @staticmethod
        def set_level(level):
            # no‐op
            return

    av.logging = _AVLogShim
except ImportError:
    pass
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import lz4.block
import zstandard
from torchvision.models import MobileNet_V2_Weights
import torch.sparse
import torch._C
import onnxruntime as ort

import rawpy

#torch.cuda.is_available = lambda: False

# Custom Imports
from xisf import XISF
from skimage.transform import resize

#import argparse  # For command-line argument parsing

GLOBAL_CLIP_TRAIL = True  # Default to False
GLOBAL_SKIP_SAVE = False

class BinaryClassificationCNN2(nn.Module):
    def __init__(self, input_channels=3):
        super(BinaryClassificationCNN2, self).__init__()
        
        # ---------------------------------------------------------
        # (A) Two initial "custom" convolutional layers
        # ---------------------------------------------------------
        # First custom convolutional layer
        self.pre_conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Second custom convolutional layer
        self.pre_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # ---------------------------------------------------------
        # (B) MobileNetV2 backbone
        # ---------------------------------------------------------
        # Load pretrained MobileNetV2
        self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # By default, MobileNetV2 expects (N, 3, H, W). But after our pre_conv2, we have 64 channels. 
        # We need to override MobileNetV2's first convolution to handle 64 -> 32.
        # mobilenet.features[0] is a Sequential of (Conv2d, BatchNorm2d, ReLU6).
        # So we replace the *first Conv2d* layer to accept 64 channels:
        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # ---------------------------------------------------------
        # (C) Replace the final classification layer
        # ---------------------------------------------------------
        # The final classifier (mobilenet.classifier) is typically:
        #   (Dropout(...), Linear(in_features=1280, out_features=1000, ...))
        # We want just 1 output for binary classification.
        in_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(in_features, 1)
        
    def forward(self, x):
        # Pass input through your custom convolutional layers
        x = self.pre_conv1(x)   # => shape (N, 32, H, W)
        x = self.pre_conv2(x)   # => shape (N, 64, H, W)
        
        # Pass through MobileNetV2 backbone (which now starts with a 64->32 conv)
        x = self.mobilenet(x)
        
        return x

# Define the Binary Classification Model
class BinaryClassificationCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(BinaryClassificationCNN, self).__init__()
        
        # First custom convolutional layer
        self.pre_conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Second custom convolutional layer
        self.pre_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # ResNet18 backbone
        self.features = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the first layer of ResNet18 to match our second custom conv layer
        self.features.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the fully connected layer for binary classification
        self.features.fc = nn.Linear(self.features.fc.in_features, 1)
        
    def forward(self, x):
        # Pass input through custom convolutional layers
        x = self.pre_conv1(x)
        x = self.pre_conv2(x)
        
        # Pass through ResNet18 backbone
        x = self.features(x)
        
        return x


# Define the ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

# Updated SharpeningCNN with Residual Blocks
class SharpeningCNN(nn.Module):
    def __init__(self):
        super(SharpeningCNN, self).__init__()
        
        # Encoder (down-sampling path)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 1st layer (3 -> 16 feature maps)
            nn.ReLU(),
            ResidualBlock(16)  # Replaced Conv2d + ReLU with ResidualBlock
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 2nd layer (16 -> 32 feature maps)
            nn.ReLU(),
            ResidualBlock(32)  # Replaced Conv2d + ReLU with ResidualBlock
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),  # 3rd layer (32 -> 64) with dilation
            nn.ReLU(),
            ResidualBlock(64)  # Replaced Conv2d + ReLU with ResidualBlock
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 4th layer (64 -> 128 feature maps)
            nn.ReLU(),
            ResidualBlock(128)  # Replaced Conv2d + ReLU with ResidualBlock
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2, dilation=2),  # 5th layer (128 -> 256) with dilation
            nn.ReLU(),
            ResidualBlock(256)  # Replaced Conv2d + ReLU with ResidualBlock
        )
        
        # Decoder (up-sampling path with skip connections)
        self.decoder5 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),  # 256 + 128 feature maps from encoder4
            nn.ReLU(),
            ResidualBlock(128)  # Replaced Conv2d + ReLU with ResidualBlock
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),  # 128 + 64 feature maps from encoder3
            nn.ReLU(),
            ResidualBlock(64)  # Replaced Conv2d + ReLU with ResidualBlock
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),  # 64 + 32 feature maps from encoder2
            nn.ReLU(),
            ResidualBlock(32)  # Replaced Conv2d + ReLU with ResidualBlock
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1),  # 32 + 16 feature maps from encoder1
            nn.ReLU(),
            ResidualBlock(16)  # Replaced Conv2d + ReLU with ResidualBlock
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # Output layer (16 -> 3 channels for RGB output)
            nn.Sigmoid()  # Ensure output values are between 0 and 1
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # First encoding block
        e2 = self.encoder2(e1)  # Second encoding block
        e3 = self.encoder3(e2)  # Third encoding block
        e4 = self.encoder4(e3)  # Fourth encoding block
        e5 = self.encoder5(e4)  # Fifth encoding block
        
        # Decoder with skip connections
        d5 = self.decoder5(torch.cat([e5, e4], dim=1))  # Concatenate with encoder4 output
        d4 = self.decoder4(torch.cat([d5, e3], dim=1))  # Concatenate with encoder3 output
        d3 = self.decoder3(torch.cat([d4, e2], dim=1))  # Concatenate with encoder2 output
        d2 = self.decoder2(torch.cat([d3, e1], dim=1))  # Concatenate with encoder1 output
        d1 = self.decoder1(d2)  # Final output layer

        return d1


# Get the directory of the executable or the script location
exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

cached_models = None

def load_model_weights(model, checkpoint_path, device):
    """
    Load the weights from a checkpoint into the model, ignoring extra keys.

    Args:
        model (torch.nn.Module): The PyTorch model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine the state_dict based on checkpoint structure
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle prefix differences (e.g., "module.")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Compare model state dict and checkpoint keys
    model_state_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model_state_dict and model_state_dict[k].shape == v.shape
    }

    # Print debug information for mismatches
    missing_keys = [k for k in model_state_dict.keys() if k not in filtered_state_dict]
    unexpected_keys = [k for k in state_dict.keys() if k not in model_state_dict]

    if missing_keys:
        print(f"Missing keys in loaded state_dict: {missing_keys[:10]}...")
    if unexpected_keys:
        print(f"Unexpected keys in checkpoint: {unexpected_keys[:10]}...")

    # Load the weights
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Loaded {len(filtered_state_dict)}/{len(model_state_dict)} layers successfully.")
    return model


def load_models(exe_dir, use_gpu=True):
    """
    Load models with separate handling for CUDA, DirectML, and CPU.

    Args:
        exe_dir (str): Path to the executable directory.
        use_gpu (bool): Whether to use GPU acceleration.

    Returns:
        dict: A dictionary containing the loaded models and their configurations.
    """
    global cached_models
    device = None

    if torch.cuda.is_available() and use_gpu:
        # Use CUDA
        if cached_models:
            return cached_models

        device = torch.device("cuda")
        print(f"Using device: {device} (CUDA)")

        # Load primary detection model (ResNet18-based)
        detection_model1 = BinaryClassificationCNN(input_channels=3).to(device)
        detection_model1 = load_model_weights(
            detection_model1,
            os.path.join(exe_dir, "satellite_trail_detector_AI3.pth"),
            device
        )
        detection_model1.eval()

        # Load secondary detection model (MobileNetV2-based)
        detection_model2 = BinaryClassificationCNN2(input_channels=3).to(device)
        detection_model2 = load_model_weights(
            detection_model2,
            os.path.join(exe_dir, "satellite_trail_detector_mobilenetv2.pth"),
            device
        )
        detection_model2.eval()

        # Load removal model
        removal_model = SharpeningCNN().to(device)
        removal_model = load_model_weights(
            removal_model,
            os.path.join(exe_dir, "satelliteremovalAI3.pth"),
            device
        )
        removal_model.eval()

        # Cache and return models
        cached_models = {
            "detection_model1": detection_model1,
            "detection_model2": detection_model2,
            "removal_model": removal_model,
            "device": device,
            "is_onnx": False,
        }
        return cached_models

    elif "DmlExecutionProvider" in ort.get_available_providers() and use_gpu:
        # Use DirectML for ONNX models
        print("Using DirectML for ONNX Runtime.")

        device = "DirectML"  # Representing device as a string for DirectML
        detection_model1 = ort.InferenceSession(
            os.path.join(exe_dir, "satellite_trail_detector_AI3.onnx"),
            providers=["DmlExecutionProvider"]
        )
        detection_model2 = ort.InferenceSession(
            os.path.join(exe_dir, "satellite_trail_detector_mobilenetv2.onnx"),
            providers=["DmlExecutionProvider"]
        )
        removal_model = ort.InferenceSession(
            os.path.join(exe_dir, "satelliteremovalAI3.onnx"),
            providers=["DmlExecutionProvider"]
        )

        return {
            "detection_model1": detection_model1,
            "detection_model2": detection_model2,
            "removal_model": removal_model,
            "device": device,
            "is_onnx": True,
        }

    else:
        # Use CPU
        print("No GPU acceleration available. Using CPU.")

        device = torch.device("cpu")

        # Load primary detection model (ResNet18-based)
        detection_model1 = BinaryClassificationCNN(input_channels=3).to(device)
        detection_model1 = load_model_weights(
            detection_model1,
            os.path.join(exe_dir, "satellite_trail_detector_AI3.pth"),
            device
        )
        detection_model1.eval()

        # Load secondary detection model (MobileNetV2-based)
        detection_model2 = BinaryClassificationCNN2(input_channels=3).to(device)
        detection_model2 = load_model_weights(
            detection_model2,
            os.path.join(exe_dir, "satellite_trail_detector_mobilenetv2.pth"),
            device
        )
        detection_model2.eval()

        # Load removal model
        removal_model = SharpeningCNN().to(device)
        removal_model = load_model_weights(
            removal_model,
            os.path.join(exe_dir, "satelliteremovalAI3.pth"),
            device
        )
        removal_model.eval()

        return {
            "detection_model1": detection_model1,
            "detection_model2": detection_model2,
            "removal_model": removal_model,
            "device": device,
            "is_onnx": False,
        }



def load_detection_model(model_path, device, is_onnx=False):
    """
    Load the pre-trained satellite detection model.

    Args:
        model_path (str): Path to the detection model.
        device (torch.device): Device to load the model onto.
        is_onnx (bool): Whether to use ONNX Runtime for DirectML.

    Returns:
        torch.nn.Module or ort.InferenceSession: The detection model.
    """
    if is_onnx:
        # Load ONNX model for DirectML
        return ort.InferenceSession(model_path, providers=["DmlExecutionProvider"])

    # Load PyTorch detection model
    detection_model = BinaryClassificationCNN(input_channels=3).to(device)
    detection_model.load_state_dict(torch.load(model_path, map_location=device))
    detection_model.eval().to(device)
    print(f"Detection model loaded onto device: {device}")
    return detection_model

def load_detection_model2(model_path, device, is_onnx=False):
    """
    Load the pre-trained satellite detection model.

    Args:
        model_path (str): Path to the detection model.
        device (torch.device): Device to load the model onto.
        is_onnx (bool): Whether to use ONNX Runtime for DirectML.

    Returns:
        torch.nn.Module or ort.InferenceSession: The detection model.
    """
    if is_onnx:
        # Load ONNX model for DirectML
        return ort.InferenceSession(model_path, providers=["DmlExecutionProvider"])

    # Load PyTorch detection model
    detection_model2 = BinaryClassificationCNN2(input_channels=3).to(device)
    detection_model2.load_state_dict(torch.load(model_path, map_location=device))
    detection_model2.eval().to(device)
    print(f"Detection model loaded onto device: {device}")
    return detection_model2


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

    def __init__(self, input_dir, output_dir, models, satellite_mode, clip_trail=None, skip_save=None, sensitivity=0.1):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.models = models
        self.satellite_mode = satellite_mode
        self.running = True
        self.is_onnx = models.get("is_onnx", False)
        self.processed_files = set()  # Initialize processed files list
        self.clip_trail = clip_trail if clip_trail is not None else GLOBAL_CLIP_TRAIL  # Use the provided or global variable
        self.skip_save = skip_save if skip_save is not None else GLOBAL_SKIP_SAVE
        self.sensitivity = sensitivity  # Store the sensitivity value

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
        self.log_message("Live monitoring started...")
        self.processed_files = set([
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(('.png', '.tif', '.tiff', '.fit', '.fits', '.xisf', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef'))
        ])

        while self.running:
            try:
                # List all eligible files in the input directory
                files = [
                    f for f in os.listdir(self.input_dir)
                    if f.lower().endswith(('.png', '.tif', '.tiff', '.fit', '.fits', '.xisf', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef'))
                ]
                # Filter for files that haven't been processed yet
                new_files = [f for f in files if f not in self.processed_files]

                for new_file in new_files:
                    # Skip files with the "_satellited" suffix to avoid reprocessing
                    if "_satellited" in new_file:
                        self.processed_files.add(new_file)
                        continue

                    file_path = os.path.join(self.input_dir, new_file)
                    self.log_signal.emit(f"Detected new file: {new_file}")

                    # Add a 5-second delay before processing
                    self.log_signal.emit(f"Waiting 5 seconds to ensure the file is fully saved...")
                    time.sleep(5)

                    try:
                        # Process the new file
                        satellited_image, original_header, bit_depth, file_extension, is_mono, file_meta, image_meta, trail_detected_in_image = satellite_image(
                            file_path,
                            self.models['device'],
                            self.models['detection_model1'],  # Primary detection model
                            self.models['detection_model2'],  # Secondary detection model
                            self.models['removal_model'],
                            self.satellite_mode,
                            sensitivity=self.sensitivity,  # Pass sensitivity
                            log_signal=self.log_signal,
                            progress_signal=self.progress_signal,
                            is_onnx=self.is_onnx
                        )


                        # Handle the Skip Save logic based on trail detection and user preference
                        if not trail_detected_in_image and self.skip_save:
                            self.log_signal.emit(f"Skipped: No satellite trail detected in {new_file}")
                            self.processed_files.add(new_file)
                            continue

                        if satellited_image is None:  # This case should not occur due to checks in satellite_image
                            self.log_signal.emit(f"Failed to process: {new_file}")
                            self.processed_files.add(new_file)
                            continue

                        # Save the processed image
                        output_image_name = os.path.splitext(new_file)[0] + "_satellited" + file_extension
                        output_file = os.path.join(self.output_dir, output_image_name)
                        save_processed_image(
                            satellited_image,
                            output_file,
                            file_extension,
                            bit_depth,
                            original_header,
                            is_mono,
                            file_meta,
                            image_meta
                        )
                        self.log_signal.emit(f"Processed and saved: {output_image_name}")

                        # Mark the output file as processed
                        self.processed_files.add(output_image_name)

                    except Exception as e:
                        self.log_signal.emit(f"Error processing file {new_file}: {e}")

                    # Mark the input file as processed
                    self.processed_files.add(new_file)

                time.sleep(2)  # Add a short delay to avoid rapid file checks
            except Exception as e:
                self.log_signal.emit(f"Error monitoring folder: {e}")



class SatelliteToolDialog(QDialog):
    log_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmic Clarity Satellite Removal Tool V2.4 AI3")

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
        self.sensitivity = 0.1  # Default value               
 

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
        self.clip_trail_checkbox.setChecked(True)  # Default unchecked
        self.clip_trail_checkbox.stateChanged.connect(self.update_clip_trail)
        layout.addWidget(self.clip_trail_checkbox)

        # Sensitivity Slider
        sensitivity_layout = QHBoxLayout()
        sensitivity_label = QLabel("Clipping Sensitivity (Smaller Value more aggressive clipping):")
        self.sensitivity_value_label = QLabel(f"{self.sensitivity:.2f}")
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)

        self.sensitivity_slider.setMinimum(1)   # Represents 0.01
        self.sensitivity_slider.setMaximum(50)  # Represents 0.5
        self.sensitivity_slider.setValue(int(self.sensitivity * 100))  # e.g., 0.1 * 100 = 10
        self.sensitivity_slider.setTickInterval(1)
        self.sensitivity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)

        sensitivity_layout.addWidget(sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        sensitivity_layout.addWidget(self.sensitivity_value_label)
        layout.addLayout(sensitivity_layout)        

        # Add "Skip Saving if No Trail Detected" checkbox
        self.skip_save_checkbox = QCheckBox("Skip Saving if No Trail Detected")
        self.skip_save_checkbox.setChecked(False)  # Default to not skipping
        self.skip_save_checkbox.stateChanged.connect(self.update_skip_save)
        layout.addWidget(self.skip_save_checkbox)


        # Control buttons
        self.process_button = QPushButton("Batch Process Input Folder")
        self.process_button.clicked.connect(self.process_input_folder)        

        # Live monitor controls in a horizontal layout
        monitor_layout = QHBoxLayout()

        # Start Live Monitor button with green light icon
        self.live_monitor_button = QPushButton("Start Live Monitor")
        self.live_monitor_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.live_monitor_button.clicked.connect(self.start_live_monitor)

        # Stop Live Monitor button with stop icon
        self.stop_monitor_button = QPushButton("Stop Live Monitor")
        self.stop_monitor_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton))
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
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter
)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(footer_label)

        self.setLayout(layout)

    def update_sensitivity(self, value):
        """
        Update the sensitivity value based on the slider position.
        """
        self.sensitivity = value / 100.0  # Convert integer back to float (0.01 to 0.5)
        self.sensitivity_value_label.setText(f"{self.sensitivity:.2f}")

    def update_skip_save(self, state):
        global GLOBAL_SKIP_SAVE
        GLOBAL_SKIP_SAVE = state == Qt.CheckState.Checked

    def update_clip_trail(self, state):
        set_clip_trail(state == Qt.CheckState.Checked)

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
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            models=self.models,
            satellite_mode=satellite_mode,
            clip_trail=self.clip_trail_checkbox.isChecked(),
            skip_save=self.skip_save_checkbox.isChecked(),
            sensitivity=self.sensitivity
        )
        self.live_monitor_thread.log_signal.connect(self.log_message)
        self.live_monitor_thread.progress_signal.connect(self.update_progress_bar)
        self.live_monitor_thread.start()

        self.sensitivity_slider.setEnabled(False)
        self.log_message("Sensitivity slider disabled during live monitoring.")

        self.live_monitor_button.setEnabled(False)
        self.stop_monitor_button.setEnabled(True)

    def stop_live_monitor(self):
        if self.live_monitor_thread and self.live_monitor_thread.isRunning():
            self.live_monitor_thread.stop()
            self.live_monitor_thread.wait()
            self.live_monitor_button.setEnabled(True)
            self.stop_monitor_button.setEnabled(False)
            self.log_message("Live monitor stopped.")
            # **Re-enable the sensitivity slider**
            self.sensitivity_slider.setEnabled(True)
            self.log_message("Sensitivity slider re-enabled after stopping live monitoring.")            

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
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            models=self.models,
            satellite_mode=self.mode_dropdown.currentText(),
            clip_trail=self.clip_trail_checkbox.isChecked(),
            skip_save=self.skip_save_checkbox.isChecked(),
            sensitivity=self.sensitivity
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
    if dialog.exec() == QDialog.Accepted:
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
    Perform a linear stretch on the image with unlinked channels for all images.
    """
    original_min = np.min(image)
    stretched_image = image - original_min  # Shift image so that the min is 0

    # Capture the original medians before any further processing
    original_medians = []
    for c in range(3):  # Assume 3-channel input
        channel_median = np.median(stretched_image[..., c])
        original_medians.append(channel_median)

    # Define the target median for stretching
    target_median = 0.25

    # Apply the stretch for each channel
    for c in range(3):
        channel_median = original_medians[c]
        if channel_median != 0:
            stretched_image[..., c] = ((channel_median - 1) * target_median * stretched_image[..., c]) / (
                channel_median * (target_median + stretched_image[..., c] - 1) - target_median * stretched_image[..., c]
            )

    # Clip stretched image to [0, 1] range
    stretched_image = np.clip(stretched_image, 0, 1)

    # Return the stretched image, original min, and original medians
    return stretched_image, original_min, original_medians



# Function to unstretch an image
def unstretch_image(image, original_medians, original_min):
    """
    Undo the stretch to return the image to the original linear state with unlinked channels.
    Handles both single-channel and 3-channel images.
    """
    was_single_channel = False  # Local flag to check if the image was originally mono



    # Check if the image is single-channel
    if image.ndim == 3 and image.shape[2] == 1:
        was_single_channel = True  # Mark the image as originally single-channel
        image = np.repeat(image, 3, axis=2)  # Convert to 3-channel by duplicating

    elif image.ndim == 2:  # True single-channel case
        was_single_channel = True
        image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel by duplicating

    # Process each channel independently
    for c in range(3):  # Assume 3-channel input at this point
        channel_median = np.median(image[..., c])
        if channel_median != 0 and original_medians[c] != 0:

            image[..., c] = ((channel_median - 1) * original_medians[c] * image[..., c]) / (
                channel_median * (original_medians[c] + image[..., c] - 1) - original_medians[c] * image[..., c]
            )

        else:
            print(f"Channel {c} - Skipping unstretch due to zero median.")

    image += original_min  # Add back the original minimum
    image = np.clip(image, 0, 1)  # Clip to [0, 1] range

    # If the image was originally single-channel, convert back to single-channel
    if was_single_channel:
        image = np.mean(image, axis=2, keepdims=True)  # Convert back to single-channel



    return image



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

def set_clip_trail(value):
    global GLOBAL_CLIP_TRAIL
    GLOBAL_CLIP_TRAIL = value




# Function to satellite the image
def satellite_image(image_path, device, detection_model1, detection_model2, removal_model, satellite_mode='luminance', sensitivity=0.1, progress_signal=None, log_signal=None, is_onnx=False):
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
    if file_extension not in ['.png', '.tif', '.tiff', '.fit', '.fits', '.xisf', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef', 'jpg', 'jpeg']:
        print(f"Ignoring non-image file: {image_path}")
        return None, None, None, None, None, None, None, None

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
                        image = image.astype(np.float32)
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
                        image = image.astype(np.float32)
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

        elif file_extension in ['.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef']:
            try:
                with rawpy.imread(image_path) as raw:
                    # Get the raw Bayer data
                    bayer_image = raw.raw_image_visible.astype(np.float32)
                    print(f"Raw Bayer image dtype: {bayer_image.dtype}, min: {bayer_image.min()}, max: {bayer_image.max()}")

                    # Normalize Bayer data to [0, 1]
                    bayer_image /= bayer_image.max()
                    print(f"Normalized Bayer image min: {bayer_image.min()}, max: {bayer_image.max()}")

                    # Convert Bayer data to a 3D array (for consistency with your pipeline)
                    image = np.stack([bayer_image] * 3, axis=-1)
                    bit_depth = "16-bit"  # Assuming 16-bit raw data
                    is_mono = True

                    # Populate `original_header` with RAW metadata
                    original_header_dict = {
                        'CAMERA': raw.camera_whitebalance[0] if raw.camera_whitebalance else 'Unknown',
                        'EXPTIME': raw.shutter if hasattr(raw, 'shutter') else 0.0,
                        'ISO': raw.iso_speed if hasattr(raw, 'iso_speed') else 0,
                        'FOCAL': raw.focal_len if hasattr(raw, 'focal_len') else 0.0,
                        'DATE': raw.timestamp if hasattr(raw, 'timestamp') else 'Unknown',
                    }

                    # Extract CFA pattern
                    cfa_pattern = raw.raw_colors_visible
                    cfa_mapping = {
                        0: 'R',  # Red
                        1: 'G',  # Green
                        2: 'B',  # Blue
                    }
                    cfa_description = ''.join([cfa_mapping.get(color, '?') for color in cfa_pattern.flatten()[:4]])

                    # Add CFA pattern to header
                    original_header_dict['CFA'] = (cfa_description, 'Color Filter Array pattern')

                    # Convert original_header_dict to fits.Header
                    original_header = fits.Header()
                    for key, value in original_header_dict.items():
                        original_header[key] = value

            except Exception as e:
                print(f"Error reading RAW file: {image_path}, {e}")
                return None, None, None, None, None, None, None, None
        


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

        #check if stretch is needed
        stretch_needed = np.median(image_with_border - np.min(image_with_border)) < 0.05
        
        if stretch_needed:
            print(f"Normalizing Linear Data")
            # Stretch the image
            stretched_image, original_min, original_medians = stretch_image(image_with_border)
        else:
            # If no stretch is needed, use the original image directly
            stretched_image = image_with_border
            original_min = np.min(image_with_border)
            original_medians = [np.median(image_with_border[..., c]) for c in range(3)] if image_with_border.ndim == 3 else [np.median(image_with_border)]

        

        # Process mono or color images
        if is_mono:
            # Convert mono image to 3 channels by stacking
            stretched_image = np.stack([stretched_image[:, :, 0]] * 3, axis=-1)

        if satellite_mode == 'luminance' and is_mono:
            # Extract luminance (single-channel processing for mono)
            satellited_image, trail_detected_in_image = satellite_channel(
                image=stretched_image,
                device=device,
                removal_model=removal_model,
                detection_model1=detection_model1,
                detection_model2=detection_model2,
                is_mono=is_mono,
                sensitivity=sensitivity,
                progress_signal=progress_signal,
                is_onnx=is_onnx
            )
            satellited_image = satellited_image[:, :, np.newaxis]  # Convert back to single channel

        elif satellite_mode == 'luminance' and not is_mono:
            # Extract luminance and process only that channel
            luminance, cb_channel, cr_channel = extract_luminance(stretched_image)
            satellited_image, trail_detected_in_image = satellite_channel(
                image=stretched_image,
                device=device,
                removal_model=removal_model,
                detection_model1=detection_model1,
                detection_model2=detection_model2,
                is_mono=is_mono,
                sensitivity=sensitivity,
                progress_signal=progress_signal,
                is_onnx=is_onnx
            )
            satellited_image = merge_luminance(satellited_image, cb_channel, cr_channel)

        else:  # Full RGB denoising mode
            # Process the 3-channel image directly
            satellited_image, trail_detected_in_image = satellite_channel(
                image=stretched_image,
                device=device,
                removal_model=removal_model,
                detection_model1=detection_model1,
                detection_model2=detection_model2,
                is_mono=is_mono,
                sensitivity=sensitivity,
                progress_signal=progress_signal,
                is_onnx=is_onnx
            )

        # Handle skipping saving if no trail detected
        if not trail_detected_in_image :
            log_message(f"No satellite trail detected in {image_path}. Proceeding with unmodified image.")
            satellited_image = stretched_image  # Use the stretched (or original) image as the output  

        # Unstretch the image
        if stretch_needed:
            print(f"De-normalizing linear data")
            satellited_image = unstretch_image(satellited_image, original_medians, original_min)       

        # Apply clip trail logic based on the `clip_trail` flag
        if GLOBAL_CLIP_TRAIL:
            min_value = np.min(satellited_image)
            satellited_image = np.where(satellited_image == min_value, 0.000, satellited_image)

        # Remove the border added around the image
        satellited_image = replace_border(original_image, satellited_image, border_size=16)

        assert satellited_image.dtype == np.float32, f"Image dtype is {satellited_image.dtype}, expected np.float32"

        log_message(f"Image {image_path} processed successfully.")
        return satellited_image, original_header, bit_depth, file_extension, is_mono, file_meta, image_meta, trail_detected_in_image

    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None, None, None, None, None, None, None


 
def satellite_channel(
    image,
    device,
    removal_model,
    detection_model1,
    detection_model2,
    is_mono=False,
    sensitivity=0.1,
    progress_signal=None,
    is_onnx=False
):
    chunk_size = 256
    overlap = 64
    chunks = split_image_into_chunks_with_overlap(image, chunk_size=chunk_size, overlap=overlap)
    total_chunks = len(chunks)
    satellited_chunks = []
    trail_detected_in_image = False  # New flag to track trail detection
    global GLOBAL_CLIP_TRAIL  # Access the global variable   

    for idx, (chunk, i, j) in enumerate(chunks):
        original_chunk = chunk.copy()

        # Detection
        if is_onnx:
            trail_detected = contains_trail_with_onnx(chunk, detection_model1)  # Modify if needed for ONNX
        else:
            trail_detected = contains_trail_with_ai(chunk, detection_model1, detection_model2, device)

        if trail_detected:
            trail_detected_in_image = True  # Set the flag if any chunk contains a trail

            # Prepare the chunk tensor
            if is_onnx:
                chunk_tensor = prepare_chunk_for_onnx(chunk)
            else:
                chunk_tensor = prepare_chunk_for_torch(chunk, device)

            # Run the removal model
            if is_onnx:
                processed_chunk = run_onnx_removal(chunk_tensor, removal_model, chunk.shape)
            else:
                processed_chunk = run_torch_removal(chunk_tensor, removal_model)

            final_chunk = apply_clip_trail_logic(processed_chunk, original_chunk, sensitivity) if GLOBAL_CLIP_TRAIL else processed_chunk
            satellited_chunks.append((final_chunk, i, j))
        else:
            satellited_chunks.append((chunk, i, j))

        # Update progress
        progress_percentage = (idx + 1) / total_chunks * 100
        if progress_signal:
            progress_signal.emit(progress_percentage)
        QApplication.processEvents()

    # Finalize by setting the progress bar to 100%
    if progress_signal:
        progress_signal.emit(100.0)

    # Stitch the chunks back together
    satellited_image = stitch_chunks_ignore_border(
        satellited_chunks, image.shape, chunk_size=chunk_size, overlap=overlap
    )
    return satellited_image, trail_detected_in_image




def contains_trail_with_onnx(chunk, detection_model):
    resized_chunk = resize(chunk, (256, 256, chunk.shape[2]), mode='reflect', anti_aliasing=True)
    input_tensor = np.transpose(resized_chunk, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    output = detection_model.run(None, {detection_model.get_inputs()[0].name: input_tensor})[0]
    return output[0] > 0.5  # Return True if trail is detected

def run_onnx_removal(chunk_tensor, removal_model, original_shape):
    output = removal_model.run(None, {removal_model.get_inputs()[0].name: chunk_tensor})[0]
    processed_chunk = np.transpose(output.squeeze(0), (1, 2, 0))
    return resize(processed_chunk, original_shape, mode='reflect', anti_aliasing=True)



def contains_trail_with_ai(chunk, detection_model1, detection_model2, device):
    """
    Detect satellite trails in a chunk using two detection models.

    Args:
        chunk (numpy.ndarray): Image chunk to analyze.
        detection_model1 (torch.nn.Module): Primary pre-trained detection model.
        detection_model2 (torch.nn.Module): Secondary pre-trained detection model.
        device (torch.device): Device to run the models on.

    Returns:
        bool: True if a trail is detected by both models, False otherwise.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # Ensure input size matches the models
    ])
    input_tensor = transform(chunk).unsqueeze(0).to(device)

    # Run the primary detection model
    with torch.no_grad():
        output1 = detection_model1(input_tensor).item()

    if output1 > 0.5:
        # If primary model detects a trail, run the secondary model
        with torch.no_grad():
            output2 = detection_model2(input_tensor).item()
        # Both models must detect a trail
        return output2 > 0.5
    else:
        # Primary model did not detect a trail; skip secondary model
        return False


def run_torch_removal(chunk_tensor, removal_model):
    with torch.no_grad():
        processed_chunk = removal_model(chunk_tensor).squeeze().cpu().numpy()

    if chunk_tensor.ndim == 4:  # RGB chunk
        return processed_chunk.transpose(1, 2, 0)
    elif chunk_tensor.ndim == 3:  # Mono chunk
        return processed_chunk[0]

def prepare_chunk_for_torch(chunk, device):
    if chunk.ndim == 2:  # Mono chunk
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return chunk_tensor.repeat(1, 3, 1, 1)  # Convert mono to 3 channels
    elif chunk.shape[-1] == 1:  # Grayscale
        chunk_tensor = torch.tensor(chunk.squeeze(-1), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return chunk_tensor.repeat(1, 3, 1, 1)  # Convert mono to 3 channels
    else:  # RGB chunk
        return torch.tensor(chunk, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)


def prepare_chunk_for_onnx(chunk):
    resized_chunk = resize(chunk, (256, 256, chunk.shape[2]), mode='reflect', anti_aliasing=True)
    return np.transpose(resized_chunk, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

def apply_clip_trail_logic(processed_chunk, original_chunk, sensitivity=0.1):
    sattrail_only_image = original_chunk - processed_chunk
    mean_val = np.mean(sattrail_only_image)
    clipped_image = np.clip((sattrail_only_image - mean_val) * 10, 0, 1)
    binary_mask = np.where(clipped_image < sensitivity, 0, 1)
    return np.clip(original_chunk - binary_mask, 0, 1)



def save_processed_image(image, output_path, file_extension, bit_depth, original_header=None, is_mono=False, file_meta=None, image_meta=None):
    """Save the processed image to the appropriate format."""
    """Save the processed image to the appropriate format."""
    if file_extension in ['.fits', '.fit']:
        # ——— 1) Build a dummy HDU purely to auto-fix the header ———
        if original_header is not None:
            # Copy your original header into a PrimaryHDU
            fixer = fits.PrimaryHDU(header=original_header.copy())
            # Ask Astropy to fix *all* non-standard cards in place
            fixer.verify('fix')
            # Now grab back the cleaned header
            hdr = fixer.header
        else:
            hdr = None
        # ——— 2) Build the FITS data array ———
        if is_mono:
            # single-channel
            if bit_depth == "16-bit":
                data = (image[:, :, 0] * 65535).astype(np.uint16)
            elif bit_depth == "32-bit unsigned":
                bzero = hdr.get('BZERO', 0)
                bscale = hdr.get('BSCALE', 1)
                data = (image[:, :, 0].astype(np.float32) * bscale + bzero).astype(np.uint32)
            else:  # 32-bit float
                data = image[:, :, 0].astype(np.float32)
        else:
            # RGB → (3, Y, X)
            arr = np.transpose(image, (2, 0, 1))
            if bit_depth == "16-bit":
                data = (arr * 65535).astype(np.uint16)
            elif bit_depth == "32-bit unsigned":
                data = arr.astype(np.float32)
                if hdr is not None:
                    hdr['BITPIX'] = -32
            else:
                data = arr.astype(np.float32)

        # ——— 3) Fix up the NAXIS keywords if we have a header ———
        if hdr is not None:
            if is_mono:
                hdr['NAXIS']  = 2
                hdr['NAXIS1'] = data.shape[1]
                hdr['NAXIS2'] = data.shape[0]
            else:
                hdr['NAXIS']  = 3
                hdr['NAXIS1'] = data.shape[2]
                hdr['NAXIS2'] = data.shape[1]
                hdr['NAXIS3'] = data.shape[0]

        # ——— 4) Write it out, letting Astropy auto-fix any tiny remaining issues ———
        hdu  = fits.PrimaryHDU(data, header=hdr)
        hdul = fits.HDUList([hdu])
        hdul.writeto(output_path, overwrite=True, output_verify='fix')
        print(f"Saved FITS to: {output_path}")
        return

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

                shuffle=True
            )
            print(f"Saved {bit_depth} XISF image as RGB with metadata to: {output_path}")

        except Exception as e:
            print(f"Error saving XISF file: {e}")

    elif file_extension in ['.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef']:
        print("RAW formats are not writable. Saving as FITS instead.")
        output_path = output_path.rsplit('.', 1)[0] + ".fits"

        # Save as FITS file with metadata
        if original_header is not None:
            # Convert original_header (dictionary) to astropy Header object
            fits_header = fits.Header()
            for key, value in original_header.items():
                fits_header[key] = value
            fits_header['BSCALE'] = 1.0  # Scaling factor
            fits_header['BZERO'] = 0.0   # Offset for brightness    

            if is_mono:  # Grayscale FITS
                if bit_depth == "16-bit":
                    satellited_image_fits = (image[:, :, 0] * 65535).astype(np.uint16)
                elif bit_depth == "32-bit unsigned":
                    bzero = fits_header.get('BZERO', 0)
                    bscale = fits_header.get('BSCALE', 1)
                    satellited_image_fits = (image[:, :, 0].astype(np.float32) * bscale + bzero).astype(np.uint32)
                else:  # 32-bit float
                    satellited_image_fits = image[:, :, 0].astype(np.float32)

                # Update header for a 2D (grayscale) image
                fits_header['NAXIS'] = 2
                fits_header['NAXIS1'] = image.shape[1]  # Width
                fits_header['NAXIS2'] = image.shape[0]  # Height
                fits_header.pop('NAXIS3', None)  # Remove if present

                hdu = fits.PrimaryHDU(satellited_image_fits, header=fits_header)
            else:  # RGB FITS
                satellited_image_transposed = np.transpose(image, (2, 0, 1))  # Channels, Height, Width
                if bit_depth == "16-bit":
                    satellited_image_fits = (satellited_image_transposed * 65535).astype(np.uint16)
                elif bit_depth == "32-bit unsigned":
                    bzero = fits_header.get('BZERO', 0)
                    bscale = fits_header.get('BSCALE', 1)
                    satellited_image_fits = satellited_image_transposed.astype(np.float32) * bscale + bzero
                    fits_header['BITPIX'] = -32
                else:  # Default to 32-bit float
                    satellited_image_fits = satellited_image_transposed.astype(np.float32)

                # Update header for a 3D (RGB) image
                fits_header['NAXIS'] = 3
                fits_header['NAXIS1'] = satellited_image_transposed.shape[2]  # Width
                fits_header['NAXIS2'] = satellited_image_transposed.shape[1]  # Height
                fits_header['NAXIS3'] = satellited_image_transposed.shape[0]  # Channels

                hdu = fits.PrimaryHDU(satellited_image_fits, header=fits_header)

            # Write the FITS file
            try:
                hdu.writeto(output_path, overwrite=True)
                print(f"RAW processed and saved as FITS to: {output_path}")
            except Exception as e:
                print(f"Error saving FITS file: {e}")



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

    def __init__(self, input_dir, output_dir, models, satellite_mode, clip_trail=None, skip_save=None, sensitivity=0.1):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.models = models
        self.satellite_mode = satellite_mode
        self.is_onnx = models.get("is_onnx", False)
        self.clip_trail = clip_trail if clip_trail is not None else GLOBAL_CLIP_TRAIL
        self.skip_save = skip_save if skip_save is not None else GLOBAL_SKIP_SAVE
        self.sensitivity = sensitivity  # Store the sensitivity value

    def run(self):
        try:
            self.log_signal.emit("Starting batch processing...")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            files = [
                f for f in os.listdir(self.input_dir)
                if f.lower().endswith(('.png', '.tif', '.tiff', '.fit', '.fits', '.xisf', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef'))
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
                    satellited_image, original_header, bit_depth, file_extension, is_mono, file_meta, image_meta, trail_detected_in_image = satellite_image(
                        image_path,
                        self.models['device'],
                        self.models["detection_model1"],  # Primary detection model
                        self.models["detection_model2"],  # Secondary detection model
                        self.models["removal_model"],
                        self.satellite_mode,
                        sensitivity=self.sensitivity,  # Pass sensitivity
                        log_signal=self.log_signal,
                        progress_signal=self.progress_signal,
                        is_onnx=self.is_onnx
                    )



                    # Handle the case where no trail was detected
                    if satellited_image is None:  # Handle cases where image processing failed entirely
                        self.log_signal.emit(f"Failed to process: {image_name}")
                        continue

                    if not trail_detected_in_image and self.skip_save:
                        # Skip saving if no trail is detected and skip save is enabled
                        self.log_signal.emit(f"Skipped: No satellite trail detected in {image_name}")
                        continue

                    # Save the processed image, either modified or unmodified
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
    parser.add_argument("--skip-save", action="store_true", default=False, help="Skip saving images if no satellite trail is detected.")
    parser.add_argument("--sensitivity", type=float, default=0.1, help="Sensitivity for clip trail logic (0.01 to 0.5).")  # New argument

    # Introduce a mutually exclusive group for clipping
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--clip-trail", action="store_true", help="Clip satellite trail to 0.000 after processing.")
    group.add_argument("--no-clip-trail", action="store_false", dest="clip_trail", help="Do not clip satellite trail to 0.000 after processing.")

    # Set default for clip_trail
    parser.set_defaults(clip_trail=True)  # Default behavior is to clip

    args = parser.parse_args()

    # Validate sensitivity range
    if not 0.01 <= args.sensitivity <= 0.5:
        parser.error("The --sensitivity value must be between 0.01 and 0.5.")    

    # Declare global before assignment
    GLOBAL_CLIP_TRAIL = args.clip_trail  # Update based on the parsed argument
    GLOBAL_SKIP_SAVE = args.skip_save 
    GLOBAL_SENSITIVITY = args.sensitivity   

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
            print(f"Skip Save: {GLOBAL_SKIP_SAVE}")      
            print(f"Sensitivity: {GLOBAL_SENSITIVITY}")      

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
                    satellite_mode=satellite_mode,
                    clip_trail=GLOBAL_CLIP_TRAIL,
                    skip_save=GLOBAL_SKIP_SAVE,
                    sensitivity=GLOBAL_SENSITIVITY  # Pass sensitivity
                )
                batch_processor.log_signal.connect(log_message)
                batch_processor.progress_signal.connect(update_progress)
                batch_processor.processing_complete.connect(progress_dialog.close)
                batch_processor.start()

                app.exec()  # Execute the Qt event loop

            elif args.monitor:
                print("Starting live folder monitoring...")

                # Create and show progress monitor
                progress_dialog = ProgressMonitorDialog()
                progress_dialog.show()

                # Define log and progress update functions
                def log_message(msg):
                    progress_dialog.log_message(msg)

                def update_progress(value):
                    progress_dialog.update_progress(value)

                # Create and start folder monitoring
                folder_monitor = FolderMonitor(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    models=models,
                    satellite_mode=satellite_mode,
                    clip_trail=GLOBAL_CLIP_TRAIL,
                    skip_save=GLOBAL_SKIP_SAVE,
                    sensitivity=GLOBAL_SENSITIVITY  # Pass sensitivity
                )
                folder_monitor.log_signal.connect(log_message)
                folder_monitor.progress_signal.connect(update_progress)

                # Stop monitoring gracefully when the progress dialog is closed
                def stop_monitoring():
                    folder_monitor.stop()
                    print("Live monitoring stopped.")

                progress_dialog.finished.connect(stop_monitoring)

                folder_monitor.start()

                app.exec()  # Execute the Qt event loop

            else:
                print("Error: Please specify --batch or --monitor.")
        except Exception as e:
            print(f"Error during headless execution: {e}")
    else:
        print("No arguments provided. Launching GUI...")
        dialog = SatelliteToolDialog()
        dialog.exec()
