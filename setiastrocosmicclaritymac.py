import warnings
import os
import sys
from xisf import XISF
import lz4.block
import zstandard
import base64
import ast
import torch
import numpy as np
import torch.nn as nn
import tifffile as tiff
from astropy.io import fits
from PIL import Image
import tkinter as tk
from tkinter import simpledialog, messagebox
import argparse  # For command-line argument parsing
import time  # For simulating progress updates
from tkinter import ttk
from tkinter import filedialog

# Suppress model loading warnings
warnings.filterwarnings("ignore")

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

# Function to initialize and load models
def load_models(exe_dir, use_gpu=True):
    # Check if MPS is available (macOS Metal Performance Shaders), else use CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) backend for GPU acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU backend.")

    stellar_model_radius_1 = SharpeningCNN()
    nonstellar_model_radius_1 = SharpeningCNN()
    nonstellar_model_radius_2 = SharpeningCNN()
    nonstellar_model_radius_4 = SharpeningCNN()
    nonstellar_model_radius_8 = SharpeningCNN()

    # Load models onto the correct device
    stellar_model_radius_1.load_state_dict(torch.load(os.path.join(exe_dir, 'deep_sharp_stellar_cnn_AI3.pth'), map_location=device))
    nonstellar_model_radius_1.load_state_dict(torch.load(os.path.join(exe_dir, 'deep_nonstellar_sharp_cnn_radius_1AI3.pth'), map_location=device))
    nonstellar_model_radius_2.load_state_dict(torch.load(os.path.join(exe_dir, 'deep_nonstellar_sharp_cnn_radius_2AI3.pth'), map_location=device))
    nonstellar_model_radius_4.load_state_dict(torch.load(os.path.join(exe_dir, 'deep_nonstellar_sharp_cnn_radius_4AI3.pth'), map_location=device))
    nonstellar_model_radius_8.load_state_dict(torch.load(os.path.join(exe_dir, 'deep_nonstellar_sharp_cnn_radius_8AI3.pth'), map_location=device))
    # Set models to evaluation mode
    stellar_model_radius_1.eval()
    nonstellar_model_radius_1.eval()
    nonstellar_model_radius_2.eval()
    nonstellar_model_radius_4.eval()
    nonstellar_model_radius_8.eval()

    # Move models to the correct device
    stellar_model_radius_1.to(device)
    nonstellar_model_radius_1.to(device)
    nonstellar_model_radius_2.to(device)
    nonstellar_model_radius_4.to(device)
    nonstellar_model_radius_8.to(device)

    return {
        "stellar_model": stellar_model_radius_1,
        "nonstellar_model_1": nonstellar_model_radius_1,
        "nonstellar_model_2": nonstellar_model_radius_2,
        "nonstellar_model_4": nonstellar_model_radius_4,
        "nonstellar_model_8": nonstellar_model_radius_8,
        "device": device
    }

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




# Function to show progress during chunk processing
def show_progress(current, total):
    progress_percentage = (current / total) * 100
    print(f"Progress: {progress_percentage:.2f}% ({current}/{total} chunks processed)", end='\r')

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
    height, width = image.shape
    chunks = []
    step_size = chunk_size - overlap  # Define how much to step over (overlapping area)

    for i in range(0, height, step_size):
        for j in range(0, width, step_size):
            end_i = min(i + chunk_size, height)
            end_j = min(j + chunk_size, width)
            chunk = image[i:end_i, j:end_j]
            is_edge = i == 0 or j == 0 or (i + chunk_size >= height) or (j + chunk_size >= width)  # Check if on edge
            chunks.append((chunk, i, j, is_edge))  # Return chunk and its position, and whether it's an edge chunk
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
    
    for chunk, i, j, is_edge in chunks:
        actual_chunk_h, actual_chunk_w = chunk.shape
        
        # Calculate the boundaries for the current chunk, ignoring the border
        border_h = min(border_size, actual_chunk_h // 2)
        border_w = min(border_size, actual_chunk_w // 2)

        # Always ignore the 5-pixel border for all chunks, including edges
        inner_chunk = chunk[border_h:actual_chunk_h-border_h, border_w:actual_chunk_w-border_w]
        stitched_image[i+border_h:i+actual_chunk_h-border_h, j+border_w:j+actual_chunk_w-border_w] += inner_chunk
        weight_map[i+border_h:i+actual_chunk_h-border_h, j+border_w:j+actual_chunk_w-border_w] += 1

    # Normalize by weight_map to blend overlapping areas
    stitched_image /= np.maximum(weight_map, 1)  # Avoid division by zero
    return stitched_image

# Function to blend two images (before and after)
def blend_images(before, after, amount):
    return (1 - amount) * before + amount * after

# Function to interpolate the results for non-stellar sharpening based on strength
def interpolate_nonstellar_sharpening(sharpened_1, sharpened_2, sharpened_4, sharpened_8, strength):
    if strength <= 2:
        return blend_images(sharpened_1, sharpened_2, strength - 1)
    elif 2 < strength <= 4:
        return blend_images(sharpened_2, sharpened_4, (strength - 2) / 2)
    else:
        return blend_images(sharpened_4, sharpened_8, (strength - 4) / 4)

# Function to get user input with GUI
def get_user_input():
    # Define global variables to store the user input
    global use_gpu, sharpening_mode, nonstellar_strength, stellar_amount, separate_rgb, nonstellar_amount

    def on_submit():
        # Update the global variables with the user's selections
        global use_gpu, sharpening_mode, nonstellar_strength, stellar_amount, separate_rgb, nonstellar_amount
        use_gpu = gpu_var.get() == "Yes"
        sharpening_mode = mode_var.get()
        nonstellar_strength = nonstellar_strength_slider.get()
        stellar_amount = stellar_amount_slider.get() if sharpening_mode != "Non-Stellar Only" else None
        nonstellar_amount = nonstellar_amount_slider.get() if sharpening_mode != "Stellar Only" else None
        separate_rgb = separate_rgb_var.get() == "Yes"
        root.destroy()

    def update_sliders(*args):
        # Show or hide sliders based on the selected sharpening mode
        mode = mode_var.get()
        if mode == "Both":
            nonstellar_strength_label.pack()
            nonstellar_strength_slider.pack()
            stellar_amount_label.pack()
            stellar_amount_slider.pack()
            nonstellar_amount_label.pack()
            nonstellar_amount_slider.pack()
        elif mode == "Stellar Only":
            nonstellar_strength_label.pack_forget()
            nonstellar_strength_slider.pack_forget()
            stellar_amount_label.pack()
            stellar_amount_slider.pack()
            nonstellar_amount_label.pack_forget()
            nonstellar_amount_slider.pack_forget()
        elif mode == "Non-Stellar Only":
            nonstellar_strength_label.pack()
            nonstellar_strength_slider.pack()
            stellar_amount_label.pack_forget()
            stellar_amount_slider.pack_forget()
            nonstellar_amount_label.pack()
            nonstellar_amount_slider.pack()

        # Ensure the submit button is always the last widget to be packed
        submit_button.pack_forget()
        submit_button.pack(pady=20)

    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Cosmic Clarity Sharpening Tool")
    root.geometry("400x600")  # Set window size

    # GPU selection
    gpu_label = ttk.Label(root, text="Use GPU Acceleration:")
    gpu_label.pack(pady=5)
    gpu_var = tk.StringVar(value="Yes")
    gpu_dropdown = ttk.OptionMenu(root, gpu_var, "Yes", "Yes", "No")
    gpu_dropdown.pack()

    # Sharpening mode selection
    mode_label = ttk.Label(root, text="Sharpening Mode:")
    mode_label.pack(pady=5)
    mode_var = tk.StringVar(value="Both")
    mode_dropdown = ttk.OptionMenu(root, mode_var, "Both", "Both", "Stellar Only", "Non-Stellar Only")
    mode_dropdown.pack()

    # Bind the update function to the mode selection
    mode_var.trace('w', update_sliders)

    # Non-Stellar strength slider
    nonstellar_strength_label = ttk.Label(root, text="Non-Stellar Sharpening PSF (1-8):")
    nonstellar_strength_slider = tk.Scale(root, from_=1, to=8, orient="horizontal", resolution=.1)
    nonstellar_strength_slider.set(3)

    # Stellar amount slider (only if Stellar or Both are selected)
    stellar_amount_label = ttk.Label(root, text="Stellar Sharpening Amount (0-1):")
    stellar_amount_slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient="horizontal")
    stellar_amount_slider.set(0.5)  # Set default value to 0.5

    # Non-Stellar amount slider
    nonstellar_amount_label = ttk.Label(root, text="Non-Stellar Sharpening Amount (0-1):")
    nonstellar_amount_slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient="horizontal")
    nonstellar_amount_slider.set(0.5)  # Set default value to 0.5

    # Separate RGB channels checkbox
    separate_rgb_label = ttk.Label(root, text="Sharpen R, G, B Channels Separately:")
    separate_rgb_label.pack(pady=5)
    separate_rgb_var = tk.StringVar(value="No")
    separate_rgb_dropdown = ttk.OptionMenu(root, separate_rgb_var, "No", "Yes", "No")
    separate_rgb_dropdown.pack()

    # Submit button
    submit_button = ttk.Button(root, text="Submit", command=on_submit)

    # Call update_sliders initially to set the correct visibility
    update_sliders()

    root.mainloop()

    return use_gpu, sharpening_mode, nonstellar_strength, stellar_amount, separate_rgb, nonstellar_amount


# Function to show progress during chunk processing
def show_progress(current, total):
    progress_percentage = (current / total) * 100
    # Use \r to overwrite the same line
    print(f"\rProgress: {progress_percentage:.2f}% ({current}/{total} chunks processed)", end='', flush=True)

# Function to replace the 5-pixel border from the original image into the processed image
def replace_border(original_image, processed_image, border_size=16):
    # Ensure the dimensions of both images match
    if original_image.shape != processed_image.shape:
        # Resize processed_image to match the original_image dimensions
        processed_image = np.resize(processed_image, original_image.shape)
        print("Warning: Resized processed image to match original image dimensions.")

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
def add_border(image, border_size=16):
    median_value = np.median(image)
    if len(image.shape) == 2:
        return np.pad(image, ((border_size, border_size), (border_size, border_size)), 'constant', constant_values=median_value)
    else:
        return np.pad(image, ((border_size, border_size), (border_size, border_size), (0, 0)), 'constant', constant_values=median_value)

# Function to remove the border added around the image
def remove_border(image, border_size=16):
    if len(image.shape) == 2:
        return image[border_size:-border_size, border_size:-border_size]
    else:
        return image[border_size:-border_size, border_size:-border_size, :]

# Function to sharpen image
def sharpen_image(image_path, sharpening_mode, nonstellar_strength, stellar_amount, nonstellar_amount, device, models, sharpen_channels_separately):
    # Only proceed if the file extension is an image format we support
    file_extension = image_path.lower().split('.')[-1]
    if file_extension not in ['png', 'tif', 'tiff', 'fit', 'fits', 'xisf', 'jpg', 'jpeg']:
        print(f"Ignoring non-image file: {image_path}")
        return None, None, None, None, None, None  # Ignore and skip non-image files
    image = None
    file_extension = image_path.lower().split('.')[-1]
    is_mono = False  # Initialize is_mono as False by default   
    original_header = None  # Initialize header for FITS files 
    bit_depth = "32-bit floating point"  # Default bit depth to 32-bit floating point for safety
    file_meta, image_meta = None, None
    

    try:
        # Load and preprocess the image based on its format
        if file_extension in ['tif', 'tiff']:
            image = tiff.imread(image_path)
            print(f"Loaded TIFF image with dtype: {image.dtype}")
            if image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
                bit_depth = "16-bit"

            elif image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
                bit_depth = "8-bit"
            elif image.dtype == np.uint32:
                image = image.astype(np.float32) / 4294967295.0
                bit_depth = "32-bit unsigned"
            else:
                image.dtype == np.float32
                bit_depth = "32-bit floating point"  # If dtype is already float

            print(f"Final bit depth set to: {bit_depth}")    


            # Check if the image has an alpha channel and remove it if necessary
            if image.shape[-1] == 4:
                print("Detected alpha channel in TIFF. Removing it.")
                image = image[:, :, :3]  # Keep only the first 3 channels (RGB)
                print(f"Loaded image bit depth: {bit_depth}")

            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
                is_mono = True
                print(f"Loaded image bit depth: {bit_depth}")

        # Check if file extension is '.xisf'
        elif file_extension == 'xisf':
            # Load XISF file
            xisf = XISF(image_path)
            image = xisf.read_image(0)  # Assuming the image data is in the first image block
            image_meta = xisf.get_images_metadata()  # List of metadata blocks for each image
            file_meta = xisf.get_file_metadata()  # File-level metadata

            # Determine bit depth
            if image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
                bit_depth = "16-bit"
            elif image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
                bit_depth = "8-bit"                
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

        elif file_extension in ['fits', 'fit']:
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
                    print("Identified 16bit FITS iamge.")
                elif image_data.dtype == np.float32:
                    bit_depth = "32-bit floating point"
                    print("Identified 32bit floating point FITS iamge.")
                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    print("Identified 32bit unsigned FITS iamge.")
                
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
        else:
            image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
            is_mono = False

    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None, None, None, None, None


    # Add a border around the image with the median value
    image_with_border = add_border(image, border_size=16)


    # Stretch the image if needed
    stretch_needed = np.median(image_with_border - np.min(image_with_border)) < 0.08
    original_median = np.median(image_with_border)
    if stretch_needed:
        stretched_image, original_min, original_median = stretch_image(image_with_border)
    else:
        stretched_image = image_with_border
        original_min = None

        

    # Apply sharpening separately to RGB channels if specified
    if sharpen_channels_separately and len(stretched_image.shape) == 3 and not is_mono:
        r_channel, g_channel, b_channel = stretched_image[:, :, 0], stretched_image[:, :, 1], stretched_image[:, :, 2]
        print("Sharpening Red Channel:")
        sharpened_r = sharpen_channel(r_channel, sharpening_mode, nonstellar_strength, stellar_amount, nonstellar_amount, device, models)
        print("Sharpening Green Channel:")
        sharpened_g = sharpen_channel(g_channel, sharpening_mode, nonstellar_strength, stellar_amount, nonstellar_amount, device, models)
        print("Sharpening Blue Channel:")
        sharpened_b = sharpen_channel(b_channel, sharpening_mode, nonstellar_strength, stellar_amount, nonstellar_amount, device, models)
        sharpened_image = np.stack([sharpened_r, sharpened_g, sharpened_b], axis=-1)
    else:
        # Extract luminance (for color images) or handle grayscale images directly
        if len(stretched_image.shape) == 3:
            luminance, cb_channel, cr_channel = extract_luminance(stretched_image)
        else:
            luminance = stretched_image

        chunks = split_image_into_chunks_with_overlap(luminance, chunk_size=256, overlap=64)
        total_chunks = len(chunks)

        print(f"Sharpening Image: {os.path.basename(image_path)}")
        print(f"Total Chunks: {total_chunks}")

        # Stellar sharpening and blending with `stellar_amount`
        stellar_sharpened_chunks = []
        if sharpening_mode == "Stellar Only" or sharpening_mode == "Both":
            print("Stellar Sharpening:")
            for idx, (chunk, i, j, is_edge) in enumerate(chunks):
                original_shape = chunk.shape
                if models.get("is_onnx"):
                    # ONNX inference
                    chunk_input = chunk[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1, 1, H, W)
                    chunk_input = np.tile(chunk_input, (1, 3, 1, 1))  # Expand to 3 channels: (1, 3, H, W)

                    # Pad chunk to 256x256 if dimensions don't match
                    if chunk_input.shape[2] != 256 or chunk_input.shape[3] != 256:
                        padded_chunk = np.zeros((1, 3, 256, 256), dtype=np.float32)  # Create a padded chunk
                        padded_chunk[:, :, :chunk_input.shape[2], :chunk_input.shape[3]] = chunk_input  # Copy original data
                        chunk_input = padded_chunk

                    input_name = models["stellar_model"].get_inputs()[0].name
                    output_name = models["stellar_model"].get_outputs()[0].name
                    try:
                        stellar_sharpened_chunk = models["stellar_model"].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                    except Exception as e:
                        print(f"ONNX inference error for stellar chunk at ({i}, {j}): {e}")
                        raise
                    
                    # Crop the processed chunk back to the original shape
                    stellar_sharpened_chunk = stellar_sharpened_chunk[:original_shape[0], :original_shape[1]]
                else:
                    # PyTorch inference
                    chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        stellar_sharpened_chunk = models["stellar_model"](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]

                blended_stellar_chunk = blend_images(chunk, stellar_sharpened_chunk, stellar_amount)
                stellar_sharpened_chunks.append((blended_stellar_chunk, i, j, is_edge))
                show_progress(idx + 1, total_chunks)

            print("")
            stellar_sharpened_luminance = stitch_chunks_ignore_border(stellar_sharpened_chunks, luminance.shape, chunk_size=256, overlap=64)
            print(f"Stellar sharpening complete. Shape: {stellar_sharpened_luminance.shape}")

            if sharpening_mode == "Stellar Only":
                sharpened_luminance = stellar_sharpened_luminance
            else:
                print("Updating luminance for non-stellar sharpening...")
                chunks = split_image_into_chunks_with_overlap(stellar_sharpened_luminance, chunk_size=256, overlap=64)  # Update luminance for non-stellar sharpening


        # Non-stellar sharpening and blending with `nonstellar_amount`
        nonstellar_sharpened_chunks = []
        model_map = {
            1: models["nonstellar_model_1"],
            2: models["nonstellar_model_2"],
            4: models["nonstellar_model_4"],
            8: models["nonstellar_model_8"]
        }

        if sharpening_mode == "Non-Stellar Only" or sharpening_mode == "Both":
            print("Non-Stellar Sharpening:")
            for idx, (chunk, i, j, is_edge) in enumerate(chunks):
                original_shape = chunk.shape  # Save original dimensions for cropping later

                if models.get("is_onnx"):
                    # Prepare ONNX input
                    chunk_input = chunk[np.newaxis, np.newaxis, :, :].astype(np.float32)  # Shape: (1, 1, H, W)
                    chunk_input = np.tile(chunk_input, (1, 3, 1, 1))  # Expand to 3 channels: (1, 3, H, W)

                    # Pad the chunk to 256x256 if necessary
                    if chunk_input.shape[2] != 256 or chunk_input.shape[3] != 256:
                        padded_chunk = np.zeros((1, 3, 256, 256), dtype=np.float32)
                        padded_chunk[:, :, :chunk_input.shape[2], :chunk_input.shape[3]] = chunk_input
                        chunk_input = padded_chunk

                    input_name = model_map[1].get_inputs()[0].name
                    output_name = model_map[1].get_outputs()[0].name

                    # Handle different nonstellar_strength cases
                    if nonstellar_strength in [1, 2, 4, 8]:
                        # Direct model usage
                        active_model = model_map[int(nonstellar_strength)]
                        sharpened_chunk = active_model.run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                    else:
                        # Interpolation between models
                        if 1 < nonstellar_strength < 2:
                            # Interpolate between models 1 and 2
                            sharpened_chunk_a = model_map[1].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                            sharpened_chunk_b = model_map[2].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                            sharpened_chunk = interpolate_nonstellar_sharpening(sharpened_chunk_a, sharpened_chunk_b, None, None, nonstellar_strength)
                        elif 2 <= nonstellar_strength < 4:
                            # Interpolate between models 2 and 4
                            sharpened_chunk_a = model_map[2].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                            sharpened_chunk_b = model_map[4].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                            sharpened_chunk = interpolate_nonstellar_sharpening(None, sharpened_chunk_a, sharpened_chunk_b, None, nonstellar_strength)
                        elif 4 <= nonstellar_strength < 8:
                            # Interpolate between models 4 and 8
                            sharpened_chunk_a = model_map[4].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                            sharpened_chunk_b = model_map[8].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                            sharpened_chunk = interpolate_nonstellar_sharpening(None, None, sharpened_chunk_a, sharpened_chunk_b, nonstellar_strength)
                        else:
                            raise ValueError(f"Invalid nonstellar_strength value: {nonstellar_strength}")

                    # Crop the processed chunk back to its original dimensions
                    sharpened_chunk = sharpened_chunk[:original_shape[0], :original_shape[1]]
                else:
                    # PyTorch inference
                    chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    if nonstellar_strength in [1, 2, 4, 8]:
                        with torch.no_grad():
                            active_model = model_map[int(nonstellar_strength)]
                            sharpened_chunk = active_model(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                    else:
                        # Interpolation between PyTorch models
                        if 1 < nonstellar_strength < 2:
                            sharpened_chunk_a = model_map[1](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                            sharpened_chunk_b = model_map[2](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                            sharpened_chunk = interpolate_nonstellar_sharpening(sharpened_chunk_a, sharpened_chunk_b, None, None, nonstellar_strength)
                        elif 2 <= nonstellar_strength < 4:
                            sharpened_chunk_a = model_map[2](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                            sharpened_chunk_b = model_map[4](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                            sharpened_chunk = interpolate_nonstellar_sharpening(None, sharpened_chunk_a, sharpened_chunk_b, None, nonstellar_strength)
                        elif 4 <= nonstellar_strength < 8:
                            sharpened_chunk_a = model_map[4](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                            sharpened_chunk_b = model_map[8](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                            sharpened_chunk = interpolate_nonstellar_sharpening(None, None, sharpened_chunk_a, sharpened_chunk_b, nonstellar_strength)



                blended_nonstellar_chunk = blend_images(chunk, sharpened_chunk, nonstellar_amount)
                nonstellar_sharpened_chunks.append((blended_nonstellar_chunk, i, j, is_edge))
                show_progress(idx + 1, total_chunks)

            print("")  # Add a newline after non-stellar sharpening progress
            nonstellar_sharpened_luminance = stitch_chunks_ignore_border(nonstellar_sharpened_chunks, luminance.shape, chunk_size=256, overlap=64)

            # Set the final sharpened luminance to the non-stellar sharpened and blended result
            sharpened_luminance = nonstellar_sharpened_luminance

        # For color images, merge back the luminance with the original chrominance (Cb, Cr)
        if len(image.shape) == 3:
            sharpened_image = merge_luminance(sharpened_luminance, cb_channel, cr_channel)
        else:
            sharpened_image = sharpened_luminance

    # Unstretch the image if necessary
    if stretch_needed:
        sharpened_image = unstretch_image(sharpened_image, original_median, original_min)

    # Replace the 5-pixel border from the original image
    sharpened_image = remove_border(sharpened_image, border_size=16)

    return sharpened_image, is_mono, original_header, bit_depth, file_meta, image_meta


# Helper function to sharpen individual R, G, B channels
def sharpen_channel(channel, sharpening_mode, nonstellar_strength, stellar_amount, nonstellar_amount, device, models):
    chunks = split_image_into_chunks_with_overlap(channel, chunk_size=256, overlap=64)
    total_chunks = len(chunks)

    # Initialize variables to hold sharpened results
    stellar_sharpened_chunks = []
    nonstellar_sharpened_chunks = []
    sharpened_channel = channel  # Initialize as the original channel

    # Apply stellar sharpening first if in "Stellar Only" or "Both" mode
    if sharpening_mode == "Stellar Only" or sharpening_mode == "Both":
        print("Stellar Sharpening Channel:")
        for idx, (chunk, i, j, is_edge) in enumerate(chunks):
            original_shape = chunk.shape
            if models.get("is_onnx"):
                # ONNX inference
                chunk_input = chunk[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1, 1, H, W)
                chunk_input = np.tile(chunk_input, (1, 3, 1, 1))  # Expand to 3 channels: (1, 3, H, W)

                # Pad chunk to 256x256 if dimensions don't match
                if chunk_input.shape[2] != 256 or chunk_input.shape[3] != 256:
                    padded_chunk = np.zeros((1, 3, 256, 256), dtype=np.float32)  # Create a padded chunk
                    padded_chunk[:, :, :chunk_input.shape[2], :chunk_input.shape[3]] = chunk_input  # Copy original data
                    chunk_input = padded_chunk

                input_name = models["stellar_model"].get_inputs()[0].name
                output_name = models["stellar_model"].get_outputs()[0].name
                try:
                    stellar_sharpened_chunk = models["stellar_model"].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                except Exception as e:
                    print(f"ONNX inference error for stellar chunk at ({i}, {j}): {e}")
                    raise
                
                # Crop the processed chunk back to the original shape
                stellar_sharpened_chunk = stellar_sharpened_chunk[:original_shape[0], :original_shape[1]]

            else:
                # PyTorch inference for stellar sharpening
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    stellar_sharpened_chunk = models["stellar_model"](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]

            # Apply stellar amount blending
            blended_stellar_chunk = blend_images(chunk, stellar_sharpened_chunk, stellar_amount)
            stellar_sharpened_chunks.append((blended_stellar_chunk, i, j, is_edge))
            show_progress(idx + 1, total_chunks)

        print("")  # Add a newline after stellar sharpening progress
        stellar_sharpened = stitch_chunks_ignore_border(stellar_sharpened_chunks, channel.shape, chunk_size=256, overlap=64)

        if sharpening_mode == "Stellar Only":
            sharpened_channel = stellar_sharpened
        else:
            channel = stellar_sharpened  # Use stellar-sharpened as input for non-stellar sharpening
            chunks = split_image_into_chunks_with_overlap(channel, chunk_size=256, overlap=64)

    # Use dictionary to select the non-stellar model based on nonstellar_strength
    model_map = {
        1: models["nonstellar_model_1"],
        2: models["nonstellar_model_2"],
        4: models["nonstellar_model_4"],
        8: models["nonstellar_model_8"]
    }

    # Apply non-stellar sharpening if applicable
    if sharpening_mode == "Non-Stellar Only" or sharpening_mode == "Both":
        print("Non-Stellar Sharpening:")
        for idx, (chunk, i, j, is_edge) in enumerate(chunks):
            original_shape = chunk.shape  # Save original dimensions for cropping later

            if models.get("is_onnx"):
                # Prepare ONNX input
                chunk_input = chunk[np.newaxis, np.newaxis, :, :].astype(np.float32)  # Shape: (1, 1, H, W)
                chunk_input = np.tile(chunk_input, (1, 3, 1, 1))  # Expand to 3 channels: (1, 3, H, W)

                # Pad the chunk to 256x256 if necessary
                if chunk_input.shape[2] != 256 or chunk_input.shape[3] != 256:
                    padded_chunk = np.zeros((1, 3, 256, 256), dtype=np.float32)
                    padded_chunk[:, :, :chunk_input.shape[2], :chunk_input.shape[3]] = chunk_input
                    chunk_input = padded_chunk

                input_name = model_map[1].get_inputs()[0].name
                output_name = model_map[1].get_outputs()[0].name

                # Handle different nonstellar_strength cases
                if nonstellar_strength in [1, 2, 4, 8]:
                    # Direct model usage
                    active_model = model_map[int(nonstellar_strength)]
                    sharpened_chunk = active_model.run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                else:
                    # Interpolation between models
                    if 1 < nonstellar_strength < 2:
                        # Interpolate between models 1 and 2
                        sharpened_chunk_a = model_map[1].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                        sharpened_chunk_b = model_map[2].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                        sharpened_chunk = interpolate_nonstellar_sharpening(sharpened_chunk_a, sharpened_chunk_b, None, None, nonstellar_strength)
                    elif 2 <= nonstellar_strength < 4:
                        # Interpolate between models 2 and 4
                        sharpened_chunk_a = model_map[2].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                        sharpened_chunk_b = model_map[4].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                        sharpened_chunk = interpolate_nonstellar_sharpening(None, sharpened_chunk_a, sharpened_chunk_b, None, nonstellar_strength)
                    elif 4 <= nonstellar_strength < 8:
                        # Interpolate between models 4 and 8
                        sharpened_chunk_a = model_map[4].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                        sharpened_chunk_b = model_map[8].run([output_name], {input_name: chunk_input})[0][0, 0, :, :]
                        sharpened_chunk = interpolate_nonstellar_sharpening(None, None, sharpened_chunk_a, sharpened_chunk_b, nonstellar_strength)
                    else:
                        raise ValueError(f"Invalid nonstellar_strength value: {nonstellar_strength}")

                # Crop the processed chunk back to its original dimensions
                sharpened_chunk = sharpened_chunk[:original_shape[0], :original_shape[1]]
            else:
                # PyTorch inference
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                if nonstellar_strength in [1, 2, 4, 8]:
                    with torch.no_grad():
                        active_model = model_map[int(nonstellar_strength)]
                        sharpened_chunk = active_model(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                else:
                    # Interpolation between PyTorch models
                    if 1 < nonstellar_strength < 2:
                        sharpened_chunk_a = model_map[1](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                        sharpened_chunk_b = model_map[2](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                        sharpened_chunk = interpolate_nonstellar_sharpening(sharpened_chunk_a, sharpened_chunk_b, None, None, nonstellar_strength)
                    elif 2 <= nonstellar_strength < 4:
                        sharpened_chunk_a = model_map[2](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                        sharpened_chunk_b = model_map[4](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                        sharpened_chunk = interpolate_nonstellar_sharpening(None, sharpened_chunk_a, sharpened_chunk_b, None, nonstellar_strength)
                    elif 4 <= nonstellar_strength < 8:
                        sharpened_chunk_a = model_map[4](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                        sharpened_chunk_b = model_map[8](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
                        sharpened_chunk = interpolate_nonstellar_sharpening(None, None, sharpened_chunk_a, sharpened_chunk_b, nonstellar_strength)

            # Blend with the input chunk and apply nonstellar_amount
            blended_nonstellar_chunk = blend_images(chunk, sharpened_chunk, nonstellar_amount)
            nonstellar_sharpened_chunks.append((blended_nonstellar_chunk, i, j, is_edge))
            show_progress(idx + 1, total_chunks)

        print("")  # Add a newline after non-stellar sharpening channel progress
        nonstellar_sharpened = stitch_chunks_ignore_border(nonstellar_sharpened_chunks, channel.shape, chunk_size=256, overlap=64)

        # Set the final sharpened channel output to the non-stellar sharpened result
        sharpened_channel = nonstellar_sharpened

    return sharpened_channel



def process_images(input_dir, output_dir, sharpening_mode=None, nonstellar_strength=None, stellar_amount=None, nonstellar_amount=None, use_gpu=True, sharpen_channels_separately=False):
    print((r"""
 *#        ___     __      ___       __                              #
 *#       / __/___/ /__   / _ | ___ / /________                      #
 *#      _\ \/ -_) _ _   / __ |(_-</ __/ __/ _ \                     #
 *#     /___/\__/\//_/  /_/ |_/___/\__/__/ \___/                     #
 *#                                                                  #
 *#              Cosmic Clarity Suite - Sharpen V6.3 AI3             # 
 *#                                                                  #
 *#                         SetiAstro                                #
 *#                    Copyright © 2024                              #
 *#                                                                  #
        """))

    # Use command-line arguments if provided, otherwise fallback to user input
    if sharpening_mode is None or nonstellar_strength is None or stellar_amount is None or nonstellar_amount is None:
        use_gpu, sharpening_mode, nonstellar_strength, stellar_amount, sharpen_channels_separately, nonstellar_amount = get_user_input()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models = load_models(exe_dir, use_gpu)

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)

        # Sharpen the image
        sharpened_image, is_mono, original_header, bit_depth, file_meta, image_meta = sharpen_image(
            image_path, sharpening_mode, nonstellar_strength, stellar_amount, nonstellar_amount,
            models['device'], models, sharpen_channels_separately
        )
        
        if sharpened_image is not None:
            file_extension = os.path.splitext(image_name)[1].lower()
            output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_sharpened" + file_extension)
            actual_bit_depth = bit_depth  # Track actual bit depth for reporting

            # Save as FITS file with header information
            if file_extension in ['.fits', '.fit']:
                if is_mono:
                    sharpened_image_fits = (sharpened_image[:, :, 0] * 65535).astype(np.uint16) if bit_depth == "16-bit" else sharpened_image[:, :, 0].astype(np.float32)
                    hdu = fits.PrimaryHDU(sharpened_image_fits, header=original_header)
                else:
                    sharpened_image_transposed = np.transpose(sharpened_image, (2, 0, 1))
                    sharpened_image_transformed = sharpened_image_transposed

                    if bit_depth == "16-bit":
                        sharpened_image_fits = (sharpened_image_transformed * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        sharpened_image_fits = sharpened_image_transformed.astype(np.float32)
                        original_header['BITPIX'] = -32
                        actual_bit_depth = "32-bit unsigned"
                    else:
                        sharpened_image_fits = sharpened_image_transformed.astype(np.float32)
                        actual_bit_depth = "32-bit float"

                    original_header['NAXIS'] = 3
                    original_header['NAXIS1'] = sharpened_image_transformed.shape[2]
                    original_header['NAXIS2'] = sharpened_image_transformed.shape[1]
                    original_header['NAXIS3'] = sharpened_image_transformed.shape[0]

                    hdu = fits.PrimaryHDU(sharpened_image_fits, header=original_header)

                hdu.writeto(output_image_path, overwrite=True)
                print(f"Saved {actual_bit_depth} sharpened image to: {output_image_path}")

            # Save as TIFF, handling mono or RGB with bit depth as specified
            elif file_extension in ['.tif', '.tiff']:
                if bit_depth == "16-bit":
                    actual_bit_depth = "16-bit"
                    if is_mono:
                        tiff.imwrite(output_image_path, (sharpened_image[:, :, 0] * 65535).astype(np.uint16))
                    else:
                        tiff.imwrite(output_image_path, (sharpened_image * 65535).astype(np.uint16))
                elif bit_depth == "8-bit":
                    actual_bit_depth = "8-bit"
                    if is_mono:
                        tiff.imwrite(output_image_path, (sharpened_image[:, :, 0] * 255.0).astype(np.uint8))
                    else:
                        tiff.imwrite(output_image_path, (sharpened_image * 255.0).astype(np.uint8))                          
                elif bit_depth == "32-bit unsigned":
                    actual_bit_depth = "32-bit unsigned"
                    if is_mono:
                        tiff.imwrite(output_image_path, (sharpened_image[:, :, 0] * 4294967295).astype(np.uint32))
                    else:
                        tiff.imwrite(output_image_path, (sharpened_image * 4294967295).astype(np.uint32))           
                else:
                    actual_bit_depth = "32-bit float"
                    if is_mono:
                        tiff.imwrite(output_image_path, sharpened_image[:, :, 0].astype(np.float32))
                    else:
                        tiff.imwrite(output_image_path, sharpened_image.astype(np.float32))
                    
                print(f"Saved {actual_bit_depth} sharpened image to: {output_image_path}")

            elif file_extension == '.xisf':
                try:
                    # Debug: Print details about the sharpened image
                    print(f"Sharpened image shape: {sharpened_image.shape}, dtype: {sharpened_image.dtype}")
                    print(f"Bit depth: {bit_depth}")

                    # Adjust bit depth for saving
                    if bit_depth == "16-bit":
                        processed_image = (sharpened_image * 65535).astype(np.uint16)
                    elif bit_depth == "8-bit":
                        processed_image = (sharpened_image * 255.0).astype(np.uint8)                        
                    elif bit_depth == "32-bit unsigned":
                        processed_image = (sharpened_image * 4294967295).astype(np.uint32)
                    else:  # Default to 32-bit float
                        processed_image = sharpened_image.astype(np.float32)

                    # Handle mono images
                    if is_mono:
                        print("Preparing mono image for XISF...")
                        processed_image = processed_image[:, :, 0]  # Extract single channel
                        processed_image = processed_image[:, :, np.newaxis]  # Add back channel dimension
                        if image_meta and isinstance(image_meta, list):
                            image_meta[0]['geometry'] = (processed_image.shape[1], processed_image.shape[0], 1)
                            image_meta[0]['colorSpace'] = 'Gray'  # Update metadata for mono images
                        else:
                            # Create default metadata for mono
                            image_meta = [{
                                'geometry': (processed_image.shape[1], processed_image.shape[0], 1),
                                'colorSpace': 'Gray'
                            }]

                    # Fallback for `image_meta` if not provided
                    if image_meta is None or not isinstance(image_meta, list):
                        image_meta = [{
                            'geometry': (processed_image.shape[1], processed_image.shape[0], processed_image.shape[2]),
                            'colorSpace': 'RGB' if not is_mono else 'Gray'
                        }]

                    # Fallback for `file_meta`
                    if file_meta is None:
                        file_meta = {}  # Ensure file_meta is a valid dictionary

                    # Debug: Print processed image details
                    print(f"Processed image shape for XISF: {processed_image.shape}, dtype: {processed_image.dtype}")

                    # Save the image
                    XISF.write(
                        output_image_path,                   # Output path
                        processed_image,                    # Final processed image
                        creator_app="Seti Astro Cosmic Clarity",
                        image_metadata=image_meta[0],       # First block of image metadata
                        xisf_metadata=file_meta,            # File-level metadata

                        shuffle=True
                    )
                    print(f"Saved {bit_depth} XISF sharpened image to: {output_image_path}")

                except Exception as e:
                    print(f"Error saving XISF file: {e}")

                            

            else:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_sharpened.png")
                
                if is_mono:
                    sharpened_image_8bit = (sharpened_image[:, :, 0] * 255).astype(np.uint8)
                    sharpened_image_pil = Image.fromarray(sharpened_image_8bit, mode='L')
                else:
                    sharpened_image_8bit = (sharpened_image * 255).astype(np.uint8)
                    sharpened_image_pil = Image.fromarray(sharpened_image_8bit)
                
                actual_bit_depth = "8-bit"
                sharpened_image_pil.save(output_image_path)

                print(f"Saved {actual_bit_depth} sharpened image to: {output_image_path}")




# Define input and output directories for PyInstaller
input_dir = os.path.join(exe_dir, 'input')
output_dir = os.path.join(exe_dir, 'output')

# Ensure the input and output directories exist
if not os.path.exists(input_dir):
    os.makedirs(input_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Add argument parsing for batch/script execution
parser = argparse.ArgumentParser(description="Stellar and Non-Stellar Sharpening Tool")
parser.add_argument('--sharpening_mode', type=str, choices=["Stellar Only", "Non-Stellar Only", "Both"],
                    help="Choose the sharpening mode: Stellar Only, Non-Stellar Only, Both")
parser.add_argument('--nonstellar_strength', type=float, help="Non-Stellar sharpening strength (1-8)")
parser.add_argument('--stellar_amount', type=float, default=0.9, help="Stellar sharpening amount (0-1)")
parser.add_argument('--nonstellar_amount', type=float, default=0.9, help="Non-Stellar sharpening amount (0-1)")
parser.add_argument('--disable_gpu', action='store_true', help="Disable GPU acceleration and use CPU only")
parser.add_argument('--sharpen_channels_separately', action='store_true', help="Sharpen R, G, and B channels separately")

args = parser.parse_args()

# Determine whether to use GPU based on command-line argument
use_gpu = not args.disable_gpu  # If --disable_gpu is passed, set use_gpu to False

# Pass arguments if provided, or fall back to user input if no command-line arguments are provided
process_images(input_dir, output_dir, args.sharpening_mode, args.nonstellar_strength, args.stellar_amount, args.nonstellar_amount, use_gpu, args.sharpen_channels_separately)
