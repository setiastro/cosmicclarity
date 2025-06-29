import warnings
from xisf import XISF
import os
import sys
import torch
import numpy as np
import torch.nn as nn
import lz4.block
import zstandard
import base64
import ast
import platform
import tifffile as tiff
from astropy.io import fits
from PIL import Image
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import ttk
import argparse  # For command-line argument parsing
import onnxruntime as ort
sys.stdout.reconfigure(encoding='utf-8')




#torch.cuda.is_available = lambda: False

# Suppress model loading warnings
warnings.filterwarnings("ignore")

# Detect broken fp16/mixed-precision support
def has_broken_fp16():
    try:
        cc = torch.cuda.get_device_capability()
        return cc[0] < 8  # For example, <8 means <Turing (RTX 20xx)
    except:
        return True

DISABLE_MIXED_PRECISION = has_broken_fp16()
if DISABLE_MIXED_PRECISION:
    print("[INFO] Mixed precision disabled due to unsupported GPU capability.")

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

# Updated DenoiseCNN with Residual Blocks
class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        
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

cached_models = None  # Cache to avoid reloading models unnecessarily

def load_models(exe_dir, use_gpu=True):
    global cached_models
    if cached_models:
        return cached_models

    # Decide device / onnx
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda"); is_onnx = False
    elif "DmlExecutionProvider" in ort.get_available_providers() and use_gpu:
        device = "DirectML"; is_onnx = True
    else:
        device = torch.device("cpu"); is_onnx = False

    mono_model, color_model = None, None

    if not is_onnx:
        # --- PyTorch path ---
        mono = DenoiseCNN().to(device)
        checkpoint = torch.load(os.path.join(exe_dir, "deep_denoise_cnn_AI3_5.pth"), map_location=device)
        mono.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        mono.eval()

        color = DenoiseCNN().to(device)
        chk2  = torch.load(os.path.join(exe_dir, "deep_denoise_cnn_AI3_5c.pth"), map_location=device)
        color.load_state_dict(chk2.get("model_state_dict", chk2))
        color.eval()

        mono_model, color_model = mono, color

    else:
        # --- ONNX path ---
        mono_model  = ort.InferenceSession(
            os.path.join(exe_dir, "deep_denoise_cnn_AI3_5.onnx"),
            providers=["DmlExecutionProvider"]
        )
        color_model = ort.InferenceSession(
            os.path.join(exe_dir, "deep_denoise_cnn_AI3_5c.onnx"),
            providers=["DmlExecutionProvider"]
        )

    cached_models = {
        "device":     device,
        "is_onnx":    is_onnx,
        "mono_model": mono_model,
        "color_model": color_model
    }
    print(f"Loaded models → ONNX={is_onnx}, Device={device}")
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
    print(f"Number of chunks: {len(chunks)}")
    print(f"First few chunks: {[type(c) if not isinstance(c, tuple) else len(c) for c in chunks[:3]]}")

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

# Function to get user input for denoise strength (Interactive)
def get_user_input():
    # Define global variables to store the user input
    global use_gpu, denoise_strength, denoise_mode

    def on_submit():
        # Update the global variables with the user's selections
        global use_gpu, denoise_strength, denoise_mode, separate_channels
        use_gpu = gpu_var.get() == "Yes"
        denoise_strength = denoise_strength_slider.get()
        denoise_mode = denoise_mode_var.get().lower()  # Convert to lowercase for consistency
        separate_channels = sep_var.get()
        root.quit()  # Quit the main loop to continue

    root = tk.Tk()
    root.title("Cosmic Clarity Denoise Tool")

    # GPU selection
    gpu_label = ttk.Label(root, text="Use GPU Acceleration:")
    gpu_label.pack(pady=5)
    gpu_var = tk.StringVar(value="Yes")
    gpu_dropdown = ttk.OptionMenu(root, gpu_var, "Yes", "Yes", "No")
    gpu_dropdown.pack()

    # Denoise strength slider
    denoise_strength_label = ttk.Label(root, text="Denoise Strength (0-1):")
    denoise_strength_label.pack(pady=5)
    denoise_strength_slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient="horizontal")
    denoise_strength_slider.set(0.9)  # Set the default value to 0.9
    denoise_strength_slider.pack()

    # Denoise mode selection
    denoise_mode_label = ttk.Label(root, text="Denoise Mode:")
    denoise_mode_label.pack(pady=5)
    denoise_mode_var = tk.StringVar(value="full")  # Set default value to "full"
    denoise_mode_dropdown = ttk.OptionMenu(root, denoise_mode_var, "full", "luminance", "full")
    denoise_mode_dropdown.pack()

    sep_label = ttk.Label(root, text="Process each RGB channel separately:")
    sep_label.pack(pady=5)
    sep_var = tk.BooleanVar(value=False)
    sep_check = ttk.Checkbutton(root, variable=sep_var, text="Separate channels")
    sep_check.pack()

    # Submit button
    submit_button = ttk.Button(root, text="Submit", command=on_submit)
    submit_button.pack(pady=20)

    root.mainloop()  # Run the main event loop
    root.destroy()  # Destroy the window after quitting the loop

    return use_gpu, denoise_strength, denoise_mode, separate_channels

# Function to show progress during chunk processing
def show_progress(current, total):
    progress_percentage = (current / total) * 100
    print(f"Progress: {progress_percentage:.2f}% ({current}/{total} chunks processed)", end='\r', flush=True)

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
def add_border(image, border_size=16):
    if image.ndim == 2:                                # mono
        med = np.median(image)
        return np.pad(image,
                      ((border_size, border_size), (border_size, border_size)),
                      mode="constant",
                      constant_values=med)

    elif image.ndim == 3 and image.shape[2] == 3:       # RGB
        meds = np.median(image, axis=(0, 1)).astype(image.dtype)  # (3,)
        padded = [np.pad(image[..., c],
                         ((border_size, border_size), (border_size, border_size)),
                         mode="constant",
                         constant_values=float(meds[c]))
                  for c in range(3)]
        return np.stack(padded, axis=-1)
    else:
        raise ValueError("add_border expects mono or RGB image.")

# Function to remove the border added around the image
def remove_border(image, border_size=5):
    if len(image.shape) == 2:
        return image[border_size:-border_size, border_size:-border_size]
    else:
        return image[border_size:-border_size, border_size:-border_size, :]

# Function to denoise the image
def denoise_image(image_path: str,
                  denoise_strength: float,
                  models: dict,
                  denoise_mode: str = 'luminance',
                  separate_channels: bool = False):
    """
    Denoises the input image using the specified model and mode.

    Args:
        image_path (str): Path to the input image.
        denoise_strength (float): Strength of the denoising effect.
        device (torch.device or str): Device for PyTorch or ONNX inference.
        model: PyTorch model or ONNX session for denoising.
        denoise_mode (str): 'luminance' or 'full' denoising mode.
        is_onnx (bool): Whether to use ONNX for inference.

    Returns:
        tuple: Denoised image, original header, bit depth, file extension, is_mono, and metadata.
    """
    device      = models["device"]
    is_onnx     = models["is_onnx"]
    mono_model  = models["mono_model"]
    color_model = models["color_model"]

    # Get file extension
    file_extension = os.path.splitext(image_path)[1].lower()
    if file_extension not in ['.png', '.tif', '.tiff', '.fit', '.fits', '.xisf', 'jpg', 'jpeg']:
        print(f"Ignoring non-image file: {image_path}")
        return None, None, None, None, None, None, None

    original_header = None
    image = None
    bit_depth = "32-bit float"  # Default bit depth
    is_mono = False
    file_meta, image_meta = None, None
    stretch_needed = None

    try:
        # Load the image based on its extension
        if file_extension in ['.tif', '.tiff']:
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
                    print("Identified 16bit FITS image.")
                elif image_data.dtype == np.float32:
                    bit_depth = "32-bit floating point"
                    print("Identified 32bit floating point FITS image.")
                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    print("Identified 32bit unsigned FITS image.")
                
                # Handle 3D FITS data (e.g., RGB or multi-layered data)
                if image_data.ndim == 3:
                    print(f"Detected 3D FITS data with shape: {image_data.shape}")
                    
                    if image_data.shape[0] == 3:
                        # Assume RGB in (3, H, W) format
                        image = np.transpose(image_data, (1, 2, 0))  # Convert to (H, W, 3)
                        
                        if bit_depth == "16-bit":
                            image = image.astype(np.float32) / 65535.0
                        elif bit_depth == "32-bit unsigned":
                            bzero = original_header.get('BZERO', 0)
                            bscale = original_header.get('BSCALE', 1)
                            image = image.astype(np.float32) * bscale + bzero
                            image_min = image.min()
                            image_max = image.max()
                            image = (image - image_min) / (image_max - image_min)
                            print(f"Image range after applying BZERO and BSCALE (RGB): min={image_min}, max={image_max}")
                        # No need to normalize float32
                        is_mono = False

                    elif image_data.shape[0] == 1:
                        # Single-channel mono FITS in (1, H, W) format
                        image = image_data[0]  # Drop channel dim, now (H, W)
                        image = np.stack([image] * 3, axis=-1)  # Convert to RGB shape (H, W, 3)
                        is_mono = True

                    elif image_data.shape[2] == 1:
                        # Shape (H, W, 1) — mono with trailing singleton dim
                        image = image_data[:, :, 0]
                        image = np.stack([image] * 3, axis=-1)
                        is_mono = True

                    else:
                        raise ValueError(f"Unsupported 3D FITS shape: {image_data.shape}")

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


        else:  # Assume 8-bit PNG
            image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
            bit_depth = "8-bit"
            print(f"Loaded {bit_depth} PNG image.")

        if image is None:
            raise ValueError(f"Failed to load image from: {image_path}")

        # Add a border around the image with the median value
        stretch_needed = np.median(image - np.min(image)) < 0.05   # decision first

        if stretch_needed:
            print("normalizing linear data")
            stretched_core, original_min, original_medians = stretch_image(image)
        else:
            stretched_core   = image.astype(np.float32, copy=False)
            original_min     = np.min(image)
            original_medians = [np.median(image[..., c]) for c in range(3)] \
                            if image.ndim == 3 else [np.median(image)]

        # pad AFTER stretch, per-channel median
        stretched_image = add_border(stretched_core, border_size=16)

        # **Apply TV Denoise on the full image unconditionally**
        #tv_weight = 0.005  # Fixed weight for TV denoising
        #print(f"Applying Total Variation Denoising (TVDenoise) on the full image with fixed weight: {tv_weight}")
        #denoised_full = np.zeros_like(stretched_image)
        #for c in range(stretched_image.shape[2]):
        #    denoised_full[:, :, c] = denoise_tv_chambolle(stretched_image[:, :, c], weight=tv_weight, channel_axis=-1)
        #stretched_image = denoised_full
        #print("TV Denoising applied.")

        # Process mono or color images
        if is_mono:
            # use mono network on channel 0
            mono_net = models["mono_model"]
            den_r = denoise_channel(stretched_image[...,0], models["device"], mono_net, True, models["is_onnx"])
            den = denoise_strength * den_r + (1-denoise_strength)*stretched_image[...,0]
            denoised_image = den[...,None]

        else:
            # 1) separate‐channels override
            if separate_channels:
                out_ch = []
                for c in range(3):
                    dch = denoise_channel(
                        stretched_image[...,c],
                        models["device"],
                        models["mono_model"],
                        True, models["is_onnx"]
                    )
                    blended = blend_images(stretched_image[...,c], dch, denoise_strength)
                    out_ch.append(blended)
                denoised_image = np.stack(out_ch, axis=-1)

            # 2) luminance only
            elif denoise_mode == 'luminance':
                y, cb, cr = extract_luminance(stretched_image)
                den_y = denoise_channel(
                    y,
                    models["device"],
                    models["mono_model"],
                    True, models["is_onnx"]
                )
                y2 = blend_images(y, den_y, denoise_strength)
                denoised_image = merge_luminance(y2, cb, cr)

            else:  # 'full'
                # full RGB network
                denoised_image = denoise_full_rgb(stretched_image, models, denoise_strength)


        # Unstretch the image
        if stretch_needed:
            print("de-normalizing linear data")
            denoised_image = unstretch_image(denoised_image, original_medians, original_min)


        # Remove the border added around the image
        
        denoised_image = remove_border(denoised_image, border_size=16)
        
        return denoised_image, original_header, bit_depth, file_extension, is_mono, file_meta, image_meta

    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None, None, None, None, None, None
    
def denoise_full_rgb(image_rgb, models, denoise_strength):
    """
    image_rgb: H×W×3 float32 in [0,1]
    models: dict from load_models()
    """
    device    = models["device"]
    is_onnx   = models["is_onnx"]
    session   = models["color_model"]
    chunk_size, overlap = 256, 64
    eps = 1e-8

    # 1) Chop into overlapping chunks
    chunks = split_image_into_chunks_with_overlap(image_rgb, chunk_size, overlap)
    denoised_chunks = []

    for idx, (chunk, i, j) in enumerate(chunks):
        h, w, _ = chunk.shape

        # 2) Compute per-channel min/max for THIS chunk: shape (1,1,3)
        #pmin  = chunk.min(axis=(0,1), keepdims=True)
        #pmax  = chunk.max(axis=(0,1), keepdims=True)
        #denom = np.maximum(pmax - pmin, eps)

        # 3) Normalize chunk per-channel
        #norm = (chunk - pmin) / denom
        norm = chunk

        # 4) Run the model on the normalized chunk
        if is_onnx:
            inp = norm.transpose(2,0,1)[None].astype(np.float32)  # (1,3,H,W)
            inp = np.pad(inp,
                         ((0,0),(0,0),(0,chunk_size-h),(0,chunk_size-w)),
                         mode="constant")
            name = session.get_inputs()[0].name
            out = session.run(None, {name: inp})[0]
            out_norm = out[0, :, :h, :w].transpose(1,2,0)
        else:
            t = torch.from_numpy(norm).permute(2,0,1)[None].to(device)
            with torch.no_grad():
                if not DISABLE_MIXED_PRECISION and device.type=="cuda":
                    with torch.cuda.amp.autocast():
                        out_t = session(t)
                else:
                    out_t = session(t)
            out_norm = out_t.squeeze(0).cpu().permute(1,2,0).numpy()

        # 5) Blend in normalized space
        blended_norm = blend_images(norm, out_norm, denoise_strength)

        # 6) Un-normalize back to original chunk range
        #blended = blended_norm * denom + pmin
        blended = blended_norm

        denoised_chunks.append((blended, i, j))
        show_progress(idx+1, len(chunks))

    # 7) Stitch all chunks back together
    fused = stitch_chunks_ignore_border(denoised_chunks, image_rgb.shape, chunk_size, overlap)
    return fused

# Function to denoise a single channel
def denoise_channel(channel, device, model, is_mono=False, is_onnx=False):
    """
    Denoises a single image channel using the specified model.

    Args:
        channel (numpy.ndarray): The input image channel to denoise.
        device (torch.device or str): Device for PyTorch or ONNX inference.
        model: PyTorch model or ONNX inference session.
        is_mono (bool): Whether the input is monochrome.
        is_onnx (bool): Whether to use ONNX for inference.

    Returns:
        numpy.ndarray: The denoised image channel.
    """
    chunk_size = 256
    overlap = 64
    chunks = split_image_into_chunks_with_overlap(channel, chunk_size=chunk_size, overlap=overlap)

    # Debug: validate chunk structure
    for idx, entry in enumerate(chunks[:3]):
        print(f"Chunk {idx} type: {type(entry)}, length: {len(entry) if isinstance(entry, tuple) else 'N/A'}")

    denoised_chunks = []

    for idx, entry in enumerate(chunks):
        try:
            chunk, i, j = entry
        except Exception as e:
            print(f"❌ Failed to unpack chunk {idx}: {entry} ({e})")
            continue

        original_chunk_shape = chunk.shape
        if is_onnx:
            # Prepare ONNX input: Expand to 3 channels
            chunk_input = chunk[np.newaxis, np.newaxis, :, :].astype(np.float32)  # Shape: (1, 1, H, W)
            chunk_input = np.tile(chunk_input, (1, 3, 1, 1))  # Shape: (1, 3, H, W)

            # Pad the chunk to 256x256 if necessary
            if chunk_input.shape[2] != chunk_size or chunk_input.shape[3] != chunk_size:
                padded_chunk = np.zeros((1, 3, chunk_size, chunk_size), dtype=np.float32)
                padded_chunk[:, :, :chunk_input.shape[2], :chunk_input.shape[3]] = chunk_input
                chunk_input = padded_chunk

            # ONNX inference
            input_name = model.get_inputs()[0].name
            try:
                denoised_chunk = model.run(None, {input_name: chunk_input})[0]
                denoised_chunk = denoised_chunk[0, 0, :original_chunk_shape[0], :original_chunk_shape[1]]
            except Exception as e:
                print(f"ONNX inference error for chunk at ({i}, {j}): {e}")
                continue

        else:
            # Prepare PyTorch input
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            if not is_mono:
                chunk_tensor = chunk_tensor.repeat(1, 3, 1, 1)
            else:
                chunk_tensor = chunk_tensor.expand(1, 3, chunk_tensor.shape[2], chunk_tensor.shape[3])

            # PyTorch inference
            try:
                with torch.no_grad():
                    if not DISABLE_MIXED_PRECISION and device.type == "cuda":
                        with torch.cuda.amp.autocast():
                            denoised_output = model(chunk_tensor).squeeze().cpu().numpy()
                    else:
                        denoised_output = model(chunk_tensor).squeeze().cpu().numpy()

                    if denoised_output.ndim == 3:
                        denoised_chunk = denoised_output[0]  # Take first channel
                    else:
                        denoised_chunk = denoised_output

            except Exception as e:
                print(f"PyTorch inference error for chunk at ({i}, {j}): {e}")
                continue

        if denoised_chunk is not None:
            denoised_chunks.append((denoised_chunk, i, j))
        else:
            print(f"⚠️ Warning: Denoised chunk at ({i}, {j}) is None — skipping")

        # Show progress update
        show_progress(idx + 1, len(chunks))

    # Debug: final validation before stitching
    if not all(isinstance(entry, tuple) and len(entry) == 3 for entry in denoised_chunks):
        print("❗ Error: One or more denoised chunks are malformed before stitching!")

    # Stitch the chunks back together
    denoised_channel = stitch_chunks_ignore_border(denoised_chunks, channel.shape, chunk_size=chunk_size, overlap=overlap)

    return denoised_channel





# Main process for denoising images
def process_images(input_dir, output_dir, denoise_strength=None, use_gpu=True, denoise_mode='luminance', separate_channels=False):
    print((r"""
 *#        ___     __      ___       __                              #
 *#       / __/___/ /__   / _ | ___ / /________                      #
 *#      _\ \/ -_) _ _   / __ |(_-</ __/ __/ _ \                     #
 *#     /___/\__/\//_/  /_/ |_/___/\__/__/ \___/                     #
 *#                                                                  #
 *#              Cosmic Clarity - Denoise V6.5 AI3.5c                # 
 *#                                                                  #
 *#                         SetiAstro                                #
 *#                    Copyright © 2025                              #
 *#                                                                  #
        """))

    if denoise_strength is None:
        # Prompt for user input
        use_gpu, denoise_strength, denoise_mode, separate_channels = get_user_input()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the denoise model
    models = load_models(exe_dir, use_gpu)

    # Determine whether we're using ONNX or PyTorch
    print(f"Using {'ONNX' if models['is_onnx'] else 'PyTorch'} models.")

    # Process each image in the input directory
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.tif', '.tiff', '.fits', '.fit', '.xisf', '.jpg', '.jpeg')):
            continue

        # Unpack exactly what denoise_image returns
        denoised_image, original_header, bit_depth, file_extension, is_mono, file_meta, image_meta = denoise_image(
            os.path.join(input_dir, fname),
            denoise_strength,
            models,
            denoise_mode,
            separate_channels
        )
        # Skip on error
        if denoised_image is None:
            continue

        # Build output path
        output_name = os.path.splitext(fname)[0] + "_denoised"
        output_image_path = os.path.join(output_dir, output_name + file_extension)

        # Save as FITS file with header information if the original was FITS
        if file_extension in ['.fits', '.fit']:
            if original_header is not None:
                if is_mono:  # Grayscale FITS
                    # Convert the grayscale image back to its original 2D format
                    if bit_depth == "16-bit":
                        denoised_image_fits = (denoised_image[:, :, 0] * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        bzero = original_header.get('BZERO', 0)
                        bscale = original_header.get('BSCALE', 1)
                        denoised_image_fits = (denoised_image[:, :, 0].astype(np.float32) * bscale + bzero).astype(np.uint32)
                        original_header['BITPIX'] = 32
                    else:  # 32-bit float
                        denoised_image_fits = denoised_image[:, :, 0].astype(np.float32)
                    
                    # Update header for a 2D (grayscale) image
                    original_header['NAXIS'] = 2
                    original_header['NAXIS1'] = denoised_image.shape[1]  # Width
                    original_header['NAXIS2'] = denoised_image.shape[0]  # Height
                    if 'NAXIS3' in original_header:
                        del original_header['NAXIS3']  # Remove if present

                    hdu = fits.PrimaryHDU(denoised_image_fits, header=original_header)
                
                else:  # RGB FITS
                    # Transpose RGB image to FITS-compatible format (channels, height, width)
                    denoised_image_transposed = np.transpose(denoised_image, (2, 0, 1))

                    if bit_depth == "16-bit":
                        denoised_image_fits = (denoised_image_transposed * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        denoised_image_fits = denoised_image_transposed.astype(np.float32)
                        original_header['BITPIX'] = -32
                        actual_bit_depth = "32-bit unsigned"
                    else:
                        denoised_image_fits = denoised_image_transposed.astype(np.float32)
                        actual_bit_depth = "32-bit float"

                    # Update header for a 3D (RGB) image
                    original_header['NAXIS'] = 3
                    original_header['NAXIS1'] = denoised_image_transposed.shape[2]  # Width
                    original_header['NAXIS2'] = denoised_image_transposed.shape[1]  # Height
                    original_header['NAXIS3'] = denoised_image_transposed.shape[0]  # Channels
                    
                    hdu = fits.PrimaryHDU(denoised_image_fits, header=original_header)

                # Write the FITS file
                hdu.writeto(output_image_path, overwrite=True)
                print(f"Saved {actual_bit_depth} denoised image to: {output_image_path}")


        # Save as TIFF based on the original bit depth if the original was TIFF
        elif file_extension in ['.tif', '.tiff']:
            if bit_depth == "16-bit":
                actual_bit_depth = "16-bit"
                if is_mono is True:  # Grayscale
                    tiff.imwrite(output_image_path, (denoised_image[:, :, 0] * 65535).astype(np.uint16))
                else:  # RGB
                    tiff.imwrite(output_image_path, (denoised_image * 65535).astype(np.uint16))
            elif bit_depth == "8-bit":
                actual_bit_depth = "8-bit"
                if is_mono:
                    tiff.imwrite(output_image_path, (denoised_image[:, :, 0] * 255.0).astype(np.uint8))
                else:
                    tiff.imwrite(output_image_path, (denoised_image * 255.0).astype(np.uint8))                               
            elif bit_depth == "32-bit unsigned":
                actual_bit_depth = "32-bit unsigned"
                if is_mono is True:  # Grayscale
                    tiff.imwrite(output_image_path, (denoised_image[:, :, 0] * 4294967295).astype(np.uint32))
                else:  # RGB
                    tiff.imwrite(output_image_path, (denoised_image * 4294967295).astype(np.uint32))           
            else:
                actual_bit_depth = "32-bit float"
                if is_mono is True:  # Grayscale
                    tiff.imwrite(output_image_path, denoised_image[:, :, 0].astype(np.float32))
                else:  # RGB
                    tiff.imwrite(output_image_path, denoised_image.astype(np.float32))

            print(f"Saved {actual_bit_depth} denoised image to: {output_image_path}")

        elif file_extension == '.xisf':
            try:
                # Debug: Print original image details
                print(f"Original image shape: {denoised_image.shape}, dtype: {denoised_image.dtype}")
                print(f"Bit depth: {bit_depth}")

                # Adjust bit depth
                if bit_depth == "16-bit":
                    processed_image = (denoised_image * 65535).astype(np.uint16)
                elif bit_depth == "8-bit":
                    processed_image = (denoised_image * 255.0).astype(np.uint8)                        
                elif bit_depth == "32-bit unsigned":
                    processed_image = (denoised_image * 4294967295).astype(np.uint32)
                else:  # Default to 32-bit float
                    processed_image = denoised_image.astype(np.float32)

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
                    output_image_path,                  # Correct output path
                    processed_image,                   # Final processed image
                    creator_app="Seti Astro Cosmic Clarity",
                    image_metadata=image_meta[0],      # First block of image metadata
                    xisf_metadata=file_meta,           # File-level metadata

                    shuffle=True
                )
                print(f"Saved {bit_depth} XISF denoised image to: {output_image_path}")

            except Exception as e:
                print(f"Error saving XISF file: {e}")




        # Save as 8-bit PNG if the original was PNG
        else:
            output_image_path = os.path.join(output_dir, output_image_name + ".png")
            denoised_image_8bit = (denoised_image * 255).astype(np.uint8)
            denoised_image_pil = Image.fromarray(denoised_image_8bit)
            actual_bit_depth = "8-bit"
            denoised_image_pil.save(output_image_path)
            print(f"Saved {actual_bit_depth} denoised image to: {output_image_path}")




# Define input and output directories
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cosmic Clarity Denoise Tool")
    parser.add_argument(
        '--denoise_strength',
        type=float,
        help="Denoise strength (0–1), overrides the GUI slider if provided"
    )
    parser.add_argument(
        '--disable_gpu',
        action='store_true',
        help="Disable GPU acceleration and force CPU usage"
    )
    parser.add_argument(
        '--denoise_mode',
        choices=['luminance','full','separate'],
        default='luminance',
        help="Denoise mode: 'luminance', 'full' color model, or 'separate' per-channel"
    )
    parser.add_argument(
        '--separate_channels',
        action='store_true',
        help="Alias for --denoise_mode separate"
    )
    
    args = parser.parse_args()

    # Ensure input/output folders exist
    input_dir  = os.path.join(exe_dir, 'input')
    output_dir = os.path.join(exe_dir, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # If the user requested separate_channels, override denoise_mode
    mode = args.denoise_mode
    if args.separate_channels:
        mode = 'separate'

    # Launch processing
    process_images(
        input_dir,
        output_dir,
        denoise_strength=args.denoise_strength,
        use_gpu=not args.disable_gpu,
        denoise_mode=mode,
        separate_channels=args.separate_channels
    )
