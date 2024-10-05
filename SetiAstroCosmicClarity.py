import warnings
import os
import sys
import torch
import numpy as np
import torch.nn as nn
import tifffile as tiff
from PIL import Image
import tkinter as tk
from tkinter import simpledialog, messagebox
import argparse  # For command-line argument parsing
import time  # For simulating progress updates

# Suppress model loading warnings
warnings.filterwarnings("ignore")

# Define the SharpeningCNN model with adjusted convolutional layers
class SharpeningCNN(nn.Module):
    def __init__(self):
        super(SharpeningCNN, self).__init__()
        
        # Encoder (down-sampling path)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 1st layer (3 -> 16 feature maps)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 2nd layer (16 -> 32 feature maps)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 3rd layer (32 -> 64 feature maps)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 4th layer (64 -> 128 feature maps)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 5th layer (128 -> 256 feature maps)
            nn.ReLU()
        )
        
        # Decoder (up-sampling path)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 1st deconvolutional layer (256 -> 128 feature maps)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 2nd deconvolutional layer (128 -> 64 feature maps)
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 3rd deconvolutional layer (64 -> 32 feature maps)
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 4th deconvolutional layer (32 -> 16 feature maps)
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # Output layer (16 -> 3 channels for RGB output)
            nn.Sigmoid()  # Ensure output values are between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Get the directory of the executable or the script location
exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

# Function to initialize and load models
def load_models(exe_dir, use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stellar_model_radius_1 = SharpeningCNN()
    nonstellar_model_radius_1 = SharpeningCNN()
    nonstellar_model_radius_2 = SharpeningCNN()
    nonstellar_model_radius_4 = SharpeningCNN()
    nonstellar_model_radius_8 = SharpeningCNN()

    # Load models onto the correct device
    stellar_model_radius_1.load_state_dict(torch.load(os.path.join(exe_dir, 'sharp_cnn_radius_1.pth'), map_location=device))
    nonstellar_model_radius_1.load_state_dict(torch.load(os.path.join(exe_dir, 'nonstellar_sharp_cnn_radius_1.pth'), map_location=device))
    nonstellar_model_radius_2.load_state_dict(torch.load(os.path.join(exe_dir, 'nonstellar_sharp_cnn_radius_2.pth'), map_location=device))
    nonstellar_model_radius_4.load_state_dict(torch.load(os.path.join(exe_dir, 'nonstellar_sharp_cnn_radius_4.pth'), map_location=device))
    nonstellar_model_radius_8.load_state_dict(torch.load(os.path.join(exe_dir, 'nonstellar_sharp_cnn_radius_8.pth'), map_location=device))

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

# Function to extract luminance (Y channel) from an RGB image in 32-bit float precision
def extract_luminance(image):
    # Ensure the image is a NumPy array before processing
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32) / 255.0  # Convert to 32-bit float

    # Convert RGB image to YCbCr and extract the Y channel (luminance)
    ycbcr_image = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8)).convert('YCbCr')
    y, cb, cr = ycbcr_image.split()

    # Convert Y back to 32-bit float
    return np.array(y).astype(np.float32) / 255.0, cb, cr



# Function to show progress during chunk processing
def show_progress(current, total):
    progress_percentage = (current / total) * 100
    print(f"Progress: {progress_percentage:.2f}% ({current}/{total} chunks processed)", end='\r')

# Function to merge the sharpened luminance (Y) with original chrominance (Cb and Cr) to reconstruct RGB image
def merge_luminance(y_sharpened, cb, cr):
    # Ensure Y is clipped between 0 and 1, as we're working in 32-bit float precision
    y_sharpened = np.clip(y_sharpened, 0.0, 1.0)

    # Convert Cb and Cr from Pillow image objects to numpy arrays (if needed)
    cb = np.array(cb).astype(np.float32) / 255.0  # Convert to 32-bit float in range [0, 1]
    cr = np.array(cr).astype(np.float32) / 255.0  # Convert to 32-bit float in range [0, 1]

    # Recreate the YCbCr image as a stack of 32-bit float arrays (Y, Cb, Cr)
    ycbcr_image = np.stack([y_sharpened, cb, cr], axis=-1)

    # Convert YCbCr to RGB while keeping everything in 32-bit float space
    rgb_image = ycbcr_to_rgb(ycbcr_image)

    return rgb_image

# Helper function to convert YCbCr (32-bit float) to RGB
def ycbcr_to_rgb(ycbcr_image):
    # Conversion matrix for YCbCr to RGB (ITU-R BT.601 standard)
    conversion_matrix = np.array([[1.0,  0.0, 1.402],
                                  [1.0, -0.344136, -0.714136],
                                  [1.0,  1.772, 0.0]])

    # Normalize Cb and Cr from [0, 1] to [-0.5, 0.5] as required by YCbCr specification
    ycbcr_image[:, :, 1:] -= 0.5

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
def stitch_chunks_ignore_border(chunks, image_shape, chunk_size, overlap, border_size=5):
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

# Function to get user input for sharpening mode, non-stellar strength, and stellar amount (Interactive)
def get_user_input():
    root = tk.Tk()
    root.withdraw()

    # Ask user if they want to disable GPU acceleration
    use_gpu = messagebox.askyesno("GPU Acceleration", "Do you want to use GPU acceleration? (Yes: Enable GPU, No: Use CPU)")

    valid_stellar_inputs = ["Stellar", "Stellar Only", "stellar"]
    valid_nonstellar_inputs = ["Non-Stellar", "Non-Stellar Only", "non-stellar"]
    sharpening_options = valid_stellar_inputs + valid_nonstellar_inputs + ["Both"]

    sharpening_mode = simpledialog.askstring("Sharpening Mode", "Choose sharpening mode (Stellar, Non-Stellar, Both):")

    # Normalize the input for "Stellar" and "Non-Stellar"
    if sharpening_mode in valid_stellar_inputs:
        sharpening_mode = "Stellar Only"
    elif sharpening_mode in valid_nonstellar_inputs:
        sharpening_mode = "Non-Stellar Only"

    if sharpening_mode not in sharpening_options:
        messagebox.showerror("Error", "Invalid choice. Defaulting to 'Both'.")
        sharpening_mode = "Both"

    # Ask for Stellar Amount only if mode is "Stellar Only" or "Both"
    stellar_amount = None
    if sharpening_mode == "Stellar Only" or sharpening_mode == "Both":
        stellar_amount = simpledialog.askfloat("Stellar Amount", "Enter stellar sharpening amount (0-1):", initialvalue=0.9, minvalue=0, maxvalue=1)

    nonstellar_strength = None
    if sharpening_mode == "Non-Stellar Only" or sharpening_mode == "Both":
        nonstellar_strength = simpledialog.askfloat("Non-Stellar Strength", "Enter non-stellar sharpening strength (1-8):", minvalue=1, maxvalue=8)

    root.destroy()

    return use_gpu, sharpening_mode, nonstellar_strength, stellar_amount

# Function to show progress during chunk processing
def show_progress(current, total):
    progress_percentage = (current / total) * 100
    print(f"Progress: {progress_percentage:.2f}% ({current}/{total} chunks processed)")
    sys.stdout.flush()  # Ensure the progress is flushed to the output immediately


# Function to show progress during chunk processing
def show_progress(current, total):
    progress_percentage = (current / total) * 100
    # Use \r to overwrite the same line
    print(f"\rProgress: {progress_percentage:.2f}% ({current}/{total} chunks processed)", end='', flush=True)

# Function to replace the 5-pixel border from the original image into the processed image
def replace_border(original_image, processed_image, border_size=5):
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


def sharpen_image(image_path, sharpening_mode, nonstellar_strength, stellar_amount, device, models):
    image = None
    file_extension = image_path.lower().split('.')[-1]

    try:
        if file_extension in ['tif', 'tiff']:
            # Load the TIFF image as a 32-bit float directly
            image = tiff.imread(image_path).astype(np.float32)
            if len(image.shape) == 2:  # Grayscale image
                image = np.stack([image] * 3, axis=-1)  # Convert grayscale to 3-channel for consistency
        else:
            # For non-TIFF files, read and convert to 32-bit float for consistency
            image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

    # Extract luminance (for color images) or handle grayscale images directly
    if len(image.shape) == 3:
        luminance, cb, cr = extract_luminance(Image.fromarray((image * 255).astype(np.uint8)))
    else:
        luminance = image

    chunks = split_image_into_chunks_with_overlap(luminance, chunk_size=256, overlap=64)
    total_chunks = len(chunks)

    print(f"Sharpening Image: {os.path.basename(image_path)}")
    print(f"Total Chunks: {total_chunks}")

    denoised_chunks = []
    nonstellar_sharpened = None
    sharpened_luminance = luminance  # Initialize in case neither path modifies it

    # Use dictionary to select the non-stellar model based on nonstellar_strength
    model_map = {
        1: models["nonstellar_model_1"],
        2: models["nonstellar_model_2"],
        4: models["nonstellar_model_4"],
        8: models["nonstellar_model_8"]
    }

    # Apply non-stellar sharpening if applicable
    if nonstellar_strength is not None:
        for idx, (chunk, i, j, is_edge) in enumerate(chunks):
            chunk_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)

            if nonstellar_strength in [1, 2, 4, 8]:
                with torch.no_grad():
                    active_model = model_map[int(nonstellar_strength)]
                    sharpened_chunk = active_model(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
            else:
                # Interpolation for nonstellar_strength between available models
                if nonstellar_strength <= 4:
                    sharpened_chunk_a = model_map[2](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
                    sharpened_chunk_b = model_map[4](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
                    sharpened_chunk = interpolate_nonstellar_sharpening(None, sharpened_chunk_a, sharpened_chunk_b, None, nonstellar_strength)
                else:
                    sharpened_chunk_a = model_map[4](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
                    sharpened_chunk_b = model_map[8](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
                    sharpened_chunk = interpolate_nonstellar_sharpening(None, None, sharpened_chunk_a, sharpened_chunk_b, nonstellar_strength)

            denoised_chunks.append((sharpened_chunk, i, j, is_edge))
            show_progress(idx + 1, total_chunks)  # Update progress after processing each chunk

        nonstellar_sharpened = stitch_chunks_ignore_border(denoised_chunks, luminance.shape, chunk_size=256, overlap=64)

    # Initialize stellar_sharpened_chunks before using it
    stellar_sharpened_chunks = []

    # Apply stellar sharpening (fixed at strength 1)
    if sharpening_mode == "Stellar Only" or sharpening_mode == "Both":
        for idx, (chunk, i, j, is_edge) in enumerate(chunks):
            chunk_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                sharpened_chunk = models["stellar_model"](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
            stellar_sharpened_chunks.append((sharpened_chunk, i, j, is_edge))
            show_progress(idx + 1, total_chunks)  # Update progress after each chunk

        stellar_sharpened_luminance = stitch_chunks_ignore_border(stellar_sharpened_chunks, luminance.shape, chunk_size=256, overlap=64)

        # Blend based on the mode:
        if sharpening_mode == "Stellar Only":
            sharpened_luminance = blend_images(luminance, stellar_sharpened_luminance, stellar_amount)
        elif sharpening_mode == "Both" and nonstellar_sharpened is not None:
            sharpened_luminance = blend_images(nonstellar_sharpened, stellar_sharpened_luminance, stellar_amount)
    elif nonstellar_sharpened is not None:
        # If only non-stellar, stitch the non-stellar sharpened image
        sharpened_luminance = nonstellar_sharpened

    # For color images, merge back the luminance with the original chrominance (Cb, Cr)
    if len(image.shape) == 3:
        sharpened_image = merge_luminance(sharpened_luminance, cb, cr)
    else:
        # For grayscale images, the luminance is the image itself
        sharpened_image = sharpened_luminance

    # Replace the 5-pixel border from the original image
    sharpened_image = replace_border(image, sharpened_image)

    return sharpened_image




def process_images(input_dir, output_dir, sharpening_mode=None, nonstellar_strength=None, stellar_amount=None, use_gpu=True):
    print((r"""
 *#        ___     __      ___       __                              #
 *#       / __/___/ /__   / _ | ___ / /________                      #
 *#      _\ \/ -_) _ _   / __ |(_-</ __/ __/ _ \                     #
 *#     /___/\__/_//_/  /_/ |_/___/\__/_/  \___/                     #
 *#                                                                  #
 *#              Cosmic Clarity - Sharpen V3.1                       # 
 *#                                                                  #
 *#                         SetiAstro                                #
 *#                    Copyright © 2024                              #
 *#                                                                  #
        """))

    # Use command-line arguments if provided, otherwise fallback to user input
    if sharpening_mode is None or nonstellar_strength is None or stellar_amount is None:
        use_gpu, sharpening_mode, nonstellar_strength, stellar_amount = get_user_input()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models = load_models(exe_dir, use_gpu)

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        sharpened_image = sharpen_image(image_path, sharpening_mode, nonstellar_strength, stellar_amount, models['device'], models)

        if sharpened_image is not None:
            file_extension = os.path.splitext(image_name)[1].lower()

            # Check if the original image was grayscale
            original_image = tiff.imread(image_path).astype(np.float32)
            is_grayscale = len(original_image.shape) == 2

            if file_extension in ['.tif', '.tiff']:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_sharpened.tif")
                
                if is_grayscale:
                    # Save only the first channel (grayscale)
                    tiff.imwrite(output_image_path, sharpened_image[:, :, 0].astype(np.float32))
                else:
                    # Save the sharpened image as 32-bit TIFF (RGB)
                    tiff.imwrite(output_image_path, sharpened_image.astype(np.float32))
                    
                print(f"Saved 32-bit sharpened image to: {output_image_path}")
            else:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_sharpened.png")
                
                if is_grayscale:
                    # Save only the first channel (grayscale)
                    sharpened_image_8bit = (sharpened_image[:, :, 0] * 255).astype(np.uint8)
                    sharpened_image_pil = Image.fromarray(sharpened_image_8bit, mode='L')  # L mode for grayscale
                else:
                    sharpened_image_8bit = (sharpened_image * 255).astype(np.uint8)
                    sharpened_image_pil = Image.fromarray(sharpened_image_8bit)
                
                sharpened_image_pil.save(output_image_path)

            print(f"Saved sharpened image to: {output_image_path}")



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
parser.add_argument('--disable_gpu', action='store_true', help="Disable GPU acceleration and use CPU only")

args = parser.parse_args()

# Determine whether to use GPU based on command-line argument
use_gpu = not args.disable_gpu  # If --disable_gpu is passed, set use_gpu to False

# Pass arguments if provided, or fall back to user input if no command-line arguments are provided
process_images(input_dir, output_dir, args.sharpening_mode, args.nonstellar_strength, args.stellar_amount, use_gpu)
