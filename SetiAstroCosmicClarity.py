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

# CNN model for sharpening
class SharpeningCNN(nn.Module):
    def __init__(self):
        super(SharpeningCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Clear the console screen
def clear_console():
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except Exception as e:
        print(f"Console clearing failed: {e}")


# Get the directory of the executable or the script location
exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

# Load models dynamically from the same directory as the executable/script
stellar_model_radius_1 = SharpeningCNN()
nonstellar_model_radius_1 = SharpeningCNN()
nonstellar_model_radius_2 = SharpeningCNN()
nonstellar_model_radius_4 = SharpeningCNN()
nonstellar_model_radius_8 = SharpeningCNN()

stellar_model_radius_1.load_state_dict(torch.load(os.path.join(exe_dir, 'sharp_cnn_radius_1.pth')))
nonstellar_model_radius_1.load_state_dict(torch.load(os.path.join(exe_dir, 'nonstellar_sharp_cnn_radius_1.pth')))
nonstellar_model_radius_2.load_state_dict(torch.load(os.path.join(exe_dir, 'nonstellar_sharp_cnn_radius_2.pth')))
nonstellar_model_radius_4.load_state_dict(torch.load(os.path.join(exe_dir, 'nonstellar_sharp_cnn_radius_4.pth')))
nonstellar_model_radius_8.load_state_dict(torch.load(os.path.join(exe_dir, 'nonstellar_sharp_cnn_radius_8.pth')))

# Set models to evaluation mode
stellar_model_radius_1.eval()
nonstellar_model_radius_1.eval()
nonstellar_model_radius_2.eval()
nonstellar_model_radius_4.eval()
nonstellar_model_radius_8.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stellar_model_radius_1.to(device)
nonstellar_model_radius_1.to(device)
nonstellar_model_radius_2.to(device)
nonstellar_model_radius_4.to(device)
nonstellar_model_radius_8.to(device)

# Function to extract luminance (Y channel) from an RGB image
def extract_luminance(image):
    ycbcr_image = image.convert('YCbCr')
    y, cb, cr = ycbcr_image.split()
    return np.array(y).astype(np.float32) / 255.0, cb, cr

# Function to show progress during chunk processing
def show_progress(current, total):
    progress_percentage = (current / total) * 100
    print(f"Progress: {progress_percentage:.2f}% ({current}/{total} chunks processed)", end='\r')


# Function to merge the sharpened luminance (Y) with original chrominance (Cb and Cr) to reconstruct RGB image
def merge_luminance(y_sharpened, cb, cr):
    y_sharpened = np.clip(y_sharpened * 255, 0, 255).astype(np.uint8)
    y_sharpened_img = Image.fromarray(y_sharpened)
    ycbcr_image = Image.merge('YCbCr', (y_sharpened_img, cb, cr))
    return ycbcr_image.convert('RGB')

# Soft blending weights for the overlap area
def generate_blend_weights(chunk_size, overlap):
    ramp = np.linspace(0, 1, overlap)
    flat = np.ones(chunk_size - 2 * overlap)
    blend_vector = np.concatenate([ramp, flat, ramp[::-1]])  # Create smooth transition
    blend_matrix = np.outer(blend_vector, blend_vector)  # 2D blending weights
    return blend_matrix

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


# Function to interpolate the results for non-stellar sharpening based on strength
def interpolate_nonstellar_sharpening(sharpened_1, sharpened_2, sharpened_4, sharpened_8, strength):
    if strength <= 2:
        return (2 - strength) * sharpened_1 + (strength - 1) * sharpened_2
    elif 2 < strength <= 4:
        return ((4 - strength) * sharpened_2 + (strength - 2) * sharpened_4) / 2
    else:
        return ((8 - strength) * sharpened_4 + (strength - 4) * sharpened_8) / 4

# Function to get user input for sharpening mode, non-stellar strength, and stellar amount (Interactive)
def get_user_input():
    root = tk.Tk()
    root.withdraw()

    valid_stellar_inputs = ["Stellar", "Stellar Only"]
    valid_nonstellar_inputs = ["Non-Stellar", "Non-Stellar Only"]
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

    return sharpening_mode, nonstellar_strength, stellar_amount


# Function to blend two images (before and after)
def blend_images(before, after, amount):
    return (1 - amount) * before + amount * after

# Function to sharpen an image
def sharpen_image(image_path, sharpening_mode, nonstellar_strength, stellar_amount):
    image = None
    file_extension = image_path.lower().split('.')[-1]

    try:
        if file_extension in ['tif', 'tiff']:
            image = tiff.imread(image_path).astype(np.float32)
            if len(image.shape) == 2:
                image = np.stack([image]*3, axis=-1)
            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

    luminance, cb, cr = extract_luminance(image)
    chunks = split_image_into_chunks_with_overlap(luminance, chunk_size=256, overlap=64)
    total_chunks = len(chunks)

    print(f"Sharpening Image: {os.path.basename(image_path)}")
    print(f"Total Chunks: {total_chunks}")

    denoised_chunks = []
    nonstellar_sharpened = None
    sharpened_luminance = luminance  # Initialize in case neither path modifies it

    # Apply non-stellar sharpening if applicable
    if nonstellar_strength is not None:
        for idx, (chunk, i, j, is_edge) in enumerate(chunks):  # Updated with enumerate to get the idx
            chunk_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)

            if nonstellar_strength in [1, 2, 4, 8]:
                with torch.no_grad():
                    active_model = eval(f'nonstellar_model_radius_{int(nonstellar_strength)}')
                    sharpened_chunk = active_model(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
            else:
                if nonstellar_strength <= 4:
                    sharpened_chunk_a = nonstellar_model_radius_2(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
                    sharpened_chunk_b = nonstellar_model_radius_4(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
                    sharpened_chunk = interpolate_nonstellar_sharpening(None, sharpened_chunk_a, sharpened_chunk_b, None, nonstellar_strength)
                else:
                    sharpened_chunk_a = nonstellar_model_radius_4(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
                    sharpened_chunk_b = nonstellar_model_radius_8(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
                    sharpened_chunk = interpolate_nonstellar_sharpening(None, None, sharpened_chunk_a, sharpened_chunk_b, nonstellar_strength)

            denoised_chunks.append((sharpened_chunk, i, j, is_edge))
            # Update progress
            show_progress(idx + 1, total_chunks)

        nonstellar_sharpened = stitch_chunks_ignore_border(denoised_chunks, luminance.shape, chunk_size=256, overlap=64)

    # Initialize stellar_sharpened_chunks before using it
    stellar_sharpened_chunks = []

    # Apply stellar sharpening (fixed at strength 1)
    if sharpening_mode == "Stellar Only" or sharpening_mode == "Both":
        for idx, (chunk, i, j, is_edge) in enumerate(chunks):  # Updated with enumerate
            chunk_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                sharpened_chunk = stellar_model_radius_1(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
            stellar_sharpened_chunks.append((sharpened_chunk, i, j, is_edge))
            # Update progress
            show_progress(idx + 1, total_chunks)

        stellar_sharpened_luminance = stitch_chunks_ignore_border(stellar_sharpened_chunks, luminance.shape, chunk_size=256, overlap=64)

        # Blend based on the mode:
        if sharpening_mode == "Stellar Only":
            sharpened_luminance = blend_images(luminance, stellar_sharpened_luminance, stellar_amount)
        elif sharpening_mode == "Both" and nonstellar_sharpened is not None:
            sharpened_luminance = blend_images(nonstellar_sharpened, stellar_sharpened_luminance, stellar_amount)
    elif nonstellar_sharpened is not None:
        # If only non-stellar, stitch the non-stellar sharpened image
        sharpened_luminance = nonstellar_sharpened

    sharpened_image = merge_luminance(sharpened_luminance, cb, cr)
    return sharpened_image


# Main process for sharpening images
def process_images(input_dir, output_dir, sharpening_mode=None, nonstellar_strength=None, stellar_amount=None):
    # Use command-line arguments if provided, otherwise fallback to user input
    if sharpening_mode is None or nonstellar_strength is None or stellar_amount is None:
        sharpening_mode, nonstellar_strength, stellar_amount = get_user_input()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        sharpened_image = sharpen_image(image_path, sharpening_mode, nonstellar_strength, stellar_amount)
        
        if sharpened_image:
            file_extension = os.path.splitext(image_name)[1].lower()
            if file_extension in ['.tif', '.tiff']:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_sharpened.tif")
                sharpened_image_np = np.array(sharpened_image).astype(np.float32) / 255.0
                tiff.imwrite(output_image_path, sharpened_image_np, dtype=np.float32)
            else:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_sharpened.png")
                sharpened_image.save(output_image_path)

            print(f"Saved sharpened image to: {output_image_path}")

# Define input and output directories for PyInstaller
input_dir = os.path.join(exe_dir, 'input')
output_dir = os.path.join(exe_dir, 'output')

# Add argument parsing for batch/script execution
parser = argparse.ArgumentParser(description="Stellar and Non-Stellar Sharpening Tool")
parser.add_argument('--sharpening_mode', type=str, choices=["Stellar Only", "Non-Stellar Only", "Both"],
                    help="Choose the sharpening mode: Stellar Only, Non-Stellar Only, Both")
parser.add_argument('--nonstellar_strength', type=float, help="Non-Stellar sharpening strength (1-8)")
parser.add_argument('--stellar_amount', type=float, default=0.9, help="Stellar sharpening amount (0-1)")

args = parser.parse_args()

# Pass arguments if provided, or fall back to user input if no command-line arguments are provided
process_images(input_dir, output_dir, args.sharpening_mode, args.nonstellar_strength, args.stellar_amount)
