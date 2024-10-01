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

# Suppress model loading warnings
warnings.filterwarnings("ignore")

# Define the DenoiseCNN model with adjusted convolutional layers
class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        
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

# Function to initialize and load the denoise model
def load_model(exe_dir, use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    denoise_model = DenoiseCNN()
    denoise_model.load_state_dict(torch.load(os.path.join(exe_dir, 'denoise_cnn.pth'), map_location=device))

    denoise_model.eval()
    denoise_model.to(device)

    return {
        "denoise_model": denoise_model,
        "device": device
    }

# Function to extract Y, Cb, Cr channels from an RGB image
def extract_ycbcr(image):
    ycbcr_image = image.convert('YCbCr')
    y, cb, cr = ycbcr_image.split()
    return np.array(y).astype(np.float32) / 255.0, np.array(cb).astype(np.float32) / 255.0, np.array(cr).astype(np.float32) / 255.0

# Function to merge the denoised Y, Cb, Cr channels back to an RGB image
def merge_ycbcr(y_denoised, cb, cr):
    y_denoised = np.clip(y_denoised * 255, 0, 255).astype(np.uint8)
    cb = np.clip(cb * 255, 0, 255).astype(np.uint8)
    cr = np.clip(cr * 255, 0, 255).astype(np.uint8)

    y_denoised_img = Image.fromarray(y_denoised)
    cb_img = Image.fromarray(cb)
    cr_img = Image.fromarray(cr)
    ycbcr_image = Image.merge('YCbCr', (y_denoised_img, cb_img, cr_img))
    return ycbcr_image.convert('RGB')

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
def stitch_chunks_ignore_border(chunks, image_shape, chunk_size, overlap, border_size=5):
    stitched_image = np.zeros(image_shape, dtype=np.float32)
    weight_map = np.zeros(image_shape, dtype=np.float32)  # Track blending weights
    
    for chunk, i, j in chunks:
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

# Function to get user input for denoise strength (Interactive)
def get_user_input():
    root = tk.Tk()
    root.withdraw()

    # Ask user if they want to disable GPU acceleration
    use_gpu = messagebox.askyesno("GPU Acceleration", "Do you want to use GPU acceleration? (Yes: Enable GPU, No: Use CPU)")

    denoise_strength = simpledialog.askfloat("Denoise Strength", "Enter denoise strength (0-1):", initialvalue=0.9, minvalue=0, maxvalue=1)

    # Ask user for denoise mode (luminance or full)
    denoise_mode = messagebox.askquestion("Denoise Mode", "Do you want full YCbCr denoise? (Yes: Full denoise, No: Luminance only)") 
    denoise_mode = "full" if denoise_mode == "yes" else "luminance"

    root.destroy()

    return use_gpu, denoise_strength, denoise_mode

# Function to show progress during chunk processing
def show_progress(current, total):
    progress_percentage = (current / total) * 100
    print(f"Progress: {progress_percentage:.2f}% ({current}/{total} chunks processed)", end='\r', flush=True)

# Function to denoise a channel (Y, Cb, Cr)
def denoise_channel(channel, device, model):
    # Split channel into chunks
    chunk_size = 256
    overlap = 64
    chunks = split_image_into_chunks_with_overlap(channel, chunk_size=chunk_size, overlap=overlap)

    denoised_chunks = []

    # Apply denoise model to each chunk
    for idx, (chunk, i, j) in enumerate(chunks):
        chunk_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            denoised_chunk = model(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
        denoised_chunks.append((denoised_chunk, i, j))

        # Show progress update
        show_progress(idx + 1, len(chunks))

    # Stitch the chunks back together
    denoised_channel = stitch_chunks_ignore_border(denoised_chunks, channel.shape, chunk_size=chunk_size, overlap=overlap)
    return denoised_channel

# Function to denoise an image (luminance or full RGB mode)
def denoise_image(image_path, denoise_strength, device, model, denoise_mode='luminance'):
    image = None
    file_extension = image_path.lower().split('.')[-1]

    try:
        if file_extension in ['tif', 'tiff']:
            image = tiff.imread(image_path).astype(np.float32)

            # Check if image is grayscale (mono) or color
            if len(image.shape) == 2:  # Grayscale image
                print("Detected grayscale (mono) image. Forcing luminance denoise mode.")
                denoise_mode = 'luminance'  # Force luminance mode for grayscale images
                image = np.stack([image] * 3, axis=-1)  # Convert grayscale to 3-channel for consistency
            else:
                print("Detected color image.")
            image = Image.fromarray((image * 255).astype(np.uint8))  # Convert float32 TIFF to uint8 image for PIL

        else:
            image = Image.open(image_path).convert('RGB')  # Convert to RGB for PIL handling
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

    if denoise_mode == 'luminance':
        print("Denoising Luminance (Y) channel only...")
        # Convert image to YCbCr, denoise Y channel (luminance)
        y, cb, cr = extract_ycbcr(image)
        denoised_y = denoise_channel(y, device, model)
        denoised_image = merge_ycbcr(denoised_y, cb, cr)
    else:
        print("Denoising full RGB channels (R, G, B)...")
        # Split image into R, G, and B channels and denoise each separately
        r, g, b = image.split()  # Split image into R, G, and B channels
        denoised_r = denoise_channel(np.array(r).astype(np.float32) / 255.0, device, model)
        denoised_g = denoise_channel(np.array(g).astype(np.float32) / 255.0, device, model)
        denoised_b = denoise_channel(np.array(b).astype(np.float32) / 255.0, device, model)

        # Merge the denoised R, G, and B channels back into an RGB image
        denoised_r = np.clip(denoised_r * 255, 0, 255).astype(np.uint8)
        denoised_g = np.clip(denoised_g * 255, 0, 255).astype(np.uint8)
        denoised_b = np.clip(denoised_b * 255, 0, 255).astype(np.uint8)
        denoised_image = Image.merge('RGB', (Image.fromarray(denoised_r), Image.fromarray(denoised_g), Image.fromarray(denoised_b)))

    return denoised_image


# Function to denoise a single channel
def denoise_channel(channel, device, model):
    # Split channel into chunks
    chunk_size = 256
    overlap = 64
    chunks = split_image_into_chunks_with_overlap(channel, chunk_size=chunk_size, overlap=overlap)

    denoised_chunks = []

    # Apply denoise model to each chunk
    for idx, (chunk, i, j) in enumerate(chunks):
        chunk_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            denoised_chunk = model(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
        denoised_chunks.append((denoised_chunk, i, j))

        # Show progress update
        show_progress(idx + 1, len(chunks))

    # Stitch the chunks back together
    denoised_channel = stitch_chunks_ignore_border(denoised_chunks, channel.shape, chunk_size=chunk_size, overlap=overlap)
    return denoised_channel


# Main process for denoising images
def process_images(input_dir, output_dir, denoise_strength=None, use_gpu=True, denoise_mode='luminance'):
    print((r"""
 *#        ___     __      ___       __                              #
 *#       / __/___/ /__   / _ | ___ / /________                      #
 *#      _\ \/ -_) _ _   / __ |(_-</ __/ __/ _ \                     #
 *#     /___/\__/_//_/  /_/ |_/___/\__/_/  \___/                     #
 *#                                                                  #
 *#              Cosmic Clarity - Denoise V1.0                       # 
 *#                                                                  #
 *#                         SetiAstro                                #
 *#                    Copyright © 2024                              #
 *#                                                                  #
        """))

    if denoise_strength is None:
        # Prompt for user input
        use_gpu, denoise_strength, denoise_mode = get_user_input()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models = load_model(exe_dir, use_gpu)

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        denoised_image = denoise_image(image_path, denoise_strength, models['device'], models["denoise_model"], denoise_mode)

        if denoised_image:
            file_extension = os.path.splitext(image_name)[1].lower()
            if file_extension in ['.tif', '.tiff']:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_denoised.tif")
                denoised_image_np = np.array(denoised_image).astype(np.float32) / 255.0
                tiff.imwrite(output_image_path, denoised_image_np, dtype=np.float32)
            else:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_denoised.png")
                denoised_image.save(output_image_path)

            print(f"Saved denoised image to: {output_image_path}")

# Define input and output directories
input_dir = os.path.join(exe_dir, 'input')
output_dir = os.path.join(exe_dir, 'output')

if not os.path.exists(input_dir):
    os.makedirs(input_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Add argument parsing for batch/script execution
parser = argparse.ArgumentParser(description="Cosmic Clarity Denoise Tool")
parser.add_argument('--denoise_strength', type=float, help="Denoise strength (0-1)")
parser.add_argument('--disable_gpu', action='store_true', help="Disable GPU acceleration and use CPU only")
parser.add_argument('--denoise_mode', type=str, choices=['luminance', 'full'], default='luminance', help="Denoise mode: luminance or full YCbCr denoising")

args = parser.parse_args()

# Determine whether to use GPU based on command-line argument
use_gpu = not args.disable_gpu

# Pass arguments if provided, or fall back to defaults
process_images(input_dir, output_dir, args.denoise_strength, use_gpu, args.denoise_mode)
