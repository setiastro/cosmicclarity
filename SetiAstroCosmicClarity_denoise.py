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
from tkinter import ttk
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

# Function to merge the denoised luminance (Y) with original chrominance (Cb and Cr) to reconstruct RGB image
def merge_luminance(y_denoised, cb, cr):
    # Ensure Y is clipped between 0 and 1, as we're working in 32-bit float precision
    y_denoised = np.clip(y_denoised, 0.0, 1.0)

    # Convert Cb and Cr from Pillow image objects to numpy arrays (if needed)
    cb = np.array(cb).astype(np.float32) / 255.0  # Convert to 32-bit float in range [0, 1]
    cr = np.array(cr).astype(np.float32) / 255.0  # Convert to 32-bit float in range [0, 1]

    # Recreate the YCbCr image as a stack of 32-bit float arrays (Y, Cb, Cr)
    ycbcr_image = np.stack([y_denoised, cb, cr], axis=-1)

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
def stitch_chunks_ignore_border(chunks, image_shape, chunk_size, overlap, border_size=5):
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
        global use_gpu, denoise_strength, denoise_mode
        use_gpu = gpu_var.get() == "Yes"
        denoise_strength = denoise_strength_slider.get()
        denoise_mode = denoise_mode_var.get().lower()  # Convert to lowercase for consistency
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

    # Submit button
    submit_button = ttk.Button(root, text="Submit", command=on_submit)
    submit_button.pack(pady=20)

    root.mainloop()  # Run the main event loop
    root.destroy()  # Destroy the window after quitting the loop

    return use_gpu, denoise_strength, denoise_mode



# Function to show progress during chunk processing
def show_progress(current, total):
    progress_percentage = (current / total) * 100
    print(f"Progress: {progress_percentage:.2f}% ({current}/{total} chunks processed)", end='\r', flush=True)


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

# Main denoise function for an image
def denoise_image(image_path, denoise_strength, device, model, denoise_mode='luminance'):
    try:
        # Load the image based on its extension
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension in ['.tif', '.tiff']:
            image = tiff.imread(image_path).astype(np.float32)  # Load as 32-bit float for TIFF
        else:
            image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0  # Load as 32-bit float for PNG

        # Check if the image is grayscale (mono) or color
        if len(image.shape) == 2:  # Grayscale image
            print("Detected grayscale (mono) image. Denoising single channel directly.")
            denoised_image = denoise_channel(image, device, model)
            denoised_image = blend_images(image, denoised_image, denoise_strength)
            return replace_border(image, denoised_image)

        print("Detected color image.")
        if denoise_mode == 'luminance':
            print("Denoising Luminance (Y) channel only...")
            y, cb, cr = extract_luminance(image)
            denoised_y = denoise_channel(y, device, model)
            denoised_y = blend_images(y, denoised_y, denoise_strength)
            denoised_image = merge_luminance(replace_border(y, denoised_y), cb, cr)
            return denoised_image

        print("Denoising full RGB channels...")
        denoised_channels = [blend_images(image[:, :, c], denoise_channel(image[:, :, c], device, model), denoise_strength)
                             for c in range(3)]
        denoised_image = np.stack([replace_border(image[:, :, c], denoised_channels[c]) for c in range(3)], axis=-1)
        return denoised_image

    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None


# Function to denoise a channel (Y, Cb, Cr)
def denoise_channel(channel, device, model):
    # Split channel into chunks
    chunk_size = 256
    overlap = 64
    chunks = split_image_into_chunks_with_overlap(channel, chunk_size=chunk_size, overlap=overlap)

    denoised_chunks = []

    # Apply denoise model to each chunk
    for idx, (chunk, i, j) in enumerate(chunks):
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            denoised_chunk = model(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]

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
 *#              Cosmic Clarity - Denoise V2.2                       # 
 *#                                                                  #
 *#                         SetiAstro                                #
 *#                    Copyright © 2024                              #
 *#                                                                  #
        """))

    if denoise_strength is None:
        # Prompt for user input
        use_gpu, denoise_strength, denoise_mode = get_user_input()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the denoise model
    models = load_model(exe_dir, use_gpu)

    # Process each image in the input directory
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        file_extension = os.path.splitext(image_name)[1].lower()
        denoised_image = denoise_image(image_path, denoise_strength, models['device'], models["denoise_model"], denoise_mode)

        if denoised_image is not None:
            output_image_name = os.path.splitext(image_name)[0] + "_denoised"

            # Save the image based on its extension
            if file_extension in ['.tif', '.tiff']:
                output_image_path = os.path.join(output_dir, output_image_name + ".tif")
                tiff.imwrite(output_image_path, denoised_image.astype(np.float32))
                print(f"Saved 32-bit denoised image to: {output_image_path}")
            else:
                output_image_path = os.path.join(output_dir, output_image_name + ".png")
                denoised_image_8bit = (denoised_image * 255).astype(np.uint8)
                denoised_image_pil = Image.fromarray(denoised_image_8bit)
                denoised_image_pil.save(output_image_path)
                print(f"Saved 8-bit denoised image to: {output_image_path}")

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
