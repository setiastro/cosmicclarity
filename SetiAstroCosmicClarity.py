import warnings
import os
import sys
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

class SharpeningCNN(nn.Module):
    def __init__(self):
        super(SharpeningCNN, self).__init__()
        
        # Encoder (down-sampling path)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 1st layer (3 -> 16 feature maps)
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Additional layer (16 -> 16)
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 2nd layer (16 -> 32 feature maps)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Additional layer (32 -> 32)
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),  # 3rd layer (32 -> 64) with dilation
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),  # Additional layer (64 -> 64) with dilation
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 4th layer (64 -> 128 feature maps)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Additional layer (128 -> 128)
            nn.ReLU()
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2, dilation=2),  # 5th layer (128 -> 256) with dilation
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2),  # Additional layer (256 -> 256) with dilation
            nn.ReLU()
        )
        
        # Decoder (up-sampling path with skip connections)
        self.decoder5 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),  # 256 + 128 feature maps from encoder4
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Additional layer
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),  # 128 + 64 feature maps from encoder3
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Additional layer
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),  # 64 + 32 feature maps from encoder2
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Additional layer
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1),  # 32 + 16 feature maps from encoder1
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Additional layer
            nn.ReLU()
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
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stellar_model_radius_1 = SharpeningCNN()
    nonstellar_model_radius_1 = SharpeningCNN()
    nonstellar_model_radius_2 = SharpeningCNN()
    nonstellar_model_radius_4 = SharpeningCNN()
    nonstellar_model_radius_8 = SharpeningCNN()

    # Load models onto the correct device
    stellar_model_radius_1.load_state_dict(torch.load(os.path.join(exe_dir, 'deep_sharp_stellar_cnn.pth'), map_location=device))
    nonstellar_model_radius_1.load_state_dict(torch.load(os.path.join(exe_dir, 'deep_nonstellar_sharp_cnn_radius_1.pth'), map_location=device))
    nonstellar_model_radius_2.load_state_dict(torch.load(os.path.join(exe_dir, 'deep_nonstellar_sharp_cnn_radius_2.pth'), map_location=device))
    nonstellar_model_radius_4.load_state_dict(torch.load(os.path.join(exe_dir, 'deep_nonstellar_sharp_cnn_radius_4.pth'), map_location=device))
    nonstellar_model_radius_8.load_state_dict(torch.load(os.path.join(exe_dir, 'deep_nonstellar_sharp_cnn_radius_8.pth'), map_location=device))

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
    stellar_amount_slider.set(0.9)  # Set default value to 0.9

    # Non-Stellar amount slider
    nonstellar_amount_label = ttk.Label(root, text="Non-Stellar Sharpening Amount (0-1):")
    nonstellar_amount_slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient="horizontal")
    nonstellar_amount_slider.set(0.9)  # Set default value to 0.9

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


# Function to stretch an image
def stretch_image(image):
    """
    Perform a linear stretch on the image.
    """
    original_min = np.min(image)
    stretched_image = image - original_min
    original_median = np.median(stretched_image, axis=(0, 1)) if image.ndim == 3 else np.median(stretched_image)
    
    target_median = 0.25
    if image.ndim == 3:
        median_color = np.mean(np.median(stretched_image, axis=(0, 1)))
        stretched_image = ((median_color - 1) * target_median * stretched_image) / (
            median_color * (target_median + stretched_image - 1) - target_median * stretched_image)
    else:
        image_median = np.median(stretched_image)
        stretched_image = ((image_median - 1) * target_median * stretched_image) / (
            image_median * (target_median + stretched_image - 1) - target_median * stretched_image)
    
    stretched_image = np.clip(stretched_image, 0, 1)
    
    return stretched_image, original_min, original_median

# Function to unstretch an image
def unstretch_image(image, original_median, original_min):
    """
    Undo the stretch to return the image to the original linear state.
    """
    if image.ndim == 3:
        median_color = np.mean(np.median(image, axis=(0, 1)))
        unstretched_image = ((median_color - 1) * original_median * image) / (
            median_color * (original_median + image - 1) - original_median * image)
    else:
        image_median = np.median(image)
        unstretched_image = ((image_median - 1) * original_median * image) / (
            image_median * (original_median + image - 1) - original_median * image)
    
    unstretched_image += original_min
    unstretched_image = np.clip(unstretched_image, 0, 1)
    
    return unstretched_image

# Function to sharpen image
def sharpen_image(image_path, sharpening_mode, nonstellar_strength, stellar_amount, nonstellar_amount, device, models, sharpen_channels_separately):
    image = None
    file_extension = image_path.lower().split('.')[-1]
    is_mono = False  # Initialize is_mono as False by default    

    try:
        if file_extension in ['tif', 'tiff']:
            # Load the TIFF image
            image = tiff.imread(image_path)
            if image.dtype == np.uint16:  # If 16-bit, convert to 32-bit float
                image = image.astype(np.float32) / 65535.0
            elif image.dtype == np.uint32:  # If 32-bit unsigned, convert to 32-bit float
                image = image.astype(np.float32) / 4294967295.0
            else:
                image = image.astype(np.float32)
            if len(image.shape) == 2:  # Grayscale image
                image = np.stack([image] * 3, axis=-1)  # Convert grayscale to 3-channel for consistency
                is_mono = True
            else:
                is_mono = False
        elif file_extension in ['fits', 'fit']:
            # Load the FITS image
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data.astype(np.float32)
                image = image_data / np.max(image_data)  # Normalize to [0, 1]
                if len(image.shape) == 2:  # Grayscale image
                    image = np.stack([image] * 3, axis=-1)  # Convert grayscale to 3-channel for consistency
                    is_mono = True
                else:
                    is_mono = False
        else:
            # For non-TIFF/FITS files, read and convert to 32-bit float for consistency
            image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
            is_mono = False
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

    # Check if the image needs stretching
    stretch_needed = np.median(image) < 0.125
    if stretch_needed:
        # Stretch the image
        stretched_image, original_min, original_median = stretch_image(image)
    else:
        stretched_image = image

    # If sharpening channels separately, split the image into R, G, and B channels
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
            luminance, cb, cr = extract_luminance(Image.fromarray((stretched_image * 255).astype(np.uint8)))
        else:
            luminance = stretched_image

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
        if sharpening_mode == "Non-Stellar Only" or sharpening_mode == "Both":
            print("Non-Stellar Sharpening:")
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
                show_progress(idx + 1, total_chunks)

            print("")  # Add a newline after non-stellar sharpening progress
            nonstellar_sharpened = stitch_chunks_ignore_border(denoised_chunks, luminance.shape, chunk_size=256, overlap=64)
            sharpened_luminance = blend_images(luminance, nonstellar_sharpened, nonstellar_amount)

        # Initialize stellar_sharpened_chunks before using it
        stellar_sharpened_chunks = []

        # Apply stellar sharpening (fixed at strength 1)
        if sharpening_mode == "Stellar Only" or sharpening_mode == "Both":
            print("Stellar Sharpening:")
            for idx, (chunk, i, j, is_edge) in enumerate(chunks):
                chunk_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    sharpened_chunk = models["stellar_model"](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
                stellar_sharpened_chunks.append((sharpened_chunk, i, j, is_edge))
                show_progress(idx + 1, total_chunks)

            print("")  # Add a newline after stellar sharpening progress
            stellar_sharpened_luminance = stitch_chunks_ignore_border(stellar_sharpened_chunks, luminance.shape, chunk_size=256, overlap=64)

            # Blend based on the mode:
            if sharpening_mode == "Stellar Only":
                sharpened_luminance = blend_images(luminance, stellar_sharpened_luminance, stellar_amount)
            elif sharpening_mode == "Both" and nonstellar_sharpened is not None:
                sharpened_luminance = blend_images(sharpened_luminance, stellar_sharpened_luminance, stellar_amount)

        # For color images, merge back the luminance with the original chrominance (Cb, Cr)
        if len(image.shape) == 3:
            sharpened_image = merge_luminance(sharpened_luminance, cb, cr)
        else:
            # For grayscale images, the luminance is the image itself
            sharpened_image = sharpened_luminance

    # Unstretch the image to return to the original linear state if stretched
    if stretch_needed:
        sharpened_image = unstretch_image(sharpened_image, original_median, original_min)

    # Replace the 5-pixel border from the original image
    sharpened_image = replace_border(image, sharpened_image)

    return sharpened_image, is_mono

# Helper function to sharpen individual R, G, B channels
def sharpen_channel(channel, sharpening_mode, nonstellar_strength, stellar_amount, nonstellar_amount, device, models):
    chunks = split_image_into_chunks_with_overlap(channel, chunk_size=256, overlap=64)
    total_chunks = len(chunks)

    denoised_chunks = []
    nonstellar_sharpened = None
    sharpened_channel = channel  # Initialize in case neither path modifies it

    # Use dictionary to select the non-stellar model based on nonstellar_strength
    model_map = {
        1: models["nonstellar_model_1"],
        2: models["nonstellar_model_2"],
        4: models["nonstellar_model_4"],
        8: models["nonstellar_model_8"]
    }

    # Apply non-stellar sharpening if applicable
    if sharpening_mode == "Non-Stellar Only" or sharpening_mode == "Both":
        print("Non-Stellar Sharpening Channel:")
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
            show_progress(idx + 1, total_chunks)

        print("")  # Add a newline after non-stellar sharpening channel progress
        nonstellar_sharpened = stitch_chunks_ignore_border(denoised_chunks, channel.shape, chunk_size=256, overlap=64)
        sharpened_channel = blend_images(channel, nonstellar_sharpened, nonstellar_amount)

    # Initialize stellar_sharpened_chunks before using it
    stellar_sharpened_chunks = []

    # Apply stellar sharpening (fixed at strength 1)
    if sharpening_mode == "Stellar Only" or sharpening_mode == "Both":
        print("Stellar Sharpening Channel:")
        for idx, (chunk, i, j, is_edge) in enumerate(chunks):
            chunk_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                sharpened_chunk = models["stellar_model"](chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().detach().numpy()[0]
            stellar_sharpened_chunks.append((sharpened_chunk, i, j, is_edge))
            show_progress(idx + 1, total_chunks)

        print("")  # Add a newline after stellar sharpening channel progress
        stellar_sharpened_channel = stitch_chunks_ignore_border(stellar_sharpened_chunks, channel.shape, chunk_size=256, overlap=64)

        # Blend based on the mode:
        if sharpening_mode == "Stellar Only":
            sharpened_channel = blend_images(channel, stellar_sharpened_channel, stellar_amount)
        elif sharpening_mode == "Both" and nonstellar_sharpened is not None:
            sharpened_channel = blend_images(sharpened_channel, stellar_sharpened_channel, stellar_amount)

    return sharpened_channel



def process_images(input_dir, output_dir, sharpening_mode=None, nonstellar_strength=None, stellar_amount=None, nonstellar_amount=None, use_gpu=True, sharpen_channels_separately=False):
    print((r"""
 *#        ___     __      ___       __                              #
 *#       / __/___/ /__   / _ | ___ / /________                      #
 *#      _\ \/ -_) _ _   / __ |(_-</ __/ __/ _ \                     #
 *#     /___/\__/\//_/  /_/ |_/___/\__/__/ \___/                     #
 *#                                                                  #
 *#              Cosmic Clarity - Sharpen V5.2                       # 
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
        
        # Capture both sharpened_image and is_mono
        sharpened_image, is_mono = sharpen_image(image_path, sharpening_mode, nonstellar_strength, stellar_amount, nonstellar_amount, models['device'], models, sharpen_channels_separately)

        if sharpened_image is not None:
            file_extension = os.path.splitext(image_name)[1].lower()

            # Check if the original image was grayscale
            if file_extension in ['.fits', '.fit']:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_sharpened.fits")
                if is_mono:
                    hdu = fits.PrimaryHDU(sharpened_image[:, :, 0])  # Save only the first channel for grayscale
                else:
                    hdu = fits.PrimaryHDU(sharpened_image)  # Save the full image for RGB
                hdu.writeto(output_image_path, overwrite=True)
                print(f"Saved 32-bit sharpened image to: {output_image_path}")
            elif file_extension in ['.tif', '.tiff']:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_sharpened.tif")
                
                if is_mono:
                    # Save only the first channel (grayscale)
                    tiff.imwrite(output_image_path, sharpened_image[:, :, 0].astype(np.float32))
                else:
                    # Save the sharpened image as 32-bit TIFF (RGB)
                    tiff.imwrite(output_image_path, sharpened_image.astype(np.float32))
                    
                print(f"Saved 32-bit sharpened image to: {output_image_path}")
            else:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_sharpened.png")
                
                if is_mono:
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
parser.add_argument('--nonstellar_amount', type=float, default=0.9, help="Non-Stellar sharpening amount (0-1)")
parser.add_argument('--disable_gpu', action='store_true', help="Disable GPU acceleration and use CPU only")
parser.add_argument('--sharpen_channels_separately', action='store_true', help="Sharpen R, G, and B channels separately")

args = parser.parse_args()

# Determine whether to use GPU based on command-line argument
use_gpu = not args.disable_gpu  # If --disable_gpu is passed, set use_gpu to False

# Pass arguments if provided, or fall back to user input if no command-line arguments are provided
process_images(input_dir, output_dir, args.sharpening_mode, args.nonstellar_strength, args.stellar_amount, args.nonstellar_amount, use_gpu, args.sharpen_channels_separately)