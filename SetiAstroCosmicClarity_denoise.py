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
from tkinter import ttk
import argparse  # For command-line argument parsing

# Suppress model loading warnings
warnings.filterwarnings("ignore")

# Define the DenoiseCNN model with adjusted convolutional layers
class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        
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

# Function to initialize and load the denoise model
def load_model(exe_dir, use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    denoise_model = DenoiseCNN()
    denoise_model.load_state_dict(torch.load(os.path.join(exe_dir, 'deep_denoise_cnn.pth'), map_location=device))

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

import numpy as np

# Function to stretch an image
def stretch_image(image):
    """
    Perform a linear stretch on the image.
    """
    original_min = np.min(image)
    stretched_image = image - original_min  # Shift image so that the min is 0
    original_median = np.median(stretched_image, axis=(0, 1)) if image.ndim == 3 else np.median(stretched_image)



    target_median = 0.25
    if image.ndim == 3:  # RGB image case
        # Calculate the overall median for color images as in original code
        median_color = np.mean(np.median(stretched_image, axis=(0, 1)))
        stretched_image = ((median_color - 1) * target_median * stretched_image) / (
            median_color * (target_median + stretched_image - 1) - target_median * stretched_image)
        stretched_medians = np.median(stretched_image, axis=(0, 1))
    else:  # Grayscale image case
        image_median = np.median(stretched_image)
        stretched_image = ((image_median - 1) * target_median * stretched_image) / (
            image_median * (target_median + stretched_image - 1) - target_median * stretched_image)
        stretched_medians = np.median(stretched_image)



    stretched_image = np.clip(stretched_image, 0, 1)  # Clip to [0, 1] range

    return stretched_image, original_min, original_median

# Function to unstretch an image with final median adjustment
def unstretch_image(image, original_median, original_min):
    """
    Undo the stretch to return the image to the original linear state.
    """
    if image.ndim == 3:  # RGB image case
        # Use the overall median to revert the stretch for color images
        median_color = np.mean(np.median(image, axis=(0, 1)))
        unstretched_image = ((median_color - 1) * original_median * image) / (
            median_color * (original_median + image - 1) - original_median * image)
        final_medians = np.median(unstretched_image, axis=(0, 1))

        # Adjust each channel to match the original median
        for c in range(3):  # R, G, B channels
            unstretched_image[..., c] *= original_median[c] / final_medians[c]
        adjusted_medians = np.median(unstretched_image, axis=(0, 1))
    else:  # Grayscale image case
        image_median = np.median(image)
        unstretched_image = ((image_median - 1) * original_median * image) / (
            image_median * (original_median + image - 1) - original_median * image)
        final_medians = np.median(unstretched_image)

        # Adjust for grayscale case
        unstretched_image *= original_median / final_medians
        adjusted_medians = np.median(unstretched_image)


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

# Function to denoise the image
def denoise_image(image_path, denoise_strength, device, model, denoise_mode='luminance'):
    # Get file extension
    file_extension = os.path.splitext(image_path)[1].lower()
    if file_extension not in ['.png', '.tif', '.tiff', '.fit', '.fits']:
        print(f"Ignoring non-image file: {image_path}")
        return None, None, None, None

    original_header = None
    image = None
    bit_depth = "32-bit float"  # Default bit depth

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
            print(f"Final bit depth set to: {bit_depth}")

            # Handle alpha channels and grayscale TIFFs
            if image.ndim == 3 and image.shape[-1] == 4:
                print("Detected alpha channel in TIFF. Removing it.")
                image = image[:, :, :3]
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)  # Convert to RGB
                is_mono = True

        elif file_extension in ['.fits', '.fit']:
            # Load the FITS image
            with fits.open(image_path) as hdul:
                image_data = hdul[0].data
                original_header = hdul[0].header  # Capture the FITS header

                # Determine the bit depth based on the data type in the FITS file
                if image_data.dtype == np.uint16:
                    bit_depth = "16-bit"
                    print("Identified 16-bit FITS image.")
                elif image_data.dtype == np.float32:
                    bit_depth = "32-bit floating point"
                    print("Identified 32-bit floating point FITS image.")
                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    print("Identified 32-bit unsigned FITS image.")

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


        else:  # Assume 8-bit PNG
            image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
            bit_depth = "8-bit"
            print(f"Loaded {bit_depth} PNG image.")

        if image is None:
            raise ValueError(f"Failed to load image from: {image_path}")

        # Add a border around the image with the median value
        image_with_border = add_border(image, border_size=5)

        # Check if the image needs stretching based on its median value
        original_median = np.median(image_with_border)
        if original_median < 0.125:
            stretched_image, original_min, original_median = stretch_image(image_with_border)
        else:
            stretched_image = image_with_border
            original_min = None

        # Process grayscale or color images
        if image.ndim == 2:
            denoised_image = denoise_channel(stretched_image, device, model)
            denoised_image = blend_images(stretched_image, denoised_image, denoise_strength)
        else:
            if denoise_mode == 'luminance':
                y, cb, cr = extract_luminance(stretched_image)
                denoised_y = denoise_channel(y, device, model)
                denoised_y = blend_images(y, denoised_y, denoise_strength)
                denoised_image = merge_luminance(denoised_y, cb, cr)
            else:
                # Denoise each RGB channel separately
                denoised_r = denoise_channel(stretched_image[:, :, 0], device, model)
                denoised_g = denoise_channel(stretched_image[:, :, 1], device, model)
                denoised_b = denoise_channel(stretched_image[:, :, 2], device, model)
                
                # Blend each denoised channel with the original channel
                denoised_r = blend_images(stretched_image[:, :, 0], denoised_r, denoise_strength)
                denoised_g = blend_images(stretched_image[:, :, 1], denoised_g, denoise_strength)
                denoised_b = blend_images(stretched_image[:, :, 2], denoised_b, denoise_strength)
                
                # Stack blended channels into the final denoised image
                denoised_image = np.stack([denoised_r, denoised_g, denoised_b], axis=-1)

        # Unstretch if stretched previously
        if original_min is not None:
            denoised_image = unstretch_image(denoised_image, original_median, original_min)

        # Remove the border added around the image
        denoised_image = remove_border(denoised_image, border_size=5)

        return denoised_image, original_header, bit_depth, file_extension

    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None, None, None

# Function to denoise a channel (Y, Cb, Cr, or individual RGB)
def denoise_channel(channel, device, model):
    # Split channel into chunks
    chunk_size = 256
    overlap = 64
    chunks = split_image_into_chunks_with_overlap(channel, chunk_size=chunk_size, overlap=overlap)

    denoised_chunks = []

    # Apply denoise model to each chunk
    for idx, (chunk, i, j) in enumerate(chunks):
        # Prepare the chunk tensor
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions

        # Run the denoising model
        with torch.no_grad():
            denoised_chunk = model(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]  # Denoise and return the first channel

        denoised_chunks.append((denoised_chunk, i, j))

        # Show progress update
        show_progress(idx + 1, len(chunks))

    print("")  # Add a newline after denoising channel progress

    # Stitch the chunks back together
    denoised_channel = stitch_chunks_ignore_border(denoised_chunks, channel.shape, chunk_size=chunk_size, overlap=overlap)
    return denoised_channel


# Main process for denoising images
def process_images(input_dir, output_dir, denoise_strength=None, use_gpu=True, denoise_mode='luminance'):
    print((r"""
 *#        ___     __      ___       __                              #
 *#       / __/___/ /__   / _ | ___ / /________                      #
 *#      _\ \/ -_) _ _   / __ |(_-</ __/ __/ _ \                     #
 *#     /___/\__/\//_/  /_/ |_/___/\__/__/ \___/                     #
 *#                                                                  #
 *#              Cosmic Clarity - Denoise V5.4.1                     # 
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
        denoised_image, original_header, bit_depth, file_extension = denoise_image(
            image_path, denoise_strength, models['device'], models["denoise_model"], denoise_mode
        )

        if denoised_image is not None:
            output_image_name = os.path.splitext(image_name)[0] + "_denoised"
            output_image_path = os.path.join(output_dir, output_image_name + file_extension)
            actual_bit_depth = bit_depth  # Track actual bit depth for reporting

            # Save as FITS file with header information if the original was FITS
            if file_extension in ['.fits', '.fit']:
                if original_header is not None:
                    if denoised_image.ndim == 2:  # Grayscale
                        denoised_image_fits = (denoised_image * 65535).astype(np.uint16) if bit_depth == "16-bit" else denoised_image.astype(np.float32)
                        hdu = fits.PrimaryHDU(denoised_image_fits, header=original_header)
                    else:  # RGB
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

                        original_header['NAXIS'] = 3
                        original_header['NAXIS1'] = denoised_image_transposed.shape[2]
                        original_header['NAXIS2'] = denoised_image_transposed.shape[1]
                        original_header['NAXIS3'] = denoised_image_transposed.shape[0]

                        hdu = fits.PrimaryHDU(denoised_image_fits, header=original_header)

                    hdu.writeto(output_image_path, overwrite=True)
                    print(f"Saved {actual_bit_depth} denoised image to: {output_image_path}")

            # Save as TIFF based on the original bit depth if the original was TIFF
            elif file_extension in ['.tif', '.tiff']:
                if bit_depth == "16-bit":
                    actual_bit_depth = "16-bit"
                    if denoised_image.ndim == 2:  # Grayscale
                        tiff.imwrite(output_image_path, (denoised_image[:, :, 0] * 65535).astype(np.uint16))
                    else:  # RGB
                        tiff.imwrite(output_image_path, (denoised_image * 65535).astype(np.uint16))
                elif bit_depth == "32-bit unsigned":
                    actual_bit_depth = "32-bit unsigned"
                    if denoised_image.ndim == 2:  # Grayscale
                        tiff.imwrite(output_image_path, (denoised_image[:, :, 0] * 4294967295).astype(np.uint32))
                    else:  # RGB
                        tiff.imwrite(output_image_path, (denoised_image * 4294967295).astype(np.uint32))           
                else:
                    actual_bit_depth = "32-bit float"
                    if denoised_image.ndim == 2:  # Grayscale
                        tiff.imwrite(output_image_path, denoised_image[:, :, 0].astype(np.float32))
                    else:  # RGB
                        tiff.imwrite(output_image_path, denoised_image.astype(np.float32))

                print(f"Saved {actual_bit_depth} denoised image to: {output_image_path}")

            # Save as 8-bit PNG if the original was PNG
            else:
                output_image_path = os.path.join(output_dir, output_image_name + ".png")
                denoised_image_8bit = (denoised_image * 255).astype(np.uint8)
                denoised_image_pil = Image.fromarray(denoised_image_8bit)
                actual_bit_depth = "8-bit"
                denoised_image_pil.save(output_image_path)
                print(f"Saved {actual_bit_depth} denoised image to: {output_image_path}")




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
