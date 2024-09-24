import os
import torch
import numpy as np
import torch.nn as nn
import tifffile as tiff  # For handling .tif files
from PIL import Image
import tkinter as tk
from tkinter import simpledialog, messagebox

# CNN model for sharpening
class SharpeningCNN(nn.Module):
    def __init__(self):
        super(SharpeningCNN, self).__init__()
        
        # Encoder (down-sampling path)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 1st convolutional layer for 3 channels -> 64 feature maps
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 2nd convolutional layer 64 -> 128 feature maps
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 3rd convolutional layer 128 -> 256 feature maps
            nn.ReLU()
        )
        
        # Decoder (up-sampling path)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 1st deconvolutional layer 256 -> 128 feature maps
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 2nd deconvolutional layer 128 -> 64 feature maps
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Output layer for 64 -> 3 channels (RGB)
            nn.Sigmoid()  # Ensure output values are between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load models for stellar sharpening (radius 1, 2, 4)
stellar_model_radius_1 = SharpeningCNN()
stellar_model_radius_2 = SharpeningCNN()
stellar_model_radius_4 = SharpeningCNN()
stellar_model_radius_1.load_state_dict(torch.load('sharp_cnn_radius_1.pth'))
stellar_model_radius_2.load_state_dict(torch.load('sharp_cnn_radius_2.pth'))
stellar_model_radius_4.load_state_dict(torch.load('sharp_cnn_radius_4.pth'))
stellar_model_radius_1.eval()
stellar_model_radius_2.eval()
stellar_model_radius_4.eval()

# Load models for non-stellar sharpening (radius 1, 2, 4, 8)
nonstellar_model_radius_1 = SharpeningCNN()
nonstellar_model_radius_2 = SharpeningCNN()
nonstellar_model_radius_4 = SharpeningCNN()
nonstellar_model_radius_8 = SharpeningCNN()
nonstellar_model_radius_1.load_state_dict(torch.load('nonstellar_sharp_cnn_radius_1.pth'))
nonstellar_model_radius_2.load_state_dict(torch.load('nonstellar_sharp_cnn_radius_2.pth'))
nonstellar_model_radius_4.load_state_dict(torch.load('nonstellar_sharp_cnn_radius_4.pth'))
nonstellar_model_radius_8.load_state_dict(torch.load('nonstellar_sharp_cnn_radius_8.pth'))
nonstellar_model_radius_1.eval()
nonstellar_model_radius_2.eval()
nonstellar_model_radius_4.eval()
nonstellar_model_radius_8.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stellar_model_radius_1.to(device)
stellar_model_radius_2.to(device)
stellar_model_radius_4.to(device)
nonstellar_model_radius_1.to(device)
nonstellar_model_radius_2.to(device)
nonstellar_model_radius_4.to(device)
nonstellar_model_radius_8.to(device)

# Function to extract luminance (Y channel) from an RGB image
def extract_luminance(image):
    ycbcr_image = image.convert('YCbCr')
    y, cb, cr = ycbcr_image.split()
    return np.array(y).astype(np.float32) / 255.0, cb, cr  # Normalize Y to [0, 1]

# Function to merge the sharpened luminance (Y) with original chrominance (Cb and Cr) to reconstruct RGB image
def merge_luminance(y_sharpened, cb, cr):
    y_sharpened = np.clip(y_sharpened * 255, 0, 255).astype(np.uint8)  # Ensure values are in 0-255 range
    y_sharpened_img = Image.fromarray(y_sharpened)  # Convert Y back to 0-255 image
    ycbcr_image = Image.merge('YCbCr', (y_sharpened_img, cb, cr))
    return ycbcr_image.convert('RGB')  # Convert back to RGB

# Function to interpolate the results based on strength for stellar models
def interpolate_stellar_sharpening(sharpened_1, sharpened_2, sharpened_4, strength):
    """ Interpolates between stellar models based on strength """
    if strength <= 2:
        return (2 - strength) * sharpened_1 + (strength - 1) * sharpened_2
    else:
        return ((4 - strength) * sharpened_2 + (strength - 2) * sharpened_4) / 2

# Function to interpolate the results based on strength for non-stellar models
def interpolate_nonstellar_sharpening(sharpened_1, sharpened_2, sharpened_4, sharpened_8, strength):
    """ Interpolates between non-stellar models based on strength """
    if strength <= 2:
        return (2 - strength) * sharpened_1 + (strength - 1) * sharpened_2
    elif 2 < strength <= 4:
        return ((4 - strength) * sharpened_2 + (strength - 2) * sharpened_4) / 2
    else:
        return ((8 - strength) * sharpened_4 + (strength - 4) * sharpened_8) / 4

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

# Function to stitch overlapping chunks back together with soft blending while ignoring borders for inner chunks
def stitch_chunks_ignore_border(chunks, image_shape, chunk_size, overlap, border_size=5):
    stitched_image = np.zeros(image_shape, dtype=np.float32)
    weight_map = np.zeros(image_shape, dtype=np.float32)  # Track blending weights
    
    for chunk, i, j, is_edge in chunks:
        actual_chunk_h, actual_chunk_w = chunk.shape
        
        # Calculate the boundaries for the current chunk, ignoring the border
        border_h = min(border_size, actual_chunk_h // 2)
        border_w = min(border_size, actual_chunk_w // 2)

        if is_edge:
            # Don't ignore the borders for edge chunks, just add them directly
            stitched_image[i:i+actual_chunk_h, j:j+actual_chunk_w] += chunk
            weight_map[i:i+actual_chunk_h, j:j+actual_chunk_w] += 1
        else:
            # Ignore the 5-pixel border for interior chunks
            inner_chunk = chunk[border_h:actual_chunk_h-border_h, border_w:actual_chunk_w-border_w]
            stitched_image[i+border_h:i+actual_chunk_h-border_h, j+border_w:j+actual_chunk_w-border_w] += inner_chunk
            weight_map[i+border_h:i+actual_chunk_h-border_h, j+border_w:j+actual_chunk_w-border_w] += 1

    # Normalize by weight_map to blend overlapping areas
    stitched_image /= np.maximum(weight_map, 1)  # Avoid division by zero
    return stitched_image

# Function to get user input for stellar and non-stellar strength and sharpening mode
def get_user_input():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Allow "Stellar" or "Stellar Only" and "Non-Stellar" or "Non-Stellar Only" as valid inputs
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
    
    stellar_strength = simpledialog.askfloat("Stellar PSF Spread", "Enter stellar sharpening: Fine to Coarse (1-2):", minvalue=1, maxvalue=4)
    nonstellar_strength = simpledialog.askfloat("Non-Stellar Detail Level", "Enter non-stellar detail enhancement level: Fine to Coarse (1-8):", minvalue=1, maxvalue=8)
    
    root.destroy()
    
    return sharpening_mode, stellar_strength, nonstellar_strength



# Function to sharpen an image
def sharpen_image(image_path, sharpening_mode, stellar_strength, nonstellar_strength):
    image = None
    file_extension = image_path.lower().split('.')[-1]

    try:
        if file_extension in ['tif', 'tiff']:
            image = tiff.imread(image_path).astype(np.float32)
            if len(image.shape) == 2:  # If grayscale, convert to RGB
                image = np.stack([image]*3, axis=-1)
            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Could not read image: {image_path}. Error: {e}")
        return None

    luminance, cb, cr = extract_luminance(image)
    chunks = split_image_into_chunks_with_overlap(luminance, chunk_size=256, overlap=64)

    denoised_chunks = []

    # Apply non-stellar sharpening
    if sharpening_mode == "Non-Stellar Only" or sharpening_mode == "Both":
        for chunk, i, j, is_edge in chunks:
            chunk_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
            if nonstellar_strength in [1, 2, 4, 8]:
                with torch.no_grad():
                    active_model = eval(f'nonstellar_model_radius_{int(nonstellar_strength)}')
                    sharpened_chunk = active_model(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]
            else:
                if nonstellar_strength <= 4:
                    sharpened_chunk_a = nonstellar_model_radius_2(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]
                    sharpened_chunk_b = nonstellar_model_radius_4(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]
                    sharpened_chunk = interpolate_nonstellar_sharpening(None, sharpened_chunk_a, sharpened_chunk_b, None, nonstellar_strength)
                else:
                    sharpened_chunk_a = nonstellar_model_radius_4(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]
                    sharpened_chunk_b = nonstellar_model_radius_8(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]
                    sharpened_chunk = interpolate_nonstellar_sharpening(None, None, sharpened_chunk_a, sharpened_chunk_b, nonstellar_strength)
        
            denoised_chunks.append((sharpened_chunk, i, j, is_edge))

    # Apply stellar sharpening
    if sharpening_mode == "Stellar Only" or sharpening_mode == "Both":
        for chunk, i, j, is_edge in chunks:
            chunk_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
            if stellar_strength in [1, 2, 4]:
                with torch.no_grad():
                    active_model = eval(f'stellar_model_radius_{int(stellar_strength)}')
                    sharpened_chunk = active_model(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]
            else:
                if stellar_strength <= 2:
                    sharpened_chunk_a = stellar_model_radius_1(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]
                    sharpened_chunk_b = stellar_model_radius_2(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]
                    sharpened_chunk = interpolate_stellar_sharpening(sharpened_chunk_a, sharpened_chunk_b, None, stellar_strength)
                else:
                    sharpened_chunk_a = stellar_model_radius_2(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]
                    sharpened_chunk_b = stellar_model_radius_4(chunk_tensor.repeat(1, 3, 1, 1)).squeeze().cpu().numpy()[0]
                    sharpened_chunk = interpolate_stellar_sharpening(None, sharpened_chunk_a, sharpened_chunk_b, stellar_strength)
        
            denoised_chunks.append((sharpened_chunk, i, j, is_edge))
        
    # Stitch the sharpened chunks back together
    sharpened_luminance = stitch_chunks_ignore_border(denoised_chunks, luminance.shape, chunk_size=256, overlap=64)

    # Merge the sharpened luminance with the chrominance channels
    sharpened_image = merge_luminance(sharpened_luminance, cb, cr)
    return sharpened_image

# Main process for sharpening images
def process_images(input_dir, output_dir):
    sharpening_mode, stellar_strength, nonstellar_strength = get_user_input()

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        sharpened_image = sharpen_image(image_path, sharpening_mode, stellar_strength, nonstellar_strength)
        
        if sharpened_image:
            # Extract file extension
            file_extension = os.path.splitext(image_name)[1].lower()

            # Adjust the output format to match the input format
            if file_extension in ['.tif', '.tiff']:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_sharpened.tif")
                sharpened_image_np = np.array(sharpened_image).astype(np.float32) / 255.0  # Normalize to [0, 1]
                tiff.imwrite(output_image_path, sharpened_image_np, dtype=np.float32)
            else:
                output_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "_sharpened.png")
                sharpened_image.save(output_image_path)

            print(f"Saved sharpened image to: {output_image_path}")


# Example directory setup
input_dir = r'C:/Users/Gaming/Desktop/Python Code/input'
output_dir = r'C:/Users/Gaming/Desktop/Python Code/output'

process_images(input_dir, output_dir)
