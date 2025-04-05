# patch_torch.py
import warnings
from xisf import XISF
import os
import sys
import torch
import numpy as np
import torch.nn as nn
import tifffile as tiff
from astropy.io import fits
from PIL import Image
import argparse  # For command-line argument parsing
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton,
    QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
# Suppress model loading warnings
warnings.filterwarnings("ignore")
try:
    # Check if the attribute exists; if not, set it to a dummy value
    if not hasattr(torch._C._distributed_c10d.BackendType, 'XCCL'):
        setattr(torch._C._distributed_c10d.BackendType, 'XCCL', None)
except Exception:
    # If any error occurs, pass silently so it doesn't impact execution
    pass



class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
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

class SuperResolutionCNN(nn.Module):
    def __init__(self):
        super(SuperResolutionCNN, self).__init__()
        # ENCODER
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(16),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(32),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            ResidualBlock(64),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=2, dilation=2),
            nn.ReLU(),
            ResidualBlock(256),
        )
        # DECODER (with skip connections)
        self.decoder5 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(32),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(16),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        d5 = self.decoder5(torch.cat([e5, e4], dim=1))
        d4 = self.decoder4(torch.cat([d5, e3], dim=1))
        d3 = self.decoder3(torch.cat([d4, e2], dim=1))
        d2 = self.decoder2(torch.cat([d3, e1], dim=1))
        d1 = self.decoder1(d2)
        return d1

# Cascaded model: two U-Nets in series.
class CascadedStarRemovalNetCombined(nn.Module):
    def __init__(self, stage1_path, stage2_path):
        super(CascadedStarRemovalNetCombined, self).__init__()
        # Load Stage 1 weights and remove any "stage1." prefix.
        self.stage1 = SuperResolutionCNN()
        map_fn = lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage
        state_dict1 = torch.load(stage1_path, map_location=map_fn)
        new_state_dict1 = {}
        for key, value in state_dict1.items():
            if key.startswith("stage1."):
                new_key = key[len("stage1."):]
            else:
                new_key = key
            new_state_dict1[new_key] = value
        self.stage1.load_state_dict(new_state_dict1)
        
        # ---------------------------------------------------------------------------
        # Stage 2 code commented out for now.
        """
        # Load Stage 2 weights (if you need it) and remove prefixes.
        self.stage2 = SuperResolutionCNN()
        state_dict2 = torch.load(stage2_path, map_location=map_fn)
        new_state_dict2 = {}
        for key, value in state_dict2.items():
            if key.startswith("stage2."):
                new_key = key[len("stage2."):]
            elif key.startswith("stage1."):
                new_key = key[len("stage1."):]
            else:
                new_key = key
            new_state_dict2[new_key] = value
        self.stage2.load_state_dict(new_state_dict2)
        """
        # ---------------------------------------------------------------------------
    
    def forward(self, x):
        coarse = self.stage1(x)
        # The Stage 2 refinement is currently disabled.
        # refined = self.stage2(coarse)
        return coarse


# Get the directory of the executable or the script location.
exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

def load_model(exe_dir, use_gpu=True):
    print(torch.__version__)
    
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Specify path for Stage 1 weights.
    stage1_path = os.path.join(exe_dir, 'darkstar_v1.pth')
    # Stage 2 weights path is commented out until needed.
    # stage2_path = os.path.join(exe_dir, 'darkstar_v2.pth')
    
    # Initialize the cascaded model.
    # Since Stage 2 is disabled, a dummy value (None) is passed for stage2_path.
    starremoval_model = CascadedStarRemovalNetCombined(stage1_path, None)
    starremoval_model.eval()
    starremoval_model.to(device)
    
    return {
        "starremoval_model": starremoval_model,
        "device": device
    }



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

# Function to stitch overlapping chunks back together with soft blending
def stitch_chunks_ignore_border(chunks, padded_shape, chunk_size, overlap, border_size=16):
    """
    Stitch chunks back together using soft blending. Instead of always removing a fixed border,
    this function checks if a chunk is at the edge of the padded image and removes only what’s available.
    
    Parameters:
      chunks: list of (chunk, i, j) tuples.
      padded_shape: shape of the padded image (height, width, channels)
      chunk_size: size used when splitting chunks.
      overlap: overlap used.
      border_size: the intended border size for interior chunks.
      
    Returns:
      The stitched image.
    """
    stitched_image = np.zeros(padded_shape, dtype=np.float32)
    weight_map = np.zeros(padded_shape, dtype=np.float32)
    blend_weights = generate_blend_weights(chunk_size, overlap)
    
    H, W, _ = padded_shape
    
    for chunk, i, j in chunks:
        actual_chunk_h, actual_chunk_w = chunk.shape[:2]
        
        # Determine how much border to remove adaptively:
        border_top = 0 if i == 0 else border_size
        border_left = 0 if j == 0 else border_size
        # For bottom/right, if the chunk goes to the padded edge, remove 0 from that side:
        border_bottom = 0 if (i + actual_chunk_h) >= H else border_size
        border_right = 0 if (j + actual_chunk_w) >= W else border_size
        
        # Determine region in the chunk to blend (the "inner chunk")
        inner_chunk = chunk[border_top:actual_chunk_h - border_bottom, border_left:actual_chunk_w - border_right]
        
        # Compute where this inner_chunk should be placed in the stitched image:
        region_i_start = i + border_top
        region_j_start = j + border_left
        region_i_end = region_i_start + inner_chunk.shape[0]
        region_j_end = region_j_start + inner_chunk.shape[1]
        
        # Adjust blend weights to match the inner chunk size (if smaller than the full blend_weights, use slicing)
        bw = blend_weights[:inner_chunk.shape[0], :inner_chunk.shape[1]]
        bw = bw[:, :, np.newaxis]
        
        stitched_image[region_i_start:region_i_end, region_j_start:region_j_end] += inner_chunk * bw
        weight_map[region_i_start:region_i_end, region_j_start:region_j_end] += bw

    # Normalize to blend overlapping areas smoothly.
    stitched_image = np.divide(stitched_image, weight_map, where=weight_map != 0)
    return stitched_image


class ProcessingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_dir, output_dir, use_gpu, star_removal_mode, show_extracted_stars):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.star_removal_mode = star_removal_mode
        self.show_extracted_stars = show_extracted_stars

    def run(self):
        def progress_callback(message):
            self.progress_signal.emit(message)

        process_images(
            self.input_dir,
            self.output_dir,
            use_gpu=self.use_gpu,
            star_removal_mode=self.star_removal_mode,
            show_extracted_stars=self.show_extracted_stars,
            progress_callback=progress_callback
        )
        self.finished_signal.emit()


class StarRemovalUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmic Clarity - Dark Star")
        self.setMinimumSize(400, 300)
        self.use_gpu = True
        self.star_removal_mode = "unscreen"
        self.show_extracted_stars = False

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.gpu_dropdown = QComboBox()
        self.gpu_dropdown.addItems(["Yes", "No"])
        layout.addWidget(QLabel("Use GPU Acceleration:"))
        layout.addWidget(self.gpu_dropdown)

        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems(["unscreen", "additive"])
        layout.addWidget(QLabel("Star Removal Mode:"))
        layout.addWidget(self.mode_dropdown)

        self.stars_checkbox = QCheckBox("Show Extracted Stars")
        layout.addWidget(self.stars_checkbox)

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)

        self.progress_label = QLabel("Ready.")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

    def start_processing(self):
        self.use_gpu = self.gpu_dropdown.currentText() == "Yes"
        self.star_removal_mode = self.mode_dropdown.currentText()
        self.show_extracted_stars = self.stars_checkbox.isChecked()

        # Call your actual processing here
        self.progress_label.setText("Processing started...")
        self.run_processing()

    def run_processing(self):
        input_dir = os.path.join(exe_dir, 'input')
        output_dir = os.path.join(exe_dir, 'output')

        self.thread = ProcessingThread(
            input_dir=input_dir,
            output_dir=output_dir,
            use_gpu=self.use_gpu,
            star_removal_mode=self.star_removal_mode,
            show_extracted_stars=self.show_extracted_stars
        )

        self.thread.progress_signal.connect(self.progress_label.setText)
        self.thread.finished_signal.connect(lambda: self.progress_label.setText("Done!"))
        self.thread.start()


# Function to create a stars-only image, ensuring both images have the same dimensions
def create_starless_and_stars_only_images(original, starless, mode):
    # If original is 2D and starless is 3D, stack original to make it 3D.
    if original.ndim == 2 and starless.ndim == 3:
        original = np.stack([original] * 3, axis=-1)
    
    # Ensure both images have the same dimensions.
    if original.shape != starless.shape:
        border_h = (original.shape[0] - starless.shape[0]) // 2
        border_w = (original.shape[1] - starless.shape[1]) // 2
        original_cropped = original[border_h:original.shape[0]-border_h, border_w:original.shape[1]-border_w]
    else:
        original_cropped = original

    if mode == 'additive':
        stars_only = original_cropped - starless
    elif mode == 'unscreen':
        stars_only = (original_cropped - starless) / (1 - starless)
    stars_only = np.clip(stars_only, 0, 1)
    return stars_only


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
        unstretched_image = ((median_color - 1) * original_median * image) / \
                            (median_color * (original_median + image - 1) - original_median * image)
    else:
        image_median = np.median(image)
        unstretched_image = ((image_median - 1) * original_median * image) / \
                            (image_median * (original_median + image - 1) - original_median * image)

    unstretched_image = np.clip(unstretched_image, 0, 1)
    unstretched_image += original_min
    unstretched_image = np.clip(unstretched_image, 0, 1)
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



# Main starremoval function for an image
def starremoval_image(image_path, starremoval_strength, device, model, border_size=16, progress_callback=None):
    try:
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension not in ['.png', '.tif', '.tiff', '.fit', '.fits', '.xisf']:
            print(f"Ignoring non-image file: {image_path}")
            return None, None, None, None, None, None

        original_header = None
        image = None
        bit_depth = "32-bit float"
        is_mono = False
        file_meta, image_meta = None, None
        original_image = None

        try:
            # Load the image based on its extension
            if file_extension in ['.tif', '.tiff']:
                image = tiff.imread(image_path)
                original_image = image.copy()
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
                else:
                    is_mono = False

            elif file_extension in ['.fits', '.fit']:
                # Load the FITS image
                with fits.open(image_path) as hdul:
                    image_data = hdul[0].data
                    original_header = hdul[0].header  # Capture the FITS header
                    image = image_data
                    original_image = image.copy()

                    # Ensure the image data uses the native byte order
                    if image_data.dtype.byteorder not in ('=', '|'):  # Check if byte order is non-native
                        image_data = image_data.astype(image_data.dtype.newbyteorder('='))  # Convert to native byte order


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

            # Check if file extension is '.xisf'
            elif file_extension == '.xisf':
                # Load XISF file
                xisf = XISF(image_path)
                image = xisf.read_image(0)  # Assuming the image data is in the first image block
                original_image = image.copy()
                original_header = xisf.get_images_metadata()[0]  # Load metadata for header reconstruction
                file_meta = xisf.get_file_metadata()  # For potential use if saving with the same meta
                image_meta = xisf.get_images_metadata()[0]

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


            else:  # Assume 8-bit PNG
                image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
                original_image = image.copy()
                bit_depth = "8-bit"
                print(f"Loaded {bit_depth} PNG image.")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None, None, None, None, None, None


        # Add a border around the image with the median value
        # Add a border around the image with the median value
        image_with_border = add_border(image, border_size=5)
        original_median = np.median(image_with_border)
        if original_median < 0.125:
            stretched_image, original_min, original_median = stretch_image(image_with_border)
        else:
            stretched_image = image_with_border
            original_min = None

        def process_chunks(chunks, processed_image_shape):
            starless_chunks = []
            for idx, (chunk, i, j) in enumerate(chunks):
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
                if chunk_tensor.dim() == 3:
                    chunk_tensor = chunk_tensor.unsqueeze(0)
                chunk_tensor = chunk_tensor.permute(0, 3, 1, 2).to(device)
                with torch.no_grad():
                    starless_chunk = model(chunk_tensor).squeeze().cpu().numpy().transpose(1, 2, 0)
                starless_chunks.append((starless_chunk, i, j))
                if progress_callback:
                    progress_callback(f"Progress: {(idx + 1) / len(chunks) * 100:.2f}% ({idx + 1}/{len(chunks)} chunks processed)")
            return stitch_chunks_ignore_border(starless_chunks, processed_image_shape, chunk_size, overlap, border_size=5)

        chunk_size = 256
        overlap = 64

        if stretched_image.ndim == 2:
            processed_image = np.stack([stretched_image] * 3, axis=-1)
            chunks = split_image_into_chunks_with_overlap(processed_image, chunk_size, overlap)
            starless_image = process_chunks(chunks, processed_image.shape)
            if original_min is not None:
                starless_image = unstretch_image(starless_image, original_median, original_min)
            final_starless = starless_image[5:5 + original_image.shape[0], 5:5 + original_image.shape[1]]

        elif stretched_image.ndim == 3 and stretched_image.shape[-1] == 1:
            processed_image = np.concatenate([stretched_image] * 3, axis=-1)
            chunks = split_image_into_chunks_with_overlap(processed_image, chunk_size, overlap)
            starless_image = process_chunks(chunks, processed_image.shape)
            if original_min is not None:
                starless_image = unstretch_image(starless_image, original_median, original_min)
            final_starless = starless_image[5:5 + original_image.shape[0], 5:5 + original_image.shape[1]]

        elif stretched_image.ndim == 3 and stretched_image.shape[-1] == 3:
            if (np.allclose(stretched_image[..., 0], stretched_image[..., 1]) and
                np.allclose(stretched_image[..., 0], stretched_image[..., 2])):
                print("Detected 3-channel mono image (channels are identical). Processing only the first channel.")
                channel_data = stretched_image[..., 0]
                processed_image = np.stack([channel_data] * 3, axis=-1)
                chunks = split_image_into_chunks_with_overlap(processed_image, chunk_size, overlap)
                starless_image = process_chunks(chunks, processed_image.shape)
                if original_min is not None:
                    starless_image = unstretch_image(starless_image, original_median, original_min)
                final_starless = starless_image[5:5 + original_image.shape[0], 5:5 + original_image.shape[1]]
            else:
                print("Detected true RGB image with 3 channels. Processing each channel individually.")
                processed_channels = []
                for ch in range(3):
                    if progress_callback:
                        progress_callback(f"Processing channel {ch + 1}/3")
                    channel_data = stretched_image[..., ch]
                    channel_input = np.stack([channel_data] * 3, axis=-1)
                    chunks = split_image_into_chunks_with_overlap(channel_input, chunk_size, overlap)
                    channel_starless = process_chunks(chunks, channel_input.shape)
                    if original_min is not None:
                        channel_starless = unstretch_image(channel_starless, original_median, original_min)
                    channel_starless_final = channel_starless[5:5 + original_image.shape[0], 5:5 + original_image.shape[1]]
                    processed_channels.append(channel_starless_final)
                final_starless = np.stack(processed_channels, axis=-1)
        else:
            raise ValueError("Unsupported image format.")

        return final_starless, original_header, bit_depth, file_extension, is_mono, file_meta, original_image

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None, None, None, None, None
    
# Main process for denoising images
def process_images(input_dir, output_dir, starremoval_strength=None, use_gpu=True, star_removal_mode='additive', show_extracted_stars=False,
                   progress_callback=None):

    print((r"""
 *#        ___     __      ___       __                              #
 *#       / __/___/ /__   / _ | ___ / /________                      #
 *#      _\ \/ -_) _ _   / __ |(_-</ __/ __/ _ \                     #
 *#     /___/\__/\//_/  /_/ |_/___/\__/__/ \___/                     #
 *#                                                                  #
 *#              Cosmic Clarity - Dark Star V1.0                     # 
 *#                                                                  #
 *#                         SetiAstro                                #
 *#                    Copyright © 2024                              #
 *#                                                                  #
        """))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models = load_model(exe_dir, use_gpu)

    all_images = [img for img in os.listdir(input_dir) if img.lower().endswith(('.tif', '.tiff', '.fits', '.fit', '.xisf', '.png'))]

    for idx, image_name in enumerate(all_images):
        image_path = os.path.join(input_dir, image_name)

        if progress_callback:
            progress_callback(f"Processing image {idx + 1}/{len(all_images)}: {image_name}")

        starless_image, original_header, bit_depth, file_extension, is_mono, file_meta, original_image = starremoval_image(
            image_path, 1.0, models['device'], models["starremoval_model"], progress_callback=progress_callback
        )

        if starless_image is not None:
            output_image_name = os.path.splitext(image_name)[0] + "_starless"
            output_image_path = os.path.join(output_dir, output_image_name + file_extension)
            actual_bit_depth = bit_depth

# Save as FITS file with header information if the original was FITS
            if file_extension in ['.fits', '.fit']:
                if original_header is not None:
                    if is_mono:  # Grayscale FITS
                        # Convert the grayscale image back to its original 2D format
                        if bit_depth == "16-bit":
                            starless_image_fits = (starless_image[:, :, 0] * 65535).astype(np.uint16)
                        elif bit_depth == "32-bit unsigned":
                            bzero = original_header.get('BZERO', 0)
                            bscale = original_header.get('BSCALE', 1)
                            starless_image_fits = (starless_image[:, :, 0].astype(np.float32) * bscale + bzero).astype(np.uint32)
                            original_header['BITPIX'] = 32
                        else:  # 32-bit float
                            starless_image_fits = starless_image[:, :, 0].astype(np.float32)
                        
                        # Update header for a 2D (grayscale) image
                        original_header['NAXIS'] = 2
                        original_header['NAXIS1'] = starless_image.shape[1]  # Width
                        original_header['NAXIS2'] = starless_image.shape[0]  # Height
                        if 'NAXIS3' in original_header:
                            del original_header['NAXIS3']  # Remove if present

                        hdu = fits.PrimaryHDU(starless_image_fits, header=original_header)
                    
                    else:  # RGB FITS
                        # Transpose RGB image to FITS-compatible format (channels, height, width)
                        starless_image_transposed = np.transpose(starless_image, (2, 0, 1))

                        if bit_depth == "16-bit":
                            starless_image_fits = (starless_image_transposed * 65535).astype(np.uint16)
                        elif bit_depth == "32-bit unsigned":
                            starless_image_fits = starless_image_transposed.astype(np.float32)
                            original_header['BITPIX'] = -32
                            actual_bit_depth = "32-bit unsigned"
                        else:
                            starless_image_fits = starless_image_transposed.astype(np.float32)
                            actual_bit_depth = "32-bit float"

                        # Update header for a 3D (RGB) image
                        original_header['NAXIS'] = 3
                        original_header['NAXIS1'] = starless_image_transposed.shape[2]  # Width
                        original_header['NAXIS2'] = starless_image_transposed.shape[1]  # Height
                        original_header['NAXIS3'] = starless_image_transposed.shape[0]  # Channels
                        
                        hdu = fits.PrimaryHDU(starless_image_fits, header=original_header)

                    # Write the FITS file
                    hdu.writeto(output_image_path, overwrite=True)
                    print(f"Saved {actual_bit_depth} starless image to: {output_image_path}")


            # Save as TIFF based on the original bit depth if the original was TIFF
            elif file_extension in ['.tif', '.tiff']:
                if bit_depth == "16-bit":
                    actual_bit_depth = "16-bit"
                    if is_mono is True:  # Grayscale
                        tiff.imwrite(output_image_path, (starless_image[:, :, 0] * 65535).astype(np.uint16))
                    else:  # RGB
                        tiff.imwrite(output_image_path, (starless_image * 65535).astype(np.uint16))
                elif bit_depth == "32-bit unsigned":
                    actual_bit_depth = "32-bit unsigned"
                    if is_mono is True:  # Grayscale
                        tiff.imwrite(output_image_path, (starless_image[:, :, 0] * 4294967295).astype(np.uint32))
                    else:  # RGB
                        tiff.imwrite(output_image_path, (starless_image * 4294967295).astype(np.uint32))           
                else:
                    actual_bit_depth = "32-bit float"
                    if is_mono is True:  # Grayscale
                        tiff.imwrite(output_image_path, starless_image[:, :, 0].astype(np.float32))
                    else:  # RGB
                        tiff.imwrite(output_image_path, starless_image.astype(np.float32))

                print(f"Saved {actual_bit_depth} starless image to: {output_image_path}")

            elif file_extension == '.xisf':
                try:
                    # If the image is mono, we replicate the single channel into RGB format for consistency
                    if is_mono:
                        rgb_image = np.stack([starless_image[:, :, 0]] * 3, axis=-1).astype(np.float32)
                    else:
                        rgb_image = starless_image.astype(np.float32)
                    
                    # Save the starless image in XISF format using the XISF write method, including metadata if available
                    XISF.write(output_image_path, rgb_image, xisf_metadata=file_meta)  # Replace `original_header` with the appropriate metadata if it's named differently

                    print(f"Saved {bit_depth} XISF starless image to: {output_image_path}")
                    
                except Exception as e:
                    print(f"Error saving XISF file: {e}")



            # Save as 8-bit PNG if the original was PNG
            else:
                output_image_path = os.path.join(output_dir, output_image_name + ".png")
                starless_image_8bit = (starless_image * 255).astype(np.uint8)
                starless_image_pil = Image.fromarray(starless_image_8bit)
                actual_bit_depth = "8-bit"
                starless_image_pil.save(output_image_path)
                print(f"Saved {actual_bit_depth} starless image to: {output_image_path}")

            # Generate and save the stars-only image with matching file type and bit depth if `show_extracted_stars` is enabled
            if show_extracted_stars:
                print("Generating stars-only image...")
                stars_only_image = create_starless_and_stars_only_images(
                    original=original_image,
                    starless=starless_image,
                    mode=star_removal_mode
                )
                output_stars_only_name = os.path.splitext(image_name)[0] + "_stars_only"
                stars_only_path = os.path.join(output_dir, output_stars_only_name + file_extension)

                if file_extension in ['.tif', '.tiff']:
                    if bit_depth == "16-bit":
                        tiff.imwrite(stars_only_path, (stars_only_image * 65535).astype(np.uint16))
                    elif bit_depth == "32-bit unsigned":
                        tiff.imwrite(stars_only_path, (stars_only_image * 4294967295).astype(np.uint32))
                    else:
                        tiff.imwrite(stars_only_path, stars_only_image.astype(np.float32))

                elif file_extension in ['.fits', '.fit']:
                    if is_mono:
                        if bit_depth == "16-bit":
                            stars_only_image_fits = (stars_only_image[:, :, 0] * 65535).astype(np.uint16)
                        elif bit_depth == "32-bit unsigned":
                            stars_only_image_fits = stars_only_image[:, :, 0].astype(np.float32)
                        else:
                            stars_only_image_fits = stars_only_image[:, :, 0].astype(np.float32)
                        hdu = fits.PrimaryHDU(stars_only_image_fits, header=original_header)
                    else:
                        stars_only_image_fits = np.transpose(stars_only_image, (2, 0, 1)).astype(np.float32)
                        hdu = fits.PrimaryHDU(stars_only_image_fits, header=original_header)
                    hdu.writeto(stars_only_path, overwrite=True)
                    print(f"Saved {bit_depth} stars-only image to: {stars_only_path}")

                elif file_extension == '.xisf':
                    if is_mono:
                        stars_only_image_rgb = np.stack([stars_only_image[:, :, 0]] * 3, axis=-1).astype(np.float32)
                    else:
                        stars_only_image_rgb = stars_only_image.astype(np.float32)
                    XISF.write(stars_only_path, stars_only_image_rgb, xisf_metadata=file_meta)
                    print(f"Saved {bit_depth} XISF stars-only image to: {stars_only_path}")

                else:  # PNG
                    stars_only_image_8bit = (stars_only_image * 255).astype(np.uint8)
                    stars_only_image_pil = Image.fromarray(stars_only_image_8bit)
                    stars_only_image_pil.save(stars_only_path)
                    print(f"Saved 8-bit stars-only image to: {stars_only_path}")

            if progress_callback:
                progress_callback(f"Saved starless image: {output_image_path}")

    if progress_callback:
        progress_callback("All images processed.")

# Define input and output directories
input_dir = os.path.join(exe_dir, 'input')
output_dir = os.path.join(exe_dir, 'output')

if not os.path.exists(input_dir):
    os.makedirs(input_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


if __name__ == "__main__":
    headless = len(sys.argv) > 1  # True if user passed any arguments (even --disable_gpu)

    if headless:
        # CLI mode: run without GUI
        parser = argparse.ArgumentParser(description="Cosmic Clarity - Dark Star Tool")
        parser.add_argument('--disable_gpu', action='store_true', help="Disable GPU acceleration and use CPU only")
        parser.add_argument('--star_removal_mode', type=str, choices=['additive', 'unscreen'], default='unscreen', help="Star Removal Mode")
        parser.add_argument('--show_extracted_stars', action='store_true', help="Output an additional image with only the extracted stars")
        args = parser.parse_args()

        use_gpu = not args.disable_gpu
        process_images(
            input_dir, output_dir,
            starremoval_strength=1.0,
            use_gpu=use_gpu,
            star_removal_mode=args.star_removal_mode,
            show_extracted_stars=args.show_extracted_stars
        )
        sys.exit(0)
    else:
        # GUI mode: launch interactive UI
        app = QApplication(sys.argv)
        window = StarRemovalUI()
        window.show()
        sys.exit(app.exec())

