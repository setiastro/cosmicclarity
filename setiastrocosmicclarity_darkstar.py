import warnings
from xisf import XISF
import os
import sys
import torch
import numpy as np
import lz4.block
import zstandard
import base64
import ast
import platform
import torch.nn as nn
import tifffile as tiff
from astropy.io import fits
import torch.nn.functional as F
from PIL import Image
import cv2

import argparse  # For command-line argument parsing
import time  # For simulating progress updates

import onnxruntime as ort
sys.stdout.reconfigure(encoding='utf-8')

#torch.cuda.is_available = lambda: False

# Suppress model loading warnings
warnings.filterwarnings("ignore")
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton,
    QComboBox, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont

class RefinementCNN(nn.Module):
    def __init__(self, channels: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=8, dilation=8), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=8, dilation=8), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(channels, 3,      3, padding=1, dilation=1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + x)

class DarkStarCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ── ENCODER ───────────────────────────────────────
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
        )

        # ── DECODER ───────────────────────────────────────
        self.decoder5 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16),
        )

        # ── FINAL OUTPUT with EXTRA RESIDUALS ──────────────────────
        # Instead of going straight 16→3, we lift back to 16, run a couple of
        # ResidualBlocks, then project down to 3.
        self.decoder1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(16),
            ResidualBlock(16),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # encode
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # decode
        d5 = self.decoder5(torch.cat([e5, e4], dim=1))
        d4 = self.decoder4(torch.cat([d5, e3], dim=1))
        d3 = self.decoder3(torch.cat([d4, e2], dim=1))
        d2 = self.decoder2(torch.cat([d3, e1], dim=1))
        out = self.decoder1(d2)
        return out

# Cascaded model: two U-Nets in series.
class CascadedStarRemovalNetCombined(nn.Module):
    def __init__(self, stage1_path, stage2_path=None):
        super().__init__()

        # --- Stage 1 (coarse) ---
        self.stage1 = DarkStarCNN()
        ckpt1 = torch.load(stage1_path, map_location='cpu')
        # strip any "stage1." prefix if present
        sd1 = {k[len("stage1."):]: v for k, v in ckpt1.items() if k.startswith("stage1.")}
        self.stage1.load_state_dict(sd1)

        # --- Stage 2 (refinement) ---
        self.stage2 = RefinementCNN()
        if stage2_path and os.path.exists(stage2_path):
            ckpt2 = torch.load(stage2_path, map_location='cpu')
            # if it’s a dict with extra info, pull out just the model_state
            if isinstance(ckpt2, dict) and 'model_state' in ckpt2:
                sd2 = ckpt2['model_state']
            else:
                sd2 = ckpt2
            self.stage2.load_state_dict(sd2)

        # --- Freeze only stage1 ---
        for p in self.stage1.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            coarse = self.stage1(x)
            #coarse = self.stage2(x)
        #return self.stage2(coarse)
        return coarse

# Get the directory of the executable or the script location.
exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

def load_model(exe_dir, use_gpu=True, color=False):
    print(torch.__version__)

    is_windows = platform.system() == "Windows"
    ckpt_name = 'darkstar_v2.1c.pth' if color else 'darkstar_v2.1.pth'
    stage1_pth_path = os.path.join(exe_dir, ckpt_name)
    stage1_onnx_path = os.path.splitext(stage1_pth_path)[0] + '.onnx'
    stage2_pth = None

    # Priority: Windows + CUDA → ONNX → CPU
    if is_windows:
        if use_gpu and torch.cuda.is_available():
            print(f"Windows + CUDA detected. Using PyTorch with GPU for {ckpt_name}")
            device = torch.device("cuda")
            model = CascadedStarRemovalNetCombined(stage1_pth_path, stage2_pth)
            model.eval()
            model.to(device)
            return {
                "starremoval_model": model,
                "device": device,
                "is_onnx": False
            }
        elif os.path.exists(stage1_onnx_path):
            print(f"Windows detected. Using ONNX Runtime for {stage1_onnx_path}.")
            providers = ['DmlExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
            session = ort.InferenceSession(stage1_onnx_path, providers=providers)
            return {
                "starremoval_model": session,
                "device": 'onnx',
                "is_onnx": True
            }
        else:
            print("Windows fallback. Using PyTorch on CPU.")
            device = torch.device("cpu")
            model = CascadedStarRemovalNetCombined(stage1_pth_path, stage2_pth)
            model.eval()
            model.to(device)
            return {
                "starremoval_model": model,
                "device": device,
                "is_onnx": False
            }

    # Non-Windows: default PyTorch behavior
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Non-Windows system using device: {device}")
    model = CascadedStarRemovalNetCombined(stage1_pth_path, None)
    model.eval()
    model.to(device)
    return {
        "starremoval_model": model,
        "device": device,
        "is_onnx": False
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
        # Default values; these will be overwritten from the UI if needed.
        self.chunk_size = 512
        self.overlap = int(round(0.125 * self.chunk_size))

    def run(self):
        def progress_callback(message):
            self.progress_signal.emit(message)
        process_images(
            self.input_dir,
            self.output_dir,
            use_gpu=self.use_gpu,
            star_removal_mode=self.star_removal_mode,
            show_extracted_stars=self.show_extracted_stars,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            progress_callback=progress_callback
        )
        self.finished_signal.emit()


class StarRemovalUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmic Clarity - Dark Star V2.1c")
        self.setMinimumSize(400, 300)
        self.use_gpu = True
        self.star_removal_mode = "unscreen"
        self.show_extracted_stars = False

        self.chunk_size_default = 512  # default chunk size
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # GPU dropdown
        self.gpu_dropdown = QComboBox()
        self.gpu_dropdown.addItems(["Yes", "No"])
        layout.addWidget(QLabel("Use GPU Acceleration:"))
        layout.addWidget(self.gpu_dropdown)

        # Mode dropdown
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems(["unscreen", "additive"])
        layout.addWidget(QLabel("Star Removal Mode:"))
        layout.addWidget(self.mode_dropdown)

        # Checkbox to show extracted stars
        self.stars_checkbox = QCheckBox("Show Extracted Stars")
        layout.addWidget(self.stars_checkbox)

        # Add chunk size control
        self.chunk_size_spinbox = QSpinBox()
        self.chunk_size_spinbox.setMinimum(128)
        self.chunk_size_spinbox.setMaximum(4096)
        self.chunk_size_spinbox.setSingleStep(64)
        self.chunk_size_spinbox.setValue(self.chunk_size_default)
        layout.addWidget(QLabel("Chunk Size (pixels):"))
        layout.addWidget(self.chunk_size_spinbox)

        # Add a label to display the computed overlap.
        overlap = int(round(0.2 * self.chunk_size_spinbox.value()))
        self.overlap_label = QLabel(f"Overlap: {overlap} pixels")
        layout.addWidget(self.overlap_label)

        # Update overlap when the chunk size changes.
        self.chunk_size_spinbox.valueChanged.connect(self.update_overlap_label)

        # Start Processing Button
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)

        self.progress_label = QLabel("Ready.")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

    def update_overlap_label(self):
        chunk_size = self.chunk_size_spinbox.value()
        overlap = int(round(0.125 * chunk_size))
        self.overlap_label.setText(f"Overlap: {overlap} pixels")

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

        # Extract chunk_size from UI and compute overlap
        chunk_size = self.chunk_size_spinbox.value()
        overlap = int(round(0.125 * chunk_size))

        self.thread = ProcessingThread(
            input_dir=input_dir,
            output_dir=output_dir,
            use_gpu=self.use_gpu,
            star_removal_mode=self.star_removal_mode,
            show_extracted_stars=self.show_extracted_stars,
            # You can add additional parameters here by modifying your ProcessingThread to pass them.
        )

        # Pass chunk_size and overlap via a lambda or by modifying your process_images signature.
        self.thread.progress_signal.connect(self.progress_label.setText)

        # Here we modify the ProcessingThread (or adjust process_images) so that these parameters are passed.
        # For example, if you update ProcessingThread.run() to pass these values:
        self.thread.chunk_size = chunk_size
        self.thread.overlap = overlap

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
def stretch_image(image, target_median: float = 0.25):
    """
    Apply the same un-linked stretch that SetiAstroSuite uses.

    Returns
    -------
    stretched_image : float32  in [0,1]
    original_mins   : (3,) or scalar – per-channel black points
    original_medians: (3,) or scalar – *median after rescale*, the value
                      needed for perfect unstretch.
    """
    img = image.astype(np.float32, copy=False)

    if img.ndim == 2:                    # mono
        orig_min = img.min()
        rescaled = (img - orig_min) / (1.0 - orig_min + 1e-12)

        median_rescaled = np.median(rescaled)
        numer  = (median_rescaled - 1.0) * target_median * rescaled
        denom  = (median_rescaled *
                  (target_median + rescaled - 1.0) -
                  target_median * rescaled)
        denom[ np.abs(denom) < 1e-12 ] = 1e-12
        stretched = numer / denom

        stretched = np.clip(stretched, 0.0, 1.0).astype(np.float32)
        return stretched, np.float32(orig_min), np.float32(median_rescaled)

    elif img.ndim == 3 and img.shape[2] == 3:   # RGB
        H, W, C = img.shape
        orig_min = img.reshape(-1, 3).min(axis=0)            # (3,)

        # broadcast to H×W×3
        rescaled = (img - orig_min.reshape(1, 1, 3)) / \
                   (1.0 - orig_min.reshape(1, 1, 3) + 1e-12)

        medians_rescaled = np.median(rescaled, axis=(0, 1))  # (3,)
        # broadcast for formula
        med_b = medians_rescaled.reshape(1, 1, 3)

        numer = (med_b - 1.0) * target_median * rescaled
        denom = med_b * (target_median + rescaled - 1.0) - target_median * rescaled
        denom[np.abs(denom) < 1e-12] = 1e-12

        stretched = numer / denom
        stretched = np.clip(stretched, 0.0, 1.0).astype(np.float32)
        return stretched, orig_min.astype(np.float32), medians_rescaled.astype(np.float32)

    else:
        raise ValueError("stretch_image expects mono or RGB image.")


def unstretch_image(image, stretch_original_medians, stretch_original_mins):
    """
    Undo the un-linked stretch (mono or RGB).  Arguments must be the
    *same* arrays returned by stretch_image.
    """
    img = image.astype(np.float32, copy=False)

    if img.ndim == 2:                              # mono
        cmed_stretched = np.median(img)
        orig_med = float(stretch_original_medians)
        orig_min = float(stretch_original_mins)

        numer = (cmed_stretched - 1.0) * orig_med * img
        denom = cmed_stretched * (orig_med + img - 1.0) - orig_med * img
        denom[np.abs(denom) < 1e-12] = 1e-12

        out = numer / denom
        out += orig_min
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    elif img.ndim == 3 and img.shape[2] == 3:      # RGB
        H, W, C = img.shape
        out = np.empty_like(img, dtype=np.float32)

        for c in range(C):
            cmed_stretched = np.median(img[..., c])
            orig_med = stretch_original_medians[c]
            orig_min = stretch_original_mins[c]

            numer = (cmed_stretched - 1.0) * orig_med * img[..., c]
            denom = cmed_stretched * (orig_med + img[..., c] - 1.0) \
                    - orig_med * img[..., c]
            mask = np.abs(denom) < 1e-12
            denom[mask] = 1e-12

            out[..., c] = numer / denom + orig_min

        return np.clip(out, 0.0, 1.0).astype(np.float32)

    else:
        raise ValueError("unstretch_image expects mono or RGB image.")

# Function to add a border of median value around the image
def add_border(image: np.ndarray, border_size: int = 5) -> np.ndarray:
    """
    Pad with each channel’s own median so the stretch stays un-linked.
    Works for mono or RGB.
    """
    if image.ndim == 2:                           # mono
        med = np.median(image)
        return np.pad(
            image,
            ((border_size, border_size), (border_size, border_size)),
            mode="constant",
            constant_values=med
        )

    elif image.ndim == 3 and image.shape[2] == 3: # RGB
        # channel-wise medians  → shape (3,)
        meds = np.median(image, axis=(0, 1)).astype(image.dtype)

        # pad each channel separately
        padded_channels = []
        for c in range(3):
            ch = np.pad(
                image[..., c],
                ((border_size, border_size), (border_size, border_size)),
                mode="constant",
                constant_values=float(meds[c])     # scalar for this channel
            )
            padded_channels.append(ch)

        return np.stack(padded_channels, axis=-1)

    else:
        raise ValueError("add_border expects mono or RGB image.")

# Function to remove the border added around the image
def remove_border(image, border_size=5):
    if len(image.shape) == 2:
        return image[border_size:-border_size, border_size:-border_size]
    else:
        return image[border_size:-border_size, border_size:-border_size, :]



# Main starremoval function for an image
def starremoval_image(image_path, starremoval_strength,
                      models_mono, models_color,
                      border_size=16, chunk_size=512, overlap=None,
                      progress_callback=None):

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
                # 1) Read the FITS
                with fits.open(image_path, memmap=False) as hdul:
                    data = hdul[0].data
                    original_header = hdul[0].header.copy()
                # 2) Make sure byteorder is native
                if data.dtype.byteorder not in ('=', '|'):
                    data = data.astype(data.dtype.newbyteorder('='))

                # 3) Figure out bit depth for later
                if data.dtype == np.uint16:
                    bit_depth = "16-bit"
                elif data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                elif data.dtype == np.float32:
                    bit_depth = "32-bit float"
                else:
                    raise ValueError(f"Unsupported FITS dtype: {data.dtype}")

                # 4) Convert to float32 [0..1], handling BSCALE/BZERO on unsigned and floats
                def apply_bscale(arr):
                    bzero = original_header.get('BZERO', 0)
                    bscale = original_header.get('BSCALE', 1)
                    arr = arr.astype(np.float32) * bscale + bzero
                    # normalize full dynamic range
                    mn, mx = arr.min(), arr.max()
                    return (arr - mn) / (mx - mn) if mx>mn else arr

                if data.ndim == 2:
                    # grayscale → stack into 3 channels
                    mono = data.astype(np.float32)
                    if bit_depth == "16-bit":
                        mono /= 65535.0
                    elif bit_depth == "32-bit unsigned":
                        mono = apply_bscale(mono)
                    # (32-bit float stays as-is)
                    image = np.stack([mono]*3, axis=-1)
                    is_mono = True

                elif data.ndim == 3 and data.shape[-1] == 3:
                    # already H×W×3
                    img = data.astype(np.float32)
                    if bit_depth == "16-bit":
                        img /= 65535.0
                    elif bit_depth == "32-bit unsigned":
                        img = apply_bscale(img)
                    image = img
                    is_mono = False

                elif data.ndim == 3 and data.shape[0] == 3:
                    # 3×H×W → transpose to H×W×3
                    img = np.transpose(data, (1,2,0)).astype(np.float32)
                    if bit_depth == "16-bit":
                        img /= 65535.0
                    elif bit_depth == "32-bit unsigned":
                        img = apply_bscale(img)
                    image = img
                    is_mono = False

                else:
                    raise ValueError(f"Unsupported FITS shape: {data.shape}")

                original_image = image.copy()
                print(f"Loaded FITS image: {bit_depth}, shape {image.shape}, mono={is_mono}")

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
        # ─── stretch & pad ─────────────────────────────────────────────────
        original_median = np.median(image)
        if original_median < 0.125:
            stretched_core, original_min, original_median = stretch_image(image)
        else:
            stretched_core = image
            original_min = None
        stretched_image = add_border(stretched_core, border_size=5)

        # ─── pick mono vs color model ───────────────────────────────────────
        # true RGB if 3-channel and not all channels identical
        is_true_rgb = (
            stretched_image.ndim == 3 and stretched_image.shape[2] == 3 and
            not (np.allclose(stretched_image[...,0], stretched_image[...,1]) and
                 np.allclose(stretched_image[...,0], stretched_image[...,2]))
        )
        chosen = models_color if is_true_rgb else models_mono
        device            = chosen["device"]
        is_onnx           = chosen["is_onnx"]
        starremoval_model = chosen["starremoval_model"]
        session           = starremoval_model if is_onnx else None

        # ─── chunk processor ────────────────────────────────────────────────
        def process_chunks(chunks, processed_shape):
            starless_chunks = []
            for idx, (chunk, i, j) in enumerate(chunks):

                # NO per-chunk min/max normalization – just make sure we’re float32
                work = chunk.astype(np.float32, copy=False)
                h0, w0 = work.shape[:2]

                # pad for ONNX if needed (model expects fixed chunk_size×chunk_size)
                if is_onnx and (h0 != chunk_size or w0 != chunk_size):
                    pad = np.zeros((chunk_size, chunk_size, work.shape[2]), dtype=work.dtype)
                    pad[:h0, :w0] = work
                    work = pad

                # [H,W,C] → [1,C,H,W]
                tensor = torch.from_numpy(work)
                tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)

                if is_onnx:
                    inp_name = session.get_inputs()[0].name
                    out = session.run(None, {inp_name: tensor.cpu().numpy()})[0][0]
                    res = out.transpose(1, 2, 0)
                    if (h0, w0) != (chunk_size, chunk_size):
                        res = res[:h0, :w0, :]
                else:
                    tensor = tensor.to(device)
                    with torch.no_grad():
                        out_t = starremoval_model(tensor)
                    res = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                # model already outputs [0,1] in the same scale as input
                starless = res  # no re-scaling per tile
                starless_chunks.append((starless, i, j))

                if progress_callback:
                    progress_callback(
                        f"Progress: {(idx + 1) / len(chunks) * 100:.2f}% "
                        f"({idx + 1}/{len(chunks)} chunks)"
                    )

            return stitch_chunks_ignore_border(
                starless_chunks, processed_shape,
                chunk_size, overlap, border_size=5
            )


        # ─── dispatch by image shape ────────────────────────────────────────
        if stretched_image.ndim == 2:
            # mono → stack→one pass
            proc = np.stack([stretched_image]*3, axis=-1)
            chunks = split_image_into_chunks_with_overlap(proc, chunk_size, overlap)
            starless = process_chunks(chunks, proc.shape)
            if original_min is not None:
                starless = unstretch_image(starless, original_median, original_min)
            final_starless = starless[5:5+original_image.shape[0], 5:5+original_image.shape[1]]

        elif stretched_image.ndim == 3 and stretched_image.shape[2] == 1:
            # single‐channel in 3D → same as mono
            proc = np.concatenate([stretched_image]*3, axis=-1)
            chunks = split_image_into_chunks_with_overlap(proc, chunk_size, overlap)
            starless = process_chunks(chunks, proc.shape)
            if original_min is not None:
                starless = unstretch_image(starless, original_median, original_min)
            final_starless = starless[5:5+original_image.shape[0], 5:5+original_image.shape[1]]

        else:
            # stretched_image.ndim==3 && shape[2]==3
            if not is_true_rgb:
                # 3-chan mono → treat like mono
                ch = stretched_image[...,0]
                proc = np.stack([ch]*3, axis=-1)
                chunks = split_image_into_chunks_with_overlap(proc, chunk_size, overlap)
                starless = process_chunks(chunks, proc.shape)
                if original_min is not None:
                    starless = unstretch_image(starless, original_median, original_min)
                final_starless = starless[5:5+original_image.shape[0], 5:5+original_image.shape[1]]
            else:
                # true RGB → full‐color one pass
                proc = stretched_image
                chunks = split_image_into_chunks_with_overlap(proc, chunk_size, overlap)
                starless = process_chunks(chunks, proc.shape)
                if original_min is not None:
                    starless = unstretch_image(starless, original_median, original_min)
                final_starless = starless[5:5+original_image.shape[0], 5:5+original_image.shape[1]]

        return final_starless, original_header, bit_depth, file_extension, is_mono, file_meta, original_image

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None, None, None, None, None
    
# Main process for denoising images
def process_images(input_dir, output_dir, starremoval_strength=None,
                   use_gpu=True, star_removal_mode='additive', show_extracted_stars=False,
                   chunk_size=512, overlap=None, progress_callback=None):

    print((r"""
 *#        ___     __      ___       __                              #
 *#       / __/___/ /__   / _ | ___ / /________                      #
 *#      _\ \/ -_) _ _   / __ |(_-</ __/ __/ _ \                     #
 *#     /___/\__/\//_/  /_/ |_/___/\__/__/ \___/                     #
 *#                                                                  #
 *#              Cosmic Clarity - Dark Star V2.2c                    # 
 *#                                                                  #
 *#                         SetiAstro                                #
 *#                    Copyright © 2025                              #
 *#                                                                  #
        """))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ─── load both mono and color versions ───────────────────────────
    models_mono  = load_model(exe_dir, use_gpu, color=False)
    models_color = load_model(exe_dir, use_gpu, color=True)

    # ─── if either model is ONNX, pick it to force chunk_size & overlap ───
    onnx_info = None
    if models_mono["is_onnx"]:
        onnx_info = models_mono
    elif models_color["is_onnx"]:
        onnx_info = models_color

    if onnx_info:
        session = onnx_info["starremoval_model"]
        inp = session.get_inputs()[0]
        # inp.shape == [1, 3, H, W]
        chunk_size = inp.shape[2]
        overlap    = int(round(0.125 * chunk_size))
        print(f"[ONNX] forcing chunk_size={chunk_size}, overlap={overlap}")

    all_images = [
        img for img in os.listdir(input_dir)
        if img.lower().endswith(('.tif', '.tiff', '.fits', '.fit', '.xisf', '.png'))
    ]

    for idx, image_name in enumerate(all_images):
        image_path = os.path.join(input_dir, image_name)

        if progress_callback:
            progress_callback(f"Processing image {idx + 1}/{len(all_images)}: {image_name}")

        # ─── dispatch to starremoval_image with both model infos ───
        starless_image, original_header, bit_depth, file_extension, \
        is_mono, file_meta, original_image = starremoval_image(
            image_path,
            starremoval_strength=1.0,
            models_mono=models_mono,
            models_color=models_color,
            border_size=16,
            chunk_size=chunk_size,
            overlap=overlap,
            progress_callback=progress_callback
        )

        if starless_image is not None:
            output_image_name = os.path.splitext(image_name)[0] + "_starless"
            output_image_path = os.path.join(output_dir, output_image_name + file_extension)
            actual_bit_depth = bit_depth

            # Save as FITS file with header information if the original was FITS
            # Save as FITS file with header information if the original was FITS
            if file_extension in ['.fits', '.fit'] and original_header is not None:
                # 1) Prepare the 2D or 3D data array
                if is_mono:
                    # Grayscale → 2D
                    if bit_depth == "16-bit":
                        data = (starless_image[:, :, 0] * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        bzero  = original_header.get('BZERO', 0)
                        bscale = original_header.get('BSCALE', 1)
                        data = (starless_image[:, :, 0].astype(np.float32) * bscale + bzero).astype(np.uint32)
                    else:  # 32-bit float
                        data = starless_image[:, :, 0].astype(np.float32)
                else:
                    # RGB → (3, H, W)
                    arr = np.transpose(starless_image, (2, 0, 1))
                    if bit_depth == "16-bit":
                        data = (arr * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        data = arr.astype(np.float32)   # float32 is fine, BITPIX will be set to -32
                    else:
                        data = arr.astype(np.float32)

                # 2) Copy & clean the header
                hdr = original_header.copy()
                for key in ("BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BSCALE", "BZERO"):
                    if key in hdr:
                        del hdr[key]

                # 3) Create a new PrimaryHDU with the cleaned header and new data
                hdu = fits.PrimaryHDU(data=data, header=hdr)
                hdu.writeto(output_image_path, overwrite=True)
                print(f"Saved FITS starless image to: {output_image_path}")



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
        parser.add_argument('--chunk_size', type=int, default=512, help="Chunk size in pixels (default: 512)")
        parser.add_argument('--overlap', type=int, default=None, help="Overlap size in pixels (default: 0.125 * chunk_size)")
        args = parser.parse_args()

        use_gpu = not args.disable_gpu

        # Determine the overlap: if user didn't provide one, compute as 0.125 * chunk_size.
        if args.overlap is None:
            overlap = int(round(0.125 * args.chunk_size))
        else:
            overlap = args.overlap

        def cli_progress_callback(message):
            print(message, flush=True)

        process_images(
            input_dir, 
            output_dir,
            starremoval_strength=1.0,
            use_gpu=use_gpu,
            star_removal_mode=args.star_removal_mode,
            show_extracted_stars=args.show_extracted_stars,
            chunk_size=args.chunk_size,
            overlap=overlap,
            progress_callback=cli_progress_callback
        )
        sys.exit(0)

    else:
        # GUI mode: launch interactive UI
        app = QApplication(sys.argv)
        window = StarRemovalUI()
        window.show()
        sys.exit(app.exec())


