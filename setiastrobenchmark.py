import os
import sys
import time
import json
import torch
import requests
import numpy as np
import onnxruntime as ort
from astropy.io import fits
from numba import njit, jit, prange
import psutil
import platform
import cpuinfo
import multiprocessing
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton,
    QProgressBar, QTextEdit, QDialog, QHBoxLayout, QMessageBox
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from PIL import Image
from PIL.ImageQt import ImageQt  # Converts PIL image to QImage

# Check for CUDA, DirectML, or MPS (Metal Performance Shaders)
USE_CUDA = torch.cuda.is_available()
USE_MPS = torch.backends.mps.is_available() if platform.system() == "Darwin" else False

# Load Benchmark Image
BENCHMARK_IMAGE_PATH = "benchmarkimage.fit"
with fits.open(BENCHMARK_IMAGE_PATH) as hdul:
    image = hdul[0].data.astype(np.float32)
    if image.ndim == 2:  # Convert grayscale to 3-channel for CNN
        image = np.stack([image] * 3, axis=0)

H, W = image.shape[1:]  # Image height and width
PATCH_SIZE = 256  # CNN processes 256x256 patches

# ✅ Function to Tile Image into 256x256 Chunks
def tile_image(image_array, patch_size=256):
    """Splits an image into 256x256 patches for batch processing."""
    c, h, w = image_array.shape  # (Channels, Height, Width)
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image_array[:, i:i+patch_size, j:j+patch_size]  # Extract patch
            if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
                patches.append(patch)
    return np.array(patches)  # Shape: (num_patches, 3, 256, 256)

image_patches = tile_image(image)  # Tile into patches
image_tensor = torch.tensor(image_patches)  # Shape (N, 3, 256, 256)

# GitHub Repo Details
GITHUB_USERNAME = "setiastro"
GITHUB_REPO = "setiastrosuite"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # ⚠️ Store securely as an environment variable!
JSON_FILE_PATH = "benchmark_results.json"

# ✅ Fetch System Information
def get_system_info():
    """Retrieve OS, CPU, GPU, RAM, and available acceleration backends."""
    system_info = {
        "OS": platform.system() + " " + platform.release(),
        "CPU": cpuinfo.get_cpu_info()["brand_raw"],
        "RAM": f"{round(psutil.virtual_memory().total / (1024 ** 3), 1)} GB",
        "CUDA Available": torch.cuda.is_available(),
        "MPS Available": torch.backends.mps.is_available() if platform.system() == "Darwin" else False,
        "ONNX Providers": ort.get_available_providers()
    }

    # Get GPU details if CUDA is available
    if torch.cuda.is_available():
        system_info["GPU"] = torch.cuda.get_device_name(0)

    return system_info

def open_submission_page():
    """Open the Seti Astro benchmark submission page."""
    import webbrowser
    webbrowser.open("https://setiastro.com/benchmark-submit")

def save_results_locally(results):
    """Save benchmark results locally."""
    filename = "benchmark_results.json"

    # Load existing data if available
    if os.path.exists(filename):
        with open(filename, "r") as file:
            try:
                all_results = json.load(file)
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []

    # Append new results
    all_results.append(results)

    # Save updated JSON
    with open(filename, "w") as file:
        json.dump(all_results, file, indent=4)

    print("✅ Results saved to", filename)


# Define ResidualBlock for CNN
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

# Define SharpeningCNN for Stellar Sharpening
class SharpeningCNN(torch.nn.Module):
    def __init__(self):
        super(SharpeningCNN, self).__init__()

        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(16)
        )
        self.encoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(32)
        )
        self.encoder3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            torch.nn.ReLU(),
            ResidualBlock(64)
        )
        self.encoder4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(128)
        )
        self.encoder5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=2, dilation=2),
            torch.nn.ReLU(),
            ResidualBlock(256)
        )

        self.decoder5 = torch.nn.Sequential(
            torch.nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(128)
        )
        self.decoder4 = torch.nn.Sequential(
            torch.nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(64)
        )
        self.decoder3 = torch.nn.Sequential(
            torch.nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(32)
        )
        self.decoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            ResidualBlock(16)
        )
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 3, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
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

# ✅ Load Model
def load_model(exe_dir, use_gpu=True):
    """Load Stellar SharpeningCNN Model with CUDA, MPS, or CPU"""
    if USE_CUDA and use_gpu:
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration.")
    elif USE_MPS and use_gpu:
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    model = SharpeningCNN()
    model_path = os.path.join(exe_dir, 'deep_sharp_stellar_cnn_AI3_5.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    return model, device



# ✅ GPU Benchmark (CUDA) with 256x256 Chunks & AMP
def gpu_benchmark(model, device, image_patches, progress_callback):
    """Run GPU Benchmark using CUDA or MPS (Metal Performance Shaders) with AMP"""
    image_patches = image_patches.to(device)

    total_time = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # Enable Mixed Precision
            for i in range(len(image_patches)):
                patch = image_patches[i:i+1]
                start_time = time.time()
                _ = model(patch)

                if USE_CUDA:
                    torch.cuda.synchronize()
                elif USE_MPS:
                    torch.mps.synchronize()

                total_time += (time.time() - start_time) * 1000
                progress_callback(f"GPU Benchmark: {i+1}/{len(image_patches)} patches...", clear=True)

    avg_time = total_time / len(image_patches)
    return avg_time, total_time




# ✅ ONNX Benchmark (DirectML, CUDA, or CPU)
def onnx_benchmark(image_patches, progress_callback):
    """Run ONNX Benchmark using DirectML, CUDA, or CPU with proper model loading."""
    model_path = "deep_sharp_stellar_cnn_AI3_5.onnx"
    if not os.path.exists(model_path):
        return "ONNX Model Not Found"

    available_providers = ort.get_available_providers()

    # Prioritize DirectML, then CUDA, then CPU
    if "DmlExecutionProvider" in available_providers:
        provider = "DmlExecutionProvider"
    elif "CUDAExecutionProvider" in available_providers:
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    print(f"Using ONNX Provider: {provider}")

    # ✅ Load ONNX Model
    ort_session = ort.InferenceSession(model_path, providers=[provider])
    input_name = ort_session.get_inputs()[0].name

    total_time = 0
    for i in range(len(image_patches)):
        patch = image_patches[i:i+1].numpy().astype(np.float32)  # Convert patch to numpy
        start_time = time.time()
        ort_session.run(None, {input_name: patch})  # Run inference
        total_time += (time.time() - start_time) * 1000  # Convert to ms

        # ✅ Show Progress
        progress_callback(f"ONNX Benchmark: {i+1}/{len(image_patches)} patches...", clear=True)

    avg_time = total_time / len(image_patches)
    return avg_time, total_time


@njit
def mad_cpu_jit(image_array, median_val):
    """Compute Median Absolute Deviation (MAD)."""
    return np.median(np.abs(image_array - median_val))

def mad_cpu(image_array, runs=3):
    """Run MAD multiple times to separate JIT compilation overhead."""
    times = []
    for _ in range(runs):
        start_time = time.time()
        median_val = np.median(image_array)
        _ = mad_cpu_jit(image_array, median_val)
        times.append((time.time() - start_time) * 1000)  # Convert to ms
    return times  # Return all timings (first includes JIT compile time)

@njit(parallel=True)
def flat_field_correction_jit(image_array, flat_frame, median_flat):
    """Flat field correction using multi-threading."""
    return image_array / (flat_frame / median_flat)

def flat_field_correction(image_array, flat_frame, runs=3):
    """Run Flat-Field Correction multiple times to measure JIT overhead."""
    times = []
    for _ in range(runs):
        start_time = time.time()
        median_flat = np.median(flat_frame)
        _ = flat_field_correction_jit(image_array, flat_frame, median_flat)
        times.append((time.time() - start_time) * 1000)  # Convert to ms
    return times  # Return all timings

# ✅ Benchmark GUI
def resource_path(relative_path):
    """ Get the absolute path to a resource, works for dev and PyInstaller builds. """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class BenchmarkWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seti Astro Suite Benchmark")
        self.setGeometry(100, 100, 500, 600)

        # Main layout
        layout = QVBoxLayout(self)

        # Load and process image using Pillow
        image_path = resource_path("benchmark.png")
        original_image = Image.open(image_path)
        resized_image = original_image.resize((200, 200), Image.LANCZOS)

        # Convert PIL image to QPixmap via QImage
        qimage = ImageQt(resized_image)
        pixmap = QPixmap.fromImage(qimage)

        # Display the image in a QLabel
        self.logo_label = QLabel(self)
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.logo_label)

        # Main title label
        self.label = QLabel("Seti Astro Benchmark", self)
        title_font = QFont("Arial", 14)
        self.label.setFont(title_font)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        # Version label at the bottom
        self.version_label = QLabel("Version 1.0", self)
        version_font = QFont("Arial", 10)
        self.version_label.setFont(version_font)
        self.version_label.setStyleSheet("color: gray;")
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.version_label)

        # Dropdown / ComboBox for benchmark options
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(["CPU", "GPU", "Both"])
        self.dropdown.setCurrentText("Both")
        layout.addWidget(self.dropdown)

        # Run Benchmark button
        self.start_button = QPushButton("Run Benchmark", self)
        self.start_button.clicked.connect(self.run_benchmark)
        layout.addWidget(self.start_button)

        # Save Locally button (initially disabled)
        self.save_button = QPushButton("Save Locally", self)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_results_locally)
        layout.addWidget(self.save_button)

        # Submit Benchmark button
        self.submit_button = QPushButton("Submit Benchmark", self)
        self.submit_button.clicked.connect(open_submission_page)
        layout.addWidget(self.submit_button)

        # Progress bar
        self.progress = QProgressBar(self)
        self.progress.setOrientation(Qt.Orientation.Horizontal)
        self.progress.setFixedWidth(400)
        layout.addWidget(self.progress)

        # Multi-line text area for results
        self.result_text = QTextEdit(self)
        self.result_text.setFixedHeight(150)
        layout.addWidget(self.result_text)

        self.results = {}

    def run_benchmark(self):
        # Reset progress and clear previous results
        self.progress.setValue(0)
        self.result_text.clear()
        results = {}

        # Define a progress callback to update the UI
        def progress_callback(status, clear=False):
            if clear:
                self.result_text.clear()
            self.result_text.append(status)
            QApplication.processEvents()

        # Run CPU Benchmarks if selected
        if self.dropdown.currentText() in ["CPU", "Both"]:
            progress_callback("Running CPU Benchmarks...")
            QApplication.processEvents
            start_time = time.time()

            # Run MAD and Flat-Field Correction benchmarks (assumed external functions)
            cpu_times_mad = mad_cpu(image)
            cpu_times_flat = flat_field_correction(image, image)

            progress_callback("Completed CPU Benchmark.")
            results["CPU MAD (Single Core)"] = f"First: {cpu_times_mad[0]:.2f} ms | Avg: {np.mean(cpu_times_mad[1:]):.2f} ms"
            results["CPU Flat-Field (Multi-Core)"] = f"First: {cpu_times_flat[0]:.2f} ms | Avg: {np.mean(cpu_times_flat[1:]):.2f} ms"
            QApplication.processEvents

        # Run GPU Benchmarks if selected
        if self.dropdown.currentText() in ["GPU", "Both"]:
            progress_callback("Running CUDA Benchmarks...")

            # Load model properly before running the benchmark
            model, device = load_model(os.getcwd(), use_gpu=True)

            avg_gpu_time, total_gpu_time = gpu_benchmark(model, device, image_tensor, progress_callback)
            progress_callback("Completed CUDA Benchmark.")

            # Only run ONNX benchmark on Windows
            if platform.system() == "Windows":
                avg_onnx_time, total_onnx_time = onnx_benchmark(image_tensor, progress_callback)
                progress_callback("Completed ONNX Benchmark.", clear=True)
                results["ONNX Time"] = f"Avg: {avg_onnx_time:.2f} ms | Total: {total_onnx_time:.2f} ms"
            else:
                results["ONNX Time"] = "ONNX benchmark only available on Windows."

            results["GPU Time (CUDA)"] = f"Avg: {avg_gpu_time:.2f} ms | Total: {total_gpu_time:.2f} ms"


        # Collect System Info
        results["System Info"] = get_system_info()

        # Display Final Results
        final_results = "\n".join([f"{k}: {v}" for k, v in results.items()])
        self.result_text.append(final_results)
        self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())

        # Store Results for Saving/Uploading and enable the Save button
        self.results = results
        self.save_button.setEnabled(True)

        self.show_results_popup()

    def show_results_popup(self):
        """Display a pop-up with the benchmark JSON and Copy/Submit buttons."""
        popup = QDialog(self)
        popup.setWindowTitle("Benchmark Results")
        layout = QVBoxLayout(popup)

        label = QLabel("Your Benchmark JSON Data", popup)
        label.setFont(QFont("Arial", 12))
        layout.addWidget(label)

        # Convert results to JSON string
        json_string = json.dumps([self.results], indent=4)
        text_box = QTextEdit(popup)
        text_box.setReadOnly(True)
        text_box.setPlainText(json_string)
        layout.addWidget(text_box)

        button_frame = QHBoxLayout()
        copy_button = QPushButton("Copy to Clipboard", popup)
        copy_button.clicked.connect(lambda: self.copy_to_clipboard(json_string))
        button_frame.addWidget(copy_button)

        submit_button = QPushButton("Submit Benchmark", popup)
        submit_button.clicked.connect(open_submission_page)
        button_frame.addWidget(submit_button)

        layout.addLayout(button_frame)
        popup.exec()

    def copy_to_clipboard(self, text):
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "Copied", "Benchmark JSON copied to clipboard!")

    def save_results_locally(self):
        save_results_locally(self.results)
        self.result_text.append("\n✅ Results saved!\n")


icon_file = resource_path("benchmark.ico")
png_file = resource_path("benchmark.png")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    multiprocessing.freeze_support()
    window = BenchmarkWindow()
    window.show()
    sys.exit(app.exec())
