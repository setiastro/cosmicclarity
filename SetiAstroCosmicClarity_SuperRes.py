#!/usr/bin/env python
"""
Super-Resolution Upscaling Tool with Robust Load/Save

This script:
  • Loads an image using robust methods (supporting FITS, TIFF, XISF, RAW, PNG, JPG)
  • Applies a per-channel linear stretch (and later unstretches the result)
  • Adds a border, bicubically upscales the image by a specified factor (2×, 3×, or 4×)
  • Splits the upscaled image into overlapping 256×256 patches, processes each patch
    with a pre-trained SuperResolutionCNN model, and stitches them back together
  • Uses our robust save_image function to write out the final result in the same format as input
  • Provides both a PyQt6 GUI and command-line (headless) modes.
"""

import os, sys, time, math, argparse, gzip
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile as tiff
import onnxruntime as ort
# Additional dependencies for robust I/O
import rawpy
from astropy.io import fits
# Assuming you have a module named 'xisf' installed for XISF handling
from xisf import XISF
# ----- QtWidgets -----
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QGraphicsView,
    QGraphicsScene,
    QMessageBox,
    QInputDialog,
    QTreeWidget,
    QTreeWidgetItem,
    QCheckBox,
    QDialog,
    QFormLayout,
    QSpinBox,
    QDialogButtonBox,
    QGridLayout,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsPathItem,
    QDoubleSpinBox,
    QColorDialog,
    QFontDialog,
    QStyle,
    QSlider,
    QTabWidget,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QAbstractItemView,
    QToolBar,
    QGraphicsPixmapItem,
    QRubberBand,
    QGroupBox,
    QGraphicsTextItem,
    QComboBox,
    QLineEdit,
    QRadioButton,
    QButtonGroup,
    QHeaderView,
    QStackedWidget,
    QSplitter,
    QMenuBar,
    QTextEdit,
    QPlainTextEdit,      
    QProgressBar,
    QGraphicsItem,
    QToolButton,
    QStatusBar,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
    QListWidget,
    QListWidgetItem,
    QSplashScreen,
    QProgressDialog
)

# ----- QtGui -----
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QPainter,
    QPen,
    QColor,
    QTransform,
    QIcon,
    QPainterPath,
    QKeySequence,
    QFont,
    QMovie,
    QCursor,
    QBrush,
    QShortcut,
    QPolygon,
    QPolygonF,
    QPalette, 
    QWheelEvent, 
    QDoubleValidator,
    QAction  # NOTE: In PyQt6, QAction is in QtGui (moved from QtWidgets)
)

# ----- QtCore -----
from PyQt6.QtCore import (
    Qt,
    QRectF,
    QLineF,
    QPointF,
    QThread,
    pyqtSignal,
    QCoreApplication,
    QPoint,
    QTimer,
    QRect,
    QFileSystemWatcher,
    QEvent,
    pyqtSlot,
    QProcess,
    QSize,
    QObject,
    QSettings,
    QRunnable,
    QThreadPool
)
##########################################
# 1. Robust Load & Save Functions
##########################################

def resource_path(relative_path):
    """Get absolute path to resource, works for development and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_valid_header(file_path):
    """
    Opens the FITS file (handling compressed files as needed), finds the first HDU
    with image data, and then searches through all HDUs for additional keywords (e.g. BAYERPAT).
    Returns a composite header (a copy of the image HDU header updated with extra keywords)
    and the extension index of the image data.
    """
    if file_path.lower().endswith(('.fits.gz', '.fit.gz')):
        with gzip.open(file_path, 'rb') as f:
            file_content = f.read()
        hdul = fits.open(BytesIO(file_content))
    else:
        hdul = fits.open(file_path)
    with hdul as hdul:
        image_hdu = None
        image_index = None
        for i, hdu in enumerate(hdul):
            if hdu.data is not None:
                image_hdu = hdu
                image_index = i
                break
        if image_hdu is None:
            raise ValueError("No image data found in FITS file.")
        composite_header = image_hdu.header.copy()
        for i, hdu in enumerate(hdul):
            if 'BAYERPAT' in hdu.header:
                composite_header['BAYERPAT'] = hdu.header['BAYERPAT']
                break
    return composite_header, image_index

def get_bayer_header(file_path):
    """
    Iterates through all HDUs in the FITS file to find a header that contains 'BAYERPAT'.
    Returns the header if found, otherwise None.
    """
    try:
        if file_path.lower().endswith(('.fits.gz', '.fit.gz')):
            with gzip.open(file_path, 'rb') as f:
                file_content = f.read()
            hdul = fits.open(BytesIO(file_content))
        else:
            hdul = fits.open(file_path)
        with hdul as hdul:
            for hdu in hdul:
                if 'BAYERPAT' in hdu.header:
                    return hdu.header
    except Exception as e:
        print(f"Error in get_bayer_header: {e}")
    return None

def load_image(filename, max_retries=3, wait_seconds=3):
    """
    Loads an image from the specified filename with support for various formats.
    If a "buffer is too small for requested array" error occurs, it retries loading after waiting.

    Returns:
        tuple: (image, original_header, bit_depth, is_mono) or (None, None, None, None) on failure.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            image = None
            bit_depth = None
            is_mono = False
            original_header = None

            if filename.lower().endswith(('.fits', '.fit', '.fits.gz', '.fit.gz', '.fz', '.fz')):
                original_header, ext_index = get_valid_header(filename)
                if filename.lower().endswith(('.fits.gz', '.fit.gz')):
                    print(f"Loading compressed FITS file: {filename}")
                    with gzip.open(filename, 'rb') as f:
                        file_content = f.read()
                    hdul = fits.open(BytesIO(file_content))
                else:
                    if filename.lower().endswith(('.fz', '.fz')):
                        print(f"Loading Rice-compressed FITS file: {filename}")
                    else:
                        print(f"Loading FITS file: {filename}")
                    hdul = fits.open(filename)
                with hdul as hdul:
                    image_data = hdul[ext_index].data
                    if image_data is None:
                        raise ValueError(f"No image data found in FITS file in extension {ext_index}.")
                    if image_data.dtype.byteorder not in ('=', '|'):
                        image_data = image_data.astype(image_data.dtype.newbyteorder('='))
                    if image_data.dtype == np.uint8:
                        bit_depth = "8-bit"
                        print("Identified 8-bit FITS image.")
                        image = image_data.astype(np.float32) / 255.0
                    elif image_data.dtype == np.uint16:
                        bit_depth = "16-bit"
                        print("Identified 16-bit FITS image.")
                        image = image_data.astype(np.float32) / 65535.0
                    elif image_data.dtype == np.int32:
                        bit_depth = "32-bit signed"
                        print("Identified 32-bit signed FITS image.")
                        bzero  = original_header.get('BZERO', 0)
                        bscale = original_header.get('BSCALE', 1)
                        image = image_data.astype(np.float32) * bscale + bzero
                    elif image_data.dtype == np.uint32:
                        bit_depth = "32-bit unsigned"
                        print("Identified 32-bit unsigned FITS image.")
                        bzero  = original_header.get('BZERO', 0)
                        bscale = original_header.get('BSCALE', 1)
                        image = image_data.astype(np.float32) * bscale + bzero
                    elif image_data.dtype == np.float32:
                        bit_depth = "32-bit floating point"
                        print("Identified 32-bit floating point FITS image.")
                        image = image_data
                    else:
                        raise ValueError(f"Unsupported FITS data type: {image_data.dtype}")
                    image = np.squeeze(image)
                    if image.ndim == 2:
                        is_mono = True
                    elif image.ndim == 3:
                        if image.shape[0] == 3 and image.shape[1] > 1 and image.shape[2] > 1:
                            image = np.transpose(image, (1, 2, 0))
                            is_mono = False
                        elif image.shape[-1] == 3:
                            is_mono = False
                        else:
                            raise ValueError(f"Unsupported 3D shape after squeeze: {image.shape}")
                    else:
                        raise ValueError(f"Unsupported FITS dimensions after squeeze: {image.shape}")
                    print(f"Loaded FITS image: shape={image.shape}, bit depth={bit_depth}, mono={is_mono}")
                    return image, original_header, bit_depth, is_mono

            elif filename.lower().endswith(('.tiff', '.tif')):
                print(f"Loading TIFF file: {filename}")
                image_data = tiff.imread(filename)
                print(f"Loaded TIFF image with dtype: {image_data.dtype}")
                if image_data.dtype == np.uint8:
                    bit_depth = "8-bit"
                    image = image_data.astype(np.float32) / 255.0
                elif image_data.dtype == np.uint16:
                    bit_depth = "16-bit"
                    image = image_data.astype(np.float32) / 65535.0
                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    image = image_data.astype(np.float32) / 4294967295.0
                elif image_data.dtype == np.float32:
                    bit_depth = "32-bit floating point"
                    image = image_data
                else:
                    raise ValueError("Unsupported TIFF format!")
                if image_data.ndim == 2:
                    is_mono = True
                elif image_data.ndim == 3 and image_data.shape[2] == 3:
                    is_mono = False
                else:
                    raise ValueError("Unsupported TIFF image dimensions!")
            elif filename.lower().endswith('.xisf'):
                print(f"Loading XISF file: {filename}")
                xisf = XISF(filename)
                image_data = xisf.read_image(0)
                image_meta = xisf.get_images_metadata()[0]
                file_meta = xisf.get_file_metadata()
                if image_data.dtype == np.uint8:
                    bit_depth = "8-bit"
                    print("Detected 8-bit dtype. Normalizing by 255.")
                    image = image_data.astype(np.float32) / 255.0
                elif image_data.dtype == np.uint16:
                    bit_depth = "16-bit"
                    print("Detected 16-bit dtype. Normalizing by 65535.")
                    image = image_data.astype(np.float32) / 65535.0
                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    print("Detected 32-bit unsigned dtype. Normalizing by 4294967295.")
                    image = image_data.astype(np.float32) / 4294967295.0
                elif image_data.dtype in [np.float32, np.float64]:
                    bit_depth = "32-bit floating point"
                    print("Detected float dtype. Casting to float32 (no normalization).")
                    image = image_data.astype(np.float32)
                else:
                    raise ValueError(f"Unsupported XISF data type: {image_data.dtype}")
                if image_data.ndim == 2:
                    is_mono = True
                    image = np.stack([image] * 3, axis=-1)
                elif image_data.ndim == 3 and image_data.shape[2] == 1:
                    is_mono = True
                    image = np.squeeze(image, axis=2)
                    image = np.stack([image] * 3, axis=-1)
                elif image_data.ndim == 3 and image_data.shape[2] == 3:
                    is_mono = False
                else:
                    raise ValueError("Unsupported XISF image dimensions!")
                original_header = {"file_meta": file_meta, "image_meta": image_meta}
                print(f"Loaded XISF image: shape={image.shape}, bit depth={bit_depth}, mono={is_mono}")
                return image, original_header, bit_depth, is_mono
            elif filename.lower().endswith(('.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
                print(f"Loading RAW file: {filename}")
                with rawpy.imread(filename) as raw:
                    bayer_image = raw.raw_image_visible.astype(np.float32)
                    print(f"Raw Bayer image: min {bayer_image.min():.2f}, max {bayer_image.max():.2f}")
                    black_levels = raw.black_level_per_channel
                    white_level  = raw.white_level
                    avg_black = float(np.mean(black_levels))
                    bayer_image -= avg_black
                    bayer_image = np.clip(bayer_image, 0, None)
                    scale_val = float(white_level - avg_black)
                    if scale_val <= 0:
                        scale_val = 1.0
                    bayer_image /= scale_val
                    if bayer_image.ndim == 2:
                        image = bayer_image
                        is_mono = True
                    elif bayer_image.ndim == 3 and bayer_image.shape[2] == 3:
                        image = bayer_image
                        is_mono = False
                    else:
                        raise ValueError(f"Unexpected RAW Bayer image shape: {bayer_image.shape}")
                    bit_depth = "16-bit"
                    original_header_dict = {
                        'CAMERA': raw.camera_whitebalance[0] if raw.camera_whitebalance else 'Unknown',
                        'EXPTIME': getattr(raw, 'shutter', 0.0),
                        'ISO': getattr(raw, 'iso_speed', 0),
                        'FOCAL': getattr(raw, 'focal_len', 0.0),
                        'DATE': getattr(raw, 'timestamp', 'Unknown'),
                    }
                    cfa_pattern = raw.raw_colors_visible
                    cfa_mapping = {0: 'R', 1: 'G', 2: 'B'}
                    cfa_description = ''.join([cfa_mapping.get(color, '?') for color in cfa_pattern.flatten()[:4]])
                    original_header_dict['CFA'] = (cfa_description, 'CFA pattern')
                    original_header = fits.Header()
                    for key, value in original_header_dict.items():
                        original_header[key] = value
                    print(f"RAW file loaded with CFA: {cfa_description}")
                    return image, original_header, bit_depth, is_mono
            elif filename.lower().endswith('.png'):
                print(f"Loading PNG file: {filename}")
                img = Image.open(filename)
                if img.mode not in ('L', 'RGB'):
                    print(f"Unsupported PNG mode: {img.mode}, converting to RGB")
                    img = img.convert("RGB")
                image = np.array(img, dtype=np.float32) / 255.0
                bit_depth = "8-bit"
                if len(image.shape) == 2:
                    is_mono = True
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    is_mono = False
                else:
                    raise ValueError(f"Unsupported PNG dimensions: {image.shape}")
            elif filename.lower().endswith(('.jpg', '.jpeg')):
                print(f"Loading JPG file: {filename}")
                img = Image.open(filename)
                if img.mode == 'L':
                    is_mono = True
                    image = np.array(img, dtype=np.float32) / 255.0
                    bit_depth = "8-bit"
                elif img.mode == 'RGB':
                    is_mono = False
                    image = np.array(img, dtype=np.float32) / 255.0
                    bit_depth = "8-bit"
                else:
                    raise ValueError("Unsupported JPG format!")
            else:
                raise ValueError("Unsupported file format!")
            print(f"Loaded image: shape={image.shape}, bit depth={bit_depth}, mono={is_mono}")
            return image, original_header, bit_depth, is_mono

        except Exception as e:
            error_message = str(e)
            if "buffer is too small for requested array" in error_message.lower():
                if attempt < max_retries:
                    attempt += 1
                    print(f"Error reading image {filename}: {e}")
                    print(f"Retrying in {wait_seconds} seconds... (Attempt {attempt}/{max_retries})")
                    time.sleep(wait_seconds)
                    continue
                else:
                    print(f"Error reading image {filename} after {max_retries} retries: {e}")
            else:
                print(f"Error reading image {filename}: {e}")
            return None, None, None, None

def ensure_native_byte_order(arr):
    """Ensure the NumPy array uses native byte order."""
    if arr.dtype.byteorder not in ('=', '|'):
        return arr.astype(arr.dtype.newbyteorder('='))
    return arr

def save_image(img_array, filename, original_format, bit_depth=None, original_header=None, is_mono=False, image_meta=None, file_meta=None):
    """
    Save an image array to a file in the specified format and bit depth.
    Uses robust methods to preserve metadata.
    """
    img_array = ensure_native_byte_order(img_array)
    is_xisf = False
    if original_header:
        for key in original_header.keys():
            if key.startswith("XISF:"):
                is_xisf = True
                break
    if image_meta and "XISFProperties" in image_meta:
        is_xisf = True
    try:
        if original_format == 'png':
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            img.save(filename)
            print(f"Saved 8-bit PNG image to: {filename}")
        elif original_format in ['jpg', 'jpeg']:
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            img.save(filename)
            print(f"Saved 8-bit JPG image to: {filename}")
        elif original_format in ['tiff', 'tif']:
            if bit_depth == "8-bit":
                tiff.imwrite(filename, (img_array * 255).astype(np.uint8))
            elif bit_depth == "16-bit":
                tiff.imwrite(filename, (img_array * 65535).astype(np.uint16))
            elif bit_depth == "32-bit unsigned":
                tiff.imwrite(filename, (img_array * 4294967295).astype(np.uint32))
            elif bit_depth == "32-bit floating point":
                tiff.imwrite(filename, img_array.astype(np.float32))
            else:
                raise ValueError("Unsupported bit depth for TIFF!")
            print(f"Saved {bit_depth} TIFF image to: {filename}")
        elif original_format in ['fits', 'fit']:
            if not filename.lower().endswith(f".{original_format}"):
                filename = filename.rsplit('.', 1)[0] + f".{original_format}"
            # Prepare the FITS header
            if is_xisf:
                print("Detected XISF metadata. Converting to FITS header...")
                fits_header = fits.Header()
                if image_meta and 'XISFProperties' in image_meta:
                    xisf_props = image_meta['XISFProperties']
                    if 'PCL:AstrometricSolution:ReferenceCoordinates' in xisf_props:
                        ref_coords = xisf_props['PCL:AstrometricSolution:ReferenceCoordinates']['value']
                        fits_header['CRVAL1'] = ref_coords[0]
                        fits_header['CRVAL2'] = ref_coords[1]
                    if 'PCL:AstrometricSolution:ReferenceLocation' in xisf_props:
                        ref_pixel = xisf_props['PCL:AstrometricSolution:ReferenceLocation']['value']
                        fits_header['CRPIX1'] = ref_pixel[0]
                        fits_header['CRPIX2'] = ref_pixel[1]
                    if 'PCL:AstrometricSolution:PixelSize' in xisf_props:
                        pixel_size = xisf_props['PCL:AstrometricSolution:PixelSize']['value']
                        fits_header['CDELT1'] = -pixel_size / 3600.0
                        fits_header['CDELT2'] = pixel_size / 3600.0
                    if 'PCL:AstrometricSolution:LinearTransformationMatrix' in xisf_props:
                        linear_transform = xisf_props['PCL:AstrometricSolution:LinearTransformationMatrix']['value']
                        fits_header['CD1_1'] = linear_transform[0][0]
                        fits_header['CD1_2'] = linear_transform[0][1]
                        fits_header['CD2_1'] = linear_transform[1][0]
                        fits_header['CD2_2'] = linear_transform[1][1]
                fits_header.setdefault('CTYPE1', 'RA---TAN')
                fits_header.setdefault('CTYPE2', 'DEC--TAN')
            elif original_header is not None:
                print("Detected FITS format. Preserving original FITS header.")
                fits_header = fits.Header()
                for key, value in original_header.items():
                    if key.startswith("XISF:"):
                        continue
                    if key in ["RANGE_LOW", "RANGE_HIGH"]:
                        continue
                    if isinstance(value, dict) and 'value' in value:
                        value = value['value']
                    try:
                        fits_header[key] = value
                    except Exception as e:
                        print(f"Skipping key {key}")
            else:
                # Create a default header if none was provided.
                print("No original header found; creating a default header.")
                fits_header = fits.Header()

            fits_header['BSCALE'] = 1.0
            fits_header['BZERO'] = 0.0


            if is_mono or img_array.ndim == 2:
                if img_array.ndim == 3:
                    img_array_fits = img_array[:, :, 0]
                else:
                    img_array_fits = img_array
                fits_header['NAXIS'] = 2
                fits_header.pop('NAXIS3', None)
            else:
                # Explicitly check and fix the shape to (3, H, W)
                if img_array.ndim == 3:
                    if img_array.shape[2] == 3:
                        img_array_fits = np.transpose(img_array, (2, 0, 1))
                    elif img_array.shape[0] == 3:
                        img_array_fits = img_array
                    else:
                        raise ValueError(f"Unexpected color image shape {img_array.shape} when saving FITS!")
                else:
                    raise ValueError(f"Unexpected array dimensions {img_array.ndim} when saving FITS!")

                fits_header['NAXIS'] = 3
                fits_header['NAXIS3'] = 3

            fits_header['NAXIS1'] = img_array_fits.shape[-1]  # width
            fits_header['NAXIS2'] = img_array_fits.shape[-2]  # height

            hdu = fits.PrimaryHDU(img_array_fits, header=fits_header)
            hdu.writeto(filename, overwrite=True)
            print(f"Saved FITS image to: {filename}")
            return
        elif original_format in ['.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef']:
            print("RAW formats are not writable. Saving as FITS instead.")
            filename = filename.rsplit('.', 1)[0] + ".fits"
            if original_header is not None:
                fits_header = fits.Header()
                for key, value in original_header.items():
                    fits_header[key] = value
                fits_header['BSCALE'] = 1.0
                fits_header['BZERO'] = 0.0
                if is_mono:
                    if bit_depth == "16-bit":
                        img_array_fits = (img_array[:, :, 0] * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        img_array_fits = (img_array[:, :, 0].astype(np.float32)).astype(np.uint32)
                    else:
                        img_array_fits = img_array[:, :, 0].astype(np.float32)
                    fits_header['NAXIS'] = 2
                    fits_header['NAXIS1'] = img_array.shape[1]
                    fits_header['NAXIS2'] = img_array.shape[0]
                    fits_header.pop('NAXIS3', None)
                    hdu = fits.PrimaryHDU(img_array_fits, header=fits_header)
                else:
                    img_array_transposed = np.transpose(img_array, (2,0,1))
                    if bit_depth == "16-bit":
                        img_array_fits = (img_array_transposed * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        img_array_fits = img_array_transposed.astype(np.float32).astype(np.uint32)
                        fits_header['BITPIX'] = -32
                    else:
                        img_array_fits = img_array_transposed.astype(np.float32)
                    fits_header['NAXIS'] = 3
                    fits_header['NAXIS1'] = img_array_transposed.shape[2]
                    fits_header['NAXIS2'] = img_array_transposed.shape[1]
                    fits_header['NAXIS3'] = img_array_transposed.shape[0]
                    hdu = fits.PrimaryHDU(img_array_fits, header=fits_header)
                try:
                    hdu.writeto(filename, overwrite=True)

                except Exception as e:
                    print(f"Error saving FITS file: {e}")
            else:
                raise ValueError("Original header is required for FITS format!")
        elif original_format == 'xisf':
            try:
                print(f"Processed image shape: {img_array.shape}, dtype: {img_array.dtype}")
                if bit_depth == "16-bit":
                    processed_image = (img_array * 65535).astype(np.uint16)
                elif bit_depth == "32-bit unsigned":
                    processed_image = (img_array * 4294967295).astype(np.uint32)
                else:
                    processed_image = img_array.astype(np.float32)
                if is_mono:
                    if processed_image.ndim == 3 and processed_image.shape[2] > 1:
                        processed_image = processed_image[:, :, 0]
                    processed_image = processed_image[:, :, np.newaxis]
                    if image_meta and isinstance(image_meta, list):
                        image_meta[0]['geometry'] = (processed_image.shape[1], processed_image.shape[0], 1)
                        image_meta[0]['colorSpace'] = 'Gray'
                    else:
                        image_meta = [{'geometry': (processed_image.shape[1], processed_image.shape[0], 1), 'colorSpace': 'Gray'}]
                else:
                    if image_meta and isinstance(image_meta, list):
                        image_meta[0]['geometry'] = (processed_image.shape[1], processed_image.shape[0], processed_image.shape[2])
                        image_meta[0]['colorSpace'] = 'RGB'
                    else:
                        image_meta = [{'geometry': (processed_image.shape[1], processed_image.shape[0], processed_image.shape[2]), 'colorSpace': 'RGB'}]
                if image_meta is None or not isinstance(image_meta, list):
                    image_meta = [{'geometry': (processed_image.shape[1], processed_image.shape[0], 1 if is_mono else 3), 'colorSpace': 'Gray' if is_mono else 'RGB'}]
                if file_meta is None:
                    file_meta = {}
                print(f"Processed image shape for XISF: {processed_image.shape}, dtype: {processed_image.dtype}")
                XISF.write(
                    filename,
                    processed_image,
                    creator_app="Seti Astro Cosmic Clarity",
                    image_metadata=image_meta[0],
                    xisf_metadata=file_meta,
                    shuffle=True
                )
                print(f"Saved {bit_depth} XISF image to: {filename}")
            except Exception as e:
                print(f"Error saving XISF file: {e}")
                raise
        else:
            raise ValueError("Unsupported file format!")
    except Exception as e:
        print(f"Error saving image to {filename}: {e}")
        raise

##########################################
# 2. Stretch / Unstretch Functions
##########################################
def stretch_image_custom(image):
    """
    Perform a linear stretch on the image with unlinked channels.
    Returns (stretched_image, original_min, original_medians)
    """
    original_min = np.min(image)
    stretched_image = image - original_min
    original_medians = []
    for c in range(3):
        channel_median = np.median(stretched_image[..., c])
        original_medians.append(channel_median)
    target_median = 0.25
    for c in range(3):
        channel_median = original_medians[c]
        if channel_median != 0:
            stretched_image[..., c] = ((channel_median - 1) * target_median * stretched_image[..., c]) / (
                channel_median * (target_median + stretched_image[..., c] - 1) - target_median * stretched_image[..., c]
            )
    stretched_image = np.clip(stretched_image, 0, 1)
    return stretched_image, original_min, original_medians

def unstretch_image_custom(image, original_medians, original_min):
    """
    Undo the linear stretch.
    """
    was_single_channel = False
    if image.ndim == 3 and image.shape[2] == 1:
        was_single_channel = True
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 2:
        was_single_channel = True
        image = np.stack([image]*3, axis=-1)
    for c in range(3):
        channel_median = np.median(image[..., c])
        if channel_median != 0 and original_medians[c] != 0:
            image[..., c] = ((channel_median - 1) * original_medians[c] * image[..., c]) / (
                channel_median * (original_medians[c] + image[..., c] - 1) - original_medians[c] * image[..., c]
            )
        else:
            print(f"Channel {c} - Skipping unstretch due to zero median.")
    image += original_min
    image = np.clip(image, 0, 1)
    if was_single_channel:
        image = np.mean(image, axis=2, keepdims=True)
    return image

##########################################
# 3. Other Utility Functions
##########################################
def add_border(image, border_size=16):
    median_val = np.median(image)
    if image.ndim == 2:
        return np.pad(image, ((border_size, border_size), (border_size, border_size)), mode='constant', constant_values=median_val)
    else:
        return np.pad(image, ((border_size, border_size), (border_size, border_size), (0,0)), mode='constant', constant_values=median_val)

def remove_border(image, border_size):
    if image.ndim == 2:
        return image[border_size:-border_size, border_size:-border_size]
    else:
        return image[border_size:-border_size, border_size:-border_size, :]

def split_image_into_chunks_with_overlap(image, chunk_size=256, overlap=64):
    h, w = image.shape[:2]
    chunks = []
    step = chunk_size - overlap
    for i in range(0, h, step):
        for j in range(0, w, step):
            end_i = min(i+chunk_size, h)
            end_j = min(j+chunk_size, w)
            patch = image[i:end_i, j:end_j]
            is_edge = (i==0 or j==0 or end_i==h or end_j==w)
            chunks.append((patch, i, j, is_edge))
    return chunks

def stitch_chunks_ignore_border(chunks, image_shape, chunk_size=256, overlap=64, border_size=16):
    stitched = np.zeros(image_shape, dtype=np.float32)
    weight_map = np.zeros(image_shape, dtype=np.float32)
    for patch, i, j, is_edge in chunks:
        ph, pw = patch.shape[:2]
        b_h = min(border_size, ph//2)
        b_w = min(border_size, pw//2)
        inner = patch[b_h:ph-b_h, b_w:pw-b_w]
        h_inner, w_inner = inner.shape[:2]
        stitched[i+b_h:i+b_h+h_inner, j+b_w:j+b_w+w_inner] += inner
        weight_map[i+b_h:i+b_h+h_inner, j+b_w:j+b_w+w_inner] += 1
    stitched /= np.maximum(weight_map, 1)
    return stitched

##########################################
# 4. Neural Net Model
##########################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class SuperResolutionCNN(nn.Module):
    def __init__(self):
        super(SuperResolutionCNN, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(16)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(32)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            ResidualBlock(256)
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(256+128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128+64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64+32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(32)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32+16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(16)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
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

# ---------------------------------------------
# Model Loading Helper Function
# ---------------------------------------------
def load_superres_model(scale, model_dir):
    """
    Load the super-resolution model for the given scale and model directory.
    Supports PyTorch and ONNX fallback, with PyInstaller compatibility.
    """
    import sys

    # 🛠 Make model_dir work with PyInstaller --onefile
    try:
        if hasattr(sys, "_MEIPASS"):
            model_dir = sys._MEIPASS
    except Exception:
        pass
    """
    Load the super-resolution model for the given scale and model directory.
    
    On Windows:
      - If CUDA is available, load the PyTorch .pth model.
      - Otherwise, if ONNX runtime has DirectML available, load the ONNX model.
      - Otherwise, fall back on CPU PyTorch.
      
    On Linux:
      - Use CUDA if available, else CPU.
      
    On macOS:
      - Use MPS if available, else CPU.
      
    Returns:
        (model, device, use_pytorch) where use_pytorch is a bool.
    """
    import sys
    if sys.platform.startswith("win"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            use_pytorch = True
        else:
            providers = ort.get_available_providers()
            if "DmlExecutionProvider" in providers:
                device = "DirectML"
                use_pytorch = False
            else:
                device = torch.device("cpu")
                use_pytorch = True
    elif sys.platform.startswith("linux"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        use_pytorch = True
    elif sys.platform == "darwin":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        use_pytorch = True
    else:
        device = torch.device("cpu")
        use_pytorch = True

    if use_pytorch:
        model_filename = os.path.join(model_dir, f"superres_{scale}x.pth")
        if not os.path.exists(model_filename):
            raise ValueError(f"Model file not found: {model_filename}")
        model = SuperResolutionCNN().to(device)
        model.load_state_dict(torch.load(model_filename, map_location=device, weights_only=True))
        model.eval()
        return model, device, True
    else:
        # Using ONNX runtime with DirectML on Windows
        model_filename = os.path.join(model_dir, f"superres_{scale}x.onnx")
        if not os.path.exists(model_filename):
            raise ValueError(f"ONNX model file not found: {model_filename}")
        sess = ort.InferenceSession(model_filename, providers=["DmlExecutionProvider"])
        return sess, None, False


# ---------------------------------------------
# Updated process_image function
# ---------------------------------------------
def process_image(input_path, scale, model, device, use_pytorch, progress_callback=None):
    """
    Process an image:
      - Loads the image using robust load_image
      - Adds a border and applies a linear stretch
      - Upscales using bicubic interpolation by the given scale
      - If the original image is colored, processes each channel separately 
        (by duplicating the single channel to match the network’s expected 3-channel input)
      - Splits the upscaled image into overlapping 256×256 patches
      - Processes each patch with the neural net (using PyTorch or ONNX)
      - Stitches patches back together, unstretches the result, and removes the border
    Returns the final image as a NumPy array in [0,1].
    """
    load_result = load_image(input_path)
    if load_result[0] is None:
        return None, None, None, None, None
    image, original_header, bit_depth, is_mono = load_result
    print(f"Loaded image: shape={image.shape}, bit depth={bit_depth}-bit, mono={is_mono}")
    ext = os.path.splitext(input_path)[1].lower().strip('.')

    def process_single_channel(channel_image):

        
        # Add border and stretch
        channel_border = add_border(channel_image, border_size=16)

        stretched, orig_min, orig_medians = stretch_image_custom(channel_border)


        # Upscale using bicubic interpolation
        h, w = stretched.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        upscaled = cv2.resize(stretched, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


        # Split into chunks and process each patch
        chunks = split_image_into_chunks_with_overlap(upscaled, chunk_size=256, overlap=64)
        processed_chunks = []

        for idx, (patch, i, j, is_edge) in enumerate(chunks):
            ph, pw = patch.shape[:2]
            # Prepare patch input for network: duplicate single channel to 3 channels
            patch_input = np.zeros((256, 256, 3), dtype=np.float32)
            patch_input[:ph, :pw, :] = np.repeat(patch[..., np.newaxis], 3, axis=2)


            # Process the patch with the model
            if use_pytorch:
                patch_tensor = torch.from_numpy(patch_input.transpose(2, 0, 1)).unsqueeze(0).to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' if device else False)):
                    output = model(patch_tensor)
                out_np = output.squeeze().detach().cpu().numpy()
            else:
                patch_input_np = np.expand_dims(patch_input.transpose(2, 0, 1), axis=0).astype(np.float32)
                result = model.run(None, {model.get_inputs()[0].name: patch_input_np})[0]
                out_np = result.squeeze()


            # Extract the first channel (since output is 3-channel grayscale)
            if out_np.ndim == 3:
                if out_np.shape[0] == 3:
                    out_np = out_np[0, :, :]
                elif out_np.shape[-1] == 3:
                    out_np = out_np[..., 0]


            # Crop to original patch size
            out_np = out_np[:ph, :pw]

            processed_chunks.append((out_np, i, j, is_edge))

            if progress_callback:
                progress_callback(idx+1, len(chunks))

        # Stitch the processed patches back together
        stitched = stitch_chunks_ignore_border(
            processed_chunks, upscaled.shape[:2], chunk_size=256, overlap=64, border_size=16
        )

        unstretched = unstretch_image_custom(stitched, orig_medians, orig_min)

        border = int(16 * scale)
        final_channel = remove_border(unstretched, border_size=border)

        # Squeeze out extra channel dimension if present
        if final_channel.ndim == 3 and final_channel.shape[-1] == 1:
            final_channel = final_channel[..., 0]

        return final_channel

    if is_mono:
        final = process_single_channel(image)

    else:
        final_channels = []
        for c in range(3):
            print(f"\n[DEBUG] Processing color channel {c+1}/3")
            channel_result = process_single_channel(image[..., c])
            print(f"[DEBUG] Color channel {c+1} result shape: {channel_result.shape}")
            final_channels.append(channel_result)
        # Combine channels into final RGB image
        final = np.stack(final_channels, axis=-1)
        print(f"[DEBUG] Final RGB image shape after stacking: {final.shape}")

    return final, original_header, bit_depth, ext, is_mono

# ---------------------------------------------
# ProcessingThread remains largely the same; now also pass use_pytorch
# ---------------------------------------------
class ProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object, object, object, object, object)  # final_image, original_header, bit_depth, original_format, is_mono
    def __init__(self, input_path, scale, model, device, use_pytorch):
        super().__init__()
        self.input_path = input_path
        self.scale = scale
        self.model = model
        self.device = device
        self.use_pytorch = use_pytorch
    def run(self):
        def progress_callback(current, total):
            pct = int((current/total)*100)
            self.progress_signal.emit(pct)
        result = process_image(self.input_path, self.scale, self.model, self.device, self.use_pytorch, progress_callback)
        self.finished_signal.emit(*result)

class UpscalingApp(QMainWindow):
    def __init__(self, model, device, use_pytorch):
        super().__init__()
        self.model = model
        self.device = device
        self.use_pytorch = use_pytorch  # Save the flag for later use
        self.setWindowTitle("Cosmic Clarity Super-Resolution Upscaling Tool")
        self.setWindowIcon(QIcon(resource_path("upscale.ico")))
        self.resize(600, 300)
        self.initUI()

    def initUI(self):
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Input file selection
        input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        btn_browse_input = QPushButton("Browse...")
        btn_browse_input.clicked.connect(self.select_input)
        input_layout.addWidget(QLabel("Input File:"))
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(btn_browse_input)
        layout.addLayout(input_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        self.outdir_edit = QLineEdit()
        btn_browse_outdir = QPushButton("Browse Directory...")
        btn_browse_outdir.clicked.connect(self.select_output_directory)
        output_layout.addWidget(QLabel("Output Directory:"))
        output_layout.addWidget(self.outdir_edit)
        output_layout.addWidget(btn_browse_outdir)
        layout.addLayout(output_layout)

        # Output file type selection
        type_layout = QHBoxLayout()
        self.outtype_combo = QComboBox()
        self.outtype_combo.addItems(["FITS", "TIFF", "PNG"])
        type_layout.addWidget(QLabel("Output File Type:"))
        type_layout.addWidget(self.outtype_combo)
        layout.addLayout(type_layout)

        # Bit depth selection
        depth_layout = QHBoxLayout()
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["32-bit floating point", "32-bit unsigned", "16-bit", "8-bit"])
        depth_layout.addWidget(QLabel("Bit Depth:"))
        depth_layout.addWidget(self.depth_combo)
        layout.addLayout(depth_layout)

        # Upscale factor selection
        scale_layout = QHBoxLayout()
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["2", "3", "4"])
        scale_layout.addWidget(QLabel("Upscale Factor:"))
        scale_layout.addWidget(self.scale_combo)
        layout.addLayout(scale_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Run button
        btn_run = QPushButton("Run Upscaling")
        btn_run.clicked.connect(self.run_processing)
        layout.addWidget(btn_run)

        # Authorship/Version info with clickable link
        authorship_label = QLabel()
        authorship_label.setText(
            "Written by Franklin Marek © 2025 - <a href='http://www.setiastro.com'>www.setiastro.com</a>"
        )
        authorship_label.setOpenExternalLinks(True)
        authorship_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(authorship_label)

    def select_input(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Input Image", "", 
            "Images (*.png *.tif *.tiff *.fits *.fit *.jpg *.jpeg *.xisf)"
        )
        if file_name:
            self.input_edit.setText(file_name)

    def select_output_directory(self):
        # If an input file has already been selected, use its directory as default.
        if self.input_edit.text():
            initial_dir = os.path.dirname(self.input_edit.text())
        else:
            initial_dir = os.path.expanduser("~")
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory", initial_dir)
        if dir_name:
            self.outdir_edit.setText(dir_name)

    def generate_output_filename(self, input_path, output_dir, scale, output_type):
        # If no output directory is specified, use the input file's directory.
        if not output_dir:
            output_dir = os.path.dirname(input_path)
        else:
            output_dir = os.path.abspath(output_dir)
        base = os.path.splitext(os.path.basename(input_path))[0]
        suffix = f"_upscaled{int(scale)}x"  # Convert scale to int (e.g. 2 instead of 2.0)
        ext = output_type.lower()
        # Map file type to the proper extension.
        if ext == "fits":
            ext = ".fit"
        elif ext == "tiff":
            ext = ".tif"
        else:
            ext = ".png"
        return os.path.join(output_dir, base + suffix + ext)

    def run_processing(self):
        input_path = self.input_edit.text().strip()
        outdir = self.outdir_edit.text().strip()
        if not input_path or not outdir:
            return
        scale = float(self.scale_combo.currentText())
        out_type = self.outtype_combo.currentText().lower()
        bit_depth = self.depth_combo.currentText()
        self.progress_bar.setValue(0)
        self.thread = ProcessingThread(input_path, scale, self.model, self.device, self.use_pytorch)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.finished_signal.connect(lambda img, hdr, bd, orig_fmt, mono: self.save_result(img, hdr, bit_depth, out_type, outdir, input_path, scale, mono))
        self.thread.start()

    def save_result(self, img, original_header, bit_depth, original_format, outdir, input_path, scale, is_mono):
        # Now there are 8 parameters besides self.
        if img is None:
            print("Processing failed.")
            return
        out_filename = self.generate_output_filename(input_path, outdir, scale, self.outtype_combo.currentText())
        try:
            save_image(img, out_filename, original_format, bit_depth=bit_depth,
                       original_header=original_header, is_mono=is_mono)
            print(f"Saved upscaled image to {out_filename}")
            QMessageBox.information(self, "Processing Complete",
                                    f"Saved upscaled image to:\n{out_filename}")
        except Exception as e:
            print(f"Error saving image: {e}")
            QMessageBox.critical(self, "Error", f"Error saving image:\n{e}")

##########################################
# Main Function & Command-Line Interface
##########################################
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path("upscale.png")))
    parser = argparse.ArgumentParser(description="Cosmic Clarity Super-Resolution Upscaling Tool")
    parser.add_argument("--input", type=str, help="Path to input image")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--scale", type=int, choices=[2,3,4], help="Upscale factor: 2, 3, or 4")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory containing superres model files")
    args = parser.parse_args()
    try:
        model, device, use_pytorch = load_superres_model(args.scale if args.scale else 2, args.model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    print(f"Using device: {device}")
    if args.input and args.output_dir:
        # Headless mode...
        result = process_image(args.input, args.scale if args.scale else 2, model, device, use_pytorch,
                                progress_callback=lambda cur, tot: print(f"PROGRESS: {int((cur/tot)*100)}%", flush=True)
)
        if result[0] is not None:
            final_img, orig_hdr, bd, orig_fmt, mono = result
            base = os.path.splitext(os.path.basename(args.input))[0]
            suffix = f"_upscaled{int(args.scale)}x"
            out_type = "tif"  # explicitly TIFF format
            output_filename = os.path.join(os.path.abspath(args.output_dir), base + suffix + "." + out_type)
            try:
                save_image(final_img, output_filename, "tiff", bit_depth="32-bit floating point", original_header=orig_hdr, is_mono=mono)
                print(f"\nSaved upscaled image to {output_filename}")
            except Exception as e:
                print(f"Error saving image: {e}")
        sys.exit(0)
    else:
        window = UpscalingApp(model, device, use_pytorch)
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
