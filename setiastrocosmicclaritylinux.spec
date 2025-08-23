# -*- mode: python ; coding: utf-8 -*-

import os
import glob
from PyInstaller.utils.hooks import (
    collect_dynamic_libs,
    collect_submodules,
    collect_data_files,
)

# ---------- Runtime libs to collect ----------
# PyTorch + friends
torch_bins  = collect_dynamic_libs('torch')
torch_data  = collect_data_files('torch')
tv_bins     = collect_dynamic_libs('torchvision')
tv_data     = collect_data_files('torchvision')
ta_bins     = collect_dynamic_libs('torchaudio')
ta_data     = collect_data_files('torchaudio')

# ONNX Runtime (handles both onnxruntime & onnxruntime-gpu)
try:
    ort_bins  = collect_dynamic_libs('onnxruntime')
    ort_data  = collect_data_files('onnxruntime')
except Exception:
    ort_bins, ort_data = [], []

# OpenCV (some wheels need these .so’s when frozen)
try:
    cv2_bins = collect_dynamic_libs('cv2')
    cv2_data = collect_data_files('cv2')
except Exception:
    cv2_bins, cv2_data = [], []

# imagecodecs for TIFF LZW/Deflate/etc
try:
    ic_bins  = collect_dynamic_libs('imagecodecs')
    ic_data  = collect_data_files('imagecodecs')
except Exception:
    ic_bins, ic_data = [], []

# (Optional) low-level libs used by your code; usually auto, but safe to include
try:
    lz4_bins = collect_dynamic_libs('lz4')
    lz4_data = collect_data_files('lz4')
except Exception:
    lz4_bins, lz4_data = [], []
try:
    zstd_bins = collect_dynamic_libs('zstandard')
    zstd_data = collect_data_files('zstandard')
except Exception:
    zstd_bins, zstd_data = [], []

# Hidden imports so lazy loaders don’t break at runtime
hidden = (
    collect_submodules('torch') +
    collect_submodules('torchvision') +
    collect_submodules('torchaudio') +
    collect_submodules('onnxruntime') +
    collect_submodules('cv2') +
    collect_submodules('imagecodecs') +
    collect_submodules('lz4') +
    collect_submodules('zstandard')
)

# ---------- Your data / models ----------
datas_extra = [('xisf.py', '.')]  # your XISF helper

# Add your model files here (edit list to match your filenames)
MODEL_FILES = [
    # examples — replace with your actual sharpen model files
    'deep_sharpen_cnn_AI3_6.pth',
    'deep_sharpen_cnn_AI3_6.onnx',
    # if you share configs, add them too:
    # 'sharpen_config.json',
]
for pat in MODEL_FILES:
    for f in glob.glob(pat):
        datas_extra.append((f, '.'))

# ---------- Build graph ----------
all_bins  = torch_bins + tv_bins + ta_bins + ort_bins + cv2_bins + ic_bins + lz4_bins + zstd_bins
all_datas = torch_data + tv_data + ta_data + ort_data + cv2_data + ic_data + lz4_data + zstd_data + datas_extra

a = Analysis(
    ['setiastrocosmicclaritylinux.py'],
    pathex=[],
    binaries=all_bins,
    datas=all_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],   # optionally add a runtime hook; see note below
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SetiAstroCosmicClarity',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # <<< IMPORTANT: don't UPX CUDA/torch/.so’s
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['sharpen.png'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,  # <<< keep false
    upx_exclude=['libcud*.so*','libtorch*.so*','libnv*.so*','libonnx*.so*','libopencv_*.so*','libimagecodecs*.so*'],
    name='SetiAstroCosmicClarity',
)
