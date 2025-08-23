# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import (
    collect_dynamic_libs,
    collect_submodules,
    collect_data_files,
)

# --- Collect GPU runtime bits ---
torch_bins  = collect_dynamic_libs('torch')          # libtorch_cuda.so, libc10.so, cudnn, cublas, etc.
torch_data  = collect_data_files('torch')
tv_bins     = collect_dynamic_libs('torchvision')
tv_data     = collect_data_files('torchvision')
ta_bins     = collect_dynamic_libs('torchaudio')
ta_data     = collect_data_files('torchaudio')

# ONNX Runtime (GPU or CPU)
try:
    ort_bins  = collect_dynamic_libs('onnxruntime')  # pulls CUDA EP .so if onnxruntime-gpu is installed
    ort_data  = collect_data_files('onnxruntime')
except Exception:
    ort_bins, ort_data = [], []

# OpenCV often needs its .so’s explicitly when frozen (depends on distro/wheel)
try:
    cv2_bins = collect_dynamic_libs('cv2')
    cv2_data = collect_data_files('cv2')
except Exception:
    cv2_bins, cv2_data = [], []

# Hiddenimports so lazy loaders don’t break
hidden = (
    collect_submodules('torch') +
    collect_submodules('torchvision') +
    collect_submodules('torchaudio') +
    collect_submodules('onnxruntime')
)

# Include your model files next to the exe
# adjust paths if you keep them elsewhere
datas_extra = []
for fname in [
    'deep_denoise_cnn_AI3_6.pth',
    'deep_denoise_cnn_AI3_6.onnx',
]:
    if os.path.exists(fname):
        datas_extra.append((fname, '.'))

a = Analysis(
    ['SetiAstroCosmicClarity_denoise.py'],
    pathex=[],
    binaries=torch_bins + tv_bins + ta_bins + ort_bins + cv2_bins,
    datas=torch_data + tv_data + ta_data + ort_data + cv2_data + datas_extra,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],      # (optional) add a runtime hook below if you want env vars set
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
    name='SetiAstroCosmicClarity_denoise',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,             # <<<< DO NOT UPX CUDA / torch libs
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,             # <<<< keep false
    upx_exclude=['libcud*.so*','libtorch*.so*','libnv*.so*','libonnx*.so*','libopencv_*.so*'],
    name='SetiAstroCosmicClarity_denoise',
)
