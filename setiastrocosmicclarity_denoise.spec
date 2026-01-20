# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_dynamic_libs

# Collect ONNX Runtime provider libraries
onnx_binaries = collect_dynamic_libs('onnxruntime')

a = Analysis(
    ['setiastrocosmicclarity_denoise.py'],
    pathex=[],
    binaries=onnx_binaries,  # Add the ONNX binaries here
    datas=[('xisf.py', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='setiastrocosmicclarity_denoise',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['cosmicclaritydenoise.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='setiastrocosmicclarity_denoise',
)
