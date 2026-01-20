# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_dynamic_libs

# Collect dynamic libraries for torch and onnxruntime
torch_binaries = collect_dynamic_libs('torch')
onnx_binaries = collect_dynamic_libs('onnxruntime')
# Combine them if needed
binaries = torch_binaries + onnx_binaries

a = Analysis(
    ['setiastrocosmicclarity_darkstar.py'],
    pathex=[],
    binaries=binaries,  # include torch's and onnxruntime's dynamic libraries
    datas=[],
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
    name='setiastrocosmicclarity_darkstar',
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
    icon=['darkstar.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='setiastrocosmicclarity_darkstar',
)
