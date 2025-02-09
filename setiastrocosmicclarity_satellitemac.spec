# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['setiastrocosmicclarity_satellite'],
    pathex=[],
    binaries=[],
    datas=[
        ('xisf.py', '.'),
        ('satellite_trail_detector.pth', '.'),
        ('satelliteremoval128featuremaps.pth', '.'),
        ('satellite.png', '.'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='setiastrocosmicclarity_satellite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='satellite.icns',  # Add the path to your icon file here
)
