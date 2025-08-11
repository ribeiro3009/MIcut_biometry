# -*- mode: python ; coding: utf-8 -%-

a = Analysis(
    ['src\u005cpipeline.py'],
    pathex=['src'],
    binaries=[],
    datas=[('bin', 'bin'), ('src', 'src')],
    hiddenimports=[
        'numpy._testing',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'scipy.special.cython_special',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow',
        'tensorboard',
        'keras',
        'lib2to3',
        'tkinter',
        'PyQt5',
        'fix_imports',
        'torchaudio',
        'soundfile',
        'librosa',
        'torch.utils.tensorboard',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='pipeline',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pipeline',
)
