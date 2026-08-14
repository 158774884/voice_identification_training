# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for VoiceModelTool — single-file Windows .exe
Usage:
    pyinstaller VoiceModelTool.spec
"""

import sys
from pathlib import Path

BASE = Path('.').resolve()

a = Analysis(
    ['app/main.py'],
    pathex=[str(BASE)],
    binaries=[],
    datas=[
        # QSS stylesheet
        ('app/ui/styles/default.qss', 'app/ui/styles'),
        # Chip database init SQL
        ('chip_db/chip_db_init.sql', 'chip_db'),
        # 两阶段 KWS 导出所需模型 + 语法文件
        ('checkpoints/stage1/final_model.pt', 'checkpoints/stage1'),
        ('checkpoints/stage2/final_model.pt', 'checkpoints/stage2'),
        ('checkpoints/stage2/grammar.json', 'checkpoints/stage2'),
        # 移植 demo 模板
        ('rtl8713e_deploy/two_stage_kws/demo/kws_pipeline.c', 'rtl8713e_deploy/two_stage_kws/demo'),
        ('rtl8713e_deploy/two_stage_kws/demo/main_demo_ac7916.c', 'rtl8713e_deploy/two_stage_kws/demo'),
        ('rtl8713e_deploy/two_stage_kws/demo/main_demo_generic.c', 'rtl8713e_deploy/two_stage_kws/demo'),
        ('rtl8713e_deploy/two_stage_kws/demo/README.md', 'rtl8713e_deploy/two_stage_kws/demo'),
    ],
    hiddenimports=[
        # Project internals
        'app', 'app.main', 'app.app_config',
        'app.models', 'app.models.project', 'app.models.dataset_model', 'app.models.chip_database',
        'app.controllers', 'app.controllers.dataset_controller', 'app.controllers.test_controller',
        'app.controllers.training_controller', 'app.controllers.chip_controller', 'app.controllers.export_controller',
        'app.workers', 'app.workers.mic_worker', 'app.workers.inference_worker', 'app.workers.training_worker',
        'app.ui', 'app.ui.main_window',
        'app.ui.panels', 'app.ui.panels.log_panel', 'app.ui.panels.dataset_panel',
        'app.ui.panels.test_panel', 'app.ui.panels.training_panel',
        'app.ui.panels.chip_panel', 'app.ui.panels.export_panel', 'app.ui.panels.model_version_panel',
        'app.ui.dialogs', 'app.ui.dialogs.project_dialog', 'app.ui.dialogs.hyperparam_dialog',
        'app.ui.dialogs.chip_editor',
        'app.ui.widgets', 'app.ui.widgets.audio_waveform', 'app.ui.widgets.chart_widget',
        'app.utils', 'app.utils.logger', 'app.utils.report_generator', 'app.utils.datasheet_parser',

        # Existing project modules (may be imported by workers)
        'model', 'model.multi_task_model', 'model.shared_backbone',
        'model.asr_branch', 'model.dialect_branch', 'model.speaker_branch',
        'data', 'data.preprocessing', 'data.augmentation', 'data.dataset', 'data.vocab',
        'training', 'training.trainer', 'training.losses', 'training.config',
        'inference', 'inference.pipeline', 'inference.asr_inference',
        'inference.dialect_inference', 'inference.speaker_inference',
        'deployment', 'deployment.export_onnx', 'deployment.quantization',
        'utils', 'utils.metrics',

        # KWS modules
        'rtl8713e_deploy', 'rtl8713e_deploy.two_stage_kws',
        'rtl8713e_deploy.two_stage_kws.export_ac7916',
        'rtl8713e_deploy.two_stage_kws.stage1_wakeword',
        'rtl8713e_deploy.two_stage_kws.stage2_command',
        'rtl8713e_deploy.two_stage_kws.wfst_decoder',

        # PySide6
        'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',

        # multiprocessing (needed for freeze_support + DataLoader workers)
        'multiprocessing',

        # Common ML/data
        'torch', 'numpy', 'scipy', 'soundfile', 'sounddevice',
        'matplotlib', 'matplotlib.backends.backend_qtagg',
        'pyqtgraph',

        # Datasheet PDF parsing (PyMuPDF)
        'fitz', 'pymupdf',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VoiceModelTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,              # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,                  # Add icon path here if available
)
