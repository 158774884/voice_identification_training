"""
Application-wide constants and configuration.
"""
import os
import sys

# App metadata
APP_NAME = "语音模型训练与芯片适配工具"
APP_VERSION = "1.0.0"
APP_ORG = "VoiceModelTool"
APP_DESCRIPTION = "离线语音识别模型定制训练、芯片适配评估、模型验证、固件导出与SDK移植一体化工具"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHIP_DB_DIR = os.path.join(BASE_DIR, "chip_db")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data")
STYLES_DIR = os.path.join(BASE_DIR, "app", "ui", "styles")

# Project file extension
PROJECT_EXTENSION = ".vproj"
PROJECT_FILTER = f"Voice Model Project (*{PROJECT_EXTENSION})"

# Window defaults
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 800
MIN_WIDTH = 1024
MIN_HEIGHT = 680

# Panel names (tab labels)
TAB_DATASET = "📊 数据集管理"
TAB_TRAINING = "🧠 模型训练"
TAB_TEST = "🎤 语音测试"
TAB_CHIP = "🔬 芯片评估"
TAB_EXPORT = "📦 固件导出"

# Log categories
LOG_CATEGORIES = ["系统", "数据集", "训练", "测试", "芯片", "导出", "错误"]

# Training presets
TRAINING_PRESETS = ["tiny", "standard", "large", "custom"]

# Supported audio formats
AUDIO_FORMATS = ["*.wav", "*.flac", "*.mp3", "*.ogg"]

# Supported export formats
EXPORT_FORMATS = ["ONNX", "C Header", "RKNN", "Custom"]

# Quantization methods
QUANT_METHODS = ["INT8 (静态)", "INT16", "动态 INT8", "FP32 (无量化)"]
