#!/usr/bin/env python3
"""
VoiceModelTool — Windows Desktop GUI Application Entry Point

离线语音识别模型定制训练、芯片适配评估、模型验证、固件导出与SDK移植一体化工具

Usage:
    python -m app.main
    python app/main.py
"""

import sys
import os

# Ensure the project root is on sys.path so existing modules can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap, QPainter, QColor

from app.app_config import APP_NAME, APP_VERSION, APP_ORG
from app.ui.main_window import MainWindow
from app.utils.logger import LogManager


def create_splash_screen() -> QSplashScreen:
    """Create a simple programmatic splash screen."""
    pixmap = QPixmap(480, 280)
    pixmap.fill(QColor("#ffffff"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # Title
    title_font = QFont("Microsoft YaHei", 22, QFont.Bold)
    painter.setFont(title_font)
    painter.setPen(QColor("#1a73e8"))
    painter.drawText(pixmap.rect().adjusted(0, 30, 0, -180),
                     Qt.AlignHCenter | Qt.AlignBottom, APP_NAME)

    # Subtitle
    sub_font = QFont("Microsoft YaHei", 11)
    painter.setFont(sub_font)
    painter.setPen(QColor("#5f6368"))
    painter.drawText(pixmap.rect().adjusted(0, 80, 0, -140),
                     Qt.AlignHCenter | Qt.AlignTop,
                     "离线语音识别 | 模型训练 | 芯片适配 | 固件导出")

    # Version
    ver_font = QFont("Microsoft YaHei", 10)
    painter.setFont(ver_font)
    painter.setPen(QColor("#9aa0a6"))
    painter.drawText(pixmap.rect().adjusted(0, 0, -20, -20),
                     Qt.AlignBottom | Qt.AlignRight, f"v{APP_VERSION}")

    # Loading text
    load_font = QFont("Microsoft YaHei", 10)
    painter.setFont(load_font)
    painter.setPen(QColor("#9aa0a6"))
    painter.drawText(pixmap.rect().adjusted(0, -60, 0, 0),
                     Qt.AlignHCenter | Qt.AlignBottom, "正在加载...")

    painter.end()

    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    return splash


def main():
    # High-DPI support
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_ORG)

    # Set default font
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # Splash screen
    splash = create_splash_screen()
    splash.show()
    app.processEvents()

    # Init log manager early
    log = LogManager()
    log.info("系统", f"正在启动 {APP_NAME} v{APP_VERSION}...")

    # Create and show main window
    window = MainWindow()

    # Close splash and show window after a short delay
    def show_main():
        splash.close()
        window.show()

    QTimer.singleShot(600, show_main)

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
