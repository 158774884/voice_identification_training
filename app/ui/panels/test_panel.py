"""
Voice Test Panel — real-time mic testing, batch testing, and model comparison.
"""
import os
import numpy as np
from typing import List, Dict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QTextEdit, QComboBox, QProgressBar, QTableView,
    QHeaderView, QAbstractItemView, QFileDialog, QMessageBox,
    QSplitter, QListWidget, QListWidgetItem,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

from app.controllers.test_controller import TestController
from app.ui.widgets.audio_waveform import AudioWaveformWidget
from app.utils.report_generator import generate_batch_test_report
from app.utils.logger import LogManager


class TestPanel(QWidget):
    """Voice recognition test and comparison panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._controller = TestController(self)
        self._log = LogManager()
        self._batch_results: List[Dict] = []
        self._recording_path: str | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # === Top: Model Selection ===
        model_group = QGroupBox("模型选择")
        model_layout = QHBoxLayout(model_group)

        model_layout.addWidget(QLabel("当前模型:"))
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(250)
        self._model_combo.setPlaceholderText("选择已训练的模型...")
        model_layout.addWidget(self._model_combo)

        self._load_model_btn = QPushButton("加载模型")
        model_layout.addWidget(self._load_model_btn)

        browse_btn = QPushButton("浏览...")
        browse_btn.setObjectName("secondaryBtn")
        model_layout.addWidget(browse_btn)

        self._model_status_label = QLabel("未加载")
        self._model_status_label.setStyleSheet("color: #9aa0a6;")
        model_layout.addWidget(self._model_status_label)

        model_layout.addStretch()

        main_layout.addWidget(model_group)

        # === Middle: Real-time Test + Waveform ===
        splitter = QSplitter(Qt.Horizontal)

        # -- Left: Real-time test --
        rt_widget = QWidget()
        rt_layout = QVBoxLayout(rt_widget)

        rt_header = QHBoxLayout()
        rt_header.addWidget(QLabel("实时语音测试"))
        rt_header.addStretch()

        self._mic_btn = QPushButton("🎤 开始录音")
        self._mic_btn.setObjectName("successBtn")
        rt_header.addWidget(self._mic_btn)

        self._stop_mic_btn = QPushButton("⏹ 停止")
        self._stop_mic_btn.setObjectName("dangerBtn")
        self._stop_mic_btn.setEnabled(False)
        rt_header.addWidget(self._stop_mic_btn)

        rt_layout.addLayout(rt_header)

        # Level meter
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("音量:"))
        self._level_bar = QProgressBar()
        self._level_bar.setRange(0, 100)
        self._level_bar.setMaximumHeight(14)
        level_layout.addWidget(self._level_bar)
        rt_layout.addLayout(level_layout)

        # Transcript display
        rt_layout.addWidget(QLabel("实时识别结果:"))
        self._transcript_display = QTextEdit()
        self._transcript_display.setReadOnly(True)
        self._transcript_display.setMinimumHeight(120)
        self._transcript_display.setStyleSheet(
            "font-size: 16px; background-color: #1e1e1e; color: #d4d4d4; "
            "border-radius: 6px; padding: 10px;"
        )
        rt_layout.addWidget(self._transcript_display)

        # Dialect/confidence
        info_layout = QHBoxLayout()
        self._dialect_label = QLabel("方言: --")
        info_layout.addWidget(self._dialect_label)
        self._confidence_label = QLabel("置信度: --")
        info_layout.addWidget(self._confidence_label)
        info_layout.addStretch()
        rt_layout.addLayout(info_layout)

        # Save recording
        save_layout = QHBoxLayout()
        self._save_recording_btn = QPushButton("💾 保存录音")
        self._save_recording_btn.setObjectName("secondaryBtn")
        self._save_recording_btn.setEnabled(False)
        save_layout.addWidget(self._save_recording_btn)
        save_layout.addStretch()
        rt_layout.addLayout(save_layout)

        splitter.addWidget(rt_widget)

        # -- Right: Waveform + History --
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        right_layout.addWidget(QLabel("音频波形"))
        self._rt_waveform = AudioWaveformWidget(self)
        right_layout.addWidget(self._rt_waveform)

        right_layout.addWidget(QLabel("识别历史:"))
        self._history_list = QListWidget()
        right_layout.addWidget(self._history_list)

        splitter.addWidget(right_widget)
        splitter.setSizes([500, 400])
        main_layout.addWidget(splitter)

        # === Bottom: Batch Test ===
        batch_group = QGroupBox("批量测试")
        batch_layout = QVBoxLayout(batch_group)

        batch_ctrl = QHBoxLayout()
        batch_ctrl.addWidget(QLabel("测试文件夹:"))
        self._batch_dir_label = QLabel("未选择")
        self._batch_dir_label.setStyleSheet("color: #9aa0a6;")
        batch_ctrl.addWidget(self._batch_dir_label)
        select_dir_btn = QPushButton("选择文件夹")
        select_dir_btn.setObjectName("secondaryBtn")
        batch_ctrl.addWidget(select_dir_btn)
        self._batch_run_btn = QPushButton("▶ 开始批量测试")
        self._batch_run_btn.setEnabled(False)
        batch_ctrl.addWidget(self._batch_run_btn)
        self._export_report_btn = QPushButton("📄 导出报告")
        self._export_report_btn.setObjectName("secondaryBtn")
        self._export_report_btn.setEnabled(False)
        batch_ctrl.addWidget(self._export_report_btn)
        batch_ctrl.addStretch()
        batch_layout.addLayout(batch_ctrl)

        self._batch_progress = QProgressBar()
        self._batch_progress.setVisible(False)
        batch_layout.addWidget(self._batch_progress)

        batch_summary = QHBoxLayout()
        self._batch_total_label = QLabel("总计: 0")
        batch_summary.addWidget(self._batch_total_label)
        self._batch_accuracy_label = QLabel("准确率: --")
        batch_summary.addWidget(self._batch_accuracy_label)
        self._batch_latency_label = QLabel("平均延迟: --")
        batch_summary.addWidget(self._batch_latency_label)
        batch_summary.addStretch()
        batch_layout.addLayout(batch_summary)

        main_layout.addWidget(batch_group)

        # === Connections ===
        self._load_model_btn.clicked.connect(self._on_load_model)
        browse_btn.clicked.connect(self._on_browse_model)
        self._mic_btn.clicked.connect(self._on_start_mic)
        self._stop_mic_btn.clicked.connect(self._on_stop_mic)
        self._save_recording_btn.clicked.connect(self._on_save_recording)
        select_dir_btn.clicked.connect(self._on_select_batch_dir)
        self._batch_run_btn.clicked.connect(self._on_run_batch)
        self._export_report_btn.clicked.connect(self._on_export_report)

    def _connect_signals(self):
        self._controller.transcription_update.connect(self._on_transcription)
        self._controller.dialect_update.connect(self._on_dialect)
        self._controller.level_update.connect(self._on_level)
        self._controller.mic_status_changed.connect(self._on_mic_status)
        self._controller.model_loaded.connect(self._on_model_loaded)
        self._controller.model_error.connect(self._on_model_error)
        self._controller.batch_result.connect(self._on_batch_result)
        self._controller.batch_complete.connect(self._on_batch_complete)

    # ================================================================
    # Slots
    # ================================================================

    @Slot(str)
    def _on_transcription(self, text: str):
        self._transcript_display.append(text)
        self._history_list.insertItem(0, QListWidgetItem(text))

    @Slot(str, float)
    def _on_dialect(self, dialect: str, confidence: float):
        self._dialect_label.setText(f"方言: {dialect}")
        self._confidence_label.setText(f"置信度: {confidence:.1%}")

    @Slot(float)
    def _on_level(self, level: float):
        self._level_bar.setValue(int(level * 100))

    @Slot(bool)
    def _on_mic_status(self, recording: bool):
        self._mic_btn.setEnabled(not recording)
        self._stop_mic_btn.setEnabled(recording)
        self._save_recording_btn.setEnabled(not recording)

    @Slot(str)
    def _on_model_loaded(self, name: str):
        self._model_status_label.setText(f"✅ {name}")
        self._model_status_label.setStyleSheet("color: #28a745;")
        self._model_combo.addItem(name)

    @Slot(str)
    def _on_model_error(self, error: str):
        self._model_status_label.setText(f"❌ {error}")
        self._model_status_label.setStyleSheet("color: #dc3545;")
        QMessageBox.warning(self, "模型加载失败", error)

    @Slot()
    def _on_load_model(self):
        path = self._model_combo.currentText()
        if path:
            self._controller.load_model(path)

    @Slot()
    def _on_browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "Model Files (*.pt *.pth);;All Files (*.*)"
        )
        if path:
            self._model_combo.setCurrentText(path)
            self._controller.load_model(path)

    @Slot()
    def _on_start_mic(self):
        self._controller.start_mic_test()
        self._transcript_display.clear()
        self._history_list.clear()

    @Slot()
    def _on_stop_mic(self):
        self._controller.stop_mic_test()

    @Slot()
    def _on_save_recording(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存录音", "recording.wav",
            "WAV Files (*.wav);;All Files (*.*)"
        )
        if path:
            self._controller.save_recording(path)

    # Batch test
    @Slot()
    def _on_select_batch_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择测试文件夹")
        if directory:
            self._batch_test_dir = directory
            self._batch_dir_label.setText(directory)
            self._batch_run_btn.setEnabled(True)

    @Slot()
    def _on_run_batch(self):
        directory = getattr(self, '_batch_test_dir', None)
        if not directory:
            return

        # Collect audio files
        audio_files = []
        for root, dirs, filenames in os.walk(directory):
            for fn in filenames:
                if fn.lower().endswith(('.wav', '.flac', '.mp3')):
                    audio_files.append(os.path.join(root, fn))

        if not audio_files:
            QMessageBox.information(self, "提示", "所选文件夹中没有音频文件")
            return

        self._batch_results.clear()
        self._batch_progress.setVisible(True)
        self._batch_progress.setMaximum(len(audio_files))
        self._batch_run_btn.setEnabled(False)

        self._controller.run_batch_test(audio_files)

    @Slot(dict)
    def _on_batch_result(self, result: dict):
        self._batch_results.append(result)
        self._batch_progress.setValue(len(self._batch_results))

    @Slot(dict)
    def _on_batch_complete(self, summary: dict):
        self._batch_progress.setVisible(False)
        self._batch_run_btn.setEnabled(True)
        self._export_report_btn.setEnabled(True)

        total = summary.get("total", 0)
        accuracy = summary.get("accuracy", 0)
        avg_lat = summary.get("avg_latency_ms", 0)

        self._batch_total_label.setText(f"总计: {total}")
        self._batch_accuracy_label.setText(f"准确率: {accuracy:.1f}%")
        self._batch_latency_label.setText(f"平均延迟: {avg_lat:.0f}ms")

        self._log.info("测试", f"批量测试完成: {total}个文件, 准确率{accuracy:.1f}%")

    @Slot()
    def _on_export_report(self):
        if not self._batch_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出测试报告", "test_report.html",
            "HTML Files (*.html);;All Files (*.*)"
        )
        if path:
            generate_batch_test_report(
                self._batch_results, path,
                model_name=self._model_combo.currentText() or "当前模型",
            )
            self._log.info("测试", f"报告已导出: {path}")
            QMessageBox.information(self, "导出成功", f"报告已保存到:\n{path}")

    # ================================================================
    # Public API
    # ================================================================

    @property
    def controller(self) -> TestController:
        return self._controller
