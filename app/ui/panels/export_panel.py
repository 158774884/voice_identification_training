"""
Firmware Export Panel — model quantization, ONNX/C export, SDK embedding.
"""
import os
import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QLineEdit, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, Slot

from app.controllers.export_controller import ExportController
from app.app_config import EXPORT_FORMATS, QUANT_METHODS
from app.utils.logger import LogManager


def _project_root():
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


PROJECT_ROOT = _project_root()

DEFAULT_STAGE1_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "stage1", "final_model.pt")
DEFAULT_STAGE2_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "stage2", "final_model.pt")
DEFAULT_GRAMMAR = os.path.join(PROJECT_ROOT, "checkpoints", "stage2", "grammar.json")


def _output_root():
    """导出输出目录基准：源码用项目根，打包用 exe 所在目录。"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return _project_root()


ONNX_OUTPUT_DIR = os.path.join(_output_root(), "导出ONNX")
FIRMWARE_OUTPUT_DIR = os.path.join(_output_root(), "导出C固件")


class ExportPanel(QWidget):
    """Firmware export and SDK embedding panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._controller = ExportController(self)
        self._log = LogManager()
        self._firmware_output_dir: str = ""
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # === Source ===
        source_group = QGroupBox("模型源")
        source_layout = QHBoxLayout(source_group)

        source_layout.addWidget(QLabel("模型文件:"))
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setPlaceholderText("选择训练好的模型检查点 (.pt)")
        source_layout.addWidget(self._model_path_edit)

        browse_model_btn = QPushButton("浏览...")
        browse_model_btn.setObjectName("secondaryBtn")
        source_layout.addWidget(browse_model_btn)

        source_layout.addWidget(QLabel("目标芯片:"))
        self._chip_combo = QComboBox()
        self._chip_combo.setMinimumWidth(130)
        self._chip_combo.addItems(["AC7916AB", "RTL8713E", "RK3588", "A311D", "Hi3559A", "ESP32-S3", "通用"])
        source_layout.addWidget(self._chip_combo)

        main_layout.addWidget(source_group)

        # === C 固件源 (两阶段 KWS) ===
        kws_group = QGroupBox("C 固件源（两阶段 KWS）")
        kws_layout = QVBoxLayout(kws_group)

        def _add_kws_row(label, default, edit_attr, filter_str):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            edit = QLineEdit()
            edit.setText(default)
            setattr(self, edit_attr, edit)
            row.addWidget(edit)
            btn = QPushButton("浏览...")
            btn.setObjectName("secondaryBtn")
            btn.clicked.connect(lambda _=False, e=edit, f=filter_str: self._browse_kws_file(e, f))
            row.addWidget(btn)
            kws_layout.addLayout(row)

        _add_kws_row("Stage1 模型:", DEFAULT_STAGE1_CKPT, "_stage1_edit", "Checkpoints (*.pt)")
        _add_kws_row("Stage2 模型:", DEFAULT_STAGE2_CKPT, "_stage2_edit", "Checkpoints (*.pt)")
        _add_kws_row("语法文件:", DEFAULT_GRAMMAR, "_grammar_edit", "Grammar (*.json)")
        main_layout.addWidget(kws_group)

        # === Export Configuration ===
        config_group = QGroupBox("导出配置")
        config_layout = QVBoxLayout(config_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("导出格式:"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(EXPORT_FORMATS)
        row1.addWidget(self._format_combo)

        row1.addWidget(QLabel("量化方法:"))
        self._quant_combo = QComboBox()
        self._quant_combo.addItems(QUANT_METHODS)
        row1.addWidget(self._quant_combo)

        row1.addWidget(QLabel("导出对象:"))
        self._onnx_target_combo = QComboBox()
        self._onnx_target_combo.addItems(["多任务模型 (full)", "两阶段 KWS (stage1+stage2)"])
        row1.addWidget(self._onnx_target_combo)

        row1.addStretch()
        config_layout.addLayout(row1)

        # Export buttons
        btn_layout = QHBoxLayout()
        self._export_onnx_btn = QPushButton("📦 导出 ONNX")
        self._export_onnx_btn.setObjectName("successBtn")
        btn_layout.addWidget(self._export_onnx_btn)

        self._export_quant_btn = QPushButton("⚡ 量化模型")
        self._export_quant_btn.setObjectName("secondaryBtn")
        btn_layout.addWidget(self._export_quant_btn)

        self._export_firmware_btn = QPushButton("🔧 导出 C 固件")
        btn_layout.addWidget(self._export_firmware_btn)

        self._embed_sdk_btn = QPushButton("📂 嵌入 SDK Demo")
        self._embed_sdk_btn.setObjectName("secondaryBtn")
        btn_layout.addWidget(self._embed_sdk_btn)

        btn_layout.addStretch()
        config_layout.addLayout(btn_layout)

        main_layout.addWidget(config_group)

        # === Progress ===
        progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout(progress_group)

        self._phase_label = QLabel("就绪")
        progress_layout.addWidget(self._phase_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        progress_layout.addWidget(self._progress_bar)

        main_layout.addWidget(progress_group)

        # === Export Log ===
        log_group = QGroupBox("导出日志")
        log_layout = QVBoxLayout(log_group)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(200)
        log_layout.addWidget(self._log_view)

        main_layout.addWidget(log_group)

        # === Connections ===
        browse_model_btn.clicked.connect(self._browse_model)

        self._export_onnx_btn.clicked.connect(self._on_export_onnx)
        self._export_quant_btn.clicked.connect(self._on_export_quant)
        self._export_firmware_btn.clicked.connect(self._on_export_firmware)
        self._embed_sdk_btn.clicked.connect(self._on_embed_sdk)

    def _connect_signals(self):
        self._controller.export_progress.connect(self._on_progress)
        self._controller.export_complete.connect(self._on_complete)
        self._controller.export_error.connect(self._on_error)

    # ================================================================
    # Slots
    # ================================================================

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "Model Files (*.pt *.pth *.onnx);;All Files (*.*)"
        )
        if path:
            self._model_path_edit.setText(path)

    def _browse_kws_file(self, edit, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filter_str)
        if path:
            edit.setText(path)

    @Slot()
    def _on_export_onnx(self):
        target = self._onnx_target_combo.currentText()
        self._progress_bar.setVisible(True)
        self._log_view.clear()
        if "两阶段" in target:
            stage1 = self._stage1_edit.text().strip()
            stage2 = self._stage2_edit.text().strip()
            if not (stage1 and stage2):
                QMessageBox.warning(self, "缺少源文件", "请填写 Stage1 / Stage2 模型路径")
                self._progress_bar.setVisible(False)
                return
            self._controller.export_kws_onnx(stage1, stage2, ONNX_OUTPUT_DIR)
        else:
            model = self._model_path_edit.text()
            if not model:
                QMessageBox.warning(self, "缺少模型", "请先选择模型文件")
                self._progress_bar.setVisible(False)
                return
            self._controller.export_onnx(model, ONNX_OUTPUT_DIR,
                                         export_mode="all")

    @Slot()
    def _on_export_quant(self):
        model = self._model_path_edit.text()
        if not model:
            QMessageBox.warning(self, "缺少模型", "请先选择模型文件")
            return
        quant = self._quant_combo.currentText().split(" ")[0]  # "INT8", etc.
        self._progress_bar.setVisible(True)
        self._log_view.clear()
        self._controller.export_quantized(model, ONNX_OUTPUT_DIR,
                                          quant_method=quant)

    @Slot()
    def _on_export_firmware(self):
        stage1 = self._stage1_edit.text().strip()
        stage2 = self._stage2_edit.text().strip()
        grammar = self._grammar_edit.text().strip()
        if not (stage1 and stage2 and grammar):
            QMessageBox.warning(self, "缺少源文件", "请填写 Stage1 / Stage2 模型和语法文件路径")
            return
        chip = self._chip_combo.currentText()
        self._progress_bar.setVisible(True)
        self._log_view.clear()
        self._controller.export_c_firmware(stage1, stage2, grammar,
                                           FIRMWARE_OUTPUT_DIR,
                                           chip_name=chip)

    @Slot()
    def _on_embed_sdk(self):
        if not self._firmware_output_dir:
            QMessageBox.warning(self, "提示", "请先导出固件")
            return
        sdk_dir = QFileDialog.getExistingDirectory(self, "选择 SDK 目录")
        if sdk_dir:
            ok = self._controller.embed_sdk_demo(self._firmware_output_dir, sdk_dir)
            if ok:
                self._log_view.append(f"SDK Demo 已嵌入: {sdk_dir}")
                QMessageBox.information(self, "完成", f"固件已嵌入 SDK 工程:\n{sdk_dir}")

    @Slot(int, int, str)
    def _on_progress(self, current: int, total: int, phase: str):
        self._phase_label.setText(phase)
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)

    @Slot(str, dict)
    def _on_complete(self, output_path: str, summary: dict):
        self._progress_bar.setVisible(False)
        self._phase_label.setText("完成!")
        self._firmware_output_dir = output_path
        fmt = summary.get("format", "")
        self._log_view.append(f"✅ 导出成功: {output_path}")
        self._log_view.append(f"   格式: {fmt}")
        self._log.info("导出", f"导出成功: {output_path} ({fmt})")

    @Slot(str)
    def _on_error(self, error: str):
        self._progress_bar.setVisible(False)
        self._phase_label.setText("错误")
        self._log_view.append(f"❌ {error}")
        self._log.error("导出", error)

    @property
    def controller(self) -> ExportController:
        return self._controller
