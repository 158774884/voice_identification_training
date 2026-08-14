"""
Training Panel — visual training config and real-time monitoring.
Model version management is now in the bottom ModelVersionPanel.
"""
import os
import re
from datetime import datetime
from typing import Dict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, Slot

from app.controllers.training_controller import TrainingController
from app.ui.dialogs.hyperparam_dialog import HyperparamDialog
from app.ui.widgets.chart_widget import TrainingChartWidget
from app.app_config import CHECKPOINTS_DIR
from app.utils.logger import LogManager


class TrainingPanel(QWidget):
    """Complete training management panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._controller = TrainingController(self)
        self._log = LogManager()
        self._current_config: Dict = {}
        self._data_root: str = ""
        self._current_phase: str = ""
        # 当前项目上下文：用于把训练出的模型保存到项目目录，并用项目名+日期命名
        self._project_name: str = ""
        self._project_dir: str = ""
        # 训练监控状态 (用于 Loss / LR / Accuracy 显示)
        self._current_step = 0
        self._chart_step = 0          # 图表横坐标 (单调递增，跨 epoch 不重置)
        self._last_loss = 0.0
        self._current_lr = 0.0
        self._current_accuracy = 0.0
        # Callbacks for notifying main window of state changes
        self._toolbar_start_cb = None
        self._toolbar_stop_cb = None
        self._toolbar_finish_cb = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # === Top: Config Section ===
        config_group = QGroupBox("训练配置")
        config_layout = QVBoxLayout(config_group)

        # Preset + key params
        params_layout = QHBoxLayout()

        params_layout.addWidget(QLabel("预设:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(["standard", "tiny", "large", "custom"])
        params_layout.addWidget(self._preset_combo)

        params_layout.addWidget(QLabel("Epochs:"))
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 500)
        self._epochs_spin.setValue(100)
        params_layout.addWidget(self._epochs_spin)

        params_layout.addWidget(QLabel("Batch:"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 256)
        self._batch_spin.setValue(32)
        params_layout.addWidget(self._batch_spin)

        params_layout.addWidget(QLabel("LR:"))
        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(1e-6, 1e-1)
        self._lr_spin.setDecimals(6)
        self._lr_spin.setSingleStep(1e-4)
        self._lr_spin.setValue(1e-3)
        params_layout.addWidget(self._lr_spin)

        params_layout.addWidget(QLabel("数据根目录:"))
        self._data_dir_label = QLabel("未选择")
        self._data_dir_label.setStyleSheet("color: #9aa0a6;")
        params_layout.addWidget(self._data_dir_label)

        browse_data_btn = QPushButton("浏览...")
        browse_data_btn.setObjectName("secondaryBtn")
        params_layout.addWidget(browse_data_btn)

        params_layout.addStretch()
        config_layout.addLayout(params_layout)

        # Advanced + control buttons
        ctrl_layout = QHBoxLayout()
        self._advanced_btn = QPushButton("⚙ 高级配置")
        self._advanced_btn.setObjectName("secondaryBtn")
        ctrl_layout.addWidget(self._advanced_btn)

        self._start_btn = QPushButton("▶ 开始训练")
        self._start_btn.setObjectName("successBtn")
        ctrl_layout.addWidget(self._start_btn)

        self._pause_btn = QPushButton("⏸ 暂停")
        self._pause_btn.setEnabled(False)
        ctrl_layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.setObjectName("dangerBtn")
        self._stop_btn.setEnabled(False)
        ctrl_layout.addWidget(self._stop_btn)

        ctrl_layout.addStretch()
        config_layout.addLayout(ctrl_layout)

        main_layout.addWidget(config_group)

        # === Middle: Training Chart + Progress ===
        monitor_group = QGroupBox("训练监控")
        monitor_layout = QVBoxLayout(monitor_group)

        self._chart = TrainingChartWidget(self)
        monitor_layout.addWidget(self._chart)

        progress_layout = QHBoxLayout()
        self._epoch_progress = QProgressBar()
        progress_layout.addWidget(QLabel("进度:"))
        progress_layout.addWidget(self._epoch_progress)

        self._eta_label = QLabel("预计剩余: --")
        self._eta_label.setStyleSheet("color: #5f6368;")
        progress_layout.addWidget(self._eta_label)
        progress_layout.addStretch()

        self._current_loss_label = QLabel("Loss: --")
        progress_layout.addWidget(self._current_loss_label)
        self._current_lr_label = QLabel("LR: --")
        progress_layout.addWidget(self._current_lr_label)

        monitor_layout.addLayout(progress_layout)

        main_layout.addWidget(monitor_group)

        # === Connections ===
        self._advanced_btn.clicked.connect(self._on_advanced_config)
        self._start_btn.clicked.connect(self._on_start_training)
        self._pause_btn.clicked.connect(self._on_pause_training)
        self._stop_btn.clicked.connect(self._on_stop_training)
        browse_data_btn.clicked.connect(self._on_browse_data)

    def _connect_signals(self):
        self._controller.progress_update.connect(self._on_progress)
        self._controller.loss_update.connect(self._on_loss)
        self._controller.lr_update.connect(self._on_lr)
        self._controller.accuracy_update.connect(self._on_accuracy)
        self._controller.training_started.connect(self._on_training_started)
        self._controller.training_paused.connect(self._on_training_paused)
        self._controller.training_resumed.connect(self._on_training_resumed)
        self._controller.training_finished.connect(self._on_training_finished)
        self._controller.training_stopped.connect(self._on_training_stopped)
        self._controller.training_error.connect(self._on_training_error)
        self._controller.checkpoint_saved.connect(self._on_checkpoint)
        self._controller.phase_changed.connect(self._on_phase_changed)

    # ================================================================
    # Slots
    # ================================================================

    @Slot()
    def _on_advanced_config(self):
        dialog = HyperparamDialog(self._current_config, self)
        if dialog.exec() == HyperparamDialog.Accepted:
            self._current_config = dialog.get_config()
            preset = self._current_config.get("preset", "custom")
            self._preset_combo.setCurrentText(preset)
            self._epochs_spin.setValue(self._current_config.get("num_epochs", 100))
            self._batch_spin.setValue(self._current_config.get("batch_size", 32))
            lr = self._current_config.get("learning_rate", 1e-3)
            self._lr_spin.setValue(lr)
            self._log.info("训练", "高级配置已更新")

    @Slot()
    def _on_start_training(self):
        try:
            self._do_start_training()
        except Exception as e:
            self._log.error("训练", f"启动训练失败: {e}")
            QMessageBox.critical(self, "训练启动失败", str(e))

    def _do_start_training(self):
        if not self._data_root:
            QMessageBox.warning(self, "缺少数据", "请先在训练配置中选择数据根目录（需包含 train.jsonl）")
            return

        # Verify the data root contains a train.jsonl
        train_jsonl = os.path.join(self._data_root, 'train.jsonl')
        alt_jsonl = os.path.join(self._data_root, 'cmd_data', 'train.jsonl')
        if not os.path.exists(train_jsonl) and not os.path.exists(alt_jsonl):
            QMessageBox.warning(self, '缺少训练数据',
                f'数据根目录下未找到 train.jsonl 文件。\n\n'
                f'请先在 [数据集管理] 面板中：\n'
                f'1. 导入音频文件夹\n'
                f'2. 点击 [导出 JSONL]\n'
                f'3. 将导出的 train.jsonl 放到数据根目录\n\n'
                f'查找路径:\n{train_jsonl}\n{alt_jsonl}')
            return

        self._log.info("训练", f"数据根目录: {self._data_root}")

        # Build config from current values
        config = dict(self._current_config) if self._current_config else {}
        config["preset"] = self._preset_combo.currentText()
        config["num_epochs"] = self._epochs_spin.value()
        config["batch_size"] = self._batch_spin.value()
        config["learning_rate"] = self._lr_spin.value()
        config["device"] = "cpu"

        self._current_config = config
        self._chart.clear()
        self._chart_step = 0
        self._epoch_progress.setValue(0)
        self._eta_label.setText("正在准备...")

        # 模型保存位置与命名：优先放在当前项目目录下，文件名加「项目名_日期」尾缀
        date_str = datetime.now().strftime("%Y%m%d")
        safe_name = self._safe_filename_part(self._project_name)
        config["model_name_suffix"] = f"_{safe_name}_{date_str}"

        if self._project_dir:
            models_root = os.path.join(self._project_dir, "models")
            checkpoint_dir = os.path.join(
                models_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
        else:
            checkpoint_dir = os.path.join(
                CHECKPOINTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))

        self._controller.start_training(config, self._data_root, checkpoint_dir)

    @Slot()
    def _on_pause_training(self):
        if self._controller.is_paused:
            self._controller.resume_training()
        else:
            self._controller.pause_training()

    @Slot()
    def _on_stop_training(self):
        reply = QMessageBox.question(
            self, "确认停止", "确定要停止训练吗？已训练的检查点将保留。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._controller.stop_training()

    @Slot()
    def _on_browse_data(self):
        directory = QFileDialog.getExistingDirectory(self, "选择数据根目录")
        if directory:
            self._data_root = directory
            self._data_dir_label.setText(directory)

    @Slot(int, int, int)
    def _on_progress(self, epoch: int, step: int, total: int):
        self._current_step = step
        self._epoch_progress.setMaximum(total)
        self._epoch_progress.setValue(step)
        phase = getattr(self, '_current_phase', '')
        phase_str = f"[{phase}] " if phase else ""
        if step > 0:
            pct = step / total * 100
            self._eta_label.setText(f"{phase_str}Epoch {epoch} | {pct:.0f}%")

    @Slot(str)
    def _on_phase_changed(self, phase: str):
        self._current_phase = phase
        self._log.info("训练", f"进入训练阶段: {phase}")

    @Slot(float, float, float, float)
    def _on_loss(self, total: float, asr: float, dialect: float, speaker: float):
        self._last_loss = total
        self._update_metric_label()
        self._chart.add_step(self._chart_step, total, self._current_accuracy,
                             asr, dialect, speaker, self._current_lr)
        self._chart_step += 1

    @Slot(float)
    def _on_accuracy(self, acc: float):
        self._current_accuracy = acc
        self._update_metric_label()

    @Slot(float)
    def _on_lr(self, lr: float):
        self._current_lr = lr
        self._current_lr_label.setText(f"LR: {lr:.2e}")

    def _update_metric_label(self):
        text = f"Loss: {self._last_loss:.4f}"
        if self._current_accuracy > 0:
            text += f" | Acc: {self._current_accuracy:.1f}%"
        self._current_loss_label.setText(text)

    @Slot()
    def _on_training_started(self):
        self._log.info("训练", "[面板] training_started 信号收到")
        self._start_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._pause_btn.setText("⏸ 暂停")
        if self._toolbar_start_cb:
            self._toolbar_start_cb()

    @Slot()
    def _on_training_paused(self):
        self._pause_btn.setText("▶ 继续")

    @Slot()
    def _on_training_resumed(self):
        self._pause_btn.setText("⏸ 暂停")

    @Slot(dict)
    def _on_training_finished(self, summary: dict):
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        # 完成：进度条整条置满并停止滚动
        self._epoch_progress.setRange(0, 100)
        self._epoch_progress.setValue(100)
        self._eta_label.setText("训练完成!")
        if self._toolbar_finish_cb:
            self._toolbar_finish_cb(summary)
        QMessageBox.information(self, "训练完成", "模型训练已完成!")

    @Slot()
    def _on_training_stopped(self):
        """训练被用户手动停止."""
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._eta_label.setText("已停止")
        self._log.info("训练", "训练已停止")
        if self._toolbar_stop_cb:
            self._toolbar_stop_cb()

    @Slot(str)
    def _on_training_error(self, error: str):
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        if self._toolbar_stop_cb:
            self._toolbar_stop_cb()
        QMessageBox.critical(self, "训练错误", error)

    @Slot(str)
    def _on_checkpoint(self, path: str):
        self._log.info("训练", f"检查点已保存: {os.path.basename(path)}")

    # ================================================================
    # Public API
    # ================================================================

    def set_project_context(self, project_name: str, project_dir: str):
        """Set the current project name/dir so trained models are saved beside it."""
        self._project_name = project_name or ""
        self._project_dir = project_dir or ""

    @staticmethod
    def _safe_filename_part(name: str) -> str:
        """Sanitize a string for use inside a filename."""
        return re.sub(r'[\\/:*?"<>|\s]+', '_', name or '').strip('_') or 'model'

    @property
    def controller(self) -> TrainingController:
        return self._controller
