"""
Hyperparameter configuration dialog — full TrainingConfig editor.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QDialogButtonBox,
    QScrollArea, QWidget, QTabWidget, QLabel,
)
from PySide6.QtCore import Qt

from app.app_config import TRAINING_PRESETS


class HyperparamDialog(QDialog):
    """Detailed hyperparameter configuration dialog."""

    def __init__(self, current_config: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("超参数详细配置")
        self.setMinimumSize(600, 500)
        self._config = current_config or {}
        self._setup_ui()
        if current_config:
            self._load_config(current_config)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # === Tab 1: Model Architecture ===
        arch_tab = QScrollArea()
        arch_tab.setWidgetResizable(True)
        arch_widget = QWidget()
        arch_vbox = QVBoxLayout(arch_widget)
        arch_vbox.setSpacing(12)

        # Architecture group
        arch_group = QGroupBox("骨干网络")
        arch_form = QFormLayout(arch_group)
        arch_form.setHorizontalSpacing(12)
        arch_form.setVerticalSpacing(10)

        self._preset_combo = QComboBox()
        self._preset_combo.addItems(TRAINING_PRESETS)
        arch_form.addRow("模型预设:", self._preset_combo)

        self._backbone_dim = QSpinBox()
        self._backbone_dim.setRange(32, 512)
        self._backbone_dim.setValue(256)
        arch_form.addRow("Backbone 维度:", self._backbone_dim)

        self._num_blocks = QSpinBox()
        self._num_blocks.setRange(1, 12)
        self._num_blocks.setValue(4)
        arch_form.addRow("Conformer Blocks:", self._num_blocks)

        self._vocab_size = QSpinBox()
        self._vocab_size.setRange(1000, 10000)
        self._vocab_size.setValue(6000)
        arch_form.addRow("词汇表大小:", self._vocab_size)

        self._embed_dim = QSpinBox()
        self._embed_dim.setRange(64, 512)
        self._embed_dim.setValue(256)
        arch_form.addRow("嵌入维度:", self._embed_dim)

        self._backbone_dropout = QDoubleSpinBox()
        self._backbone_dropout.setRange(0.0, 0.5)
        self._backbone_dropout.setSingleStep(0.05)
        self._backbone_dropout.setValue(0.1)
        arch_form.addRow("Dropout:", self._backbone_dropout)

        self._causal_cb = QCheckBox("启用流式推理 (Causal Conv)")
        self._causal_cb.setChecked(True)
        arch_form.addRow("", self._causal_cb)
        arch_vbox.addWidget(arch_group)
        arch_vbox.addStretch()

        arch_tab.setWidget(arch_widget)
        tabs.addTab(arch_tab, "模型架构")

        # === Tab 2: Training ===
        train_tab = QScrollArea()
        train_tab.setWidgetResizable(True)
        train_widget = QWidget()
        train_vbox = QVBoxLayout(train_widget)
        train_vbox.setSpacing(12)

        # Training params group
        train_group = QGroupBox("训练控制")
        train_form = QFormLayout(train_group)
        train_form.setHorizontalSpacing(12)
        train_form.setVerticalSpacing(10)

        self._num_epochs = QSpinBox()
        self._num_epochs.setRange(1, 500)
        self._num_epochs.setValue(100)
        train_form.addRow("训练轮数:", self._num_epochs)

        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 256)
        self._batch_size.setValue(32)
        train_form.addRow("Batch Size:", self._batch_size)

        self._learning_rate = QDoubleSpinBox()
        self._learning_rate.setRange(1e-6, 1e-1)
        self._learning_rate.setSingleStep(1e-4)
        self._learning_rate.setDecimals(6)
        self._learning_rate.setValue(1e-3)
        train_form.addRow("学习率:", self._learning_rate)

        self._weight_decay = QDoubleSpinBox()
        self._weight_decay.setRange(1e-6, 1e-1)
        self._weight_decay.setDecimals(6)
        self._weight_decay.setValue(1e-4)
        train_form.addRow("权重衰减:", self._weight_decay)

        self._optimizer_combo = QComboBox()
        self._optimizer_combo.addItems(["AdamW", "Adam", "SGD"])
        train_form.addRow("优化器:", self._optimizer_combo)

        self._lr_scheduler_combo = QComboBox()
        self._lr_scheduler_combo.addItems(["cosine", "step", "plateau", "warmup_cosine"])
        train_form.addRow("学习率调度:", self._lr_scheduler_combo)

        self._warmup_steps = QSpinBox()
        self._warmup_steps.setRange(0, 20000)
        self._warmup_steps.setValue(5000)
        train_form.addRow("Warmup Steps:", self._warmup_steps)

        self._grad_accum = QSpinBox()
        self._grad_accum.setRange(1, 16)
        self._grad_accum.setValue(1)
        train_form.addRow("梯度累积:", self._grad_accum)
        train_vbox.addWidget(train_group)
        train_vbox.addStretch()

        train_tab.setWidget(train_widget)
        tabs.addTab(train_tab, "训练参数")

        # === Tab 3: Loss Weights ===
        loss_tab = QScrollArea()
        loss_tab.setWidgetResizable(True)
        loss_widget = QWidget()
        loss_vbox = QVBoxLayout(loss_widget)
        loss_vbox.setSpacing(12)

        loss_group = QGroupBox("多任务损失权重")
        loss_form = QFormLayout(loss_group)
        loss_form.setHorizontalSpacing(12)
        loss_form.setVerticalSpacing(10)

        self._asr_weight = QDoubleSpinBox()
        self._asr_weight.setRange(0.0, 10.0)
        self._asr_weight.setSingleStep(0.1)
        self._asr_weight.setValue(1.0)
        loss_form.addRow("ASR 损失权重:", self._asr_weight)

        self._dialect_weight = QDoubleSpinBox()
        self._dialect_weight.setRange(0.0, 10.0)
        self._dialect_weight.setSingleStep(0.1)
        self._dialect_weight.setValue(0.3)
        loss_form.addRow("方言损失权重:", self._dialect_weight)

        self._speaker_weight = QDoubleSpinBox()
        self._speaker_weight.setRange(0.0, 10.0)
        self._speaker_weight.setSingleStep(0.1)
        self._speaker_weight.setValue(0.5)
        loss_form.addRow("声纹损失权重:", self._speaker_weight)
        loss_vbox.addWidget(loss_group)
        loss_vbox.addStretch()

        loss_tab.setWidget(loss_widget)
        tabs.addTab(loss_tab, "损失权重")

        layout.addWidget(tabs)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Preset switching
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)

    def _on_preset_changed(self, preset: str):
        """Apply preset values."""
        presets = {
            "tiny": {"backbone_dim": 128, "num_blocks": 2, "vocab_size": 4000,
                     "embed_dim": 128, "num_epochs": 50, "batch_size": 64},
            "standard": {"backbone_dim": 256, "num_blocks": 4, "vocab_size": 6000,
                         "embed_dim": 256, "num_epochs": 100, "batch_size": 32},
            "large": {"backbone_dim": 320, "num_blocks": 6, "vocab_size": 8000,
                      "embed_dim": 320, "num_epochs": 150, "batch_size": 16},
        }
        if preset in presets:
            p = presets[preset]
            self._backbone_dim.setValue(p["backbone_dim"])
            self._num_blocks.setValue(p["num_blocks"])
            self._vocab_size.setValue(p["vocab_size"])
            self._embed_dim.setValue(p["embed_dim"])
            self._num_epochs.setValue(p["num_epochs"])
            self._batch_size.setValue(p["batch_size"])

    def _load_config(self, config: dict):
        """Load values from existing config dict."""
        for key, widget in [
            ('backbone_dim', self._backbone_dim),
            ('num_blocks', self._num_blocks),
            ('vocab_size', self._vocab_size),
            ('embed_dim', self._embed_dim),
            ('num_epochs', self._num_epochs),
            ('batch_size', self._batch_size),
            ('learning_rate', self._learning_rate),
            ('weight_decay', self._weight_decay),
            ('asr_loss_weight', self._asr_weight),
            ('dialect_loss_weight', self._dialect_weight),
            ('speaker_loss_weight', self._speaker_weight),
        ]:
            if key in config:
                if isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(config[key]))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(config[key]))

    def get_config(self) -> dict:
        """Return the current configuration as a dict."""
        return {
            "preset": self._preset_combo.currentText(),
            "backbone_dim": self._backbone_dim.value(),
            "num_blocks": self._num_blocks.value(),
            "vocab_size": self._vocab_size.value(),
            "embed_dim": self._embed_dim.value(),
            "backbone_dropout": self._backbone_dropout.value(),
            "causal": self._causal_cb.isChecked(),
            "num_epochs": self._num_epochs.value(),
            "batch_size": self._batch_size.value(),
            "learning_rate": self._learning_rate.value(),
            "weight_decay": self._weight_decay.value(),
            "optimizer": self._optimizer_combo.currentText(),
            "lr_scheduler": self._lr_scheduler_combo.currentText(),
            "warmup_steps": self._warmup_steps.value(),
            "gradient_accumulation_steps": self._grad_accum.value(),
            "asr_loss_weight": self._asr_weight.value(),
            "dialect_loss_weight": self._dialect_weight.value(),
            "speaker_loss_weight": self._speaker_weight.value(),
            "device": "cpu",
        }
