"""
Chip editor dialog — add or edit chip specifications.
"""
import json
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QTextEdit, QDialogButtonBox, QLabel, QScrollArea, QWidget,
)
from PySide6.QtCore import Qt

from app.models.chip_database import ChipSpec


class ChipEditorDialog(QDialog):
    """Dialog for adding/editing chip specifications."""

    def __init__(self, chip: ChipSpec = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑芯片" if chip else "添加新芯片")
        self.setMinimumSize(500, 550)
        self._chip = chip
        self._setup_ui()
        if chip:
            self._load_chip(chip)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        widget = QWidget()
        main_vbox = QVBoxLayout(widget)
        main_vbox.setSpacing(12)

        # === Basic Info ===
        basic_group = QGroupBox("基本信息")
        basic_form = QFormLayout(basic_group)
        basic_form.setHorizontalSpacing(12)
        basic_form.setVerticalSpacing(8)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("芯片型号")
        basic_form.addRow("名称:", self._name_edit)

        self._mfr_edit = QLineEdit()
        self._mfr_edit.setPlaceholderText("制造商")
        basic_form.addRow("制造商:", self._mfr_edit)

        self._arch_combo = QComboBox()
        self._arch_combo.addItems(["MCU", "SoC", "NPU", "DSP"])
        basic_form.addRow("架构:", self._arch_combo)

        main_vbox.addWidget(basic_group)

        # === CPU ===
        cpu_group = QGroupBox("CPU / 处理器")
        cpu_form = QFormLayout(cpu_group)
        cpu_form.setHorizontalSpacing(12)
        cpu_form.setVerticalSpacing(8)

        self._cpu_cores = QSpinBox()
        self._cpu_cores.setRange(1, 16)
        self._cpu_cores.setValue(1)
        cpu_form.addRow("核心数:", self._cpu_cores)

        self._cpu_freq = QSpinBox()
        self._cpu_freq.setRange(10, 5000)
        self._cpu_freq.setValue(200)
        self._cpu_freq.setSuffix(" MHz")
        cpu_form.addRow("主频:", self._cpu_freq)
        main_vbox.addWidget(cpu_group)

        # === Memory ===
        mem_group = QGroupBox("存储")
        mem_form = QFormLayout(mem_group)
        mem_form.setHorizontalSpacing(12)
        mem_form.setVerticalSpacing(8)

        self._ram_kb = QSpinBox()
        self._ram_kb.setRange(1, 16777216)
        self._ram_kb.setValue(512)
        self._ram_kb.setSuffix(" KB")
        mem_form.addRow("RAM / SRAM:", self._ram_kb)

        self._flash_kb = QSpinBox()
        self._flash_kb.setRange(1, 134217728)
        self._flash_kb.setValue(4096)
        self._flash_kb.setSuffix(" KB")
        mem_form.addRow("Flash:", self._flash_kb)

        self._max_model_kb = QSpinBox()
        self._max_model_kb.setRange(1, 1048576)
        self._max_model_kb.setValue(1024)
        self._max_model_kb.setSuffix(" KB")
        mem_form.addRow("最大模型大小:", self._max_model_kb)
        main_vbox.addWidget(mem_group)

        # === AI Acceleration ===
        ai_group = QGroupBox("AI 加速")
        ai_form = QFormLayout(ai_group)
        ai_form.setHorizontalSpacing(12)
        ai_form.setVerticalSpacing(8)

        self._npu_tops = QDoubleSpinBox()
        self._npu_tops.setRange(0.0, 100.0)
        self._npu_tops.setDecimals(2)
        self._npu_tops.setValue(0.0)
        self._npu_tops.setSuffix(" TOPS")
        ai_form.addRow("NPU 算力:", self._npu_tops)

        self._dsp_cb = QCheckBox("具备 DSP")
        ai_form.addRow("", self._dsp_cb)

        self._quant_edit = QLineEdit()
        self._quant_edit.setPlaceholderText("INT8, INT16, FP16 (逗号分隔)")
        ai_form.addRow("支持量化:", self._quant_edit)

        self._power_spin = QSpinBox()
        self._power_spin.setRange(1, 50000)
        self._power_spin.setValue(500)
        self._power_spin.setSuffix(" mW")
        ai_form.addRow("典型功耗:", self._power_spin)
        main_vbox.addWidget(ai_group)

        # === Operators ===
        ops_group = QGroupBox("算子兼容")
        ops_form = QFormLayout(ops_group)
        ops_form.setHorizontalSpacing(12)
        ops_form.setVerticalSpacing(8)

        self._ops_edit = QTextEdit()
        self._ops_edit.setPlaceholderText(
            "支持的 ONNX 算子 (每行一个):\nConv\nConv1d\nBatchNorm\nReLU\nGRU\n..."
        )
        self._ops_edit.setMaximumHeight(100)
        ops_form.addRow("支持算子:", self._ops_edit)
        main_vbox.addWidget(ops_group)

        # === Price ===
        price_group = QGroupBox("其他")
        price_form = QFormLayout(price_group)
        price_form.setHorizontalSpacing(12)
        price_form.setVerticalSpacing(8)

        self._price_spin = QDoubleSpinBox()
        self._price_spin.setRange(0.0, 10000.0)
        self._price_spin.setDecimals(2)
        self._price_spin.setPrefix("¥ ")
        price_form.addRow("参考价格:", self._price_spin)

        self._notes_edit = QTextEdit()
        self._notes_edit.setPlaceholderText("备注信息...")
        self._notes_edit.setMaximumHeight(60)
        price_form.addRow("备注:", self._notes_edit)
        main_vbox.addWidget(price_group)

        main_vbox.addStretch()

        scroll.setWidget(widget)
        layout.addWidget(scroll)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_chip(self, chip: ChipSpec):
        self._name_edit.setText(chip.name)
        self._mfr_edit.setText(chip.manufacturer)
        self._arch_combo.setCurrentText(chip.architecture)
        self._cpu_cores.setValue(chip.cpu_cores)
        self._cpu_freq.setValue(chip.cpu_freq_mhz)
        self._ram_kb.setValue(chip.ram_kb)
        self._flash_kb.setValue(chip.flash_kb)
        self._max_model_kb.setValue(chip.max_model_size_kb)
        self._npu_tops.setValue(chip.npu_tops)
        self._dsp_cb.setChecked(chip.dsp)
        self._quant_edit.setText(", ".join(chip.supported_quant))
        self._power_spin.setValue(chip.power_consumption_mw)
        self._ops_edit.setText("\n".join(chip.supported_ops))
        self._price_spin.setValue(chip.price_cny)
        self._notes_edit.setText(chip.notes)

    def get_chip(self) -> ChipSpec:
        """Build ChipSpec from form values."""
        ops_text = self._ops_edit.toPlainText().strip()
        ops_list = [op.strip() for op in ops_text.split("\n") if op.strip()] if ops_text else []

        quant_text = self._quant_edit.text().strip()
        quant_list = [q.strip() for q in quant_text.split(",") if q.strip()] if quant_text else []

        return ChipSpec(
            id=self._chip.id if self._chip else 0,
            name=self._name_edit.text().strip(),
            manufacturer=self._mfr_edit.text().strip(),
            architecture=self._arch_combo.currentText(),
            cpu_cores=self._cpu_cores.value(),
            cpu_freq_mhz=self._cpu_freq.value(),
            ram_kb=self._ram_kb.value(),
            flash_kb=self._flash_kb.value(),
            npu_tops=self._npu_tops.value(),
            dsp=self._dsp_cb.isChecked(),
            supported_quant=quant_list,
            max_model_size_kb=self._max_model_kb.value(),
            power_consumption_mw=self._power_spin.value(),
            supported_ops=ops_list,
            price_cny=self._price_spin.value(),
            notes=self._notes_edit.toPlainText().strip(),
        )
