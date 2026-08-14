"""
Chip Evaluation Panel — chip database browser, compatibility assessment, batch comparison.
"""
from typing import List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QTableView, QHeaderView, QAbstractItemView, QLineEdit,
    QComboBox, QTextEdit, QProgressBar, QSplitter, QMessageBox,
    QToolButton, QMenu, QFileDialog,
)
from PySide6.QtCore import Qt, Signal, Slot, QAbstractTableModel, QModelIndex

from app.controllers.chip_controller import ChipController, AssessmentResult
from app.models.chip_database import ChipSpec
from app.ui.dialogs.chip_editor import ChipEditorDialog
from app.utils.datasheet_parser import parse_datasheet
from app.utils.logger import LogManager


class ChipTableModel(QAbstractTableModel):
    """Table model for chip list."""

    COLUMNS = ["名称", "制造商", "架构", "CPU(MHz)", "RAM(KB)", "Flash(KB)",
               "NPU(TOPS)", "功耗(mW)", "价格(¥)"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._chips: list[ChipSpec] = []

    def set_chips(self, chips: list[ChipSpec]):
        self.beginResetModel()
        self._chips = chips
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._chips)

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLUMNS)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        c = self._chips[index.row()]
        if role == Qt.DisplayRole:
            col = index.column()
            vals = [c.name, c.manufacturer, c.architecture, c.cpu_freq_mhz,
                    c.ram_kb, c.flash_kb, c.npu_tops, c.power_consumption_mw, c.price_cny]
            return vals[col]
        if role == Qt.UserRole:
            return c
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section]
        return None


class ChipPanel(QWidget):
    """Chip evaluation and comparison panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._controller = ChipController(self)
        self._log = LogManager()
        self._setup_ui()
        self._connect_signals()
        self._refresh_chip_list()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # === Top: Chip Database ===
        chip_group = QGroupBox("芯片数据库")
        chip_layout = QVBoxLayout(chip_group)

        # Search bar
        search_layout = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("搜索芯片名称/制造商...")
        search_layout.addWidget(self._search_input)

        self._arch_filter = QComboBox()
        self._arch_filter.addItem("全部架构", None)
        search_layout.addWidget(self._arch_filter)

        search_btn = QPushButton("搜索")
        search_layout.addWidget(search_btn)

        add_chip_btn = QToolButton()
        add_chip_btn.setText("+ 添加芯片")
        add_chip_btn.setObjectName("successBtn")
        add_chip_btn.setPopupMode(QToolButton.InstantPopup)
        add_menu = QMenu(add_chip_btn)
        add_menu.addAction("手动添加", self._on_add_chip)
        add_menu.addAction("从数据手册导入", self._on_import_datasheet)
        add_chip_btn.setMenu(add_menu)
        search_layout.addWidget(add_chip_btn)

        edit_chip_btn = QPushButton("编辑")
        edit_chip_btn.setObjectName("secondaryBtn")
        search_layout.addWidget(edit_chip_btn)

        delete_btn = QPushButton("删除")
        delete_btn.setObjectName("dangerBtn")
        search_layout.addWidget(delete_btn)

        chip_layout.addLayout(search_layout)

        self._chip_table = QTableView()
        self._chip_model = ChipTableModel(self)
        self._chip_table.setModel(self._chip_model)
        self._chip_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._chip_table.setAlternatingRowColors(True)
        self._chip_table.horizontalHeader().setStretchLastSection(True)
        self._chip_table.verticalHeader().setVisible(False)
        self._chip_table.setMaximumHeight(200)
        chip_layout.addWidget(self._chip_table)

        main_layout.addWidget(chip_group)

        # === Middle: Assessment ===
        assess_group = QGroupBox("芯片适配评估")
        assess_layout = QHBoxLayout(assess_group)

        self._assess_btn = QPushButton("🔍 评估选中的芯片")
        self._assess_btn.setObjectName("successBtn")
        assess_layout.addWidget(self._assess_btn)

        self._batch_compare_btn = QPushButton("📊 批量对比")
        self._batch_compare_btn.setObjectName("secondaryBtn")
        assess_layout.addWidget(self._batch_compare_btn)

        self._model_params_spin_label = QLabel("模型参数 (百万):")
        assess_layout.addWidget(self._model_params_spin_label)

        from PySide6.QtWidgets import QSpinBox
        self._model_params_spin = QSpinBox()
        self._model_params_spin.setRange(1, 100)
        self._model_params_spin.setValue(5)  # ~4.5M default
        self._model_params_spin.setSuffix(" M")
        assess_layout.addWidget(self._model_params_spin)

        assess_layout.addStretch()
        main_layout.addWidget(assess_group)

        # === Bottom: Results ===
        splitter = QSplitter(Qt.Horizontal)

        # Left: Dimension scores
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)

        self._result_verdict = QLabel("选择芯片并点击评估")
        self._result_verdict.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        result_layout.addWidget(self._result_verdict)

        self._result_details = QTextEdit()
        self._result_details.setReadOnly(True)
        self._result_details.setMaximumHeight(250)
        result_layout.addWidget(self._result_details)

        splitter.addWidget(result_widget)

        # Right: Suggestions
        self._suggestions_text = QTextEdit()
        self._suggestions_text.setReadOnly(True)
        self._suggestions_text.setPlaceholderText("优化建议将在此显示...")
        splitter.addWidget(self._suggestions_text)

        splitter.setSizes([400, 300])
        main_layout.addWidget(splitter)

        # === Connections ===
        search_btn.clicked.connect(self._on_search)
        edit_chip_btn.clicked.connect(self._on_edit_chip)
        delete_btn.clicked.connect(self._on_delete_chip)
        self._assess_btn.clicked.connect(self._on_assess)
        self._batch_compare_btn.clicked.connect(self._on_batch_compare)

    def _connect_signals(self):
        self._controller.chip_list_updated.connect(self._refresh_chip_list)
        self._controller.assessment_complete.connect(self._on_assessment_done)

    # ================================================================
    # Slots
    # ================================================================

    def _refresh_chip_list(self):
        chips = self._controller.list_chips()
        self._chip_model.set_chips(chips)
        # Update arch filter
        current = self._arch_filter.currentText()
        self._arch_filter.clear()
        self._arch_filter.addItem("全部架构", None)
        for arch in self._controller.database.get_architectures():
            self._arch_filter.addItem(arch, arch)
        idx = self._arch_filter.findText(current)
        if idx >= 0:
            self._arch_filter.setCurrentIndex(idx)

    @Slot()
    def _on_search(self):
        query = self._search_input.text()
        arch = self._arch_filter.currentData()
        chips = self._controller.list_chips(query=query, arch=arch)
        self._chip_model.set_chips(chips)

    @Slot()
    def _on_add_chip(self):
        dialog = ChipEditorDialog(parent=self)
        if dialog.exec() == ChipEditorDialog.Accepted:
            chip = dialog.get_chip()
            self._controller.add_chip(chip)

    @Slot()
    def _on_import_datasheet(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择芯片数据手册", "",
            "数据手册 (*.pdf *.txt *.md *.docx);;所有文件 (*.*)",
        )
        if not path:
            return

        try:
            result = parse_datasheet(path)
        except Exception as e:
            self._log.error("芯片", f"数据手册解析失败: {e}")
            QMessageBox.critical(self, "解析失败", f"无法解析数据手册文件：\n{e}")
            return

        if result.confidence < 0.3:
            found = "、".join(result.found_fields) if result.found_fields else "无"
            QMessageBox.warning(
                self, "解析结果较少",
                f"从数据手册中抽取到的信息较少（已识别字段：{found}）。\n\n"
                f"将打开编辑对话框，请手动补充参数。",
            )

        dialog = ChipEditorDialog(result.chip, self, title="从数据手册导入芯片")
        if dialog.exec() == ChipEditorDialog.Accepted:
            chip = dialog.get_chip()
            if not chip.name:
                QMessageBox.warning(self, "缺少名称", "请填写芯片名称。")
                return
            self._controller.add_chip(chip)
            self._select_chip_by_name(chip.name)
            # 立即评估刚导入的芯片
            self._on_assess()

    def _select_chip_by_name(self, name: str):
        """Select the table row whose chip name matches."""
        for row in range(self._chip_model.rowCount()):
            chip = self._chip_model.data(self._chip_model.index(row, 0), Qt.UserRole)
            if chip and chip.name == name:
                self._chip_table.selectRow(row)
                return

    @Slot()
    def _on_edit_chip(self):
        indexes = self._chip_table.selectionModel().selectedRows()
        if not indexes:
            return
        chip: ChipSpec = self._chip_model.data(indexes[0], Qt.UserRole)
        if chip:
            dialog = ChipEditorDialog(chip, self)
            if dialog.exec() == ChipEditorDialog.Accepted:
                updated = dialog.get_chip()
                self._controller.update_chip(updated)

    @Slot()
    def _on_delete_chip(self):
        indexes = self._chip_table.selectionModel().selectedRows()
        if not indexes:
            return
        chip: ChipSpec = self._chip_model.data(indexes[0], Qt.UserRole)
        if chip:
            reply = QMessageBox.question(
                self, "确认删除", f"确定要删除芯片 '{chip.name}' 吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._controller.delete_chip(chip.id)

    @Slot()
    def _on_assess(self):
        indexes = self._chip_table.selectionModel().selectedRows()
        if not indexes:
            QMessageBox.information(self, "提示", "请先选择一个芯片")
            return
        chip: ChipSpec = self._chip_model.data(indexes[0], Qt.UserRole)
        if chip:
            params = self._model_params_spin.value() * 1_000_000
            self._result_verdict.setText(f"正在评估 {chip.name}...")
            self._controller.assess(chip, model_params=params)

    @Slot()
    def _on_batch_compare(self):
        chips = self._chip_model._chips[:10]  # limit to 10
        if not chips:
            return
        params = self._model_params_spin.value() * 1_000_000
        results = self._controller.batch_compare(chips, model_params=params)
        # Show comparison summary
        text = "=== 批量对比结果 ===\n\n"
        text += f"{'芯片':15s} {'总分':>6s} {'推理':>6s} {'内存':>6s} {'闪存':>6s} {'结果':>6s}\n"
        text += "-" * 60 + "\n"
        for r in sorted(results, key=lambda x: x.overall_score, reverse=True):
            verdict = "通过" if r.overall_pass else "不通过"
            text += (f"{r.chip_name:15s} {r.overall_score:5.0f}% "
                     f"{r.inference_score:5.0f}% {r.memory_score:5.0f}% "
                     f"{r.flash_score:5.0f}% {verdict:>6s}\n")
        self._result_details.setText(text)

    @Slot(AssessmentResult)
    def _on_assessment_done(self, result: AssessmentResult):
        verdict_text = "✅ 通过" if result.overall_pass else "❌ 不通过"
        self._result_verdict.setText(
            f"{result.chip_name}: {verdict_text} "
            f"(总分: {result.overall_score:.0f}%)"
        )

        details = (
            f"模型大小: {result.model_size_kb:.0f} KB\n"
            f"推理时间: {result.inference_time_ms:.1f} ms\n"
            f"峰值RAM: {result.peak_ram_kb:.0f} KB\n"
            f"算子兼容: {result.ops_compatibility:.0f}%\n\n"
            f"各维度评分:\n"
            f"  推理耗时: {result.inference_score:.0f}%\n"
            f"  内存占用: {result.memory_score:.0f}%\n"
            f"  闪存占用: {result.flash_score:.0f}%\n"
            f"  算力匹配: {result.compute_score:.0f}%\n"
            f"  功耗适配: {result.power_score:.0f}%\n"
            f"  架构兼容: {result.arch_score:.0f}%"
        )
        if result.unsupported_ops:
            details += f"\n\n不支持的算子: {', '.join(result.unsupported_ops)}"

        self._result_details.setText(details)

        if result.suggestions:
            self._suggestions_text.setText(
                "优化建议:\n\n" + "\n".join(f"• {s}" for s in result.suggestions)
            )
        else:
            self._suggestions_text.setText("无需优化建议。")

    @property
    def controller(self) -> ChipController:
        return self._controller
