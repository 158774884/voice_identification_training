"""
Model version panel — standalone widget for browsing and managing checkpoints.
Displayed alongside the log panel at the bottom of the main window.
"""
import os
import shutil
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QHeaderView,
    QPushButton, QLabel, QAbstractItemView, QMessageBox, QFileDialog,
)
from PySide6.QtCore import Qt, Signal, Slot, QAbstractTableModel, QModelIndex

from app.app_config import CHECKPOINTS_DIR
from app.utils.logger import LogManager


class VersionTableModel(QAbstractTableModel):
    """Table model for checkpoint versions."""

    COLUMNS = ["", "模型名称", "时间", "路径"]
    CURRENT_MARK = "★"  # indicator for the currently-active model

    def __init__(self, parent=None):
        super().__init__(parent)
        self._versions: list[dict] = []
        self._current_path: str = ""

    def set_current(self, checkpoint_path: str):
        """Highlight the row whose checkpoint_path matches."""
        old_row = self._row_for_path(self._current_path)
        new_row = self._row_for_path(checkpoint_path)
        self._current_path = checkpoint_path
        if old_row >= 0:
            self.dataChanged.emit(
                self.index(old_row, 0), self.index(old_row, self.columnCount() - 1)
            )
        if new_row >= 0:
            self.dataChanged.emit(
                self.index(new_row, 0), self.index(new_row, self.columnCount() - 1)
            )

    def _row_for_path(self, path: str) -> int:
        for i, v in enumerate(self._versions):
            if v.get("checkpoint_path", "") == path:
                return i
        return -1

    def set_versions(self, versions: list[dict]):
        self.beginResetModel()
        self._versions = versions
        # Re-resolve current path row after refresh
        self._current_row = self._row_for_path(self._current_path)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._versions)

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLUMNS)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        v = self._versions[index.row()]
        is_current = v.get("checkpoint_path", "") == self._current_path

        if role == Qt.DisplayRole:
            col = index.column()
            if col == 0:
                return self.CURRENT_MARK if is_current else ""
            elif col == 1:
                return v.get("name", "")
            elif col == 2:
                return v.get("timestamp", "")
            elif col == 3:
                return v.get("checkpoint_path", "")

        if role == Qt.FontRole and is_current:
            from PySide6.QtGui import QFont
            font = QFont()
            font.setBold(True)
            return font

        if role == Qt.ForegroundRole and is_current:
            from PySide6.QtGui import QColor
            return QColor("#1a73e8")  # blue highlight

        if role == Qt.ToolTipRole:
            col = index.column()
            if col == 0:
                return "当前模型" if is_current else ""
            elif col == 1:
                return v.get("name", "")
            elif col == 2:
                return v.get("timestamp", "")
            elif col == 3:
                return v.get("checkpoint_path", "")

        if role == Qt.UserRole:
            return v
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section]
        return None


class ModelVersionPanel(QWidget):
    """Bottom panel: browse, activate, and delete model checkpoints."""

    version_activated = Signal(str)   # checkpoint path
    version_deleted = Signal(str)     # checkpoint path
    version_imported = Signal(dict)   # imported model version metadata
    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = LogManager()
        self._project_dir = ""
        self._setup_ui()
        self.scan()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        title = QLabel("模型版本")
        title.setStyleSheet("font-weight: bold; color: #5f6368;")
        header.addWidget(title)
        header.addStretch()

        import_btn = QPushButton("导入模型")
        import_btn.setObjectName("secondaryBtn")
        import_btn.setMinimumWidth(88)
        import_btn.clicked.connect(self._on_import)
        header.addWidget(import_btn)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setObjectName("secondaryBtn")
        refresh_btn.setMinimumWidth(64)
        refresh_btn.clicked.connect(self.scan)
        header.addWidget(refresh_btn)

        self._activate_btn = QPushButton("设为当前")
        self._activate_btn.setObjectName("secondaryBtn")
        self._activate_btn.setMinimumWidth(96)
        self._activate_btn.clicked.connect(self._on_activate)
        header.addWidget(self._activate_btn)

        delete_btn = QPushButton("删除")
        delete_btn.setObjectName("dangerBtn")
        delete_btn.setMinimumWidth(64)
        delete_btn.clicked.connect(self._on_delete)
        header.addWidget(delete_btn)

        layout.addLayout(header)

        # Table
        self._table = QTableView()
        self._model = VersionTableModel(self)
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setTextElideMode(Qt.ElideRight)   # long paths -> "..."
        self._table.setWordWrap(False)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)                   # ★ 标记
        hh.setSectionResizeMode(1, QHeaderView.Interactive)             # 模型名称
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)        # 时间
        hh.setSectionResizeMode(3, QHeaderView.Stretch)                # 路径
        hh.setStretchLastSection(True)
        self._table.setColumnWidth(0, 24)   # ★ 列
        self._table.setColumnWidth(1, 160)  # 名称
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

    # ================================================================
    # Actions
    # ================================================================

    def set_project_dir(self, project_dir: str):
        """Set the current project directory so its models/ folder is also scanned."""
        self._project_dir = project_dir or ""

    def scan(self):
        """Scan the global checkpoints dir and the current project's models/ dir."""
        versions = []
        scan_dirs = [CHECKPOINTS_DIR]
        if self._project_dir:
            scan_dirs.append(os.path.join(self._project_dir, "models"))
        for base_dir in scan_dirs:
            if os.path.exists(base_dir):
                for root, dirs, files in os.walk(base_dir):
                    for fn in files:
                        if fn.endswith(('.pt', '.pth', '.onnx')):
                            path = os.path.join(root, fn)
                            try:
                                stat = os.stat(path)
                                ts = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                            except Exception:
                                ts = ""
                            versions.append({
                                "name": fn,
                                "checkpoint_path": path,
                                "timestamp": ts,
                            })
        # Sort by mtime descending (newest first)
        versions.sort(key=lambda v: v["checkpoint_path"], reverse=True)
        self._model.set_versions(versions)
        self.refresh_requested.emit()

    @Slot()
    def _on_import(self):
        """Import an externally trained model file into the project."""
        path, _ = QFileDialog.getOpenFileName(
            self, "导入模型", "",
            "模型文件 (*.pt *.pth *.onnx);;所有文件 (*.*)",
        )
        if not path:
            return

        try:
            path = os.path.abspath(path)

            # Destination: checkpoints/imported/
            ckpt_root = os.path.abspath(CHECKPOINTS_DIR)
            dest_dir = os.path.join(ckpt_root, "imported")
            os.makedirs(dest_dir, exist_ok=True)

            fn = os.path.basename(path)
            dest_path = os.path.join(dest_dir, fn)

            # If the file is already inside the checkpoints dir, register in place.
            # NOTE: avoid os.path.commonpath — it raises ValueError across drives on Windows.
            src_norm = os.path.normcase(path)
            root_norm = os.path.normcase(ckpt_root) + os.sep
            already_managed = src_norm.startswith(root_norm)

            if already_managed:
                dest_path = path
            else:
                # Resolve filename collisions in the destination.
                if os.path.exists(dest_path) and os.path.normcase(os.path.abspath(dest_path)) != src_norm:
                    stem, ext = os.path.splitext(fn)
                    i = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(dest_dir, f"{stem}_{i}{ext}")
                        i += 1
                shutil.copy2(path, dest_path)

            try:
                ts = datetime.fromtimestamp(os.stat(dest_path).st_mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts = ""

            version = {
                "name": os.path.basename(dest_path),
                "checkpoint_path": dest_path,
                "timestamp": ts,
                "phase": "imported",
                "notes": f"导入自: {path}" if not already_managed else "",
            }

            self.scan()
            self.version_imported.emit(version)
            self._log.info("系统", f"已导入模型: {version['name']}")
            QMessageBox.information(self, "导入成功", f"已导入模型:\n{version['name']}")
        except Exception as e:
            self._log.error("系统", f"导入模型失败: {e}")
            QMessageBox.critical(self, "导入失败", f"导入模型时出错:\n{e}")

    @Slot()
    def _on_activate(self):
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            QMessageBox.information(self, "提示", "请先在列表中选择一个模型")
            return
        v = self._model.data(indexes[0], Qt.UserRole)
        if v:
            path = v["checkpoint_path"]
            self._model.set_current(path)
            self.version_activated.emit(path)
            self._log.info("系统", f"当前模型设置为: {v['name']}")

    @Slot()
    def _on_delete(self):
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return
        v = self._model.data(indexes[0], Qt.UserRole)
        if not v:
            return
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除模型 '{v['name']}' 吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                os.remove(v["checkpoint_path"])
                self._log.info("系统", f"已删除: {v['name']}")
                self.version_deleted.emit(v["checkpoint_path"])
                self.scan()
            except Exception as e:
                QMessageBox.critical(self, "删除失败", str(e))
