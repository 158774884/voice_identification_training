"""
Log output panel — displays categorized log messages in a table view
with filtering and search capabilities.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QHeaderView,
    QLineEdit, QPushButton, QComboBox, QCheckBox, QLabel,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, Signal, Slot
from PySide6.QtGui import QColor, QFont

from app.utils.logger import LogManager, LogEntry, LogLevel
from app.app_config import LOG_CATEGORIES


class LogTableModel(QAbstractTableModel):
    """Table model for log entries."""

    COLUMNS = ["时间", "类别", "级别", "消息"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list[LogEntry] = []

    def set_entries(self, entries: list[LogEntry]):
        self.beginResetModel()
        self._entries = entries
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._entries)

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLUMNS)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        entry = self._entries[index.row()]

        if role == Qt.DisplayRole:
            if index.column() == 0:
                return entry.timestamp
            elif index.column() == 1:
                return entry.category
            elif index.column() == 2:
                return entry.level.value
            elif index.column() == 3:
                return entry.message

        if role == Qt.ForegroundRole:
            if entry.level == LogLevel.ERROR:
                return QColor("#dc3545")
            elif entry.level == LogLevel.WARNING:
                return QColor("#e6a817")
            elif entry.level == LogLevel.DEBUG:
                return QColor("#6c757d")

        if role == Qt.FontRole and entry.level == LogLevel.ERROR:
            font = QFont()
            font.setBold(True)
            return font

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section]
        return None


class LogPanel(QWidget):
    """Dockable log panel with filtering, search, and auto-scroll."""

    log_added = Signal(LogEntry)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log_manager = LogManager()
        self._auto_scroll = True
        self._setup_ui()
        self._connect_signals()
        # Load initial entries
        self._refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Toolbar ---
        toolbar = QHBoxLayout()

        # Category filter
        toolbar.addWidget(QLabel("类别:"))
        self._category_combo = QComboBox()
        self._category_combo.addItem("全部", None)
        for cat in LOG_CATEGORIES:
            self._category_combo.addItem(cat, cat)
        self._category_combo.setMaximumWidth(100)
        toolbar.addWidget(self._category_combo)

        # Level filter
        toolbar.addWidget(QLabel("级别:"))
        self._level_combo = QComboBox()
        self._level_combo.addItem("全部", None)
        for lvl in LogLevel:
            self._level_combo.addItem(lvl.value, lvl)
        self._level_combo.setMaximumWidth(100)
        toolbar.addWidget(self._level_combo)

        # Search
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("搜索日志...")
        self._search_input.setMaximumWidth(200)
        toolbar.addWidget(self._search_input)

        toolbar.addStretch()

        # Buttons
        self._auto_scroll_cb = QCheckBox("自动滚动")
        self._auto_scroll_cb.setChecked(True)
        toolbar.addWidget(self._auto_scroll_cb)

        clear_btn = QPushButton("清空")
        clear_btn.setMaximumWidth(60)
        toolbar.addWidget(clear_btn)

        export_btn = QPushButton("导出")
        export_btn.setMaximumWidth(60)
        toolbar.addWidget(export_btn)

        layout.addLayout(toolbar)

        # --- Table ---
        self._table = QTableView()
        self._model = LogTableModel(self)
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.setColumnWidth(0, 150)  # timestamp
        self._table.setColumnWidth(1, 60)   # category
        self._table.setColumnWidth(2, 60)   # level
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        # --- Connections ---
        clear_btn.clicked.connect(self._clear_logs)
        export_btn.clicked.connect(self._export_logs)
        self._search_input.textChanged.connect(self._on_filter_changed)
        self._category_combo.currentIndexChanged.connect(self._on_filter_changed)
        self._level_combo.currentIndexChanged.connect(self._on_filter_changed)
        self._auto_scroll_cb.toggled.connect(self._set_auto_scroll)

    def _connect_signals(self):
        """Listen for new log entries."""
        self._log_manager.add_listener(self._on_new_log)

    def _on_new_log(self, entry: LogEntry):
        """Called when a new log entry arrives. Refresh if it passes filters."""
        # Check if entry matches current filters
        cat_filter = self._category_combo.currentData()
        lvl_filter = self._level_combo.currentData()
        search = self._search_input.text().lower()

        if cat_filter and entry.category != cat_filter:
            return
        if lvl_filter and entry.level != lvl_filter:
            return
        if search and search not in entry.message.lower():
            return

        # Append single row
        self._model.beginInsertRows(QModelIndex(), self._model.rowCount(), self._model.rowCount())
        self._model._entries.append(entry)
        self._model.endInsertRows()

        if self._auto_scroll:
            self._table.scrollToBottom()

    @Slot()
    def _refresh(self):
        """Reload all entries with current filters."""
        cat_filter = self._category_combo.currentData()
        lvl_filter = self._level_combo.currentData()
        search = self._search_input.text()

        categories = [cat_filter] if cat_filter else None
        levels = [lvl_filter] if lvl_filter else None
        entries = self._log_manager.get_entries(
            categories=categories, levels=levels, search=search
        )
        self._model.set_entries(entries)
        if self._auto_scroll:
            self._table.scrollToBottom()

    @Slot()
    def _on_filter_changed(self):
        self._refresh()

    @Slot(bool)
    def _set_auto_scroll(self, checked: bool):
        self._auto_scroll = checked

    @Slot()
    def _clear_logs(self):
        self._log_manager.clear()
        self._model.set_entries([])

    @Slot()
    def _export_logs(self):
        from PySide6.QtWidgets import QFileDialog
        import os

        path, _ = QFileDialog.getSaveFileName(
            self, "导出日志", "voice_model_log.txt",
            "Log Files (*.log *.txt);;All Files (*.*)"
        )
        if not path:
            return
        try:
            entries = self._log_manager.get_entries()
            with open(path, "w", encoding="utf-8") as f:
                for e in entries:
                    f.write(e.formatted() + "\n")
            self._log_manager.info("系统", f"日志已导出到: {path}")
        except Exception as ex:
            self._log_manager.error("系统", f"导出日志失败: {ex}")

    def append_log(self, category: str, message: str, level: LogLevel = LogLevel.INFO):
        """Programmatic log insertion (convenience)."""
        self._log_manager.log(category, message, level)
