"""
New / Open project dialogs.
"""
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QFormLayout,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt

from app.app_config import PROJECT_EXTENSION, PROJECT_FILTER


class NewProjectDialog(QDialog):
    """Dialog for creating a new project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建项目")
        self.setMinimumWidth(450)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._name_edit = QLineEdit("voice_project")
        self._name_edit.setPlaceholderText("项目名称")
        form.addRow("项目名称:", self._name_edit)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("选择保存位置...")
        self._path_edit.setReadOnly(True)
        path_layout = QHBoxLayout()
        path_layout.addWidget(self._path_edit)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(browse_btn)
        form.addRow("保存位置:", path_layout)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_path(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", self._name_edit.text() + PROJECT_EXTENSION,
            PROJECT_FILTER,
        )
        if path:
            self._path_edit.setText(path)

    def _validate_and_accept(self):
        if not self._name_edit.text().strip():
            QMessageBox.warning(self, "输入错误", "请输入项目名称")
            return
        self.accept()

    def get_project_path(self) -> str:
        """Return the full project file path."""
        if self._path_edit.text():
            path = self._path_edit.text()
        else:
            path = os.path.join(os.getcwd(), self._name_edit.text() + PROJECT_EXTENSION)
        if not path.endswith(PROJECT_EXTENSION):
            path += PROJECT_EXTENSION
        return path

    def get_project_name(self) -> str:
        return self._name_edit.text().strip()
