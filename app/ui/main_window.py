"""
MainWindow — QMainWindow with tabbed panels, menu bar, toolbar,
status bar, and bottom splitter (logs + model versions).
"""
import os
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QMenuBar, QMenu, QToolBar, QStatusBar,
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox,
    QFileDialog, QProgressBar, QSplitter,
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence, QIcon

from app.app_config import (
    APP_NAME, APP_VERSION, DEFAULT_WIDTH, DEFAULT_HEIGHT, MIN_WIDTH, MIN_HEIGHT,
    TAB_DATASET, TAB_TRAINING, TAB_TEST, TAB_CHIP, TAB_EXPORT,
    PROJECT_FILTER, PROJECT_EXTENSION, STYLES_DIR,
)
from app.utils.logger import LogManager, LogLevel, log_info, log_error
from app.ui.panels.log_panel import LogPanel
from app.ui.panels.model_version_panel import ModelVersionPanel
from app.ui.panels.dataset_panel import DatasetPanel
from app.ui.panels.test_panel import TestPanel
from app.ui.panels.training_panel import TrainingPanel
from app.ui.panels.chip_panel import ChipPanel
from app.ui.panels.export_panel import ExportPanel
from app.models.project import Project


class MainWindow(QMainWindow):
    """Top-level application window."""

    # Signals for inter-module communication
    project_loaded = Signal(str)  # project file path
    project_saved = Signal(str)

    def __init__(self):
        super().__init__()
        self._log_manager = LogManager()
        self._project: Project | None = None
        self._current_project_path: str | None = None
        self._project_dirty = False

        self._setup_window()
        self._setup_menu_bar()
        self._setup_tool_bar()
        self._setup_status_bar()
        self._setup_central_layout()
        self._load_stylesheet()

        # Log startup
        self._log_manager.info("系统", f"{APP_NAME} v{APP_VERSION} 启动成功")
        self._log_manager.info("系统", "欢迎使用语音模型训练与芯片适配工具")

    # ================================================================
    # Window Setup
    # ================================================================

    def _setup_window(self):
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(DEFAULT_WIDTH, DEFAULT_HEIGHT)
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)
        # Center on screen
        screen = self.screen().availableGeometry()
        self.move(
            (screen.width() - DEFAULT_WIDTH) // 2,
            (screen.height() - DEFAULT_HEIGHT) // 2,
        )

    def _load_stylesheet(self):
        """Load QSS stylesheet from file."""
        qss_path = os.path.join(STYLES_DIR, "default.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    # ================================================================
    # Menu Bar
    # ================================================================

    def _setup_menu_bar(self):
        menubar = self.menuBar()

        # === File Menu ===
        file_menu = menubar.addMenu("文件(&F)")

        new_action = QAction("新建项目(&N)", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction("打开项目(&O)", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("保存项目(&S)", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction("另存为...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # === View Menu ===
        view_menu = menubar.addMenu("视图(&V)")

        self._toggle_log_action = QAction("日志面板", self)
        self._toggle_log_action.setCheckable(True)
        self._toggle_log_action.setChecked(True)
        self._toggle_log_action.triggered.connect(self._toggle_log_panel)
        view_menu.addAction(self._toggle_log_action)

        # === Help Menu ===
        help_menu = menubar.addMenu("帮助(&H)")

        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # ================================================================
    # Tool Bar
    # ================================================================

    def _setup_tool_bar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        new_btn = toolbar.addAction("📄 新建")
        new_btn.triggered.connect(self._on_new_project)

        open_btn = toolbar.addAction("📂 打开")
        open_btn.triggered.connect(self._on_open_project)

        save_btn = toolbar.addAction("💾 保存")
        save_btn.triggered.connect(self._on_save_project)

        toolbar.addSeparator()

        # Training controls — wired to training panel
        self._train_action = toolbar.addAction("▶ 开始训练")
        self._train_action.triggered.connect(self._on_toolbar_start_training)

        self._stop_action = toolbar.addAction("⏹ 停止")
        self._stop_action.triggered.connect(self._on_toolbar_stop_training)
        self._stop_action.setEnabled(False)

        toolbar.addSeparator()

        self._export_action = toolbar.addAction("📦 导出固件")
        self._export_action.setEnabled(False)

    # ================================================================
    # Status Bar
    # ================================================================

    def _setup_status_bar(self):
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("就绪")
        self._status_bar.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setMaximumHeight(16)
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)

        self._worker_label = QLabel("")
        self._status_bar.addPermanentWidget(self._worker_label)

    # ================================================================
    # Central Layout: Tabs on top, Logs + Model Versions on bottom
    # ================================================================

    def _setup_central_layout(self):
        """Create vertical splitter: tabs (top) | logs + versions (bottom)."""
        self._tab_widget = QTabWidget()
        self._panels = {}

        # Tab 0: Dataset Panel
        self._dataset_panel = DatasetPanel(self)
        self._tab_widget.addTab(self._dataset_panel, TAB_DATASET)
        self._panels["dataset"] = self._dataset_panel

        # Tab 1: Test Panel
        self._test_panel = TestPanel(self)
        self._tab_widget.addTab(self._test_panel, TAB_TEST)
        self._panels["test"] = self._test_panel

        # Tab 2: Training Panel
        self._training_panel = TrainingPanel(self)
        self._tab_widget.addTab(self._training_panel, TAB_TRAINING)
        self._panels["training"] = self._training_panel

        # Tab 3: Chip Panel
        self._chip_panel = ChipPanel(self)
        self._tab_widget.addTab(self._chip_panel, TAB_CHIP)
        self._panels["chip"] = self._chip_panel

        # Tab 4: Export Panel
        self._export_panel = ExportPanel(self)
        self._tab_widget.addTab(self._export_panel, TAB_EXPORT)
        self._panels["export"] = self._export_panel

        # === Bottom panel: Logs (left) + Model Versions (right) ===
        self._log_panel = LogPanel(self)
        self._version_panel = ModelVersionPanel(self)

        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(self._log_panel)
        bottom_splitter.addWidget(self._version_panel)
        bottom_splitter.setSizes([600, 400])
        bottom_splitter.setChildrenCollapsible(False)
        self._bottom_splitter = bottom_splitter

        # === Main vertical splitter ===
        self._main_splitter = QSplitter(Qt.Vertical)
        self._main_splitter.addWidget(self._tab_widget)
        self._main_splitter.addWidget(bottom_splitter)
        self._main_splitter.setSizes([520, 240])
        self._main_splitter.setChildrenCollapsible(False)

        self.setCentralWidget(self._main_splitter)

        # Wire version panel signals
        self._version_panel.version_activated.connect(self._on_version_activated)
        self._version_panel.version_imported.connect(self._on_version_imported)

        # Wire training panel → toolbar sync (direct method calls, not signals)
        if "training" in self._panels:
            self._panels["training"]._toolbar_start_cb = self._on_training_started
            self._panels["training"]._toolbar_stop_cb = self._on_training_stopped
            self._panels["training"]._toolbar_finish_cb = self._on_training_finished

    # ================================================================
    # Toolbar Training Slots
    # ================================================================

    @Slot()
    def _on_toolbar_start_training(self):
        """Toolbar '开始训练' → delegate to training panel."""
        if "training" in self._panels:
            self._panels["training"]._on_start_training()

    @Slot()
    def _on_toolbar_stop_training(self):
        """Toolbar '停止' → delegate to training panel."""
        if "training" in self._panels:
            self._panels["training"]._on_stop_training()

    @Slot()
    def _on_training_started(self):
        self._log_manager.info("系统", "[工具栏] 收到 training_started 信号")
        self._train_action.setEnabled(False)
        self._stop_action.setEnabled(True)
        self.set_status("训练运行中...")

    @Slot()
    def _on_training_stopped(self):
        self._log_manager.info("系统", "[工具栏] 收到 training_stopped 信号")
        self._train_action.setEnabled(True)
        self._stop_action.setEnabled(False)
        self.set_status("就绪")

    @Slot(dict)
    def _on_training_finished(self, summary: dict):
        self._log_manager.info("系统", "[工具栏] 收到 training_finished 信号")
        self._train_action.setEnabled(True)
        self._stop_action.setEnabled(False)
        self.set_status("训练完成")
        # Refresh the model version list to pick up newly trained checkpoints
        if hasattr(self, '_version_panel'):
            self._version_panel.scan()

    @Slot(bool)
    def _toggle_log_panel(self, checked: bool):
        self._bottom_splitter.setVisible(checked)

    @Slot(str)
    def _on_version_activated(self, checkpoint_path: str):
        """When a model version is activated, update the test panel's current model."""
        if "test" in self._panels:
            test_panel = self._panels["test"]
            name = os.path.basename(checkpoint_path)
            # Add/update combo with full path stored as userData
            combo = test_panel._model_combo
            idx = combo.findData(checkpoint_path)
            if idx < 0:
                combo.insertItem(0, name, userData=checkpoint_path)
                idx = 0
            combo.setCurrentIndex(idx)
            test_panel._model_status_label.setText(f"✅ {name}")
            test_panel._model_status_label.setStyleSheet("color: #28a745;")
        self._log_manager.info("系统", f"当前模型: {os.path.basename(checkpoint_path)}")

    @Slot(dict)
    def _on_version_imported(self, version: dict):
        """Register an imported model into the current project's model versions."""
        if not self._project:
            self._log_manager.info(
                "系统", "已导入模型，但当前无打开的项目；请先新建/打开项目再保存。"
            )
            return
        # Avoid duplicate entries for the same checkpoint path.
        existing = [
            v for v in self._project.model_versions
            if v.get("checkpoint_path") == version.get("checkpoint_path")
        ]
        if not existing:
            self._project.add_model_version(version)
            self.mark_dirty()
            self._log_manager.info("系统", f"模型已加入项目: {version.get('name', '')}")

    # ================================================================
    # Project Management Actions
    # ================================================================

    @Slot()
    def _on_new_project(self):
        if self._check_unsaved():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "新建项目", f"voice_project{PROJECT_EXTENSION}", PROJECT_FILTER
        )
        if path:
            if not path.endswith(PROJECT_EXTENSION):
                path += PROJECT_EXTENSION
            self._current_project_path = path
            self._project_dirty = True
            self._save_empty_project(path)
            self.project_loaded.emit(path)
            self._log_manager.info("系统", f"新建项目: {os.path.basename(path)}")
            self._sync_project_context()
            self._update_title()

    @Slot()
    def _on_open_project(self):
        if self._check_unsaved():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", PROJECT_FILTER
        )
        if path and os.path.exists(path):
            self._current_project_path = path
            self._project = Project(path)
            self._project_dirty = False
            self._restore_project_to_panels()
            self.project_loaded.emit(path)
            self._log_manager.info("系统", f"打开项目: {os.path.basename(path)}")
            self._sync_project_context()
            self._update_title()

    @Slot()
    def _on_save_project(self):
        if self._project and self._current_project_path:
            self._save_project_data()
        else:
            self._on_save_project_as()

    @Slot()
    def _on_save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "另存为", "", PROJECT_FILTER
        )
        if path:
            if not path.endswith(PROJECT_EXTENSION):
                path += PROJECT_EXTENSION
            self._current_project_path = path
            self._project = Project(path)
            self._save_project_data()
            self._log_manager.info("系统", f"项目另存为: {os.path.basename(path)}")
            self._sync_project_context()
            self._update_title()

    def _save_empty_project(self, path: str):
        """Write initial empty project JSON."""
        self._project = Project(path)
        self._project.name = os.path.splitext(os.path.basename(path))[0]
        self._project.save()

    def _save_project_data(self):
        """Collect current state from ALL panels and save to project file."""
        if not self._project:
            return

        # --- Dataset config ---
        if "dataset" in self._panels and hasattr(self._panels["dataset"], 'get_dataset_config'):
            ds_config = self._panels["dataset"].get_dataset_config()
            self._project.update_dataset_config(**ds_config)

        # --- Training config ---
        tp = self._panels.get("training")
        if tp and hasattr(tp, '_current_config') and tp._current_config:
            self._project.update_training_config(tp._current_config)
        if tp and hasattr(tp, '_data_root') and tp._data_root:
            self._project._data.training_config['data_root'] = tp._data_root

        # --- Active model ---
        test_panel = self._panels.get("test")
        if test_panel:
            active_path = test_panel._model_combo.currentData()
            if active_path:
                self._project._data.active_version = os.path.basename(active_path)

        # --- Model versions (already collected via add_model_version during training/import) ---

        # --- Selected chip ---
        chip_panel = self._panels.get("chip")
        if chip_panel and hasattr(chip_panel, '_selected_chip_name'):
            self._project.selected_chip = chip_panel._selected_chip_name

        self._project.save()
        self._project_dirty = False
        self._log_manager.info("系统", "项目已保存")
        self._update_title()

    def _restore_project_to_panels(self):
        """After loading a project, push saved state back to all panels."""
        if not self._project:
            return
        data = self._project._data

        # --- Restore dataset config ---
        ds_config = data.dataset_config
        if ds_config and "dataset" in self._panels:
            ds_panel = self._panels["dataset"]
            if hasattr(ds_panel, '_controller'):
                ctrl = ds_panel._controller
                if ds_config.get("min_duration"):
                    ctrl._dataset.min_duration = ds_config["min_duration"]
                if ds_config.get("max_duration"):
                    ctrl._dataset.max_duration = ds_config["max_duration"]
                if ds_config.get("train_ratio"):
                    ctrl._dataset.train_ratio = ds_config["train_ratio"]
                if ds_config.get("val_ratio"):
                    ctrl._dataset.val_ratio = ds_config["val_ratio"]
                if ds_config.get("test_ratio"):
                    ctrl._dataset.test_ratio = ds_config["test_ratio"]
                if ds_config.get("data_root"):
                    ctrl._dataset.data_root = ds_config["data_root"]
                    # Update UI
                    ds_panel._min_dur_spin.setValue(ds_config.get("min_duration", 0.5))
                    ds_panel._max_dur_spin.setValue(ds_config.get("max_duration", 15.0))
                    ds_panel._train_slider.setValue(int(ds_config.get("train_ratio", 0.80) * 100))
                    ds_panel._val_slider.setValue(int(ds_config.get("val_ratio", 0.10) * 100))

        # --- Restore training config ---
        train_config = data.training_config
        if train_config and "training" in self._panels:
            tp = self._panels["training"]
            tp._current_config = train_config
            if train_config.get("data_root"):
                tp._data_root = train_config["data_root"]
                tp._data_dir_label.setText(train_config["data_root"])
            if train_config.get("preset"):
                tp._preset_combo.setCurrentText(train_config["preset"])
            if train_config.get("num_epochs"):
                tp._epochs_spin.setValue(train_config["num_epochs"])
            if train_config.get("batch_size"):
                tp._batch_spin.setValue(train_config["batch_size"])
            if train_config.get("learning_rate"):
                tp._lr_spin.setValue(train_config["learning_rate"])

        # --- Restore model versions ---
        for v in data.model_versions:
            self._version_panel._model._versions.append(v)
        self._version_panel._model.set_versions(data.model_versions)
        # Highlight active version
        if data.active_version:
            for v in data.model_versions:
                if v.get("name") == data.active_version:
                    self._version_panel._model.set_current(v.get("checkpoint_path", ""))
                    break

        # --- Restore active model to test panel ---
        active_path = self._project.active_model_path
        if active_path and os.path.exists(active_path):
            test_panel = self._panels.get("test")
            if test_panel:
                name = os.path.basename(active_path)
                combo = test_panel._model_combo
                idx = combo.findData(active_path)
                if idx < 0:
                    combo.insertItem(0, name, userData=active_path)
                else:
                    combo.setCurrentIndex(idx)
                test_panel._model_status_label.setText(f"[项目] {name}")
                test_panel._model_status_label.setStyleSheet("color: #1a73e8;")

        # --- Restore selected chip ---
        if data.selected_chip and "chip" in self._panels:
            chip_panel = self._panels["chip"]
            if hasattr(chip_panel, '_chip_combo'):
                chip_panel._chip_combo.setCurrentText(data.selected_chip)

        self._log_manager.info("系统", "项目配置已恢复")

    def _check_unsaved(self) -> bool:
        """Return True if user wants to cancel the operation."""
        if self._project_dirty:
            reply = QMessageBox.question(
                self, "未保存的更改",
                "当前项目有未保存的更改。是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            return reply != QMessageBox.Yes
        return False

    def _update_title(self):
        title = f"{APP_NAME} v{APP_VERSION}"
        if self._current_project_path:
            name = os.path.basename(self._current_project_path)
            dirty = " *" if self._project_dirty else ""
            title = f"{name}{dirty} — {title}"
        self.setWindowTitle(title)

    def _sync_project_context(self):
        """Push current project name/dir to panels that store models/checkpoints."""
        name = ""
        proj_dir = ""
        if self._project and self._current_project_path:
            name = self._project.name
            if not name or name == "Untitled":
                name = os.path.splitext(os.path.basename(self._current_project_path))[0]
            proj_dir = os.path.dirname(self._current_project_path)

        tp = self._panels.get("training")
        if tp and hasattr(tp, 'set_project_context'):
            tp.set_project_context(name, proj_dir)
        if hasattr(self, '_version_panel'):
            self._version_panel.set_project_dir(proj_dir)

    def mark_dirty(self):
        """Mark the current project as having unsaved changes."""
        self._project_dirty = True
        self._update_title()

    # ================================================================
    # Status Helpers
    # ================================================================

    def set_status(self, text: str):
        self._status_label.setText(text)

    def set_progress(self, value: int, maximum: int = 100):
        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(maximum)
        self._progress_bar.setValue(value)
        if value >= maximum:
            QTimer.singleShot(1500, self._progress_bar.hide)

    def set_worker_status(self, text: str):
        self._worker_label.setText(text)

    @property
    def current_project_path(self) -> str | None:
        return self._current_project_path

    # ================================================================
    # About Dialog
    # ================================================================

    @Slot()
    def _on_about(self):
        QMessageBox.about(
            self, f"关于 {APP_NAME}",
            f"<h3>{APP_NAME}</h3>"
            f"<p>版本 {APP_VERSION}</p>"
            f"<p>离线语音识别模型定制训练、芯片适配评估、"
            f"模型验证、固件导出与SDK移植一体化工具</p>"
            f"<p>纯本地运行 | Windows 10/11</p>"
        )

    # ================================================================
    # Close Event
    # ================================================================

    def closeEvent(self, event):
        # Prevent accidental close during training
        tp = self._panels.get("training")
        if tp and tp._controller.is_running:
            reply = QMessageBox.question(
                self, "训练进行中",
                "模型训练正在运行中！\n\n"
                "关闭应用程序将丢失当前训练进度。\n"
                "确定要停止训练并退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            # Stop training before closing
            tp._controller.stop_training()
            self._log_manager.info("系统", "训练已停止，正在关闭...")

        if self._project_dirty and self._current_project_path:
            reply = QMessageBox.question(
                self, "保存项目",
                "项目有未保存的更改，是否保存后退出？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Save:
                self._save_project_data()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        self._log_manager.info("系统", f"{APP_NAME} 关闭")
        event.accept()
