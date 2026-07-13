"""
Training Controller — manages model training lifecycle.
"""
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any

from PySide6.QtCore import QObject, Signal, Slot

from app.workers.training_worker import TrainingWorker
from app.utils.logger import LogManager


class TrainingController(QObject):
    """Controls training workflow — config, start, pause, resume, stop."""

    # Signals
    training_started = Signal()
    training_paused = Signal()
    training_resumed = Signal()
    training_stopped = Signal()
    training_finished = Signal(dict)       # summary
    training_error = Signal(str)           # error
    progress_update = Signal(int, int, int)  # epoch, step, total
    loss_update = Signal(float, float, float, float)
    accuracy_update = Signal(float)
    lr_update = Signal(float)
    checkpoint_saved = Signal(str)
    model_version_added = Signal(str)       # version name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = LogManager()
        self._worker: Optional[TrainingWorker] = None
        self._training_running = False
        self._training_paused = False

    def start_training(self, config: Dict[str, Any], data_root: str,
                       checkpoint_dir: str):
        """Launch training in background thread."""
        if self._training_running:
            self._log.warning("训练", "训练已在运行中")
            return

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save config alongside checkpoints
        config_path = os.path.join(checkpoint_dir, 'training_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # Create and start worker
        self._worker = TrainingWorker(config, data_root, checkpoint_dir)
        self._worker.progress.connect(self.progress_update)
        self._worker.loss_update.connect(self.loss_update)
        self._worker.accuracy_update.connect(self.accuracy_update)
        self._worker.lr_update.connect(self.lr_update)
        self._worker.epoch_complete.connect(self._on_epoch_done)
        self._worker.phase_changed.connect(lambda p: self._log.info("训练", f"阶段: {p}"))
        self._worker.checkpoint_saved.connect(self._on_checkpoint)
        self._worker.training_complete.connect(self._on_training_done)
        self._worker.error_occurred.connect(self._on_training_error)
        self._worker.log_message.connect(
            lambda cat, msg: self._log.info(cat, msg)
        )

        self._worker.start()
        self._training_running = True
        self._training_paused = False
        self.training_started.emit()
        self._log.info("训练", "训练已启动")

    def pause_training(self):
        """Pause running training."""
        if self._worker and self._training_running and not self._training_paused:
            self._worker.pause()
            self._training_paused = True
            self.training_paused.emit()
            self._log.info("训练", "训练已暂停")

    def resume_training(self):
        """Resume paused training."""
        if self._worker and self._training_paused:
            self._worker.resume()
            self._training_paused = False
            self.training_resumed.emit()
            self._log.info("训练", "训练已恢复")

    def stop_training(self):
        """Stop training completely."""
        if self._worker:
            self._worker.cancel()
            self._worker.wait(3000)
            self._training_running = False
            self._training_paused = False
            self._worker = None
            self.training_stopped.emit()
            self._log.info("训练", "训练已停止")

    @Slot(int, float)
    def _on_epoch_done(self, epoch: int, avg_loss: float):
        self._log.info("训练", f"Epoch {epoch} 完成, 平均 Loss: {avg_loss:.4f}")

    @Slot(str)
    def _on_checkpoint(self, path: str):
        name = os.path.basename(path)
        self._log.info("训练", f"检查点已保存: {name}")
        # Register model version
        version_entry = {
            "name": name.replace('.pt', ''),
            "checkpoint_path": path,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
        }
        self.model_version_added.emit(name)

    @Slot(dict)
    def _on_training_done(self, summary: dict):
        self._training_running = False
        self._training_paused = False
        self._worker = None
        self.training_finished.emit(summary)
        self._log.info("训练", "训练完成")

    @Slot(str)
    def _on_training_error(self, error: str):
        self._training_running = False
        self._training_paused = False
        self.training_error.emit(error)
        self._log.error("训练", error)

    @property
    def is_running(self) -> bool:
        return self._training_running

    @property
    def is_paused(self) -> bool:
        return self._training_paused
