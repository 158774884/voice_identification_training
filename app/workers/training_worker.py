"""
TrainingWorker — runs model training in a background QThread.
Wraps the existing training pipeline (train.py, training/trainer.py).
"""
import os
import sys
import time
import threading
import torch
from typing import Optional, Dict, Any

from PySide6.QtCore import QThread, Signal


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TrainingWorker(QThread):
    """Runs the training loop in a background thread."""

    # Signals emitted to the UI
    progress = Signal(int, int, int)       # epoch, step, total_steps
    loss_update = Signal(float, float, float, float)  # total, asr, dialect, speaker
    accuracy_update = Signal(float)         # validation accuracy
    lr_update = Signal(float)              # current learning rate
    epoch_complete = Signal(int, float)    # epoch number, avg loss
    phase_changed = Signal(str)            # phase name
    checkpoint_saved = Signal(str)         # checkpoint path
    training_complete = Signal(dict)       # summary metrics
    log_message = Signal(str, str)         # category, message
    error_occurred = Signal(str)           # error message

    def __init__(self, config: Dict[str, Any], data_root: str,
                 checkpoint_dir: str, parent=None):
        super().__init__(parent)
        self._config = config
        self._data_root = data_root
        self._checkpoint_dir = checkpoint_dir
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially
        self._device = config.get('device', 'cpu')

    def run(self):
        """Main training loop — runs in background thread."""
        try:
            self._run_training()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error_occurred.emit(f"训练出错: {e}\n{tb}")

    def _run_training(self):
        """Execute the training pipeline using existing project code."""
        from training.config import TrainingConfig
        from model.multi_task_model import create_model
        from training.losses import MultiTaskLoss
        from training.trainer import Trainer
        from data.vocab import get_default_vocab
        from data.dataset import MultiTaskDataset, create_dataloader
        from data.preprocessing import AudioPreprocessor

        cfg_dict = self._config

        # Build TrainingConfig
        config = TrainingConfig()
        config.batch_size = cfg_dict.get('batch_size', 32)
        config.learning_rate = cfg_dict.get('learning_rate', 1e-3)
        config.num_epochs = cfg_dict.get('num_epochs', 100)
        config.checkpoint_dir = self._checkpoint_dir
        config.device = self._device

        preset = cfg_dict.get('preset', 'standard')
        if preset == 'tiny':
            from training.config import get_tiny_config
            config = get_tiny_config()
            config.num_epochs = cfg_dict.get('num_epochs', 100)
            config.batch_size = cfg_dict.get('batch_size', 64)
        elif preset == 'large':
            from training.config import get_large_config
            config = get_large_config()
            config.num_epochs = cfg_dict.get('num_epochs', 100)
            config.batch_size = cfg_dict.get('batch_size', 16)

        self.log_message.emit("训练", f"配置: preset={preset}, epochs={config.num_epochs}, "
                              f"batch={config.batch_size}, lr={config.learning_rate}")

        # Build vocab
        self.log_message.emit("训练", "构建词汇表...")
        vocab = get_default_vocab()
        config.vocab_size = len(vocab)

        # Build model
        self.log_message.emit("训练", "创建模型...")
        model_config = {
            'frontend_channels': config.frontend_channels,
            'backbone_dim': config.backbone_dim,
            'num_blocks': config.num_blocks,
            'conv_kernel': config.conv_kernel,
            'gru_layers': config.gru_layers,
            'backbone_dropout': config.backbone_dropout,
            'causal': config.causal,
            'vocab_size': config.vocab_size,
            'asr_hidden_dim': config.asr_hidden_dim,
            'blank_id': config.blank_id,
            'num_dialects': config.num_dialects,
            'dialect_hidden_dim': config.dialect_hidden_dim,
            'embed_dim': config.embed_dim,
            'num_speakers': config.num_speakers,
            'speaker_dropout': config.speaker_dropout,
        }
        model = create_model(model_config)
        model.summary()

        # Loss
        loss_fn = MultiTaskLoss(
            asr_weight=config.asr_loss_weight,
            dialect_weight=config.dialect_loss_weight,
            speaker_weight=config.speaker_loss_weight,
            blank_id=config.blank_id,
        )

        # Trainer
        trainer = Trainer(model, loss_fn, config, progress_callback=self._on_progress)

        # Load datasets
        self.log_message.emit("训练", "加载训练数据...")
        preprocessor = AudioPreprocessor(target_sr=config.sample_rate)

        # Try to find metadata files
        train_meta = os.path.join(self._data_root, 'train.jsonl')
        val_meta = os.path.join(self._data_root, 'val.jsonl')

        if not os.path.exists(train_meta):
            # Check alternative paths
            alt_train = os.path.join(self._data_root, 'cmd_data', 'train.jsonl')
            if os.path.exists(alt_train):
                train_meta = alt_train
                self._data_root = os.path.dirname(alt_train)

        if not os.path.exists(train_meta):
            self.error_occurred.emit(f"找不到训练元数据文件: {train_meta}")
            return

        train_dataset = MultiTaskDataset(
            data_root=self._data_root,
            metadata_file=os.path.basename(train_meta),
            vocab=vocab,
            preprocessor=preprocessor,
            training=True,
            max_audio_length=config.max_audio_length,
        )
        train_loader = create_dataloader(
            train_dataset, config.batch_size, shuffle=True,
            num_workers=config.num_workers
        )

        val_loader = None
        if os.path.exists(val_meta):
            val_dataset = MultiTaskDataset(
                data_root=self._data_root,
                metadata_file=os.path.basename(val_meta),
                vocab=vocab,
                preprocessor=preprocessor,
                training=False,
                max_audio_length=config.max_audio_length,
            )
            val_loader = create_dataloader(
                val_dataset, config.batch_size, shuffle=False,
                num_workers=config.num_workers
            )

        self.log_message.emit("训练",
            f"训练集: {len(train_dataset)} 样本, "
            f"验证集: {len(val_dataset) if val_loader else 'N/A'} 样本"
        )

        # === Run training ===
        self.log_message.emit("训练", "开始训练...")

        try:
            trainer.train(train_loader, val_loader)
        except Exception as e:
            self.error_occurred.emit(f"训练过程中出错: {e}")
            return

        # Save final model
        final_path = os.path.join(self._checkpoint_dir, 'final_model.pt')
        trainer.save_checkpoint(final_path)
        self.checkpoint_saved.emit(final_path)

        summary = {
            "epochs_completed": config.num_epochs,
            "final_loss": 0.0,
            "best_checkpoint": os.path.join(self._checkpoint_dir, 'best_model.pt'),
            "final_checkpoint": final_path,
        }
        self.training_complete.emit(summary)
        self.log_message.emit("训练", "训练完成!")

    def pause(self):
        """Pause training."""
        self._pause_event.clear()

    def resume(self):
        """Resume training."""
        self._pause_event.set()

    def cancel(self):
        """Stop training."""
        self._cancel_event.set()
        self._pause_event.set()  # unpause to allow clean exit

    def _on_progress(self, phase: str, epoch: int, step: int,
                     total_steps: int, loss_dict: dict, lr: float):
        """Progress callback invoked from trainer at each log interval."""
        if self._cancel_event.is_set():
            return
        # Wait if paused
        self._pause_event.wait()

        self.progress.emit(epoch, step, total_steps)
        self.loss_update.emit(
            loss_dict.get('total_loss', 0.0),
            loss_dict.get('asr_loss', 0.0),
            loss_dict.get('dialect_loss', 0.0),
            loss_dict.get('speaker_loss', 0.0),
        )
        if lr > 0:
            self.lr_update.emit(lr)
        self.log_message.emit("训练",
            f"[{phase}] Epoch {epoch} Step {step}/{total_steps} "
            f"Loss {loss_dict.get('total_loss', 0):.4f} LR {lr:.2e}")

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()
