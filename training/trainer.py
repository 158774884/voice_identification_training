"""
多任务训练器 —— 支持三阶段训练策略

Phase 1: 预训练 (ASR + Dialect, 冻结 Speaker)
Phase 2: 联合训练 (所有任务, 冻结 Backbone 前 N 层)
Phase 3: 全参数微调 (小学习率)

支持:
- 混合精度训练 (AMP FP16)
- 梯度累积
- 学习率 warmup + cosine 衰减
- 定期评估和保存
- TensorBoard 日志
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.multi_task_model import MultiTaskVoiceModel
from training.losses import MultiTaskLoss, UncertaintyWeightingLoss
from training.config import TrainingConfig


class Trainer:
    """
    多任务训练器

    使用方式:
        config = TrainingConfig()
        model = MultiTaskVoiceModel(...)
        loss_fn = MultiTaskLoss(...)
        trainer = Trainer(model, loss_fn, config)
        trainer.train(train_loader, val_loader)
    """

    def __init__(self,
                 model: MultiTaskVoiceModel,
                 loss_fn: MultiTaskLoss,
                 config: TrainingConfig,
                 device: Optional[torch.device] = None,
                 progress_callback: Optional[callable] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config

        self.device = device or torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器 (在 train() 中初始化)
        self.scheduler = None

        # 混合精度
        self.scaler = GradScaler(enabled=config.use_amp)

        # 日志
        self.writer = None
        self.global_step = 0
        self.current_epoch = 0

        # 最佳验证损失
        self.best_val_loss = float('inf')

        # 进度回调: (phase, epoch, step, total_steps, loss_dict, lr) -> None
        self.progress_callback = progress_callback

        # 确保 checkpoint 目录存在
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _create_optimizer(self):
        """创建优化器"""
        cfg = self.config

        # 分组参数 (不同模块可用不同学习率)
        backbone_params = list(self.model.backbone.parameters())
        other_params = [p for n, p in self.model.named_parameters()
                        if not n.startswith('backbone')]

        param_groups = [
            {'params': backbone_params, 'lr': cfg.learning_rate, 'name': 'backbone'},
            {'params': other_params, 'lr': cfg.learning_rate, 'name': 'heads'},
        ]

        if cfg.optimizer == 'AdamW':
            return torch.optim.AdamW(
                param_groups,
                lr=cfg.learning_rate,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == 'Adam':
            return torch.optim.Adam(
                param_groups,
                lr=cfg.learning_rate,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == 'SGD':
            return torch.optim.SGD(
                param_groups,
                lr=cfg.learning_rate,
                momentum=0.9,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _create_scheduler(self, total_steps: int):
        """创建学习率调度器"""
        cfg = self.config

        if cfg.lr_scheduler == 'warmup_cosine':
            # Warmup + Cosine Annealing
            def lr_lambda(step):
                # Linear warmup
                if step < cfg.warmup_steps:
                    return float(step) / float(max(1, cfg.warmup_steps))
                # Cosine decay
                progress = float(step - cfg.warmup_steps) / float(
                    max(1, total_steps - cfg.warmup_steps))
                return max(cfg.min_lr / cfg.learning_rate,
                           0.5 * (1.0 + math.cos(math.pi * progress)))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda)

        elif cfg.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=cfg.min_lr)

        elif cfg.lr_scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=total_steps // 3, gamma=0.1)

        elif cfg.lr_scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5,
                min_lr=cfg.min_lr)

        else:
            self.scheduler = None

    def train(self, train_loader, val_loader=None):
        """
        完整训练流程 (三阶段)

        Phase 1: 预训练 (ASR + Dialect)
        Phase 2: 联合训练
        Phase 3: 微调
        """
        cfg = self.config
        total_phases = 3

        # Phase 1: 预训练
        print("\n" + "=" * 60)
        print("Phase 1: Pretraining (ASR + Dialect)")
        print("=" * 60)
        if cfg.pretrain_freeze_speaker:
            # 冻结 speaker branch
            for param in self.model.speaker_branch.parameters():
                param.requires_grad = False
            print("[Phase 1] Speaker branch frozen")

        # 暂设 speaker loss 权重为 0
        original_speaker_weight = self.loss_fn.speaker_weight
        self.loss_fn.speaker_weight = 0.0

        self._run_training_loop(
            train_loader, val_loader,
            num_epochs=cfg.pretrain_epochs,
            phase_name="Phase1_Pretrain",
        )

        # 恢复 speaker weight
        self.loss_fn.speaker_weight = original_speaker_weight
        # 解冻 speaker
        for param in self.model.speaker_branch.parameters():
            param.requires_grad = True

        # Phase 2: 联合训练
        print("\n" + "=" * 60)
        print("Phase 2: Joint Training (All Tasks)")
        print("=" * 60)
        if cfg.freeze_backbone_layers > 0:
            self.model.freeze_backbone_layers(cfg.freeze_backbone_layers)
            print(f"[Phase 2] Frozen backbone layers: 0-{cfg.freeze_backbone_layers-1}")

        # 重设学习率 (joint training 可以用较低学习率)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cfg.learning_rate

        self._run_training_loop(
            train_loader, val_loader,
            num_epochs=cfg.joint_epochs,
            phase_name="Phase2_Joint",
        )

        # Phase 3: 全参数微调
        print("\n" + "=" * 60)
        print("Phase 3: Full Finetuning")
        print("=" * 60)
        self.model.unfreeze_all()

        # 更小的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cfg.finetune_lr

        self._run_training_loop(
            train_loader, val_loader,
            num_epochs=cfg.finetune_epochs,
            phase_name="Phase3_Finetune",
        )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Final model saved to: {cfg.checkpoint_dir}")
        print("=" * 60)

    def _run_training_loop(self, train_loader, val_loader,
                           num_epochs: int, phase_name: str = ""):
        """单个训练阶段的主循环"""
        cfg = self.config

        # 估算总步数
        total_steps = num_epochs * min(cfg.steps_per_epoch, len(train_loader))
        self._create_scheduler(total_steps)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.model.train()

            epoch_loss = 0.0
            epoch_asr_loss = 0.0
            epoch_dialect_loss = 0.0
            epoch_speaker_loss = 0.0
            start_time = time.time()

            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= cfg.steps_per_epoch:
                    break

                # 移到设备
                batch = self._to_device(batch)

                # 累积梯度时 loss 需要除以累积步数
                loss, loss_dict = self._train_step(batch)

                # 梯度累积
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if cfg.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), cfg.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self.scheduler is not None and cfg.lr_scheduler != 'plateau':
                        self.scheduler.step()

                    self.global_step += 1

                # 记录
                epoch_loss += loss_dict.get('total_loss', 0.0)
                epoch_asr_loss += loss_dict.get('asr_loss', 0.0)
                epoch_dialect_loss += loss_dict.get('dialect_loss', 0.0)
                epoch_speaker_loss += loss_dict.get('speaker_loss', 0.0)

                # 日志
                if batch_idx % cfg.log_interval == 0 and batch_idx > 0:
                    avg_loss = epoch_loss / (batch_idx + 1) * cfg.gradient_accumulation_steps
                    lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time

                    print(f"[{phase_name}] Epoch {epoch+1}/{num_epochs} | "
                          f"Step {batch_idx} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"ASR: {epoch_asr_loss/(batch_idx+1)*cfg.gradient_accumulation_steps:.4f} | "
                          f"Dial: {epoch_dialect_loss/(batch_idx+1)*cfg.gradient_accumulation_steps:.4f} | "
                          f"Spk: {epoch_speaker_loss/(batch_idx+1)*cfg.gradient_accumulation_steps:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Time: {elapsed:.1f}s")

                    # Progress callback (for GUI integration)
                    if self.progress_callback:
                        try:
                            self.progress_callback(
                                phase_name, epoch + 1, batch_idx,
                                min(cfg.steps_per_epoch, len(train_loader)),
                                {
                                    'total_loss': avg_loss,
                                    'asr_loss': epoch_asr_loss / (batch_idx + 1) * cfg.gradient_accumulation_steps,
                                    'dialect_loss': epoch_dialect_loss / (batch_idx + 1) * cfg.gradient_accumulation_steps,
                                    'speaker_loss': epoch_speaker_loss / (batch_idx + 1) * cfg.gradient_accumulation_steps,
                                },
                                lr,
                            )
                        except Exception:
                            pass  # callback failure shouldn't crash training

                    # TensorBoard
                    if self.writer:
                        self.writer.add_scalar(f'{phase_name}/loss', avg_loss, self.global_step)
                        self.writer.add_scalar(f'{phase_name}/lr', lr, self.global_step)

                # 验证
                if val_loader is not None and batch_idx % cfg.eval_interval == 0 and batch_idx > 0:
                    val_loss = self.evaluate(val_loader)
                    print(f"[{phase_name}] Validation Loss: {val_loss:.4f}")

                    if self.writer:
                        self.writer.add_scalar(f'{phase_name}/val_loss', val_loss,
                                               self.global_step)

                    # 保存最佳模型
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(f'best_model.pt')

                    # Plateau scheduler
                    if self.scheduler is not None and cfg.lr_scheduler == 'plateau':
                        self.scheduler.step(val_loss)

                    self.model.train()

                # 定期保存
                if batch_idx % cfg.save_interval == 0 and batch_idx > 0:
                    self.save_checkpoint(f'{phase_name}_step{self.global_step}.pt')

            # Epoch 结束
            epoch_loss = epoch_loss / min(batch_idx + 1, cfg.steps_per_epoch)
            print(f"[{phase_name}] Epoch {epoch+1}/{num_epochs} completed | "
                  f"Avg Loss: {epoch_loss:.4f}")

        # Phase 结束保存
        self.save_checkpoint(f'{phase_name}_final.pt')

    def _train_step(self, batch: Dict[str, torch.Tensor]):
        """单步训练"""
        cfg = self.config

        with autocast(enabled=cfg.use_amp):
            # 前向
            outputs = self.model(
                audio=batch['audio'],
                audio_lengths=batch['audio_lengths'],
                asr_labels=batch['asr_labels'],
                asr_label_lengths=batch['asr_label_lengths'],
                dialect_labels=batch['dialect_labels'],
                speaker_labels=batch['speaker_labels'],
            )

            # 计算损失
            total_loss, loss_dict = self.loss_fn(outputs, {
                'asr_labels': batch['asr_labels'],
                'asr_label_lengths': batch['asr_label_lengths'],
                'feat_lengths': outputs['feat_lengths'],
                'dialect_labels': batch['dialect_labels'],
            })

        # 缩放 loss (梯度累积)
        if cfg.gradient_accumulation_steps > 1:
            total_loss = total_loss / cfg.gradient_accumulation_steps

        # 反向传播
        self.scaler.scale(total_loss).backward()

        return total_loss, loss_dict

    @torch.no_grad()
    def evaluate(self, val_loader) -> float:
        """验证"""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = self._to_device(batch)

            with autocast(enabled=self.config.use_amp):
                outputs = self.model(
                    audio=batch['audio'],
                    audio_lengths=batch['audio_lengths'],
                    asr_labels=batch['asr_labels'],
                    asr_label_lengths=batch['asr_label_lengths'],
                    dialect_labels=batch['dialect_labels'],
                    speaker_labels=batch['speaker_labels'],
                )

                _, loss_dict = self.loss_fn(outputs, {
                    'asr_labels': batch['asr_labels'],
                    'asr_label_lengths': batch['asr_label_lengths'],
                    'feat_lengths': outputs['feat_lengths'],
                    'dialect_labels': batch['dialect_labels'],
                })

            total_loss += loss_dict.get('total_loss', 0.0)
            num_batches += 1

            # 最多评估 100 个 batch
            if num_batches >= 100:
                break

        return total_loss / max(num_batches, 1)

    def _to_device(self, batch: Dict) -> Dict:
        """将 batch 移到设备"""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
        }, path)
        print(f"[Checkpoint] Saved to {path}")

        # Notify progress callback
        if self.progress_callback:
            try:
                self.progress_callback(
                    'checkpoint', self.current_epoch, self.global_step, 0,
                    {'total_loss': 0, 'asr_loss': 0, 'dialect_loss': 0,
                     'speaker_loss': 0, 'checkpoint_path': path},
                    0,
                )
            except Exception:
                pass

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"[Checkpoint] Loaded from {path} (step {self.global_step})")

    def setup_tensorboard(self, log_dir: str = './runs'):
        """初始化 TensorBoard"""
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir)
            print(f"[TensorBoard] Logging to {log_dir}")
        else:
            print("[TensorBoard] tensorboard not installed, skipping")
