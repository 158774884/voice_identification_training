"""
训练超参数配置

所有配置集中管理，支持从 YAML/JSON 文件加载
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class TrainingConfig:
    """训练超参数"""

    # ===== 模型架构 =====
    # Shared Backbone
    frontend_channels: Tuple[int, ...] = (64, 128, 256)
    backbone_dim: int = 256
    num_blocks: int = 4
    conv_kernel: int = 31
    gru_layers: int = 1
    backbone_dropout: float = 0.1
    causal: bool = True  # 因果卷积 (支持流式)

    # ASR
    vocab_size: int = 6000
    asr_hidden_dim: int = 320
    blank_id: int = 0

    # Dialect
    num_dialects: int = 10
    dialect_hidden_dim: int = 128

    # Speaker
    embed_dim: int = 256
    num_speakers: int = 1000
    speaker_dropout: float = 0.1

    # ===== 音频 =====
    sample_rate: int = 16000
    max_audio_length: int = 16000 * 15  # 最长 15 秒
    min_audio_length: int = 16000 * 1   # 最短 1 秒

    # ===== 数据处理 =====
    # SpecAugment (频域+时域遮蔽)
    spec_augment: bool = True
    freq_mask_width: int = 27
    time_mask_width: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 2

    # 数据增强
    speed_perturb: bool = True
    speed_perturb_rates: Tuple[float, float, float] = (0.9, 1.0, 1.1)
    noise_augment: bool = True
    noise_snr_range: Tuple[float, float] = (5.0, 20.0)
    reverb_augment: bool = False

    # 特征
    use_raw_waveform: bool = True  # 使用原始波形 (前端可学习)
    normalize_audio: bool = True
    remove_silence: bool = False

    # ===== 训练 =====
    # 优化器
    optimizer: str = 'AdamW'  # AdamW | Adam | SGD
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.98)

    # 学习率调度
    lr_scheduler: str = 'cosine'  # cosine | step | plateau | warmup_cosine
    warmup_steps: int = 5000
    min_lr: float = 1e-6

    # 批次
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 5.0

    # 训练阶段
    num_epochs: int = 100
    steps_per_epoch: int = 2000

    # 混合精度
    use_amp: bool = True  # Automatic Mixed Precision (FP16)

    # ===== 损失权重 =====
    asr_loss_weight: float = 1.0
    dialect_loss_weight: float = 0.3
    speaker_loss_weight: float = 0.5
    use_uncertainty_weighting: bool = False  # 是否使用不确定性自动加权

    # ===== 多阶段训练策略 =====
    # Phase 1: 预训练 (仅 ASR + Dialect, 冻结 speaker branch)
    pretrain_epochs: int = 10
    pretrain_freeze_speaker: bool = True

    # Phase 2: 联合训练 (所有任务, 冻结 backbone 前 N 层)
    joint_epochs: int = 60
    freeze_backbone_layers: int = 2

    # Phase 3: 全参数微调
    finetune_epochs: int = 30
    finetune_lr: float = 1e-4

    # ===== 日志 =====
    log_interval: int = 100  # 每 N 步打印一次
    eval_interval: int = 1000  # 每 N 步验证一次
    save_interval: int = 5000  # 每 N 步保存一次
    checkpoint_dir: str = './checkpoints'

    # ===== 设备 =====
    device: str = 'cuda'
    num_workers: int = 4

    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# 预设配置
def get_tiny_config():
    """极小模型 (适合超低功耗 MCU, < 2M params)"""
    return TrainingConfig(
        frontend_channels=(32, 64, 128),
        backbone_dim=128,
        num_blocks=2,
        conv_kernel=15,
        asr_hidden_dim=192,
        vocab_size=4000,
        embed_dim=128,
        num_dialects=10,
        batch_size=64,
    )


def get_standard_config():
    """标准配置 (适合通用 SOC NPU, ~4.5M params)"""
    return TrainingConfig()


def get_large_config():
    """较大配置 (适合边缘计算盒子, ~8M params)"""
    return TrainingConfig(
        frontend_channels=(64, 128, 256, 384),
        backbone_dim=320,
        num_blocks=6,
        conv_kernel=31,
        asr_hidden_dim=384,
        vocab_size=8000,
        embed_dim=320,
        batch_size=16,
    )
