#!/usr/bin/env python3
"""
训练入口脚本

用法:
    # 从头训练
    python train.py --config configs/standard.json

    # 从检查点恢复
    python train.py --resume checkpoints/best_model.pt

    # 仅推理测试
    python train.py --eval --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import argparse
import torch

from model.multi_task_model import MultiTaskVoiceModel, create_model
from training.losses import MultiTaskLoss
from training.trainer import Trainer
from training.config import TrainingConfig
from data.dataset import MultiTaskDataset, create_dataloader
from data.preprocessing import AudioPreprocessor
from data.augmentation import AudioAugmentor
from data.vocab import ChineseVocab, get_default_vocab


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Task Voice Model Training')

    # 模式
    parser.add_argument('--train', action='store_true', default=True,
                        help='Training mode')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluation mode only')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # 数据
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--train_metadata', type=str, default='train.jsonl',
                        help='Training metadata file')
    parser.add_argument('--val_metadata', type=str, default='val.jsonl',
                        help='Validation metadata file')
    parser.add_argument('--noise_dir', type=str, default=None,
                        help='Noise directory for augmentation')

    # 配置预设
    parser.add_argument('--preset', type=str, default='standard',
                        choices=['tiny', 'standard', 'large'],
                        help='Model size preset')

    # 超参数覆盖
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')

    # 检查点
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint for evaluation')

    return parser.parse_args()


def main():
    args = parse_args()

    # ===== 创建配置 =====
    if args.preset == 'tiny':
        from training.config import get_tiny_config
        config = get_tiny_config()
    elif args.preset == 'large':
        from training.config import get_large_config
        config = get_large_config()
    else:
        config = TrainingConfig()

    # 命令行覆盖
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.epochs:
        config.num_epochs = args.epochs
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    config.device = args.device

    # ===== 创建词汇表 =====
    print("Building vocabulary...")
    vocab = get_default_vocab()
    config.vocab_size = len(vocab)
    print(f"Vocabulary size: {len(vocab)}")

    # ===== 创建模型 =====
    print("Creating model...")
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

    # 打印模型信息
    model.summary()

    # ===== 创建损失函数 =====
    loss_fn = MultiTaskLoss(
        asr_weight=config.asr_loss_weight,
        dialect_weight=config.dialect_loss_weight,
        speaker_weight=config.speaker_loss_weight,
        blank_id=config.blank_id,
    )

    # ===== 创建训练器 =====
    trainer = Trainer(model, loss_fn, config)

    # 加载检查点
    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.eval:
        # 仅评估
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        # 创建验证数据集
        preprocessor = AudioPreprocessor(target_sr=config.sample_rate)
        val_dataset = MultiTaskDataset(
            data_root=args.data_root,
            metadata_file=args.val_metadata,
            vocab=vocab,
            preprocessor=preprocessor,
            training=False,
            max_audio_length=config.max_audio_length,
        )
        val_loader = create_dataloader(val_dataset, config.batch_size,
                                       shuffle=False, num_workers=config.num_workers)

        val_loss = trainer.evaluate(val_loader)
        print(f"Validation loss: {val_loss:.4f}")
        return

    # ===== 创建数据集 =====
    preprocessor = AudioPreprocessor(target_sr=config.sample_rate)

    augmentor = None
    if config.speed_perturb or config.noise_augment:
        augmentor = AudioAugmentor(
            speed_perturb=config.speed_perturb,
            speed_rates=config.speed_perturb_rates,
            noise_augment=config.noise_augment,
            noise_snr_range=config.noise_snr_range,
            reverb_augment=config.reverb_augment,
            spec_augment=config.spec_augment,
            freq_mask_width=config.freq_mask_width,
            time_mask_width=config.time_mask_width,
            num_freq_masks=config.num_freq_masks,
            num_time_masks=config.num_time_masks,
        )

        if args.noise_dir and os.path.isdir(args.noise_dir):
            augmentor.load_noise_samples(args.noise_dir)

    print("Loading training data...")
    train_dataset = MultiTaskDataset(
        data_root=args.data_root,
        metadata_file=args.train_metadata,
        vocab=vocab,
        preprocessor=preprocessor,
        augmentor=augmentor,
        training=True,
        max_audio_length=config.max_audio_length,
        min_audio_length=config.min_audio_length,
    )
    train_loader = create_dataloader(train_dataset, config.batch_size,
                                     shuffle=True, num_workers=config.num_workers)

    # 验证集
    val_loader = None
    val_meta = os.path.join(args.data_root, args.val_metadata)
    if os.path.exists(val_meta):
        print("Loading validation data...")
        val_augmentor = AudioAugmentor(speed_perturb=False, noise_augment=False,
                                        spec_augment=False)
        val_dataset = MultiTaskDataset(
            data_root=args.data_root,
            metadata_file=args.val_metadata,
            vocab=vocab,
            preprocessor=preprocessor,
            augmentor=val_augmentor,
            training=False,
            max_audio_length=config.max_audio_length,
        )
        val_loader = create_dataloader(val_dataset, config.batch_size,
                                       shuffle=False, num_workers=config.num_workers)
    else:
        print(f"Warning: No validation data found at {val_meta}")

    # ===== 训练 =====
    trainer.setup_tensorboard(os.path.join(config.checkpoint_dir, 'runs'))
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
