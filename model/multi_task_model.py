"""
多任务语音模型 —— 共享主干 + 三任务分支

总参数量: ~4.5M (极端轻量, 适合 SOC 部署)

网络结构 (文字版):

┌─────────────────────────────────────────────────────────┐
│                   Input: 16kHz Raw Audio                 │
│                    [Batch, 1, T_audio]                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Shared Backbone (~2.5M)                     │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Learnable Conv Frontend                         │   │
│  │  Conv1d(1→64, K=400, S=160)   → 100Hz           │   │
│  │  Conv1d(64→128, K=3, S=2)    → 50Hz             │   │
│  │  Conv1d(128→256, K=3, S=2)   → 25Hz             │   │
│  │  Conv1d(256→256, K=3, S=1)   → 25Hz             │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Tiny Conformer Blocks × 4                       │   │
│  │  FFN(GLUGated) → DepthwiseConv → GRU → FFN       │   │
│  │  (无 MHA, 纯 Conv+GRU, ONNX/SOC 友好)            │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│         Output: [Batch, 256, T_feat] @ 25Hz              │
└────────────┬──────────────┬──────────────┬───────────────┘
             │              │              │
      ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────────┐
      │ Task 1: ASR │ │Task 2:     │ │Task 3: Speaker│
      │ CTC Head    │ │Dialect Cls │ │TDNN Embedding │
      │             │ │            │ │               │
      │ Conv1d × 2  │ │Attentive   │ │TDNN × 4      │
      │ LayerNorm   │ │Stats Pool  │ │SE Blocks     │
      │ Linear→Vocab│ │            │ │Attentive     │
      │ CTC Decode  │ │FC × 2      │ │Stats Pool    │
      │             │ │Softmax     │ │              │
      │ ~1.3M param │ │~50K param  │ │FC→256-dim    │
      │             │ │            │ │L2-Norm       │
      └──────┬──────┘ └─────┬──────┘ │AAM-Softmax   │
             │              │        │~800K param   │
             ▼              ▼        └────┬──────────┘
    ┌────────────┐ ┌──────────┐          ▼
    │ 中文字序列 │ │ 方言标签 │  ┌──────────────┐
    │ + CTC 后处理│ │ + 概率  │  │ 声纹嵌入向量 │
    └────────────┘ └──────────┘  │ 256-dim      │
                                 │ 余弦比对     │
                                 └──────────────┘

损失函数:
  Total Loss = λ_asr * CTC_Loss
             + λ_dialect * CrossEntropy_Loss
             + λ_speaker * AAM_Softmax_Loss

训练策略:
  1. Phase 1: 预训练共享主干 (自监督或大模型蒸馏)
  2. Phase 2: 联合训练 (冻结主干前2层)
  3. Phase 3: 全参数微调 (小学习率)
"""

import torch
import torch.nn as nn

from .shared_backbone import SharedBackbone
from .asr_branch import ASRBranch
from .dialect_branch import DialectBranch
from .speaker_branch import SpeakerBranch


class MultiTaskVoiceModel(nn.Module):
    """
    多任务语音模型 —— 单模型同时支持 ASR / 方言识别 / 声纹提取
    """

    def __init__(self,
                 # Shared backbone
                 frontend_channels=(64, 128, 256),
                 backbone_dim=256,
                 num_blocks=4,
                 conv_kernel=31,
                 gru_layers=1,
                 backbone_dropout=0.1,
                 causal=True,
                 # ASR
                 vocab_size=6000,
                 asr_hidden_dim=320,
                 blank_id=0,
                 # Dialect
                 num_dialects=10,
                 dialect_hidden_dim=128,
                 # Speaker
                 embed_dim=256,
                 num_speakers=1000,
                 speaker_dropout=0.1,
                 ):
        super().__init__()

        # ===== Shared Backbone =====
        self.backbone = SharedBackbone(
            input_dim=1,
            frontend_channels=frontend_channels,
            output_dim=backbone_dim,
            num_blocks=num_blocks,
            conv_kernel=conv_kernel,
            gru_layers=gru_layers,
            dropout=backbone_dropout,
            causal=causal,
        )

        # ===== Task 1: ASR Branch =====
        self.asr_branch = ASRBranch(
            input_dim=backbone_dim,
            hidden_dim=asr_hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
        )

        # ===== Task 2: Dialect Classification Branch =====
        self.dialect_branch = DialectBranch(
            input_dim=backbone_dim,
            num_dialects=num_dialects,
            hidden_dim=dialect_hidden_dim,
        )

        # ===== Task 3: Speaker Embedding Branch =====
        self.speaker_branch = SpeakerBranch(
            input_dim=backbone_dim,
            embed_dim=embed_dim,
            num_speakers=num_speakers,
            dropout=speaker_dropout,
        )

        self.backbone_dim = backbone_dim
        self.vocab_size = vocab_size
        self.num_dialects = num_dialects
        self.embed_dim = embed_dim
        self.blank_id = blank_id

    def forward(self, audio, audio_lengths=None,
                asr_labels=None, asr_label_lengths=None,
                dialect_labels=None, speaker_labels=None,
                task_mask=None):
        """
        多任务前向传播

        Args:
            audio: [B, 1, T_audio] 原始音频
            audio_lengths: [B] 音频采样点长度
            asr_labels: [B, max_text_len] ASR 标签 (token ids)
            asr_label_lengths: [B] ASR 标签长度
            dialect_labels: [B] 方言标签
            speaker_labels: [B] 说话人标签
            task_mask: dict {'asr': bool, 'dialect': bool, 'speaker': bool}
                       控制哪些任务参与本次前向

        Returns:
            outputs: dict 包含各任务输出
        """
        if task_mask is None:
            task_mask = {'asr': True, 'dialect': True, 'speaker': True}

        outputs = {}

        # ===== Shared Backbone =====
        features, feat_lengths = self.backbone(audio, audio_lengths)

        # ===== ASR Branch =====
        if task_mask.get('asr', False):
            asr_log_probs, _ = self.asr_branch(features, feat_lengths)
            outputs['asr_log_probs'] = asr_log_probs  # [T, B, V]

        # ===== Dialect Branch =====
        if task_mask.get('dialect', False):
            dialect_logits = self.dialect_branch(features, feat_lengths)
            outputs['dialect_logits'] = dialect_logits  # [B, num_dialects]

        # ===== Speaker Branch =====
        if task_mask.get('speaker', False):
            speaker_embedding = self.speaker_branch(features, feat_lengths, speaker_labels)
            if isinstance(speaker_embedding, tuple):
                speaker_embedding, speaker_loss = speaker_embedding
                outputs['speaker_embedding'] = speaker_embedding
                outputs['speaker_loss'] = speaker_loss
            else:
                outputs['speaker_embedding'] = speaker_embedding

        outputs['features'] = features
        outputs['feat_lengths'] = feat_lengths

        return outputs

    def freeze_backbone_layers(self, num_layers_to_freeze=2):
        """
        冻结主干网络的前 N 层 (渐进式微调策略)

        Args:
            num_layers_to_freeze: 冻结前几个 TinyConformer block
        """
        frozen_count = 0

        # 冻结 frontend
        for param in self.backbone.frontend.parameters():
            param.requires_grad = False
            frozen_count += 1

        # 冻结前 N 个 Conformer blocks
        for i, block in enumerate(self.backbone.blocks):
            if i < num_layers_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False
                frozen_count += 1

        print(f"[Model] Frozen: frontend + {num_layers_to_freeze} conformer blocks")

    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] All parameters unfrozen")

    def get_param_counts(self):
        """统计各模块参数量"""
        counts = {
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'asr_branch': sum(p.numel() for p in self.asr_branch.parameters()),
            'dialect_branch': sum(p.numel() for p in self.dialect_branch.parameters()),
            'speaker_branch': sum(p.numel() for p in self.speaker_branch.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts

    def summary(self):
        """打印模型结构摘要"""
        param_counts = self.get_param_counts()
        print("=" * 60)
        print("Multi-Task Voice Model Summary")
        print("=" * 60)
        print(f"  Backbone dim:         {self.backbone_dim}")
        print(f"  Vocab size:           {self.vocab_size}")
        print(f"  Num dialects:         {self.num_dialects}")
        print(f"  Speaker embed dim:    {self.embed_dim}")
        print("-" * 60)
        for name, count in param_counts.items():
            print(f"  {name:25s}: {count:>10,} params ({count/1e6:.2f}M)")
        print(f"  {'TOTAL':25s}: {param_counts['total']:>10,} params ({param_counts['total']/1e6:.2f}M)")
        print("=" * 60)
        print(f"  Subsampling rate:     4x (100Hz → 25Hz)")
        print(f"  Frontend receptive:   ~400 samples (25ms @ 16kHz)")
        print(f"  Output frame rate:    25Hz")
        print(f"  Causal (streaming):   {self.backbone.causal}")
        print("=" * 60)


def create_model(config=None):
    """
    工厂函数: 根据配置创建模型

    Args:
        config: 模型配置字典, 为 None 则使用默认值
    Returns:
        model: MultiTaskVoiceModel
    """
    if config is None:
        config = get_default_config()

    return MultiTaskVoiceModel(
        frontend_channels=config.get('frontend_channels', (64, 128, 256)),
        backbone_dim=config.get('backbone_dim', 256),
        num_blocks=config.get('num_blocks', 4),
        conv_kernel=config.get('conv_kernel', 31),
        gru_layers=config.get('gru_layers', 1),
        backbone_dropout=config.get('backbone_dropout', 0.1),
        causal=config.get('causal', True),
        vocab_size=config.get('vocab_size', 6000),
        asr_hidden_dim=config.get('asr_hidden_dim', 320),
        blank_id=config.get('blank_id', 0),
        num_dialects=config.get('num_dialects', 10),
        dialect_hidden_dim=config.get('dialect_hidden_dim', 128),
        embed_dim=config.get('embed_dim', 256),
        num_speakers=config.get('num_speakers', 1000),
        speaker_dropout=config.get('speaker_dropout', 0.1),
    )


def get_default_config():
    """默认配置 (~4.5M 参数)"""
    return {
        # Backbone
        'frontend_channels': (48, 96, 192),
        'backbone_dim': 192,
        'num_blocks': 4,
        'conv_kernel': 31,
        'gru_layers': 1,
        'backbone_dropout': 0.1,
        'causal': True,
        # ASR
        'vocab_size': 5000,
        'asr_hidden_dim': 256,
        'blank_id': 0,
        # Dialect
        'num_dialects': 10,
        'dialect_hidden_dim': 128,
        # Speaker
        'embed_dim': 192,
        'num_speakers': 1000,
        'speaker_dropout': 0.1,
    }
