"""
Stage 2: 命令词识别 (唤醒后运行)

支持两套方案:
  Plan A: 卷积分类器 (简单, 50条命令, 全加速)
  Plan B: 小型CTC声学模型 + WFST语法解码 (灵活, 200条命令)

AC7916AB:
  Plan A: 加速器上跑, ~17ms延迟, 50条命令以内
  Plan B: 加速器跑声学模型 + CPU跑WFST解码, ~14ms延迟, 200条命令
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    from .wfst_decoder import WFSTGrammarDecoder
except ImportError:
    from wfst_decoder import WFSTGrammarDecoder


# ===== Plan A: 卷积分类器 (训练版 CommandClassifierV2) =====
class CommandClassifierV2(nn.Module):
    """
    命令词分类器 (DS-CNN, 与 train_stage2_classifier.py 一致)

    192 类, ~300K 参数, 3 个 DS-Conv block
    卷积结构原生加速, 无 GRU, 无循环依赖
    """
    def __init__(self, num_classes=192, n_mels=40, n_frames=200):
        super().__init__()
        self.num_classes = num_classes

        # Stem: initial conv + stride
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )

        # Block 1: DS-Conv (32 → 64)
        self.dw1 = nn.Conv2d(32, 32, (3, 3), stride=(2, 2), padding=1,
                             groups=32, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pw1 = nn.Conv2d(32, 64, 1, bias=False)
        self.bn1b = nn.BatchNorm2d(64)

        # Block 2: DS-Conv (64 → 128)
        self.dw2 = nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1,
                             groups=64, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pw2 = nn.Conv2d(64, 128, 1, bias=False)
        self.bn2b = nn.BatchNorm2d(128)

        # Block 3: DS-Conv (128 → 128)
        self.dw3 = nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=1,
                             groups=128, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.pw3 = nn.Conv2d(128, 128, 1, bias=False)
        self.bn3b = nn.BatchNorm2d(128)

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = F.relu(self.bn1b(self.pw1(F.relu(self.bn1(self.dw1(x))))))
        x = F.relu(self.bn2b(self.pw2(F.relu(self.bn2(self.dw2(x))))))
        x = F.relu(self.bn3b(self.pw3(F.relu(self.bn3(self.dw3(x))))))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x


# Compat alias
CommandClassifier = CommandClassifierV2


# ===== Plan B: 小型 CTC 声学模型 =====
class CTCEncoder(nn.Module):
    """
    小型 CTC 声学模型 (可加速的 Conv 结构)

    输出: 每帧的子词后验概率 (支持 200-500 个输出单元)
    """
    def __init__(self, input_dim=40, hidden_dim=128, num_tokens=256,
                 num_layers=3):
        super().__init__()
        self.num_tokens = num_tokens

        # Frontend: 降采样 4x
        self.frontend = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Encoder blocks (膨胀 Conv 替代 GRU, 加速器友好)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.blocks.append(_DilatedConvBlock(hidden_dim, dilation))

        # Output projection
        self.output = nn.Conv1d(hidden_dim, num_tokens, 1)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, n_mels, T]  Mel features
        x = self.frontend(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.output(x)
        # CTC output: [B, num_tokens, T']
        return F.log_softmax(x, dim=1)


class _DilatedConvBlock(nn.Module):
    """膨胀卷积块 (替代 GRU, 卷积原生加速)"""
    def __init__(self, dim, dilation):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, 3, dilation=dilation,
                            padding=dilation, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm1d(dim)
        self.pw1 = nn.Conv1d(dim, dim * 2, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(dim * 2)
        self.pw2 = nn.Conv1d(dim * 2, dim, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw1(x)))
        x = self.bn3(self.pw2(x))
        return x + residual


class CommandRecognizer:
    """
    Stage 2 命令识别器 (封装 Plan A 和 Plan B)

    Plan A (classifier): 50 条命令以内, 简单
    Plan B (ctc+grammar): 200 条命令, 灵活
    """

    def __init__(self, plan='A', **kwargs):
        self.plan = plan
        self.model = None
        self.decoder = None
        self.id2cmd = {}  # class_id → command text

    def load_classifier(self, model: CommandClassifier,
                        id2cmd: Dict[int, str]):
        self.plan = 'A'
        self.model = model.eval()
        self.id2cmd = id2cmd

    def load_ctc(self, encoder: CTCEncoder,
                 decoder: WFSTGrammarDecoder,
                 id2token: Dict[int, str]):
        self.plan = 'B'
        self.encoder = encoder.eval()
        self.decoder = decoder
        self.id2token = id2token

    @torch.no_grad()
    def recognize(self, mel: torch.Tensor,
                  top_k: int = 3) -> List[Tuple[str, float]]:
        """
        识别命令

        Args:
            mel: Mel 特征 [1, n_mels, T_frames] 或 [1, 1, n_mels, T]
            top_k: 返回 top-k 候选

        Returns:
            [(command_text, confidence), ...]
        """
        if self.plan == 'A':
            return self._recognize_classifier(mel, top_k)
        else:
            return self._recognize_ctc(mel, top_k)

    def _recognize_classifier(self, mel, top_k):
        if mel.dim() == 3:
            mel = mel.unsqueeze(0)  # [1, 1, 40, T]

        logits = self.model(mel)  # [1, num_classes]
        probs = F.softmax(logits, dim=-1)

        topk_p, topk_i = torch.topk(probs[0], k=min(top_k, probs.size(-1)))

        results = []
        for i in range(len(topk_i)):
            cls_id = topk_i[i].item()
            conf = topk_p[i].item()
            cmd = self.id2cmd.get(cls_id, f'cmd_{cls_id}')
            results.append((cmd, conf))

        return results

    def _recognize_ctc(self, mel, top_k):
        # mel: [1, n_mels, T]
        log_probs = self.encoder(mel)  # [1, num_tokens, T']

        # 转 numpy for CTC decoder
        lp_np = log_probs[0].permute(1, 0).cpu().numpy()  # [T', num_tokens]

        # Grammar-constrained beam search
        candidates = self.decoder.decode(lp_np, top_k=top_k)

        results = []
        for token_ids, score in candidates:
            # 解码 token IDs 为命令文本
            text = ''.join(self.id2token.get(t, '?') for t in token_ids)
            conf = float(np.exp(score / max(len(token_ids), 1)))
            results.append((text, conf))

        return results

    def get_stats(self) -> dict:
        total = sum(p.numel() for p in
                    (self.encoder if self.plan == 'B' else self.model).parameters())
        return {
            'plan': self.plan,
            'params': total,
            'int8_kb': total / 1024,
            'macs_per_inference': 15e6 if self.plan == 'B' else 10e6,
            'mva_accelerated': True,  # both plans use Conv-only, accelerator compatible
        }


# Re-export for convenience
CTCCommandASR = None  # placeholder, use CTCEncoder + WFSTGrammarDecoder
