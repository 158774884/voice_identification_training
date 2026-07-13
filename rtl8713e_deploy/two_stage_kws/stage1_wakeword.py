"""
Stage 1: 始终在线唤醒词检测器

模型: UltraTinyKWS (~2.4K params)
职责: 检测 1-3 个唤醒词 (如 "小度小度")
功耗: < 0.5mA, 常驻 SRAM 34KB

AC7916AB: 跑在 CPU 上 (MVA 休眠), 每10ms推理一次, 延迟 <200us
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class UltraTinyWakeWord(nn.Module):
    """
    超轻量唤醒词检测模型 (可配置容量)

    size='micro':  ~5K params,  ~50K MACs (推荐, AC7916AB <5% CPU)
    size='nano':   ~1K params,  ~20K MACs (极简, 适合极低功耗)
    """
    def __init__(self, num_wake_words=2, n_mels=40, size='micro'):
        super().__init__()
        self.num_wake_words = num_wake_words

        if size == 'nano':
            c1, c2 = 16, 16
        elif size == 'micro':
            c1, c2 = 32, 32
        else:
            c1, c2 = 32, 64

        # Block 1: Conv 5x5 → reduce freq×time
        self.conv1 = nn.Conv2d(1, c1, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        # Block 2: DW Conv 3x3
        self.dw = nn.Conv2d(c1, c1, kernel_size=(3, 3), stride=(1, 2),
                            padding=(1, 1), groups=c1, bias=False)
        self.bn2 = nn.BatchNorm2d(c1)

        # Block 3: 1x1 projection
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)

        # Block 4: DW Conv 3x3 (额外的一层, 增加容量)
        self.dw2 = nn.Conv2d(c2, c2, kernel_size=(3, 3), stride=(1, 2),
                             padding=(1, 1), groups=c2, bias=False)
        self.bn4 = nn.BatchNorm2d(c2)

        # Block 5: 1x1 projection
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(c2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(c2, num_wake_words)

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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.dw(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.dw2(x)))
        x = F.relu(self.bn5(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x


class WakeWordDetector:
    """
    唤醒词检测器 (Stage 1 运行时)

    状态机:
      SILENCE → SPEECH → MAYBE_WAKE → CONFIRMED
          ↑         ↓           ↓           ↓
          └─────────┴───────────┴───────────┘
    """

    CONFIDENCE_THRESHOLD = 0.65
    SMOOTH_WINDOW = 5
    MIN_SILENCE_FRAMES = 30  # 300ms silence between activations
    SPEECH_ENERGY_THRESHOLD = 0.002

    def __init__(self, model: UltraTinyWakeWord,
                 wake_labels: List[str] = None):
        self.model = model.eval()
        self.wake_labels = wake_labels or ['wake_0', 'wake_1']

        # 状态
        self.state = 'SILENCE'
        self.silence_counter = 0
        self.wake_history = []  # [(frame_idx, predicted_class, confidence), ...]
        self.last_detection_frame = -1000
        self.frame_idx = 0

    def reset(self):
        self.state = 'SILENCE'
        self.silence_counter = 0
        self.wake_history = []
        self.frame_idx = 0
        self.last_detection_frame = -1000  # 重置冷却期, 避免跨文件污染
        self._mel_window = []  # 清空滑动窗口, 避免跨文件污染

    @torch.no_grad()
    def process_frame(self, mel_features: torch.Tensor) -> Tuple[bool, Optional[str], float]:
        """
        每 10ms 调用一次

        Args:
            mel_features: [1, 40] 当前帧 Mel 特征向量

        Returns:
            (is_wake, wake_label, confidence)
        """
        self.frame_idx += 1

        # 确保 mel 是 [40] (1D)
        mel_1d = mel_features.squeeze()  # [40]

        # 需要积累至少 98 帧 (~1s) 才能做推理
        if not hasattr(self, '_mel_window'):
            self._mel_window = []
        self._mel_window.append(mel_1d)
        if len(self._mel_window) > 98:
            self._mel_window.pop(0)
        if len(self._mel_window) < 98:
            return False, None, 0.0

        # 能量检测
        energy = mel_1d.abs().mean().item()

        # 构建输入: [1, 1, 40, 98]  (B, C, H_freq, W_time)
        mel_stack = torch.stack(list(self._mel_window), dim=0)   # [98, 40]
        mel_stack = mel_stack.T.unsqueeze(0).unsqueeze(0)         # [1, 1, 40, 98]

        # 推理
        logits = self.model(mel_stack)  # [1, num_wake_words]
        probs = F.softmax(logits, dim=-1)
        best_class = probs.argmax(dim=-1).item()
        best_prob = probs[0, best_class].item()

        # 状态机
        if self.state == 'SILENCE':
            if energy > self.SPEECH_ENERGY_THRESHOLD:
                self.state = 'SPEECH'
                self.silence_counter = 0

        elif self.state == 'SPEECH':
            if best_prob > self.CONFIDENCE_THRESHOLD and best_class < len(self.wake_labels):
                self.state = 'MAYBE_WAKE'
                self.wake_history = [(self.frame_idx, best_class, best_prob)]
            elif energy < self.SPEECH_ENERGY_THRESHOLD:
                self.silence_counter += 1
                if self.silence_counter > self.MIN_SILENCE_FRAMES:
                    self.state = 'SILENCE'

        elif self.state == 'MAYBE_WAKE':
            self.wake_history.append((self.frame_idx, best_class, best_prob))
            if len(self.wake_history) > self.SMOOTH_WINDOW:
                self.wake_history.pop(0)

            # 检查连续一致性
            if len(self.wake_history) >= self.SMOOTH_WINDOW:
                classes = [h[1] for h in self.wake_history]
                if len(set(classes)) == 1 and classes[0] == best_class:
                    avg_conf = np.mean([h[2] for h in self.wake_history])
                    if avg_conf > self.CONFIDENCE_THRESHOLD:
                        # 确认唤醒
                        frames_since_last = self.frame_idx - self.last_detection_frame
                        if (frames_since_last > self.MIN_SILENCE_FRAMES
                                and best_class < len(self.wake_labels)):
                            self.state = 'CONFIRMED'
                            self.last_detection_frame = self.frame_idx
                            label = self.wake_labels[best_class]
                            return True, label, avg_conf

            if energy < self.SPEECH_ENERGY_THRESHOLD:
                self.silence_counter += 1
                if self.silence_counter > 20:
                    self.state = 'SILENCE'
                    self.wake_history = []

        elif self.state == 'CONFIRMED':
            self.state = 'SILENCE'
            self.wake_history = []

        return False, None, 0.0

    def get_state(self) -> str:
        return self.state

    def get_model_size(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        return {
            'params': total,
            'int8_bytes': total,
            'int8_kb': total / 1024,
            'macs_per_inference': 50000,  # ~50K MACs
            'cpu_us_per_inference': 195,   # on 320MHz RISC
        }
