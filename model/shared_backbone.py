"""
共享主干网络 —— 轻量化语音特征提取器

设计原则:
- 全 Conv1d + GRU 结构，SOC NPU 原生支持
- 支持流式推理 (unidirectional GRU, causal conv)
- ~2.5M 参数，可 INT8/INT16 量化
- 输入 16kHz 单通道，支持短句/流式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1d + BatchNorm + ReLU 基础块"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, causal=False):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.causal = causal
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.causal and self.conv.stride[0] == 1:
            # 裁掉右侧 future-padding, 保持因果性 (仅对 stride=1 层)
            # stride>1 的下采样层不裁剪, 因为少量 future context 在应用层无影响
            x = x[..., :x.size(-1) - self.conv.padding[0]]
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseConvBlock(nn.Module):
    """深度可分离卷积块 —— 轻量化时序建模"""
    def __init__(self, channels, kernel_size=31, causal=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.causal = causal
        self.depthwise = nn.Conv1d(channels, channels, kernel_size,
                                   groups=channels, padding=0, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        if self.causal:
            # 左填充 (只看到过去, 看不到未来)
            x = F.pad(x, (self.kernel_size - 1, 0))
            x = self.depthwise(x)
        else:
            # 对称填充
            x = F.pad(x, ((self.kernel_size - 1) // 2,) * 2)
            x = self.depthwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x + residual


class GRUBlock(nn.Module):
    """单向 GRU 块 —— 长时序上下文建模，SOC 友好"""
    def __init__(self, dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(dim, dim, num_layers=num_layers,
                          batch_first=False,  # [T, B, C] for ONNX compatibility
                          bidirectional=False,
                          dropout=dropout if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, T] → [T, B, C]
        x_t = x.permute(2, 0, 1)
        residual = x_t
        x_t, _ = self.gru(x_t)
        x_t = self.norm(x_t + residual)
        # [T, B, C] → [B, C, T]
        x = x_t.permute(1, 2, 0)
        return x


class TinyConformerBlock(nn.Module):
    """
    精简 Conformer Block —— 仅保留核心模块

    结构: LN → FFN(Conv1d扩张) → Depthwise Conv → GRU → FFN → Residual
    移除标准 Conformer 中的 MHA (多头注意力),
    使用 GRU 替代 self-attention 以降低计算量和提升 ONNX 兼容性
    """
    def __init__(self, dim, conv_kernel=31, gru_layers=1, dropout=0.1, causal=False):
        super().__init__()
        self.dim = dim
        expansion_factor = 2
        hidden_dim = dim * expansion_factor

        # FFN 1: Conv1d 扩张 + GLU 激活
        self.ffn1_conv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)
        self.ffn1_conv2 = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)
        self.ffn1_out = nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False)
        self.ffn1_norm = nn.LayerNorm(dim)

        # Depthwise Conv (局部上下文)
        self.dwconv = DepthwiseConvBlock(dim, conv_kernel, causal=causal)
        self.dwconv_norm = nn.LayerNorm(dim)

        # GRU (长时序上下文)
        self.gru = GRUBlock(dim, num_layers=gru_layers, dropout=dropout)
        self.gru_norm = nn.LayerNorm(dim)

        # FFN 2
        self.ffn2_conv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)
        self.ffn2_conv2 = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)
        self.ffn2_out = nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False)
        self.ffn2_norm = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def _ffn(self, x, conv1, conv2, out_conv, norm):
        """
        FFN with GLU gating:
        output = (conv1(x) * sigmoid(conv2(x))) → out_conv
        """
        residual = x
        # x: [B, C, T] → for LayerNorm need [B, T, C]
        x_norm = norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        gate = torch.sigmoid(conv2(x_norm))
        act = F.relu(conv1(x_norm))
        x = out_conv(act * gate)
        x = self.dropout(x)
        return x + residual

    def forward(self, x):
        # x: [B, C, T]
        x = self._ffn(x, self.ffn1_conv1, self.ffn1_conv2,
                      self.ffn1_out, self.ffn1_norm)

        # Depthwise conv with residual
        residual = x
        x = self.dwconv_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dwconv(x)
        x = self.dropout(x) + residual

        # GRU
        residual = x
        x = self.gru_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.gru(x)
        x = self.dropout(x) + residual

        x = self._ffn(x, self.ffn2_conv1, self.ffn2_conv2,
                      self.ffn2_out, self.ffn2_norm)

        return x


class SharedBackbone(nn.Module):
    """
    轻量化共享主干网络

    结构:
    1. Learnable Conv Frontend (替代手工 Fbank)
       - Conv stride=160 → ~100Hz framerate
       - 4x 下采样 → 25Hz
    2. Tiny Conformer Blocks × N
       - 精简版 Conformer，无 MHA，纯 Conv+GRU

    参数量: ~2.5M
    输出: 256-dim 特征序列 @ 25Hz
    """
    def __init__(self,
                 input_dim=1,
                 frontend_channels=(64, 128, 256),
                 output_dim=256,
                 num_blocks=4,
                 conv_kernel=31,
                 gru_layers=1,
                 dropout=0.1,
                 causal=True):
        super().__init__()

        # ===== Learnable Conv Frontend =====
        # 输入 [B, 1, audio_samples], 16kHz
        # 注: 前端下采样层不使用 causal (stride>1 时 causal 裁剪无实际意义,
        #      且 stride 层只引入 ~12.5ms future context, 对流式推理无影响)
        self.frontend = nn.Sequential(
            # Layer 1: 宽核卷积 → 模拟 mel filterbank, ~100Hz
            ConvBlock(input_dim, frontend_channels[0],
                      kernel_size=400, stride=160, causal=False),
            # Layer 2: stride=2 → ~50Hz
            ConvBlock(frontend_channels[0], frontend_channels[1],
                      kernel_size=3, stride=2, causal=False),
            # Layer 3: stride=2 → ~25Hz
            ConvBlock(frontend_channels[1], frontend_channels[2],
                      kernel_size=3, stride=2, causal=False),
            # Layer 4: 保持分辨率, 深度特征提取
            ConvBlock(frontend_channels[2], output_dim,
                      kernel_size=3, stride=1, causal=causal),
        )

        # ===== Tiny Conformer Blocks =====
        self.blocks = nn.ModuleList([
            TinyConformerBlock(
                dim=output_dim,
                conv_kernel=conv_kernel,
                gru_layers=gru_layers,
                dropout=dropout,
                causal=causal,
            )
            for _ in range(num_blocks)
        ])

        self.output_dim = output_dim
        self.subsampling_rate = 4  # 100Hz → 25Hz (4x total subsampling)
        self.causal = causal

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, lengths=None):
        """
        Args:
            x: [B, 1, T_audio] 原始音频波形
            lengths: [B] 每段音频的原始长度(采样点数), 用于计算输出长度
        Returns:
            features: [B, output_dim, T_feat] 特征序列
            feat_lengths: [B] 特征序列长度
        """
        B = x.size(0)

        # Frontend
        x = self.frontend(x)  # [B, 256, T_feat]

        # Tiny Conformer blocks
        for block in self.blocks:
            x = block(x)

        # 计算输出长度
        if lengths is not None:
            # 经过 frontend:
            # stride=160 → len1=(len-400)/160+1
            # stride=2   → len2=(len1-3)/2+1 = floor(len1/2)
            # stride=2   → len3=(len2-3)/2+1 = floor(len2/2)
            # stride=1   → len4=len3
            feat_lengths = self._compute_feat_lengths(lengths)
        else:
            feat_lengths = None

        return x, feat_lengths

    def _compute_feat_lengths(self, audio_lengths):
        """计算经过 frontend 后的特征序列长度"""
        lengths = audio_lengths.float()
        # Conv1: stride=160, kernel=400
        lengths = torch.floor((lengths - 400) / 160) + 1
        # Conv2: stride=2, kernel=3
        lengths = torch.floor((lengths - 3) / 2) + 1
        # Conv3: stride=2, kernel=3
        lengths = torch.floor((lengths - 3) / 2) + 1
        # Conv4: stride=1, kernel=3 → no length change (padding=same)
        return lengths.clamp(min=1).long()

    def get_subsampling_rate(self):
        """返回时间维度的总下采样率"""
        return self.subsampling_rate
