"""
ASR 分支 —— CTC 中文+多方言语音识别

设计:
- 输入: 共享主干输出的 256-dim 特征序列
- 1-2 层 Conv1d + Linear 投影到字符空间
- CTC 解码 (greedy / beam search)
- 参数量: ~1.3M (含 5000+ 字符词典)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ASRBranch(nn.Module):
    """
    CTC-based ASR 分支

    支持:
    - 中文汉字 + 常用标点
    - 方言特有字符
    - Greedy / Beam Search 解码
    """
    def __init__(self,
                 input_dim=256,
                 hidden_dim=320,
                 vocab_size=6000,
                 num_conv_layers=2,
                 conv_kernel=5,
                 dropout=0.1,
                 blank_id=0):
        super().__init__()

        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.blank_id = blank_id

        # Conv 投影层 (平滑特征)
        convs = OrderedDict()
        for i in range(num_conv_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            convs[f'conv{i}'] = nn.Conv1d(in_dim, hidden_dim, conv_kernel,
                                          stride=1, padding=conv_kernel // 2, bias=False)
            convs[f'bn{i}'] = nn.BatchNorm1d(hidden_dim)
            convs[f'relu{i}'] = nn.ReLU(inplace=True)
            if dropout > 0:
                convs[f'dropout{i}'] = nn.Dropout(dropout)
        self.conv_stack = nn.Sequential(convs)

        # 输出投影: hidden_dim → vocab_size
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features, feat_lengths=None):
        """
        Args:
            features: [B, input_dim, T_feat] 共享主干输出
            feat_lengths: [B] 特征长度
        Returns:
            log_probs: [T, B, vocab_size] log-softmax 输出 (CTC 需要)
            feat_lengths: [B] 经过 conv 投影后的长度 (与输入相同, 因为 padding=same)
        """
        # Conv 投影: [B, C, T] → [B, hidden_dim, T]
        x = self.conv_stack(features)

        # 投影到字符空间: [B, T, hidden_dim] → [B, T, vocab_size]
        x = x.permute(0, 2, 1)  # [B, T, C]
        logits = self.output_proj(x)  # [B, T, vocab_size]

        # CTC 需要 [T, B, vocab_size] 格式的 log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)  # [T, B, vocab_size]

        return log_probs, feat_lengths

    def greedy_decode(self, log_probs):
        """
        Greedy CTC 解码
        Args:
            log_probs: [T, B, vocab_size] log-probabilities
        Returns:
            decoded: List[List[int]] 每个样本的解码 token ids
        """
        # [T, B, V] → 每个时间步取最大概率
        best_tokens = log_probs.argmax(dim=-1)  # [T, B]

        # 转置为 [B, T] 方便处理
        best_tokens = best_tokens.permute(1, 0)  # [B, T]

        decoded = []
        for bt in best_tokens:
            # 合并连续重复, 去除 blank
            result = []
            prev = self.blank_id
            for token in bt:
                token = token.item()
                if token != prev and token != self.blank_id:
                    result.append(token)
                prev = token
            decoded.append(result)

        return decoded

    def beam_search_decode(self, log_probs, beam_width=5, blank_id=None):
        """
        CTC Beam Search 解码 (简化版)

        Args:
            log_probs: [T, 1, vocab_size] 单个样本
            beam_width: beam 宽度
            blank_id: blank token id
        Returns:
            best_sequences: List[tuple(tokens, score)]
        """
        if blank_id is None:
            blank_id = self.blank_id

        # 简化实现: 返回 top beam_width 个 greedy 附近的解
        T, B, V = log_probs.shape
        log_probs = log_probs[:, 0, :]  # [T, V]

        # 初始 beam
        beams = [([], 0.0, blank_id)]  # (tokens, score, last_token)

        for t in range(T):
            frame_log_probs = log_probs[t]  # [V]
            new_beams = []

            for tokens, score, last_token in beams:
                topk_scores, topk_ids = frame_log_probs.topk(beam_width)

                for k in range(beam_width):
                    token = topk_ids[k].item()
                    new_score = score + topk_scores[k].item()
                    new_tokens = tokens[:]

                    if token != last_token and token != blank_id:
                        new_tokens.append(token)

                    new_beams.append((new_tokens, new_score, token))

            # 保留 top beam_width
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        return [(tokens, score) for tokens, score, _ in beams]
