"""
方言 / 口音分类分支

支持分类:
- 普通话 (Mandarin)
- 粤语 (Cantonese)
- 川渝话 (Southwestern Mandarin / Sichuanese)
- 吴语 (Wu / Shanghainese)
- 闽南语 (Hokkien / Min Nan)
- 客家话 (Hakka)
- 湘语 (Xiang)
- 赣语 (Gan)
- 其他方言 (可扩展)

设计:
- 注意力统计池化 (Attentive Statistics Pooling)
- 2 层 FC 分类器
- 参数量: ~50K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveStatisticsPooling(nn.Module):
    """
    注意力统计池化

    对时间维度做加权平均和加权标准差，
    输出 [mean; std] 拼接形成的固定维度表征
    """
    def __init__(self, input_dim, bottleneck_dim=128):
        super().__init__()
        self.bottleneck = nn.Linear(input_dim, bottleneck_dim)
        self.attention = nn.Linear(bottleneck_dim, 1)

    def forward(self, x, lengths=None):
        """
        Args:
            x: [B, C, T]
            lengths: [B] 有效长度 (可选, 用于 mask)
        Returns:
            pooled: [B, C * 2]  (mean; std 拼接)
        """
        x_t = x.permute(0, 2, 1)  # [B, T, C]

        # 注意力权重
        attn_input = self.bottleneck(x_t)  # [B, T, bottleneck]
        attn_input = torch.tanh(attn_input)
        attn_weights = self.attention(attn_input).squeeze(-1)  # [B, T]

        # 长度 mask
        if lengths is not None:
            max_len = x_t.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # [B, 1, T]

        # 加权均值
        mean = torch.bmm(attn_weights, x_t).squeeze(1)  # [B, C]

        # 加权标准差
        x_t_sq = x_t * x_t
        mean_sq = torch.bmm(attn_weights, x_t_sq).squeeze(1)  # [B, C]
        var = mean_sq - mean * mean
        var = torch.clamp(var, min=1e-8)
        std = torch.sqrt(var)

        # 拼接
        pooled = torch.cat([mean, std], dim=1)  # [B, C*2]

        return pooled


class DialectBranch(nn.Module):
    """
    方言/口音分类分支

    支持开集分类，输出 softmax 概率
    """
    def __init__(self,
                 input_dim=256,
                 num_dialects=10,
                 hidden_dim=128,
                 bottleneck_dim=128,
                 dropout=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.num_dialects = num_dialects

        # 注意力统计池化 → 固定维度
        self.pooling = AttentiveStatisticsPooling(input_dim, bottleneck_dim)
        pooled_dim = input_dim * 2  # mean + std

        # 分类器 (LayerNorm 兼容 batch_size=1, 更适合 ONNX 部署)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_dialects),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features, feat_lengths=None):
        """
        Args:
            features: [B, input_dim, T_feat]
            feat_lengths: [B] (optional)
        Returns:
            logits: [B, num_dialects]
        """
        # Attentive statistics pooling
        pooled = self.pooling(features, feat_lengths)  # [B, input_dim * 2]

        # 分类
        logits = self.classifier(pooled)  # [B, num_dialects]

        return logits

    def predict(self, features, feat_lengths=None):
        """推理: 返回 top-k 预测结果"""
        logits = self.forward(features, feat_lengths)
        probs = F.softmax(logits, dim=-1)

        # 返回概率最高的 3 个类别及其概率
        topk_probs, topk_indices = torch.topk(probs, k=min(3, self.num_dialects), dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'topk_indices': topk_indices,
            'topk_probs': topk_probs,
        }
