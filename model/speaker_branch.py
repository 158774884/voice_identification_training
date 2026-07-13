"""
声纹说话人嵌入分支 —— 轻量化 TDNN/ECAPA 精简版

设计:
- 精简 TDNN (Time Delay Neural Network) 用于说话人特征提取
- 注意力统计池化 → 256-dim 声纹嵌入
- AAM-Softmax (Additive Angular Margin) 用于训练
- 推理时输出 L2 归一化嵌入向量
- 支持 1:1 余弦相似度比对 + 开集验证

参数量: ~800K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TDNNBlock(nn.Module):
    """
    TDNN 一层: Conv1d + BN + ReLU

    TDNN 本质是在时间轴上做 dilated 卷积,
    等价于跳过固定帧的上下文窗口
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation,
                              padding=(kernel_size - 1) * dilation // 2,
                              bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block —— 通道注意力

    轻量化实现, 仅需 ~2K 参数
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        bottleneck = channels // reduction
        self.fc1 = nn.Linear(channels, bottleneck)
        self.fc2 = nn.Linear(bottleneck, channels)

    def forward(self, x):
        # x: [B, C, T]
        # Squeeze: global avg pooling
        se = x.mean(dim=2)  # [B, C]
        # Excitation
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        # Scale
        return x * se.unsqueeze(2)


class SpeakerEmbedding(nn.Module):
    """
    精简 TDNN + 统计池化 → 声纹嵌入

    结构:
    - TDNN blocks (多层递增 dilation)
    - SE 通道注意力
    - 注意力统计池化
    - FC → 256-dim 嵌入
    """
    def __init__(self, input_dim=256, embed_dim=256, dropout=0.1):
        super().__init__()

        # TDNN 特征提取 (逐步增大 dilation 扩大感受野)
        self.tdnn1 = TDNNBlock(input_dim, 256, kernel_size=5, dilation=1, dropout=dropout)
        self.se1 = SEBlock(256)

        self.tdnn2 = TDNNBlock(256, 256, kernel_size=3, dilation=2, dropout=dropout)
        self.se2 = SEBlock(256)

        self.tdnn3 = TDNNBlock(256, 256, kernel_size=3, dilation=3, dropout=dropout)
        self.se3 = SEBlock(256)

        self.tdnn4 = TDNNBlock(256, 256, kernel_size=1, dilation=1, dropout=dropout)

        # 注意力统计池化
        self.attn_pool = AttentiveStatsPool(256, bottleneck_dim=128)

        # 嵌入层: [mean; std] (512-dim) → embed_dim
        pooled_dim = 256 * 2
        self.embed_fc = nn.Sequential(
            nn.Linear(pooled_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        self.embed_dim = embed_dim

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

    def forward(self, features, feat_lengths=None, return_embedding=True):
        """
        Args:
            features: [B, C, T]
            feat_lengths: [B]
            return_embedding: 是否返回嵌入 (训练时需要 logits for AAM-Softmax)
        Returns:
            embedding: [B, embed_dim] L2 归一化嵌入向量
        """
        # TDNN
        x = self.tdnn1(features)
        x = self.se1(x)

        x = self.tdnn2(x)
        x = self.se2(x)

        x = self.tdnn3(x)
        x = self.se3(x)

        x = self.tdnn4(x)

        # 注意力统计池化
        pooled = self.attn_pool(x, feat_lengths)  # [B, C*2]

        # 嵌入
        embedding = self.embed_fc(pooled)  # [B, embed_dim]

        # L2 归一化 (声纹嵌入标准做法)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class AttentiveStatsPool(nn.Module):
    """
    注意力统计池化 (用于声纹分支)

    对时间维度的帧做注意力加权, 输出 [weighted_mean; weighted_std]
    """
    def __init__(self, input_dim, bottleneck_dim=128):
        super().__init__()
        self.bottleneck = nn.Conv1d(input_dim, bottleneck_dim, kernel_size=1)
        self.attention = nn.Conv1d(bottleneck_dim, input_dim, kernel_size=1)

    def forward(self, x, lengths=None):
        """
        Args:
            x: [B, C, T]
            lengths: [B] (optional)
        Returns:
            pooled: [B, C*2]
        """
        # 计算注意力权重
        attn = self.bottleneck(x)  # [B, 128, T]
        attn = torch.tanh(attn)
        attn = self.attention(attn)  # [B, C, T]

        # Length mask
        if lengths is not None:
            max_len = x.size(2)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0).unsqueeze(1) >= lengths.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn, dim=2)  # [B, C, T]

        # 加权均值
        mean = torch.sum(x * attn_weights, dim=2)  # [B, C]

        # 加权标准差
        mean_sq = torch.sum(x * x * attn_weights, dim=2)
        var = mean_sq - mean * mean
        var = torch.clamp(var, min=1e-8)
        std = torch.sqrt(var)

        return torch.cat([mean, std], dim=1)  # [B, C*2]


class AAMSoftmaxLoss(nn.Module):
    """
    Additive Angular Margin Softmax Loss

    用于训练声纹嵌入, 增大类间间距, 压缩类内间距

    公式: cos(θ + m) 替代标准 Softmax 中的 cos(θ)
    实际计算使用: s * (cosθ * cosm - sinθ * sinm)

    Args:
        embed_dim: 嵌入维度
        num_speakers: 训练集中说话人数
        margin: 角度 margin (推荐 0.2-0.3)
        scale: 缩放因子 (推荐 30.0)
    """
    def __init__(self, embed_dim, num_speakers, margin=0.2, scale=30.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_speakers = num_speakers
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        # 分类权重
        self.weight = nn.Parameter(torch.FloatTensor(num_speakers, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding, labels):
        """
        Args:
            embedding: [B, embed_dim] L2 归一化嵌入
            labels: [B] 说话人标签
        Returns:
            loss: scalar
        """
        # Cosine similarity: [B, num_speakers]
        cosine = F.linear(embedding, F.normalize(self.weight, p=2, dim=1))
        cosine = cosine.clamp(-1, 1)

        sine = torch.sqrt(1.0 - cosine ** 2).clamp(0, 1)

        # cos(θ + m) = cosθ * cosm - sinθ * sinm
        phi = cosine * self.cos_m - sine * self.sin_m

        # 只对正类添加 margin
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output * self.scale

        loss = F.cross_entropy(output, labels)

        return loss


class SpeakerBranch(nn.Module):
    """
    声纹分支: 训练时输出 AAM-Softmax logits
             推理时输出 L2 归一化嵌入向量
    """
    def __init__(self, input_dim=256, embed_dim=256, num_speakers=1000, dropout=0.1):
        super().__init__()

        self.embedding = SpeakerEmbedding(input_dim, embed_dim, dropout)
        self.embed_dim = embed_dim

        # 训练时用于 AAM-Softmax 的分类权重
        self.aam_loss = None  # 由训练器设置

    def forward(self, features, feat_lengths=None, labels=None):
        """
        Args:
            features: [B, C, T]
            feat_lengths: [B]
            labels: [B] speaker ids (training only)
        Returns:
            embedding: [B, embed_dim]
            loss: AAM-Softmax loss (if labels provided)
        """
        embedding = self.embedding(features, feat_lengths)

        if labels is not None and self.aam_loss is not None:
            loss = self.aam_loss(embedding, labels)
            return embedding, loss

        return embedding

    def extract_embedding(self, features, feat_lengths=None):
        """仅提取嵌入向量 (推理用)"""
        return self.embedding(features, feat_lengths)

    def set_aam_loss(self, num_speakers, margin=0.2, scale=30.0):
        """设置 AAM-Softmax 损失"""
        self.aam_loss = AAMSoftmaxLoss(self.embed_dim, num_speakers, margin, scale)


def cosine_similarity(emb1, emb2):
    """
    余弦相似度

    Args:
        emb1, emb2: [B, D] or [D] L2 归一化嵌入
    Returns:
        similarity: [B] or scalar, range [-1, 1]
    """
    # 嵌入应该已经 L2 归一化, 所以余弦相似度 = 内积
    if emb1.dim() == 1:
        emb1 = emb1.unsqueeze(0)
    if emb2.dim() == 1:
        emb2 = emb2.unsqueeze(0)

    # 确保 L2 归一化
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)

    similarity = (emb1 * emb2).sum(dim=1)

    return similarity


def speaker_verification(enroll_embedding, test_embedding, threshold=0.6):
    """
    声纹 1:1 验证

    Args:
        enroll_embedding: [D] 注册说话人嵌入
        test_embedding: [D] 测试说话人嵌入
        threshold: 判定阈值 (需要根据验证集调优)
    Returns:
        is_same_speaker: bool
        similarity: float
    """
    sim = cosine_similarity(enroll_embedding, test_embedding)

    if sim.dim() > 0:
        sim = sim.item()

    is_same = sim >= threshold
    return is_same, sim
