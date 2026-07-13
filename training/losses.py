"""
多任务联合损失函数

Total Loss = λ_asr * CTC_Loss
           + λ_dialect * CrossEntropy Loss
           + λ_speaker * AAM-Softmax Loss

支持:
- 动态任务权重调整
- 单任务训练 (某任务 loss_weight=0 则跳过)
- 梯度缩放
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    多任务联合损失

    配置:
        asr_weight:     CTC 损失权重
        dialect_weight: 方言分类损失权重
        speaker_weight: 声纹 AAM-Softmax 损失权重
    """

    def __init__(self,
                 asr_weight=1.0,
                 dialect_weight=0.3,
                 speaker_weight=0.5,
                 blank_id=0):
        super().__init__()

        self.asr_weight = asr_weight
        self.dialect_weight = dialect_weight
        self.speaker_weight = speaker_weight

        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        """
        Args:
            outputs: dict from MultiTaskVoiceModel.forward()
                - asr_log_probs: [T, B, V] log-probabilities
                - dialect_logits: [B, num_dialects]
                - speaker_loss: scalar (pre-computed by SpeakerBranch)
            labels: dict
                - asr_labels: [B, max_label_len]
                - asr_label_lengths: [B]
                - feat_lengths: [B]
                - dialect_labels: [B]
        Returns:
            total_loss: scalar
            loss_dict: {name: value} 各子损失
        """
        total_loss = 0.0
        loss_dict = {}

        # ===== ASR CTC Loss =====
        if self.asr_weight > 0 and 'asr_log_probs' in outputs:
            asr_log_probs = outputs['asr_log_probs']  # [T, B, V]
            asr_labels = labels.get('asr_labels')
            feat_lengths = labels.get('feat_lengths')
            asr_label_lengths = labels.get('asr_label_lengths')

            if asr_labels is not None and feat_lengths is not None:
                # CTC: input_lengths = 特征序列长度 (已经过 conv 处理)
                asr_loss = self.ctc_loss(
                    asr_log_probs,           # [T, B, V]
                    asr_labels,              # [B, max_label_len]
                    feat_lengths,            # [B] 特征帧数
                    asr_label_lengths,       # [B] 标签长度
                )

                # CTCLoss 在 batch 中所有样本标签长度为 0 时会返回 inf
                if not torch.isinf(asr_loss) and not torch.isnan(asr_loss):
                    asr_loss = asr_loss * self.asr_weight
                    total_loss = total_loss + asr_loss
                    loss_dict['asr_loss'] = asr_loss.item()

        # ===== Dialect Classification Loss =====
        if self.dialect_weight > 0 and 'dialect_logits' in outputs:
            dialect_logits = outputs['dialect_logits']
            dialect_labels = labels.get('dialect_labels')

            if dialect_labels is not None:
                dialect_loss = self.ce_loss(dialect_logits, dialect_labels)
                dialect_loss = dialect_loss * self.dialect_weight
                total_loss = total_loss + dialect_loss
                loss_dict['dialect_loss'] = dialect_loss.item()

        # ===== Speaker AAM-Softmax Loss =====
        if self.speaker_weight > 0 and 'speaker_loss' in outputs:
            speaker_loss = outputs['speaker_loss']

            if not torch.isinf(speaker_loss) and not torch.isnan(speaker_loss):
                speaker_loss = speaker_loss * self.speaker_weight
                total_loss = total_loss + speaker_loss
                loss_dict['speaker_loss'] = speaker_loss.item()

        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        return total_loss, loss_dict

    def update_weights(self, asr_weight=None, dialect_weight=None, speaker_weight=None):
        """动态调整任务权重"""
        if asr_weight is not None:
            self.asr_weight = asr_weight
        if dialect_weight is not None:
            self.dialect_weight = dialect_weight
        if speaker_weight is not None:
            self.speaker_weight = speaker_weight


class UncertaintyWeightingLoss(nn.Module):
    """
    基于不确定性加权的多任务损失 (Kendall et al. 2018)

    自动学习各任务的最优权重:
    Total Loss = Σ (1/(2*σ_i²)) * Loss_i + log(σ_i)

    优势:
    - 自动平衡各任务
    - 避免人工调权
    - σ_i 可解释 (任务不确定度)
    """

    def __init__(self, num_tasks=3, blank_id=0):
        super().__init__()

        # 可学习的 log-variance 参数 (保证 σ² 为正)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        """
        使用不确定性加权计算总损失
        """
        total_loss = 0.0
        loss_dict = {}

        # Task 0: ASR
        if 'asr_log_probs' in outputs:
            precision = torch.exp(-self.log_vars[0])
            asr_loss = self.ctc_loss(
                outputs['asr_log_probs'],
                labels['asr_labels'],
                labels['feat_lengths'],
                labels['asr_label_lengths'],
            )
            if not torch.isinf(asr_loss) and not torch.isnan(asr_loss):
                task_loss = precision * asr_loss + self.log_vars[0]
                total_loss = total_loss + task_loss
                loss_dict['asr_loss'] = asr_loss.item()
                loss_dict['asr_weight'] = precision.item()

        # Task 1: Dialect
        if 'dialect_logits' in outputs:
            precision = torch.exp(-self.log_vars[1])
            dialect_loss = self.ce_loss(outputs['dialect_logits'], labels['dialect_labels'])
            task_loss = precision * dialect_loss + self.log_vars[1]
            total_loss = total_loss + task_loss
            loss_dict['dialect_loss'] = dialect_loss.item()
            loss_dict['dialect_weight'] = precision.item()

        # Task 2: Speaker
        if 'speaker_loss' in outputs:
            precision = torch.exp(-self.log_vars[2])
            speaker_loss = outputs['speaker_loss']
            if not torch.isinf(speaker_loss) and not torch.isnan(speaker_loss):
                task_loss = precision * speaker_loss + self.log_vars[2]
                total_loss = total_loss + task_loss
                loss_dict['speaker_loss'] = speaker_loss.item()
                loss_dict['speaker_weight'] = precision.item()

        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        return total_loss, loss_dict
