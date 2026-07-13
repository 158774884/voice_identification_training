"""
方言 / 口音分类推理模块

支持:
- 单条音频方言识别
- 批量识别
- Top-K 概率输出
- 流式音频累积识别
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional

from data.vocab import DIALECT_LABELS, DIALECT_NAMES_ZH


class DialectInference:
    """
    方言分类推理器

    Args:
        model: MultiTaskVoiceModel 或仅 Dialect 分支
        device: 运行设备
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, audio: torch.Tensor,
                audio_lengths: Optional[torch.Tensor] = None,
                top_k: int = 3) -> List[Dict]:
        """
        预测方言类别

        Args:
            audio: [B, 1, T] 音频
            audio_lengths: [B]
            top_k: 返回 top-K 结果

        Returns:
            results: List[Dict] 每个样本的预测结果
                - dialect: 最佳方言名
                - dialect_zh: 中文方言名
                - confidence: 置信度
                - top_k: [(dialect, prob), ...]
        """
        audio = audio.to(self.device)
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(self.device)

        self.model.eval()
        outputs = self.model(audio, audio_lengths,
                             task_mask={'asr': False, 'dialect': True, 'speaker': False})

        logits = outputs['dialect_logits']  # [B, num_dialects]
        probs = F.softmax(logits, dim=-1)   # [B, num_dialects]

        # Top-K
        topk_probs, topk_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)

        results = []
        for b in range(logits.size(0)):
            best_idx = topk_indices[b][0].item()
            best_prob = topk_probs[b][0].item()

            top_k_list = []
            for k in range(topk_indices.size(1)):
                idx = topk_indices[b][k].item()
                prob = topk_probs[b][k].item()
                top_k_list.append({
                    'dialect': DIALECT_LABELS.get(idx, f'unknown_{idx}'),
                    'dialect_zh': DIALECT_NAMES_ZH.get(idx, f'未知_{idx}'),
                    'probability': round(prob, 4),
                })

            results.append({
                'dialect': DIALECT_LABELS.get(best_idx, f'unknown_{best_idx}'),
                'dialect_zh': DIALECT_NAMES_ZH.get(best_idx, f'未知_{best_idx}'),
                'confidence': round(best_prob, 4),
                'top_k': top_k_list,
            })

        return results

    def predict_single(self, audio: torch.Tensor,
                       top_k: int = 3) -> Dict:
        """
        单条音频方言识别

        Args:
            audio: [T] or [1, T]
            top_k: 返回 top-K

        Returns:
            result: Dict
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)

        results = self.predict(audio, top_k=top_k)
        return results[0]

    def streaming_predict(self, audio_chunks: List[torch.Tensor],
                          accumulation: str = 'mean') -> Dict:
        """
        流式: 逐 chunk 累积识别

        args:
            audio_chunks: 音频片段列表
            accumulation: 累积方式 'mean' | 'max_vote' | 'last'

        Returns:
            final_result: Dict
        """
        all_logits = []

        for chunk in audio_chunks:
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0).unsqueeze(0)

            self.model.eval()
            outputs = self.model(chunk.to(self.device), None,
                                 task_mask={'asr': False, 'dialect': True, 'speaker': False})
            all_logits.append(outputs['dialect_logits'].cpu())

        # 累积
        if accumulation == 'mean':
            # 平均所有 chunk 的 logits
            avg_logits = torch.stack(all_logits).mean(dim=0)  # [1, num_dialects]
            probs = F.softmax(avg_logits, dim=-1)

        elif accumulation == 'max_vote':
            # 投票: 每个 chunk 取 argmax
            votes = torch.stack([l.argmax(dim=-1) for l in all_logits])  # [N, 1]
            # 多数表决
            mode = votes.mode(dim=0).values  # [1]
            probs = torch.zeros(1, all_logits[0].size(-1))
            probs[0, mode] = 1.0

        else:  # 'last'
            probs = F.softmax(all_logits[-1], dim=-1)

        best_idx = probs.argmax(dim=-1).item()
        best_prob = probs[0, best_idx].item()

        return {
            'dialect': DIALECT_LABELS.get(best_idx, f'unknown_{best_idx}'),
            'dialect_zh': DIALECT_NAMES_ZH.get(best_idx, f'未知_{best_idx}'),
            'confidence': round(best_prob, 4),
        }
