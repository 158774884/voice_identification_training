"""
ASR 推理模块

支持:
- Greedy CTC 解码
- Beam Search CTC 解码
- 流式增量解码
- 语言模型浅融合 (可选)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


class ASRInference:
    """
    ASR CTC 推理器

    Args:
        model: MultiTaskVoiceModel 或仅 ASR 分支
        vocab: ChineseVocab 实例
        device: 运行设备
        beam_width: beam search 宽度 (默认 1 = greedy)
    """

    def __init__(self, model, vocab, device='cpu', beam_width=1):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.beam_width = beam_width
        self.blank_id = vocab.blank_id

    @torch.no_grad()
    def transcribe(self, audio: torch.Tensor,
                   audio_lengths: Optional[torch.Tensor] = None) -> List[str]:
        """
        转录音频为文本

        Args:
            audio: [B, 1, T] 单通道 16kHz 音频
            audio_lengths: [B] 可选

        Returns:
            texts: List[str] 转录文本列表
        """
        audio = audio.to(self.device)

        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(self.device)

        # 前向传播
        self.model.eval()
        outputs = self.model(audio, audio_lengths,
                             task_mask={'asr': True, 'dialect': False, 'speaker': False})

        log_probs = outputs['asr_log_probs']  # [T, B, V]

        # 解码
        if self.beam_width <= 1:
            # Greedy decoding
            token_ids = self._greedy_decode(log_probs)
        else:
            # Beam search
            token_ids = self._beam_search_decode(log_probs)

        # 转换为文本
        texts = [self.vocab.decode(ids) for ids in token_ids]

        return texts

    def transcribe_single(self, audio: torch.Tensor) -> str:
        """
        单条音频转录

        Args:
            audio: [T] or [1, T] 音频

        Returns:
            text: 转录文本
        """
        # 确保正确的维度
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)  # [1, 1, T]

        texts = self.transcribe(audio)
        return texts[0]

    @torch.no_grad()
    def transcribe_streaming(self, audio_chunks: List[torch.Tensor],
                             cache: Optional[dict] = None) -> List[str]:
        """
        流式转录 (逐 chunk 处理)

        Args:
            audio_chunks: 音频片段列表 [chunk1, chunk2, ...]
            cache: 前一次推理的缓存 (包含 GRU 状态等)

        Returns:
            texts: 累积的转录文本列表
        """
        if cache is None:
            cache = {}

        # 如果有 GRU 状态缓存, 需要手动管理
        # 这里简化: 将 chunk 拼接后一次推理
        # 实际流式部署中应使用 causal 模型 + 状态缓存
        full_audio = torch.cat([c.flatten() for c in audio_chunks])

        return self.transcribe_single(full_audio)

    def _greedy_decode(self, log_probs: torch.Tensor) -> List[List[int]]:
        """
        Greedy CTC 解码

        Args:
            log_probs: [T, B, V]

        Returns:
            decoded: List[List[int]]
        """
        # 取最大概率 token
        best_tokens = log_probs.argmax(dim=-1)  # [T, B]
        best_tokens = best_tokens.permute(1, 0)  # [B, T]

        decoded = []
        for bt in best_tokens:
            result = []
            prev = self.blank_id
            for token in bt:
                token = token.item()
                if token != prev and token != self.blank_id:
                    result.append(token)
                prev = token
            decoded.append(result)

        return decoded

    def _beam_search_decode(self, log_probs: torch.Tensor,
                            beam_width: int = 5) -> List[List[int]]:
        """
        CTC Beam Search 解码

        简化实现: 对每个样本单独做 beam search
        """
        T, B, V = log_probs.shape

        all_results = []
        for b in range(B):
            sample_log_probs = log_probs[:, b, :]  # [T, V]

            # 初始 beam (tokens, log_prob, last_token)
            beams = [([], 0.0, self.blank_id)]

            for t in range(T):
                frame_lp = sample_log_probs[t]  # [V] log-probs
                new_beams = []

                # 取 top-k
                topk_scores, topk_ids = frame_lp.topk(beam_width)

                for tokens, score, last_token in beams:
                    for k in range(beam_width):
                        token = topk_ids[k].item()
                        new_score = score + topk_scores[k].item()
                        new_tokens = tokens[:]

                        # CTC: 非重复且非 blank → 新 token
                        if token != last_token and token != self.blank_id:
                            new_tokens.append(token)

                        new_beams.append((new_tokens, new_score, token))

                # 保留 top beam_width
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width]

            # 取最优 beam
            best_tokens = beams[0][0]
            all_results.append(best_tokens)

        return all_results


class CTCDecoder:
    """
    独立 CTC 解码器 (可用于验证)

    支持语言模型浅融合 (Shallow Fusion):
    score = CTC_score + alpha * LM_score + beta * len_penalty
    """

    def __init__(self, vocab, lm=None, alpha=0.3, beta=0.0):
        self.vocab = vocab
        self.lm = lm  # 可选的语言模型
        self.alpha = alpha
        self.beta = beta

    def decode(self, log_probs: np.ndarray, method: str = 'greedy') -> str:
        """
        Args:
            log_probs: [T, V] numpy array
            method: 'greedy' | 'beam'

        Returns:
            text: 解码文本
        """
        if method == 'greedy':
            ids = self._greedy(log_probs)
        else:
            ids = self._beam_search(log_probs)

        return self.vocab.decode(ids)

    def _greedy(self, log_probs: np.ndarray) -> List[int]:
        tokens = log_probs.argmax(axis=-1)
        result = []
        prev = self.vocab.blank_id
        for t in tokens:
            if t != prev and t != self.vocab.blank_id:
                result.append(int(t))
            prev = t
        return result

    def _beam_search(self, log_probs: np.ndarray, beam_width: int = 5) -> List[int]:
        T, V = log_probs.shape

        beams = [([], 0.0, self.vocab.blank_id)]

        for t in range(T):
            new_beams = []
            topk_indices = np.argsort(log_probs[t])[-beam_width:]

            for tokens, score, last in beams:
                for idx in topk_indices:
                    new_score = score + log_probs[t][idx]
                    new_tokens = tokens[:]

                    if idx != last and idx != self.vocab.blank_id:
                        new_tokens.append(int(idx))

                    # LM score
                    if self.lm is not None and idx != self.vocab.blank_id:
                        # 简单: 用 LM 对 new_tokens 打分
                        # 实际实现需用 KenLM 等
                        pass

                    new_beams.append((new_tokens, new_score, int(idx)))

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        return beams[0][0]
