"""
统一推理流水线 —— 一次前向传播完成三项任务

使用示例:
    pipeline = VoiceInferencePipeline(model, vocab)
    results = pipeline(audio)  # dict with asr, dialect, speaker keys
"""

import torch
import numpy as np
from typing import Dict, Optional

from .asr_inference import ASRInference
from .dialect_inference import DialectInference
from .speaker_inference import SpeakerInference
from data.vocab import ChineseVocab
from data.preprocessing import AudioPreprocessor


class VoiceInferencePipeline:
    """
    统一语音推理流水线

    单次前向传播 → 同时输出:
    1. ASR 文字识别结果
    2. 方言/口音分类
    3. 声纹嵌入向量

    Args:
        model: MultiTaskVoiceModel (已训练)
        vocab: ChineseVocab 实例
        device: 推理设备 ('cpu', 'cuda')
        beam_width: ASR beam search 宽度
        sample_rate: 输入采样率
    """

    def __init__(self, model, vocab: ChineseVocab, device='cpu',
                 beam_width=1, sample_rate=16000):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.sample_rate = sample_rate

        # 子推理器
        self.asr = ASRInference(model, vocab, device, beam_width)
        self.dialect = DialectInference(model, device)
        self.speaker = SpeakerInference(model, device)

        # 音频预处理器 (处理非标准输入)
        self.preprocessor = AudioPreprocessor(target_sr=sample_rate)

    @torch.no_grad()
    def __call__(self, audio: torch.Tensor,
                 audio_lengths: Optional[torch.Tensor] = None,
                 tasks: str = 'all') -> Dict:
        """
        统一推理入口

        Args:
            audio: [1, 1, T] 或 [T] 16kHz 音频
            audio_lengths: [1] 可选
            tasks: 'all' | 'asr' | 'dialect' | 'speaker'

        Returns:
            results: Dict
        """
        # 确保维度正确
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)  # [1, 1, T]

        audio = audio.to(self.device)
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(self.device)

        # 任务掩码 — 只计算需要的分支
        task_mask = {
            'asr': tasks in ('all', 'asr'),
            'dialect': tasks in ('all', 'dialect'),
            'speaker': tasks in ('all', 'speaker'),
        }

        self.model.eval()
        outputs = self.model(audio, audio_lengths, task_mask=task_mask)

        results = {}

        # ASR
        if task_mask['asr']:
            log_probs = outputs['asr_log_probs']  # [T, B, V]
            token_ids = self.asr._greedy_decode(log_probs)
            results['asr_text'] = [self.vocab.decode(ids) for ids in token_ids]
            results['asr_tokens'] = token_ids

        # Dialect
        if task_mask['dialect']:
            import torch.nn.functional as F
            logits = outputs['dialect_logits']  # [B, num_dialects]
            probs = F.softmax(logits, dim=-1)
            best_idx = probs.argmax(dim=-1).item()

            from data.vocab import DIALECT_LABELS, DIALECT_NAMES_ZH
            results['dialect'] = DIALECT_LABELS.get(best_idx, f'unknown_{best_idx}')
            results['dialect_zh'] = DIALECT_NAMES_ZH.get(best_idx, f'未知_{best_idx}')
            results['dialect_confidence'] = probs[0, best_idx].item()
            results['dialect_probs'] = probs.cpu().numpy()

        # Speaker
        if task_mask['speaker']:
            embedding = outputs['speaker_embedding']  # [B, embed_dim]
            results['speaker_embedding'] = embedding.cpu().numpy()

        return results

    def from_file(self, audio_path: str, tasks: str = 'all') -> Dict:
        """
        从音频文件推理

        Args:
            audio_path: 音频文件路径
            tasks: 推理任务

        Returns:
            results: Dict
        """
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
        except ImportError:
            raise ImportError("Please install soundfile: pip install soundfile")

        # 预处理
        audio = self.preprocessor.process(audio.astype(np.float32), sr)

        # 转 tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)

        return self(audio_tensor, tasks=tasks)

    def from_microphone(self, chunk_duration: float = 3.0,
                        tasks: str = 'all') -> Dict:
        """
        从麦克风实时推理 (简化实现)

        实际嵌入式部署中，由硬件 DMA 直接喂入音频
        """
        try:
            import sounddevice as sd
            duration = int(self.sample_rate * chunk_duration)
            audio = sd.rec(duration, samplerate=self.sample_rate,
                           channels=1, dtype='float32')
            sd.wait()
        except ImportError:
            raise ImportError("Please install sounddevice: pip install sounddevice")

        audio = audio.flatten()
        audio = self.preprocessor.process(audio, self.sample_rate)
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)

        return self(audio_tensor, tasks=tasks)
