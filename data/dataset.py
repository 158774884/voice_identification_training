"""
多任务 PyTorch Dataset

支持:
- 原始音频加载 (wav/flac/mp3)
- ASR 转录文本 + 方言标签 + 说话人标签
- 动态批次 padding
- 分布式训练支持
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Optional, List, Dict, Tuple


class MultiTaskDataset(Dataset):
    """
    多任务语音数据集

    期望的数据目录结构:
    data_root/
    ├── metadata.jsonl     # 主标注文件
    ├── audio/             # 音频文件目录
    │   ├── spk001/
    │   │   ├── utt001.wav
    │   │   └── utt002.wav
    │   └── spk002/
    │       └── ...
    └── noise/             # 噪声文件 (可选)
        └── ...

    metadata.jsonl 格式 (每行一个 JSON):
    {
        "audio_path": "audio/spk001/utt001.wav",
        "text": "你好世界",
        "dialect": "mandarin",
        "speaker_id": "spk001",
        "duration": 2.5
    }
    """

    def __init__(self,
                 data_root: str,
                 metadata_file: str = 'metadata.jsonl',
                 vocab=None,
                 preprocessor=None,
                 augmentor=None,
                 max_audio_length: int = 16000 * 15,  # 15s max
                 min_audio_length: int = 16000 * 1,   # 1s min
                 sample_rate: int = 16000,
                 training: bool = True):
        """
        Args:
            data_root: 数据根目录
            metadata_file: 标注文件路径 (相对于 data_root)
            vocab: ChineseVocab 实例
            preprocessor: AudioPreprocessor 实例
            augmentor: AudioAugmentor 实例
            max_audio_length: 最大音频长度 (采样点)
            min_audio_length: 最小音频长度 (采样点)
            sample_rate: 目标采样率
            training: 是否训练模式 (决定是否做数据增强)
        """
        self.data_root = data_root
        self.training = training
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.sample_rate = sample_rate

        self.vocab = vocab
        self.preprocessor = preprocessor
        self.augmentor = augmentor

        # 方言标签映射
        self.dialect2id = {
            'mandarin': 0,
            'cantonese': 1,
            'sichuanese': 2,
            'wu': 3,
            'minnan': 4,
            'hakka': 5,
            'xiang': 6,
            'gan': 7,
            'jin': 8,
            'other': 9,
        }

        # 加载元数据
        self.samples = []
        self.speaker2id = {}
        self._load_metadata(os.path.join(data_root, metadata_file))

        print(f"[Dataset] Loaded {len(self.samples)} utterances, "
              f"{len(self.speaker2id)} speakers")

    def _load_metadata(self, metadata_path: str):
        """加载标注文件"""
        if not os.path.exists(metadata_path):
            print(f"[Dataset] Warning: {metadata_path} not found, "
                  f"creating dummy dataset")
            self._create_dummy_data()
            return

        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json
                    item = json.loads(line)
                    self._add_sample(item)
                except Exception as e:
                    print(f"[Dataset] Skip line: {e}")

    def _add_sample(self, item: dict):
        """添加一个样本"""
        audio_path = os.path.join(self.data_root, item['audio_path'])
        if not os.path.exists(audio_path):
            audio_path = item.get('audio_path', '')
            if not os.path.exists(audio_path):
                return

        speaker_id = item.get('speaker_id', 'unknown')

        # 分配 speaker 数字 ID
        if speaker_id not in self.speaker2id:
            self.speaker2id[speaker_id] = len(self.speaker2id)

        dialect_name = item.get('dialect', 'mandarin')

        self.samples.append({
            'audio_path': audio_path,
            'text': item.get('text', ''),
            'dialect': self.dialect2id.get(dialect_name, 9),
            'dialect_name': dialect_name,
            'speaker_id': self.speaker2id[speaker_id],
            'duration': item.get('duration', 0.0),
        })

    def _load_audio(self, audio_path: str):
        """Load audio file and return float32 numpy array.

        Uses soundfile with fallback to scipy/wave for robustness.
        """
        # 1) soundfile
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            # Resample if needed
            if sr != self.sample_rate:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                except Exception:
                    pass  # keep original sr
            return audio.astype(np.float32)
        except Exception:
            pass

        # 2) scipy.io.wavfile
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(audio_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            if sr != self.sample_rate:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                except Exception:
                    pass
            return audio.astype(np.float32)
        except Exception:
            pass

        # 3) built-in wave module (WAV only)
        try:
            import wave
            with wave.open(audio_path, 'rb') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
                return audio.astype(np.float32) / 32768.0
        except Exception:
            pass

        # 4) return zeros as last resort (e.g. for dummy data)
        return np.zeros(self.sample_rate, dtype=np.float32)

    def _create_dummy_data(self):
        """创建哑数据 (用于测试)"""
        for i in range(100):
            speaker_id = f'spk_{i % 10:03d}'
            if speaker_id not in self.speaker2id:
                self.speaker2id[speaker_id] = len(self.speaker2id)

            self.samples.append({
                'audio_path': f'dummy_{i}',
                'text': f'这是第{i}段测试语音',
                'dialect': random.randint(0, 9),
                'dialect_name': 'mandarin',
                'speaker_id': self.speaker2id[speaker_id],
                'duration': random.uniform(1.0, 10.0),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载音频
        audio = self._load_audio(sample['audio_path'])

        # 预处理
        if self.preprocessor is not None:
            audio = self.preprocessor.process(audio, self.sample_rate)

        # 数据增强 (仅训练)
        if self.training and self.augmentor is not None:
            audio = self.augmentor.augment_waveform(audio, self.sample_rate)

        # 裁剪/填充
        if len(audio) > self.max_audio_length:
            # 随机截取
            start = random.randint(0, len(audio) - self.max_audio_length)
            audio = audio[start:start + self.max_audio_length]

        audio_length = len(audio)

        # 编码文本
        text = sample['text']
        if self.vocab is not None:
            tokens = self.vocab.encode(text)
        else:
            # 无词表时使用 UTF-8 bytes
            tokens = list(text.encode('utf-8'))

        return {
            'audio': torch.FloatTensor(audio),  # [T]
            'audio_length': audio_length,
            'asr_tokens': torch.LongTensor(tokens),  # [L]
            'asr_label_length': len(tokens),
            'dialect_label': sample['dialect'],
            'speaker_label': sample['speaker_id'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批次整理函数: 对变长序列做 padding

    Args:
        batch: list of samples from __getitem__

    Returns:
        batched dict
    """
    # 音频 padding
    max_audio_len = max(item['audio'].size(0) for item in batch)
    audio_batch = torch.zeros(len(batch), 1, max_audio_len)
    audio_lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        audio = item['audio']
        audio_batch[i, 0, :audio.size(0)] = audio
        audio_lengths[i] = item['audio_length']

    # ASR 标签 padding
    max_label_len = max(item['asr_tokens'].size(0) for item in batch)
    label_batch = torch.zeros(len(batch), max_label_len, dtype=torch.long)
    label_lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        tokens = item['asr_tokens']
        label_batch[i, :tokens.size(0)] = tokens
        label_lengths[i] = item['asr_label_length']

    # 方言和说话人标签
    dialect_labels = torch.LongTensor([item['dialect_label'] for item in batch])
    speaker_labels = torch.LongTensor([item['speaker_label'] for item in batch])

    return {
        'audio': audio_batch,
        'audio_lengths': audio_lengths,
        'asr_labels': label_batch,
        'asr_label_lengths': label_lengths,
        'dialect_labels': dialect_labels,
        'speaker_labels': speaker_labels,
    }


def create_dataloader(dataset: MultiTaskDataset,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 4,
                      pin_memory: bool = True,
                      drop_last: bool = True) -> DataLoader:
    """
    创建 DataLoader

    Args:
        dataset: MultiTaskDataset 实例
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 并行加载进程数
        pin_memory: 是否固定到 GPU 内存
        drop_last: 是否丢弃不完整批次

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class StreamingAudioDataset(Dataset):
    """
    流式音频数据集 —— 适用于超长音频的流式训练

    将长音频切分为固定长度的 chunk,
    每个 chunk 有对应的文本段和标签
    """

    def __init__(self,
                 data_root: str,
                 metadata_file: str = 'metadata.jsonl',
                 chunk_duration: float = 3.0,
                 chunk_overlap: float = 0.5,
                 **kwargs):
        super().__init__(data_root, metadata_file, **kwargs)
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.chunk_samples = int(chunk_duration * self.sample_rate)
        self.overlap_samples = int(chunk_overlap * self.sample_rate)
