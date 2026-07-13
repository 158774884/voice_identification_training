"""
关键词 / 指令词识别数据集

数据格式 (metadata.jsonl):
{"audio_path": "wav/001.wav", "label": "打开灯", "label_id": 0}
{"audio_path": "wav/002.wav", "label": "xiao_du_xiao_du", "label_id": 1}
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Dict, List, Optional


class KwsDataset(Dataset):
    """指令词识别数据集"""

    def __init__(self,
                 data_root: str,
                 metadata_file: str = 'metadata.jsonl',
                 feature_extractor=None,
                 augmentor=None,
                 label_list: Optional[List[str]] = None,
                 sample_rate: int = 16000,
                 n_frames: int = 98,      # ~1 second of audio
                 training: bool = True,
                 background_dir: Optional[str] = None,
                 unknown_prob: float = 0.1,
                 ):
        self.data_root = data_root
        self.fe = feature_extractor
        self.augmentor = augmentor
        self.sample_rate = sample_rate
        self.n_frames = n_frames
        self.training = training
        self.unknown_prob = unknown_prob

        # 标签映射
        if label_list is None:
            label_list = []

        self.label2id = {lbl: i for i, lbl in enumerate(label_list)}
        self.id2label = {i: lbl for i, lbl in enumerate(label_list)}

        # 加载样本
        self.samples = []
        self._load(os.path.join(data_root, metadata_file))

        # 未知词标签 ID
        self.unknown_id = len(self.label2id)
        self.num_classes = len(self.label2id) + 1  # +1 for <unknown>

        # 背景噪声
        self.backgrounds = []
        if background_dir and os.path.isdir(background_dir):
            for f in os.listdir(background_dir):
                if f.endswith(('.wav', '.flac')):
                    self.backgrounds.append(os.path.join(background_dir, f))

        print(f"[KwsDataset] {len(self.samples)} samples, "
              f"{len(self.label2id)} keywords (+1 unknown), "
              f"classes={self.num_classes}")

    def _load(self, metadata_path: str):
        if not os.path.exists(metadata_path):
            print(f"[KwsDataset] No metadata found, creating dummy data")
            for i in range(500):
                self.samples.append({
                    'audio_path': f'dummy_{i}.wav',
                    'label_id': random.randint(0, max(0, self.num_classes - 2)),
                    'is_keyword': True,
                })
            return

        with open(metadata_path, 'r', encoding='utf-8') as f:
            import json
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                label = item.get('label', '')
                if label not in self.label2id:
                    self.label2id[label] = len(self.label2id)
                    self.id2label[self.label2id[label]] = label

                self.samples.append({
                    'audio_path': os.path.join(self.data_root, item['audio_path']),
                    'label_id': self.label2id[label],
                    'is_keyword': True,
                    'label': label,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载音频
        try:
            audio = self._load_audio(sample['audio_path'])
        except Exception:
            audio = np.random.randn(self.sample_rate).astype(np.float32) * 0.01

        # 随机偏移 (时间增强)
        if self.training and len(audio) > self.n_frames * 160:
            max_offset = len(audio) - self.n_frames * 160
            offset = random.randint(0, max_offset)
            audio = audio[offset:offset + self.n_frames * 160]

        # 填充/截断到固定长度
        target_len = self.n_frames * 160 + 400  # extra for windowing
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # 背景噪声混合
        if self.training and self.backgrounds and random.random() < 0.5:
            audio = self._mix_background(audio)

        # 提取 Mel 特征
        audio_tensor = torch.FloatTensor(audio)
        if self.fe is not None:
            mel = self.fe.extract(audio_tensor)  # [1, 1, n_mels, T]
            mel = mel.squeeze(0)  # [1, n_mels, T]

            # 时间维度对齐
            if mel.size(2) > self.n_frames:
                if self.training:
                    start = random.randint(0, mel.size(2) - self.n_frames)
                else:
                    start = 0
                mel = mel[:, :, start:start + self.n_frames]
            elif mel.size(2) < self.n_frames:
                pad_w = self.n_frames - mel.size(2)
                mel = torch.nn.functional.pad(mel, (0, pad_w))
        else:
            mel = audio_tensor.unsqueeze(0).unsqueeze(0)

        # <unknown> 标签替换 (训练时随机)
        label_id = sample['label_id']
        if self.training and random.random() < self.unknown_prob:
            label_id = self.unknown_id

        return {
            'mel': mel,                     # [1, n_mels, n_frames]
            'label_id': label_id,
            'is_keyword': sample['is_keyword'],
        }

    def _load_audio(self, path: str) -> np.ndarray:
        if 'dummy_' in path:
            return np.random.randn(self.sample_rate).astype(np.float32) * 0.01

        try:
            import soundfile as sf
            audio, sr = sf.read(path)
            if sr != self.sample_rate:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * self.sample_rate / sr))
            return audio.astype(np.float32)
        except ImportError:
            return np.random.randn(self.sample_rate).astype(np.float32) * 0.01

    def _mix_background(self, audio: np.ndarray) -> np.ndarray:
        if not self.backgrounds:
            return audio
        bg_file = random.choice(self.backgrounds)
        try:
            bg, _ = __import__('soundfile').read(bg_file)
            bg = bg.astype(np.float32)
            if len(bg) < len(audio):
                bg = np.tile(bg, len(audio) // len(bg) + 1)
            bg = bg[:len(audio)]
            snr = random.uniform(5, 15)
            signal_power = np.mean(audio ** 2)
            noise_power = np.mean(bg ** 2)
            if noise_power > 1e-10:
                scale = np.sqrt(signal_power / (10 ** (snr / 10)) / noise_power)
                bg = bg * scale
            return np.clip(audio + bg, -1, 1).astype(np.float32)
        except Exception:
            return audio


def kws_collate_fn(batch: List[Dict]) -> Dict:
    """批次整理"""
    mel_batch = torch.stack([item['mel'] for item in batch])
    label_batch = torch.LongTensor([item['label_id'] for item in batch])
    return {'mel': mel_batch, 'label_id': label_batch}


def create_kws_dataloader(dataset: KwsDataset, batch_size=64,
                          shuffle=True, num_workers=4) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=kws_collate_fn,
                      drop_last=True, pin_memory=True)
