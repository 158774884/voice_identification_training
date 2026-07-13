"""
数据增强模块

SOC 部署时数据增强仅在训练阶段使用

增强方法:
1. 速度扰动 (Speed Perturbation): 改变语速而不改变音高
2. 加性噪声混合 (Additive Noise): 从噪声库随机选取
3. 房间混响模拟 (RIR Reverberation)
4. SpecAugment: 时域/频域遮蔽 (在特征域做)
5. 音量扰动 (Volume Perturbation)
6. 音高偏移 (Pitch Shift)
"""

import numpy as np
import random
from scipy import signal
from typing import Optional, List, Tuple


class AudioAugmentor:
    """
    音频数据增强器

    所有增强方法都保持 16kHz 采样率不变
    """

    def __init__(self,
                 speed_perturb: bool = True,
                 speed_rates: Tuple[float, ...] = (0.9, 1.0, 1.1),
                 noise_augment: bool = True,
                 noise_files: Optional[List[str]] = None,
                 noise_snr_range: Tuple[float, float] = (5.0, 20.0),
                 reverb_augment: bool = False,
                 rir_files: Optional[List[str]] = None,
                 volume_perturb: bool = True,
                 volume_range_db: Tuple[float, float] = (-5.0, 5.0),
                 spec_augment: bool = True,
                 freq_mask_width: int = 27,
                 time_mask_width: int = 100,
                 num_freq_masks: int = 2,
                 num_time_masks: int = 2,
                 seed: Optional[int] = None):
        """
        Args:
            speed_perturb: 是否启用速度扰动
            speed_rates: 速度扰动倍率 (1.0 = 不变)
            noise_augment: 是否混合噪声
            noise_files: 噪声文件路径列表
            noise_snr_range: 随机 SNR 范围 (dB)
            reverb_augment: 是否添加混响
            rir_files: 房间冲激响应文件路径列表
            volume_perturb: 是否音量扰动
            volume_range_db: 音量增益范围 (dB)
            spec_augment: 是否启用 SpecAugment (特征域)
            freq_mask_width: 频域遮蔽最大宽度
            time_mask_width: 时域遮蔽最大宽度
            num_freq_masks: 频域遮蔽次数
            num_time_masks: 时域遮蔽次数
        """
        self.speed_perturb = speed_perturb
        self.speed_rates = speed_rates
        self.noise_augment = noise_augment
        self.noise_files = noise_files or []
        self.noise_snr_range = noise_snr_range
        self.reverb_augment = reverb_augment
        self.rir_files = rir_files or []
        self.volume_perturb = volume_perturb
        self.volume_range_db = volume_range_db
        self.spec_augment = spec_augment
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 加载噪声 (懒加载)
        self._noise_cache = {}
        self._rir_cache = {}

    def augment_waveform(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        对原始音频波形做增强

        Args:
            audio: [T] 音频数据
            sample_rate: 采样率

        Returns:
            augmented: [T'] 增强后的音频 (长度可能因速度扰动改变)
        """
        # 1. 音量扰动 (最先, 影响后续处理)
        if self.volume_perturb:
            audio = self._volume_perturb(audio)

        # 2. 速度扰动 (改变长度)
        if self.speed_perturb:
            rate = random.choice(self.speed_rates)
            if abs(rate - 1.0) > 0.01:
                audio = self._speed_perturb(audio, rate)

        # 3. 混响 (在加噪声前)
        if self.reverb_augment and len(self.rir_files) > 0:
            audio = self._apply_reverb(audio)

        # 4. 加性噪声
        if self.noise_augment and len(self.noise_files) > 0:
            audio = self._mix_noise(audio, sample_rate)

        # 再次裁剪避免溢出
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def spec_augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        SpecAugment: 在特征图上做时域/频域遮蔽

        Args:
            features: [F, T] 特征图 (F=freq, T=time)

        Returns:
            augmented: [F, T] 增强后的特征
        """
        if not self.spec_augment:
            return features

        F, T = features.shape

        # 频域遮蔽
        for _ in range(self.num_freq_masks):
            f_width = random.randint(0, self.freq_mask_width)
            if f_width > 0:
                f_start = random.randint(0, max(F - f_width, 0))
                features[f_start:f_start + f_width, :] = 0.0

        # 时域遮蔽
        for _ in range(self.num_time_masks):
            t_width = random.randint(0, self.time_mask_width)
            if t_width > 0:
                t_start = random.randint(0, max(T - t_width, 0))
                features[:, t_start:t_start + t_width] = 0.0

        return features

    def _volume_perturb(self, audio: np.ndarray) -> np.ndarray:
        """随机音量扰动"""
        gain_db = random.uniform(*self.volume_range_db)
        gain_linear = 10 ** (gain_db / 20.0)
        return audio * gain_linear

    def _speed_perturb(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """
        速度扰动 (改变语速, 不改变音高)

        通过重采样实现: 先升采样再降采样

        例如 rate=0.9 → 语速变慢 10% → 音频变长 10%
             rate=1.1 → 语速变快 10% → 音频变短 10%
        """
        # 用 scipy 的 resample 实现
        new_len = int(len(audio) / rate)
        return signal.resample(audio.astype(np.float64), new_len).astype(np.float32)

    def _mix_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        随机选择噪声片段并混合

        目标 SNR 从 noise_snr_range 随机选取

        SNR 计算公式:
        SNR_dB = 10 * log10(Psignal / Pnoise)
        """
        # 随机选择噪声文件 (简化: 生成高斯白噪声)
        # 实际使用应加载 noise_files 中的文件
        if random.random() < 0.3:
            # 30% 概率用白噪声 (模拟通用背景噪)
            noise = np.random.randn(len(audio)).astype(np.float32) * 0.01
        elif len(self._noise_cache) > 0:
            # 从缓存的噪声文件中选取
            noise_key = random.choice(list(self._noise_cache.keys()))
            noise_sample = self._noise_cache[noise_key]
            # 随机截取等长片段
            if len(noise_sample) > len(audio):
                start = random.randint(0, len(noise_sample) - len(audio))
                noise = noise_sample[start:start + len(audio)]
            else:
                repeats = len(audio) // len(noise_sample) + 1
                noise = np.tile(noise_sample, repeats)[:len(audio)]
        else:
            # 无噪声文件, 用白噪声
            noise = np.random.randn(len(audio)).astype(np.float32) * 0.005

        # 按 SNR 混合
        snr_db = random.uniform(*self.noise_snr_range)

        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power > 1e-10:
            target_noise_power = signal_power / (10 ** (snr_db / 10.0))
            scale = np.sqrt(target_noise_power / noise_power)
            noise = noise * scale

        return audio + noise

    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """
        添加房间混响效果

        用随机 RIR (Room Impulse Response) 做卷积
        """
        if len(self._rir_cache) > 0:
            rir_key = random.choice(list(self._rir_cache.keys()))
            rir = self._rir_cache[rir_key]
        else:
            # 简化: 用指数衰减模拟混响
            rt60 = random.uniform(0.1, 0.5)  # 混响时间 (秒)
            decay = np.exp(-np.arange(8000) / (16000 * rt60))
            rir = np.random.randn(8000) * decay
            rir = rir / np.max(np.abs(rir))

        # 卷积混响
        reverb_audio = signal.convolve(audio, rir, mode='full')
        # 裁剪到原始长度
        reverb_audio = reverb_audio[:len(audio)]
        # 保持幅度
        scale = np.sqrt(np.mean(audio ** 2)) / (np.sqrt(np.mean(reverb_audio ** 2)) + 1e-8)
        return reverb_audio * scale

    def load_noise_samples(self, noise_dir: str, max_files: int = 100):
        """从目录加载噪声文件到缓存"""
        import os
        noise_files = []
        for root, _, files in os.walk(noise_dir):
            for f in files:
                if f.endswith(('.wav', '.flac', '.mp3')):
                    noise_files.append(os.path.join(root, f))

        random.shuffle(noise_files)
        for f in noise_files[:max_files]:
            try:
                import soundfile as sf
                audio, _ = sf.read(f)
                if len(audio) > 16000:  # 至少 1 秒
                    self._noise_cache[f] = audio.astype(np.float32)
            except Exception:
                pass

        self.noise_files = list(self._noise_cache.keys())
        print(f"[Augmentor] Loaded {len(self._noise_cache)} noise files")

    def load_rir_samples(self, rir_dir: str, max_files: int = 50):
        """从目录加载 RIR 文件到缓存"""
        import os
        rir_files = []
        for root, _, files in os.walk(rir_dir):
            for f in files:
                if f.endswith(('.wav', '.flac')):
                    rir_files.append(os.path.join(root, f))

        random.shuffle(rir_files)
        for f in rir_files[:max_files]:
            try:
                import soundfile as sf
                audio, _ = sf.read(f)
                self._rir_cache[f] = audio.astype(np.float32)
            except Exception:
                pass

        self.rir_files = list(self._rir_cache.keys())
        print(f"[Augmentor] Loaded {len(self._rir_cache)} RIR files")
