"""
音频预处理流水线

处理流程:
  原始音频 → 重采样(16kHz) → 降噪 → 音量归一化 → 去静音 → 分段

所有操作在 CPU 完成，支持批量处理。
SOC 部署时这些操作在音频采集硬件/驱动完成。
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple


class AudioPreprocessor:
    """
    音频预处理器

    支持:
    - 重采样到 16kHz
    - DC 偏移去除
    - 预加重
    - 幅度归一化 (RMS/Peak)
    - VAD 静音检测/去除
    - 频谱减法降噪
    """

    def __init__(self, target_sr=16000, normalize='rms', target_rms=0.1,
                 preemphasis=0.97, remove_dc=True):
        self.target_sr = target_sr
        self.normalize = normalize
        self.target_rms = target_rms
        self.preemphasis = preemphasis
        self.remove_dc = remove_dc

    def process(self, audio: np.ndarray, orig_sr: int = 16000) -> np.ndarray:
        """
        完整预处理流水线

        Args:
            audio: 原始音频 [T] or [1, T]
            orig_sr: 原始采样率

        Returns:
            processed: 处理后的音频 [T]
        """
        audio = np.squeeze(audio).astype(np.float32)

        # 1. 重采样
        if orig_sr != self.target_sr:
            audio = self._resample(audio, orig_sr, self.target_sr)

        # 2. DC 偏移去除
        if self.remove_dc:
            audio = audio - np.mean(audio)

        # 3. 预加重
        if self.preemphasis > 0:
            audio = self._preemphasis_filter(audio, self.preemphasis)

        # 4. 幅度归一化
        if self.normalize == 'rms':
            audio = self._rms_normalize(audio, self.target_rms)
        elif self.normalize == 'peak':
            audio = self._peak_normalize(audio)

        # 5. 裁剪异常值
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """高质量重采样 (使用 scipy)"""
        if orig_sr == target_sr:
            return audio

        # 计算重采样比例
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd

        # 多级重采样以获得最佳质量
        return signal.resample_poly(audio.astype(np.float64), up, down).astype(np.float32)

    def _preemphasis_filter(self, audio: np.ndarray, coeff: float) -> np.ndarray:
        """预加重: y[n] = x[n] - coeff * x[n-1]"""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    def _rms_normalize(self, audio: np.ndarray, target_rms: float) -> np.ndarray:
        """RMS 归一化到目标值"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-8:
            audio = audio * (target_rms / rms)
        return audio

    def _peak_normalize(self, audio: np.ndarray) -> np.ndarray:
        """峰值归一化到 ±1.0"""
        peak = np.max(np.abs(audio))
        if peak > 1e-8:
            audio = audio / peak
        return audio

    def remove_silence(self, audio: np.ndarray, threshold_db: float = -40.0,
                       frame_ms: int = 25, hop_ms: int = 10,
                       min_duration_ms: int = 200) -> np.ndarray:
        """
        基于能量的 VAD 静音去除

        Args:
            audio: 音频信号
            threshold_db: 静音阈值 (dB)
            frame_ms: 帧长 (ms)
            hop_ms: 帧移 (ms)
            min_duration_ms: 最小有效语音段 (ms)

        Returns:
            去除前后静音后的音频
        """
        frame_len = int(self.target_sr * frame_ms / 1000)
        hop_len = int(self.target_sr * hop_ms / 1000)
        min_frames = int(min_duration_ms / hop_ms)

        if len(audio) < frame_len:
            return audio

        # 计算每帧能量
        energy = []
        for i in range(0, len(audio) - frame_len + 1, hop_len):
            frame = audio[i:i + frame_len]
            e = np.sum(frame ** 2) / frame_len
            energy.append(e)

        energy = np.array(energy)
        energy_db = 10 * np.log10(energy + 1e-10)

        # 找到语音段
        is_speech = energy_db > threshold_db

        # 平滑 (去除孤立帧)
        min_speech_frames = max(3, min_frames)
        for i in range(len(is_speech)):
            if i > 0 and i < len(is_speech) - 1:
                if not is_speech[i] and is_speech[i-1] and is_speech[i+1]:
                    is_speech[i] = True

        # 找到第一个和最后一个语音帧
        speech_frames = np.where(is_speech)[0]

        if len(speech_frames) == 0:
            return audio  # 全是静音, 返回原始

        start_frame = speech_frames[0]
        end_frame = speech_frames[-1]

        start_sample = max(0, start_frame * hop_len - frame_len)
        end_sample = min(len(audio), (end_frame + 1) * hop_len + frame_len)

        return audio[start_sample:end_sample]

    def spectral_subtraction(self, audio: np.ndarray, noise_sample: np.ndarray = None,
                             n_fft: int = 512, reduction: float = 2.0) -> np.ndarray:
        """
        频谱减法降噪

        Args:
            audio: 带噪音频
            noise_sample: 纯噪声样本 (如未提供, 取音频前100ms作为噪声估计)
            n_fft: FFT 点数
            reduction: 降噪强度 (1.0-3.0)

        Returns:
            降噪后的音频
        """
        if noise_sample is None:
            # 用前 100ms 作为噪声估计
            noise_len = min(self.target_sr // 10, len(audio) // 4)
            noise_sample = audio[:noise_len]

        # 噪声频谱估计
        noise_stft = np.abs(self._stft(noise_sample, n_fft))
        noise_spec = np.mean(noise_stft, axis=1)

        # 信号 STFT
        audio_stft = self._stft(audio, n_fft)
        audio_mag = np.abs(audio_stft)
        audio_phase = np.angle(audio_stft)

        # 频谱减法 (过减法)
        gain = np.maximum(audio_mag - reduction * noise_spec[:, np.newaxis], 1e-6) / \
               np.maximum(audio_mag, 1e-6)

        # 重建
        cleaned_mag = gain * audio_mag
        cleaned_stft = cleaned_mag * np.exp(1j * audio_phase)
        cleaned = self._istft(cleaned_stft)

        return cleaned[:len(audio)].astype(np.float32)

    def _stft(self, x: np.ndarray, n_fft: int, hop: int = None) -> np.ndarray:
        if hop is None:
            hop = n_fft // 4
        window = np.hanning(n_fft)
        return np.array([np.fft.rfft(window * x[i:i + n_fft])
                         for i in range(0, len(x) - n_fft, hop)]).T

    def _istft(self, stft: np.ndarray, hop: int = None) -> np.ndarray:
        n_fft = (stft.shape[0] - 1) * 2
        if hop is None:
            hop = n_fft // 4
        frames = stft.shape[1]
        window = np.hanning(n_fft)
        result = np.zeros(hop * (frames - 1) + n_fft)
        for i in range(frames):
            result[i * hop:i * hop + n_fft] += window * np.fft.irfft(stft[:, i])
        return result


def preprocess_audio(audio: np.ndarray, orig_sr: int = 16000,
                     do_denoise: bool = False,
                     do_vad: bool = False,
                     **kwargs) -> np.ndarray:
    """
    便捷函数: 单次调用完成音频预处理

    Args:
        audio: 原始音频
        orig_sr: 原始采样率
        do_denoise: 是否降噪
        do_vad: 是否去静音

    Returns:
        处理后的音频 [T_samples] @ 16kHz
    """
    preprocessor = AudioPreprocessor(**kwargs)

    audio = preprocessor.process(audio, orig_sr)

    if do_denoise:
        audio = preprocessor.spectral_subtraction(audio)

    if do_vad:
        audio = preprocessor.remove_silence(audio)

    return audio
