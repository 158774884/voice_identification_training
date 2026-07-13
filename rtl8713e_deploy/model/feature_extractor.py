"""
Mel 特征提取器 —— 训练和 HiFi 5 DSP 双端对齐

训练端: PyTorch (torchaudio MelSpectrogram)
DSP 端: HiFi 5 FFT 硬件 + Mel 滤波器组

两者参数必须严格一致, 确保训练和部署的特征分布一致

HiFi 5 DSP 实现 (C 伪代码):
```c
// 1. Pre-emphasis: y[n] = x[n] - 0.97*x[n-1]
// 2. Framing: 25ms window, 10ms hop
// 3. Hann window
// 4. 512-point FFT (HiFi 5 hardware accelerated)
// 5. Mel filterbank (40 bins, 80Hz-7600Hz)
// 6. Log magnitude
```
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


class MelFeatureExtractor:
    """
    Log-Mel 特征提取器

    参数与 HiFi 5 DSP 实现严格对齐

    训练时: 使用 torchaudio (GPU 加速)
    推理时: 导出参数到 DSP C 代码
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 n_mels: int = 40,
                 n_fft: int = 512,
                 win_length_ms: float = 25.0,
                 hop_length_ms: float = 10.0,
                 f_min: float = 80.0,
                 f_max: float = 7600.0,
                 preemphasis: float = 0.97,
                 window_fn: str = 'hann',
                 power: float = 2.0,
                 ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = int(win_length_ms * sample_rate / 1000)  # 400 points
        self.hop_length = int(hop_length_ms * sample_rate / 1000)  # 160 points
        self.f_min = f_min
        self.f_max = f_max
        self.preemphasis = preemphasis
        self.power = power

        # Torchaudio MelSpectrogram (or fallback)
        if HAS_TORCHAUDIO:
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
                power=power,
                window_fn=getattr(torch, f'{window_fn}_window'),
                center=True,
                pad_mode='reflect',
                norm='slaney',
                mel_scale='slaney',
            )
            self.mel_fb = self.mel_spec.mel_scale.fb
        else:
            # Fallback: 使用 torch.stft + 手动 mel 滤波器
            self.mel_spec = None
            self.mel_fb = self._create_mel_filterbank(n_mels, n_fft // 2 + 1)

        # 存参数
        self.window = torch.hann_window(self.win_length)
        self.n_fft = n_fft
        self.hop_length = self.hop_length
        self.win_length = self.win_length
        self.power = power

    def _create_mel_filterbank(self, n_mels: int, n_freq: int) -> torch.Tensor:
        """创建 Mel 滤波器组 (不依赖 torchaudio)"""
        # Mel scale
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_points = np.linspace(hz_to_mel(self.f_min), hz_to_mel(self.f_max), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        fbank = np.zeros((n_mels, n_freq))
        for m in range(n_mels):
            for k in range(bin[m], bin[m + 1]):
                fbank[m, k] = (k - bin[m]) / max(bin[m + 1] - bin[m], 1)
            for k in range(bin[m + 1], min(bin[m + 2], n_freq)):
                fbank[m, k] = (bin[m + 2] - k) / max(bin[m + 2] - bin[m + 1], 1)

        return torch.FloatTensor(fbank)

    def extract(self, audio: torch.Tensor) -> torch.Tensor:
        """
        提取 log-Mel 特征

        Args:
            audio: [B, T] 或 [T] 16kHz 音频

        Returns:
            features: [B, 1, n_mels, T_frames] 特征图
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # 预加重
        if self.preemphasis > 0:
            audio = self._preemphasis(audio)

        if HAS_TORCHAUDIO:
            # Mel spectrogram: [B, n_mels, T]
            mel = self.mel_spec(audio)
        else:
            # Fallback: torch.stft + manual mel filterbank
            mel = self._extract_mel_torch(audio)

        # Log scaling (+1e-6 for numerical stability)
        mel = torch.log(mel + 1e-6)

        # 添加 channel 维度: [B, 1, n_mels, T]
        mel = mel.unsqueeze(1)

        return mel

    def _extract_mel_torch(self, audio: torch.Tensor) -> torch.Tensor:
        """使用 torch.stft 提取 Mel 特征 (torchaudio 不可用时)"""
        # STFT
        stft = torch.stft(
            audio.squeeze(1) if audio.dim() > 2 else audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(audio.device),
            center=True,
            pad_mode='reflect',
            return_complex=True,
        )
        # Power spectrum: [B, n_fft//2+1, T]
        power_spec = stft.abs() ** self.power

        # Mel filterbank: [n_mels, n_fft//2+1]
        mel_fb = self.mel_fb.to(audio.device)

        # Apply filterbank: [B, n_mels, T]
        mel = torch.matmul(mel_fb, power_spec)

        return mel

    def _preemphasis(self, audio: torch.Tensor) -> torch.Tensor:
        """预加重: y[n] = x[n] - preemph * x[n-1]"""
        return torch.cat([
            audio[:, :1],
            audio[:, 1:] - self.preemphasis * audio[:, :-1],
        ], dim=1)

    def get_dsp_params(self) -> dict:
        """导出 DSP C 代码所需参数"""
        return {
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'win_length': self.win_length,
            'hop_length': self.hop_length,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'preemphasis': self.preemphasis,
            'mel_filterbank': self.mel_fb.cpu().numpy() if self.mel_fb is not None else None,
        }

    def generate_dsp_c_header(self) -> str:
        """生成 DSP C 头文件"""
        params = self.get_dsp_params()
        mel_fb = params['mel_filterbank']

        lines = []
        lines.append('// Auto-generated Mel Feature Extractor config for HiFi 5 DSP')
        lines.append('// Do not edit manually')
        lines.append('')
        lines.append('#ifndef MEL_FEATURE_CONFIG_H')
        lines.append('#define MEL_FEATURE_CONFIG_H')
        lines.append('')
        lines.append(f'#define MEL_SAMPLE_RATE    {params["sample_rate"]}')
        lines.append(f'#define MEL_N_MELS         {params["n_mels"]}')
        lines.append(f'#define MEL_N_FFT          {params["n_fft"]}')
        lines.append(f'#define MEL_WIN_LENGTH     {params["win_length"]}')
        lines.append(f'#define MEL_HOP_LENGTH     {params["hop_length"]}')
        lines.append(f'#define MEL_F_MIN          {params["f_min"]}f')
        lines.append(f'#define MEL_F_MAX          {params["f_max"]}f')
        lines.append(f'#define MEL_PREEMPHASIS     {params["preemphasis"]}f')
        lines.append(f'#define MEL_N_FFT_HALF     ({params["n_fft"]} / 2 + 1)')
        lines.append('')

        # Mel 滤波器组矩阵
        if mel_fb is not None:
            lines.append(f'// Mel filterbank matrix [{mel_fb.shape[0]} x {mel_fb.shape[1]}]')
            lines.append(f'static const float mel_filterbank[{mel_fb.shape[0]}][{mel_fb.shape[1]}] = {{')
            for i in range(mel_fb.shape[0]):
                row = ', '.join(f'{v:.8f}f' for v in mel_fb[i])
                lines.append(f'    {{{row}}},')
            lines.append('};')

        lines.append('')
        lines.append('#endif // MEL_FEATURE_CONFIG_H')
        lines.append('')

        return '\n'.join(lines)


def streaming_mel_extract(audio_chunk: np.ndarray,
                          extractor: MelFeatureExtractor,
                          prev_buffer: Optional[np.ndarray] = None,
                          ) -> tuple:
    """
    流式 Mel 特征提取 (模拟 DSP 逐帧处理)

    Args:
        audio_chunk: 新音频 chunk [T] @ 16kHz
        extractor: MelFeatureExtractor 实例
        prev_buffer: 上一帧的尾部缓冲区
            (用于窗口重叠, 包含最后的 win_length - hop_length 个样本)

    Returns:
        mel_frames: [n_mels, n_new_frames] 新产生的特征帧
        next_buffer: 更新后的缓冲区 (传给下一帧)
    """
    if prev_buffer is None:
        # 初始缓冲区 (填充零)
        audio = audio_chunk
    else:
        audio = np.concatenate([prev_buffer, audio_chunk])

    # 提取特征
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
    mel = extractor.extract(audio_tensor)  # [1, 1, 40, T]

    # 更新缓冲区: 保留最后 win_length - hop_length 个样本
    buffer_len = extractor.win_length - extractor.hop_length
    if len(audio) >= buffer_len:
        next_buffer = audio[-buffer_len:]
    else:
        next_buffer = audio

    n_frames = mel.size(3)
    return mel.squeeze(0).numpy(), next_buffer
