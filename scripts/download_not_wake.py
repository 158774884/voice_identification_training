#!/usr/bin/env python3
"""
下载非唤醒词语音数据 (not_wake) + 背景噪音

用法:
    # 只下载噪音 (小, 快):
    python download_not_wake.py --data_root ../data/wake_data --noise_only

    # 下载噪音 + 语音 (推荐):
    python download_not_wake.py --data_root ../data/wake_data

    # 仅生成合成数据 (零下载, 即刻可用):
    python download_not_wake.py --data_root ../data/wake_data --synthetic_only

输出:
    wake_data/
    ├── not_wake/
    │   ├── synth_noise_0001.wav     ← 人工合成噪音
    │   ├── synth_speech_0001.wav    ← 合成语音
    │   └── musan_noise_0001.wav     ← MUSAN 真实噪音 (如果下载了)
    └── noise/
        ├── white_noise_0001.wav     ← 白噪声 (用于数据增强)
        ├── pink_noise_0001.wav      ← 粉红噪声
        ├── babble_noise_0001.wav    ← 模拟人声噪音
        └── ...
"""

import os, sys, argparse, random, subprocess, tarfile, io, glob
import numpy as np
from scipy import signal
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
try:
    import soundfile as sf
except ImportError:
    sf = None


# ============== 合成数据生成 ==============

def generate_synthetic_noise(data_root, num_files=500):
    """生成合成噪音文件 (白噪声, 粉红噪声, 棕色噪声, 多频率音)"""
    sr = 16000
    not_wake_dir = os.path.join(data_root, 'not_wake')
    noise_dir = os.path.join(data_root, 'noise')
    os.makedirs(not_wake_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)

    noise_types = ['white', 'pink', 'brown', 'tonal', 'chirp', 'impulse']
    durations = [1.0, 1.5, 2.0, 3.0, 5.0]

    print(f"[Synth] Generating {num_files} synthetic noise files...")
    for i in range(num_files):
        duration = random.choice(durations)
        n_samples = int(sr * duration)
        noise_type = random.choice(noise_types)

        if noise_type == 'white':
            audio = np.random.randn(n_samples).astype(np.float32)
        elif noise_type == 'pink':
            audio = _pink_noise(n_samples)
        elif noise_type == 'brown':
            audio = _brown_noise(n_samples)
        elif noise_type == 'tonal':
            freqs = [random.uniform(100, 4000) for _ in range(random.randint(1, 5))]
            t = np.arange(n_samples) / sr
            audio = sum(np.sin(2 * np.pi * f * t) * random.uniform(0.1, 0.5)
                       for f in freqs).astype(np.float32)
        elif noise_type == 'chirp':
            t = np.arange(n_samples) / sr
            f0, f1 = random.uniform(100, 500), random.uniform(1000, 4000)
            audio = signal.chirp(t, f0, t[-1], f1).astype(np.float32)
        elif noise_type == 'impulse':
            audio = np.zeros(n_samples, dtype=np.float32)
            for _ in range(random.randint(3, 20)):
                pos = random.randint(0, n_samples - 100)
                audio[pos:pos + random.randint(1, 50)] = random.uniform(0.3, 0.9)

        # 归一化
        peak = np.max(np.abs(audio)) + 1e-8
        audio = audio / peak * random.uniform(0.1, 0.8)

        # 决定放 not_wake 还是 noise
        if random.random() < 0.7:
            dst = os.path.join(not_wake_dir, f'synth_{noise_type}_{i:04d}.wav')
        else:
            dst = os.path.join(noise_dir, f'synth_{noise_type}_{i:04d}.wav')

        _write_wav(dst, audio, sr)

    print(f"[Synth] Done: {num_files} files in {not_wake_dir} + {noise_dir}")

    # 额外: 生成模拟"人声噪音" (多人混合语音, 模拟公共场所)
    print(f"[Synth] Generating babble noise...")
    for i in range(50):
        duration = random.uniform(2.0, 5.0)
        n_s = int(sr * duration)
        # 多路随机振荡模拟多人说话
        babble = np.zeros(n_s, dtype=np.float32)
        for _ in range(random.randint(3, 8)):
            f0 = random.uniform(80, 300)
            # FM 调制模拟语音基频变化
            t = np.arange(n_s) / sr
            mod = np.sin(2 * np.pi * random.uniform(3, 8) * t)
            voice = np.sin(2 * np.pi * f0 * t + mod * 50)
            # 谐波
            for h in range(2, random.randint(2, 6)):
                voice += np.sin(2 * np.pi * f0 * h * t) * 0.3 / h
            # 振幅包络
            env = np.abs(signal.hilbert(np.random.randn(n_s).astype(np.float32)))
            env = signal.savgol_filter(env, min(51, len(env) // 2 * 2 + 1), 2)
            env = env / (np.max(env) + 1e-8)
            babble += voice * env * random.uniform(0.1, 0.3)

        babble = babble / (np.max(np.abs(babble)) + 1e-8) * random.uniform(0.2, 0.7)
        dst = os.path.join(not_wake_dir, f'synth_babble_{i:04d}.wav')
        _write_wav(dst, babble, sr)

    print(f"[Synth] Babble noise done")


def _pink_noise(n):
    """生成粉红噪声 (1/f 频谱)"""
    white = np.random.randn(n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    # 避免 DC 除零
    freqs[0] = freqs[1]
    fft = fft / np.sqrt(freqs)
    result = np.fft.irfft(fft, n=n)
    return (result / np.max(np.abs(result))).astype(np.float32)


def _brown_noise(n):
    """生成棕色噪声 (随机游走)"""
    white = np.random.randn(n).astype(np.float32)
    brown = np.cumsum(white)
    return (brown / np.max(np.abs(brown))).astype(np.float32)


# ============== MUSAN 下载 (可选) ==============

def download_musan(data_root, noise_only=True):
    """从 OpenSLR 下载 MUSAN 噪声子集"""
    if not HAS_REQUESTS:
        print("[MUSAN] requests not installed, skip download")
        print("  Install: pip install requests")
        return

    url = 'https://www.openslr.org/resources/17/musan.tar.gz'
    # 如果只要噪声, 只下载 noise/ 子集 (约 1.5GB 的 tar.gz, 我们提取需要的部分)
    dst_zip = os.path.join(data_root, 'musan.tar.gz')

    # 尝试用 wget/curl (Windows 常见问题)
    print(f"[MUSAN] Downloading noise subset from {url}")
    print(f"  Size: ~1.5 GB (noise only), this may take a while...")
    print(f"  If too slow, use --synthetic_only or download manually")

    try:
        # 只下载 noise/ 目录下的文件 (使用 HTTP Range 不太可行)
        # 改为完整下载, 用户可 Ctrl+C 中断后用手动下载
        resp = requests.get(url, stream=True, timeout=30)
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(dst_zip, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        mb = downloaded / 1024 / 1024
                        print(f"\r  Downloading... {mb:.1f} MB ({pct:.1f}%)", end='')

        print(f"\n[MUSAN] Downloaded. Extracting noise files...")

        # 只提取 noise/ 子目录
        _extract_musan_noise(dst_zip, data_root)

        # 删除 tar.gz (保留噪声 wav)
        os.remove(dst_zip)
        print(f"[MUSAN] Done. Files in {data_root}/not_wake/")

    except Exception as e:
        print(f"\n[MUSAN] Download failed: {e}")
        print(f"  Falling back to synthetic data. Use --synthetic_only next time.")
        return


def _extract_musan_noise(tar_path, data_root):
    """从 MUSAN tar.gz 中提取 noise/ 目录"""
    not_wake_dir = os.path.join(data_root, 'not_wake')
    os.makedirs(not_wake_dir, exist_ok=True)

    with tarfile.open(tar_path, 'r:gz') as tar:
        noise_members = [m for m in tar.getmembers()
                         if m.name.startswith('musan/noise/') and m.name.endswith('.wav')]
        print(f"  Extracting {len(noise_members)} noise files...")

        for i, member in enumerate(noise_members):
            tar.extract(member, data_root)
            # 移动到 not_wake/
            src = os.path.join(data_root, member.name)
            dst = os.path.join(not_wake_dir, f'musan_{member.name.split("/")[-1]}')
            if os.path.exists(src):
                os.rename(src, dst)

            if (i + 1) % 100 == 0:
                print(f"\r  Extracted {i + 1}/{len(noise_members)}", end='')

    print(f"\n  Done: {len(noise_members)} noise files")


# ============== 工具函数 ==============

def _write_wav(path, audio, sr):
    if sf is not None:
        sf.write(path, audio, sr)
    else:
        # Fallback: 写原始 PCM (16-bit)
        import struct
        raw = (np.clip(audio * 32767, -32768, 32767)).astype(np.int16)
        with open(path.replace('.wav', '.raw'), 'wb') as f:
            f.write(raw.tobytes())

# ============== 主入口 ==============

def parse_args():
    p = argparse.ArgumentParser(description='Download/generate not_wake data for wake word training')
    p.add_argument('--data_root', required=True,
                   help='Path to wake_data directory (e.g. ../data/wake_data)')
    p.add_argument('--noise_only', action='store_true',
                   help='Only generate noise files, no speech-like samples')
    p.add_argument('--synthetic_only', action='store_true',
                   help='Only generate synthetic data (no download, instant)')
    p.add_argument('--num_synthetic', type=int, default=500,
                   help='Number of synthetic samples to generate')
    p.add_argument('--download_musan', action='store_true',
                   help='Download MUSAN noise (1.5GB, requires internet)')
    return p.parse_args()


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    print(f"Target: {data_root}")
    print(f"Mode: {'synthetic only' if args.synthetic_only else 'synthetic + download'}")

    # 1. 合成数据 (总是生成, 确保有数据可用)
    if not args.noise_only:
        generate_synthetic_noise(data_root, num_files=args.num_synthetic)
    else:
        # 只生成纯噪音
        noise_dir = os.path.join(data_root, 'noise')
        not_wake_dir = os.path.join(data_root, 'not_wake')
        os.makedirs(noise_dir, exist_ok=True)
        os.makedirs(not_wake_dir, exist_ok=True)
        for i in range(args.num_synthetic // 2):
            duration = random.uniform(1, 5)
            n_s = int(16000 * duration)
            for ntype in ['white', 'pink', 'brown']:
                if ntype == 'white':
                    audio = np.random.randn(n_s).astype(np.float32) * 0.3
                elif ntype == 'pink':
                    audio = _pink_noise(n_s) * 0.3
                else:
                    audio = _brown_noise(n_s) * 0.3
                peak = np.max(np.abs(audio)) + 1e-8
                audio = audio / peak * random.uniform(0.1, 0.7)
                dst = os.path.join(noise_dir, f'synth_{ntype}_{i:04d}.wav')
                _write_wav(dst, audio, 16000)
        print(f"[Synth] Noise only: {args.num_synthetic // 2 * 3} files")

    # 2. MUSAN 下载 (可选)
    if args.download_musan and not args.synthetic_only:
        download_musan(data_root, noise_only=True)

    # 3. 统计
    print(f"\n{'='*50}")
    print(f"Data ready:")
    for sub in ['not_wake', 'noise']:
        d = os.path.join(data_root, sub)
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.endswith(('.wav', '.flac', '.raw'))]
            print(f"  {sub}/: {len(files)} files")
        else:
            print(f"  {sub}/: (not created)")
    print(f"\nNext step:")
    print(f"  1. Add your wake word recordings to: {data_root}/wake/")
    print(f"  2. Run Stage1 training: python rtl8713e_deploy/two_stage_kws/train_stage1_wakeword.py --data_root {data_root} --wake_words \"your_wake_word\"")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
