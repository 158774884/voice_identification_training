#!/usr/bin/env python3
"""
Stage 1 唤醒词检测 — 测试脚本

用法:
    python tests/test_stage1_wake.py                          # 自动评估
    python tests/test_stage1_wake.py --wav path/to/test.wav   # 单文件测试
"""

import os, sys, argparse, time
import torch, json, random
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rtl8713e_deploy', 'two_stage_kws'))
from stage1_wakeword import UltraTinyWakeWord, WakeWordDetector


def load_model(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    n_classes = ckpt['num_classes']
    model = UltraTinyWakeWord(num_wake_words=n_classes, n_mels=40, size='micro')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    wake_words = ckpt.get('wake_words', ['wake_0'])
    return model, wake_words


def extract_mel(audio, sr=16000):
    """提取 40-dim mel, 复现训练时的特征提取"""
    if sr != 16000:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))

    # Pre-emphasis
    emph = np.zeros_like(audio)
    emph[0] = audio[0]
    emph[1:] = audio[1:] - 0.97 * audio[:-1]

    n_fft, hop, win = 512, 160, 400
    window = np.hanning(win)

    frames = []
    for i in range(0, len(emph) - win + 1, hop):
        frame = emph[i:i+win] * window
        spec = np.fft.rfft(frame, n=n_fft)
        power = np.abs(spec) ** 2
        frames.append(power)

    if not frames:
        return np.zeros((40, 98), dtype=np.float32)

    power_spec = np.array(frames).T  # [257, T]

    # Mel filterbank
    mel_fb = _mel_fb(40, n_fft // 2 + 1, 16000)
    mel = np.dot(mel_fb, power_spec)
    mel = np.log(mel + 1e-6)

    # 时间对齐到 98 帧
    if mel.shape[1] > 98:
        mel = mel[:, :98]
    elif mel.shape[1] < 98:
        mel = np.pad(mel, ((0, 0), (0, 98 - mel.shape[1])))

    return mel.astype(np.float32)


def _mel_fb(n_mels, n_freq, sr):
    mel = np.linspace(2595 * np.log10(1 + 80 / 700),
                       2595 * np.log10(1 + 7600 / 700), n_mels + 2)
    hz = 700 * (10 ** (mel / 2595) - 1)
    bins = np.floor(n_freq * hz / (sr / 2)).astype(int)
    fb = np.zeros((n_mels, n_freq))
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m, k] = (k - bins[m]) / max(bins[m + 1] - bins[m], 1)
        for k in range(bins[m + 1], min(bins[m + 2], n_freq)):
            fb[m, k] = (bins[m + 2] - k) / max(bins[m + 2] - bins[m + 1], 1)
    return fb


def test_single_file(model, wake_words, wav_path):
    """测试单个文件"""
    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio[:, 0]

    mel = extract_mel(audio, sr)
    mel_tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)  # [1, 1, 40, 98]

    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    print(f"\n  File:  {os.path.basename(wav_path)}")
    print(f"  Duration: {len(audio)/sr:.1f}s")
    print(f"  Results:")
    for i, w in enumerate(wake_words):
        marker = " <-- WAKE" if i < len(wake_words)-1 and probs[i].item() > 0.5 else ""
        print(f"    {w:<15s}: {probs[i].item():.4f}{marker}")
    print(f"    not_wake      : {probs[-1].item():.4f}")

    best = probs.argmax().item()
    best_w = wake_words[best] if best < len(wake_words) else 'not_wake'
    return best_w, probs[best].item()


def evaluate(model, wake_words, data_root, num_samples=200):
    """批量评估"""
    wake_dir = os.path.join(data_root, 'wake')
    not_wake_dir = os.path.join(data_root, 'not_wake')

    wake_files = []
    if os.path.isdir(wake_dir):
        wake_files = [os.path.join(wake_dir, f) for f in os.listdir(wake_dir)
                      if f.endswith('.wav')]

    nw_files = []
    if os.path.isdir(not_wake_dir):
        nw_files = [os.path.join(not_wake_dir, f) for f in os.listdir(not_wake_dir)
                    if f.endswith(('.wav', '.raw'))]

    random.shuffle(wake_files)
    random.shuffle(nw_files)

    wake_test = wake_files[:min(len(wake_files), num_samples//2)]
    nw_test = nw_files[:min(len(nw_files), num_samples//2)]

    print(f"\n  Evaluating: {len(wake_test)} wake + {len(nw_test)} not_wake")
    detector = WakeWordDetector(model, wake_labels=wake_words)

    tp, tn, fp, fn = 0, 0, 0, 0
    t0 = time.time()

    # Test wake files
    for path in wake_test:
        try:
            audio, sr = sf.read(path, dtype='float32')
            if audio.ndim > 1: audio = audio[:, 0]
            mel = extract_mel(audio, sr)
            mel_t = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(mel_t)
                pred = logits.argmax(-1).item()
            if pred < len(wake_words):
                tp += 1
            else:
                fn += 1
        except Exception as e:
            pass

    # Test not_wake files
    for path in nw_test:
        try:
            if path.endswith('.raw'):
                audio = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768
                sr = 16000
            else:
                audio, sr = sf.read(path, dtype='float32')
                if audio.ndim > 1: audio = audio[:, 0]
            mel = extract_mel(audio, sr)
            mel_t = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(mel_t)
                pred = logits.argmax(-1).item()
            if pred >= len(wake_words):
                tn += 1
            else:
                fp += 1
        except Exception:
            pass

    elapsed = time.time() - t0
    total = tp + tn + fp + fn
    acc = (tp + tn) / max(total, 1)
    far = fp / max(fp + tn, 1)  # False accept rate
    frr = fn / max(fn + tp, 1)  # False reject rate

    print(f"\n  {'='*45}")
    print(f"  Stage 1 Wake Word Test Results")
    print(f"  {'='*45}")
    print(f"  Wake word: {wake_words}")
    print(f"  Total:     {total} files")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Accuracy:  {acc:.2%}")
    print(f"  FAR:       {far:.4f} (误唤醒率)")
    print(f"  FRR:       {frr:.4f} (漏唤醒率)")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/max(total,1)*1000:.0f}ms/file)")
    print(f"  {'='*45}")

    return {'acc': acc, 'far': far, 'frr': frr, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def main():
    parser = argparse.ArgumentParser(description='Stage 1 Wake Word Test')
    parser.add_argument('--wav', type=str, help='Single wav file to test')
    parser.add_argument('--checkpoint', default='checkpoints/stage1/best_model.pt')
    parser.add_argument('--data_root', default='data/wake_data')
    parser.add_argument('--num_samples', type=int, default=200)
    args = parser.parse_args()

    model, wake_words = load_model(args.checkpoint)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Stage 1 Model: {n_p:,} params ({n_p/1024:.1f}KB)")
    print(f"Wake words: {wake_words}")

    if args.wav:
        test_single_file(model, wake_words, args.wav)
    else:
        evaluate(model, wake_words, args.data_root, args.num_samples)


if __name__ == '__main__':
    main()
