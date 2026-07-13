#!/usr/bin/env python3
"""
两级语音流水线 — 端到端测试

模拟完整使用场景: 先唤醒, 再说命令

用法:
    python tests/test_pipeline.py                           # 交互测试
    python tests/test_pipeline.py --scene "wake_then_cmd"   # 场景: 唤醒+识别
    python tests/test_pipeline.py --mic                     # 实时麦克风

场景模式:
    wake_only:   只测试唤醒词检测 (用真实唤醒录音)
    cmd_only:    只测试命令识别 (跳过唤醒)
    wake_then_cmd: 先唤醒, 再识别命令 (完整流程)
"""

import os, sys, argparse, time, json, random
import torch, numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rtl8713e_deploy', 'two_stage_kws'))
from stage1_wakeword import UltraTinyWakeWord, WakeWordDetector
from stage2_command import CTCEncoder
from wfst_decoder import WFSTGrammarDecoder

# ==================== 模型加载 ====================

def load_stage1(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    n_classes = ckpt['num_classes']
    model = UltraTinyWakeWord(num_wake_words=n_classes, n_mels=40, size='micro')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    wake_words = ckpt.get('wake_words', ['wake_0'])
    detector = WakeWordDetector(model, wake_labels=wake_words)
    return detector, wake_words

def load_stage2(checkpoint_path, grammar_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    n_tokens = len(ckpt['tokenizer']['c2i'])
    model = CTCEncoder(num_tokens=n_tokens)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    decoder = WFSTGrammarDecoder.load(grammar_path)
    id2token = {int(k): v for k, v in ckpt['tokenizer']['i2c'].items()}
    return model, decoder, id2token


# ==================== Mel 特征 (复用) ====================

def extract_mel_online(audio_generator, sr=16000):
    """流式 Mel 提取, 逐帧 yield (无 pre-emphasis, 与训练一致)"""
    n_fft, hop, win = 512, 160, 400
    window = np.hanning(win)
    mel_fb = _mel_fb(40, n_fft // 2 + 1, sr)

    audio_buffer = np.zeros(0, dtype=np.float32)

    for chunk in audio_generator:
        chunk = chunk.astype(np.float32).flatten()
        audio_buffer = np.concatenate([audio_buffer, chunk])

        while len(audio_buffer) >= win:
            frame = audio_buffer[:win].copy()
            # Window (no pre-emphasis — matches training)
            frame_w = frame * window
            # FFT
            spec = np.fft.rfft(frame_w, n=n_fft)
            power = np.abs(spec) ** 2
            # Mel
            mel = np.dot(mel_fb, power)
            mel = np.log(mel + 1e-6)
            yield mel.astype(np.float32)
            # Hop
            audio_buffer = audio_buffer[hop:]


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


def _mel_fb_tensor(n_mels, n_freq, sr):
    """同 _mel_fb 但返回 torch.FloatTensor"""
    fb = _mel_fb(n_mels, n_freq, sr)
    return torch.FloatTensor(fb)


# ==================== CTC 贪婪解码 + 命令匹配 ====================

# 加载命令列表 (module-level, 只加载一次)
_cmd_list = None

def _load_commands():
    global _cmd_list
    if _cmd_list is None:
        cmd_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cmd_data', 'commands.txt')
        with open(cmd_path, encoding='utf-8') as f:
            _cmd_list = [l.strip() for l in f if l.strip()]


def _ctc_greedy_match(log_probs, id2token):
    """CTC 贪婪解码, 然后匹配到最近的命令词"""
    _load_commands()

    # CTC greedy decode: argmax per frame, collapse consecutive duplicates, remove blank
    ids = log_probs.argmax(axis=1)
    collapsed = []
    for tid in ids:
        if tid != 0 and (not collapsed or tid != collapsed[-1]):
            collapsed.append(int(tid))
    raw = ''.join(id2token.get(t, '?') for t in collapsed)

    # 如果在命令列表中直接命中
    if raw in _cmd_list:
        return raw

    # 否则编辑距离最近匹配
    if not _cmd_list:
        return raw
    best = min(_cmd_list, key=lambda c: _edit_distance(raw, c))
    return best


def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return dp[m][n]


# ==================== 场景测试 ====================

def scene_wake_only(detector, wake_words):
    """场景 1: 只用唤醒录音测试唤醒检测"""
    wake_dir = 'data/wake_data/wake'
    files = [os.path.join(wake_dir, f) for f in os.listdir(wake_dir)
             if f.endswith('.wav')]
    random.shuffle(files)

    print(f"\n{'='*50}")
    print(f"  Scene: Wake Word Detection")
    print(f"  Files: {len(files[:20])} (random subset)")
    print(f"  Expected: 检测到 '{wake_words[0]}'")
    print(f"{'='*50}\n")

    correct = 0
    for i, path in enumerate(files[:20]):
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1: audio = audio[:, 0]

        print(f"  [{i+1:2d}] {os.path.basename(path)[:40]}...", end=' ')

        detector.reset()
        any_wake = False
        for mel in extract_mel_online([audio[s:s+160] for s in range(0, len(audio), 160)], sr):
            is_wake, label, conf = detector.process_frame(torch.FloatTensor([mel]))

            if is_wake:
                print(f"WAKE! '{label}' (conf={conf:.3f})")
                any_wake = True
                break

        if any_wake:
            correct += 1
        else:
            print("MISS")

    print(f"\n  Result: {correct}/{len(files[:20])} detected ({correct/20:.0%})")


def scene_cmd_only(model2, decoder, id2token):
    """场景 2: 直接识别命令 (跳过唤醒)"""
    val_path = 'data/cmd_data/val.jsonl'
    with open(val_path, encoding='utf-8') as f:
        samples = [json.loads(l) for l in f]
    random.shuffle(samples)

    print(f"\n{'='*50}")
    print(f"  Scene: Command Recognition (skip wake)")
    print(f"  Samples: {len(samples[:10])}")
    print(f"{'='*50}\n")

    correct = 0
    for i, s in enumerate(samples[:10]):
        audio, sr = sf.read(s['path'], dtype='float32')
        if audio.ndim > 1: audio = audio[:, 0]

        mel = extract_mel_full(audio, sr)
        mel_t = torch.FloatTensor(mel).unsqueeze(0)

        with torch.no_grad():
            log_probs = model2(mel_t)
            lp_np = log_probs[0].permute(1, 0).cpu().numpy()

        # CTC 贪婪解码 + 命令匹配 (比 WFST 语法解码更可靠)
        pred = _ctc_greedy_match(lp_np, id2token)

        match = 'OK' if pred == s['text'] else 'MISS'
        print(f"  [{i+1:2d}] pred: {pred:<20s} | true: {s['text']:<20s} {match}")
        if pred == s['text']:
            correct += 1

    print(f"\n  Result: {correct}/{len(samples[:10])} correct ({correct/10:.0%})")


def scene_full_pipeline(detector, model2, decoder, id2token, wake_words):
    """场景 3: 完整两级流程 (唤醒 + 识别)"""
    wake_dir = 'data/wake_data/wake'
    wake_files = [os.path.join(wake_dir, f) for f in os.listdir(wake_dir)
                  if f.endswith('.wav')][:5]

    # 随机选几个命令文件作为"唤醒后的命令"
    val_path = 'data/cmd_data/val.jsonl'
    with open(val_path, encoding='utf-8') as f:
        cmd_samples = [json.loads(l) for l in f]
    random.shuffle(cmd_samples)

    print(f"\n{'='*50}")
    print(f"  Scene: Full Pipeline (Wake -> Command)")
    print(f"  5 test rounds")
    print(f"{'='*50}\n")

    for round_idx in range(5):
        print(f"--- Round {round_idx+1} ---")

        # Step 1: 播放唤醒词录音
        wake_path = wake_files[round_idx % len(wake_files)]
        audio, sr = sf.read(wake_path, dtype='float32')
        if audio.ndim > 1: audio = audio[:, 0]

        print(f"  1. Playing wake word: {os.path.basename(wake_path)[:30]}...")
        detector.reset()
        woke = False

        for mel in extract_mel_online([audio[s:s+160] for s in range(0, len(audio), 160)], sr):
            is_wake, label, conf = detector.process_frame(torch.FloatTensor([mel]))
            if is_wake:
                print(f"     -> Wake! '{label}' (conf={conf:.3f})")
                woke = True
                break

        if not woke:
            print("     -> MISS - skip command")
            continue

        # Step 2: 播放命令录音
        cmd = cmd_samples[round_idx % len(cmd_samples)]
        audio2, sr2 = sf.read(cmd['path'], dtype='float32')
        if audio2.ndim > 1: audio2 = audio2[:, 0]

        mel = extract_mel_full(audio2, sr2)
        mel_t = torch.FloatTensor(mel).unsqueeze(0)

        with torch.no_grad():
            log_probs = model2(mel_t)
            lp_np = log_probs[0].permute(1, 0).cpu().numpy()

        pred = _ctc_greedy_match(lp_np, id2token)
        match = 'OK' if pred == cmd['text'] else 'MISMATCH'

        print(f"  2. Command: pred='{pred}' | true='{cmd['text']}' -> {match}")
        print()


def extract_mel_full(audio, sr=16000):
    """全量 Mel 提取 (非流式, 用于 Stage2)

    与训练代码完全一致: torch.stft, center=True, 无 pre-emphasis
    """
    if sr != 16000:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
    if audio.ndim > 1:
        audio = audio[:, 0]

    n_fft, hop, win = 512, 160, 400
    window = torch.hann_window(win)
    x = torch.FloatTensor(audio).unsqueeze(0)
    stft = torch.stft(x, n_fft, hop, win, window=window,
                      return_complex=True, center=True)
    power = stft.abs() ** 2  # [1, 257, T]
    mel_fb = _mel_fb_tensor(40, n_fft // 2 + 1, sr)
    mel = torch.matmul(mel_fb, power.squeeze(0))  # [40, T]
    mel = torch.log(mel + 1e-6).numpy()

    max_t = 200
    if mel.shape[1] > max_t:
        mel = mel[:, :max_t]
    elif mel.shape[1] < max_t:
        mel = np.pad(mel, ((0, 0), (0, max_t - mel.shape[1])))
    return mel.astype(np.float32)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Two-Stage Pipeline Test')
    parser.add_argument('--scene', default='all',
                        choices=['wake_only', 'cmd_only', 'wake_then_cmd', 'all'])
    parser.add_argument('--stage1_ckpt', default='checkpoints/stage1/best_model.pt')
    parser.add_argument('--stage2_ckpt', default='checkpoints/stage2/best_model.pt')
    parser.add_argument('--grammar', default='checkpoints/stage2/grammar.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("Loading models...")
    detector, wake_words = load_stage1(args.stage1_ckpt, args.device)
    model2, decoder, id2token = load_stage2(args.stage2_ckpt, args.grammar, args.device)

    print(f"Stage 1: {sum(p.numel() for p in detector.model.parameters()):,} params")
    print(f"Stage 2: {sum(p.numel() for p in model2.parameters())/1000:.0f}K params")
    print(f"Grammar: {len(decoder.grammar)} states")
    print(f"Device:  {args.device}")

    if args.scene in ('wake_only', 'all'):
        scene_wake_only(detector, wake_words)

    if args.scene in ('cmd_only', 'all'):
        scene_cmd_only(model2, decoder, id2token)

    if args.scene in ('wake_then_cmd', 'all'):
        scene_full_pipeline(detector, model2, decoder, id2token, wake_words)


if __name__ == '__main__':
    main()
