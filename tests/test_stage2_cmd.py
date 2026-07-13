#!/usr/bin/env python3
"""
Stage 2 命令词识别 — 测试脚本

用法:
    python tests/test_stage2_cmd.py                          # 在 val 集评估
    python tests/test_stage2_cmd.py --wav test.wav           # 单文件测试
    python tests/test_stage2_cmd.py --mic                    # 实时麦克风
"""

import os, sys, argparse, time
import torch, json, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rtl8713e_deploy', 'two_stage_kws'))
from stage2_command import CTCEncoder


def load_models(checkpoint_path, grammar_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    n_tokens = len(ckpt['tokenizer']['c2i'])
    model = CTCEncoder(num_tokens=n_tokens)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)

    from wfst_decoder import WFSTGrammarDecoder
    decoder = WFSTGrammarDecoder.load(grammar_path)

    id2token = {int(k): v for k, v in ckpt['tokenizer']['i2c'].items()}
    return model, decoder, id2token


def extract_mel(audio, sr=16000):
    if sr != 16000:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
    if audio.ndim > 1:
        audio = audio[:, 0]

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

    power_spec = np.array(frames).T
    mel_fb = _mel_fb(40, n_fft // 2 + 1, 16000)
    mel = np.dot(mel_fb, power_spec)
    mel = np.log(mel + 1e-6)

    max_t = 200
    if mel.shape[1] > max_t:
        mel = mel[:, :max_t]
    elif mel.shape[1] < max_t:
        mel = np.pad(mel, ((0, 0), (0, max_t - mel.shape[1])))
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


def test_single_file(model, decoder, id2token, wav_path):
    """单文件测试"""
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype='float32')
    mel = extract_mel(audio, sr)
    mel_t = torch.FloatTensor(mel).unsqueeze(0)  # [1, 40, T]

    with torch.no_grad():
        log_probs = model(mel_t)  # [1, V, T']
        lp_np = log_probs[0].permute(1, 0).cpu().numpy()  # [T', V]

    candidates = decoder.decode(lp_np, top_k=5)

    print(f"\n  File:  {os.path.basename(wav_path)}")
    print(f"  Duration: {len(audio)/sr:.1f}s")
    print(f"  Top-5 candidates:")
    for i, (tokens, score) in enumerate(candidates):
        text = ''.join(id2token.get(t, '?') for t in tokens)
        marker = " <-- BEST" if i == 0 else ""
        print(f"    {i+1}. {text:<20s}  score={score:.3f}{marker}")

    return candidates[0] if candidates else ([], -999)


def evaluate(model, decoder, id2token, val_jsonl, num_samples=500):
    """评估: 检查识别准确率"""
    with open(val_jsonl, encoding='utf-8') as f:
        samples = [json.loads(l) for l in f]

    random.shuffle(samples)
    samples = samples[:num_samples]

    import soundfile as sf
    correct = 0
    total = 0
    t0 = time.time()

    for s in samples:
        try:
            audio, sr = sf.read(s['path'], dtype='float32')
            mel = extract_mel(audio, sr)
            mel_t = torch.FloatTensor(mel).unsqueeze(0)

            with torch.no_grad():
                log_probs = model(mel_t)
                lp_np = log_probs[0].permute(1, 0).cpu().numpy()

            candidates = decoder.decode(lp_np, top_k=1)
            if candidates:
                pred_text = ''.join(id2token.get(t, '?') for t in candidates[0][0])
                if pred_text == s['text']:
                    correct += 1
            total += 1
        except Exception:
            continue

        if total % 50 == 0:
            print(f"\r  Progress: {total}/{len(samples)}", end='', flush=True)

    elapsed = time.time() - t0
    acc = correct / max(total, 1)
    print(f"\n\n  {'='*45}")
    print(f"  Stage 2 Command Recognition Test")
    print(f"  {'='*45}")
    print(f"  Samples:   {total}")
    print(f"  Correct:   {correct}")
    print(f"  Accuracy:  {acc:.2%}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/max(total,1)*1000:.0f}ms/file)")
    print(f"  {'='*45}")
    return {'acc': acc, 'correct': correct, 'total': total}


def main():
    parser = argparse.ArgumentParser(description='Stage 2 Command Test')
    parser.add_argument('--wav', type=str, help='Single wav file to test')
    parser.add_argument('--checkpoint', default='checkpoints/stage2/best_model.pt')
    parser.add_argument('--grammar', default='checkpoints/stage2/grammar.json')
    parser.add_argument('--val_jsonl', default='data/cmd_data/val.jsonl')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model, decoder, id2token = load_models(args.checkpoint, args.grammar, args.device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Stage 2 Model: {n_p/1000:.0f}K params")
    print(f"Tokens: {len(id2token)}")

    if args.wav:
        test_single_file(model, decoder, id2token, args.wav)
    else:
        evaluate(model, decoder, id2token, args.val_jsonl, args.num_samples)


if __name__ == '__main__':
    main()
