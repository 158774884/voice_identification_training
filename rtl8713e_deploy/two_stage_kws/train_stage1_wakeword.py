#!/usr/bin/env python3
"""
Stage 1 唤醒词训练

用法:
    # 准备数据见 train_stage1_wakeword.py --help-data
    python train_stage1_wakeword.py \
        --data_root ./wake_data \
        --wake_words "小度小度,你好小智" \
        --epochs 30 --batch_size 256

数据准备:
    wake_data/
    ├── wake/          # 唤醒词音频 (每个唤醒词至少 500 条)
    │   ├── speaker001_xiaoduxiaodu_01.wav
    │   └── ...
    ├── not_wake/      # 非唤醒词语音 (日常对话/噪音, 至少 2000 条)
    │   └── ...
    └── noise/         # 纯背景噪声 (可选)
        └── ...

输出:
    checkpoints/stage1/
    ├── best_model.pt          # PyTorch 模型
    ├── wake_labels.txt        # 唤醒词标签
    └── model_c.h              # C 数组 (AC7916 CPU 可直接用)
"""

import os, sys, argparse, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)
from stage1_wakeword import UltraTinyWakeWord

# ==================== Dataset ====================

class WakeWordDataset(Dataset):
    """唤醒词数据集"""

    def __init__(self, data_root, wake_words, training=True):
        self.data_root = data_root
        self.wake_words = [w.strip() for w in wake_words.split(',')]
        self.training = training
        self.samples = []
        self.sample_rate = 16000
        self.target_frames = 98

        self._scan()

    def _scan(self):
        wake_dir = os.path.join(self.data_root, 'wake')
        not_wake_dir = os.path.join(self.data_root, 'not_wake')

        wake_files = []
        if os.path.isdir(wake_dir):
            for f in sorted(os.listdir(wake_dir)):
                if f.endswith(('.wav', '.flac', '.pcm', '.npy', '.raw')):
                    wake_files.append(os.path.join(wake_dir, f))

        # Wake samples: 取录音中间段 (包含唤醒词)
        for path in wake_files:
            label = 0
            for i, w in enumerate(self.wake_words):
                if w in path:
                    label = i; break
            self.samples.append({
                'path': path, 'label': label, 'is_wake': True,
                'offset': 'center',
            })

        # Not-wake from wake files: 开头+结尾段 (大概率不含唤醒词)
        # 始终生成, 因为这才是真正的"同类声学环境不同内容"的负样本
        for path in wake_files:
            self.samples.append({
                'path': path, 'label': len(self.wake_words),
                'is_wake': False, 'offset': 'start',
            })
            self.samples.append({
                'path': path, 'label': len(self.wake_words),
                'is_wake': False, 'offset': 'end',
            })

        # External not_wake files
        if os.path.isdir(not_wake_dir):
            for f in sorted(os.listdir(not_wake_dir)):
                if f.endswith(('.wav', '.flac', '.pcm', '.npy', '.raw')):
                    self.samples.append({
                        'path': os.path.join(not_wake_dir, f),
                        'label': len(self.wake_words),
                        'is_wake': False, 'offset': 'random',
                    })

        print(f"[Stage1 Dataset] {len(self.samples)} samples, "
              f"{len([s for s in self.samples if s['is_wake']])} wake / "
              f"{len([s for s in self.samples if not s['is_wake']])} not-wake")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = self._load_audio(s['path'])

        # --- 根据 offset 选择音频段 ---
            # Wake: longer window to catch full wake word; Not-wake: shorter edge window
        if s.get('is_wake'):
            target_samples = int(self.sample_rate * 1.2)
        else:
            target_samples = int(self.sample_rate * 0.6)  # short edge = less overlap
        offset = s.get('offset', 'random')

        if len(audio) > target_samples:
            if offset == 'center':
                # 正样本: 取中间 (包含唤醒词)
                center = len(audio) // 2
                start = max(0, center - target_samples // 2)
                start = min(start, len(audio) - target_samples)
            elif offset == 'start':
                # 负样本: 取开头 (大概率不含唤醒词)
                start = 0
            elif offset == 'end':
                # 负样本: 取结尾
                start = max(0, len(audio) - target_samples)
            else:
                # 随机段
                start = random.randint(0, len(audio) - target_samples)
            audio = audio[start:start + target_samples]
        else:
            # 音频太短, 补零
            audio = np.pad(audio, (0, max(0, target_samples - len(audio))))

        # 数据增强
        if self.training:
            audio = self._augment(audio)

        # 提取 Mel 特征
        mel = self._extract_mel(audio)  # [40, T]

        # 时间对齐
        if mel.shape[1] > self.target_frames:
            start = random.randint(0, mel.shape[1] - self.target_frames)
            mel = mel[:, start:start + self.target_frames]
        elif mel.shape[1] < self.target_frames:
            pad = self.target_frames - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad)))

        return {
            'mel': torch.FloatTensor(mel).unsqueeze(0),  # [1, 40, 98]
            'label': s['label'],
        }

    def _load_audio(self, path):
        if path.endswith('.npy'):
            return np.load(path).astype(np.float32)
        try:
            import soundfile as sf
            if path.endswith(('.raw', '.pcm')):
                # raw PCM: assume int16 @ 16kHz mono
                raw = np.fromfile(path, dtype=np.int16)
                audio = raw.astype(np.float32) / 32768.0
                sr = self.sample_rate
            else:
                audio, sr = sf.read(path, dtype='float32')
            audio = np.asarray(audio, dtype=np.float32).flatten()
            if sr != self.sample_rate:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * self.sample_rate / sr))
            return audio.astype(np.float32)
        except Exception:
            return np.random.randn(self.sample_rate).astype(np.float32) * 0.01

    def _augment(self, audio):
        if random.random() < 0.5:
            speed = random.uniform(0.9, 1.1)
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) / speed)).astype(np.float32)
        if random.random() < 0.3:
            noise = np.random.randn(len(audio)).astype(np.float32) * 0.005
            audio = audio + noise
        snr = random.uniform(5, 20)
        signal_power = np.mean(audio ** 2) + 1e-10
        noise_power = signal_power / (10 ** (snr / 10))
        audio = audio + np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
        return np.clip(audio, -1, 1).astype(np.float32)

    def _extract_mel(self, audio):
        # 简化 Mel (使用 torch.stft, 不依赖 torchaudio)
        n_fft, hop, win = 512, 160, 400
        window = torch.hann_window(win)
        x = torch.FloatTensor(audio).unsqueeze(0)
        stft = torch.stft(x, n_fft, hop, win, window=window,
                          return_complex=True, center=True)
        power = stft.abs() ** 2  # [1, 257, T]
        # 简化的 mel filter (40 bins)
        mel_fb = self._mel_filterbank(40, n_fft // 2 + 1, self.sample_rate)
        mel = torch.matmul(mel_fb, power.squeeze(0))  # [40, T]
        return torch.log(mel + 1e-6).numpy()

    def _mel_filterbank(self, n_mels, n_freq, sr):
        mel = np.linspace(self._hz_to_mel(80), self._hz_to_mel(7600), n_mels + 2)
        hz = self._mel_to_hz(mel)
        bins = np.floor((n_freq) * hz / (sr / 2)).astype(int)
        fb = np.zeros((n_mels, n_freq))
        for m in range(n_mels):
            for k in range(bins[m], bins[m + 1]):
                fb[m, k] = (k - bins[m]) / max(bins[m + 1] - bins[m], 1)
            for k in range(bins[m + 1], min(bins[m + 2], n_freq)):
                fb[m, k] = (bins[m + 2] - k) / max(bins[m + 2] - bins[m + 1], 1)
        return torch.FloatTensor(fb)

    @staticmethod
    def _hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


def collate_fn(batch):
    mels = torch.stack([b['mel'] for b in batch])
    labels = torch.LongTensor([b['label'] for b in batch])
    return {'mel': mels, 'label': labels}


# ==================== Training ====================

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    wake_words = [w.strip() for w in args.wake_words.split(',')]
    num_classes = len(wake_words) + 1  # +1 for not_wake

    # Model
    model = UltraTinyWakeWord(num_wake_words=num_classes, n_mels=40, size='micro')
    model = model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stage1 Model: {n_params} params ({n_params/1024:.1f}KB INT8)")
    print(f"Classes: {num_classes} ({len(wake_words)} wake + 1 not_wake)")

    # Data
    train_ds = WakeWordDataset(args.data_root, args.wake_words, training=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    # No class weights — data is balanced enough with start/end augmentation

    scaler = GradScaler(enabled=args.amp)
    best_acc = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            mel = batch['mel'].to(args.device)
            labels = batch['label'].to(args.device)

            with autocast(enabled=args.amp):
                logits = model(mel)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1:3d} | Step {batch_idx:4d} | "
                      f"Loss {loss.item():.4f} | Acc {correct/max(total,1):.3f}")

        scheduler.step()
        acc = correct / max(total, 1)
        print(f"--- Epoch {epoch+1}/{args.epochs} | Avg Loss {total_loss/len(train_loader):.4f} | Acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'wake_words': wake_words,
                'num_classes': num_classes,
                'params': n_params,
            }
            torch.save(ckpt, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f"  -> Best model saved (acc={best_acc:.4f})")

    # Final save
    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'final_model.pt'))

    # Export C header
    _export_c_array(model, wake_words, args.checkpoint_dir)

    print(f"\nDone! Best acc: {best_acc:.4f}")
    print(f"Model: {args.checkpoint_dir}/best_model.pt")
    print(f"C code: {args.checkpoint_dir}/stage1_model.h")


def _export_c_array(model, wake_words, out_dir):
    """导出为 C 数组 (AC7916 CPU 推理)"""
    model.eval()
    path = os.path.join(out_dir, 'stage1_model.h')
    with open(path, 'w') as f:
        f.write('// Auto-generated Stage1 WakeWord model for AC7916\n')
        f.write(f'// Wake words: {wake_words}\n\n')
        f.write('#ifndef STAGE1_MODEL_H\n#define STAGE1_MODEL_H\n\n')
        f.write('#include <stdint.h>\n\n')

        for name, param in model.named_parameters():
            data = param.data.cpu().numpy()
            # INT8 quantization
            scale = max(abs(data.min()), abs(data.max())) / 127.0
            if scale < 1e-8:
                scale = 1e-8
            q = np.clip(np.round(data / scale), -128, 127).astype(np.int8)

            cname = name.replace('.', '_')
            f.write(f'// {name} shape={list(data.shape)} scale={scale:.6f}\n')
            f.write(f'static const int8_t {cname}[{q.size}] = {{\n  ')
            f.write(', '.join(str(v) for v in q.flatten()))
            f.write(f'\n}};\nstatic const float {cname}_scale = {scale:.6f}f;\n\n')

        f.write(f'#define STAGE1_NUM_CLASSES {len(wake_words) + 1}\n')
        f.write(f'static const char* stage1_labels[{len(wake_words) + 1}] = {{\n')
        for i, w in enumerate(wake_words):
            f.write(f'  "{w}",\n')
        f.write(f'  "not_wake"\n}};\n\n')
        f.write('#endif\n')

    import os as _os
    print(f"[Export] C header: {path} ({_os.path.getsize(path)} bytes)")


def parse_args():
    p = argparse.ArgumentParser(description='Train Stage 1 Wake Word Detector')
    p.add_argument('--data_root', required=True, help='wake_data directory')
    p.add_argument('--wake_words', default='xiao_du_xiao_du', help='comma-separated wake words')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint_dir', default='./checkpoints/stage1')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
