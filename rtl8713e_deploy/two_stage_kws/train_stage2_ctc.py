#!/usr/bin/env python3
"""
Stage 2 命令词 CTC 模型训练

用法:
    python train_stage2_ctc.py \
        --data_root ./cmd_data \
        --commands commands.txt \
        --epochs 50 --batch_size 64

数据准备:
    cmd_data/
    ├── commands.txt         # 命令列表, 一行一个
    ├── wav/                 # 音频文件
    │   ├── spk01_cmd001.wav
    │   └── ...
    ├── train.jsonl          # {"path":"wav/spk01_cmd001.wav","text":"打开客厅的灯","speaker":"spk01"}
    └── val.jsonl

输出:
    checkpoints/stage2/
    ├── best_model.pt       # CTCEncoder PyTorch 模型
    ├── grammar.json        # WFST 语法图
    ├── token_map.json      # token id -> token 映射
    └── stage2_model.h      # C 数组 (AC7916 用)
"""

import os, sys, argparse, random, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from collections import Counter

_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)
from stage2_command import CTCEncoder
from wfst_decoder import WFSTGrammarDecoder, build_command_grammar


# ==================== Tokenizer ====================

class SimpleTokenizer:
    """简单汉字 tokenizer (训练CTC用)"""
    def __init__(self, texts=None):
        self.blank_id = 0
        self.space_id = 1
        self.specials = ['<blank>', '<space>', '<unk>', '<sos>', '<eos>']

        chars = set()
        if texts:
            for t in texts:
                chars.update(t)
        self.chars = self.specials + sorted(chars)
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.i2c = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.c2i.get(c, self.c2i['<unk>']) for c in text]

    def decode(self, ids):
        return ''.join(self.i2c.get(i, '?') for i in ids
                       if i not in (0, 3, 4))

    def __len__(self):
        return len(self.chars)


# ==================== Dataset ====================

class CommandDataset(Dataset):
    """命令词 CTC 训练集"""

    def __init__(self, data_root, metadata_file, tokenizer, training=True):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.training = training
        self.sr = 16000
        self.max_frames = 200  # max mel frames (~2s)

        self.samples = []
        with open(os.path.join(data_root, metadata_file), encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.samples.append(item)

        print(f"[Stage2 Dataset] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = self._load(os.path.join(self.data_root, s['path']))
        if self.training:
            audio = self._augment(audio)
        mel = self._extract_mel(audio)  # [40, T]

        if mel.shape[1] > self.max_frames:
            start = random.randint(0, mel.shape[1] - self.max_frames)
            mel = mel[:, start:start + self.max_frames]
        elif mel.shape[1] < self.max_frames:
            pad = self.max_frames - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad)))

        tokens = self.tokenizer.encode(s['text'])
        return {
            'mel': torch.FloatTensor(mel),       # [40, T]
            'tokens': torch.LongTensor(tokens),   # [L]
            'token_len': len(tokens),
        }

    def _load(self, path):
        try:
            import soundfile as sf
            a, sr = sf.read(path, dtype='float32')
            if sr != self.sr:
                from scipy import signal
                a = signal.resample(a, int(len(a) * self.sr / sr))
            return a.astype(np.float32)
        except:
            return np.random.randn(self.sr).astype(np.float32) * 0.01

    def _augment(self, a):
        if random.random() < 0.5:
            from scipy import signal
            a = signal.resample(a, int(len(a) * random.uniform(0.9, 1.1))).astype(np.float32)
        snr = random.uniform(5, 20)
        sp = np.mean(a**2) + 1e-10
        a = a + np.random.randn(len(a)).astype(np.float32) * np.sqrt(sp / 10**(snr/10))
        return np.clip(a, -1, 1).astype(np.float32)

    def _extract_mel(self, audio):
        n_fft, hop, win = 512, 160, 400
        window = torch.hann_window(win)
        x = torch.FloatTensor(audio).unsqueeze(0)
        stft = torch.stft(x, n_fft, hop, win, window=window,
                          return_complex=True, center=True)
        power = stft.abs() ** 2
        mel_fb = self._mel_fb(40, n_fft // 2 + 1, self.sr)
        mel = torch.matmul(mel_fb, power.squeeze(0))
        return torch.log(mel + 1e-6).numpy()

    @staticmethod
    def _mel_fb(n_mels, n_freq, sr):
        mel = np.linspace(2595*np.log10(1+80/700), 2595*np.log10(1+7600/700), n_mels+2)
        hz = 700*(10**(mel/2595)-1)
        bins = np.floor(n_freq * hz / (sr/2)).astype(int)
        fb = np.zeros((n_mels, n_freq))
        for m in range(n_mels):
            for k in range(bins[m], bins[m+1]):
                fb[m, k] = (k - bins[m]) / max(bins[m+1] - bins[m], 1)
            for k in range(bins[m+1], min(bins[m+2], n_freq)):
                fb[m, k] = (bins[m+2] - k) / max(bins[m+2] - bins[m+1], 1)
        return torch.FloatTensor(fb)


def ctc_collate(batch):
    max_t = max(b['mel'].shape[1] for b in batch)
    max_l = max(b['token_len'] for b in batch)

    mels = torch.zeros(len(batch), 40, max_t)
    tokens = torch.zeros(len(batch), max_l, dtype=torch.long)
    mel_lens = torch.zeros(len(batch), dtype=torch.long)
    token_lens = torch.zeros(len(batch), dtype=torch.long)

    for i, b in enumerate(batch):
        t = b['mel'].shape[1]
        mels[i, :, :t] = b['mel']
        mel_lens[i] = t // 4  # 4x subsampling in encoder

        l = b['token_len']
        tokens[i, :l] = b['tokens']
        token_lens[i] = l

    return {'mel': mels, 'mel_lens': mel_lens,
            'tokens': tokens, 'token_lens': token_lens}


# ==================== Training ====================

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load commands
    with open(args.commands, encoding='utf-8') as f:
        commands = [line.strip() for line in f if line.strip()]
    all_texts = commands.copy()

    # Also collect texts from training data
    train_json = os.path.join(args.data_root, args.train_metadata)
    if os.path.exists(train_json):
        with open(train_json, encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                all_texts.append(item['text'])

    tokenizer = SimpleTokenizer(all_texts)
    print(f"Tokenizer: {len(tokenizer)} tokens")
    print(f"Commands: {len(commands)}")

    # Model
    model = CTCEncoder(input_dim=40, hidden_dim=128,
                       num_tokens=len(tokenizer), num_layers=3)
    model = model.to(args.device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Stage2 Model: {n_p/1000:.0f}K params ({n_p/1024:.1f}KB INT8)")

    # Data
    train_ds = CommandDataset(args.data_root, args.train_metadata,
                               tokenizer, training=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=ctc_collate,
                              drop_last=True, num_workers=2)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, reduction='mean',
                          zero_infinity=True)

    scaler = GradScaler(enabled=args.amp)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss, steps = 0, 0

        for batch in train_loader:
            mel = batch['mel'].to(args.device)
            mel_lens = batch['mel_lens'].to(args.device)
            tokens = batch['tokens'].to(args.device)
            token_lens = batch['token_lens'].to(args.device)

            with autocast(enabled=args.amp):
                log_probs = model(mel)  # [B, V, T']
                # CTC needs [T, B, V]
                log_probs_ctc = log_probs.permute(2, 0, 1)
                loss = ctc_loss(log_probs_ctc, tokens, mel_lens, token_lens)

            if torch.isinf(loss) or torch.isnan(loss):
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1

            if steps % 50 == 0:
                print(f"Epoch {epoch+1:3d} | Step {steps:4d} | "
                      f"Loss {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(steps, 1)
        print(f"--- Epoch {epoch+1}/{args.epochs} | Avg Loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'tokenizer': {k: tokenizer.__dict__[k] for k in ['c2i', 'i2c', 'blank_id']},
            }
            torch.save(ckpt, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f"  -> Best model")

    # Final
    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'final_model.pt'))

    # Build & save WFST grammar
    print(f"\nBuilding WFST grammar...")
    grammar = build_command_grammar(commands, tokenizer,
                                     os.path.join(args.checkpoint_dir, 'grammar.json'))

    # Export
    _export_cto_c(model, tokenizer, args.checkpoint_dir)

    print(f"\nDone! Files in {args.checkpoint_dir}/:")
    for f in sorted(os.listdir(args.checkpoint_dir)):
        print(f"  {f}")
    print(f"\nDeploy to AC7916AB:")
    print(f"  1. Copy stage2_model.h to SDK include/")
    print(f"  2. Copy grammar.json to SDK data/")
    print(f"  3. Accelerator runs encoder, CPU runs WFST decoder")


def _export_cto_c(model, tokenizer, out_dir):
    """导出 CTCEncoder 为 C 数组"""
    model.eval()
    path = os.path.join(out_dir, 'stage2_model.h')
    with open(path, 'w') as f:
        f.write('// Auto-generated Stage2 CTC Encoder for AC7916\n')
        f.write(f'// Tokens: {len(tokenizer)}\n\n')
        f.write('#ifndef STAGE2_MODEL_H\n#define STAGE2_MODEL_H\n')
        f.write('#include <stdint.h>\n\n')

        for name, param in model.named_parameters():
            data = param.data.cpu().numpy()
            scale = max(abs(data.min()), abs(data.max())) / 127.0 or 1e-8
            q = np.clip(np.round(data / scale), -128, 127).astype(np.int8)
            cname = name.replace('.', '_')
            f.write(f'static const int8_t {cname}[{q.size}] = {{')
            f.write(', '.join(str(v) for v in q.flatten()))
            f.write(f'}};\nstatic const float {cname}_scale = {scale:.6f}f;\n\n')

        f.write(f'#define STAGE2_NUM_TOKENS {len(tokenizer)}\n')
        f.write(f'#define STAGE2_BLANK_ID {tokenizer.blank_id}\n')
        f.write('#endif\n')

    print(f"[Export] C header: {path} ({os.path.getsize(path)} bytes)")


def parse_args():
    p = argparse.ArgumentParser(description='Train Stage 2 Command CTC ASR')
    p.add_argument('--data_root', required=True)
    p.add_argument('--commands', required=True, help='commands.txt, one per line')
    p.add_argument('--train_metadata', default='train.jsonl')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint_dir', default='./checkpoints/stage2')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
