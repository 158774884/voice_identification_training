#!/usr/bin/env python3
"""
Stage 2 命令分类器训练 (替代CTC, 避免对齐崩溃)

192 条命令 → 192 个输出类
模型: DS-CNN 分类器, ~300K params

用法:
    python train_stage2_classifier.py \
        --data_root ./data/cmd_data \
        --commands ./data/cmd_data/commands.txt \
        --epochs 60 --batch_size 64
"""

import os, sys, argparse, random, json
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)

# ==================== 分类器模型 ====================

class CommandClassifierV2(nn.Module):
    """命令分类器: 192 类, ~300K 参数"""
    def __init__(self, num_classes=192, n_mels=40, n_frames=200):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), stride=(1,2), padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )

        # Block 1: DS-Conv
        self.dw1 = nn.Conv2d(32, 32, (3,3), stride=(2,2), padding=1, groups=32, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pw1 = nn.Conv2d(32, 64, 1, bias=False)
        self.bn1b = nn.BatchNorm2d(64)

        # Block 2: DS-Conv
        self.dw2 = nn.Conv2d(64, 64, (3,3), stride=(2,2), padding=1, groups=64, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pw2 = nn.Conv2d(64, 128, 1, bias=False)
        self.bn2b = nn.BatchNorm2d(128)

        # Block 3: DS-Conv
        self.dw3 = nn.Conv2d(128, 128, (3,3), stride=(2,2), padding=1, groups=128, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.pw3 = nn.Conv2d(128, 128, 1, bias=False)
        self.bn3b = nn.BatchNorm2d(128)

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = F.relu(self.bn1b(self.pw1(F.relu(self.bn1(self.dw1(x))))))
        x = F.relu(self.bn2b(self.pw2(F.relu(self.bn2(self.dw2(x))))))
        x = F.relu(self.bn3b(self.pw3(F.relu(self.bn3(self.dw3(x))))))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x


# ==================== Dataset ====================

class CommandClsDataset(Dataset):
    def __init__(self, data_root, metadata_file, label_map, training=True):
        self.data_root = data_root
        self.label_map = label_map
        self.training = training
        self.sr = 16000
        self.max_frames = 200

        self.samples = []
        with open(os.path.join(data_root, metadata_file), encoding='utf-8') as f:
            for line in f:
                s = json.loads(line.strip())
                if s['text'] in label_map:
                    self.samples.append(s)

        print(f"[Classifier DS] {len(self.samples)} samples, {len(label_map)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = self._load(s['path'])
        if self.training: audio = self._aug(audio)
        mel = self._mel(audio)
        mel = torch.FloatTensor(mel).unsqueeze(0)  # [1, 40, T]
        return {'mel': mel, 'label': self.label_map[s['text']]}

    def _load(self, path):
        try:
            import soundfile as sf
            a, sr = sf.read(path, dtype='float32')
            if sr != self.sr:
                from scipy import signal
                a = signal.resample(a, int(len(a) * self.sr / sr))
            return a.astype(np.float32)
        except: return np.random.randn(self.sr).astype(np.float32)*0.01

    def _aug(self, a):
        if random.random() < 0.5:
            from scipy import signal
            a = signal.resample(a, int(len(a)*random.uniform(0.9,1.1))).astype(np.float32)
        snr = random.uniform(5, 20)
        a = a + np.random.randn(len(a)).astype(np.float32)*np.sqrt(np.mean(a**2)/(10**(snr/10))+1e-10)
        return np.clip(a, -1, 1).astype(np.float32)

    def _mel(self, audio):
        emph = np.zeros_like(audio); emph[0]=audio[0]; emph[1:]=audio[1:]-0.97*audio[:-1]
        n_fft, hop, win = 512, 160, 400
        window = np.hanning(win)
        frames = []
        for i in range(0, len(emph)-win+1, hop):
            spec = np.fft.rfft(emph[i:i+win]*window, n=n_fft)
            frames.append(np.abs(spec)**2)
        if not frames: return np.zeros((40, self.max_frames), dtype=np.float32)
        power = np.array(frames).T
        fb = _mel_fb(40, n_fft//2+1, self.sr)
        mel = np.log(np.dot(fb, power)+1e-6)
        if mel.shape[1] > self.max_frames: mel = mel[:,:self.max_frames]
        elif mel.shape[1] < self.max_frames: mel = np.pad(mel,((0,0),(0,self.max_frames-mel.shape[1])))
        return mel.astype(np.float32)


def _mel_fb(n_mels, n_freq, sr):
    mel = np.linspace(2595*np.log10(1+80/700), 2595*np.log10(1+7600/700), n_mels+2)
    hz = 700*(10**(mel/2595)-1)
    bins = np.floor(n_freq*hz/(sr/2)).astype(int)
    fb = np.zeros((n_mels, n_freq))
    for m in range(n_mels):
        for k in range(bins[m], bins[m+1]):
            fb[m,k] = (k-bins[m])/max(bins[m+1]-bins[m],1)
        for k in range(bins[m+1], min(bins[m+2], n_freq)):
            fb[m,k] = (bins[m+2]-k)/max(bins[m+2]-bins[m+1],1)
    return fb


def collate(batch):
    mels = torch.stack([b['mel'] for b in batch])
    labels = torch.LongTensor([b['label'] for b in batch])
    return {'mel': mels, 'label': labels}


# ==================== Training ====================

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open(args.commands, encoding='utf-8') as f:
        cmds = [l.strip() for l in f if l.strip()]
    label_map = {c: i for i, c in enumerate(cmds)}
    print(f"Commands: {len(cmds)}")

    model = CommandClassifierV2(num_classes=len(cmds))
    start_epoch = 0
    best_acc = 0

    # Resume from checkpoint if exists
    resume_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_acc = ckpt.get('best_acc', 0)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    model = model.to(args.device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_p/1000:.0f}K params ({n_p/1024:.1f}KB)")

    train_ds = CommandClsDataset(args.data_root, args.train_metadata, label_map, training=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            mel = batch['mel'].to(args.device)
            labels = batch['label'].to(args.device)

            logits = model(mel)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1:3d} | Step {batch_idx:4d} | "
                      f"Loss {loss.item():.4f} | Acc {correct/max(total,1):.3f}")

        scheduler.step()
        acc = correct / max(total, 1)
        print(f"--- Epoch {epoch+1}/{args.epochs} | Loss {total_loss/len(train_loader):.4f} | Acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'label_map': label_map, 'num_classes': len(cmds),
                'commands': cmds, 'best_acc': best_acc,
            }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f"  -> Best (acc={best_acc:.4f})")

    torch.save({
        'model_state_dict': model.state_dict(),
        'label_map': label_map, 'num_classes': len(cmds),
        'commands': cmds,
    }, os.path.join(args.checkpoint_dir, 'final_model.pt'))

    print(f"\nDone! Best acc: {best_acc:.4f}")
    print(f"Model: {args.checkpoint_dir}/best_model.pt")


def parse_args():
    p = argparse.ArgumentParser(description='Train Stage 2 Command Classifier')
    p.add_argument('--data_root', required=True)
    p.add_argument('--commands', required=True)
    p.add_argument('--train_metadata', default='train.jsonl')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint_dir', default='./checkpoints/stage2_cls')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
