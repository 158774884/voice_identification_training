"""
RTL8713E TinyKWS 训练脚本

用法:
    python train_kws.py --data_root ./kws_data --num_classes 50 --epochs 50
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rtl8713e_deploy.model.tiny_kws import TinyKWS, UltraTinyKWS, create_tiny_kws
from rtl8713e_deploy.model.feature_extractor import MelFeatureExtractor
from rtl8713e_deploy.data.kws_dataset import KwsDataset, create_kws_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='./kws_data')
    p.add_argument('--num_classes', type=int, default=50)
    p.add_argument('--n_mels', type=int, default=40)
    p.add_argument('--preset', type=str, default='standard',
                   choices=['micro', 'standard', 'large'])
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--checkpoint_dir', type=str, default='./kws_checkpoints')
    p.add_argument('--export', action='store_true', help='Export to ONNX after training')
    return p.parse_args()


def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ===== Feature Extractor =====
    fe = MelFeatureExtractor(sample_rate=16000, n_mels=args.n_mels)

    # ===== Model =====
    model = create_tiny_kws(num_classes=args.num_classes,
                            n_mels=args.n_mels,
                            preset=args.preset)
    model = model.to(args.device)
    model.summary()

    # Generate DSP C header
    c_header = fe.generate_dsp_c_header()
    with open(os.path.join(args.checkpoint_dir, 'mel_config.h'), 'w') as f:
        f.write(c_header)
    print(f"[Train] DSP Mel config saved to mel_config.h")

    # ===== Dataset =====
    train_ds = KwsDataset(
        data_root=args.data_root,
        metadata_file='train.jsonl',
        feature_extractor=fe,
        n_frames=98,
        training=True,
    )
    train_loader = create_kws_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Validation
    val_meta = os.path.join(args.data_root, 'val.jsonl')
    val_loader = None
    if os.path.exists(val_meta):
        val_ds = KwsDataset(
            data_root=args.data_root,
            metadata_file='val.jsonl',
            feature_extractor=fe,
            n_frames=98,
            training=False,
        )
        # Copy label mapping from train
        val_ds.label2id = train_ds.label2id
        val_ds.id2label = train_ds.id2label
        val_ds.num_classes = train_ds.num_classes
        val_ds.unknown_id = train_ds.unknown_id
        val_loader = create_kws_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Label list for inference
    label_list_path = os.path.join(args.checkpoint_dir, 'labels.txt')
    with open(label_list_path, 'w', encoding='utf-8') as f:
        for idx, label in sorted(train_ds.id2label.items()):
            f.write(f'{idx}\t{label}\n')
        f.write(f'{train_ds.unknown_id}\t<unknown>\n')
    print(f"[Train] Labels saved to {label_list_path}")

    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = GradScaler(enabled=True)

    best_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            mel = batch['mel'].to(args.device)
            labels = batch['label_id'].to(args.device)

            with autocast():
                logits = model(mel)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if batch_idx % 100 == 0:
                acc = train_correct / max(train_total, 1)
                print(f"Epoch {epoch+1:3d} | Batch {batch_idx:4d} | "
                      f"Loss: {loss.item():.4f} | Acc: {acc:.3f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        epoch_acc = train_correct / max(train_total, 1)
        print(f"--- Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {epoch_acc:.4f} ---")

        # Validation
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    mel = batch['mel'].to(args.device)
                    labels = batch['label_id'].to(args.device)
                    logits = model(mel)
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = val_correct / max(val_total, 1)
            print(f"--- Val Acc: {val_acc:.4f} ---")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'label2id': train_ds.label2id,
                    'id2label': train_ds.id2label,
                    'num_classes': args.num_classes,
                    'n_mels': args.n_mels,
                    'best_acc': best_acc,
                }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
                print(f"  → Best model saved (acc={best_acc:.4f})")

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'label2id': train_ds.label2id,
        'id2label': train_ds.id2label,
        'num_classes': args.num_classes,
    }, os.path.join(args.checkpoint_dir, 'final_model.pt'))

    # Export ONNX
    if args.export:
        model.eval()
        onnx_path = os.path.join(args.checkpoint_dir, 'tiny_kws.onnx')
        model.export_onnx_slim(onnx_path)

    print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")
    print(f"Model saved to: {args.checkpoint_dir}")
    return best_acc


if __name__ == '__main__':
    args = parse_args()
    train(args)
