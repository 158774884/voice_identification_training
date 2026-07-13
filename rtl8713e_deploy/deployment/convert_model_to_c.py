#!/usr/bin/env python3
"""
PyTorch 模型 → C 数组导出 (用于 HiFi 5 DSP)

将训练好的 TinyKWS 模型导出为:
1. model_weights.h  — INT8 权重的 C 数组
2. mel_config.h     — Mel 滤波器组参数
3. kws_config.h     — 类别标签和推理配置

用法:
    python convert_model_to_c.py \
        --checkpoint kws_checkpoints/best_model.pt \
        --output_dir ./dsp_codegen
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rtl8713e_deploy.model.tiny_kws import TinyKWS, UltraTinyKWS
from rtl8713e_deploy.model.feature_extractor import MelFeatureExtractor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained model checkpoint (.pt)')
    p.add_argument('--output_dir', type=str, default='./dsp_codegen',
                   help='Output directory for C source files')
    p.add_argument('--quantize', action='store_true', default=True,
                   help='Quantize weights to INT8')
    p.add_argument('--input_scale', type=float, default=0.0625,
                   help='Input quantization scale (1/16 default)')
    return p.parse_args()


def quantize_tensor_int8(tensor: np.ndarray) -> tuple:
    """
    将 FP32 tensor 量化为 INT8

    Returns:
        (int8_weights, scale, zero_point)
    """
    w_min = tensor.min()
    w_max = tensor.max()

    # 对称量化 (zero_point=0, 适合 DSP)
    scale = max(abs(w_min), abs(w_max)) / 127.0
    if scale < 1e-8:
        scale = 1e-8

    q = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
    return q, scale


def write_c_array(f, name, data, dtype='int8_t', per_line=12):
    """将 numpy 数组写入为 C 数组"""
    if dtype == 'int8_t':
        vals = data.astype(np.int8).flatten()
    elif dtype == 'int32_t':
        vals = data.astype(np.int32).flatten()
    elif dtype == 'float':
        vals = data.astype(np.float32).flatten()
    else:
        vals = np.array(data).flatten()

    f.write(f'static const {dtype} {name}[{len(vals)}] = {{\n')

    for i in range(0, len(vals), per_line):
        chunk = vals[i:i + per_line]
        if dtype == 'int8_t':
            f.write('    ' + ', '.join(f'{v:4d}' for v in chunk) + ',\n')
        elif dtype == 'int32_t':
            f.write('    ' + ', '.join(f'{v:8d}' for v in chunk) + ',\n')
        else:
            f.write('    ' + ', '.join(f'{v:.8f}f' for v in chunk) + ',\n')

    f.write('};\n\n')


def convert(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    # 重建模型
    num_classes = ckpt.get('num_classes', 50)
    n_mels = ckpt.get('n_mels', 40)
    model = TinyKWS(num_classes=num_classes, n_mels=n_mels)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    label2id = ckpt.get('label2id', {})
    id2label = ckpt.get('id2label', {})

    print(f"Model: {num_classes} classes, {n_mels} mel bins")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ===== 导出 model_weights.h =====
    weights_path = os.path.join(args.output_dir, 'model_weights.h')
    with open(weights_path, 'w') as f:
        f.write('// Auto-generated TinyKWS model weights for HiFi 5 DSP\n')
        f.write('// DO NOT EDIT MANUALLY\n')
        f.write(f'// Model: {sum(p.numel() for p in model.parameters()):,} params\n')
        f.write(f'// Classes: {num_classes}\n')
        f.write(f'// Quantization: INT8 symmetric\n\n')
        f.write('#ifndef MODEL_WEIGHTS_H\n')
        f.write('#define MODEL_WEIGHTS_H\n\n')
        f.write('#include <stdint.h>\n\n')

        total_bytes = 0
        layer_idx = 0

        for name, param in model.named_parameters():
            if 'bn' in name or 'batch_norm' in name:
                continue  # BN 可融合到 Conv, 不单独导出

            data = param.data.cpu().numpy()
            shape_str = 'x'.join(str(s) for s in data.shape)

            if args.quantize:
                q_data, scale = quantize_tensor_int8(data)
                total_bytes += q_data.nbytes

                c_name = name.replace('.', '_').replace('-', '_')
                f.write(f'// {name}  shape=[{shape_str}]  '
                        f'scale={scale:.6f}  zero_point=0\n')

                if 'bias' in name:
                    # Bias 用 INT32 (量化后的)
                    bias_int32 = (data / scale * 65536).astype(np.int32)
                    write_c_array(f, c_name, bias_int32, 'int32_t')
                else:
                    write_c_array(f, c_name, q_data, 'int8_t')

                # 保存 scale 信息
                scale_name = c_name + '_scale'
                f.write(f'static const float {scale_name} = {scale:.8f}f;\n\n')
            else:
                # FP32 导出 (调试用)
                write_c_array(f, name.replace('.', '_'), data, 'float')

            layer_idx += 1

        f.write(f'// Total weight bytes (INT8): {total_bytes} ({total_bytes/1024:.1f} KB)\n')
        f.write(f'// DTCM budget: 256 KB\n')
        f.write(f'// DTCM usage:  {total_bytes} bytes (weights)\n')
        f.write(f'//              + ~80 KB (activations, reusable)\n')
        f.write(f'//              + ~32 KB (audio buffers)\n')
        f.write(f'//              = ~{total_bytes/1024 + 80 + 32:.0f} KB total\n')
        if total_bytes + 80*1024 + 32*1024 < 256*1024:
            f.write('// ✅ FITS within 256KB DTCM\n')
        else:
            f.write('// ⚠ WARNING: May not fit in DTCM!\n')

        f.write('\n#endif // MODEL_WEIGHTS_H\n')

    print(f"  → {weights_path} ({total_bytes/1024:.1f} KB INT8 weights)")

    # ===== 导出 Mel 滤波器组配置 =====
    fe = MelFeatureExtractor(n_mels=n_mels)
    mel_config = fe.generate_dsp_c_header()
    mel_config_path = os.path.join(args.output_dir, 'mel_config.h')
    with open(mel_config_path, 'w') as f:
        f.write(mel_config)
    print(f"  → {mel_config_path}")

    # ===== 导出 kws_config.h (类别标签) =====
    config_path = os.path.join(args.output_dir, 'kws_config.h')
    with open(config_path, 'w') as f:
        f.write('// Auto-generated KWS labels\n')
        f.write('#ifndef KWS_CONFIG_H\n')
        f.write('#define KWS_CONFIG_H\n\n')
        f.write(f'#define KWS_NUM_CLASSES  {num_classes}\n')
        f.write(f'#define KWS_N_MELS       {n_mels}\n')
        f.write(f'#define KWS_N_FRAMES     98\n')
        f.write(f'#define KWS_CONFIDENCE_THRESHOLD  0.6f\n')
        f.write(f'#define KWS_SMOOTH_WINDOW         5\n\n')
        f.write('// Class labels\n')
        f.write('static const char* kws_labels[KWS_NUM_CLASSES] = {\n')
        for i in range(num_classes):
            label = id2label.get(i, f'class_{i}')
            # Escape special chars
            label = label.replace('\\', '\\\\').replace('"', '\\"')
            f.write(f'    "{label}",\n')
        f.write('};\n\n')
        f.write('#endif // KWS_CONFIG_H\n')
    print(f"  → {config_path}")

    # ===== 导出 ONNX (可选, 用于 Cadence XNNC 编译器) =====
    onnx_path = os.path.join(args.output_dir, 'tiny_kws.onnx')
    try:
        model.export_onnx_slim(onnx_path)
    except Exception as e:
        print(f"  ⚠ ONNX export failed: {e}")

    print("\nDone! Generated files:")
    for f in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, f)
        size = os.path.getsize(fpath)
        print(f"  {f:30s} {size:>8,d} bytes")

    print(f"\nCopy these files to your RTL8713E project's include/ directory.")
    print(f"Then compile with Cadence XCC and link against Nature DSP library.")


if __name__ == '__main__':
    args = parse_args()
    convert(args)
