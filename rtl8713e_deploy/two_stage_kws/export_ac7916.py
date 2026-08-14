#!/usr/bin/env python3
"""
导出两阶段模型到 AC7916AB — 生成完整 C 代码包

用法:
    python export_ac7916.py \
        --stage1_ckpt checkpoints/stage1/best_model.pt \
        --stage2_ckpt checkpoints/stage2/best_model.pt \
        --grammar checkpoints/stage2/grammar.json \
        --output ./ac7916_firmware/

输出:
    ac7916_firmware/
    ├── stage1_model.h        # Stage1 INT8 权重 (CPU)
    ├── stage2_model.h        # Stage2 INT8 权重 (MVA)
    ├── grammar.h             # WFST 语法图 C 数组
    ├── mel_config.h          # Mel 滤波器组参数
    ├── kws_pipeline.h        # 两级流水线 C 代码
    └── flash_layout.txt      # Flash/PSRAM 布局建议
"""

import os, sys, argparse, json
import torch
import numpy as np

_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)

# 优先包内相对导入（打包 / 包导入环境），回退顶层导入（源码 CLI 环境）
try:
    from .stage1_wakeword import UltraTinyWakeWord
    from .stage2_command import CTCEncoder
    from .wfst_decoder import WFSTGrammarDecoder
except ImportError:
    from stage1_wakeword import UltraTinyWakeWord
    from stage2_command import CTCEncoder
    from wfst_decoder import WFSTGrammarDecoder


def export(args):
    os.makedirs(args.output, exist_ok=True)
    flash_map = {}

    # === Stage 1 ===
    ckpt1 = torch.load(args.stage1_ckpt, map_location='cpu')
    n_classes = ckpt1['num_classes']
    wake_words = ckpt1.get('wake_words', [])
    model1 = UltraTinyWakeWord(num_wake_words=n_classes, n_mels=40)
    model1.load_state_dict(ckpt1['model_state_dict'])
    model1.eval()
    _fold_batchnorms(model1)
    stage1_h = os.path.join(args.output, 'stage1_model.h')
    _write_c_weights(model1, stage1_h, 'STAGE1', 'CPU @ 320MHz')
    flash_map['stage1_weights'] = os.path.getsize(stage1_h)

    # === Stage 2 ===
    ckpt2 = torch.load(args.stage2_ckpt, map_location='cpu')
    token_info = ckpt2['tokenizer']
    n_tokens = len(token_info['c2i'])
    model2 = CTCEncoder(input_dim=40, num_tokens=n_tokens)
    model2.load_state_dict(ckpt2['model_state_dict'])
    model2.eval()
    _fold_batchnorms(model2)
    stage2_h = os.path.join(args.output, 'stage2_model.h')
    _write_c_weights(model2, stage2_h, 'STAGE2', 'MVA @ 360MHz')
    flash_map['stage2_weights'] = os.path.getsize(stage2_h)

    # === WFST Grammar ===
    grammar = WFSTGrammarDecoder.load(args.grammar)
    grammar_h = os.path.join(args.output, 'grammar.h')
    _write_grammar_c(grammar, token_info, grammar_h)
    flash_map['grammar'] = os.path.getsize(grammar_h)

    # === Mel Config ===
    mel_h = os.path.join(args.output, 'mel_config.h')
    _write_mel_config(mel_h)
    flash_map['mel_config'] = os.path.getsize(mel_h)

    # === Pipeline C code ===
    pipeline_h = os.path.join(args.output, 'kws_pipeline.h')
    _write_pipeline_c(pipeline_h, n_classes, n_tokens)
    flash_map['pipeline_code'] = os.path.getsize(pipeline_h)

    # === KWS Config (类别/token/标签) ===
    _write_kws_config(os.path.join(args.output, 'kws_config.h'),
                      n_classes, wake_words, n_tokens,
                      token_info.get('blank_id', 0), token_info.get('i2c', {}))

    # === 移植 Demo ===
    _copy_demo_files(args.output)

    # === Flash Layout ===
    _write_flash_layout(os.path.join(args.output, 'flash_layout.txt'), flash_map)

    # === Summary ===
    print("\n" + "=" * 60)
    print("AC7916AB Firmware Package Generated!")
    print("=" * 60)
    total_bytes = sum(flash_map.values())
    print(f"  Total Flash:  {total_bytes/1024:.0f} KB / 8 MB")
    print(f"  Total PSRAM:  {flash_map['stage2_weights']/1024 + flash_map['grammar']/1024:.0f} KB / 2 MB")
    print(f"\n  Files in {args.output}/:")
    for f in sorted(os.listdir(args.output)):
        size = os.path.getsize(os.path.join(args.output, f))
        print(f"    {f:<25s} {size:>8,d} bytes")
    print(f"\n  Copy all .h files to your AC7916 SDK project.")
    print(f"  Include kws_pipeline.h in your main.c.")
    print("=" * 60)


def _fold_batchnorms(model):
    """把 eval 模式下的 BatchNorm 融合进前面的 Conv（就地修改）。

    融合后 BN 变为恒等，导出时跳过即可，推理端无需再单独算 BN。
    """
    import torch
    import torch.nn as nn

    def _fold(conv, bn):
        w = conv.weight.data.clone()
        b = conv.bias.data.clone() if conv.bias is not None \
            else torch.zeros(conv.out_channels, device=w.device)
        std = torch.sqrt(bn.running_var + bn.eps)
        s = (bn.weight.data / std).to(w.dtype)
        if w.dim() == 4:
            w = w * s.view(-1, 1, 1, 1)
        elif w.dim() == 3:
            w = w * s.view(-1, 1, 1)
        else:
            w = w * s.view(-1, 1)
        b = (b - bn.running_mean) * s + bn.bias.data
        conv.weight.data = w
        if conv.bias is None:
            conv.bias = nn.Parameter(b)
        else:
            conv.bias.data = b
        # BN 变恒等
        bn.weight.data.fill_(1.0)
        bn.bias.data.fill_(0.0)
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)

    def _walk(container):
        children = list(container.children())
        i = 0
        while i < len(children):
            child = children[i]
            if isinstance(child, (nn.Conv1d, nn.Conv2d)) and i + 1 < len(children) \
                    and isinstance(children[i + 1], (nn.BatchNorm1d, nn.BatchNorm2d)):
                _fold(child, children[i + 1])
                i += 2
                continue
            _walk(child)
            i += 1

    _walk(model)
    return model


def _write_c_weights(model, path, prefix, target):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'// {prefix} Model Weights for {target}\n')
        f.write(f'// Auto-generated by export_ac7916.py\n\n')
        f.write(f'#ifndef {prefix}_MODEL_H\n#define {prefix}_MODEL_H\n')
        f.write('#include <stdint.h>\n\n')

        total_bytes = 0
        scale_map = {}
        for name, param in model.named_parameters():
            if 'bn' in name or 'batch_norm' in name:
                continue  # BN 已融合进 Conv
            data = param.data.cpu().numpy()
            cname = name.replace('.', '_')

            if 'bias' in name:
                # bias 用同层 weight 的 scale，量化为 int32 保精度
                wname = name.replace('.bias', '.weight')
                scale = scale_map.get(
                    wname, max(abs(data.min()), abs(data.max())) / 127.0 or 1e-8)
                q = np.round(data / scale * 65536).astype(np.int32)
                dtype = 'int32_t'
            else:
                scale = max(abs(data.min()), abs(data.max())) / 127.0 or 1e-8
                scale_map[name] = scale
                q = np.clip(np.round(data / scale), -128, 127).astype(np.int8)
                dtype = 'int8_t'

            total_bytes += q.nbytes
            f.write(f'// {name}  shape={list(data.shape)}  scale={scale:.6f}\n')
            flat = q.flatten()
            f.write(f'static const {dtype} {cname}[{len(flat)}] = {{\n')
            for i in range(0, len(flat), 16):
                f.write('    ' + ','.join(str(int(v)) for v in flat[i:i+16]) + ',\n')
            f.write('};\n')
            f.write(f'static const float {cname}_scale = {scale:.6f}f;\n\n')

        f.write(f'// Total: {total_bytes} bytes ({total_bytes/1024:.1f} KB)\n')
        f.write(f'#endif\n')
    print(f"[Export] {path} ({total_bytes/1024:.1f} KB)")


def _write_grammar_c(grammar, token_info, path):
    with open(path, 'w') as f:
        f.write('// WFST Grammar for AC7916 CPU Decoder\n')
        f.write('#ifndef GRAMMAR_H\n#define GRAMMAR_H\n')
        f.write('#include <stdint.h>\n\n')

        # token mapping
        c2i = token_info['c2i']
        i2c = token_info['i2c']
        f.write(f'#define GRAMMAR_NUM_TOKENS {len(c2i)}\n')
        f.write(f'#define GRAMMAR_NUM_STATES {len(grammar.grammar)}\n\n')

        # Token to char mapping
        f.write('static const char grammar_tokens[GRAMMAR_NUM_TOKENS][4] = {\n')
        for i in range(len(i2c)):
            ch = i2c.get(str(i), '?')
            f.write(f'  "{ch}",\n')
        f.write('};\n\n')

        # Grammar transitions (sparse adjacency)
        f.write('// Grammar transitions: prev_token -> [next_tokens]\n')
        f.write('// Format: {prev_token, num_next, next_token_list}\n')
        trans_list = []
        for k, v in grammar.grammar.items():
            trans_list.append({'prev': int(k), 'next': list(v)})
        trans_list.sort(key=lambda x: x['prev'])

        f.write(f'#define GRAMMAR_NUM_TRANS {len(trans_list)}\n')
        all_next = []
        offsets = []
        for t in trans_list:
            offsets.append(len(all_next))
            all_next.extend(t['next'])
        offsets.append(len(all_next))

        f.write('static const uint16_t grammar_next_tokens[] = {')
        f.write(', '.join(str(n) for n in all_next))
        f.write('};\n')

        f.write('static const uint16_t grammar_offsets[] = {')
        f.write(', '.join(str(o) for o in offsets))
        f.write('};\n')

        f.write('static const uint16_t grammar_prevs[] = {')
        f.write(', '.join(str(t['prev']) for t in trans_list))
        f.write('};\n')

        # Terminal states
        f.write('static const uint16_t grammar_terminals[] = {')
        f.write(', '.join(str(t) for t in sorted(grammar.terminals)))
        f.write('};\n')
        f.write(f'#define GRAMMAR_NUM_TERMINALS {len(grammar.terminals)}\n')

        f.write('\n#endif\n')
    print(f"[Export] {path}")


def _write_mel_config(path):
    with open(path, 'w') as f:
        f.write('// Mel Feature Extractor Config for AC7916\n')
        f.write('#ifndef MEL_CONFIG_H\n#define MEL_CONFIG_H\n\n')
        f.write('#define MEL_SAMPLE_RATE  16000\n')
        f.write('#define MEL_N_MELS       40\n')
        f.write('#define MEL_N_FFT        512\n')
        f.write('#define MEL_WIN_LENGTH   400\n')
        f.write('#define MEL_HOP_LENGTH   160\n')
        f.write('#define MEL_F_MIN        80.0f\n')
        f.write('#define MEL_F_MAX        7600.0f\n')
        f.write('#define MEL_N_FRAMES     98\n')
        f.write('#define MEL_PREEMPHASIS   0.97f\n')
        # Mel filterbank (40x257)
        mel = _gen_mel_fb(40, 257, 16000)
        f.write(f'\n// Mel filterbank [{mel.shape[0]}x{mel.shape[1]}]\n')
        f.write(f'static const float mel_filterbank[{mel.shape[0]}][{mel.shape[1]}] = {{\n')
        for row in mel:
            f.write('  {' + ', '.join(f'{v:.8f}f' for v in row[:8]) + ', ...},\n')
        f.write('};\n')
        f.write('\n#endif\n')


def _gen_mel_fb(n_mels, n_freq, sr):
    mel = np.linspace(2595*np.log10(1+80/700), 2595*np.log10(1+7600/700), n_mels+2)
    hz = 700*(10**(mel/2595)-1)
    bins = np.floor(n_freq * hz / (sr/2)).astype(int)
    fb = np.zeros((n_mels, n_freq))
    for m in range(n_mels):
        for k in range(bins[m], bins[m+1]):
            fb[m, k] = (k - bins[m]) / max(bins[m+1] - bins[m], 1)
        for k in range(bins[m+1], min(bins[m+2], n_freq)):
            fb[m, k] = (bins[m+2] - k) / max(bins[m+2] - bins[m+1], 1)
    return fb


def _write_pipeline_c(path, n_wake_classes, n_tokens):
    with open(path, 'w') as f:
        f.write('// Two-Stage KWS Pipeline for AC7916AB\n')
        f.write('// Stage1: CPU @ 320MHz, always-on\n')
        f.write('// Stage2: MVA @ 360MHz, on-demand\n\n')
        f.write('#ifndef KWS_PIPELINE_H\n#define KWS_PIPELINE_H\n\n')
        f.write('#include "stage1_model.h"\n')
        f.write('#include "stage2_model.h"\n')
        f.write('#include "grammar.h"\n')
        f.write('#include "mel_config.h"\n')
        f.write('#include "kws_config.h"\n\n')

        f.write('typedef enum { KWS_IDLE, KWS_LISTENING, KWS_WOKE, KWS_COMMAND } kws_state_t;\n\n')
        f.write(f'#define KWS_N_WAKE_CLASSES {n_wake_classes}\n')
        f.write(f'#define KWS_N_TOKENS {n_tokens}\n\n')

        f.write('// Forward declarations\n')
        f.write('void kws_pipeline_init(void);\n')
        f.write('int  kws_pipeline_feed(const int16_t *pcm_10ms);  // returns cmd_id or -1\n')
        f.write('const char* kws_get_command_text(int cmd_id);\n')
        f.write('kws_state_t kws_get_state(void);\n\n')
        f.write('#endif\n')
    print(f"[Export] {path}")


def _write_kws_config(path, n_classes, wake_words, n_tokens, blank_id, i2c):
    """生成 kws_config.h：类别数 / token 数 / blank id / 标签映射。"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('// KWS model configuration (auto-generated)\n')
        f.write('#ifndef KWS_CONFIG_H\n#define KWS_CONFIG_H\n\n')
        f.write(f'#define STAGE1_NUM_CLASSES {n_classes}\n')
        f.write(f'#define STAGE2_NUM_TOKENS  {n_tokens}\n')
        f.write(f'#define STAGE2_BLANK_ID    {blank_id}\n\n')

        f.write('// Stage1 唤醒词标签\n')
        f.write('static const char* stage1_labels[STAGE1_NUM_CLASSES] = {\n')
        for i in range(n_classes):
            label = wake_words[i] if i < len(wake_words) else f'wake_{i}'
            label = label.replace('\\', '\\\\').replace('"', '\\"')
            f.write(f'    "{label}",\n')
        f.write('};\n\n')

        f.write('// Stage2 token -> 字符映射\n')
        f.write('static const char* stage2_tokens[STAGE2_NUM_TOKENS] = {\n')
        for i in range(n_tokens):
            ch = i2c.get(str(i), i2c.get(i, '?'))
            ch = ch.replace('\\', '\\\\').replace('"', '\\"')
            f.write(f'    "{ch}",\n')
        f.write('};\n\n')
        f.write('#endif\n')
    print(f"[Export] {path}")


def _copy_demo_files(output_dir):
    """复制移植 demo 模板到输出目录 demo/ 子目录。"""
    import shutil
    if getattr(sys, 'frozen', False):
        base = os.path.join(getattr(sys, '_MEIPASS', ''), 'rtl8713e_deploy', 'two_stage_kws')
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    demo_src = os.path.join(base, 'demo')
    demo_dst = os.path.join(output_dir, 'demo')
    os.makedirs(demo_dst, exist_ok=True)
    copied = 0
    if os.path.isdir(demo_src):
        for fn in os.listdir(demo_src):
            src = os.path.join(demo_src, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(demo_dst, fn))
                copied += 1
    print(f"[Export] demo files -> {demo_dst} ({copied} files)")


def _write_flash_layout(path, flash_map):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('AC7916AB Flash / PSRAM Memory Layout\n')
        f.write('=' * 50 + '\n\n')
        f.write('注：以下 Stage/Config 数值为导出的源文件大小（.h 文本文件），\n')
        f.write('    芯片 Flash 实际二进制占用（int8 量化后）约为其 1/3~1/4。\n\n')
        f.write(f'Flash (8 MB total):\n')
        offset = 0
        items = [
            ('Bootloader', 65536),
            ('Firmware', 262144),
            ('Stage1 Weights', flash_map['stage1_weights']),
            ('Stage2 Weights', flash_map['stage2_weights']),
            ('WFST Grammar', flash_map['grammar']),
            ('Mel Config', flash_map['mel_config']),
            ('Pipeline Code', flash_map['pipeline_code']),
            ('Reserved (OTA)', 8*1024*1024 - sum(flash_map.values()) - 65536 - 262144),
        ]
        for name, size in items:
            f.write(f'  0x{offset:06X}  {name:<20s} {size:>10,d} B\n')
            offset += size
        f.write(f'  {"─"*50}\n')
        f.write(f'  Total: {offset/1024:.0f} KB / 8192 KB\n\n')

        f.write(f'PSRAM (2 MB, runtime):\n')
        f.write(f'  Stage2 Weights:  {flash_map["stage2_weights"]/1024:.0f} KB\n')
        f.write(f'  WFST Grammar:    {flash_map["grammar"]/1024:.0f} KB\n')
        f.write(f'  Mel Window:      16 KB\n')
        f.write(f'  PCM Buffer:       8 KB\n')
        f.write(f'  Scratch:          64 KB\n')
        used = flash_map['stage2_weights'] + flash_map['grammar'] + 88*1024
        f.write(f'  {"─"*50}\n')
        f.write(f'  Total: {used/1024:.0f} KB / 2048 KB\n')


def parse_args():
    p = argparse.ArgumentParser(description='Export 2-Stage KWS to AC7916AB')
    p.add_argument('--stage1_ckpt', required=True)
    p.add_argument('--stage2_ckpt', required=True)
    p.add_argument('--grammar', required=True)
    p.add_argument('--output', default='./ac7916_firmware')
    return p.parse_args()


if __name__ == '__main__':
    export(parse_args())
