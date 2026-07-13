# RTL8713E HiFi 5 DSP 语音指令识别 — 部署与集成指南

## 目录
1. [架构总览](#1-架构总览)
2. [链路流程](#2-完整链路流程)
3. [训练模型](#4-训练自己的模型)
4. [导出部署代码](#5-导出部署代码)
5. [RTL8713E SDK 集成](#6-rtl8713e-sdk-集成)
6. [性能与内存验证](#7-性能与内存验证)
7. [调试与排查](#8-调试与排查)

---

## 1. 架构总览

```
┌────────────────────────────────────────────────────────┐
│                  RTL8713E 芯片内部                       │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ HiFi 5 DSP @ 500MHz  (256KB DTCM, 512KB SRAM)   │   │
│  │                                                  │   │
│  │  ① Audio Frontend       ② KWS Engine            │   │
│  │  ┌──────────────┐      ┌──────────────────┐     │   │
│  │  │ PDM/I²S → PCM│      │ Mel Feature Extr │     │   │
│  │  │ ↓            │      │ ↓                │     │   │
│  │  │ Pre-emphasis │      │ TinyKWS CNN      │     │   │
│  │  │ ↓            │      │ ↓                │     │   │
│  │  │ Hann + FFT   │      │ Softmax + Top-1  │     │   │
│  │  │ ↓            │      │ ↓                │     │   │
│  │  │ Mel Filter   │      │ Post-smooth      │     │   │
│  │  └──────────────┘      └────────┬─────────┘     │   │
│  │                                 │               │   │
│  └─────────────────────────────────┼───────────────┘   │
│                                    │                    │
│  ┌─────────────────────────────────▼────────────────┐  │
│  │ KM4 (Armv8.1-M @ 400MHz)                          │  │
│  │  - 接收 DSP 的指令结果                             │  │
│  │  - 业务逻辑 / 网络通信 / 状态管理                  │  │
│  └──────────────────────────────────────────────────┘  │
│                                    │                    │
│  ┌─────────────────────────────────▼────────────────┐  │
│  │ KR4 (RISC-V @ 400MHz)                            │  │
│  │  - Wi-Fi 6 / BT 5.2 无线通信                     │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 原因 |
|------|------|
| **Mel 特征而非 learnable frontend** | HiFi 5 的 FFT 有硬件加速, 比 Conv frontend 快 5-10x |
| **INT8 对称量化 (zp=0)** | Nature DSP 对 zp=0 有特殊优化, 省去 subtraction 操作 |
| **每 10 帧推理一次** (100ms) | 实时率 ~30x, 留足余量给其他 DSP 任务 (降噪/AEC) |
| **滑动窗口 98 帧** (~1s) | 覆盖中文字词完整发音, 既不过长也不损失响应速度 |

---

## 2. 完整链路流程

```
时间轴 (ms):  0    10   20   30    ...   980  990  1000
              │    │    │    │          │    │    │
PCM 输入:   [160][160][160][160]......[160][160][160]
              │    │    │    │          │    │    │
Mel 提取:     M0   M1   M2   M3  ..... M97  M98  M99
              │         │              │         │
滑动窗口:  [M0..M97] [M1..M98] [M2..M99] (每帧更新)
              │         │              │
CNN 推理:     ───跳过───  ───推理────  ───跳过────
 (每100ms)                    │
                          logits
                             │
                          softmax
                             │
                      top-1: "打开灯" (prob=0.89)
                             │
                      连续5帧一致? → YES → 触发指令
                             │
                      发送消息给 KM4 主控
```

---

## 3. Python 训练侧

### 3.1 准备训练数据

```
kws_data/
├── train.jsonl        # 训练标注
├── val.jsonl          # 验证标注
├── noise/             # 背景噪声 (可选)
└── wav/
    ├── cmd_001.wav
    ├── cmd_002.wav
    └── ...
```

`train.jsonl` 格式 (每行一个 JSON):
```json
{"audio_path": "wav/cmd_001.wav", "label": "打开灯", "label_id": 0}
{"audio_path": "wav/cmd_002.wav", "label": "关闭空调", "label_id": 1}
{"audio_path": "wav/cmd_003.wav", "label": "温度调高", "label_id": 2}
{"audio_path": "wav/bg_noise_01.wav", "label": "<unknown>", "label_id": -1}
```

### 3.2 训练

```bash
cd rtl8713e_deploy/training

python train_kws.py \
    --data_root ../../kws_data \
    --num_classes 50 \
    --preset standard \
    --batch_size 128 \
    --epochs 80 \
    --lr 1e-3 \
    --export
```

### 3.3 导出 DSP C 代码

```bash
cd rtl8713e_deploy/deployment

python convert_model_to_c.py \
    --checkpoint ../training/kws_checkpoints/best_model.pt \
    --output_dir ./dsp_codegen
```

输出:
```
dsp_codegen/
├── model_weights.h    # INT8 权重 C 数组
├── mel_config.h       # Mel 滤波器组
├── kws_config.h       # 类别标签
└── tiny_kws.onnx      # (可选) 用于 Cadence XNNC
```

---

## 4. RTL8713E SDK 集成

### 4.1 SDK 文件结构

将生成的文件放入 Realtek Ameba SDK 项目:

```
ameba_pro2_sdk/
└── project/realtek_amebaPro2_va0_example/
    └── src/
        ├── kws/
        │   ├── hifi5_inference.c      ← 我们的推理代码
        │   ├── model_weights.h        ← 生成的权重
        │   ├── mel_config.h           ← 生成的 Mel 配置
        │   ├── kws_config.h           ← 生成的类别标签
        │   └── kws_inference.h        ← 对外接口
        ├── main.c                     ← KM4 主控代码
        └── ...
```

### 4.2 HiFi 5 DSP 侧 (kws_inference.h)

```c
// kws_inference.h — KM4 调用 DSP 的接口
#ifndef KWS_INFERENCE_H
#define KWS_INFERENCE_H

#include <stdint.h>

// 初始化 KWS 引擎 (启动时调用一次)
void kws_init(void);

// 处理一帧 PCM 数据 (每 10ms 调用)
// 返回: >=0 检测到的指令ID, -1 无指令, -2 处理中
int kws_process_frame(const int16_t *pcm_chunk);

// 获取上次检测的指令文本 (KM4 侧调用)
const char* kws_get_label(int class_id);

// 获取所有指令列表
int kws_get_num_classes(void);

#endif
```

### 4.3 KM4 主控侧代码 (main.c)

```c
// main.c — RTL8713E KM4 核心

#include "kws/kws_inference.h"
#include "ameba_audio.h"

// DMA 音频回调: 每 10ms 触发一次
static void audio_dma_callback(const int16_t *pcm_data, uint32_t len) {
    int result = kws_process_frame(pcm_data);

    if (result >= 0) {
        // 检测到指令！
        const char *command = kws_get_label(result);
        printf("[KWS] Detected: %s (class=%d)\n", command, result);

        // 执行业务逻辑
        execute_command(result);

        // 可选: 通过 Wi-Fi / BLE 上报
        // send_command_to_cloud(result);

        // 可选: 播放确认音
        // audio_play_ack();
    }
}

void app_main(void) {
    // 1. 初始化 KWS (DSP 侧)
    kws_init();
    printf("[KWS] Initialized on HiFi 5 DSP\n");

    // 2. 配置音频输入
    audio_config_t cfg = {
        .sample_rate = 16000,
        .channels = 1,
        .format = AUDIO_FORMAT_PCM_16BIT,
        .chunk_size = 160,  // 10ms @ 16kHz
        .callback = audio_dma_callback,
    };
    audio_init(&cfg);

    // 3. 启动音频流
    audio_start();

    printf("[App] Listening for commands...\n");

    // 4. 主循环 (处理 Wi-Fi / BT / 其他业务)
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
```

### 4.4 Cadence XCC 编译配置

```makefile
# Makefile 片段 — HiFi 5 DSP 编译

HIFI5_CC = xt-xcc
HIFI5_CFLAGS = -mhifi5 -O3 -DUSE_NATURE_DSP
HIFI5_LDFLAGS = -lnature -lm

# 将数据放在 DTCM
HIFI5_CFLAGS += -DWEIGHTS_IN_DTCM

KWS_SRCS = kws/hifi5_inference.c
KWS_OBJS = $(KWS_SRCS:.c=.o)

kws_dsp.bin: $(KWS_OBJS)
    $(HIFI5_CC) $(HIFI5_CFLAGS) -o $@ $^ $(HIFI5_LDFLAGS)

# 检查 DTCM 使用
size_check: kws_dsp.bin
    @echo "DTCM usage:"
    @xt-size kws_dsp.bin | grep -E "dtcm|\.rodata|\.bss"
```

### 4.5 Cadence XNNC 编译器 (高级方案)

如果不想手写 C 推理代码, 可以用 Cadence XNNC 编译器直接从 ONNX 生成 HiFi 5 优化代码:

```bash
# Cadence XNNC (Xtensa Neural Network Compiler)
# 将 ONNX 编译为 HiFi 5 优化 C 代码

xnnc \
    --model tiny_kws.onnx \
    --input-shape "mel_features:1,1,40,98" \
    --target hifi5 \
    --quantization int8 \
    --output-dir ./hifi5_generated \
    --optimization-level 3 \
    --use-nature-dsp

# 输出:
# ./hifi5_generated/
#   ├── model.c          # 优化的推理代码 (Nature DSP API)
#   ├── model.h
#   ├── weights.bin      # INT8 权重
#   └── model.params
```

XNNC 生成的代码性能:
- 自动融合 Conv+BN+ReLU
- 自动分配 DTCM 内存
- 自动生成量化/反量化节点
- 比手写 C 快 2-3x

---

## 5. 内存预算验证

### DTCM 256KB 分配明细

| 区域 | 大小 | 说明 |
|------|------|------|
| 模型权重 (.rodata) | ~100 KB | INT8 权重, 在 DTCM (通过 section attribute) |
| 激活缓冲 (复用) | ~80 KB | 32×40×98 最大张量, 各层复用 |
| Mel 滑动窗口 | ~16 KB | 40×98×4 bytes (float32) |
| PCM DMA 双缓冲 | ~8 KB | 2×4096×2 bytes (int16) |
| FFT 工作区 | ~4 KB | 512 float + 257 complex |
| 预加重/窗历史 | ~1 KB | prev_audio + prev_sample |
| 后处理状态 | ~1 KB | prev_predictions 等 |
| DSP 栈 | ~16 KB | 函数调用栈 |
| **总计** | **~226 KB** | ✅ 在 256KB 以内, 余量 30KB |

### 推理延迟

| 阶段 | 延迟 | 说明 |
|------|------|------|
| Mel 特征提取 | ~0.3 ms | FFT 硬件加速 |
| TinyKWS CNN (INT8) | ~2.5 ms | Nature DSP 优化卷积 |
| Softmax + 后处理 | ~0.1 ms | |
| **每帧总计** | **~2.9 ms** | 帧间隔 10ms → 实时率 ~3.4x ✅ |

---

## 6. 高级优化

### 6.1 BN 融合

训练完成后, 将 BatchNorm 参数融合到 Conv 层:

```python
# bn_fusion.py
def fuse_conv_bn(conv_weight, conv_bias, bn_weight, bn_bias,
                  bn_mean, bn_var, bn_eps):
    """Conv + BN → Conv (合并为一个 Conv 层)"""
    scale = bn_weight / np.sqrt(bn_var + bn_eps)
    fused_weight = conv_weight * scale.reshape(-1, 1, 1, 1)
    fused_bias = (conv_bias - bn_mean) * scale + bn_bias if conv_bias is not None \
                 else -bn_mean * scale + bn_bias
    return fused_weight, fused_bias
```

### 6.2 激活缓存复用

不同层之间激活缓冲复用同一块内存:

```c
// 规划最大激活大小, 所有层共用
#define MAX_ACT_SIZE  (32 * 40 * 98)  // Stem 输出最大
int8_t act_buf[MAX_ACT_SIZE] __attribute__((section(".dtcm.bss")));
```

### 6.3 稀疏化 (可选)

训练后对权重做稀疏化 (50% 稀疏度):
- 移除绝对值 < threshold 的权重
- 模型大小再减 40-50%
- HiFi 5 Nature DSP 支持稀疏 Conv

---

## 7. 调试与排查

### 精度下降 (> 5%)

```
原因: 量化精度损失
排查:
1. 对比 Python FP32 vs INT8 输出差异
2. 增大校准数据集 (至少 100 条)
3. 对敏感层 (第一层 Conv) 使用 per-channel 量化
```

### DTCM 溢出

```
原因: 权重 + 激活超过 256KB
排查:
1. 减小模型 (用 micro preset)
2. 部分权重放到 SRAM (512KB)
3. 激活缓冲减小 (减少 n_frames)
```

### 实时性不足

```
原因: 推理时间超过 10ms 帧间隔
排查:
1. 检查是否每帧都在推理 → 改为每 N 帧推理
2. 确认 Nature DSP 优化已开启
3. 减小模型尺寸
```

### Cadence XCC 编译错误

```
常见问题:
- "undefined reference to nature_*" → 链接 libnature.a
- "DTCM overflow" → 用 __attribute__((section(".sram"))) 移走大数组
- "misaligned access" → 检查数据对齐 (64-byte for HiFi 5 DMA)
```
