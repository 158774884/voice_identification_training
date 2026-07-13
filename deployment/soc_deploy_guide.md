# SOC 嵌入式芯片部署完整指南

## 目录
1. [部署总览](#1-部署总览)
2. [模型导出量化](#2-模型导出与量化)
3. [芯片特定适配](#3-芯片特定适配)
4. [推理运行时](#4-推理运行时)
5. [性能优化](#5-性能优化)
6. [常见问题](#6-常见问题)

---

## 1. 部署总览

### 部署流水线

```
PyTorch 训练模型 (FP32, ~4.5M params)
        │
        ▼
   ONNX 导出 (opset 14)
        │
        ├──→ ONNX Runtime INT8 量化 ──→ CPU 推理 (x86/ARM)
        │
        ├──→ RKNN Toolkit2 ──→ Rockchip NPU 推理 (RK3588)
        │
        ├──→ Amlogic NPU SDK ──→ Amlogic NPU 推理 (A311D)
        │
        └──→ 海思 NNIE Mapper ──→ 海思 NNIE 推理 (Hi3559A)
```

### 目标芯片规格对比

| 芯片 | NPU 算力 | 支持量化 | 模型限制 | 典型功耗 |
|------|---------|---------|---------|---------|
| RK3588 | 3 TOPS | INT8/INT16/FP16 | < 10MB | 3-5W |
| A311D | 5 TOPS | INT8/INT16 | < 20MB | 3-5W |
| Hi3559A | 4 TOPS | INT8 | < 8MB | 2-4W |
| RV1126 | 1 TOPS | INT8 | < 4MB | 1-2W |

---

## 2. 模型导出与量化

### Step 1: 训练验证

```python
# 训练完成后，确认模型精度达标
from model.multi_task_model import create_model

model = create_model()
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 验证 CER < 15%, Dialect Acc > 80%, EER < 5%
```

### Step 2: 导出 ONNX

```python
from deployment.export_onnx import export_to_onnx

# 导出完整模型或多分支模型
paths = export_to_onnx(
    model,
    output_dir='./onnx_models',
    export_mode='all',          # full/asr/dialect/speaker
    opset_version=14,           # 支持 GRU
    dynamic_audio_len=True,     # 动态音频长度
)

# 导出结果:
# - voice_model_full.onnx       (~18MB FP32)
# - voice_model_asr.onnx        (~15MB FP32)
# - voice_model_dialect.onnx    (~12MB FP32)
# - voice_model_speaker.onnx    (~13MB FP32)
```

### Step 3: INT8 量化

```python
from deployment.quantization import ModelQuantizer

quantizer = ModelQuantizer(model)

# 3a. ONNX Runtime INT8 量化 (推荐)
quantizer.onnx_quantize_int8(
    'onnx_models/voice_model_full.onnx',
    calibration_data=cal_samples,  # 需要 100-500 条校准音频
    output_path='onnx_models/voice_model_int8.onnx',
    quant_format='QOperator',  # 或 'QDQ'
)

# 3b. 对含 GRU 的模型, 推荐动态量化
quantizer.onnx_dynamic_quantize(
    'onnx_models/voice_model_full.onnx',
    'onnx_models/voice_model_dynamic_int8.onnx',
)

# 量化后模型体积: ~18MB (FP32) → ~4.5MB (INT8)
```

### Step 4: 量化精度验证

```python
# 对比 FP32 vs INT8 输出差异
import onnxruntime as ort
import numpy as np

fp32_session = ort.InferenceSession('voice_model_full.onnx')
int8_session = ort.InferenceSession('voice_model_int8.onnx')

# 随机测试 100 条
for i in range(100):
    audio = test_audios[i]  # [1, 1, T] numpy
    lengths = np.array([audio.shape[-1]], dtype=np.int64)

    fp32_out = fp32_session.run(None, {'audio': audio, 'audio_lengths': lengths})
    int8_out = int8_session.run(None, {'audio': audio, 'audio_lengths': lengths})

    # ASR log_probs 差异 < 0.5% 则可接受
    max_diff = np.max(np.abs(fp32_out[0] - int8_out[0]))
    assert max_diff < 0.5, f"Quantization error too large: {max_diff}"
```

---

## 3. 芯片特定适配

### 3.1 Rockchip RK3588 (推荐)

```bash
# 安装 rknn-toolkit2
# 参考: https://github.com/rockchip-linux/rknn-toolkit2

# 转换为 RKNN 格式
python -c "
from rknn.api import RKNN

rknn = RKNN()

# 配置
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[1, 1, 1]],
    target_platform='rk3588',
    quantized_dtype='asymmetric_quantized-8',  # INT8
)

# 加载 ONNX
ret = rknn.load_onnx(model='voice_model_full.onnx')
print(f'Load ONNX: {ret}')

# 构建 RKNN
ret = rknn.build(
    do_quantization=True,
    dataset='./calibration_dataset.txt',  # 校准图片路径列表
)
print(f'Build RKNN: {ret}')

# 导出
ret = rknn.export_rknn('voice_model.rknn')
print(f'Export: {ret}')

# 精度分析 (可选)
ret = rknn.accuracy_analysis(inputs=['./test_data/'])
print(f'Accuracy: {ret}')

rknn.release()
"
```

### 3.2 Amlogic A311D

```bash
# 使用 Amlogic NPU SDK
# 参考: https://gitlab.com/amlogic/aml_npu_sdk

# 转换流程:
# ONNX → AML IR → Quantize → Compile → .nb 文件

python3 convert.py \
    --model voice_model_full.onnx \
    --output voice_model.nb \
    --target A311D \
    --quantization INT8 \
    --calibration_data ./cal_data/
```

### 3.3 海思 Hi3559A (需算子替换)

⚠️ Hi3559A NNIE 不支持 GRU 算子, 需要将 GRU 层替换为 Conv1D 等效层

```python
# 部署前替换方案:
# 原始: GRU(256, 256) → Conv1d(256, 256, kernel=3, dilation=N)
# 通过增大 dilation 弥补 GRU 的长期依赖能力

from deployment.utils import replace_gru_with_conv_for_nnie

model_nnie = replace_gru_with_conv_for_nnie(model)
# GRU → Dilated Conv1d with larger kernel
```

---

## 4. 推理运行时

### 4.1 ONNX Runtime (CPU/通用 ARM)

```python
# 嵌入式 ARM Linux 推理
import onnxruntime as ort
import numpy as np

# ARM 优化: 使用 ARM NN 后端
session_options = ort.SessionOptions()
session_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)
session_options.intra_op_num_threads = 4  # 多核并行

session = ort.InferenceSession(
    'voice_model_int8.onnx',
    session_options,
    providers=['CPUExecutionProvider'],
)

# 推理
audio = np.random.randn(1, 1, 16000 * 3).astype(np.float32)
lengths = np.array([16000 * 3], dtype=np.int64)

outputs = session.run(None, {
    'audio': audio,
    'audio_lengths': lengths,
})

asr_log_probs, feat_lengths, dialect_logits, speaker_emb = outputs

# 后处理
# 1. ASR: CTC Greedy Decode
# 2. Dialect: softmax → argmax
# 3. Speaker: 直接使用 embedding 做余弦比对
```

### 4.2 RKNN Runtime API

```python
# RK3588 NPU 推理
from rknnlite.api import RKNNLite

rknn = RKNNLite()
rknn.load_rknn('voice_model.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

# 预处理音频 → INT8 量化输入
audio_int8 = quantize_audio_to_int8(audio_float32)

outputs = rknn.inference(inputs=[audio_int8])
# outputs 已经是 NPU 加速后的结果

rknn.release()
```

### 4.3 流式推理循环

```c
// C 语言嵌入式推理伪代码 (RK3588 为例)
// 参考 rknn_api.h

rknn_context ctx;
rknn_input inputs[2];
rknn_output outputs[4];

// 加载模型
rknn_init(&ctx, model_data, model_size, 0, NULL);

// 音频缓冲 (滑动窗口)
float audio_buffer[WINDOW_SIZE];  // e.g. 16k * 0.5s = 8000 samples

while (audio_available) {
    // 1. DMA 读取音频 chunk → audio_buffer
    dma_read(audio_buffer, WINDOW_SIZE);

    // 2. 设置输入
    inputs[0].buf = audio_buffer;
    inputs[0].size = WINDOW_SIZE * sizeof(float);
    inputs[0].fmt = RKNN_TENSOR_FLOAT32;

    inputs[1].buf = &window_size;
    inputs[1].size = sizeof(int);
    inputs[1].fmt = RKNN_TENSOR_INT64;

    // 3. NPU 推理
    rknn_run(ctx, NULL);

    // 4. 获取输出
    rknn_get_output(ctx, 0, outputs[0].buf, &outputs[0].size); // ASR log_probs
    rknn_get_output(ctx, 1, outputs[1].buf, &outputs[1].size); // dialect_logits
    rknn_get_output(ctx, 2, outputs[2].buf, &outputs[2].size); // speaker_embedding

    // 5. CPU 后处理 (CTC decode / softmax / cosine)
    ctc_greedy_decode(outputs[0].buf);
    softmax_argmax(outputs[1].buf);
    // speaker embedding 直接使用
}

rknn_destroy(ctx);
```

---

## 5. 性能优化

### 5.1 内存优化

| 优化项 | 方法 | 效果 |
|-------|------|------|
| 模型剪枝 | 移除贡献 < 1% 的 channel | -20% 参数 |
| 权值共享 | 聚类量化到 256 个码本 | -75% 存储 |
| 激活缓存复用 | 流式推理重用 buffer | -50% 内存 |
| 内存池 | 预分配固定大小 workspace | 零动态分配 |

### 5.2 速度优化

| 芯片 | FP32 (CPU) | INT8 (CPU) | INT8 (NPU) | 实时率 |
|------|-----------|-----------|-----------|--------|
| RK3588 CPU | 0.8x | 2.5x | - | 勉强实时 |
| RK3588 NPU | - | - | 15x | 远超实时 |
| A311D NPU | - | - | 20x | 远超实时 |
| RV1126 NPU | - | - | 8x | 远超实时 |

> 实时率 = 推理速度 / 音频长度, > 1x 即可实时

### 5.3 延迟优化

```
单次推理延迟 (1s 音频):
  NPU 推理:  5-20ms
  CPU 后处理: 1-5ms
  总延迟:    6-25ms (可接受)
```

### 5.4 精度对齐检查清单

- [ ] CER 下降 < 2% (INT8 量化后)
- [ ] 方言分类准确率下降 < 1%
- [ ] 声纹 EER 上升 < 1%
- [ ] 模型输出与 PyTorch FP32 差异 < 0.5%

---

## 6. 常见问题

### Q1: ONNX 导出 GRU 失败
```
错误: "Unsupported: GRU"
解决: 升级到 opset >= 14
      torch.onnx.export(..., opset_version=14)
```

### Q2: 量化后精度大幅下降
```
原因1: 校准数据不够 (需要 100+ 条)
原因2: GRU 层 INT8 溢出 → 改为 INT16
原因3: LayerNorm 不支持 → 替换为 BatchNorm
```

### Q3: NPU 推理崩溃
```
检查:
1. 模型 size 是否超过芯片限制
2. 是否存在不支持的算子
3. 输入格式是否匹配 (NCHW vs NHWC)
4. 内存是否对齐到 64 bytes
```

### Q4: 流式推理状态管理
```
GRU hidden state 需要在 chunk 间传递
方案: 导出时将 hidden state 也作为输入/输出
      或者使用 causal conv 替代 GRU
```

### Q5: 如何选择合适的芯片
```
低功耗 IoT (< 2W):    RV1126 (1 TOPS) — tiny 配置
智能音箱 (3-5W):      RK3588 (3 TOPS) — standard 配置
边缘网关 (5-10W):     A311D (5 TOPS) — standard/large 配置
```
