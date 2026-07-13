# 轻量化多任务语音深度学习模型

## 项目概述

一套面向 **SOC 嵌入式芯片部署** 的轻量化多任务语音深度学习模型，单模型同时支持:

| 任务 | 描述 | 输出 |
|------|------|------|
| **ASR 语音识别** | 中文 + 多方言 CTC 识别 | 中文字符序列 |
| **方言分类** | 粤语/川渝/吴语/闽南语/普通话等 10 类 | 方言标签 + 置信度 |
| **声纹识别** | TDNN 精简版声纹嵌入提取 | 256-dim 嵌入向量 |

### 核心特点

- **极致轻量**: ~4.5M 参数，INT8 量化后 ~4.5MB
- **ONNX 兼容**: 全 Conv1d/BN/ReLU/GRU/FC 算子，NPU 原生支持
- **流式推理**: Causal Conv + UniGRU，逐帧/逐段处理
- **三阶段训练**: 预训练 → 联合训练 → 全参数微调
- **开箱即用**: 完整训练/推理/部署代码

## 网络架构

```
Input: 16kHz Raw Audio [B, 1, T]
        │
        ▼
┌──────────────────────────────┐
│  Shared Backbone (~2.5M)     │
│  ┌────────────────────────┐  │
│  │ Conv Frontend          │  │
│  │ Conv1d(1→64, K=400,    │  │
│  │        S=160) → 100Hz  │  │
│  │ Conv1d(64→128, S=2)    │  │
│  │ Conv1d(128→256, S=2)   │  │
│  │ → 25Hz feature rate    │  │
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │ Tiny Conformer ×4      │  │
│  │ FFN→DWConv→GRU→FFN     │  │
│  │ (无 MHA, ONNX 友好)     │  │
│  └────────────────────────┘  │
│  Output: [B, 256, T'] @25Hz  │
└──┬──────────┬──────────┬─────┘
   │          │          │
┌──▼────┐ ┌──▼────┐ ┌───▼──────┐
│ ASR   │ │Dialect│ │ Speaker  │
│ CTC   │ │Attn   │ │ TDNN×4   │
│ Conv×2│ │Pool   │ │+SE Block │
│→Vocab │ │→FC×2  │ │→256-dim  │
│~1.3M  │ │~50K   │ │~800K     │
└───────┘ └───────┘ └──────────┘
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 模型创建和测试

```python
from model.multi_task_model import create_model
import torch

# 创建模型
model = create_model()
model.summary()

# 测试前向传播
audio = torch.randn(2, 1, 16000 * 3)  # 2 条 3 秒音频
lengths = torch.tensor([16000 * 3, 16000 * 2])

outputs = model(audio, lengths)
print(f"ASR log_probs: {outputs['asr_log_probs'].shape}")
print(f"Dialect logits: {outputs['dialect_logits'].shape}")
print(f"Speaker embedding: {outputs['speaker_embedding'].shape}")
```

### 训练

```bash
# 标准配置训练
python train.py \
    --data_root ./data \
    --preset standard \
    --batch_size 32 \
    --device cuda

# 极小模型 (超低功耗)
python train.py --preset tiny

# 从检查点恢复
python train.py --resume checkpoints/best_model.pt
```

### 推理

```python
from inference.pipeline import VoiceInferencePipeline
from data.vocab import get_default_vocab
import torch

# 加载模型
model = create_model()
model.load_state_dict(torch.load('checkpoints/best_model.pt')['model_state_dict'])
vocab = get_default_vocab()

# 创建推理流水线
pipeline = VoiceInferencePipeline(model, vocab, device='cpu')

# 从音频文件推理
results = pipeline.from_file('test.wav')
print(f"ASR: {results['asr_text']}")
print(f"Dialect: {results['dialect_zh']} ({results['dialect_confidence']:.2%})")
print(f"Speaker emb shape: {results['speaker_embedding'].shape}")
```

### 声纹比对

```python
from inference.speaker_inference import SpeakerInference, SpeakerVerification

extractor = SpeakerInference(model)
verifier = SpeakerVerification(extractor, threshold=0.65)

# 注册说话人
verifier.enroll('张三', audio_zhangsan)

# 1:1 验证
is_match, similarity = verifier.verify('张三', test_audio)
print(f"Same speaker: {is_match} (similarity: {similarity:.3f})")

# 1:N 识别
results = verifier.identify(test_audio, top_k=3)
for name, sim in results:
    print(f"  {name}: {sim:.3f}")
```

### ONNX 导出

```python
from deployment.export_onnx import export_to_onnx

paths = export_to_onnx(model, output_dir='./onnx_models', export_mode='all')
# → voice_model_full.onnx, voice_model_asr.onnx, ...
```

## 项目结构

```
voice_identfication/
├── model/                    # 模型定义
│   ├── shared_backbone.py    # 共享主干 (Conv+GRU)
│   ├── asr_branch.py         # ASR CTC 分支
│   ├── dialect_branch.py     # 方言分类分支
│   ├── speaker_branch.py     # 声纹 TDNN 分支
│   └── multi_task_model.py   # 组合模型 + 工厂函数
├── data/                     # 数据处理
│   ├── preprocessing.py      # 音频预处理 (降噪/重采样/归一化)
│   ├── augmentation.py       # 数据增强 (变速/加噪/混响/SpecAug)
│   ├── dataset.py            # PyTorch Dataset + DataLoader
│   └── vocab.py              # 中文字符词汇表 + 方言标签
├── training/                 # 训练
│   ├── losses.py             # 多任务损失 (CTC+CE+AAM-Softmax)
│   ├── trainer.py            # 三阶段训练器 (预训练/联合/微调)
│   └── config.py             # 超参数 (tiny/standard/large 预设)
├── inference/                # 推理
│   ├── pipeline.py           # 统一推理流水线
│   ├── asr_inference.py      # ASR 解码 (Greedy+Beam Search)
│   ├── dialect_inference.py  # 方言识别
│   └── speaker_inference.py  # 声纹比对 (1:1+1:N)
├── deployment/               # 部署
│   ├── export_onnx.py        # ONNX 导出 (全量/分支/验证)
│   ├── quantization.py       # INT8/INT16 量化 (PyTorch/ONNX)
│   └── soc_deploy_guide.md   # SOC 芯片部署完整指南
├── utils/                    # 工具
│   └── metrics.py            # CER/WER/EER/minDCF 评估
├── train.py                  # 训练入口
├── requirements.txt
└── README.md
```

## 训练策略

### 三阶段训练

| 阶段 | Epochs | 策略 | 学习率 |
|------|--------|------|--------|
| Phase 1: 预训练 | 10 | ASR + Dialect, 冻结 Speaker | 1e-3 |
| Phase 2: 联合训练 | 60 | 所有任务, 冻结 Backbone前2层 | 1e-3 |
| Phase 3: 微调 | 30 | 解冻全部, 低学习率精调 | 1e-4 |

### 损失函数

```
Total Loss = 1.0 * CTC_Loss        (ASR)
           + 0.3 * CrossEntropy     (Dialect)
           + 0.5 * AAM-Softmax      (Speaker)
```

### 推荐超参

| 参数 | Tiny | Standard | Large |
|------|------|----------|-------|
| Backbone dim | 128 | 256 | 320 |
| Blocks | 2 | 4 | 6 |
| Vocab size | 4000 | 6000 | 8000 |
| Embed dim | 128 | 256 | 320 |
| 参数量 | ~1.2M | ~4.5M | ~8M |
| Batch size | 64 | 32 | 16 |

## SOC 部署状态

| 算子 | ONNX | RK3588 | A311D | Hi3559A |
|------|------|--------|-------|---------|
| Conv1d | ✅ | ✅ | ✅ | ✅ |
| BatchNorm | ✅ | ✅ | ✅ | ✅ |
| ReLU | ✅ | ✅ | ✅ | ✅ |
| GRU | ✅ | ✅ | ✅ | ❌ |
| Linear | ✅ | ✅ | ✅ | ✅ |
| Softmax | ✅ | ✅ | ✅ | ✅ |
| CTC (外部) | N/A | CPU | CPU | CPU |

> CTC 解码在 NPU 外部 CPU 完成，计算量极小

## 数据集准备

```jsonl
{"audio_path": "audio/spk001/utt001.wav", "text": "你好世界", "dialect": "mandarin", "speaker_id": "spk001", "duration": 2.5}
{"audio_path": "audio/spk001/utt002.wav", "text": "今日天气好好", "dialect": "cantonese", "speaker_id": "spk002", "duration": 3.1}
```

## License

MIT
