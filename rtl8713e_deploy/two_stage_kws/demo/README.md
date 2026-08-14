# 两阶段 KWS 芯片移植 Demo

本目录是「导出 C 固件」时自动生成的移植示例，展示如何把导出的模型权重
（`stage1_model.h` / `stage2_model.h` / `grammar.h` / `mel_config.h`）集成到芯片平台。

## 文件清单

| 文件 | 说明 |
|---|---|
| `main_demo_ac7916.c` | AC7916AB 平台主程序示例（含命令映射、音频回调、`app_main` 入口） |
| `main_demo_generic.c` | 通用 C 平台主程序（无芯片 SDK 依赖，从 WAV 读 PCM，PC 可编译） |
| `kws_pipeline.c` | 两阶段流水线实现：Mel 提取 + 唤醒/命令状态机 + CTC + 语法解码 |
| `kws_pipeline.h` | 流水线对外接口（导出时生成，含 `kws_config.h` 引用） |
| `kws_config.h` | 模型配置宏：类别数、token 数、blank id、唤醒词/命令标签（导出时生成） |

## 快速验证（PC）

```bash
# 在导出目录下，把 demo/ 里的 .c 复制出来和 .h 放一起（或直接在该目录编译）
gcc main_demo_generic.c kws_pipeline.c -lm -o kws_demo
./kws_demo 你的.wav        # 16kHz / 16bit / 单声道
```

## 移植到芯片平台

1. 把 `stage1_model.h` / `stage2_model.h` / `grammar.h` / `mel_config.h` /
   `kws_config.h` / `kws_pipeline.h` 加入工程 `include/`。
2. 把 `kws_pipeline.c` 加入工程 `src/`。
3. 参考 `main_demo_ac7916.c` 写你的 `main.c`：初始化 `kws_pipeline_init()`，
   在音频回调里每 10ms 调 `kws_pipeline_feed(pcm_10ms)`，根据返回的
   `cmd_id` 执行命令。
4. 替换两处 **TODO**（`kws_pipeline.c` 中）：
   - `compute_fft` → 芯片硬件 FFT API（如 `JL_FFT_R2C`）
   - `stage1_inference` / `stage2_inference` → 芯片 NPU/MVA 硬件加速 API
     （如 `JL_MVA_Conv1D`），或按 `stage*_model.h` 里的权重数组写纯 C 推理

## 关键配置宏（`kws_config.h`）

- `STAGE1_NUM_CLASSES` — 唤醒词类别数
- `STAGE2_NUM_TOKENS`  — 命令词 token 数
- `STAGE2_BLANK_ID`   — CTC blank token id
- `stage1_labels[]`   — 唤醒词标签
- `stage2_tokens[]`   — token → 字符映射

## 内存布局参考

见导出目录下的 `flash_layout.txt`（各权重文件大小 + Flash/PSRAM 占用建议）。
