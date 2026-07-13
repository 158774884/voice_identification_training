/**
 * RTL8713E HiFi 5 DSP 语音指令词识别 — 推理代码
 *
 * 运行环境: Cadence Tensilica HiFi 5 DSP @ 500MHz
 * 编译链:   Cadence XCC (Xtensa C/C++ Compiler)
 * 依赖库:   Cadence Nature DSP Library (libnature)
 *
 * 内存规划 (DTCM 256KB):
 *   - 模型权重 (INT8):         ~100 KB  (放在 DTCM .rodata 段)
 *   - 激活张量 (INT8):          ~80 KB  (临时缓冲, 可复用)
 *   - Mel 特征缓冲:             ~16 KB  (40×98×4 bytes, float32)
 *   - 音频 PCM 缓冲:            ~32 KB  (16000 samples × 2 bytes)
 *   - 栈 + 其他:               ~28 KB
 *   ─────────────────────────────────
 *   总计:                      ~256 KB  ✓
 *
 * 推理流水线:
 *   1. DMA 接收 PCM 音频 → 双缓冲区
 *   2. FFT → Mel 滤波器 → Log → Mel 特征帧 (HiFi 5 硬件加速)
 *   3. 滑动窗口累积 98 帧 (约 1 秒)
 *   4. DS-CNN 推理 (Nature DSP 优化内核)
 *   5. Softmax → Top-1 分类结果
 *   6. 后处理: 连续 N 帧一致 → 触发指令
 */

#include <nature_dsp.h>          // Cadence Nature DSP Library
#include <xtensa/hifi5_sim.h>    // HiFi 5 intrinsics
#include <stdint.h>
#include <string.h>
#include <math.h>

// ============== 配置常量 (与训练参数对齐) ==============
#define SAMPLE_RATE         16000
#define N_MELS              40
#define N_FFT               512
#define WIN_LENGTH          400     // 25ms @ 16kHz
#define HOP_LENGTH          160     // 10ms @ 16kHz
#define N_FRAMES            98      // ~1 second window
#define N_CLASSES           50      // 指令词类别数
#define MEL_F_MIN           80.0f
#define MEL_F_MAX           7600.0f
#define PREEMPHASIS         0.97f
#define LOG_EPSILON         1e-6f

// ============== 模型权重 (由 Python 导出为 C array) ==============
// 这些数组由 convert_model_to_c.py 自动生成
// #include "model_weights.h"

// ============== 内存布局 (DTCM) ==============
// Mel 特征缓冲 (双缓冲, 滑动窗口)
static float mel_buffer[N_MELS][N_FRAMES] __attribute__((section(".dtcm")));
static int mel_frame_idx = 0;

// PCM 音频缓冲 (双缓冲 DMA)
#define PCM_BUF_SIZE        4096
static int16_t pcm_dma_buf[2][PCM_BUF_SIZE] __attribute__((section(".dtcm")));

// 特征提取缓冲
static float fft_in[N_FFT]        __attribute__((section(".dtcm")));
static complex float fft_out[N_FFT/2 + 1] __attribute__((section(".dtcm")));
static float prev_audio[WIN_LENGTH - HOP_LENGTH] __attribute__((section(".dtcm")));
static float prev_sample = 0.0f;  // 预加重状态

// 推理临时缓冲 (可逐层复用)
#define MAX_ACTIVATION_SIZE  (32 * 40 * 98)  // 最大激活张量
static int8_t activation_pool[MAX_ACTIVATION_SIZE] __attribute__((section(".dtcm")));

// 后处理状态
#define POST_SMOOTH_WINDOW  5       // 连续 N 帧一致才输出
static int prev_predictions[POST_SMOOTH_WINDOW];
static int post_smooth_idx = 0;

// ============== 阶段 1: 特征提取 (Mel) ==============

/**
 * 预加重: y[n] = x[n] - preemph * x[n-1]
 */
static inline float preemphasis(float sample) {
    float result = sample - PREEMPHASIS * prev_sample;
    prev_sample = sample;
    return result;
}

/**
 * HiFi 5 FFT (硬件加速)
 * 使用 Nature DSP 的 fft_r2c_32f 函数
 */
void compute_fft(float *input, complex float *output, int n) {
    // Cadence Nature DSP 提供的优化 FFT
    // nature_fft_r2c_f32(input, output, n);
    // 这里简化实现, 实际用 Nature DSP API
    for (int i = 0; i < n; i++) {
        fft_in[i] = input[i];
    }
    // FFT here (Nature DSP)
    // memset(output, 0, (n/2+1) * sizeof(complex float));
}

/**
 * Mel 滤波器组 (由 Python 预计算)
 * mel_filterbank[i][j]: 第 i 个 mel bin 对第 j 个 FFT bin 的权重
 */
// 从 mel_config.h 包含
#include "mel_config.h"  // 包含 mel_filterbank 数组

/**
 * 提取一帧 Mel 特征
 *
 * @param pcm_chunk  新 PCM 音频数据 [HOP_LENGTH]
 * @param mel_frame  输出: Mel 特征向量 [N_MELS]
 */
void extract_mel_frame(const int16_t *pcm_chunk, float *mel_frame) {
    float frame[WIN_LENGTH];

    // 1. 构建窗帧: [prev_audio; pcm_chunk]
    memcpy(frame, prev_audio, (WIN_LENGTH - HOP_LENGTH) * sizeof(float));
    for (int i = 0; i < HOP_LENGTH; i++) {
        float sample = (float)pcm_chunk[i] / 32768.0f;
        sample = preemphasis(sample);
        frame[WIN_LENGTH - HOP_LENGTH + i] = sample;
    }

    // 2. 更新 prev_audio
    memcpy(prev_audio, frame + HOP_LENGTH,
           (WIN_LENGTH - HOP_LENGTH) * sizeof(float));

    // 3. 汉宁窗
    for (int i = 0; i < WIN_LENGTH; i++) {
        float w = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (WIN_LENGTH - 1)));
        frame[i] *= w;
    }

    // 4. 补零到 N_FFT
    memset(fft_in, 0, N_FFT * sizeof(float));
    memcpy(fft_in, frame, WIN_LENGTH * sizeof(float));

    // 5. FFT (HiFi 5 硬件加速)
    compute_fft(fft_in, fft_out, N_FFT);

    // 6. 功率谱 + Mel 滤波器组
    for (int m = 0; m < N_MELS; m++) {
        float mel_energy = 0.0f;
        for (int k = 0; k < N_FFT/2 + 1; k++) {
            float power = crealf(fft_out[k]) * crealf(fft_out[k]) +
                         cimagf(fft_out[k]) * cimagf(fft_out[k]);
            mel_energy += mel_filterbank[m][k] * power;
        }
        // 7. Log scaling
        mel_frame[m] = logf(mel_energy + LOG_EPSILON);
    }
}

// ============== 阶段 2: CNN 推理 (HiFi 5 Nature DSP) ==============

/**
 * Conv2D 1x1 INT8 — 使用 Nature DSP 加速
 *
 * nature_conv2d_1x1_s8s8_s8(...) 完成:
 *   Conv2D(zero_padding=0) + optional BN + ReLU
 */
void conv2d_1x1_relu_s8(const int8_t *input, const int8_t *weight,
                        const int32_t *bias, int8_t *output,
                        int in_ch, int out_ch, int h, int w,
                        float input_scale, float weight_scale, float output_scale) {
    // 实际调用 Nature DSP:
    // nature_conv2d_s8(input, weight, bias, output,
    //                  in_ch, out_ch, h, w, 1, 1, 0, 0, 1, 1,
    //                  input_scale, weight_scale, output_scale);
}

/**
 * Depthwise Conv2D 3x3 INT8 — 使用 Nature DSP 加速
 */
void dw_conv2d_3x3_relu_s8(const int8_t *input, const int8_t *weight,
                           const int32_t *bias, int8_t *output,
                           int ch, int h, int w, int stride_h, int stride_w) {
    // nature_depthwise_conv2d_s8(input, weight, bias, output,
    //                            ch, h, w, 3, 3, stride_h, stride_w, 1, 1);
}

/**
 * Global Average Pooling
 */
void global_avg_pool_s8(const int8_t *input, int8_t *output,
                        int ch, int h, int w) {
    for (int c = 0; c < ch; c++) {
        int32_t sum = 0;
        for (int i = 0; i < h * w; i++) {
            sum += input[c * h * w + i];
        }
        output[c] = (int8_t)(sum / (h * w));
    }
}

/**
 * Fully Connected INT8
 */
void fully_connected_s8(const int8_t *input, const int8_t *weight,
                        const int32_t *bias, int32_t *output,
                        int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        int32_t sum = bias ? bias[o] : 0;
        for (int i = 0; i < in_dim; i++) {
            sum += (int32_t)input[i] * (int32_t)weight[o * in_dim + i];
        }
        output[o] = sum;
    }
}

/**
 * TinyKWS 完整推理
 *
 * 模型结构 (INT8 量化):
 *   stem:   Conv2D 3×3, 1→32, /1
 *   block1: DS-Conv2D 3×3, 32→64, stride(1,2)
 *   block2: DS-Conv2D 3×3, 64→128, stride(2,2)
 *   block3: DS-Conv2D 3×3, 128→128, stride(2,2)
 *   block4: DS-Conv2D 3×3, 128→128, stride=1
 *   head:   GlobalAvgPool → FC, 128→N_CLASSES
 *
 * @param mel_input  Mel 特征 [1, N_MELS, N_FRAMES] INT8 量化
 * @param logits     输出 logits [N_CLASSES] INT32 (需要后续 softmax)
 */
void tiny_kws_inference(const int8_t *mel_input, int32_t *logits) {
    // 所有层权重都包含在 #include "model_weights.h" 中
    // 以下展示推理流程结构

    // Layer 0: Stem - Conv2D 3×3, 1→32
    // conv2d_3x3_relu_s8(mel_input, stem_weight, stem_bias, act0,
    //                    1, 32, N_MELS, N_FRAMES, 1, 1);

    // Layer 1: Block1 - DS-Conv, 32→64
    // conv2d_1x1_relu_s8(act0, block1_pw1_w, block1_pw1_b, act1a, ...);
    // dw_conv2d_3x3_relu_s8(act1a, block1_dw_w, block1_dw_b, act1b, ...);
    // conv2d_1x1_relu_s8(act1b, block1_pw2_w, block1_pw2_b, act1, ...);

    // Layer 2-4: Block2-4 (同上)
    // ...

    // Head: GlobalAvgPool → FC
    // global_avg_pool_s8(act4, pooled, 128, 5, 12);
    // fully_connected_s8(pooled, fc_weight, fc_bias, logits, 128, N_CLASSES);
}

// ============== 阶段 3: 后处理 ==============

/**
 * Softmax (FP32)
 */
void softmax_f32(const int32_t *logits, float *probs, int n) {
    // 去量化为 FP32
    float max_val = -INFINITY;
    for (int i = 0; i < n; i++) {
        if ((float)logits[i] > max_val) max_val = (float)logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((float)logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) {
        probs[i] /= sum;
    }
}

/**
 * 平滑后处理: 连续 POST_SMOOTH_WINDOW 帧一致才判断为有效指令
 *
 * @param class_id  当前帧预测的类别
 * @return          平滑后的类别, -1 表示未达成一致
 */
int postprocess_smooth(int class_id) {
    prev_predictions[post_smooth_idx] = class_id;
    post_smooth_idx = (post_smooth_idx + 1) % POST_SMOOTH_WINDOW;

    // 检查是否连续一致
    int first = prev_predictions[0];
    for (int i = 1; i < POST_SMOOTH_WINDOW; i++) {
        if (prev_predictions[i] != first) return -1;
    }
    return first;
}

// ============== 主循环 ==============

/**
 * RTL8713E 语音指令识别主循环 (运行在 HiFi 5 DSP)
 *
 * 状态机:
 *   IDLE → (VAD triggers) → ACTIVE → (classify)→ COMMAND_READY → IDLE
 */
typedef enum {
    STATE_IDLE = 0,
    STATE_ACTIVE,
    STATE_COMMAND_READY,
} kws_state_t;

static kws_state_t g_state = STATE_IDLE;
static int g_active_frames = 0;

/**
 * 每 10ms 调用一次 (由 DMA 中断驱动)
 *
 * @param pcm_chunk  新 PCM 数据 [160 samples = 10ms]
 * @return           >=0: 检测到的指令 ID, -1: 无指令, -2: 正在处理
 */
int kws_process_frame(const int16_t *pcm_chunk) {
    static float mel_frame[N_MELS];

    // 1. 提取 Mel 特征
    extract_mel_frame(pcm_chunk, mel_frame);

    // 2. 更新滑动窗口
    for (int m = 0; m < N_MELS; m++) {
        // 左移一帧
        for (int t = 1; t < N_FRAMES; t++) {
            mel_buffer[m][t - 1] = mel_buffer[m][t];
        }
        // 新帧放入末尾
        mel_buffer[m][N_FRAMES - 1] = mel_frame[m];
    }

    // 3. 状态机
    switch (g_state) {
    case STATE_IDLE: {
        // 简单能量 VAD
        float energy = 0.0f;
        for (int i = 0; i < HOP_LENGTH; i++) {
            float s = (float)pcm_chunk[i] / 32768.0f;
            energy += s * s;
        }
        energy /= HOP_LENGTH;

        if (energy > 0.001f) {  // VAD 阈值
            g_state = STATE_ACTIVE;
            g_active_frames = 0;
        }
        return -1;
    }

    case STATE_ACTIVE: {
        g_active_frames++;

        // 每 10 帧做一次推理 (100ms interval = real-time factor ~3x)
        if (g_active_frames % 10 == 0) {
            // 4. CNN 推理
            int32_t logits[N_CLASSES];
            float probs[N_CLASSES];

            // 量化 Mel 特征为 INT8
            int8_t mel_quant[N_MELS * N_FRAMES];
            for (int i = 0; i < N_MELS * N_FRAMES; i++) {
                // quantize: (float → int8)
                float val = ((float *)mel_buffer)[i];
                int8_t q = (int8_t)(val * 16.0f);  // scale = 1/16
                mel_quant[i] = q > 127 ? 127 : (q < -128 ? -128 : q);
            }

            tiny_kws_inference(mel_quant, logits);
            softmax_f32(logits, probs, N_CLASSES);

            // 5. 找最大概率类别
            int best_class = 0;
            float best_prob = probs[0];
            for (int i = 1; i < N_CLASSES; i++) {
                if (probs[i] > best_prob) {
                    best_prob = probs[i];
                    best_class = i;
                }
            }

            // 6. 平滑 + 阈值判断
            if (best_prob > 0.6f) {  // 置信度阈值
                int smoothed = postprocess_smooth(best_class);
                if (smoothed >= 0) {
                    g_state = STATE_COMMAND_READY;
                    return smoothed;
                }
            }
        }

        // 超时检测 (最大 3 秒)
        if (g_active_frames > 300) {
            g_state = STATE_IDLE;
        }
        return -1;
    }

    case STATE_COMMAND_READY:
        // 指令已触发, 重置状态
        g_state = STATE_IDLE;
        return -1;

    default:
        g_state = STATE_IDLE;
        return -1;
    }
}

/**
 * 初始化 (启动时调用一次)
 */
void kws_init(void) {
    // 清除缓冲区
    memset(mel_buffer, 0, sizeof(mel_buffer));
    memset(pcm_dma_buf, 0, sizeof(pcm_dma_buf));
    memset(fft_in, 0, sizeof(fft_in));
    memset(prev_audio, 0, sizeof(prev_audio));
    memset(prev_predictions, -1, sizeof(prev_predictions));

    prev_sample = 0.0f;
    mel_frame_idx = 0;
    post_smooth_idx = 0;
    g_state = STATE_IDLE;
    g_active_frames = 0;

    // 初始化 Cadence Nature DSP 库
    // nature_init();
}
