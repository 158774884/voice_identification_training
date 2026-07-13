/**
 * kws_pipeline_jl.c — 对接杰理 WL82 SDK 的两级语音识别
 *
 * 部署: 复制到 SDK/apps/common/asr/jlkws/ 或自建目录
 * 编译: 加入 SDK Makefile
 *
 * SDK 依赖 (已在 SDK 中):
 *   #include "server/audio_server.h"      // 音频采集
 *   #include "server/server_core.h"       // server 框架
 *   #include "generic/circular_buf.h"     // 环形缓冲
 *   #include "os/os_api.h"               // 线程/信号量
 *   #include "event.h"                    // 事件
 *   #include "app_config.h"              // 配置
 *
 * 自建头文件 (从 ac7916_firmware/ 复制):
 *   #include "stage1_model.h"            // Stage1 INT8 权重
 *   #include "stage2_model.h"            // Stage2 INT8 权重
 *   #include "grammar.h"                 // WFST 语法图
 *   #include "mel_config.h"              // Mel 滤波器组
 *   #include "kws_pipeline.h"            // API 声明
 */

#include "server/audio_server.h"
#include "server/server_core.h"
#include "generic/circular_buf.h"
#include "os/os_api.h"
#include "event.h"
#include "app_config.h"

#include "stage1_model.h"
#include "stage2_model.h"
#include "grammar.h"
#include "mel_config.h"
#include "kws_pipeline.h"

#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

// ==================== 配置 ====================

#define ONCE_SR_POINTS     160     // 每帧 160 samples (10ms @ 16kHz)
#define AISP_BUF_SIZE       (ONCE_SR_POINTS * 2 * 4)  // PCM 缓冲
#define WINDOW_FRAMES       98      // Mel 窗口帧数 (~1s)
#define STAGE2_MAX_T        50      // Stage2 最大时间帧

// ==================== 运行时状态 ====================

static struct {
    // 线程
    int pid;
    u8   exit_flag;
    u8   run_flag;
    OS_SEM sem;

    // 音频
    u16  sample_rate;
    void *mic_enc;
    s16  mic_buf[AISP_BUF_SIZE * 2];
    cbuffer_t mic_cbuf;

    // 特征
    float prev_sample;
    float audio_ring[MEL_WIN_LENGTH];
    int   audio_ring_pos;
    float mel_window[MEL_N_MELS][WINDOW_FRAMES];
    int   mel_frame_count;

    // 状态机
    kws_state_t state;
    int state_frames;
    int cmd_frames;
    int last_speech_frame;
    int total_frames;

    // 平滑
    int smooth_hist[5];
    int smooth_idx;

    // 结果
    char command_text[128];
    int  command_id;

    // Stage2 激活缓冲 (在 PSRAM 分配)
    int8_t *s2_scratch;
} kws;

// ==================== Mel 特征提取 ====================

// 纯 C 的 RDFT (可替换为 CMSIS-DSP arm_rfft_fast_f32)
static void rdft_c(const float *in, float *real, float *imag, int n)
{
    int k, j;
    for (k = 0; k < n/2 + 1; k++) {
        real[k] = 0; imag[k] = 0;
        for (j = 0; j < n; j++) {
            float angle = -6.2831853f * k * j / n;
            real[k] += in[j] * cosf(angle);
            imag[k] += in[j] * sinf(angle);
        }
    }
}

static void extract_mel_frame(const int16_t *pcm, float *mel_out)
{
    int i, m, k;
    float frame[MEL_N_FFT];
    float real[MEL_N_FFT/2 + 1], imag[MEL_N_FFT/2 + 1];

    // 1. 更新环形缓冲 + pre-emphasis
    for (i = 0; i < MEL_HOP_LENGTH; i++) {
        float s = (float)pcm[i] / 32768.0f;
        float emph = s - MEL_PREEMPHASIS * kws.prev_sample;
        kws.prev_sample = s;
        kws.audio_ring[kws.audio_ring_pos] = emph;
        kws.audio_ring_pos = (kws.audio_ring_pos + 1) % MEL_WIN_LENGTH;
    }

    // 2. 汉宁窗
    for (i = 0; i < MEL_WIN_LENGTH; i++) {
        int idx = (kws.audio_ring_pos + i) % MEL_WIN_LENGTH;
        float w = 0.5f * (1.0f - cosf(6.2831853f * i / (MEL_WIN_LENGTH - 1)));
        frame[i] = kws.audio_ring[idx] * w;
    }
    for (i = MEL_WIN_LENGTH; i < MEL_N_FFT; i++) frame[i] = 0;

    // 3. FFT → 功率谱 → Mel 滤波
    rdft_c(frame, real, imag, MEL_N_FFT);
    for (m = 0; m < MEL_N_MELS; m++) {
        float energy = 0;
        for (k = 0; k < MEL_N_FFT/2 + 1; k++)
            energy += (real[k]*real[k] + imag[k]*imag[k]) * mel_filterbank[m][k];
        mel_out[m] = logf(energy + 1e-6f);
    }
}

// ==================== 轻量 INT8 推理引擎 ====================

// Conv2D INT8: input[H][W] × weight[OC][IC][KH][KW] → output
// 简化: 针对 Stage1 的特定形状手写
static void conv2d_s1(const int8_t *in, int ih, int iw, int ic,
                       const int8_t *w, const float *scale,
                       int8_t *out, int *oh, int *ow, int oc,
                       int kh, int kw, int sh, int sw)
{
    int o_h = (ih - kh) / sh + 1;
    int o_w = (iw - kw) / sw + 1;
    int o, h, w, c, r, s;

    for (o = 0; o < oc; o++) {
        float s_out = scale[o];
        for (h = 0; h < o_h; h++) {
            for (w = 0; w < o_w; w++) {
                int sum = 0;
                for (c = 0; c < ic; c++) {
                    for (r = 0; r < kh; r++) {
                        for (s_ = 0; s_ < kw; s_++) {
                            int in_h = h * sh + r;
                            int in_w = w * sw + s_;
                            sum += (int)in[(c * ih + in_h) * iw + in_w] *
                                   (int)w[((o * ic + c) * kh + r) * kw + s_];
                        }
                    }
                }
                out[(o * o_h + h) * o_w + w] = (int8_t)(sum / (ic * kh * kw));
            }
        }
    }
    *oh = o_h; *ow = o_w;
}

// Fully Connected INT8
static void fc_s1(const int8_t *in, int in_dim,
                   const int8_t *w, const int8_t *b,
                   int32_t *out, int out_dim)
{
    int o, i;
    for (o = 0; o < out_dim; o++) {
        int32_t sum = b ? (int32_t)b[o] << 8 : 0;
        for (i = 0; i < in_dim; i++)
            sum += (int32_t)in[i] * (int32_t)w[o * in_dim + i];
        out[o] = sum;
    }
}

// ==================== Stage1 推理 ====================

// 1x1 Conv2D (用于 Stage1 中的 pw conv)
static void conv2d_1x1(const int8_t *in, int ih, int iw, int ic,
                        const int8_t *w, int8_t *out, int oc)
{
    int o, h, w, c;
    for (o = 0; o < oc; o++) {
        for (h = 0; h < ih; h++) {
            for (w = 0; w < iw; w++) {
                int sum = 0;
                for (c = 0; c < ic; c++)
                    sum += (int)in[(c * ih + h) * iw + w] * (int)w[o * ic + c];
                out[(o * ih + h) * iw + w] = (int8_t)(sum / ic);
            }
        }
    }
}

// DW Conv2D (depthwise, groups=ic)
static void dw_conv2d(const int8_t *in, int ih, int iw, int ic,
                       const int8_t *w, int kh, int kw,
                       int8_t *out, int *oh, int *ow, int sh, int sw)
{
    int o_h = (ih - kh) / sh + 1;
    int o_w = (iw - kw) / sw + 1;
    int c, h, w, r, s;
    for (c = 0; c < ic; c++) {
        for (h = 0; h < o_h; h++) {
            for (w = 0; w < o_w; w++) {
                int sum = 0;
                for (r = 0; r < kh; r++)
                    for (s = 0; s < kw; s++)
                        sum += (int)in[(c * ih + h * sh + r) * iw + w * sw + s] *
                               (int)w[(c * kh + r) * kw + s];
                out[(c * o_h + h) * o_w + w] = (int8_t)(sum / (kh * kw));
            }
        }
    }
    *oh = o_h; *ow = o_w;
}

// Global Avg Pool
static void global_avg_pool(const int8_t *in, int ih, int iw, int ic,
                             int8_t *out)
{
    int c, h, w;
    for (c = 0; c < ic; c++) {
        int sum = 0;
        for (h = 0; h < ih; h++)
            for (w = 0; w < iw; w++)
                sum += (int)in[(c * ih + h) * iw + w];
        out[c] = (int8_t)(sum / (ih * iw));
    }
}

// ReLU INT8
static void relu(int8_t *x, int n)
{
    int i;
    for (i = 0; i < n; i++)
        if (x[i] < 0) x[i] = 0;
}

static int stage1_inference(const int8_t *mel_flat)
{
    int oh, ow;
    int8_t act1[32][20][49];
    int8_t act2[32][20][25];
    int8_t act3[32][20][25];
    int8_t act4[32][10][13];
    int8_t act5[32][10][13];
    int8_t pooled[32];
    int32_t logits[STAGE1_NUM_CLASSES];

    // Block 1: Conv2D(1→32, 5×5, stride 2, pad 2)
    conv2d_s1(mel_flat, MEL_N_MELS, WINDOW_FRAMES, 1,
              stage1_conv1_weight, &stage1_conv1_weight_scale,
              (int8_t*)act1, &oh, &ow, 32, 5, 5, 2, 2);
    relu((int8_t*)act1, 32 * oh * ow);

    // Block 2: DW Conv2D(32→32, 3×3, stride(1,2))
    dw_conv2d((int8_t*)act1, oh, ow, 32,
              stage1_dw_weight, 3, 3,
              (int8_t*)act2, &oh, &ow, 1, 2);
    relu((int8_t*)act2, 32 * oh * ow);

    // Block 3: 1×1 Conv
    conv2d_1x1((int8_t*)act2, oh, ow, 32,
               stage1_conv2_weight, (int8_t*)act3, 32);
    relu((int8_t*)act3, 32 * oh * ow);

    // Block 4: DW Conv2D(32→32, 3×3, stride(1,2))
    int oh2, ow2;
    dw_conv2d((int8_t*)act3, oh, ow, 32,
              stage1_dw2_weight, 3, 3,
              (int8_t*)act4, &oh2, &ow2, 1, 2);
    relu((int8_t*)act4, 32 * oh2 * ow2);

    // Block 5: 1×1 Conv
    conv2d_1x1((int8_t*)act4, oh2, ow2, 32,
               stage1_conv3_weight, (int8_t*)act5, 32);
    relu((int8_t*)act5, 32 * oh2 * ow2);

    // Pool + FC
    global_avg_pool((int8_t*)act5, oh2, ow2, 32, pooled);
    fc_s1(pooled, 32, stage1_fc_weight, NULL, logits, STAGE1_NUM_CLASSES);

    // argmax
    int best = 0;
    for (int i = 1; i < STAGE1_NUM_CLASSES; i++)
        if (logits[i] > logits[best]) best = i;

    return best;
}

// ==================== Stage2 推理 (简化 CPU 版) ====================

// Conv1D
static void conv1d(const int8_t *in, int t_in, int c_in,
                    const int8_t *w, int c_out, int k,
                    int8_t *out, int *t_out)
{
    int pad = k / 2;
    *t_out = t_in;
    int o, i, t, c;
    for (o = 0; o < c_out; o++) {
        for (t = 0; t < *t_out; t++) {
            int sum = 0;
            for (c = 0; c < c_in; c++) {
                for (i = 0; i < k; i++) {
                    int ti = t + i - pad;
                    if (ti >= 0 && ti < t_in)
                        sum += (int)in[c * t_in + ti] *
                               (int)w[((o * c_in + c) * k) + i];
                }
            }
            out[o * (*t_out) + t] = (int8_t)(sum / (c_in * k));
        }
    }
}

static int stage2_inference(const int8_t *mel, int t_frames,
                             int32_t *logits_out)
{
    // 简化: 仅做前向 Conv + log_softmax 框架
    // 实际完整 Stage2 需展开 encode 的所有层
    // 当前为占位, 返回全零 (无检测)
    (void)mel; (void)t_frames;
    memset(logits_out, 0, STAGE2_NUM_TOKENS * STAGE2_MAX_T * sizeof(int32_t));
    return STAGE2_MAX_T;
}

// ==================== CTC Greedy + Grammar ====================

static int ctc_decode(const int32_t *logits, int t_max, int n_tokens,
                       int blank, char *out, int out_max)
{
    int t, prev = blank, pos = 0;
    for (t = 0; t < t_max; t++) {
        int best_v = blank;
        int32_t best_s = logits[t * n_tokens + blank];
        for (int v = 1; v < n_tokens; v++) {
            if (logits[t * n_tokens + v] > best_s) {
                best_s = logits[t * n_tokens + v];
                best_v = v;
            }
        }
        if (best_v != blank && best_v != prev && pos < out_max - 1) {
            if (best_v < GRAMMAR_NUM_TOKENS && grammar_tokens[best_v][0])
                out[pos++] = grammar_tokens[best_v][0];
        }
        prev = best_v;
    }
    out[pos] = '\0';
    return pos;
}

// ==================== Pipeline ====================

void kws_pipeline_init(void)
{
    memset(&kws, 0, sizeof(kws));
    kws.sample_rate = 16000;
    kws.state = KWS_IDLE;
    memset(kws.smooth_hist, -1, sizeof(kws.smooth_hist));
}

int kws_pipeline_feed(const int16_t *pcm_10ms)
{
    float mel_40[MEL_N_MELS];
    kws.total_frames++;

    // 1. Mel 特征
    extract_mel_frame(pcm_10ms, mel_40);

    // 2. 滑动窗口
    int m, t;
    for (m = 0; m < MEL_N_MELS; m++) {
        for (t = 1; t < WINDOW_FRAMES; t++)
            kws.mel_window[m][t-1] = kws.mel_window[m][t];
        kws.mel_window[m][WINDOW_FRAMES-1] = mel_40[m];
    }
    kws.mel_frame_count++;

    // 3. 能量
    float energy = 0;
    for (int i = 0; i < MEL_HOP_LENGTH; i++) {
        float s = (float)pcm_10ms[i] / 32768.0f;
        energy += s * s;
    }

    // 4. 状态机
    switch (kws.state) {
    case KWS_IDLE:
        if (energy > 0.001f) {
            kws.state = KWS_LISTENING;
            kws.state_frames = 0;
        }
        return -1;

    case KWS_LISTENING:
        kws.state_frames++;
        if (kws.state_frames % 10 == 0 && kws.mel_frame_count > WINDOW_FRAMES) {
            // 量化 Mel
            int8_t mel_q[MEL_N_MELS * WINDOW_FRAMES];
            for (int i = 0; i < MEL_N_MELS * WINDOW_FRAMES; i++)
                mel_q[i] = (int8_t)(((float*)kws.mel_window)[i] * 16.0f);

            int cls = stage1_inference(mel_q);

            // 平滑
            kws.smooth_hist[kws.smooth_idx] = cls;
            kws.smooth_idx = (kws.smooth_idx + 1) % 5;
            int first = kws.smooth_hist[0], all_same = 1;
            for (int i = 1; i < 5; i++)
                if (kws.smooth_hist[i] != first) all_same = 0;

            if (all_same && first >= 0 && first < STAGE1_NUM_CLASSES - 1) {
                kws.state = KWS_WOKE;
                kws.cmd_frames = 0;
                kws.command_id = first;
                printf("[KWS] Wake! %s\n", stage1_labels[first]);
                return -1;
            }
        }
        if (kws.state_frames > 300) kws.state = KWS_IDLE;
        return -1;

    case KWS_WOKE:
        kws.cmd_frames++;
        if (energy > 0.001f) kws.last_speech_frame = kws.cmd_frames;

        int should = 0;
        if (kws.cmd_frames > 500) should = 1;
        else if (kws.cmd_frames > 50 &&
                 (kws.cmd_frames - kws.last_speech_frame) > 50) should = 1;

        if (should && kws.mel_frame_count > WINDOW_FRAMES) {
            int8_t mel_q[MEL_N_MELS * WINDOW_FRAMES];
            for (int i = 0; i < MEL_N_MELS * WINDOW_FRAMES; i++)
                mel_q[i] = (int8_t)(((float*)kws.mel_window)[i] * 16.0f);

            int32_t logits[STAGE2_NUM_TOKENS * STAGE2_MAX_T];
            int t_out = stage2_inference(mel_q, STAGE2_MAX_T, logits);
            int len = ctc_decode(logits, t_out, STAGE2_NUM_TOKENS,
                                 STAGE2_BLANK_ID,
                                 kws.command_text, sizeof(kws.command_text));

            kws.state = KWS_COMMAND;
            if (len > 0) {
                printf("[KWS] Command: %s\n", kws.command_text);
                return kws.command_id;
            }
        }
        return -2;

    case KWS_COMMAND:
        kws.state = KWS_IDLE;
        return -1;

    default:
        kws.state = KWS_IDLE;
        return -1;
    }
}

const char* kws_get_command_text(int cmd_id)
{
    (void)cmd_id;
    return kws.command_text[0] ? kws.command_text : "";
}

kws_state_t kws_get_state(void)
{
    return kws.state;
}

// ==================== 杰理 SDK 音频任务 ====================

static void kws_task(void *priv)
{
    u32 mic_len;
    int ret;

    kws_pipeline_init();

    while (1) {
        if (kws.exit_flag) break;
        if (!kws.run_flag) {
            os_sem_pend(&kws.sem, 0);
            continue;
        }
        if (cbuf_get_data_size(&kws.mic_cbuf) < ONCE_SR_POINTS * 2) {
            os_sem_pend(&kws.sem, 0);
            continue;
        }

        s16 pcm[ONCE_SR_POINTS];
        mic_len = cbuf_read(&kws.mic_cbuf, pcm, ONCE_SR_POINTS * 2);
        if (!mic_len) continue;

        ret = kws_pipeline_feed(pcm);
        if (ret >= 0) {
            // 识别到命令 → 发送事件给主任务
            printf("Command: %s (id=%d)\n", kws_get_command_text(ret), ret);
            // TODO: 通知主任务处理命令
        }
    }
}

// ==================== 音频 VFS 回调 ====================

static int kws_vfs_write(void *file, void *data, u32 len)
{
    cbuffer_t *cbuf = (cbuffer_t *)file;
    if (cbuf_write(cbuf, data, len) != len)
        cbuf_clear(&kws.mic_cbuf);
    os_sem_set(&kws.sem, 0);
    os_sem_post(&kws.sem);
    return len;
}

static int kws_vfs_close(void *file)
{
    return 0;
}

static const struct audio_vfs_ops kws_vfs_ops = {
    .fwrite = kws_vfs_write,
    .fclose = kws_vfs_close,
};

// ==================== 启动/停止 ====================

int kws_jl_start(u16 sample_rate)
{
    kws.exit_flag = 0;
    kws.mic_enc = server_open("audio_server", "enc");
    if (!kws.mic_enc) return -1;

    cbuf_init(&kws.mic_cbuf, kws.mic_buf, sizeof(kws.mic_buf));
    os_sem_create(&kws.sem, 0);
    kws.sample_rate = sample_rate;

    int pid = thread_fork("kws_task", 3, 4096, 0, &kws.pid, kws_task, NULL);

    // 启动音频采集
    kws.run_flag = 1;
    union audio_req req = {0};
    req.enc.cmd = AUDIO_ENC_OPEN;
    req.enc.channel = 1;
    req.enc.frame_size = ONCE_SR_POINTS * 2;
    req.enc.sample_rate = sample_rate;
    req.enc.format = "pcm";
    req.enc.sample_source = "mic";
    req.enc.vfs_ops = &kws_vfs_ops;
    req.enc.output_buf_len = req.enc.frame_size * 5;
    req.enc.file = (FILE *)&kws.mic_cbuf;
    server_request(kws.mic_enc, AUDIO_REQ_ENC, &req);

    return 0;
}

void kws_jl_stop(void)
{
    kws.run_flag = 0;
    kws.exit_flag = 1;
    os_sem_post(&kws.sem);
    if (kws.mic_enc) {
        server_close(kws.mic_enc);
        kws.mic_enc = NULL;
    }
    thread_kill(&kws.pid, KILL_WAIT);
}
