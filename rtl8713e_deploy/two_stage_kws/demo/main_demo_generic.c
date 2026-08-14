/**
 * main_demo_generic.c — 通用 C 平台移植 Demo（无芯片 SDK 依赖）
 *
 * 用途：在 PC 或任意 MCU/DSP 上验证 / 移植两阶段 KWS 流水线。
 * 它从 16kHz / 16bit / 单声道 WAV 文件读取 PCM，按 10ms 帧喂给
 * kws_pipeline_feed()，打印识别结果。
 *
 * 编译（PC 上测试，与 kws_pipeline.c 一起，.h 文件在同目录）:
 *   gcc main_demo_generic.c kws_pipeline.c -lm -o kws_demo
 *
 * 注意：
 *   - kws_pipeline.c 里的推理核心 (stage1/stage2_inference) 是占位实现
 *     （标注 TODO），移植到具体芯片时请替换为硬件加速 API（如 JL_MVA_*）。
 *   - FFT 为纯 C 参考实现，实际部署请替换为芯片硬件 FFT。
 */

#include "kws_pipeline.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/* ---- 极简 WAV 解析（16bit PCM）---- */
typedef struct {
    int16_t *data;
    int n_samples;
} wav_t;

static int load_wav(const char *path, wav_t *out)
{
    FILE *f = fopen(path, "rb");
    unsigned char header[44];
    if (!f) return -1;
    if (fread(header, 1, 44, f) != 44) { fclose(f); return -1; }

    /* 简化处理：假定 16bit PCM（实际项目请按 header 字段解析 channels/sample_rate） */
    int channels = header[22] | (header[23] << 8);
    int bits     = header[34] | (header[35] << 8);
    if (channels != 1 || bits != 16) {
        printf("[WARN] 假定 16bit 单声道，实际 channels=%d bits=%d\n", channels, bits);
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f) - 44;
    fseek(f, 44, SEEK_SET);

    out->n_samples = (int)(size / 2);
    out->data = (int16_t *)malloc((size_t)(size > 0 ? size : 1));
    if (!out->data) { fclose(f); return -1; }
    if (fread(out->data, 1, (size_t)size, f) != (size_t)size) { /* 容忍短读 */ }
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    const char *wav_path = (argc > 1) ? argv[1] : "test.wav";
    wav_t wav;

    if (load_wav(wav_path, &wav) != 0) {
        printf("用法: %s <16k_16bit_mono.wav>\n", argv[0]);
        printf("无法读取 WAV: %s\n", wav_path);
        return 1;
    }

    printf("=== 通用 C KWS Demo ===\n");
    printf("音频: %s (%d samples, %.2fs)\n",
           wav_path, wav.n_samples, wav.n_samples / 16000.0);

    kws_pipeline_init();

    int16_t frame[160];   /* 10ms @ 16kHz */
    int pos = 0;
    int frame_idx = 0;

    while (pos + 160 <= wav.n_samples) {
        for (int i = 0; i < 160; i++)
            frame[i] = wav.data[pos + i];
        pos += 160;

        int result = kws_pipeline_feed(frame);
        if (result >= 0) {
            const char *text = kws_get_command_text(result);
            printf("[KWS] 识别到命令: %s (id=%d)\n", text, result);
        }
        frame_idx++;
    }

    printf("处理完成（%d 帧）\n", frame_idx);
    free(wav.data);
    return 0;
}
