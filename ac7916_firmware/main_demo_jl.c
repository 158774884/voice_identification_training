/**
 * main_demo_jl.c — AC7916AB 语音唤醒+命令识别 Demo
 *
 * 集成到杰理 WL82 SDK:
 *   1. 复制本文件到 SDK/apps/common/asr/jlkws/ (或新建目录)
 *   2. 复制 ac7916_firmware/*.h 到同目录
 *   3. 复制 kws_pipeline_jl.c 到同目录
 *   4. 在 app_config.h 添加:
 *      #define CONFIG_ASR_ALGORITHM
 *      #define CONFIG_CUSTOM_KWS_ENABLE  1
 *   5. 在 Makefile 添加编译
 *
 * 运行:
 *   上电 → 说 "小倍小倍" → 听到"滴" → 说命令 → 执行
 */

#include "server/audio_server.h"
#include "server/server_core.h"
#include "os/os_api.h"
#include "event.h"
#include "app_config.h"
#include "event/key_event.h"

#include "kws_pipeline.h"
#include "grammar.h"

// ==================== 命令处理 ====================

static void on_command(const char *text, int confidence_pct)
{
    printf("\n========================================\n");
    printf("  Command: %s (confidence: %d%%)\n", text, confidence_pct);
    printf("========================================\n\n");

    // 根据命令文本执行业务逻辑
    // 192 条命令的完整映射见 commands.txt
    if (strstr(text, "关机") || strstr(text, "停止运行") || strstr(text, "关闭设备")) {
        printf("  -> Power off\n");
        // system_power_off();
    } else if (strstr(text, "打开") && strstr(text, "灯")) {
        printf("  -> Light on\n");
        // gpio_light_on();
    } else if (strstr(text, "关闭") && strstr(text, "灯")) {
        printf("  -> Light off\n");
        // gpio_light_off();
    } else if (strstr(text, "温度") && (strstr(text, "高") || strstr(text, "升"))) {
        printf("  -> Temperature up\n");
    } else if (strstr(text, "温度") && (strstr(text, "低") || strstr(text, "降"))) {
        printf("  -> Temperature down\n");
    } else if (strstr(text, "模式")) {
        printf("  -> Switch mode\n");
    } else if (strstr(text, "音量") || strstr(text, "声音")) {
        printf("  -> Volume control\n");
    } else if (strstr(text, "力度")) {
        printf("  -> Intensity control\n");
    } else if (strstr(text, "眼保健操")) {
        printf("  -> Eye exercise mode\n");
    } else if (strstr(text, "舒缓养眼")) {
        printf("  -> Relax mode\n");
    } else if (strstr(text, "活力护眼")) {
        printf("  -> Active mode\n");
    } else if (strstr(text, "睡眠") || strstr(text, "助眠")) {
        printf("  -> Sleep mode\n");
    } else if (strstr(text, "热敷") || strstr(text, "加热")) {
        printf("  -> Heat control\n");
    } else if (strstr(text, "WiFi") || strstr(text, "联网") || strstr(text, "网络")) {
        printf("  -> Network control\n");
    } else if (strstr(text, "静音")) {
        printf("  -> Mute\n");
    } else {
        printf("  -> Unknown command, forwarding...\n");
    }
}

// ==================== 音频回调 ====================

static void audio_vad_callback(int event, void *arg)
{
    switch (event) {
    case AUDIO_SERVER_EVENT_SPEAK_START:
        printf("[VAD] Speech start\n");
        break;
    case AUDIO_SERVER_EVENT_SPEAK_STOP:
        printf("[VAD] Speech stop\n");
        break;
    default:
        break;
    }
}

// ==================== 主任务 ====================

// 声明 kws_pipeline_jl.c 中的函数
extern int  kws_jl_start(u16 sample_rate);
extern void kws_jl_stop(void);
extern int  kws_pipeline_feed(const int16_t *pcm);
extern const char* kws_get_command_text(int cmd_id);

static void voice_main_task(void *priv)
{
    printf("\n");
    printf("  ====================================\n");
    printf("    AC7916AB Voice KWS Demo\n");
    printf("    Chip:  WL82 (Dual RISC 320MHz)\n");
    printf("    Wake:  'xiao bei xiao bei'\n");
    printf("    Cmds:  192\n");
    printf("    Model: 3.8K + 285K (CPU only)\n");
    printf("    Flash: ~314 KB / 8 MB\n");
    printf("  ====================================\n\n");

    // 启动语音引擎
    int ret = kws_jl_start(16000);
    if (ret != 0) {
        printf("[Error] KWS init failed\n");
        return;
    }
    printf("[Demo] Listening for wake word...\n");

    // 主循环
    while (1) {
        os_time_dly(10);  // 100ms

        // 检查状态
        kws_state_t st = kws_get_state();
        static kws_state_t last_st = KWS_IDLE;
        if (st != last_st) {
            printf("[State] %d -> %d\n", last_st, st);
            last_st = st;

            if (st == KWS_WOKE) {
                printf("  (beep) Say your command...\n");
            }
        }
    }
}

// ==================== 应用入口 ====================

void app_main(void)
{
    thread_fork("voice_main", 4, 2048, 0, NULL, voice_main_task, NULL);
}
