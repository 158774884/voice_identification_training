/**
 * AC7916AB 两级语音唤醒+命令识别 — Demo 示例
 *
 * 复制到 AC7916 SDK: project/.../src/main.c
 *
 * 硬件连接:
 *   - PDM DMIC → GPIO_PDM_CLK / GPIO_PDM_DATA
 *   - 或 I2S MIC → GPIO_I2S_*
 *   - 状态 LED → GPIO_PA0 (可选)
 *
 * 编译: 用 Cadence XCC / Realtek Ameba SDK Makefile
 */

#include "kws_pipeline.h"   // 我们的模型
#include "ameba_audio.h"      // AC7916 SDK 音频驱动 (示例)
#include "ameba_gpio.h"       // GPIO 驱动
#include "ameba_wifi.h"       // WiFi 驱动
#include "ameba_bt.h"         // 蓝牙驱动
#include "os_task.h"          // FreeRTOS 任务

// ============== 业务逻辑 ==============

#define LED_PIN      GPIO_PA0
#define LED_ON()     gpio_write(LED_PIN, 1)
#define LED_OFF()    gpio_write(LED_PIN, 0)

// 命令ID → 执行函数映射 (根据你的 commands.txt 定义)
typedef void (*cmd_handler_t)(void);

// 示例: 命令处理
static void cmd_open_living_light(void)   { /* 打开客厅灯 */ }
static void cmd_close_living_light(void)  { /* 关闭客厅灯 */ }
static void cmd_open_ac(void)             { /* 打开空调 */ }
static void cmd_close_ac(void)            { /* 关闭空调 */ }
static void cmd_temp_up(void)             { /* 温度调高 */ }
static void cmd_temp_down(void)           { /* 温度调低 */ }
static void cmd_mode_switch(void)         { /* 切换模式 */ }
static void cmd_power_off(void)           { /* 关机 */ }
static void cmd_start_massage(void)       { /* 开始按摩 */ }
static void cmd_query_status(void)        { /* 查询状态 */ }

// 命令映射表
typedef struct {
    const char *text;
    cmd_handler_t handler;
} cmd_map_t;

static const cmd_map_t cmd_map[] = {
    {"打开客厅的灯",  cmd_open_living_light},
    {"关闭客厅的灯",  cmd_close_living_light},
    {"打开空调",      cmd_open_ac},
    {"关闭空调",      cmd_close_ac},
    {"温度调高",      cmd_temp_up},
    {"温度调低",      cmd_temp_down},
    {"切换模式",      cmd_mode_switch},
    {"关机",          cmd_power_off},
    {"开始按摩",      cmd_start_massage},
    {"查询状态",      cmd_query_status},
    // ... 补充所有命令
    {NULL, NULL},
};

// ============== 音频回调 (DMA 中断, 每 10ms 触发) ==============

static void audio_frame_callback(const int16_t *pcm_data, uint32_t frame_len)
{
    // frame_len = 160 samples (10ms @ 16kHz)
    int result = kws_pipeline_feed(pcm_data);

    if (result >= 0) {
        // 识别到命令!
        const char *cmd_text = kws_get_command_text(result);
        kws_state_t state = kws_get_state();

        printf("[KWS] Command: %s (id=%d)\n", cmd_text, result);

        // 执行命令
        for (const cmd_map_t *cm = cmd_map; cm->text; cm++) {
            if (strcmp(cm->text, cmd_text) == 0) {
                cm->handler();
                break;
            }
        }
    }

    // 状态指示灯
    kws_state_t state = kws_get_state();
    switch (state) {
    case KWS_IDLE:
        LED_OFF();
        break;
    case KWS_LISTENING:
        // 呼吸灯 (简化: 闪烁)
        static int led_tick = 0;
        if (++led_tick % 50 == 0) gpio_toggle(LED_PIN);
        break;
    case KWS_WOKE:
        LED_ON();  // 唤醒后常亮
        break;
    case KWS_COMMAND:
        LED_OFF();
        break;
    }
}

// ============== 主任务 ==============

static void voice_task(void *arg)
{
    printf("[Demo] Initializing KWS pipeline...\n");

    // 1. 初始化 KWS 引擎 (DSP 侧加载模型权重)
    kws_pipeline_init();
    printf("[Demo] KWS ready. Listening for 'xiao bei xiao bei'...\n");

    // 2. 配置音频输入
    audio_config_t audio_cfg = {
        .sample_rate   = 16000,
        .channels      = 1,
        .bit_depth     = 16,
        .chunk_samples = 160,         // 10ms chunk
        .input         = AUDIO_IN_PDM, // PDM 数字麦
        .callback      = audio_frame_callback,
    };
    audio_init(&audio_cfg);
    audio_start();

    // 3. 主循环: 处理其他业务
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(100));

        // 可选: 检查是否需要升级模型
        // 可选: 上报状态到云端
        // 可选: 播放 TTS 反馈
    }
}

// ============== 应用入口 ==============

void app_main(void)
{
    printf("\n");
    printf("========================================\n");
    printf("  AC7916AB Voice KWS Demo\n");
    printf("  Wake word: 'xiao bei xiao bei'\n");
    printf("  Commands:  192\n");
    printf("  Model:     285K params\n");
    printf("  Flash:     314 KB / 8 MB\n");
    printf("========================================\n\n");

    // 初始化外设
    gpio_init(LED_PIN, GPIO_OUTPUT);
    LED_OFF();

    // 可选: 初始化 WiFi / BLE
    // wifi_init();
    // bt_init();

    // 创建语音任务 (栈 4KB)
    xTaskCreate(voice_task, "voice", 4096, NULL, 5, NULL);

    vTaskStartScheduler();
}
