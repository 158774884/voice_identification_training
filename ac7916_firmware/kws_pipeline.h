// Two-Stage KWS Pipeline for AC7916AB
// Stage1: always-on 唤醒词检测
// Stage2: on-demand 命令词识别

#ifndef KWS_PIPELINE_H
#define KWS_PIPELINE_H

#include "stage1_model.h"
#include "stage2_model.h"
#include "grammar.h"
#include "mel_config.h"

typedef enum { KWS_IDLE, KWS_LISTENING, KWS_WOKE, KWS_COMMAND } kws_state_t;

#define KWS_N_WAKE_CLASSES 2
#define KWS_N_TOKENS 149

// Forward declarations
void kws_pipeline_init(void);
int  kws_pipeline_feed(const int16_t *pcm_10ms);  // returns cmd_id or -1
const char* kws_get_command_text(int cmd_id);
kws_state_t kws_get_state(void);

#endif
