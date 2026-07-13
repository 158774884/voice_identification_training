// Two-Stage KWS Pipeline for AC7916AB
// Stage1: CPU @ 320MHz, always-on
// Stage2: MVA @ 360MHz, on-demand

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
