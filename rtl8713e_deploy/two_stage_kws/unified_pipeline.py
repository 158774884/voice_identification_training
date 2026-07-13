"""
两级语音系统统一流水线 + AC7916AB 部署适配

┌─────────────────────────────────────────────────┐
│                 AC7916AB 内部                     │
│                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  │
│  │ Stage1: WakeWord   │  │ Stage2: Command    │  │
│  │ (始终在线, CPU跑)   │  │ (唤醒后激活, MVA跑) │  │
│  │                    │  │                    │  │
│  │ UltraTinyKWS 2.4K  │  │ TinyKWS-MVA 780K   │  │
│  │ 34KB SRAM          │  │ 50KB SRAM + 762KB  │  │
│  │ <200us/帧          │  │ PSRAM              │  │
│  │ <0.5mA             │  │ <18ms/推理          │  │
│  └────────┬───────────┘  └────────┬───────────┘  │
│           │ 唤醒!                  │ 识别结果      │
│           ▼                        ▼              │
│  ┌─────────────────────────────────────────────┐  │
│  │           Pipeline State Machine            │  │
│  │  IDLE → LISTENING → WOKE → COMMAND → IDLE │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘

状态转换:
  IDLE ──(VAD)──→ LISTENING ──(wake detected)──→ WOKE
    ↑                                                 │
    │                                        ┌────────┘
    │                                        ▼
    └──(timeout/no cmd)── COMMAND ──(result)──┘
"""

import time
from enum import Enum
from typing import Optional, List, Tuple, Dict
import torch
import numpy as np


class PipelineState(Enum):
    IDLE = "idle"               # 待机, Stage1 运行
    LISTENING = "listening"      # 检测到语音, 等唤醒词
    WOKE = "woke"               # 已唤醒, 等 Stage2 识别
    COMMAND = "command"          # 识别完成, 输出结果
    TIMEOUT = "timeout"          # 超时, 回到 IDLE


class TwoStagePipeline:
    """
    两级语音唤醒+指令识别统一流水线

    使用方式:
        pipeline = TwoStagePipeline(wake_detector, cmd_recognizer)
        pipeline.start()

        # 每 10ms 喂一帧 PCM
        while audio_available:
            pcm_frame = get_next_frame()  # 160 samples @16kHz
            result = pipeline.feed(pcm_frame)
            if result:
                print(f"Command: {result}")

        pipeline.stop()
    """

    def __init__(self,
                 wake_detector,
                 cmd_recognizer,
                 mel_extractor=None,
                 cmd_timeout_ms: int = 3000,      # 唤醒后 3 秒内说出命令
                 max_cmd_duration_ms: int = 5000,  # 单次命令最长 5 秒
                 min_silence_ms: int = 500,        # 命令间最少静音 500ms
                 ):
        self.wake = wake_detector
        self.cmd = cmd_recognizer
        self.mel_extractor = mel_extractor

        self.cmd_timeout_ms = cmd_timeout_ms
        self.max_cmd_duration_ms = max_cmd_duration_ms
        self.min_silence_ms = min_silence_ms

        # 状态
        self.state = PipelineState.IDLE
        self.state_start_time = 0
        self.wake_label = None
        self.wake_confidence = 0.0

        # 命令音频缓冲
        self._cmd_audio_buffer = []  # [(mel_frame), ...]
        self._cmd_start_time = 0
        self._last_speech_time = 0
        self._total_frames = 0

        # 回调
        self.on_wake = None     # callback(wake_label, confidence)
        self.on_command = None  # callback(command_text, confidence)
        self.on_state_change = None  # callback(old_state, new_state)

    def start(self):
        """启动流水线 (IDLE 状态, Stage1 开始运行)"""
        self._transition(PipelineState.IDLE)
        self.wake.reset()
        self._cmd_audio_buffer = []
        self._total_frames = 0

    def stop(self):
        """停止流水线"""
        self._transition(PipelineState.IDLE)

    def feed(self, mel_frame: np.ndarray) -> Optional[Dict]:
        """
        每 10ms 喂入一帧 Mel 特征

        Args:
            mel_frame: [40] Mel 特征向量 (一帧)

        Returns:
            None: 无结果
            {'type': 'wake', 'label': ..., 'confidence': ...}: 唤醒
            {'type': 'command', 'text': ..., 'confidence': ..., 'alt': [...]}: 识别
        """
        self._total_frames += 1
        mel_tensor = torch.FloatTensor(mel_frame).unsqueeze(0)  # [1, 40]

        if self.state in (PipelineState.IDLE, PipelineState.LISTENING):
            # Stage 1: 检测唤醒词
            is_wake, label, conf = self.wake.process_frame(mel_tensor)

            if self.wake.state == 'SPEECH' and self.state == PipelineState.IDLE:
                self._transition(PipelineState.LISTENING)

            if is_wake:
                self._transition(PipelineState.WOKE)
                self.wake_label = label
                self.wake_confidence = conf
                self._cmd_audio_buffer = []
                self._cmd_start_time = self._total_frames
                self._last_speech_time = self._total_frames

                if self.on_wake:
                    self.on_wake(label, conf)

                return {'type': 'wake', 'label': label, 'confidence': conf}

        elif self.state == PipelineState.WOKE:
            # Stage 2: 收集命令音频 + 执行识别
            self._cmd_audio_buffer.append(mel_frame)

            # 检测静音 (能量)
            energy = np.mean(mel_frame ** 2)
            if energy > 0.001:
                self._last_speech_time = self._total_frames

            elapsed_ms = (self._total_frames - self._cmd_start_time) * 10
            silence_ms = (self._total_frames - self._last_speech_time) * 10

            # 触发识别: (有语音 + 足够时长) 或 (超时) 或 (静音足够长)
            should_recognize = False

            if elapsed_ms > self.max_cmd_duration_ms:
                should_recognize = True  # 超时, 强制识别
            elif len(self._cmd_audio_buffer) >= 49 and silence_ms > self.min_silence_ms:
                should_recognize = True  # 说了话 + 静音间隔, 触发识别
            elif elapsed_ms > self.cmd_timeout_ms:
                # 超时无命令 → 回到 IDLE
                self._transition(PipelineState.TIMEOUT)
                self._transition(PipelineState.IDLE)
                self._cmd_audio_buffer = []
                return {'type': 'timeout'}

            if should_recognize and len(self._cmd_audio_buffer) >= 49:
                # 执行 Stage 2 识别
                mel_stack = np.stack(self._cmd_audio_buffer, axis=1)  # [40, T]
                mel_tensor = torch.FloatTensor(mel_stack).unsqueeze(0)  # [1, 40, T]

                # 长度对齐 (截断或填充到 98 帧)
                if mel_tensor.size(2) > 98:
                    mel_tensor = mel_tensor[:, :, :98]
                elif mel_tensor.size(2) < 98:
                    pad_w = 98 - mel_tensor.size(2)
                    mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_w))

                results = self.cmd.recognize(mel_tensor, top_k=3)

                self._transition(PipelineState.COMMAND)

                output = {
                    'type': 'command',
                    'text': results[0][0],
                    'confidence': results[0][1],
                    'alternatives': [(t, c) for t, c in results[1:]],
                }

                if self.on_command:
                    self.on_command(results[0][0], results[0][1])

                # 识别完成 → 回到 IDLE
                self._transition(PipelineState.IDLE)
                self._cmd_audio_buffer = []

                return output

        elif self.state == PipelineState.TIMEOUT:
            # 回到 IDLE
            self._transition(PipelineState.IDLE)
            self._cmd_audio_buffer = []

        return None

    def _transition(self, new_state: PipelineState):
        old = self.state
        self.state = new_state
        self.state_start_time = self._total_frames

        if self.on_state_change and old != new_state:
            self.on_state_change(old, new_state)

    def get_state(self) -> PipelineState:
        return self.state

    def get_stats(self) -> dict:
        """获取系统资源占用统计"""
        s1 = self.wake.get_model_size()
        s2 = self.cmd.get_stats()

        return {
            'stage1': {
                'params': s1['params'],
                'sram_kb': 34,       # weights + activation + buffers
                'cpu_usage_pct': 2.0,  # CPU usage %
                'power_ma': 0.5,       # always-on current
            },
            'stage2': {
                'params': s2['params'],
                'sram_kb': 55,         # activation only
                'psram_kb': s2['int8_kb'],  # weights in PSRAM
                'power_ma': 10,        # active current
                'duration_ms_per_cmd': 3000,
            },
            'total_sram_kb': 89,       # peak (34 + 55)
            'avg_power_ma': 1.5,       # typical daily average
        }
