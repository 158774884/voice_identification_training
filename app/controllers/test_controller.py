"""
Test Controller — manages real-time and batch voice testing workflows.
Coordinates MicWorker and InferenceWorker.
"""
import os
import numpy as np
from collections import deque
from typing import List, Optional, Dict

from PySide6.QtCore import QObject, Signal, Slot

from app.workers.mic_worker import MicWorker
from app.workers.inference_worker import InferenceWorker
from app.utils.logger import LogManager


class TestController(QObject):
    """Controls voice recognition testing workflows."""

    # Signals for UI updates
    transcription_update = Signal(str)      # real-time transcript text
    dialect_update = Signal(str, float)     # dialect name, confidence
    level_update = Signal(float)            # audio level 0-1
    mic_status_changed = Signal(bool)       # True=recording, False=stopped
    batch_progress = Signal(int, int)       # current, total
    batch_result = Signal(dict)             # single result
    batch_complete = Signal(dict)           # summary
    model_loaded = Signal(str)              # model name
    model_error = Signal(str)               # error
    recording_saved = Signal(str)           # path to saved recording

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = LogManager()
        self._mic_worker: Optional[MicWorker] = None
        self._inference_worker: Optional[InferenceWorker] = None
        self._audio_buffer: deque = deque(maxlen=30)  # ~3 seconds of 100ms chunks
        self._is_recording = False
        self._recording_data: List[np.ndarray] = []
        self._transcript_history: List[dict] = []

    # ================================================================
    # Model Management
    # ================================================================

    def load_model(self, checkpoint_path: str, device: str = 'cpu'):
        """Load a model checkpoint for inference."""
        if not os.path.exists(checkpoint_path):
            self.model_error.emit(f"模型文件不存在: {checkpoint_path}")
            return

        self._inference_worker = InferenceWorker()
        self._inference_worker.load_model(checkpoint_path, device)
        self._inference_worker.model_loaded.connect(self._on_model_loaded)
        self._inference_worker.model_load_failed.connect(self.model_error)
        self._inference_worker.start()
        self._log.info("测试", f"正在加载模型: {os.path.basename(checkpoint_path)}")

    @Slot(str)
    def _on_model_loaded(self, name: str):
        self._log.info("测试", f"模型已加载: {name}")
        self.model_loaded.emit(name)

    # ================================================================
    # Real-time Mic Testing
    # ================================================================

    def start_mic_test(self):
        """Begin real-time microphone capture and inference."""
        if self._is_recording:
            return

        self._audio_buffer.clear()
        self._recording_data.clear()

        self._mic_worker = MicWorker(sample_rate=16000, block_size=1600)
        self._mic_worker.audio_chunk.connect(self._on_audio_chunk)
        self._mic_worker.level_meter.connect(self.level_update)
        self._mic_worker.mic_error.connect(self.model_error)
        self._mic_worker.mic_started.connect(lambda: self.mic_status_changed.emit(True))
        self._mic_worker.mic_stopped.connect(lambda: self.mic_status_changed.emit(False))
        self._mic_worker.start()

        self._is_recording = True
        self._log.info("测试", "麦克风测试已启动")

    def stop_mic_test(self):
        """Stop microphone capture."""
        if self._mic_worker:
            self._mic_worker.stop()
            self._mic_worker.wait(2000)
            self._mic_worker = None

        self._is_recording = False
        self._log.info("测试", "麦克风测试已停止")

    @Slot(np.ndarray)
    def _on_audio_chunk(self, chunk: np.ndarray):
        """Process incoming audio chunk."""
        self._audio_buffer.append(chunk)
        self._recording_data.append(chunk)

        # Run inference on accumulated buffer every ~500ms (5 chunks)
        if len(self._audio_buffer) >= 5 and self._inference_worker:
            buffer = np.concatenate(list(self._audio_buffer))
            try:
                result = self._inference_worker.infer_chunk(buffer)
                if result.get("status") == "success":
                    text = result.get("asr_text", "")
                    if text:
                        self.transcription_update.emit(text)
                    dialect = result.get("dialect", "")
                    confidence = result.get("dialect_confidence", 0.0)
                    if dialect:
                        self.dialect_update.emit(dialect, confidence)

                    # Store in history
                    self._transcript_history.append({
                        "text": text,
                        "dialect": dialect,
                        "confidence": confidence,
                    })
            except Exception:
                pass  # Inference may fail if model not loaded; skip silently

    def save_recording(self, output_path: str):
        """Save captured audio to WAV file."""
        if not self._recording_data:
            self._log.warning("测试", "没有录音数据可保存")
            return

        try:
            import soundfile as sf
            audio = np.concatenate(self._recording_data)
            sf.write(output_path, audio, 16000)
            self._log.info("测试", f"录音已保存: {output_path}")
            self.recording_saved.emit(output_path)
        except Exception as e:
            self.model_error.emit(f"保存录音失败: {e}")

    # ================================================================
    # Batch Testing
    # ================================================================

    def run_batch_test(self, file_list: List[str],
                       ground_truth: Optional[Dict[str, str]] = None):
        """Run inference on a batch of audio files."""
        if not self._inference_worker:
            self.model_error.emit("请先加载模型")
            return

        self._log.info("测试", f"开始批量测试: {len(file_list)} 个文件")
        # Run in the inference worker thread
        self._inference_worker.run_batch(file_list, ground_truth)

    # ================================================================
    # History
    # ================================================================

    def clear_history(self):
        self._transcript_history.clear()

    @property
    def history(self) -> List[dict]:
        return list(self._transcript_history)
