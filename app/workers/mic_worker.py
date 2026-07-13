"""
MicWorker — captures real-time audio from the default microphone using sounddevice.
Runs in a QThread, emits audio chunks via signals.
"""
import numpy as np
import threading
from collections import deque

from PySide6.QtCore import QThread, Signal


class MicWorker(QThread):
    """Captures microphone audio in a background thread."""

    # Signals
    audio_chunk = Signal(np.ndarray)  # float32 audio chunk [samples]
    level_meter = Signal(float)       # RMS level (0.0 - 1.0)
    mic_error = Signal(str)           # error message
    mic_started = Signal()
    mic_stopped = Signal()

    def __init__(self, sample_rate: int = 16000,
                 block_size: int = 1600,  # 100ms @ 16kHz
                 parent=None):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._cancel_event = threading.Event()
        self._stream = None

    def run(self):
        try:
            import sounddevice as sd

            # List available devices for debugging
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.block_size,
                callback=self._audio_callback,
            )

            self._stream.start()
            self.mic_started.emit()

            # Keep thread alive while streaming
            while not self._cancel_event.is_set():
                self._cancel_event.wait(0.05)

            self._stream.stop()
            self._stream.close()
            self._stream = None
            self.mic_stopped.emit()

        except Exception as e:
            self.mic_error.emit(f"麦克风初始化失败: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if self._cancel_event.is_set():
            return

        if status:
            # Log warning but continue
            pass

        # Copy data (it may be reused after callback returns)
        chunk = indata.copy().flatten()

        # Calculate RMS level
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        # Normalize to 0-1 range (typical speech ~0.01-0.2)
        normalized_level = min(1.0, rms * 5.0)

        self.audio_chunk.emit(chunk)
        self.level_meter.emit(normalized_level)

    def stop(self):
        """Signal the worker to stop capturing."""
        self._cancel_event.set()

    def cancel(self):
        """Alias for stop."""
        self.stop()
