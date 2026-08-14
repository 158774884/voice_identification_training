"""
Audio waveform display widget using matplotlib embedded in Qt.
"""
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class WaveformCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for waveform rendering."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 1.5), dpi=100, facecolor="#f8f9fa")
        self.ax = self.fig.add_axes([0.02, 0.1, 0.96, 0.82])
        self.ax.set_facecolor("#f8f9fa")
        self.ax.tick_params(labelsize=7, colors="#5f6368")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.spines["bottom"].set_color("#d0d5dd")
        self.ax.xaxis.set_tick_params(width=0)
        self.ax.yaxis.set_ticks([])

        super().__init__(self.fig)
        self.setParent(parent)
        self._line = None
        self._data = np.zeros(1000)

    def plot_waveform(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Render waveform from numpy audio array.

        Args:
            audio_data: 1D float32 audio samples
            sample_rate: Sample rate in Hz
        """
        self.ax.clear()
        self.ax.set_facecolor("#f8f9fa")
        self.ax.tick_params(labelsize=7, colors="#5f6368")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.spines["bottom"].set_color("#d0d5dd")
        self.ax.yaxis.set_ticks([])

        if len(audio_data) == 0:
            self.draw()
            return

        # Downsample for display if too many points
        max_points = 10000
        if len(audio_data) > max_points:
            step = len(audio_data) // max_points
            audio_data = audio_data[::step]

        # Time axis in seconds
        duration = len(audio_data) / sample_rate
        time_axis = np.linspace(0, duration, len(audio_data))

        self.ax.plot(time_axis, audio_data, color="#1a73e8", linewidth=0.8)
        self.ax.set_xlabel("时间 (秒)", fontsize=8, color="#9aa0a6")
        self.ax.set_xlim(0, duration)

        # Set y limits with some padding
        peak = max(abs(audio_data.max()), abs(audio_data.min())) or 1.0
        self.ax.set_ylim(-peak * 1.1, peak * 1.1)

        self.fig.tight_layout()
        self.draw()

    def clear(self):
        """Clear the waveform display."""
        self.ax.clear()
        self.draw()


class AudioWaveformWidget(QWidget):
    """Widget containing waveform canvas with label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = WaveformCanvas(self)
        layout.addWidget(self.canvas)

    def load_audio(self, filepath: str):
        """Load and display audio file waveform."""
        try:
            audio, sr = self._read_audio(filepath)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # mono
            self.canvas.plot_waveform(audio.astype(np.float32), sr)
        except Exception:
            self.canvas.clear()

    @staticmethod
    def _read_audio(filepath: str):
        """Read audio with multi-backend fallback."""
        # 1) soundfile
        try:
            import soundfile as sf
            return sf.read(filepath)
        except Exception:
            pass
        # 2) scipy
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(filepath)
            return audio, sr
        except Exception:
            pass
        # 3) built-in wave
        import wave
        import numpy as np
        with wave.open(filepath, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
            return audio.astype(np.float32) / 32768.0, sr

    def load_array(self, audio: np.ndarray, sample_rate: int = 16000):
        """Display waveform from numpy array."""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self.canvas.plot_waveform(audio.astype(np.float32), sample_rate)

    def clear(self):
        self.canvas.clear()
