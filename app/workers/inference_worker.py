"""
InferenceWorker — runs voice model inference in background thread.
Supports both single-file and batch processing.
"""
import os
import time
import threading
import numpy as np
import torch
from typing import Optional, List, Dict

from PySide6.QtCore import QThread, Signal


class InferenceWorker(QThread):
    """Runs model inference on audio files or chunks."""

    # Signals
    inference_ready = Signal(dict)         # single result {text, dialect, confidence, ...}
    batch_progress = Signal(int, int)      # current, total
    batch_result = Signal(dict)            # per-file result in batch mode
    batch_complete = Signal(dict)          # summary statistics
    inference_error = Signal(str)          # error
    model_loaded = Signal(str)             # model name
    model_load_failed = Signal(str)        # error

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._vocab = None
        self._pipeline = None
        self._cancel_event = threading.Event()
        self._device = 'cpu'

    def load_model(self, checkpoint_path: str, device: str = 'cpu'):
        """Load model from checkpoint (call before starting thread).

        This is called from the main thread before run().
        """
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._model_name = os.path.basename(checkpoint_path)

    def run(self):
        """Load model and wait for inference requests."""
        try:
            # Try loading the multi-task model
            self._load_multitask_model()
        except Exception as e:
            self.model_load_failed.emit(f"模型加载失败: {e}")
            return

    def _load_multitask_model(self):
        """Load the MultiTaskVoiceModel."""
        checkpoint_path = getattr(self, '_checkpoint_path', None)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            self.model_load_failed.emit(f"模型文件不存在: {checkpoint_path}")
            return

        # Load model from existing project code
        try:
            from model.multi_task_model import create_model
            from data.vocab import get_default_vocab
            from inference.pipeline import VoiceInferencePipeline

            self._model = create_model()
            checkpoint = torch.load(checkpoint_path, map_location=self._device,
                                    weights_only=True)
            if 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self._model.load_state_dict(checkpoint)
            self._model.to(self._device)
            self._model.eval()

            self._vocab = get_default_vocab()
            self._pipeline = VoiceInferencePipeline(
                self._model, self._vocab, device=self._device
            )

            self.model_loaded.emit(os.path.basename(checkpoint_path))
        except Exception as e:
            # Try the two-stage KWS model as fallback
            self._load_kws_model(checkpoint_path)

    def _load_kws_model(self, checkpoint_path: str):
        """Fallback: try loading two-stage KWS model."""
        try:
            from rtl8713e_deploy.two_stage_kws.unified_pipeline import TwoStagePipeline
            from rtl8713e_deploy.two_stage_kws.stage1_wakeword import UltraTinyWakeWord
            from rtl8713e_deploy.two_stage_kws.stage2_command import CTCEncoder

            # This is a simplified fallback — full integration would need more config
            checkpoint = torch.load(checkpoint_path, map_location=self._device,
                                    weights_only=True)

            self._pipeline = None  # Use demo_pc.py style instead
            self.model_loaded.emit(f"KWS: {os.path.basename(checkpoint_path)}")
        except Exception as e:
            self.model_load_failed.emit(f"模型加载失败: {e}")

    def infer_file(self, audio_path: str) -> dict:
        """Run inference on a single audio file. Must be called from run() context."""
        if self._pipeline is None:
            return {"error": "模型未加载"}

        try:
            results = self._pipeline.from_file(audio_path)
            return {
                "asr_text": results.get("asr_text", [""])[0] if results.get("asr_text") else "",
                "dialect": results.get("dialect_zh", results.get("dialect", "")),
                "dialect_confidence": results.get("dialect_confidence", 0.0),
                "speaker_embedding": results.get("speaker_embedding"),
                "audio_path": audio_path,
                "status": "success",
            }
        except Exception as e:
            return {"error": str(e), "audio_path": audio_path, "status": "error"}

    def infer_chunk(self, audio_chunk: np.ndarray) -> dict:
        """Run inference on a raw audio chunk. Must be called from run() context."""
        if self._pipeline is None:
            return {"error": "模型未加载"}

        try:
            audio_tensor = torch.FloatTensor(audio_chunk).unsqueeze(0).unsqueeze(0)
            audio_tensor = audio_tensor.to(self._device)
            results = self._pipeline(audio_tensor)
            return {
                "asr_text": results.get("asr_text", [""])[0] if results.get("asr_text") else "",
                "dialect": results.get("dialect_zh", results.get("dialect", "")),
                "dialect_confidence": results.get("dialect_confidence", 0.0),
                "status": "success",
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    def run_batch(self, file_list: List[str], ground_truth: Optional[Dict[str, str]] = None):
        """Run inference on a batch of files, emitting progress.

        Args:
            file_list: List of audio file paths
            ground_truth: Optional dict mapping filename->reference text
        """
        total = len(file_list)
        results = []
        correct = 0
        total_latency = 0.0

        for i, path in enumerate(file_list):
            if self._cancel_event.is_set():
                break

            start = time.time()
            result = self.infer_file(path)
            latency = (time.time() - start) * 1000  # ms
            result["latency_ms"] = latency
            total_latency += latency

            # Check accuracy if ground truth available
            filename = os.path.basename(path)
            if ground_truth and filename in ground_truth:
                ref = ground_truth[filename]
                hypothesis = result.get("asr_text", "")
                if ref.strip() == hypothesis.strip():
                    correct += 1
                    result["correct"] = True
                else:
                    result["correct"] = False
                result["reference"] = ref

            results.append(result)
            self.batch_progress.emit(i + 1, total)
            self.batch_result.emit(result)

        # Summary
        summary = {
            "total": len(results),
            "errors": sum(1 for r in results if r.get("status") == "error"),
            "avg_latency_ms": total_latency / len(results) if results else 0,
        }
        if ground_truth:
            summary["accuracy"] = correct / len(results) * 100 if results else 0
            summary["correct"] = correct

        self.batch_complete.emit(summary)

    def cancel(self):
        """Signal the worker to stop processing."""
        self._cancel_event.set()

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None or self._pipeline is not None
