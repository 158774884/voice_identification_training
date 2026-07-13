"""
Chip Controller — manages chip database and runs compatibility assessments.
Reuses existing evaluation logic from ac7916_feasibility.py and deployment/quantization.py.
"""
import os
import sys
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal, Slot

from app.models.chip_database import ChipDatabase, ChipSpec
from app.utils.logger import LogManager


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class AssessmentResult:
    """Result of a chip-model compatibility assessment."""

    chip_name: str = ""
    model_name: str = ""

    # Dimension scores (0-100)
    inference_score: float = 0.0
    memory_score: float = 0.0
    flash_score: float = 0.0
    compute_score: float = 0.0
    power_score: float = 0.0
    arch_score: float = 0.0

    # Details
    inference_time_ms: float = 0.0
    peak_ram_kb: float = 0.0
    model_size_kb: float = 0.0
    ops_compatibility: float = 0.0
    unsupported_ops: List[str] = field(default_factory=list)

    # Overall
    overall_score: float = 0.0
    overall_pass: bool = False
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chip_name": self.chip_name,
            "model_name": self.model_name,
            "inference_score": self.inference_score,
            "memory_score": self.memory_score,
            "flash_score": self.flash_score,
            "compute_score": self.compute_score,
            "power_score": self.power_score,
            "arch_score": self.arch_score,
            "inference_time_ms": self.inference_time_ms,
            "peak_ram_kb": self.peak_ram_kb,
            "model_size_kb": self.model_size_kb,
            "ops_compatibility": self.ops_compatibility,
            "unsupported_ops": self.unsupported_ops,
            "overall_score": self.overall_score,
            "overall_pass": self.overall_pass,
            "suggestions": self.suggestions,
        }


class ChipController(QObject):
    """Manages chip database and evaluation."""

    # Signals
    chip_list_updated = Signal()
    assessment_complete = Signal(AssessmentResult)
    assessment_error = Signal(str)
    batch_comparison_complete = Signal(list)  # list of AssessmentResult

    # Weights for dimensions
    WEIGHTS = {
        "inference": 0.30,
        "memory": 0.25,
        "flash": 0.20,
        "compute": 0.15,
        "power": 0.10,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = ChipDatabase()
        self._log = LogManager()

    # ================================================================
    # Chip DB Operations
    # ================================================================

    @property
    def database(self) -> ChipDatabase:
        return self._db

    def list_chips(self, **filters) -> List[ChipSpec]:
        return self._db.search(**filters) if filters else self._db.list_all()

    def get_chip(self, chip_id: int) -> Optional[ChipSpec]:
        return self._db.get_by_id(chip_id)

    def add_chip(self, chip: ChipSpec) -> int:
        chip_id = self._db.add(chip)
        self.chip_list_updated.emit()
        self._log.info("芯片", f"已添加芯片: {chip.name}")
        return chip_id

    def update_chip(self, chip: ChipSpec):
        self._db.update(chip)
        self.chip_list_updated.emit()
        self._log.info("芯片", f"已更新芯片: {chip.name}")

    def delete_chip(self, chip_id: int):
        chip = self._db.get_by_id(chip_id)
        self._db.delete(chip_id)
        self.chip_list_updated.emit()
        if chip:
            self._log.info("芯片", f"已删除芯片: {chip.name}")

    # ================================================================
    # Assessment Engine
    # ================================================================

    def assess(self, chip: ChipSpec,
               model_checkpoint_path: Optional[str] = None,
               model_params: Optional[int] = None) -> AssessmentResult:
        """Evaluate whether a model can run on a given chip.

        Args:
            chip: Target chip specification
            model_checkpoint_path: Optional path to model for accurate analysis
            model_params: Override model parameter count (if checkpoint not available)

        Returns:
            AssessmentResult with scores, verdict, and suggestions
        """
        result = AssessmentResult(chip_name=chip.name,
                                  model_name=os.path.basename(model_checkpoint_path or ""))

        # Estimate model properties
        total_params = model_params or 4_500_000  # default ~4.5M (standard)
        int8_size_kb = total_params * 1 / 1024   # INT8: 1 byte/param
        fp32_size_kb = total_params * 4 / 1024   # FP32: 4 bytes/param
        model_size_kb = int8_size_kb  # assume INT8 for embedded

        # Estimate MACs (very rough: ~10 MACs/param/inference for CNN-like models)
        total_macs = total_params * 10
        result.model_size_kb = model_size_kb

        # === 1. Inference Time ===
        if chip.npu_tops > 0:
            # NPU available
            npu_macs_per_sec = chip.npu_tops * 1e12 * 0.6  # 60% efficiency
            inference_ms = total_macs / npu_macs_per_sec * 1000
        elif chip.dsp:
            # DSP estimate: ~100 MMACs/s per 100MHz
            dsp_macs = chip.cpu_freq_mhz * 1_000_000 * 0.5
            inference_ms = total_macs / dsp_macs * 1000
        else:
            # CPU-only: ~2 instructions/MAC, ~0.5 MACs/Hz
            cpu_macs = chip.cpu_freq_mhz * 1_000_000 / 4
            inference_ms = total_macs / cpu_macs * 1000

        result.inference_time_ms = inference_ms
        target_ms = 100  # target: <100ms for real-time
        result.inference_score = min(100.0, target_ms / max(inference_ms, 0.01) * 100)

        # === 2. Memory (RAM/SRAM) ===
        # Peak: weights + activations + buffers (~3x weight size for inference)
        peak_ram_kb = int8_size_kb * 3
        result.peak_ram_kb = peak_ram_kb
        result.memory_score = min(100.0, chip.ram_kb / max(peak_ram_kb, 1) * 100)

        # === 3. Flash ===
        result.flash_score = min(100.0, chip.flash_kb / max(model_size_kb, 1) * 100)

        # === 4. Compute Match ===
        if chip.npu_tops > 0:
            required_tops = total_macs / (target_ms / 1000) / 1e12
            result.compute_score = min(100.0, chip.npu_tops / max(required_tops, 0.001) * 100)
        else:
            result.compute_score = 50.0 if chip.cpu_freq_mhz > 200 else 20.0

        # === 5. Power ===
        # Very rough: inference power ~ NPU/CPU power during active inference
        est_power_mw = chip.power_consumption_mw * 0.8
        result.power_score = min(100.0, 100.0)  # simplified

        # === 6. Architecture / Op Compatibility ===
        # Check common ops needed by our model
        required_ops = {"Conv", "Conv1d", "BatchNorm", "ReLU", "GRU",
                        "FC", "Softmax", "Pool", "Concat", "Gemm"}
        chip_ops = set(chip.supported_ops)

        supported = required_ops & chip_ops
        unsupported = required_ops - chip_ops
        result.ops_compatibility = len(supported) / len(required_ops) * 100
        result.unsupported_ops = list(unsupported)
        result.arch_score = result.ops_compatibility

        # === Overall ===
        dim = result
        scores = [
            (dim.inference_score, "inference"),
            (dim.memory_score, "memory"),
            (dim.flash_score, "flash"),
            (dim.compute_score, "compute"),
            (dim.power_score, "power"),
        ]
        # Architecture compatibility is a hard gate
        if dim.arch_score < 80:
            result.overall_pass = False
            dim.suggestions.append(
                f"算子兼容性不足 ({dim.arch_score:.0f}%),"
                f"不支持的算子: {', '.join(dim.unsupported_ops)}"
            )

        weighted = sum(s * self.WEIGHTS[w] for s, w in scores)
        result.overall_score = weighted

        # Pass/fail: all dimensions must be >= 50
        all_pass = all(s >= 50 for s, _ in scores) and dim.arch_score >= 80
        result.overall_pass = all_pass

        # Generate suggestions
        if dim.inference_score < 50:
            result.suggestions.append(
                f"推理时间过长 ({dim.inference_time_ms:.0f}ms),"
                f"考虑模型轻量化或选择更高算力芯片"
            )
        if dim.memory_score < 50:
            result.suggestions.append(
                f"RAM不足 ({dim.peak_ram_kb:.0f}KB > {chip.ram_kb}KB),"
                f"考虑INT8量化或减少模型参数"
            )
        if dim.flash_score < 50:
            result.suggestions.append(
                f"Flash不足 ({dim.model_size_kb:.0f}KB > {chip.flash_kb}KB),"
                f"考虑模型裁剪或使用更大Flash芯片"
            )

        self._log.info("芯片", f"评估完成: {chip.name} -> "
                      f"{'通过' if result.overall_pass else '不通过'} "
                      f"({result.overall_score:.0f}分)")

        self.assessment_complete.emit(result)
        return result

    def batch_compare(self, chips: List[ChipSpec],
                      model_params: Optional[int] = None) -> List[AssessmentResult]:
        """Compare model compatibility across multiple chips."""
        results = []
        for chip in chips:
            result = self.assess(chip, model_params=model_params)
            results.append(result)
        self.batch_comparison_complete.emit(results)
        return results
