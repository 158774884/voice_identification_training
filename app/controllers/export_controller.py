"""
Export Controller — manages model export, quantization, firmware generation.
Reuses existing deployment/export_onnx.py, deployment/quantization.py,
and rtl8713e_deploy/two_stage_kws/export_ac7916.py.
"""
import os
import sys
import json
import shutil
from datetime import datetime
from typing import Optional, Dict, Any

from PySide6.QtCore import QObject, Signal, Slot

from app.utils.logger import LogManager


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class ExportController(QObject):
    """Controls firmware export workflow."""

    # Signals
    export_progress = Signal(int, int, str)     # current, total, phase description
    export_complete = Signal(str, dict)          # output_path, summary
    export_error = Signal(str)                   # error
    log_message = Signal(str, str)               # category, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log = LogManager()

    def export_onnx(self, checkpoint_path: str, output_dir: str,
                    export_mode: str = "full") -> Optional[str]:
        """Export model to ONNX format.

        Args:
            checkpoint_path: Path to model checkpoint (.pt)
            output_dir: Output directory
            export_mode: 'full', 'asr', 'dialect', 'speaker', 'all'

        Returns:
            Path to the exported ONNX file, or None on failure
        """
        os.makedirs(output_dir, exist_ok=True)

        try:
            from deployment.export_onnx import export_to_onnx
            from model.multi_task_model import create_model
            import torch

            self.export_progress.emit(1, 4, "加载模型...")
            self._log.info("导出", "正在加载模型...")

            model = create_model()
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()

            self.export_progress.emit(2, 4, "导出 ONNX...")
            paths = export_to_onnx(model, output_dir=output_dir, export_mode=export_mode)

            self.export_progress.emit(4, 4, "完成")
            output_path = paths.get('full') or paths.get(export_mode)
            summary = {"format": "ONNX", "paths": paths}

            self.export_complete.emit(output_dir, summary)
            self._log.info("导出", f"ONNX 导出完成: {output_dir}")
            return output_path

        except Exception as e:
            self.export_error.emit(f"ONNX 导出失败: {e}")
            self._log.error("导出", f"ONNX 导出失败: {e}")
            return None

    def export_quantized(self, onnx_path: str, output_dir: str,
                        quant_method: str = "INT8") -> Optional[str]:
        """Quantize an ONNX model.

        Args:
            onnx_path: Path to FP32 ONNX model
            output_dir: Output directory
            quant_method: 'INT8', 'INT16', 'dynamic'

        Returns:
            Path to quantized model
        """
        os.makedirs(output_dir, exist_ok=True)

        try:
            from deployment.quantization import ModelQuantizer
            from model.multi_task_model import create_model
            import torch

            self.export_progress.emit(1, 3, "加载模型...")

            model = create_model()
            quantizer = ModelQuantizer(model)

            output_path = os.path.join(output_dir, "voice_model_int8.onnx")

            self.export_progress.emit(2, 3, f"正在{quant_method}量化...")

            if quant_method == "dynamic":
                quantizer.onnx_dynamic_quantize(onnx_path, output_path)
            else:
                quantizer.onnx_quantize_int8(onnx_path, [], output_path)

            self.export_progress.emit(3, 3, "完成")
            self.export_complete.emit(output_path, {
                "format": f"ONNX ({quant_method})",
                "size_kb": os.path.getsize(output_path) // 1024,
            })
            self._log.info("导出", f"量化完成: {output_path}")
            return output_path

        except Exception as e:
            self.export_error.emit(f"量化失败: {e}")
            self._log.error("导出", f"量化失败: {e}")
            return None

    def export_c_firmware(self, stage1_ckpt: str, stage2_ckpt: str,
                          grammar_path: str, output_dir: str,
                          chip_name: str = "AC7916AB") -> Optional[str]:
        """Export the lightweight two-stage KWS model as C firmware headers.

        Uses rtl8713e_deploy/two_stage_kws/export_ac7916.py to generate
        stage1_model.h / stage2_model.h / grammar.h / mel_config.h /
        kws_pipeline.h / flash_layout.txt. This is far smaller than the
        full multi-task model (tens of MB -> ~1 MB).

        Args:
            stage1_ckpt: Stage 1 (wake-word) checkpoint path
            stage2_ckpt: Stage 2 (CTC command) checkpoint path
            grammar_path: WFST grammar JSON path
            output_dir: Output directory
            chip_name: Target chip name

        Returns:
            Path to firmware package
        """
        os.makedirs(output_dir, exist_ok=True)

        # Validate inputs
        for label, path in (("Stage1 模型", stage1_ckpt),
                            ("Stage2 模型", stage2_ckpt),
                            ("语法文件", grammar_path)):
            if not path or not os.path.exists(path):
                self.export_error.emit(f"{label} 不存在: {path}")
                return None

        try:
            self.export_progress.emit(1, 3, "导出两阶段 KWS C 固件...")
            self._log.info("导出", f"两阶段 KWS 导出: stage1={os.path.basename(stage1_ckpt)}, "
                                   f"stage2={os.path.basename(stage2_ckpt)}")

            kws_dir = os.path.join(PROJECT_ROOT, "rtl8713e_deploy", "two_stage_kws")
            if kws_dir not in sys.path:
                sys.path.insert(0, kws_dir)

            import export_ac7916
            from types import SimpleNamespace

            self.export_progress.emit(2, 3, "生成 INT8 权重 / 语法 / Mel 配置...")
            export_ac7916.export(SimpleNamespace(
                stage1_ckpt=stage1_ckpt,
                stage2_ckpt=stage2_ckpt,
                grammar=grammar_path,
                output=output_dir,
            ))

            self.export_progress.emit(3, 3, "完成")

            # Summarize generated files
            generated = sorted(
                f for f in os.listdir(output_dir)
                if os.path.isfile(os.path.join(output_dir, f))
            )
            total_bytes = sum(os.path.getsize(os.path.join(output_dir, f))
                              for f in generated)

            summary = {
                "format": "C Firmware (2-Stage KWS)",
                "chip": chip_name,
                "output_dir": output_dir,
                "files": generated,
                "total_kb": total_bytes // 1024,
            }
            self.export_complete.emit(output_dir, summary)
            self._log.info("导出", f"固件导出完成: {output_dir} ({total_bytes/1024:.0f} KB)")
            return output_dir

        except Exception as e:
            self.export_error.emit(f"固件导出失败: {e}")
            self._log.error("导出", f"固件导出失败: {e}")
            return None

    def _generate_model_header(self, checkpoint_path: str, output_dir: str):
        """Generate model weights as C header file."""
        import torch
        import numpy as np

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            return

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if not isinstance(state_dict, dict):
            return

        header_path = os.path.join(output_dir, "model_weights.h")
        with open(header_path, "w", encoding="utf-8") as f:
            f.write("// Auto-generated model weights\n")
            f.write(f"// Source: {os.path.basename(checkpoint_path)}\n")
            f.write(f"// Generated: {datetime.now().isoformat()}\n\n")
            f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
            f.write("#include <stdint.h>\n\n")

            for name, tensor in state_dict.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                safe_name = name.replace('.', '_').replace('-', '_')
                data = tensor.cpu().numpy()

                # INT8 quantization (simple per-tensor)
                if data.dtype in (np.float32, np.float64):
                    scale = max(abs(data.min()), abs(data.max())) / 127.0
                    if scale == 0:
                        scale = 1.0
                    quant = np.clip(np.round(data / scale), -128, 127).astype(np.int8)
                    f.write(f"// {safe_name}: shape {list(data.shape)}, scale={scale:.6f}\n")
                    f.write(f"static const float {safe_name}_scale = {scale:.6f}f;\n")
                else:
                    quant = data

                f.write(f"static const int8_t {safe_name}[] = {{\n    ")
                flat = quant.flatten()
                for i, val in enumerate(flat):
                    f.write(f"{int(val)}, ")
                    if (i + 1) % 16 == 0 and i < len(flat) - 1:
                        f.write("\n    ")
                f.write(f"\n}}; // {len(flat)} values\n\n")

            f.write("#endif // MODEL_WEIGHTS_H\n")

    def _generate_firmware_config(self, output_dir: str, chip_name: str):
        """Generate firmware configuration files."""
        config_path = os.path.join(output_dir, "firmware_config.h")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("// Firmware configuration\n")
            f.write(f"// Target chip: {chip_name}\n")
            f.write(f"// Generated: {datetime.now().isoformat()}\n\n")
            f.write("#ifndef FIRMWARE_CONFIG_H\n#define FIRMWARE_CONFIG_H\n\n")
            f.write("#define SAMPLE_RATE 16000\n")
            f.write("#define FRAME_SIZE_MS 25\n")
            f.write("#define FRAME_SHIFT_MS 10\n")
            f.write("#define MEL_BANDS 40\n")
            f.write("#define MAX_COMMANDS 200\n")
            f.write("\n#endif // FIRMWARE_CONFIG_H\n")

    def embed_sdk_demo(self, firmware_dir: str, sdk_dir: str) -> bool:
        """Copy firmware files into SDK demo project structure.

        Args:
            firmware_dir: Directory with generated firmware files
            sdk_dir: Path to chip SDK directory

        Returns:
            True if successful
        """
        if not os.path.exists(sdk_dir):
            self.export_error.emit(f"SDK 目录不存在: {sdk_dir}")
            return False

        model_dir = os.path.join(sdk_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        # Copy firmware files
        copied = []
        for fn in os.listdir(firmware_dir):
            src = os.path.join(firmware_dir, fn)
            dst = os.path.join(model_dir, fn)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied.append(fn)

        self._log.info("导出", f"已嵌入 SDK Demo: {len(copied)} 个文件 -> {model_dir}")
        return True
