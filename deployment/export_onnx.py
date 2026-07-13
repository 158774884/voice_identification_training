"""
ONNX 导出模块

将 PyTorch 模型导出为 ONNX 格式，用于:
1. 模型检查与验证
2. INT8/INT16 量化
3. NPU 推理引擎部署

注意事项:
- 导出前必须替换动态控制流 (if/for) 为静态
- 仅支持 ONNX opset 12+ (支持 GRU)
- 支持动态 batch / 固定长度两种模式
- CTC 后处理在 ONNX 外部完成
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple


class ONNXExporter:
    """
    PyTorch → ONNX 导出器

    支持:
    - 整个多任务模型导出
    - 单任务分支导出
    - 动态/固定 shape
    - ONNX 验证和优化
    """

    def __init__(self, model: nn.Module, device='cpu'):
        self.model = model.to(device).eval()
        self.device = device

    def export_full_model(self, output_path: str,
                          dynamic_audio_len: bool = True,
                          opset_version: int = 14,
                          simplify: bool = True) -> str:
        """
        导出完整多任务模型

        Args:
            output_path: 输出 ONNX 文件路径
            dynamic_audio_len: 是否支持动态音频长度
            opset_version: ONNX opset 版本
            simplify: 是否用 onnx-simplifier 简化

        Returns:
            output_path: 导出的文件路径
        """
        # 示例输入
        if dynamic_audio_len:
            # 动态长度: 用 symbolic shape
            dummy_audio = torch.randn(1, 1, 16000 * 5).to(self.device)  # 5s
            dummy_lengths = torch.tensor([16000 * 5], dtype=torch.long).to(self.device)

            dynamic_axes = {
                'audio': {0: 'batch', 2: 'audio_len'},
                'audio_lengths': {0: 'batch'},
                'feat_lengths': {0: 'batch'},
                'asr_log_probs': {1: 'batch'},
                'dialect_logits': {0: 'batch'},
                'speaker_embedding': {0: 'batch'},
            }
        else:
            # 固定长度: 更高效但灵活性低
            fixed_len = 16000 * 10
            dummy_audio = torch.randn(1, 1, fixed_len).to(self.device)
            dummy_lengths = torch.tensor([fixed_len], dtype=torch.long).to(self.device)
            dynamic_axes = None

        # 包装模型 (移除内部 task_mask 逻辑)
        wrapped_model = ONNXExportWrapper(self.model)

        print(f"[ONNX] Exporting to {output_path}")
        print(f"  Opset: {opset_version}")
        print(f"  Dynamic audio length: {dynamic_audio_len}")

        torch.onnx.export(
            wrapped_model,
            (dummy_audio, dummy_lengths),
            output_path,
            input_names=['audio', 'audio_lengths'],
            output_names=['asr_log_probs', 'feat_lengths',
                          'dialect_logits', 'speaker_embedding'],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )

        # 验证导出
        self._verify_onnx(output_path)

        # 简化 (需要 onnx-simplifier)
        if simplify:
            try:
                from onnxsim import simplify as onnx_simplify
                import onnx
                model_onnx = onnx.load(output_path)
                model_simp, check = onnx_simplify(model_onnx)
                if check:
                    onnx.save(model_simp, output_path)
                    print(f"[ONNX] Simplified model saved to {output_path}")
                else:
                    print("[ONNX] Simplification check failed, using original")
            except ImportError:
                print("[ONNX] onnx-simplifier not installed, skipping simplification")
                print("  Install: pip install onnx-simplifier")

        # 文件大小
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[ONNX] Model size: {size_mb:.2f} MB")

        return output_path

    def export_asr_only(self, output_path: str, **kwargs) -> str:
        """仅导出 ASR 分支 (更小体积)"""
        wrapper = ASROnlyWrapper(self.model)
        dummy_audio = torch.randn(1, 1, 16000 * 5).to(self.device)
        dummy_lengths = torch.tensor([16000 * 5], dtype=torch.long).to(self.device)

        torch.onnx.export(
            wrapper, (dummy_audio, dummy_lengths),
            output_path,
            input_names=['audio', 'audio_lengths'],
            output_names=['log_probs', 'feat_lengths'],
            opset_version=kwargs.get('opset_version', 14),
            dynamic_axes={
                'audio': {0: 'batch', 2: 'audio_len'},
                'audio_lengths': {0: 'batch'},
                'feat_lengths': {0: 'batch'},
                'log_probs': {1: 'batch'},
            },
        )
        return output_path

    def export_speaker_only(self, output_path: str, **kwargs) -> str:
        """仅导出声纹分支"""
        wrapper = SpeakerOnlyWrapper(self.model)
        dummy_audio = torch.randn(1, 1, 16000 * 5).to(self.device)
        dummy_lengths = torch.tensor([16000 * 5], dtype=torch.long).to(self.device)

        torch.onnx.export(
            wrapper, (dummy_audio, dummy_lengths),
            output_path,
            input_names=['audio', 'audio_lengths'],
            output_names=['speaker_embedding'],
            opset_version=kwargs.get('opset_version', 14),
            dynamic_axes={
                'audio': {0: 'batch', 2: 'audio_len'},
                'audio_lengths': {0: 'batch'},
                'speaker_embedding': {0: 'batch'},
            },
        )
        return output_path

    def export_dialect_only(self, output_path: str, **kwargs) -> str:
        """仅导出方言分类分支"""
        wrapper = DialectOnlyWrapper(self.model)
        dummy_audio = torch.randn(1, 1, 16000 * 5).to(self.device)
        dummy_lengths = torch.tensor([16000 * 5], dtype=torch.long).to(self.device)

        torch.onnx.export(
            wrapper, (dummy_audio, dummy_lengths),
            output_path,
            input_names=['audio', 'audio_lengths'],
            output_names=['dialect_logits'],
            opset_version=kwargs.get('opset_version', 14),
            dynamic_axes={
                'audio': {0: 'batch', 2: 'audio_len'},
                'audio_lengths': {0: 'batch'},
                'dialect_logits': {0: 'batch'},
            },
        )
        return output_path

    def _verify_onnx(self, onnx_path: str):
        """验证 ONNX 模型"""
        try:
            import onnx
            import onnxruntime as ort

            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print("[ONNX] Model verification PASSED ✓")

            # 尝试推理
            session = ort.InferenceSession(onnx_path)
            dummy_input = np.random.randn(1, 1, 16000 * 3).astype(np.float32)
            dummy_lengths = np.array([16000 * 3], dtype=np.int64)

            outputs = session.run(None, {
                'audio': dummy_input,
                'audio_lengths': dummy_lengths,
            })

            print(f"[ONNX] Test inference OK ({len(outputs)} outputs)")
            for i, o in enumerate(outputs):
                print(f"  Output {i}: shape={o.shape}, dtype={o.dtype}")

        except ImportError:
            print("[ONNX] onnx/onnxruntime not installed, skipping verification")
            print("  Install: pip install onnx onnxruntime")
        except Exception as e:
            print(f"[ONNX] Warning: verification failed: {e}")


class ONNXExportWrapper(nn.Module):
    """
    ONNX 导出包装器 —— 移除 Python 控制流, 输出所有任务固定格式
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, audio_lengths):
        outputs = self.model(
            audio, audio_lengths,
            task_mask={'asr': True, 'dialect': True, 'speaker': True}
        )
        return (outputs['asr_log_probs'],
                outputs['feat_lengths'],
                outputs['dialect_logits'],
                outputs['speaker_embedding'])


class ASROnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, audio_lengths):
        outputs = self.model(
            audio, audio_lengths,
            task_mask={'asr': True, 'dialect': False, 'speaker': False}
        )
        return outputs['asr_log_probs'], outputs['feat_lengths']


class SpeakerOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, audio_lengths):
        outputs = self.model(
            audio, audio_lengths,
            task_mask={'asr': False, 'dialect': False, 'speaker': True}
        )
        return outputs['speaker_embedding']


class DialectOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, audio_lengths):
        outputs = self.model(
            audio, audio_lengths,
            task_mask={'asr': False, 'dialect': True, 'speaker': False}
        )
        return outputs['dialect_logits']


def export_to_onnx(model: nn.Module, output_dir: str = './onnx_models',
                   export_mode: str = 'full', **kwargs):
    """
    便捷导出函数

    Args:
        model: 训练好的 MultiTaskVoiceModel
        output_dir: 输出目录
        export_mode: 'full' | 'asr' | 'dialect' | 'speaker' | 'all'
    """
    os.makedirs(output_dir, exist_ok=True)
    exporter = ONNXExporter(model)

    paths = {}

    if export_mode in ('full', 'all'):
        path = os.path.join(output_dir, 'voice_model_full.onnx')
        paths['full'] = exporter.export_full_model(path, **kwargs)

    if export_mode in ('asr', 'all'):
        path = os.path.join(output_dir, 'voice_model_asr.onnx')
        paths['asr'] = exporter.export_asr_only(path, **kwargs)

    if export_mode in ('dialect', 'all'):
        path = os.path.join(output_dir, 'voice_model_dialect.onnx')
        paths['dialect'] = exporter.export_dialect_only(path, **kwargs)

    if export_mode in ('speaker', 'all'):
        path = os.path.join(output_dir, 'voice_model_speaker.onnx')
        paths['speaker'] = exporter.export_speaker_only(path, **kwargs)

    return paths
