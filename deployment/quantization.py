"""
模型量化模块 —— 为 SOC NPU 部署准备

支持的量化方案:
1. PyTorch 静态量化 (INT8) — CPU 推理加速
2. ONNX Runtime INT8 量化 — 通用嵌入式推理
3. 芯片特定 NPU 量化 (模拟) — Rockchip / Amlogic / 海思 等

量化流程:
  FP32 Model → Calibration (校准数据) → INT8/INT16 Quantized Model → NPU 推理

注意事项:
- Conv/FC 层: INT8 量化精度损失 < 1%
- GRU 层: 推荐 INT16 量化 (INT8 易溢出)
- CTC Beam Search 在 ONNX 外部执行
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict


class ModelQuantizer:
    """
    模型量化器

    支持 PyTorch native quantization 和 ONNX Runtime quantization
    """

    def __init__(self, model: nn.Module, device='cpu'):
        self.model = model.to(device).eval()
        self.device = device

    def pytorch_static_quantize(self, calibration_loader,
                                 output_path: str,
                                 backend: str = 'fbgemm') -> nn.Module:
        """
        PyTorch 静态 INT8 量化

        Args:
            calibration_loader: 校准数据 DataLoader
            output_path: 保存路径
            backend: 'fbgemm' (x86) | 'qnnpack' (ARM)

        Returns:
            quantized_model: 量化后的模型
        """
        print("[Quantize] PyTorch Static INT8 Quantization")

        # 配置
        if backend == 'qnnpack':
            torch.backends.quantized.engine = 'qnnpack'
        else:
            torch.backends.quantized.engine = 'fbgemm'

        # 设置量化配置
        model_fp32 = self.model

        # 为 Conv/Linear 设置量化观测器
        model_fp32.qconfig = torch.quantization.get_default_qconfig(backend)
        # GRU 层使用 dynamic quantization (INT8 weight, FP32 activation)
        # model_fp32.gru.qconfig = torch.quantization.default_dynamic_qconfig

        # 准备
        model_prepared = torch.quantization.prepare(model_fp32, inplace=False)

        # 校准 (用校准数据集跑推理)
        print("[Quantize] Calibrating...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= 100:  # 100 batches 足够校准
                    break
                audio = batch['audio'].to(self.device)
                lengths = batch['audio_lengths'].to(self.device)
                _ = model_prepared(audio, lengths)
                if (i + 1) % 20 == 0:
                    print(f"  Calibration: {i+1}/100 batches")

        # 转换为 INT8
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)

        # 保存
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'quantized': True,
        }, output_path)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Quantize] INT8 model saved to {output_path} ({size_mb:.2f} MB)")

        return quantized_model

    def onnx_quantize_int8(self, onnx_model_path: str,
                            calibration_data: List[np.ndarray],
                            output_path: str,
                            quant_format: str = 'QOperator') -> str:
        """
        ONNX Runtime INT8 量化

        Args:
            onnx_model_path: FP32 ONNX 模型路径
            calibration_data: 校准音频数据列表 [numpy arrays]
            output_path: 输出路径
            quant_format: 'QOperator' | 'QDQ' (Quantize-Dequantize)

        Returns:
            output_path: 量化后的 ONNX 路径
        """
        try:
            from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
            import onnx
        except ImportError:
            print("[Quantize] onnxruntime quantization tools not available")
            print("  Install: pip install onnxruntime-tools")
            return onnx_model_path

        print(f"[Quantize] ONNX Runtime INT8 Quantization")
        print(f"  Format: {quant_format}")

        # 创建校准数据读取器
        class CalibrationDataReader:
            def __init__(self, data_list, input_names):
                self.data = data_list
                self.input_names = input_names
                self.iter = iter(self.data)

            def get_next(self):
                try:
                    batch = next(self.iter)
                    return {name: batch[i] for i, name in enumerate(self.input_names)}
                except StopIteration:
                    return None

        # 获取输入名称
        model_onnx = onnx.load(onnx_model_path)
        input_names = [inp.name for inp in model_onnx.graph.input]

        # 校准数据格式: [(audio, lengths), ...]
        calibration_reader = CalibrationDataReader(
            [(d[0], d[1]) for d in calibration_data],
            input_names
        )

        # 量化
        quantized_model = quantize_static(
            onnx_model_path,
            output_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QOperator if quant_format == 'QOperator' else QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Quantize] INT8 ONNX model saved to {output_path} ({size_mb:.2f} MB)")

        return output_path

    def onnx_dynamic_quantize(self, onnx_model_path: str,
                               output_path: str) -> str:
        """
        ONNX Runtime 动态 INT8 量化

        仅量化权重，activation 保持 FP32
        适合 GRU/LSTM 等递归层

        Args:
            onnx_model_path: FP32 模型
            output_path: 输出路径

        Returns:
            output_path
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            return onnx_model_path

        print("[Quantize] ONNX Dynamic Quantization")

        quantize_dynamic(
            onnx_model_path,
            output_path,
            weight_type=QuantType.QInt8,
        )

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Quantize] Dynamic INT8 model saved ({size_mb:.2f} MB)")

        return output_path

    def prepare_for_soc_npu(self, onnx_path: str,
                             chip_type: str = 'rockchip_rk3588') -> str:
        """
        SOC NPU 适配准备

        针对不同芯片做算子替换和优化:
        - Rockchip RK3588 (3 TOPS NPU)
        - Amlogic A311D (5 TOPS NPU)
        - 海思 Hi3559A (4 TOPS NPU)

        实际部署时需用芯片厂商提供的转换工具:
        - Rockchip: rknn-toolkit2 (onnx → rknn)
        - Amlogic:   aml_npu_sdk (onnx → nb)
        - 海思:      nnie_mapper (caffe → wk)

        Args:
            onnx_path: FP32 ONNX 模型
            chip_type: 芯片型号

        Returns:
            soc_config: 芯片配置字典
        """
        chip_configs = {
            'rockchip_rk3588': {
                'npu_name': 'RKNN (3 TOPS)',
                'tool': 'rknn-toolkit2',
                'quant_type': 'INT8 (per-channel)',
                'max_model_size': 10,  # MB
                'supported_ops': [
                    'Conv', 'Conv1d', 'BatchNorm', 'ReLU', 'Sigmoid', 'Tanh',
                    'GRU', 'Gemm', 'MatMul', 'Add', 'Mul', 'Concat', 'Reshape',
                    'Softmax', 'AveragePool', 'GlobalAveragePool', 'Slice',
                    'Transpose', 'Gather', 'Unsqueeze', 'Squeeze',
                ],
                'unsupported_ops': ['LayerNorm', 'InstanceNorm', 'Dropout'],
            },
            'amlogic_a311d': {
                'npu_name': 'Amlogic NPU (5 TOPS)',
                'tool': 'aml_npu_sdk',
                'quant_type': 'INT8 / INT16',
                'max_model_size': 20,
                'supported_ops': [
                    'Conv', 'Conv1d', 'BN', 'ReLU', 'GRU', 'FC', 'Softmax',
                    'Pool', 'Concat', 'Eltwise',
                ],
            },
            'hisilicon_hi3559a': {
                'npu_name': 'NNIE (4 TOPS)',
                'tool': 'nnie_mapper',
                'quant_type': 'INT8 (per-layer)',
                'max_model_size': 8,
                'supported_ops': [
                    'Conv', 'BN', 'ReLU', 'FC', 'Pool', 'Concat', 'Eltwise',
                ],
                'unsupported_ops': ['GRU', 'LayerNorm'],
            },
        }

        config = chip_configs.get(chip_type, chip_configs['rockchip_rk3588'])

        print(f"[SOC] Preparing for {chip_type}")
        print(f"  NPU: {config['npu_name']}")
        print(f"  Quantization: {config['quant_type']}")
        print(f"  Max model size: {config['max_model_size']} MB")

        # 算子兼容性检查
        print(f"  Supported ops: {len(config['supported_ops'])}")
        if 'unsupported_ops' in config:
            print(f"  ⚠ Unsupported ops: {config['unsupported_ops']}")
            print(f"    → Will need op substitution before conversion")

        return config


def quantize_model_int8(model: nn.Module, calibration_loader,
                         output_path: str) -> nn.Module:
    """便捷量化函数"""
    quantizer = ModelQuantizer(model)
    return quantizer.pytorch_static_quantize(calibration_loader, output_path)
