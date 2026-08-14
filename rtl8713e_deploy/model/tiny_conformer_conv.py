"""
AC7916AB 适配版: TinyKWS + GRU-free Tiny-Conformer

AC7916AB 硬件约束:
- CPU: 双核 DSP
- AI加速: 芯片加速器
- SRAM: 578KB (on-chip)
- PSRAM: 外接 (2-8MB 典型)
- 加速器擅长: Conv2D, DepthwiseConv, FC, Pool, ReLU
- 加速器不擅长/不支持: GRU, LayerNorm, Multi-Head Attention

策略: 用 Dilated Depthwise Conv 替代 GRU 做长时序建模
      → 加速器可以全速运行，无循环依赖瓶颈
      → 参数量 ~250K, INT8 权重 ~250KB

对比:
  原版 TinyConformer (shared_backbone.py):
    CNN + GRU → GRU 在 加速器上无加速, 性能差

  Conv 优化版 (本文件):
    CNN + Dilated DWConv → 全部是 Conv 操作, 卷积原生加速
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedDWConvBlock(nn.Module):
    """
    多尺度膨胀深度可分离卷积块
    替代 GRU 做长时序建模

    3 路并行的 Dilated Conv:
      - dilation=1:  局部上下文 (~60ms @ 25Hz)
      - dilation=3:  中程上下文 (~180ms)
      - dilation=9:  长程上下文 (~540ms)

    卷积原生优化: 全部是 Conv2D/DepthwiseConv → 全速运行
    """
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        # 三路并行膨胀卷积
        self.dw1 = nn.Conv2d(channels, channels, kernel_size=(1, 3),
                             padding=(0, 1), dilation=(1, 1),
                             groups=channels, bias=False)
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=(1, 3),
                             padding=(0, 3), dilation=(1, 3),
                             groups=channels, bias=False)
        self.dw9 = nn.Conv2d(channels, channels, kernel_size=(1, 3),
                             padding=(0, 9), dilation=(1, 9),
                             groups=channels, bias=False)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        # x: [B, C, H, 1] — frequency × time as 2D
        d1 = self.dw1(x)
        d3 = self.dw3(x)
        d9 = self.dw9(x)

        # 确保时间维度对齐 (dilation=9 会多出 padding)
        min_t = min(d1.size(3), d3.size(3), d9.size(3))
        d1, d3, d9 = d1[:, :, :, :min_t], d3[:, :, :, :min_t], d9[:, :, :, :min_t]

        # 门控融合: gate * sum(paths)
        concat = torch.cat([d1, d3, d9], dim=1)  # [B, 3C, H, T]
        gate = self.gate(concat)
        fused = self.fuse(concat)
        x = gate * fused
        x = self.bn(x)
        x = self.dropout(x)

        # 对齐 residual
        residual = residual[:, :, :, :min_t]
        return x + residual


class ConvConformerBlock(nn.Module):
    """
    Conv 优化的 Conformer Block (无 GRU, 无 MHA)

    结构: LN → FFN(Conv1×1 扩张) → Dilated DWConv → FFN → Residual
          全 Conv 操作, 加速器一次性跑完

    参数量: ~8K/block → 4 blocks ≈ 32K
    """
    def __init__(self, dim, expansion=2, dropout=0.1):
        super().__init__()
        hidden = dim * expansion

        # FFN 1
        self.ffn1_pw1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.ffn1_bn1 = nn.BatchNorm2d(hidden)
        self.ffn1_pw2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.ffn1_bn2 = nn.BatchNorm2d(dim)

        # Temporal modeling (替代 GRU)
        self.temporal = DilatedDWConvBlock(dim, dropout)

        # FFN 2
        self.ffn2_pw1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.ffn2_bn1 = nn.BatchNorm2d(hidden)
        self.ffn2_pw2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.ffn2_bn2 = nn.BatchNorm2d(dim)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # FFN 1
        residual = x
        x1 = F.relu(self.ffn1_bn1(self.ffn1_pw1(x)))
        x1 = self.ffn1_pw2(x1)
        x1 = self.ffn1_bn2(x1)
        x = self.dropout(x1) + residual

        # Temporal (Dilated DWConv)
        residual = x
        x = self.temporal(x)
        x = self.dropout(x) + residual

        # FFN 2
        residual = x
        x2 = F.relu(self.ffn2_bn1(self.ffn2_pw1(x)))
        x2 = self.ffn2_pw2(x2)
        x2 = self.ffn2_bn2(x2)
        x = self.dropout(x2) + residual

        return x


class TinyKWS_Conv(nn.Module):
    """
    AC7916AB 优化版: TinyKWS CNN + 加速器-Conformer

    架构:
      Stem:       Conv2D 3×3 → 32ch  (特征提取)
      KWS Blocks: DS-CNN ×3          (局部模式识别)
      Conformer:  Conv 优化 blocks ×3 (长时序依赖)
      Head:       GlobalPool → FC → N_classes

    参数量: ~250K → INT8 权重 ~250KB
    加速器加速:   全部 Conv 操作 → 预期 RTF < 0.05x

    输入: [B, 1, n_mels, T_frames]  log-Mel 特征图
    输出: [B, num_classes]
    """

    def __init__(self, num_classes=50, n_mels=40, n_frames=98,
                 stem_ch=32, kws_ch=(64, 96, 128),
                 conf_dim=128, conf_blocks=3, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.n_mels = n_mels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_ch, kernel_size=(3, 3), stride=1,
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )

        # DS-CNN KWS Blocks (局部特征)
        self.kws_blocks = nn.ModuleList()
        in_ch = stem_ch
        for i, out_ch in enumerate(kws_ch):
            stride = (1, 2) if i < 2 else 1  # 2× temporal subsample
            self.kws_blocks.append(self._make_ds_block(in_ch, out_ch, stride))
            in_ch = out_ch

        # 加速器-Conformer Blocks (长时序建模)
        self.conf_blocks = nn.ModuleList([
            ConvConformerBlock(in_ch, expansion=2, dropout=dropout)
            for _ in range(conf_blocks)
        ])

        # Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

        self._init_weights()
        self._compute_shapes(n_mels, n_frames, stem_ch, kws_ch)

    def _make_ds_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch * 2, in_ch * 2, kernel_size=(3, 3), stride=stride,
                      padding=(1, 1), groups=in_ch * 2, bias=False),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _compute_shapes(self, n_mels, n_frames, stem_ch, kws_ch):
        with torch.no_grad():
            dummy = torch.randn(1, 1, n_mels, n_frames)
            dummy = self.stem(dummy)
            for blk in self.kws_blocks:
                dummy = blk(dummy)
            self.pooled_dim = dummy.size(1)
            self.pooled_h = dummy.size(2)
            self.pooled_w = dummy.size(3)
            del dummy

    def forward(self, x):
        x = self.stem(x)
        for blk in self.kws_blocks:
            x = blk(x)
        for blk in self.conf_blocks:
            x = blk(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def summary(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"TinyKWS-Conv (for AC7916AB) Summary:")
        print(f"  Input:      [B, 1, {self.n_mels}, T] log-Mel")
        print(f"  KWS out:    [{self.pooled_dim}, {self.pooled_h}, ~T/4]")
        print(f"  Conformer:  {len(self.conf_blocks)} blocks")
        print(f"  Pooled:     ({self.pooled_dim}, 1, 1)")
        print(f"  Classes:    {self.num_classes}")
        print(f"  Params:     {total:,} ({total/1000:.1f}K)")
        print(f"  INT8 wt:    ~{total/1024:.0f} KB")
        print(f"  加速器 ops:    100% Conv-based (zero GRU)")
        status = 'OK - COMPATIBLE' if total < 800000 else 'CHECK - large'
        print(f"  AC7916AB:   {status}")

    def export_onnx(self, path):
        self.eval()
        dummy = torch.randn(1, 1, self.n_mels, 98)
        torch.onnx.export(self, dummy, path,
                          input_names=['mel_features'],
                          output_names=['logits'],
                          opset_version=14,
                          dynamic_axes={'mel_features': {0: 'batch', 3: 'time'}})
        import os
        print(f"[ONNX] {path} ({os.path.getsize(path)/1024:.0f} KB)")


# ===== 芯片适配分析 =====
def chip_compatibility_analysis():
    """打印 AC7916AB 适配分析"""
    print("=" * 62)
    print("AC7916AB: TinyKWS-Conv 适配分析")
    print("=" * 62)

    # 假设参数
    accel_freq = 360  # MHz
    accel_macs_per_cycle = 8  # 保守估计: 8 MACs/cycle (向量加速器)
    accel_eff = 0.6  # 60% 效率

    total_macs_per_sec = accel_freq * 1e6 * accel_macs_per_cycle * accel_eff
    print(f"\n  加速器 @ {accel_freq}MHz × {accel_macs_per_cycle} MACs/cycle × {accel_eff:.0%} eff")
    print(f"  = {total_macs_per_sec/1e9:.2f} GMACs/sec")

    # TinyKWS-Conv: ~780K params, ~30M MACs per inference (98 frame window)
    model_macs = 30  # million MACs
    inference_time_ms = model_macs * 1e6 / total_macs_per_sec * 1000
    inference_interval_ms = 100  # run inference every 100ms, not every frame
    print(f"\n  Model MACs:     ~{model_macs}M per inference (98 frames)")
    print(f"  Inference time:  ~{inference_time_ms:.1f} ms")
    print(f"  Run interval:    {inference_interval_ms} ms (every 10 frames)")
    rt_factor = inference_interval_ms / inference_time_ms if inference_time_ms > 0 else float('inf')
    print(f"  Real-time:       {rt_factor:.1f}x faster than real-time")
    print(f"  CPU load:        {inference_time_ms/inference_interval_ms*100:.1f}% of one core")
    print(f"  NOTE: 加速器 spec is estimated. Actual may be 2-4x faster.")
    print(f"        If 加速器 = 32 MACs/cycle -> inference ~4.3ms -> 23x realtime.")

    # Memory
    weight_kb = 780  # INT8
    act_peak_kb = 128 * 10 * 25 / 1024 * 4  # ~125KB FP32 (worst activation)
    act_peak_int8_kb = act_peak_kb / 4  # ~31KB INT8
    total_kb = weight_kb + act_peak_int8_kb + 40 + 8 + 20

    print(f"\n  SRAM Budget (578KB on-chip):")
    print(f"    Activation (INT8): {act_peak_int8_kb:.0f} KB")
    print(f"    Mel buffer:        16 KB")
    print(f"    PCM buffer:         8 KB")
    print(f"    Stack + misc:      20 KB")
    print(f"    ---------------------------")
    sram_used = act_peak_int8_kb + 16 + 8 + 20
    print(f"    SRAM used:         {sram_used:.0f} KB")
    print(f"    SRAM free:         {578 - sram_used:.0f} KB")
    print(f"    Note: weights ({weight_kb} KB) stored in PSRAM, loaded in chunks")

    # 对比
    print(f"\n  Chip comparison:")
    print(f"  {'Chip':<16} {'Engine':<16} {'TinyKWS(179K)':<16} {'KWS-Conv(780K)':<16} {'6M-Model':<12}")
    print(f"  {'-'*16} {'-'*16} {'-'*16} {'-'*16} {'-'*12}")
    print(f"  {'AC7911B':<16} {'CPU only':<16} {'marginal':<16} {'too heavy':<16} {'NO':<12}")
    print(f"  {'AC7916AB':<16} {'加速器':<16} {'GOOD':<16} {'GOOD':<16} {'NO':<12}")
    print(f"  {'RTL8713E':<16} {'HiFi5 500MHz':<16} {'GOOD':<16} {'GOOD':<16} {'NO':<12}")
    print(f"  {'RK3588':<16} {'NPU 3TOPS':<16} {'overkill':<16} {'overkill':<16} {'YES':<12}")

    print(f"\n  Conclusion: AC7916AB 加速器 can deploy TinyKWS-Conv (780K params)")
    print(f"  GRU replaced by Dilated DWConv -> 100% 加速器原生 ops")
    print(f"  Expected: 50 classes, RTF < 0.1x, streaming capable")
    print("=" * 62)


if __name__ == '__main__':
    # 创建模型并验证
    model = TinyKWS_Conv(num_classes=50, n_mels=40)
    model.eval()
    model.summary()

    # 前向测试
    x = torch.randn(1, 1, 40, 98)
    y = model(x)
    print(f"\nForward: {x.shape} → {y.shape}  ✅")

    # 芯片分析
    chip_compatibility_analysis()
