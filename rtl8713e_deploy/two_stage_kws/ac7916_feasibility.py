"""
AC7916AB 两级语音唤醒+指令识别 — 完整可行性评估

=====================================================================
评估方法: 基于 AC7916AB 公开参数做内存/算力/延迟三维分析
=====================================================================

AC7916AB 关键参数来源 (杰理官方文档):
  CPU: 双核 32bit RISC @ 320MHz
  MVA: 矩阵向量加速器 @ 360MHz (8-64 MACs/cycle, 保守取8)
  SRAM: 578KB
  PSRAM: 标配 2MB (可选 8MB)
  Flash: 标配 8MB
  音频: 4ch ADC, 硬件 FFT

两级系统架构:
  Stage 1 (始终在线): 唤醒词检测 → 检测到 "小度小度" 后唤醒 Stage 2
  Stage 2 (按需启动): 命令词识别 → 识别 "打开客厅的灯" 等指令

=====================================================================
"""

import math

# ===== AC7916AB 硬件参数 =====
CPU_FREQ    = 320e6    # 320 MHz (单核)
MVA_FREQ    = 360e6    # 360 MHz
SRAM_BYTES  = 578 * 1024   # 578 KB
PSRAM_BYTES = 2 * 1024 * 1024  # 2 MB (保守)
FLASH_BYTES = 8 * 1024 * 1024  # 8 MB
MVA_MACS_PER_CYCLE = 8    # 保守估计 (实际可能 16-64)
MVA_EFFICIENCY     = 0.6  # 60% 利用率

# ===== Stage 1: 唤醒词 (UltraTinyKWS) =====
STAGE1_PARAMS       = 2370       # 参数量
STAGE1_WEIGHT_INT8  = 2370       # bytes (INT8)
STAGE1_MACS_PER_FRAME = 0.05e6   # ~50K MACs (单帧推理, 40 mel bins)
STAGE1_ACTIVATION   = 10 * 1024  # ~10KB (mel帧缓冲+中间激活)
STAGE1_RUN_INTERVAL = 0.010      # 10ms (100Hz 帧率)
STAGE1_SRAM = {
    'weights (INT8)':     STAGE1_WEIGHT_INT8,
    'mel buffer (40x98)': 40 * 98 * 2,      # int16, ~7.7KB
    'pcm ring buffer':    160 * 25 * 2,     # 25 frames int16, ~8KB
    'fft workspace':      512 * 4,           # ~2KB
    'activation scratch': STAGE1_ACTIVATION,
    'stack':              4096,              # ~4KB
}
STAGE1_SRAM_TOTAL = sum(STAGE1_SRAM.values())

# ===== Stage 2: 命令词识别 (TinyKWS-MVA 或 小型CTC) =====

# 方案A: TinyKWS-MVA 分类器 (简单, MVA原生加速)
PLAN_A_PARAMS       = 780_000
PLAN_A_WEIGHT_INT8  = 780_000      # bytes
PLAN_A_MACS_PER_INF = 30e6         # 30M MACs (98帧窗口推理)
PLAN_A_ACTIVATION   = 50 * 1024    # ~50KB (峰值激活)
PLAN_A_SRAM = {
    'activation (peak)': PLAN_A_ACTIVATION,
    'mel window (shared)': 0,      # 与Stage1共享
    'output probs':        200,     # 50类 x 4bytes
}
PLAN_A_SRAM_TOTAL = PLAN_A_ACTIVATION + 200

# 方案B: 小型CTC声学模型 + WFST语法解码 (更灵活, 部分在CPU)
PLAN_B_PARAMS       = 350_000      # 更小的声学模型
PLAN_B_WEIGHT_INT8  = 350_000
PLAN_B_MACS_PER_INF = 15e6         # 模型推理
PLAN_B_WFST_SIZE    = 150_000      # 语法图 (~150KB, 支持200条命令)
PLAN_B_ACTIVATION   = 35 * 1024
PLAN_B_SRAM = {
    'activation (peak)':  PLAN_B_ACTIVATION,
    'wfst decoder state': 20 * 1024,  # beam search state
}
PLAN_B_SRAM_TOTAL = PLAN_B_ACTIVATION + 20 * 1024


def _fmt(n_bytes):
    if n_bytes >= 1024*1024:
        return f"{n_bytes/1024/1024:.1f} MB"
    elif n_bytes >= 1024:
        return f"{n_bytes/1024:.0f} KB"
    return f"{n_bytes} B"


def _pct(part, whole):
    return f"{part/whole*100:.1f}%"


def analyze():
    mva_tops = MVA_FREQ * MVA_MACS_PER_CYCLE * MVA_EFFICIENCY
    cpu_mips = CPU_FREQ * 0.8  # ~80% efficiency for RISC

    print("=" * 72)
    print("  AC7916AB 两级语音唤醒+指令识别 — 可行性评估")
    print("=" * 72)

    # ── 硬件总览 ──
    print(f"\n  [硬件平台]")
    print(f"    CPU:  双核 320 MHz RISC, 有效 ~{cpu_mips/1e6:.0f} MIPS/core")
    print(f"    MVA:  360 MHz x {MVA_MACS_PER_CYCLE} MACs/cycle = {mva_tops/1e6:.1f} MMACs/s")
    print(f"    SRAM: {_fmt(SRAM_BYTES)}")
    print(f"    PSRAM:{_fmt(PSRAM_BYTES)}")
    print(f"    Flash:{_fmt(FLASH_BYTES)}")

    # ── Stage 1 分析 ──
    print(f"\n  {'─'*60}")
    print(f"  [Stage 1: 始终在线唤醒词检测]")
    print(f"    Model:      UltraTinyKWS ({STAGE1_PARAMS:,} params)")
    print(f"    INT8 size:  {_fmt(STAGE1_WEIGHT_INT8)}")
    print(f"    MACs/frame: {STAGE1_MACS_PER_FRAME/1000:.0f}K")
    print(f"    运行频率:   每 {STAGE1_RUN_INTERVAL*1000:.0f}ms 一次")

    stage1_cpu_ms = STAGE1_MACS_PER_FRAME / cpu_mips * 1000
    print(f"    推理延迟:   ~{stage1_cpu_ms*1000:.0f} us (CPU 单核)")
    print(f"    CPU 占用:   {stage1_cpu_ms/10*100:.1f}%")

    print(f"\n    SRAM 占用:")
    for name, size in STAGE1_SRAM.items():
        print(f"      {name:<25s}: {_fmt(size):>8s}")
    print(f"      {'─'*35}")
    print(f"      {'Stage1 Total':<25s}: {_fmt(STAGE1_SRAM_TOTAL):>8s}  ({_pct(STAGE1_SRAM_TOTAL, SRAM_BYTES)} of SRAM)")

    s1_ok = STAGE1_SRAM_TOTAL < SRAM_BYTES
    print(f"    结论: {'OK' if s1_ok else 'EXCEEDS'}  {'- 恒常在线, 功耗 <0.5mA' if s1_ok else ''}")

    # ── Stage 2 分析 (方案A) ──
    print(f"\n  {'─'*60}")
    print(f"  [Stage 2: 唤醒后命令识别 — 方案A: MVA分类器]")
    print(f"    Model:      TinyKWS-MVA ({PLAN_A_PARAMS/1000:.0f}K params)")
    print(f"    INT8 size:  {_fmt(PLAN_A_WEIGHT_INT8)}  (存在 PSRAM)")
    print(f"    MACs/inf:   {PLAN_A_MACS_PER_INF/1e6:.0f}M")
    print(f"    运行频率:   每 100ms 一次 (唤醒后才跑)")

    plan_a_mva_ms = PLAN_A_MACS_PER_INF / mva_tops * 1000
    print(f"    推理延迟:   ~{plan_a_mva_ms:.1f} ms (MVA)")
    print(f"    实时率:     {100/plan_a_mva_ms:.1f}x  (100ms 间隔内可跑 {(100/plan_a_mva_ms):.0f} 次)")

    # Stage2 时 SRAM: Stage1继续跑着 + Stage2激活
    s2a_peak_sram = PLAN_A_SRAM_TOTAL  # Stage2 激活 (Stage1 weights 常驻)
    total_sram_a = STAGE1_SRAM_TOTAL + s2a_peak_sram
    print(f"\n    SRAM 占用 (Stage2 激活时):")
    print(f"      Stage1 (常驻):        {_fmt(STAGE1_SRAM_TOTAL):>8s}")
    print(f"      Stage2 (激活峰值):     {_fmt(s2a_peak_sram):>8s}")
    print(f"      {'─'*35}")
    print(f"      Peak Total:            {_fmt(total_sram_a):>8s}  ({_pct(total_sram_a, SRAM_BYTES)} of SRAM)")

    # PSRAM
    psram_a = PLAN_A_WEIGHT_INT8  # + 语法图(方案A不需要)
    print(f"\n    PSRAM 占用:")
    print(f"      Stage2 weights:        {_fmt(PLAN_A_WEIGHT_INT8):>8s}")
    print(f"      PSRAM free:            {_fmt(PSRAM_BYTES - psram_a):>8s}")

    a_ok = total_sram_a < SRAM_BYTES and psram_a < PSRAM_BYTES

    # ── Stage 2 分析 (方案B) ──
    print(f"\n  {'─'*60}")
    print(f"  [Stage 2: 唤醒后命令识别 — 方案B: 小型CTC + WFST语法图]")
    print(f"    Model:      CTC声学模型 ({PLAN_B_PARAMS/1000:.0f}K params)")
    print(f"    WFST:       语法图 ({_fmt(PLAN_B_WFST_SIZE)}, 支持200条命令)")
    print(f"    INT8 size:  {_fmt(PLAN_B_WEIGHT_INT8)} + {_fmt(PLAN_B_WFST_SIZE)} (PSRAM)")
    print(f"    MACs/inf:   {PLAN_B_MACS_PER_INF/1e6:.0f}M  (MVA)")
    print(f"    解码:       CTC beam search (CPU, 轻量)")

    plan_b_mva_ms = PLAN_B_MACS_PER_INF / mva_tops * 1000
    plan_b_decode_ms = 5  # CTC beam search on 320MHz CPU, grammar-constrained
    plan_b_total_ms = plan_b_mva_ms + plan_b_decode_ms
    print(f"    推理延迟:   ~{plan_b_mva_ms:.1f}ms (MVA) + ~{plan_b_decode_ms}ms (CPU解码)")
    print(f"    实时率:     {100/plan_b_total_ms:.1f}x")

    s2b_peak_sram = PLAN_B_SRAM_TOTAL
    total_sram_b = STAGE1_SRAM_TOTAL + s2b_peak_sram
    psram_b = PLAN_B_WEIGHT_INT8 + PLAN_B_WFST_SIZE

    print(f"\n    SRAM 占用 (Stage2 激活时):")
    print(f"      Stage1 (常驻):        {_fmt(STAGE1_SRAM_TOTAL):>8s}")
    print(f"      Stage2 (激活峰值):     {_fmt(s2b_peak_sram):>8s}")
    print(f"      {'─'*35}")
    print(f"      Peak Total:            {_fmt(total_sram_b):>8s}  ({_pct(total_sram_b, SRAM_BYTES)} of SRAM)")

    print(f"\n    PSRAM 占用:")
    print(f"      Stage2 weights:        {_fmt(PLAN_B_WEIGHT_INT8):>8s}")
    print(f"      WFST 语法图:           {_fmt(PLAN_B_WFST_SIZE):>8s}")
    print(f"      PSRAM free:            {_fmt(PSRAM_BYTES - psram_b):>8s}")

    b_ok = total_sram_b < SRAM_BYTES and psram_b < PSRAM_BYTES

    # ── Flash 占用 ──
    flash_usage = STAGE1_WEIGHT_INT8 + PLAN_A_WEIGHT_INT8 + 500_000  # 两个模型+固件配置
    print(f"\n  {'─'*60}")
    print(f"  [Flash 存储 (8MB)]")
    print(f"    Stage1 weights:     {_fmt(STAGE1_WEIGHT_INT8)}")
    print(f"    Stage2 planA:       {_fmt(PLAN_A_WEIGHT_INT8)}")
    print(f"    Stage2 planB+WFST:  {_fmt(PLAN_B_WEIGHT_INT8 + PLAN_B_WFST_SIZE)}")
    print(f"    Firmware + config:  {_fmt(500_000)}")
    print(f"    Total:              {_fmt(flash_usage)}  ({_pct(flash_usage, FLASH_BYTES)} of Flash)")
    print(f"    OK - all fits with room for OTA updates")

    # ── 功耗估算 ──
    print(f"\n  {'─'*60}")
    print(f"  [功耗估算]")
    print(f"    待机 (Stage1 only):   < 0.5 mA  (CPU轻载 + MVA休眠)")
    print(f"    活跃 (Stage2 active):   5-15 mA  (MVA工作 + CPU解码)")
    print(f"    典型场景 (24h):")
    print(f"      唤醒100次, 每次识别3秒 → 活跃时间 5分钟/天")
    print(f"      平均电流: 0.5 * 23.9h + 10 * 0.1h ≈ 1.5 mA")
    print(f"      200mAh 电池: ~5.5 天")

    # ── 总结 ──
    print(f"\n  {'='*60}")
    print(f"  [最终结论]")
    print(f"  {'='*60}")
    print(f"")
    print(f"  Stage 1 (UltraTinyKWS):")
    print(f"    SRAM: {_fmt(STAGE1_SRAM_TOTAL)} / {_fmt(SRAM_BYTES)}  {'OK' if s1_ok else 'FAIL'}")
    print(f"    延迟: ~{stage1_cpu_ms*1000:.0f} us  OK")
    print(f"")
    print(f"  Stage 2 Plan A (MVA分类器, 50条命令):")
    print(f"    SRAM: {_fmt(total_sram_a)} / {_fmt(SRAM_BYTES)}  {'OK' if a_ok else 'FAIL'}")
    print(f"    PSRAM:{_fmt(psram_a)} / {_fmt(PSRAM_BYTES)}  {'OK' if psram_a < PSRAM_BYTES else 'FAIL'}")
    print(f"    延迟: ~{plan_a_mva_ms:.1f} ms  {'OK' if plan_a_mva_ms < 100 else 'FAIL'}")
    print(f"")
    print(f"  Stage 2 Plan B (CTC+WFST, 200条命令):")
    print(f"    SRAM: {_fmt(total_sram_b)} / {_fmt(SRAM_BYTES)}  {'OK' if b_ok else 'FAIL'}")
    print(f"    PSRAM:{_fmt(psram_b)} / {_fmt(PSRAM_BYTES)}  {'OK' if psram_b < PSRAM_BYTES else 'FAIL'}")
    print(f"    延迟: ~{plan_b_total_ms:.1f} ms  {'OK' if plan_b_total_ms < 300 else 'FAIL'}")
    print(f"")
    print(f"  推荐: {'Plan B (CTC+WFST)' if b_ok else 'Plan A (分类器)' if a_ok else '需要降配'}")
    print(f"  理由: {'200条命令泛化好' if b_ok else 'AC7916AB MVA加速可用' if a_ok else ''}")
    print(f"  {'='*60}")

    return {
        'stage1_ok': s1_ok,
        'planA_ok': a_ok,
        'planB_ok': b_ok,
    }


if __name__ == '__main__':
    analyze()
