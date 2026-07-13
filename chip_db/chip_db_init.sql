-- Chip Database Initialization
-- Pre-populated with common embedded voice/AI chips

CREATE TABLE IF NOT EXISTS chips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    manufacturer TEXT NOT NULL,
    architecture TEXT NOT NULL,        -- MCU, SoC, NPU, DSP
    cpu_cores INTEGER DEFAULT 1,
    cpu_freq_mhz INTEGER DEFAULT 100,
    ram_kb INTEGER DEFAULT 128,        -- SRAM/DTCM in KB
    flash_kb INTEGER DEFAULT 1024,     -- Flash in KB
    npu_tops REAL DEFAULT 0.0,         -- NPU TOPS (0 = no NPU)
    dsp INTEGER DEFAULT 0,             -- Has DSP (boolean)
    supported_quant TEXT DEFAULT '[]',  -- JSON array: ["INT8","INT16","FP16"]
    max_model_size_kb INTEGER DEFAULT 1024,
    power_consumption_mw INTEGER DEFAULT 500,
    supported_ops TEXT DEFAULT '[]',    -- JSON array of supported ONNX ops
    price_cny REAL DEFAULT 0.0,
    notes TEXT DEFAULT ''
);

-- ================================================================
-- Initial Data: 15 widely-used embedded voice/AI chips
-- ================================================================

INSERT OR IGNORE INTO chips (name, manufacturer, architecture, cpu_cores, cpu_freq_mhz,
    ram_kb, flash_kb, npu_tops, dsp, supported_quant, max_model_size_kb,
    power_consumption_mw, supported_ops, price_cny, notes)
VALUES
-- High-end NPU SoCs
('RK3588', 'Rockchip', 'SoC', 8, 2400, 8388608, 33554432, 6.0, 1,
 '["INT8","INT16","FP16"]', 10240, 5000,
 '["Conv","Conv1d","BatchNorm","ReLU","Sigmoid","Tanh","GRU","Gemm","MatMul","Add","Mul","Concat","Softmax","AveragePool","GlobalAveragePool","Reshape","Transpose","Gather","Unsqueeze","Squeeze"]',
 350, '8核旗舰 SoC, 3 TOPS NPU, 适合高性能语音识别'),

('A311D', 'Amlogic', 'SoC', 6, 2200, 4194304, 16777216, 5.0, 1,
 '["INT8","INT16"]', 20480, 5000,
 '["Conv","Conv1d","BN","ReLU","GRU","FC","Softmax","Pool","Concat","Eltwise"]',
 280, '5 TOPS NPU, 适合边缘AI音箱'),

('Hi3559A', 'HiSilicon', 'SoC', 4, 2000, 2097152, 8388608, 4.0, 1,
 '["INT8"]', 8192, 4000,
 '["Conv","BN","ReLU","FC","Pool","Concat","Eltwise"]',
 250, 'NNIE 4 TOPS, 需替换GRU为Conv1D'),

('RV1126', 'Rockchip', 'SoC', 4, 1500, 1048576, 4194304, 1.0, 1,
 '["INT8"]', 4096, 2000,
 '["Conv","Conv1d","BN","ReLU","FC","Pool","Concat","Eltwise","Softmax"]',
 80, '1 TOPS NPU, 适合低功耗设备'),

-- Dedicated AI Accelerator MCUs
('AC7916AB', 'JieLi', 'MCU', 2, 320, 578, 8192, 0.0, 0,
 '["INT8"]', 400, 15,
 '["Conv2D","DWConv","FC","ReLU","Pool","Concat","Eltwise"]',
 12, 'MVA加速器, 360MHz, 适合超低功耗KWS'),

('RTL8713E', 'Realtek', 'SoC', 2, 500, 768, 16384, 0.0, 1,
 '["INT8","INT16"]', 512, 300,
 '["Conv2D","DWConv","FC","ReLU","Pool","Concat","Softmax"]',
 25, 'HiFi 5 DSP 500MHz, Wi-Fi 6, 适合智能IoT'),

-- MCU-class (no NPU, lightweight models only)
('ESP32-S3', 'Espressif', 'MCU', 2, 240, 512, 16384, 0.0, 0,
 '["INT8"]', 256, 300,
 '["Conv2D","DWConv","FC","ReLU","Pool","Softmax"]',
 15, '双核Xtensa LX7, 512KB SRAM, 内置Wi-Fi/BT'),

('STM32H743', 'STMicro', 'MCU', 1, 480, 1024, 2048, 0.0, 0,
 '["INT8"]', 512, 200,
 '["Conv2D","FC","ReLU","Pool","Softmax"]',
 45, 'Cortex-M7 480MHz, 1MB SRAM, 适合轻量KWS'),

('K210', 'Kendryte', 'MCU', 1, 400, 8192, 16384, 0.5, 0,
 '["INT8","INT16"]', 4096, 500,
 '["Conv2D","DWConv","FC","ReLU","Pool","Concat","Softmax"]',
 35, 'RISC-V 64双核, KPU 0.5 TOPS, 低成本AI视觉'),

('BK7258', 'Beken', 'MCU', 1, 240, 512, 4096, 0.0, 0,
 '["INT8"]', 256, 150,
 '["Conv2D","DWConv","FC","ReLU","Pool"]',
 8, 'ARM968 240MHz, 集成Wi-Fi, 适合超低成本IoT'),

-- Mid-range
('i.MX RT1060', 'NXP', 'MCU', 1, 600, 1024, 8192, 0.0, 0,
 '["INT8"]', 768, 300,
 '["Conv2D","DWConv","FC","ReLU","Pool","Concat","Softmax"]',
 60, 'Cortex-M7 600MHz, 1MB SRAM, 适合中端KWS'),

('BL602', 'Bouffalo', 'MCU', 1, 192, 276, 2048, 0.0, 0,
 '["INT8"]', 128, 120,
 '["Conv2D","DWConv","FC","ReLU","Pool"]',
 6, 'RISC-V 192MHz, 超低成本Wi-Fi/BT'),

-- DSP-focused
('CSK4002', 'XMOS', 'DSP', 2, 500, 256, 2048, 0.0, 1,
 '["INT8","INT16"]', 256, 200,
 '["Conv2D","DWConv","FC","ReLU","Pool","Concat"]',
 20, 'xCORE DSP 500MHz, 超低延迟音频处理'),

-- High-end edge
('Jetson Nano', 'NVIDIA', 'SoC', 4, 1430, 4194304, 16777216, 0.47, 0,
 '["INT8","FP16","FP32"]', 20480, 10000,
 '["Conv","Conv1d","BatchNorm","ReLU","GRU","Gemm","MatMul","Add","Mul","Concat","Softmax","Pool","Reshape","Transpose","Gather","Unsqueeze","Squeeze"]',
 600, 'Maxwell GPU 128核, 适合边缘AI全栈部署'),

('AX650A', 'Axera', 'SoC', 4, 1500, 2097152, 8388608, 3.6, 1,
 '["INT8","INT16","FP16"]', 8192, 3000,
 '["Conv","Conv1d","BatchNorm","ReLU","GRU","Gemm","MatMul","Add","Mul","Concat","Softmax","Pool","Reshape","Transpose"]',
 120, '3.6 TOPS NPU, 高性价比边缘AI');
