"""
Datasheet parser tests — verify offline extraction of chip specs from text.

Usage:
    python tests/test_datasheet_parser.py   # self-contained run
    pytest tests/test_datasheet_parser.py
"""
import os
import sys
import shutil
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.utils.datasheet_parser import parse_text, parse_datasheet


SAMPLE = """
NovaX1000 Series Datasheet

Manufacturer: Rockchip
Processor: Arm Cortex-M55 dual-core @ 480 MHz
Memory: 512 KB SRAM, 4 MB Flash
Neural Processing Unit (NPU): 2.5 TOPS
Integrated HiFi DSP
Power: 350 mW typical
Supports INT8 / INT16 quantization
"""


def test_parse_text_extracts_specs():
    result = parse_text(SAMPLE, filename="NovaX1000_datasheet.txt")
    chip = result.chip

    assert chip.name == "NovaX1000"
    assert chip.manufacturer == "Rockchip"
    assert chip.cpu_freq_mhz == 480
    assert chip.cpu_cores == 2
    assert chip.ram_kb == 512
    assert chip.flash_kb == 4096
    assert chip.npu_tops == 2.5
    assert chip.power_consumption_mw == 350
    assert chip.architecture == "MCU"
    assert chip.dsp is True
    assert result.confidence >= 0.7


def test_parse_text_unit_conversions():
    text = "1.2 GHz CPU, 1 MB RAM, 500 GOPS NPU, 1.5 W power"
    result = parse_text(text, filename="chip.txt")
    chip = result.chip
    assert chip.cpu_freq_mhz == 1200
    assert chip.ram_kb == 1024
    assert chip.npu_tops == 0.5
    assert chip.power_consumption_mw == 1500


def test_parse_datasheet_txt_file():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "chip_datasheet.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(SAMPLE)
        result = parse_datasheet(path)
        assert result.chip.cpu_freq_mhz == 480
        assert result.chip.flash_kb == 4096
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


AC7916_SAMPLE = """
AC7916AB 规格书
杰理 JieLi

处理器: 双核 32bit RISC @ 320MHz
片上 SRAM: 578KB
PSRAM: 2MB (可选 8MB)
Flash: 8MB
支持蓝牙 2.4GHz
工作电压: 3.3V
工作电流: 15mA, 待机 0.5mA
"""


def test_parse_ac7916_like_datasheet():
    result = parse_text(AC7916_SAMPLE, filename="AC7916AB_datasheet.txt")
    chip = result.chip
    # 关键: 不能把 2.4GHz 蓝牙频段当 CPU 主频
    assert chip.cpu_freq_mhz == 320
    # 不能把 2MB PSRAM 当片上 SRAM
    assert chip.ram_kb == 578
    assert chip.flash_kb == 8192
    assert chip.cpu_cores == 2
    assert chip.manufacturer == "JieLi"
    assert chip.npu_tops == 0.0
    assert chip.power_consumption_mw == 49 or chip.power_consumption_mw == 50
    assert "MVA" not in chip.notes


def main():
    test_parse_text_extracts_specs()
    test_parse_text_unit_conversions()
    test_parse_datasheet_txt_file()
    test_parse_ac7916_like_datasheet()
    print("All datasheet parser tests passed.")


if __name__ == "__main__":
    main()
