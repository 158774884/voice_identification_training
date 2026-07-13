"""
Test report generator — creates HTML reports from batch test results.
"""
import os
from datetime import datetime
from typing import List, Dict, Optional


def generate_batch_test_report(
    results: List[Dict],
    output_path: str,
    title: str = "语音识别批量测试报告",
    model_name: str = "",
    extra_info: Optional[Dict] = None,
) -> str:
    """Generate an HTML report from batch test results.

    Args:
        results: List of per-file results from InferenceWorker
        output_path: Where to save the .html report
        title: Report title
        model_name: Name of the model tested
        extra_info: Optional dict of additional info to include

    Returns:
        Path to the generated report file
    """
    total = len(results)
    if total == 0:
        return ""

    success = [r for r in results if r.get("status") == "success"]
    errors = [r for r in results if r.get("status") == "error"]
    correct = [r for r in results if r.get("correct", False)]
    accuracy = len(correct) / len(success) * 100 if success else 0
    avg_latency = sum(r.get("latency_ms", 0) for r in success) / len(success) if success else 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: 'Microsoft YaHei', sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; color: #2c3e50; background: #f5f6fa; }}
.header {{ text-align: center; margin-bottom: 30px; }}
.header h1 {{ color: #1a73e8; margin-bottom: 4px; }}
.header p {{ color: #5f6368; margin: 2px 0; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: 12px; margin-bottom: 24px; }}
.card {{ background: white; border-radius: 10px; padding: 18px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.card .value {{ font-size: 28px; font-weight: bold; color: #1a73e8; }}
.card .label {{ font-size: 12px; color: #5f6368; margin-top: 4px; }}
.card.pass .value {{ color: #28a745; }}
.card.fail .value {{ color: #dc3545; }}
.card.warn .value {{ color: #e6a817; }}
h2 {{ color: #1a73e8; border-bottom: 2px solid #e0e4e8; padding-bottom: 6px; margin-top: 24px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; font-size: 13px; }}
th {{ background: #f8f9fa; font-weight: bold; color: #5f6368; }}
tr:hover {{ background: #e8f0fe; }}
.correct {{ color: #28a745; font-weight: bold; }}
.incorrect {{ color: #dc3545; }}
.latency {{ font-family: monospace; }}
.footer {{ text-align: center; color: #9aa0a6; font-size: 11px; margin-top: 30px; }}
"""
    # Summary cards
    html += f"""
<div class="header">
<h1>{title}</h1>
<p>模型: {model_name or 'N/A'} | 测试时间: {now}</p>
</div>
<div class="summary">
<div class="card"><div class="value">{total}</div><div class="label">总测试数</div></div>
<div class="card {'pass' if accuracy >= 85 else 'warn' if accuracy >= 60 else 'fail'}"><div class="value">{accuracy:.1f}%</div><div class="label">准确率</div></div>
<div class="card"><div class="value">{avg_latency:.0f}ms</div><div class="label">平均延迟</div></div>
<div class="card"><div class="value">{len(errors)}</div><div class="label">错误数</div></div>
</div>
"""

    # Extra info
    if extra_info:
        html += "<h2>附加信息</h2><table>"
        for k, v in extra_info.items():
            html += f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
        html += "</table>"

    # Results table
    html += "<h2>详细结果</h2><table>"
    html += "<tr><th>#</th><th>文件</th><th>识别结果</th><th>参考文本</th><th>延迟</th><th>状态</th></tr>"

    for i, r in enumerate(results):
        filename = os.path.basename(r.get("audio_path", ""))
        text = r.get("asr_text", r.get("error", "-"))
        ref = r.get("reference", "-")
        latency = r.get("latency_ms", 0)
        status = r.get("status", "error")
        is_correct = r.get("correct")

        correct_class = "correct" if is_correct else "incorrect" if is_correct is False else ""
        status_class = "correct" if status == "success" else "incorrect"

        html += f"<tr>"
        html += f"<td>{i+1}</td>"
        html += f"<td>{filename}</td>"
        html += f"<td class='{correct_class}'>{text}</td>"
        html += f"<td>{ref}</td>"
        html += f"<td class='latency'>{latency:.1f}ms</td>"
        html += f"<td class='{status_class}'>{status}</td>"
        html += "</tr>"

    html += "</table>"

    # Dialect analysis if available
    dialects = {}
    for r in success:
        d = r.get("dialect", "unknown")
        dialects[d] = dialects.get(d, 0) + 1
    if dialects:
        html += "<h2>方言分布</h2><table>"
        html += "<tr><th>方言</th><th>数量</th><th>占比</th></tr>"
        for d, count in sorted(dialects.items(), key=lambda x: -x[1]):
            pct = count / len(success) * 100
            html += f"<tr><td>{d}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"
        html += "</table>"

    html += f"<div class='footer'>由 VoiceModelTool 自动生成 | {now}</div></body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


def generate_comparison_report(
    before_results: List[Dict],
    after_results: List[Dict],
    output_path: str,
    model_before: str = "训练前模型",
    model_after: str = "训练后模型",
) -> str:
    """Generate a comparison report between two models."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calc_stats(results):
        total = len(results)
        success = [r for r in results if r.get("status") == "success"]
        correct = [r for r in results if r.get("correct", False)]
        accuracy = len(correct) / len(success) * 100 if success else 0
        avg_latency = sum(r.get("latency_ms", 0) for r in success) / len(success) if success else 0
        return total, accuracy, avg_latency

    t1, a1, l1 = calc_stats(before_results)
    t2, a2, l2 = calc_stats(after_results)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>模型对比报告</title>
<style>
body {{ font-family:'Microsoft YaHei',sans-serif; max-width:800px; margin:40px auto; padding:20px; color:#2c3e50; background:#f5f6fa; }}
.header {{ text-align:center; margin-bottom:30px; }}
.header h1 {{ color:#1a73e8; }}
table {{ width:100%; border-collapse:collapse; background:white; border-radius:8px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.08); margin-bottom:20px; }}
th, td {{ padding:10px 14px; text-align:center; border-bottom:1px solid #f0f0f0; }}
th {{ background:#f8f9fa; font-weight:bold; color:#5f6368; }}
.improvement {{ color:#28a745; font-weight:bold; }}
.degradation {{ color:#dc3545; font-weight:bold; }}
.card {{ background:white; border-radius:10px; padding:18px; text-align:center; }}
</style></head><body>
<div class="header"><h1>模型对比测试报告</h1><p>{now}</p></div>
<div class="card"><h2>核心指标对比</h2>
<table>
<tr><th>指标</th><th>{model_before}</th><th>{model_after}</th><th>变化</th></tr>
<tr><td>准确率</td><td>{a1:.1f}%</td><td>{a2:.1f}%</td>
    <td class="{'improvement' if a2>a1 else 'degradation'}">{a2-a1:+.1f}%</td></tr>
<tr><td>平均延迟</td><td>{l1:.0f}ms</td><td>{l2:.0f}ms</td>
    <td class="{'improvement' if l2<l1 else 'degradation'}">{l2-l1:+.0f}ms</td></tr>
</table></div>
<div class='footer'>由 VoiceModelTool 自动生成</div></body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
