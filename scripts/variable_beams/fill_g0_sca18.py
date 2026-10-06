"""Fill report 14's G0 placeholders from the saved SCA result files.

Run once ``g0_sca_per_method.json`` (and optionally ``beam_tails_G0.json``) exist; the
numbers are read from those files so the report always matches its recorded source.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ANALYSIS = Path(r"D:\paper\outputs\experiments\variable_beams_s2p18_20261006\analysis")
REPORT = Path(r"D:\paper\docs\第二阶段\实验报告\14_可变束数泛化第二轮S2-P18三臂对照实验报告.md")
LABELS = {"none": "原分配（不调整）", "sca": "功率优化 SCA", "sca_si_lp": "SI优先 SCA"}


def replace_block(text: str, begin: str, end: str, payload: str) -> str:
    start = text.index(begin)
    stop = text.index(end) + len(end)
    return text[:start] + payload + text[stop:]


def main() -> None:
    sca_path = ANALYSIS / "g0_sca_per_method.json"
    tails_path = ANALYSIS / "beam_tails_G0.json"
    if not sca_path.is_file():
        raise SystemExit("g0_sca_per_method.json 尚未生成，暂不填数")
    sca = json.loads(sca_path.read_text(encoding="utf-8-sig"))["methods"]
    power_route = sca.get("sca") or {}
    si_route = sca.get("sca_si_lp") or {}
    baseline = sca.get("none") or {}

    text = REPORT.read_text(encoding="utf-8")
    note = (f"自 {baseline.get('si')} 升至 {power_route.get('si')}（+"
            f"{(power_route.get('si') or 0) - (baseline.get('si') or 0):.4f}）")
    text = replace_block(text, "<!--G0-SCA-SCA-->", "<!--/G0-SCA-SCA-->",
                         f"{power_route.get('si')}")
    text = replace_block(text, "<!--G0-SCA-SI-->", "<!--/G0-SCA-SI-->",
                         f"{si_route.get('si')}")
    text = replace_block(text, "<!--G0-SCA-NOTE-->", "<!--/G0-SCA-NOTE-->", note)
    text = replace_block(text, "<!--G0-SCA-STATUS-->", "<!--/G0-SCA-STATUS-->",
                         f"（G0 的 SCA 已完成：原分配 SI {baseline.get('si')} → "
                         f"功率优化 {power_route.get('si')}／SI优先 {si_route.get('si')}，"
                         f"功率 {power_route.get('power_w')} W）")

    if tails_path.is_file():
        tails = json.loads(tails_path.read_text(encoding="utf-8-sig"))["methods"]
        rows = []
        for method in ("none", "sca", "sca_si_lp"):
            entry = tails.get(method)
            if not entry:
                continue
            rows.append(f"| G0 | {LABELS[method]} | {entry['ratio_mean']:.3f} | "
                        f"{entry['ratio_min']:.3f} | {entry['sample_p50']:.3f} | "
                        f"{entry['ratio_max']:.3f} | {entry['share_below_1']:.1%} | "
                        f"{entry['share_below_0_8']:.1%} |")
        text = replace_block(text, "<!--G0-TAILS-ROW-->", "<!--/G0-TAILS-ROW-->",
                            "\n".join(rows))
        print("3.5 节已填入 G0 的三行")
    else:
        print("beam_tails_G0.json 尚未生成，3.5 节保持占位")

    REPORT.write_text(text, encoding="utf-8")
    print("报告 3.4 与第 9 节的 G0 数值已填入")


if __name__ == "__main__":
    sys.exit(main())
