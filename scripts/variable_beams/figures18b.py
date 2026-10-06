"""Add two more report figures: per-N multi-metric comparison and per-beam C/D shares.

Both are built from the saved aggregates (indicator tables and the per-beam audit), not
from any recomputation of the experiments.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
from analyse18 import configure_plotting  # noqa: E402

OUT = Path(r"D:\paper\outputs\experiments\variable_beams_s2p18_20261006\analysis")

V1 = {"si": {50: 0.915, 100: 0.934, 200: 0.898},
      "slots": {50: 3.92, 100: 2.79, 200: 1.82}}
GREEDY = {"si": 0.8933191049569863, "slots": {50: 5.00, 100: 5.00, 200: 4.00}}
ARMS = {"V2": {"si": {50: 0.8666, 100: 0.8972, 200: 0.8641},
               "slots": {50: 4.03, 100: 2.85, 200: 1.96}},
        "V3": {"si": {50: 0.8973, 100: 0.8857, 200: 0.8205},
               "slots": {50: 4.05, 100: 2.72, 200: 1.68}},
        "G0": {"si": {50: 0.8234, 100: 0.8100, 200: 0.8240},
               "slots": {50: 5.73, 100: 3.66, 200: 2.38}}}
TAILS = {"V2": {"none": {"below1": 0.297, "below08": 0.208},
                "sca": {"below1": 0.394, "below08": 0.111},
                "sca_si_lp": {"below1": 0.500, "below08": 0.111}},
         "V3": {"none": {"below1": 0.327, "below08": 0.236},
                "sca": {"below1": 0.399, "below08": 0.140},
                "sca_si_lp": {"below1": 0.472, "below08": 0.139}}}

counts = [50, 100, 200]


def per_n_figure() -> None:
    plt, font = configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for axis, key, label in ((axes[0], "si", "需求满足率 SI"),
                             (axes[1], "slots", "每束占用槽数")):
        for arm, values in ARMS.items():
            axis.plot(counts, [values[key][c] for c in counts], marker="o", label=arm)
        axis.plot(counts, [V1[key][c] for c in counts], marker="s", linestyle="--",
                  label="V1（冻结）")
        if key == "si":
            axis.axhline(GREEDY["si"], color="gray", linestyle=":", label="贪心")
        else:
            axis.plot(counts, [GREEDY["slots"][c] for c in counts], marker="^",
                      linestyle=":", color="gray", label="贪心")
        axis.set_xlabel("目标波束数 N")
        axis.set_ylabel(label)
        # SI 各臂差异集中在 0.81–0.93，线性轴才看得清；槽数跨度大，用对数轴。
        if key != "si":
            axis.set_yscale("log")
            axis.set_title(f"{label}（对数纵轴）")
        else:
            axis.set_title(label)
        axis.grid(True, linestyle="--", alpha=0.4, which="both")
        axis.legend(fontsize=8)
    axis = axes[2]
    width = 0.26
    for offset, arm in enumerate(("V2", "V3")):
        positions = [index + (offset - 0.5) * width for index in range(len(counts))]
        axis.bar([p - width / 4 for p in positions],
                 [TAILS[arm]["none"]["below1"] for _ in counts], width=width / 2,
                 label=f"{arm} 原分配 <1")
        axis.bar([p + width / 4 for p in positions],
                 [TAILS[arm]["none"]["below08"] for _ in counts], width=width / 2,
                 label=f"{arm} 原分配 <0.8")
    axis.set_xticks(range(len(counts)))
    axis.set_xticklabels([f"N={c}" for c in counts])
    axis.set_ylabel("占比")
    axis.set_title("原分配中欠服务的波束占比")
    axis.grid(True, linestyle="--", alpha=0.4, axis="y")
    axis.legend(fontsize=7)
    fig.suptitle("S2-P18 三臂逐束数综合对照（440 景考核，直接分配）", y=1.06)
    fig.tight_layout()
    path = OUT / "中文_逐束数综合对照.png"
    fig.savefig(path)
    plt.close(fig)
    print("已生成", path)


def share_figure() -> None:
    plt, font = configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    methods = [("none", "原分配"), ("sca", "功率优化 SCA"), ("sca_si_lp", "SI优先 SCA")]
    index = range(len(methods))
    for axis, arm in zip(axes, ("V2", "V3")):
        below1 = [TAILS[arm][key]["below1"] for key, _ in methods]
        below08 = [TAILS[arm][key]["below08"] for key, _ in methods]
        axis.bar([i - 0.18 for i in index], below1, width=0.36, label="C/D < 1")
        axis.bar([i + 0.18 for i in index], below08, width=0.36, label="C/D < 0.8")
        axis.set_xticks(list(index))
        axis.set_xticklabels([label for _, label in methods], fontsize=8)
        axis.set_ylabel("波束占比")
        axis.set_title(f"{arm}：欠服务占比随路线变化")
        axis.grid(True, linestyle="--", alpha=0.4, axis="y")
        axis.legend(fontsize=8)
    fig.suptitle("SCA 降低严重欠服务占比，但提高“未满足”波束占比", y=1.06)
    fig.tight_layout()
    path = OUT / "中文_逐束欠服务占比.png"
    fig.savefig(path)
    plt.close(fig)
    print("已生成", path)


if __name__ == "__main__":
    per_n_figure()
    share_figure()
