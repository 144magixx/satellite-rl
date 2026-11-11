# -*- coding: utf-8 -*-
"""W01 图5：三种网络的原公式和修正公式（正目标熵＋完整联合动作取双Q最小值）训练40万步后的对比。
数字取自 docs/资料/06_中期研究进展与路线调整_导师讨论版.md 附录乙“三网络与学习公式对照”表
（同100个场景、每景3次随机动作；原报告 sac_comparison_20260924_r2 目录本机已清理，按该表重画）。
斜线柱表示该组100%的回合超过了功率预算而提前结束。
运行：python outputs/weekly_reports_2026/figure_scripts/W01_公式修正前后对比.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt = setup_mpl()

nets = ["T-SAC", "CNN版SAC", "MLP版SAC"]
si_old = [67.18, 52.53, 51.92]
si_new = [62.97, 41.45, 47.39]
sg_old = [0.6949, 0.5422, 0.5399]
sg_new = [0.6786, 0.3975, 0.4645]
over_old = [True, True, True]
over_new = [False, True, False]
g_si, f_si, g_sg, f_sg = 68.48, 70.90, 0.6356, 0.5821  # 贪心、固定规则

fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.5))
x = np.arange(3)
w = 0.36
for ax, old, new, g, f, ttl, fmt, ylim in [
        (axs[0], si_old, si_new, g_si, f_si, "需求满足率（%）", "{:.1f}", (0, 90)),
        (axs[1], sg_old, sg_new, g_sg, f_sg, "供需匹配程度（0到1）", "{:.2f}", (0, 0.92))]:
    for i in range(3):
        for dx, v, ov, c in [(-w / 2, old[i], over_old[i], C_ORANGE), (w / 2, new[i], over_new[i], C_BLUE)]:
            ax.bar(x[i] + dx, v, w, color=c, hatch="////" if ov else None, edgecolor="white" if ov else c, lw=0)
            ax.text(x[i] + dx, v + ylim[1] * 0.012, fmt.format(v), ha="center", fontsize=8.5, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
    ax.axhline(g, color="#333", ls="--", lw=1.1)
    ax.axhline(f, color=C_GREY, ls=":", lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(nets)
    ax.set_ylim(*ylim)
    ax.set_title(ttl)
    ax.grid(axis="x", visible=False)
h = [Patch(color=C_ORANGE, label="原公式"), Patch(color=C_BLUE, label="修正公式"),
     Patch(facecolor="#bbb", hatch="////", edgecolor="white", label="斜线：回合超功率"),
     Line2D([], [], color="#333", ls="--", lw=1.1, label="贪心（虚线）"),
     Line2D([], [], color=C_GREY, ls=":", lw=1.4, label="固定分配（点线）")]
fig.legend(handles=h, ncol=3, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)
fig.tight_layout(rect=(0, 0.13, 1, 1))
save_fig(fig, "W01_公式修正前后对比.png")
