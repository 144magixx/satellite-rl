# -*- coding: utf-8 -*-
"""W02 四张按报告表格数字重画的图（逐回合原始记录本机没有）。

数字来源（都在 docs/第二阶段/实验报告/ 下）：
  01_四策略20随机场景对照分析报告.md
    第3节表格：平均需求满足率（SI）、供需匹配程度（SGM）、达到需求10%的波束数、零槽动作比例、每束实际槽数、实际总功率；
    第4节正文：旧奖励/融合奖励每束请求槽数 6.67/6.17、前50束与后50束零槽比例（融合 14.60%/33.43%，旧奖励 4.23%/14.70%）。
  02_原T-SAC联合掩码对照实验.md
    “2026-09-26 正式终点结果”表格（100个新考核场景）：四种方法的 SI、SGM、达标束数、零容量束数、实际总功率；
    正文：联合掩码相对无掩码的配对平均差（+9.73个百分点、+10.52个百分点、+10.20束、功率+507.60 W）、
    相对固定策略少用 471.98 W、全体100束达标的次数（联合掩码 106/300，固定 40/100，无掩码 0/300，贪心 0/100）。
输出：figures/W02_四种策略平均成绩.png、W02_融合奖励的零槽问题.png、W02_联合掩码前后对比.png、W02_联合掩码后的全体达标与功率.png
运行：python outputs/weekly_reports_2026/figure_scripts/W02_表格数字重画.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np

plt = setup_mpl()

BLUE, ORANGE, GREEN, PURPLE, GOLD = "#2070b4", "#d66a31", "#269368", "#8b55a3", "#e9b44c"


def bars(ax, labels, vals, cols, title, fmt="{:.1f}", ylim=None, unit_pad=0.02):
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.62, color=cols)
    top = ylim[1] if ylim else max(vals) * 1.18
    ax.set_ylim(0, top)
    for i, v in enumerate(vals):
        ax.text(i, v + top * unit_pad, fmt.format(v + 1e-9), ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.grid(axis="x", visible=False)


# ---------- 图A：四种策略平均成绩（报告01第3节表）----------
def fig_overview():
    labels = ["旧奖励\nT-SAC", "融合奖励\nT-SAC", "贪心", "固定策略"]
    cols = [BLUE, ORANGE, GREEN, PURPLE]
    si = [65.98, 35.31, 67.31, 70.94]
    sgm = [69.36, 32.35, 62.05, 57.58]
    ok = [89.47, 74.03, 95.50, 97.70]
    pw = [2093.58, 450.00, 2418.25, 3000.00]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.0))
    bars(axs[0, 0], labels, si, cols, "需求满足率（%）", ylim=(0, 90))
    bars(axs[0, 1], labels, sgm, cols, "供需匹配度（%）", ylim=(0, 90))
    bars(axs[1, 0], labels, ok, cols, "基本服务达标束数（共100束）", ylim=(0, 118))
    bars(axs[1, 1], labels, pw, cols, "实际总功率（W）", fmt="{:.0f}", ylim=(0, 3600))
    axs[1, 1].axhline(3000, color="#555", ls="--", lw=1)
    axs[1, 1].text(-0.42, 3060, "上限3000 W", ha="left", va="bottom", fontsize=8.5, color="#555")
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    save_fig(fig, "W02_四种策略平均成绩.png")


# ---------- 图B：融合奖励的零槽问题（报告01第3、4节）----------
def fig_zero_slot():
    labels4 = ["旧奖励\nT-SAC", "融合奖励\nT-SAC", "贪心", "固定策略"]
    cols4 = [BLUE, ORANGE, GREEN, PURPLE]
    zero = [9.47, 24.02, 2.00, 0.0]
    real_slots = [5.68, 3.92, 4.90, 7.84]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.0))
    bars(axs[0, 0], labels4, zero, cols4, "零槽动作比例（%）", ylim=(0, 30))
    bars(axs[0, 1], labels4, real_slots, cols4, "每束实际分到的频率槽数", fmt="{:.2f}", ylim=(0, 9.5))
    # 请求槽数 vs 实际槽数
    ax = axs[1, 0]
    x = np.arange(2)
    w = 0.34
    req = [6.67, 6.17]
    act = [5.68, 3.92]
    ax.bar(x - w / 2, req, w, color=["#9bbfe0", "#efb593"])
    ax.bar(x + w / 2, act, w, color=[BLUE, ORANGE])
    for i in range(2):
        ax.text(i - w / 2, req[i] + 0.12, f"{req[i]:.2f}", ha="center", fontsize=9.5)
        ax.text(i + w / 2, act[i] + 0.12, f"{act[i]:.2f}", ha="center", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["旧奖励 T-SAC", "融合奖励 T-SAC"], fontsize=9)
    ax.set_ylim(0, 9.6)
    ax.set_title("每束请求的槽数和实际分到的槽数", fontsize=10.5, loc="left")
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c9c9c9", label="请求的（浅色）"), Patch(color="#666666", label="实际分到的（深色）")],
              loc="upper right", fontsize=8.5, frameon=False, ncol=1, bbox_to_anchor=(1.02, 1.04))
    # 前50束/后50束
    ax = axs[1, 1]
    first = [4.23, 14.60]
    last = [14.70, 33.43]
    ax.bar(x - w / 2, first, w, color="#9bbfe0", label="前50束")
    ax.bar(x + w / 2, last, w, color="#2c5d8f", label="后50束")
    for i in range(2):
        ax.text(i - w / 2, first[i] + 0.7, f"{first[i]:.1f}", ha="center", fontsize=9.5)
        ax.text(i + w / 2, last[i] + 0.7, f"{last[i]:.1f}", ha="center", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["旧奖励 T-SAC", "融合奖励 T-SAC"], fontsize=9)
    ax.set_ylim(0, 41)
    ax.set_title("按处理顺序分的零槽动作比例（%）", fontsize=10.5, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    save_fig(fig, "W02_融合奖励的零槽问题.png")


# ---------- 图C：联合掩码前后对比（报告02正式终点结果表，100个新考核场景）----------
def fig_mask():
    labels = ["无掩码\nT-SAC", "联合掩码\nT-SAC", "贪心", "固定策略"]
    cols = [BLUE, GOLD, GREEN, PURPLE]
    si = [65.82, 75.55, 68.27, 72.16]
    sgm = [70.03, 80.55, 63.49, 59.08]
    ok = [88.74, 98.94, 95.43, 98.24]
    pw = [2020.42, 2528.02, 2409.75, 3000.00]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.0))
    bars(axs[0, 0], labels, si, cols, "需求满足率（%）", ylim=(0, 92))
    bars(axs[0, 1], labels, sgm, cols, "供需匹配度（%）", ylim=(0, 100))
    bars(axs[1, 0], labels, ok, cols, "基本服务达标束数（共100束）", ylim=(0, 118))
    bars(axs[1, 1], labels, pw, cols, "实际总功率（W）", fmt="{:.0f}", ylim=(0, 3600))
    axs[1, 1].axhline(3000, color="#555", ls="--", lw=1)
    axs[1, 1].text(-0.42, 3060, "上限3000 W", ha="left", va="bottom", fontsize=8.5, color="#555")
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    save_fig(fig, "W02_联合掩码前后对比.png")


# ---------- 图D：联合掩码后的全体达标与功率（报告02）----------
def fig_mask_detail():
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.0))
    labels = ["无掩码\nT-SAC", "联合掩码\nT-SAC", "贪心", "固定策略"]
    cols = [BLUE, GOLD, GREEN, PURPLE]
    # (a) 100束全部达标的次数占比
    ax = axs[0, 0]
    frac = [0 / 300 * 100, 106 / 300 * 100, 0 / 100 * 100, 40 / 100 * 100]
    txt = ["0/300", "106/300", "0/100", "40/100"]
    order = [0, 1, 2, 3]
    bars(ax, labels, frac, cols, "100束全部达标的评价次数占比（%）", ylim=(0, 52))
    for t in list(ax.texts):
        t.remove()
    for i, (v, s) in enumerate(zip(frac, txt)):
        ax.text(i, v + 1.2, s, ha="center", va="bottom", fontsize=9.5)
    # (b) 零容量束数
    zc = [10.37, 0.0, 2.00, 0.0]
    bars(axs[0, 1], labels, zc, cols, "平均每个场景的零容量波束数", fmt="{:.1f}", ylim=(0, 12.5))
    # (c) 联合掩码相对无掩码的配对平均提高
    ax = axs[1, 0]
    d_lab = ["需求满足率\n（百分点）", "供需匹配度\n（百分点）", "达标束数\n（束）"]
    d_val = [9.73, 10.52, 10.20]
    ax.bar(np.arange(3), d_val, 0.55, color=GOLD)
    for i, v in enumerate(d_val):
        ax.text(i, v + 0.25, f"+{v:.2f}", ha="center", fontsize=9.5)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(d_lab, fontsize=9)
    ax.set_ylim(0, 13.5)
    ax.set_title("联合掩码比无掩码的逐场景平均提高", fontsize=10.5, loc="left")
    ax.grid(axis="x", visible=False)
    # (d) 功率差
    ax = axs[1, 1]
    p_lab = ["比无掩码\nT-SAC", "比固定策略"]
    p_val = [507.60, -471.98]
    ax.bar(np.arange(2), p_val, 0.5, color=[C_ORANGE, C_TEAL])
    ax.axhline(0, color="#444", lw=0.8)
    ax.text(0, 507.6 + 25, "多用 508 W", ha="center", fontsize=9.5)
    ax.text(1, -471.98 - 25, "少用 472 W", ha="center", va="top", fontsize=9.5)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(p_lab, fontsize=9)
    ax.set_ylim(-700, 700)
    ax.set_title("联合掩码T-SAC的平均功率差（W）", fontsize=10.5, loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    save_fig(fig, "W02_联合掩码后的全体达标与功率.png")


if __name__ == "__main__":
    fig_overview()
    fig_zero_slot()
    fig_mask()
    fig_mask_detail()
