# -*- coding: utf-8 -*-
"""W03 四张按报告表格数字重画的图（训练和评价的逐条原始记录本机没有）。

数字来源（docs/第二阶段/ 下）：
  方案/05_欧洲ADS-B首轮训练说明.md 第7节“实际数据与运行记录”表：
    训练/验证/考核的实际场景数 576/20/100、每场有效飞机范围 140–1887/122–1815/133–1764、
    平均总需求 22.482/23.590/22.851 Gbit/s、飞机覆盖比例 99.66/99.64/99.69%、需求覆盖比例 99.79/99.79/99.81%。
  实验报告/03_欧洲ADS-B首轮训练结果.md “结论与限制”补充段：
    固定验证场景上 1万步 到 40万步：需求满足率 86.92% -> 88.69%（期间最高约89.15%），回报 57.11 -> 28.47。
  实验报告/04_欧洲模型与基线逐场景对比.md：
    “平均指标与约束”表（需求满足率、供需匹配度、达标束数、全部100束达标比例、实际功率）、
    约束表现段（固定分配每场景4次动作裁剪、贪心2次动作裁剪和2个零容量束、模型都是0）、
    配对段（相对固定分配：满足率更高62个/更低37个/相同1个，匹配度更高99个/更低1个；
    相对贪心：满足率更高49个/更低51个，匹配度更高98个/更低2个）。
输出：figures/W03_数据划分与场景规模.png、W03_训练前后对比.png、W03_模型与基线平均对比.png、W03_逐场景配对与约束表现.png
运行：python outputs/weekly_reports_2026/figure_scripts/W03_表格数字重画.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np

plt = setup_mpl()

GOLD, GREEN, PURPLE, BLUE = "#e9b44c", "#269368", "#8b55a3", "#2070b4"


def bars(ax, labels, vals, cols, title, fmt="{:.1f}", ylim=None, texts=None):
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.6, color=cols)
    top = ylim[1] if ylim else max(vals) * 1.18
    ax.set_ylim(ylim[0] if ylim else 0, top)
    for i, v in enumerate(vals):
        s = texts[i] if texts else fmt.format(v + 1e-9)
        ax.text(i, v + top * 0.02, s, ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.grid(axis="x", visible=False)


# ---------- 图A：三套场景的规模（方案05第7节表）----------
def fig_data():
    labels = ["训练场景", "验证场景", "考核场景"]
    cols = [C_GREY, C_TEAL, GOLD]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.0))
    bars(axs[0, 0], labels, [576, 20, 100], cols, "场景数（个）", fmt="{:.0f}", ylim=(0, 680))
    # 每场有效飞机范围
    ax = axs[0, 1]
    lo, hi = [140, 122, 133], [1887, 1815, 1764]
    for i in range(3):
        ax.bar(i, hi[i] - lo[i], 0.6, bottom=lo[i], color=cols[i])
        ax.text(i, hi[i] + 40, f"{lo[i]}～{hi[i]}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 2250)
    ax.set_title("每个场景的有效飞机数（最少～最多）", fontsize=10.5, loc="left")
    ax.grid(axis="x", visible=False)
    bars(axs[1, 0], labels, [22.482, 23.590, 22.851], cols, "平均每个场景的总需求（Gbit/s）", fmt="{:.1f}", ylim=(0, 29))
    # 覆盖比例
    ax = axs[1, 1]
    x = np.arange(3)
    w = 0.36
    ac = [99.66, 99.64, 99.69]
    dc = [99.79, 99.79, 99.81]
    ax.bar(x - w / 2, ac, w, color="#b9bfca")
    ax.bar(x + w / 2, dc, w, color="#5b6b82")
    for i in range(3):
        ax.text(i - w / 2, ac[i] + 1.5, f"{ac[i]:.1f}", ha="center", fontsize=9)
        ax.text(i + w / 2, dc[i] + 1.5, f"{dc[i]:.1f}", ha="center", fontsize=9)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("被100个波束覆盖到的比例（%）", fontsize=10.5, loc="left")
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#b9bfca", label="飞机"), Patch(color="#5b6b82", label="需求")],
              loc="upper right", fontsize=8.5, frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    save_fig(fig, "W03_数据划分与场景规模.png")


# ---------- 图B：训练前后（报告03补充段）----------
def fig_before_after():
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0))
    labels = ["训练1万步时", "训练40万步后"]
    cols = [C_GREY, BLUE]
    bars(axs[0], labels, [86.92, 88.69], cols, "验证场景的需求满足率（%）", fmt="{:.1f}", ylim=(0, 108))
    bars(axs[1], labels, [57.11, 28.47], cols, "验证场景的回合回报", fmt="{:.1f}", ylim=(0, 70))
    fig.tight_layout(w_pad=1.6)
    save_fig(fig, "W03_训练前后对比.png")


# ---------- 图C：模型与基线平均对比（报告04表）----------
def fig_compare():
    labels = ["T-SAC\n（联合掩码）", "固定分配", "贪心"]
    cols = [GOLD, PURPLE, GREEN]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.0))
    bars(axs[0, 0], labels, [88.94, 88.04, 89.27], cols, "需求满足率（%）", ylim=(0, 108))
    bars(axs[0, 1], labels, [0.4213, 0.2436, 0.3240], cols, "供需匹配度（0～1）", fmt="{:.2f}", ylim=(0, 0.52))
    bars(axs[1, 0], labels, [99.14, 98.38, 97.78], cols, "基本服务达标束数（共100束）", ylim=(0, 120))
    bars(axs[1, 1], labels, [2885.23, 3000.00, 2066.60], cols, "实际总功率（W）", fmt="{:.0f}", ylim=(0, 3600))
    axs[1, 1].axhline(3000, color="#555", ls="--", lw=1)
    axs[1, 1].text(2.45, 3060, "上限3000 W", ha="right", va="bottom", fontsize=8.5, color="#555")
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    save_fig(fig, "W03_模型与基线平均对比.png")


# ---------- 图D：逐场景配对和约束表现（报告04）----------
def fig_pair():
    fig = plt.figure(figsize=(6.4, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    rows = [("对固定分配：需求满足率", 62, 1, 37), ("对固定分配：供需匹配度", 99, 0, 1),
            ("对贪心：需求满足率", 49, 0, 51), ("对贪心：供需匹配度", 98, 0, 2)]
    y = np.arange(len(rows))[::-1]
    for yy, (name, hi, eq, lo) in zip(y, rows):
        ax.barh(yy, hi, 0.55, color=GOLD)
        ax.barh(yy, eq, 0.55, left=hi, color="#bdbdbd")
        ax.barh(yy, lo, 0.55, left=hi + eq, color="#6b7a90")
        ax.text(hi / 2, yy, f"{hi}", ha="center", va="center", fontsize=9.5)
        if lo >= 5:
            ax.text(hi + eq + lo / 2, yy, f"{lo}", ha="center", va="center", fontsize=9.5, color="white")
        else:
            ax.text(100.8, yy, f"{lo}", ha="left", va="center", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 106)
    ax.set_title("同一场景里T-SAC和规则基线比，各占多少个场景（共100个）", fontsize=10.5, loc="left")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GOLD, label="T-SAC更高"), Patch(color="#bdbdbd", label="相同"),
                       Patch(color="#6b7a90", label="T-SAC更低")], loc="lower center", fontsize=8.5,
              frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.36))
    labels = ["T-SAC\n（联合掩码）", "固定分配", "贪心"]
    cols = [GOLD, PURPLE, GREEN]
    ax2 = fig.add_subplot(gs[1, 0])
    bars(ax2, labels, [145 / 300 * 100, 42 / 100 * 100, 0.0], cols, "100束全部达标的评价次数占比（%）", ylim=(0, 62),
         texts=["145/300", "42/100", "0/100"])
    ax2.set_title("100束全部达标的评价次数占比（%）", fontsize=10, loc="left")
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(3)
    w = 0.36
    clip = [0, 4, 2]
    zero = [0, 0, 2]
    ax3.bar(x - w / 2, clip, w, color="#9aa6b8")
    ax3.bar(x + w / 2, zero, w, color="#c0392b")
    for i in range(3):
        ax3.text(i - w / 2, clip[i] + 0.1, str(clip[i]), ha="center", fontsize=9.5)
        ax3.text(i + w / 2, zero[i] + 0.1, str(zero[i]), ha="center", fontsize=9.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylim(0, 5.6)
    ax3.set_title("每个场景平均出现的次数", fontsize=10, loc="left")
    ax3.grid(axis="x", visible=False)
    ax3.legend(handles=[Patch(color="#9aa6b8", label="动作被裁剪"), Patch(color="#c0392b", label="零容量波束")],
               loc="upper right", fontsize=8.5, frameon=False, bbox_to_anchor=(1.02, 1.04))
    fig.tight_layout(h_pad=2.6, w_pad=1.4)
    save_fig(fig, "W03_逐场景配对与约束表现.png")


if __name__ == "__main__":
    fig_data()
    fig_before_after()
    fig_compare()
    fig_pair()
