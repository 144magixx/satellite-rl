# -*- coding: utf-8 -*-
"""W05-W08 重画图的公共设置。图都是按报告表格里的数字画的，不是原始记录。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11.5
plt.rcParams["axes.titlesize"] = 12.5
plt.rcParams["axes.labelsize"] = 11.5
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

ROOT = r"E:\satellite-rl\outputs\weekly_reports_2026"
FIG_DIR = os.path.join(ROOT, "figures")
SCRIPT_DIR = os.path.join(ROOT, "figure_scripts")

# 配色
C_REF = "#8c9db5"      # 参照模型
C_BASE = "#c9c9c9"     # 固定/贪心基线
C1 = "#1f77b4"; C2 = "#ff7f0e"; C3 = "#2ca02c"; C4 = "#d62728"
C5 = "#9467bd"; C6 = "#8c564b"; C7 = "#e377c2"; C8 = "#17becf"

def save(fig, name, dpi=220):
    path = os.path.join(FIG_DIR, name + ".png")
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    from PIL import Image
    im = Image.open(path)
    assert im.format == "PNG" and im.width <= 2400, (im.format, im.size)
    size = os.path.getsize(path)
    assert size <= 1.5 * 1024 * 1024, size
    print(name, im.size, round(size / 1024), "KB")
    return path

def note(name, text):
    with open(os.path.join(SCRIPT_DIR, name + ".md"), "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

def hbar(ax, labels, values, colors, fmt="{:.1f}", xlim=None, hatches=None, xlabel=None, pad=0.012, show_ylabels=True, fs=10.5):
    """横向柱状图，柱右侧标数字。labels 自上而下排列。"""
    n = len(labels)
    ys = list(range(n))
    bars = ax.barh(ys, values, color=colors, height=0.66)
    if hatches:
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h); b.set_edgecolor("white")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels if show_ylabels else [""] * n)
    ax.invert_yaxis()
    if xlim:
        ax.set_xlim(*xlim)
    xmax = ax.get_xlim()[1]
    for y, v in zip(ys, values):
        ax.text(v + (xmax - ax.get_xlim()[0]) * pad, y, fmt.format(v), va="center", ha="left", fontsize=fs)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def two_panel(name, labels, left, right, colors, hatches, titles, fmts, xlims, figsize=(7.2, 3.9), left_margin=0.27, ticks=None, wspace=0.16):
    """左右两栏横向柱图（共用左边的类别名）。left/right 是数值列表。"""
    fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": wspace})
    hbar(axs[0], labels, left, colors, fmt=fmts[0], xlim=xlims[0], hatches=hatches)
    axs[0].set_title(titles[0])
    hbar(axs[1], labels, right, colors, fmt=fmts[1], xlim=xlims[1], hatches=hatches, show_ylabels=False)
    axs[1].set_title(titles[1])
    if ticks:
        axs[0].set_xticks(ticks[0]); axs[1].set_xticks(ticks[1])
    fig.subplots_adjust(left=left_margin, right=0.985, top=0.9, bottom=0.09)
    return save(fig, name)
