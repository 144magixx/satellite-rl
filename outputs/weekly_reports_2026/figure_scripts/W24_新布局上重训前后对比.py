"""W24 图：第二阶段在新布局上重训前后（按报告表格重画）。
数字来源：docs/第二阶段/实验报告/24_FLOW-AO布局下的第二阶段重训与三阶段一致性对照.md 第 2 节表格
  “主要结果（开发 20 景，15 万步终点，三种子平均）”，括号里的区间是三个种子的最小到最大（图上的黑色细线）。
  s2_w32 实验目录里没有留下原始 csv，只有报告图（图里图例带代号，且图例在绘图区内部），所以不用原图，按表重画。
  报告里的“P00/P01”对应：原档位／原档位加 0.5 瓦；“M0/M1”对应：无／有余量摘要。
  “旧模型直接用”是 W31 旧布局上训练的模型不重训直接搬到新布局（报告表里的 W31 旧模型 P00-M0、P01-M1 两行）。
"""
from _w24_common import *

# (标签, 满足率%, 区间下, 区间上, 供需匹配, 区间下, 区间上, 颜色)
G, O, R = "#9A9A9A", "#E69F00", BLUE
rows = [
    ("贪心规则", 98.7, None, None, 0.437, None, None, G),
    ("随机规则", 92.4, None, None, 0.474, None, None, G),
    ("固定规则", 89.5, None, None, 0.363, None, None, G),
    ("旧模型直接用（原档位）", 84.0, None, None, 0.635, None, None, O),
    ("旧模型直接用（加0.5瓦＋摘要）", 91.0, None, None, 0.781, None, None, O),
    ("重训（原档位）", 96.9, 96.4, 97.2, 0.782, 0.779, 0.784, R),
    ("重训（原档位＋摘要）", 97.6, 97.6, 97.7, 0.787, 0.784, 0.788, R),
    ("重训（加0.5瓦）", 97.2, 97.0, 97.5, 0.853, 0.851, 0.855, R),
    ("重训（加0.5瓦＋摘要）", 97.6, 97.3, 97.9, 0.857, 0.849, 0.861, R),
]
ypos = np.arange(len(rows))[::-1]

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True, gridspec_kw={"wspace": 0.06})
for ax, (iv, lo, hi, xl, fmt, xlim) in zip(axes, [(1, 2, 3, "需求满足率（%）", "{:.1f}", (0, 118)),
                                                   (4, 5, 6, "供需匹配程度（越高越好）", "{:.2f}", (0, 1.18))]):
    for y, r in zip(ypos, rows):
        v, c = r[iv], r[7]
        ax.barh(y, v, color=c, height=0.66)
        if r[lo] is not None:
            ax.plot([r[lo], r[hi]], [y, y], color="black", lw=1.6)
        ax.text((r[hi] if r[hi] is not None else v) + xlim[1] * 0.012, y, fmt.format(v + 1e-9), va="center", fontsize=9.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xl)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
for ax in axes:
    for yb in (ypos[2] - 0.5, ypos[4] - 0.5):
        ax.axhline(yb, color="#BBBBBB", lw=0.8)
axes[0].set_xticks([0, 20, 40, 60, 80, 100])
axes[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([r[0] for r in rows], fontsize=10)
fig.text(0.5, -0.05, "摘要：多给模型一份已分配波束的受扰和供需缺口信息。加0.5瓦：功率档里多一个0.5瓦的低功率选项。\n重训的数字是三个种子的平均，黑色细线是种子范围。",
         ha="center", va="top", fontsize=8.6, color="#444444")
save_png(fig, "W24_新布局上重训前后对比.png", **SAVE_KW)
