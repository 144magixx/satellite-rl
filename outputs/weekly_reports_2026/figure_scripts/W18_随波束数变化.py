# -*- coding: utf-8 -*-
"""W18 图：七种方法的需求满足率随目标波束数变化（细调前、细调后两个面板）。

数据说明：本机没有这个实验的逐场景原始csv，只有报告15里引用的成品图
  D:/paper/outputs/experiments/sac_routes_s2p19_20261007/analysis/publication/si_by_n_panels.png
（每个点是该波束数下40个母时刻×2种负载的平均）。这里把原图里每个标记点的位置读出来，
按坐标轴刻度换算成数值，再用中文标签重画。读点方法：用图例里的线条颜色找出每个波束数处
（50/60/75/90/100/125/140/150/175/180/200）的标记中心，纵向刻度用坐标轴刻度线线性换算。
校核：每条线11个点的平均值应等于报告15第4节表里880场景的平均需求满足率（11个波束数
每个都是40母×2负载，权重相同）。本脚本运行时会逐条比对，差值超过0.05个百分点就报错。
读出的数值同时存成 figure_scripts/data/W18_随波束数变化_读点.csv。
方法名对应：C0=从空构造（只改奖励）、C1=从空构造＋贪心样本、I0=贪心起点＋SAC调整、V2=上一轮旧模型。
"""
import numpy as np
from _c1821 import *

SRC = D_EXP / "sac_routes_s2p19_20261007/analysis/publication/si_by_n_panels.png"
NS = [50, 60, 75, 90, 100, 125, 140, 150, 175, 180, 200]
# 图例里读到的线条颜色
COLORS = {"贪心": (96, 96, 96), "规则搜索": (198, 124, 0), "随机搜索": (72, 169, 166),
          "上一轮旧模型": (146, 117, 181), "从空构造（只改奖励）": (207, 110, 126),
          "从空构造＋贪心样本": (57, 115, 172), "贪心起点＋SAC调整": (194, 43, 35)}
# 报告15第4节表：直接SI、最终B SI（%）
TABLE = {"贪心": (90.7907, 94.2675), "规则搜索": (92.1045, 95.5844), "随机搜索": (93.2082, 96.6486),
         "上一轮旧模型": (89.0485, 93.1220), "从空构造（只改奖励）": (86.6018, 91.8427),
         "从空构造＋贪心样本": (88.4616, 92.5744), "贪心起点＋SAC调整": (94.0516, 97.3334)}
# 原图坐标：x轴 60->227.5px, 200->1107.5px（左）/1383.5, 2262.5（右）；
# y轴 左 96->212.5px, 84->753.5px；右 98->235.5px, 90->751.5px（由刻度线位置读出）
X60 = {"L": (227.5, 1107.5), "R": (1383.5, 2262.5)}
Y = {"L": (212.5, 96, 753.5, 84), "R": (235.5, 98, 751.5, 90)}


def digitize():
    im = np.array(Image.open(SRC).convert("RGB")).astype(int)
    out = {}
    for panel in "LR":
        for name, c in COLORS.items():
            vals = []
            for n in NS:
                x60, x200 = X60[panel]
                x = int(round(x60 + (n - 60) * (x200 - x60) / 140))
                sub = im[196:803, x - 4:x + 5]
                m = np.abs(sub - np.array(c)).sum(axis=2) < 30
                ys = np.sort(np.where(m)[0] + 196)
                cl, cur = [], [ys[0]]
                for yy in ys[1:]:
                    if yy - cur[-1] > 6:
                        cl.append(cur)
                        cur = [yy]
                    else:
                        cur.append(yy)
                cl.append(cur)
                best = max(cl, key=len)
                y0, v0, y1, v1 = Y[panel]
                vals.append(v0 + (np.mean(best) - y0) * (v1 - v0) / (y1 - y0))
            out[(panel, name)] = np.array(vals)
    return out


data = digitize()
rows = []
for name, (tb, ta) in TABLE.items():
    for panel, t in (("L", tb), ("R", ta)):
        d = abs(data[(panel, name)].mean() - t)
        assert d < 0.05, (name, panel, d)
for name in COLORS:
    for i, n in enumerate(NS):
        rows.append([name, n, round(data[("L", name)][i], 3), round(data[("R", name)][i], 3)])
with open(DATA_DIR / "W18_随波束数变化_读点.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["方法", "目标波束数", "细调前需求满足率(%)", "细调后需求满足率(%)"])
    w.writerows(rows)

style = {"贪心": (C_GREY, 2.0, "o"), "规则搜索": (C_ORANGE, 1.8, "o"), "随机搜索": (C_TEAL, 1.8, "o"),
         "上一轮旧模型": (C_PURPLE, 1.6, "o"), "从空构造（只改奖励）": (C_PINK, 1.6, "o"),
         "从空构造＋贪心样本": (C_BLUE, 1.6, "o"), "贪心起点＋SAC调整": (C_RED, 3.0, "o")}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.9), gridspec_kw={"wspace": 0.28, "bottom": 0.33, "top": 0.93})
for ax, panel, title in ((axes[0], "L", "细调前（直接分配）"), (axes[1], "R", "细调后（功率优化后）")):
    for name, (c, lw, mk) in style.items():
        ax.plot(NS, data[(panel, name)], color=c, lw=lw, marker=mk, ms=3.2, label=name)
    ax.set_title(title)
    ax.set_xlabel("目标波束数")
    ax.set_xticks([50, 100, 150, 200])
axes[0].set_ylabel("需求满足率（%）")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
           columnspacing=1.2, handlelength=1.6)
save_png(fig, "W18_随波束数变化.png")
