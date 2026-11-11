# -*- coding: utf-8 -*-
"""W20 图：多宽度候选＋整数补全先导试验的执行状态。

数据：第一阶段实验报告15 两张表/正文（该轮的逐槽原始表不在本机可直接读取的位置，数字取自报告）：
  上：预定的 24个案例 × 6种方法 = 144个方法槽：12个发布了布局、3个在选择阶段出接口错误、129个未执行；
  下：72个既有布局点（每个案例的共同起点、较大邻域结果、只改归属结果）的离散菜单投影：36个能投影、36个投影失败
      （失败的都是匹配完成后菜单角域超过上限）。
"""
from _c1821 import *

fig, ax = plt.subplots(figsize=(6.2, 2.9))
rows = [("144个方法槽\n（24个案例×6种方法）", [(12, "发布了布局", C_GREEN), (3, "接口错误", C_ORANGE), (129, "未执行", C_LIGHT)], 144),
        ("72个既有布局点\n投影到候选菜单", [(36, "能投影", C_GREEN), (36, "投影失败", C_RED)], 72)]
for yi, (lab, parts, tot) in zip((1, 0), rows):
    left = 0
    for n, name, c in parts:
        w = n / tot * 100
        ax.barh(yi, w, left=left, color=c, height=0.52)
        if w > 12:
            ax.text(left + w / 2, yi, f"{name} {n}", ha="center", va="center", fontsize=10.5,
                    color="white" if c in (C_GREEN, C_RED) else "black")
        else:
            tx = 1 if name == "发布了布局" else 20
            ax.annotate(f"{name} {n}", (left + w / 2, yi + 0.26), xytext=(tx, yi + 0.5), fontsize=10.5,
                        arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.8), va="bottom")
        left += w
ax.set_yticks([1, 0])
ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
ax.set_xlim(0, 100)
ax.set_ylim(-0.45, 1.85)
ax.set_xlabel("占比/%")
ax.grid(axis="y", visible=False)
save_png(fig, "W20_整数补全执行状态.png")
