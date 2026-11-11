# -*- coding: utf-8 -*-
"""W18 图：七种方法的供需匹配程度，以及容量不到需求八成的波束比例。

数据来源：第二阶段实验报告15
  左图：第4节表“直接SGM / 最终B SGM”两列（0~1，越高越好）；
  右图：第5节表“直接/B的C/D<0.8比例/%”（越低越好）。
逐束原始表不在本机，取报告表格数字。
"""
from _c1821 import *

names = ["贪心", "规则搜索", "随机搜索", "上一轮\n旧模型",
         "从空构造\n（只改奖励）", "从空构造\n＋贪心样本", "贪心起点\n＋SAC调整"]
sgm_b = [0.33634, 0.35123, 0.35635, 0.63024, 0.41396, 0.39538, 0.36196]
sgm_a = [0.94076, 0.95476, 0.96566, 0.93187, 0.91525, 0.92379, 0.97304]
low_b = [16.388, 14.696, 13.262, 18.421, 22.379, 19.923, 12.006]
low_a = [8.699, 7.052, 5.766, 9.824, 10.945, 10.785, 4.811]

fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.5), sharey=True, gridspec_kw={"wspace": 0.08})
y = list(range(len(names)))[::-1]
h = 0.38
for ax, b, a, title, fmt, xmax in ((axes[0], sgm_b, sgm_a, "供需匹配程度（越高越好）", "{:.2f}", 1.25),
                                   (axes[1], low_b, low_a, "容量不到需求八成的波束占比/%\n（越低越好）", "{:.1f}", 28.5)):
    for yi, vb, va, n in zip(y, b, a, names):
        key = n.startswith("贪心起点")
        ax.barh(yi + h / 2, vb, h, color="#9db8d6" if not key else "#e8a79f")
        ax.barh(yi - h / 2, va, h, color=C_BLUE if not key else C_RED)
        ax.text(vb + xmax * 0.012, yi + h / 2, fmt.format(vb), va="center", fontsize=9.5)
        ax.text(va + xmax * 0.012, yi - h / 2, fmt.format(va), va="center", fontsize=9.5)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", visible=False)
axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axes[0].set_yticks(y)
axes[0].set_yticklabels(names)
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color="#9db8d6", label="细调前"), Patch(color=C_BLUE, label="细调后")],
               loc="lower right", frameon=False)
save_png(fig, "W18_供需匹配与欠服务.png")
