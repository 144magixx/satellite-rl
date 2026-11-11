# -*- coding: utf-8 -*-
"""W21 图：让图神经网络直接预测分组，再交给同一个修复器，20个开发场景上的结果。

数据：第一阶段实验报告20 开发评价表（20个场景全部保留）：
  新提议可用/回退：模型 1/19，规则 20/0，随机 0/20（教师布局是参照，不涉及“提议”）；
  平均峰值请求负载（Mbps）：模型+修复器 2250.72、规则+修复器 892.08、随机+修复器 2255.04、教师布局 960.12。
“回退”指修复器没能补全，发布的是共同起点布局。逐景原始表不在本机可读位置，取报告表格数字。
"""
from _c1821 import *

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"wspace": 0.38})
ax = axes[0]
names = ["模型直接\n预测", "规则分组", "随机分组"]
ok = [1, 20, 0]
cols = [C_RED, C_BLUE, C_GREY]
for i, (v, c) in enumerate(zip(ok, cols)):
    ax.bar(i, v, 0.6, color=c)
    ax.text(i, v + 0.5, f"{v}/20", ha="center", fontsize=11)
ax.set_xticks(range(3))
ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0, 24)
ax.set_title("修复后能用上新提议的场景数", fontsize=10.5)
ax.grid(axis="x", visible=False)

ax = axes[1]
names = ["模型直接\n预测", "规则分组", "随机分组", "教师布局"]
peak = [2250.72, 892.08, 2255.04, 960.12]
cols = [C_RED, C_BLUE, C_GREY, C_GREEN]
for i, (v, c) in enumerate(zip(peak, cols)):
    ax.bar(i, v, 0.62, color=c)
    ax.text(i, v + 40, f"{v:.0f}", ha="center", fontsize=10.5)
ax.set_xticks(range(4))
ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylim(0, 2600)
ax.set_title("平均峰值负载/Mbps", fontsize=10.5)
ax.grid(axis="x", visible=False)
save_png(fig, "W21_直接预测布局.png")
