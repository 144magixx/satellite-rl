# -*- coding: utf-8 -*-
"""W05_七模型终点对比：数字取自 docs/第二阶段/实验报告/07_欧洲六组独立场景对照.md 第2节主结果表（1月18日100个独立场景，每场景3次动作平均）。"""
import matplotlib.pyplot as plt
from _plot_common import *

labels = ["原模型", "A 关dropout", "B 打乱场景", "C1 少量全局信息", "C2 资源概况", "C3 布局与需求", "C4 布局、分配与干扰", "固定分配", "贪心分配"]
si  = [89.92, 91.40, 90.99, 85.90, 85.12, 82.87, 88.41, 88.48, 90.32]
sgm = [0.4033, 0.5741, 0.5198, 0.4049, 0.3797, 0.3924, 0.3638, 0.2430, 0.3081]
colors = [C_REF, C2, C3, C5, C5, C5, C5, C_BASE, C_BASE]
hat = [None] * 7 + ["//", "//"]

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.16})
hbar(axs[0], labels, si, colors, fmt="{:.1f}%", xlim=(0, 112), hatches=hat)
axs[0].set_xticks([0, 25, 50, 75, 100])
axs[0].set_title("需求满足率（%）")
hbar(axs[1], labels, sgm, colors, fmt="{:.2f}", xlim=(0, 0.78), hatches=hat, show_ylabels=False)
axs[1].set_title("供需匹配程度（0到1）")
fig.subplots_adjust(left=0.30, right=0.985, top=0.9, bottom=0.07)
save(fig, "W05_七模型终点对比")
note("W05_七模型终点对比", """
按报告表格数字重画（没有逐回合原始记录）。数字取自 docs/第二阶段/实验报告/07_欧洲六组独立场景对照.md 第2节“01-18独立共同考核主结果”表：
需求满足率（SI）和供需匹配程度（SGM）两列，共9行（7个模型加固定、贪心两个基线）。
模型名按 W05 周报表格里的叫法，灰色斜线柱是基线。脚本：W05_七模型终点对比.py。
""")
