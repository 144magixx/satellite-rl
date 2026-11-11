# -*- coding: utf-8 -*-
"""W04 图：逐段计时里，参数更新和有限值检查（反复让CPU等GPU回答“有没有NaN”）在原版和优化后各花了多少秒。

数字取自 docs/第二阶段/实验报告/05_异步采集与训练提速验证.md “同工作量稳态训练”一节正文（函数累计耗时中位数，单位秒）：
  204维（13个输入令牌）：参数更新 原版约 53.8、优化后约 28.3；有限值检查 原版约 33.7、优化后约 0.65
  2946维（32个输入令牌）：参数更新 原版约 67.9、优化后约 38.5；有限值检查 原版约 39.7、优化后约 0.74
报告说明函数计时有嵌套关系，这些秒数不能相加，图里两项并排画、不叠加。
（逐函数计时的原始记录本机没有，按报告正文数字重画。）
运行：python outputs/weekly_reports_2026/figure_scripts/W04_更新耗时拆分.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np

plt = setup_mpl()

panels = [("204维状态（13个输入令牌）", [53.8, 28.3], [33.7, 0.65]),
          ("2946维状态（32个输入令牌）", [67.9, 38.5], [39.7, 0.74])]

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.8), sharey=True)
w = 0.34
for ax, (title, upd, chk) in zip(axes, panels):
    x = np.arange(2)       # 0=参数更新，1=有限值检查
    ax.bar(0 - w / 2, upd[0], w, color=C_ORANGE, label="原版")
    ax.bar(0 + w / 2, upd[1], w, color=C_TEAL, label="优化后")
    ax.bar(1 - w / 2, chk[0], w, color=C_ORANGE)
    ax.bar(1 + w / 2, chk[1], w, color=C_TEAL)
    for xs, v in ((0 - w / 2, upd[0]), (0 + w / 2, upd[1]), (1 - w / 2, chk[0]), (1 + w / 2, chk[1])):
        ax.text(xs, v + 1.2, f"{v:g}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["参数更新\n累计耗时", "有限值检查\n累计耗时"])
    ax.set_title(title)
    ax.set_ylim(0, 80)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("500步里的累计耗时（秒）")
axes[0].legend(loc="upper right", frameon=False)
fig.text(0.5, -0.02, "两项是分别计时的，有互相包含的部分，不能加起来", ha="center", fontsize=8.5, color="#555")
fig.tight_layout()
save_fig(fig, "W04_更新耗时拆分.png")
