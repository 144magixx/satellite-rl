# -*- coding: utf-8 -*-
"""W25 图：为什么最初的环境里T-SAC比贪心好（事后诊断，一次只改贪心的一处）。

数据来源：docs/Claude维护/原版框架复现/20_原版框架上八种方法不重训对比报告.md 第3节“事后诊断”表
（原版环境，100个场景，细调前）。表里没有的“T-SAC直接输出”一行取报告第3节诊断图里的数
（96.0%、60.0%）。逐景原始记录 s2_orig_compare_20261022/main/diagnostics.jsonl 本机缺失，按报告数字重画。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *
from matplotlib.patches import Patch

data = [  # 名称, 满足率, 最差10%波束
    ("容量感知贪心（原样）", 88.2, 19.8),
    ("贪心+把本组剩余槽补满", 87.5, 21.4),
    ("贪心+每束统一20 W", 82.9, 30.3),
    ("贪心+固定分成8组", 81.4, 15.4),
    ("贪心+8组+组内按干扰挑空位", 99.7, 95.2),
    ("T-SAC直接输出", 96.0, 60.0),
    ("T-SAC的频率方案+按需求分功率", 96.0, 44.1),
]
n = len(data)
y = np.arange(n)[::-1]
h = 0.38
fig, ax = plt.subplots(figsize=(5.8, 3.7))
key = [("按干扰" in d[0]) for d in data]
ax.barh(y + h / 2, [d[1] for d in data], height=h, color=[C_KEY if k else C_BLUE for k in key])
ax.barh(y - h / 2, [d[2] for d in data], height=h, color=["#e8a49a" if k else "#9dbbd8" for k in key])
for yy, d in zip(y, data):
    ax.text(d[1] + 1, yy + h / 2, f"{d[1]:.1f}", va="center", fontsize=8.5)
    ax.text(d[2] + 1, yy - h / 2, f"{d[2]:.1f}", va="center", fontsize=8.5)
ax.set_xlim(0, 112)
ax.set_yticks(y)
ax.set_yticklabels([d[0] for d in data], fontsize=8.5)
ax.set_xlabel("%")
ax.xaxis.grid(True, color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=C_BLUE, label="需求满足率"), Patch(color="#9dbbd8", label="最差10%波束满足率"),
                   Patch(color=C_KEY, label="分散摆放规则")],
          loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.45, 1.0), columnspacing=1.2, handlelength=1.4)
fig.subplots_adjust(left=0.40, right=0.985, top=0.93, bottom=0.12)
save_png(fig, "W25_原版环境诊断.png", dpi=220)
