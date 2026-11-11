# -*- coding: utf-8 -*-
"""W25 图：把“8个频率组+组内按干扰挑空位”的分散摆放规则放进新环境，不搜索也比搜索320次高。

数据来源：docs/Claude维护/原版框架复现/20_原版框架上八种方法不重训对比报告.md 第4节
“顺带的发现”表（新环境，确认88景，接同一套SCA功率细调之后）。
逐景原始结果 s2_orig_compare_20261022/cross/free8_in_v2_conf_*.jsonl 本机缺失，按报告表格数字重画。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *
from matplotlib.patches import Patch

data = [  # 名称, 满足率, 最差10%飞机, 供需匹配程度
    ("容量感知贪心", 92.5, 67.2, 0.973),
    ("贪心+禁忌搜索320次", 94.9, 83.9, 0.990),
    ("改进式SAC", 94.9, 84.4, 0.990),
    ("T-SAC直接输出", 88.3, 50.0, 0.919),
    ("分散摆放规则（不搜索）", 95.7, 86.6, 0.991),
]
n = len(data)
y = np.arange(n)[::-1]
h = 0.38
fig, ax = plt.subplots(figsize=(5.8, 3.0))
key = [("分散" in d[0]) for d in data]
ax.barh(y + h / 2, [d[1] for d in data], height=h, color=[C_KEY if k else C_BLUE for k in key])
ax.barh(y - h / 2, [d[2] for d in data], height=h, color=["#e8a49a" if k else "#9dbbd8" for k in key])
for yy, d in zip(y, data):
    ax.text(d[1] + 1, yy + h / 2, f"{d[1]:.1f}", va="center", fontsize=8.5)
    ax.text(d[2] + 1, yy - h / 2, f"{d[2]:.1f}", va="center", fontsize=8.5)
    ax.text(124, yy, f"{d[3]:.3f}", va="center", ha="right", fontsize=8.5)
ax.text(124, n - 0.35, "供需匹配程度", va="bottom", ha="right", fontsize=8.5)
ax.set_xlim(0, 126)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_yticks(y)
ax.set_yticklabels([d[0] for d in data], fontsize=8.5)
ax.set_xlabel("%（细调之后，确认88景）")
ax.xaxis.grid(True, color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=C_BLUE, label="需求满足率"), Patch(color="#9dbbd8", label="最差10%飞机满足率")],
          loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.45, 1.0))
fig.subplots_adjust(left=0.30, right=0.985, top=0.85, bottom=0.17)
save_png(fig, "W25_分散摆放规则新环境.png", dpi=220)
