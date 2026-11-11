# -*- coding: utf-8 -*-
"""W08_九组达标与功率：数字取自 docs/第二阶段/实验报告/12_温度记忆与干扰图独立场景对照.md 里 T组、M组、G组三张结果表：
全100束达标比例（300次分配）和功率W；行同 W08_九组终点对比。"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from _plot_common import *
labels = ["原模型（15万步参照）", "联合Q（15万步参照）", "固定温度＋加性Q", "固定温度＋联合Q", "不加记忆", "加上一束动作", "GRU看16步", "Transformer看16步", "整段编码", "逐束令牌", "逐束令牌＋干扰关系"]
ok = [88.00, 38.67, 82.67, 60.67, 83.67, 85.67, 83.00, 81.67, 74.67, 77.33, 85.00]
pw = [2491.67, 2993.70, 2129.18, 2990.00, 2678.18, 2525.15, 2705.87, 2479.32, 2811.33, 2664.32, 2772.07]
colors = [C_REF, C_REF, C2, C2, C3, C3, C3, C3, C5, C5, C5]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.4), gridspec_kw={"wspace": 0.16})
hbar(axs[0], labels, ok, colors, fmt="{:.1f}%", xlim=(0, 118)); axs[0].set_xticks([0, 25, 50, 75, 100]); axs[0].set_title("100束全部达到基本服务（%）")
hbar(axs[1], labels, pw, colors, fmt="{:.0f}", xlim=(0, 3900), show_ylabels=False); axs[1].set_xticks([0, 1000, 2000, 3000]); axs[1].set_title("平均总功率（W）")
fig.legend(handles=[Patch(color=C_REF, label="15万步参照"), Patch(color=C2, label="固定温度（15万步）"), Patch(color=C3, label="分配记忆（40万步）"), Patch(color=C5, label="干扰图（40万步）")],
           loc="lower center", ncol=4, frameon=False, fontsize=10, columnspacing=1.0, handlelength=1.2, bbox_to_anchor=(0.55, -0.005))
fig.subplots_adjust(left=0.29, right=0.985, top=0.93, bottom=0.14)
save(fig, "W08_九组达标与功率")
note("W08_九组达标与功率", """
按报告表格数字重画。数字取自 docs/第二阶段/实验报告/12_温度记忆与干扰图独立场景对照.md 里 T组、M组、G组三张结果表：
“全100束达标比例”和“功率W”，行同 W08_九组终点对比。脚本：W08_九组达标与功率.py。
""")
