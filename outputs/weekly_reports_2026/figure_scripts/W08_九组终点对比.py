# -*- coding: utf-8 -*-
"""W08_九组终点对比：数字取自 docs/第二阶段/实验报告/12_温度记忆与干扰图独立场景对照.md 里 T组、M组、G组三张结果表
（1月24日100个独立场景，每场景3次动作平均）：需求满足率SI、供需匹配程度SGM。
加上T组表里两行15万步参照（J0_150k、J1_150k，即周报里的“原模型（15万步参照）”和“原策略＋联合Q（15万步参照）”）。"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from _plot_common import *
labels = ["原模型（15万步参照）", "联合Q（15万步参照）", "固定温度＋加性Q", "固定温度＋联合Q", "不加记忆", "加上一束动作", "GRU看16步", "Transformer看16步", "整段编码", "逐束令牌", "逐束令牌＋干扰关系"]
si  = [92.86, 84.82, 91.90, 87.50, 92.97, 92.90, 92.51, 92.29, 91.89, 92.10, 92.71]
sgm = [0.5658, 0.4461, 0.6973, 0.5831, 0.5725, 0.5763, 0.5505, 0.5720, 0.5685, 0.5603, 0.5644]
colors = [C_REF, C_REF, C2, C2, C3, C3, C3, C3, C5, C5, C5]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.4), gridspec_kw={"wspace": 0.16})
hbar(axs[0], labels, si, colors, fmt="{:.1f}%", xlim=(0, 118)); axs[0].set_xticks([0, 25, 50, 75, 100]); axs[0].set_title("需求满足率（%）")
hbar(axs[1], labels, sgm, colors, fmt="{:.2f}", xlim=(0, 0.84), show_ylabels=False); axs[1].set_xticks([0, 0.2, 0.4, 0.6]); axs[1].set_title("供需匹配程度（0到1）")
fig.legend(handles=[Patch(color=C_REF, label="15万步参照"), Patch(color=C2, label="固定温度（15万步）"), Patch(color=C3, label="分配记忆（40万步）"), Patch(color=C5, label="干扰图（40万步）")],
           loc="lower center", ncol=4, frameon=False, fontsize=10, columnspacing=1.0, handlelength=1.2, bbox_to_anchor=(0.55, -0.005))
fig.subplots_adjust(left=0.29, right=0.985, top=0.93, bottom=0.14)
save(fig, "W08_九组终点对比")
note("W08_九组终点对比", """
按报告表格数字重画（没有逐回合原始记录）。数字取自 docs/第二阶段/实验报告/12_温度记忆与干扰图独立场景对照.md 里 T组、M组、G组三张结果表
（需求满足率SI、供需匹配程度SGM），另加T组表里两行15万步参照（J0_150k、J1_150k）。固定温度两组只训了15万步，其余40万步，图里用颜色分开，不是同预算排名。
脚本：W08_九组终点对比.py。
""")
