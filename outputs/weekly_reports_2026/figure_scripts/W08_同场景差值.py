# -*- coding: utf-8 -*-
"""W08_同场景差值：数字取自 docs/第二阶段/实验报告/12_温度记忆与干扰图独立场景对照.md 里 T、M、G 三组的“同场景差值”表
（1月24日100个共同场景，先对每个场景3次动作取平均，再两两相减）：SI差（百分点）、SGM差、功率差（W）。"""
import numpy as np
import matplotlib.pyplot as plt
from _plot_common import *
labels = ["固定温度＋加性Q\n对 原模型（15万步）", "固定温度＋联合Q\n对 联合Q（15万步）", "加上一束动作\n对 不加记忆", "GRU看16步\n对 加上一束动作", "Transformer看16步\n对 GRU看16步", "逐束令牌\n对 整段编码", "逐束令牌＋干扰关系\n对 逐束令牌"]
si  = [-0.963, 2.678, -0.069, -0.385, -0.225, 0.209, 0.615]
sgm = [0.1315, 0.1369, 0.0038, -0.0258, 0.0215, -0.0082, 0.0041]
pw  = [-362.48, -3.70, -153.03, 180.72, -226.55, -147.02, 107.75]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 4.2), gridspec_kw={"wspace": 0.16})
ys = np.arange(len(labels))
specs = [(si, "需求满足率差\n（百分点）", "{:+.2f}", (-3.0, 4.6)), (sgm, "供需匹配程度差", "{:+.3f}", (-0.105, 0.25)), (pw, "平均功率差（W）", "{:+.0f}", (-700, 520))]
for k, (ax, (vals, title, fmt, xl)) in enumerate(zip(axs, specs)):
    cols = ["#4c78a8" if v >= 0 else "#d95f5f" for v in vals]
    ax.barh(ys, vals, color=cols, height=0.62)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlim(*xl)
    span = xl[1] - xl[0]
    for y, v in zip(ys, vals):
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), y, fmt.format(v), va="center", ha="left" if v >= 0 else "right", fontsize=9.5)
    ax.set_yticks(ys); ax.set_yticklabels(labels if k == 0 else [""] * len(labels), fontsize=10)
    ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=11.5)
    ax.grid(axis="x", color="#e6e6e6"); ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9.5)
axs[0].set_xticks([0, 2, 4]); axs[1].set_xticks([0, 0.1, 0.2]); axs[2].set_xticks([-400, 0, 400])
fig.subplots_adjust(left=0.265, right=0.995, top=0.88, bottom=0.07)
save(fig, "W08_同场景差值")
note("W08_同场景差值", """
按报告表格数字重画。数字取自 docs/第二阶段/实验报告/12_温度记忆与干扰图独立场景对照.md 里 T组、M组、G组各自的“同场景差值”表
（T0−J0_150k、T1−J1_150k、M1−M0、M2−M1、M3−M2、G1−G0、G2−G1）：SI百分点差、SGM差、功率W差。
蓝色是正差，红色是负差。脚本：W08_同场景差值.py。
""")
