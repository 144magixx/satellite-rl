# -*- coding: utf-8 -*-
"""W21 图：直接预测布局的图神经网络模型，2000次更新的训练损失（总损失）。

数据说明：本机没有逐更新的原始损失文件（raw_training_loss.csv 在远端归档卷里），只有报告20引用的成品图
  D:/paper/outputs/experiments/first_stage_ai_search_20261013/round7/learning/train2000_curve_assets/train2000_raw_loss.png
的 (a) 面板（总损失，黑色折线）。这里按坐标轴刻度线把折线每个像素列的位置换算成数值，再用中文标签重画
（每个像素列约对应 2 次更新，所以是原图精度的读点，不是逐步原始值）。
校核：报告20给出第 1 步损失 12.0475、第 2000 步 3.8724；脚本核对读点开头约 11~13、结尾 3.4~4.4，
并且整条曲线下降。读点存成 figure_scripts/data/W21_训练损失_读点.csv。
"""
import numpy as np
from _c1821 import *

SRC = D_EXP / "first_stage_ai_search_20261013/round7/learning/train2000_curve_assets/train2000_raw_loss.png"
# 刻度线像素位置：y 轴 12->203.5px、0->710.5px；x 轴 1步->106.5px、2000步->1143.5px
Y12, Y0 = 203.5, 710.5
X1, X2000 = 106.5, 1143.5

a = np.array(Image.open(SRC).convert("RGB")).astype(int)
dark = a.sum(axis=2) < 150
steps, vals = [], []
for x in range(109, 1144):
    ys = np.where(dark[170:709, x])[0] + 170
    if len(ys) == 0:
        continue
    steps.append(1 + (x - X1) / (X2000 - X1) * 1999)
    vals.append((Y0 - ys.mean()) / (Y0 - Y12) * 12)
steps, vals = np.array(steps), np.array(vals)
assert 11 < vals[:5].mean() < 13.5 and 3.4 < vals[-20:].mean() < 4.4 and vals[:100].mean() > vals[-100:].mean() + 3

with open(DATA_DIR / "W21_训练损失_读点.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["更新次数(近似)", "总损失"])
    for s, v in zip(steps, vals):
        w.writerow([round(float(s), 1), round(float(v), 3)])

fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.plot(steps, vals, color=C_LIGHT, lw=0.8)
k = 25
sm = np.convolve(vals, np.ones(k) / k, mode="valid")
ax.plot(steps[k // 2:k // 2 + len(sm)], sm, color=C_BLUE, lw=2)
ax.set_xlabel("训练更新次数")
ax.set_ylabel("训练总损失")
ax.set_xlim(0, 2000)
ax.set_ylim(0, 13)
ax.annotate("约12", (steps[0], vals[:3].mean()), xytext=(90, vals[:3].mean() - 0.6), fontsize=10.5)
ax.annotate("约4", (steps[-1], vals[-20:].mean()), xytext=(1880, vals[-20:].mean() + 1.2), fontsize=10.5)
save_png(fig, "W21_训练损失.png")
