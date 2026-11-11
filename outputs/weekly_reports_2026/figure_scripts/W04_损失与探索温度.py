# -*- coding: utf-8 -*-
"""W04 图：第一轮欧洲训练里策略网络、两个价值网络的损失和探索温度随训练步数的变化。

原图：D:/paper/outputs/experiments/europe_adsb_20260928/curves/learning_diagnostics.png
（报告 docs/第二阶段/实验报告/03_欧洲ADS-B首轮训练结果.md 引用；本机转存在 .scratch_w1234/curves__learning_diagnostics.png）。
逐更新的原始日志本机没有，只有这张报告图。
做法：绘图区（曲线、网格、刻度数字）一个像素都不动，只改绘图区外的英文文字：
  顶部标题 "G07: parameter-update diagnostics (symmetric-log y axes)" -> “参数更新过程诊断（纵轴为对称对数刻度）”（去掉代号 G07）
  图例 "Logged value" / "Trailing mean (100 logged points)" -> “每50步的原始值” / “最近100个记录点的向后平均”（图例框和线样本按原色重画）
  四个面板标题 Actor loss / Critic 1 loss / Critic 2 loss / Temperature alpha -> 中文
  下排 x 轴标签 "Environment steps" -> “环境交互步数”
运行：python outputs/weekly_reports_2026/figure_scripts/W04_损失与探索温度.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *

SCR = os.path.normpath(os.path.join(HERE, "..", ".scratch_w1234"))
im = load_any(os.path.join(SCR, "curves__learning_diagnostics.png"))

# 标题
erase(im, (480, 5, 1740, 62))
write(im, "参数更新过程诊断（纵轴为对称对数刻度）", (1112, 33), 34)

# 图例：整块盖掉后重画
erase(im, (728, 80, 1497, 138))
d = ImageDraw.Draw(im)
f = font(26)
t1, t2 = "每50步的原始值", "最近100个记录点的向后平均"
hw, gap, sep, pad = 50, 12, 36, 12
w1, w2 = d.textlength(t1, font=f), d.textlength(t2, font=f)
total = pad + hw + gap + w1 + sep + hw + gap + w2 + pad
x0 = 1112 - total / 2
y0, y1 = 86, 132
d.rectangle([x0, y0, x0 + total, y1], outline=(214, 214, 214), width=2, fill=WHITE)
x = x0 + pad
d.line([(x, 109), (x + hw, 109)], fill=(190, 214, 232), width=3)
d.text((x + hw + gap, 109), t1, font=f, fill=BLACK, anchor="lm")
x += hw + gap + w1 + sep
d.line([(x, 108), (x + hw, 108)], fill=(18, 104, 154), width=4)
d.text((x + hw + gap, 109), t2, font=f, fill=BLACK, anchor="lm")

# 面板标题
f30 = font(30)
for cx, cy, new in [(593, 207, "策略网络（Actor）损失"),
                    (1700, 207, "价值网络1（Critic 1）损失"),
                    (594, 787, "价值网络2（Critic 2）损失"),
                    (1700, 787, "探索温度 α")]:
    half = max(d.textlength(new, font=f30) / 2 + 12, 150)
    erase(im, (int(cx - half), cy - 20, int(cx + half), cy + 20))
    write(im, new, (cx, cy), 30)
# 下排 x 轴标签
for cx in (594, 1700):
    erase(im, (cx - 160, 1360, cx + 160, 1398))
    write(im, "环境交互步数", (cx, 1379), 26)

save_png(im, "W04_损失与探索温度.png")
