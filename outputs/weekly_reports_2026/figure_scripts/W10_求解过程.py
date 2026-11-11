# -*- coding: utf-8 -*-
"""W10_求解过程.png：报告现成图改文字，不是重画。

原图：D:\\paper\\outputs\\experiments\\stage3_all_routes_20261003\\report_20261003\\total_accepted_progress.png
（报告 第三阶段/实验报告/02 里的“3000 W预算预选场景已接受候选变化”，webp 格式，转存成真 PNG）。
文字改动（绘图区的线一个像素没动，也没缩放）：
  1. 标题（英文，带“total”“scene 0/action 0”）换成中文；
  2. 纵轴“SI”→“需求满足率”，“SGM”→“供需匹配程度”；横轴“Optimizer elapsed time (s)”→“求解耗时（秒）”；
  3. 右图图例方法英文名换成中文：discrete→原档位调整，de→差分进化，pso→粒子群；slsqp、sca、wmmse 保留算法名。
运行：python W10_求解过程.py
"""
from _patch_tools import *

SRC = EXP + r"\stage3_all_routes_20261003\report_20261003\total_accepted_progress.png"
im = load_any(SRC)

erase(im, (480, 20, 1400, 55))
write(im, "3000 W总功率预算：预先指定的一个场景、一次动作的求解过程（只画已接受的可行解）", (937, 37), 25)

for (x0, y0, x1, y1), s in [((26, 338, 41, 353), "需求满足率"), ((963, 323, 978, 367), "供需匹配程度")]:
    erase(im, (x0 - 4, y0 - 4, x1 + 4, y1 + 4))
    write(im, s, ((x0 + x1) / 2, (y0 + y1) / 2), 21, rot=90)

for (x0, y0, x1, y1) in [(395, 648, 674, 667), (1305, 648, 1584, 667)]:
    erase(im, (x0 - 3, y0 - 3, x1 + 3, y1 + 3))
    write(im, "求解耗时（秒）", ((x0 + x1) / 2, (y0 + y1) / 2), 21)

legend = [
    ((1625, 526, 1689, 537), "原档位调整", 15, 531), ((1770, 526, 1788, 537), "差分进化", 15, 531),
    ((1626, 549, 1665, 563), "SLSQP", 14, 555), ((1771, 549, 1797, 563), "粒子群", 15, 555),
    ((1626, 575, 1650, 583), "SCA", 14, 580), ((1770, 575, 1832, 583), "WMMSE", 14, 580),
]
for (x0, y0, x1, y1), s, size, cy in legend:
    erase(im, (x0 - 2, y0 - 3, int(max(x1 + 3, x0 + font(size).getlength(s) + 2)), y1 + 5))
    write(im, s, (x0, cy), size, anchor="lm")

save_png(im, "W10_求解过程.png")
