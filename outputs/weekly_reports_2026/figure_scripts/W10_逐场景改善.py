# -*- coding: utf-8 -*-
"""W10_逐场景改善.png：报告现成图改文字，不是重画。

原图：D:\\paper\\outputs\\experiments\\stage3_all_routes_20261003\\report_20261003\\total_paired_delta.png
（报告 第三阶段/实验报告/02 里的“3000 W预算相对原分配的配对差值”，webp 格式，转存成真 PNG）。
文字改动（绘图区的点一个像素没动，也没缩放）：
  1. 两个子图标题（英文，带“total”）换成中文，上方补了一条白边放总标题；
  2. 纵轴“Delta SI vs none”→“需求满足率的提高量”，“Delta SGM vs none”→“供需匹配程度的提高量”；
     横轴“Common scene index”→“场景序号”；
  3. 右图图例的方法英文名换成中文：discrete→原档位调整，de→差分进化，pso→粒子群；slsqp、sca、wmmse 保留算法名。
运行：python W10_逐场景改善.py
"""
from _patch_tools import *

SRC = EXP + r"\stage3_all_routes_20261003\report_20261003\total_paired_delta.png"
im = load_any(SRC)
TOP = 40
im = pad(im, top=TOP)
dy = TOP

write(im, "3000 W总功率预算：每个场景相比“不细调”提高了多少（1月26日的100个场景，每景先平均3次动作）", (1022, 16), 25)
for (x0, y0, x1, y1), s in [((412, 26, 720, 49), "需求满足率"), ((1414, 26, 1722, 49), "供需匹配程度")]:
    erase(im, (x0 - 3, y0 - 3 + dy, x1 + 3, y1 + 3 + dy))
    write(im, s, ((x0 + x1) / 2, (y0 + y1) / 2 + dy), 24)

for (x0, y0, x1, y1), s in [((24, 240, 39, 409), "需求满足率的提高量"), ((1040, 226, 1055, 424), "供需匹配程度的提高量")]:
    erase(im, (x0 - 3, y0 - 4 + dy, x1 + 3, y1 + 4 + dy))
    write(im, s, ((x0 + x1) / 2, (y0 + y1) / 2 + dy), 21, rot=90)

for (x0, y0, x1, y1) in [(455, 637, 681, 652), (1456, 637, 1682, 652)]:
    erase(im, (x0 - 3, y0 - 3 + dy, x1 + 3, y1 + 3 + dy))
    write(im, "场景序号（共100个）", ((x0 + x1) / 2, (y0 + y1) / 2 + dy), 21)

# 右图图例文字
legend = [
    ((1182, 74, 1246, 88), "原档位调整", 17), ((1329, 74, 1353, 88), "SCA", 16), ((1436, 74, 1462, 88), "粒子群", 16),
    ((1184, 97, 1223, 111), "SLSQP", 16), ((1328, 97, 1346, 111), "差分进化", 15), ((1435, 97, 1497, 111), "WMMSE", 16),
]
for (x0, y0, x1, y1), s, size in legend:
    erase(im, (x0 - 2, y0 - 2 + dy, int(max(x1 + 4, x0 + font(size).getlength(s) + 2)), y1 + 2 + dy))
    write(im, s, (x0, (y0 + y1) / 2 + dy), size, anchor="lm")

save_png(im, "W10_逐场景改善.png")
