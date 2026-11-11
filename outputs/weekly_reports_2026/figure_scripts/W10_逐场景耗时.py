# -*- coding: utf-8 -*-
"""W10_逐场景耗时.png：报告现成图改文字，不是重画。

原图：D:\\paper\\outputs\\experiments\\stage3_all_routes_20261003\\report_20261003\\total_scene_time.png
（报告 第三阶段/实验报告/02 里的“3000 W预算逐场景三动作合计时间”，webp 格式，转存成真 PNG）。
文字改动（绘图区的线一个像素没动，也没缩放）：
  1. 标题（英文，带“total”）换成中文标题，上方补了一条白边来放两行标题；
  2. 纵轴“3-action measured-stage sum (s)”→“三次动作的耗时合计（秒）”，横轴“Common scene index (SHA order)”→“场景序号（共100个）”；
  3. 图例里的方法英文名换成中文：none→不细调，discrete→原档位调整，de→差分进化，pso→粒子群；slsqp、sca、wmmse 保留算法名的写法。
运行：python W10_逐场景耗时.py
"""
from _patch_tools import *

SRC = EXP + r"\stage3_all_routes_20261003\report_20261003\total_scene_time.png"
im = load_any(SRC)
TOP = 40
im = pad(im, top=TOP)          # 在上面补一条白边放标题，原图整体下移 TOP 像素
dy = TOP

# 原标题（英文）擦掉
erase(im, (80, 20 + dy, im.width, 54 + dy))
write(im, "3000 W总功率预算：每个场景三次动作的耗时合计（七种做法）", (889, 16), 26)
write(im, "1月26日的100个场景，每次动作的实测耗时相加，不平滑", (889, 46), 22)

# 纵轴、横轴标签
erase(im, (20, 190 + dy, 44, 546 + dy))
write(im, "三次动作的耗时合计（秒）", (32, 368 + dy), 21, rot=90)
erase(im, (700, 718 + dy, 1076, 746 + dy))
write(im, "场景序号（共100个）", (888, 731 + dy), 21)

# 图例文字：(原文字框, 新文字, 字号)
legend = [
    ((1160, 76, 1204, 94), "不细调", 18), ((1326, 76, 1371, 94), "SLSQP", 16),
    ((1465, 76, 1486, 94), "差分进化", 16), ((1589, 76, 1660, 94), "WMMSE", 16),
    ((1159, 101, 1232, 127), "原档位调整", 18), ((1326, 101, 1354, 127), "SCA", 16),
    ((1466, 101, 1496, 127), "粒子群", 16),
]
for (x0, y0, x1, y1), s, size in legend:
    erase(im, (x0 - 2, y0 - 2 + dy, int(max(x1 + 4, x0 + font(size).getlength(s) + 2)), y1 + 2 + dy))
    cy = (y0 + y1) / 2 + dy if y0 < 90 else 114 + dy
    write(im, s, (x0, cy), size, anchor="lm")

save_png(im, "W10_逐场景耗时.png")
