# -*- coding: utf-8 -*-
"""W11_逐场景耗时.png：报告现成图改文字，不是重画。

原图：D:\\paper\\outputs\\experiments\\stage3_si_priority_20261003\\report_20261003\\total_scene_time.png
（报告 第三阶段/实验报告/03 里的“3000 W预算每景完整与追加时间”，webp 格式，转存成真 PNG）。
文字改动（绘图区的线一个像素没动，也没缩放）：
  1. 横轴“场景序号（SHA排序）”→“场景序号”；
  2. 两个图例里的 A/B/C 和“SI”换成人话：A: 原SCA→原SCA，B: SI优先+原估计→先提满足率＋原近似，
     C: SI优先+改进估计→先提满足率＋新近似。
纵轴标签原本就是中文，没动。左图是每个场景三次动作的完整耗时合计，右图只算追加的那一段。
运行：python W11_逐场景耗时.py
"""
from _patch_tools import *

SRC = EXP + r"\stage3_si_priority_20261003\report_20261003\total_scene_time.png"
im = load_any(SRC)

for cx in (525, 1491):
    erase(im, (cx - 110, 664, cx + 110, 694))
    write(im, "场景序号（共100个）", (cx, 679), 22)

rows1 = [(784, 52, "原SCA"), (784, 82, "先提满足率＋原近似"), (784, 113, "先提满足率＋新近似")]
for x0, cy, s in rows1:
    erase(im, (778, cy - 11, 952, cy + 11))
    write(im, s, (x0, cy), 18, anchor="lm")
rows2 = [(1751, 52, "先提满足率＋原近似"), (1751, 82, "先提满足率＋新近似")]
for x0, cy, s in rows2:
    erase(im, (1745, cy - 11, 1918, cy + 11))
    write(im, s, (x0, cy), 18, anchor="lm")

save_png(im, "W11_逐场景耗时.png")
