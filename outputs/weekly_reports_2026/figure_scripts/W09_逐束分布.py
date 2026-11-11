# -*- coding: utf-8 -*-
"""W09_逐束分布.png：报告现成图改文字，不是重画。

原图：D:\\paper\\outputs\\experiments\\stage3_power_20261003\\report_20261003\\beams_total.png
（报告 第三阶段/实验报告/01 里的“全部目标束容量/需求与功率分布”，webp 格式，转存成真 PNG）。
文字改动（绘图区一个像素没动，也没缩放）：
  1. 标题两行（带“total预算”“T0训练seed 51001”“Jan25”）换成中文标题；
  2. 左图横轴“log10(C/D)，零容量置于−6”→“log10(容量/需求)，……”，并补上两条竖线的含义（虚线是需求的0.1倍，点线是1倍）。
右图横轴、图例原本就是中文，没动。
运行：python W09_逐束分布.py
"""
from _patch_tools import *

SRC = EXP + r"\stage3_power_20261003\report_20261003\beams_total.png"
im = load_any(SRC)

erase(im, (0, 0, im.width, 66))
write(im, "3000 W总功率预算：全部目标束的容量/需求比（左）和整束功率（右）分布", (975, 17), 26)
write(im, "1月25日的100个场景×每个场景3次动作的全部束，原始值，不平滑；重复动作不当作独立场景", (975, 47), 24)

erase(im, (395, 716, 672, 748))
write(im, "log10(容量/需求)，零容量置于−6；虚线0.1倍，点线1倍", (533, 733), 22)

save_png(im, "W09_逐束分布.png")
