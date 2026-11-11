# -*- coding: utf-8 -*-
"""W12_负载不均分布.png：报告现成图改文字，不是重画。

原图：D:\\paper\\outputs\\experiments\\first_stage_s1_01_20260930\\audit_precision_review\\figures\\load_cv_distribution.png
（报告 第一阶段/实验报告/02 里的“负载CV场景分布”，webp 格式，转存成真 PNG）。
改动（绘图区的箱线一个像素没动，也没缩放）：
  1. 英文标题换成中文标题；
  2. 纵轴英文标签“Std / mean of all 100 beam loads”→“负载不均程度（标准差÷平均）”；
  3. 横轴刻度的实验代号 R0/R1/R2/C0/C1/C2/P0 换成人话（规则不调整、规则＋局部搜索……），下方补了一条白边。
运行：python W12_负载不均分布.py
"""
from _patch_tools import *

SRC = EXP + r"\first_stage_s1_01_20260930\audit_precision_review\figures\load_cv_distribution.png"
im = load_any(SRC)
im = pad(im, bottom=22)

NAMES = ["规则\n不调整", "规则\n局部搜索", "规则\nPPO微调", "聚类\n不调整", "聚类\n局部搜索", "聚类\nPPO微调", "纯PPO\n空布局"]
XS = [199, 386, 573, 760, 947, 1134, 1321]

erase(im, (300, 18, 1220, 58))
write(im, "七组在100个考核场景上的负载不均程度（越小越均匀）", (760, 38), 27)

erase(im, (20, 170, 46, 550))
write(im, "负载不均程度（100个波束负载的标准差÷平均）", (32, 360), 20, rot=90)

erase(im, (107, 670, 1415, im.height))
for x, n in zip(XS, NAMES):
    write(im, n, (x, 672), 21, anchor="ma", spacing=2)

save_png(im, "W12_负载不均分布.png")
