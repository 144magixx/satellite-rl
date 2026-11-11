# -*- coding: utf-8 -*-
"""W12_逐场景覆盖分布.png：报告现成图改文字，不是重画。

原图：D:\\paper\\outputs\\experiments\\first_stage_s1_01_20260930\\audit_precision_review\\figures\\all_scene_distributions.png
（报告 第一阶段/实验报告/02 里的“七组逐场景覆盖与可行分布”，webp 格式，转存成真 PNG）。
改动（绘图区的箱线一个像素没动，也没缩放）：
  1. 只保留前两个小图（需求覆盖、没覆盖的需求量），裁掉第三个“可行场景”小图（除了纯PPO那一格，其余六组都是同一条线，没有信息，
     可行场景数在表里有）；
  2. 两个小图的英文标题换成中文；
  3. 横轴刻度的实验代号 R0/R1/R2/C0/C1/C2/P0 换成人话两行字（规则不调整、规则＋局部搜索……），下方补了一条白边放两行字。
运行：python W12_逐场景覆盖分布.py
"""
from _patch_tools import *

SRC = EXP + r"\first_stage_s1_01_20260930\audit_precision_review\figures\all_scene_distributions.png"
im = load_any(SRC).crop((0, 0, 1620, 704))
im = pad(im, bottom=34)

NAMES = ["规则\n不调整", "规则\n局部搜索", "规则\nPPO微调", "聚类\n不调整", "聚类\n局部搜索", "聚类\nPPO微调", "纯PPO\n空布局"]
XS1 = [166, 266, 365, 465, 565, 665, 765]
XS2 = [946, 1046, 1146, 1245, 1345, 1445, 1545]

for (x0, y0, x1, y1), s in [((346, 26, 587, 51), "各场景的需求覆盖比例"), ((1068, 26, 1424, 51), "各场景里没被覆盖的需求量（Mbps）")]:
    erase(im, (x0 - 4, y0 - 4, x1 + 4, y1 + 4))
    write(im, s, ((x0 + x1) / 2, 39), 26)

erase(im, (118, 652, 1610, im.height))
for xs in (XS1, XS2):
    for x, n in zip(xs, NAMES):
        write(im, n, (x, 655), 20, anchor="ma", spacing=2)

save_png(im, "W12_逐场景覆盖分布.png")
