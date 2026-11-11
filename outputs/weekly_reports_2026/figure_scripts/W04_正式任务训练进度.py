# -*- coding: utf-8 -*-
"""W04 图：三个正在跑的训练任务，换成新版代码前后各观察约120秒内完成的交互和更新次数。

原图：D:/paper/outputs/verification/async_telemetry_20260929/figures/production_windows.png
（报告 docs/第二阶段/实验报告/05_异步采集与训练提速验证.md “正式任务切换前后的观察”一节引用；本机转存在 .scratch_w1234/figures__production_windows.png）。
每5秒一个点的原始读数本机没有，只有这张报告图（折线终点和报告表里的步/秒大致对得上：
13.57步/秒×120秒≈1630，6.99×120≈840）。
做法：绘图区（折线、网格、图内图例）一个像素都不动，只改绘图区外的面板标题（原来是 A / B / C1 代号）：
  “A: 关闭dropout” -> “关闭dropout的训练”
  “B: 场景顺序随机” -> “每轮随机场景顺序的训练”
  “C1: 少量全局状态” -> “只加少量全局状态的训练”
顶部总标题、轴标签原本就是中文且不含代号，原样保留。
运行：python outputs/weekly_reports_2026/figure_scripts/W04_正式任务训练进度.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *

SCR = os.path.normpath(os.path.join(HERE, "..", ".scratch_w1234"))
im = load_any(os.path.join(SCR, "figures__production_windows.png"))

# 面板标题在 y=121～151，坐标轴顶在 y=157；三个面板 x 范围 125～748、779～1402、1433～2056
panels = [((125 + 748) / 2, "关闭dropout的训练"),
          ((779 + 1402) / 2, "每轮随机场景顺序的训练"),
          ((1433 + 2056) / 2, "只加少量全局状态的训练")]
for cx, new in panels:
    erase(im, (int(cx - 300), 114, int(cx + 300), 154))
    write(im, new, (cx, 136), 29)

save_png(im, "W04_正式任务训练进度.png")
