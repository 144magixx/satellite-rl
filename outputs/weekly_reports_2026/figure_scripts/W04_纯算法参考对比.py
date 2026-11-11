# -*- coding: utf-8 -*-
"""W04 图：优化后的训练和“关掉观测日志的纯算法参考”相比，每500次更新的耗时。

原图：D:/paper/outputs/verification/async_telemetry_20260929/figures/pure_reference.png
（报告 docs/第二阶段/实验报告/05_异步采集与训练提速验证.md “验证/保存边界与纯算法参考”一节引用；本机转存在 .scratch_w1234/figures__pure_reference.png）。
原始数值本机没有，只有这张报告图（柱上数字36.00/35.74、46.12/45.16 与报告正文一致）。
做法：绘图区（柱、柱顶数字、网格）一个像素都不动，只改绘图区外的文字：
  面板标题 “13令牌” / “32令牌” -> “204维状态（13个输入令牌）” / “2946维状态（32个输入令牌）”，和周报表格对上
顶部标题、柱下标签、y 轴标签、底部脚注原本就是中文且不含代号，原样保留。
运行：python outputs/weekly_reports_2026/figure_scripts/W04_纯算法参考对比.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *

SCR = os.path.normpath(os.path.join(HERE, "..", ".scratch_w1234"))
im = load_any(os.path.join(SCR, "figures__pure_reference.png"))

# 面板标题在 y=104～130，坐标轴顶在 y=140；左面板中心 x≈484，右面板中心 x≈1352
for cx, new in [(484, "204维状态（13个输入令牌）"), (1352, "2946维状态（32个输入令牌）")]:
    erase(im, (cx - 300, 98, cx + 300, 137))
    write(im, new, (cx, 118), 27)

save_png(im, "W04_纯算法参考对比.png")
