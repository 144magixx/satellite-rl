# -*- coding: utf-8 -*-
"""W04 图：工作台监控页读取训练指标的耗时（原版全读、新版首次、缓存命中、追加少量数据后增量读取）。

原图：D:/paper/outputs/verification/async_telemetry_20260929/figures/monitor_reading.png
（报告 docs/第二阶段/实验报告/05_异步采集与训练提速验证.md “读端单独收益与失败记录”一节引用；本机转存在 .scratch_w1234/figures__monitor_reading.png）。
读取耗时的原始记录本机没有，只有这张报告图（柱上数字1.603/1.563/0.053/0.068/0.065 和报告正文的
冷读1.72、1.60、1.50 秒（均值约1.6）、新冷读1.56 秒、缓存命中0.047～0.058 秒、追加后0.068 秒、TensorBoard 增量0.065 秒对得上）。
做法：图里的标题、轴标签、柱下标签原本就是中文且不含代号，没有改任何文字，只是用 Pillow 重存成标准 PNG。
运行：python outputs/weekly_reports_2026/figure_scripts/W04_监控页读取耗时.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *

SCR = os.path.normpath(os.path.join(HERE, "..", ".scratch_w1234"))
im = load_any(os.path.join(SCR, "figures__monitor_reading.png"))
save_png(im, "W04_监控页读取耗时.png")
