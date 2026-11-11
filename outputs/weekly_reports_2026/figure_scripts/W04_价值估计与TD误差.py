# -*- coding: utf-8 -*-
"""W04 图：第一轮欧洲训练里价值估计（Q值）、平均TD误差、梯度裁剪比例和探索温度随训练步数的变化。

原图：D:/paper/outputs/experiments/europe_adsb_20260928/diagnosis/curves/g07_q_td_clip_diagnosis.png
（报告 docs/第二阶段/方案/06_欧洲训练异常诊断与下一阶段方案.md 第1.1节引用；本机转存在 .scratch_w1234/curves__g07_q_td_clip_diagnosis.png）。
逐更新原始日志本机没有，只有这张报告图。
做法：只裁掉顶部一行总标题 “G07价值学习与梯度诊断 | 398001次更新 | 梯度范数记录为裁剪前”（带实验代号，图题写在文档里）；
四个面板的标题、图例、轴标签原本就是中文且不含代号，一个像素都没动；Actor / Critic 是网络名，保留。
运行：python outputs/weekly_reports_2026/figure_scripts/W04_价值估计与TD误差.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *

SCR = os.path.normpath(os.path.join(HERE, "..", ".scratch_w1234"))
im = load_any(os.path.join(SCR, "curves__g07_q_td_clip_diagnosis.png"))
W, H = im.size
im = im.crop((0, 50, W, H))     # 总标题在 y=7～43，面板标题从 y=57 开始
save_png(im, "W04_价值估计与TD误差.png")
