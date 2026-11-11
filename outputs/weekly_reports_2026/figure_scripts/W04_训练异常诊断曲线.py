# -*- coding: utf-8 -*-
"""W04 图：第一轮欧洲训练的异常诊断（同场景回报、第2轮到第6轮的需求满足率差、固定验证、更新损失）。

原图：D:/paper/outputs/experiments/europe_adsb_20260928/diagnosis/curves/g07_training_diagnosis.png
（报告 docs/第二阶段/方案/06_欧洲训练异常诊断与下一阶段方案.md 第1.2节引用；本机转存在 .scratch_w1234/curves__g07_training_diagnosis.png）。
逐回合原始日志本机没有，只有这张报告图。
做法：只裁掉顶部一行总标题 “G07日志诊断 | 训练种子51001 | 训练576场景循环，验证20固定场景”（带代号和种子，图题写在文档里）；
四个面板的标题、图例、轴标签原本就是中文且不含代号，一个像素都没动；Actor / Critic 是网络名，保留。
运行：python outputs/weekly_reports_2026/figure_scripts/W04_训练异常诊断曲线.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *

SCR = os.path.normpath(os.path.join(HERE, "..", ".scratch_w1234"))
im = load_any(os.path.join(SCR, "curves__g07_training_diagnosis.png"))
W, H = im.size
im = im.crop((0, 50, W, H))     # 总标题在 y=7～43，面板标题从 y=57 开始
save_png(im, "W04_训练异常诊断曲线.png")
