# -*- coding: utf-8 -*-
"""W04 图：提速测试里每一次的原始计时（原版和优化后，两种状态布局，各3轮）。

原图：D:/paper/outputs/verification/async_telemetry_20260929/figures/steady_raw_samples.png
（报告 docs/第二阶段/实验报告/05_异步采集与训练提速验证.md “同工作量稳态训练”一节引用；本机转存在 .scratch_w1234/figures__steady_raw_samples.png）。
每次计时的原始数值本机没有，只有这张报告图（三轮中位数见报告表）。
做法：绘图区（折线、点、星号、网格）一个像素都不动，只改绘图区外的文字：
  顶部总标题 “同工作量交替测速：原始样本，无平滑（训练种子51001）” 整行裁掉（含种子，图题写在文档里）
  面板标题 “原状态网络（13令牌）” / “扩展状态网络（32令牌）” -> “204维状态（13个输入令牌）” / “2946维状态（32个输入令牌）”，和周报表格对上
  x 轴标签 “布局内实测顺序（每轮 A/B/B/A，共3轮）” -> “测试顺序（每轮：原版、优化、优化、原版，共3轮）”（去掉 A/B 代号）
图例（原版同步 / 组合优化＋异步）和底部脚注原样保留。
运行：python outputs/weekly_reports_2026/figure_scripts/W04_每轮测速原始点.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *

SCR = os.path.normpath(os.path.join(HERE, "..", ".scratch_w1234"))
im = load_any(os.path.join(SCR, "figures__steady_raw_samples.png"))
W, H = im.size

# 面板标题（左面板 x 112～988，中心 550；右面板 x 1020～1896，中心 1458；标题在 y=105～132，坐标轴顶在 y=140）
for cx, new in [(550, "204维状态（13个输入令牌）"), (1458, "2946维状态（32个输入令牌）")]:
    erase(im, (cx - 300, 98, cx + 300, 137))
    write(im, new, (cx, 119), 27)

# x 轴标签（y=626～648）
for cx in (550, 1458):
    erase(im, (cx - 380, 622, cx + 380, 652))
    write(im, "测试顺序（每轮：原版、优化、优化、原版，共3轮）", (cx, 637), 22)

im = im.crop((0, 70, W, H))     # 总标题在 y=13～40
save_png(im, "W04_每轮测速原始点.png")
