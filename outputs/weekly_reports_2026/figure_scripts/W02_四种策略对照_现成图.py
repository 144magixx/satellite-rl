# -*- coding: utf-8 -*-
"""W02 四张“报告现成图”的中文改标签版本（逐回合原始记录本机没有，只有报告里的图）。

原图：D:/paper/outputs/experiments/coverage_reward_20260924/comparison/visualization_random20_20260925/
（报告 docs/第二阶段/实验报告/01_四策略20随机场景对照分析报告.md 第3、4、5节引用；本机转存在 .scratch_w1234/visualizatio__*.png）
做法：绘图区（曲线、柱、热力块、网格）一个像素都不动；只改绘图区外的文字，最后整张图等比缩小一次到 2400 px 宽以内。
  1. si_sgm_20_scenes.png -> figures/W02_四种策略逐场景满足率与匹配度.png
     去掉顶部两行标题（图题写在文档里）；y 轴标签“需求满足率 SI（%）”“供需匹配程度 SGM（%）”改成
     “需求满足率（%）”“供需匹配程度（%）”；x 轴刻度的场景编号（83001…83098）改成 1～20，轴标签改成“20个随机场景（四种策略共用）”。
  2. frequency_occupancy_20_scenes.png -> figures/W02_四种策略频率槽占用.png   只去掉顶部两行标题，其余文字原样。
  3. power_distribution_20_scenes.png -> figures/W02_四种策略功率选择分布.png   只去掉顶部两行标题，其余文字原样。
  4. power_total_20_scenes.png -> figures/W02_四种策略每场景总功率.png
     去掉顶部标题（原图里标题和图例叠在一起），图例用同样的线型颜色重画；x 轴场景编号改成 1～20，轴标签改成“20个随机场景”。
面板标题里的“T-SAC”是本周对比里的方法名，保留。
运行：python outputs/weekly_reports_2026/figure_scripts/W02_四种策略对照_现成图.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np

SCR = os.path.normpath(os.path.join(HERE, "..", ".scratch_w1234"))


def src(name):
    return os.path.join(SCR, "visualizatio__" + name)


def x_ticks(im, y_off=5):
    """在横轴底脊线下方 y_off 处找刻度线的 x 位置（只返回绘图区内的）。"""
    a = np.asarray(im).min(axis=2)
    h, w = a.shape
    rows = np.where((a < 80).sum(axis=1) > w * 0.5)[0]
    spine = rows[-1]
    cols = np.where(a[spine + y_off] < 150)[0]
    groups, s, pv = [], cols[0], cols[0]
    for c in cols[1:]:
        if c - pv > 2:
            groups.append((s + pv) / 2)
            s = c
        pv = c
    groups.append((s + pv) / 2)
    return spine, [g for g in groups if g > 200]


def fig_si_sgm():
    im = load_any(src("si_sgm_20_scenes.png"))
    W, H = im.size
    spine, ticks = x_ticks(im)
    assert len(ticks) == 20, len(ticks)
    erase(im, (0, 0, W, 103))                       # 两行标题（图例在 105～136，保留）
    # y 轴标签
    erase(im, (0, 430, 62, 800))
    erase(im, (0, 1130, 62, 1560))
    write(im, "需求满足率（%）", (33, 595), 30, rot=90)
    write(im, "供需匹配程度（%）", (33, 1341), 30, rot=90)
    # x 轴刻度和标签
    erase(im, (0, spine + 13, W, H))
    for i, x in enumerate(ticks, 1):
        write(im, str(i), (x, spine + 38), 26)
    write(im, "20个随机场景（四种策略共用）", ((136 + 3083) / 2, spine + 112), 30)
    im = im.crop((0, 95, W, spine + 150))
    save_png(im, "W02_四种策略逐场景满足率与匹配度.png")


def fig_freq():
    im = load_any(src("frequency_occupancy_20_scenes.png"))
    W, H = im.size
    erase(im, (650, 0, 2800, 103))                  # 两行标题；左上角面板标题在 x<=583，保留
    im = im.crop((0, 70, W, H))
    save_png(im, "W02_四种策略频率槽占用.png")


def fig_power_dist():
    im = load_any(src("power_distribution_20_scenes.png"))
    W, H = im.size
    erase(im, (1000, 0, 2200, 103))                 # 两行标题；两个面板标题在 x<=954 和 x>=2247，保留
    im = im.crop((0, 70, W, H))
    save_png(im, "W02_四种策略功率选择分布.png")


def fig_power_total():
    im = load_any(src("power_total_20_scenes.png"))
    W, H = im.size
    spine, ticks = x_ticks(im)
    assert len(ticks) == 20, len(ticks)
    # 图例：标题和图例叠在一起，整块盖掉，用同样的线型颜色重画
    erase(im, (0, 0, W, 140))
    items = [("旧奖励 T-SAC", (32, 112, 180), "solid"), ("融合奖励 T-SAC", (214, 106, 49), "solid"),
             ("贪心", (38, 147, 104), "solid"), ("固定策略", (139, 85, 163), "solid"),
             ("总功率上限 3000 W", (85, 85, 85), "dash")]
    f = font(28)
    d = ImageDraw.Draw(im)
    hw, gap, sep = 66, 14, 56
    widths = [hw + gap + d.textlength(t, font=f) for t, _, _ in items]
    x = (136 + 3083) / 2 - (sum(widths) + sep * (len(items) - 1)) / 2
    y = 70
    for (t, col, kind), wd in zip(items, widths):
        if kind == "solid":
            d.line([(x, y), (x + hw, y)], fill=col, width=4)
            d.ellipse([x + hw / 2 - 7, y - 7, x + hw / 2 + 7, y + 7], fill=col)
        else:
            for k in range(0, hw, 22):
                d.line([(x + k, y), (x + min(k + 12, hw), y)], fill=col, width=3)
        d.text((x + hw + gap, y), t, font=f, fill=BLACK, anchor="lm")
        x += wd + sep
    # x 轴刻度和标签
    erase(im, (0, spine + 13, W, H))
    for i, tx in enumerate(ticks, 1):
        write(im, str(i), (tx, spine + 38), 26)
    write(im, "20个随机场景", ((150 + 3085) / 2, spine + 112), 30)
    im = im.crop((0, 35, W, spine + 150))
    save_png(im, "W02_四种策略每场景总功率.png")


if __name__ == "__main__":
    fig_si_sgm()
    fig_freq()
    fig_power_dist()
    fig_power_total()
