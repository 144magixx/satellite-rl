# -*- coding: utf-8 -*-
"""W08_固定温度学习量：报告现成图改标签。
原图：D:\paper\outputs\experiments\europe_tmg_20261001\verified_report_20261002\figures\T_学习量原始采样.png
（报告 12_温度记忆与干扰图独立场景对照.md 引用，实为 webp，转存成真 PNG）。
改的文字：8个子图标题开头的 T0、T1 去掉；图左边加空白边，写“固定温度＋加性Q”“固定温度＋联合Q”两行标签。
曲线、坐标轴、网格、图内图例（当前Q均值、目标Q均值等）没动。
"""
import numpy as np
import _img_edit as E
from _w08_common import title_rows
SRC = r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b\orig\w08_T_learn.png"
OUT = r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W08_固定温度学习量.png"
im = E.load(SRC)
rows = title_rows(im)
print("标题行", rows)
assert len(rows) == 2
tails = ["温度α（每50交互步记录原值）", "绝对TD误差（每50交互步记录原值）", "当前/目标Q（每50交互步原值）", "双Critic损失（每50交互步原值）"]
for (y0, y1), code in zip(rows, ["T0", "T1"]):
    segs = E.text_segments(im, (0, y0 - 2, im.width, y1 + 1), gap=100)
    print(segs)
    assert len(segs) == 4
    for sg, t in zip(segs, tails):
        E.replace_text(im, (sg[0] - 3, y0 - 4, sg[2] + 3, y1 + 6), t, old_text=f"{code} {t}")
im2 = E.extend_canvas(im, left=110)
bounds = [r[0] for r in rows] + [im.height - 75]
for i, name in enumerate(["固定温度＋加性Q", "固定温度＋联合Q"]):
    E.put_text_rot(im2, name, 50, (rows[i][0] + 40 + bounds[i + 1] - 20) / 2, 36, angle=90, bold=True)
E.finish(im2, OUT)
