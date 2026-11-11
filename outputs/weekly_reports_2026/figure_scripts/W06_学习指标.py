# -*- coding: utf-8 -*-
"""W06_学习指标：报告现成图改标签。
原图：D:\paper\outputs\verification\g09_completion_20261001\figures\learning_diagnostics.png
（报告 10_欧洲自然维状态与独立场景对照.md 第5节引用，实为 webp，转存成真 PNG）。
改的文字：第三个子图标题和图内“实线Q，虚线target”里的 target 改“目标值”；总标题里的“n=1”“无对数轴”改口语；第一个子图里带 N0到N4 代号的图例整块盖白，
在图下方加中文图例条（204维到2946维），颜色按原图图例取样。曲线、点、坐标轴没动。
"""
import _img_edit as E
SRC = r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b\orig\w06_learning_diagnostics.png"
OUT = r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W06_学习指标.png"
im = E.load(SRC)
cols = [im.getpixel((x, 103)) for x in (395, 510, 625, 740, 855)]
E.replace_text(im, (300, 0, 1700, 36), "每1万个交互步取一次学习日志的均值，圆点是最后一步的单独值", old_text="每10000交互步区间的日志均值；首窗8000次更新，400000端点n=1单独标点；无对数轴")
E.replace_text(im, (300, 626, 750, 652), "Q及目标值均值", old_text="Q及target均值")
E.replace_text_gray(im, (60, 672, 300, 712), "实线Q，虚线目标值", old_text="实线Q，虚线target")
E.cover(im, (366, 86, 934, 125))
names = ["204维", "207维", "246维", "746维", "2946维"]
im = E.legend_strip(im, list(zip(cols, names)), px=24, handle_w=50, gap=50, line_w=4, marker=False, y_pad=16)
E.finish(im, OUT)
