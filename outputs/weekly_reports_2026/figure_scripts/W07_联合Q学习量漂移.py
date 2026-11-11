# -*- coding: utf-8 -*-
"""W07_联合Q学习量漂移：报告现成图改标签。
原图：D:\paper\outputs\experiments\europe_joint_action_20261001\analysis\root_diagnostics\G10_学习诊断对数尺度.png
（报告 11_联合动作四组独立场景对照.md 第5节引用，实为 webp，转存成真 PNG）。
改的文字：总标题去掉实验组代号和训练种子编号；第一个子图里带 J0到J3 代号的图例整块盖白，在图下方加中文图例条；
最下面两个子图标题和底部说明里的“symlog”改“对称对数轴”。曲线、坐标轴、网格没动。
"""
import _img_edit as E
SRC = r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b\orig\w07_G10_learning.png"
OUT = r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W07_联合Q学习量漂移.png"
im = E.load(SRC)
E.replace_text(im, (500, 20, 2400, 100), "四组模型的学习指标：用完整联合Q的两组后期越变越大", old_text="G10学习诊断：联合Q组后期数值漂移（训练种子51001）")
E.replace_text(im, (400, 1380, 1000, 1428), "当前Q均值（对称对数轴）", old_text="当前Q均值（symlog）")
E.replace_text(im, (1800, 1380, 2500, 1428), "目标Q均值（对称对数轴）", old_text="目标Q均值（symlog）")
E.replace_text(im, (400, 2100, 2500, 2152), "淡线是每50个交互步记录的原始值，实线是向后100个记录点的均值，约5000步。Q值用对称对数轴（线性阈值1）；图里是采样点，不是全部398001次更新。",
               old_text="淡线=每50交互步记录的原始学习诊断；实线=向后100记录点的算术均值，约5000步。Q与目标Q采用symlog（线性阈值1）；这些诊断点并非全部398001次更新。")
E.cover(im, (98, 206, 542, 250))
names = ["原策略＋加性Q", "原策略＋联合Q", "条件策略＋加性Q", "条件策略＋联合Q"]
im = E.legend_strip(im, [(E.TAB10[i], names[i]) for i in range(4)], px=30, handle_w=60, gap=60, line_w=5, marker=False, y_pad=20)
E.finish(im, OUT)
