# -*- coding: utf-8 -*-
"""W05_验证曲线：报告现成图改标签。
原图：D:\paper\outputs\experiments\europe_ablation_20260929\analysis\figures\validation_curves.png
（报告 07_欧洲六组独立场景对照.md 第5节引用，文件实为 webp，用 Pillow 转存成真 PNG）。
只动文字：把绘图区里带代号的图例框（G07、A_dropout_zero 等）盖白，在图下方加一条中文图例条，
颜色与原图曲线一一对应（G07蓝、A橙、B绿、C1红、C2紫、C3棕、C4粉）。曲线、坐标轴、网格、标题没有改动。
"""
import _img_edit as E
SRC = r"D:\paper\outputs\experiments\europe_ablation_20260929\analysis\figures\validation_curves.png"
OUT = r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W05_验证曲线.png"
im = E.load(SRC)
E.cover(im, (796, 530, 1170, 650))
names = ["原模型", "A 关dropout", "B 打乱场景", "C1 少量全局信息", "C2 资源概况", "C3 布局与需求", "C4 布局、分配与干扰"]
entries = [(E.TAB10[i], names[i]) for i in range(7)]
im = E.legend_strip(im, entries, px=24, handle_w=50, gap=36, line_w=5, y_pad=18)
E.finish(im, OUT)
