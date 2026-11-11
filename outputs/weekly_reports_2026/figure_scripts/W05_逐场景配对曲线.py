# -*- coding: utf-8 -*-
"""W05_逐场景配对曲线：报告现成图改标签。
原图：D:\paper\outputs\experiments\europe_ablation_20260929\analysis\figures\independent_test_summary_paired.png
（报告 07_欧洲六组独立场景对照.md 第4节引用，实为 webp，转存成真 PNG）。
改的文字：标题里的“新01-18”“SI与SGM”换成中文说法，横轴说明里的“SHA”改口语，
图内带代号的图例（G07、A_dropout_zero 等）盖白后在图下方加中文图例条。曲线、坐标轴、网格没有动。
"""
import _img_edit as E
SRC = r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b\orig\w05_independent_paired.png"
OUT = r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W05_逐场景配对曲线.png"
im = E.load(SRC)
E.replace_text(im, (300, 0, 2100, 60), "1月18日的100个独立场景：逐场景的需求满足率和供需匹配程度", old_text="新01-18独立共同场景：逐场景SI与SGM")
E.replace_text(im, (300, 1325, 2100, 1360), "场景序号（按场景编号排序）；每个模型先对3次动作取平均", old_text="按场景SHA排序的配对序号；每模型先平均3次动作")
E.cover(im, (112, 574, 887, 646))
names = ["原模型", "A 关dropout", "B 打乱场景", "C1 少量全局信息", "C2 资源概况", "C3 布局与需求", "C4 布局、分配与干扰"]
im = E.legend_strip(im, [(E.TAB10[i], names[i]) for i in range(7)], px=24, handle_w=50, gap=36, line_w=4, marker=False, y_pad=18)
E.finish(im, OUT)
