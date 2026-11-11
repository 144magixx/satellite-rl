# -*- coding: utf-8 -*-
"""W08_T组训练与验证：报告现成图改标签，说明见 _w08_common.py 开头。
原图：D:\paper\outputs\experiments\europe_tmg_20261001\verified_report_20261002\figures\T_训练与固定验证.png
改的文字：子图标题开头的代号换成周报表格里的叫法，其余不动。"""
from _w08_common import relabel
relabel(r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b\orig\w08_T_trainval.png",
        r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W08_T组训练与验证.png",
        ["T0", "T1"], ["固定温度＋加性Q", "固定温度＋联合Q"])
