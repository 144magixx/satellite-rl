# -*- coding: utf-8 -*-
"""W08_M组训练与验证：报告现成图改标签，说明见 _w08_common.py 开头。
原图：D:\paper\outputs\experiments\europe_tmg_20261001\verified_report_20261002\figures\M_训练与固定验证.png
改的文字：子图标题开头的代号换成周报表格里的叫法，其余不动。"""
from _w08_common import relabel
relabel(r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b\orig\w08_M_trainval.png",
        r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W08_M组训练与验证.png",
        ["M0", "M1", "M2", "M3"], ["不加记忆", "加上一束动作", "GRU看16步", "Transformer看16步"])
