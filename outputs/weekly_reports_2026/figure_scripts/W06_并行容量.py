# -*- coding: utf-8 -*-
"""W06_并行容量：报告现成图改标签。
原图：D:\paper\outputs\verification\g09_remote_migration_20260930\parallel_capacity.png
（报告 09_4090D迁移与并行验证.md 第5节引用，实为 webp，转存成真 PNG）。
改的文字：总标题里的 seed 编号去掉，“活跃模型”改口语。柱子、数字、坐标轴没动。
"""
import _img_edit as E
SRC = r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b\orig\w06_parallel_capacity.png"
OUT = r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W06_并行容量.png"
im = E.load(SRC)
E.replace_text(im, (400, 0, 1700, 42), "4090D服务器短测：每个模型各更新24次", old_text="4090D工程短测：每个活跃模型24次更新，seed51001")
E.finish(im, OUT)
