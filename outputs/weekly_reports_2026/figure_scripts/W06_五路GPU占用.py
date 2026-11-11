# -*- coding: utf-8 -*-
"""W06_五路GPU占用：报告现成图裁剪。
原图：D:\paper\outputs\verification\g09_remote_migration_20260930\production_gpu_updates.png
（报告 09_4090D迁移与并行验证.md 第6节引用，实为 webp，转存成真 PNG）。
只保留上面两个子图（GPU利用率、显存），裁掉第三个带 N0到N4 代号图例的子图。没有改任何文字和像素。
"""
import _img_edit as E
SRC = r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b\orig\w06_production_gpu_updates.png"
OUT = r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W06_五路GPU占用.png"
im = E.load(SRC)
im = im.crop((0, 0, im.width, 900))
E.finish(im, OUT)
