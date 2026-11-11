# -*- coding: utf-8 -*-
"""W12_覆盖图对比.png：报告现成图改文字，不是重画（真实生成的地图类图，不是截图）。

原图：D:\\paper\\outputs\\experiments\\first_stage_s1_01_20260930\\coverage_maps_10_scenes\\compare_scene_055.png
（报告 第一阶段/实验报告/03 里的“场景055七组覆盖对照”，webp 格式，转存成真 PNG，原宽2416，右侧白边裁到2400，没有缩放）。
文字改动（地图、波束轮廓、飞机点、色条一个像素没动）：
  1. 总标题“Scene 055: paired seven-group geometric coverage”换成中文；
  2. 七个小图标题：实验代号 R0/R1/R2/C0/C1/C2/P0 换成方法名，第二行 “cover / CV / missed” 换成 需求覆盖 / 负载不均 / 漏覆盖，
     数字照原图抄回（99.76%/0.943/6、99.91%/0.903/3、99.69%/0.944/7、99.70%/0.843/6、99.88%/0.802/3、99.89%/0.836/3、98.22%/0.752/26），
     其中聚类＋局部搜索（0.802、3架）和纯PPO（98.22%、0.752、26架）与报告03正文一致；
  3. 右下角图例块（Shared symbols……）换成中文；色条标签“Beam demand (Mbps)”→“波束需求（Mbps）”。
运行：python W12_覆盖图对比.py
"""
from _patch_tools import *

SRC = EXP + r"\first_stage_s1_01_20260930\coverage_maps_10_scenes\compare_scene_055.png"
im = load_any(SRC)

erase(im, (760, 14, 1495, 52))
write(im, "同一个考核场景（1月22日13:17 UTC）下七组方法的波束覆盖对照", (1126, 33), 28)

groups = [
    ("规则布局，不调整", "需求覆盖99.76%  负载不均0.943  漏覆盖6架"),
    ("规则布局＋局部搜索", "需求覆盖99.91%  负载不均0.903  漏覆盖3架"),
    ("规则布局＋PPO微调", "需求覆盖99.69%  负载不均0.944  漏覆盖7架"),
    ("聚类布局，不调整", "需求覆盖99.70%  负载不均0.843  漏覆盖6架"),
    ("聚类布局＋局部搜索", "需求覆盖99.88%  负载不均0.802  漏覆盖3架"),
    ("聚类布局＋PPO微调", "需求覆盖99.89%  负载不均0.836  漏覆盖3架"),
    ("纯PPO，从空布局构建", "需求覆盖98.22%  负载不均0.752  漏覆盖26架"),
]
centers_row1 = [302, 876, 1450, 2024]
centers_row2 = [302, 876, 1450]
slots = [(cx, 98, 118) for cx in centers_row1] + [(cx, 741, 760) for cx in centers_row2]
for (cx, y1, y2), (name, stat) in zip(slots, groups):
    erase(im, (cx - 195, y1 - 9, cx + 195, y2 + 11))
    write(im, f"{name}  13:17 UTC", (cx, y1), 20)
    write(im, stat, (cx, y2 + 1), 17)

# 右下角图例块
erase(im, (1790, 770, im.width, 1170))
write(im, "图例", (1803, 790), 30, anchor="lm", bold=True)
RED = (200, 35, 50)
lines = [
    (852, "• 深色点：被波束几何覆盖的飞机", BLACK),
    (900, "× 红色：没被覆盖的飞机", RED),
    (950, "+ ：波束中心", BLACK),
    (1007, "轮廓：静止卫星射线圆锥与球形地球表面的交线", BLACK),
    (1063, "颜色：该束汇总的需求（Mbps），七组共用同一色标", BLACK),
    (1121, "开口的轮廓：被地球边缘、欧洲范围或10°仰角截断。", BLACK),
    (1148, "这里的覆盖只是几何覆盖，不是容量，也不是干扰。", BLACK),
]
for y, s, c in lines:
    write(im, s, (1803, y), 20, anchor="lm", fill=c)

erase(im, (1020, 1364, 1236, 1393))
write(im, "波束需求（Mbps）", (1127, 1379), 22)

im = im.crop((0, 0, 2400, im.height))
save_png(im, "W12_覆盖图对比.png")
