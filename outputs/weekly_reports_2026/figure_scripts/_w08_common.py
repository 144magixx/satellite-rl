# -*- coding: utf-8 -*-
"""W08 训练与固定验证图（T、M、G 三张）的共用处理：报告现成图改标签。
原图：D:\paper\outputs\experiments\europe_tmg_20261001\verified_report_20261002\figures\{T,M,G}_训练与固定验证.png
（报告 12_温度记忆与干扰图独立场景对照.md 引用，实为 webp，转存成真 PNG）。
改的文字：每个子图标题开头的 T0、T1、M0到M3、G0到G2 去掉，标题只留“训练回报”“固定验证”等；
在图左边加一条空白边，用周报表格里的叫法（固定温度＋加性Q、不加记忆、整段编码等）写每一行是哪个模型。
曲线、点、坐标轴、网格、图例没动（原图图例是“终态需求满足率 %”“供需匹配程度 ×100”，没有代号）。
"""
import numpy as np
import _img_edit as E

def title_rows(im):
    a = np.asarray(im.crop((225, 0, 625, im.height))).astype(int)
    m = (a.min(axis=2) < 235).any(axis=1)
    runs = []; s = None
    for y, v in enumerate(m):
        if v and s is None: s = y
        if (not v) and s is not None:
            runs.append([s, y]); s = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 6: merged[-1][1] = r[1]
        else: merged.append(r)
    rows = []
    for i, r in enumerate(merged):
        h = r[1] - r[0]
        if 20 <= h <= 32 and i + 1 < len(merged) and (merged[i + 1][1] - merged[i + 1][0]) <= 4 and merged[i + 1][0] - r[1] < 15:
            rows.append(r)
    return rows

def relabel(src, out, codes, names, row_px=40):
    im = E.load(src)
    rows = title_rows(im)
    assert len(rows) == len(codes), (len(rows), len(codes))
    for (y0, y1), code in zip(rows, codes):
        segs = E.text_segments(im, (0, y0 - 2, im.width, y1 + 1), gap=120)
        assert len(segs) == 3, segs
        olds = [f"{code} 训练回报：原值与向后100回合均值", f"{code} 固定验证：每节点20场景×3动作", f"{code} 固定验证：终态服务"]
        news = ["训练回报：原值与向后100回合均值", "固定验证：每节点20场景×3动作", "固定验证：终态服务"]
        for sg, old, new in zip(segs, olds, news):
            E.replace_text(im, (sg[0] - 3, y0 - 4, sg[2] + 3, y1 + 6), new, old_text=old)
    # 左边加一条行标签
    LEFT = 130
    im2 = E.extend_canvas(im, left=LEFT)
    bounds = [r[0] for r in rows] + [im.height - 75]
    for i, name in enumerate(names):
        top = rows[i][0] + 40; bot = bounds[i + 1] - 20
        E.put_text_rot(im2, name, 55, (top + bot) / 2, row_px, angle=90, bold=True)
    E.finish(im2, out)
