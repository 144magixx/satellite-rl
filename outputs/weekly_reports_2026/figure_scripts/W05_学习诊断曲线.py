# -*- coding: utf-8 -*-
"""W05_学习诊断曲线：报告现成图改标签。
原图：D:\paper\outputs\experiments\europe_ablation_20260929\analysis\final_audit\learning_log_view\learning_metrics_4panel.png
（报告 07_欧洲六组独立场景对照.md 第5节引用，实为 webp，转存成真 PNG）。
改的文字：总标题（去掉“G07与G08”和训练种子编号）、顶部图例（G07、A_dropout_zero 等换成中文模型名）、
第三个子图标题和纵轴“Alpha”改“温度α”、右下子图标题和纵轴里的“target/symlog/linthresh”改中文、
横轴“env_step”去掉、纵轴“Critic 1 loss”改中文、底部说明文字改口语、右下子图里图例的“target均值”改“目标均值”。
曲线、坐标轴、刻度、网格没有动。
"""
from PIL import ImageDraw
import _img_edit as E
SRC = r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b\orig\w05_learning_4panel.png"
OUT = r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W05_学习诊断曲线.png"
im = E.load(SRC)

# 总标题
E.replace_text(im, (800, 5, 2000, 75), "七个模型的学习指标曲线（对数坐标）", old_text="G07与G08六组学习日志曲线（训练种子51001）")
# 子图标题
E.replace_text(im, (100, 1140, 1400, 1183), "温度α（对数轴）", old_text="Alpha（对数轴）")
E.replace_text(im, (1550, 1140, 2800, 1183), "Q均值与目标均值（对称对数轴，线性阈值10）", old_text="Q均值与target均值（symlog，linthresh=10）")
# 横轴
for box in [(630, 1085, 910, 1125), (2065, 1085, 2345, 1125), (630, 1962, 910, 2002), (2065, 1962, 2345, 2002)]:
    E.replace_text(im, box, "环境交互步数", old_text="环境交互步数 env_step")
# 纵轴
E.replace_text(im, (10, 500, 48, 900), "Critic 1 损失", old_text="Critic 1 loss", rotate=90)
E.replace_text(im, (10, 1500, 48, 1900), "温度α", old_text="Alpha", rotate=90)
E.replace_text(im, (1430, 1400, 1480, 1950), "Q均值 / 目标均值", old_text="Q均值 / target均值", rotate=90)
# 底部说明
E.replace_text(im, (600, 2060, 2200, 2105), "淡线是原始采样点，粗线是向后100个采样点的均值（每50步一个点，约5000个交互步）。图里是采样值，不是全部参数更新。",
               old_text="淡线：原始 learning 采样点；粗线：100采样点尾随均值（50步/点，约5000 env_step）。日志是采样值，不包含全部398001次参数更新。")
# 右下子图图例里的 target均值
E.replace_text(im, (1630, 1238, 1736, 1266), "目标均值", old_text="target均值", align="left")
# 顶部图例
E.cover(im, (560, 105, 2300, 152))
names = ["原模型", "A 关dropout", "B 打乱场景", "C1 少量全局信息", "C2 资源概况", "C3 布局与需求", "C4 布局、分配与干扰"]
px = 26; f = E.font(px)
hw = 52; gap = 44
widths = [hw + 10 + f.getlength(n) for n in names]
total = sum(widths) + gap * (len(names) - 1)
x = (im.width - total) / 2; cy = 129
d = ImageDraw.Draw(im)
for i, (n, w) in enumerate(zip(names, widths)):
    col = E.hexrgb(E.TAB10[i])
    d.line([x, cy, x + hw, cy], fill=col, width=5)
    d.text((x + hw + 10, cy), n, font=f, fill=(0, 0, 0), anchor="lm")
    x += w + gap
E.finish(im, OUT)
