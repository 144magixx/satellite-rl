# -*- coding: utf-8 -*-
"""W07_奖励训练与验证：报告现成图改标签。
原图：D:\paper\outputs\experiments\europe_joint_action_20261001\analysis\root_diagnostics\G10_奖励训练与固定验证分面.png
（报告 11_联合动作四组独立场景对照.md 第4节引用，实为 webp，转存成真 PNG）。
改的文字：总标题里的实验组代号和训练种子编号去掉；8个子图标题里的 J0到J3 换成周报表格里的叫法
（原策略＋加性Q、原策略＋联合Q、条件策略＋加性Q、条件策略＋联合Q）。曲线、点、坐标轴、网格没动。
"""
import _img_edit as E
SRC = r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b\orig\w07_G10_reward.png"
OUT = r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W07_奖励训练与验证.png"
im = E.load(SRC)
E.replace_text(im, (400, 30, 2500, 110), "四组模型的训练回报和固定验证回报：用完整联合Q的两组后期回落", old_text="G10训练奖励与固定验证原生累计回报：联合Q组后段回落（训练种子51001）")
names = ["原策略＋加性Q", "原策略＋联合Q", "条件策略＋加性Q", "条件策略＋联合Q"]
old_l = ["J0 训练：前500均值79.6，后500均值85.3", "J1 训练：前500均值63.8，后500均值25.4", "J2 训练：前500均值77.3，后500均值83.1", "J3 训练：前500均值54.9，后500均值23.6"]
old_r = ["J0 固定验证：后段高于早段4.7（后5点−早5点）", "J1 固定验证：后段回落36.4（后5点−早5点）", "J2 固定验证：后段高于早段3.1（后5点−早5点）", "J3 固定验证：后段回落32.5（后5点−早5点）"]
ys = [194, 791, 1388, 1985]
for i, y in enumerate(ys):
    E.replace_text(im, (300, y - 18, 1300, y + 18), old_l[i].replace(f"J{i}", names[i]), old_text=old_l[i])
    E.replace_text(im, (1700, y - 18, 2700, y + 18), old_r[i].replace(f"J{i}", names[i]), old_text=old_r[i])
E.finish(im, OUT)
