# -*- coding: utf-8 -*-
"""W06_验证回报分面：报告现成图改标签。
原图：D:\paper\outputs\verification\g09_completion_20261001\figures\reward_by_state.png
（报告 10_欧洲自然维状态与独立场景对照.md 第4节引用，实为 webp，转存成真 PNG）。
改的文字：总标题（去掉实验组代号和种子编号）、10个子图标题里的 N0到N4 去掉，只留“204维”等状态维数。
左边是每回合训练回报，右边是固定验证节点，曲线、点、网格、坐标轴一个像素没动。
"""
import _img_edit as E
SRC = r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b\orig\w06_reward_by_state.png"
OUT = r"E:\satellite-rl\outputs\weekly_reports_2026\figures\W06_验证回报分面.png"
im = E.load(SRC)
E.replace_text(im, (200, 0, 1800, 40), "按状态维数分面：训练回报和验证回报都不平滑，验证每点是20个场景×3次动作", old_text="G09单种子51001：按模型分面保留原始波动；验证每点20场景×3动作，首节点10000步")
dims = ["204", "207", "246", "746", "2946"]
old_l = ["N0 204维：训练旧奖励回报", "N1 207维：训练旧奖励回报", "N2 246维：训练旧奖励回报", "N3 746维：训练旧奖励回报", "N4 2946维：训练旧奖励回报"]
old_r = ["N0 204维：固定验证回报原节点", "N1 207维：固定验证回报原节点", "N2 246维：固定验证回报原节点", "N3 746维：固定验证回报原节点", "N4 2946维：固定验证回报原节点"]
for i, y in enumerate([58, 460, 861, 1263, 1664]):
    E.replace_text(im, (200, y - 17, 1000, y + 17), f"{dims[i]}维：训练回报", old_text=old_l[i])
    E.replace_text(im, (1100, y - 17, 1900, y + 17), f"{dims[i]}维：固定验证回报", old_text=old_r[i])
E.finish(im, OUT)
