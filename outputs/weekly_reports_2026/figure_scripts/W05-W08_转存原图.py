# -*- coding: utf-8 -*-
"""W05到W08：把报告引用的原图（D:\paper 下，实际是 webp 或 png）转存成真 PNG，放到 .work_w58b\orig。
各图脚本（W05_验证曲线.py 等）都从这个目录读原图。看完图可以把 .work_w58b 整个删掉，要重做时再跑这个脚本。"""
import os
from PIL import Image

base = r"E:\satellite-rl\outputs\weekly_reports_2026\.work_w58b"
os.makedirs(base + r"\orig", exist_ok=True)
A = r"D:\paper\outputs\experiments\europe_ablation_20260929"
src = {
    "w05_independent_paired": A + r"\analysis\figures\independent_test_summary_paired.png",
    "w05_validation_curves": A + r"\analysis\figures\validation_curves.png",
    "w05_learning_4panel": A + r"\analysis\final_audit\learning_log_view\learning_metrics_4panel.png",
    "w06_independent_scene_curves": r"D:\paper\outputs\verification\g09_completion_20261001\archive\outputs\experiments\europe_natural_state_20260930\analysis\figures\independent_scene_curves.png",
    "w06_reward_by_state": r"D:\paper\outputs\verification\g09_completion_20261001\figures\reward_by_state.png",
    "w06_learning_diagnostics": r"D:\paper\outputs\verification\g09_completion_20261001\figures\learning_diagnostics.png",
    "w06_parallel_capacity": r"D:\paper\outputs\verification\g09_remote_migration_20260930\parallel_capacity.png",
    "w06_production_gpu_updates": r"D:\paper\outputs\verification\g09_remote_migration_20260930\production_gpu_updates.png",
    "w07_jan23_paired": r"D:\paper\outputs\experiments\europe_joint_action_20261001\analysis\attempt_v2\figures\jan23_paired_scenes.png",
    "w07_G10_reward": r"D:\paper\outputs\experiments\europe_joint_action_20261001\analysis\root_diagnostics\G10_奖励训练与固定验证分面.png",
    "w07_G10_learning": r"D:\paper\outputs\experiments\europe_joint_action_20261001\analysis\root_diagnostics\G10_学习诊断对数尺度.png",
}
for g in "TMG":
    for k, n in (("trainval", "训练与固定验证"), ("learn", "学习量原始采样")):
        src[f"w08_{g}_{k}"] = rf"D:\paper\outputs\experiments\europe_tmg_20261001\verified_report_20261002\figures\{g}_{n}.png"
for k, p in src.items():
    im = Image.open(p)
    print(k, im.format, im.size)
    im.convert("RGB").save(base + rf"\orig\{k}.png", format="PNG")
