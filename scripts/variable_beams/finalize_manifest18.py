"""Update the run manifest to the final factual state after all three arms finished."""
import json
import sys

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

PATH = ops18.LOCAL / "run_manifest.json"
data = json.loads(PATH.read_text(encoding="utf-8-sig"))

ARMS = {
    "V2": {"run_id": "train_sac_20261006_084454_365eff71", "port": 20418,
           "assessment": "evaluate_sac_20261006_181949_137a0534",
           "sca": "refine_sac_20261006_224531_316265cd"},
    "V3": {"run_id": "train_sac_20261006_084726_c004cd09", "port": 20424,
           "assessment": "evaluate_sac_20261006_162655_a661dd3d",
           "sca": "refine_sac_20261006_173515_4b10636d"},
    "G0": {"run_id": "train_sac_20261006_092142_142c6953", "port": 20418,
           "assessment": "evaluate_sac_20261007_021119_846087b2",
           "sca": "refine_sac_20261007_024934_39422e19"},
}
data["status"] = "training_and_assessment_complete_sca_finishing"
data["final_state"] = {
    "checked_utc": ops18.utc_now(),
    "arms": {arm: {"train_run": entry["run_id"], "port": entry["port"],
                   "train_stage": "completed", "train_env_step": 400000,
                   "assessment_run": entry["assessment"],
                   "sca_run": entry["sca"],
                   "note": "训练与440景共同考核均完成；SCA 三路线按各臂实际进度记录"}
             for arm, entry in ARMS.items()},
    "budget": {"pilot": 300000, "extension": 900000, "approved_cap": 1200000,
               "note": "试点三臂各10万＋补足三臂各增30万；故障中断尝试的消耗另计于审计轨"},
    "verdicts": {"G1": "未达标", "G2": "未达标",
                 "G3": "部分达标（G0 槽数 +31% 达标；尾部与绝对水平未改善）",
                 "G4": "按方案转下一手段，不追加同手段预算"},
    "headline": {"direct": {"V2": 0.8795, "V3": 0.8660, "G0": 0.8152,
                            "greedy": 0.8933, "fixed": 0.8368},
                 "after_sca": {"V2": 0.9236, "V3": 0.9090, "G0": "见 g0_sca_per_method.json"}},
}
# 补足阶段记录更新为已完成
stage2 = data.setdefault("stage2", {})
for arm in ARMS:
    entry = stage2.get(arm)
    if isinstance(entry, dict):
        entry["state"] = "completed_400000"
stage2["note"] = ("三臂均在同一任务 ID 上从 10 万步断点续跑到 40 万步；"
                  "重定标只改总步数预算，原断点与试点终点快照改名保留并记录 SHA")
data["updated_utc"] = ops18.utc_now()
ops18.write_json(PATH, data)
print("运行清单已更新为最终事实状态")
print(json.dumps(data["final_state"]["verdicts"], ensure_ascii=False))
