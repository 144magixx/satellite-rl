"""W15 数据提取：从归档 zip 里流式读取十二条覆盖路线的独立静态考核结果、两条学习路线的训练日志，
写成小 CSV 放到 figure_scripts/data/W15/，画图脚本只读这些 CSV。

数据来源（只读）：E:\\paper_archives\\outputs\\experiments\\first_stage_s1_v10_20261009\\job_archives
用法：python W15_extract.py
"""
import csv
import glob
import json
import os
import zipfile

ARCH = r"E:\paper_archives\outputs\experiments\first_stage_s1_v10_20261009\job_archives"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "W15")
os.makedirs(OUT, exist_ok=True)


def iter_members(job_dir, suffix, exclude=("recoveries",)):
    """遍历一个作业目录下所有分卷 zip，找到文件名以 suffix 结尾的成员。"""
    for z in sorted(glob.glob(os.path.join(job_dir, "*", "*.zip"))):
        zf = zipfile.ZipFile(z)
        for info in zf.infolist():
            if info.filename.endswith(suffix) and not any(e in info.filename for e in exclude):
                yield zf, info


def job_name(job_dir):
    for zf, info in iter_members(job_dir, "job/config.json"):
        return json.loads(zf.read(info))["name"]
    return None


def iter_jsonl(zf, info):
    with zf.open(info) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def band_share(demands):
    pos = [d for d in demands if d > 0]
    if not pos:
        return float("nan")
    m = sum(pos) / len(pos)
    return sum(1 for d in pos if 0.8 * m <= d <= 1.2 * m) / len(pos)


def extract_static():
    rows = []
    for jd in sorted(glob.glob(os.path.join(ARCH, "*", "evaluate_ppo_*"))):
        name = job_name(jd)
        if not name or " test_R" not in name:
            continue
        route = name.split("test_")[1]
        n = 0
        for zf, info in iter_members(jd, "job/results.partial.jsonl"):
            for r in iter_jsonl(zf, info):
                m = r["metrics"]
                d = r["downstream"]
                rows.append({
                    "route": route, "scene_id": r["scene_id"], "mult": r["demand_multiplier"],
                    "N": m["beam_count"], "load_cv": m["load_cv"], "peak_ratio": m["peak_load_ratio"],
                    "coverage_100": int(bool(m["coverage_100"])),
                    "band_share": band_share(m["beam_demand_bps"]),
                    "carried_aircraft": d["aircraft_service"]["demand_weighted_carried_fraction"],
                    "carried_beam": d["metrics"]["demand_weighted_carried_fraction"],
                    "si": d["metrics"]["beam_mean_si"], "sgm": d["metrics"]["beam_mean_sgm"],
                    "power_w": d["metrics"]["total_power_w"],
                    "init_s": r["initialization_seconds"], "algo_s": r["algorithm_seconds"],
                    "final_s": r["final_check_seconds"], "total_s": r["total_seconds"],
                    "down_s": d["timing"]["total_seconds"],
                })
                n += 1
        print("static", route, n, flush=True)
    with open(os.path.join(OUT, "static_per_scene.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def extract_sequence():
    rows = []
    for jd in sorted(glob.glob(os.path.join(ARCH, "*", "evaluate_ppo_*"))):
        name = job_name(jd)
        if not name or " sequence_R" not in name:
            continue
        route = name.split("sequence_")[1]
        for zf, info in iter_members(jd, "job/results.partial.jsonl"):
            for k, r in enumerate(iter_jsonl(zf, info)):
                if k == 0:
                    json.dump({kk: str(vv)[:80] for kk, vv in r.items()},
                              open(os.path.join(OUT, f"seq_{route}_first_record_keys.json"), "w", encoding="utf-8"),
                              ensure_ascii=False, indent=1)
                m = r["metrics"]
                d = r["downstream"]
                ii = r.get("initialization_info") or {}
                rows.append({
                    "route": route, "scene_id": r.get("scene_id"), "sequence_id": r.get("sequence_id"),
                    "time_utc": r.get("selected_time_utc"), "mult": r.get("demand_multiplier"),
                    "temporal_mode": r.get("temporal_mode"),
                    "warm_start_fallback": ii.get("warm_start_fallback"),
                    "created_beams": len(ii.get("created_beam_ids") or []),
                    "retired_beams": len(ii.get("retired_beam_ids") or []),
                    "N": m["beam_count"], "load_cv": m["load_cv"],
                    "carried_aircraft": d["aircraft_service"]["demand_weighted_carried_fraction"],
                    "total_s": r.get("total_seconds"), "down_s": d["timing"]["total_seconds"],
                })
        print("sequence", route, len(rows), flush=True)
    with open(os.path.join(OUT, "sequence_per_frame.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def extract_training():
    jobs = {"R08": os.path.join(ARCH, "20418", "train_ppo_20261009_095314_19a69288"),
            "R09": os.path.join(ARCH, "20424", "train_ppo_20261009_095159_714e5c20")}
    for route, jd in jobs.items():
        # 学习指标
        for zf, info in iter_members(jd, "job/updates.jsonl"):
            rows = list(iter_jsonl(zf, info))
            with open(os.path.join(OUT, f"train_{route}_updates.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["env_step", "update_step", "episode", "policy_loss", "value_loss", "entropy"])
                for r in rows:
                    w.writerow([r["env_step"], r["update_step"], r["episode"], r["policy_loss"],
                                r["value_loss"], r["entropy"]])
        # 固定验证节点（每节点 60 条）
        for zf, info in iter_members(jd, "job/evaluations.jsonl"):
            rows = list(iter_jsonl(zf, info))
            with open(os.path.join(OUT, f"train_{route}_validation.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["env_step", "scene_count", "coverage_100_count", "mean_load_cv", "mean_beam_count"])
                for r in rows:
                    s = r["summary"]
                    w.writerow([r["env_step"], s["scene_count"], s["coverage_100_count"],
                                s["mean_load_cv"], s["mean_beam_count"]])
        # 逐步动作：按 16384 步一块统计停止比例、接受比例、平均单步回报
        blocks = {}
        for zf, info in iter_members(jd, "job/actions.jsonl"):
            for a in iter_jsonl(zf, info):
                b = (int(a["env_step"]) - 1) // 16384
                s = blocks.setdefault(b, {"n": 0, "stop": 0, "accepted": 0, "reward": 0.0})
                s["n"] += 1
                op = (a.get("requested_action") or a.get("action") or {}).get("operation")
                s["stop"] += int(op == "stop")
                s["accepted"] += int(bool(a.get("accepted")))
                s["reward"] += float(a["reward"])
        with open(os.path.join(OUT, f"train_{route}_action_blocks.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["block", "env_step_end", "n", "stop", "accepted", "reward_sum"])
            for b in sorted(blocks):
                s = blocks[b]
                w.writerow([b, (b + 1) * 16384, s["n"], s["stop"], s["accepted"], s["reward"]])
        print("training", route, "done", flush=True)
    # R09 完整回合回报
    jd = jobs["R09"]
    for zf, info in iter_members(jd, "job/episodes.jsonl"):
        with open(os.path.join(OUT, "train_R09_episodes.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["episode", "terminal_env_step", "episode_return", "N", "load_cv", "stop_reason", "completed"])
            for r in iter_jsonl(zf, info):
                w.writerow([r["episode"], r["terminal_env_step"], r["episode_return"], r["metrics"]["beam_count"],
                            r["metrics"]["load_cv"], r["stop_reason"], int(bool(r["completed_episode"]))])
    print("R09 episodes done", flush=True)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["static", "sequence", "training"]
    if "static" in which:
        extract_static()
    if "sequence" in which:
        extract_sequence()
    if "training" in which:
        extract_training()
