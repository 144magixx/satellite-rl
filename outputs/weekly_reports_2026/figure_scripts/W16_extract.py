"""W16 数据提取：从归档 zip 读取六种做法在 100 个母场景（每个三档需求）上的独立考核结果，
以及两个 PPO 的训练日志、行为克隆的拟合曲线，写成小 CSV 放到 figure_scripts/data/W16/。

数据来源（只读）：E:\\paper_archives\\outputs\\experiments\\first_stage_s1_d12a_20261010\\job_archives
用法：python W16_extract.py
"""
import csv
import glob
import json
import os
import zipfile

import numpy as np

ARCH = r"E:\paper_archives\outputs\experiments\first_stage_s1_d12a_20261010\job_archives"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "W16")
os.makedirs(OUT, exist_ok=True)


def members(job_dir):
    for z in sorted(glob.glob(os.path.join(job_dir, "*.zip"))):
        zf = zipfile.ZipFile(z)
        for info in zf.infolist():
            yield zf, info


def job_name(job_dir):
    for zf, info in members(job_dir):
        if info.filename.endswith("/config.json") and info.filename.count("/") == 2:
            return json.loads(zf.read(info)).get("name")


def iter_jsonl(zf, info):
    with zf.open(info) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def band(vals, lo=0.8, hi=1.2):
    v = np.asarray([x for x in vals if x > 0], float)
    if v.size == 0:
        return float("nan")
    m = v.mean()
    return float(np.mean((v >= lo * m) & (v <= hi * m)))


def extract_static():
    rows = []
    for jd in sorted(glob.glob(os.path.join(ARCH, "*", "evaluate_ppo_*"))):
        name = job_name(jd) or ""
        if " test_" not in name or "preparation" in name:
            continue
        method = name.split("test_")[1].split("_")[0]
        for zf, info in members(jd):
            if not info.filename.endswith("results.jsonl"):
                continue
            for r in iter_jsonl(zf, info):
                tp = r["terminal_physics"]
                best = tp["layouts_by_sha256"][tp["layout_aliases"]["best"]]
                beams = best["beams"]
                D = np.array([b["demand_bps"] for b in beams], float)
                C = np.array([b["capacity_bps"] for b in beams], float)
                pos = D > 0
                cv = float(D.std() / D.mean())
                rows.append({
                    "method": method, "mother": r["scene_source_sha256"], "mult": r["demand_multiplier"],
                    "N": len(beams), "cv": cv,
                    "band_demand": band(D), "band_capacity": band(C[pos] / D[pos]) if pos.any() else float("nan"),
                    "carried_center": float(np.minimum(C, D).sum() / D.sum()),
                    "carried_aircraft": best["aircraft_service"]["demand_weighted_carried_fraction"],
                    "si": best["metrics"]["beam_mean_si"], "sgm": best["metrics"]["beam_mean_sgm"],
                    "power_w": best["metrics"]["total_power_w"],
                    "slots": int(sum(b["slots"] for b in beams)),
                    "initial_j": r["initial_j"], "best_j": r["best_j"], "commits": r["commits"],
                    "stop_reason": r["stop_reason"],
                    "algo_s": r["algorithm_seconds"], "total_s": r["total_seconds"],
                    "published_phys_s": best["timing"]["total_seconds"],
                })
        print("static", method, len(rows), flush=True)
    with open(os.path.join(OUT, "static_per_scene.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def extract_training():
    jobs = {"P0": os.path.join(ARCH, "20418", "train_ppo_20261010_034306_03e7f00f"),
            "P1": os.path.join(ARCH, "20424", "train_ppo_20261010_034937_d980cee4")}
    for tag, jd in jobs.items():
        for zf, info in members(jd):
            fn = info.filename
            if fn.endswith("/updates.jsonl"):
                rows = list(iter_jsonl(zf, info))
                with open(os.path.join(OUT, f"train_{tag}_updates.csv"), "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["env_step", "episode", "policy_loss", "value_loss", "entropy"])
                    for r in rows:
                        w.writerow([r["env_step"], r["episode"], r["policy_loss"], r["value_loss"], r["entropy"]])
            elif fn.endswith("/evaluations.jsonl"):
                rows = list(iter_jsonl(zf, info))
                with open(os.path.join(OUT, f"train_{tag}_validation.csv"), "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["env_step", "mean_best_j", "first_step_stop_count", "best_j_improved_count",
                                "three_level_improved_mother_count"])
                    for r in rows:
                        w.writerow([r["env_step"], r["mean_best_j"], r["first_step_stop_count"],
                                    r["best_j_improved_count"], r["three_level_improved_mother_count"]])
            elif fn.endswith("/episodes.jsonl"):
                with open(os.path.join(OUT, f"train_{tag}_episodes.csv"), "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["episode", "terminal_env_step", "episode_return", "initial_j", "best_j",
                                "commits", "stop_reason", "completed"])
                    for r in iter_jsonl(zf, info):
                        w.writerow([r["episode"], r["terminal_env_step"], r["episode_return"], r["initial_j"],
                                    r["best_j"], r["commits"], r["stop_reason"], int(bool(r["completed_episode"]))])
        print("training", tag, flush=True)
    bc = os.path.join(ARCH, "20424", "train_ppo_20261010_034521_2e4a6f55")
    for zf, info in members(bc):
        if info.filename.endswith("bc_updates.jsonl"):
            rows = list(iter_jsonl(zf, info))
            with open(os.path.join(OUT, "bc_updates.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "update_step", "labels", "mean_cross_entropy"])
                for r in rows:
                    w.writerow([r["epoch"], r["update_step"], r["labels"], r["mean_cross_entropy"]])
    print("bc done", flush=True)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["static", "training"]
    if "static" in which:
        extract_static()
    if "training" in which:
        extract_training()
