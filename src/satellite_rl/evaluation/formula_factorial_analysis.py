"""Read-only, single-seed 2×2 SAC formula comparison from completed run artifacts.

The original six runs and the six additional runs remain in separate experiment
directories. This module never starts training or substitutes missing results.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from satellite_rl.evaluation.experiment_report import MAIN_FIELDS, MODEL_NAMES
from satellite_rl.evaluation.single_seed_scope import require_completed as require_original_six
from satellite_rl.training.experiment_runner import ACTION_SEEDS, MODELS, sha256_file
from satellite_rl.training.run_monitor import atomic_json


SEED = 51001
STEPS = 400000
UPDATES = 398000
OLD_EXPERIMENT = "sac_comparison_20260924_r2"
NEW_EXPERIMENT = "sac_formula_factorial_20260925"
NEW_TRAIN_COMMIT = "09505290aae1a754a0e4b9d4b9a9ae3c39bf1425"
FORMULAS = ("legacy", "negative_joint", "positive_per_head", "corrected")
NEW_FORMULAS = ("negative_joint", "positive_per_head")
FORMULA_LABELS = {
    "legacy": "−6.9，逐头双Q（A）",
    "negative_joint": "−6.9，联合双Q（B）",
    "positive_per_head": "+6.9，逐头双Q（C）",
    "corrected": "+6.9，联合双Q（D）",
}
METRICS = ("si_mean", "sgm_mean", "throughput_bps", "total_power_w",
           "power_exceeded", "completed", "served_beams", "allocated_slots",
           "zero_slot_rate", "occupancy_fraction")
DIAGNOSTICS = ("alpha", "entropy", "actor_loss", "critic_1_loss",
               "critic_2_loss", "q_mean", "target_mean")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _git_head_clean(root, expected_commit):
    root = Path(root).resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                           capture_output=True, text=True).stdout.strip()
    if head != expected_commit or dirty:
        raise RuntimeError(f"冻结训练源码提交不符或工作树已改动：{root}")
    return root


def _scene_hashes(plan, group="test_scenes"):
    refs = plan.get(group, {})
    expected = ({str(i) for i in range(81000, 81100)} if group == "test_scenes"
                else {str(i) for i in range(71000, 71020)})
    if set(refs) != expected:
        raise RuntimeError(f"{group} 必须是预定的固定场景")
    result = {}
    for scene, reference in refs.items():
        path = Path(reference["path"])
        digest = reference["sha256"]
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"{group} 场景缺失或SHA变化：{path}")
        result[scene] = digest
    return result


def _check_plan(plan, experiment_id, formulas, frozen_root):
    expected = {f"{model}_{formula}_s{SEED}"
                for model in MODELS for formula in formulas}
    jobs = plan.get("jobs", [])
    if (plan.get("experiment_id") != experiment_id
            or plan.get("steps") != STEPS
            or plan.get("action_seeds") != list(ACTION_SEEDS)
            or len(jobs) != len(expected)
            or {job.get("run_id") for job in jobs} != expected):
        raise RuntimeError("新实验计划必须精确包含三网络×两公式、同预算同动作种子")
    for job in jobs:
        if job.get("run_id") != f"{job.get('model')}_{job.get('formula')}_s{SEED}":
            raise RuntimeError(f"新实验运行身份不一致：{job}")
    source = plan["source"]
    if source.get("git_commit") != NEW_TRAIN_COMMIT:
        raise RuntimeError("补充训练源码不是已冻结的实验提交")
    _git_head_clean(frozen_root, source["git_commit"])
    if (Path(source["source_root"]).resolve() != Path(frozen_root).resolve()
            or not source.get("torch_cuda")
            or "dqn" not in (part.lower() for part in Path(source.get("python", "")).parts)):
        raise RuntimeError("新实验计划的冻结源码或DQN/CUDA身份不符")
    _scene_hashes(plan)
    _scene_hashes(plan, "validation_scenes")
    return jobs


def _check_config_matrix(runs, old_plan, new_plan):
    """Keep formula and its entropy target as the only within-model choices."""
    old_data = old_plan["source"].get("cities_data", {})
    new_data = new_plan["source"].get("cities_data", {})
    if not old_data.get("sha256") or old_data["sha256"] != new_data.get("sha256"):
        raise RuntimeError("训练城市输入数据SHA不一致")
    ignored = {"experiment_id", "run_id", "formula", "target_entropy",
               "q_min_variant", "runtime"}
    for model in MODELS:
        reference = runs[model, "legacy"]["manifest"]["config"]
        baseline = {key: value for key, value in reference.items() if key not in ignored}
        for formula in FORMULAS:
            config = runs[model, formula]["manifest"]["config"]
            stripped = {key: value for key, value in config.items() if key not in ignored}
            expected_target = -6.9 if formula in ("legacy", "negative_joint") else 6.9
            expected_q = ("per_head" if formula in ("legacy", "positive_per_head")
                          else "joint")
            if (stripped != baseline or config.get("formula") != formula
                    or config.get("target_entropy") != expected_target
                    or config.get("q_min_variant", expected_q) != expected_q
                    or config.get("train_seed") != SEED):
                raise RuntimeError(f"{model}/{formula} 除公式外训练配置与原方案不一致")


def _check_result(result, plan, run_id):
    keys = {(str(scene), "sample", str(seed))
            for scene in plan["test_scenes"] for seed in plan["action_seeds"]}
    keys.update((str(scene), "argmax", "argmax") for scene in plan["test_scenes"])
    rows = result.get("rows", [])
    observed = [(str(row["scene_id"]), row["action_mode"],
                 str(row["action_seed"])) for row in rows]
    if (result.get("checkpoint") != "final_test" or result.get("env_step") != STEPS
            or result.get("include_argmax") is not True
            or result.get("scene_references") != plan["test_scenes"]
            or result.get("action_seeds") != plan["action_seeds"]
            or len(observed) != 400 or len(set(observed)) != 400
            or set(observed) != keys):
        raise RuntimeError(f"{run_id} 缺少共同的100场景×3动作考核或最大概率补充考核")
    for path_key, sha_key in (("arrays_path", "arrays_sha256"),
                              ("trace_path", "trace_sha256")):
        path = Path(result[path_key])
        if not path.is_file() or sha256_file(path) != result[sha_key]:
            raise RuntimeError(f"{run_id} 考核原始资产校验失败：{path}")
    sampled = {}
    for row in rows:
        if set(MAIN_FIELDS) - set(row):
            raise RuntimeError(f"{run_id} 考核指标缺失")
        if row["action_mode"] != "sample":
            continue
        key = (str(row["scene_id"]), str(row["action_seed"]))
        if int(row["total_beams"]) != 100:
            raise RuntimeError(f"{run_id} 终态需求满足率分母不是全部100束")
        if not math.isclose(float(row["si_sum"]) / 100, float(row["si_mean"]),
                            rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(f"{run_id} SI总量与全束均值不一致")
        if not all(math.isfinite(float(row[field])) for field in METRICS):
            raise RuntimeError(f"{run_id} 终态指标含非有限值")
        sampled[key] = row
    with np.load(result["arrays_path"]) as arrays:
        for (scene, seed), row in sampled.items():
            prefix = f"scene_{scene}_action_{seed}_"
            for name, field in (("satisfaction", "si_mean"), ("sgm", "sgm_mean")):
                values = arrays[prefix + name]
                if (values.shape != (100,) or not np.isfinite(values).all()
                        or not math.isclose(float(values.mean()), float(row[field]),
                                            rel_tol=0, abs_tol=1e-8)):
                    raise RuntimeError(f"{run_id} 终态全束数组与{field}不一致")
    if len(sampled) != 300:
        raise RuntimeError(f"{run_id} 采样考核数不是300")
    return sampled


def _check_checkpoint(model_dir, config_hash, run_id):
    import torch
    paths = {name: model_dir / name for name in
             ("initial.pt", "latest.pt", "final.pt", "final_actor.pt")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"{run_id} 缺少权重：{path}")
    checks = {}
    for name, expected in (("initial.pt", 0), ("latest.pt", STEPS),
                           ("final.pt", STEPS)):
        data = torch.load(paths[name], map_location="cpu", weights_only=False)
        if (data.get("config_hash") != config_hash
                or data.get("progress", {}).get("env_step") != expected
                or not isinstance(data.get("agent", {}).get("networks", {}).get("actor"), dict)):
            raise RuntimeError(f"{run_id} 权重配置、结构或预算不符：{name}")
        checks[name] = sha256_file(paths[name])
        if name == "final.pt":
            saved_actor = data["agent"]["networks"]["actor"]
    actor = torch.load(paths["final_actor.pt"], map_location="cpu", weights_only=True)
    if set(actor) != set(saved_actor) or any(
        not torch.equal(actor[key], saved_actor[key]) for key in actor
    ):
        raise RuntimeError(f"{run_id} 独立Actor与完整断点不一致")
    checks["final_actor.pt"] = sha256_file(paths["final_actor.pt"])
    return checks


def _load_new_run(exp_dir, plan, job, archive_root):
    run_id = job["run_id"]
    run_dir = exp_dir / run_id
    status = _read_json(run_dir / "status.json")
    manifest = _read_json(run_dir / "manifest.json")
    config = manifest["config"]
    source = plan["source"]
    runtime = config["runtime"]
    if (status.get("stage") != "completed" or status.get("env_step") != STEPS
            or status.get("update_step") != UPDATES
            or config.get("experiment_id") != NEW_EXPERIMENT
            or config.get("run_id") != run_id
            or config.get("model") != job["model"]
            or config.get("formula") != job["formula"]
            or config.get("train_seed") != SEED
            or config.get("budget_env_steps") != STEPS
            or config.get("target_entropy") != (-6.9 if job["formula"] == "negative_joint" else 6.9)
            or config.get("q_min_variant") != ("joint" if job["formula"] == "negative_joint"
                                                  else "per_head")
            or config.get("test_scene_seeds") != [int(x) for x in plan["test_scenes"]]
            or config.get("action_seeds") != plan["action_seeds"]
            or runtime.get("git_commit") != source["git_commit"]
            or Path(runtime.get("source_root", "")).resolve() != Path(source["source_root"]).resolve()
            or not Path(runtime.get("package_file", "")).resolve().is_relative_to(
                Path(source["source_root"]) / "src")
            or not runtime.get("torch_cuda")
            or "dqn" not in (part.lower() for part in Path(runtime.get("python", "")).parts)):
        raise RuntimeError(f"{run_id} 未以预定公式、冻结DQN/CUDA源码完成同预算训练")
    model_dir = archive_root / "models" / "runs" / NEW_EXPERIMENT / run_id
    weights = _check_checkpoint(model_dir, manifest["config_hash"], run_id)
    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        episodes = list(csv.DictReader(stream))
    if (not episodes or int(episodes[-1]["env_step"]) > STEPS
            or any(int(row["env_step"]) > STEPS for row in episodes)):
        raise RuntimeError(f"{run_id} 逐回合训练记录缺失或超预算")
    result_path = run_dir / "evaluation" / f"final_test_{STEPS}.json"
    result = _read_json(result_path)
    rows = _check_result(result, plan, run_id)
    return {"job": job, "run_dir": run_dir, "status": status, "manifest": manifest,
            "result": result, "by_key": rows, "weights": weights,
            "result_sha256": sha256_file(result_path)}


def paired_factorial(by_formula, field):
    """Paired scene/action differences for all four cells, including interaction."""
    if set(by_formula) != set(FORMULAS):
        raise RuntimeError("四组合不完整")
    keys = set(by_formula["legacy"])
    if not keys or any(set(rows) != keys for rows in by_formula.values()):
        raise RuntimeError("四组合的场景和动作种子键不一致")
    scene_ids = {key[0] for key in keys}
    if any(len([key for key in keys if key[0] == scene]) != len(ACTION_SEEDS)
           for scene in scene_ids):
        raise RuntimeError("每场景动作种子不完整")
    coefficients = {
        "entropy_at_per_head": {"positive_per_head": 1, "legacy": -1},
        "entropy_at_joint": {"corrected": 1, "negative_joint": -1},
        "joint_q_at_negative": {"negative_joint": 1, "legacy": -1},
        "joint_q_at_positive": {"corrected": 1, "positive_per_head": -1},
        "interaction": {"corrected": 1, "positive_per_head": -1,
                        "negative_joint": -1, "legacy": 1},
    }
    output = {}
    for label, weights in coefficients.items():
        action_delta = {key: sum(weight * float(by_formula[formula][key][field])
                                 for formula, weight in weights.items())
                        for key in keys}
        scene_delta = [statistics.mean(action_delta[key] for key in keys if key[0] == scene)
                       for scene in sorted(scene_ids)]
        output[label] = {
            "mean": statistics.mean(scene_delta),
            "scene_sample_std": statistics.stdev(scene_delta) if len(scene_delta) > 1 else None,
            "higher_scenes": sum(delta > 0 for delta in scene_delta),
            "lower_scenes": sum(delta < 0 for delta in scene_delta),
            "independent_scenes": len(scene_delta),
            "paired_action_trials": len(keys),
            "scene_differences": dict(zip(sorted(scene_ids), scene_delta)),
        }
    return output


def _read_updates(run_dir, run_id):
    with (run_dir / "updates.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = set(range(100, UPDATES + 1, 100))
    actual = [int(row["update_step"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError(f"{run_id} 缺少3980个更新诊断点")
    points = []
    for row in rows:
        point = {"run_id": run_id, "env_step": int(row["env_step"]),
                 "update_step": int(row["update_step"]),
                 **{field: float(row[field]) for field in DIAGNOSTICS}}
        if (not 0 <= point["env_step"] <= STEPS
                or any(not math.isfinite(point[field]) for field in DIAGNOSTICS)):
            raise RuntimeError(f"{run_id} 更新诊断数值无效")
        points.append(point)
    return points


def _validation_curve(run_dir, run_id, plan):
    with (run_dir / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_keys = {(str(scene), str(seed)) for scene in plan["validation_scenes"]
                     for seed in plan["action_seeds"]}
    if len(expected_keys) != 60:
        raise RuntimeError("验证场景与动作种子不是20×3")
    grouped = defaultdict(list)
    for row in rows:
        if row["checkpoint"] in ("validation", "final_validation") and row["action_mode"] == "sample":
            grouped[(int(row["env_step"]), row["checkpoint"])].append(row)
    points = {}
    for (step, phase), group in grouped.items():
        keys = [(str(row["scene_id"]), str(row["action_seed"])) for row in group]
        if len(keys) != 60 or set(keys) != expected_keys:
            raise RuntimeError(f"{run_id} {step}步验证缺失或重复场景")
        bucket = step // int(plan["validation_interval"])
        if bucket not in range(1, 41):
            raise RuntimeError(f"{run_id} 验证超出40万步")
        current = {"run_id": run_id, "round": bucket, "env_step": step,
                   "si_mean": statistics.mean(float(row["si_mean"]) for row in group),
                   "sgm_mean": statistics.mean(float(row["sgm_mean"]) for row in group)}
        if bucket in points:
            previous = points[bucket]
            if (bucket != 40 or step != STEPS or previous["env_step"] != STEPS
                    or any(not math.isclose(current[field], previous[field], rel_tol=1e-8)
                           for field in ("si_mean", "sgm_mean"))):
                raise RuntimeError(f"{run_id} 验证轮次重复且数值不同")
        points[bucket] = current
    if set(points) != set(range(1, 41)):
        raise RuntimeError(f"{run_id} 缺少完整40轮验证曲线")
    return [points[i] for i in range(1, 41)]


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _figures(target, models, curves, updates):
    colors = {"legacy": "#4c78a8", "negative_joint": "#f58518",
              "positive_per_head": "#54a24b", "corrected": "#e45756"}
    for model in models:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for formula in FORMULAS:
            run_id = f"{model}_{formula}_s{SEED}"
            color = colors[formula]
            label = FORMULA_LABELS[formula]
            axes[0, 0].plot([p["env_step"] for p in curves[run_id]],
                            [p["si_mean"] * 100 for p in curves[run_id]],
                            label=label, color=color)
            axes[0, 1].plot([p["env_step"] for p in curves[run_id]],
                            [p["sgm_mean"] * 100 for p in curves[run_id]],
                            label=label, color=color)
            for axis, field in ((axes[1, 0], "alpha"), (axes[1, 1], "critic_1_loss")):
                axis.plot([p["env_step"] for p in updates[run_id]],
                          [p[field] for p in updates[run_id]], label=label,
                          color=color, linewidth=1)
        for axis, title in zip(axes.flat, ("验证需求满足率 / %", "验证供需匹配程度 / %",
                                           "温度系数 α", "Critic 1 损失")):
            axis.set_title(title)
            axis.set_xlabel("环境交互步数")
            axis.grid(alpha=0.2)
        for axis in axes[1]:
            axis.set_yscale("symlog", linthresh=1e-4)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.savefig(target / f"{model}_四组合学习曲线.png", dpi=160)
        plt.close(fig)


def _report(summary, effects, old_commit, new_commit):
    indexed = {(r["model"], r["formula"]): r for r in summary}
    lines = ["# SAC 学习公式四组合对照（单一训练种子 51001）", "",
             f"旧训练提交：`{old_commit}`；补充训练提交：`{new_commit}`。",
             "三种网络各比较目标熵 −6.9 / +6.9 和逐头 / 联合双Q。"
             "每组40万次环境交互、39.8万次参数更新；终点模型在同100个固定场景上，"
             "各使用3个动作采样种子。下表为全部100束的终态需求满足率，"
             "包含后续干扰和未服务波束；SGM 单列。", "",
             "| 网络 | 组合 | 需求满足率 / % | SGM / % | 功率 / W | 超功率 / % | 完成100束 / % | 零槽 / % |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in MODELS:
        for formula in FORMULAS:
            m = indexed[model, formula]["metrics"]
            lines.append(f"| {MODEL_NAMES[model]} | {FORMULA_LABELS[formula]} | "
                         f"{m['si_mean']*100:.2f} | {m['sgm_mean']*100:.2f} | "
                         f"{m['total_power_w']:.1f} | {m['power_exceeded']*100:.1f} | "
                         f"{m['completed']*100:.1f} | {m['zero_slot_rate']*100:.1f} |")
    lines += ["", "## 同场景配对差值", "",
              "差值以终点需求满足率的百分点表示。正值表示左侧比较组合较高；"
              "仍须结合上表功率与完整分配情况判断。", "",
              "| 网络 | 改目标熵（逐头，C−A） | 改目标熵（联合，D−B） | 改双Q（−6.9，B−A） | 改双Q（+6.9，D−C） | 两项交互（D−C−B+A） |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    order = ("entropy_at_per_head", "entropy_at_joint", "joint_q_at_negative",
             "joint_q_at_positive", "interaction")
    for model in MODELS:
        e = effects[model]["si_mean"]
        lines.append(f"| {MODEL_NAMES[model]} | " + " | ".join(
            f"{e[key]['mean']*100:+.2f}" for key in order) + " |")
    lines += ["", "`paired_effects.json` 保存每100场景的配对差值、改善和下降场景数，"
              "并分别记录 SGM、功率、超功率和完成率等指标。交互项表示两项改动共同实施时"
              "与单独实施之和有多大偏离，不能仅凭一个种子把它解释为普遍规律。", "",
              "## 收敛与数值诊断", "",
              "`validation_curves.csv` 为40轮固定验证原始均值；`learning_diagnostics.csv`"
              "保留每100次更新记录的温度系数、动作随机程度、损失与价值。"
              "图直接从这两份原始数据绘制，损失与温度图用对称对数刻度，未改写原始值。", ""]
    for model in MODELS:
        lines += [f"![{MODEL_NAMES[model]}四组合学习曲线]({model}_四组合学习曲线.png)", ""]
    lines += ["## 结论边界", "",
              "这里只使用一个训练种子。100个场景的配对差值反映本次模型在固定考核场景"
              "上的差异，不能当作训练重复的稳定性证据。目标熵符号与代码熵定义是否一致"
              "属于计算检查；同场景表现及约束是否改善属于实验效果，两者分别判断。"
              "四组合以两次冻结提交训练，已核对共同场景内容SHA、预算、评价动作种子和"
              "运行配置；源码差异及剩余限制应结合 `analysis_manifest.json` 检查。", ""]
    return "\n".join(lines)


def generate(archive_root="D:/paper",
             old_frozen_root="D:/paper_worktrees/frozen-sac-20260924-r2",
             new_frozen_root="D:/paper_worktrees/frozen-sac-factorial-20260925"):
    archive_root = Path(archive_root).resolve()
    new_exp = archive_root / "outputs" / "experiments" / NEW_EXPERIMENT
    new_plan = _read_json(new_exp / "suite_plan.json")
    new_jobs = _check_plan(new_plan, NEW_EXPERIMENT, NEW_FORMULAS, new_frozen_root)
    for job in new_jobs:
        status_path = new_exp / job["run_id"] / "status.json"
        if not status_path.is_file():
            raise RuntimeError(f"补充训练尚未完成：{job['run_id']} 无状态文件")
        status = _read_json(status_path)
        if (status.get("stage") != "completed" or status.get("env_step") != STEPS
                or status.get("update_step") != UPDATES):
            raise RuntimeError(f"补充训练尚未完成：{job['run_id']}")
    old_exp, old_plan, old_final = require_original_six(archive_root, old_frozen_root)
    if (new_plan.get("reference_plan_sha256") != sha256_file(old_exp / "suite_plan.json")
            or Path(new_plan.get("reference_suite", "")).resolve() != old_exp.resolve()):
        raise RuntimeError("新实验的旧计划引用或SHA不一致")
    if _scene_hashes(old_plan) != _scene_hashes(new_plan):
        raise RuntimeError("新旧考核场景内容SHA不一致")
    if _scene_hashes(old_plan, "validation_scenes") != _scene_hashes(
            new_plan, "validation_scenes"):
        raise RuntimeError("新旧验证场景内容SHA不一致")
    if old_plan["action_seeds"] != new_plan["action_seeds"]:
        raise RuntimeError("新旧评价动作种子不一致")
    runs = {}
    input_files = [old_exp / "suite_plan.json", new_exp / "suite_plan.json"]
    for job in old_plan["jobs"]:
        run_id = job["run_id"]
        run_dir = old_exp / run_id
        status = _read_json(run_dir / "status.json")
        if status.get("update_step") != UPDATES:
            raise RuntimeError(f"{run_id} 更新次数不足")
        result_path = run_dir / "evaluation" / f"final_test_{STEPS}.json"
        result = _read_json(result_path)
        runs[(job["model"], job["formula"])] = {
            "job": job, "run_dir": run_dir,
            "by_key": _check_result(result, old_plan, run_id),
            "manifest": _read_json(run_dir / "manifest.json"),
            "result": result,
        }
        input_files += [run_dir / "status.json", run_dir / "manifest.json", result_path,
                        Path(result["arrays_path"]), Path(result["trace_path"]),
                        run_dir / "updates.csv", run_dir / "evaluations.csv",
                        run_dir / "episodes.csv",
                        Path(old_final[run_id]["checkpoint"]), Path(old_final[run_id]["actor"])]
    for job in new_jobs:
        run = _load_new_run(new_exp, new_plan, job, archive_root)
        run_id = job["run_id"]
        runs[(job["model"], job["formula"])] = run
        run_dir = run["run_dir"]
        result = run["result"]
        input_files += [run_dir / "status.json", run_dir / "manifest.json",
                        run_dir / "evaluation" / f"final_test_{STEPS}.json",
                        Path(result["arrays_path"]), Path(result["trace_path"]),
                        run_dir / "updates.csv", run_dir / "evaluations.csv",
                        run_dir / "episodes.csv"]
        input_files += [archive_root / "models" / "runs" / NEW_EXPERIMENT / run_id / name
                        for name in run["weights"]]
    if len(runs) != 12:
        raise RuntimeError("必须有三网络×四组合共12个真实终点模型")
    _check_config_matrix(runs, old_plan, new_plan)
    curves = {}
    updates = {}
    summary = []
    effects = {}
    for model in MODELS:
        by_formula = {formula: runs[model, formula]["by_key"] for formula in FORMULAS}
        if len({frozenset(rows) for rows in by_formula.values()}) != 1:
            raise RuntimeError(f"{model} 四组合场景与动作种子不完全一致")
        effects[model] = {field: paired_factorial(by_formula, field) for field in METRICS}
        for formula in FORMULAS:
            run = runs[model, formula]
            run_id = run["job"]["run_id"]
            rows = list(run["by_key"].values())
            summary.append({"run_id": run_id, "model": model, "formula": formula,
                            "train_seed": SEED, "scene_count": 100, "action_trials": 300,
                            "metrics": {field: statistics.mean(float(row[field]) for row in rows)
                                        for field in METRICS},
                            "manifest_config": run["manifest"]["config"]})
            plan = old_plan if formula in ("legacy", "corrected") else new_plan
            curves[run_id] = _validation_curve(run["run_dir"], run_id, plan)
            updates[run_id] = _read_updates(run["run_dir"], run_id)
    inputs = {str(path): sha256_file(path) for path in sorted(set(input_files))}
    target = new_exp / "analysis"
    if target.exists():
        prior_path = target / "analysis_manifest.json"
        if prior_path.is_file():
            prior = _read_json(prior_path)
            if prior.get("input_sha256") == inputs and all(
                (target / name).is_file() and sha256_file(target / name) == digest
                for name, digest in prior.get("output_sha256", {}).items()
            ):
                return target / "中文四组合公式对照报告.md"
        raise RuntimeError(f"分析目录已存在且输入或输出变化，保留现场拒绝覆盖：{target}")
    staging = new_exp / ("analysis_staging_" + uuid.uuid4().hex[:10])
    staging.mkdir(parents=True)
    try:
        atomic_json(staging / "run_summary.json", summary)
        atomic_json(staging / "paired_effects.json", effects)
        _write_csv(staging / "validation_curves.csv",
                   [point for run_id in sorted(curves) for point in curves[run_id]],
                   ("run_id", "round", "env_step", "si_mean", "sgm_mean"))
        _write_csv(staging / "learning_diagnostics.csv",
                   [point for run_id in sorted(updates) for point in updates[run_id]],
                   ("run_id", "env_step", "update_step", *DIAGNOSTICS))
        _figures(staging, MODELS, curves, updates)
        report = _report(summary, effects, old_plan["source"]["git_commit"],
                         new_plan["source"]["git_commit"])
        (staging / "中文四组合公式对照报告.md").write_text(report, encoding="utf-8")
        outputs = {path.name: sha256_file(path) for path in staging.iterdir() if path.is_file()}
        atomic_json(staging / "analysis_manifest.json", {
            "experiment_id": NEW_EXPERIMENT, "training_seed": SEED,
            "old_training_commit": old_plan["source"]["git_commit"],
            "new_training_commit": new_plan["source"]["git_commit"],
            "old_experiment": str(old_exp), "new_experiment": str(new_exp),
            "test_scene_sha256": _scene_hashes(old_plan),
            "action_seeds": old_plan["action_seeds"],
            "input_sha256": inputs, "output_sha256": outputs,
        })
        staging.rename(target)
    except BaseException:
        # Keep partial output for diagnosis; never replace an existing report.
        raise
    return target / "中文四组合公式对照报告.md"


def main():
    parser = argparse.ArgumentParser(description="十二组单种子SAC公式对照的离线分析")
    parser.add_argument("--archive-root", default="D:/paper")
    parser.add_argument("--old-frozen-root", default="D:/paper_worktrees/frozen-sac-20260924-r2")
    parser.add_argument("--new-frozen-root", default="D:/paper_worktrees/frozen-sac-factorial-20260925")
    args = parser.parse_args()
    print(generate(args.archive_root, args.old_frozen_root, args.new_frozen_root))


if __name__ == "__main__":
    main()
