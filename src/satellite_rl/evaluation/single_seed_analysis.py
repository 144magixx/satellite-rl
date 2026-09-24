"""Offline report for the six approved SAC runs at training seed 51001.

All results come from saved terminal arrays, traces, checkpoints, and existing
same-scene baseline files. Nothing in this module trains or uses CUDA.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

import numpy as np

from satellite_rl.evaluation.experiment_report import (
    FORMULA_NAMES, MODEL_NAMES, MAIN_FIELDS, _load_run,
)
from satellite_rl.evaluation.final_analysis import (
    _distribution_rows, behavior_for_run, timing_for_run,
)
from satellite_rl.evaluation.post_training_checks import source_guard
from satellite_rl.evaluation.single_seed_scope import (
    EXPERIMENT_ID, RUN_IDS, SEED, TRAIN_COMMIT, require_completed,
)
from satellite_rl.training import experiment_runner as runner
from satellite_rl.training.run_monitor import atomic_json, replace_with_retry, utc_now


BASELINE_ROOT = Path(r"D:/paper/outputs/verification/baseline_recheck_sac_r2_20260924")
VALIDATION_FIELDS = ("si_mean", "sgm_mean", "throughput_bps", "return")
BEHAVIOR_FIELDS = ("final_si_mean", "unserved_rate", "requested_power_mean_w",
                   "requested_slots_mean", "actual_slots_mean", "zero_slot_rate", "clipped_rate")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _average(rows, field):
    return statistics.mean(float(row[field]) for row in rows)


def _scene_means(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["scene_id"])].append(float(row[field]))
    return {scene: statistics.mean(values) for scene, values in grouped.items()}


def load_same_scene_baselines(plan, frozen_root, baseline_root=BASELINE_ROOT):
    """Verify the previously saved deterministic greedy/fixed rollouts."""
    baseline_root = Path(baseline_root)
    config_path = baseline_root / "config_and_summary.json"
    records_path = baseline_root / "per_scene_results.jsonl"
    config = _read(config_path)
    if (config.get("git_commit") != TRAIN_COMMIT
            or config.get("scene_ids") != sorted(int(value) for value in plan["test_scenes"])
            or config.get("scene_count") != len(plan["test_scenes"])
            or Path(config.get("code_root", "")).resolve() != Path(frozen_root).resolve()
            or set(config.get("policies", {})) != {"greedy", "fixed"}):
        raise RuntimeError("既有贪心/固定策略清单与本轮冻结场景不一致")
    for source, expected in config["source_sha256"].items():
        if not Path(source).is_file() or runner.sha256_file(source) != expected:
            raise RuntimeError(f"基线源码发生变化：{source}")
    by_policy = defaultdict(dict)
    with records_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            policy, scene = row["policy"], str(row["scene_id"])
            if policy not in ("greedy", "fixed") or scene not in plan["test_scenes"]:
                raise RuntimeError("基线包含未知策略或场景")
            if scene in by_policy[policy]:
                raise RuntimeError("基线包含重复策略/场景")
            reference = plan["test_scenes"][scene]
            if (Path(row["scene_path"]).resolve() != Path(reference["path"]).resolve()
                    or row["scene_sha256"] != reference["sha256"]
                    or len(row["satisfaction_per_beam"]) != 100
                    or len(row["sgm_per_beam"]) != 100
                    or row["total_beams"] != 100
                    or not math.isclose(sum(row["satisfaction_per_beam"]), row["si_sum"],
                                        rel_tol=0, abs_tol=1e-7)
                    or not math.isclose(row["si_sum"] / 100, row["si_mean"],
                                        rel_tol=0, abs_tol=1e-9)):
                raise RuntimeError("基线场景、身份或终态需求满足率不一致")
            by_policy[policy][scene] = row
    if any(set(by_policy[policy]) != set(plan["test_scenes"]) for policy in ("greedy", "fixed")):
        raise RuntimeError("贪心或固定策略缺少共同的100个场景")
    for policy in ("greedy", "fixed"):
        for field in ("si_mean", "sgm_mean", "throughput_bps", "total_power_w"):
            observed = statistics.mean(row[field] for row in by_policy[policy].values())
            saved = config["summary"][policy][field]["mean"]
            if not math.isclose(observed, saved, rel_tol=1e-11, abs_tol=1e-9):
                raise RuntimeError(f"{policy} 基线汇总与逐场景原始记录不一致：{field}")
    return {"config_path": str(config_path), "config_sha256": runner.sha256_file(config_path),
            "records_path": str(records_path), "records_sha256": runner.sha256_file(records_path),
            "config": config, "by_policy": dict(by_policy)}


def summarize_runs(runs):
    """One result per trained model: scenario spread is not seed variation."""
    output = []
    for run in runs:
        job = run["job"]
        sampled = run["rows"]
        argmax = run["argmax_rows"]
        best = [row for row in run["best_test"]["rows"]
                if row["action_mode"] == "sample"]
        scene_si = _scene_means(sampled, "si_mean")
        output.append({"run_id": job["run_id"], "model": job["model"],
                       "formula": job["formula"], "train_seed": SEED,
                       "sampled_trials": len(sampled), "independent_test_scenes": len(scene_si),
                       "scene_si_sample_std": statistics.stdev(scene_si.values()),
                       "metrics": {field: _average(sampled, field) for field in MAIN_FIELDS},
                       "argmax": {field: _average(argmax, field)
                                  for field in ("si_mean", "sgm_mean", "power_exceeded",
                                                "completed", "zero_slot_rate")},
                       "best_validation_selected": {field: _average(best, field)
                                                    for field in ("si_mean", "sgm_mean",
                                                                  "power_exceeded", "completed")}})
    if len(output) != 6 or {row["run_id"] for row in output} != RUN_IDS:
        raise RuntimeError("主表不是精确六个终点模型")
    return output


def paired_formula_differences(runs, plan):
    indexed = {(run["job"]["model"], run["job"]["formula"]): run for run in runs}
    output = []
    for model in runner.MODELS:
        old, new = indexed[(model, "legacy")], indexed[(model, "corrected")]
        if old["by_key"].keys() != new["by_key"].keys():
            raise RuntimeError(f"{model} 原/修公式考核的场景动作键不一致")
        item = {"model": model, "train_seed": SEED,
                "paired_action_trials": len(old["by_key"]),
                "independent_scenes": len(plan["test_scenes"]),
                "difference": {}, "higher_scenes": {}}
        for field in ("si_mean", "sgm_mean", "throughput_bps", "total_power_w",
                      "power_exceeded", "completed", "allocated_slots", "zero_slot_rate"):
            per_action = [float(new["by_key"][key][field])
                          - float(old["by_key"][key][field])
                          for key in sorted(old["by_key"])]
            per_scene_old = _scene_means(old["rows"], field)
            per_scene_new = _scene_means(new["rows"], field)
            if set(per_scene_old) != set(plan["test_scenes"]):
                raise RuntimeError("配对公式差值缺少100个场景")
            scene_delta = [per_scene_new[scene] - per_scene_old[scene]
                           for scene in sorted(per_scene_old)]
            item["difference"][field] = {"mean_over_paired_actions": statistics.mean(per_action),
                                          "sample_std_across_scenes": statistics.stdev(scene_delta)}
            item["higher_scenes"][field] = sum(value > 0 for value in scene_delta)
        output.append(item)
    return output


def compare_with_baselines(runs, baselines):
    output = []
    for run in runs:
        model_scene = _scene_means(run["rows"], "si_mean")
        for policy in ("greedy", "fixed"):
            baseline = baselines["by_policy"][policy]
            if set(model_scene) != set(baseline):
                raise RuntimeError("模型与基线考核场景不一致")
            delta = [model_scene[scene] - baseline[scene]["si_mean"]
                     for scene in sorted(model_scene)]
            output.append({"run_id": run["job"]["run_id"], "policy": policy,
                           "train_seed": SEED, "scene_count": len(delta),
                           "model_minus_baseline_si_mean": statistics.mean(delta),
                           "scene_delta_sample_std": statistics.stdev(delta),
                           "model_higher_scenes": sum(value > 0 for value in delta),
                           "model_lower_scenes": sum(value < 0 for value in delta)})
    return output


def validation_curve_for_run(run, plan):
    """Keep each of the forty real validation steps for one training seed."""
    config = run["manifest"]["config"]
    scene_ids = config["validation_scene_seeds"]
    action_seeds = config["action_seeds"]
    expected = {(str(scene), str(action)) for scene in scene_ids for action in action_seeds}
    if len(expected) != 60 or len(scene_ids) != 20 or len(action_seeds) != 3:
        raise RuntimeError("验证场景或动作种子不是20×3")
    grouped = defaultdict(list)
    for row in run["evaluations"]:
        if row["checkpoint"] in ("validation", "final_validation") and row.get("action_mode") == "sample":
            grouped[(int(row["env_step"]), row["checkpoint"])].append(row)
    rounds = {}
    for (step, phase), rows in sorted(grouped.items()):
        actual = [(str(row["scene_id"]), str(row["action_seed"])) for row in rows]
        if len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError(f"{run['job']['run_id']} 在{step}步验证缺失或重复场景")
        index = step // plan["validation_interval"]
        if index not in range(1, 41) or step > 400000:
            raise RuntimeError("验证实际交互步数超出四十轮预算")
        point = {"validation_round": index, "env_step": step,
                 **{field: _average(rows, field) for field in VALIDATION_FIELDS}}
        if not all(np.isfinite(point[field]) for field in VALIDATION_FIELDS):
            raise RuntimeError("验证曲线包含非有限数值")
        if index in rounds:
            previous = rounds[index]
            if (step != 400000 or previous["env_step"] != step
                    or any(not np.isclose(previous[field], point[field],
                                          rtol=1e-8, atol=1e-10)
                           for field in VALIDATION_FIELDS)):
                raise RuntimeError("同轮次出现不同验证数据")
        rounds[index] = point
    if set(rounds) != set(range(1, 41)):
        raise RuntimeError(f"{run['job']['run_id']} 缺少预定的四十轮验证")
    return [rounds[index] for index in sorted(rounds)]


def learning_curve_for_run(run):
    with (run["run_dir"] / "updates.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = set(range(100, 398001, 100))
    steps = [int(row["update_step"]) for row in rows]
    if len(steps) != len(expected) or set(steps) != expected:
        raise RuntimeError(f"{run['job']['run_id']} 更新诊断点缺失或重复")
    fields = ("alpha", "entropy", "entropy_freq", "entropy_slots",
              "entropy_power", "actor_loss", "critic_1_loss", "critic_2_loss",
              "alpha_loss", "q_mean", "target_mean")
    points = []
    for row in rows:
        point = {"env_step": int(row["env_step"]),
                 "update_step": int(row["update_step"]),
                 **{field: float(row[field]) for field in fields}}
        if not all(np.isfinite(point[field]) for field in fields):
            raise RuntimeError(f"{run['job']['run_id']} 更新诊断含NaN/Inf")
        points.append(point)
    return points


def _plot_validation(curves, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    mapping = (("si_mean", 100, "需求满足率 SI（%）"),
               ("sgm_mean", 100, "供需匹配程度 SGM（%）"),
               ("throughput_bps", 1e-9, "总吞吐量（Gbps）"),
               ("return", 1, "验证回合累计奖励"))
    colors = {"transformer": "tab:blue", "cnn_matched": "tab:orange",
              "mlp_matched": "tab:green"}
    for row_index, formula in enumerate(("legacy", "corrected")):
        for model in runner.MODELS:
            points = curves[f"{model}_{formula}"]
            x = [item["env_step"] for item in points]
            for axis, (field, scale, _) in zip(axes[row_index], mapping):
                axis.plot(x, [item[field] * scale for item in points],
                          color=colors[model], label=MODEL_NAMES[model], linewidth=1.4)
        for axis, (_, _, title) in zip(axes[row_index], mapping):
            axis.set_title(f"{FORMULA_NAMES[formula]}：{title}")
            axis.set_xlabel("实际环境交互步数")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
    fig.suptitle("固定场景验证曲线：单一训练种子 51001", fontsize=14)
    fig.tight_layout()
    path = output / "三网络单种子验证曲线.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_learning(learning, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    paths = []
    for model in runner.MODELS:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for row_index, formula in enumerate(("legacy", "corrected")):
            points = learning[f"{model}_{formula}"]
            x = [point["env_step"] for point in points]
            axis = axes[row_index]
            axis[0].plot(x, [point["alpha"] for point in points], linewidth=1)
            axis[0].set_yscale("symlog", linthresh=1e-6)
            axis[0].set_title("温度 α（对称对数刻度）")
            for field, label in (("entropy_freq", "频率"), ("entropy_slots", "槽数"),
                                 ("entropy_power", "功率")):
                axis[1].plot(x, [point[field] for point in points], label=label, linewidth=1)
            axis[1].set_title("三个动作头的随机程度")
            axis[1].legend(fontsize=7)
            for field, label in (("critic_1_loss", "Critic 1"),
                                 ("critic_2_loss", "Critic 2")):
                axis[2].plot(x, [point[field] for point in points], label=label, linewidth=1)
            axis[2].set_yscale("symlog", linthresh=1)
            axis[2].set_title("价值网络损失（对称对数刻度）")
            axis[2].legend(fontsize=7)
            axis[3].plot(x, [point["actor_loss"] for point in points], linewidth=1)
            axis[3].set_yscale("symlog", linthresh=1)
            axis[3].set_title("策略网络损失（对称对数刻度）")
            for panel in axis:
                panel.set_xlabel("实际环境交互步数")
                panel.grid(alpha=0.25)
            axis[0].set_ylabel(FORMULA_NAMES[formula])
        fig.suptitle(f"{MODEL_NAMES[model]}：单种子学习诊断", fontsize=14)
        fig.tight_layout()
        path = output / f"{model}_学习诊断.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_training_episodes(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    paths = []
    for model in runner.MODELS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for run in runs:
            job = run["job"]
            if job["model"] != model:
                continue
            x = np.asarray([float(row["env_step"]) for row in run["episodes"]])
            color = "tab:blue" if job["formula"] == "legacy" else "tab:orange"
            label = FORMULA_NAMES[job["formula"]]
            for axis, field, scale in ((axes[0], "return", 1),
                                        (axes[1], "si_mean", 100)):
                y = np.asarray([float(row[field]) * scale for row in run["episodes"]])
                axis.plot(x, y, color=color, alpha=0.12, linewidth=0.6)
                window = min(50, len(y))
                if window:
                    smooth = np.convolve(y, np.ones(window) / window, mode="valid")
                    axis.plot(x[window - 1:], smooth, color=color,
                              label=label + "（50回合滑动平均）", linewidth=1.2)
        for axis, title in zip(axes, ("训练回合累计奖励", "训练回合终态需求满足率 SI（%）")):
            axis.set_title(title)
            axis.set_xlabel("实际环境交互步数")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
        fig.suptitle(f"{MODEL_NAMES[model]}：单种子训练曲线", fontsize=13)
        fig.tight_layout()
        path = output / f"{model}_训练回合曲线.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _parameters(model):
    module_name, hidden = runner.MODELS[model]
    module = importlib.import_module("satellite_rl.agents." + module_name)
    actor = module.PolicyNet(204, hidden, 100, 10, 10)
    critic = module.ValueNet(204, hidden, 100, 10, 10)
    return {"actor": sum(parameter.numel() for parameter in actor.parameters()),
            "critic_each": sum(parameter.numel() for parameter in critic.parameters())}


def _write_csv(path, rows, fields):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _input_hashes(exp_dir, plan, final, baselines, selection_path):
    files = [exp_dir / "suite_plan.json", selection_path,
             baselines["config_path"], baselines["records_path"]]
    files += [Path(reference["path"]) for reference in plan["test_scenes"].values()]
    for run_id, info in final.items():
        run_dir = exp_dir / run_id
        files += [run_dir / name for name in
                  ("status.json", "manifest.json", "events.jsonl", "updates.csv",
                   "episodes.csv", "evaluations.csv")]
        files += [Path(info[name]) for name in
                  ("initial_checkpoint", "latest_checkpoint", "checkpoint", "actor",
                   "best_checkpoint", "best_actor",
                   "original_test", "best_test")]
        for key in ("original_test", "best_test"):
            result = _read(info[key])
            files += [Path(result["arrays_path"]), Path(result["trace_path"])]
    paths = sorted({Path(path) for path in files})
    return {str(path): runner.sha256_file(path) for path in paths}


def _selection(exp_dir, plan):
    path = exp_dir / "scope_change_single_seed_20260924" / "single_seed_selection.json"
    selected = _read(path)
    before = Path(selected["source_plan_before_path"])
    if (selected.get("train_seed") != SEED or selected.get("training_commit") != TRAIN_COMMIT
            or set(selected.get("run_ids", [])) != RUN_IDS
            or selected.get("post_training_output") != plan["single_seed_output_dir"]
            or not before.is_file()
            or runner.sha256_file(before) != selected["source_plan_before_sha256"]):
        raise RuntimeError("用户收窄为六组的原始计划及校验值不符")
    return path, selected


def _report_text(summary, paired, baseline, baseline_pairs, timing, behavior,
                 learning, figures, provenance):
    indexed = {(row["model"], row["formula"]): row for row in summary}
    times = {row["run_id"]: row for row in timing}
    diagnostics = {key: values[-1] for key, values in learning.items()}
    lines = ["# 单一种子 51001 的六组 SAC 对照", "",
             f"冻结训练提交：`{TRAIN_COMMIT}`；离线分析提交：`{provenance['post_commit']}`。"
             "三种网络各以原、修两套学习公式训练40万次环境交互。"
             "只有一个训练种子，因此下列差异是本次固定场景的观察值，"
             "**不能估计训练种子间波动，也不能宣称普遍或稳定优势**。", "",
             "## 预算终点：全部波束的终态服务与约束", "",
             "需求满足率按100束最终速率计算，含后续干扰，未处理束记零；"
             "SGM（供需匹配程度）单列。每组使用100个共同场景×3个动作采样种子。", "",
             "| 网络与公式 | 终态 SI / % | SGM / % | 吞吐 / Gbps | 功率 / W | 超功率 / % | 完成100束 / % | 零槽 / % | 实际槽数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in runner.MODELS:
        for formula in ("legacy", "corrected"):
            m = indexed[(model, formula)]["metrics"]
            lines.append(f"| {MODEL_NAMES[model]} · {FORMULA_NAMES[formula]} | "
                         f"{m['si_mean'] * 100:.2f} | {m['sgm_mean'] * 100:.2f} | "
                         f"{m['throughput_bps'] * 1e-9:.2f} | {m['total_power_w']:.1f} | "
                         f"{m['power_exceeded'] * 100:.1f} | {m['completed'] * 100:.1f} | "
                         f"{m['zero_slot_rate'] * 100:.1f} | {m['allocated_slots']:.1f} |")
    lines += ["", "贪心与固定策略是**同一修正环境、同100场景**的已有确定性参照，"
              "每场景各执行一次；模型每场景三个动作种子。两种参照的请求槽数裁剪率"
              "未由已存基线记录完整提供，不猜测该列。", "",
              "| 参照策略 | 终态 SI / % | SGM / % | 功率 / W | 超功率 / % | 完成100束 / % | 实际槽数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for policy, label in (("greedy", "贪心"), ("fixed", "固定")):
        m = baseline["config"]["summary"][policy]
        lines.append(f"| {label} | {m['si_mean']['mean'] * 100:.2f} | "
                     f"{m['sgm_mean']['mean'] * 100:.2f} | {m['total_power_w']['mean']:.1f} | "
                     f"{m['power_exceeded']['mean'] * 100:.1f} | "
                     f"{m['completed']['mean'] * 100:.1f} | {m['allocated_slots']['mean']:.1f} |")
    lines += ["", "`baseline_pairs.csv` 按同一场景将模型三个动作种子的平均 SI 与单次确定性参照配对。"
              "参照的资源与约束结果须和 SI 一起判断。", "",
              "## 同网络内修正公式减原公式", "",
              "| 网络 | SI 差 / 百分点 | SGM 差 / 百分点 | 功率差 / W | 超功率率差 / 百分点 | 完成率差 / 百分点 | SI 较高场景 / 100 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in paired:
        d = item["difference"]
        lines.append(f"| {MODEL_NAMES[item['model']]} | "
                     f"{d['si_mean']['mean_over_paired_actions'] * 100:+.2f} | "
                     f"{d['sgm_mean']['mean_over_paired_actions'] * 100:+.2f} | "
                     f"{d['total_power_w']['mean_over_paired_actions']:+.1f} | "
                     f"{d['power_exceeded']['mean_over_paired_actions'] * 100:+.1f} | "
                     f"{d['completed']['mean_over_paired_actions'] * 100:+.1f} | "
                     f"{item['higher_scenes']['si_mean']} |")
    lines += ["", "差值在同一训练种子、场景和动作种子上逐一计算；"
              "`paired_formula.json` 另存100场景间的差值样本标准差，"
              "它**不是训练种子间标准差**。两组学习公式使用同一环境奖励口径；"
              "训练回报可作过程诊断，但最终效果须以相同考核场景的终态需求满足率和资源约束判断。", "",
              "## 网络规模、耗时与数值诊断", "",
             "| 网络与公式 | Actor参数 | 单Critic参数 | 日历时长 / 小时 | 断点累计进程时长 / 小时 | 末次α | 末次Critic 1损失 | 末次Q均值 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in runner.MODELS:
        for formula in ("legacy", "corrected"):
            row = indexed[(model, formula)]
            p = row["parameters"]
            t = times[row["run_id"]]
            d = diagnostics[f"{model}_{formula}"]
            lines.append(f"| {MODEL_NAMES[model]} · {FORMULA_NAMES[formula]} | "
                         f"{p['actor']} | {p['critic_each']} | "
                         f"{t['calendar_seconds_including_retries'] / 3600:.2f} | "
                         f"{t['recorded_active_seconds'] / 3600:.2f} | "
                         f"{d['alpha']:.3e} | {d['critic_1_loss']:.3e} | "
                         f"{d['q_mean']:.3e} |")
    lines += ["", "日历时长可能包含已确认的断电停机；断点累计时长不含断电前"
              "最后一个断点之后被回滚的尾段。三路共享GPU作业的耗时不是独占GPU网络速度排名。"
              "周期验证与终点评价时间块及实际更新次数见 `timing_by_run.csv`。", "",
              "首四组的已登记现象包括 CNN 修正公式 α 与 Critic 损失极大、验证成绩回落，"
              "以及三组超功率提前终止。上表从当前原始 `updates.csv` 再列终点数值；"
              "完整约束、零槽、请求裁剪和 Q 值另见原始CSV与本报告数据。"
              "一次更新的数学检查通过不等于整个训练稳定。", "",
              "## 需求与处理阶段的动作行为", "",
              "`behavior_by_run.csv` 使用终态数组和完整逐步日志逐场景复核后，"
              "按业务需求档与波束处理前/中/后段列出全部目标束数、未处理率、"
              "终态 SI、请求功率、请求/实际槽数、零槽和裁剪率。"
              "未处理波束纳入服务成绩分母，动作平均只对已执行分配计数；"
              "`behavior_distributions.csv` 保留功率、槽数、频率起点的完整计数。", "",
              "需求与处理顺序本身有关联，表中关联仅描述实际动作，不能证明因果。", "",
              "## 收敛与动作方式", "",
              "固定验证场景的每轮实际交互步数及原始均值在 `validation_curves.json`；"
              "不插值、不跨种子平均，预算终点重复验证只计一次。"
              "训练回合图的淡线为逐回合原始值，深线为预先固定的50回合滑动平均；"
              "原始回合数据仍保留在各run的 `episodes.csv`。"
              "更新诊断原始数值在 `learning_diagnostics.csv`，"
              "损失与温度图使用对称对数坐标且不改变原数据。", "",
              "![单种子三网络验证曲线](三网络单种子验证曲线.png)", "",
              "逐头最大概率动作与按验证选中的模型仅作补充，"
              "`run_summary.json` 单列两者的 SI、SGM、约束及验证选中步数，"
              "不用最终考核挑选保存点。", "",
              "## 数学与环境计算的独立证据", "",
              "两项 SAC 公式方向与完整联合双 Q 的正确性，以及功率、干扰和终态统计修正，"
              "分别依赖已登记的确定性检查，包括目标熵两侧温度更新方向、"
              "完整联合动作双Q期望与梯度，以及功率/干扰、零槽和终态全束统计。"
              "本报告没有把六组模型分数当作这些计算正确性的证明。"
              "旧错误环境权重的历史分数不进入本次六模型排名。"
              "已登记的依据见 `Q&A/算法与训练/匹配规模SAC训练与公式修复对照.md`"
              "与 `docs/算法与环境问题分级清单.md`。", "",
              "## 结论边界", "",
              "这只是一个训练种子在固定仿真场景下的受控对照。"
              "相同场景上的高低与约束差异可被复核，"
              "但无法估计重新训练时的波动，也不能据此宣称稳定优势或真实通信性能提升。"
              "补充需求/次序条件、旧权重跨环境和独立CUDA耗时结果若尚未执行，"
              "应与本主报告分开标为待验证。", "",
              "原始输入及每个输出文件的 SHA-256 保存在 `analysis_manifest.json`。", ""]
    return "\n".join(lines)


def generate_single_seed_analysis(archive_root="D:/paper",
                                  frozen_root="D:/paper_worktrees/frozen-sac-20260924-r2",
                                  baseline_root=BASELINE_ROOT):
    """Generate an isolated, immutable report only after all six runs finish."""
    provenance = source_guard(frozen_root)
    exp_dir, plan, final = require_completed(archive_root, frozen_root)
    selection_path, selection = _selection(exp_dir, plan)
    baselines = load_same_scene_baselines(plan, frozen_root, baseline_root)
    input_hashes = _input_hashes(exp_dir, plan, final, baselines, selection_path)
    target = Path(plan["single_seed_output_dir"]) / "analysis"
    if target.exists():
        manifest_path = target / "analysis_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"已有部分单种子分析目录，保留现场并拒绝覆盖：{target}")
        prior = _read(manifest_path)
        if (prior.get("analysis_commit") != provenance["post_commit"]
                or prior.get("input_sha256") != input_hashes):
            raise RuntimeError("已有单种子分析使用不同源码或原始输入，拒绝覆盖")
        if all((target / name).is_file() and runner.sha256_file(target / name) == digest
               for name, digest in prior["output_sha256"].items()):
            return target / "中文单种子六组报告.md"
        raise RuntimeError("已有单种子分析文件损坏，保留现场并拒绝覆盖")
    runs = [_load_run(exp_dir, job, plan) for job in plan["jobs"]]
    summary = summarize_runs(runs)
    parameters = {model: _parameters(model) for model in runner.MODELS}
    for row in summary:
        row["parameters"] = parameters[row["model"]]
        row["best_checkpoint_step"] = final[row["run_id"]]["best_checkpoint_step"]
    paired = paired_formula_differences(runs, plan)
    baseline_pairs = compare_with_baselines(runs, baselines)
    timings = [timing_for_run(run, plan) for run in runs]
    behavior = [record for run in runs for record in behavior_for_run(run, plan)]
    distributions = _distribution_rows(behavior)
    curves = {f"{run['job']['model']}_{run['job']['formula']}":
              validation_curve_for_run(run, plan) for run in runs}
    learning = {f"{run['job']['model']}_{run['job']['formula']}":
                learning_curve_for_run(run) for run in runs}
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name("analysis_staging_" + uuid.uuid4().hex[:10])
    staging.mkdir()
    try:
        atomic_json(staging / "run_summary.json", summary)
        atomic_json(staging / "paired_formula.json", paired)
        atomic_json(staging / "baseline_reference.json", {
            "config_path": baselines["config_path"],
            "config_sha256": baselines["config_sha256"],
            "records_path": baselines["records_path"],
            "records_sha256": baselines["records_sha256"],
            "summary": baselines["config"]["summary"],
        })
        atomic_json(staging / "validation_curves.json", {
            "training_seed": SEED, "training_repeats": 1,
            "actual_step_per_run": True, "validation_rounds_per_run": 40,
            "metrics": list(VALIDATION_FIELDS), "curves": curves,
        })
        _write_csv(staging / "validation_curves.csv",
                   [{"run_id": f"{model}_{formula}_s{SEED}", "model": model,
                     "formula": formula, **point}
                    for model in runner.MODELS for formula in ("legacy", "corrected")
                    for point in curves[f"{model}_{formula}"]],
                   ("run_id", "model", "formula", "validation_round", "env_step",
                    *VALIDATION_FIELDS))
        _write_csv(staging / "learning_diagnostics.csv",
                   [{"run_id": f"{model}_{formula}_s{SEED}", "model": model,
                     "formula": formula, **point}
                    for model in runner.MODELS for formula in ("legacy", "corrected")
                    for point in learning[f"{model}_{formula}"]],
                   ("run_id", "model", "formula", "env_step", "update_step",
                    "alpha", "entropy", "entropy_freq", "entropy_slots", "entropy_power",
                    "actor_loss", "critic_1_loss", "critic_2_loss", "alpha_loss",
                    "q_mean", "target_mean"))
        _write_csv(staging / "timing_by_run.csv", timings,
                   tuple(timings[0].keys()))
        _write_csv(staging / "baseline_pairs.csv", baseline_pairs,
                   tuple(baseline_pairs[0].keys()))
        behavior_fields = ("run_id", "model", "formula", "train_seed", "axis", "group",
                           "target_beams", "attempted_beams", "unserved_beams", "si_sum",
                           *BEHAVIOR_FIELDS)
        _write_csv(staging / "behavior_by_run.csv", behavior, behavior_fields)
        _write_csv(staging / "behavior_distributions.csv", distributions,
                   tuple(distributions[0].keys()))
        _plot_validation(curves, staging)
        _plot_learning(learning, staging)
        _plot_training_episodes(runs, staging)
        (staging / "中文单种子六组报告.md").write_text(
            _report_text(summary, paired, baselines, baseline_pairs, timings,
                         behavior, learning, None, provenance), encoding="utf-8")
        outputs = [path for path in staging.iterdir() if path.is_file()
                   and path.name != "analysis_manifest.json"]
        manifest = {
            "generated_at": utc_now(), "analysis_commit": provenance["post_commit"],
            "training_commit": TRAIN_COMMIT, "training_seed": SEED,
            "training_repeats": 1, "experiment_id": EXPERIMENT_ID,
            "source_plan": plan["source_plan_path"],
            "single_seed_selection": str(selection_path),
            "single_seed_selection_sha256": runner.sha256_file(selection_path),
            "run_ids": sorted(RUN_IDS), "fixed_scenes": len(plan["test_scenes"]),
            "action_seeds": plan["action_seeds"],
            "matched_training_source_sha256": provenance["matched_file_sha256"],
            "input_sha256": input_hashes,
            "output_sha256": {path.name: runner.sha256_file(path) for path in outputs},
            "limitations": [
                "只有一个训练种子；场景间波动不是训练种子间波动。",
                "旧十八组计划状态保留历史，不作为六组完成证据。",
                "已登记CNN修正公式的数值失稳和约束超限需要与终态分数共同解释。",
                "旧错误环境的历史分数不纳入本轮六模型排名。",
            ],
        }
        atomic_json(staging / "analysis_manifest.json", manifest)
        staging.rename(target)
        return target / "中文单种子六组报告.md"
    except Exception as error:
        atomic_json(staging / "failure.json", {"at": utc_now(),
                                                 "type": type(error).__name__,
                                                 "message": str(error),
                                                 "staging_dir": str(staging)})
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description="种子51001六组SAC终点的离线报告与曲线")
    parser.add_argument("--archive-root", default="D:/paper")
    parser.add_argument("--frozen-root", default="D:/paper_worktrees/frozen-sac-20260924-r2")
    parser.add_argument("--baseline-root", default=str(BASELINE_ROOT))
    args = parser.parse_args()
    print(generate_single_seed_analysis(args.archive_root, args.frozen_root,
                                        args.baseline_root))


if __name__ == "__main__":
    main()
