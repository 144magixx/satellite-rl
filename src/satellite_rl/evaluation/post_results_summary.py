"""Summarize only completed, saved post-training evaluations; never run models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from satellite_rl.evaluation.experiment_report import FORMULA_NAMES, MODEL_NAMES
from satellite_rl.evaluation.post_training_checks import (
    ACTION_SEEDS, CONDITIONS, EXPECTED_RUN_IDS, OLD_ENV_COMMIT, SCENE_IDS, TRAIN_COMMIT,
    _load_json, require_completed, source_guard, verify_cached_post_result,
)
from satellite_rl.training.experiment_runner import sha256_file
from satellite_rl.training.run_monitor import atomic_json, utc_now


FIELDS = ("si_mean", "sgm_mean", "throughput_bps", "total_power_w",
          "power_exceeded", "completed")
CONDITION_NAMES = {"demand_075": "需求 ×0.75", "demand_125": "需求 ×1.25",
                   "within_group_order": "同组内次序变化"}
ENV_NAMES = {"before_c4963ba": "修正前环境", "corrected_r2": "修正后环境"}


def _hash_into(record, path):
    path = Path(path)
    record[str(path.resolve())] = sha256_file(path)


def _sample_rows(result):
    rows = [row for row in result["rows"] if row.get("action_mode") == "sample"]
    expected = {(str(scene), str(seed)) for scene in SCENE_IDS for seed in ACTION_SEEDS}
    keys = [(str(row["scene_id"]), str(row["action_seed"])) for row in rows]
    if len(keys) != 60 or set(keys) != expected:
        raise RuntimeError("预定20场景×3动作种子的采样结果不完整或重复")
    return {key: row for key, row in zip(keys, rows)}


def paired_means(original, changed):
    """Return one mean difference per metric over exactly 60 matched trials."""
    if original.keys() != changed.keys() or len(original) != 60:
        raise RuntimeError("条件与原条件未按场景及动作种子完整配对")
    return {field: statistics.mean(float(changed[key][field]) - float(original[key][field])
                                   for key in sorted(original))
            for field in FIELDS}


def summarize_three_seeds(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["condition"], row["model"], row["formula"])
        if row["train_seed"] in grouped[key]:
            raise RuntimeError("同一配置出现重复训练种子")
        grouped[key][row["train_seed"]] = row
    if len(grouped) != len(CONDITIONS) * 6:
        raise RuntimeError("条件、网络或学习公式不完整")
    summary = []
    for (condition, model, formula), by_seed in sorted(grouped.items()):
        if set(by_seed) != {51001, 51002, 51003}:
            raise RuntimeError("条件对照缺少预定三个独立训练种子")
        item = {"condition": condition, "model": model, "formula": formula,
                "train_seeds": sorted(by_seed)}
        for field in FIELDS:
            values = [by_seed[seed][field] for seed in sorted(by_seed)]
            item[field + "_difference_mean"] = statistics.mean(values)
            item[field + "_difference_std_across_train_seeds"] = statistics.stdev(values)
        summary.append(item)
    return summary


def _read_condition_results(exp_dir, post_root, plan, final, eval_commit, hashes):
    originals = {}
    for job in plan["jobs"]:
        result_path = Path(final[job["run_id"]]["original_test"])
        originals[job["run_id"]] = _sample_rows(_load_json(result_path))
        _hash_into(hashes, result_path)
    by_seed = []
    for condition in CONDITIONS:
        scene_manifest_path = post_root / "scenes" / condition / "manifest.json"
        scene_manifest = _load_json(scene_manifest_path)
        if (scene_manifest["condition"] != condition or
                scene_manifest["post_commit"] != eval_commit or
                scene_manifest["training_commit"] != TRAIN_COMMIT or
                len(scene_manifest["scenes"]) != 20):
            raise RuntimeError(f"{condition} 场景清单与固定补评版本不符")
        references = scene_manifest["scenes"]
        _hash_into(hashes, scene_manifest_path)
        for scene_id, reference in references.items():
            if (scene_id not in plan["test_scenes"] or
                    reference["base_sha256"] != plan["test_scenes"][scene_id]["sha256"] or
                    sha256_file(reference["path"]) != reference["sha256"]):
                raise RuntimeError(f"{condition} 场景或原场景校验失败：{scene_id}")
            _hash_into(hashes, reference["path"])
            _hash_into(hashes, reference["base_path"])
        for job in plan["jobs"]:
            run_id = job["run_id"]
            task = post_root / "conditions" / condition / run_id
            status_path, manifest_path = task / "status.json", task / "manifest.json"
            status, manifest = _load_json(status_path), _load_json(manifest_path)
            if (status.get("stage") != "completed" or
                    manifest["post_commit"] != eval_commit or
                    manifest["training_commit"] != TRAIN_COMMIT or
                    manifest["actor_sha256"] != final[run_id]["actor_sha256"] or
                    manifest["weight_training_steps"] != 400000):
                raise RuntimeError(f"{condition}/{run_id} 结果状态、权重或版本不符")
            result_path = task / "evaluation" / "final_test_400000.json"
            if Path(status["result_path"]).resolve() != result_path.resolve():
                raise RuntimeError("补评状态指向了其他结果文件")
            result = verify_cached_post_result(result_path, references)
            changed = _sample_rows(result)
            differences = paired_means(originals[run_id], changed)
            absolute = {
                "original_" + field: statistics.mean(float(row[field]) for row in originals[run_id].values())
                for field in FIELDS}
            absolute.update({
                "condition_" + field: statistics.mean(float(row[field]) for row in changed.values())
                for field in FIELDS})
            by_seed.append({"condition": condition, "model": job["model"],
                            "formula": job["formula"], "train_seed": job["seed"],
                            "run_id": run_id, "paired_scenes": 20,
                            "action_seeds_per_scene": 3, **absolute, **differences})
            for path in (status_path, manifest_path, result_path,
                         result["arrays_path"], result["trace_path"], final[run_id]["actor"]):
                _hash_into(hashes, path)
    return by_seed, summarize_three_seeds(by_seed)


def _read_environment_pair(post_root, plan, eval_commit, hashes):
    source_path = post_root / "environment_comparison" / "source_manifest.json"
    source = _load_json(source_path)
    if (source["post_commit"] != eval_commit or
            source["training_commit"] != TRAIN_COMMIT or
            source["old_environment_commit"] != OLD_ENV_COMMIT or
            source["old_actor_training_steps"] != "unknown" or
            len(source["scenes"]) != 20 or
            sha256_file(source["old_actor"]) != source["old_actor_sha256"] or
            sha256_file(source["old_environment_source"]) != source["old_environment_sha256"]):
        raise RuntimeError("历史权重或环境源码清单不完整")
    _hash_into(hashes, source_path)
    _hash_into(hashes, source["old_actor"])
    _hash_into(hashes, source["old_environment_source"])
    for scene_id, reference in source["scenes"].items():
        if reference != plan["test_scenes"][scene_id]:
            raise RuntimeError("历史权重对照没有使用相同固定场景")
        _hash_into(hashes, reference["path"])
    results = {}
    for label in ENV_NAMES:
        task = post_root / "environment_comparison" / label
        status_path, manifest_path = task / "status.json", task / "manifest.json"
        status, manifest = _load_json(status_path), _load_json(manifest_path)
        if (status.get("stage") != "completed" or
                manifest["post_commit"] != eval_commit or
                manifest["training_commit"] != TRAIN_COMMIT or
                manifest["actor_sha256"] != source["old_actor_sha256"] or
                manifest["weight_training_steps"] != "unknown"):
            raise RuntimeError(f"历史权重 {label} 的版本或状态不一致")
        result_path = task / "evaluation" / "final_test_unknown.json"
        if Path(status["result_path"]).resolve() != result_path.resolve():
            raise RuntimeError("历史权重状态指向了其他结果文件")
        result = verify_cached_post_result(result_path, source["scenes"],
                                           expected_step="unknown")
        results[label] = _sample_rows(result)
        for path in (status_path, manifest_path, result_path,
                     result["arrays_path"], result["trace_path"]):
            _hash_into(hashes, path)
    old, corrected = results["before_c4963ba"], results["corrected_r2"]
    differences = paired_means(old, corrected)
    per_trial = [{"scene_id": scene, "action_seed": action,
                  **{field: float(corrected[(scene, action)][field])
                     - float(old[(scene, action)][field]) for field in FIELDS}}
                 for scene, action in sorted(old)]
    return {"weight_training_steps": "unknown", "independent_training_repeats": 1,
            "paired_scenes": 20, "action_seeds_per_scene": 3,
            "difference_definition": "corrected_environment_minus_old_environment",
            "mean_differences": differences,
            "std_across_60_paired_evaluations": {
                field: statistics.stdev(row[field] for row in per_trial) for field in FIELDS},
            "old_actor_sha256": source["old_actor_sha256"],
            "old_environment_sha256": source["old_environment_sha256"],
            "paired_trials": per_trial}


def _read_timing(post_root, final, eval_commit, hashes):
    timing_dir = post_root / "timing"
    index_path = timing_dir / "result_index.json"
    index = _load_json(index_path)
    manifest_path = Path(index["manifest"])
    manifest = _load_json(manifest_path)
    if (sha256_file(manifest_path) != index["manifest_sha256"] or
            manifest["training_commit"] != TRAIN_COMMIT or
            manifest["post_commit"] != eval_commit or
            not manifest["runtime"].get("torch_cuda") or
            "synchronized" not in manifest["scope_forward"] or
            "synchronized" not in manifest["scope_take_action"] or
            sha256_file(manifest["state_path"]) != manifest["state_sha256"] or
            set(index["runs"]) != EXPECTED_RUN_IDS):
        raise RuntimeError("统一推理计时清单或状态数组不完整")
    for path in (index_path, manifest_path, manifest["state_path"]):
        _hash_into(hashes, path)
    rows = []
    for run_id, reference in sorted(index["runs"].items()):
        path = Path(reference["path"])
        if sha256_file(path) != reference["sha256"]:
            raise RuntimeError(f"计时结果校验值不符：{run_id}")
        result = _load_json(path)
        config = result["config"]
        if (config["run_id"] != run_id or config["post_commit"] != eval_commit or
                config["actor_sha256"] != final[run_id]["actor_sha256"] or
                config["state_sha256"] != manifest["state_sha256"] or
                config["warmup"] != manifest["warmup_per_scope"] or
                config["repeats"] != manifest["repeats_per_scope"]):
            raise RuntimeError(f"计时结果的权重、状态或重复次数不一致：{run_id}")
        job = final[run_id]["job"]
        if not isinstance(result["parameters"], int) or result["parameters"] <= 0:
            raise RuntimeError(f"Actor参数量无效：{run_id}")
        entry = {"run_id": run_id, "model": job["model"], "formula": job["formula"],
                 "train_seed": job["seed"], "actor_parameters": result["parameters"],
                 "gpu_before": result["gpu_before"], "gpu_after": result["gpu_after"]}
        for scope in ("forward_device_tensor", "take_action_host_to_action"):
            distribution = result["timing"][scope]
            for key in ("mean_ms", "median_ms", "p90_ms", "sample_std_ms", "min_ms", "max_ms"):
                value = float(distribution[key])
                if not math.isfinite(value) or value < 0:
                    raise RuntimeError("计时分布含非有限或负值")
                entry[scope + "_" + key] = value
        rows.append(entry)
        _hash_into(hashes, path)
        _hash_into(hashes, final[run_id]["actor"])
    return rows


def _write_csv(path, rows):
    if not rows:
        raise RuntimeError("没有可保存的数值行")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _show(value, scale=1):
    return f"{value * scale:.2f}"


def _condition_table(rows):
    lines = ["| 条件 | 网络 | 学习公式 | SI差/百分点 | SGM差/百分点 | 吞吐差/Gbps | 功率差/W | 超功率率差/百分点 | 完整分配率差/百分点 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        def pair(field, scale=1):
            return (_show(row[field + "_difference_mean"], scale) + " ± " +
                    _show(row[field + "_difference_std_across_train_seeds"], scale))
        lines.append("| " + " | ".join((
            CONDITION_NAMES[row["condition"]], MODEL_NAMES[row["model"]],
            FORMULA_NAMES[row["formula"]], pair("si_mean", 100), pair("sgm_mean", 100),
            pair("throughput_bps", 1e-9), pair("total_power_w"),
            pair("power_exceeded", 100), pair("completed", 100),
        )) + " |")
    return "\n".join(lines)


def _timing_table(rows):
    lines = ["| 网络 | 公式 | 种子 | Actor参数量 | 纯前向均值/中位/P90 ms | 完整决策均值/中位/P90 ms | 计时前/后GPU利用率 | GPU进程快照 |",
             "| --- | --- | ---: | ---: | --- | --- | --- | --- |"]
    for row in rows:
        def scope(name):
            return "/".join(_show(row[name + "_" + field])
                            for field in ("mean_ms", "median_ms", "p90_ms"))
        gpu = str(row["gpu_before"].get("utilization_percent", "?")) + "/" + str(
            row["gpu_after"].get("utilization_percent", "?"))
        query = str(row["gpu_before"].get("compute_process_query", "?"))
        query += " → " + str(row["gpu_after"].get("compute_process_query", "?"))
        query = query.replace("\n", "; ").replace("|", "/")
        lines.append("| " + " | ".join((MODEL_NAMES[row["model"]],
                                         FORMULA_NAMES[row["formula"]],
                                         str(row["train_seed"]), str(row["actor_parameters"]),
                                         scope("forward_device_tensor"),
                                         scope("take_action_host_to_action"), gpu,
                                         query)) + " |")
    return "\n".join(lines)


def generate_summary(archive_root, frozen_root, output_directory=None):
    archive_root, frozen_root = Path(archive_root).resolve(), Path(frozen_root).resolve()
    # This gate checks every formal endpoint, raw final evaluation and Actor/critic
    # checkpoint before any new output directory is created.
    provenance = source_guard(frozen_root)
    exp_dir, plan, final = require_completed(archive_root, frozen_root)
    post_root = exp_dir / "post_training_checks"
    runtime_path = post_root / "runtime.json"
    runtime = _load_json(runtime_path)
    eval_commit = runtime["post_commit"]
    if (runtime["training_commit"] != TRAIN_COMMIT or
            not runtime["runtime"].get("torch_cuda")):
        raise RuntimeError("补充评价生成版本与训练冻结提交不一致")
    hashes = {}
    for path in (exp_dir / "suite_plan.json", exp_dir / "suite_status.json", runtime_path):
        _hash_into(hashes, path)
    by_seed, condition_summary = _read_condition_results(
        exp_dir, post_root, plan, final, eval_commit, hashes)
    environment = _read_environment_pair(post_root, plan, eval_commit, hashes)
    timing = _read_timing(post_root, final, eval_commit, hashes)
    output = (Path(output_directory) if output_directory else post_root / "summary").resolve()
    manifest = {"created_at": utc_now(), "training_commit": TRAIN_COMMIT,
                "evaluation_generation_commit": eval_commit,
                "report_code_commit": provenance["post_commit"],
                "report_code_root": provenance["post_source_root"],
                "evaluation_matched_source_sha256": runtime["source_hashes"],
                "report_matched_source_sha256": provenance["matched_file_sha256"],
                "input_sha256": hashes, "formal_training_runs": 18,
                "historical_weight_training_steps": "unknown"}
    existing = output / "summary_manifest.json"
    if output.exists() and any(output.iterdir()) and not existing.exists():
        raise RuntimeError("汇总目录已有无清单文件，拒绝覆盖未知结果")
    if existing.exists():
        previous = _load_json(existing)
        if (previous["training_commit"] != manifest["training_commit"] or
                previous["evaluation_generation_commit"] != eval_commit or
                previous["report_code_commit"] != manifest["report_code_commit"] or
                previous["input_sha256"] != hashes):
            raise RuntimeError("汇总目录已有不同输入或代码版本，不覆盖")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "conditions_by_seed.csv", by_seed)
    _write_csv(output / "conditions_summary.csv", condition_summary)
    _write_csv(output / "historical_environment_trials.csv", environment["paired_trials"])
    _write_csv(output / "timing_by_run.csv", timing)
    atomic_json(output / "summary_data.json", {
        "conditions_by_seed": by_seed, "conditions_summary": condition_summary,
        "historical_environment": environment, "timing_by_run": timing})
    lines = ["# 十八组正式实验的固定补充评价汇总", "",
             f"训练固定提交：`{TRAIN_COMMIT}`；补充评价生成提交：`{eval_commit}`；"
             f"本报告代码提交：`{provenance['post_commit']}`。", "",
             "## 三种预定条件与原条件的同场景配对差", "",
             "每个模型与学习公式先在同一训练种子的20个场景、3个动作种子上计算“变化条件减原条件”，"
             "再对三个独立训练种子给均值与样本标准差。SI 是全体目标波束终态需求满足率，SGM 为供需匹配程度。", "",
             _condition_table(condition_summary), "",
             "## 历史 Transformer 权重的环境计算对照", "",
             "历史权重训练交互步数为 unknown；这里仅一份权重的60个配对评价，"
             "不是三个独立训练种子，也不参与三网络排名。差值定义为修正后环境减修正前环境。", "",
             "| SI差/百分点 | SGM差/百分点 | 吞吐差/Gbps | 功率差/W | 超功率率差/百分点 | 完整分配率差/百分点 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines.append("| " + " | ".join((
        _show(environment["mean_differences"]["si_mean"], 100),
        _show(environment["mean_differences"]["sgm_mean"], 100),
        _show(environment["mean_differences"]["throughput_bps"], 1e-9),
        _show(environment["mean_differences"]["total_power_w"]),
        _show(environment["mean_differences"]["power_exceeded"], 100),
        _show(environment["mean_differences"]["completed"], 100),
    )) + " |")
    lines += ["", "60项逐场景/动作差值保存在 `historical_environment_trials.csv`；"
              "它们是20个场景各重复3个动作种子，并非60个独立场景。"
              "跨样本波动在 `summary_data.json`，不能当作训练种子间波动。", "",
              "## 统一 CUDA 单次决策计时", "",
              _timing_table(timing), "",
              "纯前向从已驻留GPU的单状态张量到 Actor 输出；完整决策从主机状态到动作返回。"
              "两项均在边界同步CUDA。计时前后的GPU快照与其他计算进程列表保存在 `summary_data.json`；"
              "快照只能提示可见争用，不能证明整个计时窗口无其他负载。", "",
              "所有输入文件、旧环境源码、权重和场景的 SHA-256 见 `summary_manifest.json`。"
              "数值直接来自保存的原始结果，没有重新评价模型。", ""]
    report = output / "中文补充评价汇总.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    atomic_json(output / "summary_manifest.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description="仅汇总已完成的固定补充评价和CUDA计时")
    parser.add_argument("--archive-root", default="D:/paper")
    parser.add_argument("--frozen-root", default="D:/paper_worktrees/frozen-sac-20260924-r2")
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    print(generate_summary(args.archive_root, args.frozen_root, args.output_directory))


if __name__ == "__main__":
    main()
