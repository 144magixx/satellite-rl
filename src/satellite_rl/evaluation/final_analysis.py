"""Read-only timing and beam-group analysis after all 18 SAC runs finish."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np

from satellite_rl.evaluation.experiment_curves import (
    build_comparison_curves, write_comparison_curves,
)
from satellite_rl.evaluation.experiment_report import (
    FORMULA_NAMES, MODEL_NAMES, _load_run, _read_json,
)
from satellite_rl.training.experiment_runner import sha256_file
from satellite_rl.training.run_monitor import atomic_json, utc_now


DEMAND_BANDS = ("低需求（<150 Mbps）", "中需求（150–250 Mbps）", "高需求（≥250 Mbps）")
PHASES = ("前段", "中段", "后段")
AXIS_NAMES = {"demand": "需求", "phase": "处理阶段"}
METRICS = ("final_si_mean", "unserved_rate", "requested_power_mean_w",
           "requested_slots_mean", "actual_slots_mean", "zero_slot_rate", "clipped_rate")
TIMING_METRICS = ("calendar_seconds_including_retries", "recorded_active_seconds",
                  "effective_env_steps_per_active_second", "effective_updates_per_active_second",
                  "post_warmup_update_steps_per_second",
                  "periodic_validation_block_seconds_approx", "final_assessment_block_seconds")


def _seconds(start, end):
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def _read_events(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_updates(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _require_finished(directory):
    plan = _read_json(directory / "suite_plan.json")
    suite_status = _read_json(directory / "suite_status.json")
    if (len(plan["jobs"]) != 18 or
            any(job["state"] != "completed" for job in plan["jobs"]) or
            suite_status.get("counts", {}).get("completed") != 18 or
            not suite_status.get("all_completed")):
        raise RuntimeError("18 组训练与考核尚未全部完成；拒绝生成正式补充报告")
    if len(plan["test_scenes"]) != 100 or len(plan["action_seeds"]) != 3:
        raise RuntimeError("固定考核场景或动作种子与批准方案不一致")
    return plan, suite_status


def timing_for_run(run, plan):
    events = _read_events(run["run_dir"] / "events.jsonl")
    first = next((event for event in events if event["kind"] == "started"), None)
    final_checkpoint = next((event for event in reversed(events)
                             if event["kind"] == "checkpoint" and event["name"] == "final"), None)
    completed = next((event for event in reversed(events) if event["kind"] == "completed"), None)
    if not first or not final_checkpoint or not completed:
        raise RuntimeError(f"{run['job']['run_id']} 缺少开始、最终断点或完成事件")
    if final_checkpoint["env_step"] != plan["steps"]:
        raise RuntimeError(f"{run['job']['run_id']} 最终断点未达到交互预算")
    checkpoint_path = Path(final_checkpoint["path"])
    if not checkpoint_path.exists() or sha256_file(checkpoint_path) != final_checkpoint["sha256"]:
        raise RuntimeError(f"{run['job']['run_id']} 最终权重与事件校验值不符")
    actor_path = checkpoint_path.with_name("final_actor.pt")
    if not actor_path.exists():
        raise RuntimeError(f"{run['job']['run_id']} 缺少预算终点 Actor 权重")
    updates = _read_updates(run["run_dir"] / "updates.csv")
    if not updates:
        raise RuntimeError(f"{run['job']['run_id']} 缺少真实参数更新记录")
    first_update, last_update = updates[0], updates[-1]
    update_delta = int(last_update["update_step"]) - int(first_update["update_step"])
    update_duration = float(last_update["wall_seconds"]) - float(first_update["wall_seconds"])
    if update_delta <= 0 or update_duration <= 0:
        raise RuntimeError(f"{run['job']['run_id']} 更新记录无法计算有效吞吐")
    # Each validation event follows its boundary checkpoint; the interval includes
    # status I/O and validation, so it is an approximate block, not pure inference.
    validation_intervals = {}
    for index, event in enumerate(events):
        if event["kind"] != "validation":
            continue
        checkpoint = next((prior for prior in reversed(events[:index])
                           if prior["kind"] == "checkpoint" and prior.get("name") == "latest"
                           and prior.get("env_step") == event.get("env_step")), None)
        if checkpoint:
            duration = _seconds(checkpoint["at"], event["at"])
            if duration < 0:
                raise RuntimeError("验证事件时间早于断点")
            validation_intervals[int(event["env_step"])] = duration
    first_at = first["at"]
    completed_at = run["status"].get("completed_at", completed["at"])
    elapsed_recorded = float(run["status"].get("elapsed_seconds", 0))
    if elapsed_recorded <= 0:
        raise RuntimeError(f"{run['job']['run_id']} 缺少累计进程计时")
    final_block = _seconds(final_checkpoint["at"], completed["at"])
    calendar = _seconds(first_at, completed_at)
    if final_block < 0 or calendar <= 0:
        raise RuntimeError("运行事件时间顺序无效")
    return {
        "run_id": run["job"]["run_id"], "model": run["job"]["model"],
        "formula": run["job"]["formula"], "train_seed": run["job"]["seed"],
        "first_started_at": first_at, "completed_at": completed_at,
        "calendar_seconds_including_retries": calendar,
        "recorded_active_seconds": elapsed_recorded,
        "attempts": int(run["job"]["attempts"]),
        "resume_events": sum(event["kind"] == "resumed" for event in events),
        "env_steps": int(run["status"]["env_step"]),
        "update_steps": int(run["status"]["update_step"]),
        "effective_env_steps_per_active_second": plan["steps"] / elapsed_recorded,
        "effective_updates_per_active_second": int(run["status"]["update_step"]) / elapsed_recorded,
        "post_warmup_update_steps_per_second": update_delta / update_duration,
        "post_warmup_update_span_seconds": update_duration,
        "periodic_validation_count_with_timing": len(validation_intervals),
        "periodic_validation_block_seconds_approx": sum(validation_intervals.values()),
        "final_assessment_block_seconds": final_block,
        "final_checkpoint_at": final_checkpoint["at"],
        "final_checkpoint_path": str(checkpoint_path),
        "final_checkpoint_sha256": final_checkpoint["sha256"],
        "final_actor_path": str(actor_path),
        "final_actor_sha256": sha256_file(actor_path),
    }


def demand_band(required_bps):
    if required_bps < 150e6:
        return DEMAND_BANDS[0]
    if required_bps < 250e6:
        return DEMAND_BANDS[1]
    return DEMAND_BANDS[2]


def phase(index, total):
    return PHASES[min(2, index * 3 // total)]


def _new_accumulator():
    return {"target_beams": 0, "attempted_beams": 0, "unserved_beams": 0,
            "si_sum": 0.0, "requested_power_sum_w": 0.0,
            "requested_slots_sum": 0, "actual_slots_sum": 0,
            "zero_slot_count": 0, "clipped_count": 0,
            "requested_power_w_counts": {}, "requested_slots_counts": {},
            "actual_slots_counts": {}, "requested_start_counts": {}}


def _finish_accumulator(raw):
    targets = raw["target_beams"]
    attempted = raw["attempted_beams"]
    if targets <= 0:
        raise RuntimeError("分组没有目标波束")
    return {**raw, "final_si_mean": raw["si_sum"] / targets,
            "unserved_rate": raw["unserved_beams"] / targets,
            "requested_power_mean_w": raw["requested_power_sum_w"] / attempted if attempted else None,
            "requested_slots_mean": raw["requested_slots_sum"] / attempted if attempted else None,
            "actual_slots_mean": raw["actual_slots_sum"] / attempted if attempted else None,
            "zero_slot_rate": raw["zero_slot_count"] / attempted if attempted else None,
            "clipped_rate": raw["clipped_count"] / attempted if attempted else None}


def _accumulate_trial(accumulators, required, satisfaction, steps):
    total = len(required)
    if len(satisfaction) != total or len(steps) > total:
        raise RuntimeError("终态数组或逐步日志的波束数量不一致")
    if [int(step["beam_index"]) for step in steps] != list(range(len(steps))):
        raise RuntimeError("逐步日志的波束编号不连续")
    for index in range(total):
        action = steps[index] if index < len(steps) else None
        if action is None and abs(float(satisfaction[index])) > 1e-10:
            raise RuntimeError("未处理波束的终态满意度不是零")
        for axis, group in (("demand", demand_band(float(required[index]))),
                            ("phase", phase(index, total))):
            raw = accumulators[(axis, group)]
            raw["target_beams"] += 1
            raw["si_sum"] += float(satisfaction[index])
            if action is None:
                raw["unserved_beams"] += 1
                continue
            requested, actual = action["requested"], action["actual"]
            requested_slots, actual_slots = int(requested[2]), int(actual[2])
            if not 0 <= actual_slots <= requested_slots:
                raise RuntimeError("实际槽数不在请求范围内")
            raw["attempted_beams"] += 1
            raw["requested_power_sum_w"] += 10 ** (float(requested[3]) / 10)
            raw["requested_slots_sum"] += requested_slots
            raw["actual_slots_sum"] += actual_slots
            raw["zero_slot_count"] += int(actual_slots == 0)
            raw["clipped_count"] += int(actual_slots < requested_slots)
            for field, value in (("requested_power_w_counts", round(10 ** (float(requested[3]) / 10), 3)),
                                 ("requested_slots_counts", requested_slots),
                                 ("actual_slots_counts", actual_slots),
                                 ("requested_start_counts", int(requested[1]))):
                counts = raw[field]
                key = str(value)
                counts[key] = counts.get(key, 0) + 1


def behavior_for_run(run, plan):
    result = run["result"]
    traces = defaultdict(list)
    with gzip.open(result["trace_path"], "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("action_mode") == "sample":
                traces[(str(item["scene_id"]), str(item["action_seed"]))].append(item)
    if traces.keys() != run["by_key"].keys():
        raise RuntimeError(f"{run['job']['run_id']} 逐步日志与最终考核行不匹配")
    raw = defaultdict(_new_accumulator)
    with np.load(result["arrays_path"]) as arrays:
        for (scene_id, action_seed), row in run["by_key"].items():
            prefix = f"scene_{scene_id}_action_{action_seed}_"
            required = np.asarray(arrays[prefix + "required_bps"], dtype=float).reshape(-1)
            rates = np.asarray(arrays[prefix + "rates_bps"], dtype=float).reshape(-1)
            satisfaction = np.asarray(arrays[prefix + "satisfaction"], dtype=float).reshape(-1)
            if (len(required) != len(rates) or len(required) != len(satisfaction)
                    or not np.isfinite(required).all() or not np.isfinite(rates).all()
                    or not np.isfinite(satisfaction).all() or np.any(required <= 0)
                    or np.any(rates < 0)):
                raise RuntimeError("终态速率、需求或满意度数组无效")
            if not np.allclose(satisfaction, np.minimum(rates / required, 1),
                               rtol=1e-11, atol=1e-12):
                raise RuntimeError("终态 SI 与逐束速率/需求独立计算不一致")
            with np.load(plan["test_scenes"][scene_id]["path"]) as scene:
                np.testing.assert_allclose(required, np.asarray(scene["rate"]).reshape(-1),
                                           rtol=0, atol=0)
            if not np.isclose(float(satisfaction.sum()), float(row["si_sum"]), atol=1e-7):
                raise RuntimeError("终态数组与每场景满意度记录不一致")
            if not np.isclose(float(rates.sum()), float(row["throughput_bps"]),
                              rtol=1e-11, atol=1e-6):
                raise RuntimeError("逐束速率之和与总吞吐量记录不一致")
            steps = traces[(scene_id, action_seed)]
            if (not np.isclose(float(satisfaction.mean()), float(row["si_mean"]), atol=1e-9)
                    or len(steps) != int(row["served_beams"])):
                raise RuntimeError("逐步日志、已处理束数与终态满意度不一致")
            if np.any(rates[len(steps):] != 0):
                raise RuntimeError("未处理波束的终态速率不是零")
            _accumulate_trial(raw, required, satisfaction, steps)
    return [{"run_id": run["job"]["run_id"], "model": run["job"]["model"],
             "formula": run["job"]["formula"], "train_seed": run["job"]["seed"],
             "axis": axis, "group": group, **_finish_accumulator(values)}
            for (axis, group), values in sorted(raw.items())]


def _aggregate_seeds(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["model"], row["formula"], row["axis"], row["group"])][row["train_seed"]] = row
    output = []
    for (model, formula, axis, group), by_seed in sorted(grouped.items()):
        if len(by_seed) != 3:
            raise RuntimeError(f"{model}/{formula}/{axis}/{group} 不足三个独立训练种子")
        item = {"model": model, "formula": formula, "axis": axis, "group": group,
                "train_seeds": sorted(by_seed)}
        for field in METRICS:
            values = [by_seed[seed][field] for seed in sorted(by_seed)]
            if any(value is None for value in values):
                item[field + "_mean"] = None
                item[field + "_std_across_seeds"] = None
            else:
                item[field + "_mean"] = statistics.mean(values)
                item[field + "_std_across_seeds"] = statistics.stdev(values)
        output.append(item)
    return output


def _aggregate_timing(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["model"], row["formula"])][row["train_seed"]] = row
    result = []
    for (model, formula), by_seed in sorted(grouped.items()):
        if len(by_seed) != 3:
            raise RuntimeError(f"{model}/{formula} 的耗时记录不足三个训练种子")
        item = {"model": model, "formula": formula, "train_seeds": sorted(by_seed),
                "total_retries": sum(by_seed[seed]["attempts"] - 1 for seed in by_seed)}
        for field in TIMING_METRICS:
            values = [by_seed[seed][field] for seed in sorted(by_seed)]
            item[field + "_mean"] = statistics.mean(values)
            item[field + "_std_across_seeds"] = statistics.stdev(values)
        result.append(item)
    return result


def _formula_differences(rows):
    indexed = {(row["model"], row["formula"], row["axis"], row["group"], row["train_seed"]): row
               for row in rows}
    output = []
    groups = {(row["model"], row["axis"], row["group"]) for row in rows}
    for model, axis, group in sorted(groups):
        seeds = sorted({row["train_seed"] for row in rows
                        if row["model"] == model and row["axis"] == axis and row["group"] == group})
        if len(seeds) != 3:
            raise RuntimeError("公式差值不足三个配对训练种子")
        item = {"model": model, "axis": axis, "group": group, "train_seeds": seeds,
                "difference_definition": "corrected_minus_legacy"}
        for field in METRICS:
            differences = []
            for seed in seeds:
                old = indexed.get((model, "legacy", axis, group, seed))
                new = indexed.get((model, "corrected", axis, group, seed))
                if old is None or new is None:
                    raise RuntimeError("公式差值缺少配对运行")
                differences.append(None if old[field] is None or new[field] is None
                                   else new[field] - old[field])
            item[field + "_by_train_seed"] = differences
            item[field + "_mean"] = (statistics.mean(differences)
                                     if all(value is not None for value in differences) else None)
            item[field + "_std_across_seeds"] = (statistics.stdev(differences)
                                                 if all(value is not None for value in differences) else None)
        output.append(item)
    return output


def _display(value, scale=1.0):
    if value is None:
        return "—"
    return f"{value * scale:.2f}"


def _timing_table(rows):
    lines = ["| 网络 | 公式 | 训练种子 | 实际开始 UTC | 完成 UTC | 重试 | 日历小时 | 累计进程小时 | 后预热更新/秒 | 最终考核合并分钟 |",
             "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append("| " + " | ".join((
            MODEL_NAMES[row["model"]], FORMULA_NAMES[row["formula"]], str(row["train_seed"]),
            row["first_started_at"], row["completed_at"], str(row["attempts"] - 1),
            _display(row["calendar_seconds_including_retries"], 1 / 3600),
            _display(row["recorded_active_seconds"], 1 / 3600),
            _display(row["post_warmup_update_steps_per_second"]),
            _display(row["final_assessment_block_seconds"], 1 / 60),
        )) + " |")
    return "\n".join(lines)


def _behavior_table(rows):
    lines = ["| 网络 | 公式 | 分组 | 档位/阶段 | 终态 SI / % | 未处理 / % | 请求功率 / W | 请求槽数 | 实际槽数 | 零槽 / % | 裁剪 / % |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        def stat(field, scale=1):
            mean = row[field + "_mean"]
            std = row[field + "_std_across_seeds"]
            return "—" if mean is None else f"{_display(mean, scale)} ± {_display(std, scale)}"
        lines.append("| " + " | ".join((
            MODEL_NAMES[row["model"]], FORMULA_NAMES[row["formula"]],
            AXIS_NAMES[row["axis"]], row["group"],
            stat("final_si_mean", 100), stat("unserved_rate", 100),
            stat("requested_power_mean_w"), stat("requested_slots_mean"),
            stat("actual_slots_mean"), stat("zero_slot_rate", 100),
            stat("clipped_rate", 100),
        )) + " |")
    return "\n".join(lines)


def _difference_table(rows):
    lines = ["| 网络 | 分组 | 档位/阶段 | SI 差 / 百分点 | 未处理率差 / 百分点 | 请求功率差 / W | 实际槽数差 | 裁剪率差 / 百分点 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        def stat(field, scale=1):
            mean = row[field + "_mean"]
            std = row[field + "_std_across_seeds"]
            return "—" if mean is None else f"{_display(mean, scale)} ± {_display(std, scale)}"
        lines.append("| " + " | ".join((
            MODEL_NAMES[row["model"]], AXIS_NAMES[row["axis"]], row["group"],
            stat("final_si_mean", 100), stat("unserved_rate", 100),
            stat("requested_power_mean_w"), stat("actual_slots_mean"),
            stat("clipped_rate", 100),
        )) + " |")
    return "\n".join(lines)


def _write_csv(path, rows):
    if not rows:
        raise RuntimeError("没有可写入的原始数字行")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _distribution_rows(rows):
    output = []
    fields = ("requested_power_w_counts", "requested_slots_counts",
              "actual_slots_counts", "requested_start_counts")
    for row in rows:
        for field in fields:
            for value, count in sorted(row[field].items(), key=lambda pair: float(pair[0])):
                output.append({"run_id": row["run_id"], "model": row["model"],
                               "formula": row["formula"], "train_seed": row["train_seed"],
                               "axis": row["axis"], "group": row["group"],
                               "dimension": field.removesuffix("_counts"),
                               "value": value, "count": count,
                               "attempted_beams": row["attempted_beams"]})
    return output


def _analysis_commit():
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if status:
        raise RuntimeError("后处理源码有未提交改动，不能写正式分析清单")
    return revision


def _input_hashes(directory, plan, timing_rows):
    paths = [directory / "suite_plan.json", directory / "suite_status.json",
             Path(plan["source"]["cities_data"]["path"])]
    if (directory / "suite_events.jsonl").exists():
        paths.append(directory / "suite_events.jsonl")
    paths += [Path(reference["path"]) for group in ("validation_scenes", "test_scenes")
              for reference in plan[group].values()]
    for row in timing_rows:
        run_dir = directory / row["run_id"]
        paths += [run_dir / name for name in ("manifest.json", "status.json", "events.jsonl",
                                             "updates.csv", "episodes.csv", "evaluations.csv")]
        paths += [run_dir / "evaluation" / f"final_test_{plan['steps']}{suffix}"
                  for suffix in (".json", ".npz", "_steps.jsonl.gz")]
        paths += [Path(row["final_checkpoint_path"]), Path(row["final_actor_path"])]
    return {str(path): sha256_file(path) for path in sorted(set(paths))}


def generate_final_analysis(suite_directory, output_directory=None):
    directory = Path(suite_directory).resolve()
    plan, _ = _require_finished(directory)
    analysis_commit = _analysis_commit()
    runs = [_load_run(directory, job, plan) for job in plan["jobs"]]
    curves = build_comparison_curves(
        runs, validation_interval=plan["validation_interval"], budget_steps=plan["steps"])
    timing = [timing_for_run(run, plan) for run in runs]
    timing_summary = _aggregate_timing(timing)
    behavior_by_seed = [item for run in runs for item in behavior_for_run(run, plan)]
    behavior_summary = _aggregate_seeds(behavior_by_seed)
    behavior_differences = _formula_differences(behavior_by_seed)
    behavior_distributions = _distribution_rows(behavior_by_seed)
    input_hashes = _input_hashes(directory, plan, timing)
    output = (Path(output_directory) if output_directory else directory / "analysis").resolve()
    manifest = {"generated_at": utc_now(), "analysis_commit": analysis_commit,
                "training_frozen_commit": plan["source"]["git_commit"],
                "suite_directory": str(directory), "input_sha256": input_hashes,
                "timing_limits": [
                    "累计进程秒数跨断点接续，但异常退出后未写入断点的尾段可能缺失。",
                    "周期验证块由同一步最新断点事件到验证完成事件估算，含状态写入，非纯推理时间。",
                    "最终考核块由 final 断点到 completed 事件估算，合并终态验证、最终考核、最佳模型考核及文件写入，不能可靠拆分。",
                    "三路作业共享一张 GPU；有效步/秒和更新/秒不能解释为独占网络吞吐。",
                    "本分析没有测量独占推理耗时。",
                ]}
    if output.exists() and (output / "analysis_manifest.json").exists():
        previous = _read_json(output / "analysis_manifest.json")
        if (previous["analysis_commit"] != analysis_commit or
                previous["input_sha256"] != input_hashes):
            raise RuntimeError("analysis/ 已存在不同版本或输入的结果，拒绝覆盖")
    output.mkdir(parents=True, exist_ok=True)
    curve_paths = write_comparison_curves(curves, output)
    _write_csv(output / "timing_by_run.csv", timing)
    _write_csv(output / "timing_summary.csv", timing_summary)
    _write_csv(output / "behavior_by_seed.csv", behavior_by_seed)
    _write_csv(output / "behavior_summary.csv", behavior_summary)
    _write_csv(output / "behavior_formula_differences.csv", behavior_differences)
    _write_csv(output / "behavior_distributions_by_seed.csv", behavior_distributions)
    atomic_json(output / "analysis_data.json", {
        "timing_by_run": timing, "timing_summary": timing_summary,
        "behavior_by_seed": behavior_by_seed,
        "behavior_summary": behavior_summary,
        "behavior_formula_differences": behavior_differences,
        "behavior_distributions_by_seed": behavior_distributions,
        "comparison_curves": curve_paths})
    manifest["generated_outputs"] = [*curve_paths.values(),
                                      str(output / "analysis_data.json"),
                                      str(output / "timing_by_run.csv"),
                                      str(output / "timing_summary.csv"),
                                      str(output / "behavior_by_seed.csv"),
                                      str(output / "behavior_summary.csv"),
                                      str(output / "behavior_formula_differences.csv"),
                                      str(output / "behavior_distributions_by_seed.csv"),
                                      str(output / "中文补充分析.md")]
    lines = ["# 十八组 SAC 实验补充分析", "",
             f"冻结训练提交：`{plan['source']['git_commit']}`；后处理提交：`{analysis_commit}`。", "",
             "本报告只读取已完成的十八组原始记录。SI 为全体目标波束的终态需求满足率；未处理波束按零计。",
             "低/中/高需求分界固定为 150 与 250 Mbps。处理阶段按波束编号把 100 束约等分为前、中、后段。", "",
             "## 耗时与更新速度的解释边界", ""]
    lines.extend("- " + statement for statement in manifest["timing_limits"])
    lines += ["", _timing_table(timing), "",
              "每组有效环境步/秒、更新/秒和周期验证近似时长见 `timing_by_run.csv`；"
              "按网络与公式汇总的三训练种子均值和标准差见 `timing_summary.csv`。",
              "", "## 波束需求与处理阶段", "",
              "`behavior_by_seed.csv` 先按每个训练种子汇总相同考核场景的三个采样动作；"
              "`behavior_summary.csv` 再给三个训练种子的均值与样本标准差。"
              "终态 SI 和未处理率以全部目标波束为分母；请求功率、请求/实际槽数、零槽与裁剪率以实际执行动作数为分母。",
              "若某分组没有执行动作，动作指标留空，不把它当成零。", "",
              "每个训练种子在各需求档、各处理阶段的请求功率、请求槽数、实际槽数、频率起点完整计数见"
              " `behavior_distributions_by_seed.csv`；没有把不同训练种子的动作合并成一次长轨迹。", "",
              _behavior_table(behavior_summary), "",
              "## 同网络下的公式差值", "",
              "下表为同一网络、同一训练种子下修正公式减原公式，再汇总三个训练种子的均值和标准差。", "",
              _difference_table(behavior_differences), "",
              "这些是观察到的分配选择与终态结果；两者同时变化不能单独证明因果。", "",
              "## 三种网络的固定验证曲线", "",
              "同一验证轮次先汇总每个训练种子的固定场景与采样动作，再报告三个训练种子的均值和样本标准差。"
              "横轴是三次训练实际交互步数的均值，最小/最大实际步数保存在数字文件；没有插值或额外平滑。", "",
              "![三种网络的验证曲线](三种网络验证曲线.png)", "",
              "逐轮完整数字见 `三种网络验证曲线.json`。", "",
              "原始逐步日志、终态数组、固定场景及最终权重的文件校验值见 `analysis_manifest.json`。", ""]
    report_path = output / "中文补充分析.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    atomic_json(output / "analysis_manifest.json", manifest)
    return report_path


def main():
    parser = argparse.ArgumentParser(description="十八组 SAC 实验完成后的只读补充分析")
    parser.add_argument("suite_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    print(generate_final_analysis(args.suite_directory, args.output_directory))


if __name__ == "__main__":
    main()
