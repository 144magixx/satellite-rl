"""Typed, read-only S2-P19 saved-allocation sources for fixed-slot SCA."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .storage import digest, read_json, validate_id


GROUPS = ("model", "greedy", "rule", "random")
PARTITIONS = ("development", "validation", "test")
I0_VALIDATION_COUNTS = (50, 80, 100, 120, 160, 200)
I0_VALIDATION_LOADS = (0.75, 1.25)


def _body_sha(body):
    payload = json.dumps(body, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"S2-P19来源{name}不是有限实数")
    return float(value)


def _verify_beams(beams, metrics, count):
    if not isinstance(beams, list) or len(beams) != count or not 50 <= count <= 200:
        raise ValueError("S2-P19来源缺全体目标波束")
    capacities, demands, powers = [], [], []
    occupied = np.zeros((8, 100), dtype=np.bool_)
    for index, beam in enumerate(beams):
        if beam.get("beam_id") != index or beam.get("group") != index % 8:
            raise ValueError("S2-P19来源束身份或组号不一致")
        slots, freq = beam.get("slots"), beam.get("freq")
        if type(slots) is not int or not 0 <= slots <= 10 or type(freq) is not int or not 0 <= freq < 100:
            raise ValueError("S2-P19来源频率起点/长度无效")
        if slots and freq + slots > 100:
            raise ValueError("S2-P19来源频率块越界")
        if slots:
            if occupied[index % 8, freq:freq + slots].any():
                raise ValueError("S2-P19来源组内频率槽冲突")
            occupied[index % 8, freq:freq + slots] = True
        demand = _number(beam.get("required_bps"), "逐束需求")
        capacity = _number(beam.get("rate_bps"), "逐束容量")
        power = _number(beam.get("power_w"), "逐束功率")
        if demand <= 0 or capacity < -1e-6 or power < -1e-8:
            raise ValueError("S2-P19来源逐束服务/功率无效")
        if not slots and (abs(capacity) > 1e-6 or abs(power) > 1e-8):
            raise ValueError("S2-P19无槽束仍有容量或功率")
        capacities.append(max(0, capacity)); demands.append(demand); powers.append(max(0, power))
    if sum(powers) > 3000 + 1e-6:
        raise ValueError("S2-P19来源总功率超3000W")
    from satellite_rl.workbench.environment import _sgm
    si = float(np.minimum(np.asarray(capacities) / np.asarray(demands), 1).mean())
    sgm = float(_sgm(np.asarray(capacities), np.asarray(demands),
                     float(np.mean(demands))).mean())
    expected_si = metrics.get("final_si_mean", metrics.get("si"))
    expected_sgm = metrics.get("final_sgm_mean", metrics.get("sgm"))
    if (abs(si - _number(expected_si, "SI")) > 1e-7 or
            abs(sgm - _number(expected_sgm, "SGM")) > 1e-7):
        raise ValueError("S2-P19来源逐束复算SI/SGM与摘要不一致")
    if "power_w" in metrics and abs(sum(powers) - _number(metrics["power_w"], "总功率")) > 1e-5:
        raise ValueError("S2-P19来源逐束功率和与摘要不一致")
    return {"final_si_mean": si, "final_sgm_mean": sgm,
            "basic_met_count": int(sum(c >= 0.1 * d for c, d in zip(capacities, demands))),
            "power_w": float(sum(powers)), "allocated_slots": int(sum(b["slots"] for b in beams)),
            "complete": True, "overload": False}


def _validate_i0_validation_grid(dataset):
    """The six held-out mothers must each have the frozen 6×2 validation grid."""
    mothers = {}
    expected = {(n, load) for n in I0_VALIDATION_COUNTS for load in I0_VALIDATION_LOADS}
    for item in dataset["files"]:
        mother = item.get("mother_scene_id")
        source_sha = item.get("source_sha256")
        selected = item.get("selected_time_utc")
        key = (item.get("beam_count"), item.get("load_multiplier"))
        if (not isinstance(mother, str) or not mother or
                not isinstance(source_sha, str) or len(source_sha) != 64 or
                any(c not in "0123456789abcdef" for c in source_sha) or
                not isinstance(selected, str) or not selected or
                item.get("partition_role") != "validation" or
                key not in expected):
            raise ValueError("I0验证场景缺冻结母来源或N×负载身份")
        details = mothers.setdefault(mother, {"source_sha256": source_sha,
                                              "selected_time_utc": selected, "grid": set()})
        if (details["source_sha256"] != source_sha or
                details["selected_time_utc"] != selected or key in details["grid"]):
            raise ValueError("I0验证母场景来源或N×负载网格重复")
        details["grid"].add(key)
    if (len(mothers) != 6 or any(row["grid"] != expected for row in mothers.values()) or
            len({row["source_sha256"] for row in mothers.values()}) != 6 or
            len({row["selected_time_utc"] for row in mothers.values()}) != 6):
        raise ValueError("I0验证须为六母各6种束数×双负载完整网格")


def _case_row(reference, item):
    path = Path(item["path"]).resolve(strict=True)
    source = Path(reference["source_directory"]).resolve(strict=True)
    if not path.is_relative_to(source):
        raise ValueError("S2-P19初解case不在原任务目录")
    if digest(path) != item["sha256"]:
        raise ValueError("S2-P19初解case字节SHA变化")
    if reference["format"] == "s2p19_inline_validation_v1":
        envelope = read_json(path, {})
        body = envelope.get("body") or {}
        contract = body.get("contract") or {}
        if (envelope.get("sha256") != _body_sha(body) or
                contract.get("env_step") != reference["validation_step"] or
                contract.get("scene_index") != item["scene_index"] or
                contract.get("repeat_index") != item["repeat_index"] or
                contract.get("source_scene_sha256") != item["scene_sha256"] or
                contract.get("action_seed") != item["action_seed"] or
                contract.get("policy_sha256") != reference["policy_sha256"] or
                contract.get("config_hash") != reference["source_config_hash"]):
            raise ValueError("S2-P19 inline验证case合同不一致")
        original = body.get("result") or {}
        scene = original.get("scene") or {}
        if scene.get("source_sha256") != item["scene_sha256"]:
            raise ValueError("S2-P19 inline验证case场景SHA不一致")
        duration = original.get("initial_allocation_seconds")
        metrics = original.get("metrics") or {}
    else:
        original = read_json(path, {})
        if (original.get("scene_sha256") != item["scene_sha256"] or
                original.get("method") != item["method"] or
                original.get("seed") != item["action_seed"] or
                original.get("action_seed") != item["action_seed"] or
                original.get("repeat_index") != item["repeat_index"]):
            raise ValueError("S2-P19 I0 case场景、方法或动作种子不一致")
        if reference.get("source_kind") == "i0_train_validation":
            if (original.get("env_step") != reference["validation_step"] or
                    original.get("checkpoint_sha256") != reference["source_checkpoint_sha256"] or
                    original.get("scene_metadata") != item["scene_metadata"]):
                raise ValueError("S2-P19 I0试点case节点、断点或场景元数据不一致")
        elif (reference["group"] == "model" and
              original.get("model_sha256") != reference.get("source_model_sha256")):
            raise ValueError("S2-P19 I0模型case与冻结权重SHA不一致")
        metrics = original.get("metrics") or {}
        duration = metrics.get("total_seconds")
        if not isinstance(original.get("trace"), list):
            raise ValueError("S2-P19 I0 case缺宏步轨迹")
        for key in ("initialization_seconds", "candidate_generation_seconds", "probe_seconds"):
            if _number(metrics.get(key), key) < 0:
                raise ValueError("S2-P19 I0 case分段耗时为负")
        for key in ("decisions", "probes", "invalid"):
            if type(metrics.get(key)) is not int or metrics[key] < 0:
                raise ValueError("S2-P19 I0 case调用/动作计数无效")
        if metrics["decisions"] > 32 or metrics["probes"] > 32:
            raise ValueError("S2-P19 I0 case超局部搜索预算")
    duration = _number(duration, "完整初解耗时")
    if duration < 0:
        raise ValueError("S2-P19来源耗时为负")
    beams = deepcopy(original.get("beams"))
    count = item["beam_count"]
    normalized = _verify_beams(beams, metrics, count)
    row = {"scene_id": original.get("scene_id", item["case_id"]),
           "scene_metadata": item.get("scene_metadata"),
           "source_scene_sha256": item["scene_sha256"],
           "repeat_index": item["repeat_index"], "action_seed": item["action_seed"],
           "beam_count": count, "metrics": normalized, "beams": beams,
           "trace": deepcopy(original.get("trace") or []),
           "initial_allocation_seconds": duration, "inference_seconds": duration,
           "policy_inference_seconds": original.get("inference_seconds")
                                       if reference["format"] == "s2p19_inline_validation_v1" else None,
           "original_case_path": str(path), "original_case_file_sha256": item["sha256"],
           "service_timing": "after_last_recorded_action_including_later_interference"}
    return row


def source_rows(reference):
    if reference.get("experiment_id") != "S2-P19" or reference.get("format") not in (
            "s2p19_inline_validation_v1", "s2p19_i0_case_v1"):
        raise ValueError("不是S2-P19已冻结初解来源")
    if (reference.get("source_kind") == "i0_train_validation" and
            digest(reference["summary_path"]) != reference["summary_sha256"]):
        raise ValueError("S2-P19 I0试点验证摘要SHA变化")
    rows = {}
    for item in reference["case_files"]:
        row = _case_row(reference, item)
        key = (item["scene_sha256"], item["repeat_index"])
        if key in rows:
            raise ValueError("S2-P19初解场景/动作重复")
        rows[key] = row
    if len(rows) != reference["scene_count"] * reference["repeats"]:
        raise ValueError("S2-P19初解未覆盖冻结场景×动作全集")
    return rows


def prepare_source(app, body):
    """Resolve S2-P19 inline validation or I0 direct cases with exact partition."""
    run_id = body.get("source_run_id")
    validate_id(run_id)
    group = body.get("source_group")
    partition = body.get("partition")
    if group not in GROUPS or partition not in PARTITIONS:
        raise ValueError("S2-P19初解须明确来源组和development/validation/test分区")
    directory, readonly = app.resolve_run(run_id)
    job, status = read_json(directory / "job.json", {}), read_json(directory / "status.json", {})
    if readonly or job.get("experiment_id") != "S2-P19":
        raise ValueError("S2-P19初解须来自本工作台同组任务")
    from .datasets import assert_scene_profile, scene_set_files
    if (job.get("kind") == "train" and job.get("algorithm") == "sac" and
            job.get("research_stage") == "sac_improvement" and job.get("mode") == "train"):
        step = body.get("validation_step")
        episodes = status.get("episodes")
        if (group != "model" or partition != "validation" or type(step) is not int or
                not 0 < step <= 100000 or type(episodes) is not int or
                not 0 <= episodes <= 3125 or
                not (step == 100000 or episodes == 3125) or
                status.get("stage") != "stopped" or
                status.get("stop_reason") != "s2p19_pilot_gate_pending" or
                status.get("env_step") != step or status.get("checkpoint_step") != step or
                job.get("action_seed") != 53001 or job.get("repeats") != 1):
            raise ValueError("I0初解仅接受已停于实际100k/3125受限试点的验证")
        source_config = read_json(directory / "config.json", {})
        from satellite_rl.sac_improvement.config import validate_config
        controls = validate_config(source_config.get("i0"))
        if (source_config.get("research_stage") != "sac_improvement" or
                controls["device"] != "cuda" or
                controls != job.get("i0_config")):
            raise ValueError("I0试点配置与冻结任务不一致")
        checkpoint_sha = digest(directory / "checkpoint.pt")
        if status.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("I0试点断点与停止状态SHA不一致")
        dataset = (job.get("scene_sets") or {}).get("validation")
        from .s2p19_service import constructive_config
        model_config = constructive_config("C0")
        summary_path = directory / "validation" / f"step_{step:09d}" / "summary.json"
        reference = {"experiment_id": "S2-P19", "format": "s2p19_i0_case_v1",
                     "source_kind": "i0_train_validation", "run_id": run_id,
                     "source_directory": str(directory), "group": group,
                     "partition": partition, "validation_step": step,
                     "repeats": 1, "action_mode": "stochastic", "action_seed": 53001,
                     "source_checkpoint_sha256": checkpoint_sha,
                     "summary_path": str(summary_path), "summary_sha256": digest(summary_path)}
    elif job.get("kind") == "train" and job.get("algorithm") == "sac":
        step = body.get("validation_step")
        if (group != "model" or partition != "validation" or type(step) is not int or
                step not in (100000, 400000) or status.get("env_step") != step or
                (step == 100000 and (status.get("stage") != "stopped" or
                                     status.get("stop_reason") != "s2p19_pilot_gate_pending")) or
                (step == 400000 and status.get("stage") != "completed")):
            raise ValueError("C0/C1初解仅接受已核试点/终点的inline validation")
        dataset = (job.get("scene_sets") or {}).get("validation")
        source_config = read_json(directory / "config.json", {})
        if source_config.get("experiment", {}).get("arm") not in ("C0", "C1"):
            raise ValueError("训练任务不是S2-P19 C0/C1")
        from .checkpoints import config_hash
        reference = {"experiment_id": "S2-P19", "format": "s2p19_inline_validation_v1",
                     "run_id": run_id, "source_directory": str(directory), "group": group,
                     "partition": partition, "validation_step": step,
                     "source_config_hash": config_hash(source_config), "repeats": 1,
                     "action_mode": "stochastic", "action_seed": 53001,
                     "source_checkpoint_sha256": digest(directory / "checkpoint.pt")}
        first_case = directory / "validation_cases" / f"step_{step:09d}" / "scene_00000_r0.json"
        first_contract = (read_json(first_case, {}).get("body") or {}).get("contract") or {}
        policy_sha = first_contract.get("policy_sha256")
        if not isinstance(policy_sha, str) or len(policy_sha) != 64 or any(
                char not in "0123456789abcdef" for char in policy_sha):
            raise ValueError("S2-P19 inline验证case缺冻结actor权重SHA")
        reference["policy_sha256"] = policy_sha
        model_config = deepcopy(source_config)
    elif job.get("research_stage") == "sac_improvement" and job.get("mode") in ("gate", "evaluate", "baselines"):
        mode = job["mode"]
        required_partition = ("development" if mode == "gate" else "test" if mode == "evaluate"
                              else job.get("partition"))
        if partition != required_partition or status.get("stage") != "completed":
            raise ValueError("I0初解任务未完成或分区不符")
        dataset = (job.get("scene_sets") or {}).get(partition)
        if (mode == "gate" and group not in ("greedy", "rule", "random") or
                mode == "evaluate" and group != "model" or
                mode == "baselines" and (group not in ("greedy", "rule", "random") or
                                          group not in job.get("methods", []))):
            raise ValueError("I0来源组与gate/evaluate/baselines用途不符")
        if mode == "baselines" and partition not in ("validation", "test"):
            raise ValueError("I0共同基线只允许validation/test分区")
        repeat_count = (1 if mode == "gate" else job["repeats"] if mode == "evaluate"
                        else (job.get("method_repeats") or {}).get(group))
        if type(repeat_count) is not int or not 1 <= repeat_count <= 3:
            raise ValueError("I0来源方法缺合法冻结动作重复数")
        from .s2p19_service import constructive_config
        model_config = constructive_config("C0")
        reference = {"experiment_id": "S2-P19", "format": "s2p19_i0_case_v1",
                     "run_id": run_id, "source_directory": str(directory), "group": group,
                     "partition": partition, "repeats": repeat_count,
                     "action_mode": "deterministic" if group in ("greedy", "rule") else "stochastic",
                     "action_seed": 53001, "source_model_sha256":
                     next((item["sha256"] for item in job.get("model_files", [])
                           if Path(item["path"]).name == "policy.pt"), None)}
        if mode == "evaluate" and reference["source_model_sha256"] is None:
            raise ValueError("I0最终模型初解缺冻结policy.pt来源SHA")
    else:
        raise ValueError("S2-P19已保存初解须来自C0/C1验证或I0 gate/evaluate")
    if not dataset or (dataset.get("source") or {}).get("partition_role") != partition:
        raise ValueError("S2-P19初解缺对应真实分区的冻结场景")
    assert_scene_profile(dataset, "europe_v1")
    files = scene_set_files(dataset)
    dates = app.scene_dates(dataset)
    if not dates:
        raise ValueError("S2-P19初解场景缺UTC日期来源")
    if reference.get("source_kind") == "i0_train_validation" and dates != {"2024-01-12"}:
        raise ValueError("I0试点验证须来自冻结Jan12留出日期")
    if reference.get("source_kind") == "i0_train_validation" and len(files) != 72:
        raise ValueError("I0试点验证须覆盖完整72景")
    if reference.get("source_kind") == "i0_train_validation":
        _validate_i0_validation_grid(dataset)
    reference["scene_count"] = len(files)
    items = []
    if reference.get("source_kind") == "i0_train_validation":
        summary = read_json(reference["summary_path"], {})
        listed = summary.get("cases")
        if (summary.get("env_step") != reference["validation_step"] or
                summary.get("checkpoint_sha256") != reference["source_checkpoint_sha256"] or
                summary.get("source_set_id") != dataset["id"] or
                summary.get("count") != 72 or not isinstance(listed, list) or
                len(listed) != 72):
            raise ValueError("I0试点验证摘要与冻结72景/断点不一致")
        for index, (_, sha) in enumerate(files):
            path = directory / "validation" / f"step_{reference['validation_step']:09d}" / f"scene_{index:05d}.json"
            if listed[index] != {"path": str(path), "sha256": digest(path)}:
                raise ValueError("I0试点验证摘要case顺序、路径或SHA不一致")
            items.append({"path": str(path), "sha256": listed[index]["sha256"],
                          "scene_index": index, "repeat_index": 0,
                          "method": "I0_validation", "scene_sha256": sha,
                          "action_seed": 53001,
                          "beam_count": int(dataset["files"][index]["beam_count"]),
                          "scene_metadata": dataset["files"][index],
                          "case_id": path.stem})
    elif reference["format"] == "s2p19_inline_validation_v1":
        for index, (_, sha) in enumerate(files):
            path = directory / "validation_cases" / f"step_{reference['validation_step']:09d}" / f"scene_{index:05d}_r0.json"
            items.append({"path": str(path), "sha256": digest(path), "scene_index": index,
                          "repeat_index": 0, "scene_sha256": sha,
                          "action_seed": reference["action_seed"] + index,
                          "beam_count": int(dataset["files"][index]["beam_count"]),
                          "scene_metadata": dataset["files"][index], "case_id": path.stem})
    else:
        summary_path = directory / "results.json"
        reference["summary_path"] = str(summary_path)
        reference["summary_sha256"] = digest(summary_path)
        summary = read_json(summary_path, {})
        selected = [row for row in summary.get("scenes", []) if row.get("method") == group]
        by_sha = {sha: index for index, (_, sha) in enumerate(files)}
        for row in selected:
            sha = row.get("scene_sha256")
            if sha not in by_sha:
                raise ValueError("I0摘要case不属于冻结分区场景")
            index = by_sha[sha]
            path = Path(row.get("case_path", ""))
            repeat = int(row.get("repeat_index", 0))
            if not 0 <= repeat < reference["repeats"]:
                raise ValueError("I0 case动作重复编号无效")
            expected_seed = job["action_seed"] + repeat
            if row.get("action_seed") != expected_seed or row.get("case_sha256") != digest(path):
                raise ValueError("I0摘要case SHA不一致")
            items.append({"path": str(path), "sha256": row["case_sha256"],
                          "scene_index": index, "repeat_index": repeat, "method": group,
                          "scene_sha256": sha, "action_seed": expected_seed,
                          "beam_count": int(dataset["files"][index]["beam_count"]),
                          "scene_metadata": dataset["files"][index],
                          "case_id": row.get("case_id", path.stem)})
    reference["case_files"] = items
    indexed = source_rows(reference)
    expected = {(sha, repeat) for _, sha in files for repeat in range(reference["repeats"])}
    if set(indexed) != expected:
        raise ValueError("S2-P19初解未覆盖冻结分区场景×动作全集")
    if reference.get("source_kind") == "i0_train_validation":
        summary = read_json(reference["summary_path"], {})
        actual_si = float(np.mean([row["metrics"]["final_si_mean"] for row in indexed.values()]))
        actual_sgm = float(np.mean([row["metrics"]["final_sgm_mean"] for row in indexed.values()]))
        initial = [_number(read_json(item["path"], {}).get("metrics", {}).get("initial_si"),
                           "I0原始SI") for item in items]
        actual_delta = actual_si - float(np.mean(initial))
        if (abs(actual_si - _number(summary.get("mean_si"), "I0平均SI")) > 1e-7 or
                abs(actual_sgm - _number(summary.get("mean_sgm"), "I0平均SGM")) > 1e-7 or
                abs(actual_delta - _number(summary.get("mean_si_delta_vs_b1"), "I0平均改善")) > 1e-7):
            raise ValueError("I0试点摘要指标与完整逐景结果不一致")
    return job, model_config, dataset, reference
