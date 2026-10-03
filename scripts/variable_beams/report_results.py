"""Read-only S2-P17 audit and Chinese report from frozen workbench evidence.

No model inference, optimization, training, or scene generation is performed.
The complete report is written only after every approved job and case passes
the input, arithmetic, identity, and budget gates below.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import numpy as np


EXPERIMENT = "S2-P17"
N_VALUES = (50, 60, 75, 90, 100, 125, 140, 150, 175, 180, 200)
HELD_OUT = (60, 90, 140, 180)
LOADS = (0.75, 1.25)
ROLES = ("V0", "V1", "H0", "fixed", "greedy")
LABELS = {
    "V0": "V0 数量感知T-SAC＋固定100束训练",
    "V1": "V1 数量感知T-SAC＋混合束数训练",
    "H0": "H0 冻结G11 T0＋历史模型跨束数评价",
    "fixed": "B0 固定规则＋共同可行域投影",
    "greedy": "B1 贪心规则＋共同可行域投影",
}
JOB_IDS = ("V0_train", "V1_train", "V0_eval", "V1_eval", "H0_eval",
           "V0_sca", "V1_sca", "H0_sca", "fixed_sca", "greedy_sca")
EVAL_JOB = {"V0": "V0_eval", "V1": "V1_eval", "H0": "H0_eval",
            "fixed": "V0_eval", "greedy": "V0_eval"}
SCA_JOB = {role: f"{role}_sca" for role in ROLES}
REPEATS = {"V0": 3, "V1": 3, "H0": 3, "fixed": 1, "greedy": 1}
METRICS = ("si", "sgm", "basic_fraction", "zero_capacity_fraction",
           "effective_carried_bps", "power_w", "complete_fraction")
SCENE_FIELDS = ("sha256", "beam_count", "mother_scene_id", "load_multiplier",
                "partition_role", "selected_time_utc", "source_sha256")


class AuditError(ValueError):
    """A missing, inconsistent, or altered formal input blocks complete reporting."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256_file(path):
    path = Path(path)
    require(path.is_file(), f"缺少冻结文件：{path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path):
    path = Path(path)
    require(path.is_file(), f"缺少JSON：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AuditError(f"JSON损坏：{path}") from error
    require(isinstance(value, dict), f"JSON根对象无效：{path}")
    return value


def verified_json(path, expected_sha):
    require(sha256_file(path) == expected_sha, f"审计期间案例文件发生变化：{path}")
    return json_file(path)


def canonical_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def same_scene(left, right):
    return isinstance(left, dict) and isinstance(right, dict) and all(
        left.get(key) == right.get(key) for key in SCENE_FIELDS)


def verify_scene_files(refs, inputs, label):
    for ref in refs:
        path = Path(ref["path"])
        actual = sha256_file(path)
        require(actual == ref.get("sha256"), f"{label}冻结场景文件SHA不符：{path}")
        inputs[str(path.resolve())] = actual


def close(actual, expected, label, *, rtol=1e-7, atol=1e-7):
    require(isinstance(actual, (int, float)) and math.isfinite(actual), f"{label}缺失或非有限")
    require(math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol),
            f"{label}与逐束复算不一致：记录{actual}，复算{expected}")


def _sgm(capacity, demand, beta):
    if capacity <= 0:
        return 0.0
    ratio = capacity / demand
    real = ratio - 1 if ratio >= 1 else 1 - 1 / ratio
    imag = (capacity - demand) / beta
    return 1 - (1 - math.exp(-math.hypot(real, imag))) ** 3


def audit_beams(beams, metrics, expected_n, *, continuous=False):
    """Independently recompute every terminal D/C/service/SI/SGM/resource field."""
    require(isinstance(beams, list) and len(beams) == expected_n, "终态目标束数量不完整")
    require([beam.get("beam_id") for beam in beams] == list(range(expected_n)),
            "终态束顺序或身份错位")
    original = [beam.get("original_index") for beam in beams]
    require(len(set(original)) == expected_n, "原始波束身份重复")
    demands, capacities, powers = [], [], []
    for beam in beams:
        demand = beam.get("required_bps")
        capacity = beam.get("capacity_bps")
        power = beam.get("power_w")
        require(all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in (demand, capacity, power)), "逐束需求/容量/功率缺失或非有限")
        require(demand > 0 and capacity >= -1e-6 and power >= -1e-8,
                "逐束需求非正或容量/功率负值")
        capacity = max(0.0, float(capacity))
        demand, power = float(demand), max(0.0, float(power))
        ratio = capacity / demand
        close(beam.get("rate_bps"), capacity, "逐束容量rate_bps", atol=1e-3)
        close(beam.get("effective_served_bps"), min(capacity, demand),
              "逐束有效承载", atol=1e-3)
        close(beam.get("capacity_demand_ratio"), ratio, "逐束未封顶C/D")
        close(beam.get("si"), min(ratio, 1.0), "逐束SI")
        require(beam.get("basic_met") is (capacity >= 0.1 * demand), "逐束基本服务判断错误")
        slots = beam.get("slots")
        require(type(slots) is int and slots >= 0, "逐束频率槽数无效")
        if slots == 0:
            require(power <= 1e-8 and capacity <= 1e-3, "空槽束出现功率或容量")
        demands.append(demand)
        capacities.append(capacity)
        powers.append(power)
    beta = statistics.mean(demands)
    sgm = [_sgm(c, d, beta) for c, d in zip(capacities, demands)]
    for index, (beam, value) in enumerate(zip(beams, sgm)):
        close(beam.get("sgm"), value, f"第{index}束SGM", atol=1e-7)
    si = [min(c / d, 1.0) for c, d in zip(capacities, demands)]
    basic = sum(c >= 0.1 * d for c, d in zip(capacities, demands))
    zero = sum(c <= 0 for c in capacities)
    power_total = sum(powers)
    require(power_total <= 3000 + 1e-6, "终态超过3000W总预算")
    close(metrics.get("final_si_mean"), statistics.mean(si), "终态平均SI")
    close(metrics.get("final_sgm_mean"), statistics.mean(sgm), "终态平均SGM")
    close(metrics.get("power_w"), power_total, "终态总功率", atol=1e-4)
    require(metrics.get("basic_met_count") == basic, "基本服务束数与逐束结果不一致")
    if "zero_capacity_count" in metrics:
        require(metrics["zero_capacity_count"] == zero, "零容量束数与逐束结果不一致")
    if "effective_carried_bps" in metrics:
        close(metrics["effective_carried_bps"], sum(min(c, d) for c, d in zip(capacities, demands)),
              "有效承载总量", atol=1e-3)
    if "final_si_total" in metrics:
        close(metrics["final_si_total"], sum(si), "SI总分")
    if "final_sgm_total" in metrics:
        close(metrics["final_sgm_total"], sum(sgm), "SGM总分")
    if "all_basic_met" in metrics:
        require(metrics["all_basic_met"] is (basic == expected_n), "全束基本服务标记错误")
    if "complete" in metrics:
        require(type(metrics["complete"]) is bool, "完整分配标记无效")
    if not continuous:
        require(all(any(math.isclose(p, level, abs_tol=1e-6) for level in range(5, 51, 5))
                    for p, beam in zip(powers, beams) if beam["slots"]),
                "第二阶段初解功率不在原5W档位")
    return {"si": statistics.mean(si), "sgm": statistics.mean(sgm),
            "basic_fraction": basic / expected_n, "zero_capacity_fraction": zero / expected_n,
            "effective_carried_bps": sum(min(c, d) for c, d in zip(capacities, demands)),
            "power_w": power_total, "complete_fraction": float(metrics.get("complete", True)),
            "beam_count": expected_n, "basic_count": basic, "zero_count": zero,
            "si_total": sum(si), "sgm_total": sum(sgm)}


def read_jsonl(path):
    path = Path(path)
    require(path.is_file(), f"缺少原始日志：{path}")
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(line.endswith("\n"), f"JSONL末行不完整：{path}:{number}")
            try:
                yield json.loads(line)
            except ValueError as error:
                raise AuditError(f"JSONL损坏：{path}:{number}") from error


def load_manifest(path, inputs):
    manifest = json_file(path)
    inputs[str(Path(path).resolve())] = sha256_file(path)
    require(manifest.get("experiment_id") == EXPERIMENT, "控制清单实验编号不符")
    require(set(manifest.get("jobs", {})) == set(JOB_IDS), "控制清单缺少或多出正式任务")
    require(set(manifest.get("datasets", {})) == {"train", "validation", "test"},
            "控制清单缺少冻结数据分区")
    require((manifest.get("target") or {}).get("ssh_port") == 20424,
            "正式执行目标须为唯一20424实例")
    require(isinstance(manifest.get("scientific_commit"), str) and
            len(manifest["scientific_commit"]) == 40, "缺少冻结科学提交")
    scope_path = Path(path).resolve().parent / "approved_scope.json"
    scope_sha = sha256_file(scope_path)
    require(scope_sha == manifest.get("scope_sha256"), "用户确认范围合同SHA缺失或不符")
    inputs[str(scope_path)] = scope_sha
    scope = json_file(scope_path)
    summary = manifest.get("scope") or {}
    allowed = {"eligible520": (520, 7280), "replace576": (576, 8064)}
    decision = scope.get("decision")
    require(scope.get("experiment_id") == EXPERIMENT and
            scope.get("status") == "user_confirmed" and
            decision in allowed and
            (scope.get("mother_count"), scope.get("derived_scene_count")) == allowed.get(decision) and
            scope.get("derived_scene_count") == scope.get("mother_count") * 14 and
            isinstance(scope.get("user_message_reference"), str) and
            bool(scope["user_message_reference"]) and
            scope.get("h0_compatibility_status") == "passed" and
            scope.get("h0_compatibility_receipt"),
            "训练母景范围缺少有效用户确认或与520/576决策不符")
    require(all(summary.get(key) == scope[key] for key in
                ("decision", "mother_count", "derived_scene_count", "user_message_reference")),
            "控制清单范围摘要与用户确认合同不一致")
    train_source = Path(scope["train_record_path"])
    train_sha = sha256_file(train_source)
    require(train_sha == scope.get("train_record_sha256"), "用户确认训练数据SHA不符")
    inputs[str(train_source.resolve())] = train_sha
    return manifest


def load_datasets(manifest, inputs):
    expected_counts = {"train": manifest["scope"]["derived_scene_count"],
                       "validation": 144, "test": 440}
    records = {}
    for role, count in expected_counts.items():
        entry = manifest["datasets"][role]
        path = Path(entry["record_path"])
        digest = sha256_file(path)
        require(digest == entry.get("record_sha256"), f"{role}数据记录SHA不符")
        inputs[str(path.resolve())] = digest
        record = json_file(path)
        refs = record.get("files")
        require(isinstance(refs, list) and len(refs) == count and entry.get("count") == count,
                f"{role}冻结场景数量不符")
        require((record.get("source") or {}).get("partition_role") == role,
                f"{role}冻结场景分区身份不符")
        if role == "train":
            require(len({ref["mother_scene_id"] for ref in refs}) ==
                    manifest["scope"]["mother_count"],
                    "训练母景数与用户确认范围不一致")
        source_path = entry.get("source_record_path")
        if source_path:
            source_digest = sha256_file(source_path)
            require(source_digest == entry.get("source_record_sha256"),
                    f"{role}准备数据记录SHA不符")
            inputs[str(Path(source_path).resolve())] = source_digest
            prepared = json_file(source_path)
            require(len(prepared.get("files", [])) == count and all(
                same_scene(a, b) for a, b in zip(refs, prepared["files"])),
                f"{role}准备→导入场景SHA/语义映射不一致")
        seen_sha = set()
        for ref in refs:
            require(ref.get("partition_role") == role, f"{role}逐场景分区身份不符")
            scene_path = Path(ref["path"])
            actual = sha256_file(scene_path)
            require(actual == ref.get("sha256") and actual not in seen_sha,
                    f"{role}逐场景SHA错误或重复")
            seen_sha.add(actual)
            inputs[str(scene_path.resolve())] = actual
        records[role] = record
    refs = records["test"]["files"]
    grid = {(ref["mother_scene_id"], ref["beam_count"], ref["load_multiplier"])
            for ref in refs}
    mothers = {ref["mother_scene_id"] for ref in refs}
    require(len(mothers) == 20 and len(grid) == 440 and
            grid == {(mother, n, load) for mother in mothers for n in N_VALUES for load in LOADS},
            "独立考核不是20母景×11束数×2负载的完整共同矩阵")
    require({ref["selected_time_utc"][:10] for ref in refs} == {"2024-01-29"},
            "独立考核日期不是冻结Jan29")
    for role in ("train", "validation"):
        require(not any(ref["beam_count"] in HELD_OUT for ref in records[role]["files"]),
                "最终留出束数混入训练或验证")
    return records


def load_jobs(manifest, datasets, inputs):
    jobs = {}
    unique_run_ids = set()
    expected_units = {"V0_train": 400000, "V1_train": 400000,
                      "V0_eval": 2200, "V1_eval": 1320, "H0_eval": 1320,
                      "V0_sca": 3960, "V1_sca": 3960, "H0_sca": 3960,
                      "fixed_sca": 1320, "greedy_sca": 1320}
    for name in JOB_IDS:
        entry = manifest["jobs"][name]
        require(entry.get("stage") == "completed" and entry.get("run_id"),
                f"{name}未完成；拒绝完整报告")
        require(entry["run_id"] not in unique_run_ids, "正式任务ID重复")
        unique_run_ids.add(entry["run_id"])
        directory = Path(entry["job_path"])
        job_path, status_path = directory / "job.json", Path(entry["status_path"])
        config_path, result_path = directory / "config.json", Path(entry["results_path"])
        for field, path in (("job_sha256", job_path), ("status_sha256", status_path),
                            ("config_sha256", config_path), ("results_sha256", result_path)):
            actual = sha256_file(path)
            require(entry.get(field) == actual, f"{name}的{field}缺失或不符")
            inputs[str(path.resolve())] = actual
        job, status, config = json_file(job_path), json_file(status_path), json_file(config_path)
        require(job.get("run_id") == entry["run_id"] and
                job.get("experiment_id") == EXPERIMENT,
                f"{name}任务来源身份不符")
        require(status.get("stage") == "completed" and
                status.get("env_step") == expected_units[name],
                f"{name}任务状态或预算不完整")
        expected_kind = "train" if name.endswith("_train") else "evaluate" if name.endswith("_eval") else "refine"
        require(job.get("kind") == expected_kind, f"{name}工作台任务类型不符")
        require(job.get("git_commit") == manifest["scientific_commit"],
                f"{name}科学源码提交不一致")
        if name.endswith(("_eval", "_sca")):
            frozen = (job.get("scene_sets") or {}).get("test") or {}
            require(len(frozen.get("files", [])) == 440 and all(
                same_scene(a, b) for a, b in zip(frozen["files"], datasets["test"]["files"])),
                    f"{name}共同独立场景不一致")
            verify_scene_files(frozen["files"], inputs, name)
        if name.endswith("_train"):
            require(status.get("update_step") == 398001,
                    f"{name}实际学习更新次数不符")
            trained = (job.get("scene_sets") or {}).get("train", {})
            require(len(trained.get("files", [])) == manifest["scope"]["derived_scene_count"] and all(
                same_scene(a, b) for a, b in zip(trained["files"], datasets["train"]["files"])),
                f"{name}训练数据集不符")
            verify_scene_files(trained["files"], inputs, name + " train")
            validation = (job.get("scene_sets") or {}).get("validation", {})
            require(len(validation.get("files", [])) == 144 and all(
                same_scene(a, b) for a, b in zip(validation["files"], datasets["validation"]["files"])),
                f"{name}验证数据集不符")
            verify_scene_files(validation["files"], inputs, name + " validation")
        model_path = entry.get("model_path")
        if model_path:
            actual = sha256_file(model_path)
            require(actual == entry.get("model_sha256"), f"{name}模型权重SHA不符")
            inputs[str(Path(model_path).resolve())] = actual
            meta_path = Path(model_path).with_name("model.json")
            actual_meta = sha256_file(meta_path)
            require(actual_meta == entry.get("model_metadata_sha256"),
                    f"{name}模型元数据SHA不符")
            inputs[str(meta_path.resolve())] = actual_meta
        jobs[name] = {"entry": entry, "job": job, "status": status,
                      "config": config, "directory": directory,
                      "results_path": result_path,
                      "results_sha256": entry["results_sha256"]}
    require(sum(expected_units[name] for name in ("V0_eval", "V1_eval", "H0_eval")) == 4840,
            "独立初解预算账错误")
    require(sum(expected_units[name] for name in SCA_JOB.values()) == 14520,
            "SCA工作台方法记录预算账错误")
    require(sum(expected_units[name] for name in ("V0_train", "V1_train")) == 800000,
            "训练预算账错误")
    for role in ROLES:
        name = SCA_JOB[role]
        source = jobs[name]["job"].get("initial_source") or {}
        expected_eval = jobs[EVAL_JOB[role]]["job"]["run_id"]
        expected_group = role if role in ("fixed", "greedy") else "model"
        require(source.get("run_id") == expected_eval and source.get("group") == expected_group,
                f"{name}初解来源组错配")
        require(source.get("sha256") == jobs[EVAL_JOB[role]]["results_sha256"],
                f"{name}第二阶段原始结果SHA错配")
        require(jobs[name]["config"].get("power_refinement", {}).get("methods") ==
                ["none", "sca", "sca_si_lp"], f"{name}缺少原/A/B完整路线")
    return jobs


def _read_evaluation_case(path, contract_sha, scene_ref, seed, repeat, group, inputs):
    file_sha = sha256_file(path)
    inputs[str(path.resolve())] = file_sha
    envelope = json_file(path)
    body = envelope.get("body")
    require(isinstance(body, dict) and envelope.get("sha256") == canonical_hash(body),
            f"评价case内容SHA不符：{path}")
    result, expected = body.get("result"), body.get("expected")
    require(body.get("contract_sha256") == contract_sha and
            isinstance(result, dict) and isinstance(expected, dict),
            f"评价case合同不符：{path}")
    source_sha = scene_ref["sha256"]
    require(expected.get("scene_sha256") == source_sha and
            expected.get("beam_count") == scene_ref["beam_count"] and
            expected.get("action_seed") == seed and expected.get("repeat_index") == repeat and
            expected.get("kind") == group and same_scene(expected.get("scene_metadata"), scene_ref),
            f"评价case场景/动作/负载身份错配：{path}")
    scene = result.get("scene") or {}
    require(scene.get("source_sha256") == source_sha and
            result.get("action_seed") == seed and result.get("repeat_index") == repeat and
            same_scene(result.get("scene_metadata"), scene_ref) and
            result.get("beam_count") == scene_ref["beam_count"],
            f"评价结果场景/动作身份错配：{path}")
    require(sha256_file(scene["path"]) == scene.get("sha256"),
            f"评价已归档场景SHA不符：{path}")
    inputs[str(Path(scene["path"]).resolve())] = scene["sha256"]
    require(isinstance(result.get("trace"), list) and
            len(result["trace"]) > 0, f"评价case缺少逐动作原始轨迹：{path}")
    for field in ("initial_allocation_seconds", "inference_seconds"):
        value = result.get(field)
        require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                f"评价case缺少实测{field}：{path}")
    recomputed = audit_beams(result.get("beams"), result.get("metrics") or {},
                             scene_ref["beam_count"])
    return {"metrics": recomputed, "beams_sha256": canonical_hash(result["beams"]),
            "case_path": str(path), "case_sha256": file_sha,
            "initial_allocation_seconds": float(result["initial_allocation_seconds"]),
            "inference_seconds": float(result["inference_seconds"]),
            "scene_metadata": scene_ref, "source_scene_sha256": source_sha,
            "action_seed": seed, "repeat_index": repeat,
            "source_group": group}


def load_evaluations(jobs, test_refs, inputs):
    direct = {}
    seen_paths = set()
    expected_counts = {"V0_eval": 2200, "V1_eval": 1320, "H0_eval": 1320}
    for name, expected_count in expected_counts.items():
        directory = jobs[name]["directory"]
        case_dir = directory / "cases"
        contract = json_file(case_dir / "contract.json")
        contract_data = contract.get("contract")
        require(isinstance(contract_data, dict) and
                contract.get("sha256") == canonical_hash(contract_data),
                f"{name}冻结评价合同SHA不符")
        require(contract_data.get("run_id") == jobs[name]["job"]["run_id"] and
                contract_data.get("repeats") == 3 and
                contract_data.get("action_seed_base") == 53001,
                f"{name}冻结评价预算不符")
        inputs[str((case_dir / "contract.json").resolve())] = sha256_file(case_dir / "contract.json")
        groups = ("model", "fixed", "greedy") if name == "V0_eval" else ("model",)
        require(set(jobs[name]["job"].get("baselines", [])) ==
                ({"fixed", "greedy"} if name == "V0_eval" else set()),
                f"{name}基线组合预算错配")
        for index, ref in enumerate(test_refs):
            for group in groups:
                role = group if group != "model" else name.split("_")[0]
                for repeat in range(REPEATS[role]):
                    seed = 53001 + repeat
                    case_name = f"{group}_{index:05d}_{repeat:02d}.json"
                    path = case_dir / case_name
                    case = _read_evaluation_case(path, contract["sha256"], ref,
                                                 seed, repeat, group, inputs)
                    key = (role, ref["sha256"], repeat)
                    require(key not in direct, f"第二阶段初解重复：{key}")
                    direct[key] = case
                    seen_paths.add(path.name)
        found = {p.name for p in case_dir.glob("*.json") if p.name != "contract.json"}
        require(found == seen_paths,
                f"{name}评价case数量/清单不符")
        require(len(found) == expected_count, f"{name}评价case预算不完整")
        seen_paths.clear()
    require(len(direct) == 4840, "第二阶段初始分配未齐4840")
    return direct


def _fingerprint(value):
    return canonical_hash(value)


def _check_timing(row, method, first):
    timing = row.get("timing") or {}
    fields = ("initialization_seconds", "physics_setup_seconds",
              "optimizer_wall_seconds", "verification_seconds", "total_pipeline_seconds")
    for field in fields:
        value = timing.get(field)
        require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                f"{method}缺少实测{field}")
    close(timing["initialization_seconds"], first["initial_allocation_seconds"],
          f"{method}公共初解耗时", atol=1e-4)
    components = sum(timing[field] for field in fields[:-1])
    if method == "sca_si_lp":
        for field in ("reference_optimizer_wall_seconds", "reference_verification_seconds"):
            value = timing.get(field)
            require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                    f"B路线缺少实测A段{field}")
            components += value
    close(timing["total_pipeline_seconds"], components, f"{method}分段合计耗时", atol=1e-4)
    return timing


def _read_refinement_case(path, role, method, ref, repeat, source_case, inputs):
    file_sha = sha256_file(path)
    inputs[str(path.resolve())] = file_sha
    row = json_file(path)
    require(row.get("method") == method and row.get("repeat") == repeat and
            row.get("source_scene_sha256") == ref["sha256"] and
            row.get("beam_count") == ref["beam_count"] and
            same_scene(row.get("scene_metadata"), ref),
            f"SCA案例场景/方法/动作错配：{path}")
    expected_group = role if role in ("fixed", "greedy") else "model"
    require(row.get("initial_source_group") == expected_group and
            row.get("initializer_kind") == ("model" if expected_group == "model"
                                                 else "baseline_" + expected_group),
            f"SCA初解方法来源错配：{path}")
    require(row.get("source_initial_case_path") and
            Path(row["source_initial_case_path"]).resolve() ==
            Path(source_case["case_path"]).resolve() and
            row.get("source_initial_case_sha256") == source_case["case_sha256"],
            f"SCA初解源case路径/SHA错配：{path}")
    require(row.get("initial_allocation_sha256") == source_case["beams_sha256"],
            f"SCA初解逐束SHA错配：{path}")
    require(row.get("power_budget_mode") == "total", "SCA功率预算口径不是3000W总预算")
    recomputed = audit_beams(row.get("beams"), row.get("metrics") or {},
                             ref["beam_count"], continuous=True)
    solver = row.get("solver") or {}
    count = solver.get("evaluations")
    require(type(count) is int and 0 <= count <= 2000,
            f"{method}容量调用超过2000或缺失")
    elapsed = solver.get("elapsed_seconds")
    require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0,
            f"{method}求解耗时缺失")
    require(elapsed <= 31.0, f"{method}求解超过30秒硬时限加1秒记录容差")
    timing = _check_timing(row, method, source_case)
    highest_power = None
    highest_sha = None
    highest_metrics = None
    if method == "sca_si_lp":
        highest_power = solver.get("highest_si_power_w")
        highest = solver.get("highest_si_metrics") or {}
        n = ref["beam_count"]
        require(isinstance(highest_power, list) and len(highest_power) == n and
                all(isinstance(value, (int, float)) and math.isfinite(value) and value >= -1e-8
                    for value in highest_power), "最高SI候选缺少完整非负功率向量")
        arrays = ("required_bps", "capacity_bps", "effective_served_bps",
                  "capacity_demand_ratio", "beam_si", "beam_sgm")
        require(all(isinstance(highest.get(field), list) and len(highest[field]) == n
                    for field in arrays), "最高SI候选缺少逐束D/C/服务/SI/SGM")
        candidate_beams = []
        for index, original in enumerate(row["beams"]):
            capacity = highest["capacity_bps"][index]
            demand = highest["required_bps"][index]
            candidate_beams.append({**original,
                "required_bps": demand, "capacity_bps": capacity, "rate_bps": capacity,
                "effective_served_bps": highest["effective_served_bps"][index],
                "capacity_demand_ratio": highest["capacity_demand_ratio"][index],
                "si": highest["beam_si"][index], "sgm": highest["beam_sgm"][index],
                "basic_met": capacity >= 0.1 * demand, "power_w": highest_power[index]})
        highest_metrics = audit_beams(candidate_beams, highest, n, continuous=True)
        close(highest_metrics["si"], solver.get("best_feasible_si_mean"),
              "最高SI候选与求解器best_feasible_si_mean")
        highest_sha = canonical_hash(highest_power)
    return {"metrics": recomputed, "beams_sha256": _fingerprint(row["beams"]),
            "case_path": str(path), "case_sha256": file_sha,
            "solver_evaluations": count, "timing": timing,
            "solver": solver, "source_scene_sha256": ref["sha256"],
            "scene_metadata": ref, "repeat_index": repeat,
            "source_group": expected_group,
            "reference_case_sha256": row.get("reference_case_sha256"),
            "reference_allocation_sha256": row.get("reference_allocation_sha256"),
            "reference_solver_evaluations": row.get("reference_solver_evaluations"),
            "complete_route_solver_evaluations": row.get("complete_route_solver_evaluations"),
            "highest_si_power_sha256": highest_sha,
            "highest_si_mean": highest_metrics["si"] if highest_metrics else None}


def load_refinements(jobs, test_refs, direct, inputs):
    refined = {}
    role_case_counts = {}
    for role in ROLES:
        name = SCA_JOB[role]
        directory = jobs[name]["directory"]
        job = jobs[name]["job"]
        case_dir = directory / "power_cases"
        manifest_path = directory / "refinement_manifest.json"
        manifest = json_file(manifest_path)
        require(manifest.get("budget_cases") == 440 * REPEATS[role] * 3,
                f"{name}细调案例预算不符")
        require([sha for _, sha in manifest.get("scene_files", [])] ==
                [ref["sha256"] for ref in test_refs],
                f"{name}细调冻结场景SHA清单不符")
        inputs[str(manifest_path.resolve())] = sha256_file(manifest_path)
        count = 0
        expected_files = set()
        for index, ref in enumerate(test_refs):
            for repeat in range(REPEATS[role]):
                source = direct[(role, ref["sha256"], repeat)]
                stages = {}
                for method in ("none", "sca", "sca_si_lp"):
                    path = case_dir / f"scene_{index:05d}_r{repeat}_{method}.json"
                    expected_files.add(path.name)
                    stage = _read_refinement_case(path, role, method, ref, repeat, source, inputs)
                    stages[method] = stage
                    count += 1
                raw, none, a, b = source, stages["none"], stages["sca"], stages["sca_si_lp"]
                for metric in ("si", "sgm", "power_w", "effective_carried_bps"):
                    close(none["metrics"][metric], raw["metrics"][metric],
                          f"{role}原分配none与第二阶段{metric}", atol=1e-4)
                require(a["metrics"]["si"] >= raw["metrics"]["si"] - 1e-8 and
                        a["metrics"]["sgm"] >= raw["metrics"]["sgm"] - 1e-8,
                        f"{role}中间A低于原初解的保护门槛")
                require(b["metrics"]["si"] >= a["metrics"]["si"] - 1e-8,
                        f"{role}最终B的SI低于公共A")
                require(b["reference_case_sha256"] == a["case_sha256"] and
                        b["reference_allocation_sha256"] == a["beams_sha256"],
                        f"{role}最终B与公共A案例/SHA错配")
                require(b["reference_solver_evaluations"] == a["solver_evaluations"] and
                        b["complete_route_solver_evaluations"] ==
                        a["solver_evaluations"] + b["solver_evaluations"] <= 4000,
                        f"{role}最终B完整A+B容量调用账错误")
                best = b["solver"].get("phase1_best_si_mean")
                require(isinstance(best, (int, float)) and math.isfinite(best) and
                        best >= a["metrics"]["si"] - 1e-8 and
                        b["metrics"]["si"] >= best - 1e-8,
                        f"{role}最终B最高SI候选或守护门槛错配")
                close(b["timing"]["reference_optimizer_wall_seconds"],
                      a["timing"]["optimizer_wall_seconds"], "B路线引用A求解耗时", atol=1e-4)
                close(b["timing"]["reference_verification_seconds"],
                      a["timing"]["verification_seconds"], "B路线引用A核验耗时", atol=1e-4)
                refined[(role, ref["sha256"], repeat)] = stages
        found = {path.name for path in case_dir.glob("*.json")}
        require(found == expected_files and count == 440 * REPEATS[role] * 3,
                f"{name}细调案例缺失/多出")
        role_case_counts[role] = count
        require((job.get("initial_source") or {}).get("run_id") ==
                jobs[EVAL_JOB[role]]["job"]["run_id"], f"{name}来源评价任务错配")
    require(sum(role_case_counts.values()) == 14520 and len(refined) == 4840,
            "SCA任务记录不是14520或独立初解不是4840")
    return refined


def audit_physics(jobs, test_refs, direct, refined):
    """Re-evaluate link capacities with the frozen stage-three physical core."""
    from satellite_rl.power_refinement.physics import FixedAllocationPhysics

    checked = 0
    candidate_checked = 0
    for role in ROLES:
        model_config = jobs[SCA_JOB[role]]["job"]["model_config"]
        for ref in test_refs:
            with np.load(ref["path"], allow_pickle=False) as archive:
                scene = {key: archive[key].copy() for key in archive.files}
            for repeat in range(REPEATS[role]):
                key = (role, ref["sha256"], repeat)
                raw = direct[key]
                raw_beams = verified_json(raw["case_path"], raw["case_sha256"])["body"]["result"]["beams"]
                physics = FixedAllocationPhysics.from_scene(scene, model_config, raw_beams)
                for stage in (raw, *[refined[key][name] for name in ("none", "sca", "sca_si_lp")]):
                    beams = (raw_beams if stage is raw else
                             verified_json(stage["case_path"], stage["case_sha256"])["beams"])
                    powers = np.asarray([beam["power_w"] for beam in beams], dtype=float)
                    capacities = physics.evaluate(powers).capacity_bps
                    saved = np.asarray([beam["capacity_bps"] for beam in beams], dtype=float)
                    require(np.allclose(capacities, saved, rtol=1e-8, atol=1e-3),
                            f"物理容量复算不符：{stage['case_path']}")
                    checked += 1
                highest = refined[key]["sca_si_lp"]
                solver = verified_json(highest["case_path"], highest["case_sha256"])["solver"]
                power = np.asarray(solver["highest_si_power_w"], dtype=float)
                require(canonical_hash(solver["highest_si_power_w"]) ==
                        highest["highest_si_power_sha256"], "最高SI候选功率SHA错配")
                capacity = physics.evaluate(power).capacity_bps
                require(np.allclose(capacity, solver["highest_si_metrics"]["capacity_bps"],
                                    rtol=1e-8, atol=1e-3),
                        "最高SI候选逐束容量物理复算不符")
                close(float(np.minimum(capacity / physics.demand_bps, 1).mean()),
                      highest["highest_si_mean"], "最高SI候选物理复算SI")
                candidate_checked += 1
    require(checked == 19360, "物理链路复算未覆盖4840原始＋14520方法终态")
    require(candidate_checked == 4840, "最高SI候选物理复算未覆盖4840条B路线")
    return checked, candidate_checked


def mean_rows(rows):
    require(rows, "共同统计分组为空")
    return {key: float(statistics.mean(row[key] for row in rows)) for key in METRICS}


def summarize_common(test_refs, direct, refined):
    """Average repeats per derivative scene before N/load/mother comparisons."""
    records = []
    pairs = []
    by_role_stage_scene = {}
    for role in ROLES:
        for ref in test_refs:
            sha = ref["sha256"]
            action_keys = [(role, sha, repeat) for repeat in range(REPEATS[role])]
            for stage in ("direct", "final"):
                values = [direct[key]["metrics"] if stage == "direct" else
                          refined[key]["sca_si_lp"]["metrics"] for key in action_keys]
                aggregate = mean_rows(values)
                row = {"role": role, "combination": LABELS[role], "stage": stage,
                       "scene_sha256": sha, "mother_scene_id": ref["mother_scene_id"],
                       "beam_count": ref["beam_count"], "load_multiplier": ref["load_multiplier"],
                       "action_repeats": REPEATS[role],
                       "covered_aircraft_fraction": ref["covered_aircraft_fraction"],
                       "covered_demand_fraction": ref["covered_demand_fraction"],
                       "total_demand_bps": ref["total_demand_bps"],
                       "covered_demand_bps": ref["covered_demand_bps"],
                       "uncovered_demand_bps": ref["uncovered_demand_bps"],
                       **aggregate}
                records.append(row)
                by_role_stage_scene[(role, stage, sha)] = row
    require(len(records) == len(ROLES) * 440 * 2,
            "同场景动作均值数量不符")
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["role"], row["stage"], row["beam_count"],
                 row["load_multiplier"])].append(row)
    by_n_load = []
    for (role, stage, n, load), rows in sorted(grouped.items()):
        require(len(rows) == 20 and len({row["mother_scene_id"] for row in rows}) == 20,
                "N/负载分组不是20个独立母快照")
        by_n_load.append({"role": role, "combination": LABELS[role],
                          "stage": stage, "beam_count": n, "load_multiplier": load,
                          "mother_count": 20, **mean_rows(rows),
                          "covered_aircraft_fraction": statistics.mean(
                              row["covered_aircraft_fraction"] for row in rows),
                          "covered_demand_fraction": statistics.mean(
                              row["covered_demand_fraction"] for row in rows)})
    require(len(by_n_load) == 5 * 2 * 11 * 2, "N/负载共同汇总缺失")
    for stage in ("direct", "final"):
        for ref in test_refs:
            sha = ref["sha256"]
            base = by_role_stage_scene[("V0", stage, sha)]
            candidate = by_role_stage_scene[("V1", stage, sha)]
            pairs.append({"stage": stage, "scene_sha256": sha,
                          "mother_scene_id": ref["mother_scene_id"],
                          "beam_count": ref["beam_count"],
                          "load_multiplier": ref["load_multiplier"],
                          **{key + "_delta_V1_minus_V0": candidate[key] - base[key]
                             for key in METRICS}})
    require(len(pairs) == 880, "V1/V0逐场景配对数量不符")
    return records, by_n_load, pairs


def percentile_summary(values):
    array = np.asarray(values, dtype=float)
    require(len(array) and np.isfinite(array).all() and np.all(array >= 0), "耗时分布缺失或非有限")
    return {"mean": float(np.mean(array)), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95))}


def summarize_timing(test_refs, direct, refined):
    stages = ("initial_allocation_seconds", "physics_setup_seconds",
              "reference_optimizer_wall_seconds", "reference_verification_seconds",
              "optimizer_wall_seconds", "verification_seconds", "total_pipeline_seconds")
    raw_rows = []
    for role in ROLES:
        for ref in test_refs:
            for repeat in range(REPEATS[role]):
                key = (role, ref["sha256"], repeat)
                source, final = direct[key], refined[key]["sca_si_lp"]
                timing = final["timing"]
                raw_rows.append({"role": role, "combination": LABELS[role],
                                 "scene_sha256": ref["sha256"],
                                 "mother_scene_id": ref["mother_scene_id"],
                                 "beam_count": ref["beam_count"],
                                 "load_multiplier": ref["load_multiplier"],
                                 "repeat_index": repeat,
                                 "initial_allocation_seconds": source["initial_allocation_seconds"],
                                 **{field: timing[field] for field in stages if field != "initial_allocation_seconds"}})
    require(len(raw_rows) == 4840, "完整耗时原始动作数量不符")
    groups = defaultdict(list)
    for row in raw_rows:
        groups[(row["role"], row["beam_count"])].append(row)
    summary = []
    for (role, n), rows in sorted(groups.items()):
        expected = 20 * 2 * REPEATS[role]
        require(len(rows) == expected, "逐N耗时动作数量不符")
        result = {"role": role, "combination": LABELS[role], "beam_count": n,
                  "single_action_count": len(rows), "scene_count": 40}
        by_scene = defaultdict(list)
        for row in rows:
            by_scene[row["scene_sha256"]].append(row)
        require(len(by_scene) == 40 and all(len(group) == REPEATS[role]
                                            for group in by_scene.values()),
                "逐场景重复动作耗时数量不符")
        for field in stages:
            one = percentile_summary([row[field] for row in rows])
            summed = percentile_summary([sum(row[field] for row in group)
                                         for group in by_scene.values()])
            result.update({field + "_" + key: value for key, value in one.items()})
            result.update({field + "_scene_sum_" + key: value for key, value in summed.items()})
        summary.append(result)
    return raw_rows, summary


def load_training_curves(jobs, inputs):
    curves = {}
    required_nodes = set(range(0, 400001, 20000))
    required_counts = {50, 80, 100, 120, 160, 200}
    for role in ("V0", "V1"):
        directory = jobs[f"{role}_train"]["directory"]
        episodes_path = directory / "episodes.jsonl"
        counts_path = directory / "count_metrics.jsonl"
        inputs[str(episodes_path.resolve())] = sha256_file(episodes_path)
        inputs[str(counts_path.resolve())] = sha256_file(counts_path)
        episodes = []
        last_step = -1
        for row in read_jsonl(episodes_path):
            step = row.get("env_step")
            n = row.get("beam_count")
            value = row.get("return")
            require(type(step) is int and step > last_step and step <= 400000 and
                    type(n) is int and 50 <= n <= 200 and
                    isinstance(value, (int, float)) and math.isfinite(value),
                    f"{role}训练回合日志步数/N/回报无效")
            last_step = step
            episodes.append({"env_step": step, "beam_count": n, "return": float(value),
                             "terminal": row.get("terminal")})
        require(episodes and last_step == 400000, f"{role}训练回合日志缺400000终点")
        validation = {}
        for row in read_jsonl(counts_path):
            require(row.get("group") == "validation" and row.get("env_step") in required_nodes,
                    f"{role}验证节点越界")
            step = row["env_step"]
            require(step not in validation, f"{role}验证节点重复")
            by_n = row.get("by_beam_count") or {}
            require({int(n) for n in by_n} == required_counts,
                    f"{role}验证缺N分档")
            for n, values in by_n.items():
                require(values.get("scene_actions") == 24,
                        f"{role}验证N={n}不是12母景×2负载")
                for metric in ("final_si_mean", "final_sgm_mean"):
                    value = (values.get("metrics") or {}).get(metric)
                    require(isinstance(value, (int, float)) and math.isfinite(value),
                            f"{role}验证节点缺{metric}")
            validation[step] = by_n
        require(set(validation) == required_nodes, f"{role}验证不是0–400000共21节点")
        curves[role] = {"episodes": episodes, "validation": validation}
    return curves


def _write_csv(path, rows):
    require(rows, f"输出表为空：{path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _line_by_n(ax, records, stage, metric, *, roles=ROLES):
    for role in roles:
        ys = []
        for n in N_VALUES:
            selected = [row[metric] for row in records if row["stage"] == stage and
                        row["role"] == role and row["beam_count"] == n]
            require(len(selected) == 40, f"绘图数据缺{role}/{stage}/N={n}")
            ys.append(statistics.mean(selected))
        ax.plot(N_VALUES, ys, marker="o", markersize=3, label=role)
    ax.set(xlabel="Beam count N", ylabel=metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)


def render_figures(output, records, pairs, timing_rows, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    for metric, title in (("si", "Demand satisfaction SI"), ("sgm", "Supply-demand match SGM"),
                          ("basic_fraction", "Basic-service fraction"),
                          ("covered_demand_fraction", "Covered demand fraction"),
                          ("power_w", "Total power W")):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, constrained_layout=True)
        for ax, stage, label in zip(axes, ("direct", "final"), ("Stage 2 direct", "Final SCA B")):
            _line_by_n(ax, records, stage, metric)
            ax.set_title(f"{title} · {label}")
            if metric == "power_w":
                ax.axhline(3000, linestyle="--", color="black", linewidth=0.8)
        path = output / f"{metric}_by_n.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures.append(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, stage in zip(axes, ("direct", "final")):
        means = [statistics.mean(row["si_delta_V1_minus_V0"] for row in pairs
                                 if row["stage"] == stage and row["beam_count"] == n)
                 for n in N_VALUES]
        ax.bar(N_VALUES, means, width=5, color=["#2a9d8f" if value >= 0 else "#d66b5d" for value in means])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Beam count N", ylabel="Paired V1 - V0 SI", title=stage)
    path = output / "paired_si_delta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for role in ROLES:
        ys = [statistics.mean(row["total_pipeline_seconds"] for row in timing_rows
                              if row["role"] == role and row["beam_count"] == n)
              for n in N_VALUES]
        ax.plot(N_VALUES, ys, marker="o", markersize=3, label=role)
    ax.set(xlabel="Beam count N", ylabel="Full route seconds per action", title="Measured initialization + A + B + checks")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    path = output / "total_time_by_n.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for role, color in (("V0", "#5277b8"), ("V1", "#df8b34")):
        episodes = curves[role]["episodes"]
        x = [row["env_step"] for row in episodes]
        y = [row["return"] for row in episodes]
        axes[0].plot(x, y, color=color, alpha=0.18, linewidth=0.5, label=f"{role} raw")
        window = 100
        smooth = np.convolve(y, np.ones(window) / window, mode="valid") if len(y) >= window else []
        if len(smooth):
            axes[0].plot(x[window - 1:], smooth, color=color, linewidth=1.5,
                         label=f"{role} {window}-episode mean")
        for n in (50, 100, 200):
            vx = sorted(curves[role]["validation"])
            vy = [curves[role]["validation"][step][str(n)]["metrics"]["final_si_mean"]
                  for step in vx]
            axes[1].plot(vx, vy, marker=".", label=f"{role} N={n}")
    axes[0].set(xlabel="Training interactions", ylabel="Episode return", title="Seed 51001; raw and smoothed")
    axes[1].set(xlabel="Training interactions", ylabel="Validation mean SI by N", title="144 scenes at 0 and every 20k steps")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7, ncol=3)
    path = output / "training_validation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for role in ("V0", "V1"):
        for n in (50, 100, 200):
            vx = sorted(curves[role]["validation"])
            vy = [curves[role]["validation"][step][str(n)]["metrics"]["final_sgm_mean"]
                  for step in vx]
            ax.plot(vx, vy, marker=".", label=f"{role} N={n}")
    ax.set(xlabel="Training interactions", ylabel="Validation mean SGM by N", title="Supply-demand match on 144 scenes")
    ax.legend(fontsize=7, ncol=3)
    path = output / "validation_sgm.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)
    return figures


def _n_average(records, role, stage, n, metric):
    values = [row[metric] for row in records if row["role"] == role and
              row["stage"] == stage and row["beam_count"] == n]
    require(len(values) == 40, f"报告表缺{role}/{stage}/N={n}")
    return statistics.mean(values)


def write_report(path, manifest, records, pairs, timing_summary, physics_checked):
    direct_held = statistics.mean(_n_average(records, "V1", "direct", n, "si") -
                                  _n_average(records, "V0", "direct", n, "si")
                                  for n in HELD_OUT)
    final_held = statistics.mean(_n_average(records, "V1", "final", n, "si") -
                                 _n_average(records, "V0", "final", n, "si")
                                 for n in HELD_OUT)
    n100_si = _n_average(records, "V1", "final", 100, "si") - _n_average(records, "V0", "final", 100, "si")
    n100_sgm = _n_average(records, "V1", "final", 100, "sgm") - _n_average(records, "V0", "final", 100, "sgm")
    local_failures = [n for n in HELD_OUT if _n_average(records, "V1", "final", n, "si") -
                      _n_average(records, "V0", "final", n, "si") < -0.01]
    continuation = direct_held > 0 and final_held > 0
    lines = ["# S2-P17 可变波束数量共同考核与功率细调报告" +
             ("" if physics_checked == 19360 else "（物理链路复算未完成）"), "",
             f"科学提交：`{manifest['scientific_commit']}`。正式目标：10.160.17.102:20424。",
             "训练仅seed51001，两个新模型各400000交互；考核为2024-01-29同日20个母快照，",
             "每母景11个束数×2需求档，共440衍生场景。V0/V1/H0每景3次动作，",
             "固定/贪心每景1次；动作种子53001–53003。重复动作和衍生N不是独立母场景。", "",
             "## 证据和预算", "",
             "工作台完成800000训练交互、6048次训练验证分配、4840次共同初始分配、",
             "14520条第三阶段方法记录（原分配none、SCA中间A、最终B各4840）。",
             "主终态比较只取4840条第二阶段直接结果与4840条最终B，共9680条；A只作中间证据。",
             "按每个衍生场景先平均3次模型动作，再按N、负载和20母快照汇总；基线每景只有一次。",
             "所有成绩均按该场景全部N个目标束计算，包含未服务束和后续干扰。不同奖励的训练回报不跨组排名。", "",
             "逐SHA验证了场景、冻结任务、初解case、SCA案例和源A引用。逐束独立复算D、C、",
             "min(C,D)、C/D、SI、SGM、基本服务和总功率；未记录的字段不补零。",
             (f"物理链路容量全量复算：已核{physics_checked}/19360个终态。" if physics_checked == 19360 else
              "物理链路容量全量复算：未完成；本报告只能作为算术与来源审计后的阶段结果。"), "",
             "## 同N共同结果", "",
             "| 编号＋实验组合 | N | 第二阶段SI | 最终B SI | 第二阶段SGM | 最终B SGM |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for role in ROLES:
        for n in N_VALUES:
            lines.append(f"| {LABELS[role]} | {n} | "
                         f"{_n_average(records, role, 'direct', n, 'si'):.6f} | "
                         f"{_n_average(records, role, 'final', n, 'si'):.6f} | "
                         f"{_n_average(records, role, 'direct', n, 'sgm'):.6f} | "
                         f"{_n_average(records, role, 'final', n, 'sgm'):.6f} |")
    lines += ["", "同场景V1−V0的四个未见束数等权平均SI差：",
              f"第二阶段{direct_held:+.6f}、最终B{final_held:+.6f}；"
              f"100束最终SI差{n100_si:+.6f}、SGM差{n100_sgm:+.6f}。",
              ("按方案初轮门槛，混合训练值得继续观察。" if continuation else
               "按方案初轮门槛，混合训练未同时提高第二阶段与最终B的未见N平均SI。"),
              ("最终B逐N SI退步超过1个百分点的留出N：" + ", ".join(map(str, local_failures)) + "。"
               if local_failures else "四个留出N没有最终SI退步超过1个百分点的局部失败。"),
              "100束保留门槛分别为SI退步≤0.5个百分点、SGM退步≤0.01；"
              "每个N与负载的完整数值见by_n_load.csv。", "",
              "## 约束、覆盖和容量", "",
              "逐束资源与数值检查均通过后才生成本报告。covered_aircraft_fraction和",
              "covered_demand_fraction来自冻结母景/布局元数据，50束可能覆盖较少；",
              "不能只用已覆盖束SI推断整体用户业务改善。有效承载为环境估计min(C,D)，不是实测流量。",
              "基本服务比例、零容量比例、总功率、完整分配比例与供需匹配程度按N及负载",
              "列在per_scene_means.csv和by_n_load.csv。", "",
              "## 运行时间", "",
              "初解完整耗时包含环境准备、动作推理和终态核验；A求解及核验、B追加求解及核验",
              "分别取实测段，完整时间为共享初解和分段实测合计。公共初解与A只执行一次，",
              "各路线账上引用同一实测时间；无调整的none并非模型生成时间。",
              "逐N的单次动作、同场景重复动作合计的均值/中位/P95及原值见timing_summary.csv、timing_raw.csv。", "",
              "| 编号＋实验组合 | N | 初解均时s | A求解均时s | B追加均时s | 完整均时/中位/P95s |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in timing_summary:
        lines.append(f"| {row['combination']} | {row['beam_count']} | "
                     f"{row['initial_allocation_seconds_mean']:.4f} | "
                     f"{row['reference_optimizer_wall_seconds_mean']:.4f} | "
                     f"{row['optimizer_wall_seconds_mean']:.4f} | "
                     f"{row['total_pipeline_seconds_mean']:.4f}/"
                     f"{row['total_pipeline_seconds_median']:.4f}/"
                     f"{row['total_pipeline_seconds_p95']:.4f} |")
    lines += ["", "## 图表", "",
              "![SI随束数变化](si_by_n.png)", "![SGM随束数变化](sgm_by_n.png)",
              "![基本服务束比例](basic_fraction_by_n.png)",
              "![业务需求覆盖比例](covered_demand_fraction_by_n.png)",
              "![总功率](power_w_by_n.png)",
              "![逐场景配对SI差](paired_si_delta.png)",
              "![完整耗时](total_time_by_n.png)",
              "![训练原始回报与验证SI](training_validation.png)",
              "![验证SGM](validation_sgm.png)", "",
              "训练回报图保留每回合原始波动，粗线为100回合移动平均；训练/验证/独立考核",
              "分别标注，验证21节点只用于诊断，主模型固定400000步终点。", "",
              "## 限制与后续", "",
              "本轮只有单训练种子、单考核日期、20个母快照，四个留出N是训练范围内",
              "未见具体束数的插值证据，不是超过200束的外推结论。H0仅作为15万步历史锚点，",
              "与V0/V1的40万步预算不同。固定/贪心初解来自原规则加共同可行域投影，",
              "不是V0模型学到的分配。SCA是局部可行细调，不声称全局最优。",
              "本脚本只读原始资产，输入与所有产物SHA见artifact_manifest.json。"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(manifest_path, output_dir, *, skip_physics=False):
    inputs = {}
    manifest = load_manifest(manifest_path, inputs)
    datasets = load_datasets(manifest, inputs)
    jobs = load_jobs(manifest, datasets, inputs)
    refs = datasets["test"]["files"]
    direct = load_evaluations(jobs, refs, inputs)
    refined = load_refinements(jobs, refs, direct, inputs)
    candidate_hashes = {stages["sca_si_lp"]["case_path"]:
                        stages["sca_si_lp"]["highest_si_power_sha256"]
                        for stages in refined.values()}
    require(len(candidate_hashes) == 4840 and all(candidate_hashes.values()),
            "最高SI候选SHA缺失")
    curves = load_training_curves(jobs, inputs)
    records, by_n_load, pairs = summarize_common(refs, direct, refined)
    timing_rows, timing_summary = summarize_timing(refs, direct, refined)
    physics_checked, candidate_checked = ((0, 0) if skip_physics else
                                          audit_physics(jobs, refs, direct, refined))
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "报告目标目录已存在；为保留旧证据，请指定新目录")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s2_p17_report_pending_", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        _write_csv(stage / "per_scene_means.csv", records)
        _write_csv(stage / "by_n_load.csv", by_n_load)
        _write_csv(stage / "paired_v1_minus_v0.csv", pairs)
        _write_csv(stage / "timing_raw.csv", timing_rows)
        _write_csv(stage / "timing_summary.csv", timing_summary)
        figures = render_figures(stage, records, pairs, timing_rows, curves)
        write_report(stage / "report.md", manifest, records, pairs,
                     timing_summary, physics_checked)
        script_path = Path(__file__).resolve()
        inputs[str(script_path)] = sha256_file(script_path)
        outputs = {path.name: sha256_file(path) for path in sorted(stage.iterdir()) if path.is_file()}
        receipt = {"experiment_id": EXPERIMENT, "scientific_commit": manifest["scientific_commit"],
                   "status": "complete" if physics_checked == 19360 else "physics_audit_incomplete",
                   "training_interactions": 800000, "validation_allocations": 6048,
                   "independent_initial_allocations": 4840,
                   "workbench_refinement_method_records": 14520,
                   "main_terminal_records": 9680,
                   "physical_terminal_recalculations": physics_checked,
                   "highest_si_candidate_recalculations": candidate_checked,
                   "inputs_sha256": dict(sorted(inputs.items())),
                   "highest_si_candidate_power_sha256": dict(sorted(candidate_hashes.items())),
                   "outputs_sha256": outputs,
                   "figure_count": len(figures)}
        (stage / "artifact_manifest.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        stage.rename(output_dir)
    return {"report": str(output_dir / "report.md"),
            "receipt": str(output_dir / "artifact_manifest.json"),
            "status": receipt["status"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-physics", action="store_true",
                        help="Only for a marked incomplete arithmetic/source audit")
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.controller_manifest, args.output_dir,
                              skip_physics=args.skip_physics), ensure_ascii=False))


if __name__ == "__main__":
    main()
