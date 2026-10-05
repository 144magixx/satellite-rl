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
           "effective_carried_bps", "power_w", "slots_total", "zero_slot_fraction",
           "complete_fraction")
SCENE_FIELDS = ("sha256", "beam_count", "mother_scene_id", "load_multiplier",
                "partition_role", "selected_time_utc", "source_sha256")
SCENE_SEMANTIC_FIELDS = tuple(key for key in SCENE_FIELDS if key != "sha256")
REPAIR_ID = "sca_basic_threshold_exact_v1"
FAILED_RUN = "refine_sac_20261004_215930_e814197c"
FAILED_CASE = "scene_00033_r0_sca_si_lp.json"
FAILED_CALLS_LOWER_BOUND = 684
FAILED_SECONDS_LOWER_BOUND = 1.0373822273686528
INTERRUPTED_RUN = "refine_sac_20261005_014002_932212d0"
INTERRUPTED_REQUEST = "s2p17-v1-jan29-sca"
INTERRUPTED_SCENE_INDEX = 146
INTERRUPTED_CASE = "scene_00146_r0_sca"
INTERRUPTION_SCHEMA = "s2p17_user_pause_interruption_v1"
INTERRUPTION_FILES = ("V1_status.json", "V1_refinement_checkpoint.json",
                      "V1_refinement_manifest.json", "V1_power_iterations.jsonl",
                      "V1_prefix_sha256.json", "V1_job.json", "user_pause.json")


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


def verify_reimported_files(prepared, imported, inputs, role):
    """Allow NPZ recompression only with exact source SHA and equal typed arrays."""
    old_refs, new_refs = prepared.get("files"), imported.get("files")
    require(isinstance(old_refs, list) and isinstance(new_refs, list) and
            len(old_refs) == len(new_refs), f"{role}准备→导入场景数量不符")
    source_files = ((imported.get("source") or {}).get("reimport") or {}).get("files")
    require(isinstance(source_files, list) and
            [item.get("sha256") for item in source_files] ==
            [item.get("sha256") for item in old_refs],
            f"{role}工作台重导入缺原NPZ逐SHA来源链")
    for index, (old, new) in enumerate(zip(old_refs, new_refs)):
        require(all(old.get(key) == new.get(key) for key in SCENE_SEMANTIC_FIELDS),
                f"{role}准备→导入场景N/母景/负载/日期/来源语义错配：{index}")
        old_path, new_path = Path(old["path"]), Path(new["path"])
        old_sha, new_sha = sha256_file(old_path), sha256_file(new_path)
        require(old_sha == old.get("sha256") and new_sha == new.get("sha256"),
                f"{role}准备或导入NPZ文件SHA错误：{index}")
        inputs[str(old_path.resolve())] = old_sha
        inputs[str(new_path.resolve())] = new_sha
        with np.load(old_path, allow_pickle=False) as left, np.load(new_path, allow_pickle=False) as right:
            require(set(left.files) == set(right.files) and all(
                left[key].dtype == right[key].dtype and np.array_equal(left[key], right[key])
                for key in left.files), f"{role}准备→导入NPZ数组不等价：{index}")


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


def _si_mean(capacities, demands):
    """Match the frozen physical core's float64 vector division and mean."""
    values = np.minimum(np.asarray(capacities, dtype=np.float64) /
                        np.asarray(demands, dtype=np.float64), 1.0)
    return float(values.mean()), float(values.sum())


def _phase2_si_guard(best, best_feasible, locked_floor, a_si, b_si):
    """Check the original inclusive SI floor, including a raised phase-two anchor."""
    return (all(isinstance(value, (int, float)) and math.isfinite(value)
                for value in (best, best_feasible, locked_floor)) and
            best >= a_si - 1e-8 and best_feasible >= best and
            locked_floor == best_feasible - 1e-8 and
            b_si >= best - 1e-8 and b_si >= locked_floor)


def audit_beams(beams, metrics, expected_n, *, continuous=False):
    """Independently recompute every terminal D/C/service/SI/SGM/resource field."""
    require(isinstance(beams, list) and len(beams) == expected_n, "终态目标束数量不完整")
    require([beam.get("beam_id") for beam in beams] == list(range(expected_n)),
            "终态束顺序或身份错位")
    original = [beam.get("original_index") for beam in beams]
    require(len(set(original)) == expected_n, "原始波束身份重复")
    demands, capacities, powers, slot_counts = [], [], [], []
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
        group, freq = beam.get("group"), beam.get("freq")
        require(type(group) is int and 0 <= group < 8,
                "逐束频率复用组无效")
        require((freq is None or type(freq) is int and 0 <= freq < 100
                 if slots == 0 else
                 type(freq) is int and 0 <= freq < 100 and freq + slots <= 100),
                "逐束频率槽起点或范围无效")
        if slots == 0:
            require(power <= 1e-8 and capacity <= 1e-3, "空槽束出现功率或容量")
        demands.append(demand)
        capacities.append(capacity)
        powers.append(power)
        slot_counts.append(slots)
    beta = statistics.mean(demands)
    sgm = [_sgm(c, d, beta) for c, d in zip(capacities, demands)]
    for index, (beam, value) in enumerate(zip(beams, sgm)):
        close(beam.get("sgm"), value, f"第{index}束SGM", atol=1e-7)
    si_mean, si_total = _si_mean(capacities, demands)
    basic = sum(c >= 0.1 * d for c, d in zip(capacities, demands))
    zero = sum(c <= 0 for c in capacities)
    power_total = sum(powers)
    slots_total = sum(slot_counts)
    zero_slots = sum(count == 0 for count in slot_counts)
    require(power_total <= 3000 + 1e-6, "终态超过3000W总预算")
    close(metrics.get("final_si_mean"), si_mean, "终态平均SI")
    close(metrics.get("final_sgm_mean"), statistics.mean(sgm), "终态平均SGM")
    close(metrics.get("power_w"), power_total, "终态总功率", atol=1e-4)
    if "allocated_slots" in metrics:
        require(metrics["allocated_slots"] == slots_total, "终态分配槽数与逐束结果不一致")
    if "actual_slots" in metrics:
        require(metrics["actual_slots"] == slots_total, "终态实际槽数与逐束结果不一致")
    if "zero_slot_count" in metrics:
        require(metrics["zero_slot_count"] == zero_slots, "终态零槽束数与逐束结果不一致")
    require(metrics.get("basic_met_count") == basic, "基本服务束数与逐束结果不一致")
    if "zero_capacity_count" in metrics:
        require(metrics["zero_capacity_count"] == zero, "零容量束数与逐束结果不一致")
    if "effective_carried_bps" in metrics:
        close(metrics["effective_carried_bps"], sum(min(c, d) for c, d in zip(capacities, demands)),
              "有效承载总量", atol=1e-3)
    if "final_si_total" in metrics:
        close(metrics["final_si_total"], si_total, "SI总分")
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
    return {"si": si_mean, "sgm": statistics.mean(sgm),
            "basic_fraction": basic / expected_n, "zero_capacity_fraction": zero / expected_n,
            "effective_carried_bps": sum(min(c, d) for c, d in zip(capacities, demands)),
            "power_w": power_total, "slots_total": slots_total,
            "zero_slot_fraction": zero_slots / expected_n,
            "complete_fraction": float(metrics.get("complete", True)),
            "beam_count": expected_n, "basic_count": basic, "zero_count": zero,
            "si_total": si_total, "sgm_total": sum(sgm)}


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
    repair = manifest.get("boundary_repair")
    if repair is not None:
        require(repair.get("repair_id") == REPAIR_ID and
                repair.get("origin_science_commit") == manifest["scientific_commit"] and
                repair.get("failed_run_id") == FAILED_RUN and
                isinstance(repair.get("repair_commit"), str) and
                len(repair["repair_commit"]) == 40 and
                repair["repair_commit"] != manifest["scientific_commit"],
                "SCA边界修复来源身份错误")
        for path_key, hash_key in (("repair_proof_path", "repair_proof_sha256"),
                                   ("switch_contract_path", "switch_contract_sha256")):
            evidence_path = Path(repair.get(path_key, ""))
            evidence_sha = sha256_file(evidence_path)
            require(evidence_sha == repair.get(hash_key), f"边界修复{path_key} SHA不符")
            inputs[str(evidence_path.resolve())] = evidence_sha
        proof = json_file(repair["repair_proof_path"])
        require(proof.get("repair_id") == REPAIR_ID and
                proof.get("origin_science_commit") == manifest["scientific_commit"] and
                proof.get("repair_commit") == repair["repair_commit"] and
                proof.get("fallback_contract_sha256") == repair.get("fallback_contract_sha256"),
                "SCA边界修复证明与控制清单不一致")
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
            require(len(prepared.get("files", [])) == count,
                    f"{role}准备场景数量不符")
            verify_reimported_files(prepared, record, inputs, role)
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


def _job_files_from_manifest(entry):
    """Controller paths name files, not the job directory."""
    job_path = Path(entry["job_path"]).resolve()
    directory = job_path.parent
    status_path = Path(entry["status_path"]).resolve()
    result_path = Path(entry["results_path"]).resolve()
    require(job_path.name == "job.json" and directory.name == entry["run_id"] and
            directory.parent.name == "jobs" and
            status_path == directory / "status.json" and
            result_path == directory / "results.json",
            "工作台任务文件路径与运行目录不一致")
    return directory, job_path, status_path, directory / "config.json", result_path


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
        directory, job_path, status_path, config_path, result_path = _job_files_from_manifest(entry)
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
        repair = manifest.get("boundary_repair")
        expected_commit = (repair["repair_commit"] if repair and name.endswith("_sca")
                           else manifest["scientific_commit"])
        require(job.get("git_commit") == expected_commit,
                f"{name}实际冻结源码提交不一致")
        source_path = Path(job.get("source_root", "")) / "source_manifest.json"
        source_sha = sha256_file(source_path)
        source_info = json_file(source_path)
        require(source_info.get("git_commit") == expected_commit,
                f"{name}冻结源码清单提交不一致")
        inputs[str(source_path.resolve())] = source_sha
        if repair and name.endswith("_sca"):
            require(job.get("scientific_origin_commit") == manifest["scientific_commit"],
                    f"{name}原科学来源提交缺失")
            if name == "V0_sca":
                proof = job.get("boundary_repair") or {}
                require(job.get("run_id") == FAILED_RUN and
                        job.get("original_git_commit") == manifest["scientific_commit"] and
                        proof.get("repair_id") == REPAIR_ID and
                        proof.get("repair_commit") == expected_commit and
                        proof.get("origin_science_commit") == manifest["scientific_commit"] and
                        proof.get("fallback_contract_sha256") == repair["fallback_contract_sha256"] and
                        proof.get("new_source_root") == job.get("source_root") and
                        proof.get("new_manifest_sha256") == source_sha and
                        Path(proof.get("archive_path", "")) / "repair_proof.json" ==
                        Path(repair["repair_proof_path"]) and
                        json_file(repair["repair_proof_path"]) == proof,
                        "V0原任务修复证明不一致")
                old_source_path = Path(proof.get("old_source_root", "")) / "source_manifest.json"
                old_source_sha = sha256_file(old_source_path)
                require(old_source_sha == proof.get("old_manifest_sha256") and
                        json_file(old_source_path).get("git_commit") == manifest["scientific_commit"],
                        "V0原冻结源码清单SHA不符")
                inputs[str(old_source_path.resolve())] = old_source_sha
            else:
                require(job.get("repair_profile") == REPAIR_ID and
                        job.get("repair_reference_run_id") == FAILED_RUN and
                        job.get("repair_proof_sha256") == repair["repair_proof_sha256"],
                        f"{name}修复来源证明缺失")
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
            "allocation_layout": [(beam["group"], beam["freq"], beam["slots"])
                                  for beam in result["beams"]],
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


def _load_failed_fallback(manifest, jobs, inputs):
    repair = manifest.get("boundary_repair")
    if not repair:
        return None
    directory = jobs["V0_sca"]["directory"]
    path = directory / "failed_attempt_fallback.json"
    digest = sha256_file(path)
    require(digest == repair.get("fallback_contract_sha256"), "失败B回退合同SHA不符")
    inputs[str(path.resolve())] = digest
    contract = json_file(path)
    exact = {"repair_type": REPAIR_ID,
             "origin_science_commit": manifest["scientific_commit"],
             "original_run_id": FAILED_RUN,
             "request_id": "s2p17-v0-jan29-sca",
             "case_key": Path(FAILED_CASE).stem,
             "scene_sha": "d7b8af50073632dcaeaa69b9af229feb4859d74a3e17f17f7b16cf092f3c0e8d",
             "initial_sha": "9fd3d48531cbdd5e29d63ae0cc24f2bdeb491e6996009c740a147b80c49d4e35",
             "A_case_sha256": "55398b6907276b949373712dbb3aa7da6388e96958e20b7a3ed857e6dcf170c9",
             "failure_sha256": "7fbacd63da943f4539df73dd8fd014552b9f0ae1c2646d363557ee710b8a9a41",
             "iterations_sha256": "ad3f1c46e2488d96180aea8d228bc431f85c25f587dbc25e0699b7fd4e00812e"}
    require(all(contract.get(key) == value for key, value in exact.items()) and
            contract.get("fallback") == "failed_attempt_fallback_to_A" and
            contract.get("completed_prefix_case_count") == 299 and
            set(contract.get("unknown_exact_fields", [])) ==
            {"original_attempt_capacity_calls", "original_attempt_optimizer_wall_seconds"} and
            bool(contract.get("missing_reason")), "失败B唯一case合同身份不符")
    bounds = contract.get("callback_lower_bounds") or {}
    require(bounds.get("capacity_calls") == FAILED_CALLS_LOWER_BOUND and
            bounds.get("accepted_callbacks") == 49 and
            isinstance(bounds.get("elapsed_seconds"), (int, float)) and
            math.isclose(bounds["elapsed_seconds"], FAILED_SECONDS_LOWER_BOUND,
                         rel_tol=0, abs_tol=1e-12), "失败B原始尝试下界不符")
    recovery_root = (directory / "recoveries").resolve()
    for path_key, hash_key in (("failure_path", "failure_sha256"),
                               ("iterations_path", "iterations_sha256")):
        evidence = Path(contract[path_key]).resolve()
        require(evidence.is_relative_to(recovery_root) and
                sha256_file(evidence) == contract[hash_key], "失败B原始证据SHA不符")
        inputs[str(evidence)] = contract[hash_key]
    a_path = (directory / "power_cases" / "scene_00033_r0_sca.json").resolve()
    require(Path(contract["A_case_path"]).resolve() == a_path and
            sha256_file(a_path) == contract["A_case_sha256"], "失败B公共A证据SHA不符")
    return {"path": str(path.resolve()), "sha256": digest, "content": contract}


def _check_timing(row, method, first, *, fallback=False):
    if fallback:
        require("timing" not in row, "失败B精确总耗时不能补造")
        return None
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


def _read_refinement_case(path, role, method, ref, repeat, source_case, inputs,
                          *, expected_commit=None, fallback=None):
    file_sha = sha256_file(path)
    inputs[str(path.resolve())] = file_sha
    row = json_file(path)
    if expected_commit is not None:
        require(row.get("git_commit") == expected_commit, f"SCA案例源码来源错配：{path}")
    is_fallback = fallback is not None
    require(bool(row.get("failed_attempt_fallback")) == is_fallback,
            f"SCA回退标志仅允许唯一失败B：{path}")
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
    elapsed = solver.get("elapsed_seconds")
    if is_fallback:
        evidence = row["failed_attempt_fallback"]
        contract = fallback["content"]
        require(method == "sca_si_lp" and path.name == FAILED_CASE and
                role == "V0" and repeat == 0 and
                row.get("source_scene_sha256") == contract["scene_sha"] and
                row.get("initial_allocation_sha256") == contract["initial_sha"] and
                evidence.get("repair_type") == REPAIR_ID and
                Path(evidence.get("contract_path", "")).resolve() == Path(fallback["path"]) and
                evidence.get("contract_sha256") == fallback["sha256"] and
                evidence.get("source_A_case_sha256") == contract["A_case_sha256"] and
                evidence.get("original_attempt_capacity_calls_lower_bound") ==
                FAILED_CALLS_LOWER_BOUND and
                evidence.get("original_attempt_optimizer_seconds_lower_bound") ==
                FAILED_SECONDS_LOWER_BOUND and
                evidence.get("original_attempt_exact_cost_unknown") is True and
                evidence.get("additional_optimizer_calls") == 0 and
                evidence.get("missing_reason") == contract["missing_reason"] and
                count is None and elapsed is None and
                solver.get("termination_reason") == "failed_attempt_fallback_to_A" and
                solver.get("fallback_used") is True and
                solver.get("converged") is False and
                solver.get("history") == [] and
                solver.get("highest_si_power_w") is None and
                solver.get("highest_si_metrics") is None and
                solver.get("phase1_best_si_mean") is None and
                solver.get("best_feasible_si_mean") is None and
                solver.get("additional_optimizer_calls") == 0 and
                solver.get("original_attempt_capacity_calls_exact") is None and
                solver.get("original_attempt_optimizer_seconds_exact") is None and
                solver.get("original_attempt_capacity_calls_lower_bound") ==
                FAILED_CALLS_LOWER_BOUND and
                solver.get("original_attempt_optimizer_seconds_lower_bound") ==
                FAILED_SECONDS_LOWER_BOUND and
                row.get("complete_route_solver_evaluations") is None,
                "失败B缺时/调用合同与唯一原尝试不符")
        require(isinstance(row.get("repair_verification_seconds"), (int, float)) and
                math.isfinite(row["repair_verification_seconds"]) and
                row["repair_verification_seconds"] >= 0,
                "失败B回退核验耗时缺失")
    else:
        require(type(count) is int and 0 <= count <= 2000,
                f"{method}容量调用超过2000或缺失")
        require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0,
                f"{method}求解耗时缺失")
        require(elapsed <= 31.0, f"{method}求解超过30秒硬时限加1秒记录容差")
    timing = _check_timing(row, method, source_case, fallback=is_fallback)
    highest_power = None
    highest_sha = None
    highest_metrics = None
    if method == "sca_si_lp" and not is_fallback:
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
            "allocation_layout": [(beam["group"], beam["freq"], beam["slots"])
                                  for beam in row["beams"]],
            "case_path": str(path), "case_sha256": file_sha,
            "solver_evaluations": count, "timing": timing,
            "source_commit": row.get("git_commit"), "failed_attempt_fallback": is_fallback,
            "repair_verification_seconds": row.get("repair_verification_seconds"),
            "complete_route_solver_evaluations_lower_bound":
                row.get("complete_route_solver_evaluations_lower_bound"),
            "solver": solver, "source_scene_sha256": ref["sha256"],
            "scene_metadata": ref, "repeat_index": repeat,
            "source_group": expected_group,
            "reference_case_sha256": row.get("reference_case_sha256"),
            "reference_allocation_sha256": row.get("reference_allocation_sha256"),
            "reference_solver_evaluations": row.get("reference_solver_evaluations"),
            "complete_route_solver_evaluations": row.get("complete_route_solver_evaluations"),
            "highest_si_power_sha256": highest_sha,
            "highest_si_mean": highest_metrics["si"] if highest_metrics else None}


def load_refinements(jobs, test_refs, direct, inputs, manifest=None):
    refined = {}
    repair = (manifest or {}).get("boundary_repair")
    failed = _load_failed_fallback(manifest, jobs, inputs) if repair else None
    original_inventory = {}
    if repair:
        proof = json_file(repair["repair_proof_path"])
        inventory = proof.get("original_299_inventory")
        expected_names = [f"scene_{index:05d}_r{repeat}_{method}.json"
                          for index in range(440) for repeat in range(3)
                          for method in ("none", "sca", "sca_si_lp")][:299]
        require(isinstance(inventory, list) and len(inventory) == 299 and
                [item.get("name") for item in inventory] == expected_names and
                canonical_hash(inventory) == proof.get("original_299_inventory_sha256") and
                proof.get("original_A_case_sha256") == failed["content"]["A_case_sha256"] and
                proof.get("original_failure_sha256") == failed["content"]["failure_sha256"] and
                proof.get("original_iterations_sha256") == failed["content"]["iterations_sha256"] and
                failed["content"].get("completed_prefix_inventory_sha256") ==
                proof.get("original_299_inventory_sha256"),
                "原299条案例清单或失败证据证明不符")
        original_inventory = {item["name"]: item["sha256"] for item in inventory}
        backup_root = Path(proof["archive_path"]) / "original"
        for item in inventory:
            backup = backup_root / "power_cases" / item["name"]
            require(sha256_file(backup) == item["sha256"],
                    "原299条案例恢复前备份SHA不符")
            inputs[str(backup.resolve())] = item["sha256"]
        for name, field in (("failure.json", "original_failure_sha256"),
                            ("power_iterations.jsonl", "original_iterations_sha256"),
                            ("status.json", "original_status_sha256")):
            backup = backup_root / name
            digest = sha256_file(backup)
            require(digest == proof.get(field), f"失败现场备份{name} SHA不符")
            inputs[str(backup.resolve())] = digest
    role_case_counts = {}
    for role in ROLES:
        name = SCA_JOB[role]
        directory = jobs[name]["directory"]
        job = jobs[name]["job"]
        case_dir = directory / "power_cases"
        manifest_path = directory / "refinement_manifest.json"
        refinement_manifest = json_file(manifest_path)
        require(refinement_manifest.get("budget_cases") == 440 * REPEATS[role] * 3,
                f"{name}细调案例预算不符")
        require([sha for _, sha in refinement_manifest.get("scene_files", [])] ==
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
                    ordinal = ((index * REPEATS[role] + repeat) * 3 +
                               ("none", "sca", "sca_si_lp").index(method))
                    original_row = role == "V0" and ordinal < 299
                    source_commit = (manifest["scientific_commit"] if original_row or not repair
                                     else repair["repair_commit"])
                    fallback = (failed if role == "V0" and path.name == FAILED_CASE else None)
                    stage = _read_refinement_case(path, role, method, ref, repeat, source, inputs,
                                                  expected_commit=source_commit, fallback=fallback)
                    if original_row:
                        require(stage["case_sha256"] == original_inventory[path.name],
                                "原299条案例SHA与恢复前冻结清单不符")
                    stages[method] = stage
                    count += 1
                raw, none, a, b = source, stages["none"], stages["sca"], stages["sca_si_lp"]
                require(all(stage["allocation_layout"] == raw["allocation_layout"]
                            for stage in (none, a, b)),
                        f"{role} SCA固定的复用组/频率起点/槽数与初解不同")
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
                require(b["reference_solver_evaluations"] == a["solver_evaluations"],
                        f"{role}最终B公共A调用账错误")
                if b["failed_attempt_fallback"]:
                    require(b["beams_sha256"] == a["beams_sha256"] and
                            b["solver"].get("power_w") == a["solver"].get("power_w") and
                            b["complete_route_solver_evaluations_lower_bound"] ==
                            a["solver_evaluations"] + FAILED_CALLS_LOWER_BOUND,
                            "失败B必须原样回退公共A并保留原尝试调用下界")
                else:
                    require(b["complete_route_solver_evaluations"] ==
                            a["solver_evaluations"] + b["solver_evaluations"] <= 4000,
                            f"{role}最终B完整A+B容量调用账错误")
                    best = b["solver"].get("phase1_best_si_mean")
                    require(_phase2_si_guard(best,
                                             b["solver"].get("best_feasible_si_mean"),
                                             b["solver"].get("phase2_si_floor"),
                                             a["metrics"]["si"], b["metrics"]["si"]),
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


def load_interruption_ledger(path, controller_path, manifest, jobs, test_refs,
                             direct, refined, inputs):
    """Bind the separate V1 user-pause attempt to immutable pre-pause evidence."""
    require(path is not None, "S2-P17修复版正式报告缺少V1主动暂停中断账本")
    path = Path(path).resolve()
    ledger_sha = sha256_file(path)
    ledger = json_file(path)
    inputs[str(path)] = ledger_sha
    job = jobs["V1_sca"]["job"]
    ref = test_refs[INTERRUPTED_SCENE_INDEX]
    key = ("V1", ref["sha256"], 0)
    stages = refined[key]
    original = direct[key]
    expected_snapshot = (Path(controller_path).resolve().parent /
                         "user_resume_20261004T185523Z").resolve()
    snapshot = Path(ledger.get("snapshot_dir", "")).resolve()
    require(ledger.get("schema_version") == INTERRUPTION_SCHEMA and
            ledger.get("experiment_id") == EXPERIMENT and
            ledger.get("event") == "user_requested_pause" and
            ledger.get("run_id") == INTERRUPTED_RUN == job.get("run_id") and
            ledger.get("request_id") == INTERRUPTED_REQUEST == job.get("request_id") and
            ledger.get("role") == "V1" and
            ledger.get("scene_index") == INTERRUPTED_SCENE_INDEX and
            ledger.get("repeat") == 0 and ledger.get("method") == "sca" and
            ledger.get("scene_id") == "scene_0000146_r0" and
            ledger.get("source_scene_sha256") == ref["sha256"] and
            ledger.get("source_initial_case_sha256") == original["case_sha256"] and
            ledger.get("initial_allocation_sha256") == original["beams_sha256"] and
            ledger.get("origin_science_commit") == manifest["scientific_commit"] and
            ledger.get("repair_commit") == manifest["boundary_repair"]["repair_commit"] and
            snapshot == expected_snapshot and
            isinstance(ledger.get("missing_reason"), str) and ledger["missing_reason"],
            "V1主动暂停账本身份/场景/初解/来源不符")
    hashes = ledger.get("files_sha256") or {}
    require(set(hashes) == set(INTERRUPTION_FILES), "V1暂停账本缺原始证据文件SHA")
    for name in INTERRUPTION_FILES:
        evidence = snapshot / name
        digest = sha256_file(evidence)
        require(digest == hashes[name], f"V1暂停原始证据SHA不符：{name}")
        inputs[str(evidence.resolve())] = digest
    saved_job = json_file(snapshot / "V1_job.json")
    status = json_file(snapshot / "V1_status.json")
    checkpoint = json_file(snapshot / "V1_refinement_checkpoint.json")
    paused_manifest = json_file(snapshot / "V1_refinement_manifest.json")
    prefix = json_file(snapshot / "V1_prefix_sha256.json")
    pause = json_file(snapshot / "user_pause.json")
    none = stages["none"]
    a = stages["sca"]
    b = stages["sca_si_lp"]
    expected_cases = {"none": none["case_sha256"], "sca": a["case_sha256"],
                      "sca_si_lp": b["case_sha256"]}
    expected_none_path = (jobs["V1_sca"]["directory"] / "power_cases" /
                          f"scene_{INTERRUPTED_SCENE_INDEX:05d}_r0_none.json").resolve()
    require(ledger.get("resumed_case_sha256") == expected_cases and
            none["case_sha256"] == ledger.get("pre_pause_last_none_sha256") and
            checkpoint.get("completed_cases") == status.get("env_step") == 1315 and
            Path(checkpoint.get("last_case", "")).resolve() == expected_none_path and
            checkpoint.get("last_case_sha256") == none["case_sha256"] and
            len(prefix) == 1315 and
            prefix.get(f"scene_{INTERRUPTED_SCENE_INDEX:05d}_r0_none.json") ==
            none["case_sha256"] and
            paused_manifest.get("identity") == checkpoint.get("identity") and
            paused_manifest.get("budget_cases") == 3960 and
            paused_manifest.get("scene_files", [])[INTERRUPTED_SCENE_INDEX][1] ==
            ref["sha256"] and
            saved_job.get("run_id") == job.get("run_id") and
            saved_job.get("request_id") == job.get("request_id") and
            saved_job.get("git_commit") == job.get("git_commit") and
            saved_job.get("scientific_origin_commit") == manifest["scientific_commit"] and
            status.get("run_id") == job.get("run_id") and
            status.get("stage") == "stopped" and
            status.get("scene_index") == INTERRUPTED_SCENE_INDEX and
            status.get("repeat") == 0 and status.get("method") == "sca" and
            (status.get("optimizer_progress") or {}).get("phase") == "si" and
            any(item.get("run_id") == INTERRUPTED_RUN and
                item.get("before_stage") == "evaluating"
                for item in pause.get("stop_requests", [])) and
            str(pause.get("reason", "")).startswith("User requested pause"),
            "V1暂停前断点、原none与续跑案例来源不一致")
    none_row = verified_json(none["case_path"], none["case_sha256"])
    require(none_row.get("source_scene_sha256") == ref["sha256"] and
            none_row.get("source_initial_case_sha256") == original["case_sha256"] and
            none_row.get("initial_allocation_sha256") == original["beams_sha256"] and
            b["reference_case_sha256"] == a["case_sha256"] and
            b["reference_allocation_sha256"] == a["beams_sha256"] and
            not any(stage["failed_attempt_fallback"] for stage in stages.values()),
            "V1暂停原none/A/B链路不完整或混入V0失败B回退")
    progress = status.get("optimizer_progress") or {}
    callback_rows = [row for row in read_jsonl(snapshot / "V1_power_iterations.jsonl")
                     if row.get("scene_id") == ledger["scene_id"] and
                     row.get("repeat") == 0 and row.get("method") == "sca"]
    require(callback_rows and all(row.get("case_identity") ==
                                  ledger.get("interrupted_case_identity")
                                  for row in callback_rows),
            "V1原暂停尝试回调身份缺失")
    last = callback_rows[-1]
    bounds = ledger.get("callback_lower_bounds") or {}
    require(type(bounds.get("capacity_calls")) is int and
            bounds["capacity_calls"] >= 571 and
            bounds["capacity_calls"] == progress.get("evaluation") == last.get("evaluation") and
            last.get("phase") == "si" and
            bounds.get("accepted_callbacks") == len(callback_rows) and
            isinstance(bounds.get("elapsed_seconds"), (int, float)) and
            bounds["elapsed_seconds"] >= 1.4651551842689514 and
            math.isclose(bounds["elapsed_seconds"], progress.get("elapsed_seconds"),
                         rel_tol=0, abs_tol=1e-12) and
            math.isclose(bounds["elapsed_seconds"], last.get("elapsed_seconds"),
                         rel_tol=0, abs_tol=1e-12),
            "V1原暂停A试算调用/时间下界与原回调日志不符")
    return {"ledger_path": str(path), "ledger_sha256": ledger_sha,
            "snapshot_dir": str(snapshot), "run_id": job["run_id"],
            "request_id": job["request_id"], "scene_index": INTERRUPTED_SCENE_INDEX,
            "scene_sha256": ref["sha256"], "scene_id": ledger["scene_id"],
            "repeat_index": 0, "method": "sca", "case_identity":
                ledger["interrupted_case_identity"],
            "saved_none_case_sha256": none["case_sha256"],
            "successful_A_case_sha256": a["case_sha256"],
            "successful_B_case_sha256": b["case_sha256"],
            "original_A_capacity_calls_lower_bound": bounds["capacity_calls"],
            "original_A_optimizer_seconds_lower_bound": bounds["elapsed_seconds"],
            "original_A_capacity_calls_exact": None,
            "original_A_optimizer_seconds_exact": None,
            "successful_attempt_A_capacity_calls": a["solver_evaluations"],
            "successful_attempt_B_capacity_calls": b["solver_evaluations"],
            "successful_attempt_complete_route_calls":
                b["complete_route_solver_evaluations"],
            "cumulative_A_capacity_calls_exact": None,
            "cumulative_A_capacity_calls_lower_bound":
                a["solver_evaluations"] + bounds["capacity_calls"],
            "cumulative_complete_route_calls_exact": None,
            "cumulative_complete_route_calls_lower_bound":
                b["complete_route_solver_evaluations"] + bounds["capacity_calls"],
            "missing_reason": ledger["missing_reason"],
            "original_status_sha256": hashes["V1_status.json"],
            "original_checkpoint_sha256": hashes["V1_refinement_checkpoint.json"],
            "original_iterations_sha256": hashes["V1_power_iterations.jsonl"],
            "pre_pause_snapshot_sha256": canonical_hash(hashes)}


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
                if highest["failed_attempt_fallback"]:
                    require(highest["beams_sha256"] == refined[key]["sca"]["beams_sha256"],
                            "失败B回退终态与冻结A逐束不同")
                    continue
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
    fallback_count = sum(stages["sca_si_lp"]["failed_attempt_fallback"]
                         for stages in refined.values())
    require(candidate_checked == 4840 - fallback_count and fallback_count <= 1,
            "最高SI候选复算数量与唯一缺失候选不符")
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


def summarize_timing(test_refs, direct, refined, interruptions=()):
    stages = ("initial_allocation_seconds", "physics_setup_seconds",
              "reference_optimizer_wall_seconds", "reference_verification_seconds",
              "optimizer_wall_seconds", "verification_seconds", "total_pipeline_seconds")
    raw_rows = []
    interruption_by_action = {(row["scene_sha256"], row["repeat_index"]): row
                              for row in interruptions}
    require(len(interruption_by_action) == len(interruptions), "主动暂停中断账本重复")
    for role in ROLES:
        for ref in test_refs:
            for repeat in range(REPEATS[role]):
                key = (role, ref["sha256"], repeat)
                source, final = direct[key], refined[key]["sca_si_lp"]
                fallback = final["failed_attempt_fallback"]
                interruption = (interruption_by_action.get((ref["sha256"], repeat))
                                if role == "V1" else None)
                require(not (fallback and interruption), "V0失败B与V1主动暂停不能合并")
                timing = final["timing"]
                if fallback:
                    a_timing = refined[key]["sca"]["timing"]
                    timing = {"physics_setup_seconds": a_timing["physics_setup_seconds"],
                              "reference_optimizer_wall_seconds": a_timing["optimizer_wall_seconds"],
                              "reference_verification_seconds": a_timing["verification_seconds"],
                              "optimizer_wall_seconds": None,
                              "verification_seconds": None,
                              "total_pipeline_seconds": None}
                successful_timing = timing if interruption else None
                if interruption:
                    timing = {**timing, "reference_optimizer_wall_seconds": None,
                              "total_pipeline_seconds": None}
                raw_rows.append({"role": role, "combination": LABELS[role],
                                 "scene_sha256": ref["sha256"],
                                 "mother_scene_id": ref["mother_scene_id"],
                                 "beam_count": ref["beam_count"],
                                 "load_multiplier": ref["load_multiplier"],
                                 "repeat_index": repeat,
                                 "initial_allocation_seconds": source["initial_allocation_seconds"],
                                 **{field: timing[field] for field in stages if field != "initial_allocation_seconds"},
                                 "timing_missing_reason": (
                                     "原失败B已完成但未保存精确求解与全路线时间；仅有原callback下界"
                                     if fallback else interruption["missing_reason"]
                                     if interruption else None),
                                 "timing_loss_kind": ("failed_B_fallback_to_A" if fallback else
                                                      "user_pause_interrupted_A" if interruption else None),
                                 "original_B_calls_lower_bound":
                                     FAILED_CALLS_LOWER_BOUND if fallback else None,
                                 "original_B_optimizer_seconds_lower_bound":
                                     FAILED_SECONDS_LOWER_BOUND if fallback else None,
                                 "total_pipeline_seconds_lower_bound":
                                     (a_timing["total_pipeline_seconds"] +
                                      FAILED_SECONDS_LOWER_BOUND if fallback else
                                      successful_timing["total_pipeline_seconds"] +
                                      interruption["original_A_optimizer_seconds_lower_bound"]
                                      if interruption else None),
                                 "original_A_calls_lower_bound": (
                                     interruption["original_A_capacity_calls_lower_bound"]
                                     if interruption else None),
                                 "cumulative_A_calls_lower_bound": (
                                     interruption["cumulative_A_capacity_calls_lower_bound"]
                                     if interruption else None),
                                 "cumulative_complete_route_calls_lower_bound": (
                                     interruption["cumulative_complete_route_calls_lower_bound"]
                                     if interruption else None),
                                 "original_A_optimizer_seconds_lower_bound": (
                                     interruption["original_A_optimizer_seconds_lower_bound"]
                                     if interruption else None),
                                 "cumulative_A_optimizer_seconds_lower_bound": (
                                     successful_timing["reference_optimizer_wall_seconds"] +
                                     interruption["original_A_optimizer_seconds_lower_bound"]
                                     if interruption else None),
                                 "successful_attempt_A_optimizer_seconds": (
                                     successful_timing["reference_optimizer_wall_seconds"]
                                     if interruption else None),
                                 "successful_attempt_initial_allocation_seconds": (
                                     source["initial_allocation_seconds"] if interruption else None),
                                 "successful_attempt_physics_setup_seconds": (
                                     successful_timing["physics_setup_seconds"]
                                     if interruption else None),
                                 "successful_attempt_A_verification_seconds": (
                                     successful_timing["reference_verification_seconds"]
                                     if interruption else None),
                                 "successful_attempt_B_optimizer_seconds": (
                                     successful_timing["optimizer_wall_seconds"]
                                     if interruption else None),
                                 "successful_attempt_B_verification_seconds": (
                                     successful_timing["verification_seconds"]
                                     if interruption else None),
                                 "successful_attempt_total_pipeline_seconds": (
                                     successful_timing["total_pipeline_seconds"]
                                     if interruption else None),
                                 "interruption_ledger_sha256": (
                                     interruption["ledger_sha256"] if interruption else None),
                                 "repair_verification_seconds":
                                     final["repair_verification_seconds"] if fallback else None,
                                 "sca_source_commit": final["source_commit"]})
    require(len(raw_rows) == 4840, "完整耗时原始动作数量不符")
    require(sum(row["timing_loss_kind"] == "user_pause_interrupted_A"
                for row in raw_rows) == len(interruptions), "V1主动暂停案例未唯一进入缺时统计")
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
        route_complete = [group for group in by_scene.values()
                          if all(row["total_pipeline_seconds"] is not None for row in group)]
        require(route_complete, f"{role}/N={n}无完整路线计时场景")
        for field in stages:
            complete = [row[field] for row in rows if row[field] is not None]
            scene_complete = [sum(row[field] for row in group)
                              for group in by_scene.values()
                              if all(row[field] is not None for row in group)]
            require(complete and scene_complete, f"{role}/N={n}/{field}无完整计时样本")
            one = percentile_summary(complete)
            summed = percentile_summary(scene_complete)
            result.update({field + "_" + key: value for key, value in one.items()})
            result.update({field + "_scene_sum_" + key: value for key, value in summed.items()})
            scene_mean = percentile_summary([value / REPEATS[role] for value in scene_complete])
            result.update({field + "_scene_mean_" + key: value
                           for key, value in scene_mean.items()})
            result[field + "_complete_count"] = len(complete)
            result[field + "_missing_count"] = len(rows) - len(complete)
            result[field + "_scene_sum_complete_count"] = len(scene_complete)
            result[field + "_scene_sum_missing_count"] = len(by_scene) - len(scene_complete)
            route_means = percentile_summary([
                sum(row[field] for row in group) / REPEATS[role]
                for group in route_complete])
            result.update({field + "_route_scene_mean_" + key: value
                           for key, value in route_means.items()})
            result[field + "_route_scene_complete_count"] = len(route_complete)
            result[field + "_route_scene_missing_count"] = len(by_scene) - len(route_complete)
        lower = [row for row in rows if row["total_pipeline_seconds_lower_bound"] is not None]
        result["total_pipeline_lower_bound_count"] = len(lower)
        result["total_pipeline_lower_bound_min_seconds"] = (
            min(row["total_pipeline_seconds_lower_bound"] for row in lower) if lower else None)
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


def _timing_curve_values(timing_summary, role):
    by_n = {row["beam_count"]: row for row in timing_summary if row["role"] == role}
    require(set(by_n) == set(N_VALUES), f"{role}完整场景计时曲线缺N")
    return [by_n[n]["total_pipeline_seconds_route_scene_mean_mean"] for n in N_VALUES]


def render_figures(output, records, pairs, timing_summary, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    for metric, title in (("si", "Demand satisfaction SI"), ("sgm", "Supply-demand match SGM"),
                          ("basic_fraction", "Basic-service fraction"),
                          ("covered_demand_fraction", "Covered demand fraction"),
                          ("power_w", "Total power W"),
                          ("slots_total", "Allocated frequency slots across 8 groups"),
                          ("zero_slot_fraction", "Zero-slot beam fraction")):
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
        ys = _timing_curve_values(timing_summary, role)
        ax.plot(N_VALUES, ys, marker="o", markersize=3, label=role)
    ax.set(xlabel="Beam count N", ylabel="Seconds per action in complete scenes",
           title="Complete-scene mean: initialization + A + B + checks")
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


def write_report(path, manifest, records, pairs, timing_summary, physics_checked,
                 interruptions=()):
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
             ("（原失败B精确耗时缺失）" if manifest.get("boundary_repair") else
              "" if physics_checked == 19360 else "（物理链路复算未完成）"), "",
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
             "min(C,D)、C/D、SI、SGM、基本服务、总功率和8组复用下的分配槽数；未记录的字段不补零。",
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
    lines += ["", "## 频率槽与功率资源", "",
              "槽数为8个复用组内逐束分配槽数之和，不是去重后的频率编号数；",
              "每组均使用原100槽频率轴。SCA只调整功率，逐束复用组、起始频率和槽数与初解逐一核对。", "",
              "| 编号＋实验组合 | N | 第二阶段功率 W | 最终B功率 W | 第二阶段槽数 | 最终B槽数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for role in ROLES:
        for n in N_VALUES:
            lines.append(f"| {LABELS[role]} | {n} | "
                         f"{_n_average(records, role, 'direct', n, 'power_w'):.3f} | "
                         f"{_n_average(records, role, 'final', n, 'power_w'):.3f} | "
                         f"{_n_average(records, role, 'direct', n, 'slots_total'):.3f} | "
                         f"{_n_average(records, role, 'final', n, 'slots_total'):.3f} |")
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
              "主表和总时间曲线先在每个场景核齐全部重复动作，再计算场景内单次平均；",
              "任何一个动作缺时，则该场景全部重复动作不进入主计时比较。",
              "单动作已观测样本和同场景重复动作合计的均值/中位/P95及完整/缺失数仍列在timing_summary.csv；",
              "逐动作实测段、下界与缺失原因见timing_raw.csv。", "",
              "| 编号＋实验组合 | N | 初解均时s | 物理准备均时s | A求解均时s | A核验均时s | B追加均时s | B核验均时s | 完整均时/中位/P95s | 单动作完整/缺失 | 场景完整/缺失 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in timing_summary:
        lines.append(f"| {row['combination']} | {row['beam_count']} | "
                     f"{row['initial_allocation_seconds_route_scene_mean_mean']:.4f} | "
                     f"{row['physics_setup_seconds_route_scene_mean_mean']:.4f} | "
                     f"{row['reference_optimizer_wall_seconds_route_scene_mean_mean']:.4f} | "
                     f"{row['reference_verification_seconds_route_scene_mean_mean']:.4f} | "
                     f"{row['optimizer_wall_seconds_route_scene_mean_mean']:.4f} | "
                     f"{row['verification_seconds_route_scene_mean_mean']:.4f} | "
                     f"{row['total_pipeline_seconds_route_scene_mean_mean']:.4f}/"
                     f"{row['total_pipeline_seconds_route_scene_mean_median']:.4f}/"
                     f"{row['total_pipeline_seconds_route_scene_mean_p95']:.4f} | "
                     f"{row['total_pipeline_seconds_complete_count']}/"
                     f"{row['total_pipeline_seconds_missing_count']} | "
                     f"{row['total_pipeline_seconds_scene_sum_complete_count']}/"
                     f"{row['total_pipeline_seconds_scene_sum_missing_count']} |")
    repair = manifest.get("boundary_repair")
    if repair:
        lines += ["", "## 边界修复来源与缺时限制", "",
                  f"训练和第二阶段共同初解使用原科学提交`{manifest['scientific_commit']}`；"
                  f"五项SCA任务的修复实现提交为`{repair['repair_commit']}`，"
                  f"冻结证明SHA为`{repair['repair_proof_sha256']}`。",
                  "V0细调原任务前299条方法记录保留原实现，其中99条B完整路线；"
                  "第300条scene_00033_r0的B使用已保存A作为失败回退，后续V0及另四项SCA使用修复实现。",
                  "原失败B的优化已执行，但最终功率向量、精确容量调用数、精确优化/全路线耗时未保存；"
                  "回退额外优化调用为0，原回调日志仅证明至少684次容量调用、1.037382秒。",
                  "失败B回退终态计入全部4840个最终动作及同场景三动作性能均值；"
                  "其完整时间和对应同场景三动作时间缺失，不能以恢复核验时间或其余动作代替。",
                  "本次候选判断修复可能改变边界行为，混合实现与缺时限制使实验不能称为无行为差异或完整计时交付。",
                  "各计时段的均值/中位/P95只针对完整样本，精确数量及单列下界见CSV。"]
    if interruptions:
        interruption = interruptions[0]
        lines += ["", "## V1用户主动暂停的独立计时缺口", "",
                  f"V1工作台任务`{interruption['run_id']}`在scene_00146_r0的A试算中按用户要求暂停；"
                  f"原迭代日志SHA为`{interruption['original_iterations_sha256']}`。",
                  f"原尝试至少{interruption['original_A_capacity_calls_lower_bound']}次容量调用、"
                  f"{interruption['original_A_optimizer_seconds_lower_bound']:.9f}秒，精确调用数及耗时未保存。",
                  "续跑成功的none/A/B终态与实测分段继续严格核验，且全部计入SI、SGM、约束和逐场景性能。",
                  "timing_raw.csv保留成功尝试的A、B及其完整路线实测时间；另列原中断尝试下界。",
                  "原A累计求解时间和该动作全程总耗时为null，仅列下界；该场景三动作时间",
                  "不参与主时间表与曲线。此事与V0唯一失败B回退无关，详细证据见interruptions.csv。"]
    lines += ["", "## 图表", "",
              "![SI随束数变化](si_by_n.png)", "![SGM随束数变化](sgm_by_n.png)",
              "![基本服务束比例](basic_fraction_by_n.png)",
              "![业务需求覆盖比例](covered_demand_fraction_by_n.png)",
              "![总功率](power_w_by_n.png)",
              "![8组复用下的分配槽数](slots_total_by_n.png)",
              "![零槽束比例](zero_slot_fraction_by_n.png)",
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


def generate(manifest_path, output_dir, *, interruption_ledger=None, skip_physics=False):
    inputs = {}
    manifest = load_manifest(manifest_path, inputs)
    require(bool(manifest.get("boundary_repair")) == bool(interruption_ledger),
            "本轮S2-P17修复报告必须提供且仅提供V1主动暂停中断账本")
    datasets = load_datasets(manifest, inputs)
    jobs = load_jobs(manifest, datasets, inputs)
    refs = datasets["test"]["files"]
    direct = load_evaluations(jobs, refs, inputs)
    refined = load_refinements(jobs, refs, direct, inputs, manifest)
    interruptions = ([load_interruption_ledger(interruption_ledger, manifest_path,
                                               manifest, jobs, refs, direct, refined, inputs)]
                     if interruption_ledger else [])
    candidate_hashes = {stages["sca_si_lp"]["case_path"]:
                        stages["sca_si_lp"]["highest_si_power_sha256"]
                        for stages in refined.values()}
    missing_candidate_count = sum(value is None for value in candidate_hashes.values())
    require(len(candidate_hashes) == 4840 and
            missing_candidate_count == (1 if manifest.get("boundary_repair") else 0),
            "最高SI候选SHA缺失数量与唯一失败B合同不符")
    curves = load_training_curves(jobs, inputs)
    records, by_n_load, pairs = summarize_common(refs, direct, refined)
    timing_rows, timing_summary = summarize_timing(refs, direct, refined, interruptions)
    if interruptions:
        require(sum(row["total_pipeline_seconds"] is None for row in timing_rows) == 2,
                "V0失败B与V1主动暂停须各保留一条精确总耗时缺失")
    interruption_rows = []
    for interruption in interruptions:
        measured = next(row for row in timing_rows
                        if row["role"] == "V1" and
                        row["scene_sha256"] == interruption["scene_sha256"] and
                        row["repeat_index"] == interruption["repeat_index"])
        interruption_rows.append({**interruption,
            "successful_attempt_initial_allocation_seconds":
                measured["successful_attempt_initial_allocation_seconds"],
            "successful_attempt_physics_setup_seconds":
                measured["successful_attempt_physics_setup_seconds"],
            "successful_attempt_A_optimizer_seconds":
                measured["successful_attempt_A_optimizer_seconds"],
            "successful_attempt_A_verification_seconds":
                measured["successful_attempt_A_verification_seconds"],
            "successful_attempt_B_optimizer_seconds":
                measured["successful_attempt_B_optimizer_seconds"],
            "successful_attempt_B_verification_seconds":
                measured["successful_attempt_B_verification_seconds"],
            "successful_attempt_total_pipeline_seconds":
                measured["successful_attempt_total_pipeline_seconds"],
            "cumulative_A_optimizer_seconds_exact": None,
            "cumulative_A_optimizer_seconds_lower_bound":
                measured["cumulative_A_optimizer_seconds_lower_bound"],
            "cumulative_total_pipeline_seconds_exact": None,
            "cumulative_total_pipeline_seconds_lower_bound":
                measured["total_pipeline_seconds_lower_bound"]})
    provenance_rows = [{"role": role, "scene_sha256": ref["sha256"],
                        "repeat_index": repeat, "method": method,
                        "case_sha256": refined[(role, ref["sha256"], repeat)][method]["case_sha256"],
                        "source_commit": refined[(role, ref["sha256"], repeat)][method]["source_commit"],
                        "failed_attempt_fallback":
                            refined[(role, ref["sha256"], repeat)][method]["failed_attempt_fallback"]}
                       for role in ROLES for ref in refs for repeat in range(REPEATS[role])
                       for method in ("none", "sca", "sca_si_lp")]
    require(len(provenance_rows) == 14520, "逐案例源码来源记录不完整")
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
        _write_csv(stage / "refinement_provenance.csv", provenance_rows)
        if interruption_rows:
            _write_csv(stage / "interruptions.csv", interruption_rows)
        figures = render_figures(stage, records, pairs, timing_summary, curves)
        write_report(stage / "report.md", manifest, records, pairs,
                     timing_summary, physics_checked, interruptions)
        script_path = Path(__file__).resolve()
        inputs[str(script_path)] = sha256_file(script_path)
        outputs = {path.name: sha256_file(path) for path in sorted(stage.iterdir()) if path.is_file()}
        receipt = {"experiment_id": EXPERIMENT, "scientific_commit": manifest["scientific_commit"],
                   "status": ("results_audited_timing_incomplete" if
                              physics_checked == 19360 and manifest.get("boundary_repair") else
                              "complete" if physics_checked == 19360 else "physics_audit_incomplete"),
                   "training_interactions": 800000, "validation_allocations": 6048,
                   "independent_initial_allocations": 4840,
                   "workbench_refinement_method_records": 14520,
                   "main_terminal_records": 9680,
                   "physical_terminal_recalculations": physics_checked,
                   "highest_si_candidate_recalculations": candidate_checked,
                   "highest_si_candidate_missing_count": missing_candidate_count,
                   "boundary_repair": manifest.get("boundary_repair"),
                   "exact_total_time_missing_actions": sum(
                       row["total_pipeline_seconds"] is None for row in timing_rows),
                   "independent_user_pause_interruption_count": len(interruptions),
                   "interruption_ledger_sha256": (interruptions[0]["ledger_sha256"]
                                                  if interruptions else None),
                   "refinement_old_source_records": sum(
                       row["source_commit"] == manifest["scientific_commit"]
                       for row in provenance_rows),
                   "refinement_repair_source_records": sum(
                       row["source_commit"] == (manifest.get("boundary_repair") or {}).get("repair_commit")
                       for row in provenance_rows) if manifest.get("boundary_repair") else 0,
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
    parser.add_argument("--interruption-ledger", type=Path,
                        help="Required SHA-bound V1 user-pause ledger for this repaired S2-P17 report")
    parser.add_argument("--skip-physics", action="store_true",
                        help="Only for a marked incomplete arithmetic/source audit")
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.controller_manifest, args.output_dir,
                              interruption_ledger=args.interruption_ledger,
                              skip_physics=args.skip_physics), ensure_ascii=False))


if __name__ == "__main__":
    main()
