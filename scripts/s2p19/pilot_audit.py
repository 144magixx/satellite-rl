"""Build an S2-P19 pilot-gate audit solely from completed, frozen workbench evidence.

Input is a JSON file with ``experiment_id=S2-P19``, ``arm`` (C0/C1/I0),
``greedy``, ``greedy_B``, and either ``C0``, ``C1``, ``C0_B``, ``C1_B``
or ``I0``, ``I0_B``. Each task source has ``run_id`` and ``job_dir``.
C0/C1 must both be stopped at 100000 before either may resume. Run
``--freeze-inputs --output <target-archive-root>`` after all B jobs complete;
it copies pilot checkpoints and completed-task identity files into immutable
``pilot_audit_inputs`` and writes a new sources_<arm>.json with snapshot refs.
Then run ``--build --sources <that-file> --output <target-archive/audit.json>``.
Build reads only those snapshots plus saved validation and completed SCA cases. It never trains,
infers, optimizes, calls the API, or sets passed without checking criteria.

The output JSON contains the exact fields required by approve-pilot:
experiment_id, run_id, checkpoint_sha256, passed, criteria, source_files.
The controller still needs a local gate receipt pointing to this remote audit
path and its SHA; a failed audit must become gate_failed without calling API.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import mean

import numpy as np


EXPERIMENT = "S2-P19"
N_VALUES = (50, 80, 100, 120, 160, 200)
LOADS = (0.75, 1.25)
LARGE_N = frozenset((120, 160, 200))
BEAM_KEYS = ("beam_location", "beam_G_Tx_max", "UE_G_Rx_max", "distances",
             "rate", "noise_temperature", "original_indices")
PHYSICS_FILES = ("src/satellite_rl/envs/satellite.py",
                 "src/satellite_rl/envs/regions.py",
                 "src/satellite_rl/workbench/environment.py",
                 "src/satellite_rl/power_refinement/physics.py",
                 "src/satellite_rl/power_refinement/optimizer.py",
                 "src/satellite_rl/power_refinement/priority.py",
                 "src/satellite_rl/power_refinement/config.py")


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file(), f"缺少已保存证据：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON根对象无效：{path}")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                  indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _config_hash(config):
    from satellite_rl.workbench.checkpoints import config_hash
    return config_hash(config)


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"{label}缺失或非有限")
    return float(value)


def scene_content(path):
    """Full physical-array hash; ZIP wrapper bytes are separately pinned."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as source:
        require(set(source.files) == set(BEAM_KEYS), "源NPZ物理字段不完整")
        for key in sorted(BEAM_KEYS):
            arr = np.asarray(source[key])
            require(np.issubdtype(arr.dtype, np.number) and np.isfinite(arr).all(),
                    f"源NPZ {key}非有限数值")
            array = np.asarray(arr, dtype="<i8" if key == "original_indices" else "<f8",
                               order="C")
            h.update(key.encode())
            h.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
            h.update(array.tobytes())
    return h.hexdigest()


def _source_files_add(ledger, path, expected=None):
    path = Path(path).resolve(strict=True)
    current = sha(path)
    if expected is not None:
        require(current == expected, f"来源SHA变化：{path}")
    previous = ledger.get(str(path))
    require(previous in (None, current), f"同路径来源SHA冲突：{path}")
    ledger[str(path)] = current
    return current


def freeze_pilot_inputs(sources, output_root):
    """Copy *both* C0/C1 stopped pilot checkpoints before either can resume."""
    arm = sources.get("arm")
    require(arm in ("C0", "C1", "I0"), "pilot arm无效")
    pilots = ("C0", "C1") if arm in ("C0", "C1") else ("I0",)
    required = (*pilots, "greedy", "greedy_B", *(("C0_B", "C1_B") if arm in ("C0", "C1")
                                                else ("I0_B",)))
    output_root = Path(output_root).resolve()
    target_archive = Path(sources[arm]["job_dir"]).resolve(strict=True)
    target_job = read_json(target_archive / "job.json")
    require(output_root.is_relative_to(Path(target_job["archive_root"]).resolve(strict=True)),
            "审计来源快照须保存于目标工作台archive")
    snapshots = {}
    for name in required:
        spec = sources.get(name)
        require(isinstance(spec, dict) and spec.get("job_dir") and spec.get("run_id"),
                f"{name}试点来源缺run_id/job_dir")
        directory = Path(spec["job_dir"]).resolve(strict=True)
        job, status = (read_json(directory / file) for file in ("job.json", "status.json"))
        is_pilot = name in pilots
        checkpoint = directory / "checkpoint.pt"
        checkpoint_sha = sha(checkpoint) if is_pilot else None
        step = status.get("env_step")
        require(job.get("run_id") == spec["run_id"] and status.get("run_id") == spec["run_id"],
                f"{name}工作台身份不符")
        if is_pilot:
            require(status.get("stage") == "stopped" and
                    status.get("stop_reason") == "s2p19_pilot_gate_pending" and
                    status.get("checkpoint_sha256", checkpoint_sha) == checkpoint_sha and
                    status.get("checkpoint_step") == step and
                    (step == 100000 if name != "I0" else
                     type(step) is int and 0 < step <= 100000),
                    f"{name}当前不是可冻结的真实试点断点")
        else:
            require(status.get("stage") == "completed" and
                    (directory / "results.json").is_file(),
                    f"{name}基线/完整B尚未完成")
        identity_sha = checkpoint_sha or sha(directory / "results.json")
        target = output_root / "pilot_audit_inputs" / spec["run_id"] / identity_sha[:16]
        target.mkdir(parents=True, exist_ok=True)
        bindings = []
        filenames = ("job.json", "config.json", "status.json", "checkpoint.pt") if is_pilot else (
            "job.json", "config.json", "status.json", "results.json")
        for filename in filenames:
            original = directory / filename
            before = sha(original)
            saved = target / filename
            if saved.exists():
                require(sha(saved) == before, f"不可变试点副本冲突：{saved}")
            else:
                pending = saved.with_name(saved.name + ".incomplete")
                shutil.copy2(original, pending)
                require(sha(pending) == before and sha(original) == before,
                        f"试点副本前后SHA不一致：{filename}")
                pending.replace(saved)
            bindings.append({"original_path": str(original), "original_sha256": before,
                             "snapshot_path": str(saved), "snapshot_sha256": before})
        manifest = {"schema": "s2p19_pilot_audit_inputs_v1", "run_id": spec["run_id"],
                    "arm": name, "observed_utc": datetime.now(timezone.utc).isoformat(),
                    "observed_pid": status.get("pid"), "env_step": step,
                    "update_step": status.get("update_step", status.get("updates")),
                    "checkpoint_sha256": checkpoint_sha, "results_sha256":
                    sha(directory / "results.json") if not is_pilot else None,
                    "bindings": bindings}
        receipt = target / "manifest.json"
        if receipt.exists():
            old = read_json(receipt)
            require(old.get("run_id") == manifest["run_id"] and
                    old.get("checkpoint_sha256") == checkpoint_sha and
                    old.get("env_step") == step and old.get("bindings") == bindings and
                    old.get("results_sha256") == manifest["results_sha256"],
                    "试点快照manifest身份不一致")
        else:
            write_json(receipt, manifest)
        snapshots[name] = {"path": str(target), "manifest_path": str(receipt),
                           "manifest_sha256": sha(receipt), "checkpoint_sha256": checkpoint_sha,
                           "results_sha256": manifest["results_sha256"], "env_step": step}
    return snapshots


def _bundle(sources, name, ledger, *, snapshot=None):
    spec = sources.get(name)
    require(isinstance(spec, dict) and spec.get("job_dir") and spec.get("run_id"),
            f"缺实际{name}工作台任务引用")
    directory = Path(spec["job_dir"]).resolve(strict=True)
    if snapshot:
        frozen = Path(snapshot["path"]).resolve(strict=True)
        require(_source_files_add(ledger, snapshot["manifest_path"],
                                  snapshot["manifest_sha256"]) == snapshot["manifest_sha256"],
                "试点快照manifest SHA不符")
        manifest = read_json(snapshot["manifest_path"])
        require(manifest["run_id"] == spec["run_id"] and
                manifest["checkpoint_sha256"] == snapshot["checkpoint_sha256"] and
                manifest.get("results_sha256") == snapshot.get("results_sha256"),
                "试点快照身份不符")
        for binding in manifest["bindings"]:
            _source_files_add(ledger, binding["snapshot_path"], binding["snapshot_sha256"])
        source = frozen
    else:
        source = directory
        for filename in ("job.json", "config.json", "status.json"):
            _source_files_add(ledger, directory / filename)
    job, config, status = (read_json(source / filename) for filename in
                           ("job.json", "config.json", "status.json"))
    require(job.get("run_id") == spec["run_id"] and status.get("run_id") == spec["run_id"]
            and job.get("experiment_id") == EXPERIMENT,
            f"{name}工作台身份与来源引用不符")
    archive = Path(job["archive_root"]).resolve(strict=True)
    require(directory == archive / "outputs" / "workbench" / "jobs" / spec["run_id"],
            f"{name}来源不是归档中的正式工作台任务")
    source_root = Path(job["source_root"])
    frozen_source = read_json(source_root / "source_manifest.json")
    _source_files_add(ledger, source_root / "source_manifest.json")
    require(frozen_source.get("git_commit") == job.get("git_commit"),
            f"{name}科学源码提交不符")
    physics_files = {key: (frozen_source.get("files") or {}).get(key)
                     for key in PHYSICS_FILES}
    for key, expected in physics_files.items():
        require(isinstance(expected, str) and len(expected) == 64,
                f"{name}冻结物理源文件缺SHA：{key}")
        _source_files_add(ledger, source_root / key, expected)
    return {"name": name, "directory": directory, "source": source,
            "job": job, "config": config, "status": status,
            "snapshot": snapshot, "checkpoint_sha256":
            sha(source / "checkpoint.pt") if snapshot and (source / "checkpoint.pt").is_file()
            else None,
            "physics_files": physics_files}


def _validation_set(bundle, ledger):
    dataset = (bundle["job"].get("scene_sets") or {}).get("validation")
    require(isinstance(dataset, dict) and dataset.get("count") == 72 and
            dataset.get("profile") == "europe_v1" and
            (dataset.get("environment_profile") or {}).get("id") == "europe_v1" and
            (dataset.get("source") or {}).get("partition_role") == "validation" and
            (dataset.get("source") or {}).get("source_dates") == ["2024-01-12"] and
            (not dataset.get("units") or
             all((dataset.get("units") or {}).get(key) == unit for key, unit in
                 (("rate", "bit/s"), ("beam_location", "degrees"),
                  ("covered_demand_bps", "bit/s")))),
            "试点验证集不是冻结欧洲72景")
    environment = bundle["config"].get("environment") or {}
    i0 = bundle["config"].get("i0") or {}
    require(environment.get("profile", i0.get("profile")) == "europe_v1" and
            environment.get("power_budget_w", i0.get("power_budget_w")) == 3000 and
            environment.get("basic_fraction", i0.get("basic_fraction")) == .1,
            "物理profile/功率/基本服务合同不符")
    files = dataset["files"]
    require(len(files) == 72, "验证NPZ目录数量不符")
    rows = {}
    for index, ref in enumerate(files):
        path = Path(ref["path"]).resolve(strict=True)
        _source_files_add(ledger, path, ref["sha256"])
        metadata = (ref["selected_time_utc"], int(ref["beam_count"]),
                    float(ref["load_multiplier"]), ref["source_sha256"])
        require(ref.get("partition_role") == "validation" and
                metadata[0].startswith("2024-01-12") and
                metadata not in rows and metadata[1] in N_VALUES and metadata[2] in LOADS,
                "验证母/N/负载重复或范围错误")
        rows[metadata] = {"scene_sha256": ref["sha256"],
                          "content_sha256": scene_content(path), "metadata": ref,
                          "scene_index": index}
    by_mother = defaultdict(list)
    for key in rows:
        by_mother[(key[0], key[3])].append((key[1], key[2]))
    require(len(by_mother) == 6 and all(set(values) ==
            {(n, load) for n in N_VALUES for load in LOADS} for values in by_mother.values()),
            "验证集不是6母×6N×2负载")
    return rows


def physical_result(beams, metrics, count):
    require(isinstance(beams, list) and len(beams) == count,
            "保存case缺全N逐束终值")
    ratios, match, watts, demands, capacities = [], [], [], [], []
    occupied = {g: set() for g in range(8)}
    basic = 0
    for index, beam in enumerate(beams):
        require(beam.get("beam_id") == index and beam.get("group") == index % 8,
                "逐束ID/复用组不符")
        slots, freq = beam.get("slots"), beam.get("freq")
        require(type(slots) is int and 0 <= slots <= 10 and
                ((slots == 0 and freq is None) or
                 (type(freq) is int and 0 <= freq <= 100 - slots)), "频率块无效")
        if freq is None:
            freq = 0
        for slot in range(freq, freq + slots):
            require(slot not in occupied[index % 8], "组内频率槽冲突")
            occupied[index % 8].add(slot)
        demand = finite(beam.get("required_bps"), "D")
        capacity = finite(beam.get("rate_bps", beam.get("capacity_bps")), "C")
        power = finite(beam.get("power_w"), "W")
        sgm = finite(beam.get("sgm"), "逐束SGM")
        require(demand > 0 and capacity >= 0 and power >= 0 and 0 <= sgm <= 1 and
                (slots > 0 or power == capacity == 0), "逐束服务/功率非法")
        rho = capacity / demand
        if beam.get("capacity_demand_ratio") is not None:
            require(abs(rho - finite(beam["capacity_demand_ratio"], "rho")) <= 1e-7,
                    "保存逐束rho不符")
        ratios.append(min(rho, 1.0)); match.append(sgm); watts.append(power)
        demands.append(demand); capacities.append(capacity)
        basic += capacity >= .1 * demand
    si, sgm = mean(ratios), mean(match)
    beta = mean(demands)
    expected_match = []
    for capacity, demand in zip(capacities, demands):
        if capacity <= 0:
            expected_match.append(0.0)
            continue
        ratio = capacity / demand
        real = ratio - 1 if ratio >= 1 else 1 - 1 / ratio
        imag = (capacity - demand) / beta
        expected_match.append(1 - (1 - math.exp(-math.hypot(real, imag))) ** 3)
    require(all(abs(saved - actual) <= 1e-7 for saved, actual in
                zip(match, expected_match)), "逐束SGM与容量/需求复算不符")
    require(abs(si - finite(metrics.get("si", metrics.get("final_si_mean")), "SI")) <= 1e-7 and
            abs(sgm - finite(metrics.get("sgm", metrics.get("final_sgm_mean")), "SGM")) <= 1e-7 and
            sum(watts) <= 3000 + 1e-6, "保存终态SI/SGM或功率账不符")
    saved_basic = metrics.get("basic_met", metrics.get("basic_met_count"))
    if saved_basic is not None:
        require(abs(basic - finite(saved_basic, "basic")) <= 1e-8,
                "严格基本服务束数不符")
    if metrics.get("power_w") is not None:
        require(abs(sum(watts) - finite(metrics["power_w"], "总功率")) <= 1e-5,
                "逐束功率与摘要不符")
    return {"si": si, "sgm": sgm, "basic_met": basic}


def _direct(bundle, validation, ledger):
    name, job, status = bundle["name"], bundle["job"], bundle["status"]
    result = {}
    if name in ("C0", "C1"):
        step = 100000
        require(status.get("env_step") == step and status.get("stage") == "stopped" and
                status.get("stop_reason") == "s2p19_pilot_gate_pending",
                f"{name}未冻结真实100k试点")
        kind = "inline"
    elif name == "I0":
        step = status.get("env_step")
        require(type(step) is int and 0 < step <= 100000 and
                status.get("stage") == "stopped" and
                status.get("stop_reason") == "s2p19_pilot_gate_pending",
                "I0未冻结实际步数试点")
        kind = "i0_validation"
    else:
        require(status.get("stage") == "completed" and job.get("mode") == "baselines",
                "同源B1验证基线未完成")
        step = None  # A non-learning baseline has no pilot checkpoint step.
        kind = "greedy"
    if kind == "i0_validation":
        summary = bundle["directory"] / "validation" / f"step_{step:09d}" / "summary.json"
        summary_data = read_json(summary)
        _source_files_add(ledger, summary)
        require(summary_data.get("env_step") == step and
                summary_data.get("checkpoint_sha256") == bundle["checkpoint_sha256"] and
                summary_data.get("count") == 72 and len(summary_data.get("cases") or []) == 72,
                "I0验证summary未绑定真实断点/72景")
    elif kind == "greedy":
        summary = bundle["source"] / "results.json"
        summary_data = read_json(summary)
        _source_files_add(ledger, summary)
        require(summary_data.get("mode") == "baselines" and
                summary_data.get("partition") == "validation" and
                summary_data.get("methods") == ["greedy"] and
                summary_data.get("method_repeats") == {"greedy": 1} and
                summary_data.get("source_set_id") ==
                (job.get("scene_sets") or {}).get("validation", {}).get("id"),
                "同源贪心验证任务用途/来源集不符")
        selected = [row for row in summary_data.get("scenes", []) if row.get("method") == "greedy"]
        require(len(selected) == 72, "同源贪心未保存72个case")
    policy_sha = set()
    for key, reference in validation.items():
        index = reference["scene_index"]
        if kind == "inline":
            path = bundle["directory"] / "validation_cases" / f"step_{step:09d}" / f"scene_{index:05d}_r0.json"
            envelope = read_json(path)
            body = envelope.get("body") or {}
            require(envelope.get("sha256") == canonical_sha(body), "inline验证case封套SHA不符")
            contract = body.get("contract") or {}
            require(contract.get("env_step") == step and contract.get("scene_index") == index and
                    contract.get("repeat_index") == 0 and
                    contract.get("source_scene_sha256") == reference["scene_sha256"] and
                    contract.get("action_seed") == 53001 + index and
                    contract.get("config_hash") == _config_hash(bundle["config"]),
                    "C0/C1 inline验证身份/动作种子不符")
            require(isinstance(contract.get("policy_sha256"), str) and
                    len(contract["policy_sha256"]) == 64,
                    "inline验证缺冻结actor权重SHA")
            policy_sha.add(contract["policy_sha256"])
            case = body.get("result") or {}
            case_scene = (case.get("scene") or {}).get("source_sha256")
            require(case_scene == reference["scene_sha256"], "inline结果场景SHA不符")
        elif kind == "i0_validation":
            item = summary_data["cases"][index]
            path = Path(item["path"])
            require(path.name == f"scene_{index:05d}.json" and
                    sha(path) == item["sha256"], "I0验证case引用/SHA不符")
            case = read_json(path)
            require(case.get("checkpoint_sha256") == bundle["checkpoint_sha256"] and
                    case.get("env_step") == step and case.get("action_seed") == 53001 and
                    case.get("seed") == 53001 and case.get("repeat_index") == 0 and
                    case.get("scene_metadata") == reference["metadata"] and
                    case.get("scene_sha256") == reference["scene_sha256"],
                    "I0验证case断点/场景/种子不符")
        else:
            item = selected[index]
            path = Path(item["case_path"])
            require(item.get("scene_sha256") == reference["scene_sha256"] and
                    item.get("action_seed") == 53001 and item.get("repeat_index") == 0 and
                    item.get("method") == "greedy" and sha(path) == item["case_sha256"],
                    "贪心验证case摘要身份/SHA不符")
            case = read_json(path)
            require(case.get("action_seed") == 53001 and case.get("seed") == 53001 and
                    case.get("repeat_index") == 0 and case.get("method") == "greedy" and
                    case.get("scene_metadata") == reference["metadata"] and
                    case.get("scene_sha256") == reference["scene_sha256"],
                    "贪心case身份不符")
        _source_files_add(ledger, path)
        observed = physical_result(case["beams"], case["metrics"], key[1])
        result[key] = {**observed, "case_path": str(path), "case_sha256": sha(path),
                       "scene_sha256": reference["scene_sha256"],
                       "content_sha256": reference["content_sha256"],
                       "mother": key[0], "n": key[1], "load": key[2]}
    require(len(result) == 72 and len(policy_sha) <= 1, "验证case缺项或跨权重混入")
    return result, {"step": step, "policy_sha256": next(iter(policy_sha)) if policy_sha else None}


def _final_B(bundle, direct_bundle, direct, ledger):
    job, status = bundle["job"], bundle["status"]
    reference = job.get("initial_source") or {}
    require(status.get("stage") == "completed" and
            status.get("env_step") == 216 and
            reference.get("run_id") == direct_bundle["job"]["run_id"] and
            reference.get("group") == ("greedy" if direct_bundle["name"] == "greedy" else "model") and
            reference.get("partition") == "validation" and
            reference.get("scene_count") == 72 and reference.get("repeats") == 1 and
            len(reference.get("case_files") or []) == 72,
            "72景完整B来源任务不匹配")
    controls = bundle["config"].get("power_refinement") or {}
    require(controls.get("methods") == ["none", "sca", "sca_si_lp"] and
            controls.get("power_budget_mode") == "total" and
            controls.get("total_power_w") == 3000 and
            controls.get("maxeval") == 2000 and controls.get("max_time_seconds") == 30.0 and
            controls.get("repeats") == 1 and controls.get("action_seed") == 53001,
            "完整B算法/预算/种子不符")
    if direct_bundle["name"] in ("C0", "C1"):
        require(reference.get("validation_step") == 100000 and
                reference.get("source_checkpoint_sha256") == direct_bundle["checkpoint_sha256"] and
                reference.get("policy_sha256") == direct_bundle["direct_policy_sha256"],
                "C0/C1完整B未绑定试点断点/策略")
    elif direct_bundle["name"] == "I0":
        require(reference.get("validation_step") == direct_bundle["direct_step"] and
                reference.get("source_checkpoint_sha256") == direct_bundle["checkpoint_sha256"],
                "I0完整B未绑定实际试点断点")
    cases = sorted((bundle["directory"] / "power_cases").glob("*_sca_si_lp.json"))
    require(len(cases) == 72, "完整B缺72个最终B案例")
    observed = {}
    direct_by_sha = {value["scene_sha256"]: (key, value) for key, value in direct.items()}
    for path in cases:
        item = read_json(path)
        source_sha = item.get("source_scene_sha256")
        require(source_sha in direct_by_sha and source_sha not in observed and
                item.get("run_id") == job["run_id"] and item.get("method") == "sca_si_lp" and
                item.get("repeat") == 0 and
                item.get("source_evaluation_run_id") == direct_bundle["job"]["run_id"] and
                item.get("source_initial_case_sha256") == direct_by_sha[source_sha][1]["case_sha256"],
                "最终B逐case初解SHA或任务来源不符")
        key, first = direct_by_sha[source_sha]
        result = physical_result(item["beams"], item["metrics"], key[1])
        for old, new in zip(read_json(first["case_path"]).get("beams") or
                            (read_json(first["case_path"]).get("body") or {}).get("result", {}).get("beams", []),
                            item["beams"], strict=True):
            require((old["freq"], old["slots"]) == (new["freq"], new["slots"]),
                    "SCA改变了初始频率槽")
        observed[source_sha] = {**result, "key": key, "case_path": str(path),
                                "case_sha256": _source_files_add(ledger, path)}
    require(len(observed) == 72, "最终B未覆盖同源72景")
    return {item["key"]: item for item in observed.values()}


def paired_mean(left, right, field="si"):
    require(set(left) == set(right) and len(left) == 72, "72景母场景配对缺失")
    return mean(left[key][field] - right[key][field] for key in left)


def mother_improvements(left, right, threshold=0.001):
    require(set(left) == set(right) and len(left) == 72, "母级配对缺72景")
    by_mother = defaultdict(list)
    for key in left:
        by_mother[(key[0], key[3])].append(left[key]["si"] - right[key]["si"])
    require(len(by_mother) == 6 and all(len(rows) == 12 for rows in by_mother.values()),
            "母级比较缺6母×12景")
    return {f"{mother}|{sha}": mean(rows) for (mother, sha), rows in by_mother.items()}


def assess(arm, direct, final_B, baseline_direct, baseline_B,
           c0_direct=None, c0_B=None, c1_direct=None, c1_B=None):
    """Scheme-19 gates; a false effect is a valid audit result, not missing data."""
    require(arm in ("C0", "C1", "I0"), "pilot arm无效")
    effect = {"vs_greedy_direct": paired_mean(direct, baseline_direct),
              "vs_greedy_final_B": paired_mean(final_B, baseline_B),
              "mean_si": {"pilot_direct": mean(row["si"] for row in direct.values()),
                          "pilot_final_B": mean(row["si"] for row in final_B.values()),
                          "greedy_direct": mean(row["si"] for row in baseline_direct.values()),
                          "greedy_final_B": mean(row["si"] for row in baseline_B.values())}}
    criteria = {"exact_completed_sources_and_checkpoint": True,
                "same_72_validation_scenes_six_mothers": True,
                "all_beam_resource_and_source_sha_checked": True}
    if arm in ("C0", "C1"):
        require(c0_direct is not None and c0_B is not None and
                c1_direct is not None and c1_B is not None,
                "C0/C1需两者同预算试点及完整B后一起判定")
        c1_gain = paired_mean(c1_direct, c0_direct)
        c1_B_gain = paired_mean(c1_B, c0_B)
        effect.update(c1_minus_c0_direct=c1_gain,
                      c1_minus_c0_final_B=c1_B_gain)
        criteria.update(c1_direct_gain_at_least_0_2pp=c1_gain >= .002,
                        c1_final_B_not_worse=c1_B_gain >= -1e-8)
    else:
        mothers = mother_improvements(final_B, baseline_B)
        improved = sum(value > .001 for value in mothers.values())
        effect.update(final_B_mother_delta=mothers,
                      mothers_above_0_1pp=improved)
        criteria.update(i0_direct_positive=effect["vs_greedy_direct"] > 0,
                        i0_final_B_positive=effect["vs_greedy_final_B"] > 0,
                        at_least_two_of_six_mothers_above_0_1pp=improved >= 2)
    return criteria, effect


def build_audit(sources, output):
    """Require immutable pilot snapshots and saved 72-case A→B evidence."""
    require(sources.get("experiment_id") == EXPERIMENT and
            sources.get("arm") in ("C0", "C1", "I0"),
            "来源说明缺实验/arm身份")
    arm = sources["arm"]
    snapshots = sources.get("pilot_snapshots") or {}
    required = {arm, "greedy", arm + "_B", "greedy_B"}
    if arm in ("C0", "C1"):
        required.update(("C0", "C1", "C0_B", "C1_B"))
    require(all(name in snapshots for name in required),
            "审核前须冻结试点checkpoint及全部基线/B任务身份原始字节")
    ledger = {}
    bundles = {name: _bundle(sources, name, ledger, snapshot=snapshots.get(name))
               for name in required}
    pilot = bundles[arm]
    target_archive = Path(pilot["job"]["archive_root"]).resolve(strict=True)
    require(all(Path(item["snapshot"]["path"]).resolve(strict=True).is_relative_to(target_archive)
                for item in bundles.values()), "审计快照必须全在目标工作台归档")
    require(pilot["checkpoint_sha256"] == sha(pilot["directory"] / "checkpoint.pt"),
            "目标试点当前断点已改变；不能向服务放行")
    physical_contracts = {tuple(bundle["physics_files"].items()) for bundle in bundles.values()}
    require(len(physical_contracts) == 1 and all(all(bundle["physics_files"].values())
                                                 for bundle in bundles.values()),
            "跨任务物理源码SHA不等价")
    validation = {name: _validation_set(bundle, ledger)
                  for name, bundle in bundles.items()}
    keys = set(validation["greedy"])
    require(all(set(rows) == keys and all(rows[key]["content_sha256"] ==
                validation["greedy"][key]["content_sha256"] for key in keys)
                for rows in validation.values()), "C0/C1/I0/贪心完整源NPZ内容不一致")
    direct = {}
    info = {}
    for name in ("greedy", arm, *(("C0", "C1") if arm in ("C0", "C1") else ())):
        if name in direct:
            continue
        direct[name], info[name] = _direct(bundles[name], validation[name], ledger)
        bundles[name]["direct_step"] = info[name]["step"]
        bundles[name]["direct_policy_sha256"] = info[name]["policy_sha256"]
    refined = {}
    for name in ("greedy", arm, *(("C0", "C1") if arm in ("C0", "C1") else ())):
        if name in refined:
            continue
        refined[name] = _final_B(bundles[name + "_B"], bundles[name], direct[name], ledger)
    if arm in ("C0", "C1"):
        c0_status, c1_status = bundles["C0"]["status"], bundles["C1"]["status"]
        require(c0_status.get("env_step") == c1_status.get("env_step") == 100000 and
                c0_status.get("update_step") == c1_status.get("update_step") and
                c0_status.get("update_step") is not None,
                "C0/C1试点步数或真实更新数不同")
        def parity(config):
            data = json.loads(json.dumps(config))
            data.pop("name", None)
            data["experiment"].pop("arm", None)
            data["training"].pop("demonstrations", None)
            return data
        require(parity(bundles["C0"]["config"]) == parity(bundles["C1"]["config"]),
                "C0/C1除示范来源外科学配置不同")
        c0_train = (bundles["C0"]["job"].get("scene_sets") or {}).get("train") or {}
        c1_train = (bundles["C1"]["job"].get("scene_sets") or {}).get("train") or {}
        def train_identity(record):
            return [(row.get("sha256"), row.get("source_sha256"),
                     row.get("mother_scene_id"), row.get("selected_time_utc"),
                     row.get("beam_count"), row.get("load_multiplier"))
                    for row in record.get("files") or []]
        require(c0_train.get("count") == c1_train.get("count") == 8064 and
                c0_train.get("profile") == c1_train.get("profile") == "europe_v1" and
                len(c0_train.get("files") or []) == 8064 and
                train_identity(c0_train) == train_identity(c1_train),
                "C0/C1训练来源集或顺序不同")
        source_set_sha = canonical_sha([row["sha256"] for row in c1_train["files"]])
        demonstration = bundles["C1"]["config"]["training"]["demonstrations"]
        teacher = bundles["C1"]["job"].get("demonstration_source") or {}
        require(demonstration.get("mode") == "half" and
                demonstration.get("source_set_sha256") == source_set_sha and
                teacher.get("manifest_path") == demonstration.get("manifest_path") and
                teacher.get("manifest_sha256") == demonstration.get("manifest_sha256") and
                teacher.get("run_id") and teacher.get("job_path") and
                teacher.get("result_path"), "C1教师来源与相同训练集未绑定")
        for path_key, sha_key in (("manifest_path", "manifest_sha256"),
                                  ("job_path", "job_sha256"),
                                  ("result_path", "result_sha256")):
            _source_files_add(ledger, teacher[path_key], teacher[sha_key])
    criteria, effect = assess(arm, direct[arm], refined[arm], direct["greedy"],
                              refined["greedy"], direct.get("C0"), refined.get("C0"),
                              direct.get("C1"), refined.get("C1"))
    # A failed performance gate must remain failed; no caller may override it.
    audit = {"experiment_id": EXPERIMENT, "arm": arm,
             "run_id": pilot["job"]["run_id"],
             "checkpoint_sha256": pilot["checkpoint_sha256"],
             "checkpoint_step": pilot["status"]["env_step"],
             "passed": all(criteria.values()), "criteria": criteria,
             "effect": effect, "scene_count": 72, "mother_count": 6,
             "source_runs": {name: bundle["job"]["run_id"] for name, bundle in bundles.items()},
             "source_files": [{"path": path, "sha256": value}
                              for path, value in sorted(ledger.items())],
             "source_scope": "Saved validation allocations and final B only; no inference or optimization",
             "next_controller_action": ("write local gate receipt then approve-pilot API"
                                        if all(criteria.values()) else
                                        "record gate_failed/not_selected; do not call approve-pilot")}
    output = Path(output).resolve()
    archive = Path(pilot["job"]["archive_root"]).resolve(strict=True)
    require(output.is_relative_to(archive) and output.suffix == ".json",
            "审计JSON须保存到目标工作台archive供approve-pilot读取")
    write_json(output, audit)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--freeze-inputs", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    sources = read_json(args.sources)
    require(args.freeze_inputs != args.build, "请单独执行freeze-inputs或build")
    if args.freeze_inputs:
        snapshots = freeze_pilot_inputs(sources, args.output)
        frozen_sources = Path(args.output) / "pilot_audit_inputs" / f"sources_{sources['arm']}.json"
        write_json(frozen_sources, {**sources, "pilot_snapshots": snapshots})
        result = {"sources_path": str(frozen_sources), "pilot_snapshots": snapshots}
    else:
        result = build_audit(sources, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
