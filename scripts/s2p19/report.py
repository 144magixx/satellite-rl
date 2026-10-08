"""Read-only S2-P19 result audit and report material from completed workbench jobs.

Usage::

    python -m scripts.s2p19.report --manifest D:/paper/outputs/experiments/sac_routes_s2p19_20261007/run_manifest.json \
        --controller D:/paper/outputs/experiments/sac_routes_s2p19_20261007/controller/state.json \
        --plan D:/paper/outputs/experiments/sac_routes_s2p19_20261007/controller/plan.json \
        --source-index D:/paper/outputs/experiments/sac_routes_s2p19_20261007/analysis/source_index.json \
        --output D:/paper/outputs/experiments/sac_routes_s2p19_20261007/analysis

The optional source-index binds each controller task ID to a locally archived
results.json and its SHA256. Formal paired reporting additionally needs
SHA-pinned job.json, config.json, source_manifest.json, a scene_root containing
the exact source NPZ bytes, and local case mirrors when case_path is advertised.
Its `jobs` mapping uses controller task IDs as keys, with `run_id`,
`results_path/results_sha256`, `job_path/job_sha256`,
`config_path/config_sha256`, `source_manifest_path/source_manifest_sha256`,
`status_path/status_sha256`, `scene_root`, and optional `case_root`.
Optional `training.C0/C1/I0` entries bind original `episodes` and `evaluations`
JSONL paths/SHA for curves. Each required SHA is checked before reading.
The full canonical scene content and frozen physics source/config/unit contract
must agree across arms, even when ZIP container SHA differs. This script
never connects to remote hosts, runs a policy, or calls an optimizer. Missing
final cases remain missing; figures are written only for a complete independent
880-scene matrix. Report prose is material for the root task, not Report 15.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


EXPERIMENT = "S2-P19"
N_VALUES = (50, 60, 75, 90, 100, 125, 140, 150, 175, 180, 200)
LOADS = (0.75, 1.25)
EXPECTED_SCENES = 880
REPEATS = {"C0": 3, "C1": 3, "I0": 3, "V2": 3,
           "greedy": 1, "rule": 1, "random": 3}
ROLE_LABELS = {"C0": "C0 终态目标对齐SAC", "C1": "C1 示范回放SAC",
               "I0": "I0 贪心起点多步改进SAC", "V2": "V2 历史锚点",
               "greedy": "B1 贪心", "rule": "同菜单规则搜索",
               "random": "同菜单随机搜索"}


class AuditError(ValueError):
    """Source identity, physical accounting, or statistical unit differs."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON根对象无效：{path}")
    return value


class _JsonStream:
    """Bounded-buffer decoder: one complete case object is the largest value."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self.pos = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _more(self):
        if self.pos:
            self.buffer = self.buffer[self.pos:]
            self.pos = 0
        chunk = self.stream.read(65536)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return None
            self._more()

    def take(self, expected):
        actual = self.peek()
        require(actual == expected, f"JSON结构错误：预期{expected!r}，实际{actual!r}")
        self.pos += 1

    def value(self):
        while True:
            self.peek()
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
            except json.JSONDecodeError as error:
                if self.eof:
                    raise AuditError("保存结果JSON损坏") from error
                self._more()
                continue
            self.pos = end
            return value


def iter_saved_scenes(path, info):
    """Stream results.scenes without copying saved raw beams into analysis."""
    with Path(path).open(encoding="utf-8") as stream:
        reader = _JsonStream(stream)
        reader.take("{")
        found = False
        while reader.peek() != "}":
            key = reader.value()
            require(isinstance(key, str), "结果JSON顶层键无效")
            reader.take(":")
            if key == "scenes":
                require(not found, "结果JSON重复scenes字段")
                found = True
                reader.take("[")
                while reader.peek() != "]":
                    row = reader.value()
                    require(isinstance(row, dict), "逐景结果不是对象")
                    yield row
                    if reader.peek() == ",":
                        reader.take(",")
                    else:
                        require(reader.peek() == "]", "逐景结果分隔符无效")
                reader.take("]")
            else:
                value = reader.value()
                if key in ("metadata", "research_stage", "summary", "mode"):
                    info[key] = value
            if reader.peek() == ",":
                reader.take(",")
            else:
                require(reader.peek() == "}", "结果顶层分隔符无效")
        reader.take("}")
        require(reader.peek() is None and found, "结果JSON缺scenes或含尾随内容")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def mean(values):
    values = [number(item) for item in values]
    return sum(values) / len(values) if values and all(item is not None for item in values) else None


def percentile(values, fraction):
    values = sorted(number(item) for item in values)
    if not values or any(item is None for item in values):
        return None
    position = (len(values) - 1) * fraction
    low, high = int(math.floor(position)), int(math.ceil(position))
    return values[low] + (values[high] - values[low]) * (position - low)


def timing_stats(values):
    numeric = [number(value) for value in values]
    available = [value for value in numeric if value is not None]
    return {"n": len(available), "missing": len(values) - len(available),
            "mean": mean(available), "median": statistics.median(available) if available else None,
            "p95": percentile(available, 0.95)}


def task_role(task_id):
    if task_id == "baseline_test_20418":
        return ("baseline", "direct")
    if task_id.startswith("baseline_") and "_B_" in task_id:
        return (task_id.split("_", 2)[1], "final_B")
    if task_id.startswith("V2_final_direct_"):
        return ("V2", "direct")
    if task_id.startswith("V2_final_B_"):
        return ("V2", "final_B")
    for arm in ("C0", "C1", "I0"):
        if task_id.startswith(arm + "_final_direct_"):
            return (arm, "direct")
        if task_id.startswith(arm + "_final_B_"):
            return (arm, "final_B")
    return None


def final_tasks(plan):
    tasks = {task["id"]: task_role(task["id"]) for task in (plan or {}).get("tasks", [])
             if task_role(task["id"]) is not None}
    for task_id in (plan or {}).get("completed_legacy", {}):
        role = task_role(task_id)
        require(role is not None and task_id not in tasks,
                f"历史完成任务ID重复或无效：{task_id}")
        tasks[task_id] = role
    return tasks


def plan_ports(plan):
    ports = {task["id"]: task["port"] for task in plan.get("tasks", [])
             if task_role(task["id"]) is not None}
    for task_id in plan.get("completed_legacy", {}):
        ports[task_id] = 20418
    for task_id, port in ports.items():
        require(port in (20418, 20424, 20473, 20474) and
                task_id.endswith("_" + str(port)),
                f"{task_id}: 计划端口与任务归属不匹配")
    return ports


SCENE_ARRAYS = ("beam_location", "beam_G_Tx_max", "UE_G_Rx_max",
                "distances", "rate", "noise_temperature", "original_indices")


@lru_cache(maxsize=2048)
def scene_content(path):
    """Hash every physical input array, independent of NPZ ZIP metadata."""
    import numpy as np

    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == set(SCENE_ARRAYS),
                f"源场景数组字段不完整：{path}")
        arrays = {}
        for key in sorted(SCENE_ARRAYS):
            array = np.asarray(data[key])
            require(np.issubdtype(array.dtype, np.number) and np.isfinite(array).all(),
                    f"源场景数组非有限数值：{key}")
            normalized = (np.asarray(array, dtype="<i8", order="C") if key == "original_indices"
                          else np.asarray(array, dtype="<f8", order="C"))
            arrays[key] = normalized
            h.update(key.encode("ascii"))
            h.update(json.dumps(list(normalized.shape), separators=(",", ":")).encode())
            h.update(normalized.tobytes())
        n = arrays["rate"].shape[0]
        require(arrays["beam_location"].shape == (n, 2) and
                arrays["rate"].shape == (n, 1) and
                arrays["original_indices"].shape == (n,),
                "源场景束维/几何/需求形状不一致")
        static = [(i, int(arrays["original_indices"][i]),
                   round(float(arrays["beam_location"][i, 0]), 12),
                   round(float(arrays["beam_location"][i, 1]), 12),
                   round(float(arrays["rate"][i, 0]), 6), i % 8)
                  for i in range(n)]
        static_sha = hashlib.sha256(json.dumps(static, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
    return h.hexdigest(), static_sha


PHYSICS_SOURCE_FILES = ("src/satellite_rl/envs/satellite.py",
                        "src/satellite_rl/envs/regions.py",
                        "src/satellite_rl/power_refinement/physics.py")


def _physical_contract(config, job, source_manifest):
    scene_sets = job.get("scene_sets") or {}
    scene_set = scene_sets.get("test")
    if not isinstance(scene_set, dict) or not isinstance(source_manifest, dict):
        return None
    files = source_manifest.get("files") or {}
    if (source_manifest.get("git_commit") != job.get("git_commit") or
            any(not isinstance(files.get(name), str) or len(files[name]) != 64
                for name in PHYSICS_SOURCE_FILES)):
        return None
    env = config.get("environment") or {}
    i0 = config.get("i0") or {}
    profile = env.get("profile", i0.get("profile"))
    budget = env.get("power_budget_w", i0.get("power_budget_w"))
    basic = env.get("basic_fraction", i0.get("basic_fraction"))
    values = {"profile": profile, "power_budget_w": budget,
              "basic_fraction": basic,
              "environment_version": scene_set.get("environment_version"),
              "environment_profile": scene_set.get("environment_profile"),
              "units": scene_set.get("units"),
              "physics_source_sha256": {name: files[name] for name in PHYSICS_SOURCE_FILES},
              "group_rule": "beam_id_mod_8",
              "slots_per_group": 100, "max_contiguous_slots": 10}
    if (values["profile"] != "europe_v1" or values["power_budget_w"] != 3000
            or values["basic_fraction"] != 0.1 or
            not values["environment_version"] or not values["environment_profile"] or
            values["units"] != {"rate": "bit/s", "beam_location": "degrees"}):
        return None
    refinement = config.get("power_refinement") or {}
    if refinement and refinement.get("total_power_w") != 3000:
        return None
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def source_for_task(task_id, run_id, sources, controller, manifest,
                    expected_port=None, require_port_evidence=False):
    spec = (sources.get("jobs") or {}).get(task_id)
    if spec is None:
        return None, "missing_local_results_reference"
    require(spec.get("run_id") == run_id, f"{task_id}: source-index run_id不匹配")
    if require_port_evidence:
        require(spec.get("port") == expected_port and spec.get("archive_receipt_path"),
                f"{task_id}: 四端归档缺少实例端口或回执")
    if expected_port is not None and spec.get("port") is not None:
        require(spec["port"] == expected_port, f"{task_id}: 归档端口与计划不匹配")
    receipt_path = spec.get("archive_receipt_path")
    if receipt_path:
        receipt = read_json(receipt_path)
        require(receipt is not None and receipt.get("run_id") == run_id and
                (expected_port is None or receipt.get("port") == expected_port) and
                receipt.get("job_stage") == "completed" and
                receipt.get("inventory_sha256") == spec.get("archive_inventory_sha256"),
                f"{task_id}: 归档回执归属或完成状态不匹配")
    manifest_task = (manifest.get("tasks") or {}).get(task_id)
    if manifest_task and manifest_task.get("run_id"):
        require(manifest_task["run_id"] == run_id, f"{task_id}: 运行清单run_id不匹配")
    path = Path(spec["results_path"]).resolve()
    if not path.is_file():
        return None, "missing_saved_results"
    require(digest(path) == spec["results_sha256"], f"{task_id}: results SHA不匹配")
    job_path = spec.get("job_path")
    if job_path:
        job = read_json(job_path)
        require(job is not None and job.get("run_id") == run_id,
                f"{task_id}: 保存job与控制器ID不匹配")
        if spec.get("job_sha256"):
            require(digest(job_path) == spec["job_sha256"], f"{task_id}: job SHA不匹配")
    else:
        job = {}
    status_path = spec.get("status_path")
    if status_path:
        status = read_json(status_path)
        require(status is not None and status.get("run_id") == run_id,
                f"{task_id}: 保存status与控制器ID不匹配")
        if spec.get("status_sha256"):
            require(digest(status_path) == spec["status_sha256"], f"{task_id}: status SHA不匹配")
        require(status.get("stage") == "completed", f"{task_id}: 逐任务状态尚未完成")
    else:
        status = {}
    config_path = spec.get("config_path")
    if config_path:
        config = read_json(config_path)
        require(config is not None, f"{task_id}: 冻结config缺失")
        require(spec.get("config_sha256") == digest(config_path),
                f"{task_id}: 冻结config SHA不匹配")
    else:
        config = {}
    source_manifest_path = spec.get("source_manifest_path")
    if source_manifest_path:
        source_manifest = read_json(source_manifest_path)
        require(source_manifest is not None and
                digest(source_manifest_path) == spec.get("source_manifest_sha256"),
                f"{task_id}: 冻结源码清单SHA不匹配")
    else:
        source_manifest = {}
    return {"spec": spec, "path": str(path), "sha256": digest(path),
            "job": job, "status": status, "config": config,
            "physical_contract_sha256": _physical_contract(config, job, source_manifest),
            "controller": controller}, None


def _resolve_case(row, spec, *, task_id):
    case_path = row.get("case_path")
    if not case_path:
        return row, None, None
    declared = Path(case_path)
    candidates = [declared]
    if spec.get("case_root"):
        candidates.append(Path(spec["case_root"]) / declared.name)
    local = next((path for path in candidates if path.is_file()), None)
    require(local is not None, f"{task_id}: 保存case缺失 {declared.name}")
    actual_sha = digest(local)
    expected_sha = row.get("case_sha256")
    if expected_sha:
        require(actual_sha == expected_sha, f"{task_id}: case SHA不匹配 {declared.name}")
    case = read_json(local)
    require(case is not None, f"{task_id}: case JSON缺失")
    # Legacy workbench cases are envelopes; I0 uses a direct case object.
    if isinstance(case.get("body"), dict):
        case = case["body"].get("result", case["body"])
    return case, str(local), actual_sha


def _resolve_scene_source(row, data, spec, expected_sha, task_id):
    metadata = data.get("scene_metadata") or row.get("scene_metadata") or {}
    scene = data.get("scene") or row.get("scene") or {}
    names = [scene.get("source_path"), metadata.get("path"),
             row.get("source_scene_path"), scene.get("path")]
    candidates = []
    for name in names:
        if not name:
            continue
        original = Path(name)
        candidates.append(original)
        if spec.get("scene_root"):
            candidates.append(Path(spec["scene_root"]) / original.name)
    local = next((path for path in candidates if path.is_file() and path.suffix == ".npz"), None)
    if local is None:
        return None, None, None
    require(digest(local) == expected_sha,
            f"{task_id}: 完整源NPZ字节SHA与逐景记录不符")
    content_sha, static_sha = scene_content(str(local.resolve()))
    return str(local.resolve()), content_sha, static_sha


def _beam_audit(beams, beam_count, metrics, task_id):
    if not isinstance(beams, list) or len(beams) != beam_count:
        return {"available": False, "reason": "missing_complete_beams"}
    ratios, si, sgm, effective, powers, slots, strict_basic = [], [], [], [], [], [], []
    static_rows = []
    seen = set()
    occupied = {group: set() for group in range(8)}
    for beam in beams:
        idx = beam.get("beam_id")
        require(type(idx) is int and 0 <= idx < beam_count and idx not in seen,
                f"{task_id}: 逐束身份缺失或重复")
        seen.add(idx)
        original = beam.get("original_index")
        lat, lon = number(beam.get("lat")), number(beam.get("lon"))
        group = beam.get("group")
        require(type(original) is int and type(group) is int and group == idx % 8
                and lat is not None and lon is not None,
                f"{task_id}: 逐束静态身份/位置/组缺失")
        demand = number(beam.get("required_bps"))
        capacity = number(beam.get("capacity_bps", beam.get("rate_bps")))
        matching = number(beam.get("sgm"))
        power = number(beam.get("power_w"))
        length = beam.get("slots")
        start = beam.get("freq")
        require(demand is not None and demand > 0 and capacity is not None and capacity >= 0
                and matching is not None and 0 <= matching <= 1 and power is not None
                and power >= 0 and type(length) is int and 0 <= length <= 10
                and (length == 0 and power == 0 or
                     length > 0 and type(start) is int and 0 <= start <= 100 - length),
                f"{task_id}: 逐束D/C/功率/槽无效")
        if length:
            block = set(range(start, start + length))
            require(not occupied[group].intersection(block), f"{task_id}: 组内频率槽冲突")
            occupied[group].update(block)
        ratio = capacity / demand
        saved_ratio = number(beam.get("capacity_demand_ratio"))
        if saved_ratio is not None:
            require(abs(saved_ratio - ratio) <= 1e-7 * max(1, ratio),
                    f"{task_id}: 逐束C/D不一致")
        ratios.append(ratio)
        si.append(min(ratio, 1.0))
        sgm.append(matching)
        effective.append(min(capacity, demand))
        powers.append(power)
        slots.append(length)
        basic = capacity >= 0.1 * demand
        if "basic_met" in beam:
            require(type(beam["basic_met"]) is bool and beam["basic_met"] == basic,
                    f"{task_id}: 逐束基本服务严格门槛不一致")
        strict_basic.append(basic)
        static_rows.append((idx, original, round(lat, 12), round(lon, 12),
                            round(demand, 6), group))
        saved_effective = number(beam.get("effective_served_bps"))
        if saved_effective is not None:
            require(abs(saved_effective - min(capacity, demand)) <=
                    max(1e-4, 1e-8 * demand), f"{task_id}: 逐束有效承载不一致")
    saved_si = number(metrics.get("final_si_mean", metrics.get("si")))
    saved_sgm = number(metrics.get("final_sgm_mean", metrics.get("sgm")))
    require(saved_si is not None and saved_sgm is not None, f"{task_id}: 保存终态SI/SGM缺失")
    require(abs(saved_si - mean(si)) <= 1e-7 and abs(saved_sgm - mean(sgm)) <= 1e-7,
            f"{task_id}: 保存终态SI/SGM与逐束不一致")
    saved_power = number(metrics.get("power_w"))
    require(sum(powers) <= 3000.0 + 1e-6 and metrics.get("overload") is not True,
            f"{task_id}: 总功率超限")
    require(metrics.get("complete", True) is not False,
            f"{task_id}: 保存分配不完整")
    if saved_power is not None:
        require(abs(saved_power - sum(powers)) <= max(1e-4, 1e-7 * saved_power),
                f"{task_id}: 总功率与逐束不一致")
    saved_slots = number(metrics.get("allocated_slots", metrics.get("slots_total")))
    if saved_slots is not None:
        require(abs(saved_slots - sum(slots)) <= 1e-8, f"{task_id}: 槽数与逐束不一致")
    saved_basic = number(metrics.get("basic_met_count", metrics.get("basic_met")))
    if saved_basic is not None:
        require(abs(saved_basic - sum(strict_basic)) <= 1e-8,
                f"{task_id}: 基本服务束数与逐束严格门槛不一致")
    static_rows.sort()
    semantic_sha = hashlib.sha256(json.dumps(static_rows, separators=(",", ":"),
                                             allow_nan=False).encode("utf-8")).hexdigest()
    return {"available": True, "si": mean(si), "sgm": mean(sgm),
            "scene_semantic_sha256": semantic_sha,
            "resource_legal": True,
            "basic_met_count": sum(strict_basic),
            "zero_capacity_count": sum(number(beam.get("capacity_bps", beam.get("rate_bps"))) <= 0
                                       for beam in beams),
            "effective_carried_bps": sum(effective), "power_w": sum(powers),
            "slots_total": sum(slots), "rho_min": min(ratios),
            "rho_p05": percentile(ratios, 0.05), "rho_median": statistics.median(ratios),
            "rho_p95": percentile(ratios, 0.95), "rho_max": max(ratios),
            "rho_below_08_fraction": sum(r < 0.8 for r in ratios) / beam_count,
            "rho_below_1_fraction": sum(r < 1 for r in ratios) / beam_count}


def _timing(row, metrics):
    timing = row.get("timing") or {}
    return {"initial_seconds": number(row.get("initial_allocation_seconds",
                    timing.get("initialization_seconds", metrics.get("initialization_seconds")))),
            "candidate_seconds": number(metrics.get("candidate_generation_seconds")),
            "physical_probe_seconds": number(metrics.get("probe_seconds")),
            "physics_setup_seconds": number(timing.get("physics_setup_seconds")),
            "optimizer_seconds": number(timing.get("optimizer_wall_seconds")),
            "verification_seconds": number(timing.get("verification_seconds")),
            "total_seconds": number(timing.get("total_pipeline_seconds",
                    metrics.get("total_seconds", row.get("initial_allocation_seconds"))))}


BEAM_COLUMNS = ("task_id", "run_id", "role", "stage", "scene_sha256",
                "source_npz_local_path", "scene_content_sha256",
                "physical_contract_sha256", "mother_source_sha256",
                "selected_time_utc", "source_evaluation_run_id",
                "mother_scene_id", "beam_count", "load_multiplier", "repeat_index",
                "beam_id", "original_index", "group", "lat", "lon",
                "required_bps", "capacity_bps", "effective_carried_bps",
                "capacity_demand_ratio", "si", "sgm", "power_w", "freq", "slots",
                "rate_unit", "power_unit", "case_path", "case_sha256",
                "result_path_sha256")


def verified_beam_rows(action, beams):
    """Emit every audited target beam; SI is capped, rho is not."""
    require(action["beam_audit_available"], "逐束原始记录未经完整审计")
    for beam in beams:
        demand = number(beam["required_bps"])
        capacity = number(beam.get("capacity_bps", beam.get("rate_bps")))
        rho = capacity / demand
        yield {**{key: action.get(key) for key in (
                   "task_id", "run_id", "role", "stage", "scene_sha256",
                   "source_npz_local_path", "scene_content_sha256",
                   "physical_contract_sha256", "mother_source_sha256",
                   "selected_time_utc", "source_evaluation_run_id",
                   "mother_scene_id", "beam_count", "load_multiplier", "repeat_index",
                   "case_path", "case_sha256", "result_path_sha256")},
               **{key: beam.get(key) for key in (
                   "beam_id", "original_index", "group", "lat", "lon",
                   "power_w", "freq", "slots", "sgm")},
               "required_bps": demand, "capacity_bps": capacity,
               "effective_carried_bps": min(capacity, demand),
               "capacity_demand_ratio": rho, "si": min(rho, 1.0),
               "rate_unit": "bit/s", "power_unit": "W"}


KNOWN_NON_TIME_TERMINATIONS = {"completed", "stop_requested", "evaluation_limit",
                               "phase_reserve", "si_stall", "no_active_beams",
                               "trust_region_small", "surrogate_stationary_direction",
                               "iteration_limit", "failed_attempt_fallback_to_A"}


def optimizer_cap_stats(actions):
    """Count exact solver time stops; proximity to budget is separate evidence."""
    grouped = defaultdict(list)
    for row in actions:
        grouped[(row["role"], row["stage"])].append(row)
    result = {}
    for (role, stage), rows in sorted(grouped.items()):
        eligible = rows if stage == "final_B" else []
        hits = sum(row.get("solver_termination") == "time_limit" for row in eligible)
        known_other = sum(row.get("solver_termination") in KNOWN_NON_TIME_TERMINATIONS
                          for row in eligible)
        unknown = len(eligible) - hits - known_other
        budgets = [(number(row.get("solver_elapsed_seconds")),
                    number(row.get("optimizer_time_budget_seconds"))) for row in eligible]
        near = sum(elapsed is not None and limit is not None and limit > 0
                   and elapsed >= 0.95 * limit for elapsed, limit in budgets)
        near_known = sum(elapsed is not None and limit is not None and limit > 0
                         for elapsed, limit in budgets)
        result[f"{role}/{stage}"] = {
            "eligible_optimizer_actions": len(eligible),
            "explicit_time_limit_count": hits,
            "known_other_termination_count": known_other,
            "unknown_termination_count": unknown,
            "explicit_time_limit_fraction": hits / len(eligible) if eligible else None,
            "near_95pct_budget_count": near,
            "near_budget_known_denominator": near_known,
            "near_budget_fraction": near / near_known if near_known else None,
            "near_budget_is_not_time_limit": True}
    return result


def normalize_result(task_id, role, stage, row, spec, run_id, result_sha,
                     physical_contract_sha256=None, beam_sink=None):
    if stage == "final_B" and row.get("method") != "sca_si_lp":
        return None
    if stage == "direct" and role == "baseline":
        role = row.get("method")
        require(role in ("greedy", "rule", "random"),
                f"{task_id}: 基线方法无效")
    case, case_path, case_sha = _resolve_case(row, spec, task_id=task_id)
    data = case if case_path else row
    metrics = data.get("metrics") or row.get("metrics") or {}
    beams = data.get("beams") or row.get("beams")
    metadata = (data.get("scene_metadata") or row.get("scene_metadata")
                or (data.get("scene") or {}).get("scene_metadata") or {})
    scene_sha = (data.get("scene_sha256") or row.get("source_scene_sha256")
                 or (data.get("scene") or {}).get("source_sha256")
                 or (row.get("scene") or {}).get("source_sha256"))
    require(isinstance(scene_sha, str) and len(scene_sha) == 64,
            f"{task_id}: 缺少真实场景SHA")
    scene_path, content_sha, source_static_sha = _resolve_scene_source(
        row, data, spec, scene_sha, task_id)
    n = int(row.get("beam_count") or data.get("beam_count") or len(beams or []))
    mother = metadata.get("mother_scene_id")
    load = number(metadata.get("load_multiplier"))
    require(n in N_VALUES and isinstance(mother, str) and mother and load in LOADS,
            f"{task_id}: 场景N/母身份/负载缺失")
    repeat = row.get("repeat_index", row.get("repeat", data.get("repeat_index", 0)))
    require(type(repeat) is int and 0 <= repeat < REPEATS[role],
            f"{task_id}: 重复动作编号无效")
    if stage == "final_B":
        source_group = row.get("initial_source_group")
        if source_group is not None:
            require(source_group == ("model" if role in ("C0", "C1", "I0", "V2") else role),
                    f"{task_id}: B初解来源组不匹配")
    try:
        audit = _beam_audit(beams, n, metrics, task_id)
    except AuditError as error:
        audit = {"available": False, "resource_legal": False,
                 "reason": str(error)}
    if source_static_sha and audit.get("scene_semantic_sha256"):
        require(source_static_sha == audit["scene_semantic_sha256"],
                f"{task_id}: 保存逐束静态身份与完整源NPZ不一致")
    action = {"task_id": task_id, "run_id": run_id, "role": role,
            "stage": stage, "method": row.get("method", role),
            "scene_sha256": scene_sha, "mother_scene_id": mother,
            "source_npz_local_path": scene_path,
            "scene_content_sha256": content_sha,
            "physical_contract_sha256": physical_contract_sha256,
            "mother_source_sha256": metadata.get("source_sha256"),
            "selected_time_utc": metadata.get("selected_time_utc"),
            "beam_count": n, "load_multiplier": load,
            "repeat_index": repeat, "action_seed": row.get("action_seed", data.get("action_seed")),
            "result_path_sha256": result_sha, "case_path": case_path,
            "case_sha256": case_sha,
            "source_evaluation_run_id": row.get("source_evaluation_run_id"),
            "source_initial_case_sha256": row.get("source_initial_case_sha256"),
            "scene_semantic_sha256": audit.get("scene_semantic_sha256"),
            "si": audit.get("si"), "sgm": audit.get("sgm"),
            "basic_met_count": audit.get("basic_met_count"),
            "effective_carried_bps": audit.get("effective_carried_bps"),
            "power_w": audit.get("power_w"), "slots_total": audit.get("slots_total"),
            "rho_min": audit.get("rho_min"), "rho_p05": audit.get("rho_p05"),
            "rho_median": audit.get("rho_median"), "rho_p95": audit.get("rho_p95"),
            "rho_max": audit.get("rho_max"),
            "rho_below_08_fraction": audit.get("rho_below_08_fraction"),
            "rho_below_1_fraction": audit.get("rho_below_1_fraction"),
            "beam_audit_available": audit["available"],
            "beam_audit_missing_reason": audit.get("reason"),
            "resource_legal": audit.get("resource_legal"),
            "zero_capacity_count": audit.get("zero_capacity_count"),
            "fallback_used": (row.get("solver") or {}).get("fallback_used"),
            "solver_termination": (row.get("solver") or {}).get("termination_reason"),
            "solver_elapsed_seconds": number((row.get("solver") or {}).get("elapsed_seconds")),
            "missing_exact_cost": (row.get("failed_attempt_fallback") or {}).get(
                "original_attempt_exact_cost_unknown"), **_timing(row, metrics)}
    if beam_sink is not None and action["beam_audit_available"]:
        beam_sink(action, beams)
    return action


def collapse_repeats(actions):
    groups = defaultdict(list)
    for row in actions:
        groups[(row["role"], row["stage"], row["scene_sha256"])].append(row)
    result = []
    numeric = ("si", "sgm", "basic_met_count", "effective_carried_bps", "power_w",
               "slots_total", "zero_capacity_count", "rho_min", "rho_p05", "rho_median", "rho_p95", "rho_max",
               "rho_below_08_fraction", "rho_below_1_fraction")
    timing = ("initial_seconds", "candidate_seconds", "physical_probe_seconds",
              "physics_setup_seconds", "optimizer_seconds", "verification_seconds",
              "total_seconds")
    for (role, stage, scene_sha), rows in sorted(groups.items()):
        first = rows[0]
        require(all(row["mother_scene_id"] == first["mother_scene_id"] and
                    row["beam_count"] == first["beam_count"] and
                    row["load_multiplier"] == first["load_multiplier"] and
                    row["scene_semantic_sha256"] == first["scene_semantic_sha256"] and
                    row["scene_content_sha256"] == first["scene_content_sha256"] and
                    row["physical_contract_sha256"] == first["physical_contract_sha256"] and
                    row["mother_source_sha256"] == first["mother_source_sha256"] and
                    row["selected_time_utc"] == first["selected_time_utc"]
                    for row in rows),
                f"{role}/{stage}: 同场景重复的母身份不同")
        expected = REPEATS[role]
        repeats = {row["repeat_index"] for row in rows}
        require(len(repeats) == len(rows), f"{role}/{stage}: 重复编号重复")
        complete = repeats == set(range(expected)) and all(row["beam_audit_available"] for row in rows)
        item = {"role": role, "stage": stage, "scene_sha256": scene_sha,
                "mother_scene_id": first["mother_scene_id"],
                "scene_semantic_sha256": first["scene_semantic_sha256"],
                "scene_content_sha256": first["scene_content_sha256"],
                "physical_contract_sha256": first["physical_contract_sha256"],
                "mother_source_sha256": first["mother_source_sha256"],
                "selected_time_utc": first["selected_time_utc"],
                "beam_count": first["beam_count"], "load_multiplier": first["load_multiplier"],
                "repeats_observed": len(rows), "repeats_expected": expected,
                "complete_repeats": complete,
                "case_paths": json.dumps([row["case_path"] for row in rows], ensure_ascii=False),
                "case_sha256s": json.dumps([row["case_sha256"] for row in rows]),
                "result_run_ids": json.dumps(sorted({row["run_id"] for row in rows}))}
        for key in numeric:
            item[key] = mean(row[key] for row in rows)
        for key in timing:
            values = [row[key] for row in rows]
            item[key + "_mean_per_action"] = mean(values)
            item[key + "_sum_repeats"] = sum(values) if all(value is not None for value in values) else None
        result.append(item)
    return result


def _full_group(rows, role, stage):
    selected = [row for row in rows if row["role"] == role and row["stage"] == stage]
    if len(selected) != EXPECTED_SCENES or not all(
            row["complete_repeats"] and row["scene_semantic_sha256"]
            and row["scene_content_sha256"] and row["physical_contract_sha256"]
            and row["mother_source_sha256"] and row["selected_time_utc"]
            for row in selected):
        return False
    keys = {(row["beam_count"], row["load_multiplier"], row["mother_scene_id"]) for row in selected}
    mothers = {row["mother_scene_id"] for row in selected}
    dates = Counter(row["selected_time_utc"][:10] for row in selected)
    mother_sources = {row["mother_scene_id"]: row["mother_source_sha256"] for row in selected}
    return (len(keys) == EXPECTED_SCENES and len(mothers) == 40 and
            sorted(dates.values()) == [440, 440] and
            len(set(mother_sources.values())) == 40 and
            all(sum(row["beam_count"] == n and row["load_multiplier"] == load
                    for row in selected) == 40 for n in N_VALUES for load in LOADS))


def by_n_load(scenes):
    groups = defaultdict(list)
    for row in scenes:
        groups[(row["role"], row["stage"], row["beam_count"], row["load_multiplier"])].append(row)
    result = []
    for (role, stage, n, load), rows in sorted(groups.items()):
        result.append({"role": role, "stage": stage, "beam_count": n,
                       "load_multiplier": load, "scene_count": len(rows),
                       "complete_40_mothers": len(rows) == 40 and all(r["complete_repeats"] for r in rows),
                       **{key: mean(row[key] for row in rows) for key in (
                           "si", "sgm", "basic_met_count", "effective_carried_bps",
                           "power_w", "slots_total", "rho_p05", "rho_below_08_fraction")}})
    return result


def mother_pair(scenes, left, right, stage, seed=51001, draws=2000):
    if not _full_group(scenes, left, stage) or not _full_group(scenes, right, stage):
        return None
    indexed = {(r["role"], r["mother_scene_id"], r["beam_count"],
                r["load_multiplier"]): r for r in scenes if r["stage"] == stage}
    differences = defaultdict(list)
    zip_only_differences = 0
    left_rows = [r for r in scenes if r["role"] == left and r["stage"] == stage]
    for row in left_rows:
        other = indexed.get((right, row["mother_scene_id"], row["beam_count"],
                             row["load_multiplier"]))
        require(other is not None and
                other["scene_semantic_sha256"] == row["scene_semantic_sha256"] and
                other["scene_content_sha256"] == row["scene_content_sha256"] and
                other["physical_contract_sha256"] == row["physical_contract_sha256"] and
                other["mother_source_sha256"] == row["mother_source_sha256"] and
                other["selected_time_utc"] == row["selected_time_utc"],
                f"{left}/{right}: 共同场景或母身份不匹配")
        if other["scene_sha256"] != row["scene_sha256"]:
            zip_only_differences += 1
        differences[row["mother_scene_id"]].append(row["si"] - other["si"])
    require(len(differences) == 40 and all(len(value) == 22 for value in differences.values()),
            f"{left}/{right}: 母级配对未覆盖11N×2负载")
    per_mother = {mother: mean(value) for mother, value in differences.items()}
    values = list(per_mother.values())
    mother_date = {}
    for row in left_rows:
        date = row["selected_time_utc"][:10]
        prior = mother_date.setdefault(row["mother_scene_id"], date)
        require(prior == date, "同一母时刻跨日期不一致")
    per_date = {date: mean(per_mother[mother] for mother, day in mother_date.items()
                           if day == date) for date in sorted(set(mother_date.values()))}
    # Correlated nearby mothers: resample fixed consecutive four-mother blocks
    # within each date as a descriptive sensitivity, not a new significance test.
    blocks_by_date = {}
    for date in sorted(per_date):
        mothers = sorted((mother for mother, day in mother_date.items() if day == date))
        blocks_by_date[date] = [[per_mother[mother] for mother in mothers[index:index + 4]]
                                for index in range(0, len(mothers), 4)]
    rng = random.Random(seed)
    boot = [mean(rng.choices(values, k=len(values))) for _ in range(draws)]
    block_boot = [mean(value for blocks in blocks_by_date.values()
                       for block in rng.choices(blocks, k=len(blocks)) for value in block)
                  for _ in range(draws)]
    return {"left": left, "right": right, "stage": stage,
            "mother_count": 40, "derived_scenes": 880,
            "different_npz_bytes_same_full_physics_count": zip_only_differences,
            "mean_delta": mean(values), "ci95_low": percentile(boot, 0.025),
            "ci95_high": percentile(boot, 0.975),
            "per_date_mean_delta": per_date,
            "four_mother_block_ci95_low": percentile(block_boot, 0.025),
            "four_mother_block_ci95_high": percentile(block_boot, 0.975),
            "per_mother": per_mother, "bootstrap_seed": seed, "bootstrap_draws": draws}


def _training_status(manifest, controller, sources):
    result = {}
    for arm in ("C0", "C1", "I0"):
        ref = (manifest.get("tasks") or {}).get(arm) or {}
        if not ref:
            candidates = [value for key, value in (controller.get("tasks") or {}).items()
                          if key.startswith(arm + "_pilot_") and value.get("run_id")]
            ref = candidates[0] if candidates else {}
        run_id = ref.get("run_id")
        value = {"run_id": run_id, "stage": ref.get("stage") or ref.get("workbench_stage"),
                 "env_step": ref.get("env_step"), "budget_env_steps": ref.get("budget_env_steps"),
                 "completed_due_to": ref.get("completed_due_to"),
                 "stop_reason": ref.get("stop_reason")}
        source = next((item for item in (sources.get("jobs") or {}).values()
                       if item.get("run_id") == run_id and item.get("status_path")), None)
        if source:
            status = read_json(source["status_path"])
            if status:
                for key in ("stage", "env_step", "budget_env_steps", "completed_due_to", "stop_reason"):
                    value[key] = status.get(key)
        value["fixed_400k_endpoint"] = value["env_step"] == 400000 and value["stage"] == "completed"
        result[arm] = value
    return result


def training_curves(sources, manifest):
    """Read only SHA-pinned original JSONL; keep raw values and their meaning."""
    rows = []
    declared = sources.get("training") or {}
    for arm, spec in declared.items():
        require(arm in ("C0", "C1", "I0"), "训练曲线来源臂无效")
        expected_run = ((manifest.get("tasks") or {}).get(arm) or {}).get("run_id")
        if expected_run:
            require(spec.get("run_id") == expected_run,
                    f"{arm}: 训练曲线run_id与清单不符")
        update_ref = spec.get("updates")
        if update_ref is not None:
            update_path = Path(update_ref["path"])
            require(update_path.is_file() and digest(update_path) == update_ref["sha256"],
                    f"{arm}: updates原始日志SHA不符")
        for kind in ("evaluations", "episodes"):
            ref = spec.get(kind)
            if ref is None:
                continue
            path = Path(ref["path"])
            require(path.is_file() and digest(path) == ref["sha256"],
                    f"{arm}: {kind}原始日志SHA不符")
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    step = item.get("env_step")
                    require(type(step) is int and step >= 0,
                            f"{arm}: {kind}:{line_number} 缺实际交互步")
                    if kind == "evaluations":
                        value = number(item.get("mean_si"))
                        if value is None and item.get("group") == "validation":
                            value = number((item.get("metrics") or {}).get("final_si_mean"))
                        if value is None:
                            continue
                        metric = "validation_si"
                    else:
                        value = number(item.get("reward_total", item.get("return")))
                        if value is None:
                            continue
                        metric = "episode_return"
                    rows.append({"arm": arm, "run_id": spec.get("run_id"),
                                 "source": kind, "source_sha256": ref["sha256"],
                                 "line_number": line_number, "env_step": step,
                                 "metric": metric, "value": value,
                                 "episode": item.get("episode"),
                                 "validation_group": item.get("group"),
                                 "smooth_window": None})
    # Keep every raw episode point. A separate trailing average is available
    # for plotting and is explicitly tagged; it never replaces raw values.
    by_arm = defaultdict(list)
    for row in rows:
        if row["metric"] == "episode_return":
            by_arm[row["arm"]].append(row)
    for arm, episodes in by_arm.items():
        episodes.sort(key=lambda row: (row["env_step"], row["line_number"]))
        for index, item in enumerate(episodes):
            window = episodes[max(0, index - 49):index + 1]
            rows.append({**item, "metric": "episode_return_trailing_mean",
                         "value": mean(row["value"] for row in window),
                         "smooth_window": len(window)})
    return sorted(rows, key=lambda row: (row["arm"], row["metric"], row["env_step"],
                                         row["line_number"]))


def _plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = summary["by_n_load"]
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    paths = []
    for metric, ylabel, filename in (("si", "全束需求满足率 SI", "si_by_n.png"),
                                     ("sgm", "供需匹配程度 SGM", "sgm_by_n.png"),
                                     ("rho_below_08_fraction", "严重欠服务束比例 C/D<0.8", "shortfall_by_n.png")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for role in summary["reported_roles"]:
            for stage, linestyle in (("direct", "--"), ("final_B", "-")):
                selected = [r for r in rows if r["role"] == role and r["stage"] == stage]
                if len(selected) != 22:
                    continue
                values = [mean(r[metric] for r in selected if r["beam_count"] == n)
                          for n in N_VALUES]
                if any(value is None for value in values):
                    continue
                ax.plot(N_VALUES, values, linestyle, marker="o", markersize=3,
                        label=f"{ROLE_LABELS[role]} {'直接' if stage == 'direct' else '最终B'}")
        ax.set(xlabel="目标波束数 N", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path = folder / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    for arm in ("C0", "C1", "I0"):
        raw = [row for row in summary["training_curves"] if row["arm"] == arm]
        if not raw:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        validation = [row for row in raw if row["metric"] == "validation_si"]
        if validation:
            by_step = defaultdict(list)
            for row in validation:
                by_step[row["env_step"]].append(row["value"])
            x = sorted(by_step)
            axes[0].plot(x, [mean(by_step[step]) for step in x], marker="o", markersize=3)
        axes[0].set(xlabel="环境交互步", ylabel="固定验证全束SI", title=f"{arm} 验证")
        original = [row for row in raw if row["metric"] == "episode_return"]
        smoothed = [row for row in raw if row["metric"] == "episode_return_trailing_mean"]
        axes[1].plot([r["env_step"] for r in original], [r["value"] for r in original],
                     alpha=0.25, linewidth=0.6, label="原值")
        axes[1].plot([r["env_step"] for r in smoothed], [r["value"] for r in smoothed],
                     linewidth=1.4, label="后向最多50回合均值")
        axes[1].set(xlabel="环境交互步", ylabel="本臂回合奖励", title=f"{arm} 训练")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.tight_layout()
        path = folder / f"{arm.lower()}_training.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _material(summary):
    lines = ["# S2-P19正式报告素材（待主任务复核）", "",
             "## 预先写定的方法出处与预期", "",
             "- C0：离散SAC与势能差分思想；在项目中使用全束终态SI增量作逐步奖励，预期减少后续干扰造成的局部得分与最终服务错位。",
             "- C1：借鉴RLPD的离线/在线回放结合；在C0相同状态、动作、奖励与更新预算下混入真实贪心轨迹，预期更早达到强基线并减少无效探索，单纯模仿本身不保证超过教师。",
             "- I0：借鉴学习局部改写与大邻域搜索，在贪心完整初解上由候选动作SAC做多步可逆槽位重排；预期在同探测和时间预算下比规则/随机找到更好的可行解。首轮训练只优化直接SI，最终B收益是独立检验。",
             "- V2：S2-P18冻结的奖励/观测/回放组合模型，仅为历史锚点，本轮不计新训练。",
             "", "## 实际保存结果与缺项", ""]
    if not summary["formal_ready"]:
        reason = ("独立880景数值矩阵已齐，但训练曲线或其他交付证据仍缺；暂不发布完整正式结论。"
                  if summary["independent_matrix_complete"] else
                  "独立880景共同考核尚未达到完整、同源且逐束可核的交付门槛；本素材不提供正式方法优劣结论或正式成绩图。")
        lines += [reason, "",
                  "缺项：" + ("；".join(summary["missing"]) if summary["missing"] else "无已确认缺项，仍待完成记录"), ""]
    elif not summary["retained_arms"]:
        lines += ["首轮没有保留SAC臂；共同考核仅记录既定贪心/搜索与V2历史锚点。不能把这些对照说成SAC超过贪心。", ""]
    else:
        lines += ["已核完整共同考核：方法每景重复动作先平均，再按40个母场景配对。直接分配与最终B分别列数，不用模型＋B对比未细调贪心。", ""]
        for arm in summary["retained_arms"]:
            for stage in ("direct", "final_B"):
                pair = summary["comparisons"].get(f"{arm}_vs_greedy_{stage}")
                if pair:
                    lines.append(f"{arm}相对B1的{stage} SI母级均值差：{pair['mean_delta'] * 100:.3f}百分点；40母bootstrap 95%区间[{pair['ci95_low'] * 100:.3f}, {pair['ci95_high'] * 100:.3f}]百分点。")
                    dates = "、".join(f"{day}:{value * 100:.3f}百分点"
                                      for day, value in pair["per_date_mean_delta"].items())
                    lines.append(f"两日期分别为{dates}；每日期固定相邻4母分块敏感性区间[{pair['four_mother_block_ci95_low'] * 100:.3f}, {pair['four_mother_block_ci95_high'] * 100:.3f}]百分点。")
        lines += ["", "这些区间只描述两个冻结日期与单训练种子的场景变化；是否达到预注册主判据仍须结合逐N、SGM、基本服务、可行性、耗时与失败。", ""]
    for arm, training in summary["training_status"].items():
        if training["run_id"]:
            lines.append(f"{arm}训练：{training.get('env_step')}/{training.get('budget_env_steps')}步，阶段{training.get('stage')}，终止原因{training.get('completed_due_to') or training.get('stop_reason') or '未记录'}。")
    for group, cap in summary["optimizer_cap"].items():
        if cap["eligible_optimizer_actions"]:
            lines.append(f"{group}优化器明确time_limit：{cap['explicit_time_limit_count']}/{cap['eligible_optimizer_actions']}；终止原因未知{cap['unknown_termination_count']}，接近95%预算{cap['near_95pct_budget_count']}/{cap['near_budget_known_denominator']}（仅耗时标记，不算超时）。")
    lines += ["", "候选生成、初解、物理准备、优化器、核验及重复动作合计时间按保存字段分段；缺失精确成本维持null，不补零。并发条件若任务记录没有现场证据则列为未知，不做跨臂速度排名。",
              "本轮计时条件限制：" + summary["timing_interpretation"], ""]
    return "\n".join(lines)


def build(manifest, controller, plan, sources, output, *, render_plots=True):
    require(manifest.get("experiment_id") == EXPERIMENT and manifest.get("authorization"),
            "缺少S2-P19正式授权清单")
    require((plan or {}).get("experiment_id", EXPERIMENT) == EXPERIMENT,
            "controller plan属于其它实验")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    planned = final_tasks(plan)
    ports = plan_ports(plan)
    receipts = controller.get("tasks") or {}
    actions, jobs, missing = [], {}, []
    beam_staging = output / "per_beam.csv.tmp"
    beam_stream = beam_staging.open("w", encoding="utf-8-sig", newline="")
    beam_writer = csv.DictWriter(beam_stream, fieldnames=BEAM_COLUMNS, extrasaction="ignore")
    beam_writer.writeheader()
    beam_rows_written = 0

    def save_beams(action, beams):
        nonlocal beam_rows_written
        for beam_row in verified_beam_rows(action, beams):
            beam_writer.writerow(beam_row)
            beam_rows_written += 1

    try:
        for task_id, (role, stage) in planned.items():
            receipt = receipts.get(task_id) or {}
            legacy_run = (plan.get("completed_legacy") or {}).get(task_id)
            if legacy_run:
                require(not receipt.get("run_id") or receipt["run_id"] == legacy_run,
                        f"{task_id}: 历史完成任务与控制器run_id不匹配")
                receipt = {"run_id": legacy_run, "state": "completed"}
            run_id = receipt.get("run_id")
            planned_run = next((item.get("adopted_run_id") for item in plan.get("tasks", [])
                                if item["id"] == task_id), None)
            if planned_run and run_id:
                require(run_id == planned_run, f"{task_id}: 实际run_id与四端计划不匹配")
            if receipt.get("state") != "completed" or not run_id:
                missing.append(f"{task_id}: controller尚未完成")
                continue
            source, reason = source_for_task(
                task_id, run_id, sources, receipt, manifest, ports[task_id],
                require_port_evidence=bool(plan.get("completed_legacy")) and not legacy_run)
            if source is None:
                missing.append(f"{task_id}: {reason}")
                continue
            jobs[task_id] = {"run_id": run_id, "results_path": source["path"],
                             "results_sha256": source["sha256"],
                             "physical_contract_sha256": source["physical_contract_sha256"],
                             "saved_method_counts": {},
                             "saved_fallback_rows": 0,
                             "unknown_exact_cost_rows": 0,
                             "port": ports[task_id],
                             "concurrency": source["status"].get("concurrency") or
                             source["job"].get("concurrency"),
                             "concurrent_run_ids": source["status"].get("concurrent_run_ids"),
                             "concurrency_observed_at": source["status"].get("concurrency_observed_at"),
                             "thread_count": source["status"].get("threads") or
                             source["job"].get("threads"),
                             "device": source["status"].get("gpu") or
                             (source["status"].get("preflight") or {}).get("gpu") or
                             source["job"].get("device"),
                             "software_preflight": source["status"].get("preflight")}
            refinement = (source["config"].get("power_refinement") or {})
            optimizer_budget = number(refinement.get("max_time_seconds"))
            method_counts = Counter()
            result_info = {}
            for row in iter_saved_scenes(source["path"], result_info):
                method_counts[row.get("method", "unknown")] += 1
                jobs[task_id]["saved_fallback_rows"] += bool((row.get("solver") or {}).get(
                    "fallback_used"))
                jobs[task_id]["unknown_exact_cost_rows"] += bool((row.get(
                    "failed_attempt_fallback") or {}).get("original_attempt_exact_cost_unknown"))
                normalized = normalize_result(task_id, role, stage, row,
                                              source["spec"], run_id, source["sha256"],
                                              source["physical_contract_sha256"], save_beams)
                if normalized is not None:
                    normalized["optimizer_time_budget_seconds"] = optimizer_budget
                    actions.append(normalized)
            jobs[task_id]["saved_method_counts"] = dict(method_counts)
            metadata_run = (result_info.get("metadata") or {}).get("run_id")
            if metadata_run is not None:
                require(metadata_run == run_id, f"{task_id}: 结果元数据run_id不匹配")
    finally:
        beam_stream.close()
    beam_staging.replace(output / "per_beam.csv")
    scenes = collapse_repeats(actions)
    nload = by_n_load(scenes)
    curves = training_curves(sources, manifest)
    direct_index = {(row["role"], row["mother_scene_id"], row["beam_count"],
                     row["load_multiplier"], row["repeat_index"]): row
                    for row in actions if row["stage"] == "direct"}
    for row in actions:
        if row["stage"] != "final_B":
            continue
        key = (row["role"], row["mother_scene_id"], row["beam_count"],
               row["load_multiplier"], row["repeat_index"])
        direct = direct_index.get(key)
        if direct is None:
            missing.append(f"{row['role']}/final_B: 对应直接分配case未保存")
            continue
        require(row["scene_semantic_sha256"] == direct["scene_semantic_sha256"],
                f"{row['role']}: 直接与最终B使用了不同静态场景")
        if row["scene_content_sha256"] and direct["scene_content_sha256"]:
            require(row["scene_content_sha256"] == direct["scene_content_sha256"],
                    f"{row['role']}: 直接与最终B的完整源NPZ不同")
        if row["physical_contract_sha256"] and direct["physical_contract_sha256"]:
            require(row["physical_contract_sha256"] == direct["physical_contract_sha256"],
                    f"{row['role']}: 直接与最终B物理合同不同")
        if row["source_evaluation_run_id"] is not None:
            require(row["source_evaluation_run_id"] == direct["run_id"],
                    f"{row['role']}: B初解任务ID与直接分配不符")
        if row["source_initial_case_sha256"] and direct["case_sha256"]:
            require(row["source_initial_case_sha256"] == direct["case_sha256"],
                    f"{row['role']}: B初解case SHA与直接分配不符")
    planned_roles = sorted({role for role, _ in planned.values() if role != "baseline"})
    retained = [arm for arm in ("C0", "C1", "I0") if arm in planned_roles]
    expected_roles = sorted(set(planned_roles) | ({"greedy", "rule", "random"}
                             if "baseline_test_20418" in planned else set()))
    matrix_ready = bool(planned) and not missing and all(
        _full_group(scenes, role, stage) for role in expected_roles
        for stage in ("direct", "final_B"))
    for role in expected_roles:
        for stage in ("direct", "final_B"):
            if not _full_group(scenes, role, stage):
                missing.append(f"{role}/{stage}: 880景×规定重复及逐束核验未齐")
                selected = [row for row in scenes if row["role"] == role and row["stage"] == stage]
                if selected and any(not row["scene_content_sha256"] for row in selected):
                    missing.append(f"{role}/{stage}: 完整源NPZ内容尚未逐SHA核验")
                if selected and any(not row["physical_contract_sha256"] for row in selected):
                    missing.append(f"{role}/{stage}: 冻结环境/源码/单位/预算合同缺失")
                if selected and any(not row["mother_source_sha256"] or
                                    not row["selected_time_utc"] for row in selected):
                    missing.append(f"{role}/{stage}: 母时刻或母来源SHA缺失")
    curve_methods = {(row["arm"], row["metric"]) for row in curves}
    for arm in retained:
        if not ((sources.get("training") or {}).get(arm) or {}).get("updates"):
            missing.append(f"{arm}: updates原始学习日志缺失")
        for metric in ("episode_return", "validation_si"):
            if (arm, metric) not in curve_methods:
                missing.append(f"{arm}: {metric}原始训练曲线缺失")
    formal_ready = matrix_ready and not missing
    comparisons = {}
    if formal_ready:
        for arm in ("C0", "C1", "I0", "V2", "rule", "random"):
            if arm not in expected_roles:
                continue
            for stage in ("direct", "final_B"):
                pair = mother_pair(scenes, arm, "greedy", stage)
                if pair:
                    comparisons[f"{arm}_vs_greedy_{stage}"] = pair
        if "C1" in retained and "C0" in retained:
            for stage in ("direct", "final_B"):
                comparisons[f"C1_vs_C0_{stage}"] = mother_pair(scenes, "C1", "C0", stage)
        if "I0" in retained:
            for baseline in ("rule", "random"):
                for stage in ("direct", "final_B"):
                    comparisons[f"I0_vs_{baseline}_{stage}"] = mother_pair(scenes, "I0", baseline, stage)
    timing = {f"{role}/{stage}": {key: timing_stats([row[key] for row in actions
                                                   if row["role"] == role and row["stage"] == stage])
                                 for key in ("initial_seconds", "candidate_seconds",
                                             "physical_probe_seconds", "physics_setup_seconds",
                                             "optimizer_seconds", "verification_seconds", "total_seconds")}
              for role in expected_roles for stage in ("direct", "final_B")}
    scene_timing = {f"{role}/{stage}": {key: timing_stats([
        row[key + "_sum_repeats"] for row in scenes
        if row["role"] == role and row["stage"] == stage])
        for key in ("initial_seconds", "candidate_seconds", "physical_probe_seconds",
                    "physics_setup_seconds", "optimizer_seconds", "verification_seconds",
                    "total_seconds")}
        for role in expected_roles for stage in ("direct", "final_B")}
    summary = {"schema": "s2p19_report_material_v1", "experiment_id": EXPERIMENT,
               "formal_ready": formal_ready,
               "independent_matrix_complete": matrix_ready,
               "status": ("formal_input_complete" if formal_ready else
                          "awaiting_final_plan" if not planned else "partial_or_unverified"),
               "retained_arms": retained,
               "reported_roles": expected_roles,
               "planned_final_tasks": planned,
               "controller_state_source": controller.get("source_path"),
               "jobs": jobs, "missing": sorted(set(missing)),
               "training_status": _training_status(manifest, controller, sources),
               "per_action_count": len(actions), "per_scene_count": len(scenes),
               "by_n_load": nload, "comparisons": comparisons,
               "timing": timing, "scene_repeat_timing": scene_timing,
               "optimizer_cap": optimizer_cap_stats(actions),
               "per_beam_rows": beam_rows_written,
               "per_beam_path": str(output / "per_beam.csv"),
               "timing_interpretation": ((manifest.get("common_assessment") or {}).get("timing_limitations")
                  or "四端并发、设备变化、重启与监测空窗须据现场记录解释；不可据此宣称严格同条件耗时。"),
               "training_curves": curves,
               "metric_semantics": "all target beams after last action and later interference; C is modeled link capacity; min(C,D) is estimated carried demand",
               "statistical_unit": "actions average within derivative scene; 22 derivatives average within mother; bootstrap resamples 40 mothers; four-mother date-stratified blocks are sensitivity only",
               "concurrency_limit": "actual job-level concurrency retained when available; missing is unknown, never inferred from ports"}
    action_cols = ("task_id", "run_id", "role", "stage", "method", "scene_sha256",
                   "source_npz_local_path", "scene_content_sha256",
                   "physical_contract_sha256", "mother_source_sha256", "selected_time_utc",
                   "scene_semantic_sha256",
                   "mother_scene_id", "beam_count", "load_multiplier", "repeat_index",
                   "action_seed", "result_path_sha256", "case_path", "case_sha256",
                   "source_evaluation_run_id", "source_initial_case_sha256",
                   "si", "sgm", "basic_met_count", "effective_carried_bps", "power_w",
                   "slots_total", "rho_min", "rho_p05", "rho_median", "rho_p95",
               "rho_max", "rho_below_08_fraction", "rho_below_1_fraction",
                   "resource_legal", "zero_capacity_count",
                   "beam_audit_available", "beam_audit_missing_reason", "fallback_used",
                   "solver_termination", "solver_elapsed_seconds",
                   "optimizer_time_budget_seconds", "missing_exact_cost", "initial_seconds",
                   "candidate_seconds", "physical_probe_seconds", "physics_setup_seconds",
                   "optimizer_seconds", "verification_seconds", "total_seconds")
    scene_cols = tuple(scenes[0]) if scenes else ("role", "stage", "scene_sha256",
                                              "mother_scene_id", "beam_count", "load_multiplier",
                                              "repeats_observed", "repeats_expected", "complete_repeats")
    nload_cols = tuple(nload[0]) if nload else ("role", "stage", "beam_count",
                                               "load_multiplier", "scene_count")
    write_csv(output / "per_action.csv", actions, action_cols)
    write_csv(output / "optimizer_cap.csv",
              [{"role": key.split("/")[0], "stage": key.split("/")[1], **value}
               for key, value in summary["optimizer_cap"].items()],
              ("role", "stage", "eligible_optimizer_actions",
               "explicit_time_limit_count", "known_other_termination_count",
               "unknown_termination_count", "explicit_time_limit_fraction",
               "near_95pct_budget_count", "near_budget_known_denominator",
               "near_budget_fraction", "near_budget_is_not_time_limit"))
    write_csv(output / "per_scene.csv", scenes, scene_cols)
    write_csv(output / "by_n_load.csv", nload, nload_cols)
    write_csv(output / "training_curves.csv", curves,
              ("arm", "run_id", "source", "source_sha256", "line_number",
               "env_step", "metric", "value", "episode", "validation_group",
               "smooth_window"))
    pair_rows = [{"comparison": name, "mother_scene_id": mother,
                  "stage": pair["stage"], "delta_si": delta,
                  "overall_delta_si": pair["mean_delta"],
                  "ci95_low": pair["ci95_low"], "ci95_high": pair["ci95_high"]}
                 for name, pair in comparisons.items() for mother, delta in pair["per_mother"].items()]
    write_csv(output / "mother_paired.csv", pair_rows,
              ("comparison", "mother_scene_id", "stage", "delta_si",
               "overall_delta_si", "ci95_low", "ci95_high"))
    if matrix_ready and render_plots:
        summary["figures"] = _plots(summary, output)
    else:
        summary["figures"] = []
    atomic_json(output / "summary.json", summary)
    (output / "report_material.md").write_text(_material(summary), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--controller", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    manifest = read_json(args.manifest)
    require(manifest is not None, "正式运行清单缺失")
    controller = read_json(args.controller, {})
    controller["source_path"] = str(args.controller) if args.controller.is_file() else None
    plan = read_json(args.plan, {})
    sources = read_json(args.source_index, {}) if args.source_index else {}
    result = build(manifest, controller, plan, sources or {}, args.output,
                   render_plots=not args.no_plots)
    print(json.dumps({"status": result["status"], "formal_ready": result["formal_ready"],
                      "missing": len(result["missing"]),
                      "summary_path": str(args.output / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
