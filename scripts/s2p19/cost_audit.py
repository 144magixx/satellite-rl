"""Read-only S2-P19 decision/probe accounting from verified local archives.

The default command checks inventory only. ``--execute`` reads every selected
case and training episode locally; it never runs a policy or optimizer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from . import archive_sources, report


DIRECT = {"I0_final_direct_20474": {"model": 2640},
          "baseline_test_20418": {"greedy": 880, "rule": 880, "random": 2640}}
TRAINING = ("C0", "C1", "I0")
EXPECTED_UNITS = {"C0": 400000, "C1": 400000, "I0": 397112}
CASE_FIELDS = ("task_id", "run_id", "port", "method", "scene_sha256",
               "mother_scene_id", "repeat_index", "beam_count", "case_member",
               "case_sha256", "decisions", "probes", "invalid", "repeats",
               "proposals_generated", "trace_rows", "stop_actions",
               "rejected_actions", "protected_output", "fallback_used",
               "last_reason")
MAX_CASE_BYTES = 16 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _receipt(path, run_id, inventory_sha):
    path = Path(path)
    receipt = read_json(path)
    require(receipt.get("schema") == "s2p19.archive.v1" and
            receipt.get("run_id") == run_id and
            receipt.get("inventory_sha256") == inventory_sha and
            receipt.get("job_stage") == "completed",
            f"archive receipt identity differs: {run_id}")
    members = {item["member"]: item for item in receipt.get("members") or []}
    require(len(members) == receipt.get("member_count") and
            len(members) == len(receipt.get("members") or []),
            f"archive member inventory differs: {run_id}")
    for volume in receipt.get("volumes") or []:
        local = path.parent / volume["name"]
        require(local.is_file() and local.stat().st_size == volume["bytes"],
                f"ZIP volume absent or wrong size: {local}")
    return receipt, members


def check_inputs(source_index, manifest, plan, state, teacher_receipt=None):
    """Bounded stat/receipt check; no large ZIP or JSONL bytes are read."""
    index, published = read_json(source_index), read_json(manifest)
    plan_data, state_data = read_json(plan), read_json(state)
    require(index.get("schema") == "s2p19.source_index.v1" and
            published.get("experiment_id") == "S2-P19" and
            plan_data.get("experiment_id") == "S2-P19" and
            state_data.get("plan_sha256") == digest(plan),
            "S2-P19 source/manifest/four-lane state binding differs")
    active = {row["id"]: row for row in plan_data.get("tasks") or []}
    legacy = plan_data.get("completed_legacy") or {}
    missing, bound = [], {}
    for task_id, methods in DIRECT.items():
        spec = (index.get("jobs") or {}).get(task_id)
        if not spec:
            missing.append(f"{task_id}: source_index.jobs missing")
            continue
        try:
            if task_id in legacy:
                expected_run, expected_port = legacy[task_id], 20418
            else:
                expected_run = state_data["tasks"][task_id]["run_id"]
                expected_port = active[task_id]["port"]
            require(spec["run_id"] == expected_run and
                    spec.get("port") == expected_port,
                    f"{task_id}: four-lane run/port differs")
            receipt, members = _receipt(spec["archive_receipt_path"], spec["run_id"],
                                        spec["archive_inventory_sha256"])
            require(receipt.get("port") == spec.get("port"), f"{task_id}: port differs")
            cases = [name for name in members if name.startswith("job/cases/") and
                     name.endswith(".json") and name != "job/cases/contract.json"]
            require(len(cases) == sum(methods.values()),
                    f"{task_id}: expected case count differs")
            for name in cases:
                require(0 < members[name]["bytes"] <= MAX_CASE_BYTES,
                        f"{task_id}: case size exceeds bounded reader")
            bound[task_id] = {"run_id": spec["run_id"], "port": spec.get("port"),
                              "case_count": len(cases), "receipt": spec["archive_receipt_path"]}
        except (KeyError, OSError, ValueError) as error:
            missing.append(f"{task_id}: {error}")
    for arm in TRAINING:
        spec = (index.get("training") or {}).get(arm)
        if not spec:
            missing.append(f"{arm}: source_index.training missing")
            continue
        try:
            expected_run = published["tasks"][arm]["run_id"]
            require(spec["run_id"] == expected_run, f"{arm}: manifest run_id differs")
            receipt, members = _receipt(spec["archive_receipt_path"], spec["run_id"],
                                        spec["archive_inventory_sha256"])
            require(receipt.get("port") == spec.get("port"), f"{arm}: port differs")
            for name in ("job/job.json", "job/config.json", "job/status.json",
                         "job/episodes.jsonl", "source/source_manifest.json"):
                require(name in members, f"{arm}: missing {name}")
            episodes = spec.get("episodes") or {}
            require(episodes.get("sha256") == members["job/episodes.jsonl"]["sha256"],
                    f"{arm}: episodes source SHA differs")
            if arm in ("C0", "C1"):
                updates = spec.get("updates") or {}
                require("job/updates.jsonl" in members and
                        updates.get("sha256") == members["job/updates.jsonl"]["sha256"],
                        f"{arm}: updates source SHA differs")
            bound[arm] = {"run_id": spec["run_id"], "port": spec.get("port"),
                          "receipt": spec["archive_receipt_path"]}
        except (KeyError, OSError, ValueError) as error:
            missing.append(f"{arm}: {error}")
    if teacher_receipt is None:
        missing.append("C1 teacher: --teacher-receipt absent")
    else:
        try:
            path = Path(teacher_receipt)
            teacher = read_json(path)
            require(teacher.get("schema") == "s2p19.archive.v1" and
                    teacher.get("experiment_id") == "S2-P19" and
                    teacher.get("job_stage") == "completed", "teacher receipt invalid")
            _, members = _receipt(path, teacher["run_id"], teacher["inventory_sha256"])
            require("job/teacher/manifest.json" in members and
                    "job/job.json" in members, "teacher manifest/job absent")
            bound["teacher"] = {"run_id": teacher["run_id"], "receipt": str(path)}
        except (KeyError, OSError, ValueError) as error:
            missing.append(f"C1 teacher: {error}")
    return {"schema": "s2p19.cost_input_check.v1", "ready": not missing,
            "missing": missing, "bound": bound,
            "plan_sha256": state_data["plan_sha256"],
            "mode": "metadata_only_no_large_member_read"}


def _member_json(receipt, name):
    reference = archive_sources.member_ref(receipt, name)
    require(reference is not None, f"archive member missing: {name}")
    return json.loads(archive_sources.read_bytes_source(reference)), reference


def _nonnegative_int(value, label, *, limit=None):
    require(type(value) is int and value >= 0 and (limit is None or value <= limit),
            f"invalid {label}: {value!r}")
    return value


def _case_row(task_id, spec, receipt_path, name):
    reference = archive_sources.member_ref(receipt_path, name)
    require(reference is not None and reference["bytes"] <= MAX_CASE_BYTES,
            f"{task_id}: case absent/oversized: {name}")
    wrapped = json.loads(archive_sources.read_bytes_source(reference))
    case = (wrapped.get("body") or {}).get("result", wrapped)
    method = case.get("method")
    require(method in DIRECT[task_id], f"{task_id}: unexpected method {method}")
    metrics = case.get("metrics") or {}
    trace = case.get("trace")
    decisions = _nonnegative_int(metrics.get("decisions"), "decisions", limit=32)
    probes = _nonnegative_int(metrics.get("probes"), "probes", limit=32)
    invalid = _nonnegative_int(metrics.get("invalid"), "invalid", limit=32)
    repeats = _nonnegative_int(metrics.get("repeats"), "repeats", limit=32)
    require(probes <= decisions and invalid + repeats <= decisions and
            isinstance(trace, list) and len(trace) == decisions,
            f"{task_id}: decision/probe/trace account differs: {name}")
    reasons = Counter()
    rejected = 0
    previous_probes = 0
    for number, step in enumerate(trace, 1):
        require(step.get("decisions") == number and
                type(step.get("probes")) is int and
                previous_probes <= step["probes"] <= previous_probes + 1,
                f"{task_id}: nonsequential trace: {name}")
        previous_probes = step["probes"]
        reasons[step.get("reason") or "unknown"] += 1
        rejected += bool((step.get("actual_action") or {}).get("rejection_reason"))
    require((not trace or previous_probes == probes) and
            reasons["repeat"] == repeats and reasons["invalid_or_stale"] == invalid,
            f"{task_id}: trace/metric counts differ: {name}")
    proposals = metrics.get("proposals_generated")
    if proposals is not None:
        proposals = _nonnegative_int(proposals, "proposals_generated")
    scene = case.get("scene_metadata") or {}
    scene_sha = case.get("scene_sha256")
    repeat = case.get("repeat_index")
    beams = case.get("beams") or []
    require(isinstance(scene_sha, str) and len(scene_sha) == 64 and
            isinstance(scene.get("mother_scene_id"), str) and
            type(repeat) is int and 0 <= repeat < (3 if method in ("model", "random") else 1) and
            len(beams) in report.N_VALUES,
            f"{task_id}: scene/repeat/beam identity absent: {name}")
    fallback = case.get("fallback_used", metrics.get("fallback_used"))
    require(fallback is None or type(fallback) is bool, "fallback field invalid")
    return {"task_id": task_id, "run_id": spec["run_id"], "port": spec.get("port"),
            "method": method, "scene_sha256": scene_sha,
            "mother_scene_id": scene["mother_scene_id"], "repeat_index": repeat,
            "beam_count": len(beams), "case_member": name,
            "case_sha256": reference["sha256"], "decisions": decisions,
            "probes": probes, "invalid": invalid, "repeats": repeats,
            "proposals_generated": proposals,
            "trace_rows": len(trace), "stop_actions": reasons["stop"],
            "rejected_actions": rejected,
            "protected_output": trace[-1].get("protected_output") if trace else None,
            "fallback_used": fallback, "last_reason": trace[-1].get("reason") if trace else None,
            "reason_counts": dict(reasons)}


def _stats(values):
    ordered = sorted(values)
    return {"count": len(ordered), "total": sum(ordered),
            "mean": sum(ordered) / len(ordered) if ordered else None,
            "median": statistics.median(ordered) if ordered else None,
            "p95": report.percentile(ordered, 0.95) if ordered else None,
            "max": ordered[-1] if ordered else None}


def _training_rows(arm, spec, receipt_path):
    job, job_ref = _member_json(receipt_path, "job/job.json")
    config, config_ref = _member_json(receipt_path, "job/config.json")
    status, status_ref = _member_json(receipt_path, "job/status.json")
    source_manifest, manifest_ref = _member_json(receipt_path,
                                                  "source/source_manifest.json")
    require(job.get("run_id") == spec["run_id"] and
            status.get("run_id") == spec["run_id"] and status.get("stage") == "completed",
            f"{arm}: frozen job/status endpoint differs")
    endpoint = _nonnegative_int(status.get("env_step"), f"{arm} env_step")
    if arm in ("C0", "C1"):
        require(endpoint == EXPECTED_UNITS[arm] and
                config.get("training", {}).get("total_steps") == 400000 and
                status.get("budget_env_steps") == 400000,
                f"{arm}: beam budget differs")
    else:
        i0 = config.get("i0") or config.get("training") or {}
        require(endpoint == EXPECTED_UNITS[arm] and
                status.get("completed_due_to") == "episode_cap" and
                status.get("budget_env_steps") == 400000 and
                status.get("episodes") == i0.get("max_episodes") == 12500 and
                i0.get("total_steps") == 400000 and
                i0.get("max_decisions") == 32 and i0.get("max_probes") == 32,
                "I0: actual macro endpoint or 32/32 contract differs")
    reference = archive_sources.member_ref(receipt_path, "job/episodes.jsonl")
    require(reference is not None and reference["sha256"] == spec["episodes"]["sha256"],
            f"{arm}: episode SHA differs from source index")
    episodes = {}
    duplicates = 0
    with archive_sources.open_text_source(reference) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            number = _nonnegative_int(row.get("episode"), f"{arm} episode")
            fingerprint = hashlib.sha256(json.dumps(
                row, sort_keys=True, ensure_ascii=False,
                separators=(",", ":")).encode("utf-8")).hexdigest()
            if number in episodes:
                require(episodes[number]["fingerprint"] == fingerprint,
                        f"{arm}: conflicting duplicate episode {number}")
                duplicates += 1
                continue
            metrics = row.get("metrics") or {}
            complete = ((row.get("terminal_snapshot") or {}).get("complete_episode")
                        if arm == "I0" else row.get("terminal", True))
            detail = {"fingerprint": fingerprint, "complete": bool(complete),
                      "env_step": row.get("env_step"),
                      "beam_count": row.get("beam_count"),
                      "decisions": metrics.get("decisions") if arm == "I0" else row.get("length"),
                      "probes": metrics.get("probes") if arm == "I0" else None}
            if arm == "I0":
                detail["decisions"] = _nonnegative_int(detail["decisions"],
                                                       "I0 episode decisions", limit=32)
                detail["probes"] = _nonnegative_int(detail["probes"],
                                                    "I0 episode probes", limit=32)
                require(detail["probes"] <= detail["decisions"] and
                        type(detail["beam_count"]) is int and
                        detail["beam_count"] in report.N_VALUES,
                        "I0 episode physical/scene cost fields missing")
            episodes[number] = detail
    completed = [row for row in episodes.values() if row["complete"]]
    partial = [row for row in episodes.values() if not row["complete"]]
    snapshots = archive_sources.member_ref(receipt_path, "job/episode_snapshots.jsonl")
    snapshot_lines, snapshot_episodes = 0, set()
    if snapshots:
        with archive_sources.open_text_source(snapshots) as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                snapshot_lines += 1
                snapshot_episodes.add(_nonnegative_int(row.get("episode"),
                                                       "I0 partial snapshot episode"))
    updates = _nonnegative_int(status.get("update_step", status.get("updates")),
                               f"{arm} update_step")
    result = {"run_id": spec["run_id"], "status_sha256": status_ref["sha256"],
              "job_sha256": job_ref["sha256"], "config_sha256": config_ref["sha256"],
              "source_manifest_sha256": manifest_ref["sha256"],
              "improvement_environment_source_sha256":
                  (source_manifest.get("files") or {}).get(
                      "src/satellite_rl/sac_improvement/environment.py"),
              "episodes_sha256": reference["sha256"], "recorded_env_step": endpoint,
              "recorded_update_step": updates,
              "completed_unique_episodes": len(completed),
              "terminal_partial_unique_episodes": len(partial),
              "duplicate_identical_episode_rows": duplicates,
              "resume_snapshot_rows_not_added_to_cost": snapshot_lines,
              "resume_snapshot_unique_episode_ids": len(snapshot_episodes),
              "snapshot_source_sha256": snapshots["sha256"] if snapshots else None,
              "unlogged_recovery_repeat_cost": None,
              "model_selection": status.get("model_selection") or
                                 job.get("model_selection")}
    if arm == "I0":
        recorded = completed + partial
        observed_steps = sum(row["decisions"] for row in recorded)
        result.update({"macro_decisions_in_unique_episode_rows": observed_steps,
                       "macro_decisions_match_status": observed_steps == endpoint,
                       "candidate_truth_probes_in_unique_episode_rows_lower_bound":
                           sum(row["probes"] for row in recorded),
                       "greedy_initializer_beam_decisions_in_unique_episode_rows_lower_bound":
                           sum(row["beam_count"] for row in recorded),
                       "greedy_initializer_unlogged_restart_extra": None,
                       "probe_unlogged_restart_extra": None,
                       "episode_cap": status.get("episodes", status.get("episode"))})
    else:
        result["beam_decisions_match_status"] = (sum(row["decisions"] for row in
                                                    completed + partial) == endpoint)
        result["beam_decisions"] = endpoint
        result["candidate_truth_probes"] = None
    model_ref = archive_sources.member_ref(receipt_path,
                                            "weights/model_dir/final/model.json")
    if model_ref:
        model = json.loads(archive_sources.read_bytes_source(model_ref))
        provenance = model.get("provenance") or {}
        require(model.get("run_id", provenance.get("run_id")) == spec["run_id"] and
                model.get("env_step", provenance.get("env_step")) == endpoint,
                f"{arm}: final model endpoint differs")
        model_updates = model.get("update_step", model.get("updates"))
        require(model_updates is None or model_updates == updates,
                f"{arm}: final model update count differs")
        result["parameter_count"] = model.get("parameter_count")
        result["model_metadata_sha256"] = model_ref["sha256"]
        result["model_selection"] = (provenance.get("selection") or
                                     result["model_selection"])
    else:
        result["parameter_count"] = None
        result["model_metadata_sha256"] = None
    return result, job, config


def _teacher(teacher_receipt, c1_job, c1_config):
    source = c1_job.get("demonstration_source") or {}
    demo = (c1_config.get("training") or {}).get("demonstrations") or {}
    receipt = read_json(teacher_receipt)
    require(source.get("run_id") == receipt.get("run_id"),
            "C1 frozen teacher run differs")
    teacher_job, job_ref = _member_json(teacher_receipt, "job/job.json")
    status, _ = _member_json(teacher_receipt, "job/status.json")
    manifest, manifest_ref = _member_json(teacher_receipt,
                                           "job/teacher/manifest.json")
    require(teacher_job.get("run_id") == source["run_id"] and
            status.get("stage") == "completed" and
            job_ref["sha256"] == source.get("job_sha256") and
            manifest_ref["sha256"] == source.get("manifest_sha256") ==
            demo.get("manifest_sha256") and
            manifest.get("source_set_sha256") == demo.get("source_set_sha256"),
            "C1 teacher job/manifest/config binding differs")
    records = manifest.get("records") or []
    require(manifest.get("format") == "s2p19_greedy_trajectory_v1" and
            len(records) == 576 and all(type(row.get("transition_count")) is int and
                                        50 <= row["transition_count"] <= 200 for row in records),
            "C1 teacher trajectory metadata incomplete")
    return {"run_id": source["run_id"], "manifest_sha256": manifest_ref["sha256"],
            "job_sha256": job_ref["sha256"], "trajectory_count": len(records),
            "transitions_from_frozen_teacher_manifest": sum(
                row["transition_count"] for row in records),
            "note": "frozen metadata count; shards are retained in verified archive"}


def execute(source_index, manifest, plan, state, teacher_receipt, output_dir):
    check = check_inputs(source_index, manifest, plan, state, teacher_receipt)
    require(check["ready"], "cost audit input incomplete: " + "; ".join(check["missing"]))
    index, published = read_json(source_index), read_json(manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "cost_per_case.csv"
    temp_csv = csv_path.with_suffix(".csv.tmp")
    grouped = defaultdict(lambda: defaultdict(list))
    reason_totals = defaultdict(Counter)
    fallback_coverage = defaultdict(Counter)
    frozen_probe_source_shas = {}
    counts = Counter()
    identity = defaultdict(set)
    with temp_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CASE_FIELDS)
        writer.writeheader()
        for task_id, expected in DIRECT.items():
            spec = index["jobs"][task_id]
            source, problem = report.source_for_task(
                task_id, spec["run_id"], index, {}, published, spec.get("port"))
            require(source is not None and problem is None and
                    source["physical_contract_sha256"],
                    f"{task_id}: physical source contract absent")
            source_manifest = read_json(spec["source_manifest_path"])
            frozen_probe_source_shas[task_id] = (source_manifest.get("files") or {}).get(
                "src/satellite_rl/sac_improvement/environment.py")
            require(isinstance(frozen_probe_source_shas[task_id], str) and
                    len(frozen_probe_source_shas[task_id]) == 64,
                    f"{task_id}: frozen probe implementation SHA absent")
            receipt_path = spec["archive_receipt_path"]
            receipt, members = _receipt(receipt_path, spec["run_id"],
                                        spec["archive_inventory_sha256"])
            for name in sorted(members):
                if (not name.startswith("job/cases/") or not name.endswith(".json") or
                        name == "job/cases/contract.json"):
                    continue
                row = _case_row(task_id, spec, receipt_path, name)
                key = (row["method"], row["scene_sha256"], row["repeat_index"])
                require(key not in identity[task_id], f"{task_id}: duplicate case identity")
                identity[task_id].add(key)
                counts[(task_id, row["method"])] += 1
                for field in ("beam_count", "decisions", "probes", "invalid", "repeats",
                              "proposals_generated", "stop_actions", "rejected_actions"):
                    if type(row[field]) is int:
                        grouped[row["method"]][field].append(row[field])
                reason_totals[row["method"]].update(row["reason_counts"])
                fallback_coverage[row["method"]]["missing" if row["fallback_used"] is None
                                                 else "true" if row["fallback_used"] else "false"] += 1
                writer.writerow({field: row[field] for field in CASE_FIELDS})
            for method, expected_count in expected.items():
                require(counts[(task_id, method)] == expected_count,
                        f"{task_id}/{method}: final case grid count differs")
    require(len(set(frozen_probe_source_shas.values())) == 1,
            "I0 and baseline probe accounting source differs")
    training = {}
    c1_job = c1_config = None
    for arm in TRAINING:
        spec = index["training"][arm]
        value, job, config = _training_rows(arm, spec, spec["archive_receipt_path"])
        training[arm] = value
        if arm == "C1":
            c1_job, c1_config = job, config
    require(training["I0"]["improvement_environment_source_sha256"] ==
            frozen_probe_source_shas["I0_final_direct_20474"],
            "I0 training/direct probe accounting source differs")
    teacher = _teacher(teacher_receipt, c1_job, c1_config)
    require(teacher["transitions_from_frozen_teacher_manifest"] ==
            published.get("budget_ledger", {}).get("teacher_transitions"),
            "teacher frozen transition count differs from run manifest")
    temp_csv.replace(csv_path)
    summary = {"schema": "s2p19.cost_audit.v1", "experiment_id": "S2-P19",
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "source_index_path": str(source_index),
               "source_index_sha256": digest(source_index),
               "frozen_probe_implementation_sha256": frozen_probe_source_shas,
               "case_csv_path": str(csv_path),
               "case_csv_sha256": digest(csv_path),
               "case_counts": {task: {method: counts[(task, method)] for method in methods}
                               for task, methods in DIRECT.items()},
               "by_method": {method: {field: _stats(values) for field, values in fields.items()}
                             for method, fields in grouped.items()},
               "trace_reason_totals": {method: dict(counter)
                                        for method, counter in reason_totals.items()},
               "fallback_field_coverage": {method: dict(counter)
                                           for method, counter in fallback_coverage.items()},
               "training": training, "teacher": teacher,
               "limits": {"per_case_decisions": 32, "per_case_probes": 32,
                          "I0_authorized_macro_budget": 400000,
                          "I0_authorized_episode_cap": 12500,
                          "teacher_transition_cap": 115200},
               "interpretation": [
                   "Direct cases are separate actions; repeats of the same scene are not independent scenes.",
                   "Decision/probe counts come from saved metrics and sequential trace, not SI/SGM.",
                   "A probe in frozen ImprovementEnv is a candidate physics evaluation; greedy initialization is counted separately.",
                   "Unique episode rows give a lower bound on greedy resets and probes; unlogged restart work is unknown.",
                   "Teacher transitions are frozen preparation metadata, separate from online training interactions.",
                   "No solver, policy, training, or new evaluation was executed."]}
    path = output / "cost_audit.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--teacher-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    check = check_inputs(args.source_index, args.manifest, args.plan,
                         args.state, args.teacher_receipt)
    if not args.execute:
        print(json.dumps(check, ensure_ascii=False))
        return 0
    if not check["ready"]:
        raise SystemExit("cost audit source incomplete; no long read: " +
                         "; ".join(check["missing"]))
    result = execute(args.source_index, args.manifest, args.plan, args.state,
                     args.teacher_receipt, args.output_dir)
    print(json.dumps({"status": "completed", "summary": str(args.output_dir / "cost_audit.json"),
                      "case_csv": result["case_csv_path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
