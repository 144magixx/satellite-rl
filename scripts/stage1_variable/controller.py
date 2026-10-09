"""Persistent, bounded four-lane S1-V10 workbench task coordinator.

``tick`` performs one observation/transition; ``run`` repeats it as a
background process. Both require the exact validated frozen commit. Each task
has an immutable request ID and resumes, when needed, through its own API ID.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.stage1_variable import data_controller, ops


ROOT = ops.LOCAL / "controller"
STATE = ROOT / "state.json"
RUN_MANIFEST = ops.LOCAL / "run_manifest.json"
LOCK = ROOT / "controller.lock"
NONLEARNING = ("R01", "R02", "R03", "R04", "R05", "R06", "R07", "R10", "R11", "R12")
LEARNING = ("R08", "R09")
DEV_PORTS = {"R01": 20473, "R02": 20474, "R03": 20473, "R04": 20474,
             "R05": 20473, "R06": 20474, "R07": 20473, "R08": 20418,
             "R09": 20424, "R10": 20473, "R11": 20474, "R12": 20474}
TEST_PORTS = DEV_PORTS


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


@contextmanager
def exclusive():
    ROOT.mkdir(parents=True, exist_ok=True)
    handle = LOCK.open("a+b")
    try:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def graph():
    result = []
    for route in NONLEARNING:
        result.append({"key": f"dev_{route}", "route": route, "port": DEV_PORTS[route],
                       "kind": "evaluate", "role": "development", "purpose": "verification",
                       "expected_count": 60})
    for route in LEARNING:
        result.append({"key": f"verify_{route}", "route": route, "port": DEV_PORTS[route],
                       "kind": "train", "role": "development", "purpose": "verification",
                       "budget": 2048, "expected_count": 60})
    for route in LEARNING:
        result.append({"key": f"train_{route}", "route": route, "port": TEST_PORTS[route],
                       "kind": "train", "role": "validation", "purpose": "formal",
                       "budget": 262144})
    for route in NONLEARNING:
        result.append({"key": f"test_{route}", "route": route, "port": TEST_PORTS[route],
                       "kind": "evaluate", "role": "test", "purpose": "formal",
                       "expected_count": 300, "cold_init": route == "R12"})
    for route in ("R04", "R12"):
        result.append({"key": f"sequence_{route}", "route": route, "port": 20474,
                       "kind": "evaluate", "role": "sequence", "purpose": "formal",
                       "expected_count": 360})
    for route in LEARNING:
        result.append({"key": f"test_{route}", "route": route, "port": TEST_PORTS[route],
                       "kind": "evaluate", "role": "test", "purpose": "formal",
                       "expected_count": 300, "requires_model": f"train_{route}"})
    return result


TASKS = {row["key"]: row for row in graph()}
DEVELOPMENT = [key for key in TASKS if key.startswith(("dev_", "verify_"))]
MAIN = [key for key in TASKS if key.startswith(("train_", "test_", "sequence_"))
        and key not in ("test_R08", "test_R09")]
LEARNED_FINAL = ("test_R08", "test_R09")
DEPENDENCIES = {
    **{f"test_{route}": (f"dev_{route}",) for route in NONLEARNING},
    "train_R08": ("verify_R08", "dev_R04"),
    "train_R09": ("verify_R09",),
    "sequence_R04": ("dev_R04",),
    "sequence_R12": ("dev_R12",),
    "test_R08": ("train_R08",),
    "test_R09": ("train_R09",),
}
THROUGHPUT_KEYS = {"train_R08", "test_R08", "train_R09", "test_R09"}


def validate_throughput_overrides(state):
    overrides = state.get("throughput_source_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - THROUGHPUT_KEYS:
        raise RuntimeError("throughput source override names a nonlearning or unknown task")
    for route in LEARNING:
        train, test = overrides.get(f"train_{route}"), overrides.get(f"test_{route}")
        if test is not None and (train is None or
                                 test.get("repair_commit") != train.get("repair_commit")):
            raise RuntimeError("learned test has no matching repaired training source")
    for key, item in overrides.items():
        if (not isinstance(item, dict) or
                item.get("origin_commit") != state["scientific_commit"] or
                not isinstance(item.get("repair_commit"), str) or
                len(item["repair_commit"]) != 40 or
                item["repair_commit"] == state["scientific_commit"]):
            raise RuntimeError(f"{key}: throughput source identity differs")
        if key.startswith("train_"):
            if (not isinstance(item.get("validation_receipt_sha256"), str) or
                    len(item["validation_receipt_sha256"]) != 64 or
                    not isinstance(item.get("origin_checkpoint_sha256"), str) or
                    len(item["origin_checkpoint_sha256"]) != 64):
                raise RuntimeError(f"{key}: checkpoint-bound validation evidence missing")
        elif (not isinstance(item.get("dataset_contract_sha256"), str) or
              len(item["dataset_contract_sha256"]) != 64):
            raise RuntimeError(f"{key}: learned-test dataset contract SHA missing")


def expected_task_source(state, key):
    return (state.get("throughput_source_overrides", {}).get(key) or {}).get(
        "repair_commit", state["scientific_commit"])


def api(port, path, body=None):
    reply = ops.api(port, path, body)
    if reply["status"] != 200:
        raise RuntimeError(f"{port} {path} HTTP {reply['status']}: {reply.get('error')}")
    return reply["body"]


def _body(task, registry, state):
    port, route, role = task["port"], task["route"], task["role"]
    config = api(port, "/api/coverage/catalog")["variable_defaults"]
    config["coverage"]["variable"]["route"] = route
    config["coverage"]["method"] = "ppo" if route in LEARNING else "local_search"
    config["name"] = f"S1-V10 {task['key']}"
    if task.get("budget") == 2048:
        config["training"]["total_steps"] = 2048
        config["training"]["validation_interval"] = 2048
    lane = registry["lanes"][str(port)]
    if task["kind"] == "train":
        selected = {"train": lane["train"]["id"], role: lane[role]["id"]}
    else:
        selected = {role: lane[role]["id"]}
    request_id = f"s1v10_{task['key']}_{port}_s51001"
    body = {"kind": task["kind"], "config": config,
            "dataset_ids": selected, "dataset_role": role,
            "experiment_role": task["purpose"], "request_id": request_id}
    if task.get("requires_model"):
        model_key = task["requires_model"]
        body["model_id"] = state["tasks"][model_key]["run_id"]
    return body


def _submit(task, registry, state):
    key, port = task["key"], task["port"]
    if key in state["tasks"]:
        return
    body = _body(task, registry, state)
    job = api(port, "/api/coverage/runs", body)
    expected_source = expected_task_source(state, key)
    if (job.get("git_commit") != expected_source or
            job.get("request_id") != body["request_id"] or
            job.get("execution_target") != ops.instance(port)["target"]):
        raise RuntimeError(f"{key}: frozen workbench task identity differs")
    if key in ("test_R08", "test_R09") and expected_source != state["scientific_commit"]:
        override = state["throughput_source_overrides"][key]
        if job.get("variable_dataset_contract", {}).get("sha256") != override.get("dataset_contract_sha256"):
            raise RuntimeError(f"{key}: throughput evaluation used another scene contract")
    state["tasks"][key] = {**task, "run_id": job["run_id"],
                           "request_id": body["request_id"],
                           "dataset_sha256": {role: registry["lanes"][str(port)][role]["sha256"]
                                              for role in body["dataset_ids"]},
                           "submitted_utc": now()}
    save(STATE, state)


def _audit_rows(port, detail, expected, task):
    directory = detail["path"]
    expected_sha = detail["artifacts"].get("raw_results_sha256")
    code = f'''
import hashlib,json,pathlib
root=pathlib.Path({directory!r});file=root/'results.partial.jsonl';expected={expected}
if not file.is_file():raise RuntimeError('raw evaluation rows missing')
raw=file.read_bytes();rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
bad=[]
for index,row in enumerate(rows):
 metric=row.get('metrics',{{}})
 if (not row.get('demand_conserved') or not metric.get('demand_conserved') or
     metric.get('beam_count',201)>200 or metric.get('aircraft_count')!=len(row.get('users',[]))):
  bad.append(index)
print(json.dumps({{'count':len(rows),'expected':expected,
 'sha256':hashlib.sha256(raw).hexdigest(),'bad_rows':bad[:30],
 'coverage_100_count':sum(bool(r.get('coverage_complete')) for r in rows),
 'temporal_mode_counts':{{mode:sum(r.get('temporal_mode')==mode for r in rows)
                          for mode in set(r.get('temporal_mode') for r in rows)}},
 'demand_multipliers':sorted(set(r.get('demand_multiplier') for r in rows))}}))
'''
    response = ops.remote_python(code, port, timeout=180)
    if response["exit_code"]:
        raise RuntimeError(f"{port} raw result audit failed: {response['stderr'][-700:]}")
    row = json.loads(response["stdout"].splitlines()[-1])
    if (row["count"] != expected or row["sha256"] != expected_sha or row["bad_rows"] or
            row["demand_multipliers"] != [.75, 1.0, 1.25]):
        raise RuntimeError("raw result count/SHA/demand conservation/budget differs")
    modes = row["temporal_mode_counts"]
    if (task["key"] == "test_R12" and modes != {"cold_init": 300} or
            task["key"] == "sequence_R12" and modes != {
                "sequence_first_frame": 60, "sequence_warm_start": 300} or
            task["key"] == "sequence_R04" and modes != {"independent_scene": 360}):
        raise RuntimeError("R12/R04 cold/sequence timing modes differ from frozen comparison")
    return row


def _audit_train(port, detail, expected_step, development=False, expected_source=None):
    directory = detail["path"]
    code = f'''
import hashlib,json,pathlib
root=pathlib.Path({directory!r});step={expected_step}
checkpoint=root/'checkpoint.pt';model=pathlib.Path(json.loads((root/'job.json').read_text())['model_dir'])/'final'
meta=json.loads((model/'model.json').read_text())
rows=root/f'validation_{{step}}.jsonl';lines=[json.loads(line) for line in rows.read_bytes().splitlines() if line.strip()] if rows.exists() else []
print(json.dumps({{'checkpoint':checkpoint.is_file(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None,
 'model_step':meta.get('env_step'),'model_sha256':hashlib.sha256((model/'policy.pt').read_bytes()).hexdigest(),
 'metadata_policy_sha256':meta.get('policy_sha256'),'model_source_commit':meta.get('source_commit'),
 'model_source_history':meta.get('source_history'),'development_rows':len(lines),
 'bad_development_rows':sum(not row.get('demand_conserved') for row in lines)}}))
'''
    response = ops.remote_python(code, port, timeout=180)
    if response["exit_code"]:
        raise RuntimeError(f"{port} training artifact audit failed: {response['stderr'][-700:]}")
    row = json.loads(response["stdout"].splitlines()[-1])
    if (not row["checkpoint"] or row["model_step"] != expected_step or
            row["model_sha256"] != row["metadata_policy_sha256"] or
            (expected_source and row["model_source_commit"] != expected_source) or
            (development and (row["development_rows"] != 60 or row["bad_development_rows"]))):
        raise RuntimeError("training endpoint/checkpoint/development audit differs")
    if expected_source and expected_source != detail["job"].get("original_git_commit", expected_source):
        history = row["model_source_history"] or []
        if not any(item.get("repair_commit") == expected_source and
                   item.get("origin_commit") == detail["job"].get("original_git_commit")
                   for item in history):
            raise RuntimeError("repaired endpoint model lacks old/new source history")
    return row


def _task_state(key, state):
    task = state["tasks"].get(key)
    if task is None:
        return "missing"
    if key in state["audits"]:
        return "audited"
    detail = api(task["port"], "/api/runs/" + task["run_id"])
    stage = detail["status"]["stage"]
    task["last_observed_utc"] = now()
    task["last_observed_stage"] = stage
    expected_source = expected_task_source(state, key)
    if detail["job"].get("git_commit") != expected_source:
        raise RuntimeError(f"{key}: scientific source commit changed")
    if key in ("train_R08", "train_R09") and expected_source != state["scientific_commit"]:
        proof = detail["job"].get("variable_throughput_repair") or {}
        override = state["throughput_source_overrides"][key]
        if (proof.get("repair_id") != "s1v10_cpu_supply_equivalent_v1" or
                proof.get("origin_commit") != state["scientific_commit"] or
                proof.get("repair_commit") != expected_source or
                proof.get("run_id") != task["run_id"] or
                proof.get("validation_receipt_sha256") != override.get("validation_receipt_sha256") or
                proof.get("checkpoint_sha256") != override.get("origin_checkpoint_sha256")):
            raise RuntimeError(f"{key}: same-ID throughput source proof differs")
    if key in ("test_R08", "test_R09") and expected_source != state["scientific_commit"]:
        override = state["throughput_source_overrides"][key]
        if detail["job"].get("variable_dataset_contract", {}).get("sha256") != override.get("dataset_contract_sha256"):
            raise RuntimeError(f"{key}: frozen learned-test scene contract differs")
    actual_datasets = {role: row.get("sha256")
                       for role, row in detail["config"].get("datasets", {}).items()}
    if actual_datasets != task["dataset_sha256"]:
        raise RuntimeError(f"{key}: frozen dataset SHA differs from controller registry")
    if stage == "completed":
        if task["kind"] == "train":
            step = task["budget"]
            if detail["status"].get("env_step") != step:
                raise RuntimeError(f"{key}: completed at wrong decision budget")
            audit = _audit_train(task["port"], detail, step,
                                 development=task["purpose"] == "verification",
                                 expected_source=expected_source)
        else:
            audit = _audit_rows(task["port"], detail, task["expected_count"], task)
        state["audits"][key] = {"run_id": task["run_id"],
                                "audited_utc": now(), **audit}
        save(STATE, state)
        return "audited"
    if stage in ("failed", "stopped", "interrupted", "cancelled"):
        state["need_attention"][key] = {"stage": stage,
                                        "run_id": task["run_id"],
                                        "error": detail["status"].get("error"),
                                        "observed_utc": now()}
        return "attention"
    return stage


def write_run_manifest(state):
    source_observation = ops.LOCAL / "preflight" / "live_observation.json"
    source = json.loads(source_observation.read_text(encoding="utf-8")) if source_observation.is_file() else {}
    manifest = {
        "experiment_id": "S1-V10",
        "status": "attention" if state["need_attention"] else state["phase"],
        "phase": state["phase"],
        "scientific_commit": state["scientific_commit"],
        "source_commit": state.get("preparation_source_commit", state["scientific_commit"]),
        "last_observed_utc": state["observed_utc"],
        "data_sha256": state.get("dataset_sha256", {}),
        "dataset_contract_sha256": state.get("dataset_contract_sha256"),
        "throughput_source_overrides": state.get("throughput_source_overrides", {}),
        "source_archive_observation": {port: {
            "date": item.get("date"),
            "status": (item.get("download") or {}).get("status"),
            "sha256": (item.get("download") or {}).get("sha256"),
            "observed_utc": item.get("observed_utc")}
            for port, item in source.get("lanes", {}).items()},
        "budget": {"formal_training_seed": 51001,
                   "formal_training_routes": list(LEARNING),
                   "formal_decisions_per_route": 262144,
                   "formal_total_decisions": 524288,
                   "verification_decisions_per_learning_route": 2048,
                   "verification_total_decisions": 4096,
                   "demand_multipliers": [.75, 1., 1.25],
                   "development_mothers": 20, "test_mothers": 100,
                   "sequence_segments": 20, "sequence_frames_per_segment": 6},
        "tasks": {**{key: {field: task.get(field) for field in
                 ("run_id", "request_id", "port", "role", "route", "kind",
                  "purpose", "budget", "expected_count", "dataset_sha256",
                  "last_observed_utc", "last_observed_stage")}
                  | {"expected_source_commit": expected_task_source(state, key),
                     "throughput_repair": state.get("throughput_source_overrides", {}).get(key)}
                  for key, task in state["tasks"].items()},
                  **{f"prepare_{port}": {"run_id": row["run_id"],
                      "request_id": row["request_id"], "port": int(port),
                      "role": row["role"], "route": "prepare", "kind": "prepare",
                      "last_observed_utc": row.get("last_observed_utc"),
                      "last_observed_stage": row.get("last_observed_stage")}
                     for port, row in state.get("prepares", {}).items()}},
        "audits": {key: {"run_id": item["run_id"],
                          "audited_utc": item["audited_utc"]}
                   for key, item in state["audits"].items()},
        "need_attention": state["need_attention"],
        "last_error": state.get("last_error"),
        "observation": {"controller_state": str(STATE),
                        "dataset_registry": str(data_controller.REGISTRY),
                        "live_preflight": str(source_observation)},
        "followup_id": "s1-v10",
        "report_path": "D:/paper/docs/第一阶段/实验报告/08_可变波束全覆盖与负载均衡十二路线对照.md",
    }
    save(RUN_MANIFEST, manifest)


def _phase_tasks(state, registry, keys):
    for key in keys:
        _submit(TASKS[key], registry, state)
    counts = {key: _task_state(key, state) for key in keys}
    state["progress"] = counts
    return all(value == "audited" for value in counts.values())


def _execution_tick(state, registry):
    for key in DEVELOPMENT:
        _submit(TASKS[key], registry, state)
    progress = {key: _task_state(key, state) for key in DEVELOPMENT}
    for key in (*MAIN, *LEARNED_FINAL):
        if key not in state["tasks"] and all(dep in state["audits"] for dep in DEPENDENCIES[key]):
            _submit(TASKS[key], registry, state)
        if key in state["tasks"]:
            progress[key] = _task_state(key, state)
        else:
            progress[key] = "waiting_dependencies"
    state["progress"] = progress
    return all(key in state["audits"] for key in TASKS)


def tick(commit):
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {
        "scientific_commit": commit, "phase": "source_preparation",
        "tasks": {}, "audits": {}, "need_attention": {}, "created_utc": now()}
    if state["scientific_commit"] != commit:
        old = data_controller.PREPARATION_SOURCE_COMMIT
        if (state["scientific_commit"] != old or state["phase"] not in
                ("source_preparation", "dataset_registration") or
                state["tasks"] or state["audits"]):
            raise RuntimeError("controller state belongs to a different frozen commit")
        data_controller.existing_prepares(commit)
        history = ROOT / "migrations" / f"prepare_{old[:12]}_to_{commit[:12]}.json"
        save(history, state)
        state["preparation_source_commit"] = old
        state["scientific_commit"] = commit
        state["phase"] = "dataset_registration"
        state["migration_receipt"] = str(history)
    state.pop("last_error", None)
    validate_throughput_overrides(state)
    try:
        if state["phase"] == "source_preparation":
            state["prepares"] = data_controller.submit_prepares(commit)
            state["preparation_source_commit"] = data_controller.PREPARATION_SOURCE_COMMIT
            state["phase"] = "dataset_registration"
        if state["phase"] == "dataset_registration":
            for port, row in state.get("prepares", {}).items():
                detail = api(int(port), "/api/runs/" + row["run_id"])
                row["last_observed_utc"] = now()
                row["last_observed_stage"] = detail["status"]["stage"]
                if detail["status"]["stage"] in ("failed", "stopped", "interrupted", "cancelled"):
                    state["need_attention"][f"prepare_{port}"] = {
                        "stage": detail["status"]["stage"], "run_id": row["run_id"],
                        "error": detail["status"].get("error"),
                        "observed_utc": row["last_observed_utc"]}
            registry = data_controller.register_datasets(commit)
            state["dataset_registry"] = str(data_controller.REGISTRY)
            state["dataset_sha256"] = registry["canonical_sha256"]
            state["dataset_contract_sha256"] = hashlib.sha256(
                data_controller.CONTRACT.read_bytes()).hexdigest()
            state["phase"] = "execution"
        if state["phase"] == "execution":
            registry = json.loads(data_controller.REGISTRY.read_text(encoding="utf-8"))
            if registry["scientific_commit"] != commit or registry["canonical_sha256"] != state["dataset_sha256"]:
                raise RuntimeError("dataset registry changed after first scientific dispatch")
            if hashlib.sha256(data_controller.CONTRACT.read_bytes()).hexdigest() != state["dataset_contract_sha256"]:
                raise RuntimeError("canonical dataset contract changed after first scientific dispatch")
            if _execution_tick(state, registry):
                state["phase"] = "completed_pending_report_archive"
    except Exception as error:
        state["last_error"] = {"message": str(error), "observed_utc": now(),
                               "phase": state["phase"]}
    state["observed_utc"] = now()
    save(STATE, state)
    write_run_manifest(state)
    return state


def resume_run(commit, key):
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state["scientific_commit"] != commit:
        raise RuntimeError("run does not belong to frozen S1-V10 controller state")
    if key.startswith("prepare_") and key.removeprefix("prepare_") in state.get("prepares", {}):
        task = state["prepares"][key.removeprefix("prepare_")]
        task = {**task, "port": int(task["port"])}
        expected_commit = state.get("preparation_source_commit", data_controller.PREPARATION_SOURCE_COMMIT)
    else:
        task = state["tasks"].get(key)
        expected_commit = expected_task_source(state, key)
    if task is None:
        raise RuntimeError("unknown frozen S1-V10 task key")
    detail = api(task["port"], "/api/runs/" + task["run_id"])
    if (detail["job"].get("git_commit") != expected_commit or
            detail["status"].get("stage") not in ("stopped", "failed", "interrupted")):
        raise RuntimeError("same-ID workbench resume requires audited stopped/failed/interrupted state")
    result = api(task["port"], f"/api/runs/{task['run_id']}/resume", {})
    state["need_attention"].pop(key, None)
    state["resume_receipts"] = state.get("resume_receipts", []) + [{
        "key": key, "run_id": task["run_id"], "observed_utc": now(), "api_result": result}]
    save(STATE, state)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "tick", "run", "resume"))
    parser.add_argument("--commit")
    parser.add_argument("--key")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps(graph(), ensure_ascii=False, indent=2))
        return
    if not args.commit or len(args.commit) != 40:
        parser.error("exact frozen --commit required")
    with exclusive():
        if args.action == "resume":
            if not args.key:
                parser.error("--key required for same-ID resume")
            result = resume_run(args.commit, args.key)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.action == "tick":
            result = tick(args.commit)
            print(json.dumps({"phase": result["phase"], "progress": result.get("progress"),
                              "need_attention": result["need_attention"],
                              "last_error": result.get("last_error")}, ensure_ascii=False, indent=2))
        else:
            if not 15 <= args.interval <= 600:
                parser.error("controller interval must be 15–600 seconds")
            while True:
                result = tick(args.commit)
                print(json.dumps({"observed_utc": result["observed_utc"],
                                  "phase": result["phase"],
                                  "audited": len(result["audits"]),
                                  "tasks": len(result["tasks"]),
                                  "need_attention": result["need_attention"],
                                  "last_error": result.get("last_error")}, ensure_ascii=False), flush=True)
                if result["phase"] == "completed_pending_report_archive":
                    break
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
