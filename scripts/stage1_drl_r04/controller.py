"""Durable, idempotent S1-D12A workbench DAG coordinator.

Run ``once`` for a bounded transition or ``loop`` as a background process.
It submits only approved workbench jobs and never invokes a scientific runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from scripts.stage1_drl_r04 import ops
from scripts.stage1_drl_r04.data_controller import REGISTRY, assert_service, save


STATE = ops.LOCAL / "controller" / "state.json"
METHODS = ("I0", "H1", "H2", "BC", "P0", "P1")
METHOD_CONFIG = {"I0": "none", "H1": "local_search", "H2": "local_search",
                 "BC": "ppo", "P0": "ppo", "P1": "ppo"}
TRAIN_PORT = {"P0": 20418, "P1": 20424, "BC": 20424}


def _snapshot(port, run_id):
    detail = ops.api(port, f"/api/runs/{run_id}")
    if (detail.get("job", {}).get("run_id") != run_id or
            detail.get("job", {}).get("experiment_id") != "S1-D12A"):
        raise ValueError(f"{port} workbench task identity differs: {run_id}")
    return detail


def _submit(state, key, port, kind, variant, datasets, *, role=None, purpose="formal",
            source_key=None, shard=None, verification=False):
    if key in state["tasks"]:
        saved = state["tasks"][key]
        if saved["port"] != port or saved["variant"] != variant or saved["kind"] != kind:
            raise ValueError("stored task ownership or variant differs")
        _snapshot(port, saved["run_id"])
        return saved
    catalog = assert_service(port, state["scientific_commit"])
    config = catalog["drl_r04_defaults"]
    config["name"] = f"S1-D12A {key}"
    config["coverage"]["method"] = METHOD_CONFIG.get(variant, "ppo")
    if verification:
        config["training"]["total_steps"] = 2048
        config["training"]["validation_interval"] = 2048
    lane = state["datasets"]["lanes"][str(port)]
    selected = {dataset: lane[dataset]["id"] for dataset in datasets}
    body = {"kind": kind, "variant": variant, "config": config,
            "dataset_ids": selected, "experiment_role": purpose,
            "request_id": f"s1d12a_{key}_s51001"}
    if role:
        body["dataset_role"] = role
    if source_key:
        source = state["tasks"][source_key]
        body.update(source_port=source["port"], source_run_id=source["run_id"])
    if kind == "train" and variant in ("P0", "P1"):
        proof = state.get("validation_reachability")
        if not proof:
            raise ValueError("PPO training requires baseline-frozen validation subset")
        body.update(reachability_contract_path=proof["remote_path"],
                    reachability_contract_sha256=proof["sha256"])
    if shard is not None:
        body.update(shard_id=shard, scene_indices=list(range(shard, 100, 4)))
    job = ops.api(port, "/api/coverage/runs", body)
    if (job.get("git_commit") != state["scientific_commit"] or
            job.get("request_id") != body["request_id"] or
            job.get("variant") != variant or job.get("experiment_id") != "S1-D12A"):
        raise ValueError(f"{key} workbench frozen task differs")
    receipt = {"run_id": job["run_id"], "request_id": body["request_id"],
               "kind": kind, "variant": variant, "port": port, "purpose": purpose,
               "dataset_ids": selected, "source_task": source_key,
               "shard_id": shard, "submitted_utc": datetime.now(timezone.utc).isoformat()}
    state["tasks"][key] = receipt
    save(STATE, state)
    return receipt


def _stage(state, key):
    if key not in state["tasks"]:
        return None
    receipt = state["tasks"][key]
    detail = _snapshot(receipt["port"], receipt["run_id"])
    return detail["status"].get("stage")


def _reachability(state, key):
    task = state["tasks"][key]
    directory = f"{ops.instance(task['port'])['archive']}/outputs/workbench/jobs/{task['run_id']}"
    code = f'''
import hashlib,json,pathlib
p=pathlib.Path({directory!r})/'results.jsonl'
rows=[json.loads(line) for line in p.read_text().splitlines()]
if len(rows)!=60:raise ValueError('development rows must be20 mothers times3 levels')
if [r['dataset_index'] for r in rows]!=[i//3 for i in range(60)]:
 raise ValueError('development mother order differs')
if [r['demand_multiplier'] for r in rows]!=[[.75,1.,1.25][i%3] for i in range(60)]:
 raise ValueError('development level order differs')
improved=[]
for i in range(20):
 group=rows[3*i:3*i+3]
 if all(r.get('initial_j') is not None and r.get('best_j') is not None for r in group):
  average=sum(r['initial_j']-r['best_j'] for r in group)/3
  if average>1e-8: improved.append(i)
print(json.dumps({{'rows':len(rows),'improved_mothers':improved,
 'improved_count':len(improved),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}))
'''
    result = ops.remote_python(code, task["port"])
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1000:])
    return json.loads(result["stdout"].splitlines()[-1])


def _freeze_validation_reachability(state):
    baseline = {variant: _reachability(state, "val_" + variant)
                for variant in ("H1", "H2")}
    subset = sorted(set(baseline["H1"]["improved_mothers"]) |
                    set(baseline["H2"]["improved_mothers"]))
    contract = {"format": "s1d12a_reachability_v1", "experiment_id": "S1-D12A",
                "scientific_commit": state["scientific_commit"],
                "validation_dataset_sha256": state["datasets"]["lanes"]["20418"]["validation"]["sha256"],
                "reachable_mother_indices": subset,
                "criterion": "three demand levels' mean initial-best J > 1e-8",
                "baselines": {variant: {"port": state["tasks"]["val_" + variant]["port"],
                                        "run_id": state["tasks"]["val_" + variant]["run_id"],
                                        "sha256": baseline[variant]["sha256"],
                                        "improved_mother_indices": baseline[variant]["improved_mothers"]}
                              for variant in ("H1", "H2")},
                "frozen_before_ppo_utc": datetime.now(timezone.utc).isoformat()}
    local = ops.LOCAL / "controller" / "validation_reachability.json"
    save(local, contract)
    sha = hashlib.sha256(local.read_bytes()).hexdigest()
    remote = f"{ops.BASE}/source/validation_reachability.json"
    transfer = ops.upload(local, remote, 20473)
    if transfer["local_sha256"] != sha or transfer["remote"] != sha:
        raise ValueError("frozen validation reachable subset upload SHA differs")
    return {"path": str(local), "remote_path": remote, "sha256": sha,
            "reachable_mother_indices": subset, "baselines": baseline}


def _training_terminal(state, key):
    if key not in state["tasks"]:
        return False
    item = state["tasks"][key]
    status = _snapshot(item["port"], item["run_id"])["status"]
    return status.get("stage") == "completed" or (
        status.get("stage") == "stopped" and status.get("scientific_stop") is True)


def _paired_pilot(state):
    if not all(key in state["tasks"] for key in ("P0", "P1")):
        return
    statuses = {key: _snapshot(state["tasks"][key]["port"], state["tasks"][key]["run_id"])["status"]
                for key in ("P0", "P1")}
    reviewable = lambda row: row.get("stage") in ("pilot_pending", "completed") or (
        row.get("stage") == "stopped" and row.get("scientific_stop") is True)
    if not all(reviewable(row) for row in statuses.values()):
        return
    for key in ("P0", "P1"):
        if statuses[key]["stage"] == "pilot_pending" and key not in state.setdefault("pilot_audits", {}):
            item = state["tasks"][key]
            outcome = ops.api(item["port"],
                              f"/api/coverage/runs/{item['run_id']}/pilot-gate", {"mode": "audit"})
            state["pilot_audits"][key] = outcome["audit"]
            save(STATE, state)
    # Audit may itself scientifically stop a full two-node STOP route.
    statuses = {key: _snapshot(state["tasks"][key]["port"], state["tasks"][key]["run_id"])["status"]
                for key in ("P0", "P1")}
    for key, peer_key in (("P0", "P1"), ("P1", "P0")):
        if statuses[key]["stage"] != "pilot_pending":
            continue
        own = state["pilot_audits"][key]
        peer = state.get("pilot_audits", {}).get(peer_key, {})
        item, paired = state["tasks"][key], state["tasks"][peer_key]
        mode = "release" if own.get("progress_pass") or peer.get("progress_pass") else "stop"
        outcome = ops.api(item["port"],
                          f"/api/coverage/runs/{item['run_id']}/pilot-gate", {
                              "mode": mode, "paired_port": paired["port"],
                              "paired_run_id": paired["run_id"]})
        state.setdefault("pilot_decisions", {})[key] = {
            "mode": mode, "audit": outcome["audit"],
            "observed_utc": datetime.now(timezone.utc).isoformat()}
        save(STATE, state)
        if mode == "release":
            ops.api(item["port"], f"/api/runs/{item['run_id']}/resume", {})


def _failed(state):
    for key, item in state["tasks"].items():
        status = _stage(state, key)
        if status in ("failed", "interrupted") or (status == "stopped" and
                not (key in ("P0", "P1") and
                     _snapshot(item["port"], item["run_id"])["status"].get("scientific_stop") is True)):
            return {"key": key, "port": item["port"], "run_id": item["run_id"],
                    "stage": status}
    return None


def tick(state):
    """Advance only dependencies whose real workbench records are complete."""
    problem = _failed(state)
    if problem:
        state.update(phase="attention_required", problem=problem)
        return state
    if not state.get("datasets"):
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        if (registry.get("scientific_commit") != state["scientific_commit"] or
                set(registry.get("lanes", {})) != set(map(str, ops.SSH_PORTS)) or
                any(set(registry["lanes"][str(port)]) < {"train", "development", "validation", "test"}
                    for port in ops.SSH_PORTS)):
            raise ValueError("four-lane data registry is not complete")
        test_sha = {registry["lanes"][str(port)]["test"]["sha256"] for port in ops.SSH_PORTS}
        if len(test_sha) != 1:
            raise ValueError("four lanes have different independent test mothers")
        state["datasets"] = registry
        state["test_sha256"] = next(iter(test_sha))
        save(STATE, state)
    state["phase"] = "development_gate"
    for variant, port in (("H1", 20473), ("H2", 20474)):
        _submit(state, "dev_" + variant, port, "evaluate", variant,
                ("development",), role="development", purpose="verification")
    if any(_stage(state, "dev_" + variant) != "completed" for variant in ("H1", "H2")):
        return state
    if "reachability" not in state:
        reach = {variant: _reachability(state, "dev_" + variant)
                 for variant in ("H1", "H2")}
        passed = max(row["improved_count"] for row in reach.values()) >= 10
        state["reachability"] = {"passed": passed, "routes": reach,
                                  "criterion": "at least 10 of 20 mothers improve J in H1 or H2"}
        save(STATE, state)
    if not state["reachability"]["passed"]:
        state["phase"] = "scientific_stop_unreachable"
        return state

    for variant, port in (("H1", 20473), ("H2", 20474)):
        _submit(state, "val_" + variant, port, "evaluate", variant,
                ("validation",), role="validation", purpose="verification")
    if any(_stage(state, "val_" + variant) != "completed" for variant in ("H1", "H2")):
        state["phase"] = "validation_baseline_gate"
        return state
    if "validation_reachability" not in state:
        state["validation_reachability"] = _freeze_validation_reachability(state)
        save(STATE, state)
    if not state["validation_reachability"]["reachable_mother_indices"]:
        state["phase"] = "scientific_stop_unreachable_validation"
        return state

    # The teacher and P0 verification can run on different instances together.
    state["phase"] = "learning"
    _submit(state, "teacher", 20473, "prepare", "teacher", ("train",))
    _submit(state, "verify_P0", 20418, "train", "P0", ("train", "validation"),
            purpose="verification", verification=True)
    if _stage(state, "teacher") == "completed":
        _submit(state, "BC", 20424, "train", "BC", ("train",), source_key="teacher")
    if _stage(state, "verify_P0") == "completed":
        _submit(state, "P0", 20418, "train", "P0", ("train", "validation"))
    if "BC" in state["tasks"] and _stage(state, "BC") == "completed":
        _submit(state, "verify_P1", 20424, "train", "P1",
                ("train", "validation"), purpose="verification",
                source_key="BC", verification=True)
    if "verify_P1" in state["tasks"] and _stage(state, "verify_P1") == "completed":
        _submit(state, "P1", 20424, "train", "P1",
                ("train", "validation"), source_key="BC")
    _paired_pilot(state)
    if (_stage(state, "teacher") != "completed" or _stage(state, "BC") != "completed" or
            not all(_training_terminal(state, key) for key in ("P0", "P1"))):
        return state

    # Final timing uses identical mother shards on identical devices for all
    # six methods, with one evaluation worker per lane and no training/teacher.
    state["phase"] = "independent_test"
    for shard, port in enumerate(ops.SSH_PORTS):
        for variant in METHODS:
            key = f"test_{variant}_{port}"
            prior = METHODS[METHODS.index(variant) - 1] if variant != METHODS[0] else None
            if prior and _stage(state, f"test_{prior}_{port}") != "completed":
                break
            source_key = variant if variant in ("BC", "P0", "P1") else None
            _submit(state, key, port, "evaluate", variant, ("test",),
                    role="test", source_key=source_key, shard=shard)
            if _stage(state, key) != "completed":
                break
    if all(_stage(state, f"test_{variant}_{port}") == "completed"
           for port in ops.SSH_PORTS for variant in METHODS):
        state["phase"] = "test_complete_report_archive_pending"
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("once", "loop"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if len(args.commit) != 40 or not 15 <= args.poll_seconds <= 600:
        parser.error("frozen commit and bounded 15–600 second poll interval required")
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {
        "experiment_id": "S1-D12A", "scientific_commit": args.commit,
        "phase": "awaiting_data_registry", "tasks": {}}
    if state.get("scientific_commit") != args.commit:
        raise ValueError("controller state belongs to another frozen commit")
    while True:
        state = tick(state)
        state["observed_utc"] = datetime.now(timezone.utc).isoformat()
        save(STATE, state)
        print(json.dumps({"phase": state["phase"], "tasks": len(state["tasks"]),
                          "observed_utc": state["observed_utc"]}, ensure_ascii=False), flush=True)
        if (args.action == "once" or state["phase"] in (
                "attention_required", "scientific_stop_unreachable",
                "scientific_stop_unreachable_validation",
                "test_complete_report_archive_pending")):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
