"""Build the S2-P19 typed API plan from frozen contracts and service defaults.

Bootstrap contains data preparation, both train-scene imports, teacher,
non-learning gate, three authorized SAC pilots and their source-backed audit
gates. It uses task-result references so no run ID or output record path is
filled by hand. Later phases are appended with a parent SHA chain.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from . import controller, ops


ROOT = Path("C:/Users/Admin/.codex/worktrees/s2-p19-sac/paper")
CONTRACTS = ops.LOCAL / "contracts"
TRAIN_RECORD = "/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424/data/prepared/train"
SCIENCE_COMMIT = "47da239910485508f7c978461314dff92f8672ec"
R1_MOTHER_COMMIT = "62202aff0e1190492338f98359cc2f274884341c"


def ref(task: str, *keys: str) -> dict:
    return {"$task": {"id": task, "path": list(keys)}}


def record_directory(task: str, partition: str, source_port: int) -> dict:
    """Resolve a preparation result's record.json to its scene directory.

    The controller accepts this typed reference only under the matching
    S2-P19 workbench job's prepared/<partition>/record.json path.
    """
    return {"$record_directory": {"id": task, "source_port": source_port,
                                  "partition": partition,
                                  "path": ["result", "summary", "outputs", partition, "path"]}}


def task(identifier: str, phase: str, port: int, kind: str, body: dict,
         disk_class: str, dependencies: list[str], **extra) -> dict:
    return {"id": identifier, "phase": phase, "port": port, "kind": kind,
            "body": body, "disk_class": disk_class, "requires": dependencies,
            "request_id": body.get("request_id") or "s2p19-" + identifier.replace("_", "-"), **extra}


def _source_commit() -> str:
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=True)
    commit = result.stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           text=True, capture_output=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError("plan builder requires clean frozen S2-P19 worktree")
    return commit


def bootstrap() -> dict:
    commit = _source_commit()
    launch_path = ops.LOCAL / "preflight" / f"workbench_{commit[:12]}.json"
    if not launch_path.is_file():
        raise RuntimeError("deploy+serve exact frozen commit before building controller plan")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch.get("scientific_commit") != commit or set(launch.get("instances", {})) != {"20418", "20424"}:
        raise RuntimeError("two-instance workbench launch does not match frozen commit")
    catalog = ops.api(20418, "/api/s2p19/catalog")
    if catalog.get("experiment_id") != "S2-P19":
        raise RuntimeError("deployed typed catalog differs")
    defaults = catalog["defaults"]
    c0 = deepcopy(defaults["C0"])
    c1 = deepcopy(defaults["C1"])
    i0 = deepcopy(defaults["I0"])
    if (c0["training"]["total_steps"] != 400000 or
            c1["training"]["total_steps"] != 400000 or
            i0["total_steps"] != 400000 or
            c0["training"]["pilot_stop_steps"] != 100000 or
            c1["training"]["pilot_stop_steps"] != 100000 or
            i0["pilot_stop_steps"] != 100000):
        raise RuntimeError("deployed defaults differ from approved pilot/total decisions")
    c1["training"]["demonstrations"].update(
        manifest_path=ref("teacher_20424", "result", "summary", "manifest_path"),
        manifest_sha256=ref("teacher_20424", "result", "summary", "manifest_sha256"),
        source_set_sha256=ref("teacher_20424", "result", "summary", "source_set_sha256"))

    dev_body = json.loads((CONTRACTS / "api_prepare_dev_20418.json").read_text(encoding="utf-8"))["body"]
    mother_body = json.loads((CONTRACTS / "api_prepare_mothers_20424.json").read_text(encoding="utf-8"))["body"]
    v2_pin = json.loads((ops.LOCAL / "preflight/p18_v2_model.json").read_text(encoding="utf-8"))
    if (v2_pin.get("provenance", {}).get("run_id") != "train_sac_20261006_084454_365eff71" or
            v2_pin.get("provenance", {}).get("env_step") != 400000 or
            v2_pin.get("files", {}).get("policy.pt", {}).get("sha256")
            != "78ed38b164141af3968c4498dc4a462b8baf060764f133076ea7c02c7d868cea"):
        raise ValueError("historical V2 final model pin differs")
    tasks = [
        task("prepare_dev_20418", "preparation", 20418, "prepare", dev_body,
             "dev_validation", []),
        task("prepare_mothers_20424", "preparation", 20424, "prepare", mother_body,
             "mother_sources", []),
        task("import_V2_model_20418", "preparation", 20418, "import_model",
             {"path": v2_pin["source_dir"], "name": "S2-P18 V2 frozen 400k historical anchor"},
             "dev_validation", [], source_path=v2_pin["source_dir"],
             expected_sha256=v2_pin["files"]["policy.pt"]["sha256"]),
    ]
    for port in ops.SSH_PORTS:
        tasks.append(task(f"import_train_{port}", "preparation", port, "import_scene",
            {"path": TRAIN_RECORD, "name": f"S2-P19 frozen 8064 train {port}", "profile": "europe_v1"},
            "dev_validation", [], source_record_id="variable_beams_train_20261004", expected_count=8064))
        for partition in ("development", "validation"):
            tasks.append(task(f"import_{partition}_{port}", "preparation", port, "import_scene",
                {"path": record_directory("prepare_dev_20418", partition, 20418),
                 "name": f"S2-P19 {partition} {port}", "profile": "europe_v1"},
                "dev_validation", ["prepare_dev_20418"],
                source_record_id=f"s2p19_{partition}_20261007", expected_count=72))
    tasks.append(task("prepare_final_20424", "preparation", 20424, "prepare",
        {"preparation": {"preparation_type": "final",
                         "source_record": {
                             "path": ref("prepare_mothers_20424", "result", "summary", "path"),
                             "sha256": ref("prepare_mothers_20424", "result", "summary", "sha256")},
                         "seed": 51001, "environment_profile": "europe_v1"}},
        "final", ["prepare_mothers_20424"]))
    for port in ops.SSH_PORTS:
        tasks.append(task(f"import_final_{port}", "preparation", port, "import_scene",
            {"path": record_directory("prepare_final_20424", "final", 20424),
             "name": f"S2-P19 final 880 {port}", "profile": "europe_v1"},
            "final", ["prepare_final_20424"],
            source_record_id="s2p19_final_20261007", expected_count=880))
    tasks.append(task("teacher_20424", "preparation", 20424, "prepare",
        {"preparation": {"preparation_type": "demonstrations"}, "teacher_config": c0,
         "scene_id": ref("import_train_20424", "scene_id")},
        "demonstrations", ["import_train_20424"]))
    tasks.append(task("i0_development_gate_20418", "engineering_gate", 20418, "improvement",
        {"mode": "gate", "partition": "development", "i0_config": i0,
         "scene_id": ref("import_development_20418", "scene_id")},
        "gate_B_216", ["import_development_20418"]))
    tasks.append({"id": "i0_scientific_gate", "phase": "engineering_gate", "port": 20418,
                  "kind": "milestone", "requires": ["i0_development_gate_20418"],
                  "receipt_file": str(ops.LOCAL / "controller/gates/i0_nonlearning_receipt.json")})
    tasks.append(task("C0_pilot_20418", "pilot", 20418, "constructive_train",
        {"config": c0, "scene_id": ref("import_train_20418", "scene_id"),
         "validation_scene_id": ref("import_validation_20418", "scene_id")},
        "C0_pilot", ["import_train_20418", "import_validation_20418"]))
    tasks.append(task("C1_pilot_20424", "pilot", 20424, "constructive_train",
        {"config": c1, "scene_id": ref("import_train_20424", "scene_id"),
         "validation_scene_id": ref("import_validation_20424", "scene_id")},
        "C1_pilot", ["teacher_20424", "import_validation_20424"]))
    tasks.append(task("I0_pilot_20418", "pilot", 20418, "improvement",
        {"mode": "train", "i0_config": i0,
         "scene_id": ref("import_train_20418", "scene_id"),
         "validation_scene_id": ref("import_validation_20418", "scene_id")},
        "I0_pilot", ["i0_scientific_gate", "import_validation_20418"],
        request_id="s2p19-i0-train-51001"))
    for arm, port in (("C0", 20418), ("C1", 20424), ("I0", 20418)):
        pilot = f"{arm}_pilot_{port}"
        tasks.append({"id": f"approve_{arm}_pilot", "phase": "pilot_gate",
                      "port": port, "kind": "approve_pilot", "pilot_task_id": pilot,
                      "requires": [pilot],
                      "gate_receipt_file": str(ops.LOCAL / f"controller/gates/{arm}_pilot_receipt.json")})
    plan = {"schema": 1, "experiment_id": "S2-P19", "scientific_commit": commit,
            "revision": 1, "terminal": False, "tasks": tasks}
    controller.load_plan  # keep shared schema import explicit
    if controller.PLAN.exists():
        prior = json.loads(controller.PLAN.read_text(encoding="utf-8"))
        if prior != plan:
            raise FileExistsError("existing controller plan differs; use append-only phase advance")
    else:
        controller.write_json(controller.PLAN, plan)
    controller.load_plan()
    return {"plan": str(controller.PLAN), "scientific_commit": commit,
            "task_count": len(tasks), "plan_sha256": controller.digest(controller.PLAN)}


def _final_model_id(port: int, train_run_id: str) -> str:
    models = ops.api(port, "/api/models").get("models") or []
    matches = [item for item in models
               if (item.get("provenance") or {}).get("run_id") == train_run_id
               and (item.get("provenance") or {}).get("env_step") == 400000
               and str(item.get("path", "")).rstrip("/").endswith("/" + train_run_id + "/final")]
    if len(matches) != 1:
        raise RuntimeError(f"{port} exact 400k final model for {train_run_id} absent/ambiguous")
    return matches[0]["id"]


def adopt(*, publish: bool = True) -> dict:
    """Adopt the live authorized jobs instead of replaying bootstrap POSTs.

    The main scientific source remains 47da; the sole 62202 exception is the
    separately owned, typed R1 mother-source prepare job. This function only
    reads API/file evidence and writes a local controller plan/state.
    """
    if controller.PLAN.exists() or controller.STATE.exists():
        raise FileExistsError("existing controller plan/state requires explicit review, not re-adoption")
    manifest_path = ops.LOCAL / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("experiment_id") != "S2-P19" or
            set(manifest.get("authorization", {}).get("scope", [])) != {"C0", "C1", "I0"}):
        raise ValueError("run manifest authorization/scope differs")
    controller._service_identity(20418, SCIENCE_COMMIT)
    controller._service_identity(20424, SCIENCE_COMMIT)
    runs = {port: ops.api(port, "/api/s2p19/runs")["runs"] for port in ops.SSH_PORTS}
    scenes = {port: ops.api(port, "/api/scenes")["scenes"] for port in ops.SSH_PORTS}
    request_owners = {}
    for port, rows in runs.items():
        for item in rows:
            request_id = item.get("request_id")
            if not request_id:
                continue
            if request_id in request_owners:
                raise RuntimeError(f"duplicate live S2-P19 request ID: {request_id}")
            request_owners[request_id] = (port, item.get("run_id"))
    tasks, receipts = [], {}

    def adopted_job(identifier: str, phase: str, port: int, kind: str,
                    workbench_kind: str, row: dict, expected_commit: str,
                    requires: list[str]) -> None:
        run_id = row.get("run_id")
        request_id = row.get("request_id")
        config_sha = row.get("config_sha256")
        if not run_id or not request_id or not config_sha:
            raise ValueError(f"{identifier}: missing saved actual run/request/config SHA")
        matches = [item for item in runs[port] if item.get("run_id") == run_id]
        if (len(matches) != 1 or matches[0].get("request_id") != request_id or
                matches[0].get("git_commit") != expected_commit or
                matches[0].get("kind") != workbench_kind or
                controller._adopted_config_sha(port, run_id) != config_sha):
            raise RuntimeError(f"{identifier}: saved job does not match live workbench API/config bytes")
        stage = matches[0].get("stage")
        state = ("completed" if stage == "completed" else
                 "pilot_gate_pending" if stage == "stopped" and
                 matches[0].get("stop_reason") == "s2p19_pilot_gate_pending" else
                 "attention_required" if stage in ("failed", "stopped", "interrupted") else
                 "running")
        if state == "attention_required":
            raise RuntimeError(f"{identifier}: live task needs diagnosis before controller adoption")
        task_row = {"id": identifier, "phase": phase, "port": port, "kind": kind,
                    "requires": requires, "adopted_only": True,
                    "adopted_run_id": run_id, "request_id": request_id,
                    "expected_config_sha256": config_sha,
                    "expected_commit": expected_commit,
                    "workbench_kind": workbench_kind}
        tasks.append(task_row)
        receipt = {"state": state, "run_id": run_id, "request_id": request_id,
                   "workbench_stage": stage, "config_sha256": config_sha,
                   "git_commit": expected_commit, "adopted_only": True,
                   "observed_at": matches[0].get("heartbeat_at")}
        if state == "completed" and kind == "prepare":
            receipt["result"] = ops.api(port, f"/api/runs/{run_id}/results")
        receipts[identifier] = receipt

    def adopted_scene(identifier: str, port: int, scene_id: str,
                      source_record_id: str, count: int, requires: list[str]) -> None:
        matches = [item for item in scenes[port] if item.get("id") == scene_id]
        if (len(matches) != 1 or matches[0].get("count") != count or
                ((matches[0].get("source") or {}).get("reimport") or {}).get("record_id")
                != source_record_id):
            raise RuntimeError(f"{identifier}: imported scene identity/count/source differs")
        tasks.append({"id": identifier, "phase": "preparation", "port": port,
                      "kind": "import_scene", "requires": requires,
                      "adopted_only": True, "scene_id": scene_id,
                      "source_record_id": source_record_id, "expected_count": count})
        receipts[identifier] = {"state": "completed", "scene_id": scene_id,
                                "adopted_only": True, "source_record_id": source_record_id}

    preparation = manifest["operations"]["preparation"]
    old = manifest["tasks"]
    imports = manifest["operations"]["scene_imports"]
    adopted_job("prepare_dev_20418", "preparation", 20418, "prepare", "prepare",
                preparation["dev_validation"], SCIENCE_COMMIT, [])
    adopted_job("prepare_mothers_R1_20424", "preparation", 20424, "prepare", "prepare",
                old["mother_sources_R1"], R1_MOTHER_COMMIT, [])
    for port in ops.SSH_PORTS:
        adopted_scene(f"import_train_{port}", port, imports[f"train_{port}"]["scene_id"],
                      "variable_beams_train_20261004", 8064, [])
        for partition in ("development", "validation"):
            adopted_scene(f"import_{partition}_{port}", port,
                          imports[f"{partition}_{port}"]["scene_id"],
                          f"s2p19_{partition}_20261007", 72, ["prepare_dev_20418"])
    adopted_job("teacher_20424", "preparation", 20424, "prepare", "prepare",
                preparation["teacher"], SCIENCE_COMMIT, ["import_train_20424"])
    adopted_job("i0_development_gate_20418", "engineering_gate", 20418,
                "improvement", "evaluate", old["I0_development_gate"], SCIENCE_COMMIT,
                ["import_development_20418"])
    for group in ("greedy", "rule", "random"):
        adopted_job(f"i0_dev_B_{group}_20418", "engineering_gate", 20418,
                    "refine", "refine", old[f"I0_dev_B_{group}"], SCIENCE_COMMIT,
                    ["i0_development_gate_20418"])
    adopted_job("C0_pilot_20418", "pilot", 20418, "constructive_train", "train",
                old["C0"], SCIENCE_COMMIT, ["import_train_20418", "import_validation_20418"])
    adopted_job("C1_pilot_20424", "pilot", 20424, "constructive_train", "train",
                old["C1"], SCIENCE_COMMIT, ["teacher_20424", "import_validation_20424"])

    # Future stages are authorized but remain fenced by completed upstream
    # API results and a source-backed scientific gate. None is dispatched now.
    tasks.append(task("prepare_final_20424", "preparation", 20424, "prepare",
        {"preparation": {"preparation_type": "final",
                         "source_record": {
                             "path": ref("prepare_mothers_R1_20424", "result", "summary", "path"),
                             "sha256": ref("prepare_mothers_R1_20424", "result", "summary", "sha256")},
                         "seed": 51001, "environment_profile": "europe_v1"}},
        "final", ["prepare_mothers_R1_20424"]))
    for port in ops.SSH_PORTS:
        tasks.append(task(f"import_final_{port}", "preparation", port, "import_scene",
            {"path": record_directory("prepare_final_20424", "final", 20424),
             "name": f"S2-P19 final 880 {port}", "profile": "europe_v1"},
            "final", ["prepare_final_20424"],
            source_record_id="s2p19_final_20261007", expected_count=880))
    v2_pin = json.loads((ops.LOCAL / "preflight/p18_v2_model.json").read_text(encoding="utf-8"))
    if (v2_pin.get("provenance", {}).get("run_id") != "train_sac_20261006_084454_365eff71" or
            v2_pin.get("provenance", {}).get("env_step") != 400000):
        raise ValueError("historical V2 pin differs")
    tasks.append(task("import_V2_model_20418", "preparation", 20418, "import_model",
        {"path": v2_pin["source_dir"], "name": "S2-P18 V2 frozen 400k historical anchor"},
        "dev_validation", [], source_path=v2_pin["source_dir"],
        expected_sha256=v2_pin["files"]["policy.pt"]["sha256"]))
    tasks.append({"id": "i0_scientific_gate", "phase": "engineering_gate", "port": 20418,
                  "kind": "milestone",
                  "requires": [f"i0_dev_B_{group}_20418" for group in ("greedy", "rule", "random")],
                  "receipt_file": str(ops.LOCAL / "controller/gates/i0_nonlearning_receipt.json")})
    i0_defaults = ops.api(20418, "/api/s2p19/catalog")["defaults"]["I0"]
    if i0_defaults.get("seed") != 51001 or i0_defaults.get("total_steps") != 400000:
        raise ValueError("deployed I0 catalog differs from authorized budget")
    if any(item.get("request_id") == "s2p19-i0-train-51001" for item in runs[20418]):
        raise RuntimeError("I0 training request already exists; refuse duplicate role")
    tasks.append(task("I0_pilot_20418", "pilot", 20418, "improvement",
        {"mode": "train", "i0_config": i0_defaults,
         "scene_id": imports["train_20418"]["scene_id"],
         "validation_scene_id": imports["validation_20418"]["scene_id"]},
        "I0_pilot", ["i0_scientific_gate", "import_validation_20418"],
        request_id="s2p19-i0-train-51001",
        engineering_receipt_file=str(ops.LOCAL / "controller/gates/i0_logging_runtime_receipt.json"),
        expected_commit_source="i0_logging_runtime_receipt"))
    for arm, port in (("C0", 20418), ("C1", 20424), ("I0", 20418)):
        pilot = f"{arm}_pilot_{port}"
        tasks.append({"id": f"approve_{arm}_pilot", "phase": "pilot_gate", "port": port,
                      "kind": "approve_pilot", "pilot_task_id": pilot,
                      "requires": [pilot],
                      "gate_receipt_file": str(ops.LOCAL / f"controller/gates/{arm}_pilot_receipt.json")})

    plan = {"schema": 1, "experiment_id": "S2-P19", "scientific_commit": SCIENCE_COMMIT,
            "science_origin_commit": SCIENCE_COMMIT, "service_commit": SCIENCE_COMMIT,
            "revision": 1, "terminal": False,
            "science_commit_exception": {"task_id": "prepare_mothers_R1_20424",
                                         "commit": R1_MOTHER_COMMIT, "kind": "prepare"},
            "tasks": tasks}
    # Validate before publishing either file; controller remains stopped.
    if len({row["id"] for row in tasks}) != len(tasks):
        raise ValueError("adopt plan task IDs duplicate")
    for index, row in enumerate(tasks):
        if any(dep not in {earlier["id"] for earlier in tasks[:index]}
               for dep in row.get("requires", [])):
            raise ValueError("adopt plan dependency order differs")
    if not publish:
        return {"status": "adopt_preview_no_dispatch_no_plan_written",
                "scientific_commit": SCIENCE_COMMIT,
                "science_commit_exception": plan["science_commit_exception"],
                "adopted_jobs": {key: value.get("run_id") for key, value in receipts.items()
                                 if value.get("run_id")},
                "adopted_scene_imports": {key: value.get("scene_id") for key, value in receipts.items()
                                           if value.get("scene_id")},
                "future_task_ids": [row["id"] for row in tasks if row["id"] not in receipts]}
    controller.write_json(controller.PLAN, plan)
    plan_sha = controller.digest(controller.PLAN)
    state = {"schema": 1, "experiment_id": "S2-P19", "scientific_commit": SCIENCE_COMMIT,
             "science_origin_commit": SCIENCE_COMMIT, "service_commit": SCIENCE_COMMIT,
             "plan_sha256": plan_sha, "plan_revision": 1,
             "task_hashes": controller._task_hashes(tasks), "tasks": receipts,
             "events": [{"event": "adopted_existing_only", "actual_run_ids":
                         {key: value.get("run_id") for key, value in receipts.items() if value.get("run_id")},
                         "no_api_post": True}], "adopted_without_dispatch": True}
    controller.write_json(controller.STATE, state)
    _, actual_sha = controller.load_plan()
    if actual_sha != plan_sha:
        raise RuntimeError("published adopt plan SHA differs")
    return {"status": "adopted_no_dispatch", "plan": str(controller.PLAN),
            "state": str(controller.STATE), "plan_sha256": plan_sha,
            "adopted_jobs": sum(bool(value.get("run_id")) for value in receipts.values()),
            "adopted_scene_imports": sum(bool(value.get("scene_id")) for value in receipts.values()),
            "future_tasks": len(tasks) - len(receipts)}


def harden_live_plan(service_commit: str) -> dict:
    """Bind the already-adopted plan to a verified server and I0 log gate.

    This is a local metadata migration while the persistent controller is
    stopped. C0/C1 adopted job commits and run IDs are never rewritten.
    """
    plan, old_sha = controller.load_plan()
    if (plan.get("revision") != 1 or plan.get("scientific_commit") != SCIENCE_COMMIT or
            plan.get("service_commit", SCIENCE_COMMIT) != SCIENCE_COMMIT or
            not controller.STATE.is_file() or
            (ops.LOCAL / "controller/latest_observation.json").exists()):
        raise RuntimeError("harden requires original adopted plan before controller start")
    state = json.loads(controller.STATE.read_text(encoding="utf-8"))
    if (state.get("plan_sha256") != old_sha or state.get("plan_revision") != 1 or
            state.get("scientific_commit") != SCIENCE_COMMIT or
            any(item.get("event") != "adopted_existing_only" for item in state.get("events", []))):
        raise RuntimeError("adopted state has already changed")
    if len(service_commit) != 40 or any(c not in "0123456789abcdef" for c in service_commit):
        raise ValueError("service commit must be an exact Git SHA")
    for port in ops.SSH_PORTS:
        controller._service_identity(port, service_commit)
    manifest = json.loads((ops.LOCAL / "run_manifest.json").read_text(encoding="utf-8"))
    saved = manifest["tasks"]["final880_preparation"]
    run_id = saved["run_id"]
    if (saved["request_id"] != "s2p19-prepare-final-20424" or
            saved["git_commit"] != SCIENCE_COMMIT):
        raise RuntimeError("manual final880 task does not match adopted science origin")
    detail = ops.api(20424, f"/api/runs/{run_id}")
    job, status = detail["job"], detail["status"]
    if (job.get("run_id") != run_id or job.get("request_id") != saved["request_id"] or
            job.get("git_commit") != SCIENCE_COMMIT or job.get("kind") != "prepare" or
            controller._adopted_config_sha(20424, run_id) != saved["config_sha256"] or
            status.get("stage") != "completed" or status.get("env_step") != 880):
        raise RuntimeError("final880 workbench task identity/completion differs")
    output = ops.api(20424, f"/api/runs/{run_id}/results")
    final = (output.get("summary") or {}).get("outputs", {}).get("final") or {}
    if final.get("count") != 880 or not final.get("path") or not final.get("sha256"):
        raise RuntimeError("final880 completed output record missing")
    replacement = {"id": "prepare_final_20424", "phase": "preparation", "port": 20424,
                   "kind": "prepare", "requires": ["prepare_mothers_R1_20424"],
                   "adopted_only": True, "adopted_run_id": run_id,
                   "request_id": saved["request_id"],
                   "expected_config_sha256": saved["config_sha256"],
                   "expected_commit": SCIENCE_COMMIT, "workbench_kind": "prepare"}
    new_plan = deepcopy(plan)
    new_plan.update(revision=2, parent_plan_sha256=old_sha,
                    science_origin_commit=SCIENCE_COMMIT,
                    service_commit=service_commit,
                    migration_kind="i0_logging_hold_service_split_v1")
    for index, row in enumerate(new_plan["tasks"]):
        if row["id"] == "prepare_final_20424":
            new_plan["tasks"][index] = replacement
        elif row["id"] == "I0_pilot_20418":
            row["engineering_receipt_file"] = str(
                ops.LOCAL / "controller/gates/i0_logging_runtime_receipt.json")
            row["expected_commit_source"] = "i0_logging_runtime_receipt"
    previous_plan = controller.PLAN.read_bytes()
    previous_state = controller.STATE.read_bytes()
    migration_dir = ops.LOCAL / "controller/migrations"
    migration_dir.mkdir(parents=True, exist_ok=True)
    backup_plan = migration_dir / "adopted_revision1_plan.json"
    backup_state = migration_dir / "adopted_revision1_state.json"
    if backup_plan.exists() or backup_state.exists():
        raise FileExistsError("existing controller migration backup needs review")
    backup_plan.write_bytes(previous_plan)
    backup_state.write_bytes(previous_state)
    try:
        controller.write_json(controller.PLAN, new_plan)
        new_sha = controller.digest(controller.PLAN)
        new_state = deepcopy(state)
        new_state.update(plan_sha256=new_sha, plan_revision=2,
                         task_hashes=controller._task_hashes(new_plan["tasks"]),
                         science_origin_commit=SCIENCE_COMMIT,
                         service_commit=service_commit)
        new_state["tasks"]["prepare_final_20424"] = {
            "state": "completed", "run_id": run_id, "request_id": saved["request_id"],
            "workbench_stage": "completed", "config_sha256": saved["config_sha256"],
            "git_commit": SCIENCE_COMMIT, "adopted_only": True,
            "result": output, "observed_at": status.get("heartbeat_at")}
        new_state.setdefault("events", []).append({"event": "i0_logging_hold_service_split_v1",
            "old_plan_sha256": old_sha, "new_plan_sha256": new_sha,
            "science_origin_commit": SCIENCE_COMMIT,
            "service_commit": service_commit,
            "final880_run_id": run_id, "no_api_post": True})
        controller.write_json(controller.STATE, new_state)
        controller.load_plan()
    except BaseException:
        controller.PLAN.write_bytes(previous_plan)
        controller.STATE.write_bytes(previous_state)
        raise
    receipt = {"migration_kind": "i0_logging_hold_service_split_v1",
               "old_plan_sha256": old_sha, "new_plan_sha256": new_sha,
               "science_origin_commit": SCIENCE_COMMIT, "service_commit": service_commit,
               "final880_run_id": run_id, "final880_config_sha256": saved["config_sha256"],
               "final880_record_sha256": final["sha256"],
               "controller_started": False, "API_POST_count": 0}
    path = migration_dir / "i0_hold_service_split_receipt.json"
    controller.write_json(path, receipt)
    return {"status": "hardened_adopted_plan_no_dispatch", "plan_sha256": new_sha,
            "receipt": str(path), "service_commit": service_commit,
            "science_origin_commit": SCIENCE_COMMIT}


def advance_common_evaluation() -> dict:
    """Append retained final-model evaluations and same-route final B tasks.

    The root scientific audit writes only a retained-arms receipt. All model
    IDs, scene IDs and task outputs are resolved from completed workbench API
    records; no run ID or result path is manually entered into the plan.
    """
    plan, old_sha = controller.load_plan()
    if any(row["id"] == "V2_final_direct_20418" for row in plan["tasks"]):
        raise ValueError("common evaluation tasks already appended")
    if not controller.STATE.is_file():
        raise RuntimeError("controller has no verified pilot/scene receipts")
    state = json.loads(controller.STATE.read_text(encoding="utf-8"))
    receipts = state.get("tasks") or {}
    decision_file = ops.LOCAL / "controller/gates/retained_arms.json"
    if not decision_file.is_file():
        raise RuntimeError("missing source-backed scientific retention decision")
    decision = json.loads(decision_file.read_text(encoding="utf-8"))
    audit_path = Path(decision["audit_path"])
    if not audit_path.is_file() or controller.digest(audit_path) != decision["audit_sha256"]:
        raise RuntimeError("retained-arms decision audit SHA differs")
    retained = decision.get("retained_arms")
    if not isinstance(retained, list) or len(set(retained)) != len(retained) or not set(retained) <= {"C0", "C1", "I0"}:
        raise ValueError("retained arms outside authorized first round")
    if "C1" in retained and "C0" not in retained:
        raise ValueError("C1 attribution requires a matched C0 final model")
    for port in ops.SSH_PORTS:
        if receipts.get(f"import_final_{port}", {}).get("state") != "completed":
            raise RuntimeError("880-scene final import incomplete on both instances")
    for arm in retained:
        port = 20424 if arm == "C1" else 20418
        pilot = receipts.get(f"{arm}_pilot_{port}") or {}
        if pilot.get("state") != "completed" or not pilot.get("run_id"):
            raise RuntimeError(f"{arm} retained without completed same-ID full training")
        if receipts.get(f"approve_{arm}_pilot", {}).get("state") != "completed":
            raise RuntimeError(f"{arm} missing saved pilot scientific approval")

    p17_configs = json.loads(Path("D:/paper/outputs/verification/s2p19_research_20261007/p17_sca_configs.json").read_text(encoding="utf-8"))
    stochastic_B = p17_configs["V1"]["config_power_refinement"]
    greedy_B = p17_configs["B1"]["config_power_refinement"]
    if (stochastic_B["methods"] != ["none", "sca", "sca_si_lp"] or
            greedy_B["methods"] != stochastic_B["methods"] or
            stochastic_B["repeats"] != 3 or greedy_B["repeats"] != 1):
        raise ValueError("saved P17 common B route differs")
    appended = []
    appended.append(task("V2_final_direct_20418", "common_evaluation", 20418,
        "constructive_evaluate",
        {"experiment_id": "S2-P19", "model_id": ref("import_V2_model_20418", "model_id"),
         "scene_id": ref("import_final_20418", "scene_id"),
         "action_mode": "stochastic", "action_seed": 53001, "repeats": 3,
         "baselines": []},
        "final_direct_model_2640", ["import_V2_model_20418", "import_final_20418"]))
    appended.append(task("V2_final_B_20418", "refinement", 20418, "refine",
        {"source_run_id": ref("V2_final_direct_20418", "run_id"),
         "source_group": "model", "partition": "test", "config": stochastic_B},
        "final_B_one_stochastic_arm_2640", ["V2_final_direct_20418"]))
    appended.append(task("baseline_test_20418", "common_evaluation", 20418, "improvement",
        {"mode": "baselines", "partition": "test", "i0_config": ops.api(20418, "/api/s2p19/catalog")["defaults"]["I0"],
         "scene_id": ref("import_final_20418", "scene_id"),
         "methods": ["greedy", "rule", "random"], "repeats": 1, "action_seed": 53001},
        "final_direct_baselines_4400", ["import_final_20418"]))
    for group in ("greedy", "rule", "random"):
        controls = stochastic_B if group == "random" else greedy_B
        disk_class = ("final_B_one_stochastic_arm_2640" if group == "random"
                      else "final_B_one_deterministic_arm_880")
        appended.append(task(f"baseline_{group}_B_20418", "refinement", 20418, "refine",
            {"source_run_id": ref("baseline_test_20418", "run_id"),
             "source_group": group, "partition": "test", "config": controls},
            disk_class, ["baseline_test_20418"]))
    for arm in retained:
        port = 20424 if arm == "C1" else 20418
        pilot = f"{arm}_pilot_{port}"
        final_scene = f"import_final_{port}"
        if arm in ("C0", "C1"):
            model_id = _final_model_id(port, receipts[pilot]["run_id"])
            body = {"experiment_id": "S2-P19", "model_id": model_id,
                    "scene_id": ref(final_scene, "scene_id"),
                    "action_mode": "stochastic", "action_seed": 53001, "repeats": 3,
                    "baselines": []}
            kind = "constructive_evaluate"
        else:
            body = {"mode": "evaluate", "partition": "test",
                    "scene_id": ref(final_scene, "scene_id"),
                    "model_run_id": ref(pilot, "run_id"),
                    "action_seed": 53001, "repeats": 3}
            kind = "improvement"
        eval_id = f"{arm}_final_direct_{port}"
        appended.append(task(eval_id, "common_evaluation", port, kind, body,
                             "final_direct_model_2640",
                             [pilot, final_scene]))
        appended.append(task(f"{arm}_final_B_{port}", "refinement", port, "refine",
            {"source_run_id": ref(eval_id, "run_id"), "source_group": "model",
             "partition": "test", "config": stochastic_B},
            "final_B_one_stochastic_arm_2640", [eval_id]))
    appended.append({"id": "formal_report_and_archive", "phase": "report_archive",
                     "port": 20418, "kind": "milestone",
                     "requires": [row["id"] for row in appended if row["phase"] == "refinement"],
                     "receipt_file": str(ops.LOCAL / "controller/gates/formal_report_archive_receipt.json")})
    runtime = plan.get("service_commit", plan["scientific_commit"])
    for row in appended:
        if row["kind"] in ("constructive_evaluate", "improvement", "refine"):
            row["expected_commit"] = runtime
    new_plan = {**plan, "revision": plan["revision"] + 1, "parent_plan_sha256": old_sha,
                "terminal": False, "retention_audit_path": str(audit_path),
                "retention_audit_sha256": decision["audit_sha256"],
                "tasks": [*plan["tasks"], *appended]}
    controller.write_json(controller.PLAN, new_plan)
    controller.load_plan()
    return {"status": "appended_common_evaluation", "retained_arms": retained,
            "new_task_count": len(appended), "plan_sha256": controller.digest(controller.PLAN),
            "unplanned_routes": [],
            "note": "Retained arms plus frozen V2 and the 1/1/3 common baselines are scheduled; report/archive milestone remains."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("bootstrap", "adopt-preview", "adopt",
                                           "harden-live", "advance-common-evaluation"))
    parser.add_argument("--service-commit")
    args = parser.parse_args()
    result = (bootstrap() if args.action == "bootstrap" else adopt(publish=False) if args.action == "adopt-preview"
              else adopt() if args.action == "adopt"
              else harden_live_plan(args.service_commit) if args.action == "harden-live"
              else advance_common_evaluation())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
