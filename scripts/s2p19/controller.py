"""Persistent, typed S2-P19 workbench progression with scientific gates.

The controller reads a frozen JSON DAG of bounded tasks. It never calls a
training script; all work goes through the workbench API. It halts on any
failed/stopped task, missing gate audit, source drift or disk forecast breach.
Use ``--once`` for a read-only snapshot before enabling dispatch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from . import disk_guard, ops


PLAN = ops.LOCAL / "controller" / "plan.json"
STATE = ops.LOCAL / "controller" / "state.json"
PHASES = ("preparation", "engineering_gate", "pilot", "pilot_gate",
          "continuation", "common_evaluation", "refinement", "report_archive")
ENDPOINTS = {
    "prepare": "/api/s2p19/prepare",
    "constructive_train": "/api/s2p19/train",
    "constructive_evaluate": "/api/evaluations",
    "improvement": "/api/s2p19/improvement",
    "refine": "/api/power-refinement/runs",
    "import_scene": "/api/scenes/import",
    "import_model": "/api/models/import",
}
ACTIVE = {"running", "training", "evaluating", "preparing", "queued"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan() -> tuple[dict, str]:
    if not PLAN.is_file():
        raise FileNotFoundError("root must freeze S2-P19 controller/plan.json before dispatch")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("schema") != 1 or plan.get("experiment_id") != "S2-P19":
        raise ValueError("controller plan identity differs")
    if type(plan.get("revision")) is not int or plan["revision"] < 1 or type(plan.get("terminal")) is not bool:
        raise ValueError("controller plan needs revision and explicit terminal flag")
    commit = plan.get("scientific_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("controller plan needs exact frozen scientific commit")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("controller plan has no typed tasks")
    seen_ids, seen_requests = set(), set()
    for index, task in enumerate(tasks):
        identifier = task.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
            raise ValueError("controller task id missing/duplicate")
        seen_ids.add(identifier)
        if task.get("phase") not in PHASES or task.get("port") not in ops.SSH_PORTS:
            raise ValueError(f"{identifier}: unknown phase or instance")
        kind = task.get("kind")
        if kind not in (*ENDPOINTS, "approve_pilot", "milestone"):
            raise ValueError(f"{identifier}: unknown typed task kind")
        if kind in ENDPOINTS:
            request = task.get("request_id")
            if kind not in ("import_scene", "import_model") and (
                    not isinstance(request, str) or request in seen_requests):
                raise ValueError(f"{identifier}: missing/duplicate request id")
            if request:
                seen_requests.add(request)
            if task.get("disk_class") not in disk_guard.PREDICTED_BYTES:
                raise ValueError(f"{identifier}: disk class missing")
            if not isinstance(task.get("body"), dict):
                raise ValueError(f"{identifier}: typed API body missing")
            if kind == "import_scene" and (not task.get("source_record_id") or
                                           not isinstance(task.get("expected_count"), int)):
                raise ValueError(f"{identifier}: source record identity/count required")
            if kind == "import_model" and (not task.get("source_path") or
                                           not task.get("expected_sha256")):
                raise ValueError(f"{identifier}: frozen model source path/SHA required")
        required = task.get("requires", [])
        if not isinstance(required, list) or any(dep not in {row["id"] for row in tasks[:index]} for dep in required):
            raise ValueError(f"{identifier}: dependency must precede task")
    return plan, digest(PLAN)


def _task_hashes(tasks: list[dict]) -> list[str]:
    return [hashlib.sha256(json.dumps(task, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()
            for task in tasks]


def _run_rows(port: int) -> list[dict]:
    response = ops.api(port, "/api/s2p19/runs")
    rows = response.get("runs")
    if not isinstance(rows, list):
        raise RuntimeError("S2-P19 workbench run catalog malformed")
    return rows


def _scene_rows(port: int) -> list[dict]:
    response = ops.api(port, "/api/scenes")
    rows = response.get("scenes")
    if not isinstance(rows, list):
        raise RuntimeError("workbench scene catalog malformed")
    return rows


def _model_rows(port: int) -> list[dict]:
    response = ops.api(port, "/api/models")
    rows = response.get("models")
    if not isinstance(rows, list):
        raise RuntimeError("workbench model catalog malformed")
    return rows


def _get_job(port: int, run_id: str) -> dict:
    return ops.api(port, "/api/runs/" + run_id)


def _result(port: int, run_id: str) -> dict:
    return ops.api(port, "/api/runs/" + run_id + "/results")


def _pilot_gate_proof(port: int, run_id: str) -> dict | None:
    if not isinstance(run_id, str) or not run_id.startswith("train_") or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in run_id):
        raise ValueError("invalid pilot run identity")
    info = ops.instance(port)
    code = f'''
import json,pathlib
path=pathlib.Path({info['archive']!r})/'outputs/workbench/jobs'/{run_id!r}/'pilot_gate.json'
print(json.dumps(json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None))
'''
    result = ops.remote_python(code, port)
    if result["exit_code"]:
        raise RuntimeError("pilot gate proof inspection failed")
    return json.loads(result["stdout"].splitlines()[-1])


def _resolve(value, receipts: dict):
    if isinstance(value, dict) and set(value) == {"$task"}:
        spec = value["$task"]
        if not isinstance(spec, dict) or set(spec) != {"id", "path"}:
            raise ValueError("$task reference must have id/path")
        source = receipts.get(spec["id"])
        if not source or source.get("state") != "completed":
            raise ValueError("dependent task result is not complete")
        selected = source
        for key in spec["path"]:
            selected = selected[key]
        return selected
    if isinstance(value, dict):
        return {key: _resolve(item, receipts) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, receipts) for item in value]
    return value


def _service_identity(port: int, commit: str) -> dict:
    info = ops.instance(port)
    launch_path = ops.LOCAL / "preflight" / f"workbench_{commit[:12]}.json"
    if not launch_path.is_file():
        raise RuntimeError("missing saved S2-P19 workbench launch receipt")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch.get("scientific_commit") != commit or str(port) not in launch.get("instances", {}):
        raise RuntimeError("workbench launch receipt is for another source/instance")
    pinned = launch["instances"][str(port)]
    code = f'''
import json,pathlib,subprocess
repo=pathlib.Path({info['repo']!r})
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
pid={pinned['pid']};proc=pathlib.Path('/proc')/str(pid)
birth=None;cmdline=''
if proc.is_dir():
 stat=(proc/'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
 cmdline=(proc/'cmdline').read_bytes().replace(b'\\x00',b' ').decode('utf-8','replace')
print(json.dumps({{'head':head,'dirty':dirty,'server_pid':pid,'start_ticks':birth,
 'cmdline_matches':{info['archive']!r} in cmdline and 'satellite_rl.workbench.server' in cmdline}}))
'''
    observed = ops.remote_python(code, port)
    if observed["exit_code"]:
        raise RuntimeError(f"{port} frozen source identity unavailable")
    identity = json.loads(observed["stdout"].splitlines()[-1])
    if (identity["head"] != commit or identity["dirty"] or
            identity["start_ticks"] != pinned["start_ticks"] or
            not identity["cmdline_matches"]):
        raise RuntimeError(f"{port} remote checkout differs from frozen commit")
    return identity


def _disk_emergency(port: int, snapshot: dict, runs: list[dict], dispatch: bool) -> dict | None:
    if snapshot["home_free_bytes"] >= disk_guard.RESERVE_BYTES:
        return None
    incident = {"port": port, "observed_utc": snapshot["observed_utc"],
                "home_free_bytes": snapshot["home_free_bytes"], "reserve_bytes": disk_guard.RESERVE_BYTES,
                "active_s2p19_runs": [row["run_id"] for row in runs if row.get("stage") in ACTIVE]}
    if dispatch:
        incident["stop_receipts"] = {}
        for run_id in incident["active_s2p19_runs"]:
            incident["stop_receipts"][run_id] = ops.api(port, f"/api/runs/{run_id}/stop", {}, "POST")
    return incident


def _task_status(task: dict, receipts: dict, runs: dict[int, list[dict]],
                 scenes: dict[int, list[dict]], models: dict[int, list[dict]]) -> dict:
    task_id = task["id"]
    saved = receipts.get(task_id)
    if task["kind"] == "milestone":
        return saved or {"state": "waiting"}
    if task["kind"] == "approve_pilot":
        if saved:
            return saved
        pilot = receipts.get(task["pilot_task_id"]) or {}
        run_id = pilot.get("run_id")
        if run_id and pilot.get("state") in ("running", "completed"):
            proof = _pilot_gate_proof(task["port"], run_id)
            if proof is not None:
                if proof.get("pass") is not True or proof.get("run_id") != run_id:
                    raise RuntimeError("saved pilot gate proof has wrong run identity")
                return {"state": "completed", "approved_run_id": run_id,
                        "recovered_from_workbench_proof": True}
        return {"state": "waiting"}
    if task["kind"] == "import_scene":
        if saved and saved.get("scene_id"):
            match = [scene for scene in scenes[task["port"]] if scene.get("id") == saved["scene_id"]]
            return {**saved, "state": "completed"} if len(match) == 1 else {**saved, "state": "attention_required"}
        expected = task.get("source_record_id")
        matches = [scene for scene in scenes[task["port"]]
                   if ((scene.get("source") or {}).get("reimport") or {}).get("record_id") == expected]
        if len(matches) > 1:
            raise RuntimeError(f"{task_id}: duplicate scene import source record")
        if len(matches) == 1:
            if task.get("expected_count") != matches[0].get("count"):
                raise RuntimeError(f"{task_id}: existing scene count differs")
            return {"state": "completed", "scene_id": matches[0]["id"],
                    "recovered_from_catalog": True}
        return {"state": "waiting"}
    if task["kind"] == "import_model":
        if saved and saved.get("model_id"):
            match = [model for model in models[task["port"]] if model.get("id") == saved["model_id"]]
            return {**saved, "state": "completed"} if len(match) == 1 else {**saved, "state": "attention_required"}
        matches = [model for model in models[task["port"]]
                   if model.get("source_path") == task["source_path"]
                   and model.get("sha256") == task["expected_sha256"]]
        if len(matches) > 1:
            raise RuntimeError(f"{task_id}: duplicate frozen model import")
        return ({"state": "completed", "model_id": matches[0]["id"],
                 "recovered_from_catalog": True} if len(matches) == 1 else {"state": "waiting"})
    matches = [row for row in runs[task["port"]] if row.get("request_id") == task["request_id"]]
    if len(matches) > 1:
        raise RuntimeError(f"{task_id}: duplicate workbench request id")
    if not matches:
        if saved and saved.get("run_id"):
            return {**saved, "state": "attention_required", "reason": "receipt has no matching live catalog run"}
        return {"state": "waiting"}
    run = matches[0]
    if saved and saved.get("run_id") not in (None, run["run_id"]):
        raise RuntimeError(f"{task_id}: request id changed run identity")
    stage = run.get("stage")
    state = ("completed" if stage == "completed" else
             "attention_required" if stage in ("failed", "stopped") else "running")
    result = {**(saved or {}), "run_id": run["run_id"], "workbench_stage": stage,
              "state": state, "observed_at": now()}
    if state == "completed" and task["kind"] == "prepare" and "result" not in result:
        result["result"] = _result(task["port"], run["run_id"])
    if stage == "stopped":
        detail = _get_job(task["port"], run["run_id"])
        result["stop_reason"] = (detail.get("status") or {}).get("stop_reason")
        if result["stop_reason"] == "s2p19_pilot_gate_pending":
            result["state"] = "pilot_gate_pending"
    return result


def step(dispatch: bool = False) -> dict:
    plan, plan_sha = load_plan()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {
        "schema": 1, "experiment_id": "S2-P19", "scientific_commit": plan["scientific_commit"],
        "plan_sha256": plan_sha, "plan_revision": plan["revision"],
        "task_hashes": _task_hashes(plan["tasks"]), "tasks": {}, "events": []}
    if state.get("scientific_commit") != plan["scientific_commit"]:
        raise RuntimeError("persistent controller source commit changed")
    if state.get("plan_sha256") != plan_sha:
        old_hashes = state.get("task_hashes") or []
        if (plan["revision"] != state.get("plan_revision", 0) + 1 or
                plan.get("parent_plan_sha256") != state["plan_sha256"] or
                _task_hashes(plan["tasks"][:len(old_hashes)]) != old_hashes or
                len(plan["tasks"]) <= len(old_hashes)):
            raise RuntimeError("controller plan changed outside an append-only phase advance")
        state["events"].append({"event": "plan_advanced", "from_sha256": state["plan_sha256"],
                                "to_sha256": plan_sha, "at": now()})
        state.update(plan_sha256=plan_sha, plan_revision=plan["revision"],
                     task_hashes=_task_hashes(plan["tasks"]))
    identities = {port: _service_identity(port, plan["scientific_commit"]) for port in ops.SSH_PORTS}
    runs = {port: _run_rows(port) for port in ops.SSH_PORTS}
    scenes = {port: _scene_rows(port) for port in ops.SSH_PORTS}
    models = {port: _model_rows(port) for port in ops.SSH_PORTS}
    disks = {port: disk_guard.sample(port) for port in ops.SSH_PORTS}
    emergencies = {port: _disk_emergency(port, disks[port], runs[port], dispatch)
                   for port in ops.SSH_PORTS}
    if any(emergencies.values()):
        receipt = {"status": "disk_attention", "observed_utc": now(),
                   "emergencies": emergencies, "identities": identities, "disks": disks}
        if dispatch:
            state["events"].append(receipt)
            write_json(STATE, state)
        return receipt
    receipts = dict(state["tasks"])
    for task in plan["tasks"]:
        receipts[task["id"]] = _task_status(task, receipts, runs, scenes, models)
    state["tasks"] = receipts
    waiting = []
    for task in plan["tasks"]:
        task_id = task["id"]
        observed = receipts[task_id]
        if observed["state"] in ("completed", "running"):
            continue
        if observed["state"] in ("attention_required", "pilot_gate_pending"):
            waiting.append({"id": task_id, **observed})
            continue
        def ready_dependency(dep):
            observed_state = receipts[dep]["state"]
            if task["kind"] == "approve_pilot" and dep == task.get("pilot_task_id"):
                return observed_state == "pilot_gate_pending"
            return observed_state == "completed"
        if not all(ready_dependency(dep) for dep in task.get("requires", [])):
            continue
        if task["kind"] == "milestone":
            reference_file = task.get("receipt_file")
            reference = (json.loads(Path(reference_file).read_text(encoding="utf-8"))
                         if reference_file and Path(reference_file).is_file() else {})
            receipt = reference.get("path")
            receipt_sha = reference.get("sha256")
            if (not receipt or not receipt_sha or not Path(receipt).is_file() or
                    digest(Path(receipt)) != receipt_sha):
                waiting.append({"id": task_id, "state": "waiting_source_backed_milestone_receipt"})
                continue
            if dispatch:
                receipts[task_id] = {"state": "completed", "phase": task["phase"],
                                     "receipt_path": receipt, "receipt_sha256": receipt_sha,
                                     "at": now()}
                state["events"].append({"event": "milestone", "task_id": task_id, "at": now()})
                write_json(STATE, state)
            else:
                waiting.append({"id": task_id, "state": "ready_milestone"})
            continue
        if task["kind"] == "approve_pilot":
            pilot = receipts[task["pilot_task_id"]]
            if pilot.get("state") != "pilot_gate_pending":
                waiting.append({"id": task_id, "state": "waiting_pilot_gate"})
                continue
            reference_file = task.get("gate_receipt_file")
            reference = (json.loads(Path(reference_file).read_text(encoding="utf-8"))
                         if reference_file and Path(reference_file).is_file() else {})
            if not reference.get("audit_path") or not reference.get("audit_sha256"):
                waiting.append({"id": task_id, "state": "waiting_scientific_audit"})
                continue
            if dispatch:
                body = {"audit_path": reference["audit_path"],
                        "audit_sha256": reference["audit_sha256"]}
                response = ops.api(task["port"], f"/api/s2p19/runs/{pilot['run_id']}/approve-pilot",
                                   body, "POST")
                receipts[task_id] = {"state": "completed", "approved_run_id": pilot["run_id"],
                                     "response": response, "at": now()}
                state["events"].append({"event": "pilot_approved", "task_id": task_id,
                                        "run_id": pilot["run_id"], "at": now()})
                write_json(STATE, state)
            else:
                waiting.append({"id": task_id, "state": "ready_gate_approval"})
            continue
        active_reservation = sum(
            disk_guard.PREDICTED_BYTES[row["disk_class"]]
            for row in plan["tasks"] if row.get("disk_class") in disk_guard.PREDICTED_BYTES
            and row["id"] != task_id and receipts[row["id"]].get("state") == "running")
        capacity = disk_guard.gate(disks[task["port"]], task["disk_class"], active_reservation)
        if not capacity["admit"]:
            waiting.append({"id": task_id, "state": "disk_capacity_hold", "capacity": capacity})
            continue
        if not dispatch:
            waiting.append({"id": task_id, "state": "ready_to_submit", "capacity": capacity})
            continue
        body = _resolve(task["body"], receipts)
        if task["kind"] not in ("import_scene", "import_model"):
            if body.get("request_id") not in (None, task["request_id"]):
                raise ValueError(f"{task_id}: API body request id mismatch")
            body["request_id"] = task["request_id"]
        result = ops.api(task["port"], ENDPOINTS[task["kind"]], body, "POST", timeout=180)
        entry = {"state": "running", "phase": task["phase"], "submitted_at": now(),
                 "request_id": task.get("request_id"), "capacity": capacity, "response": result}
        if task["kind"] == "import_scene":
            entry.update(state="completed", scene_id=result["id"])
        elif task["kind"] == "import_model":
            entry.update(state="completed", model_id=result["id"])
        else:
            entry["run_id"] = result["run_id"]
        receipts[task_id] = entry
        state["events"].append({"event": "submitted", "task_id": task_id,
                                "run_id": entry.get("run_id"), "at": now()})
        write_json(STATE, state)
        # One dependent submission per poll permits fresh API and disk observation.
        break
    state["last_observed_utc"] = now()
    state["tasks"] = receipts
    if dispatch:
        write_json(STATE, state)
    summary = {"status": "observed" if not dispatch else "controller_active",
               "observed_utc": state["last_observed_utc"],
               "scientific_commit": plan["scientific_commit"],
               "completed": sum(item.get("state") == "completed" for item in receipts.values()),
               "total": len(plan["tasks"]), "waiting": waiting,
               "tasks": {key: {k: value.get(k) for k in ("state", "run_id", "workbench_stage", "stop_reason")}
                         for key, value in receipts.items()},
               "disk_free_bytes": {str(port): disks[port]["home_free_bytes"] for port in ops.SSH_PORTS}}
    return summary


def maybe_advance_common_evaluation() -> dict | None:
    plan, _ = load_plan()
    if plan["revision"] != 1:
        return None
    decision_path = ops.LOCAL / "controller/gates/retained_arms.json"
    if not decision_path.is_file() or not STATE.is_file():
        return None
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    retained = decision.get("retained_arms")
    if not isinstance(retained, list) or not set(retained) <= {"C0", "C1", "I0"}:
        raise ValueError("retained-arms receipt has invalid scope")
    receipts = json.loads(STATE.read_text(encoding="utf-8")).get("tasks") or {}
    if any(receipts.get(f"import_final_{port}", {}).get("state") != "completed"
           for port in ops.SSH_PORTS):
        return None
    for arm in retained:
        port = 20424 if arm == "C1" else 20418
        if (receipts.get(f"{arm}_pilot_{port}", {}).get("state") != "completed" or
                receipts.get(f"approve_{arm}_pilot", {}).get("state") != "completed"):
            return None
    from .plan_builder import advance_common_evaluation
    return advance_common_evaluation()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="read-only current snapshot")
    parser.add_argument("--run", action="store_true", help="persistent API dispatch after frozen plan")
    parser.add_argument("--interval-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.once == args.run or args.interval_seconds < 30:
        parser.error("choose exactly --once or --run; interval must be >=30 seconds")
    if args.once:
        print(json.dumps(step(False), ensure_ascii=False, indent=2))
        return
    while True:
        report = step(True)
        advancement = maybe_advance_common_evaluation()
        if advancement is not None:
            report["plan_advance"] = advancement
        write_json(ops.LOCAL / "controller" / "latest_observation.json", report)
        if report["status"] == "disk_attention":
            return
        plan, _ = load_plan()
        if report["completed"] == report["total"] and plan["terminal"]:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
