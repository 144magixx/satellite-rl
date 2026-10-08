"""Independent four-instance final-assessment coordinator for S2-P19.

Only the final seven workbench jobs are managed here. Migration produces a
separate, reviewable plan; this module never edits the legacy controller.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


PORTS = (20418, 20424, 20473, 20474)
API_PORTS = {20418: 8900, 20424: 8901, 20473: 8904, 20474: 8905}
HOST = "10.160.17.102"
USER = "u2024110224"
HOST_KEY = "a6ym3EfC35pPdkqWO8lg+4As2nLDI1eKu8cur4P7WDc"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
BASE = "/home/u2024110224/second_stage_satellite_rl/sac_routes_s2p19"
ENDPOINT = {"constructive_evaluate": "/api/evaluations",
            "improvement": "/api/s2p19/improvement",
            "refine": "/api/power-refinement/runs"}
ACTIVE = {"queued", "running", "training", "evaluating", "preparing"}
FINAL_IDS = ("baseline_random_B_20418", "C0_final_direct_20473",
             "C0_final_B_20473", "C1_final_direct_20424", "C1_final_B_20424",
             "I0_final_direct_20474", "I0_final_B_20474")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def api(port: int, endpoint: str, body: dict | None = None) -> dict:
    """Pinned SSH to the instance-local API; password is never serialized."""
    if port not in PORTS or endpoint not in (*ENDPOINT.values(), "/api/s2p19/runs"):
        raise ValueError("invalid four-lane API target")
    import paramiko

    password = os.environ.get("BUPT_PASSWORD")
    if not password:
        raise RuntimeError("BUPT_PASSWORD is unavailable")

    class Pinned(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            actual = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
            if actual != HOST_KEY:
                raise paramiko.SSHException("SSH host key differs from pinned host")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(Pinned())
    client.connect(HOST, port=port, username=USER, password=password,
                   allow_agent=False, look_for_keys=False, timeout=15,
                   banner_timeout=20, auth_timeout=20)
    payload = json.dumps(body, ensure_ascii=False) if body is not None else None
    code = f'''
import json,urllib.error,urllib.request
payload={payload!r}
request=urllib.request.Request('http://127.0.0.1:{API_PORTS[port]}{endpoint}',
 data=None if payload is None else payload.encode('utf-8'),
 method='GET' if payload is None else 'POST',
 headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
try:
 with urllib.request.urlopen(request,timeout=180) as response:
  print(json.dumps({{'status':response.status,'body':json.loads(response.read().decode('utf-8'))}}))
except urllib.error.HTTPError as error:
 print(json.dumps({{'status':error.code,'error':error.read().decode('utf-8','replace')[:2000]}}))
'''
    command = (f"{DQN} -c \"import base64;exec(base64.b64decode('"
               + base64.b64encode(code.encode()).decode() + "').decode('utf-8'))\"")
    try:
        _, stdout, stderr = client.exec_command(command, timeout=220)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        status = stdout.channel.recv_exit_status()
    finally:
        client.close()
    if status:
        raise ConnectionError(f"{port} API transport failed: {err.replace(password, '[REDACTED]')[-300:]}")
    result = json.loads(out.splitlines()[-1])
    if result["status"] >= 400:
        raise RuntimeError(f"{port} API HTTP {result['status']}: {result.get('error')}")
    return result["body"]


def migrate(old_plan: Path, old_state: Path, output: Path, *, c0_model_id: str,
            c1_model_id: str, c0_scene_id: str, i0_scene_id: str,
            i0_source_archive: str, i0_runtime_commit: str,
            c0_direct_run_id: str, c1_direct_run_id: str,
            i0_direct_run_id: str) -> dict:
    """Freeze only remaining work, preserving original experiment contracts."""
    source = json.loads(old_plan.read_text(encoding="utf-8"))
    legacy = json.loads(old_state.read_text(encoding="utf-8"))
    if (source.get("experiment_id") != "S2-P19" or source.get("revision") != 6 or
            legacy.get("plan_sha256") != sha(old_plan)):
        raise ValueError("legacy plan/state identity differs")
    old = {row["id"]: row for row in source["tasks"]}
    saved = legacy["tasks"]
    for key in ("V2_final_direct_20418", "V2_final_B_20418", "baseline_test_20418",
                "baseline_greedy_B_20418", "baseline_rule_B_20418"):
        if saved[key]["state"] != "completed" or not saved[key].get("run_id"):
            raise ValueError(f"{key} is not a completed legacy workbench job")
    random_key = "baseline_random_B_20418"
    if saved[random_key].get("run_id") != "refine_sac_20261008_120914_687c01ef":
        raise ValueError("original random B run identity differs")
    if saved[random_key]["state"] not in ("running", "completed"):
        raise ValueError("original random B needs reconciliation")
    expected_archive = f"{BASE}/20418/archive"
    if i0_source_archive != expected_archive:
        raise ValueError("I0 source must be the original 20418 archive")
    if len(i0_runtime_commit) != 40 or any(c not in "0123456789abcdef" for c in i0_runtime_commit):
        raise ValueError("I0 runtime needs an exact service commit")
    for identifier in (c0_model_id, c1_model_id, c0_scene_id, i0_scene_id):
        if not isinstance(identifier, str) or not identifier or not all(
                character.isalnum() or character in "_-" for character in identifier):
            raise ValueError("model and scene IDs must be exact workbench IDs")
    for run_id in (c0_direct_run_id, c1_direct_run_id, i0_direct_run_id):
        if not run_id or not all(c.isalnum() or c in "_-" for c in run_id):
            raise ValueError("root-submitted direct run ID is required")

    spec = (
        (random_key, random_key, 20418, []),
        ("C0_final_direct_20473", "C0_final_direct_20418", 20473, []),
        ("C0_final_B_20473", "C0_final_B_20418", 20473, ["C0_final_direct_20473"]),
        ("C1_final_direct_20424", "C1_final_direct_20418", 20424, []),
        ("C1_final_B_20424", "C1_final_B_20418", 20424, ["C1_final_direct_20424"]),
        ("I0_final_direct_20474", "I0_final_direct_20418", 20474, []),
        ("I0_final_B_20474", "I0_final_B_20418", 20474, ["I0_final_direct_20474"]),
    )
    tasks = []
    for identifier, source_id, port, requires in spec:
        previous = old[source_id]
        task = {"id": identifier, "port": port, "kind": previous["kind"],
                "requires": requires, "request_id": previous["request_id"],
                "body": json.loads(json.dumps(previous["body"])),
                "expected_commit": i0_runtime_commit if identifier.startswith("I0_final_")
                else previous["expected_commit"],
                "disk_class": previous["disk_class"]}
        if identifier.startswith("C0_final_direct"):
            task["body"].update(model_id=c0_model_id, scene_id=c0_scene_id)
            task["adopted_run_id"] = c0_direct_run_id
        elif identifier.startswith("C1_final_direct"):
            task["body"].update(model_id=c1_model_id,
                                scene_id=saved["import_final_20424"]["scene_id"])
            task["adopted_run_id"] = c1_direct_run_id
        elif identifier.startswith("I0_final_direct"):
            task["body"].update(scene_id=i0_scene_id,
                                model_run_id=saved["I0_pilot_20418"]["run_id"],
                                model_source_archive=i0_source_archive)
            task["adopted_run_id"] = i0_direct_run_id
        elif identifier.endswith("_B_20473") or identifier.endswith("_B_20424") or identifier.endswith("_B_20474"):
            task["body"]["source_run_id"] = {"$task_run": requires[0]}
        elif identifier == random_key:
            task["body"]["source_run_id"] = saved["baseline_test_20418"]["run_id"]
            task["adopted_run_id"] = saved[random_key]["run_id"]
        tasks.append(task)
    new_requests = [item["request_id"] for item in tasks]
    if len(set(new_requests)) != len(new_requests):
        raise ValueError("final request IDs are not globally unique")
    for _, source_id, _, _ in spec[1:]:
        if saved.get(source_id, {}).get("run_id"):
            raise ValueError(f"{source_id} already has a legacy run; reconcile before migration")
    plan = {"schema": "s2p19.four_lane.v1", "experiment_id": "S2-P19",
            "created_utc": now(), "old_plan_sha256": sha(old_plan),
            "old_state_sha256": sha(old_state), "service_commit": source["service_commit"],
            "science_origin_commit": source["science_origin_commit"],
            "final_mothers": 40, "final_scenes": 880, "complete_B_budget": 14960,
            "completed_B_calls": 4400, "remaining_B_calls": 10560,
            "timing": "concurrent across four isolated instances; record actual device, threads and contention",
            "tasks": tasks, "completed_legacy": {key: saved[key]["run_id"] for key in (
                "V2_final_direct_20418", "V2_final_B_20418", "baseline_test_20418",
                "baseline_greedy_B_20418", "baseline_rule_B_20418")}}
    save(output, plan)
    return plan


def poll(plan_path: Path, state_path: Path, dispatch: bool) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "s2p19.four_lane.v1" or plan.get("complete_B_budget") != 14960:
        raise ValueError("four-lane plan contract differs")
    tasks = plan.get("tasks")
    if ([task.get("id") for task in tasks] != list(FINAL_IDS) or
            plan.get("completed_B_calls") != 4400 or
            plan.get("remaining_B_calls") != 10560 or
            len({task.get("request_id") for task in tasks}) != len(tasks)):
        raise ValueError("final matrix task or budget contract differs")
    for task in tasks:
        body = task["body"]
        if (task["port"] not in PORTS or task["kind"] not in ENDPOINT or
                len(task["expected_commit"]) != 40 or
                body.get("repeats", body.get("config", {}).get("repeats")) != 3):
            raise ValueError(f"{task['id']}: invalid lane, source or repeat budget")
        if task["kind"] == "refine" and body.get("config", {}).get("methods") != [
                "none", "sca", "sca_si_lp"]:
            raise ValueError(f"{task['id']}: complete B method grid differs")
    plan_sha = sha(plan_path)
    state = (json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists()
             else {"plan_sha256": plan_sha, "tasks": {}, "events": []})
    if state["plan_sha256"] != plan_sha:
        raise ValueError("plan changed after coordinator state was created")
    rows = {}
    errors = {}
    for port in PORTS:
        try:
            result = api(port, "/api/s2p19/runs")
            if not isinstance(result.get("runs"), list):
                raise ValueError("run catalog malformed")
            rows[port] = result["runs"]
        except (ConnectionError, TimeoutError, OSError) as error:
            errors[str(port)] = str(error)
    planned_requests = {task["request_id"] for task in tasks}
    observed_requests = {}
    for port, catalog in rows.items():
        for row in catalog:
            request = row.get("request_id")
            if request in planned_requests:
                observed_requests.setdefault(request, []).append((port, row.get("run_id")))
    if any(len(items) > 1 for items in observed_requests.values()):
        raise RuntimeError("four-lane request ID is duplicated across instances")
    changes = []
    for task in plan["tasks"]:
        key, port, request = task["id"], task["port"], task["request_id"]
        if port not in rows:
            continue
        matches = [row for row in rows[port] if row.get("request_id") == request]
        if len(matches) > 1:
            raise RuntimeError(f"{key}: duplicate workbench request")
        prior = state["tasks"].get(key, {})
        if matches:
            row = matches[0]
            if (prior.get("run_id") and prior["run_id"] != row.get("run_id") or
                    task.get("adopted_run_id") and task["adopted_run_id"] != row.get("run_id") or
                    row.get("git_commit") != task["expected_commit"]):
                raise RuntimeError(f"{key}: run/source identity differs")
            stage = row.get("stage")
            if stage in ("failed", "stopped"):
                raise RuntimeError(f"{key}: science job {stage}; operator attention required")
            if stage not in ACTIVE | {"completed"}:
                raise RuntimeError(f"{key}: unknown workbench stage {stage}")
            state["tasks"][key] = {"state": stage, "run_id": row["run_id"],
                                    "request_id": request, "observed_utc": now()}
            continue
        if task.get("adopted_run_id") or prior.get("run_id") or prior.get("submit_unknown"):
            raise RuntimeError(f"{key}: prior/unknown submit absent from catalog; reconcile same request")
        if any(state["tasks"].get(dep, {}).get("state") != "completed" for dep in task["requires"]):
            continue
        if not dispatch:
            changes.append({"ready": key})
            continue
        body = json.loads(json.dumps(task["body"]))
        if isinstance(body.get("source_run_id"), dict) and set(body["source_run_id"]) == {"$task_run"}:
            body["source_run_id"] = state["tasks"][body["source_run_id"]["$task_run"]]["run_id"]
        body["request_id"] = request
        state["tasks"][key] = {"state": "submitting", "request_id": request,
                                "submit_unknown": True, "at": now()}
        save(state_path, state)  # durable intent before an ambiguous POST
        try:
            result = api(port, ENDPOINT[task["kind"]], body)
        except (ConnectionError, TimeoutError, OSError):
            errors[str(port)] = f"{key}: POST outcome unknown; query same request before retry"
            continue
        if result.get("git_commit") != task["expected_commit"] or not result.get("run_id"):
            raise RuntimeError(f"{key}: POST returned wrong source or missing run ID")
        state["tasks"][key] = {"state": "submitted", "run_id": result["run_id"],
                                "request_id": request, "observed_utc": now()}
        state["events"].append({"task": key, "run_id": result["run_id"], "at": now()})
        save(state_path, state)
        changes.append({"submitted": key, "run_id": result["run_id"]})
    if dispatch:
        save(state_path, state)
    return {"observed_utc": now(), "tasks": state["tasks"], "transient_errors": errors,
            "changes": changes, "complete": all(state["tasks"].get(task["id"], {}).get("state") == "completed"
                                     for task in plan["tasks"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    migration = sub.add_parser("migrate")
    for flag in ("old-plan", "old-state", "output", "c0-model-id", "c1-model-id",
                 "c0-scene-id", "i0-scene-id", "i0-source-archive", "i0-runtime-commit",
                 "c0-direct-run-id", "c1-direct-run-id", "i0-direct-run-id"):
        migration.add_argument("--" + flag, required=True)
    for action in ("once", "run"):
        command = sub.add_parser(action)
        command.add_argument("--plan", required=True)
        command.add_argument("--state", required=True)
    args = parser.parse_args()
    if args.action == "migrate":
        result = migrate(Path(args.old_plan), Path(args.old_state), Path(args.output),
                         c0_model_id=args.c0_model_id, c1_model_id=args.c1_model_id,
                         c0_scene_id=args.c0_scene_id, i0_scene_id=args.i0_scene_id,
                         i0_source_archive=args.i0_source_archive,
                         i0_runtime_commit=args.i0_runtime_commit,
                         c0_direct_run_id=args.c0_direct_run_id,
                         c1_direct_run_id=args.c1_direct_run_id,
                         i0_direct_run_id=args.i0_direct_run_id)
        print(json.dumps({"plan": args.output, "task_count": len(result["tasks"])}))
    else:
        while True:
            report = poll(Path(args.plan), Path(args.state), args.action == "run")
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if args.action == "once" or report["complete"]:
                return
            time.sleep(180)


if __name__ == "__main__":
    main()
