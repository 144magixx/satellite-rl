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
from pathlib import Path, PurePosixPath
import re
import subprocess
import time

from . import disk_guard, ops


PLAN = ops.LOCAL / "controller" / "plan.json"
STATE = ops.LOCAL / "controller" / "state.json"
JOINT_PILOT_DECISION = ops.LOCAL / "controller" / "gates" / "joint_C0_C1_decision.json"
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
    origin = plan.get("science_origin_commit", commit)
    service_commit = plan.get("service_commit", commit)
    for label, value in (("science_origin_commit", origin), ("service_commit", service_commit)):
        if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"controller plan {label} must be an exact Git commit")
    if origin != commit:
        raise ValueError("science origin cannot be changed by a service-only handoff")
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
        adopted = task.get("adopted_only") is True
        if kind in ENDPOINTS:
            request = task.get("request_id")
            if kind not in ("import_scene", "import_model") and (
                    not isinstance(request, str) or request in seen_requests):
                raise ValueError(f"{identifier}: missing/duplicate request id")
            if request:
                seen_requests.add(request)
            if identifier == "I0_pilot_20418" and request != "s2p19-i0-train-51001":
                raise ValueError("I0 first training must reuse its single frozen request ID")
            if adopted:
                if (kind not in ("import_scene", "import_model") and
                        (not task.get("adopted_run_id") or not task.get("expected_config_sha256") or
                         not task.get("expected_commit") or not task.get("workbench_kind"))):
                    raise ValueError(f"{identifier}: adopted workbench identity incomplete")
                if "body" in task:
                    raise ValueError(f"{identifier}: adopted task cannot carry a dispatch body")
            else:
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


def _adopted_config_sha(port: int, run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in run_id):
        raise ValueError("invalid adopted run id")
    info = ops.instance(port)
    code = f'''
import hashlib,json,pathlib
path=pathlib.Path({info['archive']!r})/'outputs/workbench/jobs'/{run_id!r}/'config.json'
print(json.dumps({{'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}))
'''
    result = ops.remote_python(code, port)
    if result["exit_code"]:
        raise RuntimeError(f"{port} adopted config unavailable: " + result["stderr"][-300:])
    return json.loads(result["stdout"].splitlines()[-1])["sha256"]


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


def _verify_i0_gate_audit(audit: dict, receipts: dict) -> bool:
    expected_B = {group: receipts[f"i0_dev_B_{group}_20418"]["run_id"]
                  for group in ("greedy", "rule", "random")}
    if (audit.get("experiment_id") != "S2-P19" or
            audit.get("purpose") != "I0_development_nonlearning_gate" or
            audit.get("source_commit") != "47da239910485508f7c978461314dff92f8672ec" or
            audit.get("direct_run_id") != receipts["i0_development_gate_20418"]["run_id"] or
            audit.get("B_run_ids") != expected_B or
            audit.get("paired_scenes") != 72 or audit.get("large_N_paired_scenes") != 48):
        raise ValueError("I0 scientific gate source run identity/count differs")
    criteria = audit.get("criteria")
    expected_keys = {"source_completed_and_sha_verified", "same_72_scene_pairs_six_mothers",
                     "direct_shared_protection_and_32_probe_budget",
                     "all_final_B_resource_and_fixed_layout_checks",
                     "one_candidate_exceeds_half_pp_overall_and_one_pp_large_N"}
    if (not isinstance(criteria, dict) or set(criteria) != expected_keys or
            any(type(value) is not bool for value in criteria.values())):
        raise ValueError("I0 scientific gate criteria missing/malformed")
    performance = audit.get("performance") or {}
    if set(performance) != {"rule", "random"}:
        raise ValueError("I0 scientific gate candidate comparison missing")
    measured_pass = any(item.get("mean_delta", -1) >= .005 and
                        item.get("large_n_mean_delta", -1) >= .01
                        for item in performance.values())
    if (criteria["one_candidate_exceeds_half_pp_overall_and_one_pp_large_N"] != measured_pass or
            audit.get("passed") is not all(criteria.values())):
        raise ValueError("I0 scientific gate verdict differs from saved thresholds")
    source_files = audit.get("source_files")
    if not isinstance(source_files, list) or len(source_files) < 400:
        raise ValueError("I0 scientific gate raw source SHA list incomplete")
    allowed_ids = {audit["direct_run_id"], *expected_B.values()}
    archive = ops.instance(20418)["archive"]
    refs = []
    for item in source_files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("I0 gate raw source reference malformed")
        path, expected = item["path"], item["sha256"]
        if (not isinstance(path, str) or not isinstance(expected, str) or len(expected) != 64 or
                not any(path.startswith(f"{archive}/outputs/workbench/jobs/{run_id}/")
                        for run_id in allowed_ids)):
            raise ValueError("I0 gate raw source escapes four frozen jobs")
        refs.append(item)
    if len({item["path"] for item in refs}) != len(refs):
        raise ValueError("I0 gate raw source list contains duplicate paths")
    expected_configs = {receipts[key]["run_id"]: receipts[key]["config_sha256"]
                        for key in ("i0_development_gate_20418",
                                    "i0_dev_B_greedy_20418", "i0_dev_B_rule_20418",
                                    "i0_dev_B_random_20418")}
    by_path = {item["path"]: item["sha256"] for item in refs}
    for run_id, expected in expected_configs.items():
        path = f"{archive}/outputs/workbench/jobs/{run_id}/config.json"
        if by_path.get(path) != expected:
            raise ValueError("I0 gate source config SHA differs from adopted workbench")
    checked_total = 0
    # Paramiko invokes the remote Python body through one shell argument. Keep
    # each SHA batch well below Linux's per-argument limit without weakening
    # the full 448-file source check.
    for start in range(0, len(refs), 64):
        subset = refs[start:start + 64]
        serialized = json.dumps(subset, ensure_ascii=False)
        code = f'''
import hashlib,json,pathlib
refs=json.loads({serialized!r})
archive=pathlib.Path({archive!r}).resolve(strict=True)
allowed={sorted(allowed_ids)!r}
checked=0
for item in refs:
 path=pathlib.Path(item['path']).resolve(strict=True)
 if not any(path.is_relative_to(archive/'outputs/workbench/jobs'/run_id) for run_id in allowed):
  raise RuntimeError('I0 gate source path escaped adopted jobs')
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 if h.hexdigest()!=item['sha256']:raise RuntimeError('I0 gate raw source SHA changed')
 checked+=1
print(json.dumps({{'checked':checked}}))
'''
        result = ops.remote_python(code, 20418, timeout=120)
        if result["exit_code"] or json.loads(result["stdout"].splitlines()[-1])["checked"] != len(subset):
            raise RuntimeError("I0 gate remote raw SHA verification failed: " + result["stderr"][-500:])
        checked_total += len(subset)
    if checked_total != len(refs):
        raise RuntimeError("I0 gate SHA verification count differs")
    return bool(audit["passed"])


def _i0_logging_receipt(task: dict, plan: dict) -> dict | None:
    """Verify an independent logging-only runtime before any I0 train POST."""
    path = Path(task.get("engineering_receipt_file") or
                ops.LOCAL / "controller/gates/i0_logging_runtime_receipt.json")
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    origin = plan.get("science_origin_commit", plan["scientific_commit"])
    runtime = receipt.get("runtime_commit")
    expected_criteria = {"numeric_equivalence", "full_step_and_final_beam_logging",
                         "resume_equivalence", "workbench_wiring"}
    criteria = receipt.get("criteria")
    if (receipt.get("experiment_id") != "S2-P19" or receipt.get("arm") != "I0" or
            receipt.get("science_origin_commit") != origin or
            receipt.get("request_id") != "s2p19-i0-train-51001" or
            not isinstance(runtime, str) or len(runtime) != 40 or
            any(char not in "0123456789abcdef" for char in runtime) or
            not isinstance(criteria, dict) or set(criteria) != expected_criteria or
            any(type(value) is not bool for value in criteria.values()) or
            type(receipt.get("passed")) is not bool or
            receipt["passed"] != all(criteria.values())):
        raise ValueError("I0 logging runtime receipt identity/criteria differs")
    proof_ref = receipt.get("proof") or {}
    proof_path = Path(proof_ref.get("path", ""))
    if (not proof_path.is_absolute() or not proof_path.is_file() or
            digest(proof_path) != proof_ref.get("sha256")):
        raise ValueError("I0 logging equivalence proof path/SHA differs")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if (proof.get("science_origin_commit") != origin or
            proof.get("runtime_commit") != runtime or
            proof.get("passed") is not receipt["passed"] or
            proof.get("criteria") != criteria):
        raise ValueError("I0 logging equivalence proof content differs from receipt")
    sources = proof.get("source_files")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("I0 logging equivalence proof has no frozen source content hashes")
    allowed_commits = {origin, runtime, "62202aff0e1190492338f98359cc2f274884341c"}
    for relative, versions in sources.items():
        if (not isinstance(relative, str) or not relative.startswith("src/satellite_rl/sac_improvement/") or
                ".." in Path(relative).parts or not isinstance(versions, dict) or
                set(versions) != allowed_commits):
            raise ValueError("I0 logging source commit mapping malformed")
        for commit, expected_sha in versions.items():
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise ValueError("I0 logging source content SHA malformed")
            observed = subprocess.run(["git", "show", f"{commit}:{relative}"],
                                      cwd=Path(__file__).resolve().parents[2],
                                      capture_output=True, check=True).stdout
            if hashlib.sha256(observed).hexdigest() != expected_sha:
                raise ValueError("I0 logging frozen Git source content changed")
    evidence = proof.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("I0 logging equivalence proof lacks evidence files")
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("I0 logging evidence reference malformed")
        source = Path(item["path"])
        if not source.is_absolute() or not source.is_file() or digest(source) != item["sha256"]:
            raise ValueError("I0 logging evidence SHA changed")
    return {"passed": receipt["passed"], "runtime_commit": runtime,
            "science_origin_commit": origin, "proof_path": str(proof_path),
            "proof_sha256": proof_ref["sha256"], "receipt_path": str(path),
            "receipt_sha256": digest(path)}


def _pilot_audit_from_receipt(port: int, run_id: str, reference: dict,
                              approved: bool = False) -> dict:
    archive = ops.instance(port)["archive"]
    audit_path = reference.get("audit_path")
    audit_sha = reference.get("audit_sha256")
    if (not isinstance(audit_path, str) or not audit_path.startswith(archive + "/") or
            not isinstance(audit_sha, str) or len(audit_sha) != 64):
        raise ValueError("pilot audit receipt path/SHA is outside this workbench archive")
    code = f'''
import hashlib,json,pathlib
archive=pathlib.Path({archive!r}).resolve(strict=True)
path=pathlib.Path({audit_path!r}).resolve(strict=True)
if not path.is_relative_to(archive):raise RuntimeError('pilot audit path escaped archive')
raw=path.read_bytes()
if hashlib.sha256(raw).hexdigest()!={audit_sha!r}:raise RuntimeError('pilot audit SHA changed')
jobdir=archive/'outputs/workbench/jobs'/{run_id!r}
checkpoint=jobdir/('pilot_artifacts/checkpoint.pt' if {approved!r} else 'checkpoint.pt')
checkhash=hashlib.sha256()
with checkpoint.open('rb') as stream:
 for block in iter(lambda:stream.read(4*1024*1024),b''):checkhash.update(block)
manifest=json.loads((jobdir/'pilot_artifacts/manifest.json').read_text()) if {approved!r} else None
proof=json.loads((jobdir/'pilot_gate.json').read_text()) if {approved!r} else None
print(json.dumps({{'audit':json.loads(raw),'checkpoint_sha256':checkhash.hexdigest(),
 'manifest':manifest,'proof':proof}},ensure_ascii=False))
'''
    result = ops.remote_python(code, port, timeout=90)
    if result["exit_code"]:
        raise RuntimeError("pilot audit/checkpoint source inspection failed: " + result["stderr"][-600:])
    payload = json.loads(result["stdout"].splitlines()[-1])
    audit = payload["audit"]
    if approved:
        manifest, proof = payload["manifest"], payload["proof"]
        if (manifest.get("run_id") != run_id or proof.get("run_id") != run_id or
                proof.get("pass") is not True or
                manifest.get("checkpoint_sha256") != payload["checkpoint_sha256"] or
                proof.get("checkpoint_sha256") != payload["checkpoint_sha256"] or
                any(item.get("audit_path") != audit_path or
                    item.get("audit_sha256") != audit_sha for item in (manifest, proof))):
            raise ValueError("approved pilot artifacts/proof differ from joint audit")
    criteria = audit.get("criteria")
    if (audit.get("experiment_id") != "S2-P19" or audit.get("run_id") != run_id or
            audit.get("checkpoint_sha256") != payload["checkpoint_sha256"] or
            type(audit.get("passed")) is not bool or
            not isinstance(criteria, dict) or not criteria or
            any(type(value) is not bool for value in criteria.values()) or
            audit["passed"] != all(criteria.values())):
        raise ValueError("pilot audit identity/checkpoint/verdict differs")
    sources = audit.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise ValueError("pilot audit lacks SHA-bound raw source files")
    allowed_archives = tuple(ops.instance(allowed_port)["archive"] + "/"
                             for allowed_port in ops.SSH_PORTS)
    seen = set()
    for item in sources:
        if (not isinstance(item, dict) or set(item) != {"path", "sha256"} or
                not isinstance(item["path"], str) or
                not item["path"].startswith(allowed_archives) or
                not isinstance(item["sha256"], str) or len(item["sha256"]) != 64 or
                item["path"] in seen):
            raise ValueError("pilot audit raw source path/SHA is malformed or outside archive")
        seen.add(item["path"])
    for start in range(0, len(sources), 64):
        subset = sources[start:start + 64]
        serialized = json.dumps(subset, ensure_ascii=False)
        code = f'''
import hashlib,json,pathlib
archives=[pathlib.Path(p).resolve(strict=True) for p in {list(allowed_archives)!r}]
refs=json.loads({serialized!r})
checked=0
for item in refs:
 path=pathlib.Path(item['path']).resolve(strict=True)
 if not any(path.is_relative_to(archive) for archive in archives):
  raise RuntimeError('pilot raw source escaped both workbench archives')
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 if h.hexdigest()!=item['sha256']:raise RuntimeError('pilot raw source SHA changed')
 checked+=1
print(json.dumps({{'checked':checked}}))
'''
        checked = ops.remote_python(code, port, timeout=120)
        if (checked["exit_code"] or
                json.loads(checked["stdout"].splitlines()[-1])["checked"] != len(subset)):
            raise RuntimeError("pilot raw source SHA verification failed: " + checked["stderr"][-500:])
    return audit


def _joint_pilot_audits(plan: dict, plan_sha: str, receipts: dict,
                        dispatch: bool) -> dict | None:
    """Freeze both source-backed 100k decisions before approving either run."""
    approvals = {}
    for arm, port in (("C0", 20418), ("C1", 20424)):
        approval = next((row for row in plan["tasks"] if row["id"] == f"approve_{arm}_pilot"), None)
        if approval is None:
            raise ValueError("matched C0/C1 pilot approval task absent")
        pilot = receipts.get(approval["pilot_task_id"]) or {}
        reference_path = Path(approval["gate_receipt_file"])
        if not reference_path.is_file() or pilot.get("state") not in (
                "pilot_gate_pending", "running", "completed", "not_selected"):
            return None
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if not reference.get("audit_path") or not reference.get("audit_sha256"):
            return None
        approvals[arm] = {"port": port, "run_id": pilot.get("run_id"),
                          "state": pilot["state"], "reference": reference}
        if not approvals[arm]["run_id"]:
            return None
    saved = (json.loads(JOINT_PILOT_DECISION.read_text(encoding="utf-8"))
             if JOINT_PILOT_DECISION.is_file() else None)
    if saved is None and any(value["state"] != "pilot_gate_pending"
                             for value in approvals.values()):
        raise RuntimeError("matched pilot approval began without immutable joint decision")
    if saved is not None:
        if (saved.get("experiment_id") != "S2-P19" or
                saved.get("plan_sha256") != plan_sha or
                set(saved.get("arms", {})) != {"C0", "C1"} or
                type(saved.get("passed")) is not bool):
            raise ValueError("immutable joint pilot decision identity differs")
    audits = {}
    for arm, row in approvals.items():
        approved = row["state"] in ("running", "completed")
        audit = _pilot_audit_from_receipt(row["port"], row["run_id"],
                                          row["reference"], approved=approved)
        entry = {"run_id": row["run_id"], "audit_path": row["reference"]["audit_path"],
                 "audit_sha256": row["reference"]["audit_sha256"],
                 "checkpoint_sha256": audit["checkpoint_sha256"],
                 "passed": audit["passed"]}
        if saved is not None and saved["arms"].get(arm) != entry:
            raise ValueError("pilot evidence changed after immutable joint decision")
        audits[arm] = entry
    decision = {"experiment_id": "S2-P19", "plan_sha256": plan_sha,
                "passed": all(row["passed"] is True for row in audits.values()),
                "arms": audits}
    if saved is not None and saved != decision:
        raise ValueError("joint pilot verdict changed after approval")
    if saved is None and dispatch:
        write_json(JOINT_PILOT_DECISION, decision)
    return decision


def _resolve(value, receipts: dict):
    if isinstance(value, dict) and set(value) == {"$record_directory"}:
        spec = value["$record_directory"]
        allowed = {("prepare_dev_20418", 20418): {"development", "validation"},
                   ("prepare_final_20424", 20424): {"final"}}
        if (not isinstance(spec, dict) or
                set(spec) != {"id", "source_port", "partition", "path"} or
                type(spec["source_port"]) is not int or
                spec["partition"] not in allowed.get((spec["id"], spec["source_port"]), set()) or
                spec["path"] != ["result", "summary", "outputs", spec["partition"], "path"]):
            raise ValueError("$record_directory reference has invalid task/partition/path")
        source = receipts.get(spec["id"])
        run_id = source.get("run_id") if isinstance(source, dict) else None
        if (not source or source.get("state") != "completed" or
                not isinstance(run_id, str) or
                not re.fullmatch(r"prepare_[A-Za-z0-9_-]+", run_id)):
            raise ValueError("record directory source is not a completed preparation job")
        selected = source
        try:
            for key in spec["path"]:
                selected = selected[key]
        except (KeyError, TypeError) as error:
            raise ValueError("record directory source result is incomplete") from error
        expected = (PurePosixPath(ops.instance(spec["source_port"])["archive"])
                    / "outputs" / "workbench" / "jobs" / run_id
                    / "prepared" / spec["partition"] / "record.json")
        if not isinstance(selected, str) or selected != str(expected):
            raise ValueError("record directory source escaped its frozen preparation result")
        return str(expected.parent)
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
    service_path = ops.LOCAL / "preflight" / f"service_{commit[:4]}" / "deployment_receipt.json"
    launch_path = (service_path if service_path.is_file() else
                   ops.LOCAL / "preflight" / f"workbench_{commit[:12]}.json")
    if not launch_path.is_file():
        raise RuntimeError("missing saved S2-P19 workbench launch receipt")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    service_only = launch_path == service_path
    if ((launch.get("service_commit") if service_only else launch.get("scientific_commit")) != commit or
            str(port) not in launch.get("instances", {})):
        raise RuntimeError("workbench launch receipt is for another source/instance")
    pinned = launch["instances"][str(port)]
    repo = pinned["service_repo"] if service_only else info["repo"]
    pid = pinned["new_server_pid"] if service_only else pinned["pid"]
    ticks = pinned["new_server_start_ticks"] if service_only else pinned["start_ticks"]
    package = pinned["service_package"] if service_only else repo + "/src/satellite_rl/__init__.py"
    if not repo.startswith(info["root"] + "/") or not package.startswith(repo + "/"):
        raise RuntimeError("service launch receipt points outside this instance")
    code = f'''
import json,os,pathlib,subprocess
repo=pathlib.Path({repo!r})
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
pid={pid};proc=pathlib.Path('/proc')/str(pid)
birth=None;cmdline='';cwd='';pythonpath='';package_ok=False
if proc.is_dir():
 stat=(proc/'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
 cmdline=(proc/'cmdline').read_bytes().replace(b'\\x00',b' ').decode('utf-8','replace')
 cwd=os.readlink(proc/'cwd')
 entries=(proc/'environ').read_bytes().split(b'\\x00')
 pythonpath=next((entry.decode('utf-8','replace')[11:] for entry in entries if entry.startswith(b'PYTHONPATH=')),'')
 package_ok=pathlib.Path({package!r}).is_file()
print(json.dumps({{'head':head,'dirty':dirty,'server_pid':pid,'start_ticks':birth,
 'cmdline_matches':{info['archive']!r} in cmdline and 'satellite_rl.workbench.server' in cmdline and '--port {info['api_port']}' in cmdline,
 'cwd_matches':cwd==str(repo),'pythonpath_matches':str(repo/'src') in pythonpath.split(':'),
 'package_matches':package_ok}}))
'''
    observed = ops.remote_python(code, port)
    if observed["exit_code"]:
        raise RuntimeError(f"{port} frozen source identity unavailable")
    identity = json.loads(observed["stdout"].splitlines()[-1])
    if (identity["head"] != commit or identity["dirty"] or
            identity["start_ticks"] != ticks or
            not all(identity[key] for key in ("cmdline_matches", "cwd_matches",
                                                "pythonpath_matches", "package_matches"))):
        raise RuntimeError(f"{port} remote checkout differs from frozen commit")
    return identity


def _shared_root_bytes(port: int, snapshots: dict[int, dict]) -> int:
    identity = snapshots[port]["mount_identity"]
    roots = {item["root"]: item["s2p19_root_bytes"] for item in snapshots.values()
             if item["mount_identity"] == identity}
    return sum(roots.values())


def _disk_emergency(port: int, snapshot: dict, shared_root_bytes: int,
                    runs: list[dict], dispatch: bool) -> dict | None:
    physical_low = snapshot["archive_free_bytes"] < disk_guard.RESERVE_BYTES
    if not physical_low:
        return None
    incident = {"port": port, "observed_utc": snapshot["observed_utc"],
                "archive_free_bytes": snapshot["archive_free_bytes"],
                "mount_identity": snapshot["mount_identity"],
                "shared_root_bytes": shared_root_bytes,
                "reserve_bytes": disk_guard.RESERVE_BYTES,
                "experiment_engineering_budget_bytes": disk_guard.EXPERIMENT_ENGINEERING_BUDGET_BYTES,
                "physical_low": physical_low,
                "active_s2p19_runs": [row["run_id"] for row in runs if row.get("stage") in ACTIVE]}
    if dispatch:
        incident["stop_receipts"] = {}
        for run_id in incident["active_s2p19_runs"]:
            incident["stop_receipts"][run_id] = ops.api(port, f"/api/runs/{run_id}/stop", {}, "POST")
    return incident


def _task_status(task: dict, receipts: dict, runs: dict[int, list[dict]],
                 scenes: dict[int, list[dict]], models: dict[int, list[dict]],
                 plan: dict | None = None) -> dict:
    task_id = task["id"]
    saved = receipts.get(task_id)
    if task["kind"] == "approve_pilot":
        if saved and saved.get("state") in ("completed", "gate_failed"):
            return saved
        pilot = receipts.get(task["pilot_task_id"]) or {}
        run_id = pilot.get("run_id")
        if run_id and pilot.get("state") in ("running", "completed"):
            proof = _pilot_gate_proof(task["port"], run_id)
            if proof is not None:
                step = proof.get("checkpoint_step")
                if (proof.get("pass") is not True or proof.get("run_id") != run_id or
                        type(step) is not int or not 0 < step <= 100000 or
                        (task_id in ("approve_C0_pilot", "approve_C1_pilot") and step != 100000) or
                        not isinstance(proof.get("checkpoint_sha256"), str) or
                        len(proof["checkpoint_sha256"]) != 64 or
                        not isinstance(proof.get("audit_path"), str) or
                        not isinstance(proof.get("audit_sha256"), str) or
                        len(proof["audit_sha256"]) != 64 or
                        not isinstance(proof.get("approved_at"), str)):
                    raise RuntimeError("saved pilot gate proof identity/step/source fields differ")
                reference = {"audit_path": proof["audit_path"],
                             "audit_sha256": proof["audit_sha256"]}
                audit = _pilot_audit_from_receipt(task["port"], run_id, reference,
                                                  approved=True)
                if (audit.get("passed") is not True or audit.get("run_id") != run_id or
                        audit.get("checkpoint_sha256") != proof["checkpoint_sha256"] or
                        audit.get("checkpoint_step") != step or
                        not audit.get("source_files")):
                    raise RuntimeError("saved pilot approval lacks matched source-backed audit")
                return {"state": "completed", "approved_run_id": run_id,
                        "audit_path": proof["audit_path"],
                        "audit_sha256": proof["audit_sha256"],
                        "checkpoint_sha256": proof["checkpoint_sha256"],
                        "checkpoint_step": step,
                        "recovered_from_workbench_proof": True}
        return saved or {"state": "waiting"}
    if saved and saved.get("state") in ("gate_failed", "not_selected"):
        return saved
    if task["kind"] == "milestone":
        return saved or {"state": "waiting"}
    if task["kind"] == "import_scene":
        if saved and saved.get("scene_id"):
            match = [scene for scene in scenes[task["port"]] if scene.get("id") == saved["scene_id"]]
            valid = (len(match) == 1 and match[0].get("count") == task.get("expected_count")
                     and ((match[0].get("source") or {}).get("reimport") or {}).get("record_id")
                     == task.get("source_record_id"))
            return {**saved, "state": "completed"} if valid else {**saved, "state": "attention_required"}
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
        if task.get("adopted_only"):
            return {**(saved or {}), "state": "attention_required",
                    "reason": "adopted request missing; never recreate it"}
        if saved and saved.get("run_id"):
            return {**saved, "state": "attention_required", "reason": "receipt has no matching live catalog run"}
        return {"state": "waiting"}
    run = matches[0]
    if task_id == "I0_pilot_20418" and plan is not None:
        logging = _i0_logging_receipt(task, plan)
        if logging is None or logging["passed"] is not True or run.get("git_commit") != logging["runtime_commit"]:
            raise RuntimeError("I0 training exists without exact source-backed logging runtime")
    elif (not task.get("adopted_only") and task.get("expected_commit") and
          run.get("git_commit") != task["expected_commit"]):
        raise RuntimeError(f"{task_id}: existing task commit differs from frozen task contract")
    if task.get("adopted_only"):
        if (run.get("run_id") != task["adopted_run_id"] or
                run.get("git_commit") != task["expected_commit"] or
                run.get("kind") != task["workbench_kind"] or
                _adopted_config_sha(task["port"], run["run_id"])
                != task["expected_config_sha256"]):
            raise RuntimeError(f"{task_id}: adopted run identity/config changed; never resubmit")
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
    if state.get("science_origin_commit", state["scientific_commit"]) != plan.get(
            "science_origin_commit", plan["scientific_commit"]):
        raise RuntimeError("persistent controller science origin changed")
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
    identities = {port: _service_identity(port, plan.get("service_commit", plan["scientific_commit"]))
                  for port in ops.SSH_PORTS}
    runs = {port: _run_rows(port) for port in ops.SSH_PORTS}
    scenes = {port: _scene_rows(port) for port in ops.SSH_PORTS}
    models = {port: _model_rows(port) for port in ops.SSH_PORTS}
    disks = {port: disk_guard.sample(port) for port in ops.SSH_PORTS}
    emergencies = {port: _disk_emergency(port, disks[port], _shared_root_bytes(port, disks),
                                         runs[port], dispatch)
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
        receipts[task["id"]] = _task_status(task, receipts, runs, scenes, models, plan)
    state["tasks"] = receipts
    waiting = []
    for task in plan["tasks"]:
        task_id = task["id"]
        observed = receipts[task_id]
        if observed["state"] in ("completed", "running", "not_selected"):
            continue
        if observed["state"] in ("attention_required", "pilot_gate_pending", "gate_failed"):
            waiting.append({"id": task_id, **observed})
            continue
        if (task_id in ("I0_pilot_20418", "approve_I0_pilot") and
                any(receipts.get(dep, {}).get("state") in ("gate_failed", "not_selected")
                    for dep in task.get("requires", []))):
            receipts[task_id] = {"state": "not_selected", "reason": "I0_development_gate_failed",
                                 "at": now()}
            if dispatch:
                state["events"].append({"event": "not_selected", "task_id": task_id, "at": now()})
                write_json(STATE, state)
            continue
        def ready_dependency(dep):
            observed_state = receipts[dep]["state"]
            if task["kind"] == "approve_pilot" and dep == task.get("pilot_task_id"):
                return observed_state == "pilot_gate_pending"
            return observed_state == "completed"
        if not all(ready_dependency(dep) for dep in task.get("requires", [])):
            continue
        if task.get("adopted_only"):
            waiting.append({"id": task_id, "state": "adopted_only_attention",
                            "reason": "existing task absent or incomplete; no dispatch body exists"})
            continue
        if task_id == "I0_pilot_20418":
            logging = _i0_logging_receipt(task, plan)
            if logging is None:
                waiting.append({"id": task_id, "state": "waiting_i0_logging_equivalence_receipt"})
                continue
            if not logging["passed"]:
                if dispatch:
                    receipts[task_id] = {"state": "not_selected",
                                         "reason": "I0_logging_equivalence_failed",
                                         "logging_proof_path": logging["proof_path"], "at": now()}
                    state["events"].append({"event": "I0_logging_gate_failed",
                                            "task_id": task_id, "at": now()})
                    write_json(STATE, state)
                else:
                    waiting.append({"id": task_id, "state": "I0_logging_gate_failed_preview"})
                continue
            service_commit = plan.get("service_commit", plan["scientific_commit"])
            if service_commit != logging["runtime_commit"]:
                waiting.append({"id": task_id, "state": "waiting_exact_i0_runtime_service",
                                "required_runtime_commit": logging["runtime_commit"]})
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
            audit_passed = True
            if task_id == "i0_scientific_gate":
                audit_passed = _verify_i0_gate_audit(
                    json.loads(Path(receipt).read_text(encoding="utf-8")), receipts)
            if dispatch:
                receipts[task_id] = {"state": "completed" if audit_passed else "gate_failed",
                                     "phase": task["phase"],
                                     "receipt_path": receipt, "receipt_sha256": receipt_sha,
                                     "at": now()}
                state["events"].append({"event": "milestone" if audit_passed else "gate_failed",
                                        "task_id": task_id, "at": now()})
                write_json(STATE, state)
            else:
                waiting.append({"id": task_id,
                                "state": "ready_milestone" if audit_passed else "gate_failed_preview"})
            continue
        if task["kind"] == "approve_pilot":
            pilot = receipts[task["pilot_task_id"]]
            if pilot.get("state") != "pilot_gate_pending":
                waiting.append({"id": task_id, "state": "waiting_pilot_gate"})
                continue
            if task_id in ("approve_C0_pilot", "approve_C1_pilot"):
                joint = _joint_pilot_audits(plan, plan_sha, receipts, dispatch)
                if joint is None:
                    waiting.append({"id": task_id, "state": "waiting_joint_C0_C1_audit"})
                    continue
                audit_passed = joint["passed"]
                arm = "C0" if task_id == "approve_C0_pilot" else "C1"
                reference = joint["arms"][arm]
            else:
                reference_file = task.get("gate_receipt_file")
                reference = (json.loads(Path(reference_file).read_text(encoding="utf-8"))
                             if reference_file and Path(reference_file).is_file() else {})
                if not reference.get("audit_path") or not reference.get("audit_sha256"):
                    waiting.append({"id": task_id, "state": "waiting_scientific_audit"})
                    continue
                audit_passed = _pilot_audit_from_receipt(
                    task["port"], pilot["run_id"], reference)["passed"]
            if audit_passed is not True:
                if dispatch:
                    receipts[task["pilot_task_id"]] = {**pilot, "state": "not_selected",
                                                         "workbench_stage": "stopped",
                                                         "reason": "matched_pilot_scientific_audit_failed"}
                    receipts[task_id] = {"state": "gate_failed", "run_id": pilot["run_id"],
                                         "audit_path": reference["audit_path"],
                                         "audit_sha256": reference["audit_sha256"], "at": now()}
                    state["events"].append({"event": "pilot_gate_failed", "task_id": task_id,
                                            "run_id": pilot["run_id"], "at": now()})
                    write_json(STATE, state)
                else:
                    waiting.append({"id": task_id, "state": "pilot_gate_failed_preview"})
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
                if task_id in ("approve_C0_pilot", "approve_C1_pilot"):
                    break  # Reobserve the first resumed worker before approving its pair.
            else:
                waiting.append({"id": task_id, "state": "ready_gate_approval"})
            continue
        active_reservation = sum(
            disk_guard.PREDICTED_BYTES[row["disk_class"]]
            for row in plan["tasks"] if row.get("disk_class") in disk_guard.PREDICTED_BYTES
            and row["id"] != task_id and receipts[row["id"]].get("state") == "running"
            and disks[row["port"]]["mount_identity"] == disks[task["port"]]["mount_identity"])
        capacity = disk_guard.gate(disks[task["port"]], task["disk_class"],
                                   active_reservation, _shared_root_bytes(task["port"], disks))
        if not capacity["admit"]:
            waiting.append({"id": task_id, "state": "disk_capacity_hold", "capacity": capacity})
            continue
        if not dispatch:
            waiting.append({"id": task_id, "state": "ready_to_submit", "capacity": capacity})
            continue
        body = _resolve(task["body"], receipts)
        expected_commit = (logging["runtime_commit"] if task_id == "I0_pilot_20418" else
                           task.get("expected_commit", plan.get("service_commit", plan["scientific_commit"])))
        if task["kind"] not in ("import_scene", "import_model") and (
                plan.get("service_commit", plan["scientific_commit"]) != expected_commit):
            raise RuntimeError(f"{task_id}: service commit differs from task runtime contract")
        if task["kind"] not in ("import_scene", "import_model"):
            if body.get("request_id") not in (None, task["request_id"]):
                raise ValueError(f"{task_id}: API body request id mismatch")
            body["request_id"] = task["request_id"]
        result = ops.api(task["port"], ENDPOINTS[task["kind"]], body, "POST", timeout=180)
        if task["kind"] not in ("import_scene", "import_model") and result.get("git_commit") != expected_commit:
            raise RuntimeError(f"{task_id}: created job has wrong runtime commit")
        entry = {"state": "running", "phase": task["phase"], "submitted_at": now(),
                 "request_id": task.get("request_id"), "capacity": capacity,
                 "expected_commit": expected_commit, "response": result}
        if task_id == "I0_pilot_20418":
            entry.update(science_origin_commit=logging["science_origin_commit"],
                         runtime_commit=logging["runtime_commit"],
                         logging_equivalence_proof_sha256=logging["proof_sha256"])
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
               "science_origin_commit": plan.get("science_origin_commit", plan["scientific_commit"]),
               "service_commit": plan.get("service_commit", plan["scientific_commit"]),
               "completed": sum(item.get("state") == "completed" for item in receipts.values()),
               "terminal_count": sum(item.get("state") in ("completed", "gate_failed", "not_selected")
                                     for item in receipts.values()),
               "total": len(plan["tasks"]), "waiting": waiting,
               "tasks": {key: {k: value.get(k) for k in ("state", "run_id", "workbench_stage", "stop_reason")}
                         for key, value in receipts.items()},
               "archive_free_bytes": {str(port): disks[port]["archive_free_bytes"] for port in ops.SSH_PORTS},
               "mount_identity": {str(port): disks[port]["mount_identity"] for port in ops.SSH_PORTS},
               "shared_root_bytes": {str(port): _shared_root_bytes(port, disks) for port in ops.SSH_PORTS}}
    return summary


def maybe_advance_common_evaluation() -> dict | None:
    plan, _ = load_plan()
    if any(row["id"] == "V2_final_direct_20418" for row in plan["tasks"]):
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
        # Validation uses only the frozen 72 + 432 complete-B budget and
        # saved fixed-node cases; this hook never approves a training gate.
        if report["status"] != "disk_attention":
            from .pilot_jobs import pass_once
            validation = pass_once(True)
            report["validation_jobs"] = {
                "observed_utc": validation["observed_utc"],
                "new_submissions": [{"key": item["key"], "port": item["port"],
                                     "request_id": item["request_id"],
                                     "run_id": item["response"]["run_id"]}
                                    for item in validation["new_submissions"]]}
        advancement = maybe_advance_common_evaluation()
        if advancement is not None:
            report["plan_advance"] = advancement
        write_json(ops.LOCAL / "controller" / "latest_observation.json", report)
        if report["status"] == "disk_attention":
            return
        plan, _ = load_plan()
        if report["terminal_count"] == report["total"] and plan["terminal"]:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
