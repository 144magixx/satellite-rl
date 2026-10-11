"""Narrow, auditable execution-source repair for the authorized S1-ND12 faults.

This is an explicit workbench action, never an implicit ordinary resume. The
original configuration and scientific evidence remain immutable history.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from satellite_rl.training.run_monitor import atomic_json
from .storage import digest, inside, now, read_json

FORMAT = "s1_nd12_source_repair_v1"
REPAIR_ID = "nd12_native_deadline_teacher_reference_v1"
ORIGIN_COMMIT = "105d5f57ec241cce0c484b95f6341d6dc418b3fe"
ALLOWED_RUNS = {
    "prepare_nondrl_20261012_045319_9ffedc65": ("01", "teacher"),
    "prepare_nondrl_20261012_045228_1d87b8e0": ("02", "teacher_probe"),
    "prepare_nondrl_20261012_045232_8dca8863": ("03", "teacher_probe"),
    **{"evaluate_nondrl_20261012_" + suffix: ("10", "evaluate") for suffix in
       ("034600_ac2268e7", "034601_b3d48850", "034601_b96222ab", "034602_5f3a61a6")},
}
FUTURE_ROLES = {(r, k, "teacher" if k == "prepare" else None)
                for r in ("01", "02", "03") for k in ("prepare", "train", "evaluate")} | {
                    ("10", "evaluate", None)}
# Every scientific diff must be enumerated with both file hashes in the frozen
# contract. This second boundary excludes changes to networks, budgets or data.
DIFF_ALLOWLIST = {
    "src/satellite_rl/coverage/nondrl/native_solver.py",
    "src/satellite_rl/coverage/nondrl/classical.py",
    "src/satellite_rl/coverage/nondrl/source_lineage.py",
    "src/satellite_rl/coverage/nondrl/core.py",
    "src/satellite_rl/coverage/nondrl/runner.py",
    "src/satellite_rl/coverage/nondrl/teachers.py",
    "src/satellite_rl/coverage/nondrl/learning.py",
    "src/satellite_rl/coverage/nondrl/mechanism_audit.py",
    "src/satellite_rl/workbench/nondrl_source_repair.py",
    "src/satellite_rl/workbench/coverage_nondrl.py",
    "src/satellite_rl/workbench/storage.py",
    "src/satellite_rl/workbench/runtime.py",
}


def _sha(value, length=64):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value):
        raise ValueError("invalid source repair hash")
    return value


def _pin(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or digest(path) != _sha(expected):
        raise ValueError("source repair artifact SHA mismatch: " + str(path))
    return path


def _git(store, *args, source_root=None):
    return subprocess.run(["git", *args], cwd=source_root or store.source_root,
                          capture_output=True, check=True).stdout


def _role(item):
    if not isinstance(item, dict):
        return (None, None, None)
    return (item.get("route"), item.get("kind"), item.get("preparation_mode"))


def _job_role(job):
    return (job.get("variant"), job.get("kind"),
            job.get("preparation_mode") if job.get("kind") == "prepare" else None)


def validate_contract(store, path, sha256):
    """Verify global raw-byte contract, exact git diff and validation receipts."""
    path = _pin(path, sha256)
    contract = read_json(path, {})
    if (contract.get("format") != FORMAT or contract.get("repair_id") != REPAIR_ID or
            contract.get("origin_commit") != ORIGIN_COMMIT):
        raise ValueError("unknown ND12 source repair contract")
    execution = _sha(contract.get("execution_commit"), 40)
    if execution == ORIGIN_COMMIT:
        raise ValueError("source repair requires an honest new execution commit")
    repo = inside(contract.get("execution_source_repo", ""), store.archive_root.parent.parent)
    if (not repo.is_dir() or _git(store, "rev-parse", "HEAD", source_root=repo).decode().strip() != execution or
            _git(store, "status", "--porcelain", "--", "src", "data", "pyproject.toml", source_root=repo).strip()):
        raise ValueError("execution source repo must be an isolated clean exact checkout")
    ids = contract.get("allowed_run_ids")
    roles = contract.get("allowed_future_roles")
    if (not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) - set(ALLOWED_RUNS) or
            not isinstance(roles, list) or any(_role(r) not in FUTURE_ROLES for r in roles) or
            set(contract.get("runs", {})) != set(ids)):
        raise ValueError("source repair run/role outside the authorized scope")
    files = contract.get("diff_files")
    actual = set(_git(store, "diff", "--name-only", ORIGIN_COMMIT, execution, "--",
                      "src", "data", "pyproject.toml", source_root=repo).decode().splitlines())
    if (not isinstance(files, list) or len({f.get("path") for f in files}) != len(files) or
            {f.get("path") for f in files} != actual or not actual or actual - DIFF_ALLOWLIST):
        raise ValueError("source repair scientific diff differs from the allowlist")
    for file in files:
        for revision, key in ((ORIGIN_COMMIT, "origin_sha256"), (execution, "execution_sha256")):
            result = subprocess.run(["git", "show", revision + ":" + file["path"]],
                                    cwd=repo, capture_output=True)
            expected = file.get(key)
            if result.returncode:
                if expected is not None or key == "execution_sha256":
                    raise ValueError("source repair file missing from execution tree")
            elif hashlib.sha256(result.stdout).hexdigest() != _sha(expected):
                raise ValueError("source repair git file hash differs")
    receipts = contract.get("validation_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("source repair requires frozen validation receipts")
    for receipt in receipts:
        proof = read_json(_pin(receipt["path"], receipt["sha256"]), {})
        if proof.get("passed") is not True and proof.get("status") != "passed":
            raise ValueError("source repair validation has not passed")
    return contract, path


def freeze_execution(store, contract):
    """Archive an exact commit without resetting the service's source checkout."""
    execution = contract["execution_commit"]
    destination = store.root / "source_snapshots" / execution
    if not (destination / "source_manifest.json").is_file():
        archive = _git(store, "archive", "--format=zip", execution, "src", "data", "pyproject.toml",
                       source_root=contract["execution_source_repo"])
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            for item in bundle.infolist():
                inside(destination / item.filename, destination)
            bundle.extractall(destination)
        atomic_json(destination / "source_manifest.json", {
            "git_commit": execution, "created_at": now(), "source_root": str(store.source_root),
            "format_version": 1, "files": {
                str(p.relative_to(destination)).replace("\\", "/"): digest(p)
                for p in destination.rglob("*") if p.is_file()},
        })
    verify_snapshot(destination, execution)
    return destination, execution


def verify_snapshot(snapshot, execution):
    snapshot = Path(snapshot)
    manifest = read_json(snapshot / "source_manifest.json", {})
    if manifest.get("git_commit") != execution or not manifest.get("files"):
        raise ValueError("execution snapshot manifest differs")
    actual = {str(p.relative_to(snapshot)).replace("\\", "/") for p in snapshot.rglob("*")
              if p.is_file() and p.name != "source_manifest.json" and "__pycache__" not in p.parts}
    if actual != set(manifest["files"]):
        raise ValueError("execution snapshot file set differs")
    for name, expected in manifest["files"].items():
        _pin(inside(snapshot / name, snapshot), expected)


def select_source(store, extra, kind):
    reference = extra.get("execution_source")
    if not reference:
        return store.freeze_source()
    contract, _ = validate_contract(store, reference["contract_path"], reference["contract_sha256"])
    if _job_role({**extra, "kind": kind}) not in {_role(r) for r in contract["allowed_future_roles"]}:
        raise ValueError("future execution source role is not authorized")
    return freeze_execution(store, contract)


def bind_new_job(store, job, directory):
    """Bind the actual generated run ID and raw frozen config to a future policy."""
    reference = job.get("execution_source")
    if not reference:
        return
    contract, path = validate_contract(store, reference["contract_path"], reference["contract_sha256"])
    target = directory / "source_lineage" / "contract.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    job["execution_source"] = {**reference, "contract_path": str(target), "source_path": str(path)}
    item = {"format": FORMAT, "repair_id": REPAIR_ID, "run_id": job["run_id"],
            "origin_commit": ORIGIN_COMMIT, "execution_commit": contract["execution_commit"],
            "config_path": str(directory / "config.json"), "config_sha256": digest(directory / "config.json"),
            "origin_provenance": None, "artifacts": [{"path": str(directory / "config.json"),
                "relative_path": "config.json", "sha256": digest(directory / "config.json"), "kind": "config"}],
            "reason": "authorized repaired execution source",
            "comparison_scope": "new_execution_task", "contract_sha256": reference["contract_sha256"]}
    job.update(source_repair_contract=item, origin_commit=ORIGIN_COMMIT,
               execution_commit=contract["execution_commit"], source_history=[{
                   "origin_commit": ORIGIN_COMMIT, "execution_commit": contract["execution_commit"],
                   "repair_id": REPAIR_ID, "contract_sha256": reference["contract_sha256"],
                   "source_manifest_sha256": digest(Path(job["source_root"]) / "source_manifest.json"),
                   "at": now(), "comparison_scope": "new_execution_task"}])


def validate_job_lineage(job, job_directory, contract_path=None, *, verify_source_snapshot=True,
                         read_artifact=None):
    """Shared client/archive validator. Old 105 remains valid without repair."""
    if job.get("git_commit") == ORIGIN_COMMIT and not job.get("source_repair_contract"):
        return
    item = job.get("source_repair_contract", {})
    directory = Path(job_directory)
    def bytes_pin(relative, expected):
        if read_artifact is None:
            return _pin(directory / relative, expected).read_bytes()
        payload = read_artifact(relative)
        if hashlib.sha256(payload).hexdigest() != _sha(expected):
            raise ValueError("archived source repair artifact SHA differs: " + relative)
        return payload
    if read_artifact is not None:
        global_contract = json.loads(bytes_pin("source_lineage/contract.json", item.get("contract_sha256")))
    else:
        path = contract_path or job.get("execution_source", {}).get("contract_path")
        global_contract = read_json(_pin(path, item.get("contract_sha256")), {})
    if (job.get("experiment_id") != "S1-ND12" or item.get("format") != FORMAT or
            item.get("repair_id") != REPAIR_ID or item.get("origin_commit") != ORIGIN_COMMIT or
            item.get("run_id") != job.get("run_id") or
            item.get("execution_commit") != job.get("git_commit") or
            job.get("execution_commit") != job.get("git_commit") or
            global_contract.get("execution_commit") != job.get("git_commit") or
            global_contract.get("origin_commit") != ORIGIN_COMMIT or
            global_contract.get("format") != FORMAT or global_contract.get("repair_id") != REPAIR_ID):
        raise ValueError("mixed ND12 execution source lineage")
    bytes_pin("config.json", item.get("config_sha256"))
    if read_artifact is None and Path(item.get("config_path", "")).resolve() != (directory / "config.json").resolve():
        raise ValueError("inline repair config path differs from actual frozen config")
    future = item.get("comparison_scope") == "new_execution_task"
    if future:
        if (_job_role(job) not in FUTURE_ROLES or
                _job_role(job) not in {_role(r) for r in global_contract.get("allowed_future_roles", [])}):
            raise ValueError("future task source repair role differs")
    else:
        if (job.get("run_id") not in ALLOWED_RUNS or
                job.get("run_id") not in global_contract.get("allowed_run_ids", [])):
            raise ValueError("repair task is not allowlisted")
        original = global_contract.get("runs", {}).get(job["run_id"], {})
        for key in ("config_sha256", "origin_provenance", "artifacts", "reason", "comparison_scope"):
            if item.get(key) != original.get(key):
                raise ValueError("inline repair contract differs from global pins")
        for artifact in original.get("artifacts", []):
            bytes_pin("source_lineage/origin/artifacts/" + artifact["relative_path"], artifact["sha256"])
        for name in ("job", "status", "config", "launch"):
            bytes_pin("source_lineage/origin/" + name + ".json", original[name + "_sha256"])
        bytes_pin("source_lineage/origin/source_manifest.json", original["origin_source_manifest_sha256"])
    entries = job.get("source_history", [])
    if not entries or any(entries[-1].get(k) != item.get(k) for k in
                          ("origin_commit", "execution_commit", "repair_id", "contract_sha256")):
        raise ValueError("source history is missing or differs")
    if verify_source_snapshot:
        verify_snapshot(job["source_root"], job["git_commit"])
        _pin(Path(job["source_root"]) / "source_manifest.json", entries[-1]["source_manifest_sha256"])


def pin_run(store, run_id):
    """Read-only generation of per-run raw SHA pins; caller freezes the global contract."""
    if run_id not in ALLOWED_RUNS:
        raise ValueError("only the seven authorized existing runs may be pinned")
    directory = store.job_dir(run_id)
    job, status = read_json(directory / "job.json", {}), read_json(directory / "status.json", {})
    launch = read_json(directory / "launch.json", {})
    if job.get("git_commit") != ORIGIN_COMMIT:
        raise ValueError("pinning requires the original scientific source")
    artifacts, provenance = [], None
    for path in sorted(directory.rglob("*")):
        if (not path.is_file() or "source_lineage" in path.parts or
                path.name in ("job.json", "status.json", "launch.json")):
            continue
        # Pin full raw manifests, committed gzip shards, model/checkpoint/log and
        # dependencies. Their bytes are preserved once before source repair.
        relative = str(path.relative_to(directory)).replace("\\", "/")
        kind = "config" if relative == "config.json" else "teacher_manifest" if (
            path.name == "manifest.json" and job.get("kind") == "prepare") else "row_manifest" if (
            path.name == "manifest.json") else "teacher_wrapper" if path.name == "teacher_manifest.json" else "artifact"
        artifacts.append({"path": str(path), "relative_path": relative, "sha256": digest(path), "kind": kind})
        record = read_json(path, {}) if path.suffix == ".json" else {}
        if isinstance(record, dict) and isinstance(record.get("provenance"), dict) and (
                record["provenance"].get("run_id") == run_id):
            if provenance is None:
                provenance = record["provenance"]
            elif provenance != record["provenance"]:
                raise ValueError("original manifest provenance is already mixed")
    if provenance is None:
        raise ValueError("original raw manifest provenance is absent")
    return {**{name + "_sha256": digest(directory / (name + ".json"))
               for name in ("job", "status", "config", "launch")},
            "required_stage": status.get("stage"), "original_pid": launch.get("pid"),
            "original_process_start_time": launch.get("process_start_time"),
            "origin_provenance": provenance, "artifacts": artifacts,
            "origin_source_manifest_sha256": digest(Path(job["source_root"]) / "source_manifest.json"),
            "reason": "bounded native deadline and complete I0 reference source repair",
            "comparison_scope": "reference_only" if ALLOWED_RUNS[run_id][1] == "teacher_probe" else
                                "preserve_committed_evidence_continue"}


def pin_diff(store, execution_source_repo, execution_commit):
    """Read-only exact scientific file hashes, suitable for the global contract."""
    names = _git(store, "diff", "--name-only", ORIGIN_COMMIT, execution_commit, "--",
                 "src", "data", "pyproject.toml", source_root=execution_source_repo).decode().splitlines()
    result = []
    for name in names:
        item = {"path": name}
        for rev, key in ((ORIGIN_COMMIT, "origin_sha256"), (execution_commit, "execution_sha256")):
            output = subprocess.run(["git", "show", rev + ":" + name], cwd=execution_source_repo, capture_output=True)
            item[key] = None if output.returncode else hashlib.sha256(output.stdout).hexdigest()
        result.append(item)
    return result


def validate_execution_job(store, job):
    if job.get("source_repair_pending"):
        raise ValueError("source repair activation is not yet committed")
    validate_job_lineage(job, store.job_dir(job["run_id"]))


def repair_job(scheduler, run_id, body):
    """Preserve the old records, then queue one explicitly authorized same-ID repair."""
    from .runtime import pid_alive
    if set(body) != {"contract_path", "contract_sha256"}:
        raise ValueError("source repair accepts only an immutable contract reference")
    with scheduler.lock, scheduler.store.lock:
        store = scheduler.store
        contract, path = validate_contract(store, body["contract_path"], body["contract_sha256"])
        directory = store.job_dir(run_id)
        job, status = read_json(directory / "job.json", {}), read_json(directory / "status.json", {})
        if job.get("source_repair_contract"):
            validate_job_lineage(job, directory)
            if job["source_repair_contract"]["contract_sha256"] != body["contract_sha256"]:
                raise ValueError("same task cannot change its repair contract")
            if job.get("source_repair_pending"):
                status.update(stage="queued", pid=None, scientific_stop=False, error=None,
                              source_repair=REPAIR_ID, execution_commit=job["execution_commit"], heartbeat_at=now(),
                              queue_reason="显式S1-ND12来源修复事务继续")
                atomic_json(directory / "status.json", status)
                job["source_repair_pending"] = False
                atomic_json(directory / "job.json", job)
            return status
        item = contract.get("runs", {}).get(run_id)
        if not item or run_id not in ALLOWED_RUNS:
            raise ValueError("run ID not authorized for source repair")
        route, mode = ALLOWED_RUNS[run_id]
        expected_kind = "evaluate" if mode == "evaluate" else "prepare"
        if (job.get("experiment_id") != "S1-ND12" or job.get("git_commit") != ORIGIN_COMMIT or
                job.get("variant") != route or job.get("kind") != expected_kind or
                (expected_kind == "prepare" and job.get("preparation_mode") != mode) or
                status.get("scientific_stop") or status.get("stage") != item.get("required_stage") or
                status.get("stage") not in (("completed",) if mode == "teacher_probe" else
                                             ("failed", "interrupted", "stopped"))):
            raise ValueError("original source/task/terminal state differs from repair scope")
        launch = read_json(directory / "launch.json", {})
        if (not launch.get("pid") or not isinstance(launch.get("process_start_time"), (int, float)) or
                launch["pid"] != item.get("original_pid") or
                launch["process_start_time"] != item.get("original_process_start_time") or
                pid_alive(launch["pid"], launch["process_start_time"])):
            raise ValueError("original worker birth must be pinned and exited before repair")
        for name in ("job", "status", "config", "launch"):
            _pin(directory / (name + ".json"), item.get(name + "_sha256"))
        verify_snapshot(job["source_root"], ORIGIN_COMMIT)
        _pin(Path(job["source_root"]) / "source_manifest.json", item.get("origin_source_manifest_sha256"))
        origin = item.get("origin_provenance", {})
        if (origin.get("run_id") != run_id or origin.get("git_commit") != ORIGIN_COMMIT or
                origin.get("source_sha256") != item["origin_source_manifest_sha256"]):
            raise ValueError("original provenance must bind the exact frozen source manifest")
        artifacts = item.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts or item.get("origin_provenance") is None:
            raise ValueError("repair requires original raw evidence/provenance pins")
        pinned = set()
        for artifact in artifacts:
            source = _pin(artifact["path"], artifact["sha256"])
            relative = str(source.relative_to(directory)).replace("\\", "/")
            if artifact.get("relative_path") != relative or relative in pinned:
                raise ValueError("repair artifact path/identity differs")
            pinned.add(relative)
        mandatory = {str(p.relative_to(directory)).replace("\\", "/") for p in directory.rglob("*")
                     if p.is_file() and (p.name in ("manifest.json", "teacher_manifest.json", "initial_manifest.json",
                         "checkpoint.pt", "losses.jsonl", "worker.log", "failure.json") or
                         p.name.endswith(".manifest.json")) and "source_lineage" not in p.parts}
        if mandatory - pinned:
            raise ValueError("original manifests/checkpoint/log/failure must all be pinned")
        snapshot, execution = freeze_execution(store, contract)
        history = directory / "source_lineage" / "origin"
        history.mkdir(parents=True, exist_ok=True)
        for name in ("job", "status", "config", "launch"):
            shutil.copyfile(directory / (name + ".json"), history / (name + ".json"))
        shutil.copyfile(Path(job["source_root"]) / "source_manifest.json", history / "source_manifest.json")
        for artifact in artifacts:
            target = history / "artifacts" / artifact["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact["path"], target)
            _pin(target, artifact["sha256"])
        local_contract = directory / "source_lineage" / "contract.json"
        shutil.copyfile(path, local_contract)
        per_run = {**deepcopy(item), "format": FORMAT, "repair_id": REPAIR_ID, "run_id": run_id,
                   "origin_commit": ORIGIN_COMMIT, "execution_commit": execution,
                   "config_path": str(directory / "config.json"), "contract_sha256": body["contract_sha256"]}
        job.update(source_root=str(snapshot), git_commit=execution, origin_commit=ORIGIN_COMMIT,
                   execution_commit=execution, source_repair_contract=per_run, resume=True,
                   source_repair_pending=True,
                   execution_source={"contract_path": str(local_contract), "contract_sha256": body["contract_sha256"],
                                     "source_path": str(path)},
                   source_history=[{"origin_commit": ORIGIN_COMMIT, "execution_commit": execution,
                                    "repair_id": REPAIR_ID, "contract_sha256": body["contract_sha256"],
                                    "source_manifest_sha256": digest(snapshot / "source_manifest.json"),
                                    "at": now(), "comparison_scope": item["comparison_scope"]}])
        validate_job_lineage(job, directory)
        (directory / "stop.request").unlink(missing_ok=True)
        atomic_json(directory / "job.json", job)
        status.update(stage="queued", pid=None, scientific_stop=False, error=None,
                      source_repair=REPAIR_ID, execution_commit=execution, heartbeat_at=now(),
                      queue_reason="显式S1-ND12来源修复：原来源与失败记录保留")
        atomic_json(directory / "status.json", status)
        job["source_repair_pending"] = False
        atomic_json(directory / "job.json", job)
        return status


def validate_dependency_source(store, execution_reference, dependency_job):
    """Teacher/model production and consumption must use the same execution source."""
    if not execution_reference and not dependency_job.get("source_repair_contract"):
        return
    expected = ORIGIN_COMMIT
    if execution_reference:
        contract, _ = validate_contract(store, execution_reference["contract_path"],
                                        execution_reference["contract_sha256"])
        expected = contract["execution_commit"]
    if dependency_job.get("git_commit") != expected:
        raise ValueError("teacher/model dependency execution source differs from consumer")
    if dependency_job.get("source_repair_contract"):
        validate_job_lineage(dependency_job, Path(dependency_job["source_repair_contract"]["config_path"]).parent)
