"""Independent S1-G16 workbench contract; no downstream service metrics."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from satellite_rl.training.run_monitor import atomic_json
from .storage import digest, new_id, now, read_json, validate_id

GROUP = "S1-G16"
PARENT = Path("/home/u2024110224/first_stage_satellite_rl/s1_g16")
PORTS = ("20424", "20473")
METHODS = ("BH", "07", "09-CV", "09-LEX", "09+SWAP", "JOINT", "QUOTA", "VARIABLE", "MLM", "rule", "random")
ROLES = ("train", "development", "validation", "test")

def default_config():
    return {"schema_version": 8, "research_stage": "coverage", "algorithm": "geoload",
            "name": "S1-G16 geometry and request load", "network": {"kind": "geoload"},
            "coverage": {"environment_profile": "europe_v1", "method": "geoload",
                         "width_deg": .5, "max_beams": 200},
            "training": {"seed": 51001, "total_steps": 0, "updates": 10000,
                         "batch": 32, "microbatch": 4, "learning_rate": 3e-4, "network": {}},
            "search": {"seconds": 2., "max_attempts": 4096, "max_geometry_calls": 20000,
                       "objective_tolerance": 1e-9},
            "workers": 4, "datasets": {}}

def deployment(store):
    root = store.archive_root.resolve()
    port = root.parent.name
    if (root.name != "archive" or root.parent.parent != PARENT.resolve() or port not in PORTS or
            os.environ.get("SATELLITE_RL_SSH_PORT") != port):
        raise ValueError("S1-G16 requires isolated authorized 20424/20473 archive")
    from .coverage import _require_remote_dqn
    _require_remote_dqn()
    return port

def pin(reference):
    if not isinstance(reference, dict) or not isinstance(reference.get("sha256"), str):
        raise ValueError("immutable path/SHA reference required")
    path = Path(reference.get("path", "")).resolve()
    if not path.is_file() or len(reference["sha256"]) != 64 or digest(path) != reference["sha256"]:
        raise ValueError("input source/SHA differs")
    return path

def validate_config(value):
    value = deepcopy(value)
    default = default_config()
    if set(value) != set(default):
        raise ValueError("unknown/missing G16 configuration field")
    for key in ("schema_version", "research_stage", "algorithm", "network", "coverage", "training"):
        if value[key] != default[key]:
            raise ValueError("frozen G16 geometry/learning configuration differs: " + key)
    if type(value["workers"]) is not int or value["workers"] not in (1, 2, 4, 8):
        raise ValueError("workers must be 1/2/4/8")
    if not isinstance(value["name"], str) or not 1 <= len(value["name"]) <= 100:
        raise ValueError("invalid task name")
    if set(value["search"]) != set(default["search"]):
        raise ValueError("invalid search fields")
    if value["search"]["seconds"] not in (2., 10., 30.):
        raise ValueError("only frozen 2/10/30 second budgets")
    if any(value["search"][k] != default["search"][k] for k in default["search"] if k != "seconds"):
        raise ValueError("candidate/geometry/tolerance budget differs")
    if not isinstance(value["datasets"], dict) or set(value["datasets"]) - set(ROLES):
        raise ValueError("invalid G16 partitions")
    return value

def create_job(application, body):
    """Freeze contract files and source before publishing a schedulable job."""
    store = application.store
    port = deployment(store)
    config = validate_config(body.get("config", default_config()))
    kind = body.get("kind")
    if kind not in ("prepare", "evaluate", "train"):
        raise ValueError("unknown G16 kind")
    mode = body.get("mode", "evaluate" if kind == "evaluate" else "initial")
    if mode not in ("initial", "teacher", "evaluate", "train", "dataset", "reference"):
        raise ValueError("unknown G16 worker mode")
    if (kind, mode) not in (("prepare", "initial"), ("prepare", "teacher"), ("prepare", "dataset"),
                            ("evaluate", "evaluate"), ("evaluate", "reference"), ("train", "train")):
        raise ValueError("kind/mode differs")
    role = body.get("dataset_role")
    if role not in ROLES or (mode != "dataset" and role not in config["datasets"]):
        raise ValueError("declared partition required")
    if role == "test" and (mode not in ("evaluate","dataset","initial") or config["search"]["seconds"] != 2.):
        raise ValueError("test is only common frozen 2s evaluation")
    if config["search"]["seconds"] != 2. and role != "development":
        raise ValueError("10/30s references are development only")
    methods = body.get("methods", ["BH"])
    if not isinstance(methods, list) or not methods or len(methods) != len(set(methods)) or set(methods)-set(METHODS):
        raise ValueError("unknown/duplicate G16 method")
    request = validate_id(body.get("request_id"))
    plan_path = pin(body.get("plan"))
    plan = read_json(plan_path, {})
    if plan.get("group") != GROUP or plan.get("authorized") is not True:
        raise ValueError("frozen authorized G16 plan required")
    partition_path = pin(body.get("partition"))
    partition = read_json(partition_path, {})
    if partition.get("group") != GROUP or set(partition.get("roles", {})) != set(ROLES):
        raise ValueError("four-role G16 partition contract required")
    dates = []
    for r, spec in partition["roles"].items():
        dates.extend(spec.get("dates", []))
        if spec.get("count", 0) <= 0:
            raise ValueError("invalid partition count")
    if len(dates) != len(set(dates)):
        raise ValueError("partition dates overlap")
    for r, record in config["datasets"].items():
        pin(record)
        if record.get("role") != r or record.get("count") != partition["roles"][r]["count"]:
            raise ValueError("dataset role/count differs from partition")
    if mode == "dataset":
        spec=deepcopy(body.get("dataset_spec",{}))
        if spec.get("role") != role or spec.get("kind") not in ("import_existing","official_day"):
            raise ValueError("frozen official/import dataset specification required")
        if spec.get("count",spec.get("expected_mothers")) != partition["roles"][role].get("mother_count",partition["roles"][role]["count"]//4):
            raise ValueError("dataset mother budget differs from partition")
        if spec["kind"]=="import_existing":
            for ref in spec.get("sources",[spec]): pin(ref)
        else:
            pin(spec.get("parser"))
            for ref in spec.get("metadata_pins",{}).values():pin(ref)
    if mode == "teacher" and role != "train":
        raise ValueError("teacher labels are training only")
    if mode == "teacher" and methods != ["JOINT"]:
        raise ValueError("teacher requires frozen JOINT trajectory")
    if mode == "reference" and (role != "development" or methods != ["JOINT"]):
        raise ValueError("nested reference is development JOINT only")
    if "MLM" in methods: pin(body.get("model"))
    if kind == "train":
        if set(config["datasets"]) != {"train","validation"}:
            raise ValueError("MLM requires fixed train and validation partitions")
        gate = read_json(pin(body.get("gate")), {})
        if gate.get("group") != GROUP or gate.get("passed") is not True:
            raise ValueError("training requires independently passed teacher gate")
        pin(body.get("teacher"))
    payload_sha = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    with store.lock:
        for existing, _ in store.jobs():
            if existing.get("request_id") == request:
                if existing.get("request_hash") != payload_sha:
                    raise ValueError("request ID reused for different G16 input")
                return existing
        source, commit = store.freeze_source()
        if commit != plan.get("scientific_commit"):
            raise ValueError("science commit differs from frozen plan")
        run_id = new_id(kind + "_geoload")
        directory = store.job_dir(run_id); directory.mkdir()
        model_dir = store.models_root / run_id; model_dir.mkdir()
        refs = {}
        for key in ("plan", "partition", "gate", "teacher", "model"):
            if body.get(key):
                original = pin(body[key]); target = directory / "input_sources" / key / original.name
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(original, target)
                refs[key] = {**body[key], "path": str(target), "source_path": str(original)}
                if key == "teacher":
                    evidence = read_json(original, {})
                    if evidence.get("status") != "completed": raise ValueError("incomplete teacher manifest")
                    for sample in evidence.get("samples", []):
                        source_sample = (original.parent / sample["path"]).resolve()
                        if not source_sample.is_relative_to(original.parent.resolve()):
                            raise ValueError("teacher shard escapes source root")
                        pin({"path": str(source_sample), "sha256": sample["sha256"]})
                        destination = target.parent / sample["path"]
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source_sample, destination)
        job = {"run_id": run_id, "kind": kind, "name": config["name"], "created_at": now(),
               "request_id": request, "request_hash": payload_sha, "archive_root": str(store.archive_root),
               "source_root": str(source), "git_commit": commit, "scientific_commit": commit,
               "runtime_commit": commit, "algorithm": "geoload", "network": "geoload",
               "model_dir": str(model_dir), "experiment_id": GROUP, "research_stage": "coverage",
               "dataset_role": role, "mode": mode, "methods": methods, "references": refs,
               "execution_device": "cuda" if kind == "train" else "cpu", "ssh_port": port}
        if mode=="dataset":job["dataset_spec"]=spec
        atomic_json(directory / "config.json", config)
        atomic_json(directory / "status.json", {"run_id": run_id, "stage": "queued", "pid": None,
                    "env_step": 0, "update_step": 0, "completed_cases": 0, "heartbeat_at": now()})
        atomic_json(directory / "job.json", job)
        return job

def resume_job(scheduler, run_id):
    directory = scheduler.store.job_dir(run_id)
    job = read_json(directory / "job.json", {})
    status = read_json(directory / "status.json", {})
    deployment(scheduler.store)
    if job.get("experiment_id") != GROUP or status.get("stage") not in ("failed", "stopped", "interrupted"):
        raise ValueError("G16 task cannot resume in current state")
    from .runtime import pid_alive
    if pid_alive(status.get("pid"), read_json(directory / "launch.json", {}).get("process_start_time")):
        raise ValueError("original worker remains alive")
    for ref in job["references"].values(): pin(ref)
    config = validate_config(read_json(directory / "config.json", {}))
    for ref in config["datasets"].values(): pin(ref)
    if job["kind"] == "train" and not (Path(job["model_dir"]) / "checkpoint.pt").is_file():
        raise ValueError("complete optimizer/RNG/cursor checkpoint required")
    attempt = int(job.get("resume_attempt", 0)) + 1
    history = directory / "recovery_history" / str(attempt); history.mkdir(parents=True)
    for name in ("status.json", "failure.json"):
        if (directory / name).exists(): shutil.copyfile(directory / name, history / name)
    job.update(resume=True, resume_attempt=attempt)
    (directory / "stop.request").unlink(missing_ok=True)
    atomic_json(directory / "job.json", job)
    status.update(stage="queued", pid=None, error=None, heartbeat_at=now())
    atomic_json(directory / "status.json", status)
    return status

def api(application, method, parts, body):
    if method == "GET" and parts == ["schema"]:
        return {"group": GROUP, "config": default_config(), "methods": METHODS,
                "metrics": ["P", "T", "Q", "CV", "N", "evidence"]}
    if method == "GET" and parts == ["jobs"]:
        return [{"job": j, "status": s} for j, s in application.store.jobs() if j.get("experiment_id") == GROUP]
    if method == "POST" and parts == ["jobs"]: return create_job(application, body)
    if len(parts) == 3 and parts[0] == "runs" and parts[2] == "resume" and method == "POST":
        return application.scheduler.resume(validate_id(parts[1]))
    raise FileNotFoundError("unknown G16 API")
