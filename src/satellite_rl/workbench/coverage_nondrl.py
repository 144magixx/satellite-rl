"""Strict S1-ND12 workbench contract, separate from historical PPO schemas."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

from satellite_rl.coverage.nondrl.config import NondrlConfig
from satellite_rl.coverage.config import _merge
from satellite_rl.training.run_monitor import atomic_json
from .storage import digest, new_id, now, read_json, validate_id

GROUP = "S1-ND12"
PARENT = Path("/home/u2024110224/first_stage_satellite_rl/s1_nd12")
PORTS = ("20418", "20424", "20473", "20474")
ROLES = ("train", "development", "validation", "test")
ROUTES = tuple(f"{i:02d}" for i in range(1, 13))
LEARNED = ROUTES[:5]
UPDATE_CAPS = dict(zip(LEARNED, (10000, 10000, 12000, 8000, 4000)))
TEACHER_CAPS = {
    "01": {"max_problems": 20000, "max_trials": 20000, "max_seconds": 20000.},
    "02": {"max_problems": 5000, "max_trials": 1000000, "max_seconds": 50000.},
    "03": {"max_problems": 10000, "max_trials": 10000, "max_seconds": 40000.},
    "04": {"max_problems": 2000, "max_trials": 121600, "max_seconds": 12800.},
    "05": {"max_problems": 960, "max_trials": 11520, "max_seconds": 7200.},
}


def default_nondrl_config(route="06"):
    if route not in ROUTES:
        raise ValueError("unknown non-DRL route")
    return {"schema_version": 7, "research_stage": "coverage", "algorithm": "nondrl",
            "name": f"S1-ND12 · {route}", "network": {"kind": "nondrl"},
            "coverage": {"environment_profile": "europe_v1", "method": "nondrl",
                         "nondrl": {"route": route, "track": "fixed_count",
                                    "method": "main", "limits": NondrlConfig().to_dict()}},
            "training": {"device": "cuda", "seed": 51001, "total_steps": 0,
                         "updates": UPDATE_CAPS.get(route, 0), "batch_size": 16 if route in ("03", "04") else 32,
                         "checkpoint_interval": 1000, "validation_interval": 1000},
            "teacher": deepcopy(TEACHER_CAPS.get(route, {"max_problems": 0, "max_trials": 0,
                                                        "max_seconds": 0.})),
            "evaluation": {"demand_multipliers": [.75, 1., 1.25]}, "datasets": {}}


def validate_nondrl_config(supplied):
    route = supplied.get("coverage", {}).get("nondrl", {}).get("route", "06")
    default = default_nondrl_config(route)
    value = _merge(default, supplied)
    if any(value[key] != default[key] for key in ("schema_version", "research_stage", "algorithm", "network")):
        raise ValueError("S1-ND12 requires the independent schema-7 contract")
    if not isinstance(value["name"], str) or not 1 <= len(value["name"].strip()) <= 100:
        raise ValueError("invalid task name")
    c = value["coverage"]
    if c["environment_profile"] != "europe_v1" or c["method"] != "nondrl":
        raise ValueError("non-DRL Europe contract differs")
    if c["nondrl"]["track"] != "fixed_count":
        raise ValueError("first round supports fixed-count mechanism track only")
    # Methods are explicitly dispatched by the core, never silently interpreted as main.
    if c["nondrl"]["method"] not in ("main", "I0", "BH", "rule", "random", "ablation"):
        raise ValueError("unknown comparison method")
    NondrlConfig.from_dict(c["nondrl"]["limits"])
    if c["nondrl"]["limits"] != default["coverage"]["nondrl"]["limits"]:
        raise ValueError("formal geometry/search limits must use frozen common limits")
    if value["training"] != default["training"] or value["teacher"] != default["teacher"]:
        raise ValueError("non-DRL seed, teacher or update budget differs")
    if value["evaluation"] != default["evaluation"]:
        raise ValueError("demand multiplier contract differs")
    if not isinstance(value["datasets"], dict) or set(value["datasets"]) - set(ROLES):
        raise ValueError("invalid dataset roles")
    return value


def deployment(store):
    from .coverage import _require_remote_dqn
    archive = store.archive_root.resolve()
    port = archive.parent.name
    if (archive.name != "archive" or archive.parent.parent != PARENT.resolve() or
            port not in PORTS or os.environ.get("SATELLITE_RL_SSH_PORT") != port):
        raise ValueError("S1-ND12 archive must match isolated SSH lane")
    _require_remote_dqn()
    return port


def _pin(path, expected):
    source = Path(path).resolve()
    if not isinstance(expected, str) or len(expected) != 64 or not source.is_file() or digest(source) != expected:
        raise ValueError("source path/SHA mismatch")
    return source


def partition_contract(path, expected):
    source = _pin(path, expected)
    contract = read_json(source, {})
    if contract.get("group") != GROUP or set(contract.get("roles", {})) != set(ROLES):
        raise ValueError("S1-ND12 partition manifest must declare all four roles")
    dates = set()
    for role, item in contract["roles"].items():
        if (not isinstance(item.get("dates"), list) or not item["dates"] or
                type(item.get("count")) is not int or item["count"] <= 0 or
                len(set(item["dates"])) != len(item["dates"]) or dates.intersection(item["dates"])):
            raise ValueError("partition dates/count invalid or overlapping")
        dates.update(item["dates"])
    return contract


def import_dataset(application, body):
    from satellite_rl.coverage.variable_dataset import load_variable_dataset
    deployment(application.store)
    role = body.get("partition_role")
    if role not in ROLES:
        raise ValueError("dataset role required")
    path = _pin(body["path"], body.get("sha256"))
    manifest = partition_contract(body["partition_manifest_path"], body.get("partition_manifest_sha256"))
    dataset = load_variable_dataset(path)
    dates = sorted({d for scene in dataset.scenes for d in scene.metadata.get("source_dates", [])})
    spec = manifest["roles"][role]
    if (dates != sorted(spec["dates"]) or len(dataset.scenes) != spec["count"] or
            len({s.scene_id for s in dataset.scenes}) != len(dataset.scenes) or
            dataset.metadata.get("partition_role") != role or
            spec.get("sha256") not in (None, body["sha256"])):
        raise ValueError("dataset disagrees with frozen partition manifest")
    if not all(scene.metadata.get("selected_time_utc") and scene.source_hash for scene in dataset.scenes):
        raise ValueError("aircraft dataset lacks timestamp/source provenance")
    identifier = new_id("nondrl_scenes")
    directory = application.store.record_dir("scenes", identifier)
    directory.mkdir(parents=True)
    target = directory / "inputs.npz"
    shutil.copyfile(path, target)
    shutil.copyfile(body["partition_manifest_path"], directory / "partition_manifest.json")
    if digest(target) != body["sha256"]:
        raise ValueError("dataset copy differs")
    record = {"id": identifier, "name": body.get("name", f"S1-ND12 {role}"),
              "schema_version": 7, "record_type": "nondrl_aircraft_dataset", "research_stage": "coverage",
              "path": str(target), "sha256": body["sha256"], "profile": "europe_v1",
              "partition_role": role, "dates": dates, "count": len(dataset.scenes),
              "source_path": str(path), "source": dataset.metadata, "created_at": now(),
              "partition_manifest_path": str(directory / "partition_manifest.json"),
              "partition_manifest_sha256": body["partition_manifest_sha256"]}
    atomic_json(directory / "record.json", record)
    return application.store.save_record("scenes", record)


def _dependency(body, key, required_kind, route):
    source = body.get(key)
    if not isinstance(source, dict) or str(source.get("port")) not in PORTS:
        raise ValueError(f"{key} requires a completed source task and lane")
    run_id = validate_id(source.get("run_id"))
    directory = PARENT / str(source["port"]) / "archive/outputs/workbench/jobs" / run_id
    # Store layout is centralized rather than guessed by callers.
    from .storage import Store
    directory = Store(PARENT / str(source["port"]) / "archive").job_dir(run_id)
    job, status = read_json(directory / "job.json", {}), read_json(directory / "status.json", {})
    if (job.get("experiment_id") != GROUP or job.get("kind") != required_kind or
            job.get("variant") != route or status.get("stage") != "completed"):
        raise ValueError(f"{key} source is not a completed matching S1-ND12 task")
    return directory, job


def submit(application, body):
    config = validate_nondrl_config(body["config"])
    port = deployment(application.store)
    route = config["coverage"]["nondrl"]["route"]
    kind = body.get("kind")
    mode = body.get("preparation_mode", "teacher")
    if kind not in ("prepare", "train", "evaluate"):
        raise ValueError("unknown S1-ND12 task kind")
    if kind == "prepare" and mode == "dataset":
        from .nondrl_preparation import submit_dataset
        return submit_dataset(application, body, config, port)
    initial_mode = kind == "prepare" and mode == "initial"
    if (kind == "train" or kind == "prepare" and not initial_mode) and route not in LEARNED:
        raise ValueError("non-learning routes have no teacher or training job")
    role = body.get("dataset_role", "test") if kind == "evaluate" or initial_mode else None
    purpose = body.get("experiment_role", "formal")
    if purpose not in ("formal", "verification") or role not in (ROLES if initial_mode else (None,) if kind != "evaluate" else ROLES[1:]):
        raise ValueError("invalid task purpose or evaluation partition")
    if kind == "evaluate" and role == "test" and purpose != "formal":
        raise ValueError("test cannot be used for engineering selection")
    required = {role} if kind == "evaluate" or initial_mode else ({"development", "validation"} if kind == "train" and route == "05" else {"train", "validation"} if kind == "train" else {"development" if route == "05" else "train"})
    supplied = body.get("dataset_ids", {})
    if set(supplied) != required:
        raise ValueError(f"requires exactly {sorted(required)} datasets")
    datasets = {r: application.store.get_record("scenes", identifier) for r, identifier in supplied.items()}
    for r, record in datasets.items():
        if (record.get("schema_version") != 7 or record.get("partition_role") != r or
                record.get("record_type") != "nondrl_aircraft_dataset"):
            raise ValueError("dataset schema/role differs")
        _pin(record["path"], record["sha256"])
        partition_contract(record["partition_manifest_path"], record["partition_manifest_sha256"])
    if len({r["partition_manifest_sha256"] for r in datasets.values()}) != 1:
        raise ValueError("dataset partition contracts differ")
    config["datasets"] = datasets
    extra = {"experiment_id": GROUP, "research_stage": "coverage", "variant": route,
             "experiment_role": purpose, "dataset_role": role, "execution_target": f"u2024110224@10.160.17.102:{port}",
             "dataset_sha256": {r: d["sha256"] for r, d in datasets.items()},
             "execution_device": "cuda" if kind == "train" or route in LEARNED and kind == "evaluate" else "cpu"}
    if kind == "prepare":
        if mode not in ("teacher", "initial"):
            raise ValueError("unknown preparation mode")
        extra["preparation_mode"] = mode
    if kind == "train":
        gate = body.get("gate_source", {})
        gate_path = _pin(gate.get("path", ""), gate.get("sha256"))
        gate_record = read_json(gate_path, {})
        if (gate_record.get("experiment_id") != GROUP or gate_record.get("route") != route or
                gate_record.get("passed") is not True):
            raise ValueError("training requires a passed frozen route development gate")
        extra["reachability_source"] = {"path": str(gate_path), "sha256": gate["sha256"]}
        directory, source_job = _dependency(body, "teacher_source", "prepare", route)
        path = directory / "teacher_manifest.json"
        manifest = read_json(path, {})
        if (manifest.get("completed") is not True or
                manifest.get("dataset_sha256") not in extra["dataset_sha256"].values()):
            raise ValueError("teacher completion/data lineage differs")
        extra["teacher_source"] = {"path": str(path), "sha256": digest(path), "run_id": source_job["run_id"]}
    if kind == "evaluate":
        directory, source_job = _dependency(body, "initial_source", "prepare", body["initial_source"].get("route", route))
        path = directory / "initial_manifest.json"
        manifest = read_json(path, {})
        if (source_job.get("preparation_mode") != "initial" or
                manifest.get("dataset_sha256") != datasets[role]["sha256"] or
                manifest.get("completed") is not True):
            raise ValueError("shared I0 provenance differs")
        extra["initial_source"] = {"path": str(path), "sha256": digest(path), "run_id": source_job["run_id"]}
    if kind == "evaluate" and route in LEARNED and config["coverage"]["nondrl"]["method"] == "main":
        directory, source_job = _dependency(body, "model_source", "train", route)
        model = Path(source_job["model_dir"]) / "final"
        extra.update(model_path=str(model), model_source_run_id=source_job["run_id"],
                     model_files=[{"path": str(model / name), "sha256": digest(model / name)} for name in ("policy.pt", "model.json")])
    indices = body.get("scene_indices")
    if indices is not None:
        if kind != "evaluate" or not isinstance(indices, list) or not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i not in range(datasets[role]["count"]) for i in indices):
            raise ValueError("scene index selection invalid")
        extra["scene_indices"] = indices
    return application.store.create_job(config, kind=kind, request_id=body.get("request_id"), extra=extra)


def freeze_sources(extra, destination):
    frozen = deepcopy(extra)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for key in ("teacher_source", "dataset_source", "initial_source", "reachability_source"):
        source = frozen.get(key)
        if source:
            path = _pin(source["path"], source["sha256"])
            target = destination / path.name
            shutil.copyfile(path, target)
            frozen[key] = {**source, "source_path": str(path), "path": str(target)}
    if frozen.get("model_files"):
        target = destination / "model"
        target.mkdir()
        refs = []
        for item in frozen["model_files"]:
            source = _pin(item["path"], item["sha256"])
            output = target / source.name
            shutil.copyfile(source, output)
            refs.append({"path": str(output), "sha256": item["sha256"], "source_path": str(source)})
        frozen.update(model_files=refs, model_path=str(target))
    return frozen


def api(application, method, parts, body):
    if parts == ["catalog"] and method == "GET":
        return {"group": GROUP, "routes": list(ROUTES), "learning_routes": list(LEARNED),
                "defaults": {r: default_nondrl_config(r) for r in ROUTES}, "update_caps": UPDATE_CAPS,
                "teacher_caps": TEACHER_CAPS, "roles": list(ROLES), "kinds": ["prepare", "train", "evaluate"]}
    if parts == ["validate"] and method == "POST":
        return {"config": validate_nondrl_config(body)}
    if parts == ["datasets", "import"] and method == "POST":
        return import_dataset(application, body)
    if parts == ["datasets", "merge"] and method == "POST":
        return merge_datasets(application, body)
    if parts == ["datasets"] and method == "GET":
        return {"datasets": [r for r in application.store.records("scenes") if r.get("schema_version") == 7]}
    if parts == ["runs"] and method == "POST":
        return submit(application, body)
    if parts == ["runs"] and method == "GET":
        return {"runs": [r for r in application.app.runs()["runs"] if r.get("experiment_id") == GROUP]}
    raise FileNotFoundError("non-DRL API not found")


def resume_job(scheduler, run_id):
    from .runtime import pid_alive
    directory = scheduler.store.job_dir(run_id)
    job, status = read_json(directory / "job.json", {}), read_json(directory / "status.json", {})
    launch = read_json(directory / "launch.json", {})
    if status.get("stage") not in ("stopped", "failed", "interrupted") or status.get("scientific_stop") is True:
        raise ValueError("S1-ND12 task cannot resume from this state")
    if pid_alive(status.get("pid"), launch.get("process_start_time")):
        raise ValueError("original worker still alive")
    config = validate_nondrl_config(read_json(directory / "config.json", {}))
    for record in config["datasets"].values():
        _pin(record["path"], record["sha256"])
    if job["kind"] == "train" and not (Path(job["model_dir"]) / "checkpoint.pt").is_file():
        raise ValueError("training requires complete optimizer/RNG/cursor checkpoint")
    attempt = int(job.get("resume_attempt", 0)) + 1
    history = directory / "recovery_history" / f"attempt_{attempt:03d}"
    history.mkdir(parents=True)
    atomic_json(history / "status_before_resume.json", status)
    if (directory / "failure.json").is_file():
        shutil.copyfile(directory / "failure.json", history / "failure_before_resume.json")
    (directory / "stop.request").unlink(missing_ok=True)
    job.update(resume=True, resume_attempt=attempt)
    atomic_json(directory / "job.json", job)
    status.update(stage="queued", pid=None, queue_reason="S1-ND12同ID恢复冻结源码和完整游标", error=None,
                  heartbeat_at=now(), resume_attempt=attempt)
    atomic_json(directory / "status.json", status)
    return status


def merge_datasets(application, body):
    from satellite_rl.coverage.scenes import CoverageDataset
    from satellite_rl.coverage.variable_dataset import load_variable_dataset, save_variable_dataset
    deployment(application.store)
    ids = body.get("dataset_ids")
    if not isinstance(ids, list) or len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("merge requires distinct source dataset IDs")
    records = [application.store.get_record("scenes", i) for i in ids]
    roles = {r.get("partition_role") for r in records}
    manifests = {r.get("partition_manifest_sha256") for r in records}
    if len(roles) != 1 or len(manifests) != 1 or any(r.get("schema_version") != 7 for r in records):
        raise ValueError("merge source schema/role/manifest differs")
    role = records[0]["partition_role"]
    manifest = partition_contract(records[0]["partition_manifest_path"], records[0]["partition_manifest_sha256"])
    scenes, dates = [], []
    for record in records:
        _pin(record["path"], record["sha256"])
        scenes.extend(load_variable_dataset(record["path"]).scenes)
        dates.extend(record["dates"])
    if (len(set(dates)) != len(dates) or sorted(dates) != sorted(manifest["roles"][role]["dates"]) or
            len(scenes) != manifest["roles"][role]["count"] or len({s.scene_id for s in scenes}) != len(scenes)):
        raise ValueError("merged partition count/date/mother identity differs")
    identifier = new_id("nondrl_scenes")
    directory = application.store.record_dir("scenes", identifier)
    dataset = CoverageDataset(scenes, {"partition_role": role, "source_dates": sorted(dates),
                              "source_datasets": [{"id": r["id"], "sha256": r["sha256"]} for r in records]})
    save_variable_dataset(dataset, directory / "inputs.npz")
    shutil.copyfile(records[0]["partition_manifest_path"], directory / "partition_manifest.json")
    record = {"id": identifier, "name": f"S1-ND12 merged {role}", "schema_version": 7,
              "record_type": "nondrl_aircraft_dataset", "research_stage": "coverage", "profile": "europe_v1",
              "partition_role": role, "path": str(directory / "inputs.npz"), "sha256": digest(directory / "inputs.npz"),
              "count": len(scenes), "dates": sorted(dates), "source": dataset.metadata, "created_at": now(),
              "partition_manifest_path": str(directory / "partition_manifest.json"),
              "partition_manifest_sha256": records[0]["partition_manifest_sha256"]}
    return application.store.save_record("scenes", record)
