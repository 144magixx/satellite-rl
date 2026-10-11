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
METHODS = {"01": ("main", "M", "MO", "O", "HO", "H"),
           "02": ("main", "BC", "H", "T", "R"), "03": ("main", "D", "DR", "H", "HR"),
           "04": ("main", "L", "H", "R", "F"), "05": ("main", "L", "rule", "worst", "random"),
           "06": ("main", "assign_only"), "07": ("main", "assign_only"),
           "08": ("main", "voronoi", "integer"), "09": ("main", "distance", "demand", "geometric"),
           "10": ("main", "initial_pool"), "11": ("main", "random", "single"),
           "12": ("main", "single_objective")}
MODEL_METHODS = {"01": ("main", "M", "MO"), "02": ("main", "BC"), "03": ("main", "D", "DR"),
                 "04": ("main", "L"), "05": ("main", "L")}
TEACHER_CAPS = {
    "01": {"max_problems": 20000, "max_trials": 20000, "max_seconds": 20000.},
    "02": {"max_problems": 4940, "max_trials": 961600, "max_seconds": 49400.},
    "03": {"max_problems": 9940, "max_trials": 1272320, "max_seconds": 39760.},
    "04": {"max_problems": 2000, "max_trials": 121600, "max_seconds": 11520.},
    "05": {"max_problems": 960, "max_trials": 11520, "max_seconds": 7200.},
}
PROBE_CAPS = {"02": {"max_problems": 60, "max_trials": 38400, "max_seconds": 600.,
                     "max_steps": 20, "per_problem_seconds": 10.},
              "03": {"max_problems": 60, "max_trials": 7680, "max_seconds": 240.,
                     "max_steps": 4, "per_problem_seconds": 4.}}


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
                         "checkpoint_interval": 1000, "validation_interval": 1000, "microbatch_size": 4},
            "preparation": {"initial_workers": 8},
            "teacher": {**deepcopy(TEACHER_CAPS.get(route, {"max_problems": 0, "max_trials": 0,
                                                        "max_seconds": 0.})), "workers": 8, "worker_threads": 1,
                        "max_steps": 4 if route == "03" else 1 if route in ("04", "05") else 20,
                        "per_problem_seconds": 10. if route == "02" else 1. if route == "01" else 4.},
            "evaluation": {"demand_multipliers": [.75, 1., 1.25]}, "datasets": {}}


def validate_nondrl_config(supplied):
    if not isinstance(supplied, dict) or not isinstance(supplied.get("coverage", {}), dict):
        raise ValueError("non-DRL configuration and coverage must be objects")
    if not isinstance(supplied.get("coverage", {}).get("nondrl", {}), dict):
        raise ValueError("non-DRL route configuration must be an object")
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
    if c["nondrl"]["method"] not in METHODS[route] + ("I0", "BH"):
        raise ValueError("unknown comparison method")
    NondrlConfig.from_dict(c["nondrl"]["limits"])
    if c["nondrl"]["limits"] != default["coverage"]["nondrl"]["limits"]:
        raise ValueError("formal geometry/search limits must use frozen common limits")
    if value["training"] != default["training"] or value["teacher"] != default["teacher"]:
        raise ValueError("non-DRL seed, teacher or update budget differs")
    if value["preparation"]["initial_workers"] not in (1, 2, 4, 8) or type(value["preparation"]["initial_workers"]) is not int:
        raise ValueError("initial worker count must be 1,2,4 or8")
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


def _initial_dependencies(body, key, dataset_sha):
    supplied = body.get(key)
    values = supplied if isinstance(supplied, list) else [supplied]
    if not values or any(not isinstance(v, dict) for v in values):
        raise ValueError(f"{key} requires frozen common initial source tasks")
    records = []
    for value in values:
        directory, source_job = _dependency({key: value}, key, "prepare", value.get("route", "06"))
        path = directory / "initial_manifest.json"
        manifest = read_json(path, {})
        if (source_job.get("preparation_mode") != "initial" or manifest.get("completed") is not True or
                manifest.get("dataset_sha256") != dataset_sha):
            raise ValueError("common initial partition provenance differs")
        records.append({"path": str(path), "sha256": digest(path), "run_id": source_job["run_id"]})
    return records


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
    ranking_audit_mode = kind == "prepare" and mode == "ranking_audit"
    probe_mode = kind == "prepare" and mode == "teacher_probe"
    if probe_mode and route not in PROBE_CAPS:
        raise ValueError("development teacher probe is implemented only for02/03")
    if ranking_audit_mode and route != "04":
        raise ValueError("fixed candidate audit is only implemented for route04")
    if (kind == "train" or kind == "prepare" and not initial_mode) and route not in LEARNED:
        raise ValueError("non-learning routes have no teacher or training job")
    role = body.get("dataset_role", "test") if kind == "evaluate" or initial_mode or ranking_audit_mode or probe_mode else None
    purpose = body.get("experiment_role", "formal")
    if purpose not in ("formal", "verification") or role not in (ROLES if initial_mode else ("validation",) if ranking_audit_mode else ("development",) if probe_mode else (None,) if kind != "evaluate" else ROLES[1:]):
        raise ValueError("invalid task purpose or evaluation partition")
    if kind == "evaluate" and role == "test" and purpose != "formal":
        raise ValueError("test cannot be used for engineering selection")
    required = {role} if kind == "evaluate" or initial_mode or ranking_audit_mode or probe_mode else ({"development", "validation"} if kind == "train" and route == "05" else {"train", "validation"} if kind == "train" else {"development" if route == "05" else "train"})
    supplied = body.get("dataset_ids", {})
    if set(supplied) != required:
        raise ValueError(f"requires exactly {sorted(required)} datasets")
    datasets = {r: application.store.get_record("scenes", identifier) for r, identifier in supplied.items()}
    for r, record in datasets.items():
        if (record.get("schema_version") != 7 or record.get("partition_role") != r or
                record.get("record_type") != "nondrl_aircraft_dataset"):
            raise ValueError("dataset schema/role differs")
        _pin(record["path"], record["sha256"])
        manifest = partition_contract(record["partition_manifest_path"], record["partition_manifest_sha256"])
        spec = manifest["roles"][r]
        if record.get("count") != spec["count"] or sorted(record.get("dates", [])) != sorted(spec["dates"]):
            raise ValueError("task requires a complete frozen partition; merge daily source records first")
    if len({r["partition_manifest_sha256"] for r in datasets.values()}) != 1:
        raise ValueError("dataset partition contracts differ")
    config["datasets"] = datasets
    needs_model = config["coverage"]["nondrl"]["method"] in MODEL_METHODS.get(route, ())
    extra = {"experiment_id": GROUP, "research_stage": "coverage", "variant": route,
             "experiment_role": purpose, "dataset_role": role, "execution_target": f"u2024110224@10.160.17.102:{port}",
             "dataset_sha256": {r: d["sha256"] for r, d in datasets.items()},
             "execution_device": "cuda" if kind == "train" or needs_model and kind == "evaluate" else "cpu"}
    if kind == "evaluate" and role == "development" and route == "04" and config["coverage"]["nondrl"]["method"] == "F":
        per_case = 5120 // (datasets[role]["count"] * len(config["evaluation"]["demand_multipliers"]))
        extra["resolved_limits"] = {**config["coverage"]["nondrl"]["limits"],
                                    "max_candidates": per_case, "max_physical_calls": per_case}
        extra["offline_label_account"] = {"role": "development", "total_trial_ceiling": 5120,
                                           "per_case_trial_ceiling": per_case,
                                           "note": "shared cap applies across all mother shards and demand levels"}
    if kind == "prepare":
        if mode not in ("teacher", "initial", "ranking_audit", "teacher_probe"):
            raise ValueError("unknown preparation mode")
        extra["preparation_mode"] = mode
        if ranking_audit_mode:
            extra["teacher_limits"] = {**config["teacher"], "max_problems": 20, "max_trials": 1280,
                                       "max_seconds": 128., "max_steps": 1, "per_problem_seconds": 4.}
        if probe_mode:
            extra["teacher_limits"] = {**config["teacher"], **PROBE_CAPS[route]}
        if mode in ("teacher", "ranking_audit", "teacher_probe"):
            teacher_role = next(iter(required))
            extra["initial_source"] = _initial_dependencies(body, "initial_source", datasets[teacher_role]["sha256"])
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
        extra["validation_initial_source"] = _initial_dependencies(body, "validation_initial_source", datasets["validation"]["sha256"])
        if route == "04":
            directory, source_job = _dependency(body, "ranking_audit_source", "prepare", route)
            path = directory / "teacher_manifest.json"
            manifest = read_json(path, {})
            if (source_job.get("preparation_mode") != "ranking_audit" or manifest.get("completed") is not True or
                    manifest.get("dataset_sha256") != datasets["validation"]["sha256"]):
                raise ValueError("fixed route04 ranking audit dataset/completion differs")
            extra["ranking_audit_source"] = {"path": str(path), "sha256": digest(path), "run_id": source_job["run_id"]}
    if kind == "evaluate":
        extra["initial_source"] = _initial_dependencies(body, "initial_source", datasets[role]["sha256"])
    if kind == "evaluate" and needs_model:
        directory, source_job = _dependency(body, "model_source", "train", route)
        model = Path(source_job["model_dir"]) / "final"
        metadata = read_json(model / "model.json", {})
        source_config = read_json(directory / "config.json", {})
        if (metadata.get("route") != route or metadata.get("schema_version") != 7 or
                metadata.get("selection") != "fixed_budget_endpoint" or
                metadata.get("updates") != source_config.get("training", {}).get("updates") or
                metadata.get("policy_sha256") != digest(model / "policy.pt") or
                metadata.get("provenance", {}).get("run_id") != source_job["run_id"]):
            raise ValueError("model endpoint route/budget/provenance differs")
        extra.update(model_path=str(model), model_source_run_id=source_job["run_id"],
                     model_files=[{"path": str(model / name), "sha256": digest(model / name)} for name in ("policy.pt", "model.json")])
    indices = body.get("scene_indices")
    if indices is not None:
        if not (kind == "evaluate" or initial_mode) or not isinstance(indices, list) or not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i not in range(datasets[role]["count"]) for i in indices):
            raise ValueError("scene index selection invalid")
        extra["scene_indices"] = indices
    return application.store.create_job(config, kind=kind, request_id=body.get("request_id"), extra=extra)


def freeze_sources(extra, destination):
    frozen = deepcopy(extra)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for key in ("teacher_source", "dataset_source", "initial_source", "reachability_source", "validation_initial_source", "ranking_audit_source"):
        source = frozen.get(key)
        if source:
            values = source if isinstance(source, list) else [source]
            records = []
            for index, item in enumerate(values):
                path = _pin(item["path"], item["sha256"])
                target = destination / key / str(index) / path.name
                target.parent.mkdir(parents=True)
                shutil.copyfile(path, target)
                records.append({**item, "source_path": str(path), "path": str(target)})
            frozen[key] = records if isinstance(source, list) else records[0]
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
                "teacher_caps": TEACHER_CAPS, "teacher_probe_caps": PROBE_CAPS,
                "methods": {r: list(m) + ["I0", "BH"] for r, m in METHODS.items()},
                "roles": list(ROLES), "kinds": ["prepare", "train", "evaluate"]}
    if parts == ["validate"] and method == "POST":
        return {"config": validate_nondrl_config(body)}
    if parts == ["datasets", "import"] and method == "POST":
        return import_dataset(application, body)
    if parts == ["datasets", "merge"] and method == "POST":
        return merge_datasets(application, body)
    if parts == ["datasets"] and method == "GET":
        return {"datasets": [r for r in application.store.records("scenes") if r.get("schema_version") == 7]}
    if parts == ["models"] and method == "GET":
        models = []
        for job, status in application.store.jobs():
            if job.get("experiment_id") != GROUP or job.get("kind") != "train" or status.get("stage") != "completed":
                continue
            root = Path(job["model_dir"]) / "final"
            metadata = read_json(root / "model.json", {})
            if (root / "policy.pt").is_file() and metadata.get("policy_sha256") == digest(root / "policy.pt"):
                models.append({"run_id": job["run_id"], "route": job["variant"], "path": str(root),
                               "sha256": metadata["policy_sha256"], "metadata": metadata})
        return {"models": models}
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
    references = body.get("source_datasets")
    if references is not None:
        from .storage import Store
        if (not isinstance(references, list) or len(references) < 2 or
                any(not isinstance(r, dict) or str(r.get("port")) not in PORTS for r in references) or
                len({(str(r["port"]), r.get("id")) for r in references}) != len(references)):
            raise ValueError("cross-lane merge requires distinct authorized port/record references")
        records = []
        for reference in references:
            store = Store(PARENT / str(reference["port"]) / "archive")
            record = store.get_record("scenes", validate_id(reference.get("id")))
            directory = store.job_dir(record.get("source_run_id", ""))
            source_job = read_json(directory / "job.json", {})
            artifacts = read_json(directory / "artifacts.json", {})
            if (source_job.get("experiment_id") != GROUP or source_job.get("preparation_mode") != "dataset" or
                    read_json(directory / "status.json", {}).get("stage") != "completed" or
                    source_job.get("dataset_role") != record.get("partition_role") or
                    record.get("dates") != [source_job.get("source_date")] or
                    artifacts.get("prepared_dataset") != record.get("path") or
                    artifacts.get("prepared_dataset_sha256") != record.get("sha256")):
                raise ValueError("cross-lane daily dataset must come from completed official ND12 prepare")
            records.append({**record, "source_port": str(reference["port"]), "source_dataset_id": record["id"]})
    else:
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
                              "source_datasets": [{"id": r["id"], "sha256": r["sha256"],
                                  "source_port": r.get("source_port"), "source_run_id": r.get("source_run_id")} for r in records]})
    save_variable_dataset(dataset, directory / "inputs.npz")
    shutil.copyfile(records[0]["partition_manifest_path"], directory / "partition_manifest.json")
    record = {"id": identifier, "name": f"S1-ND12 merged {role}", "schema_version": 7,
              "record_type": "nondrl_aircraft_dataset", "research_stage": "coverage", "profile": "europe_v1",
              "partition_role": role, "path": str(directory / "inputs.npz"), "sha256": digest(directory / "inputs.npz"),
              "count": len(scenes), "dates": sorted(dates), "source": dataset.metadata, "created_at": now(),
              "partition_manifest_path": str(directory / "partition_manifest.json"),
              "partition_manifest_sha256": records[0]["partition_manifest_sha256"]}
    return application.store.save_record("scenes", record)
