"""S1-D12A workbench contract: frozen data, teacher, BC, PPO and evaluation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import getpass
from pathlib import Path
import os
import shutil
import sys

from satellite_rl.coverage.config import _merge
from satellite_rl.coverage.drl_r04_config import DrlR04Config
from satellite_rl.workbench.storage import digest, new_id, now, read_json, validate_id


GROUP = "S1-D12A"
PARENT = Path("/home/u2024110224/first_stage_satellite_rl/s1_d12a")
PORTS = ("20418", "20424", "20473", "20474")
ROLES = ("train", "development", "validation", "test")
ROLE_DATES = {"train": ("2024-01-08", "2024-01-09"),
              "development": ("2024-01-12",),
              "validation": ("2024-02-07",),
              "test": ("2024-02-08", "2024-02-09")}
ROLE_COUNTS = {"train": 576, "development": 20,
               "validation": 20, "test": 100}
MULTIPLIERS = [0.75, 1.0, 1.25]
METHODS = ("I0", "H1", "H2", "BC", "P0", "P1")
LEARNED = ("BC", "P0", "P1")


def default_drl_r04_config():
    """Fixed scientific budget; variant and phase live in the job metadata."""
    return {
        "schema_version": 6, "research_stage": "coverage", "algorithm": "ppo",
        "name": "S1-D12A · R04引导的DRL覆盖优化",
        "network": {"kind": "drl_r04_set_ppo", "hidden_size": 128},
        "coverage": {"environment_profile": "europe_v1", "method": "ppo",
                     "drl_r04": asdict(DrlR04Config())},
        "training": {"device": "cuda", "seed": 51001, "total_steps": 262144,
                     "num_envs": 32, "rollout_steps": 64, "minibatch_size": 64,
                     "epochs": 4, "lr": .0003, "gamma": 1.,
                     "gae_lambda": .95, "clip_ratio": .2, "entropy_coef": .01,
                     "value_coef": .5, "max_grad_norm": .5,
                     "validation_interval": 16384, "checkpoint_interval": 2048,
                     "torch_threads": 8, "cpu_workers": 16,
                     "bc_epochs": 20, "bc_batch_size": 256,
                     "bc_max_updates": 1600, "pilot_step": 65536,
                     "teacher_max_labels": 20000,
                     "teacher_max_trials": 442368},
        "evaluation": {"action_mode": "deterministic", "repeats": 1,
                       "demand_multipliers": MULTIPLIERS},
        "datasets": {},
    }


def validate_drl_r04_config(supplied):
    default = default_drl_r04_config()
    config = _merge(default, supplied)
    if (config["schema_version"] != 6 or config["research_stage"] != "coverage" or
            config["algorithm"] != "ppo" or config["network"] != default["network"]):
        raise ValueError("S1-D12A requires the frozen schema-6 set PPO contract")
    if not isinstance(config["name"], str) or not 1 <= len(config["name"].strip()) <= 100:
        raise ValueError("S1-D12A name must have 1–100 characters")
    coverage = config["coverage"]
    if (coverage["environment_profile"] != "europe_v1" or
            coverage["method"] not in ("ppo", "local_search", "none")):
        raise ValueError("S1-D12A Europe physics or method differs")
    if coverage["drl_r04"] != default["coverage"]["drl_r04"]:
        raise ValueError("S1-D12A geometry, menu, objective or call budget differs")
    DrlR04Config(**coverage["drl_r04"])
    expected_training = deepcopy(default["training"])
    if config["training"].get("total_steps") == 2048:
        expected_training["total_steps"] = 2048
        expected_training["validation_interval"] = 2048
    if config["training"] != expected_training or config["evaluation"] != default["evaluation"]:
        raise ValueError("S1-D12A seed, PPO, teacher, BC or evaluation budget differs")
    datasets = config["datasets"]
    if not isinstance(datasets, dict) or set(datasets) - set(ROLES):
        raise ValueError("S1-D12A dataset role differs")
    for role, record in datasets.items():
        if (not isinstance(record, dict) or record.get("partition_role") != role or
                record.get("schema_version") != 6 or not isinstance(record.get("path"), str) or
                not isinstance(record.get("sha256"), str)):
            raise ValueError(f"S1-D12A {role} dataset provenance incomplete")
    return config


def drl_r04_summary(config):
    value = validate_drl_r04_config(config)
    c = value["coverage"]["drl_r04"]
    return {"experiment_id": GROUP, "methods": list(METHODS),
            "max_beams": c["max_beams"], "fixed_width_deg": c["width_deg"],
            "max_commits": c["max_commits"], "max_trials": c["max_trials"],
            "max_geometry_calls": c["max_geometry_calls"],
            "train_cap_per_model": value["training"]["total_steps"],
            "pilot_step": value["training"]["pilot_step"],
            "demand_multipliers": MULTIPLIERS,
            "physics_version": "europe_v1_idealized_fixed_width"}


def deployment(store):
    import torch
    archive = store.archive_root.resolve()
    port = archive.parent.name if archive.name == "archive" else ""
    if (archive.parent.parent != PARENT.resolve() or port not in PORTS or
            os.environ.get("SATELLITE_RL_SSH_PORT") != port):
        raise ValueError("S1-D12A archive and SSH port must match an isolated lane")
    if (getpass.getuser() != "u2024110224" or
            Path(sys.prefix).resolve() != Path("/home/u2024110224/first_stage_satellite_rl/envs/DQN") or
            not torch.cuda.is_available() or "4090 D" not in torch.cuda.get_device_name(0)):
        raise ValueError("S1-D12A requires remote DQN/CUDA/4090D")
    return port


def import_dataset(application, body):
    """Import a source prepared elsewhere, preserving its exact bytes and SHA."""
    from satellite_rl.coverage.variable_dataset import load_variable_dataset
    deployment(application.store)
    role = body.get("partition_role")
    if role not in ROLES:
        raise ValueError("S1-D12A dataset role must be explicit")
    path = Path(body["path"]).resolve()
    expected_sha = body.get("sha256")
    if not path.is_file() or not isinstance(expected_sha, str) or digest(path) != expected_sha:
        raise ValueError("S1-D12A dataset path/SHA mismatch")
    dataset = load_variable_dataset(path)
    dates = sorted({day for scene in dataset.scenes
                    for day in scene.metadata.get("source_dates", [])})
    daily_test = role == "test" and len(dataset.scenes) == 50 and len(dates) == 1 and \
        dates[0] in ROLE_DATES["test"]
    if (not (tuple(dates) == ROLE_DATES[role] and len(dataset.scenes) == ROLE_COUNTS[role]
             or daily_test) or dataset.metadata.get("partition_role") != role):
        raise ValueError("S1-D12A mother dates, count or role differ")
    if role in ("train", "development"):
        from satellite_rl.workbench.coverage_variable import HISTORICAL_SOURCE_SHA
        provenance = read_json(path.parent / "record.json", {})
        if (not path.is_relative_to(PARENT.parent / "s1_v10") or
                provenance.get("schema_version") != 5 or
                provenance.get("sha256") != expected_sha or
                dataset.metadata.get("source_frozen_npz_sha256") != HISTORICAL_SOURCE_SHA[role]):
            raise ValueError("S1-D12A historical train/development source SHA differs")
    else:
        if not all(scene.metadata.get("selection_contract_sha256") and
                   scene.metadata.get("source_preflight_sha256") for scene in dataset.scenes):
            raise ValueError("new independent scene lacks frozen source selection lineage")
        if daily_test or role == "validation":
            source_directory, source_job = _source_job(body.get("source_port"), body.get("source_run_id"))
            artifacts = read_json(source_directory / "artifacts.json", {})
            if (source_job.get("preparation_mode") != "dataset" or
                    source_job.get("dataset_role") != role or
                    artifacts.get("prepared_dataset") != str(path) or
                    artifacts.get("prepared_dataset_sha256") != expected_sha):
                raise ValueError("new independent scene is not from completed workbench prepare")
        else:
            provenance = read_json(path.parent / "record.json", {})
            if (not path.is_relative_to(PARENT) or provenance.get("schema_version") != 6 or
                    provenance.get("sha256") != expected_sha or provenance.get("count") != 100):
                raise ValueError("merged test source lacks workbench provenance")
    identifier = new_id("drl_r04_scenes")
    directory = application.store.record_dir("scenes", identifier)
    directory.mkdir(parents=True)
    target = directory / "inputs.npz"
    shutil.copyfile(path, target)
    if digest(target) != expected_sha:
        raise ValueError("S1-D12A copied dataset SHA differs")
    record = {"id": identifier, "name": body.get("name") or f"S1-D12A {role}",
              "path": str(target), "sha256": expected_sha, "schema_version": 6,
              "research_stage": "coverage", "record_type": "drl_r04_aircraft_dataset",
              "profile": "europe_v1", "partition_role": role, "dates": dates,
              "count": len(dataset.scenes), "source_path": str(path),
              "source": dataset.metadata, "created_at": now()}
    from satellite_rl.training.run_monitor import atomic_json
    atomic_json(directory / "record.json", record)
    return application.store.save_record("scenes", record)


def merge_test_datasets(application, body):
    """Combine the two frozen 50-mother Feb08/09 workbench scene records."""
    from satellite_rl.coverage.scenes import CoverageDataset
    from satellite_rl.coverage.variable_dataset import load_variable_dataset, save_variable_dataset
    deployment(application.store)
    ids = body.get("dataset_ids")
    if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("S1-D12A test merge requires two distinct workbench records")
    records = [application.store.get_record("scenes", identifier) for identifier in ids]
    dates = []
    scenes = []
    for record in records:
        if (record.get("schema_version") != 6 or record.get("partition_role") != "test" or
                record.get("record_type") != "drl_r04_aircraft_dataset" or
                record.get("count") != 50 or len(record.get("dates", [])) != 1 or
                record["dates"][0] not in ROLE_DATES["test"] or
                digest(record["path"]) != record["sha256"]):
            raise ValueError("S1-D12A daily test source differs")
        dates.extend(record["dates"])
        scenes.extend(load_variable_dataset(record["path"]).scenes)
    if tuple(sorted(dates)) != ROLE_DATES["test"] or len({s.scene_id for s in scenes}) != 100:
        raise ValueError("S1-D12A merged test mothers or dates differ")
    identifier = new_id("drl_r04_scenes")
    root = application.store.record_dir("scenes", identifier)
    root.mkdir(parents=True)
    path = root / "inputs.npz"
    metadata = {"partition_role": "test", "source_dates": sorted(dates),
                "source_datasets": [{"id": r["id"], "sha256": r["sha256"],
                                     "dates": r["dates"]} for r in records],
                "demand_multipliers_applied": [1.0]}
    save_variable_dataset(CoverageDataset(scenes, metadata), path)
    record = {"id": identifier, "name": "S1-D12A Feb08+09 independent test",
              "path": str(path), "sha256": digest(path), "schema_version": 6,
              "research_stage": "coverage", "record_type": "drl_r04_aircraft_dataset",
              "profile": "europe_v1", "partition_role": "test", "dates": sorted(dates),
              "count": 100, "source": metadata, "created_at": now()}
    from satellite_rl.training.run_monitor import atomic_json
    atomic_json(root / "record.json", record)
    return application.store.save_record("scenes", record)


def _dataset_preparation_source(role, day):
    from hashlib import sha256
    import json
    if role not in ("validation", "test") or day not in ROLE_DATES[role]:
        raise ValueError("S1-D12A new source role/day differs")
    root = PARENT / "source" / role / day
    pin_path = root / "source_preflight.json"
    contract_path = root / "selection_contract.json"
    pin = read_json(pin_path, {})
    contract = read_json(contract_path, {})
    if (pin.get("group") != GROUP or contract.get("group") != GROUP or
            pin.get("role") != role or contract.get("role") != role or
            pin.get("date_utc") != day or contract.get("date_utc") != day or
            pin.get("selection_contract_sha256") != digest(contract_path) or
            contract.get("count") != (20 if role == "validation" else 50) or
            len(contract.get("selected_times_utc", [])) != contract["count"] or
            contract.get("no_outcome_based_replacement") is not True):
        raise ValueError("S1-D12A official source pin or selected UTC contract differs")
    records = {}
    for name, key in (("source_preflight.json", None),
                      ("selection_contract.json", "selection_contract_sha256"),
                      ("release_api.json", "release_api_sha256"),
                      ("PREFERRED_RELEASES.txt", "preferred_sha256"),
                      ("LICENSE-ODbL-official.txt", "license_sha256")):
        path = root / name
        if not path.is_file() or (key and digest(path) != pin[key]):
            raise ValueError(f"S1-D12A source {name} missing or changed")
        records[name] = {"path": str(path), "sha256": digest(path)}
    release = read_json(root / "release_api.json", {})
    assets = [item for item in release.get("assets", [])
              if item.get("name") == pin.get("archive_name")]
    preferred = (root / "PREFERRED_RELEASES.txt").read_text(encoding="utf-8")
    if (len(assets) != 1 or release.get("tag_name") != pin.get("release_tag") or
            assets[0].get("id") != pin.get("asset_id") or
            assets[0].get("size") != pin.get("expected_bytes") or
            assets[0].get("browser_download_url") != pin.get("archive_url") or
            pin.get("archive_url") not in preferred):
        raise ValueError("S1-D12A official release asset and preferred list differ")
    from satellite_rl.workbench.coverage_variable import HISTORICAL_SOURCE_SHA
    if set(contract.get("excluded_training_dates", [])) != set(ROLE_DATES["train"]) or \
            set(contract.get("excluded_development_dates", [])) != set(ROLE_DATES["development"]):
        raise ValueError("S1-D12A selection has training/development date collision")
    times_sha = sha256(json.dumps(contract["selected_times_utc"],
                             separators=(",", ":")).encode()).hexdigest()
    reference = Path(__file__).resolve().parents[3] / "scripts" / "stage1_reconstruction" / "prepare_snapshots_reference.py"
    records["parser"] = {"path": str(reference), "sha256": digest(reference)}
    return {"role": role, "date_utc": day, "files": records,
            "selected_times_sha256": times_sha,
            "asset_id": pin["asset_id"], "expected_bytes": pin["expected_bytes"],
            "archive_name": pin["archive_name"], "archive_url": pin["archive_url"],
            "historical_train_sha256": HISTORICAL_SOURCE_SHA["train"]}


def _submit_dataset_preparation(application, body, config, port):
    if config["training"]["total_steps"] != 262144:
        raise ValueError("dataset preparation retains the formal schema budget")
    role, day = body.get("dataset_role"), body.get("source_date")
    preparation = _dataset_preparation_source(role, day)
    config["datasets"] = {}
    extra = {"research_stage": "coverage", "experiment_id": GROUP,
             "experiment_role": "formal", "variant": "dataset",
             "preparation_mode": "dataset", "dataset_role": role,
             "source_date": day, "dataset_preparation": preparation,
             "scientific_phase": "data_preparation",
             "execution_target": f"u2024110224@10.160.17.102:{port}"}
    return application.store.create_job(config, kind="prepare",
                                        request_id=body.get("request_id"), extra=extra)


def _source_job(port, run_id, *, allow_scientific_stop=False):
    if str(port) not in PORTS:
        raise ValueError("source lane differs")
    validate_id(run_id)
    directory = PARENT / str(port) / "archive" / "outputs" / "workbench" / "jobs" / run_id
    job = read_json(directory / "job.json", {})
    status = read_json(directory / "status.json", {})
    allowed_stage = status.get("stage") == "completed" or (
        allow_scientific_stop and status.get("stage") == "stopped" and
        status.get("scientific_stop") is True and
        status.get("stop_reason") in ("consecutive_validation_all_stop_zero_progress",
                                       "paired_pilot_no_progress"))
    if (job.get("run_id") != run_id or job.get("experiment_id") != GROUP or
            not allowed_stage):
        raise ValueError("source workbench task is not completed S1-D12A")
    return directory, job


def _teacher_source(body, train_sha):
    directory, job = _source_job(body.get("source_port"), body.get("source_run_id"))
    if job.get("preparation_mode") != "teacher" or job.get("kind") != "prepare":
        raise ValueError("BC requires completed teacher task")
    manifest_path = directory / "teacher_manifest.json"
    manifest = read_json(manifest_path, {})
    if (manifest.get("dataset_sha256") != train_sha or
            manifest.get("completed") is not True or manifest.get("inflight") or
            manifest.get("labels", 0) > 20000 or
            manifest.get("trials", 0) + manifest.get("unverified_trials", 0) > 442368):
        raise ValueError("teacher scene SHA or label budget differs")
    demonstrations = directory / "demonstrations"
    if not demonstrations.is_dir():
        raise ValueError("teacher examples unavailable")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("teacher manifest lacks example provenance")
    for item in scenes:
        path = Path(item.get("path", "")).resolve()
        if (not path.is_file() or not path.is_relative_to(demonstrations.resolve()) or
                digest(path) != item.get("sha256")):
            raise ValueError("teacher example file/SHA differs")
    return {"source_run_id": job["run_id"], "source_port": str(body["source_port"]),
            "manifest_path": str(manifest_path), "manifest_sha256": digest(manifest_path),
            "demonstrations_dir": str(demonstrations)}


def _model_source(body, expected_variant):
    directory, job = _source_job(body.get("source_port"), body.get("source_run_id"),
                                 allow_scientific_stop=expected_variant in ("P0", "P1"))
    if (job.get("variant") != expected_variant or job.get("kind") != "train" or
            job.get("experiment_role") != "formal"):
        raise ValueError("selected model task has wrong variant")
    status = read_json(directory / "status.json", {})
    endpoint = ("bc_endpoint" if expected_variant == "BC" else
                "final" if status.get("stage") == "completed" else
                "scientific_stop_endpoint" if status.get("stop_reason") ==
                    "consecutive_validation_all_stop_zero_progress" and
                    status.get("env_step") != 65536 else
                "pilot_endpoint")
    model_dir = Path(job["model_dir"]) / endpoint
    files = [model_dir / "policy.pt", model_dir / "model.json"]
    if any(not path.is_file() for path in files):
        raise ValueError("selected source model endpoint incomplete")
    metadata = read_json(model_dir / "model.json", {})
    if (metadata.get("env_step") != status.get("env_step", 0) or
            metadata.get("selection") != (
                "bc_endpoint" if expected_variant == "BC" else
                "fixed_budget_endpoint" if endpoint == "final" else endpoint) or
            metadata.get("policy_sha256") != digest(model_dir / "policy.pt")):
        raise ValueError("selected model endpoint metadata/budget differs")
    return {"source_run_id": job["run_id"], "source_port": str(body["source_port"]),
            "model_path": str(model_dir), "selection": metadata["selection"],
            "env_step": metadata["env_step"],
            "model_files": [{"path": str(path), "sha256": digest(path)} for path in files]}


def _reachability_contract(body, validation_sha):
    import json
    import math
    from .drl_r04_gate import _real_refresh, _valid_terminal
    path = Path(body.get("reachability_contract_path", "")).resolve()
    expected_sha = body.get("reachability_contract_sha256")
    if (not path.is_file() or not path.is_relative_to(PARENT / "source") or
            not isinstance(expected_sha, str) or digest(path) != expected_sha):
        raise ValueError("S1-D12A training lacks frozen validation reachability contract")
    contract = read_json(path, {})
    if (contract.get("format") != "s1d12a_reachability_v1" or
            contract.get("validation_dataset_sha256") != validation_sha or
            contract.get("experiment_id") != GROUP or
            not isinstance(contract.get("reachable_mother_indices"), list) or
            not contract["reachable_mother_indices"] or
            any(type(index) is not int or index not in range(20)
                for index in contract["reachable_mother_indices"]) or
            len(set(contract["reachable_mother_indices"])) != len(contract["reachable_mother_indices"])):
        raise ValueError("validation reachable subset or source dataset differs")
    reachable = set()
    for variant in ("H1", "H2"):
        source = contract.get("baselines", {}).get(variant, {})
        directory, job = _source_job(source.get("port"), source.get("run_id"))
        result_path = directory / "results.jsonl"
        if (job.get("kind") != "evaluate" or job.get("variant") != variant or
                job.get("dataset_role") != "validation" or
                job.get("dataset_sha256", {}).get("validation") != validation_sha or
                digest(result_path) != source.get("sha256")):
            raise ValueError("fixed validation baseline evidence changed")
        rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
        if (len(rows) != 60 or any(
                row.get("dataset_index") != index // 3 or
                row.get("demand_multiplier") != MULTIPLIERS[index % 3] or
                not _valid_terminal(row) for index, row in enumerate(rows))):
            raise ValueError("fixed validation baseline has missing or invalid scene")
        improved = [mother for mother in range(20)
                    if math.fsum(rows[mother * 3 + level]["initial_j"] -
                                 rows[mother * 3 + level]["best_j"]
                                 for level in range(3)) / 3 > 1e-8 and
                    any(_real_refresh(rows[mother * 3 + level]) for level in range(3))]
        if improved != source.get("improved_mother_indices"):
            raise ValueError("frozen baseline reachable mothers differ from raw results")
        reachable.update(improved)
    if sorted(reachable) != contract["reachable_mother_indices"]:
        raise ValueError("combined reachable validation subset differs from baselines")
    return {"path": str(path), "sha256": expected_sha,
            "reachable_mother_indices": contract["reachable_mother_indices"],
            "validation_dataset_sha256": validation_sha}


def submit(application, body, config):
    config = validate_drl_r04_config(config)
    port = deployment(application.store)
    kind, variant = body.get("kind"), body.get("variant")
    purpose = body.get("experiment_role", "formal")
    role = body.get("dataset_role")
    if kind == "prepare" and variant == "dataset":
        return _submit_dataset_preparation(application, body, config, port)
    if (kind not in ("prepare", "train", "evaluate") or variant not in METHODS + ("teacher",) or
            purpose not in ("formal", "verification")):
        raise ValueError("S1-D12A task kind, variant or role differs")
    if (kind == "prepare" and variant != "teacher") or (kind == "train" and variant not in LEARNED):
        raise ValueError("S1-D12A teacher prepares; BC/P0/P1 train")
    if purpose == "verification" and kind in ("prepare", "train") and variant not in ("P0", "P1"):
        raise ValueError("only P0/P1 have authorized 2048-step engineering verification")
    if kind == "evaluate" and (role not in ("development", "validation", "test") or
                               (purpose == "formal" and role == "development") or
                               (purpose == "verification" and role == "test")):
        raise ValueError("S1-D12A evaluation role/purpose differs")
    if kind != "evaluate" and role is not None:
        raise ValueError("non-evaluation task must not select an evaluation role")
    if kind == "train" and variant in ("P0", "P1"):
        expected_steps = 2048 if purpose == "verification" else 262144
        if config["training"]["total_steps"] != expected_steps:
            raise ValueError("S1-D12A verification/formal decision budget differs")
    elif config["training"]["total_steps"] != 262144:
        raise ValueError("S1-D12A non-PPO job retains the formal configuration budget")
    shard = body.get("shard_id")
    if kind == "evaluate" and role == "test":
        if type(shard) is not int or shard not in range(4):
            raise ValueError("test must select one of four frozen mother-scene shards")
        indices = list(range(shard, ROLE_COUNTS["test"], 4))
        if body.get("scene_indices") != indices:
            raise ValueError("test mother-scene index list differs from frozen shard")
    elif shard is not None or body.get("scene_indices") is not None:
        raise ValueError("only test evaluation uses four-scene sharding")
    expected_method = "ppo" if variant in LEARNED + ("teacher",) else "none" if variant == "I0" else "local_search"
    if config["coverage"]["method"] != expected_method:
        raise ValueError("S1-D12A variant and coverage method differ")
    supplied = body.get("dataset_ids")
    required = {"train"} if kind == "prepare" or variant == "BC" else (
        {"train", "validation"} if kind == "train" else {role})
    if not isinstance(supplied, dict) or set(supplied) != required:
        raise ValueError(f"S1-D12A requires exactly {sorted(required)} datasets")
    datasets = {}
    for dataset_role, identifier in supplied.items():
        record = application.store.get_record("scenes", identifier)
        if (record.get("schema_version") != 6 or record.get("partition_role") != dataset_role or
                record.get("record_type") != "drl_r04_aircraft_dataset" or
                tuple(record.get("dates", ())) != ROLE_DATES[dataset_role] or
                record.get("count") != ROLE_COUNTS[dataset_role] or
                digest(record["path"]) != record["sha256"]):
            raise ValueError(f"S1-D12A {dataset_role} frozen dataset differs")
        datasets[dataset_role] = record
    config["datasets"] = datasets
    extra = {"research_stage": "coverage", "experiment_id": GROUP,
             "experiment_role": purpose, "variant": variant,
             "dataset_role": role, "execution_target": f"u2024110224@10.160.17.102:{port}",
             "dataset_sha256": {key: record["sha256"] for key, record in datasets.items()},
             "scientific_phase": "teacher" if kind == "prepare" else
                                 "bc" if variant == "BC" else
                                 "pilot" if kind == "train" else "evaluation"}
    if kind == "evaluate" and role == "test":
        extra.update(shard_id=shard, scene_indices=indices,
                     shard_source_dataset_sha256=datasets["test"]["sha256"],
                     shard_mother_count=len(indices), expected_result_rows=len(indices) * 3)
    if kind == "prepare":
        extra.update(preparation_mode="teacher", teacher_max_labels=20000,
                     teacher_max_trials=442368)
    elif kind == "train":
        extra["train_mode"] = variant
        if variant in ("P0", "P1"):
            extra["reachability_contract"] = _reachability_contract(
                body, datasets["validation"]["sha256"])
        if variant == "BC":
            teacher = _teacher_source(body, datasets["train"]["sha256"])
            extra.update(teacher_source=teacher,
                         demonstrations_dir=teacher["demonstrations_dir"])
        elif variant == "P1":
            source = _model_source(body, "BC")
            extra.update(bc_source=source, bc_model_dir=source["model_path"])
    elif variant in LEARNED:
        source = _model_source(body, variant)
        extra.update(model_source=source, model_path=source["model_path"],
                     model_files=source["model_files"])
    return application.store.create_job(config, kind=kind,
                                        request_id=body.get("request_id"), extra=extra)


def freeze_drl_r04_sources(extra, destination):
    """Pin task dependencies without copying large immutable teacher arrays."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    frozen = deepcopy(extra)
    teacher = frozen.get("teacher_source")
    if teacher:
        source = Path(teacher["manifest_path"])
        if digest(source) != teacher["manifest_sha256"]:
            raise ValueError("teacher manifest changed during job freeze")
        manifest = read_json(source, {})
        examples = Path(teacher["demonstrations_dir"]).resolve()
        for item in manifest.get("scenes", []):
            path = Path(item["path"]).resolve()
            if not path.is_relative_to(examples) or digest(path) != item["sha256"]:
                raise ValueError("teacher example changed during job freeze")
        target = destination / "teacher_manifest.json"
        shutil.copyfile(source, target)
        teacher["manifest_source_path"] = str(source)
        teacher["manifest_path"] = str(target)
    preparation = frozen.get("dataset_preparation")
    if preparation:
        root = destination / "official_source"
        root.mkdir()
        for name, item in preparation["files"].items():
            source = Path(item["path"])
            if digest(source) != item["sha256"]:
                raise ValueError("S1-D12A official source metadata changed during freeze")
            target = root / ("parser.py" if name == "parser" else name)
            shutil.copyfile(source, target)
            if digest(target) != item["sha256"]:
                raise ValueError("S1-D12A official source frozen SHA differs")
            item["source_path"], item["path"] = str(source), str(target)
    reachability = frozen.get("reachability_contract")
    if reachability:
        source = Path(reachability["path"])
        if digest(source) != reachability["sha256"]:
            raise ValueError("S1-D12A reachable subset changed during freeze")
        target = destination / "reachability_contract.json"
        shutil.copyfile(source, target)
        if digest(target) != reachability["sha256"]:
            raise ValueError("S1-D12A frozen reachable subset SHA differs")
        reachability["source_path"], reachability["path"] = str(source), str(target)
    bc = frozen.get("bc_source")
    if bc:
        source = Path(bc["model_path"])
        target = destination / "bc_model"
        target.mkdir()
        for item in bc["model_files"]:
            path = Path(item["path"])
            if path.parent.resolve() != source.resolve() or digest(path) != item["sha256"]:
                raise ValueError("BC model changed during P1 freeze")
            shutil.copyfile(path, target / path.name)
            if digest(target / path.name) != item["sha256"]:
                raise ValueError("frozen BC model SHA differs")
        bc["source_model_path"] = bc["model_path"]
        bc["model_path"] = str(target)
        frozen["bc_model_dir"] = str(target)
    return frozen
