"""S1-V10 variable-count coverage workbench contract and submission gates."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from satellite_rl.coverage.variable_config import ROUTES, VariableConfig
from satellite_rl.workbench.storage import digest, new_id, now


ROLES = ("train", "development", "validation", "test", "sequence")
LEARNED = ("R08", "R09")
LANES = ("20418", "20424", "20473", "20474")
ARCHIVE_PARENT = Path("/home/u2024110224/first_stage_satellite_rl/s1_v10")
DEMAND_MULTIPLIERS = (0.75, 1.0, 1.25)
SOURCE_DATES = {"validation": ("2024-02-03",),
                "test": ("2024-02-04", "2024-02-05"),
                "sequence": ("2024-02-06",)}
PLAN_REVISION = "s1_v10_20261009_v1"
ROLE_DATES = {"train": ["2024-01-08", "2024-01-09"],
              "development": ["2024-01-12"],
              "validation": ["2024-02-03"],
              "test": ["2024-02-04", "2024-02-05"],
              "sequence": ["2024-02-06"]}
ROLE_COUNTS = {"train": 576, "development": 20, "validation": 20,
               "test": 100, "sequence": 120}
HISTORICAL_SOURCE_SHA = {"train": "61be543ac64bf4d0b9e14fd39c7a88d1346d5540cdd3b023433e2ab57ac02ec6",
                         "development": "f73525eaab0e41bc208d4aa6fb14ad174b49c71947c837e2c04751fdec876219"}
SOURCE_CONTRACT_SHA = {"2024-02-03": "405949ef64361c307f009ab31d7df0e1f06437e2cb6157e6765f64edb5a5609d",
                       "2024-02-04": "75b57070090d837848c261c61c01ffeeb8a344097de19dc9afbc933e60003efe",
                       "2024-02-05": "02681f4498287b31af6d567cb580992872e6a766d2e7de6012c5e18fc309adae",
                       "2024-02-06": "866ead38ac0154411c765ff6e041cdb859f19319a5338af00db4235bd61c28c9"}
SOURCE_PREFLIGHT_SHA = {"2024-02-03": "d3475f54b70d3f5ab35133b7141e7a3d68aad90a804312cfd04eeff7526f0eba",
                        "2024-02-04": "79b45e365dba2b633ebedaa114e78f75b60d4b70f7734070d49096b775f7286b",
                        "2024-02-05": "db2368e0a40ab3954d997543a974a35d2c162371d61194fc970be6d974371a7c",
                        "2024-02-06": "211031cd115a4d12618a436b2c37578d3144111cb6325c3729bf0668da55864a"}


def variable_default_config():
    from satellite_rl.coverage.reconstruction_config import reconstruction_default_config

    config = reconstruction_default_config()
    config["schema_version"] = 5
    config["name"] = "S1-V10 · 可变波束全覆盖与负载均衡"
    config["network"] = {"kind": "variable_set_ppo", "hidden_size": 128,
                         "message_layers": 2}
    config["coverage"] = {
        "environment_profile": "europe_v1", "method": "local_search",
        "variable": {"route": "R01", **asdict(VariableConfig())},
    }
    config["training"].pop("mechanism_gate_step", None)
    config["training"].update(total_steps=262144, seed=51001, num_envs=32,
                              rollout_steps=64, minibatch_size=64, epochs=4,
                              lr=.0003, gamma=.99, gae_lambda=.95,
                              clip_ratio=.2, entropy_coef=.01, value_coef=.5,
                              max_grad_norm=.5, validation_interval=16384,
                              checkpoint_interval=2048,
                              demand_multipliers=list(DEMAND_MULTIPLIERS))
    config["datasets"] = {}
    config.pop("reconstruction_starts", None)
    config["evaluation"] = {"action_mode": "deterministic", "repeats": 1,
                            "demand_multipliers": list(DEMAND_MULTIPLIERS)}
    return config


def validate_variable_config(supplied):
    from satellite_rl.coverage.config import _merge

    config = _merge(variable_default_config(), supplied)
    if config["schema_version"] != 5 or config["research_stage"] != "coverage" or config["algorithm"] != "ppo":
        raise ValueError("S1-V10 requires coverage schema 5")
    if not isinstance(config["name"], str) or not 1 <= len(config["name"].strip()) <= 100:
        raise ValueError("name must contain 1–100 characters")
    if config["network"] != variable_default_config()["network"]:
        raise ValueError("S1-V10 network contract differs")
    coverage = config["coverage"]
    route = coverage["variable"]["route"]
    if route not in ROUTES:
        raise ValueError("unknown S1-V10 route")
    if coverage["environment_profile"] != "europe_v1" or coverage["method"] != (
            "ppo" if route in LEARNED else "local_search"):
        raise ValueError("route, method or Europe physics mismatch")
    raw = {key: value for key, value in coverage["variable"].items() if key != "route"}
    raw["width_choices_deg"] = tuple(raw["width_choices_deg"])
    VariableConfig(**raw)
    expected_training = variable_default_config()["training"]
    if config["training"]["total_steps"] == 2048:
        expected_training["total_steps"] = 2048
        expected_training["validation_interval"] = 2048
    if config["training"] != expected_training:
        raise ValueError("S1-V10 learning seed, budget or PPO settings differ from the frozen plan")
    if config["evaluation"] != variable_default_config()["evaluation"]:
        raise ValueError("S1-V10 demand multipliers and deterministic evaluation are fixed")
    datasets = config["datasets"]
    if not isinstance(datasets, dict) or set(datasets) - set(ROLES):
        raise ValueError("unknown variable-count dataset role")
    for role, record in datasets.items():
        if not isinstance(record, dict) or record.get("partition_role") != role:
            raise ValueError(f"{role} dataset provenance differs")
        if not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
            raise ValueError(f"{role} dataset path and SHA are required")
    return config


def variable_summary(config):
    c = validate_variable_config(config)
    v = c["coverage"]["variable"]
    return {"route": v["route"], "max_beams": v["max_beams"],
            "beam_width_deg": v["fixed_width_deg"],
            "width_physics_version": "idealized_circular_footprint_v1",
            "demand_multipliers": list(DEMAND_MULTIPLIERS),
            "coverage_goal": "all_aircraft", "geometry_call_budget": v["max_geometry_calls"],
            "training_decisions": c["training"]["total_steps"] if v["route"] in LEARNED else 0}


def _dataset_info(path: Path):
    from satellite_rl.coverage.variable_dataset import load_variable_dataset

    dataset = load_variable_dataset(path)
    if not dataset.scenes:
        raise ValueError("variable dataset is empty")
    if any(len(scene.aircraft_ids) != len(scene.demands) or
           len(scene.positions) != len(scene.demands) for scene in dataset.scenes):
        raise ValueError("aircraft positions, IDs and demands differ")
    if any(not np.all(np.isfinite(scene.demands)) or np.any(scene.demands < 0)
           for scene in dataset.scenes):
        raise ValueError("aircraft demand must be finite and nonnegative")
    dates = sorted({date for scene in dataset.scenes
                    for date in scene.metadata.get("source_dates", []) if date})
    if not dates:
        dates = sorted({date for date in dataset.metadata.get("source_dates", []) if date})
    if not dates:
        raise ValueError("aircraft dataset lacks UTC source dates")
    return dataset, dates


def import_variable_dataset(application, body):
    from satellite_rl.coverage.variable_dataset import save_variable_dataset

    _deployment(application.store)
    role = body.get("partition_role")
    if role not in ROLES:
        raise ValueError("explicit train/development/validation/test/sequence role required")
    source = Path(body["path"]).expanduser().resolve()
    dataset, dates = _dataset_info(source)
    source_role = dataset.metadata.get("partition_role")
    if source_role and source_role != role and role != "development":
        raise ValueError("source dataset role differs; historical reuse is development only")
    dataset.metadata["original_partition_role"] = source_role
    dataset.metadata["partition_role"] = role
    dataset.metadata["source_dates"] = dates
    identifier = new_id("variable_scenes")
    directory = application.store.record_dir("scenes", identifier)
    directory.mkdir(parents=True)
    destination = directory / "inputs.npz"
    if source.is_file() and source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            original_format = str(archive["format_version"])
    else:
        original_format = None
    if original_format == "coverage_aircraft_v1":
        dataset.metadata["source_frozen_npz_sha256"] = digest(source)
    if original_format == "variable_aircraft_v1" and source_role == role:
        provenance = source.parent / "record.json"
        if not provenance.is_file():
            raise ValueError("prepared variable dataset lacks immutable record.json")
        prepared = json.loads(provenance.read_text(encoding="utf-8"))
        if prepared.get("prepared_npz_sha256") != digest(source):
            raise ValueError("prepared variable dataset provenance SHA differs")
        shutil.copyfile(source, destination)
        shutil.copyfile(provenance, directory / "record.json")
        if digest(destination) != prepared["prepared_npz_sha256"]:
            raise ValueError("copied variable dataset SHA differs")
    else:
        prepared = save_variable_dataset(dataset, destination)
    record = {**prepared, "id": identifier, "name": body.get("name") or source.name,
              "path": str(destination), "sha256": digest(destination),
              "research_stage": "coverage", "schema_version": 5,
              "record_type": "variable_aircraft_dataset", "profile": "europe_v1",
              "created_at": now(), "partition_role": role, "dates": dates,
              "source_path": str(source), "source_sha256": digest(source) if source.is_file() else None,
              "source": dataset.metadata}
    return application.store.save_record("scenes", record)


def merge_variable_datasets(application, body):
    from satellite_rl.coverage.scenes import CoverageDataset
    from satellite_rl.coverage.variable_dataset import load_variable_dataset, save_variable_dataset

    _deployment(application.store)
    role = body.get("partition_role")
    ids = body.get("dataset_ids")
    if role not in ROLES or not isinstance(ids, list) or len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("merge requires two or more distinct schema-5 dataset IDs for one role")
    records = [application.store.get_record("scenes", identifier) for identifier in ids]
    if any(r.get("schema_version") != 5 or r.get("partition_role") != role or
           r.get("record_type") != "variable_aircraft_dataset" for r in records):
        raise ValueError("variable dataset schema or role differs")
    dates = []
    scenes = []
    for record in records:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError("variable dataset SHA differs")
        if set(dates) & set(record["dates"]):
            raise ValueError("merged mother UTC dates overlap")
        dates.extend(record["dates"])
        scenes.extend(load_variable_dataset(path).scenes)
    if len({scene.scene_id for scene in scenes}) != len(scenes):
        raise ValueError("merged scene IDs repeat")
    identifier = new_id("variable_scenes")
    directory = application.store.record_dir("scenes", identifier)
    directory.mkdir(parents=True)
    path = directory / "inputs.npz"
    metadata = {"partition_role": role, "source_dates": sorted(dates),
                "format_version": "variable_aircraft_v1",
                "source_datasets": [{"id": r["id"], "sha256": r["sha256"],
                                     "dates": r["dates"]} for r in records],
                "demand_multipliers_applied": [1.]}
    prepared = save_variable_dataset(CoverageDataset(scenes, metadata), path)
    record = {**prepared, "id": identifier, "name": body.get("name") or f"S1-V10 {role} combined",
              "path": str(path), "sha256": digest(path), "research_stage": "coverage",
              "schema_version": 5, "record_type": "variable_aircraft_dataset",
              "profile": "europe_v1", "partition_role": role,
              "dates": sorted(dates), "source": metadata,
              "created_at": now(), "count": len(scenes)}
    return application.store.save_record("scenes", record)


def _deployment(store):
    import getpass
    import os
    import sys
    import torch

    archive = store.archive_root.resolve()
    port = archive.name
    if (archive.parent != ARCHIVE_PARENT.resolve() or port not in LANES or
            os.environ.get("SATELLITE_RL_SSH_PORT") != port):
        raise ValueError("S1-V10 archive and SSH instance must match one approved lane")
    expected = Path("/home/u2024110224/first_stage_satellite_rl/envs/DQN")
    if (getpass.getuser() != "u2024110224" or Path(sys.prefix).resolve() != expected or
            not torch.cuda.is_available() or "4090 D" not in torch.cuda.get_device_name(0)):
        raise ValueError("S1-V10 requires remote DQN/CUDA/4090D")
    return port


def freeze_variable_preparation(source, destination):
    """Freeze small source contracts and parser code; retain full archive SHA."""
    destination = Path(destination)
    destination.mkdir(parents=True)
    frozen = deepcopy(source)
    for key in ("source_preflight", "selection_contract", "preparation_reference",
                "release_api", "preferred_releases", "license"):
        record = frozen[key]
        path = Path(record["path"])
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"S1-V10 {key} SHA differs")
        target = destination / path.name
        shutil.copyfile(path, target)
        if digest(target) != record["sha256"]:
            raise ValueError(f"S1-V10 frozen {key} SHA differs")
        record["source_path"], record["path"] = str(path), str(target)
    contract = json.loads(Path(frozen["selection_contract"]["path"]).read_text(encoding="utf-8"))
    times_sha = hashlib.sha256(json.dumps(contract["selected_times_utc"],
                              separators=(",", ":")).encode("utf-8")).hexdigest()
    if times_sha != frozen["selected_times_sha256"]:
        raise ValueError("S1-V10 frozen selected UTC times SHA differs")
    return frozen


def freeze_variable_dataset_contract(source, destination):
    destination = Path(destination)
    destination.mkdir(parents=True)
    path = Path(source["path"])
    if digest(path) != source["sha256"]:
        raise ValueError("S1-V10 canonical dataset contract changed before task freeze")
    target = destination / "canonical_dataset_contract.json"
    shutil.copyfile(path, target)
    if digest(target) != source["sha256"]:
        raise ValueError("S1-V10 frozen canonical dataset contract SHA differs")
    return {**source, "source_path": str(path), "path": str(target)}


def _canonical_contract(store):
    path = ARCHIVE_PARENT / "source" / "canonical_dataset_contract.json"
    if not path.is_file():
        raise ValueError("S1-V10 canonical dataset contract not registered")
    contract = json.loads(path.read_text(encoding="utf-8"))
    head = subprocess.run(["git", "-C", str(store.source_root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if (contract.get("experiment_id") != "S1-V10" or
            contract.get("plan_revision") != PLAN_REVISION or
            contract.get("scientific_commit") != head or
            set(contract.get("datasets", {})) != set(ROLE_DATES) or
            set(contract.get("source_dates", {})) != set(SOURCE_CONTRACT_SHA)):
        raise ValueError("S1-V10 canonical data contract revision/commit/roles differ")
    for role in ROLE_DATES:
        item = contract["datasets"][role]
        if (item.get("dates") != ROLE_DATES[role] or item.get("count") != ROLE_COUNTS[role] or
                not isinstance(item.get("sha256"), str)):
            raise ValueError(f"S1-V10 canonical {role} date/count/SHA differs")
    for day in SOURCE_CONTRACT_SHA:
        item = contract["source_dates"][day]
        if (item.get("selection_contract_sha256") != SOURCE_CONTRACT_SHA[day] or
                item.get("source_preflight_sha256") != SOURCE_PREFLIGHT_SHA[day] or
                not isinstance(item.get("archive_sha256"), str)):
            raise ValueError(f"S1-V10 canonical {day} original source differs")
    for role in ("validation", "sequence"):
        day = ROLE_DATES[role][0]
        if contract["datasets"][role]["sha256"] != contract["source_dates"][day]["prepared_dataset_sha256"]:
            raise ValueError(f"S1-V10 {role} prepared dataset SHA differs from source job")
    test_sources = contract["datasets"]["test"].get("source_datasets")
    if (not isinstance(test_sources, list) or len(test_sources) != 2 or
            {tuple(item.get("dates", [])): item.get("sha256") for item in test_sources} !=
            {(day,): contract["source_dates"][day]["prepared_dataset_sha256"]
             for day in ROLE_DATES["test"]}):
        raise ValueError("S1-V10 merged test sources differ from Feb04/05 prepared tasks")
    if contract.get("historical_source_sha256") != {
            "train": HISTORICAL_SOURCE_SHA["train"],
            "validation": HISTORICAL_SOURCE_SHA["development"]}:
        raise ValueError("S1-V10 historical train/development source SHA differs")
    return contract, {"path": str(path), "sha256": digest(path),
                      "plan_revision": PLAN_REVISION}


def _validate_dataset_lineage(role, record, contract):
    from satellite_rl.coverage.variable_dataset import load_variable_dataset

    dataset = load_variable_dataset(record["path"])
    metadata = dataset.metadata
    dates = sorted({d for scene in dataset.scenes for d in scene.metadata.get("source_dates", []) if d})
    if (len(dataset.scenes) != ROLE_COUNTS[role] or record.get("count") != ROLE_COUNTS[role] or
            record.get("dates") != ROLE_DATES[role] or dates != ROLE_DATES[role] or
            metadata.get("partition_role") != role):
        raise ValueError(f"S1-V10 {role} mother dates/count/partition differ from plan")
    if role in HISTORICAL_SOURCE_SHA:
        old_sha = metadata.get("source_frozen_npz_sha256")
        if old_sha != HISTORICAL_SOURCE_SHA[role]:
            raise ValueError(f"S1-V10 {role} historical aircraft source SHA differs")
    elif role == "test":
        if metadata.get("source_datasets") != contract["datasets"]["test"].get("source_datasets"):
            raise ValueError("S1-V10 Feb04+05 merge source IDs/SHA differ")
    else:
        day = ROLE_DATES[role][0]
        source = contract["source_dates"][day]
        if (metadata.get("selection_contract_sha256") != SOURCE_CONTRACT_SHA[day] or
                metadata.get("source_preflight_sha256") != SOURCE_PREFLIGHT_SHA[day] or
                metadata.get("source_archive", {}).get("sha256") != source["archive_sha256"]):
            raise ValueError(f"S1-V10 {role} official archive/selection lineage differs")


def _submit_preparation(application, body, config, port):
    role, day = body.get("dataset_role"), body.get("source_date")
    if role not in SOURCE_DATES or day not in SOURCE_DATES[role]:
        raise ValueError("S1-V10 source date and dataset role differ from frozen plan")
    source = ARCHIVE_PARENT / "source" / day
    pin_path, contract_path, status_path = (source / name for name in
                                            ("source_preflight.json", "selection_contract.json",
                                             "download_status.json"))
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    archive = source / pin["archive_name"]
    if (pin.get("group") != "S1-V10" or pin.get("role") != role or pin.get("date_utc") != day or
            contract.get("selected_times_utc") is None or
            digest(contract_path) != pin.get("selection_contract_sha256") or
            status.get("status") != "completed" or status.get("asset_id") != pin.get("asset_id") or
            status.get("bytes") != pin.get("expected_bytes") or
            not archive.is_file() or archive.stat().st_size != pin["expected_bytes"] or
            digest(archive) != status.get("sha256")):
        raise ValueError("S1-V10 frozen ADS-B source archive is incomplete or changed")
    if (digest(pin_path) != SOURCE_PREFLIGHT_SHA[day] or
            digest(contract_path) != SOURCE_CONTRACT_SHA[day]):
        raise ValueError("S1-V10 source selection differs from approved plan revision")
    reference = Path(__file__).resolve().parents[3] / "scripts" / "stage1_reconstruction" / "prepare_snapshots_reference.py"
    if not reference.is_file():
        raise ValueError("approved aircraft trace parser is missing")
    config["datasets"] = {}
    preparation = {
        "role": role, "date_utc": day,
        "selected_times_sha256": hashlib.sha256(json.dumps(
            contract["selected_times_utc"], separators=(",", ":")).encode("utf-8")).hexdigest(),
        "selected_time_count": contract["count"],
        "archive": {"path": str(archive), "sha256": status["sha256"],
                    "bytes": status["bytes"], "asset_id": pin["asset_id"]},
        "source_preflight": {"path": str(pin_path), "sha256": digest(pin_path)},
        "selection_contract": {"path": str(contract_path), "sha256": digest(contract_path)},
        "preparation_reference": {"path": str(reference), "sha256": digest(reference)},
        "release_api": {"path": str(source / "release_api.json"),
                        "sha256": pin["release_api_sha256"]},
        "preferred_releases": {"path": str(source / "PREFERRED_RELEASES.txt"),
                               "sha256": pin["preferred_sha256"]},
        "license": {"path": str(source / "LICENSE-ODbL-official.txt"),
                    "sha256": pin["license_sha256"]},
    }
    extra = {"research_stage": "coverage", "experiment_id": "S1-V10",
             "experiment_role": "formal", "variant": "prepare:" + role,
             "variable_route": "prepare", "dataset_role": role,
             "execution_target": f"u2024110224@10.160.17.102:{port}",
             "variable_preparation": preparation}
    return application.store.create_job(config, kind="prepare",
                                        request_id=body.get("request_id"), extra=extra)


def submit_variable(application, body, config):
    config = validate_variable_config(config)
    port = _deployment(application.store)
    route = config["coverage"]["variable"]["route"]
    kind = body.get("kind")
    if kind == "prepare":
        return _submit_preparation(application, body, config, port)
    if kind not in ("train", "evaluate") or (kind == "train" and route not in LEARNED):
        raise ValueError("only R08/R09 may train; all R01–R12 may evaluate")
    purpose = body.get("experiment_role", "formal")
    if purpose not in ("formal", "verification"):
        raise ValueError("experiment role must be formal or verification")
    contract, contract_ref = _canonical_contract(application.store)
    if kind == "train" and config["training"]["total_steps"] != (
            2048 if purpose == "verification" else 262144):
        raise ValueError("verification training is 2048 decisions; formal training is 262144")
    if kind != "train" and config["training"]["total_steps"] != 262144:
        raise ValueError("nontraining tasks retain the formal 262144-budget schema")
    role = body.get("dataset_role", "test") if kind == "evaluate" else None
    if kind == "evaluate" and role not in ("development", "validation", "test", "sequence"):
        raise ValueError("evaluation dataset role is invalid")
    if purpose == "formal" and kind == "evaluate" and role == "development":
        raise ValueError("development evaluation must be marked verification")
    if purpose == "verification" and kind == "evaluate" and role in ("test", "sequence"):
        raise ValueError("independent test and sequence dates cannot be engineering verification inputs")
    required = ({"train", "development"} if purpose == "verification" else
                {"train", "validation"}) if kind == "train" else {role}
    supplied = body.get("dataset_ids", {})
    if not isinstance(supplied, dict) or not required <= set(supplied):
        raise ValueError(f"missing variable dataset roles: {sorted(required - set(supplied))}")
    datasets = {}
    for dataset_role in required:
        record = application.store.get_record("scenes", supplied[dataset_role])
        if (record.get("research_stage") != "coverage" or record.get("schema_version") != 5 or
                record.get("partition_role") != dataset_role or
                record.get("record_type") != "variable_aircraft_dataset"):
            raise ValueError("dataset schema or partition role differs")
        if digest(Path(record["path"])) != record["sha256"]:
            raise ValueError("dataset SHA differs")
        if record["sha256"] != contract["datasets"][dataset_role]["sha256"]:
            raise ValueError(f"{dataset_role} dataset SHA differs from S1-V10 canonical contract")
        _validate_dataset_lineage(dataset_role, record, contract)
        datasets[dataset_role] = record
    for left, first in datasets.items():
        for right, second in datasets.items():
            if left < right and set(first["dates"]) & set(second["dates"]):
                raise ValueError("train/validation/test mother dates overlap")
    config["datasets"] = datasets
    extra = {"research_stage": "coverage", "experiment_id": "S1-V10",
             "experiment_role": purpose, "variant": route, "variable_route": route,
             "dataset_role": role, "execution_target": f"u2024110224@10.160.17.102:{port}",
             "variable_dataset_sha256": {r: d["sha256"] for r, d in datasets.items()},
             "variable_dataset_contract": contract_ref}
    if kind == "evaluate" and route in LEARNED:
        model = next((m for m in application.models() if m["id"] == body.get("model_id")), None)
        if not model or model["config"].get("schema_version") != 5:
            raise ValueError("R08/R09 evaluation requires an existing schema 5 model")
        trained = model["config"]
        if trained["coverage"] != config["coverage"] or trained["network"] != config["network"]:
            raise ValueError("evaluation physics and route differ from selected model")
        for prior in trained.get("datasets", {}).values():
            if set(prior.get("dates", [])) & set(datasets[role]["dates"]):
                raise ValueError("evaluation mother date overlaps model training or validation")
        model_dir = Path(model["path"])
        extra.update(model_path=str(model_dir),
                     model_files=[{"path": str(model_dir / name), "sha256": digest(model_dir / name)}
                                  for name in ("policy.pt", "model.json")])
    return application.store.create_job(config, kind=kind,
                                        request_id=body.get("request_id"), extra=extra)
