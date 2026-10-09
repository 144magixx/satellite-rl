"""Exact same-ID S1-V10 CPU-supply recovery for the two formal PPO runs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil

from satellite_rl.coverage.config import config_hash
from satellite_rl.training.run_monitor import atomic_json
from .recovery import verified_manifest
from .runtime import pid_alive
from .storage import digest, inside, now, read_json


REPAIR_ID = "s1v10_cpu_supply_equivalent_v1"
ORIGIN_COMMIT = "b31187fcaf79e90b70d85db4cdfec5a4150020de"
DATA_CONTRACT_SHA256 = "912cb6081fa10620a12566abdc61c4f638e2bb8147b02942cc5d1b9c1bdd3cff"
TRAIN_SHA256 = "ab8ab07bf3a135e7b0c06954e8e0a7a8ca30b0739cc756dc1f99b3d055749a74"
VALIDATION_SHA256 = "244ba20fc2290b28c941f401e3880a63ba4f07ab9a59b66e4389c7664d92cc71"
TARGETS = {
    "train_ppo_20261009_095314_19a69288": {
        "port": "20418", "route": "R08", "request_id": "s1v10_train_R08_20418_s51001"},
    "train_ppo_20261009_095159_714e5c20": {
        "port": "20424", "route": "R09", "request_id": "s1v10_train_R09_20424_s51001"},
}
SOURCE_ALLOWLIST = {
    "src/satellite_rl/coverage/variable_design.py",
    "src/satellite_rl/coverage/variable_runner.py",
    "src/satellite_rl/workbench/variable_throughput_recovery.py",
    "src/satellite_rl/workbench/runtime.py",
    "src/satellite_rl/workbench/coverage_variable.py",
    "src/satellite_rl/workbench/service.py",
}
REQUIRED_CHANGED = {
    "src/satellite_rl/coverage/variable_design.py",
    "src/satellite_rl/coverage/variable_runner.py",
    "src/satellite_rl/workbench/variable_throughput_recovery.py",
    "src/satellite_rl/workbench/runtime.py",
    "src/satellite_rl/workbench/coverage_variable.py",
}
CRITERIA = {
    "constructor_1728_exact",
    "pool_packet_state_rng_equal",
    "trajectory_128_weight_optimizer_rng_equal",
    "actual_checkpoint_policy_optimizer_rng_32streams_complete",
    "same_input_action_gpu_update_equal",
    "ledger_prefix_complete",
    "no_budget_seed_data_change",
}
LEDGERS = ("actions.jsonl", "episodes.jsonl", "updates.jsonl", "evaluations.jsonl")


def _same_source_chain(old_root, new_root, new_commit):
    old = verified_manifest(old_root, ORIGIN_COMMIT)
    new = verified_manifest(new_root, new_commit)
    names = set(old["files"]) | set(new["files"])
    changed = {name for name in names if old["files"].get(name) != new["files"].get(name)}
    if not REQUIRED_CHANGED <= changed or changed - SOURCE_ALLOWLIST:
        raise ValueError("S1-V10 throughput repair source exceeds exact worker/control allowlist")
    return {"old_manifest_sha256": digest(Path(old_root) / "source_manifest.json"),
            "new_manifest_sha256": digest(Path(new_root) / "source_manifest.json"),
            "changed_files": sorted(changed),
            "source_file_sha256_before": {name: old["files"].get(name) for name in sorted(changed)},
            "source_file_sha256_after": {name: new["files"].get(name) for name in sorted(changed)}}


def _checkpoint_state(directory, job, config, status):
    import torch

    checkpoint = directory / "checkpoint.pt"
    checkpoint_sha = digest(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_hash = config_hash(config)
    step = status.get("env_step")
    if (payload.get("format_version") != 2 or payload.get("run_id") != job["run_id"] or
            payload.get("config_hash") != expected_hash or
            payload.get("source_commit") != ORIGIN_COMMIT or
            type(step) is not int or step <= 0 or step >= 262144 or step % 2048 or
            payload.get("step") != step or status.get("checkpoint_sha256") != checkpoint_sha or
            status.get("update_step") != payload.get("updates") or
            status.get("episode") != payload.get("episode") or
            len(payload.get("streams", [])) != 32):
        raise ValueError("S1-V10 stopped checkpoint provenance, budget or 32-stream state differs")
    policy, optimizer, rng = (payload.get(key) for key in ("policy", "optimizer", "rng"))
    if (not isinstance(policy, dict) or not policy or not isinstance(optimizer, dict) or
            not {"state", "param_groups"} <= set(optimizer) or
            not isinstance(optimizer["state"], dict) or not optimizer["state"] or
            not isinstance(optimizer["param_groups"], list) or not optimizer["param_groups"] or
            not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"} or
            any(rng[key] is None for key in rng)):
        raise ValueError("S1-V10 checkpoint lacks policy, optimizer or complete RNG")
    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return True
    if any(not torch.isfinite(value).all() for value in policy.values()
           if isinstance(value, torch.Tensor)) or not finite(optimizer):
        raise ValueError("S1-V10 checkpoint policy/optimizer contains nonfinite parameters")
    for index, stream in enumerate(payload["streams"]):
        if (not isinstance(stream, dict) or
                not {"sampler_state", "layout", "scene_index", "level", "episode_serial"} <= set(stream) or
                not isinstance(stream["sampler_state"], dict) or
                not isinstance(stream["layout"], dict)):
            raise ValueError(f"S1-V10 stream {index} lacks sampler/layout state")
    counts = payload.get("ledger_counts")
    if not isinstance(counts, dict) or set(counts) != set(LEDGERS):
        raise ValueError("S1-V10 checkpoint ledger count map differs")
    for name in LEDGERS:
        if type(counts[name]) is not int or counts[name] < 0:
            raise ValueError("S1-V10 checkpoint ledger count invalid")
        path = directory / name
        if path.is_file():
            with path.open("rb") as handle:
                lines = sum(1 for line in handle if line.strip())
        else:
            lines = 0
        if lines != counts[name]:
            raise ValueError(f"S1-V10 {name} has {lines} rows but checkpoint committed {counts[name]}")
    interval = config["training"]["validation_interval"]
    last_validation = payload.get("last_validation")
    if (type(last_validation) is not int or last_validation < 0 or
            last_validation > step or last_validation % interval):
        raise ValueError("S1-V10 checkpoint validation position differs")
    expected_nodes = list(range(interval, last_validation + 1, interval))
    found = sorted(directory.glob("validation_*.jsonl"),
                   key=lambda path: int(path.stem.removeprefix("validation_")))
    if [path.name for path in found] != [f"validation_{node}.jsonl" for node in expected_nodes]:
        raise ValueError("S1-V10 saved validation nodes include missing/future/partial files")
    summaries = []
    path = directory / "evaluations.jsonl"
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            summaries = [json.loads(line) for line in handle if line.strip()]
    if len(summaries) != len(expected_nodes):
        raise ValueError("S1-V10 validation summary count differs from committed nodes")
    validation_nodes = []
    common_scene_grid = None
    for node, file, summary in zip(expected_nodes, found, summaries):
        identities = set()
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = (row.get("scene_id"), row.get("scene_sha256"),
                            row.get("demand_multiplier"))
                if (not row.get("demand_conserved") or row.get("route") !=
                        config["coverage"]["variable"]["route"] or
                        not isinstance(identity[0], str) or
                        not isinstance(identity[1], str) or
                        identity[2] not in (.75, 1., 1.25) or
                        identity in identities):
                    raise ValueError("S1-V10 saved validation row/source grid differs")
                identities.add(identity)
        if (len(identities) != 60 or len({(a, b) for a, b, _ in identities}) != 20 or
                common_scene_grid is not None and identities != common_scene_grid):
            raise ValueError("S1-V10 validation node lacks the same 20 mothers × three demands")
        common_scene_grid = identities
        node_sha = digest(file)
        if (summary.get("env_step") != node or summary.get("scene_count") != 60 or
                summary.get("rows_sha256") != node_sha):
            raise ValueError("S1-V10 validation summary SHA/step differs from full rows")
        validation_nodes.append({"env_step": node, "rows": 60,
                                 "sha256": node_sha})
    return {"checkpoint_sha256": checkpoint_sha, "checkpoint_step": step,
            "checkpoint_updates": payload["updates"], "checkpoint_episode": payload["episode"],
            "config_hash": expected_hash, "stream_count": len(payload["streams"]),
            "ledger_counts": dict(counts), "last_validation": last_validation,
            "saved_validation_nodes": validation_nodes}


def _validation_receipt(store, directory, path, expected_sha, *, job, config, status,
                        checkpoint_state, new_commit):
    path = inside(path, directory / "recoveries")
    if not path.is_file() or digest(path) != expected_sha:
        raise ValueError("S1-V10 throughput validation receipt SHA/path differs")
    receipt = read_json(path, {})
    expected = TARGETS[job["run_id"]]
    actual = receipt.get("actual_checkpoint") or {}
    if (receipt.get("format") != "s1v10_throughput_repair_validation_v1" or
            receipt.get("repair_id") != REPAIR_ID or
            receipt.get("origin_commit") != ORIGIN_COMMIT or
            receipt.get("repair_commit") != new_commit or
            receipt.get("old_source_root") != job["source_root"] or
            receipt.get("new_source_root") != str(
                (store.root / "source_snapshots" / new_commit).resolve()) or
            receipt.get("run_id") != job["run_id"] or
            receipt.get("request_id") != expected["request_id"] or
            receipt.get("route") != expected["route"] or
            receipt.get("config_sha256") != digest(directory / "config.json") or
            receipt.get("config_hash") != checkpoint_state["config_hash"] or
            receipt.get("dataset_sha256") != {"train": TRAIN_SHA256,
                                               "validation": VALIDATION_SHA256} or
            receipt.get("stream_count") != 32 or
            receipt.get("saved_validation_nodes") != checkpoint_state["saved_validation_nodes"] or
            actual != {"sha256": checkpoint_state["checkpoint_sha256"],
                       "step": checkpoint_state["checkpoint_step"],
                       "updates": checkpoint_state["checkpoint_updates"],
                       "episode": checkpoint_state["checkpoint_episode"]} or
            receipt.get("tests_passed") is not True or
            receipt.get("approved") is not True or
            set(receipt.get("criteria") or {}) != CRITERIA or
            any(receipt["criteria"][key] is not True for key in CRITERIA)):
        raise ValueError("S1-V10 throughput validation is not bound to this stopped checkpoint")
    files = receipt.get("evidence_files")
    if not isinstance(files, list) or len(files) < 5:
        raise ValueError("S1-V10 throughput validation lacks independent raw evidence")
    for item in files:
        evidence = inside(item["path"], store.archive_root)
        if not evidence.is_file() or digest(evidence) != item["sha256"]:
            raise ValueError("S1-V10 throughput evidence path or SHA differs")
    return receipt


def prepare_variable_throughput_recovery(store, job, validation_path,
                                         validation_sha256, expected_repair_commit):
    """Archive an exact stopped checkpoint and rebind only its future worker."""
    run_id = job.get("run_id")
    expected = TARGETS.get(run_id)
    if (not expected or job.get("request_id") != expected["request_id"] or
            job.get("kind") != "train" or job.get("research_stage") != "coverage" or
            job.get("experiment_id") != "S1-V10" or
            job.get("experiment_role") != "formal" or
            job.get("variable_route") != expected["route"] or
            job.get("execution_target") != f"u2024110224@10.160.17.102:{expected['port']}" or
            store.archive_root.name != expected["port"] or
            job.get("archive_root") != str(store.archive_root) or
            job.get("git_commit") != ORIGIN_COMMIT or
            not isinstance(expected_repair_commit, str) or
            not re.fullmatch(r"[0-9a-f]{40}", expected_repair_commit) or
            expected_repair_commit == ORIGIN_COMMIT):
        raise ValueError("S1-V10 throughput repair permits only two exact original formal PPO jobs")
    directory = store.job_dir(run_id)
    status = read_json(directory / "status.json", {})
    config = read_json(directory / "config.json", {})
    launch = read_json(directory / "launch.json", {})
    dataset = config.get("datasets") or {}
    if (config.get("schema_version") != 5 or config.get("research_stage") != "coverage" or
            config.get("coverage", {}).get("variable", {}).get("route") != expected["route"] or
            config.get("training", {}).get("seed") != 51001 or
            config.get("training", {}).get("total_steps") != 262144 or
            config.get("training", {}).get("num_envs") != 32 or
            config.get("training", {}).get("device") != "cuda" or
            set(dataset) != {"train", "validation"} or
            dataset["train"].get("sha256") != TRAIN_SHA256 or
            dataset["validation"].get("sha256") != VALIDATION_SHA256 or
            job.get("variable_dataset_sha256") != {
                "train": TRAIN_SHA256, "validation": VALIDATION_SHA256} or
            job.get("variable_dataset_contract", {}).get("sha256") != DATA_CONTRACT_SHA256 or
            digest(job["variable_dataset_contract"]["path"]) != DATA_CONTRACT_SHA256):
        raise ValueError("S1-V10 frozen input/seed/decision budget differs")
    if (status.get("stage") != "stopped" or status.get("stop_reason") != "requested" or
            launch.get("pid") != status.get("pid") or
            not isinstance(launch.get("process_start_time"), (int, float)) or
            pid_alive(status.get("pid"), launch["process_start_time"])):
        raise ValueError("S1-V10 original worker has not stopped cleanly at a checkpoint")
    old_root = inside(job["source_root"], store.root / "source_snapshots")
    state = _checkpoint_state(directory, job, config, status)
    receipt = _validation_receipt(store, directory, validation_path,
                                  validation_sha256, job=job, config=config,
                                  status=status, checkpoint_state=state,
                                  new_commit=expected_repair_commit)
    new_root, new_commit = store.freeze_source()
    if new_commit != expected_repair_commit:
        raise ValueError("S1-V10 workbench source is not the reviewed repair commit")
    source = _same_source_chain(old_root, new_root, new_commit)
    archive = inside(directory / "recoveries" /
                     f"{REPAIR_ID}_{new_commit[:12]}", directory)
    original = archive / "original"
    file_names = ["checkpoint.pt", "config.json", "job.json", "status.json",
                  "launch.json", "worker.log"]
    file_names += [name for name in LEDGERS if (directory / name).is_file()]
    file_names += [path.name for path in sorted(directory.glob("validation_*.jsonl"))]
    file_names += ["events.jsonl"] if (directory / "events.jsonl").is_file() else []
    if archive.exists():
        proof = read_json(archive / "repair_proof.json", {})
        if (not proof or proof.get("validation_receipt_sha256") != validation_sha256 or
                proof.get("checkpoint_sha256") != state["checkpoint_sha256"] or
                proof.get("repair_commit") != new_commit or
                proof.get("run_id") != run_id or
                set(proof.get("backup_files_sha256") or {}) != set(file_names) or
                any(digest(original / name) != proof["backup_files_sha256"][name]
                    for name in file_names)):
            raise ValueError("S1-V10 existing recovery archive is incomplete or differs")
    else:
        original.mkdir(parents=True)
        backup = {}
        for name in file_names:
            src = inside(directory / name, directory)
            dst = original / name
            if not src.is_file():
                raise ValueError("S1-V10 old frozen artifact missing: " + name)
            shutil.copy2(src, dst)
            if digest(src) != digest(dst):
                raise RuntimeError("S1-V10 original recovery backup SHA differs: " + name)
            backup[name] = digest(src)
        shutil.copy2(validation_path, archive / "validation_receipt.json")
        if digest(archive / "validation_receipt.json") != validation_sha256:
            raise RuntimeError("S1-V10 archived validation receipt SHA differs")
        proof = {**source, **state,
                 "repair_id": REPAIR_ID, "approved": True,
                 "origin_commit": ORIGIN_COMMIT, "repair_commit": new_commit,
                 "run_id": run_id, "request_id": expected["request_id"],
                 "route": expected["route"], "at": now(),
                 "archive_path": str(archive),
                 "old_source_root": str(old_root), "new_source_root": str(new_root),
                 "config_sha256": digest(directory / "config.json"),
                 "validation_receipt_sha256": validation_sha256,
                 "validation_receipt_path": str(validation_path),
                 "backup_files_sha256": backup,
                 "evidence_files": receipt["evidence_files"],
                 "scope": "same run/request/seed/262144 total decisions and frozen aircraft inputs; only future CPU supply changes"}
        atomic_json(archive / "repair_proof.json", proof)
    updated = deepcopy(job)
    updated.setdefault("original_git_commit", ORIGIN_COMMIT)
    updated.setdefault("source_history", []).append(proof)
    updated.update(source_root=str(new_root), git_commit=new_commit,
                   variable_throughput_repair=proof,
                   scientific_origin_commit=ORIGIN_COMMIT)
    return updated
