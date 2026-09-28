"""Strict, receipt-gated source migration for equivalent async telemetry.

This module never stops a worker or dispatches a job. The scheduler calls it
only while holding its lock and an active temporary dispatch hold.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import torch

from satellite_rl.training.run_monitor import atomic_json
from .checkpoints import config_hash
from .datasets import scene_set_files
from .recovery import verified_manifest
from .storage import digest, inside, new_id, now, read_json


UPGRADE_TYPE = "async_telemetry_v1"
OLD_COMMIT = "48022f4499823def40d573ea9436e3f7bdbf87fa"
TELEMETRY = {"mode": "async", "flush_interval_seconds": 2.0, "queue_capacity": 2048}
VERIFICATION_FOLDER = Path("outputs/verification/async_telemetry_20260929")
ALLOWED_CHANGED = {
    "src/satellite_rl/workbench/runner.py",
    "src/satellite_rl/workbench/telemetry.py",
    "src/satellite_rl/workbench/config.py",
    "src/satellite_rl/workbench/checkpoints.py",
    "src/satellite_rl/workbench/monitoring.py",
    "src/satellite_rl/workbench/learning_snapshots.py",
    "src/satellite_rl/workbench/static/app.js",
    "src/satellite_rl/agents/workbench_algorithms.py",
    "src/satellite_rl/workbench/runtime.py",
    "src/satellite_rl/workbench/service.py",
    "src/satellite_rl/workbench/performance_upgrade.py",
}
EQUIVALENCE_CHECKS = {"config_hash", "scene_hashes", "weights", "optimizer", "replay", "rng",
                      "actions", "reward", "metrics", "checkpoint_resume", "logs"}
BACKGROUND_RUNS = {
    "train_sac_20260929_022411_a4654185",
    "train_sac_20260929_022415_f5552810",
    "train_sac_20260929_022420_7df81644",
}
MEASURED_WORKER_FILES = {
    "src/satellite_rl/workbench/runner.py",
    "src/satellite_rl/workbench/telemetry.py",
    "src/satellite_rl/workbench/config.py",
    "src/satellite_rl/workbench/checkpoints.py",
    "src/satellite_rl/workbench/learning_snapshots.py",
    "src/satellite_rl/agents/workbench_algorithms.py",
}


def _hex_sha(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _hex_commit(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{40}", value))


def _read_receipt_file(path, folder):
    if not isinstance(path, (str, Path)):
        raise ValueError("性能升级凭据路径无效")
    path = inside(path, folder)
    if not path.is_file() or path.suffix.lower() != ".json" or path.stat().st_size > 1024 * 1024:
        raise ValueError("性能升级凭据必须是已授权验证目录中的小型JSON文件")
    record = read_json(path)
    if not isinstance(record, dict):
        raise ValueError("性能升级凭据JSON无效")
    return path, record


def _evidence(folder, reference, expected_type, old_commit, new_commit):
    if not isinstance(reference, dict) or not _hex_sha(reference.get("sha256")):
        raise ValueError("性能升级证据缺少文件SHA-256")
    path, record = _read_receipt_file(reference.get("path"), folder)
    if digest(path) != reference["sha256"]:
        raise ValueError("性能升级证据文件SHA-256不符")
    if (record.get("type") != expected_type or record.get("passed") is not True or
            record.get("old_commit") != old_commit or record.get("new_commit") != new_commit):
        raise ValueError("性能升级证据类型、版本或通过状态无效")
    return path, record


def _positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _verify_performance(record):
    if (record.get("measurement_mode") != "paired_interleaved_with_background" or
            record.get("same_workload") is not True or
            not isinstance(record.get("background_runs"), list) or
            len(record["background_runs"]) != len(BACKGROUND_RUNS) or
            set(record.get("background_runs", [])) != BACKGROUND_RUNS or
            not isinstance(record.get("layout_results"), dict) or
            set(record["layout_results"]) != {"13", "32"}):
        raise ValueError("性能证据缺少三任务背景下13/32令牌的配对同工作量比较")
    for layout, result in record["layout_results"].items():
        if not isinstance(result, dict) or result.get("passed") is not True:
            raise ValueError(f"{layout}令牌配对性能未通过")
        old, new = result.get("old_steps_per_second"), result.get("new_steps_per_second")
        if not _positive_number(old) or not _positive_number(new) or new < old * 1.05:
            raise ValueError(f"{layout}令牌吞吐提速未达到5%")


def verify_receipt(store, job, receipt_path, new_root, new_commit):
    """Verify frozen bytes and two hashed evidence files before touching a job."""
    folder = (Path(store.archive_root) / VERIFICATION_FOLDER).resolve()
    receipt_path, receipt = _read_receipt_file(receipt_path, folder)
    if (receipt.get("type") != UPGRADE_TYPE or receipt.get("status") != "passed" or
            receipt.get("old_commit") != OLD_COMMIT or receipt.get("old_commit") != job.get("git_commit") or
            receipt.get("new_commit") != new_commit or not _hex_commit(new_commit) or
            new_commit == OLD_COMMIT):
        raise ValueError("性能升级凭据类型、状态或源码提交不匹配")
    measured_commit = receipt.get("measured_worker_commit")
    worker_files = receipt.get("measured_worker_files")
    if not _hex_commit(measured_commit) or not isinstance(worker_files, dict) or \
            set(worker_files) != MEASURED_WORKER_FILES or not all(_hex_sha(value) for value in worker_files.values()):
        raise ValueError("性能升级凭据缺少实测worker版本与完整逐文件指纹")
    old_root = inside(job["source_root"], Path(store.root) / "source_snapshots")
    new_root = inside(new_root, Path(store.root) / "source_snapshots")
    old = verified_manifest(old_root, OLD_COMMIT)
    new = verified_manifest(new_root, new_commit)
    old_manifest_sha = digest(old_root / "source_manifest.json")
    new_manifest_sha = digest(new_root / "source_manifest.json")
    if (receipt.get("old_manifest_sha256") != old_manifest_sha or
            receipt.get("new_manifest_sha256") != new_manifest_sha):
        raise ValueError("性能升级凭据未绑定实际冻结源码清单")
    changed = {name for name in set(old["files"]) | set(new["files"])
               if old["files"].get(name) != new["files"].get(name)}
    if (not {"src/satellite_rl/workbench/runner.py", "src/satellite_rl/workbench/telemetry.py"}.issubset(changed)
            or not changed.issubset(ALLOWED_CHANGED)):
        raise ValueError("性能升级源码含未许可的环境、网络、数据或算法文件改动")
    entries = receipt.get("changed_files")
    if not isinstance(entries, list) or len(entries) != len(changed):
        raise ValueError("性能升级逐文件哈希清单不完整")
    declared = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "old_sha256", "new_sha256"}:
            raise ValueError("性能升级逐文件哈希项无效")
        name = entry["path"]
        if name in declared or name not in changed or entry["old_sha256"] != old["files"].get(name) or \
                entry["new_sha256"] != new["files"].get(name):
            raise ValueError("性能升级逐文件哈希与冻结源码不符")
        declared[name] = entry
    if set(declared) != changed:
        raise ValueError("性能升级逐文件哈希缺项")
    if any(new["files"].get(name) != expected for name, expected in worker_files.items()):
        raise ValueError("最终worker源码与配对实测版本的逐文件指纹不同")
    equivalence_path, equivalence = _evidence(folder, receipt.get("equivalence"),
        "async_telemetry_equivalence_v1", OLD_COMMIT, measured_commit)
    if equivalence.get("measured_worker_files") != worker_files:
        raise ValueError("数值等价证据未绑定实测worker逐文件指纹")
    if not isinstance(equivalence.get("checks"), dict) or not EQUIVALENCE_CHECKS.issubset(equivalence["checks"]) or \
            any(equivalence["checks"][key] is not True for key in EQUIVALENCE_CHECKS):
        raise ValueError("数值等价、日志或恢复证据未全部通过")
    performance_path, performance = _evidence(folder, receipt.get("performance"),
        "async_telemetry_performance_v1", OLD_COMMIT, measured_commit)
    if performance.get("measured_worker_files") != worker_files:
        raise ValueError("速度证据未绑定实测worker逐文件指纹")
    _verify_performance(performance)
    return {"upgrade_type": UPGRADE_TYPE, "old_commit": OLD_COMMIT, "new_commit": new_commit,
            "old_source_root": str(old_root), "new_source_root": str(new_root),
            "old_manifest_sha256": old_manifest_sha, "new_manifest_sha256": new_manifest_sha,
            "changed_files": [declared[name] for name in sorted(changed)],
            "receipt_path": str(receipt_path), "receipt_sha256": digest(receipt_path),
            "measured_worker_commit": measured_commit,
            "measured_worker_files": dict(worker_files),
            "equivalence_path": str(equivalence_path), "equivalence_sha256": digest(equivalence_path),
            "performance_path": str(performance_path), "performance_sha256": digest(performance_path),
            "measurement_mode": performance["measurement_mode"],
            "minimum_gain_percent": 5.0}


def verify_source_history_entry(entry):
    """Recheck a completed upgrade without current job/config/checkpoint state."""
    if not isinstance(entry, dict) or entry.get("upgrade_type") != UPGRADE_TYPE:
        raise ValueError("源码历史不是性能升级记录")
    receipt_path = Path(entry.get("receipt_path", "")).resolve()
    folder = receipt_path.parent
    if (folder.name != "async_telemetry_20260929" or folder.parent.name != "verification" or
            folder.parent.parent.name != "outputs"):
        raise ValueError("性能升级凭据不在指定验证目录")
    archive_root = folder.parent.parent.parent
    old_root = Path(entry.get("old_source_root", "")).resolve()
    new_root = Path(entry.get("new_source_root", "")).resolve()
    source_folder = archive_root / "outputs" / "workbench" / "source_snapshots"
    if old_root.parent != source_folder or new_root.parent != source_folder:
        raise ValueError("性能升级历史源码不在工作台冻结目录")
    store = SimpleNamespace(root=source_folder.parent, archive_root=archive_root)
    job = {"source_root": str(old_root), "git_commit": entry.get("old_commit")}
    verified = verify_receipt(store, job, receipt_path, new_root, entry.get("new_commit"))
    if any(entry.get(key) != value for key, value in verified.items()):
        raise ValueError("性能升级历史与重新核验的来源/证据不一致")
    archive = Path(entry.get("archive_path", "")).resolve()
    if not archive.is_relative_to(store.root / "jobs") or archive.parent.name != "upgrades":
        raise ValueError("性能升级归档不在对应工作台任务目录")
    if read_json(archive / "upgrade_manifest.json") != entry:
        raise ValueError("性能升级归档清单与历史记录不一致")
    archived_job = read_json(archive / "job.json")
    archived_config = read_json(archive / "config.json")
    if (not isinstance(archived_job, dict) or archived_job.get("git_commit") != entry["old_commit"] or
            archived_job.get("source_root") != entry["old_source_root"] or
            archive.parent.parent.name != archived_job.get("run_id")):
        raise ValueError("性能升级归档任务与旧源码身份不一致")
    if not isinstance(archived_config, dict) or digest(archive / "config.json") != entry.get("config_sha256") or \
            config_hash(archived_config) != entry.get("config_hash"):
        raise ValueError("性能升级归档配置SHA-256不一致")
    archived_scenes_sha = hashlib.sha256(json.dumps(archived_job.get("scene_sets", {}),
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if archived_scenes_sha != entry.get("scene_sets_sha256"):
        raise ValueError("性能升级归档场景集身份不一致")
    checkpoint = entry.get("checkpoint")
    if checkpoint is not None and digest(archive / "checkpoint.pt") != checkpoint.get("sha256"):
        raise ValueError("性能升级归档断点SHA-256不一致")
    return deepcopy(entry)


def _checkpoint_evidence(directory, config):
    path = Path(directory) / "checkpoint.pt"
    if not path.is_file():
        return None
    checksum = digest(path)
    # Trusted job-owned checkpoint only. Check the complete optimizer, replay
    # and RNG contract before allowing a source switch.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    agent = payload.get("agent", {})
    rng = payload.get("rng", {})
    progress = payload.get("progress", {})
    optimizers = agent.get("optimizers", {})
    if (payload.get("format") != "satellite_rl_workbench_checkpoint_v1" or
            payload.get("config_hash") != config_hash(config) or
            not isinstance(progress.get("env_step"), int) or
            not 0 <= progress["env_step"] <= config["training"]["total_steps"] or
            not isinstance(agent.get("update_step"), int) or
            not isinstance(optimizers, dict) or
            not {"actor_opt", "critic1_opt", "critic2_opt", "alpha_opt"}.issubset(optimizers) or
            not isinstance(agent.get("replay"), dict) or
            not isinstance(agent["replay"].get("rows"), list) or
            not agent.get("rng") or not all(rng.get(key) is not None for key in ("python", "numpy", "torch", "cuda"))):
        raise ValueError("性能升级要求完整的SAC断点及优化器/回放/随机状态")
    del payload
    return {"path": str(path), "sha256": checksum, "env_step": progress["env_step"],
            "update_step": agent["update_step"], "optimizer_names": sorted(optimizers),
            "replay_present": True, "rng_present": True}


def prepare_upgrade(store, job, receipt_path, new_root, new_commit):
    if job.get("kind") != "train" or not str(job.get("request_id", "")).startswith("g08-20260929-train-"):
        raise ValueError("性能升级仅适用于本次G08工作台训练任务")
    directory = store.job_dir(job["run_id"])
    config = read_json(directory / "config.json")
    if not isinstance(config, dict) or config.get("algorithm") != "sac":
        raise ValueError("性能升级缺少原SAC配置")
    original_hash = config_hash(config)
    upgraded_config = deepcopy(config)
    upgraded_config["telemetry"] = deepcopy(TELEMETRY)
    if config_hash(upgraded_config) != original_hash:
        raise ValueError("异步采集选项改变了科学配置哈希")
    proof = verify_receipt(store, job, receipt_path, new_root, new_commit)
    for record in job.get("scene_sets", {}).values():
        scene_set_files(record)
    checkpoint = _checkpoint_evidence(directory, config)
    proof.update(config_sha256=digest(directory / "config.json"), config_hash=original_hash,
                 scene_sets_sha256=hashlib.sha256(json.dumps(job.get("scene_sets", {}),
                     ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                 checkpoint=checkpoint, created_at=now())
    return upgraded_config, proof


def commit_upgrade(store, job, status, upgraded_config, proof):
    """Archive before publishing; roll job/config/status back on write failure."""
    directory = store.job_dir(job["run_id"])
    backup = inside(directory / "upgrades" / new_id(UPGRADE_TYPE), directory)
    backup.mkdir(parents=True)
    originals = {}
    for name in ("job.json", "config.json", "status.json", "checkpoint.pt"):
        source = directory / name
        if source.is_file():
            target = backup / name
            shutil.copy2(source, target)
            if digest(target) != digest(source):
                raise RuntimeError("性能升级备份校验失败：" + name)
            originals[name] = read_json(source) if name.endswith(".json") else None
    checkpoint = proof["checkpoint"]
    if checkpoint is not None and digest(directory / "checkpoint.pt") != checkpoint["sha256"]:
        raise ValueError("备份期间断点发生变化，拒绝升级")
    if digest(directory / "config.json") != proof["config_sha256"]:
        raise ValueError("备份期间配置发生变化，拒绝升级")
    proof = deepcopy(proof)
    proof["archive_path"] = str(backup)
    atomic_json(backup / "upgrade_manifest.json", proof)
    updated = deepcopy(job)
    updated.setdefault("original_git_commit", job["git_commit"])
    updated.setdefault("source_history", []).append(proof)
    updated.update(source_root=proof["new_source_root"], git_commit=proof["new_commit"],
                   telemetry=deepcopy(TELEMETRY), performance_upgrade=proof)
    new_status = {**status, "source_upgrade": UPGRADE_TYPE,
                  "source_upgrade_archive": str(backup), "heartbeat_at": now()}
    try:
        atomic_json(directory / "config.json", upgraded_config)
        atomic_json(directory / "job.json", updated)
        atomic_json(directory / "status.json", new_status)
    except Exception:
        for name in ("config.json", "job.json", "status.json"):
            if name in originals:
                atomic_json(directory / name, originals[name])
        raise
    return new_status
