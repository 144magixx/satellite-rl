"""Durable, versioned workbench records and frozen source snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import uuid
import zipfile

from satellite_rl.training.run_monitor import atomic_json

SOURCE_ROOT = Path(__file__).resolve().parents[3]
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,150}$")


def now():
    return datetime.now(timezone.utc).isoformat()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return clean_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_id(value):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError("无效的记录编号")
    return value


def new_id(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("文件路径超出允许的目录")
    return path


def model_input_files(path, legacy=False):
    source = Path(path).resolve()
    directory = source.parent if source.is_file() else source
    files = [source] if legacy else [directory / "policy.pt", directory / "model.json"]
    if any(not file.is_file() for file in files):
        raise ValueError("评价模型文件不完整")
    return [{"path": str(file), "sha256": digest(file)} for file in files]


class Store:
    def __init__(self, archive_root, source_root=SOURCE_ROOT):
        self.archive_root = Path(archive_root).resolve()
        self.source_root = Path(source_root).resolve()
        self.root = self.archive_root / "outputs" / "workbench"
        self.models_root = self.archive_root / "models" / "runs" / "workbench"
        self.lock = threading.RLock()
        for name in ("jobs", "configs", "scenes", "models", "source_snapshots"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)
        self._historical_cache = (0.0, {})

    def job_dir(self, run_id):
        return self.root / "jobs" / validate_id(run_id)

    def record_dir(self, category, identifier):
        if category not in ("configs", "scenes", "models"):
            raise ValueError("未知记录类型")
        return self.root / category / validate_id(identifier)

    def records(self, category):
        return [v for p in sorted((self.root / category).glob("*/record.json"))
                if (v := read_json(p)) is not None]

    def save_record(self, category, record):
        directory = self.record_dir(category, record["id"])
        directory.mkdir(exist_ok=True)
        atomic_json(directory / "record.json", clean_json(record))
        return record

    def get_record(self, category, identifier):
        record = read_json(self.record_dir(category, identifier) / "record.json")
        if record is None:
            raise FileNotFoundError("记录不存在")
        return record

    def jobs(self):
        result = []
        for path in (self.root / "jobs").glob("*/job.json"):
            job = read_json(path)
            if job:
                status = read_json(path.parent / "status.json", {})
                result.append((job, status))
        return sorted(result, key=lambda pair: pair[0].get("created_at", ""), reverse=True)

    def freeze_source(self):
        """Copy committed source/data once; mutable development files never run jobs."""
        with self.lock:
            def git(*args):
                result = subprocess.run(["git", *args], cwd=self.source_root,
                                        capture_output=True, check=True)
                return result.stdout
            dirty = git("status", "--porcelain", "--", "src", "data", "pyproject.toml")
            if dirty.strip():
                raise ValueError("训练源码有未提交改动，请先完成验证并提交；运行任务使用固定版本")
            commit = git("rev-parse", "HEAD").decode().strip()
            destination = self.root / "source_snapshots" / commit
            manifest = read_json(destination / "source_manifest.json")
            if manifest:
                for name, expected in manifest["files"].items():
                    file = inside(destination / name, destination)
                    if not file.is_file() or digest(file) != expected:
                        raise ValueError("冻结源码校验失败，请保留现场并检查：" + name)
                return destination, commit
            destination.mkdir(exist_ok=True)
            archive = git("archive", "--format=zip", commit, "src", "data", "pyproject.toml")
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                for item in bundle.infolist():
                    inside(destination / item.filename, destination)
                bundle.extractall(destination)
            files = {str(p.relative_to(destination)).replace("\\", "/"): digest(p)
                     for p in destination.rglob("*") if p.is_file()}
            atomic_json(destination / "source_manifest.json", {
                "git_commit": commit, "created_at": now(), "files": files,
                "source_root": str(self.source_root), "format_version": 1,
            })
            return destination, commit

    def create_job(self, config, *, kind="train", request_id=None, extra=None):
        if kind not in ("train", "evaluate", "refine", "prepare"):
            raise ValueError("未知任务类型")
        if kind == "prepare" and not (config.get("research_stage") == "s2p19_preparation" or
                (config.get("research_stage") == "coverage" and config.get("schema_version") == 5)):
            raise ValueError("准备任务须声明受校验的S2-P19准备类型")
        if request_id is not None:
            validate_id(request_id)
        request_payload = {"kind": kind, "config": config, "extra": extra or {}}
        request_hash = hashlib.sha256(json.dumps(request_payload, ensure_ascii=False,
                                                sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        with self.lock:
            if request_id:
                for job, _ in self.jobs():
                    if job.get("request_id") == request_id:
                        if job.get("request_hash") != request_hash:
                            raise ValueError("同一请求编号已用于不同的配置")
                        return job
            snapshot, commit = self.freeze_source()
            run_id = new_id(kind + "_" + config["algorithm"])
            directory = self.job_dir(run_id)
            directory.mkdir()
            model_dir = self.models_root / run_id
            model_dir.mkdir()
            frozen_extra = dict(extra or {})
            if config.get("research_stage") == "s2p19_preparation":
                from copy import deepcopy
                from .s2p19_service import freeze_preparation
                config = deepcopy(config)
                config["preparation"] = freeze_preparation(config["preparation"], directory / "input_sources")
            if kind == "prepare" and config.get("research_stage") == "coverage" and config.get("schema_version") == 5:
                from .coverage_variable import freeze_variable_preparation
                frozen_extra["variable_preparation"] = freeze_variable_preparation(
                    frozen_extra["variable_preparation"], directory / "input_sources")
            if kind in ("train", "evaluate") and config.get("research_stage") == "coverage" and config.get("schema_version") == 5:
                from .coverage_variable import freeze_variable_dataset_contract
                frozen_extra["variable_dataset_contract"] = freeze_variable_dataset_contract(
                    frozen_extra["variable_dataset_contract"], directory / "input_sources")
            is_coverage = config.get("research_stage") == "coverage"
            profile = (config["coverage"]["environment_profile"] if is_coverage else
                       config.get("environment", {}).get("profile", "china_legacy"))
            from satellite_rl.envs.regions import profile_metadata
            frozen_extra.update(profile=profile, environment_profile=profile_metadata(profile),
                                environment_version=profile_metadata(profile)["environment_version"])
            if frozen_extra.get("scene_sets"):
                from .datasets import freeze_scene_set
                frozen_sets = {role: freeze_scene_set(record, directory / "scene_inputs" / role, profile)
                               for role, record in frozen_extra["scene_sets"].items()}
                frozen_extra["scene_sets"] = frozen_sets
                for role, record in frozen_sets.items():
                    path_key = ("scene_path" if kind in ("evaluate", "refine") and role == "test" else
                                "training_scene_path" if role == "train" else role + "_scene_path")
                    frozen_extra[path_key] = record["path"]
            if is_coverage:
                from .coverage import freeze_balance_start, freeze_dataset
                config = dict(config)
                config["datasets"] = {role: freeze_dataset(record, directory / "scene_inputs" / role)
                                      for role, record in config["datasets"].items()}
                if config.get("schema_version") == 3:
                    config["balance_starts"] = {
                        role: freeze_balance_start(record, directory / "balance_starts" / role)
                        for role, record in config.get("balance_starts", {}).items()}
                elif config.get("schema_version") == 4:
                    config["reconstruction_starts"] = {
                        role: freeze_balance_start(record, directory / "reconstruction_starts" / role)
                        for role, record in config.get("reconstruction_starts", {}).items()}
                frozen_extra["research_stage"] = "coverage"
                frozen_extra["dataset_provenance"] = config["datasets"]
                if config.get("schema_version") == 3:
                    frozen_extra["balance_start_provenance"] = config["balance_starts"]
                elif config.get("schema_version") == 4:
                    frozen_extra["reconstruction_start_provenance"] = config["reconstruction_starts"]
            improvement_baseline = (config.get("research_stage") == "sac_improvement"
                                    and frozen_extra.get("mode") in ("gate", "baselines")
                                    and frozen_extra.get("experiment_id") == "S2-P19")
            source = frozen_extra.get("initial_source") or {}
            p19_saved_without_model = (kind == "refine" and
                                       frozen_extra.get("experiment_id") == "S2-P19" and
                                       source.get("experiment_id") == "S2-P19" and
                                       source.get("format") in ("s2p19_inline_validation_v1",
                                                                "s2p19_i0_case_v1") and
                                       source.get("group") in ("model", "greedy", "rule", "random") and
                                       bool(source.get("case_files")) and
                                       not frozen_extra.get("model_files"))
            if kind in ("evaluate", "refine") and not improvement_baseline and not p19_saved_without_model and (not is_coverage or config["coverage"]["method"] == "ppo"):
                model_target = directory / "model_input"
                model_target.mkdir()
                references = frozen_extra.get("model_files") or model_input_files(
                    frozen_extra["model_path"], bool(frozen_extra.get("legacy_type")))
                frozen_models = []
                for reference in references:
                    source = Path(reference["path"])
                    if digest(source) != reference["sha256"]:
                        raise ValueError("评价模型在提交期间发生变化")
                    name = "weights.pt" if frozen_extra.get("legacy_type") else source.name
                    destination = model_target / name
                    shutil.copyfile(source, destination)
                    if digest(destination) != reference["sha256"]:
                        raise ValueError("冻结评价模型校验失败")
                    frozen_models.append({"path": str(destination), "sha256": reference["sha256"],
                                          "source_path": str(source)})
                frozen_extra["model_source_path"] = frozen_extra["model_path"]
                frozen_extra["model_files"] = frozen_models
                frozen_extra["model_path"] = str(model_target / "weights.pt" if frozen_extra.get("legacy_type") else model_target)
            job = {"run_id": run_id, "kind": kind, "name": config.get("name", run_id),
                   "created_at": now(), "request_id": request_id,
                   "request_hash": request_hash,
                   "archive_root": str(self.archive_root), "model_dir": str(model_dir),
                   "source_root": str(snapshot), "git_commit": commit,
                   "algorithm": config["algorithm"], "network": config["network"]["kind"],
                   **frozen_extra}
            atomic_json(directory / "config.json", config)
            atomic_json(directory / "status.json", {
                "run_id": run_id, "stage": "queued", "created_at": now(),
                "env_step": 0, "update_step": 0, "episode": 0,
                "budget_env_steps": config["training"]["total_steps"],
                "heartbeat_at": now(), "queue_reason": "等待后台调度",
            })
            # Publish the discoverable job last so the scheduler cannot observe
            # a partially written configuration/status pair.
            atomic_json(directory / "job.json", job)
            return job
