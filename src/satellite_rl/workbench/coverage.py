"""First-stage API using the same durable store and persistent scheduler."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import getpass
import os
from pathlib import Path
import shutil
import sys

from satellite_rl.coverage.config import (balance_default_config, config_hash, default_config,
                                          summary, validate_config)
from satellite_rl.coverage.reconstruction_config import reconstruction_default_config
from satellite_rl.workbench.storage import digest, inside, new_id, now, read_json, validate_id
from satellite_rl.training.run_monitor import atomic_json


RECONSTRUCTION_ARCHIVE_PARENT = Path("/home/u2024110224/first_stage_satellite_rl/s1_03")


def _reconstruction_target(store):
    archive = store.archive_root.resolve()
    port = archive.name
    if (archive.parent != RECONSTRUCTION_ARCHIVE_PARENT or port not in ("20418", "20424") or
            os.environ.get("SATELLITE_RL_SSH_PORT") != port):
        raise ValueError("S1-03实例目录和启动时固定SSH端口必须一致：20418或20424")
    return f"u2024110224@10.160.17.102:{port}"


def freeze_dataset(record, destination):
    source = Path(record["path"]).resolve()
    expected = record["sha256"]
    if not source.is_file() or digest(source) != expected:
        raise ValueError("第一阶段输入数据校验失败")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    shutil.copyfile(source, target)
    if digest(target) != expected:
        raise ValueError("冻结第一阶段输入SHA不一致")
    provenance = source.parent / "record.json"
    if not provenance.is_file():
        raise ValueError("覆盖数据缺少record.json来源记录")
    shutil.copyfile(provenance, destination / "record.json")
    return {**deepcopy(record), "path": str(target), "source_path": str(source),
            "provenance_sha256": digest(destination / "record.json")}


def freeze_balance_start(record, destination):
    source = Path(record["path"]).resolve()
    if not source.is_file() or digest(source) != record["sha256"]:
        raise ValueError("A2起点来源文件或SHA不一致")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "starts.jsonl"
    shutil.copyfile(source, target)
    if digest(target) != record["sha256"]:
        raise ValueError("冻结A2起点SHA不一致")
    return {**deepcopy(record), "path": str(target), "source_path": str(source)}


def _same_balance_preparation(left, right):
    """A2 preparation is reusable across routes/menus, not across physics or budgets."""
    def canonical(coverage):
        value = deepcopy(coverage)
        value["method"] = "none"
        value["balance"]["route"] = "prepare"
        value["balance"]["action_menu"] = "X0"
        return value
    return canonical(left) == canonical(right)


def _require_remote_dqn():
    import torch

    expected_prefix = Path("/home/u2024110224/first_stage_satellite_rl/envs/DQN")
    if (Path(sys.prefix).resolve() != expected_prefix or getpass.getuser() != "u2024110224" or
            not torch.cuda.is_available() or "4090 D" not in torch.cuda.get_device_name(0)):
        raise ValueError("第一阶段请连接已批准的远端DQN/4090D工作台；本机仅保留开发源码")


class CoverageApplication:
    def __init__(self, application):
        self.app = application
        self.store = application.store

    def datasets(self):
        return [r for r in self.store.records("scenes") if r.get("research_stage") == "coverage"]

    def models(self):
        rows = []
        for path in self.store.models_root.glob("*/final/model.json"):
            metadata = read_json(path, {})
            if metadata.get("research_stage") != "coverage" or not (path.parent / "policy.pt").is_file():
                continue
            run_id = path.parent.parent.name
            config = read_json(self.store.job_dir(run_id) / "config.json", {})
            rows.append({"id": run_id, "name": config.get("name", run_id),
                         "path": str(path.parent), "metadata": metadata,
                         "config": config, "sha256": digest(path.parent / "policy.pt")})
        for record in self.store.records("models"):
            if record.get("record_type") == "reconstruction_imported_model":
                path = Path(record["path"])
                if (path / "policy.pt").is_file() and digest(path / "policy.pt") == record["sha256"]:
                    rows.append({"id": record["id"], "name": record["name"],
                                 "path": str(path), "metadata": record["metadata"],
                                 "config": record["config"], "sha256": record["sha256"],
                                 "source_run_id": record["source_run_id"],
                                 "source_target": record["source_target"]})
        return rows

    def import_dataset(self, body):
        from satellite_rl.coverage.scenes import load_dataset, save_dataset
        role = body.get("partition_role")
        if role not in ("train", "validation", "test"):
            raise ValueError("请明确数据用途：训练/验证/考核")
        config = (reconstruction_default_config() if "reconstruction" in body.get("coverage", {}) else
                  balance_default_config() if "balance" in body.get("coverage", {}) else default_config())
        config["coverage"].update(body.get("coverage", {}))
        config = validate_config(config)
        source = Path(body["path"]).expanduser().resolve()
        dataset = load_dataset(source, config["coverage"]["initialization"], config["coverage"])
        if dataset.metadata.get("partition_role") not in (None, role):
            raise ValueError("数据来源与所选用途不一致")
        dataset.metadata["partition_role"] = role
        dates = sorted(set(dataset.metadata.get("source_dates", [])) |
                       {d for s in dataset.scenes for d in s.metadata.get("source_dates", []) if d})
        if not dates:
            raise ValueError("航空数据缺少明确UTC来源日期")
        identifier = new_id("coverage_scenes")
        directory = self.store.record_dir("scenes", identifier)
        prepared = save_dataset(dataset, directory / "inputs.npz")
        record = {**prepared, "id": identifier, "name": body.get("name") or source.name,
                  "path": str(directory / "inputs.npz"), "sha256": prepared["prepared_npz_sha256"],
                  "research_stage": "coverage", "profile": "europe_v1", "created_at": now(),
                  "partition_role": role, "dates": dates,
                  "source": {**dataset.metadata, "partition_role": role, "source_dates": dates},
                  "source_path": str(source),
                  "failed_initializations": sum(not s.metadata.get("initialization_valid", True)
                                                for s in dataset.scenes)}
        return self.store.save_record("scenes", record)

    def submit(self, body):
        config = validate_config(body["config"])
        if config["schema_version"] == 4:
            return self._submit_reconstruction(body, config)
        balanced = config["schema_version"] == 3
        route = config["coverage"].get("balance", {}).get("route") if balanced else None
        kind = body.get("kind", "train")
        if kind not in ("train", "evaluate"):
            raise ValueError("未知第一阶段任务类型")
        if config["training"]["device"] != "cuda":
            raise ValueError("正式第一阶段任务必须在指定远端DQN/CUDA执行")
        _require_remote_dqn()
        if kind == "train" and config["coverage"]["method"] != "ppo":
            raise ValueError("规则与局部搜索是评价基线，不是训练任务")
        if balanced and kind == "train" and route != "ppo":
            raise ValueError("平衡PPO训练须选PPO路线")
        dataset_role = body.get("dataset_role", "test") if balanced else "test"
        if dataset_role not in ("train", "validation", "test"):
            raise ValueError("数据用途未知")
        purpose = body.get("experiment_role", "formal")
        if purpose not in ("formal", "verification"):
            raise ValueError("任务用途须为正式实验或工程验证")
        if (kind == "evaluate" and route != "prepare" and dataset_role != "test" and
                not (balanced and purpose == "verification" and dataset_role == "validation")):
            raise ValueError("正式评价必须使用独立考核场景；开发筛选须标工程验证")
        required = {"train", "validation"} if kind == "train" else {dataset_role}
        supplied = body.get("dataset_ids", {})
        if not required <= set(supplied):
            raise ValueError("第一阶段任务缺少所需用途数据集")
        datasets = {}
        for role, identifier in supplied.items():
            if role not in (required if balanced else ("train", "validation", "test")) or not identifier:
                continue
            record = self.store.get_record("scenes", identifier)
            if record.get("research_stage") != "coverage" or record.get("partition_role") != role:
                raise ValueError("场景阶段或用途不一致")
            if record.get("initialization") != config["coverage"]["initialization"]:
                raise ValueError("请选择与本组基础布局方法一致的预处理数据")
            if (record.get("initialization") == "cluster" and
                    record.get("cluster_iterations") != config["coverage"]["cluster_iterations"]):
                raise ValueError("聚类配置已变化，请重新准备相应数据集")
            datasets[role] = record
        for left, a in datasets.items():
            for right, b in datasets.items():
                if left < right and set(a["dates"]) & set(b["dates"]):
                    raise ValueError("训练/验证/考核航空日期必须隔离")
        config["datasets"] = datasets
        if balanced and route != "prepare":
            start_ids = body.get("start_run_ids") or {}
            missing = required - set(start_ids)
            if missing:
                raise ValueError(f"缺少{sorted(missing)}对应的工作台A2预处理任务")
            starts = {}
            for role in required:
                identifier = start_ids[role]
                source_job = read_json(self.store.job_dir(identifier) / "job.json", {})
                source_status = read_json(self.store.job_dir(identifier) / "status.json", {})
                source_config = read_json(self.store.job_dir(identifier) / "config.json", {})
                source_artifacts = read_json(self.store.job_dir(identifier) / "artifacts.json", {})
                if (source_job.get("research_stage") != "coverage" or
                        source_status.get("stage") != "completed" or
                        source_config.get("schema_version") != 3 or
                        source_config.get("coverage", {}).get("balance", {}).get("route") != "prepare" or
                        not _same_balance_preparation(source_config.get("coverage", {}),
                                                      config["coverage"]) or
                        source_config.get("datasets", {}).get(role, {}).get("sha256") != datasets[role]["sha256"]):
                    raise ValueError(f"{role}的A2预处理任务未完成或场景不匹配")
                path = Path(source_artifacts.get("balance_starts", ""))
                sha = source_artifacts.get("balance_starts_sha256")
                if not path.is_file() or digest(path) != sha:
                    raise ValueError(f"{role}的A2起点文件/SHA未通过核验")
                starts[role] = {"path": str(path), "sha256": sha,
                                "dataset_sha256": datasets[role]["sha256"],
                                "source_run_id": identifier}
            config["balance_starts"] = starts
        variant = ((route + ":" + config["coverage"]["balance"]["action_menu"]) if balanced else
                   ("P0" if config["coverage"]["initialization"] == "empty" else
                   ("R" if config["coverage"]["initialization"] == "rule" else "C") +
                   {"none":"0", "local_search":"1", "ppo":"2"}[config["coverage"]["method"]]))
        extra = {"research_stage": "coverage", "experiment_role": purpose, "variant": variant,
                 "execution_target": "u2024110224@10.160.17.102:20418"}
        if balanced:
            extra.update(balance_route=route, balance_action_menu=config["coverage"]["balance"]["action_menu"],
                         dataset_role=dataset_role if kind == "evaluate" else None,
                         balance_dataset_sha256={role: record["sha256"] for role, record in datasets.items()},
                         balance_start_run_ids=body.get("start_run_ids") or {})
        if kind == "evaluate" and config["coverage"]["method"] == "ppo":
            model = next((m for m in self.models() if m["id"] == body.get("model_id")), None)
            if not model:
                raise ValueError("PPO评价须选择已有第一阶段模型")
            for role in ("train", "validation"):
                prior = model["config"].get("datasets", {}).get(role, {})
                if role != dataset_role and set(prior.get("dates", [])) & set(datasets[dataset_role]["dates"]):
                    raise ValueError("考核与模型训练/验证日期重叠")
            if model["config"].get("schema_version") != config["schema_version"] or model["config"].get("coverage") != config["coverage"]:
                raise ValueError("评价物理、初始化和微调配置须与模型一致")
            extra.update(model_path=model["path"],
                         model_files=[{"path": str(Path(model["path"]) / name),
                                       "sha256": digest(Path(model["path"]) / name)}
                                      for name in ("policy.pt", "model.json")])
        return self.store.create_job(config, kind=kind, request_id=body.get("request_id"), extra=extra)

    def _reconstruction_source(self, source_target, source_run_id):
        target = _reconstruction_target(self.store)
        if source_target not in ("20418", "20424") or target.endswith(":" + source_target):
            raise ValueError("只允许从另一已批准S1-03实例只读导入")
        root = RECONSTRUCTION_ARCHIVE_PARENT / source_target
        directory = root / "outputs" / "workbench" / "jobs" / validate_id(source_run_id)
        return root, directory

    def import_reconstruction_start(self, body):
        _require_remote_dqn()
        target = _reconstruction_target(self.store)
        role = body.get("partition_role")
        if role not in ("train", "validation", "test"):
            raise ValueError("共同初始化导入须明确数据用途")
        source_root, source_dir = self._reconstruction_source(body.get("source_target"), body.get("source_run_id"))
        job = read_json(source_dir / "job.json", {})
        status = read_json(source_dir / "status.json", {})
        config = read_json(source_dir / "config.json", {})
        artifacts = read_json(source_dir / "artifacts.json", {})
        if (job.get("research_stage") != "coverage" or job.get("execution_target") !=
                f"u2024110224@10.160.17.102:{body['source_target']}" or
                status.get("stage") != "completed" or config.get("schema_version") != 4 or
                config.get("coverage", {}).get("reconstruction", {}).get("route") != "prepare" or
                role not in config.get("datasets", {})):
            raise ValueError("来源须为另一实例已完成的S1-03共同初始化任务")
        source = inside(Path(artifacts.get("reconstruction_starts", "")), source_dir)
        sha = artifacts.get("reconstruction_starts_sha256")
        if not source.is_file() or not isinstance(sha, str) or digest(source) != sha:
            raise ValueError("来源共同初始化文件与SHA不一致")
        identifier = new_id("reconstruction_start_import")
        directory = self.store.record_dir("scenes", identifier)
        directory.mkdir(parents=True)
        destination = directory / "starts.jsonl"
        shutil.copyfile(source, destination)
        if digest(destination) != sha:
            raise ValueError("跨实例共同初始化导入SHA不一致")
        return self.store.save_record("scenes", {
            "id": identifier, "record_type": "reconstruction_imported_start",
            "name": f"{body['source_run_id']} · {role}", "path": str(destination),
            "sha256": sha, "dataset_sha256": config["datasets"][role]["sha256"],
            "partition_role": role, "source_run_id": body["source_run_id"],
            "source_target": job["execution_target"], "import_target": target,
            "source_path": str(source), "created_at": now(),
            "source_config": config,
        })

    def import_reconstruction_dataset(self, body):
        _require_remote_dqn()
        target = _reconstruction_target(self.store)
        source_target = body.get("source_target")
        if source_target not in ("20418", "20424") or target.endswith(":" + source_target):
            raise ValueError("只允许从另一已批准S1-03实例只读导入数据")
        source_id = validate_id(body.get("source_dataset_id"))
        source_root = RECONSTRUCTION_ARCHIVE_PARENT / source_target
        source_record_path = source_root / "outputs" / "workbench" / "scenes" / source_id / "record.json"
        source_record = read_json(source_record_path, {})
        role = body.get("partition_role")
        if (role not in ("train", "validation", "test") or
                source_record.get("research_stage") != "coverage" or
                source_record.get("partition_role") != role or
                source_record.get("initialization") != "cluster" or
                source_record.get("cluster_iterations") != 12):
            raise ValueError("来源须为匹配分区的欧洲版S1-03聚类兼容数据")
        source = inside(source_record.get("path", ""), source_root)
        if not source.is_file() or digest(source) != source_record.get("sha256"):
            raise ValueError("来源数据不在批准归档内或SHA不一致")
        identifier = new_id("reconstruction_dataset_import")
        directory = self.store.record_dir("scenes", identifier)
        directory.mkdir(parents=True)
        destination = directory / "inputs.npz"
        shutil.copyfile(source, destination)
        if digest(destination) != source_record["sha256"]:
            raise ValueError("跨实例数据导入SHA不一致")
        record = {**deepcopy(source_record), "id": identifier,
                  "name": source_record.get("name", source_id) + " · 跨实例",
                  "path": str(destination), "source_path": str(source),
                  "source_record_sha256": digest(source_record_path),
                  "source_dataset_id": source_id,
                  "source_target": f"u2024110224@10.160.17.102:{source_target}",
                  "import_target": target, "created_at": now()}
        return self.store.save_record("scenes", record)

    def import_reconstruction_model(self, body):
        _require_remote_dqn()
        target = _reconstruction_target(self.store)
        _, source_dir = self._reconstruction_source(body.get("source_target"), body.get("source_run_id"))
        job = read_json(source_dir / "job.json", {})
        status = read_json(source_dir / "status.json", {})
        config = read_json(source_dir / "config.json", {})
        source_model = inside(RECONSTRUCTION_ARCHIVE_PARENT / body["source_target"] /
                              "models" / "runs" / "workbench" / body["source_run_id"] / "final",
                              RECONSTRUCTION_ARCHIVE_PARENT / body["source_target"])
        if (job.get("research_stage") != "coverage" or job.get("execution_target") !=
                f"u2024110224@10.160.17.102:{body['source_target']}" or
                status.get("stage") != "completed" or config.get("schema_version") != 4 or
                job.get("kind") != "train" or
                config.get("coverage", {}).get("reconstruction", {}).get("route") != "P1"):
            raise ValueError("来源须为另一实例已完成的S1-03 PPO训练任务")
        metadata = read_json(source_model / "model.json", {})
        if (metadata.get("research_stage") != "coverage" or metadata.get("schema_version") != 4 or
                metadata.get("run_id") != job.get("run_id") or
                metadata.get("execution_target") != job.get("execution_target") or
                metadata.get("source_commit") != job.get("git_commit") or
                metadata.get("env_step") != config.get("training", {}).get("total_steps") or
                not (source_model / "policy.pt").is_file()):
            raise ValueError("来源PPO模型元数据或权重缺失")
        hashes = {name: digest(source_model / name) for name in ("policy.pt", "model.json")}
        if (metadata.get("policy_sha256") != hashes["policy.pt"] or
                metadata.get("config_hash") != config_hash(config)):
            raise ValueError("来源PPO权重SHA与模型元数据不一致")
        identifier = new_id("reconstruction_model_import")
        destination = self.store.record_dir("models", identifier)
        destination.mkdir(parents=True)
        for name, expected in hashes.items():
            shutil.copyfile(source_model / name, destination / name)
            if digest(destination / name) != expected:
                raise ValueError("跨实例模型导入SHA不一致")
        return self.store.save_record("models", {
            "id": identifier, "record_type": "reconstruction_imported_model",
            "name": f"{body['source_run_id']} · S1-03 PPO", "path": str(destination),
            "sha256": hashes["policy.pt"], "model_json_sha256": hashes["model.json"],
            "source_run_id": body["source_run_id"], "source_target": job["execution_target"],
            "import_target": target, "source_path": str(source_model),
            "config": config, "metadata": metadata, "created_at": now(),
        })

    def _submit_reconstruction(self, body, config):
        kind = body.get("kind", "train")
        route = config["coverage"]["reconstruction"]["route"]
        role = body.get("dataset_role", "test")
        purpose = body.get("experiment_role", "formal")
        if kind not in ("train", "evaluate") or purpose not in ("formal", "verification") or role not in ("train", "validation", "test"):
            raise ValueError("S1-03任务类型、用途或数据分区未知")
        if config["training"]["device"] != "cuda":
            raise ValueError("S1-03工作台正式任务必须使用DQN/CUDA")
        _require_remote_dqn()
        target = _reconstruction_target(self.store)
        if body.get("execution_target") not in (None, target):
            raise ValueError("执行目标由当前部署实例确定，不接受请求伪造")
        if kind == "train" and route != "P1":
            raise ValueError("只有P1路线可训练；其余路线提交评价任务")
        if kind == "evaluate" and route != "prepare" and role != "test" and not (purpose == "verification" and role == "validation"):
            raise ValueError("正式区域重构评价须用独立考核场景，开发筛选只允许验证分区")
        required = {"train", "validation"} if kind == "train" else {role}
        supplied = body.get("dataset_ids") or {}
        if not isinstance(supplied, dict) or not required <= set(supplied):
            raise ValueError("S1-03缺少所需分区数据集")
        datasets = {}
        for partition in required:
            record = self.store.get_record("scenes", supplied[partition])
            if (record.get("research_stage") != "coverage" or record.get("partition_role") != partition or
                    record.get("initialization") != "cluster" or
                    record.get("cluster_iterations") != config["coverage"]["cluster_iterations"]):
                raise ValueError(f"{partition}场景阶段、分区或兼容预处理不一致")
            datasets[partition] = record
        if len(datasets) == 2 and set(datasets["train"]["dates"]) & set(datasets["validation"]["dates"]):
            raise ValueError("训练与验证航空日期必须隔离")
        config["datasets"] = datasets
        starts = {}
        if route != "prepare":
            identifiers = body.get("start_run_ids") or {}
            if not isinstance(identifiers, dict) or not required <= set(identifiers):
                raise ValueError("S1-03每个分区须绑定已完成共同初始化来源")
            for partition in required:
                identifier = validate_id(identifiers[partition])
                imported = read_json(self.store.record_dir("scenes", identifier) / "record.json", {})
                if imported.get("record_type") == "reconstruction_imported_start":
                    source_config = imported["source_config"]
                    path = Path(imported["path"])
                    sha = imported["sha256"]
                    source_target = imported["source_target"]
                    source_run_id = imported["source_run_id"]
                    source_dataset_sha = imported["dataset_sha256"]
                else:
                    source_dir = self.store.job_dir(identifier)
                    source_job = read_json(source_dir / "job.json", {})
                    source_status = read_json(source_dir / "status.json", {})
                    source_config = read_json(source_dir / "config.json", {})
                    artifacts = read_json(source_dir / "artifacts.json", {})
                    if source_job.get("execution_target") != target or source_status.get("stage") != "completed":
                        raise ValueError(f"{partition}初始化任务未完成或不属于当前实例")
                    path = Path(artifacts.get("reconstruction_starts", ""))
                    sha = artifacts.get("reconstruction_starts_sha256")
                    source_target = target
                    source_run_id = identifier
                    source_dataset_sha = source_config.get("datasets", {}).get(partition, {}).get("sha256")
                expected_coverage = deepcopy(config["coverage"])
                expected_coverage["method"] = "none"
                expected_coverage["reconstruction"]["route"] = "prepare"
                if (source_config.get("schema_version") != 4 or source_config.get("coverage") != expected_coverage or
                        source_dataset_sha != datasets[partition]["sha256"] or not path.is_file() or
                        not isinstance(sha, str) or digest(path) != sha):
                    raise ValueError(f"{partition}共同初始化来源、物理设置或SHA不匹配")
                starts[partition] = {"path": str(path), "sha256": sha,
                                     "dataset_sha256": datasets[partition]["sha256"],
                                     "source_run_id": source_run_id,
                                     "source_target": source_target}
        config["reconstruction_starts"] = starts
        extra = {"research_stage": "coverage", "experiment_role": purpose,
                 "variant": f"S1-03:{route}", "reconstruction_route": route,
                 "dataset_role": role if kind == "evaluate" else None,
                 "reconstruction_dataset_sha256": {r: v["sha256"] for r, v in datasets.items()},
                 "reconstruction_start_sources": starts, "execution_target": target}
        if kind == "evaluate" and route == "P1":
            model = next((m for m in self.models() if m["id"] == body.get("model_id")), None)
            if model is None or model["config"].get("schema_version") != 4 or model["config"].get("coverage") != config["coverage"]:
                raise ValueError("P1评价须选择同物理配置的S1-03 PPO模型")
            if (model["metadata"].get("policy_sha256") != model["sha256"] or
                    model["metadata"].get("env_step") != model["config"]["training"]["total_steps"] or
                    model["metadata"].get("network") != config["network"] or
                    model["metadata"].get("config_hash") != config_hash(model["config"])):
                raise ValueError("P1模型终点步数、网络或权重SHA与来源元数据不一致")
            for previous in ("train", "validation"):
                if set(model["config"].get("datasets", {}).get(previous, {}).get("dates", [])) & set(datasets[role]["dates"]):
                    raise ValueError("独立考核与模型训练/验证日期重叠")
            extra.update(model_path=model["path"], model_source_run_id=model.get("source_run_id", model["id"]),
                         model_source_target=model.get("source_target", target),
                         model_sha256=model["sha256"],
                         model_files=[{"path": str(Path(model["path"]) / name),
                                       "sha256": digest(Path(model["path"]) / name)}
                                      for name in ("policy.pt", "model.json")])
        return self.store.create_job(config, kind=kind, request_id=body.get("request_id"), extra=extra)

    def approve_reconstruction_gate(self, run_id, body):
        """Bind a controller-produced mechanism audit to the frozen checkpoint."""
        _require_remote_dqn()
        target = _reconstruction_target(self.store)
        directory = self.store.job_dir(validate_id(run_id))
        job = read_json(directory / "job.json", {})
        status = read_json(directory / "status.json", {})
        config = read_json(directory / "config.json", {})
        if (job.get("execution_target") != target or job.get("kind") != "train" or
                config.get("schema_version") != 4 or
                status.get("stage") != "stopped" or
                status.get("stop_reason") != "mechanism_gate_pending" or
                status.get("env_step") != config.get("training", {}).get("mechanism_gate_step")):
            raise ValueError("仅65k机制闸门停下的本实例S1-03训练任务可提交审计证明")
        checkpoint = directory / "checkpoint.pt"
        if not checkpoint.is_file():
            raise ValueError("闸门断点不存在")
        checkpoint_sha = digest(checkpoint)
        if checkpoint_sha != status.get("checkpoint_sha256"):
            raise ValueError("当前断点SHA与停机状态不一致")
        audit_path = Path(body.get("audit_path", "")).resolve()
        allowed = (directory, self.store.archive_root / "ops" / "audits")
        if not any(audit_path.is_relative_to(root.resolve()) for root in allowed):
            raise ValueError("审计证明路径须位于本任务或本归档ops/audits")
        audit = read_json(audit_path, {})
        audit_sha = digest(audit_path) if audit_path.is_file() else None
        if (audit.get("passed") is not True or audit.get("run_id") != run_id or
                audit.get("env_step") != status["env_step"] or
                audit.get("checkpoint_sha256") != checkpoint_sha or
                not isinstance(body.get("audit_sha256"), str) or body["audit_sha256"] != audit_sha):
            raise ValueError("机制审计未通过或任务/交互步/断点/文件SHA不一致")
        proof = {"approved": True, "run_id": run_id, "env_step": status["env_step"],
                 "checkpoint_sha256": checkpoint_sha, "audit_path": str(audit_path),
                 "audit_sha256": audit_sha, "approved_at": now(),
                 "execution_target": target}
        atomic_json(directory / "mechanism_gate.json", proof)
        artifacts = read_json(directory / "artifacts.json", {})
        artifacts.update(mechanism_gate=str(directory / "mechanism_gate.json"),
                         mechanism_gate_sha256=digest(directory / "mechanism_gate.json"),
                         mechanism_audit=str(audit_path), mechanism_audit_sha256=audit_sha)
        atomic_json(directory / "artifacts.json", artifacts)
        return proof

    def api(self, method, parts, query, body):
        if parts == ["catalog"] and method == "GET":
            return {"defaults": default_config(), "summary": summary(default_config()),
                    "balance_defaults": balance_default_config(),
                    "reconstruction_defaults": reconstruction_default_config(),
                    "reconstruction_routes": ["prepare", "R0", "R1", "R2", "R3", "P1"],
                    "balance_routes": ["prepare", "A0", "A1", "A2", "A3", "X0", "X1", "X2", "X3", "X4", "ppo"],
                    "action_menus": ["X0", "X1", "X2", "X3", "X4"],
                    "initializations": [{"id": "rule", "label": "现有需求加权覆盖规则"},
                                        {"id": "cluster", "label": "需求加权聚类"},
                                        {"id": "empty", "label": "无基础布局（纯PPO逐束构建）"}],
                    "methods": [{"id": "none", "label": "不微调（评价基线）"},
                                {"id": "local_search", "label": "局部搜索（评价基线）"},
                                {"id": "ppo", "label": "PPO微调"}]}
        if parts == ["validate"] and method == "POST":
            config = validate_config(body)
            return {"config": config, "summary": summary(config)}
        if parts == ["datasets"] and method == "GET":
            return {"datasets": self.datasets()}
        if parts == ["datasets", "import"] and method == "POST":
            return self.import_dataset(body)
        if parts == ["reconstruction", "import-start"] and method == "POST":
            return self.import_reconstruction_start(body)
        if parts == ["reconstruction", "import-dataset"] and method == "POST":
            return self.import_reconstruction_dataset(body)
        if parts == ["reconstruction", "import-model"] and method == "POST":
            return self.import_reconstruction_model(body)
        if parts == ["reconstruction", "imports"] and method == "GET":
            return {"starts": [r for r in self.store.records("scenes")
                               if r.get("record_type") == "reconstruction_imported_start"],
                    "models": [r for r in self.store.records("models")
                               if r.get("record_type") == "reconstruction_imported_model"]}
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "mechanism-gate" and method == "POST":
            return self.approve_reconstruction_gate(parts[1], body)
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "mechanism-gate" and method == "GET":
            directory = self.store.job_dir(validate_id(parts[1]))
            return {"proof": read_json(directory / "mechanism_gate.json", {}),
                    "audit": read_json(read_json(directory / "mechanism_gate.json", {}).get("audit_path", ""), {})}
        if parts == ["models"] and method == "GET":
            return {"models": self.models()}
        if parts == ["runs"]:
            if method == "GET":
                return {"runs": [r for r in self.app.runs()["runs"] if r.get("research_stage") == "coverage"]}
            if method == "POST":
                return self.submit(body)
        if parts == ["configs"]:
            if method == "GET":
                return {"configs": [r for r in self.store.records("configs")
                                    if r.get("config", {}).get("research_stage") == "coverage"]}
            if method == "POST":
                config = validate_config(body["config"])
                return self.store.save_record("configs", {"id": new_id("coverage_config"),
                    "name": config["name"], "config": config, "created_at": now()})
        raise FileNotFoundError("第一阶段接口不存在")
