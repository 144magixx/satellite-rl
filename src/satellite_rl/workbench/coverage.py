"""First-stage API using the same durable store and persistent scheduler."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import getpass
from pathlib import Path
import shutil
import sys

from satellite_rl.coverage.config import default_config, summary, validate_config
from satellite_rl.workbench.storage import digest, new_id, now, read_json


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
        return rows

    def import_dataset(self, body):
        from satellite_rl.coverage.scenes import load_dataset, save_dataset
        role = body.get("partition_role")
        if role not in ("train", "validation", "test"):
            raise ValueError("请明确数据用途：训练/验证/考核")
        config = default_config()
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
        kind = body.get("kind", "train")
        if kind not in ("train", "evaluate"):
            raise ValueError("未知第一阶段任务类型")
        if config["training"]["device"] != "cuda":
            raise ValueError("正式第一阶段任务必须在指定远端DQN/CUDA执行")
        import torch
        expected_prefix = Path("/home/u2024110224/first_stage_satellite_rl/envs/DQN")
        if (Path(sys.prefix).resolve() != expected_prefix or getpass.getuser() != "u2024110224" or
                not torch.cuda.is_available() or "4090 D" not in torch.cuda.get_device_name(0)):
            raise ValueError("第一阶段请连接u2024110224@10.160.17.102:20418的4090D工作台；本机仅保留开发源码")
        if kind == "train" and config["coverage"]["method"] != "ppo":
            raise ValueError("规则与局部搜索是评价基线，不是训练任务")
        required = {"train", "validation"} if kind == "train" else {"test"}
        supplied = body.get("dataset_ids", {})
        if not required <= set(supplied):
            raise ValueError("第一阶段任务缺少所需用途数据集")
        datasets = {}
        for role, identifier in supplied.items():
            if role not in ("train", "validation", "test") or not identifier:
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
        purpose = body.get("experiment_role", "formal")
        if purpose not in ("formal", "verification"):
            raise ValueError("任务用途须为正式实验或工程验证")
        variant = ("P0" if config["coverage"]["initialization"] == "empty" else
                   ("R" if config["coverage"]["initialization"] == "rule" else "C") +
                   {"none":"0", "local_search":"1", "ppo":"2"}[config["coverage"]["method"]])
        extra = {"research_stage": "coverage", "experiment_role": purpose, "variant": variant,
                 "execution_target": "u2024110224@10.160.17.102:20418"}
        if kind == "evaluate" and config["coverage"]["method"] == "ppo":
            model = next((m for m in self.models() if m["id"] == body.get("model_id")), None)
            if not model:
                raise ValueError("PPO评价须选择已有第一阶段模型")
            for role in ("train", "validation"):
                prior = model["config"].get("datasets", {}).get(role, {})
                if set(prior.get("dates", [])) & set(datasets["test"]["dates"]):
                    raise ValueError("考核与模型训练/验证日期重叠")
            if model["config"].get("coverage") != config["coverage"]:
                raise ValueError("评价物理、初始化和微调配置须与模型一致")
            extra.update(model_path=model["path"],
                         model_files=[{"path": str(Path(model["path"]) / name),
                                       "sha256": digest(Path(model["path"]) / name)}
                                      for name in ("policy.pt", "model.json")])
        return self.store.create_job(config, kind=kind, request_id=body.get("request_id"), extra=extra)

    def api(self, method, parts, query, body):
        if parts == ["catalog"] and method == "GET":
            return {"defaults": default_config(), "summary": summary(default_config()),
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
