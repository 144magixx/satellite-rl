"""Local application API, archived experiments, models and paired evaluation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import zipfile

from satellite_rl.training.run_monitor import atomic_json
from .monitoring import MetricsReader, log_tail, read_jsonl
from .runtime import Scheduler, pid_alive, system_info
from .storage import Store, clean_json, digest, inside, new_id, now, read_json, validate_id


class Application:
    def __init__(self, archive_root, source_root=None, *, python_executable=None, max_workers=1):
        options = {"source_root": source_root} if source_root else {}
        self.store = Store(archive_root, **options)
        scheduler_options = {"python_executable": python_executable} if python_executable else {}
        self.scheduler = Scheduler(self.store, max_workers=max_workers, **scheduler_options)
        self.metrics = MetricsReader()
        self.history_cache = (0, {})

    @staticmethod
    def observed_status(status):
        result = dict(status)
        result["process_alive"] = pid_alive(status.get("pid"))
        active = {"training", "running", "warmup", "initializing", "starting", "validating", "evaluating"}
        if status.get("stage") in active and status.get("pid") and not result["process_alive"]:
            result["reported_stage"] = status["stage"]
            result["stage"] = "interrupted"
            result["status_note"] = "原状态仍为运行中，但对应进程已退出；历史文件保持原样"
        return result

    def historical(self):
        if time.monotonic() - self.history_cache[0] < 20:
            return self.history_cache[1]
        root = self.store.archive_root / "outputs" / "experiments"
        records = {}
        if root.exists():
            for path in root.rglob("status.json"):
                status = read_json(path, {})
                if not status.get("run_id") and "env_step" not in status:
                    continue
                key = "history_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]
                records[key] = path.parent
        self.history_cache = (time.monotonic(), records)
        return records

    def resolve_run(self, identifier):
        validate_id(identifier)
        directory = self.store.job_dir(identifier)
        if (directory / "job.json").is_file():
            return directory, False
        if identifier in self.historical():
            return self.historical()[identifier], True
        raise FileNotFoundError("运行记录不存在")

    def run_detail(self, identifier):
        directory, readonly = self.resolve_run(identifier)
        status = self.observed_status(read_json(directory / "status.json", {}))
        manifest = read_json(directory / "manifest.json", {})
        config = read_json(directory / "config.json", manifest.get("config", {}))
        job = read_json(directory / "job.json", {"run_id": identifier, "name": status.get("run_id", directory.name),
                        "kind": "historical", "created_at": status.get("created_at"), "path": str(directory)})
        runtime = config.get("runtime", {})
        if readonly and isinstance(runtime, dict):
            job.update({"source_root": runtime.get("source_root"), "git_commit": runtime.get("git_commit")})
            status.setdefault("preflight", runtime)
            status.setdefault("gpu", runtime.get("cuda_device"))
        artifacts = read_json(directory / "artifacts.json", {})
        for filename in ("manifest.json", "status.json", "config.json", "metrics.jsonl", "updates.csv", "episodes.csv", "evaluations.csv", "events.jsonl", "failure.json", "worker.log", "results.json"):
            if (directory / filename).is_file():
                artifacts.setdefault(filename, str(directory / filename))
        launch = read_json(directory / "launch.json", {})
        status["process_alive"] = pid_alive(status.get("pid") or launch.get("pid"))
        return clean_json({"job": {**job, "read_only": readonly}, "status": status,
                           "config": config, "artifacts": artifacts,
                           "events": read_jsonl(directory / "events.jsonl", 100),
                           "read_only": readonly, "path": str(directory)})

    def runs(self):
        rows = []
        for job, status in self.store.jobs():
            status = self.observed_status(status)
            rows.append({**job, **status, "source": "workbench", "read_only": False,
                         "run_id": job["run_id"], "name": job.get("name", job["run_id"])})
        for key, directory in self.historical().items():
            status = self.observed_status(read_json(directory / "status.json", {}))
            manifest = read_json(directory / "manifest.json", {})
            config = manifest.get("config", {})
            rows.append({**status, "run_id": key, "original_run_id": status.get("run_id", directory.name),
                         "name": status.get("run_id", directory.name), "kind": "historical",
                         "algorithm": "sac", "network": config.get("model", "历史模型"),
                         "created_at": status.get("created_at", manifest.get("created_at", "")),
                         "source": "historical", "read_only": True, "path": str(directory)})
        return {"runs": sorted(rows, key=lambda row: row.get("created_at") or "", reverse=True)}

    def models(self):
        from .legacy import legacy_config, legacy_architecture
        records = {row["id"]: row for row in self.store.records("models")}
        for kind, filename in (("sac_transformer", "actor.pth"), ("sac_cnn", "actor.pth"),
                               ("sac_mlp", "actor.pth"), ("ppo_direct", "actor.pth"), ("dqn", "dqn_online.pth")):
            path = self.store.archive_root / "models" / "pretrained" / kind / filename
            if path.is_file():
                records["pretrained_" + kind] = {"id": "pretrained_" + kind, "name": "原预训练 · " + kind,
                    "path": str(path), "format": "legacy_state_dict", "algorithm": kind.split("_")[0],
                    "legacy_type": kind, "config": legacy_config(kind), "compatible": None,
                    "architecture": legacy_architecture(kind),
                    "note": "原204维与原动作档位，评价前严格检查；旧训练来历以历史记录为准"}
        for path in self.store.models_root.glob("*/*/model.json"):
            metadata = read_json(path, {})
            if not metadata.get("config"):
                continue
            key = "model_" + hashlib.sha256(str(path.parent).encode()).hexdigest()[:20]
            records[key] = {"id": key, "name": path.parent.parent.name + " / " + path.parent.name,
                "path": str(path.parent), "format": metadata.get("format"), "algorithm": metadata.get("algorithm"),
                "config": metadata["config"], "provenance": metadata.get("provenance"), "compatible": None}
        return {"models": list(records.values())}

    def get_model(self, identifier):
        model = next((item for item in self.models()["models"] if item["id"] == identifier), None)
        if not model:
            raise FileNotFoundError("模型不存在")
        return model

    def import_model(self, body):
        from .checkpoints import load_policy
        from .config import validate_config
        from .legacy import legacy_config, load_legacy_policy, validate_legacy_config
        source = Path(body["path"]).expanduser().resolve()
        legacy_type = body.get("legacy_type")
        identifier = new_id("model")
        if legacy_type:
            if not source.is_file() or source.stat().st_size > 512 * 1024 ** 2:
                raise ValueError("旧权重须为不超过512MB的文件")
            config = validate_config(body.get("config") or legacy_config(legacy_type))
            validate_legacy_config(config, legacy_type)
            loaded = load_legacy_policy(source, legacy_type, "cpu")
            target = self.store.record_dir("models", identifier)
            target.mkdir()
            shutil.copy2(source, target / "weights.pt")
            path = target / "weights.pt"
            format_name = "legacy_state_dict"
        else:
            policy = load_policy(source, config=body.get("config"), device="cpu")
            config = validate_config(policy.config)
            source_dir = source.parent if source.is_file() else source
            target = self.store.record_dir("models", identifier)
            target.mkdir()
            for filename in ("policy.pt", "model.json"):
                shutil.copy2(source_dir / filename, target / filename)
            path = target
            format_name = "satellite_rl_workbench_policy_v1"
        record = {"id": identifier, "name": body.get("name") or source.name,
                  "path": str(path), "format": format_name, "algorithm": config["algorithm"],
                  "config": config, "legacy_type": legacy_type, "compatible": True,
                  "architecture": loaded.architecture if legacy_type else policy.metadata.get("variant"),
                  "created_at": now(), "source_path": str(source),
                  "sha256": digest(path if path.is_file() else path / "policy.pt")}
        return self.store.save_record("models", record)

    def start_evaluation(self, body):
        from .checkpoints import load_policy
        from .config import validate_config
        from .legacy import load_legacy_policy, validate_legacy_config
        model = self.get_model(body["model_id"])
        scene = self.store.get_record("scenes", body["scene_id"])
        config = validate_config(model["config"])
        mode = body.get("action_mode", "deterministic")
        if mode == "sample":
            mode = "stochastic"
        if mode not in ("deterministic", "stochastic"):
            raise ValueError("未知评价动作方式")
        repeats = body.get("repeats", 1)
        if not isinstance(repeats, int) or not 1 <= repeats <= 30:
            raise ValueError("动作重复次数须为1–30")
        baselines = body.get("baselines", [])
        if not isinstance(baselines, list) or any(b not in ("fixed", "greedy") for b in baselines):
            raise ValueError("未知对照基线")
        if model.get("legacy_type"):
            validate_legacy_config(config, model["legacy_type"])
            load_legacy_policy(model["path"], model["legacy_type"], "cpu")
        else:
            load_policy(model["path"], config=config, device="cpu")
        return self.store.create_job(config, kind="evaluate", request_id=body.get("request_id"), extra={
            "name": "测试 · " + model["name"], "model_id": model["id"], "model_path": model["path"],
            "model_config": config, "legacy_type": model.get("legacy_type"),
            "scene_id": scene["id"], "scene_path": scene["path"], "action_mode": mode,
            "repeats": repeats, "baselines": sorted(set(baselines)),
        })

    def results(self, identifier):
        directory, readonly = self.resolve_run(identifier)
        result = read_json(directory / "results.json")
        if result:
            return result
        return {"stage": read_json(directory / "status.json", {}).get("stage"), "summary": None,
                "scenes": [], "message": "该运行尚无工作台格式的逐束结果；历史标量仍可在训练监控中查看"}

    def compare(self, identifiers):
        if not isinstance(identifiers, list) or not 2 <= len(set(identifiers)) <= 8:
            raise ValueError("请选择2–8个不同的运行")
        records, warnings, maps = [], [], []
        aliases = {"si_mean": "final_si_mean", "sgm_mean": "final_sgm_mean",
                   "basic_service_count": "basic_met_count", "total_power_w": "power_w",
                   "power_exceeded": "overload", "completed": "complete", "jain_index": "jain"}

        def normalized(metrics):
            return {**metrics, **{target: metrics[source] for source, target in aliases.items()
                                 if target not in metrics and source in metrics}}
        for identifier in dict.fromkeys(identifiers):
            detail, result = self.run_detail(identifier), self.results(identifier)
            if not result.get("scenes"):
                raise ValueError("运行 " + identifier + " 缺少可配对的逐场景结果")
            config = detail["config"]
            record = {"run_id": identifier, "name": detail["job"].get("name", identifier),
                      "config": config, "summary": normalized(result.get("summary", {})), "metadata": result.get("metadata", {})}
            records.append(record)
            by_hash = {}
            for scene in result["scenes"]:
                key = scene.get("scene", {}).get("sha256") or scene.get("scene_sha256")
                if key:
                    by_hash.setdefault(key, []).append(normalized(scene["metrics"]))
            maps.append(by_hash)
        def conditions(record):
            c = record["config"]
            account = "original" if c.get("reward", {}).get("preset") == "legacy_original" else "repaired"
            return (c.get("environment"), account, record["metadata"].get("action_mode"))
        same_conditions = all(conditions(record) == conditions(records[0]) for record in records[1:])
        common = set.intersection(*(set(item) for item in maps))
        if not same_conditions:
            warnings.append("环境、功率记账或动作评价方式不同：只展示摘要，不宣称同条件配对优劣")
        if not common:
            warnings.append("没有校验值一致的实际场景：只展示摘要，不按场景名称猜测配对")
        metrics = ("final_si_mean", "final_sgm_mean", "basic_met_count", "throughput_bps", "power_w",
                   "zero_capacity_count", "overload", "complete", "zero_slot_count", "jain")
        rows = []
        for index, record in enumerate(records):
            row = {"run_id": record["run_id"], "name": record["name"], "paired_scenes": len(common) if same_conditions else 0}
            for metric in metrics:
                if metric in record["summary"]:
                    row[metric] = record["summary"][metric]
                if index and common and same_conditions:
                    differences = []
                    for key in common:
                        current = [v[metric] for v in maps[index][key] if isinstance(v.get(metric), (int, float))]
                        baseline = [v[metric] for v in maps[0][key] if isinstance(v.get(metric), (int, float))]
                        if current and baseline:
                            differences.append(sum(current) / len(current) - sum(baseline) / len(baseline))
                    if differences:
                        row[metric + "_paired_delta"] = sum(differences) / len(differences)
            rows.append(row)
        warnings.append("配对差值相对所选第一个运行；重复动作先在同场景内平均。回报不跨奖励公式排名。单训练种子不代表跨种子稳定性。")
        output = {"runs": records, "rows": rows, "warnings": warnings,
                  "created_at": now(), "paired_scene_hashes": sorted(common) if same_conditions else []}
        directory = self.store.root / "comparisons"
        directory.mkdir(exist_ok=True)
        atomic_json(directory / (new_id("comparison") + ".json"), clean_json(output))
        return output

    def api(self, method, parts, query, body):
        from .config import catalog, default_config, validate_config, config_summary
        if parts == ["catalog"] and method == "GET":
            return {**catalog(), "defaults": default_config()}
        if parts == ["system"] and method == "GET":
            return system_info(self.store)
        if parts == ["validate"] and method == "POST":
            from .checkpoints import describe_model
            config = validate_config(body)
            return {"config": config, "summary": config_summary(config), "model": describe_model(config)}
        if parts == ["runs"]:
            if method == "GET":
                return self.runs()
            if method == "POST":
                config = validate_config(body["config"])
                if config["training"]["device"] != "cuda":
                    raise ValueError("正式训练必须使用DQN/CUDA；CPU仅供代码单元检查，工作台不会静默更换设备")
                extra = {}
                if body.get("scene_id"):
                    extra["training_scene_path"] = self.store.get_record("scenes", body["scene_id"])["path"]
                return self.store.create_job(config, request_id=body.get("request_id"), extra=extra)
        if parts == ["models"] and method == "GET":
            return self.models()
        if parts == ["models", "import"] and method == "POST":
            return self.import_model(body)
        if parts == ["evaluations"] and method == "POST":
            return self.start_evaluation(body)
        if parts == ["scenes"]:
            if method == "GET":
                return {"scenes": self.store.records("scenes")}
            if method == "POST":
                from .datasets import generate_scenes
                return generate_scenes(self.store, body.get("name"), body["seed"], body["count"])
        if parts == ["scenes", "import"] and method == "POST":
            from .datasets import import_scenes
            return import_scenes(self.store, body["path"], body.get("name"))
        if parts == ["configs"]:
            if method == "GET":
                return {"configs": self.store.records("configs")}
            if method == "POST":
                config = validate_config(body["config"])
                return self.store.save_record("configs", {"id": new_id("config"), "name": body.get("name") or config["name"],
                                              "config": config, "created_at": now()})
        if parts == ["compare"] and method == "POST":
            return self.compare(body["run_ids"])
        if len(parts) >= 2 and parts[0] == "runs":
            identifier = parts[1]
            directory, readonly = self.resolve_run(identifier)
            if len(parts) == 2 and method == "GET":
                return self.run_detail(identifier)
            if len(parts) == 3:
                action = parts[2]
                if action == "metrics" and method == "GET":
                    return self.metrics.read(directory, int(query.get("limit", [2000])[0]))
                if action == "logs" and method == "GET":
                    return log_tail(directory, min(5000, max(1, int(query.get("limit", [200])[0]))))
                if action == "results" and method == "GET":
                    return self.results(identifier)
                if method == "POST" and action in ("stop", "resume"):
                    if readonly:
                        raise ValueError("历史实验为只读，本工作台不控制其他管理进程")
                    return self.scheduler.request_stop(identifier) if action == "stop" else self.scheduler.resume(identifier)
        raise FileNotFoundError("接口不存在")

    def download(self, identifier, filename):
        directory, _ = self.resolve_run(identifier)
        allow = {"metrics.jsonl", "updates.jsonl", "episodes.jsonl", "evaluations.jsonl", "traces.jsonl",
                 "updates.csv", "episodes.csv", "evaluations.csv", "events.jsonl", "worker.log", "config.json",
                 "job.json", "manifest.json", "status.json", "results.json", "failure.json", "artifacts.json"}
        if filename == "report.md":
            detail, result = self.run_detail(identifier), self.results(identifier)
            lines = ["# 卫星资源分配实验报告", "", "运行：" + identifier,
                     "状态：" + str(detail["status"].get("stage")), "",
                     "全部目标波束的终态结果，包含后续干扰及未服务波束；单种子结果不构成稳定性结论。", "",
                     "## 配置", "", "```json", json.dumps(detail["config"], ensure_ascii=False, indent=2), "```", "",
                     "## 实际结果", "", "```json", json.dumps(result.get("summary"), ensure_ascii=False, indent=2), "```", "",
                     "## 来历", "", "```json", json.dumps(result.get("metadata", detail["job"]), ensure_ascii=False, indent=2), "```"]
            return "text/markdown; charset=utf-8", "report.md", "\n".join(lines).encode("utf-8")
        if filename == "beam_results.csv":
            result = self.results(identifier)
            rows = [{"scene_id": scene["scene_id"], **beam} for scene in result.get("scenes", []) for beam in scene.get("beams", [])]
            output = io.StringIO(newline="")
            if rows:
                writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
                writer.writeheader()
                writer.writerows(rows)
            return "text/csv; charset=utf-8", filename, ("\ufeff" + output.getvalue()).encode("utf-8")
        if filename not in allow:
            raise ValueError("不支持下载该文件")
        path = inside(directory / filename, directory)
        if not path.is_file():
            raise FileNotFoundError("该运行没有记录此文件")
        return "application/octet-stream", filename, path.read_bytes()

    def download_model(self, identifier):
        model = self.get_model(identifier)
        path = Path(model["path"])
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
            if model.get("legacy_type"):
                bundle.write(path, "weights.pt")
                bundle.writestr("legacy_model.json", json.dumps(model, ensure_ascii=False, indent=2))
            else:
                for name in ("model.json", "policy.pt"):
                    bundle.write(path / name, name)
        return "application/zip", identifier + ".zip", stream.getvalue()
