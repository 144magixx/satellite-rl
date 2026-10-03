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
from .storage import Store, clean_json, digest, inside, model_input_files, new_id, now, read_json, validate_id


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
        for filename in ("manifest.json", "status.json", "config.json", "metrics.jsonl",
                         "traces.jsonl", "episodes.jsonl", "evaluations.jsonl", "updates.csv",
                         "episodes.csv", "evaluations.csv", "power_results.csv", "power_beams.csv", "power_scene_times.csv",
                         "power_iterations.jsonl", "power_cases.jsonl", "power_report.md", "events.jsonl", "failure.json",
                         "worker.log", "results.json"):
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
                "config": metadata["config"], "provenance": metadata.get("provenance"), "compatible": None,
                "state_dim": metadata.get("state_dim"), "network_token_count": metadata.get("network_token_count"),
                "parameter_count": metadata.get("parameter_count")}
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
            if config["environment"].get("profile", "china_legacy") != "china_legacy":
                raise ValueError("中国历史权重不能以欧洲模型身份导入")
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
        if not legacy_type:
            record.update(state_dim=policy.metadata.get("state_dim"),
                          network_token_count=policy.metadata.get("network_token_count"),
                          parameter_count=policy.metadata.get("parameter_count"))
        from satellite_rl.envs.regions import profile_metadata
        metadata = profile_metadata(config["environment"].get("profile", "china_legacy"))
        record.update(profile=metadata["id"], environment_version=metadata["environment_version"],
                      environment_profile=metadata,
                      provenance=None if legacy_type else policy.metadata.get("provenance"))
        return self.store.save_record("models", record)

    def start_evaluation(self, body):
        from .checkpoints import load_policy
        from .config import validate_config
        from .legacy import load_legacy_policy, validate_legacy_config
        model = self.get_model(body["model_id"])
        scene = self.store.get_record("scenes", body["scene_id"])
        config = validate_config(model["config"])
        from .datasets import assert_scene_profile
        profile = config["environment"].get("profile", "china_legacy")
        if model.get("legacy_type") and profile != "china_legacy":
            raise ValueError("中国历史权重不能作为欧洲环境模型评价")
        assert_scene_profile(scene, profile)
        if (scene.get("source") or {}).get("partition_role") in ("train", "validation"):
            raise ValueError("最终考核须选择test分区场景")
        from .datasets import scene_set_files
        test_hashes = {value for _, value in scene_set_files(scene)}
        test_dates = self.scene_dates(scene)
        def check_provenance(provenance):
            trained_sets = (provenance or {}).get("scene_sets", {})
            for role in ("train", "validation"):
                trained = trained_sets.get(role, {})
                hashes = {ref["sha256"] for ref in trained.get("files", []) if ref.get("sha256")}
                if test_hashes & hashes:
                    raise ValueError("考核场景与模型训练/验证场景重复")
                if test_dates & self.scene_dates(trained):
                    raise ValueError("考核航空来源或快照日期与模型训练/验证分区重叠")
        provenance = model.get("provenance")
        check_provenance(provenance)
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
        model_files = model_input_files(model["path"], bool(model.get("legacy_type")))
        if model.get("legacy_type"):
            validate_legacy_config(config, model["legacy_type"])
            load_legacy_policy(model["path"], model["legacy_type"], "cpu")
        else:
            loaded = load_policy(model["path"], config=config, device="cpu")
            if isinstance(loaded.metadata, dict):
                # A live 'latest' export may have changed since catalog lookup.
                # Validate the provenance of the actual checked policy as well.
                provenance = loaded.metadata.get("provenance")
                check_provenance(provenance)
        return self.store.create_job(config, kind="evaluate", request_id=body.get("request_id"), extra={
            "name": "测试 · " + model["name"], "model_id": model["id"], "model_path": model["path"],
            "model_config": config, "legacy_type": model.get("legacy_type"),
            "scene_id": scene["id"], "scene_path": scene["path"], "action_mode": mode,
            "repeats": repeats, "baselines": sorted(set(baselines)),
            "scene_sets": {"test": scene},
            "model_files": model_files, "model_provenance": provenance,
        })

    def training_scene_sets(self, config, body):
        from .datasets import assert_scene_profile, scene_set_files
        profile = config["environment"].get("profile", "china_legacy")
        selections = {"train": body.get("scene_id"),
                      "validation": body.get("validation_scene_id"), "test": body.get("test_scene_id")}
        if config["training"].get("scene_order", "sequential") == "shuffle_each_cycle" and not selections["train"]:
            raise ValueError("每轮随机场景顺序需要选择固定训练场景集")
        counts = {role: config["evaluation"][role + "_count"] for role in ("validation", "test")}
        if profile == "europe_v1" and selections["train"]:
            for role, count in counts.items():
                if count and not selections[role]:
                    raise ValueError("欧洲自定义训练须显式选择" + role + "场景，防止自动评价改变业务范围")
        sets, hashes, dates = {}, {}, {}
        for role, identifier in selections.items():
            if not identifier:
                continue
            record = self.store.get_record("scenes", identifier)
            assert_scene_profile(record, profile)
            if role in counts and record["count"] != counts[role]:
                raise ValueError(role + "场景数与评价配置不一致")
            source = record.get("source") or {}
            if source.get("partition_role", role) != role:
                raise ValueError("航空数据分区角色与" + role + "用途不匹配")
            hashes[role] = {value for _, value in scene_set_files(record)}
            dates[role] = self.scene_dates(record)
            for previous in sets:
                if hashes[role] & hashes[previous]:
                    raise ValueError("训练、验证与考核不能复用相同实际场景")
                if dates[role] & dates[previous]:
                    raise ValueError("航空轨迹来源日期跨分区重叠")
            sets[role] = record
        return sets

    @staticmethod
    def scene_dates(record):
        source = record.get("source") or {}
        return set(source.get("source_dates", [])) | set(source.get("snapshot_dates", []))

    def import_aviation_scenes(self, body):
        from .aviation import build_aviation_scenes
        from .datasets import _store_scenes
        profile = body.get("profile", "europe_v1")
        if profile != "europe_v1":
            raise ValueError("航空轨迹导入当前仅支持独立欧洲环境")
        source_format = body.get("source_format", "csv_snapshot")
        suffixes = {"csv_snapshot": (".csv",), "readsb_trace": (".json", ".json.gz")}.get(source_format)
        if suffixes is None:
            raise ValueError("不支持的航空轨迹格式")
        path = Path(body["path"]).expanduser().resolve()
        paths = (sorted(p for p in path.iterdir() if p.name.lower().endswith(suffixes))
                 if path.is_dir() else [path])
        if not paths or len(paths) > 1000 or any(not p.is_file() or not p.name.lower().endswith(suffixes) for p in paths):
            raise ValueError("请选择1–1000个对应格式的航空文件")
        sizes = [p.stat().st_size for p in paths]
        if any(size > 100 * 1024 ** 2 for size in sizes) or sum(sizes) > 512 * 1024 ** 2:
            raise ValueError("航空输入每文件不超过100MB、合计不超过512MB")
        times = body.get("selected_times")
        if not isinstance(times, list) or not 1 <= len(times) <= 1000:
            raise ValueError("需选择1–1000个明确的UTC时刻")
        if not all(isinstance(value, str) for value in times) or len(set(times)) != len(times):
            raise ValueError("航空时刻不得重复")
        units = {"timestamp": "utc_iso8601", "lat": "degrees", "lon": "degrees", "altitude_m": "m", "persons": "count"}
        arrays, source = build_aviation_scenes(
            paths, times, source_format=source_format,
            units=units if source_format == "csv_snapshot" else None,
            demand_model=body["demand_model"], seed=body["seed"],
            max_record_age_seconds=body.get("max_record_age_seconds", 300),
            environment_profile=profile, partition_role=body["partition_role"],
            window_seconds=body.get("window_seconds", 60))
        return _store_scenes(self.store, body.get("name") or path.stem, arrays, source, profile)

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
                      "config": config, "summary": normalized(result.get("summary", {})), "metadata": result.get("metadata") or {}}
            records.append(record)
            by_hash = {}
            for scene in result["scenes"]:
                key = scene.get("scene", {}).get("sha256") or scene.get("scene_sha256")
                if key:
                    by_hash.setdefault(key, []).append(normalized(scene["metrics"]))
            maps.append(by_hash)
        def conditions(record):
            c = record["config"]
            environment = c.get("environment")
            if not isinstance(environment, dict):
                return None
            mask = environment.get("action_mask", "none")
            if mask not in ("none", "joint_feasible"):
                return None
            physical_environment = {key: value for key, value in environment.items() if key != "action_mask"}
            physical_environment.setdefault("profile", "china_legacy")
            declared = record["metadata"].get("profile", physical_environment["profile"])
            if declared != physical_environment["profile"]:
                return None
            account = "original" if c.get("reward", {}).get("preset") == "legacy_original" else "repaired"
            metadata_mode = record["metadata"].get("action_mode")
            mode = metadata_mode if metadata_mode is not None else c.get("evaluation", {}).get("action_mode")
            mode = "stochastic" if mode == "sample" else mode
            if mode not in ("deterministic", "stochastic"):
                return None
            return (physical_environment, account, mode)
        def action_mask(record):
            environment = record["config"].get("environment")
            return environment.get("action_mask", "none") if isinstance(environment, dict) else None
        masks = {record["run_id"]: action_mask(record) for record in records}
        mask_values = list(masks.values())
        mask_differs = any(mask != mask_values[0] for mask in mask_values[1:])
        condition_values = [conditions(record) for record in records]
        same_conditions = (condition_values[0] is not None
                           and all(value == condition_values[0] for value in condition_values[1:]))
        common = set.intersection(*(set(item) for item in maps))
        if mask_differs and all(mask in ("none", "joint_feasible") for mask in mask_values):
            warnings.append("有无动作掩码的配对差值反映掩码、最低资源预留及配套熵目标的总体作用；不能全部归因于网络学习。")
        if not same_conditions:
            warnings.append("物理环境、功率记账或动作评价方式不同，或包含未知掩码：只展示摘要，不宣称同条件配对优劣")
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
                  "created_at": now(), "paired_scene_hashes": sorted(common) if same_conditions else [],
                  "treatment_factors": {"action_mask": masks} if mask_differs else {}}
        directory = self.store.root / "comparisons"
        directory.mkdir(exist_ok=True)
        atomic_json(directory / (new_id("comparison") + ".json"), clean_json(output))
        return output

    def api(self, method, parts, query, body):
        if parts and parts[0] == "power-refinement":
            from .power_refinement import PowerRefinementApplication
            return PowerRefinementApplication(self).api(method, parts[1:], query, body)
        if parts and parts[0] == "coverage":
            from .coverage import CoverageApplication
            return CoverageApplication(self).api(method, parts[1:], query, body)
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
                sets = self.training_scene_sets(config, body)
                extra = {"scene_sets": sets} if sets else {}
                if "train" in sets:
                    extra["training_scene_path"] = sets["train"]["path"]
                return self.store.create_job(config, request_id=body.get("request_id"), extra=extra)
        if parts == ["models"] and method == "GET":
            return self.models()
        if parts == ["models", "import"] and method == "POST":
            return self.import_model(body)
        if parts == ["evaluations"] and method == "POST":
            return self.start_evaluation(body)
        if parts == ["scenes"]:
            if method == "GET":
                return {"scenes": [s for s in self.store.records("scenes")
                                   if s.get("research_stage") != "coverage"]}
            if method == "POST":
                from .datasets import generate_scenes
                return generate_scenes(self.store, body.get("name"), body["seed"], body["count"],
                                       profile=body.get("profile", "china_legacy"))
        if parts == ["scenes", "import"] and method == "POST":
            from .datasets import import_scenes
            return import_scenes(self.store, body["path"], body.get("name"),
                                 profile=body.get("profile", "china_legacy"))
        if parts == ["scenes", "aviation"] and method == "POST":
            return self.import_aviation_scenes(body)
        if parts == ["configs"]:
            if method == "GET":
                return {"configs": [c for c in self.store.records("configs")
                                    if c.get("config", {}).get("research_stage") != "coverage"]}
            if method == "POST":
                config = validate_config(body["config"])
                return self.store.save_record("configs", {"id": new_id("config"), "name": body.get("name") or config["name"],
                                              "config": config, "created_at": now()})
        if parts == ["compare"] and method == "POST":
            return self.compare(body["run_ids"])
        if parts == ["dispatch", "hold"]:
            if method == "GET":
                return self.scheduler.dispatch_hold()
            if method == "POST":
                return self.scheduler.hold_dispatch(body.get("ttl_seconds"), body.get("reason"))
            if method == "DELETE":
                return self.scheduler.release_dispatch()
        if parts == ["dispatch", "hold", "release"] and method == "POST":
            return self.scheduler.release_dispatch()
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
                    return (self.scheduler.request_stop(identifier) if action == "stop" else
                            self.scheduler.resume(identifier, source_repair=body.get("source_repair")))
                if action == "upgrade" and method == "POST":
                    if readonly:
                        raise ValueError("历史实验为只读，不能升级")
                    return self.scheduler.upgrade(identifier, upgrade_type=body.get("upgrade_type"),
                                                  receipt_path=body.get("receipt_path"))
        raise FileNotFoundError("接口不存在")

    def download(self, identifier, filename):
        directory, _ = self.resolve_run(identifier)
        allow = {"metrics.jsonl", "updates.jsonl", "episodes.jsonl", "evaluations.jsonl", "traces.jsonl",
                 "updates.csv", "episodes.csv", "evaluations.csv", "events.jsonl", "worker.log", "config.json",
                 "job.json", "manifest.json", "status.json", "results.json", "failure.json", "artifacts.json",
                 "power_results.csv", "power_beams.csv", "power_scene_times.csv", "power_iterations.jsonl", "power_cases.jsonl",
                 "power_report.md"}
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
