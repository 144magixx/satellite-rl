"""Typed workbench entry points for the authorized S2-P19 experiment.

Preparation, constructive training and improvement all publish ordinary durable
workbench jobs. No endpoint executes an arbitrary command or changes old workers.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

from satellite_rl.training.run_monitor import atomic_json
from .storage import digest, read_json, now, validate_id


EXPERIMENT = "S2-P19"


def constructive_config(arm="C0"):
    from .config import default_config, validate_config
    from satellite_rl.state_layouts import natural_shapes
    if arm not in ("C0", "C1"):
        raise ValueError("构造式路线须为C0/C1")
    config = default_config()
    config.update(name=f"S2-P19 {arm} " + ("终态目标对齐SAC" if arm == "C0" else "示范回放增强SAC"),
                  experiment={"id": EXPERIMENT, "arm": arm})
    config["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
        context_beam_count="actual", readout_mode="attention_pooling", beam_count_conditioning="none")
    config["state"].update(context_level="full", features=list(natural_shapes("full")))
    config["environment"].update(profile="europe_v1", beam_count_max=200, action_mask="joint_feasible")
    config["reward"]["preset"] = "terminal_si_delta"
    config["training"].update(scene_order="variable_beams", beam_count_mode="mixed_four_buckets",
        gamma=1, total_steps=400000, pilot_stop_steps=100000, checkpoint_interval=10000,
        alpha_mode="fixed", fixed_alpha=0.01,
        demonstrations={"mode": "none" if arm == "C0" else "half", "manifest_path": "",
                        "manifest_sha256": "", "source_set_sha256": ""})
    config["evaluation"].update(validation_count=72, test_count=0, interval_steps=100000,
        action_mode="stochastic", repeats=1, extra_steps=[0, 50000], validation_seed=53001, test_seed=53001)
    config["telemetry"]["mode"] = "async"
    return validate_config(config) if arm == "C0" else config


def assert_constructive_contract(config):
    """All formal C0/C1 science is identical except declared replay sources."""
    actual, expected = deepcopy(config), constructive_config()
    if actual.get("experiment", {}).get("arm") not in ("C0", "C1"):
        raise ValueError("S2-P19构造式身份无效")
    for item in (actual, expected):
        item.pop("name", None)
        item.pop("experiment", None)
        item["training"].pop("demonstrations", None)
    if actual != expected:
        raise ValueError("S2-P19 C0/C1正式科学超参、表示或验证日程偏离共同冻结合同")


def bind_teacher_job(app, config, sets):
    if config.get("experiment", {}).get("arm") != "C1":
        return None
    from .s2p19_demonstrations import validate_manifest
    from .scene_order import VariableBeamSceneOrder
    record = sets["train"]
    order = VariableBeamSceneOrder("mixed_four_buckets", 51001,
        [row["sha256"] for row in record["files"]], record["files"])
    demo = config["training"]["demonstrations"]
    manifest_path = Path(demo["manifest_path"]).resolve(strict=True)
    validate_manifest(manifest_path, demo["manifest_sha256"], config, order.source_set_sha256)
    manifest = read_json(manifest_path)
    if [row["scene_sha256"] for row in manifest["records"]] != select_teacher_scenes(record):
        raise ValueError("教师场景不是预定逐母分层选择")
    matches = []
    for job, status in app.store.jobs():
        if (job.get("kind") != "prepare" or job.get("experiment_id") != EXPERIMENT or
                status.get("stage") != "completed"):
            continue
        directory = app.store.job_dir(job["run_id"])
        prep = read_json(directory / "config.json", {}).get("preparation", {})
        if prep.get("preparation_type") != "demonstrations":
            continue
        result_path = directory / "preparation_result.json"
        result = read_json(result_path, {})
        if (not manifest_path.is_relative_to(directory.resolve()) or
                result.get("manifest_path") != str(manifest_path) or
                result.get("manifest_sha256") != demo["manifest_sha256"] or
                result.get("source_set_sha256") != order.source_set_sha256):
            continue
        matches.append({"run_id": job["run_id"], "job_path": str(directory / "job.json"),
            "job_sha256": digest(directory / "job.json"), "result_path": str(result_path),
            "result_sha256": digest(result_path), "manifest_path": str(manifest_path),
            "manifest_sha256": demo["manifest_sha256"]})
    if len(matches) != 1:
        raise ValueError("C1教师必须绑定本工作台唯一已完成准备任务，不能使用手工清单")
    return matches[0]


def select_teacher_scenes(record):
    """One input per mother, deterministic stratification without model scores."""
    from .scene_order import beam_bucket
    groups = {}
    for row in record["files"]:
        beam_bucket(int(row["beam_count"]))
        groups.setdefault(row["mother_scene_id"], []).append(row)
    if len(groups) != 576:
        raise ValueError("正式教师来源须恰为576个合格训练母场景")
    buckets = ("anchor_100", "lower", "middle", "upper")
    selected = []
    for index, mother in enumerate(sorted(groups)):
        bucket, load = buckets[index % 4], (0.75, 1.25)[(index // 4) % 2]
        choices = sorted((r for r in groups[mother] if beam_bucket(int(r["beam_count"])) == bucket
                          and float(r["load_multiplier"]) == load),
                         key=lambda r: (int(r["beam_count"]), r["sha256"]))
        if not choices:
            raise ValueError(f"训练母场景缺少预定桶/负载：{mother} {bucket} {load}")
        selected.append(choices[(index // 8) % len(choices)]["sha256"])
    return selected


def freeze_preparation(config, target):
    if config["preparation_type"] == "demonstrations":
        return deepcopy(config)  # The complete scene set is frozen by Store.
    from .s2p19_data import freeze_preparation_config
    return freeze_preparation_config(config, target)


class S2P19Application:
    def __init__(self, app):
        self.app, self.store = app, app.store

    def scene(self, identifier, partition):
        from .datasets import assert_scene_profile, scene_set_files
        record = self.store.get_record("scenes", identifier)
        assert_scene_profile(record, "europe_v1")
        if (record.get("source") or {}).get("partition_role") != partition:
            raise ValueError(f"场景必须来自{partition}分区，不能伪装成test")
        if not self.app.scene_dates(record):
            raise ValueError("场景缺少UTC日期来源")
        scene_set_files(record)
        return record

    def prepare(self, body):
        prep = deepcopy(body["preparation"])
        kind = prep.get("preparation_type")
        extra = {"experiment_id": EXPERIMENT, "research_stage": "s2p19_preparation"}
        if body.get("recovery_parent_run_id"):
            if kind != "mother_sources":
                raise ValueError("母源网络恢复只适用于mother_sources")
            from .s2p19_data import build_recovery_source
            parent = validate_id(body["recovery_parent_run_id"])
            recovery = build_recovery_source(self.store.job_dir(parent))
            if prep.get("recovery_source") not in (None, recovery):
                raise ValueError("恢复输入必须由本工作台失败任务构建")
            prep["recovery_source"] = recovery
            extra["recovery_parent_run_id"] = parent
        if kind == "demonstrations":
            from .config import validate_config
            teacher = validate_config(body.get("teacher_config") or constructive_config())
            assert_constructive_contract(teacher)
            if teacher["experiment"] != {"id": EXPERIMENT, "arm": "C0"}:
                raise ValueError("教师必须使用C0相同的物理/状态/奖励合同")
            record = self.scene(body["scene_id"], "train")
            selection = select_teacher_scenes(record)
            if body.get("selected_sha256") is not None and body["selected_sha256"] != selection:
                raise ValueError("教师选择不符合预定逐母分层规则")
            prep = {"preparation_type": kind, "teacher_config": teacher, "selected_sha256": selection}
            extra["scene_sets"] = {"train": record}
            count = len(selection)
        else:
            from .s2p19_data import validate_preparation_config
            prep = validate_preparation_config(prep)
            count = {"dev_validation": 144, "mother_sources": 40, "final": 880}[kind]
        config = {"schema_version": 1, "name": f"S2-P19 准备 · {kind}", "algorithm": "prepare",
            "network": {"kind": "none"}, "environment": {"profile": "europe_v1", "beam_count_max": 200},
            "research_stage": "s2p19_preparation", "preparation": prep,
            "training": {"total_steps": count, "device": "cuda", "seed": 51001}}
        if extra.get("recovery_parent_run_id"):
            config["name"] += " · 同资产恢复R1"
        return self.store.create_job(config, kind="prepare", request_id=body.get("request_id"), extra=extra)

    def improvement(self, body):
        from satellite_rl.sac_improvement import validate_config, DEFAULT_CONFIG
        controls = validate_config(body.get("i0_config") or DEFAULT_CONFIG)
        if controls["device"] != "cuda" or controls["seed"] != 51001:
            raise ValueError("正式I0须使用DQN/CUDA与seed51001")
        mode = body.get("mode")
        if mode not in ("gate", "train", "evaluate", "baselines"):
            raise ValueError("未知I0任务用途")
        role = "train" if mode == "train" else body.get("partition", "development" if mode == "gate" else "test")
        if role not in ("train", "development", "validation", "test") or (mode == "evaluate" and role != "test"):
            raise ValueError("I0分区用途不合法")
        if mode == "baselines" and role not in ("validation", "test"):
            raise ValueError("基线入口只接受明确validation/test分区")
        record = self.scene(body["scene_id"], role)
        sets = {role: record}
        if mode == "train":
            sets["validation"] = self.scene(body["validation_scene_id"], "validation")
            if self.app.scene_dates(record) & self.app.scene_dates(sets["validation"]):
                raise ValueError("I0训练/验证日期重叠")
        repeats = body.get("repeats", 3 if mode == "evaluate" else 1)
        if repeats != (3 if mode == "evaluate" else 1) or body.get("action_seed", 53001) != 53001:
            raise ValueError("I0动作重复与种子不符合本轮合同")
        config = {"schema_version": 1, "name": "S2-P19 I0 " + mode, "algorithm": "sac",
            "network": {"kind": "candidate_sac"}, "research_stage": "sac_improvement", "i0": controls,
            "environment": {"profile": "europe_v1", "beam_count_max": 200, "action_mask": "joint_feasible"},
            "training": {"total_steps": controls["total_steps"] if mode == "train" else record["count"] * repeats,
                         "device": "cuda", "seed": 51001}}
        extra = {"experiment_id": EXPERIMENT, "research_stage": "sac_improvement", "mode": mode,
                 "i0_config": controls, "scene_sets": sets, "seed": 51001, "device": "cuda",
                 "scene_id": record["id"],
                 "total_steps": controls["total_steps"], "repeats": repeats, "action_seed": 53001,
                 "partition": role, "checkpoint_interval": controls["checkpoint_interval"]}
        if mode == "baselines":
            methods = ["greedy"] if role == "validation" else ["greedy", "rule", "random"]
            if body.get("methods", methods) != methods:
                raise ValueError("基线方法范围与分区预算不一致")
            method_repeats = {name: (3 if name == "random" and role == "test" else 1) for name in methods}
            extra.update(methods=methods, method_repeats=method_repeats)
            config["training"]["total_steps"] = record["count"] * sum(method_repeats.values())
        if mode == "evaluate":
            run_id = validate_id(body["model_run_id"])
            directory, readonly = self.app.resolve_run(run_id)
            job, status = read_json(directory / "job.json"), read_json(directory / "status.json")
            if readonly or job.get("experiment_id") != EXPERIMENT or status.get("stage") != "completed":
                raise ValueError("I0评价须引用本工作台已完成固定终点模型")
            from .storage import model_input_files
            path = Path(job["model_dir"]) / "final"
            extra.update(model_path=str(path), model_files=model_input_files(path), model_run_id=run_id)
        return self.store.create_job(config, kind="train" if mode == "train" else "evaluate",
                                     request_id=body.get("request_id"), extra=extra)

    def approve_gate(self, run_id, body):
        """Only a SHA-bound, source-backed scientific audit releases a pilot."""
        validate_id(run_id)
        directory, readonly = self.app.resolve_run(run_id)
        job, status = read_json(directory / "job.json", {}), read_json(directory / "status.json", {})
        if readonly or job.get("experiment_id") != EXPERIMENT or status.get("stage") != "stopped":
            raise ValueError("仅S2-P19已停试点支持闸门接续")
        if status.get("stop_reason") != "s2p19_pilot_gate_pending":
            raise ValueError("该停止不是试点科学闸门")
        path = Path(body["audit_path"]).resolve(strict=True)
        if not path.is_relative_to(self.store.archive_root.resolve()) or digest(path) != body["audit_sha256"]:
            raise ValueError("闸门审计文件不属于归档或SHA不符")
        audit = read_json(path, {})
        checkpoint_sha = digest(directory / "checkpoint.pt")
        if (audit.get("experiment_id") != EXPERIMENT or audit.get("run_id") != run_id or
                audit.get("checkpoint_sha256") != checkpoint_sha or audit.get("passed") is not True or
                not audit.get("source_files") or not audit.get("criteria") or
                any(value is not True for value in audit["criteria"].values())):
            raise ValueError("试点审计未通过或与当前断点不一致")
        for ref in audit["source_files"]:
            if digest(ref["path"]) != ref["sha256"]:
                raise ValueError("闸门原始证据已改变")
        pilot = directory / "pilot_artifacts"
        pilot.mkdir(exist_ok=True)
        retained = pilot / "checkpoint.pt"
        if retained.exists():
            if digest(retained) != checkpoint_sha:
                raise ValueError("已归档试点断点与当前科学闸门不一致")
        else:
            pending = retained.with_suffix(".pending")
            shutil.copyfile(directory / "checkpoint.pt", pending)
            if digest(pending) != checkpoint_sha:
                raise ValueError("试点断点归档SHA失败")
            pending.replace(retained)
        atomic_json(pilot / "manifest.json", {"run_id": run_id, "env_step": status["env_step"],
            "checkpoint_path": str(retained), "checkpoint_sha256": checkpoint_sha,
            "config_sha256": digest(directory / "config.json"), "git_commit": job.get("git_commit"),
            "audit_path": str(path), "audit_sha256": body["audit_sha256"], "archived_at": now()})
        proof = {"pass": True, "run_id": run_id, "checkpoint_sha256": checkpoint_sha,
                 "checkpoint_step": status["env_step"], "audit_path": str(path),
                 "audit_sha256": body["audit_sha256"], "approved_at": now()}
        atomic_json(directory / "pilot_gate.json", proof)
        return self.app.scheduler.resume(run_id)

    def api(self, method, parts, query, body):
        if parts == ["catalog"] and method == "GET":
            from satellite_rl.sac_improvement import DEFAULT_CONFIG
            from .s2p19_data import OFFICIAL_ASSETS
            assets = []
            for date, asset_id, size in OFFICIAL_ASSETS:
                tag = "v" + date.replace("-", ".") + "-planes-readsb-staging-0"
                assets.append({"date_utc": date, "release_tag": tag, "asset_id": asset_id,
                    "asset_name": tag + ".tar", "expected_bytes": size,
                    "url": f"https://github.com/adsblol/globe_history_2024/releases/download/{tag}/{tag}.tar"})
            return {"experiment_id": EXPERIMENT, "defaults": {"C0": constructive_config(),
                    "C1": constructive_config("C1"), "I0": deepcopy(DEFAULT_CONFIG)},
                    "scenes": self.store.records("scenes"),
                    "official_assets": assets,
                    "preparation_types": ["dev_validation", "mother_sources", "final", "demonstrations"]}
        if parts == ["runs"] and method == "GET":
            return {"runs": [row for row in self.app.runs()["runs"] if row.get("experiment_id") == EXPERIMENT]}
        if method == "POST" and parts == ["prepare"]:
            return self.prepare(body)
        if method == "POST" and parts == ["train"]:
            body = {**body, "experiment_id": EXPERIMENT}
            if body.get("config", {}).get("experiment", {}).get("id") != EXPERIMENT:
                raise ValueError("该入口只接受S2-P19 C0/C1配置")
            return self.app.api("POST", ["runs"], {}, body)
        if method == "POST" and parts == ["improvement"]:
            return self.improvement(body)
        if method == "POST" and len(parts) == 3 and parts[0] == "runs" and parts[2] == "approve-pilot":
            return self.approve_gate(parts[1], body)
        raise FileNotFoundError("S2-P19接口不存在")
