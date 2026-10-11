"""Isolated typed S2-F20 API; every computation is a frozen workbench job."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from satellite_rl.training.run_monitor import atomic_json
from .storage import digest, read_json, now, validate_id, inside

EXPERIMENT = "S2-F20"
ALGORITHMS = ("sac", "ppo", "dqn")
NODES = (100000, 200000, 300000, 400000)
INPUT_FIELDS = ("datasets", "dataset_contract", "sourceprep", "sourceprep_run_id", "source_files", "model_files", "direct_files", "selection", "models_by_algorithm", "method_order", "prepare_run_id", "partition", "offset", "count")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def default_config(algorithm="sac", candidate="A", mode="train"):
    if algorithm not in (*ALGORITHMS, "random", "greedy", "prepare") or candidate not in ("A", "B"):
        raise ValueError("S2-F20算法或候选无效")
    if mode not in ("sourceprep", "prepare", "shorttune", "train", "evaluate", "refine", "matrix", "engineering"):
        raise ValueError("S2-F20任务类型无效")
    steps = 400000 if mode == "train" else 32768 if mode == "shorttune" else 2048 if mode == "engineering" else 0
    if mode in ("train", "shorttune", "engineering") and algorithm not in ALGORITHMS:
        raise ValueError("学习任务必须属于三算法之一")
    if mode in ("sourceprep", "prepare", "matrix") and algorithm != "prepare":
        raise ValueError("准备任务不允许伪装为模型训练")
    learning = {"adam_betas": [0.9, 0.999], "adam_eps": 1e-8, "gradient_norm_max": 1.,
                "huber_delta": 1., "gamma": 1.}
    if algorithm == "sac":
        learning.update(alpha=.01 if candidate == "A" else .05, actor_lr=1e-4, critic_lr=3e-4,
                        replay_capacity=100000, batch_size=256, update_every=4, tau=.005)
    elif algorithm == "ppo":
        learning.update(lr=1e-4 if candidate == "A" else 3e-4, rollout=2048, minibatch=256,
                        epochs=4, clip=.2, gae_lambda=.95, value_coefficient=.5, entropy_coefficient=.01)
    elif algorithm == "dqn":
        learning.update(lr=1e-4 if candidate == "A" else 3e-4, replay_capacity=100000, batch_size=256,
                        update_every=4, target_updates=1000, epsilon_start=1., epsilon_end=.05,
                        epsilon_decay_interactions=80000, double_dqn=True)
    return {"schema_version": 20, "research_stage": "fair20", "algorithm": algorithm,
            "candidate": candidate, "mode": mode, "name": f"S2-F20 {algorithm} {candidate} {mode}",
            "network": {"kind": "transformer", "hidden_dim": 128, "layers": 2, "heads": 4,
                        "feedforward_dim": 512, "dropout": 0, "joint_action_dim": 1000},
            "environment": {"profile": "europe_v1", "beam_count_max": 200,
                            "power_budget_w": 3000, "power_rule": "3000/N",
                            "action_mask": "joint_feasible_future_reservation",
                            "reward": "100_delta_global_si", "gamma": 1},
            "training": {"seed": 51001, "total_steps": steps, "warmup": 4096,
                         "checkpoint_nodes": list(NODES) if mode == "train" else [],
                         "cpu_threads": 1, "precision": "float32", "checkpoint_interval": 10000},
            "learning": learning,
            "evaluation": {"action_mode": "deterministic", "repeats": 1,
                           "action_seed": 53001, "optimizer_seed": 63001,
                           "selection": "direct_si_tie_0.001_then_sgm_then_A"}}


def validate_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("配置必须为对象")
    expected = default_config(raw.get("algorithm"), raw.get("candidate"), raw.get("mode"))
    actual = deepcopy(raw)
    expected["name"] = actual.get("name", expected["name"])
    if actual != expected:
        raise ValueError("S2-F20共同结构、动作、功率、种子或有限预算偏离方案20合同")
    return actual


def _copy_reference(ref, target):
    source = Path(ref["path"]).resolve(strict=True)
    if not source.is_file() or digest(source) != ref["sha256"]:
        raise ValueError("冻结输入缺失或SHA不符：" + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if digest(target) != ref["sha256"]:
        raise ValueError("输入冻结复制校验失败")
    return {**ref, "path": str(target), "source_path": str(source)}


def freeze_fair20_sources(extra, destination):
    """Called by Store before it publishes job.json, preventing dispatch races."""
    result = deepcopy(extra)
    destination = Path(destination)
    supplied = result.get("datasets", {})
    result["dataset_contract"] = {role: {"count": len(record["files"]),
        "input_sha256": fingerprint([row["sha256"] for row in record["files"]]),
        "mother_count": len({row.get("mother_scene_id") for row in record["files"]})}
        for role, record in supplied.items()}
    if result.get("mode") in ("train", "shorttune", "engineering"):
        result["datasets"] = {role: supplied[role] for role in ("train", "validation")}
    elif result.get("mode") == "evaluate":
        role = result["partition"]
        result["datasets"] = {role: supplied[role]}
    elif result.get("mode") == "matrix":
        result["datasets"] = {"test": supplied["test"]}
    for role, record in result.get("datasets", {}).items():
        detail_cache = {}
        references = [_copy_reference(ref, destination / role / f"scene_{i:07d}.npz")
                      for i, ref in enumerate(record["files"])]
        for ref in references:
            for field in ("observation_detail", "source_observation_detail"):
                detail = ref.get(field)
                if isinstance(detail, dict) and "path" in detail and "sha256" in detail:
                    sha = detail["sha256"]
                    if sha not in detail_cache:
                        detail_cache[sha] = _copy_reference(detail, destination / role / "details" / (sha + ".jsonl.gz"))
                    ref[field] = detail_cache[sha]
        record.update(files=references, path=str(destination / role), count=len(references))
        atomic_json(destination / role / "record.json", record)
    for key in ("model_files", "direct_files", "source_files"):
        result[key] = [_copy_reference(ref, destination / key / f"{i:05d}_{Path(ref['path']).name}")
                       for i, ref in enumerate(result.get(key, []))]
    for algorithm, source in result.get("models_by_algorithm", {}).items():
        source["model_files"] = [_copy_reference(ref, destination / "models_by_algorithm" / algorithm / Path(ref["path"]).name)
                                 for ref in source["model_files"]]
    if result.get("sourceprep"):
        prep = result["sourceprep"]
        refs = {row["source_path"]: row for row in result["source_files"]}
        prep["train_record_path"] = refs[str(Path(prep["train_record_path"]).resolve())]["path"]
        prep["original_train_record_sha256"] = prep["train_record_sha256"]
        atomic_json(prep["train_record_path"], result["datasets"]["train"])
        prep["train_record_sha256"] = digest(prep["train_record_path"])
        for asset in prep["source_assets"]:
            if asset.get("path"):
                asset["path"] = refs[str(Path(asset["path"]).resolve())]["path"]
    result["frozen_input_sha256"] = fingerprint({key: result.get(key) for key in INPUT_FIELDS})
    return result


def validate_admission_proof(proof, job):
    """Bind a fresh lane observation to the root's old-group release receipt."""
    if proof.get("admitted") is not True or proof.get("run_id") != job["run_id"]:
        raise ValueError("资源放行证明不是本任务")
    receipt_path = Path(proof.get("release_receipt_path", "")).resolve(strict=True)
    if not receipt_path.is_file() or digest(receipt_path) != proof.get("release_receipt_sha256"):
        raise ValueError("缺少已核SHA的旧组20474交还证明")
    observation = proof.get("resource_observation", {})
    try:
        checked = datetime.fromisoformat(observation["checked_at_utc"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc)-checked).total_seconds()
    except (KeyError, TypeError, ValueError):
        raise ValueError("资源观察时间无效") from None
    if (not 0 <= age <= 60 or observation.get("port") != 20474 or
            observation.get("scientific_commit") != job["git_commit"] or
            observation.get("worker_free") is not True or observation.get("gpu_free") is not True):
        raise ValueError("20474最新资源观察或冻结提交不符，禁止调度")
    from .fair20_guard import admission_reasons
    reasons=admission_reasons(observation, proof.get("release", {}), job["git_commit"], read_json(receipt_path, {}))
    if reasons:
        raise ValueError("资源放行证据矛盾：" + "; ".join(reasons))
    return proof


def select_candidate(rows):
    if {row["candidate"] for row in rows} != {"A", "B"} or len(rows) != 2:
        raise ValueError("正式训练须有两个已完成短调参候选")
    a, b = sorted(rows, key=lambda row: row["candidate"])
    if abs(a["SI"] - b["SI"]) > 0.001:
        return max(rows, key=lambda row: row["SI"])["candidate"]
    return "B" if b["SGM"] > a["SGM"] else "A"


class Fair20Application:
    def __init__(self, app):
        self.app, self.store = app, app.store

    def task(self, run_id, completed=False):
        directory = self.store.job_dir(validate_id(run_id))
        job = read_json(directory / "job.json", {})
        status = read_json(directory / "status.json", {})
        if job.get("experiment_id") != EXPERIMENT:
            raise ValueError("来源不是S2-F20任务")
        if completed and status.get("stage") != "completed":
            raise ValueError("来源任务尚未真实完成")
        return directory, job, status

    def datasets(self, prepare_run_id):
        directory, job, _ = self.task(prepare_run_id, True)
        result = read_json(directory / "results.json", {})
        if job.get("mode") not in ("prepare", "sourceprep") or not result.get("datasets"):
            raise ValueError("来源须为完成的共同数据准备任务")
        return result["datasets"]

    def matrix_model(self, identifier, algorithm, prepare_run_id, datasets):
        directory, model_job, status = self.task(identifier, True)
        if model_job.get("mode") != "train" or model_job.get("algorithm") != algorithm or status.get("env_step") != 400000:
            raise ValueError("matrix必须使用三臂各自完整400000步正式终点")
        if model_job.get("prepare_run_id") != prepare_run_id:
            raise ValueError("matrix模型与共同准备任务来源不一致")
        contract = {role: {"count":len(record["files"]),
            "input_sha256":fingerprint([row["sha256"] for row in record["files"]]),
            "mother_count":len({row.get("mother_scene_id") for row in record["files"]})} for role,record in datasets.items()}
        if model_job.get("dataset_contract") != contract:
            raise ValueError("matrix模型完整数据来源SHA合同不一致")
        controls=validate_config(read_json(directory/"config.json"))
        result=read_json(directory/"results.json",{})
        budget=result.get("budget_actual_interactions_proof",{})
        if budget.get("retained_interactions") != 400000 or budget.get("strict_actual_budget_verified") is not True:
            raise ValueError("matrix模型缺少严格同预算实际交互证据")
        model=Path(model_job["model_dir"])/"final"
        return {"model_run_id":model_job["run_id"],"model_source_git_commit":model_job["git_commit"],
                "candidate":controls["candidate"],"config":controls,
                "model_files":[{"path":str(model/name),"sha256":digest(model/name)} for name in ("policy.pt","model.json")]}

    def submit(self, body, mode):
        if not isinstance(body.get("request_id"), str) or not body["request_id"].strip():
            raise ValueError("S2-F20提交必须填写唯一request_id，以保证未知网络结果可安全重试")
        validate_id(body["request_id"])
        algorithm = body.get("algorithm", "prepare" if mode in ("prepare", "sourceprep", "matrix") else body.get("method", "sac"))
        config = validate_config(body.get("config") or default_config(algorithm, body.get("candidate", "A"), mode))
        if config["mode"] != mode:
            raise ValueError("接口与配置任务类型不符")
        if config["algorithm"] != algorithm:
            raise ValueError("接口算法与冻结配置算法不符")
        extra = {"experiment_id": EXPERIMENT, "research_stage": "fair20", "mode": mode,
                 "seed": 51001, "kind": "train" if mode in ("train", "shorttune", "engineering") else
                 "evaluate" if mode in ("evaluate", "matrix") else "refine" if mode == "refine" else "prepare",
                 "resource_hold": os.environ.get("SATELLITE_RL_F20_REQUIRE_ADMISSION") == "1"}
        if mode == "sourceprep":
            prep = {k: deepcopy(body[k]) for k in ("train_record_path", "train_record_sha256", "source_assets", "exclude_mother_ids")}
            if {a["date"] for a in prep["source_assets"]} != {"2024-02-13", "2024-02-14", "2024-02-15"}:
                raise ValueError("方案20来源日期必须为Feb13/14/15")
            extra["sourceprep"] = prep
            extra["source_files"] = [{"path": prep["train_record_path"], "sha256": prep["train_record_sha256"]},
                                     *[{"path": a["path"], "sha256": a["sha256"]} for a in prep["source_assets"] if a.get("path")]]
            extra["datasets"] = {"train": read_json(prep["train_record_path"])}
        elif mode == "prepare":
            if body.get("sourceprep_run_id"):
                _,source_job,_=self.task(body["sourceprep_run_id"],True)
                if source_job.get("mode") != "sourceprep":
                    raise ValueError("注册来源须为已完成的S2-F20 sourceprep任务")
                extra["datasets"]=self.datasets(body["sourceprep_run_id"])
                extra["sourceprep_run_id"]=body["sourceprep_run_id"]
            else:
                extra["datasets"] = deepcopy(body["datasets"])
        elif mode == "refine":
            directory, source, _ = self.task(body["direct_run_id"], True)
            if source["mode"] != "evaluate":
                raise ValueError("B须复用完成direct任务的共同初解")
            result = read_json(directory / "results.json", {})
            extra.update(direct_files=result["case_files"], direct_run_id=source["run_id"],
                         direct_results_sha256=digest(directory / "results.json"))
        else:
            extra["datasets"] = self.datasets(body["prepare_run_id"])
            extra["prepare_run_id"] = body["prepare_run_id"]
            if mode == "matrix":
                models=body.get("models_by_algorithm",{})
                if set(models) != set(ALGORITHMS):
                    raise ValueError("matrix须有sac/ppo/dqn三个正式终点任务ID")
                extra["models_by_algorithm"]={arm:self.matrix_model(models[arm],arm,body["prepare_run_id"],extra["datasets"])
                                               for arm in ALGORITHMS}
                extra["method_order"]={"methods":["sac","ppo","dqn","random","greedy"],
                                       "rule":"scene_index modulo5 rotates fixed method sequence",
                                       "warmup_network_calls_per_model":4,"all_cases_once":True}
            if mode == "train":
                rows = []
                for identifier in body["selection_run_ids"]:
                    directory, selected, status = self.task(identifier, True)
                    selected_config = read_json(directory / "config.json")
                    result = read_json(directory / "results.json")
                    if (selected["mode"] != "shorttune" or selected["algorithm"] != algorithm or
                            selected.get("prepare_run_id") != body["prepare_run_id"] or
                            status.get("env_step") != 32768):
                        raise ValueError("短调参来源、算法、数据或预算不匹配")
                    rows.append({"run_id": identifier, "candidate": selected_config["candidate"],
                                 **result["validation_summary"], "results_sha256": digest(directory / "results.json")})
                winner = select_candidate(rows)
                if "config" not in body and "candidate" not in body:
                    config = default_config(algorithm, winner, mode)
                if config["candidate"] != winner:
                    raise ValueError("正式候选不符合验证SI/SGM/A规则")
                extra["selection"] = {"candidate": winner, "rows": rows, "from_scratch": True}
            if mode == "evaluate":
                partition = body.get("partition", "test")
                if partition not in ("test", "validation"):
                    raise ValueError("只能评价冻结validation/test分区")
                count = len(extra["datasets"][partition]["files"])
                offset, number = body.get("offset", 0), body.get("count", count)
                if type(offset) is not int or type(number) is not int or not 0 <= offset < count or not 1 <= number <= count - offset:
                    raise ValueError("评价分片范围无效")
                extra.update(partition=partition, offset=offset, count=number)
                if algorithm in ALGORITHMS:
                    directory, model_job, status = self.task(body["model_run_id"], True)
                    if model_job["mode"] != "train" or model_job["algorithm"] != algorithm or status.get("env_step") != 400000:
                        raise ValueError("考核模型必须是本臂完整400000终点")
                    if model_job.get("prepare_run_id") != body["prepare_run_id"]:
                        raise ValueError("考核模型与共同数据准备任务来源不一致")
                    current_contract = {role: {"count": len(record["files"]),
                        "input_sha256": fingerprint([row["sha256"] for row in record["files"]]),
                        "mother_count": len({row.get("mother_scene_id") for row in record["files"]})}
                        for role, record in extra["datasets"].items()}
                    if model_job.get("dataset_contract") != current_contract:
                        raise ValueError("考核模型与本轮冻结训练/验证/考核输入SHA合同不一致")
                    model_config = validate_config(read_json(directory / "config.json"))
                    if (model_config["environment"] != config["environment"] or
                            model_config["network"] != config["network"]):
                        raise ValueError("考核模型环境与共同表示不匹配")
                    if "config" not in body and "candidate" not in body:
                        config = default_config(algorithm, model_config["candidate"], mode)
                    elif config["candidate"] != model_config["candidate"]:
                        raise ValueError("考核候选身份与实际终点模型不符")
                    model = Path(model_job["model_dir"]) / "final"
                    extra["model_files"] = [{"path": str(model / filename), "sha256": digest(model / filename)}
                                            for filename in ("policy.pt", "model.json")]
                    extra["model_run_id"] = model_job["run_id"]
                    extra["model_source_git_commit"] = model_job["git_commit"]
        # Store's generic science input branch is intentionally bypassed. Its
        # fair20 hook freezes all inputs before publishing the semantic job kind.
        extra["candidate"] = config["candidate"]
        return self.store.create_job(config, kind="prepare", extra=extra, request_id=body.get("request_id"))

    def resume(self, run_id):
        with self.app.scheduler.lock, self.store.lock:
            directory, job, status = self.task(run_id)
            if status.get("stage") not in ("failed", "stopped", "interrupted"):
                raise ValueError("仅异常或停止任务支持同ID恢复")
            from .runtime import pid_alive
            launch = read_json(directory / "launch.json", {})
            if pid_alive(status.get("pid"), launch.get("process_start_time")):
                raise ValueError("旧worker仍存活，拒绝重复调度")
            if job["kind"] == "train" and not (directory / "checkpoint.pt").is_file():
                raise ValueError("训练缺少完整状态断点，不能用推理权重替代")
            recovery = directory / "recovery_history" / now().replace(":", "_")
            recovery.mkdir(parents=True)
            for name in ("status.json", "failure.json", "job.json"):
                if (directory / name).exists():
                    shutil.copyfile(directory / name, recovery / name)
            job["resume"] = True
            job["resume_attempt"] = job.get("resume_attempt", 0) + 1
            job["resume_origin_stage"] = status["stage"]
            job["resume_origin_env_step"] = status.get("env_step")
            atomic_json(directory / "job.json", job)
            (directory / "stop.request").unlink(missing_ok=True)
            status.update(stage="queued_resources" if job.get("resource_hold") else "queued", pid=None,
                          error=None, queue_reason="S2-F20从完整冻结状态同ID恢复", heartbeat_at=now())
            atomic_json(directory / "status.json", status)
            return status

    def api(self, method, parts, body):
        if method == "GET" and parts == ["schema"]:
            configurations = {}
            for mode in ("sourceprep", "prepare", "shorttune", "train", "evaluate", "refine", "matrix"):
                algorithms = ("prepare",) if mode in ("sourceprep", "prepare", "matrix") else (*ALGORITHMS, "random", "greedy") if mode == "evaluate" else ALGORITHMS
                for algorithm in algorithms:
                    for candidate in ("A", "B"):
                        configurations[f"{mode}/{algorithm}/{candidate}"] = default_config(algorithm, candidate, mode)
            return {"experiment_id": EXPERIMENT, "algorithms": list(ALGORITHMS), "candidates": ["A", "B"],
                    "modes": ["sourceprep", "prepare", "shorttune", "train", "evaluate", "refine", "matrix"],
                    "default_config": default_config(), "formal_steps": 400000, "short_steps": 32768,
                    "validation_nodes": list(NODES), "warmup_included": True, "configurations": configurations}
        if method == "GET" and parts == ["tasks"]:
            summary_keys=("run_id","kind","name","mode","algorithm","candidate","git_commit","prepare_run_id","created_at","model_dir","experiment_id")
            return {"tasks": [{"job": {key:job.get(key) for key in summary_keys}, "status": self.app.observed_status(status)}
                              for job, status in self.store.jobs() if job.get("experiment_id") == EXPERIMENT]}
        if len(parts) >= 2 and parts[0] == "tasks":
            self.task(parts[1])
            if method == "GET" and len(parts) == 2:
                detail = self.app.run_detail(parts[1])
                detail["result"] = read_json(self.store.job_dir(parts[1]) / "results.json", {})
                from .monitoring import log_tail
                detail["log"] = log_tail(self.store.job_dir(parts[1]))
                return detail
            if method == "POST" and parts[2:] == ["resume"]:
                return self.resume(parts[1])
            if method == "POST" and parts[2:] == ["stop"]:
                directory, job, status = self.task(parts[1])
                if status.get("stage") == "queued_resources":
                    status.update(stage="cancelled", queue_reason="用户取消资源排队", heartbeat_at=now())
                    atomic_json(directory / "status.json", status)
                    return status
                return self.app.scheduler.request_stop(parts[1])
        if method == "POST" and parts == ["admit"]:
            with self.app.scheduler.lock:
                directory, job, status = self.task(body["run_id"])
                proof_path = Path(body["proof_path"]).resolve(strict=True)
                proof = read_json(proof_path, {})
                if digest(proof_path) != body["proof_sha256"] or proof.get("admitted") is not True or proof.get("run_id") != job["run_id"]:
                    raise ValueError("资源放行证明无效")
                validate_admission_proof(proof, job)
                if status.get("stage") not in ("queued_resources", "queued"):
                    raise ValueError("任务当前不等待资源放行")
                frozen = directory / "resource_admission.json"
                shutil.copyfile(proof_path, frozen)
                status.update(stage="queued", queue_reason="资源审查已放行", admission_sha256=digest(frozen))
                atomic_json(directory / "status.json", status)
                return status
        if method == "POST" and len(parts) == 1 and parts[0] in ("sourceprep", "prepare", "shorttune", "train", "engineering", "evaluate", "refine", "matrix"):
            return {"job": self.submit(body, parts[0])}
        raise FileNotFoundError("S2-F20接口不存在")
