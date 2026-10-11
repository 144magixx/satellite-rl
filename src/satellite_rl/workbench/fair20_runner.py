"""Persistent bounded S2-F20 workers with complete learning continuation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from satellite_rl.training.run_monitor import atomic_json, replace_with_retry
from .storage import clean_json, digest, read_json, now
from .fair20_service import EXPERIMENT, NODES, INPUT_FIELDS, validate_config, fingerprint


class Fair20RecoveryAttention(RuntimeError):
    """The retained checkpoint cannot account for all already spent interactions."""


def execution_metadata(job):
    return {"scientific_commit": job["git_commit"],
            "execution_source": deepcopy(job.get("execution_source")),
            "execution_commit": job.get("execution_source", {}).get("execution_commit", job["git_commit"])}


def validation_only_resume_verified(job, directory, saved, log_tails):
    """Require immutable SHA-bound evidence before excluding an unknown action.

    A missing or contradictory proof retains the existing conservative recovery
    rule. Validation cannot increment the learner interaction counter.
    """
    registration = job.get("validation_only_resume_proof", {})
    try:
        if log_tails or not registration or not job.get("execution_source"):
            return False
        path = Path(registration["proof_path"])
        if digest(path) != registration["proof_sha256"]:
            return False
        proof = read_json(path)
        step = saved["step"]
        if (proof["version"] != "fair20_validation_only_resume_v1" or step != 100000
                or saved["agent"]["interactions"] != step or step in saved["node_done"]
                or proof["run_id"] != job["run_id"] or saved["run_id"] != job["run_id"]
                or proof["scientific_commit"] != job["git_commit"]
                or proof["execution_commit"] != job["execution_source"]["execution_commit"]
                or proof["checkpoint_step"] != step or proof["agent_interactions"] != step
                or proof["failed_phase"] != "validation_100000"
                or proof["checkpoint_sha256"] != digest(Path(directory)/"checkpoint.pt")):
            return False
        records = proof["origin_records"]
        if set(records) != {"job", "status", "config", "launch", "failure", "checkpoint"}:
            return False
        for reference in records.values():
            if digest(Path(reference["path"])) != reference["sha256"]:
                return False
        origin_job = read_json(records["job"]["path"])
        status = read_json(records["status"]["path"])
        failure = read_json(records["failure"]["path"])
        if (origin_job["run_id"] != job["run_id"] or origin_job["git_commit"] != job["git_commit"]
                or records["checkpoint"]["sha256"] != proof["checkpoint_sha256"]
                or records["config"]["sha256"] != saved["config_sha256"]
                or records["config"]["sha256"] != digest(Path(directory)/"config.json")
                or records["status"]["sha256"] != proof["failed_status_sha256"]
                or records["failure"]["sha256"] != proof["failure_sha256"]
                or status["run_id"] != job["run_id"] or status["stage"] != "failed"
                or status["phase"] != "validation_100000" or status["env_step"] != step
                or status["checkpoint_sha256"] != proof["checkpoint_sha256"]
                or failure["run_id"] != job["run_id"] or failure["config_sha256"] != saved["config_sha256"]):
            return False
        stack = failure["traceback"]
        if not all(marker in stack for marker in ("in validate_node", "in _evaluate", "in full_b_case")):
            return False
        if job["git_commit"] not in stack:
            return False
        if any(status["clock"][key] != saved["clock"][key] for key in
               ("environment_seconds", "sampling_seconds", "update_seconds", "validation_seconds")):
            return False
        prior_io, measured_io = saved["clock"]["io_seconds"], status["clock"]["io_seconds"]
        if not np.isfinite([prior_io, measured_io]).all() or measured_io < prior_io:
            return False
        if set(proof["training_logs"]) != set(saved["log_sizes"]):
            return False
        for name, size in saved["log_sizes"].items():
            reference = proof["training_logs"][name]
            current = Path(directory)/name
            if (Path(reference["path"]).resolve() != current.resolve()
                    or current.stat().st_size != size or reference["saved_size"] != size
                    or proof["training_log_sizes"][name] != size or digest(current) != reference["sha256"]):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validation_only_recovered_clock(job, saved):
    """Called only after SHA-bound validation proof passes, preserving all work.

    The failed status contains the measured checkpoint-write overhead that was
    incremented after serialization. Incomplete validation wall time was never
    recorded; persisted per-case costs remain available but cannot replace it.
    """
    proof = read_json(job["validation_only_resume_proof"]["proof_path"])
    status = read_json(proof["origin_records"]["status"]["path"])
    clock = deepcopy(saved["clock"])
    previous_io = clock["io_seconds"]
    clock["io_seconds"] = status["clock"]["io_seconds"]
    metadata = {"post_checkpoint_measured_io_seconds_recovered": clock["io_seconds"]-previous_io,
                "checkpoint_io_seconds": previous_io, "failed_status_io_seconds": clock["io_seconds"],
                "source_failed_status_sha256": proof["failed_status_sha256"],
                "historical_validation_wall_seconds": None,
                "historical_validation_wall_missing_reason":
                    "Validation failed before its wall time was committed. Original persisted Direct/A/B/overhead case timings are retained; unrecorded wall time is unavailable and is not counted as zero or reconstructed from repeat work.",
                "validation_seconds_scope": "completed measured validation calls only; historical failed-call wall missing",
                "training_environment_sampling_update_seconds_unchanged": True}
    return clock, metadata


class FrozenActionPolicy:
    """One deployment action network for every arm; no critics, V or optimizer."""
    def __init__(self, algorithm, candidate, state, device="cuda"):
        from satellite_rl.fair_comparison.networks import TransformerNetwork
        self.algorithm,self.candidate,self.device=algorithm,candidate,torch.device(device)
        began=time.perf_counter()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(51001)
            self.policy=TransformerNetwork(1000)
        self.policy.load_state_dict(state,strict=True)
        cpu_load_seconds=time.perf_counter()-began
        began=time.perf_counter()
        self.policy=self.policy.to(self.device).eval()
        if self.device.type=="cuda":torch.cuda.synchronize()
        self.load_timing={"cpu_action_network_build_load_seconds":cpu_load_seconds,
                          "device_transfer_seconds":time.perf_counter()-began}
        self.policy.requires_grad_(False)
        self.parameter_count=sum(parameter.numel() for parameter in self.policy.parameters())

    def act(self, obs, mask, deterministic=True):
        from satellite_rl.fair_comparison.networks import tensor_observations
        if deterministic is not True:
            raise ValueError("冻结考核动作网络只允许单次合法argmax")
        legal=np.flatnonzero(np.asarray(mask,bool))
        if not len(legal):raise ValueError("考核合法集合为空")
        with torch.no_grad():
            logits=self.policy(tensor_observations(obs,self.device))[0]
            # np.argmax returns the first index, with legal indices ascending.
            action=int(legal[np.argmax(logits[legal].cpu().numpy())])
        return action,{"log_prob":0.,"value":0.,"epsilon":0.}


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False) + "\n")


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    replace_with_retry(temp, path)


def read_scene(reference):
    path = Path(reference["path"])
    if digest(path) != reference["sha256"]:
        raise ValueError("冻结场景SHA不符")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def validate_datasets(datasets, engineering=False, required_roles=("train", "validation", "test")):
    expected = {"train": (8064, 576, {"2024-01-08", "2024-01-09"}),
                "validation": (144, 12, {"2024-02-13"}),
                "test": (1320, 40, {"2024-02-14", "2024-02-15"})}
    expected = {role: expected[role] for role in required_roles}
    if set(datasets) != set(expected):
        raise ValueError("共同数据必须有train、validation和test三个分区")
    mother_sets = {}
    for role, record in datasets.items():
        refs = record.get("files", [])
        if not refs:
            raise ValueError("冻结场景集为空")
        mothers = {row.get("mother_scene_id") for row in refs}
        if None in mothers:
            raise ValueError("场景缺少母ID")
        source = record.get("source") or {}
        if source.get("partition_role") != role:
            raise ValueError("分区身份不符")
        if not engineering and (len(refs) != expected[role][0] or len(mothers) != expected[role][1]):
            raise ValueError(f"{role}数量偏离方案20")
        dates = set(source.get("dates", source.get("utc_dates", [])))
        # Per-scene date/timestamp is authoritative when the record lacks dates.
        if not dates:
            dates = {str(row.get("utc_date", row.get("date", row.get("selected_time_utc", row.get("mother_timestamp", row.get("timestamp", ""))))))[:10] for row in refs}
        if not engineering and dates != expected[role][2]:
            raise ValueError(f"{role}来源日期偏离共同合同：{dates}")
        hashes = [row["sha256"] for row in refs]
        if len(set(hashes)) != len(hashes):
            raise ValueError("分区有重复输入SHA")
        mother_sets[role] = mothers
    if any(mother_sets[a] & mother_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")) if a in mother_sets and b in mother_sets):
        raise ValueError("训练/验证/考核发生母级数据泄漏")
    return {role: fingerprint(record["files"]) for role, record in datasets.items()}


def action_for(env, obs, mask, method, agent=None, rng=None):
    if agent is not None:
        return agent.act(obs, mask, deterministic=True)[0]
    from satellite_rl.fair_comparison import baseline_action
    return baseline_action(env, method, rng)


def verify_snapshot(snapshot):
    """Independent full terminal physics and resource audit, including all N beams."""
    from satellite_rl.power_refinement.physics import FixedAllocationPhysics
    beams = snapshot["beams"]
    if any(row["slots"] < 1 for row in beams):
        raise ValueError("未完成全部目标束的非空分配")
    power = np.array([row["power_w"] for row in beams])
    if (not np.isfinite(power).all() or (power < 0).any() or power.sum() > 3000+1e-7):
        raise ValueError("功率资源核验失败")
    problem = FixedAllocationPhysics.from_scene(snapshot["scene"],
        {"environment": {"profile": "europe_v1", "power_budget_w": 3000}}, beams)
    actual = problem.evaluate(power)
    logged = np.array([row["rate_bps"] for row in beams])
    if not np.allclose(actual.capacity_bps, logged, rtol=1e-9, atol=1e-4):
        raise ValueError("终态容量独立复算不一致")
    for row, cap in zip(beams, actual.capacity_bps):
        row["capacity_bps"] = float(cap)
        row["effective_served_bps"] = float(min(cap, row["required_bps"]))
    snapshot["metrics"].update(si=actual.si, sgm=actual.sgm,
        SI=actual.si, SGM=actual.sgm, basic_fraction=float(np.mean(actual.capacity_bps >= .1*problem.demand_bps)),
        physical_violations=0, independent_terminal_verified=True)
    return problem


def direct_case(reference, method, agent, scene_index, stop):
    from satellite_rl.fair_comparison import Fair20Env
    began = time.perf_counter()
    env = Fair20Env(read_scene(reference))
    obs = env.reset()
    prepared = time.perf_counter()-began
    rng = np.random.default_rng(np.random.SeedSequence([53001, scene_index]))
    decision_seconds, mask_seconds = 0., 0.
    while not env.metrics()["complete"]:
        if stop():
            raise InterruptedError("direct评价收到停止请求")
        start = time.perf_counter()
        mask = env.action_mask()
        mask_seconds += time.perf_counter()-start
        if agent is not None and agent.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        action = action_for(env, obs, mask, method, agent, rng)
        if agent is not None and agent.device.type == "cuda":
            torch.cuda.synchronize()
        decision_seconds += time.perf_counter()-start
        obs, _, _, _ = env.step(action)
    snap = env.snapshot()
    start = time.perf_counter()
    verify_snapshot(snap)
    verification = time.perf_counter()-start
    snap.update(scene_reference=deepcopy(reference), method=method, scene_index=scene_index,
                repeat=0, action_mode="deterministic" if method != "random" else "uniform_legal",
                timing={"input_prepare_seconds": prepared, "decision_seconds": decision_seconds,
                        "mask_seconds": mask_seconds, "physics_seconds": env.physics_seconds,
                        "power_rule_seconds": None, "power_rule_timing_scope": "included_in_environment_step",
                        "verification_seconds": verification, "direct_total_seconds": time.perf_counter()-began})
    return snap


def full_b_case(initial, stop, log=None, journal=None):
    """Actual SCA A then SI-priority LP B, same allocation, no additional C."""
    from satellite_rl.power_refinement.optimizer import optimize_power
    from satellite_rl.power_refinement.priority import optimize_priority
    start = time.perf_counter()
    original = deepcopy(initial)
    problem = verify_snapshot(original)
    p0 = np.array([row["power_w"] for row in original["beams"]])
    options = {"power_cap_w": 3000., "maxiter": 300, "max_evaluations": 2000,
               "time_limit_seconds": 30., "si_tolerance": 1e-8, "sgm_tolerance": 1e-8,
               "protect_basic_service": True}
    journal = Path(journal) if journal is not None else None
    saved = read_json(journal, {}) if journal is not None else {}
    initial_sha = fingerprint(clean_json(initial))
    if saved and saved.get("initial_sha256") != initial_sha:
        raise ValueError("完整B阶段断点与冻结direct初解不符")
    if saved.get("phase") in ("A_started", "B_started"):
        raise ValueError("存在未提交的优化器尝试；原开销未知，禁止重启追加冻结预算")
    prior_overhead = saved.get("overhead_seconds", 0. if not saved else None)
    reused_a, reused_b = "A" in saved, "B" in saved
    local_a_time = local_b_time = 0.
    def commit(value):
        if journal is not None:
            atomic_json(journal, clean_json({"initial_sha256": initial_sha, **value}))
    if "A" in saved:
        A, a_time = saved["A"], saved["A_seconds"]
    else:
        commit({"phase": "A_started", "started_at": now()})
        a_start = time.perf_counter()
        A = optimize_power(problem, p0, method="sca", options=options,
                           callback=log, stop_requested=stop)
        a_time = time.perf_counter()-a_start
        local_a_time = a_time
        commit({"phase": "A_complete", "A": A, "A_seconds": a_time,
                "overhead_seconds": time.perf_counter()-start-a_time})
    if stop():
        commit({"phase": "A_complete", "A": A, "A_seconds": a_time,
                "overhead_seconds": None if prior_overhead is None else prior_overhead+time.perf_counter()-start-local_a_time})
        raise InterruptedError("完整B的A阶段收到停止请求")
    pA = np.asarray(A["power_w"])
    bopts = dict(options, si_stall_window=200, si_stall_tolerance=1e-6,
                 si_min_evaluations=200, si_max_fraction=.85, service_margin_ratio=5e-9,
                 baseline_service_floor_bps=np.minimum(np.array([row["rate_bps"] for row in original["beams"]]),
                                                      .1*problem.demand_bps))
    if "B" in saved:
        B, b_time = saved["B"], saved["B_seconds"]
    else:
        commit({"phase": "B_started", "A": A, "A_seconds": a_time, "started_at": now(),
                "overhead_seconds": None if prior_overhead is None else prior_overhead+time.perf_counter()-start-local_a_time})
        b_start = time.perf_counter()
        B = optimize_priority(problem, pA, method="sca_si_lp", options=bopts,
                              callback=log, stop_requested=stop)
        b_time = time.perf_counter()-b_start
        local_b_time = b_time
        commit({"phase": "B_complete", "A": A, "B": B, "A_seconds": a_time, "B_seconds": b_time,
                "overhead_seconds": None if prior_overhead is None else prior_overhead+time.perf_counter()-start-local_a_time-local_b_time})
    if stop():
        commit({"phase": "B_complete", "A": A, "B": B, "A_seconds": a_time, "B_seconds": b_time,
                "overhead_seconds": None if prior_overhead is None else prior_overhead+time.perf_counter()-start-local_a_time-local_b_time})
        raise InterruptedError("完整B的B阶段收到停止请求")
    for name, solver in (("A", A), ("B", B)):
        evaluations = solver.get("evaluations")
        elapsed = solver.get("elapsed_seconds")
        if (type(evaluations) is not int or not 1 <= evaluations <= 2000 or
                not isinstance(elapsed, (float, int)) or not np.isfinite(elapsed) or elapsed < 0 or
                not solver.get("termination_reason") or solver["termination_reason"] == "user_stop"):
            raise ValueError(f"{name}优化结果缺少有限预算证据或尚未完成")
    power = np.asarray(B["power_w"])
    baseline = problem.evaluate(p0)
    floor = np.minimum(baseline.capacity_bps, .1*problem.demand_bps)
    reasons = []
    valid_power = (power.shape == p0.shape and np.isfinite(power).all()
                   and not (power < 0).any() and not (power[~problem.active] != 0).any())
    if not valid_power:
        reasons.append("invalid_power_domain")
    elif power.sum() > 3000+1e-7:
        reasons.append("total_power_budget")
    candidate_ev = problem.evaluate(power) if valid_power else None
    shortfall_indices = []
    if candidate_ev is not None:
        if candidate_ev.si < baseline.si-1e-8:
            reasons.append("direct_si_protection")
        if candidate_ev.sgm < baseline.sgm-1e-8:
            reasons.append("direct_sgm_protection")
        shortfall_indices = np.flatnonzero(candidate_ev.capacity_bps < floor-1e-7).tolist()
        if shortfall_indices:
            reasons.append("direct_basic_service_floor")
    candidate = {"power_w": power.tolist(), "total_power_w": float(power.sum()),
                 "SI": None if candidate_ev is None else candidate_ev.si,
                 "SGM": None if candidate_ev is None else candidate_ev.sgm,
                 "capacity_bps": None if candidate_ev is None else candidate_ev.capacity_bps.tolist(),
                 "basic_met_count": None if candidate_ev is None else int(np.count_nonzero(
                     candidate_ev.capacity_bps >= .1*problem.demand_bps)),
                 "basic_fraction": None if candidate_ev is None else float(np.mean(
                     candidate_ev.capacity_bps >= .1*problem.demand_bps)),
                 "basic_floor_shortfall_beam_ids": shortfall_indices,
                 "basic_floor_shortfall_bps": None if candidate_ev is None else
                     np.maximum(floor-candidate_ev.capacity_bps, 0).tolist()}
    # The original strict verifier and solver budgets stay unchanged. A rejected
    # candidate is retained verbatim; the already verified Direct is the output.
    fallback = bool(reasons)
    ev = baseline if fallback else candidate_ev
    if fallback:
        power = p0.copy()
    for i, row in enumerate(original["beams"]):
        row.update(power_w=float(power[i]), rate_bps=float(ev.capacity_bps[i]),
                   si=float(ev.beam_si[i]), sgm=float(ev.beam_sgm[i]))
    verify_start = time.perf_counter()
    verify_snapshot(original)
    original["metrics"].update(total_power_w=float(power.sum()),
        total_capacity_bps=float(ev.capacity_bps.sum()), zero_service_fraction=float(np.mean(ev.capacity_bps == 0)))
    original.update(route="full_A_then_B", initial_allocation_reused=True,
                    acceptance={"accepted": not fallback, "fallback": fallback,
                        "fallback_source": "verified_direct" if fallback else None,
                        "rejection_reasons": reasons, "candidate": candidate,
                        "strict_guard": {"power_tolerance_w": 1e-7, "SI_tolerance": 1e-8,
                            "SGM_tolerance": 1e-8, "service_floor_tolerance_bps": 1e-7},
                        "returned_basic_met_count": int(np.count_nonzero(ev.capacity_bps >= .1*problem.demand_bps)),
                        "returned_basic_fraction": float(np.mean(ev.capacity_bps >= .1*problem.demand_bps)),
                        "additional_optimizer_calls": 0},
                    solver={"A": A, "B": B, "options_A": options,
                            "options_B": clean_json(bopts), "optimizer_seed": 63001,
                            "algorithm_randomness": "deterministic_LP", "no_extra_C": True})
    verification_seconds = time.perf_counter()-verify_start
    current_call_wall = time.perf_counter()-start
    local_overhead = current_call_wall-local_a_time-local_b_time
    cumulative_overhead = None if prior_overhead is None else prior_overhead+local_overhead
    route_total = None if cumulative_overhead is None else a_time+b_time+cumulative_overhead
    original["timing"].update(A_seconds=a_time, B_seconds=b_time,
        final_verification_seconds=verification_seconds, recovery_call_wall_seconds=current_call_wall,
        overhead_seconds=cumulative_overhead,
        refinement_total_seconds=route_total,
        direct_plus_refine_total_seconds=None if route_total is None else initial["timing"]["direct_total_seconds"]+route_total,
        historical_overhead_missing_reason=None if prior_overhead is not None else "prior optimizer journal did not record pre-optimizer/verification/I/O overhead",
        recovered_A=reused_a, recovered_B=reused_b,
        total_timing_scope="persisted_A+B_measured_overheads_shared_initial_once")
    commit({"phase": "complete", "A": A, "B": B, "A_seconds": a_time, "B_seconds": b_time,
            "overhead_seconds": cumulative_overhead, "timing": original["timing"],
            "acceptance": original["acceptance"]})
    return original


def _export(agent, job, config, directory, step, checkpoint_sha):
    from satellite_rl.fair_comparison import OBSERVATION_CONTRACT
    target = Path(job["model_dir"]) / ("final" if step == config["training"]["total_steps"] else f"step_{step:06d}")
    target.mkdir(parents=True, exist_ok=True)
    policy = {"format": "fair20_policy_v1", "algorithm": config["algorithm"],
              **execution_metadata(job),
              "candidate": config["candidate"], "seed": 51001,
              "models": {"policy": agent.policy.state_dict()},
              "deployment_networks":1,
              "env_step": step, "updates": agent.update_count}
    save_torch(target / "policy.pt", policy)
    model = {"format": "fair20_policy_v1", "run_id": job["run_id"], "algorithm": config["algorithm"],
             **execution_metadata(job),
             "candidate": config["candidate"], "config": config, "config_sha256": digest(directory / "config.json"),
             "git_commit": job["git_commit"], "env_step": step, "updates": agent.update_count,
             "policy_sha256": digest(target / "policy.pt"), "checkpoint_sha256": checkpoint_sha,
             "observation_contract": OBSERVATION_CONTRACT,
             "training_parameter_counts":{key:sum(parameter.numel() for parameter in model.parameters())
                                           for key,model in agent.models.items()},
             "deployment_parameter_count":sum(parameter.numel() for parameter in agent.policy.parameters()),
             "deployment_networks":1,
             "initialization_metadata": agent.initialization_metadata,
             "provenance": {"run_id": job["run_id"], "env_step": step, "seed": 51001,
                            "from_scratch": True, "source_manifest_sha256": digest(Path(job["source_root"]) / "source_manifest.json")}}
    atomic_json(target / "model.json", model)
    return target


def load_model(job, config):
    load_started=time.perf_counter()
    refs = {Path(row["source_path"]).name: row for row in job.get("model_files", [])}
    if set(refs) != {"policy.pt", "model.json"}:
        raise ValueError("模型输入缺少完整权重/元数据，拒绝随机回退")
    for row in refs.values():
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("冻结模型SHA不符")
    metadata = read_json(refs["model.json"]["path"])
    if (metadata.get("format") != "fair20_policy_v1" or metadata.get("algorithm") != config["algorithm"] or
            metadata.get("env_step") != 400000 or metadata.get("policy_sha256") != refs["policy.pt"]["sha256"]):
        raise ValueError("模型不是匹配本臂的完整400000终点")
    if (metadata.get("run_id") != job.get("model_run_id") or
            metadata.get("git_commit") != job.get("model_source_git_commit") or
            metadata.get("git_commit") != job.get("git_commit")):
        raise ValueError("模型来源run_id或科学冻结提交不匹配")
    model_config = validate_config(metadata.get("config"))
    if (model_config["mode"] != "train" or model_config["algorithm"] != config["algorithm"] or
            model_config["candidate"] != metadata.get("candidate")):
        raise ValueError("模型冻结配置身份无效")
    file_started=time.perf_counter()
    policy = torch.load(refs["policy.pt"]["path"], map_location="cpu", weights_only=False)
    file_seconds=time.perf_counter()-file_started
    if (policy.get("format") != metadata["format"] or policy.get("algorithm") != metadata["algorithm"] or
            policy.get("candidate") != metadata["candidate"] or policy.get("env_step") != 400000):
        raise ValueError("权重内部来源与冻结元数据身份不符")
    if set(policy["models"]) != {"policy"}:
        raise ValueError("模型网络角色不匹配")
    deployed=FrozenActionPolicy(config["algorithm"],metadata["candidate"],policy["models"]["policy"],device="cuda")
    deployed.load_timing.update(policy_file_read_seconds=file_seconds,
        loader_total_seconds=time.perf_counter()-load_started,deployment_action_networks=1,
        deployment_parameter_count=deployed.parameter_count,load_device="one_action_network_to_cuda")
    return deployed


def _evaluate(directory, job, config, progress, stop, agent=None, partition=None,
              prefix="evaluation", full_b=False):
    from .runner import isolated_rng
    partition = partition or job.get("partition", "test")
    refs = job["datasets"][partition]["files"]
    offset = job.get("offset", 0) if prefix == "evaluation" else 0
    count = job.get("count", len(refs)) if prefix == "evaluation" else len(refs)
    target = directory / prefix
    target.mkdir(exist_ok=True)
    state_path = target / "state.json"
    state = read_json(state_path, {"cases": []})
    cases = state["cases"]
    for i, reference in enumerate(cases):
        if reference["scene_sha256"] != refs[offset+i]["sha256"] or digest(reference["path"]) != reference["sha256"]:
            raise ValueError("评价恢复记录与冻结输入不符")
        if full_b and (not reference.get("B_path") or digest(reference["B_path"]) != reference["B_sha256"]):
            raise ValueError("节点评价完整B恢复记录缺失")
    with isolated_rng(53001):
        for index in range(len(cases), count):
            if stop():
                raise InterruptedError("评价收到停止请求")
            row = refs[offset+index]
            path = target / f"direct_{index:05d}.json"
            # A direct case atomically persisted before B is reused after failure.
            initial = read_json(path)
            if initial is None:
                initial = direct_case(row, config["algorithm"], agent, offset+index, stop)
                atomic_json(path, clean_json(initial))
            elif initial["scene_reference"]["sha256"] != row["sha256"]:
                raise ValueError("恢复初解与本场景不符")
            ref = {"path": str(path), "sha256": digest(path), "scene_sha256": row["sha256"],
                   "mother_scene_id": row["mother_scene_id"], "SI": initial["metrics"]["SI"], "SGM": initial["metrics"]["SGM"]}
            if full_b:
                bpath = target / f"B_{index:05d}.json"
                final = read_json(bpath)
                if final is None:
                    final = full_b_case(initial, stop,
                        log=lambda entry: append(directory / "power_iterations.jsonl", {"prefix": prefix, "scene_index": index, **entry}),
                        journal=target / f"optimizer_{index:05d}.json")
                    atomic_json(bpath, clean_json(final))
                ref.update(B_path=str(bpath), B_sha256=digest(bpath), B_SI=final["metrics"]["SI"], B_SGM=final["metrics"]["SGM"])
            cases.append(ref)
            atomic_json(state_path, {"cases": cases})
            progress(phase=prefix, completed_scenes=len(cases), total_scenes=count)
    return {"case_files": cases, "model_load_timing":getattr(agent,"load_timing",None),
            "model_load_timing_scope":"separate_once_per_worker; inline validation retains its already trained agent",
            "summary": {"SI": float(np.mean([r["SI"] for r in cases])),
                                               "SGM": float(np.mean([r["SGM"] for r in cases]))}}


def _matrix(directory, job, config, progress, stop):
    """Interleave five methods for each scene; every direct and B occurs once."""
    from satellite_rl.fair_comparison import Fair20Env
    from .fair20_service import default_config
    from .runtime import gpu_snapshot
    from .runner import isolated_rng
    references=job["datasets"]["test"]["files"]
    methods=("sac","ppo","dqn","random","greedy")
    order_contract={"methods":list(methods),"rule":"scene_index modulo5 rotates fixed method sequence",
                    "warmup_network_calls_per_model":4,"all_cases_once":True}
    if job.get("method_order") != order_contract:
        raise ValueError("matrix方法交错顺序偏离事前共同计时合同")
    agents={}
    for method in methods[:3]:
        source=job["models_by_algorithm"][method]
        agents[method]=load_model({**job,**source},default_config(method,source["candidate"],"evaluate"))
    target=directory/"matrix_cases";target.mkdir(exist_ok=True)
    warmup_path=directory/"matrix_warmup.json"
    warmup_previous=read_json(warmup_path,{"attempts":[]})
    warmup_environment=Fair20Env(read_scene(references[0]))
    observation=warmup_environment.reset()
    mask=warmup_environment.action_mask()
    warmup_started=time.perf_counter()
    for method,agent in agents.items():
        for _ in range(4):agent.act(observation,mask,deterministic=True)
    torch.cuda.synchronize()
    warmup_previous["attempts"].append({"at":now(),"seconds":time.perf_counter()-warmup_started,
                                      "network_calls_per_model":4,"no_environment_step":True})
    atomic_json(warmup_path,warmup_previous)
    state_path=directory/"matrix_progress.json"
    state=read_json(state_path,{"case_files":[]})
    rows=state["case_files"]
    expected=[(index,methods[(index%5+position)%5]) for index in range(len(references)) for position in range(5)]
    if len(rows)>len(expected):raise ValueError("matrix恢复账超出6600次正式考核")
    for position,row in enumerate(rows):
        index,method=expected[position]
        if (row["scene_index"] != index or row["method"] != method or row["scene_sha256"] != references[index]["sha256"] or
                digest(row["direct_path"]) != row["direct_sha256"] or digest(row["B_path"]) != row["B_sha256"]):
            raise ValueError("matrix持久案例与冻结场景、顺序或原结果SHA不一致")
    with isolated_rng(53001):
        for ordinal in range(len(rows),len(expected)):
            if stop():raise InterruptedError("matrix收到停止请求")
            scene_index,method=expected[ordinal]
            reference=references[scene_index]
            control_path=target/f"control_{scene_index:05d}.json"
            if not control_path.exists():
                gpu=gpu_snapshot()
                external=[pid for pid in gpu.get("compute_pids",[]) if pid != os.getpid()]
                atomic_json(control_path,{"at":now(),"gpu":gpu,"cpu_threads":torch.get_num_threads(),
                    "cpu_interop_threads":torch.get_num_interop_threads(),"method_order":expected[scene_index*5:(scene_index+1)*5],
                    "strict_gpu_isolation_observed":not gpu.get("error") and not external,
                    "external_gpu_processes":external,"matrix_worker_pid":os.getpid(),
                    "remaining_cpu_host_contention": "other physical-lane workloads must be audited independently"})
            direct_path=target/f"direct_{scene_index:05d}_{method}.json"
            b_path=target/f"B_{scene_index:05d}_{method}.json"
            direct=read_json(direct_path)
            if direct is None:
                direct=direct_case(reference,method,agents.get(method),scene_index,stop)
                direct.update(method_position=ordinal%5,method_order_rule=order_contract["rule"])
                atomic_json(direct_path,clean_json(direct))
            elif direct.get("scene_reference",{}).get("sha256") != reference["sha256"] or direct.get("method") != method:
                raise ValueError("matrix暂存direct来源不匹配")
            final=read_json(b_path)
            if final is None:
                final=full_b_case(direct,stop,lambda entry:append(directory/"power_iterations.jsonl",
                    {"scene_index":scene_index,"method":method,**entry}),
                    journal=target/f"optimizer_{scene_index:05d}_{method}.json")
                atomic_json(b_path,clean_json(final))
            control=read_json(control_path)
            row={"method":method,"scene_index":scene_index,"method_position":ordinal%5,
                 "scene_sha256":reference["sha256"],"mother_scene_id":reference["mother_scene_id"],
                 "beam_count":reference["beam_count"],"load_multiplier":reference["load_multiplier"],
                 "direct_path":str(direct_path),"direct_sha256":digest(direct_path),
                 "B_path":str(b_path),"B_sha256":digest(b_path),
                 "direct_SI":direct["metrics"]["SI"],"direct_SGM":direct["metrics"]["SGM"],
                 "B_SI":final["metrics"]["SI"],"B_SGM":final["metrics"]["SGM"],
                 "direct_seconds":direct["timing"]["direct_total_seconds"],
                 "direct_plus_B_seconds":final["timing"]["direct_plus_refine_total_seconds"],
                 "control_path":str(control_path),"control_sha256":digest(control_path),
                 "strict_gpu_isolation_observed":control["strict_gpu_isolation_observed"]}
            rows.append(row)
            atomic_json(state_path,{"case_files":rows,"budget_direct_cases":6600,"budget_full_B_cases":6600})
            progress(stage="evaluating",phase="matrix",completed_scenes=scene_index+1,
                     completed_cases=len(rows),total_cases=len(expected),completed_full_B=len(rows),method=method)
    summary={}
    for method in methods:
        subset=[row for row in rows if row["method"]==method]
        summary[method]={key:float(np.mean([row[key] for row in subset])) for key in
                         ("direct_SI","direct_SGM","B_SI","B_SGM","direct_seconds")}
        summary[method]["count"]=len(subset)
    return {"case_files":rows,"summary_by_method":summary,"direct_count":len(rows),"full_B_count":len(rows),
            "method_order":order_contract,"warmup":warmup_previous,
            "model_load_timing":{method:getattr(agent,"load_timing",None) for method,agent in agents.items()},
            "strict_speed_claim_ready":False,"speed_claim_limitation":"CPU contention and all scene-level isolation evidence require independent final audit."}


def _train(directory, job, config, resume, progress, stop):
    from satellite_rl.fair_comparison import Fair20Env, make_agent
    from .runner import seed_training
    seed_training(51001)
    agent = make_agent(config["algorithm"], config["candidate"], device="cuda")
    references = job["datasets"]["train"]["files"]
    order_rng = np.random.default_rng(52001)
    order = order_rng.permutation(len(references)).tolist()
    order_hash = fingerprint([references[i]["sha256"] for i in order])
    atomic_json(directory / "scene_order.json", {"seed": 52001, "order": order, "source_sha256": order_hash,
                "rule": "fixed_permutation_cycled_by_episode", "no_policy_dependent_order": True})
    step, episode, node_done, clock = 0, 0, [], {"environment_seconds": 0., "sampling_seconds": 0.,
                                                     "update_seconds": 0., "validation_seconds": 0., "io_seconds": 0.}
    env = Fair20Env(read_scene(references[order[0]]))
    obs = env.reset()
    checkpoint = directory / "checkpoint.pt"
    recovery_budget = {"retained_interactions": 0, "known_uncommitted_interactions": 0,
                       "unknown_failed_interactions_upper_bound": 0, "strict_actual_budget_verified": True}
    validation_failure_verified = False
    if resume:
        saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
        if saved["config_sha256"] != digest(directory / "config.json") or saved["order_sha256"] != order_hash or saved["run_id"] != job["run_id"]:
            raise ValueError("训练断点身份/配置/场景顺序不匹配")
        log_tails = {}
        recovery_root = directory / "recovery_history" / f"strict_budget_attempt_{job.get('resume_attempt', 0):04d}"
        for name, size in saved.get("log_sizes", {}).items():
            path = directory / name
            if not path.is_file() or path.stat().st_size < size:
                raise Fair20RecoveryAttention("持久日志短于完整断点，无法核对已消耗交互")
            if path.stat().st_size > size:
                recovery_root.mkdir(parents=True, exist_ok=True)
                target = recovery_root / (name + ".uncommitted_tail")
                with path.open("rb") as source, target.open("wb") as destination:
                    source.seek(size)
                    for block in iter(lambda: source.read(1024*1024), b""):
                        destination.write(block)
                log_tails[name] = {"path": str(target), "sha256": digest(target),
                                   "bytes": target.stat().st_size}
        # A log not yet present in the checkpoint is also uncommitted work.
        for name in ("traces.jsonl", "episodes.jsonl", "updates.jsonl"):
            path = directory / name
            if name not in saved.get("log_sizes", {}) and path.is_file() and path.stat().st_size:
                recovery_root.mkdir(parents=True, exist_ok=True)
                target = recovery_root / (name + ".uncommitted_tail")
                import shutil
                shutil.copyfile(path, target)
                log_tails[name] = {"path": str(target), "sha256": digest(target), "bytes": target.stat().st_size}
        recovery_budget = deepcopy(saved.get("recovery_budget", recovery_budget))
        if log_tails:
            committed_steps = []
            trace = log_tails.get("traces.jsonl")
            malformed = False
            if trace:
                with Path(trace["path"]).open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            committed_steps.append(json.loads(line)["env_step"])
                        except (ValueError, KeyError):
                            malformed = True
            known = len({step for step in committed_steps if step > saved["step"]})
            ledger = {"checkpoint_retained_interactions": saved["step"], "known_uncommitted_interactions": known,
                      "unknown_failed_interactions_upper_bound": 1,
                      "strict_actual_budget_verified": False, "uncommitted_logs": log_tails,
                      "malformed_trace_tail": malformed, "automatic_reexecution_forbidden": True,
                      "reason": "Offline reconstruction or an explicit equal-budget repair protocol is required; no new interactions were executed."}
            atomic_json(recovery_root / "budget_attention.json", ledger)
            raise Fair20RecoveryAttention("断点后已有未提交交互/日志，原字节与开销账已保留；禁止回滚后重复交互扩大公平预算")
        if job.get("resume_origin_stage") in ("failed", "interrupted"):
            if validation_only_resume_verified(job, directory, saved, log_tails):
                validation_failure_verified = True
                recovery_budget["validation_only_failure_verified"] = True
                recovery_budget["validation_only_resume_proof"] = deepcopy(job["validation_only_resume_proof"])
            else:
                recovery_budget["unknown_failed_interactions_upper_bound"] += 1
                recovery_budget["strict_actual_budget_verified"] = False
                recovery_budget["unknown_reason"] = "Crash may occur after environment action but before committed trace; at most one unlogged transition per failure is unknown."
        agent.load_state_dict(saved["agent"])
        env.load_state_dict(saved["environment"])
        obs = env.observation()
        step, episode, node_done, clock = saved["step"], saved["episode"], saved["node_done"], saved["clock"]
        if validation_failure_verified:
            clock, timing_recovery = validation_only_recovered_clock(job, saved)
            recovery_budget["timing_recovery"] = timing_recovery
        if agent.interactions != step:
            raise ValueError("agent与训练账交互数不一致")
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in saved["rng"]["cuda"]])
    budget = config["training"]["total_steps"]
    status_last_at=time.monotonic()
    if not 0 <= step <= budget:
        raise ValueError("断点步数超过有限预算")
    def persist(stage="training"):
        started = time.perf_counter()
        value = {"format": "fair20_checkpoint_v1", "run_id": job["run_id"], "config_sha256": digest(directory / "config.json"),
                 **execution_metadata(job),
                 "order_sha256": order_hash, "step": step, "episode": episode,
                 "agent": agent.state_dict(), "environment": env.state_dict(), "node_done": node_done,
                 "clock": clock, "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                                           "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                 "log_sizes": {name: (directory / name).stat().st_size for name in ("traces.jsonl", "episodes.jsonl", "updates.jsonl") if (directory / name).exists()}}
        recovery_budget["retained_interactions"] = step
        value["recovery_budget"] = deepcopy(recovery_budget)
        save_torch(checkpoint, value)
        sha = digest(checkpoint)
        clock["io_seconds"] += time.perf_counter()-started
        progress(stage=stage, env_step=step, update_step=agent.update_count, updates=agent.update_count,
                 **execution_metadata(job),
                 episode=episode, checkpoint_sha256=sha, clock=clock, recovery_budget=recovery_budget,
                 **({"stop_reason": "user_stop"} if stage == "stopped" else {}))
        return sha
    if not resume:
        persist()
    def validate_node():
        if step in NODES and config["mode"] == "train" and step not in node_done:
            sha = persist()
            model = _export(agent, job, config, directory, step, sha)
            started = time.perf_counter()
            result = _evaluate(directory, job, config, progress, stop, agent,
                               partition="validation", prefix=f"validation_{step}", full_b=True)
            clock["validation_seconds"] += time.perf_counter()-started
            atomic_json(directory / f"validation_{step}" / "results.json", result)
            node_done.append(step)
            append(directory / "evaluations.jsonl", {"env_step": step, "direct": result["summary"],
                    "B": {"SI": float(np.mean([r["B_SI"] for r in result["case_files"]])),
                          "SGM": float(np.mean([r["B_SGM"] for r in result["case_files"]]))},
                    "model_path": str(model), "checkpoint_sha256": sha})
            persist()
    validate_node()  # A crash during node evaluation resumes that node first.
    while step < budget:
        if stop():
            persist("stopped")
            return None
        mask = env.action_mask()
        began = time.perf_counter()
        action, metadata = agent.act(obs, mask)
        clock["sampling_seconds"] += time.perf_counter()-began
        began = time.perf_counter()
        next_obs, reward, terminal, info = env.step(action)
        clock["environment_seconds"] += time.perf_counter()-began
        next_mask = env.action_mask()
        step += 1
        truncated = not terminal and (step == budget or step in NODES)
        began = time.perf_counter()
        previous_updates = agent.update_count
        update = agent.observe(obs, mask, action, reward, next_obs, next_mask, terminal,
                               truncated=truncated, metadata=metadata)
        clock["update_seconds"] += time.perf_counter()-began
        append(directory / "traces.jsonl", {"env_step": step, "episode": episode,
                    "scene_sha256": references[order[episode % len(order)]]["sha256"],
                    "reward": reward, **info, "truncated": truncated})
        if agent.update_count != previous_updates:
            append(directory / "updates.jsonl", {"env_step": step, "updates": agent.update_count, **agent.last_update})
        obs = next_obs
        if terminal:
            snapshot = env.snapshot()
            verify_snapshot(snapshot)
            if not np.isclose(snapshot["metrics"]["reward_sum"], 100*snapshot["metrics"]["si"], atol=1e-7):
                raise ValueError("回合奖励望远镜核验失败")
            append(directory / "episodes.jsonl", {"episode": episode, "env_step": step,
                    "scene_sha256": references[order[episode % len(order)]]["sha256"],
                    "metrics": snapshot["metrics"], "beams": snapshot["beams"]})
            episode += 1
            env = Fair20Env(read_scene(references[order[episode % len(order)]]))
            obs = env.reset()
        if step in NODES or step == budget:
            began = time.perf_counter()
            agent.flush()  # PPO partial rollout uses the real tail and bootstrap.
            clock["update_seconds"] += time.perf_counter()-began
            persist()
            validate_node()
        elif step % config["training"]["checkpoint_interval"] == 0:
            persist()
        elif step % 100 == 0 or time.monotonic()-status_last_at>=5.:
            progress(stage="training", env_step=step, update_step=agent.update_count,
                     updates=agent.update_count, episode=episode, clock=clock)
            status_last_at=time.monotonic()
    sha = persist()
    model = _export(agent, job, config, directory, step, sha)
    result = {"env_step": step, "updates": agent.update_count, "episodes": episode,
              "model_path": str(model), "node_done": node_done, "clock": clock,
              "from_scratch": True, "initialization_metadata": agent.initialization_metadata}
    result["budget_actual_interactions_proof"] = recovery_budget
    if config["mode"] == "shorttune":
        validation = _evaluate(directory, job, config, progress, stop, agent,
                               partition="validation", prefix="short_validation", full_b=False)
        result["validation_summary"] = validation["summary"]
        result["case_files"] = validation["case_files"]
    return result


def run_job(job_dir, *, resume=False):
    from .runner import strict_cuda_preflight
    directory = Path(job_dir).resolve()
    job, config = read_json(directory / "job.json", {}), read_json(directory / "config.json", {})
    status = read_json(directory / "status.json", {})
    config_sha = digest(directory / "config.json")
    def progress(**values):
        if digest(directory / "config.json") != config_sha:
            raise ValueError("运行时冻结配置发生变化")
        status.update(values, run_id=job["run_id"], pid=os.getpid(), heartbeat_at=now())
        atomic_json(directory / "status.json", status)
    stop = lambda: (directory / "stop.request").is_file()
    try:
        validate_config(config)
        if job.get("frozen_input_sha256") != fingerprint({key: job.get(key) for key in INPUT_FIELDS}):
            raise ValueError("冻结任务输入合同发生变化")
        if job.get("experiment_id") != EXPERIMENT or config["mode"] != job.get("mode"):
            raise ValueError("任务身份不符")
        from .fair20_service import validate_execution_job, execution_view
        validate_execution_job(job, directory)
        preflight = strict_cuda_preflight(execution_view(job))
        torch.set_num_threads(config["training"]["cpu_threads"])
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        preflight.update(cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
                         precision="float32", cuda_matmul_tf32=False, cudnn_tf32=False,
                         float32_matmul_precision=torch.get_float32_matmul_precision())
        progress(stage="initializing", preflight=preflight, budget_env_steps=config["training"]["total_steps"],
                 **execution_metadata(job))
        mode = config["mode"]
        if mode == "sourceprep":
            from .fair20_data import run_sourceprep
            result = run_sourceprep(directory, job["sourceprep"],
                                    lambda value: progress(stage="preparing", **value), stop)
            validate_datasets(result["datasets"])
        elif mode == "prepare":
            hashes = validate_datasets(job["datasets"])
            for role, record in job["datasets"].items():
                for index, reference in enumerate(record["files"]):
                    if stop():
                        raise InterruptedError("准备收到停止请求")
                    from satellite_rl.fair_comparison import Fair20Env
                    env = Fair20Env(read_scene(reference))
                    if not np.array_equal(env.groups, np.arange(env.n) % 8):
                        raise ValueError("共同固定分组不符")
                    if index % 100 == 0:
                        progress(stage="preparing", partition=role, completed_scenes=index)
            result = {"datasets": job["datasets"], "dataset_sha256": hashes, "validated": True}
        elif mode in ("train", "shorttune", "engineering"):
            validate_datasets(job["datasets"], engineering=mode == "engineering", required_roles=("train", "validation"))
            result = _train(directory, job, config, resume, progress, stop)
            if result is None:
                return None
        elif mode == "evaluate":
            validate_datasets(job["datasets"], required_roles=(job["partition"],))
            agent = load_model(job, config) if config["algorithm"] in ("sac", "ppo", "dqn") else None
            progress(stage="evaluating")
            result = _evaluate(directory, job, config, progress, stop, agent)
        elif mode == "matrix":
            validate_datasets(job["datasets"],required_roles=("test",))
            progress(stage="evaluating",phase="matrix",budget_direct_cases=6600,budget_full_B_cases=6600)
            result=_matrix(directory,job,config,progress,stop)
        elif mode == "refine":
            progress(stage="running")
            target = directory / "power_cases"
            target.mkdir(exist_ok=True)
            result = {"case_files": []}
            for index, reference in enumerate(job["direct_files"]):
                if digest(reference["path"]) != reference["sha256"]:
                    raise ValueError("direct初解SHA不符")
                path = target / f"B_{index:05d}.json"
                final = read_json(path)
                if final is None:
                    final = full_b_case(read_json(reference["path"]), stop,
                                       lambda entry: append(directory / "power_iterations.jsonl", {"scene_index": index, **entry}),
                                       journal=target / f"optimizer_{index:05d}.json")
                    atomic_json(path, clean_json(final))
                result["case_files"].append({"path": str(path), "sha256": digest(path),
                        "direct_source_sha256": reference["sha256"], "SI": final["metrics"]["SI"], "SGM": final["metrics"]["SGM"]})
                progress(completed_scenes=index+1, total_scenes=len(job["direct_files"]))
            result["summary"] = {key: float(np.mean([r[key] for r in result["case_files"]])) for key in ("SI", "SGM")}
        else:
            raise ValueError("未知S2-F20模式")
        result.update(execution_metadata(job))
        atomic_json(directory / "results.json", clean_json(result))
        atomic_json(directory / "artifacts.json", {name: str(directory / name) for name in
            ("config.json", "status.json", "checkpoint.pt", "scene_order.json", "traces.jsonl", "episodes.jsonl",
             "updates.jsonl", "evaluations.jsonl", "power_iterations.jsonl", "results.json", "worker.log") if (directory / name).exists()})
        progress(stage="completed", completed_at=now(), results_sha256=digest(directory / "results.json"))
        return result
    except Fair20RecoveryAttention as error:
        atomic_json(directory / "attention_required.json", {"at": now(), "error": str(error),
                    "run_id": job.get("run_id"), "automatic_reexecution_forbidden": True})
        progress(stage="attention_required", error=str(error),
                 attention_path=str(directory / "attention_required.json"), strict_actual_budget_verified=False)
        return None
    except InterruptedError as error:
        progress(stage="stopped", stop_reason="user_stop", error=str(error))
        return None
    except Exception as error:
        atomic_json(directory / "failure.json", {"at": now(), "error": str(error), "traceback": traceback.format_exc(),
                    "config_sha256": config_sha, "run_id": job.get("run_id"), "resume": resume})
        status.update(stage="failed", error=str(error), failure_path=str(directory / "failure.json"), heartbeat_at=now())
        atomic_json(directory / "status.json", status)
        raise
