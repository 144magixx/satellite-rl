"""Schema-3 first-stage jobs. Every long operation runs through a frozen workbench job."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import torch

from .config import config_hash
from .runner import (_dataset, _event, _jsonl, _jsonl_many, _load_torch, _now,
                     _jsonable, _read, _restore_rng, _rng_state, _save_torch, _seed, _sha256,
                     _status, _stop_requested, _write_json)
from .ppo import CoveragePolicy, Rollout, ppo_update


STOP_REASON_LABELS = {0: "运行中", 1: "初始已达标", 2: "全局目标达到",
                      3: "最大轮数", 4: "束处理预算耗尽",
                      5: "归属尝试预算耗尽", 6: "细调时间上限",
                      7: "中心候选核验预算耗尽"}
REJECTION_REASON_LABELS = {0: "调整已执行", 1: "会失去已保护用户覆盖",
                           2: "会形成空束", 3: "该场景已停止",
                           4: "主动保持", 5: "归属尝试预算耗尽",
                           6: "中心候选核验预算耗尽",
                           7: "外部时间/初态终止快照"}


def _optional_float(value):
    number = float(value)
    return number if np.isfinite(number) else None


def _due_validation_threshold(actual_step, last_completed_threshold, interval):
    """Return one frozen target crossed by a completed rollout, never a step-0 node."""
    if interval <= 0 or last_completed_threshold < 0:
        raise ValueError("Invalid S1-02 validation threshold ledger")
    scheduled = last_completed_threshold + interval
    return scheduled if actual_step >= scheduled else None


def _starts(config, role, dataset):
    from .balancing import BalanceStart

    record = config["balance_starts"][role]
    source = Path(record["path"])
    if _sha256(source) != record["sha256"]:
        raise ValueError(f"{role} frozen A2 start SHA mismatch")
    if record["dataset_sha256"] != config["datasets"][role]["sha256"]:
        raise ValueError(f"{role} A2 start references different scenes")
    lines = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != len(dataset):
        raise ValueError(f"{role} A2 start count differs from dataset")
    starts = []
    for scene, row in zip(dataset.scenes, lines):
        if row.get("status") == "initialization_failed":
            body = {key: value for key, value in row.items() if key != "start_sha256"}
            actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
            if (actual != row.get("start_sha256") or row.get("scene_id") != scene.scene_id or
                    row.get("scene_sha256") != scene.source_hash or
                    row.get("aircraft_ids") != scene.aircraft_ids):
                raise ValueError(f"{role} failed A2 start provenance mismatch")
            starts.append(row)
        else:
            starts.append(BalanceStart.from_record(scene, row))
    return starts


def _training_eligibility(directory, dataset, starts, *, resume):
    entries = []
    eligible = []
    for index, (scene, start) in enumerate(zip(dataset.scenes, starts)):
        status = start.get("status") if isinstance(start, dict) else start.status
        accepted = status == "completed"
        if accepted:
            eligible.append(index)
        entries.append({"original_scene_index": index, "scene_id": scene.scene_id,
                        "scene_sha256": scene.source_hash, "status": status,
                        "eligible": accepted,
                        "failure_reason": start.get("failure_reason") if isinstance(start, dict)
                        else None if accepted else "A2起点未达到双99.9%覆盖门槛",
                        "start_sha256": start["start_sha256"] if isinstance(start, dict)
                        else start.start_sha256})
    if not eligible:
        raise ValueError("全部训练A2起点未达到覆盖资格，不能启动PPO")
    body = {"source_scene_count": len(dataset), "eligible_scene_count": len(eligible),
            "eligible_original_indices": eligible, "scenes": entries}
    body["eligibility_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    path = directory / "training_eligibility.json"
    if resume and _read(path) != body:
        raise ValueError("训练场景资格与冻结断点不一致")
    if not resume:
        _write_json(path, body)
    return body


def _policy(config, env):
    n = config["network"]
    return CoveragePolicy(env.obs_dim, action_dim=env.action_dim,
                          hidden_sizes=n["hidden_sizes"],
                          candidate_dim=env.candidate_dim).to("cuda")


def _row_metrics(metrics, index):
    result = {}
    for key, value in metrics.items():
        if key in ("scene_index", "done") or not torch.is_tensor(value) or value[index].numel() != 1:
            continue
        number = float(value[index].item())
        result[key] = number if np.isfinite(number) else None
    return result


def _scene_input(scene, start=None):
    result = {"aircraft_ids": scene.aircraft_ids, "positions": scene.positions,
            "demands_bps": scene.demands, "initial_centers": scene.centers,
            "metadata": scene.metadata}
    if hasattr(start, "centers"):
        result["movement_reference_centers"] = scene.centers
        result["a2_start_centers"] = start.centers
    return result


def _physical_not_applicable():
    return {"capacity_C": None, "effective_carried_min_C_D": None,
            "capacity_demand_ratio": None, "SI": None, "SGM": None,
            "reason": "第一阶段仅计算几何覆盖与业务归属，未运行链路容量模型"}


def _unready_start_row(scene, index, start, config):
    from .balancing import run_balance_route

    result = run_balance_route(scene, config["coverage"], "A2", common=start,
                               device="cuda")
    result.update(scene_index=index, scene_sha256=scene.source_hash,
                  action_trace=[], PPO_adjustment_performed=False,
                  failure_reason=(start.get("failure_reason") if isinstance(start, dict)
                                  else "共同A2起点未达到冻结覆盖门槛；PPO未执行调整"),
                  physical_metrics_not_applicable=_physical_not_applicable())
    return result


def _partial_rows(path, resume):
    path = Path(path)
    if not resume:
        path.write_text("", encoding="utf-8")
        return []
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            break  # A kill during the final append cannot publish a partial row.
        if row.get("scene_index") != len(rows):
            raise ValueError("S1-02 partial result scene order differs from frozen input")
        rows.append(row)
    # Remove a torn trailing JSONL line before appending after recovery.
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            for row in rows), encoding="utf-8")
    return rows


def _verify_partial_scenes(rows, dataset, *, wrapper=False):
    if len(rows) > len(dataset):
        raise ValueError("S1-02 partial result exceeds frozen scene count")
    for index, row in enumerate(rows):
        value = row["record"] if wrapper else row
        if (value.get("scene_id") != dataset.scenes[index].scene_id or
                value.get("scene_sha256", value.get("source_hash")) !=
                dataset.scenes[index].source_hash):
            raise ValueError("S1-02 partial result scene source differs from frozen input")


def _terminal_row(scene, local, info, trace, *, step, group, elapsed, start=None):
    raw_metrics = _row_metrics(info["episode_metrics"], local)
    metrics = _row_metrics(info.get("best_metrics", info["episode_metrics"]), local)
    for key in ("stop_reason_code",):
        if key in raw_metrics:
            metrics[key] = raw_metrics[key]
    raw_assignments = info["terminal_associations"][local]
    aircraft_mask = info["terminal_aircraft_mask"][local]
    raw_assignments = raw_assignments[aircraft_mask].detach().cpu().numpy()
    raw_layouts = info["terminal_layouts"][local].detach().cpu().numpy()
    assignments = info.get("best_associations", info["terminal_associations"])[local]
    assignments = assignments[aircraft_mask].detach().cpu().numpy()
    layouts = info.get("best_layouts", info["terminal_layouts"])[local].detach().cpu().numpy()
    beam_loads = info.get("terminal_beam_loads_bps")
    raw_beam_loads = (beam_loads[local].detach().cpu().numpy()
                      if beam_loads is not None else None)
    if "best_beam_loads_bps" in info:
        beam_loads = info["best_beam_loads_bps"]
    if beam_loads is not None:
        beam_loads = beam_loads[local].detach().cpu().numpy()
    elif start is not None:
        demands = np.asarray(scene.demands, dtype=np.float64)
        beam_loads = np.bincount(assignments[assignments >= 0].astype(int),
                                 weights=demands[assignments >= 0], minlength=100)
    uncovered = [scene.aircraft_ids[i] for i in np.flatnonzero(assignments < 0)]
    qualified_edge_ids = []
    if hasattr(start, "regions"):
        for region in start.regions:
            members = np.asarray(region["aircraft_indices"], dtype=np.int64)
            if members.size:
                beam = int(assignments[members[0]])
                if (beam >= 0 and np.all(assignments[members] == beam) and
                        np.count_nonzero(assignments == beam) == members.size):
                    qualified_edge_ids.append(beam)
    qualified_edge_ids = sorted(set(qualified_edge_ids))
    common_seconds = sum(float(value.get("seconds", 0.0)) for value in
                         start.stages.values() if isinstance(value, dict)) if hasattr(start, "stages") else 0.0
    common_stages = dict(start.stages) if hasattr(start, "stages") else {}
    stage_seconds = {key: float(value.get("seconds", 0.0)) for key, value in
                     common_stages.items() if isinstance(value, dict) and "seconds" in value}
    processing_seconds = elapsed if group != "train" else None
    return {"scene_id": scene.scene_id, "scene_sha256": scene.source_hash,
            "group": group, "terminal_env_step": step, "status": "completed",
            "stop_reason_code": metrics.get("stop_reason_code"),
            "stop_reason": STOP_REASON_LABELS.get(int(metrics.get("stop_reason_code") or 0), "未知"),
            "metrics": metrics, "raw_terminal_metrics": raw_metrics,
            "input": _scene_input(scene, start),
            "initial_a2_sha256": start.to_record()["start_sha256"] if hasattr(start, "to_record") else None,
            "final_positions": layouts, "final_associations": assignments,
            "final_centers": layouts, "associations": assignments,
            "raw_terminal_positions": raw_layouts,
            "raw_terminal_associations": raw_assignments,
            "raw_terminal_beam_loads_bps": raw_beam_loads,
            "beam_loads_bps": beam_loads, "uncovered_aircraft_ids": uncovered,
            "action_trace": trace,
            "balance_diagnostics": {"certified_regions": start.regions if hasattr(start, "regions") else [],
                                    "qualified_edge_beam_ids": qualified_edge_ids,
                                    "common_start_sha256": start.to_record()["start_sha256"] if hasattr(start, "to_record") else None,
                                    "common_stages": common_stages,
                                    "stage_seconds": stage_seconds,
                                    "total_pipeline_seconds": (common_seconds + elapsed if group != "train" else None),
                                    "assignment_attempts": metrics.get("assignment_attempts"),
                                    "raw_terminal_assignment_attempts": raw_metrics.get("assignment_attempts"),
                                    "raw_terminal_candidate_evaluations": raw_metrics.get("candidate_evaluations"),
                                    "stop_reason_code": metrics.get("stop_reason_code")},
            "common_preparation_seconds": common_seconds,
            "evaluation_call_seconds": processing_seconds,
            "elapsed_seconds": common_seconds + elapsed if group != "train" else None,
            "training_cumulative_seconds": elapsed if group == "train" else None,
            "common_start_reused": True,
            "physical_metrics_not_applicable": _physical_not_applicable()}


def _summary(rows):
    metric_rows = [r["metrics"] for r in rows if isinstance(r.get("metrics"), dict)]
    def stop_reason(row):
        stages = row.get("balance_diagnostics", {}).get("stages", {})
        adjustment = stages.get("adjustment", {})
        return row.get("stop_reason", adjustment.get("counts", {}).get(
            "stop_reason", row.get("status")))
    result = {"scene_count": len(rows), "valid_metric_scene_count": len(metric_rows),
              "failed_scene_count": sum(row.get("status") != "completed" for row in rows),
              "stop_reasons": dict(Counter(str(stop_reason(r)) for r in rows))}
    keys = set.intersection(*(set(row) for row in metric_rows)) if metric_rows else set()
    for key in keys:
        values = [row[key] for row in metric_rows]
        if all(isinstance(v, (int, float)) and np.isfinite(v) for v in values):
            result[key] = float(np.mean(values))
    requests = [float(np.asarray(row["input"]["demands_bps"], dtype=np.float64).sum())
                for row in rows]
    users = [len(row["input"]["demands_bps"]) for row in rows]
    demand_fraction = [float(row["metrics"].get("coverage_demand") or 0.0)
                       if isinstance(row.get("metrics"), dict) else 0.0 for row in rows]
    user_fraction = [float(row["metrics"].get("coverage_users") or 0.0)
                     if isinstance(row.get("metrics"), dict) else 0.0 for row in rows]
    if rows:
        result["coverage_demand"] = float(np.mean(demand_fraction))
        result["coverage_users"] = float(np.mean(user_fraction))
        result["pooled_demand_coverage"] = (float(np.dot(requests, demand_fraction) / sum(requests))
                                             if sum(requests) > 0 else 0.0)
        result["requested_demand_bps"] = float(sum(requests))
        total_uncovered_demand = float(sum(r * (1 - c) for r, c in
                                           zip(requests, demand_fraction)))
        total_uncovered_users = float(sum(n * (1 - c) for n, c in
                                         zip(users, user_fraction)))
        result["total_uncovered_demand_bps"] = total_uncovered_demand
        result["total_uncovered_users"] = total_uncovered_users
        result["uncovered_demand_bps"] = total_uncovered_demand / len(rows)
        result["uncovered_users"] = total_uncovered_users / len(rows)
        result["coverage_100_scene_count"] = sum(bool(row.get("metrics", {}).get("coverage_100"))
                                                  for row in rows if isinstance(row.get("metrics"), dict))
        result["coverage_99_9_scene_count"] = sum(bool(row.get("metrics", {}).get("coverage_99_9"))
                                                   for row in rows if isinstance(row.get("metrics"), dict))
    return result


def _prepare(directory, job, config, resume):
    from .balancing import prepare_common_a2

    role = next(iter(config["datasets"]))
    dataset = _dataset(config, role)
    path = directory / "starts.jsonl"
    partial = directory / "starts.partial.jsonl"
    prior = _partial_rows(partial, resume)
    _verify_partial_scenes(prior, dataset, wrapper=True)
    rows = [row["record"] for row in prior]
    for index in range(len(rows), len(dataset.scenes)):
        scene = dataset.scenes[index]
        if _stop_requested(directory):
            _status(directory, job, stage="stopped", evaluated_scenes=index)
            return
        start = prepare_common_a2(scene, config["coverage"], device="cuda")
        record = start.to_record() if hasattr(start, "to_record") else start
        rows.append(record)
        _jsonl(partial, {"scene_index": index, "record": record})
        _status(directory, job, stage="preparing", evaluated_scenes=index + 1,
                evaluation_total=len(dataset))
    if _stop_requested(directory):
        _status(directory, job, stage="stopped", evaluated_scenes=len(rows))
        return
    path.unlink(missing_ok=True)
    _jsonl_many(path, rows)
    sha = _sha256(path)
    _write_json(directory / "artifacts.json", {"run_id": job["run_id"],
                "balance_starts": str(path), "balance_starts_sha256": sha,
                "config": str(directory / "config.json"), "source_root": job["source_root"],
                "datasets": config["datasets"]})
    _write_json(directory / "results.json", {"summary": {"scene_count": len(rows),
                "prepared_start_count": len(rows)}, "start_sha256": sha,
                "source_run_id": job["run_id"]})
    _status(directory, job, stage="completed", evaluated_scenes=len(rows),
            evaluation_total=len(rows), balance_starts_sha256=sha)
    _event(directory, job, "balance_starts_prepared", count=len(rows), sha256=sha)


def _baseline(directory, job, config, resume):
    from .balancing import run_balance_route

    role = next(iter(config["datasets"]))
    dataset = _dataset(config, role)
    route = config["coverage"]["balance"]["route"]
    starts = _starts(config, role, dataset)
    partial = directory / "results.partial.jsonl"
    rows = _partial_rows(partial, resume)
    _verify_partial_scenes(rows, dataset)
    for index in range(len(rows), len(dataset.scenes)):
        scene, start = dataset.scenes[index], starts[index]
        if _stop_requested(directory):
            _status(directory, job, stage="stopped", evaluated_scenes=index)
            return
        began = time.monotonic()
        result = run_balance_route(scene, config["coverage"], route, common=start, device="cuda")
        row = {"scene_index": index, "scene_sha256": scene.source_hash,
               "input": _scene_input(scene), **result}
        row.setdefault("elapsed_seconds", time.monotonic() - began)
        row["physical_metrics_not_applicable"] = _physical_not_applicable()
        rows.append(row)
        _jsonl(partial, row)
        _jsonl(directory / "evaluations.jsonl", {"group": role, "scene_index": index,
               "scene_id": scene.scene_id, "status": row.get("status"),
               "metrics": row.get("metrics"), "elapsed_seconds": row["elapsed_seconds"]})
        _status(directory, job, stage="evaluating", evaluated_scenes=index + 1,
                evaluation_total=len(dataset))
    if _stop_requested(directory):
        _status(directory, job, stage="stopped", evaluated_scenes=len(rows))
        return
    _save_results(directory, job, config, rows, "baseline")


def _evaluate_policy(config, policy, dataset, starts, *, directory=None, job=None,
                     role="validation", step=0, scheduled_step=None,
                     prior_rows=None, partial_path=None, publish_rows=True):
    from .balancing_env import BalanceBatchEnv
    from .scenes import CoverageDataset

    rows = list(prior_rows or [])
    prior = policy.training
    policy.eval()
    try:
        for index in range(len(rows), len(dataset.scenes)):
            scene = dataset.scenes[index]
            if directory is not None and _stop_requested(directory):
                break
            if isinstance(starts[index], dict) or starts[index].status != "completed":
                failure = _unready_start_row(scene, index, starts[index], config)
                rows.append(failure)
                if partial_path is not None:
                    _jsonl(partial_path, failure)
                if directory is not None and publish_rows:
                    _jsonl(directory / "evaluations.jsonl", {"step": step,
                           "scheduled_step": scheduled_step,
                           "group": role, **failure})
                if directory is not None:
                    _status(directory, job, stage="evaluating" if role == "test" else "validating",
                            evaluated_scenes=len(rows), evaluation_total=len(dataset))
                continue
            began = time.monotonic()
            env = BalanceBatchEnv(CoverageDataset([scene], dataset.metadata), config["coverage"],
                                  [starts[index]],
                                  num_envs=1, device="cuda", seed=config["training"]["seed"] + 100003,
                                  auto_reset=False)
            obs, mask = env.reset(scene_indices=[0])
            candidates = env.candidate_features()
            setup_finished = time.monotonic()
            trace = []
            terminal_info = None
            common_refinement_seconds = sum(float(starts[index].stages.get(part, {}).get("seconds", 0.0))
                                            for part in ("repair", "assignment"))
            refinement_budget_seconds = float(config["coverage"]["balance"]["max_refinement_seconds"])
            remaining_seconds = max(0.0, refinement_budget_seconds - common_refinement_seconds)
            if bool(env.initial_target[0]):
                terminal_info = env.terminal_snapshot("already_target")
            for decision in range(env.horizon):
                if terminal_info is not None:
                    break
                if time.monotonic() - began >= remaining_seconds:
                    terminal_info = env.terminal_snapshot("time_limit")
                    break
                with torch.no_grad():
                    action, _, _ = policy.act(obs, mask, candidates,
                                              deterministic=config["evaluation"]["action_mode"] == "deterministic")
                next_obs, _, done, info = env.step(action)
                accepted = bool(info["action_accepted"][0])
                reason_code = int(info["rejection_reason_code"][0])
                trace.append({"decision": decision + 1, "requested_action": int(action[0]),
                              "actual_action": int(info.get("actual_action", action)[0]),
                              "action_accepted": accepted,
                              "post_move_transfer": bool(info["post_move_transfer"][0]),
                              "post_move_attempts": int(info["post_move_attempts"][0]),
                              "center_candidate_checks": int(info["center_candidate_checks"][0]),
                              "rejection_reason_code": reason_code,
                              "action_outcome": REJECTION_REASON_LABELS[reason_code],
                              "beam_processes": int(info["episode_metrics"]["beam_processes"][0]),
                              "assignment_attempts": int(info["episode_metrics"]["assignment_attempts"][0]),
                              "candidate_evaluations": int(info["episode_metrics"]["candidate_evaluations"][0]),
                              "coverage_demand": _optional_float(info["episode_metrics"]["coverage_demand"][0]),
                              "coverage_users": _optional_float(info["episode_metrics"]["coverage_users"][0]),
                              "subject_cv": _optional_float(info["episode_metrics"]["subject_cv"][0])})
                if bool(info["finished_rows"][0]):
                    terminal_info = info
                    break
                obs = next_obs
                mask = info.get("mask", env.action_mask())
                candidates = env.candidate_features()
            if terminal_info is None and time.monotonic() - began >= remaining_seconds:
                terminal_info = env.terminal_snapshot("time_limit")
            if terminal_info is None:
                raise RuntimeError(f"{scene.scene_id} did not finish within horizon")
            verification_started = time.monotonic()
            terminal = _terminal_row(scene, 0, terminal_info, trace, step=step,
                                     group=role, elapsed=verification_started - began,
                                     start=starts[index])
            torch.cuda.synchronize()
            finished_at = time.monotonic()
            evaluation_seconds = finished_at - began
            terminal["evaluation_call_seconds"] = evaluation_seconds
            terminal["elapsed_seconds"] = terminal["common_preparation_seconds"] + evaluation_seconds
            terminal["balance_diagnostics"]["evaluation_stage_seconds"] = {
                "environment_setup": setup_finished - began,
                "policy_and_adjustment": verification_started - setup_finished,
                "final_verification_and_record": finished_at - verification_started}
            terminal["balance_diagnostics"]["stage_seconds"].update(
                terminal["balance_diagnostics"]["evaluation_stage_seconds"])
            terminal["balance_diagnostics"]["total_pipeline_seconds"] = terminal["elapsed_seconds"]
            terminal["balance_diagnostics"]["refinement_budget_seconds"] = refinement_budget_seconds
            terminal["balance_diagnostics"]["common_refinement_seconds"] = common_refinement_seconds
            terminal["scene_index"] = index
            rows.append(terminal)
            if partial_path is not None:
                _jsonl(partial_path, terminal)
            if directory is not None and publish_rows:
                _jsonl(directory / "evaluations.jsonl", {"step": step,
                       "scheduled_step": scheduled_step,
                       "group": role, **terminal})
            if directory is not None:
                _status(directory, job, stage="evaluating" if role == "test" else "validating",
                        evaluated_scenes=len(rows), evaluation_total=len(dataset))
    finally:
        policy.train(prior)
    return rows, _summary(rows)


def _save_results(directory, job, config, rows, execution):
    path = directory / "results.json"
    summary = _summary(rows)
    _write_json(path, {"summary": summary, "scenes": rows,
              "metadata": {"schema_version": 3, "research_stage": "coverage",
                           "route": config["coverage"]["balance"]["route"],
                           "action_menu": config["coverage"]["balance"]["action_menu"],
                           "git_commit": job["git_commit"], "config": config,
                           "execution": execution, "metric_scope": "all input scenes; physical SI/SGM not applicable"}})
    _write_json(directory / "artifacts.json", {"run_id": job["run_id"],
                "config": str(directory / "config.json"), "results": str(path),
                "source_root": job["source_root"], "datasets": config["datasets"],
                "balance_starts": config.get("balance_starts", {})})
    _status(directory, job, stage="completed", evaluated_scenes=len(rows),
            evaluation_total=len(rows), metrics=summary, results=str(path))
    _event(directory, job, "balance_evaluation_completed", scene_count=len(rows))


def _sync_validation_records(directory, config, dataset):
    """Rebuild complete-node rows and fill missing summaries after an interruption."""
    nodes = []
    for path in sorted((directory / "validation_nodes").glob("*.json"),
                       key=lambda value: int(value.stem)):
        node = _read(path)
        if (not isinstance(node, dict) or len(node.get("rows", [])) != len(dataset) or
                node.get("scene_count") != len(dataset) or
                node.get("config_hash") != config_hash(config)):
            raise ValueError("Incomplete frozen validation node")
        _verify_partial_scenes(node["rows"], dataset)
        nodes.append(node)
    lines = []
    for node in nodes:
        for row in node["rows"]:
            lines.append(json.dumps(_jsonable({"step": node["actual_env_step"],
                       "scheduled_step": node["scheduled_step"],
                       "group": "validation", **row}), ensure_ascii=False,
                       allow_nan=False) + "\n")
    path = directory / "evaluations.jsonl"
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text("".join(lines), encoding="utf-8")
    os.replace(temp, path)
    logged = set()
    metrics_path = directory / "metrics.jsonl"
    if metrics_path.is_file():
        valid_lines = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                break  # Trim a torn final write before appending a node summary.
            valid_lines.append(line + "\n")
            if record.get("group") == "validation":
                logged.add(record.get("scheduled_step"))
        metrics_temp = metrics_path.with_name(metrics_path.name + f".tmp.{os.getpid()}")
        metrics_temp.write_text("".join(valid_lines), encoding="utf-8")
        os.replace(metrics_temp, metrics_path)
    for node in nodes:
        if node["scheduled_step"] not in logged:
            _jsonl(metrics_path, {"step": node["actual_env_step"],
                    "scheduled_step": node["scheduled_step"],
                    "group": "validation", "values": node["summary"],
                    "wall_time": node["completed_at"]})


def _model_metadata(job, config, policy, step, eligibility):
    return {"format_version": 3, "policy_format": "coverage_balance_candidate_v3",
            "research_stage": "coverage", "algorithm": "ppo",
            "obs_dim": policy.obs_dim, "action_dim": policy.action_dim,
            "candidate_dim": policy.candidate_dim,
            "hidden_sizes": list(policy.hidden_sizes),
            "training_seed": config["training"]["seed"], "env_step": step,
            "selection": "fixed_budget_endpoint", "git_commit": job["git_commit"],
            "source_root": job["source_root"], "config_hash": config_hash(config),
            "coverage": config["coverage"], "datasets": config["datasets"],
            "balance_starts": config["balance_starts"],
            "training_eligibility_sha256": eligibility["eligibility_sha256"],
            "training_eligibility_file_sha256": eligibility["file_sha256"],
            "eligible_original_indices": eligibility["eligible_original_indices"],
            "source_scene_count": eligibility["source_scene_count"],
            "eligible_scene_count": eligibility["eligible_scene_count"],
            "created_at": _now()}


def _checkpoint(directory, job, config, policy, optimizer, env, obs, mask, candidates,
                *, step, update, episode, returns, traces, elapsed,
                last_completed_validation_threshold, eligibility_sha256=None):
    path = directory / "checkpoint.pt"
    _save_torch(path, {"format_version": 3, "policy_format": "coverage_balance_candidate_v3",
                "config_hash": config_hash(config), "source_commit": job["git_commit"],
                "source_root": job["source_root"],
                "dataset_hashes": {r: x["sha256"] for r, x in config["datasets"].items()},
                "start_hashes": {r: x["sha256"] for r, x in config["balance_starts"].items()},
                "training_eligibility_sha256": eligibility_sha256,
                "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "env": env.state_dict(), "obs": obs, "mask": mask,
                "candidate_features": candidates, "step": step, "update": update,
                "episode": episode, "episode_returns": returns,
                "episode_traces": traces, "elapsed_seconds": elapsed,
                "last_completed_validation_threshold": last_completed_validation_threshold,
                "rng": _rng_state()})
    return _sha256(path)


def _restore(directory, job, config, policy, optimizer, env, eligibility_sha256=None):
    payload = _load_torch(directory / "checkpoint.pt", "cuda")
    expected = {"format_version": 3, "policy_format": "coverage_balance_candidate_v3",
                "config_hash": config_hash(config), "source_commit": job["git_commit"],
                "source_root": job["source_root"],
                "dataset_hashes": {r: x["sha256"] for r, x in config["datasets"].items()},
                "start_hashes": {r: x["sha256"] for r, x in config["balance_starts"].items()}}
    expected["training_eligibility_sha256"] = eligibility_sha256
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("S1-02 checkpoint source, model, data or A2 starts differ")
    policy.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    env.load_state_dict(payload["env"])
    if not torch.equal(payload["candidate_features"], env.candidate_features()):
        raise ValueError("S1-02 checkpoint candidates differ from restored environment")
    _restore_rng(payload["rng"])
    return payload


def _train(directory, job, config, resume, preflight):
    from .balancing_env import BalanceBatchEnv

    t = config["training"]
    train_data = _dataset(config, "train")
    val_data = _dataset(config, "validation")
    starts = _starts(config, "train", train_data)
    val_starts = _starts(config, "validation", val_data)
    eligibility = _training_eligibility(directory, train_data, starts, resume=resume)
    eligibility["file_sha256"] = _sha256(directory / "training_eligibility.json")
    eligibility_sha = eligibility["eligibility_sha256"]
    _seed(t["seed"])
    env = BalanceBatchEnv(train_data, config["coverage"], starts, num_envs=t["num_envs"],
                          device="cuda", seed=t["seed"],
                          eligible_indices=eligibility["eligible_original_indices"])
    policy = _policy(config, env)
    optimizer = torch.optim.Adam(policy.parameters(), lr=t["lr"])
    if resume:
        payload = _restore(directory, job, config, policy, optimizer, env,
                           eligibility_sha256=eligibility_sha)
        obs, mask, candidates = (payload[k] for k in ("obs", "mask", "candidate_features"))
        step, update, episode = (payload[k] for k in ("step", "update", "episode"))
        returns, traces = payload["episode_returns"], payload["episode_traces"]
        elapsed_base = payload["elapsed_seconds"]
        last_completed_validation_threshold = payload["last_completed_validation_threshold"]
    else:
        obs, mask = env.reset()
        candidates = env.candidate_features()
        step = update = episode = 0
        returns = torch.zeros(t["num_envs"], device="cuda")
        traces = [[] for _ in range(t["num_envs"])]
        elapsed_base, last_completed_validation_threshold = 0.0, 0
    began = time.monotonic()
    _status(directory, job, stage="training", env_step=step, update_step=update,
            budget_env_steps=t["total_steps"], preflight=preflight,
            last_completed_validation_threshold=last_completed_validation_threshold,
            source_training_scene_count=eligibility["source_scene_count"],
            eligible_training_scene_count=eligibility["eligible_scene_count"],
            training_eligibility_sha256=eligibility_sha,
            training_eligibility_file_sha256=eligibility["file_sha256"])
    _sync_validation_records(directory, config, val_data)
    while True:
        if _stop_requested(directory):
            sha = _checkpoint(directory, job, config, policy, optimizer, env, obs, mask,
                              candidates, step=step, update=update, episode=episode,
                              returns=returns, traces=traces,
                              elapsed=elapsed_base + time.monotonic() - began,
                              last_completed_validation_threshold=last_completed_validation_threshold,
                              eligibility_sha256=eligibility_sha)
            _status(directory, job, stage="stopped", checkpoint_sha256=sha,
                    env_step=step, update_step=update, episode=episode)
            return
        scheduled_step = _due_validation_threshold(
            step, last_completed_validation_threshold, t["validation_interval"])
        if scheduled_step is not None:
            node_path = directory / "validation_nodes" / f"{scheduled_step}.json"
            node_path.parent.mkdir(parents=True, exist_ok=True)
            node = _read(node_path)
            if node is None:
                rng_before_validation = _rng_state()
                try:
                    validation_rows, summary = _evaluate_policy(
                        config, policy, val_data, val_starts,
                        directory=directory, job=job,
                        role="validation", step=step,
                        scheduled_step=scheduled_step, publish_rows=False)
                finally:
                    _restore_rng(rng_before_validation)
                if len(validation_rows) != len(val_data):
                    sha = _checkpoint(directory, job, config, policy, optimizer, env, obs, mask,
                                      candidates, step=step, update=update, episode=episode,
                                      returns=returns, traces=traces,
                                      elapsed=elapsed_base + time.monotonic() - began,
                                      last_completed_validation_threshold=last_completed_validation_threshold,
                                      eligibility_sha256=eligibility_sha)
                    _status(directory, job, stage="stopped", env_step=step,
                            update_step=update, episode=episode,
                            checkpoint_step=step, checkpoint_sha256=sha)
                    return
                node = {"scheduled_step": scheduled_step, "actual_env_step": step,
                        "scene_count": len(val_data), "config_hash": config_hash(config),
                        "rows": validation_rows, "summary": summary,
                        "completed_at": _now()}
                _write_json(node_path, node)
            if (node.get("scheduled_step") != scheduled_step or
                    node.get("actual_env_step") != step or
                    node.get("scene_count") != len(val_data) or
                    node.get("config_hash") != config_hash(config)):
                raise ValueError("Frozen validation node differs from checkpoint and source")
            last_completed_validation_threshold = scheduled_step
            _sync_validation_records(directory, config, val_data)
            sha = _checkpoint(directory, job, config, policy, optimizer, env, obs, mask,
                              candidates, step=step, update=update, episode=episode,
                              returns=returns, traces=traces,
                              elapsed=elapsed_base + time.monotonic() - began,
                              last_completed_validation_threshold=last_completed_validation_threshold,
                              eligibility_sha256=eligibility_sha)
            _status(directory, job, stage="training", env_step=step,
                    update_step=update, episode=episode, checkpoint_step=step,
                    checkpoint_sha256=sha,
                    last_completed_validation_threshold=scheduled_step,
                    evaluated_scenes=len(val_data), evaluation_total=len(val_data))
        if step >= t["total_steps"]:
            break
        length = min(t["rollout_steps"], (t["total_steps"] - step) // t["num_envs"])
        batches = [[] for _ in range(8)]
        completed = []
        for offset in range(length):
            with torch.no_grad():
                action, logp, value = policy.act(obs, mask, candidates)
            next_obs, raw_reward, done, info = env.step(action)
            reward = raw_reward * t["reward_scale"]
            for bucket, item in zip(batches, (obs, mask, candidates, action,
                                              logp, value, reward, done)):
                bucket.append(item)
            returns += reward
            requested = info.get("requested_action", action).detach().cpu().tolist()
            actual = info.get("actual_action", action).detach().cpu().tolist()
            finished = info["finished_rows"].detach().cpu().tolist()
            telemetry_keys = ("coverage_demand", "coverage_users", "subject_cv",
                              "load_cv", "beam_processes", "assignment_attempts",
                              "candidate_evaluations", "stop_reason_code")
            telemetry = torch.stack(tuple((info["episode_metrics"].get(key,
                                  info.get(key))).to(torch.float64) for key in telemetry_keys),
                                    dim=1).detach().cpu().numpy()
            accepted = info["action_accepted"].detach().cpu().tolist()
            transferred = info["post_move_transfer"].detach().cpu().tolist()
            post_move_attempts = info["post_move_attempts"].detach().cpu().tolist()
            center_candidate_checks = info["center_candidate_checks"].detach().cpu().tolist()
            reason_codes = info["rejection_reason_code"].detach().cpu().tolist()
            for local in range(t["num_envs"]):
                event = {"env_step": step + offset * t["num_envs"] + local + 1,
                         "requested_action": int(requested[local]),
                         "actual_action": int(actual[local]),
                         "action_accepted": bool(accepted[local]),
                         "post_move_transfer": bool(transferred[local]),
                         "post_move_attempts": int(post_move_attempts[local]),
                         "center_candidate_checks": int(center_candidate_checks[local]),
                         "rejection_reason_code": int(reason_codes[local]),
                         "action_outcome": REJECTION_REASON_LABELS[int(reason_codes[local])],
                         **{key: (float(telemetry[local, j]) if
                                  np.isfinite(telemetry[local, j]) else None)
                            for j, key in enumerate(telemetry_keys)}}
                traces[local].append(event)
                if finished[local]:
                    scene_index = int(info["episode_metrics"]["scene_index"][local].item())
                    scene = train_data.scenes[scene_index]
                    row = _terminal_row(scene, local, info, traces[local],
                                        step=event["env_step"], group="train",
                                        elapsed=elapsed_base + time.monotonic() - began,
                                        start=starts[scene_index])
                    row.update(scene_index=scene_index,
                               episode_return=float(returns[local].item()),
                               training_seed=t["seed"])
                    completed.append(row)
                    traces[local] = []
                    returns[local] = 0
            obs = next_obs
            mask = info.get("mask", env.action_mask())
            candidates = env.candidate_features()
        with torch.no_grad():
            bootstrap = policy.value(obs)
        rollout = Rollout(*(torch.stack(items) for items in batches), bootstrap)
        diagnostics = ppo_update(policy, optimizer, rollout, gamma=t["gamma"],
            gae_lambda=t["gae_lambda"], clip_ratio=t["clip_ratio"],
            entropy_coef=t["entropy_coef"], value_coef=t["value_coef"],
            max_grad_norm=t["max_grad_norm"], epochs=t["epochs"],
            minibatch_size=t["minibatch_size"])
        prior_update = update
        step += length * t["num_envs"]
        minibatch_updates = diagnostics.pop("updates")
        update += minibatch_updates
        first_episode = episode
        episode += len(completed)
        for offset, row in enumerate(completed, start=1):
            row["episode"] = first_episode + offset
            row["collected_update_step"] = prior_update
        _jsonl_many(directory / "episodes.jsonl", completed)
        _jsonl(directory / "updates.jsonl", {"env_step": step, "update_step": update,
               "rollout_interactions": length * t["num_envs"],
               "minibatch_updates": minibatch_updates,
               "episode": episode, "training_seed": t["seed"], **diagnostics})
        _jsonl(directory / "metrics.jsonl", {"step": step, "group": "ppo_update",
               "values": {**diagnostics, "minibatch_updates": minibatch_updates},
               "wall_time": _now()})
        if completed:
            _jsonl(directory / "metrics.jsonl", {"step": step, "group": "terminal",
                   "values": _summary(completed), "wall_time": _now()})
        elapsed = elapsed_base + time.monotonic() - began
        if (step % t["checkpoint_interval"] < length * t["num_envs"] or
                step == t["total_steps"] or _stop_requested(directory)):
            sha = _checkpoint(directory, job, config, policy, optimizer, env, obs, mask,
                              candidates, step=step, update=update, episode=episode,
                              returns=returns, traces=traces, elapsed=elapsed,
                              last_completed_validation_threshold=last_completed_validation_threshold,
                              eligibility_sha256=eligibility_sha)
            checkpoint_fields = {"checkpoint_sha256": sha, "checkpoint_step": step}
        else:
            checkpoint_fields = {}
        _status(directory, job, stage="training", env_step=step, update_step=update,
                episode=episode, budget_env_steps=t["total_steps"],
                wall_seconds=elapsed, latest_update=diagnostics,
                last_completed_validation_threshold=last_completed_validation_threshold,
                **checkpoint_fields)
    if _stop_requested(directory):
        sha = _checkpoint(directory, job, config, policy, optimizer, env, obs, mask,
                          candidates, step=step, update=update, episode=episode,
                          returns=returns, traces=traces,
                          elapsed=elapsed_base + time.monotonic() - began,
                          last_completed_validation_threshold=last_completed_validation_threshold,
                          eligibility_sha256=eligibility_sha)
        _status(directory, job, stage="stopped", checkpoint_sha256=sha,
                env_step=step, update_step=update, episode=episode)
        return
    final = Path(job["model_dir"]) / "final"
    final.mkdir(parents=True, exist_ok=True)
    _save_torch(final / "policy.pt", policy.state_dict())
    _write_json(final / "model.json", _model_metadata(job, config, policy, step,
                                                      eligibility))
    _write_json(directory / "artifacts.json", {"run_id": job["run_id"],
                "config": str(directory / "config.json"),
                "checkpoint": str(directory / "checkpoint.pt"),
                "final_policy": str(final), "model": str(final / "policy.pt"),
                "model_metadata": str(final / "model.json"),
                "datasets": config["datasets"], "balance_starts": config["balance_starts"],
                "training_eligibility": str(directory / "training_eligibility.json"),
                "training_eligibility_sha256": eligibility_sha,
                "training_eligibility_file_sha256": eligibility["file_sha256"]})
    _status(directory, job, stage="completed", env_step=step, update_step=update,
            episode=episode, final_policy=str(final))
    _event(directory, job, "balance_training_completed", env_step=step)


def _evaluation(directory, job, config, resume):
    from .balancing_env import BalanceBatchEnv
    from .scenes import CoverageDataset

    role = next(iter(config["datasets"]))
    dataset = _dataset(config, role)
    starts = _starts(config, role, dataset)
    partial = directory / "results.partial.jsonl"
    prior_rows = _partial_rows(partial, resume)
    _verify_partial_scenes(prior_rows, dataset)
    first_valid = next((i for i, start in enumerate(starts)
                        if not isinstance(start, dict) and start.status == "completed"), None)
    if first_valid is None:
        rows = [_unready_start_row(scene, i, starts[i], config)
                for i, scene in enumerate(dataset.scenes)]
        _save_results(directory, job, config, rows, "serial_policy")
        return
    env = BalanceBatchEnv(CoverageDataset([dataset.scenes[first_valid]], dataset.metadata),
                          config["coverage"], [starts[first_valid]], num_envs=1,
                          device="cuda", seed=config["training"]["seed"], auto_reset=False)
    policy = _policy(config, env)
    metadata = _read(Path(job["model_path"]) / "model.json", {})
    if (metadata.get("format_version") != 3 or
            metadata.get("policy_format") != "coverage_balance_candidate_v3" or
            metadata.get("coverage") != config["coverage"] or
            metadata.get("action_dim") != env.action_dim or
            metadata.get("obs_dim") != env.obs_dim or
            metadata.get("candidate_dim") != env.candidate_dim):
        raise ValueError("S1-02 model/config/action menu dimensions differ")
    policy.load_state_dict(_load_torch(Path(job["model_path"]) / "policy.pt", "cuda"))
    rows, _ = _evaluate_policy(config, policy, dataset, starts, directory=directory,
                                job=job, role=role, step=metadata["env_step"],
                                prior_rows=prior_rows, partial_path=partial)
    if _stop_requested(directory):
        _status(directory, job, stage="stopped", evaluated_scenes=len(rows))
        return
    _save_results(directory, job, config, rows, "serial_policy")


def run_balance_job(directory, job, config, *, resume, preflight):
    directory = Path(directory)
    route = config["coverage"]["balance"]["route"]
    _status(directory, job, stage="initializing", preflight=preflight)
    _event(directory, job, "balance_preflight_verified", preflight=preflight)
    if job["kind"] == "train":
        if route != "ppo":
            raise ValueError("S1-02 training requires PPO route")
        _train(directory, job, config, resume, preflight)
    elif job["kind"] == "evaluate":
        if route == "prepare":
            _prepare(directory, job, config, resume)
        elif route == "ppo":
            _evaluation(directory, job, config, resume)
        else:
            _baseline(directory, job, config, resume)
    else:
        raise ValueError("Unknown S1-02 workbench task kind")
