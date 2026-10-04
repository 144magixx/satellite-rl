"""Frozen workbench jobs for S1-03 region reconstruction.

Preparation, training, validation and the five common-geometry comparisons
share the same reconstruction core. Only the workbench dispatcher calls this
module for formal jobs; it never starts a process of its own.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields
from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import torch

from .config import config_hash
from .reconstruction_ppo import (OPERATIONS, GraphRollout, RelationPolicy,
                                  action_identity, graph_observation, identity_digest,
                                  identity_table, ppo_update,
                                  stack_observations)
from .runner import (_dataset, _event, _jsonl, _jsonl_many, _load_torch, _now,
                     _read, _restore_rng, _rng_state, _save_torch, _seed,
                     _sha256, _status, _stop_requested, _write_json)


def _core_config(config):
    from .reconstruction import ReconstructionConfig
    supplied = config["coverage"]["reconstruction"]
    names = {item.name for item in fields(ReconstructionConfig)}
    return ReconstructionConfig(**{name: supplied[name] for name in names if name in supplied})


def _write_artifacts(directory, payload):
    path = directory / "artifacts.json"
    existing = _read(path, {})
    if not isinstance(existing, dict):
        raise ValueError("Existing artifact manifest is malformed")
    _write_json(path, {**existing, **payload})


def _start_record(scene, state, seconds):
    return {"scene_id": scene.scene_id, "scene_sha256": scene.source_hash,
            "initialization_valid": bool(state.initialization_valid),
            "initialization_seconds": seconds,
            "initialization_repair_geometry_calls": int(
                state.initialization_repair_geometry_calls),
            "initialization_repair_ray_pair_checks": int(
                state.initialization_repair_ray_pair_checks),
            "initialization_repair_attempts": int(state.initialization_repair_attempts),
            "state": state.to_record()}


def _finite_json(value):
    """Strict JSON for explicitly undefined statistics on failed starts."""
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _load_starts(config, role, dataset, core_config):
    from .reconstruction import ReconstructionState
    record = config["reconstruction_starts"][role]
    source = Path(record["path"])
    if not source.is_file() or _sha256(source) != record["sha256"]:
        raise ValueError(f"{role} reconstruction starts missing or SHA mismatch")
    if record["dataset_sha256"] != config["datasets"][role]["sha256"]:
        raise ValueError(f"{role} reconstruction starts refer to another dataset")
    lines = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != len(dataset):
        raise ValueError(f"{role} reconstruction start count differs from dataset")
    states = []
    for scene, line in zip(dataset.scenes, lines):
        if (line.get("scene_id") != scene.scene_id or
                line.get("scene_sha256") != scene.source_hash):
            raise ValueError(f"{role} reconstruction start scene provenance differs")
        if line.get("state") is None:
            if line.get("initialization_valid") or not line.get("initialization_error"):
                raise ValueError(f"{role} missing start lacks explicit failure reason")
            state = None
        else:
            state = ReconstructionState.from_record(scene, line["state"], core_config)
            if bool(state.initialization_valid) != bool(line["initialization_valid"]):
                raise ValueError(f"{role} reconstruction start validity differs")
            for field_name in ("initialization_repair_geometry_calls",
                               "initialization_repair_ray_pair_checks",
                               "initialization_repair_attempts"):
                if int(line.get(field_name, -1)) != int(getattr(state, field_name)):
                    raise ValueError(f"{role} reconstruction start accounting differs: {field_name}")
        states.append(state)
    return states, lines


def _row(scene, state, metrics, *, route, actions, initialization_seconds,
         policy_seconds, reconstruction_seconds, final_check_seconds,
         observation_ray_pairs=0, observation_seconds=0.0,
         initial_metrics=None):
    user_rows = [{"aircraft_id": identifier, "demand_bps": float(demand),
                  "assigned_beam": int(owner) if owner >= 0 else None}
                 for identifier, demand, owner in zip(
                     scene.aircraft_ids, scene.demands, state.assignment)]
    return {"scene_id": scene.scene_id, "scene_sha256": scene.source_hash,
            "route": route, "initialization_valid": bool(state.initialization_valid),
            "metrics": _finite_json(metrics),
            "initial_metrics": _finite_json(initial_metrics or {}),
            "stop_reason": (actions[-1]["details"].get("terminal_reason")
                            if actions else "initial_state"),
            "macro_steps": int(state.steps), "geometry_calls": int(state.geometry_calls),
            "attempts": int(state.attempts), "centers_latlon_deg": state.centers,
            "beam_load_demand_bps": state.loads, "users": user_rows,
            "best_quality_diagnostic": _finite_json(state.best_quality),
            "best_record_diagnostic": _finite_json(state.best_record),
            "fit_candidate_checks": int(state.fit_candidate_checks),
            "assignment_candidate_checks": int(state.assignment_candidate_checks),
            "ray_pair_checks": int(state.ray_pair_checks),
            "candidate_builds": int(state.candidate_builds),
            "candidate_screen_ray_pairs": int(state.candidate_screen_ray_pairs),
            "candidate_screen_seconds": float(state.elapsed_candidate_screen_s),
            "observation_ray_pairs": int(observation_ray_pairs),
            "observation_seconds": float(observation_seconds),
            "initialization_repair_geometry_calls": int(
                state.initialization_repair_geometry_calls),
            "initialization_repair_ray_pair_checks": int(
                state.initialization_repair_ray_pair_checks),
            "initialization_repair_attempts": int(state.initialization_repair_attempts),
            "actions": actions, "initialization_seconds": initialization_seconds,
            "policy_seconds": policy_seconds,
            "reconstruction_seconds": reconstruction_seconds,
            "final_check_seconds": final_check_seconds,
            "total_seconds": initialization_seconds + policy_seconds +
                             reconstruction_seconds + final_check_seconds}


def _candidate_digest(candidates):
    return identity_digest(identity_table(candidates))


def _actual_action(action, info):
    if action.operation == "stop" or info.get("accepted"):
        return {"operation": action.operation, "beams": list(action.beams),
                "template": int(action.template)}
    return {"operation": "no_change", "beams": [], "template": None,
            "reason": info.get("reason")}


def _policy(config, device):
    network = config["network"]
    if (network["kind"] != "reconstruction_relation_ppo" or
            network["hidden_size"] != 128 or network["message_layers"] != 2 or
            float(network.get("dropout", 0)) != 0):
        raise ValueError("S1-03 requires two 128-wide relation layers without dropout")
    return RelationPolicy().to(device)


def _metrics(state, core_config):
    from .reconstruction import state_metrics
    return state_metrics(state, core_config)


def _training_eligibility(directory, config, dataset, starts, start_records,
                          core_config, *, resume):
    rows = []
    for index, (scene, state, record) in enumerate(zip(dataset.scenes, starts, start_records)):
        rows.append({"scene_index": index, "scene_id": scene.scene_id,
                     "scene_sha256": scene.source_hash,
                     "eligible": bool(state is not None and state.initialization_valid),
                     "initialization_error": record.get("initialization_error"),
                     "initial_metrics": (_finite_json(_metrics(state, core_config))
                                         if state is not None else None)})
    payload = {"dataset_sha256": config["datasets"]["train"]["sha256"],
               "start_sha256": config["reconstruction_starts"]["train"]["sha256"],
               "scene_count": len(rows), "eligible_scene_count": sum(row["eligible"] for row in rows),
               "rows": rows}
    path = directory / "training_eligibility.json"
    if resume:
        if _read(path) != payload:
            raise ValueError("Frozen training eligibility ledger differs on resume")
    else:
        if path.exists():
            raise FileExistsError("Existing eligibility ledger requires same-ID resume")
        _write_json(path, payload)
    return payload, path


def _prepare(directory, job, config, resume):
    from .reconstruction import initial_state
    role = next(iter(config["datasets"]))
    dataset = _dataset(config, role)
    core_config = _core_config(config)
    partial = directory / "starts.partial.jsonl"
    if partial.exists() and not resume:
        raise FileExistsError("Existing prepared prefix requires same-ID resume")
    rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines() if line] if resume and partial.exists() else []
    for index, row in enumerate(rows):
        scene = dataset.scenes[index]
        if row.get("scene_id") != scene.scene_id or row.get("scene_sha256") != scene.source_hash:
            raise ValueError("Prepared prefix does not match frozen scenes")
    _status(directory, job, stage="preparing", prepared_scenes=len(rows),
            preparation_total=len(dataset))
    for index in range(len(rows), len(dataset)):
        if _stop_requested(directory):
            _status(directory, job, stage="stopped", prepared_scenes=index)
            return
        scene = dataset.scenes[index]
        began = time.monotonic()
        try:
            state = initial_state(scene, core_config)
            row = _start_record(scene, state, time.monotonic() - began)
        except ValueError as error:
            row = {"scene_id": scene.scene_id, "scene_sha256": scene.source_hash,
                   "initialization_valid": False,
                   "initialization_error": str(error),
                   "initialization_seconds": time.monotonic() - began,
                   "initialization_repair_geometry_calls": None,
                   "initialization_repair_ray_pair_checks": None,
                   "initialization_repair_attempts": None,
                   "state": None}
        _jsonl(partial, row)
        rows.append(row)
        _status(directory, job, stage="preparing", prepared_scenes=len(rows),
                preparation_total=len(dataset))
    output = directory / "starts.jsonl"
    output.write_bytes(partial.read_bytes())
    _write_artifacts(directory, {"run_id": job["run_id"],
                "reconstruction_starts": str(output),
                "reconstruction_starts_sha256": _sha256(output),
                "dataset_sha256": config["datasets"][role]["sha256"],
                "source_root": job["source_root"]})
    _status(directory, job, stage="completed", prepared_scenes=len(rows),
            preparation_total=len(dataset), starts_sha256=_sha256(output))
    _event(directory, job, "reconstruction_preparation_completed", scenes=len(rows))


def _choose_rule(state, candidates):
    """Cheap deterministic whole-action heuristic; no outcome simulation."""
    loads = np.asarray(state.loads, dtype=np.float64)
    mean = max(float(loads.mean()), 1e-9)
    def score(action):
        if action.operation == "stop":
            return (-10.0,)
        beams = action.beams
        if action.operation == "merge_split":
            i, j, k = beams
            value = loads[k] / mean - (loads[i] + loads[j]) / mean
        elif action.operation in ("repartition2", "repartition3", "assign"):
            value = float(np.ptp(loads[list(beams)])) / mean
        else:
            value = -1e9
        geometric_key = tuple(-float(value) for beam in beams
                              for value in state.centers[beam])
        return (value, -OPERATIONS.index(action.operation),
                geometric_key, -int(action.template))
    return max(candidates, key=score)


def _choose_search(state, candidates, core_config, *, width):
    """Bounded full-action trial search, charging every geometry call.

    This is intentionally an outcome-aware baseline. Policy inference and R1
    never receive its trial results. Trial count is logged separately from
    geometry-call budget; no candidate gets a free reconstruction.
    """
    from .reconstruction import apply_action
    began = time.monotonic()
    non_stop = [item for item in candidates if item.operation != "stop"]
    ordered = sorted(non_stop, key=lambda item: (
        OPERATIONS.index(item.operation),
        tuple(float(value) for beam in item.beams
              for value in state.centers[beam]), int(item.template)))
    if not ordered:
        stop = next(item for item in candidates if item.operation == "stop")
        stopped, info, _, _, metrics = apply_action(state, stop, core_config)
        return stop, stopped, {**info, "search_trials": 0, "trial_actions": [],
                                "search_seconds": 0.0,
                                "fit_candidate_checks_total": stopped.fit_candidate_checks,
                                "assignment_candidate_checks_total": stopped.assignment_candidate_checks,
                                "ray_pair_checks_total": stopped.ray_pair_checks}, 0, metrics
    # Include the same cheap rule choice, then cover each operation and spread
    # the remaining slots across identities. This is still a frozen selector.
    trials = []
    rule = _choose_rule(state, candidates)
    if rule.operation != "stop":
        trials.append(rule)
    for operation in OPERATIONS[1:]:
        group = [item for item in ordered if item.operation == operation]
        if group and len(trials) < width:
            middle = group[len(group) // 2]
            if middle not in trials:
                trials.append(middle)
    spread = np.linspace(0, len(ordered) - 1, min(width * 2, len(ordered)), dtype=int)
    for index in spread:
        action = ordered[int(index)]
        if action not in trials:
            trials.append(action)
        if len(trials) >= width:
            break
    charged = state.copy()
    best = None
    best_quality = float(_metrics(state, core_config)["quality"])
    tried = 0
    trials_log = []
    for action in trials:
        if charged.geometry_calls >= core_config.max_geometry_calls_total:
            break
        candidate_state, info, _, _, metrics = apply_action(charged.copy(), action,
                                                             core_config)
        tried += 1
        charged.geometry_calls = candidate_state.geometry_calls
        charged.attempts = candidate_state.attempts
        for name in ("fit_candidate_checks", "assignment_candidate_checks",
                     "ray_pair_checks", "elapsed_fit_s", "elapsed_validate_s",
                     "elapsed_repair_s"):
            setattr(charged, name, getattr(candidate_state, name))
        quality = float(metrics.get("quality", -float("inf")))
        trials_log.append({"action": action_identity(action),
                           "accepted": bool(info.get("accepted", False)),
                           "reason": info.get("reason"),
                           "quality": _finite_json(quality),
                           "geometry_calls": int(info.get("geometry_calls", 0)),
                           "fit_candidate_checks": int(info.get("fit_candidate_checks", 0)),
                           "assignment_candidate_checks": int(info.get("assignment_candidate_checks", 0)),
                           "ray_pair_checks": int(info.get("ray_pair_checks", 0))})
        if quality > best_quality:
            best, best_quality = (action, candidate_state, info, metrics), quality
    if best is None:
        stop = next(item for item in candidates if item.operation == "stop")
        state, info, _, _, metrics = apply_action(charged, stop, core_config)
        return stop, state, {**info, "search_trials": tried,
                             "trial_actions": trials_log,
                             "search_seconds": time.monotonic() - began,
                             "fit_candidate_checks_total": state.fit_candidate_checks,
                             "assignment_candidate_checks_total": state.assignment_candidate_checks,
                             "ray_pair_checks_total": state.ray_pair_checks}, tried, metrics
    action, chosen, info, metrics = best
    chosen.geometry_calls = charged.geometry_calls
    chosen.attempts = charged.attempts
    for name in ("fit_candidate_checks", "assignment_candidate_checks",
                 "ray_pair_checks", "elapsed_fit_s", "elapsed_validate_s",
                 "elapsed_repair_s"):
        setattr(chosen, name, getattr(charged, name))
    chosen.done = bool(chosen.done or chosen.geometry_calls >= core_config.max_geometry_calls_total)
    metrics = _metrics(chosen, core_config)
    return action, chosen, {**info, "search_trials": tried,
                            "trial_actions": trials_log,
                            "search_seconds": time.monotonic() - began,
                            "geometry_calls_total": chosen.geometry_calls,
                            "fit_candidate_checks_total": chosen.fit_candidate_checks,
                            "assignment_candidate_checks_total": chosen.assignment_candidate_checks,
                            "ray_pair_checks_total": chosen.ray_pair_checks}, tried, metrics


def _evaluate_scene(scene, start, start_record, route, core_config, *, policy, rng,
                    device, search_width=8):
    from .reconstruction import action_candidates, apply_action
    if start is None:
        return {"scene_id": scene.scene_id, "scene_sha256": scene.source_hash,
                "route": route, "initialization_valid": False,
                "initialization_error": start_record["initialization_error"],
                "metrics": {}, "initial_metrics": {},
                "macro_steps": 0, "geometry_calls": 0,
                "attempts": 0, "actions": [], "users": [
                    {"aircraft_id": identifier, "demand_bps": float(demand),
                     "assigned_beam": None}
                    for identifier, demand in zip(scene.aircraft_ids, scene.demands)],
                "initialization_seconds": float(start_record["initialization_seconds"]),
                "policy_seconds": 0.0, "reconstruction_seconds": 0.0,
                "final_check_seconds": 0.0,
                "initialization_repair_geometry_calls": start_record.get(
                    "initialization_repair_geometry_calls"),
                "initialization_repair_ray_pair_checks": start_record.get(
                    "initialization_repair_ray_pair_checks"),
                "initialization_repair_attempts": start_record.get(
                    "initialization_repair_attempts"),
                "total_seconds": float(start_record["initialization_seconds"])}
    state = start.copy()
    actions = []
    policy_seconds = 0.0
    reconstruction_seconds = 0.0
    observation_ray_pairs = 0
    observation_seconds = 0.0
    began_check = time.monotonic()
    metrics = _metrics(state, core_config)
    initial_check_seconds = time.monotonic() - began_check
    if not state.initialization_valid:
        return _row(scene, state, metrics, route=route, actions=actions,
                    initialization_seconds=float(start_record["initialization_seconds"]),
                    policy_seconds=0.0, reconstruction_seconds=0.0,
                    final_check_seconds=initial_check_seconds,
                    initial_metrics=metrics)
    while not state.done and route != "R0" and state.steps < core_config.max_macro_steps:
        metrics_before = metrics
        began_candidates = time.monotonic()
        candidates = action_candidates(state, core_config)
        candidate_hash = _candidate_digest(candidates) if route != "P1" else None
        policy_seconds += time.monotonic() - began_candidates
        began = time.monotonic()
        choice = None
        if route == "R1":
            choice = _choose_rule(state, candidates)
        elif route == "R2":
            choice = candidates[int(rng.integers(len(candidates)))]
        elif route == "R3":
            choice, next_state, info, trials, metrics = _choose_search(
                state, candidates, core_config, width=search_width)
        elif route == "P1":
            observation_began = time.monotonic()
            observation = stack_observations([graph_observation(state, core_config)], device)
            observation_seconds += time.monotonic() - observation_began
            observation_ray_pairs += len(scene.demands) * 100 + 100 * 100
            choice, _, _, _, _, digests = policy.act(observation, [candidates],
                                                     deterministic=True)
            candidate_hash = digests[0]
            choice = choice[0]
        else:
            raise ValueError("Unknown S1-03 comparison route")
        if route != "R3":
            policy_seconds += time.monotonic() - began
        if route != "R3":
            began = time.monotonic()
            next_state, info, _, _, metrics = apply_action(state, choice, core_config)
            reconstruction_seconds += time.monotonic() - began
        else:
            reconstruction_seconds += float(info.get("search_seconds", 0))
        actions.append({"requested_action": action_identity(choice),
                        "requested_action_fields": {"operation": choice.operation,
                                                    "beams": choice.beams,
                                                    "template": choice.template},
                        "actual_action": _actual_action(choice, info),
                        "accepted": bool(info.get("accepted", False)),
                        "reason": info.get("reason"),
                        "candidate_count": len(candidates),
                        "candidate_identity_sha256": candidate_hash,
                        "search_trials": trials if route == "R3" else 0,
                        "geometry_calls_after": int(next_state.geometry_calls),
                        "attempts_after": int(next_state.attempts),
                        "quality_before": metrics_before.get("quality"),
                        "quality_after": metrics.get("quality"),
                        "j_before": metrics_before.get("j"),
                        "j_after": metrics.get("j"),
                        "details": info})
        state = next_state
        if choice.operation == "stop":
            break
    began = time.monotonic()
    metrics = _metrics(state, core_config)
    final_check_seconds = initial_check_seconds + time.monotonic() - began
    return _row(scene, state, metrics, route=route, actions=actions,
                initialization_seconds=float(start_record["initialization_seconds"]),
                policy_seconds=policy_seconds,
                reconstruction_seconds=reconstruction_seconds,
                final_check_seconds=final_check_seconds,
                observation_ray_pairs=observation_ray_pairs,
                observation_seconds=observation_seconds,
                initial_metrics=_metrics(start, core_config))


def _summary(rows):
    valid = sum(bool(row["initialization_valid"]) for row in rows)
    result = {"scenes": len(rows), "initialization_valid": valid,
              "initialization_failed": len(rows) - valid,
              "macro_steps": sum(row["macro_steps"] for row in rows),
              "geometry_calls": sum(row["geometry_calls"] for row in rows),
              "attempts": sum(row["attempts"] for row in rows)}
    for key in ("initialization_repair_geometry_calls",
                "initialization_repair_ray_pair_checks",
                "initialization_repair_attempts", "fit_candidate_checks",
                "assignment_candidate_checks", "ray_pair_checks",
                "candidate_builds", "candidate_screen_ray_pairs",
                "observation_ray_pairs"):
        available = [row.get(key) for row in rows if row.get(key) is not None]
        result[key + "_total"] = (sum(int(value) for value in available)
                                   if len(available) == len(rows) else None)
        result[key + "_defined_total"] = sum(int(value) for value in available)
        result[key + "_available_count"] = len(available)
    for key in ("quality", "j", "coverage_users", "coverage_demand",
                "subject_cv", "load_cv", "subject_in_band_beams"):
        values = [row["metrics"].get(key) for row in rows]
        finite = [float(value) for value in values if value is not None and np.isfinite(value)]
        result[key + "_mean"] = float(np.mean(finite)) if len(finite) == len(rows) and rows else None
        result[key + "_defined_mean"] = float(np.mean(finite)) if finite else None
        result[key + "_available_count"] = len(finite)
        result[key + "_undefined_count"] = len(rows) - len(finite)
    result["strict_coverage_scenes"] = sum(bool(row["metrics"].get("coverage_100", False))
                                            for row in rows)
    result["target_scenes"] = sum(bool(row["metrics"].get("target_met", False))
                                   for row in rows)
    result["strict_coverage_missing_metrics"] = sum(
        "coverage_100" not in row["metrics"] for row in rows)
    result["target_missing_metrics"] = sum("target_met" not in row["metrics"]
                                            for row in rows)
    result["metric_mean_denominator"] = "all_scenes_only_when_all_values_defined"
    result["metric_defined_mean_denominator"] = "available_finite_scenes"
    attempted = Counter(action["requested_action_fields"]["operation"]
                        for row in rows for action in row.get("actions", []))
    accepted = Counter(action["requested_action_fields"]["operation"]
                       for row in rows for action in row.get("actions", [])
                       if action.get("accepted"))
    result["action_attempts_by_operation"] = dict(attempted)
    result["action_accepted_by_operation"] = dict(accepted)
    result["search_trials"] = sum(int(action.get("search_trials", 0))
                                  for row in rows for action in row.get("actions", []))
    result["action_acceptance_by_operation"] = {
        operation: accepted[operation] / count for operation, count in attempted.items()
        if count and operation != "stop"}
    result["explicit_stop_scenes"] = sum(any(action["requested_action_fields"]["operation"] ==
                                              "stop" for action in row.get("actions", []))
                                         for row in rows)
    result["immediate_stop_scenes"] = sum(len(row.get("actions", [])) == 1 and
                                          row["actions"][0]["requested_action_fields"]["operation"] ==
                                          "stop" for row in rows)
    result["terminal_macro_step_distribution"] = dict(Counter(
        int(row["macro_steps"]) for row in rows))
    quality_changes = []
    for row in rows:
        old = row.get("initial_metrics", {}).get("quality")
        new = row.get("metrics", {}).get("quality")
        if old is not None and new is not None and np.isfinite(old) and np.isfinite(new):
            quality_changes.append(float(new - old))
    result["quality_change_defined_mean"] = (float(np.mean(quality_changes))
                                             if quality_changes else None)
    result["quality_change_available_count"] = len(quality_changes)
    for key in ("initialization_seconds", "policy_seconds", "reconstruction_seconds",
                "final_check_seconds", "total_seconds"):
        values = [float(row[key]) for row in rows]
        result[key] = {"mean": float(np.mean(values)) if values else None,
                       "median": float(np.median(values)) if values else None,
                       "p95": float(np.percentile(values, 95)) if values else None}
    return result


def _evaluate_rows(directory, job, config, *, role, route, policy=None,
                   step=0, node_path=None, resume=False):
    dataset = _dataset(config, role)
    core_config = _core_config(config)
    starts, start_records = _load_starts(config, role, dataset, core_config)
    partial = directory / "results.partial.jsonl" if node_path is None else None
    if partial is not None and partial.exists() and not resume:
        raise FileExistsError("Existing evaluation prefix requires same-ID resume")
    rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines() if line] if resume and partial and partial.exists() else []
    for index, row in enumerate(rows):
        if row.get("scene_id") != dataset.scenes[index].scene_id:
            raise ValueError("Evaluation partial prefix differs from frozen scenes")
    for index in range(len(rows), len(dataset)):
        if _stop_requested(directory):
            break
        # Scene-specific frozen random source: resume and validation traversal
        # cannot alter another scene's random route or training RNG state.
        scene_rng = np.random.default_rng(int(config["training"]["seed"]) + 1729 + index)
        row = _evaluate_scene(dataset.scenes[index], starts[index], start_records[index],
                              route, core_config, policy=policy, rng=scene_rng,
                              device="cuda" if policy is not None else "cpu",
                              search_width=int(config["coverage"]["reconstruction"].get(
                                  "search_width", 8)))
        row.update(scene_index=index, group=role, model_step=step,
                   training_seed=config["training"]["seed"])
        rows.append(row)
        if partial is not None:
            _jsonl(partial, row)
            _status(directory, job, stage="evaluating", evaluated_scenes=len(rows),
                    evaluation_total=len(dataset), model_step=step)
    if node_path is not None and len(rows) == len(dataset):
        _write_json(node_path, {"scheduled_step": step, "actual_env_step": step,
                                "scene_count": len(dataset), "config_hash": config_hash(config),
                                "rows": rows, "summary": _summary(rows),
                                "completed_at": _now()})
    return rows, _summary(rows), len(dataset)


class ReconstructionVectorEnv:
    """CPU geometry with exact, checkpointable per-row scene state."""

    def __init__(self, dataset, starts, core_config, *, num_envs, seed):
        if len(starts) != len(dataset):
            raise ValueError("A frozen start is required for every training scene")
        self.dataset = dataset
        self.starts = starts
        self.core_config = core_config
        self.num_envs = num_envs
        self.eligible = [index for index, state in enumerate(starts)
                         if state is not None and state.initialization_valid]
        if not self.eligible:
            raise ValueError("No eligible training scene meets initialization gate")
        self.rng = np.random.default_rng(seed)
        self.scene_indices = [self._sample_index() for _ in range(num_envs)]
        self.states = [starts[index].copy() for index in self.scene_indices]
        self.episode_serial = [0 for _ in range(num_envs)]

    def _sample_index(self):
        return self.eligible[int(self.rng.integers(len(self.eligible)))]

    def observation(self, device):
        return stack_observations([graph_observation(state, self.core_config)
                                   for state in self.states], device)

    def candidates(self):
        from .reconstruction import action_candidates
        return [action_candidates(state, self.core_config) for state in self.states]

    def step(self, actions):
        from .reconstruction import apply_action
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        events, completed = [], []
        for row, action in enumerate(actions):
            state = self.states[row]
            scene_index = self.scene_indices[row]
            metrics_before = _metrics(state, self.core_config)
            began = time.monotonic()
            next_state, info, reward, done, metrics = apply_action(
                state, action, self.core_config)
            event = {"row": row, "scene_index": scene_index,
                     "scene_id": self.dataset.scenes[scene_index].scene_id,
                     "episode_serial": self.episode_serial[row],
                     "requested_action": action_identity(action),
                     "requested_action_fields": {"operation": action.operation,
                                                 "beams": action.beams,
                                                 "template": action.template},
                     "actual_action": _actual_action(action, info),
                     "accepted": bool(info.get("accepted", False)),
                     "reason": info.get("reason"),
                     "reward": float(reward), "metrics_after": metrics,
                     "quality_before": metrics_before.get("quality"),
                     "j_before": metrics_before.get("j"),
                     "geometry_calls_after": int(next_state.geometry_calls),
                     "attempts_after": int(next_state.attempts),
                     "reconstruction_seconds": time.monotonic() - began,
                     "details": info}
            rewards[row] = reward
            dones[row] = bool(done)
            events.append(event)
            if done:
                completed.append((row, scene_index, next_state, metrics,
                                  self.episode_serial[row]))
                self.episode_serial[row] += 1
                self.scene_indices[row] = self._sample_index()
                self.states[row] = self.starts[self.scene_indices[row]].copy()
            else:
                self.states[row] = next_state
        return rewards, dones, events, completed

    def state_dict(self):
        return {"scene_indices": list(self.scene_indices), "states": [state.to_record()
                 for state in self.states], "rng": dict(self.rng.bit_generator.state),
                "episode_serial": list(self.episode_serial),
                "eligible_indices": list(self.eligible)}

    def load_state_dict(self, payload):
        from .reconstruction import ReconstructionState
        if payload["eligible_indices"] != self.eligible:
            raise ValueError("Training eligibility changed across resume")
        self.scene_indices = [int(value) for value in payload["scene_indices"]]
        self.states = [ReconstructionState.from_record(self.dataset.scenes[index], record,
                                                       self.core_config)
                       for index, record in zip(self.scene_indices, payload["states"])]
        self.episode_serial = [int(value) for value in payload["episode_serial"]]
        if len(self.states) != self.num_envs:
            raise ValueError("Checkpoint vector environment width differs")
        self.rng.bit_generator.state = payload["rng"]


def _checkpoint(directory, job, config, policy, optimizer, env, *, step, update,
                episode, last_validation, elapsed, episode_returns,
                episode_traces, gate_passed):
    payload = {"format_version": 1, "policy_format": "reconstruction_relation_ppo_v1",
               "config_hash": config_hash(config), "source_commit": job["git_commit"],
               "source_root": job["source_root"], "run_id": job["run_id"],
               "dataset_hashes": {key: value["sha256"] for key, value in config["datasets"].items()},
               "start_hashes": {key: value["sha256"] for key, value in config["reconstruction_starts"].items()},
               "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
               "env": env.state_dict(), "step": step, "update": update,
               "episode": episode, "last_validation": last_validation,
               "elapsed_seconds": elapsed, "episode_returns": episode_returns,
               "episode_traces": episode_traces, "gate_passed": gate_passed,
               "rng": _rng_state()}
    path = directory / "checkpoint.pt"
    _save_torch(path, payload)
    _write_artifacts(directory, {"run_id": job["run_id"],
                "config": str(directory / "config.json"), "checkpoint": str(path),
                "datasets": config["datasets"],
                "reconstruction_starts": config["reconstruction_starts"]})
    return _sha256(path)


def _restore(directory, job, config, policy, optimizer, env, *, device="cuda"):
    payload = _load_torch(directory / "checkpoint.pt", device)
    if (payload.get("format_version") != 1 or
            payload.get("policy_format") != "reconstruction_relation_ppo_v1" or
            payload.get("config_hash") != config_hash(config) or
            payload.get("source_commit") != job["git_commit"] or
            payload.get("source_root") != job["source_root"] or
            payload.get("run_id") != job["run_id"] or
            payload.get("dataset_hashes") != {key: value["sha256"] for key, value in config["datasets"].items()} or
            payload.get("start_hashes") != {key: value["sha256"] for key, value in config["reconstruction_starts"].items()}):
        raise ValueError("S1-03 checkpoint source/config/data/start mismatch")
    policy.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    env.load_state_dict(payload["env"])
    _restore_rng(payload["rng"])
    return payload


def _truncate_to_checkpoint(path, key, maximum, *, validation_maximum=None):
    if not path.exists():
        return
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    retained = [row for row in lines if
                int(row.get(key, 0)) <= maximum and
                (validation_maximum is None or row.get("group") != "validation" or
                 int(row.get("scheduled_step", 0)) <= validation_maximum)]
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            for row in retained), encoding="utf-8")
    os.replace(temp, path)


def _train(directory, job, config, resume, preflight):
    training = config["training"]
    if training["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal S1-03 training requires actual CUDA")
    train_data = _dataset(config, "train")
    val_data = _dataset(config, "validation")
    core_config = _core_config(config)
    train_starts, train_start_records = _load_starts(config, "train", train_data,
                                                    core_config)
    _load_starts(config, "validation", val_data, core_config)
    eligibility, eligibility_path = _training_eligibility(
        directory, config, train_data, train_starts, train_start_records,
        core_config, resume=resume)
    _write_artifacts(directory, {"training_eligibility": str(eligibility_path),
                                 "training_eligibility_sha256": _sha256(eligibility_path)})
    _seed(training["seed"])
    env = ReconstructionVectorEnv(train_data, train_starts, core_config,
                                  num_envs=training["num_envs"], seed=training["seed"])
    policy = _policy(config, "cuda")
    optimizer = torch.optim.Adam(policy.parameters(), lr=training["lr"])
    if resume:
        payload = _restore(directory, job, config, policy, optimizer, env)
        step, update, episode = payload["step"], payload["update"], payload["episode"]
        last_validation = payload["last_validation"]
        elapsed_base = payload["elapsed_seconds"]
        returns = payload["episode_returns"]
        traces = payload["episode_traces"]
        gate_passed = payload["gate_passed"]
        for filename, key in (("actions.jsonl", "env_step"),
                              ("episodes.jsonl", "terminal_env_step"),
                              ("updates.jsonl", "env_step")):
            _truncate_to_checkpoint(directory / filename, key, step)
        _truncate_to_checkpoint(directory / "metrics.jsonl", "step", step,
                                validation_maximum=last_validation)
        _truncate_to_checkpoint(directory / "evaluations.jsonl", "scheduled_step",
                                last_validation)
    else:
        step = update = episode = last_validation = 0
        elapsed_base = 0.0
        returns = np.zeros(env.num_envs, dtype=np.float64)
        traces = [[] for _ in range(env.num_envs)]
        gate_passed = False
    gate_step = int(training["mechanism_gate_step"])
    began = time.monotonic()
    _status(directory, job, stage="training", env_step=step,
            update_step=update, episode=episode, budget_env_steps=training["total_steps"],
            source_training_scenes=eligibility["scene_count"],
            eligible_training_scenes=eligibility["eligible_scene_count"],
            training_eligibility_sha256=_sha256(eligibility_path),
            preflight=preflight)
    while True:
        if _stop_requested(directory):
            sha = _checkpoint(directory, job, config, policy, optimizer, env,
                              step=step, update=update, episode=episode,
                              last_validation=last_validation,
                              elapsed=elapsed_base + time.monotonic() - began,
                              episode_returns=returns, episode_traces=traces,
                              gate_passed=gate_passed)
            _status(directory, job, stage="stopped", stop_reason="requested",
                    env_step=step, checkpoint_step=step, checkpoint_sha256=sha)
            return
        due = last_validation + training["validation_interval"]
        if step >= due:
            node = directory / "validation_nodes" / f"{due}.json"
            node.parent.mkdir(parents=True, exist_ok=True)
            saved = _read(node)
            if saved is None:
                rng_before = _rng_state()
                was_training = policy.training
                policy.eval()
                try:
                    rows, summary, count = _evaluate_rows(
                        directory, job, config, role="validation", route="P1",
                        policy=policy, step=due, node_path=node)
                finally:
                    policy.train(was_training)
                    _restore_rng(rng_before)
                if len(rows) != count:
                    sha = _checkpoint(directory, job, config, policy, optimizer, env,
                                      step=step, update=update, episode=episode,
                                      last_validation=last_validation,
                                      elapsed=elapsed_base + time.monotonic() - began,
                                      episode_returns=returns, episode_traces=traces,
                                      gate_passed=gate_passed)
                    _status(directory, job, stage="stopped", stop_reason="validation_interrupted",
                            checkpoint_sha256=sha, env_step=step)
                    return
            else:
                if (saved.get("scheduled_step") != due or
                        saved.get("config_hash") != config_hash(config) or
                        saved.get("scene_count") != len(val_data)):
                    raise ValueError("Frozen validation node differs from current job")
                rows, summary = saved["rows"], saved["summary"]
            _jsonl(directory / "evaluations.jsonl", {"scheduled_step": due,
                    "actual_env_step": step, "scene_count": len(rows),
                    "summary": summary, "node": str(node), "node_sha256": _sha256(node)})
            _jsonl(directory / "metrics.jsonl", {"step": step,
                    "group": "validation", "scheduled_step": due,
                    "values": summary, "wall_time": _now()})
            last_validation = due
            sha = _checkpoint(directory, job, config, policy, optimizer, env,
                              step=step, update=update, episode=episode,
                              last_validation=last_validation,
                              elapsed=elapsed_base + time.monotonic() - began,
                              episode_returns=returns, episode_traces=traces,
                              gate_passed=gate_passed)
            _status(directory, job, stage="training", env_step=step,
                    last_validation_step=due, checkpoint_step=step,
                    checkpoint_sha256=sha)
            continue
        if step >= gate_step and not gate_passed:
            checkpoint = directory / "checkpoint.pt"
            if not checkpoint.is_file():
                raise ValueError("Mechanism gate requires a frozen checkpoint")
            sha = _sha256(checkpoint)
            approval = _read(directory / "mechanism_gate.json", {})
            audit_path = Path(approval.get("audit_path", ""))
            if not audit_path.is_absolute():
                audit_path = directory / audit_path
            audit_valid = (audit_path.is_file() and
                           approval.get("audit_sha256") == _sha256(audit_path))
            if not (approval.get("approved") is True and
                    approval.get("checkpoint_sha256") == sha and audit_valid):
                _status(directory, job, stage="stopped", stop_reason="mechanism_gate_pending",
                        env_step=step, checkpoint_step=step, checkpoint_sha256=sha)
                _event(directory, job, "mechanism_gate_pending", env_step=step,
                       checkpoint_sha256=sha)
                return
            gate_passed = True
            _event(directory, job, "mechanism_gate_approved", env_step=step,
                   checkpoint_sha256=sha)
        if step >= training["total_steps"]:
            break
        # Rollouts end exactly at every validation/checkpoint/gate boundary.
        next_boundary = min(training["total_steps"], due,
                            gate_step if not gate_passed else training["total_steps"],
                            ((step // training["checkpoint_interval"]) + 1) *
                            training["checkpoint_interval"])
        remaining = (next_boundary - step) // env.num_envs
        length = min(training["rollout_steps"], remaining)
        if length < 1:
            raise ValueError("S1-03 boundaries must align to num_envs")
        buckets = {key: [] for key in ("beam", "relation", "global")}
        choices_batch, masks_batch, logp_batch = [], [], []
        values_batch, rewards_batch, dones_batch = [], [], []
        completed_rows = []
        action_rows = []
        for offset in range(length):
            observation_began = time.monotonic()
            observation = env.observation("cuda")
            observation_batch_seconds = time.monotonic() - observation_began
            selection_began = time.monotonic()
            candidates = env.candidates()
            with torch.no_grad():
                actions, choices, masks, logp, values, candidate_digests = policy.act(
                    observation, candidates)
            selection_batch_seconds = time.monotonic() - selection_began
            rewards, dones, events, completed = env.step(actions)
            for key in buckets:
                buckets[key].append(observation[key])
            choices_batch.append(choices)
            masks_batch.append(masks)
            logp_batch.append(logp)
            values_batch.append(values)
            rewards_batch.append(torch.as_tensor(rewards * training["reward_scale"], device="cuda"))
            dones_batch.append(torch.as_tensor(dones, device="cuda"))
            choice_values = choices.detach().cpu().tolist()
            logp_values = logp.detach().cpu().tolist()
            mask_values = masks.detach().cpu().numpy()
            for row, event in enumerate(events):
                env_step = step + offset * env.num_envs + row + 1
                packed_masks = [np.packbits(mask_values[row, head]).tobytes().hex()
                                for head in range(5)]
                event.update(env_step=env_step, joint_log_prob=float(logp_values[row]),
                             action_identity=choice_values[row],
                             conditional_masks_packbits_hex=packed_masks,
                             candidate_count=len(candidates[row]),
                             observation_ray_pairs=(len(train_data.scenes[
                                 event["scene_index"]].demands) * 100 + 100 * 100),
                             observation_batch_seconds=observation_batch_seconds,
                             selection_batch_seconds=selection_batch_seconds,
                             candidate_identity_sha256=candidate_digests[row])
                action_rows.append(event)
                traces[row].append(event)
                returns[row] += float(rewards[row] * training["reward_scale"])
            for row, scene_index, terminal_state, terminal_metrics, serial in completed:
                terminal_env_step = step + offset * env.num_envs + row + 1
                completed_rows.append(_row(train_data.scenes[scene_index], terminal_state,
                                           terminal_metrics, route="P1",
                                           actions=traces[row], initialization_seconds=0.0,
                                           policy_seconds=sum(float(item["observation_batch_seconds"] +
                                                                    item["selection_batch_seconds"])
                                                              for item in traces[row]) / env.num_envs,
                                           reconstruction_seconds=sum(float(item["reconstruction_seconds"])
                                                                      for item in traces[row]),
                                           final_check_seconds=0.0,
                                           observation_ray_pairs=sum(int(item["observation_ray_pairs"])
                                                                     for item in traces[row]),
                                           observation_seconds=sum(float(item["observation_batch_seconds"])
                                                                   for item in traces[row]) /
                                                               env.num_envs,
                                           initial_metrics=_metrics(train_starts[scene_index],
                                                                    core_config)) |
                                      {"terminal_env_step": terminal_env_step,
                                       "episode_serial": serial,
                                       "episode_return": float(returns[row]),
                                       "training_seed": training["seed"],
                                       "scene_index": scene_index})
                traces[row] = []
                returns[row] = 0.0
        with torch.no_grad():
            bootstrap = policy.value(env.observation("cuda"))
        rollout = GraphRollout({key: torch.stack(value) for key, value in buckets.items()},
                               torch.stack(choices_batch), torch.stack(masks_batch),
                               torch.stack(logp_batch), torch.stack(values_batch),
                               torch.stack(rewards_batch), torch.stack(dones_batch), bootstrap)
        diagnostics = ppo_update(policy, optimizer, rollout,
                                 gamma=training["gamma"],
                                 gae_lambda=training["gae_lambda"],
                                 clip_ratio=training["clip_ratio"],
                                 entropy_coef=training["entropy_coef"],
                                 value_coef=training["value_coef"],
                                 max_grad_norm=training["max_grad_norm"],
                                 epochs=training["epochs"],
                                 minibatch_size=training["minibatch_size"])
        updates = diagnostics.pop("updates")
        step += length * env.num_envs
        update += updates
        for index, row in enumerate(completed_rows, start=1):
            row["episode"] = episode + index
        episode += len(completed_rows)
        _jsonl_many(directory / "actions.jsonl", action_rows)
        _jsonl_many(directory / "episodes.jsonl", completed_rows)
        _jsonl(directory / "updates.jsonl", {"env_step": step, "update_step": update,
                "minibatch_updates": updates, "rollout_interactions": length * env.num_envs,
                "episode": episode, "training_seed": training["seed"], **diagnostics})
        _jsonl(directory / "metrics.jsonl", {"step": step, "group": "ppo_update",
                "values": diagnostics | {"minibatch_updates": updates}, "wall_time": _now()})
        if completed_rows:
            _jsonl(directory / "metrics.jsonl", {"step": step, "group": "terminal",
                    "values": _summary(completed_rows), "wall_time": _now()})
        sha = _checkpoint(directory, job, config, policy, optimizer, env,
                          step=step, update=update, episode=episode,
                          last_validation=last_validation,
                          elapsed=elapsed_base + time.monotonic() - began,
                          episode_returns=returns, episode_traces=traces,
                          gate_passed=gate_passed)
        _status(directory, job, stage="training", env_step=step,
                update_step=update, episode=episode,
                budget_env_steps=training["total_steps"],
                checkpoint_step=step, checkpoint_sha256=sha,
                latest_update=diagnostics,
                wall_seconds=elapsed_base + time.monotonic() - began)
    final = Path(job["model_dir"]) / "final"
    final.mkdir(parents=True, exist_ok=True)
    _save_torch(final / "policy.pt", policy.state_dict())
    metadata = {"format_version": 1, "policy_format": "reconstruction_relation_ppo_v1",
                "schema_version": 4, "research_stage": "coverage",
                "run_id": job["run_id"],
                "network": config["network"], "coverage": config["coverage"],
                "training_seed": training["seed"], "env_step": step,
                "selection": "fixed_budget_endpoint", "source_commit": job["git_commit"],
                "source_root": job["source_root"], "config_hash": config_hash(config),
                "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "torch_threads": torch.get_num_threads(),
                "datasets": config["datasets"],
                "reconstruction_starts": config["reconstruction_starts"],
                "python_prefix": preflight.get("python_prefix"),
                "gpu": preflight.get("gpu"),
                "execution_target": job["execution_target"],
                "execution_profile": preflight.get("execution_target"),
                "policy_sha256": _sha256(final / "policy.pt"),
                "created_at": _now()}
    _write_json(final / "model.json", metadata)
    _write_artifacts(directory, {"run_id": job["run_id"],
                "config": str(directory / "config.json"),
                "checkpoint": str(directory / "checkpoint.pt"),
                "final_policy": str(final), "model": str(final / "policy.pt"),
                "model_metadata": str(final / "model.json"),
                "model_sha256": _sha256(final / "policy.pt"),
                "model_metadata_sha256": _sha256(final / "model.json"),
                "datasets": config["datasets"],
                "reconstruction_starts": config["reconstruction_starts"]})
    _status(directory, job, stage="completed", env_step=step,
            update_step=update, episode=episode, final_policy=str(final))
    _event(directory, job, "reconstruction_training_completed", env_step=step)


def _evaluate(directory, job, config, resume):
    route = config["coverage"]["reconstruction"]["route"]
    role = next(iter(config["datasets"]))
    policy = None
    model_step = 0
    if route == "P1":
        if config["training"]["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Formal S1-03 policy evaluation requires CUDA")
        metadata = _read(Path(job["model_path"]) / "model.json", {})
        if (metadata.get("policy_format") != "reconstruction_relation_ppo_v1" or
                metadata.get("research_stage") != "coverage" or
                metadata.get("schema_version") != 4 or
                metadata.get("network") != config["network"] or
                metadata.get("coverage", {}).get("reconstruction") !=
                config["coverage"]["reconstruction"]):
            raise ValueError("S1-03 model network or reconstruction geometry differs")
        if _sha256(Path(job["model_path"]) / "policy.pt") != metadata.get("policy_sha256"):
            raise ValueError("S1-03 policy file SHA differs from model metadata")
        policy = _policy(config, "cuda")
        policy.load_state_dict(_load_torch(Path(job["model_path"]) / "policy.pt", "cuda"))
        policy.eval()
        model_step = int(metadata["env_step"])
    rows, summary, count = _evaluate_rows(directory, job, config, role=role,
                                          route=route, policy=policy,
                                          step=model_step, resume=resume)
    if len(rows) < count:
        _status(directory, job, stage="stopped", evaluated_scenes=len(rows))
        return
    raw_path = directory / "results.partial.jsonl"
    raw_sha256 = _sha256(raw_path)
    result = {"run_id": job["run_id"], "route": route,
              "scene_count": len(rows), "summary": summary,
              "rows_path": str(raw_path), "rows_sha256": raw_sha256,
              "source_commit": job["git_commit"],
              "config_hash": config_hash(config),
              "execution_target": job["execution_target"],
              "python_prefix": sys.prefix,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "torch_threads": torch.get_num_threads(), "created_at": _now()}
    _write_json(directory / "results.json", result)
    _jsonl(directory / "metrics.jsonl", {"step": model_step,
            "group": "evaluation", "values": summary, "wall_time": _now()})
    _write_artifacts(directory, {"run_id": job["run_id"],
                "config": str(directory / "config.json"),
                "results": str(directory / "results.json"),
                "raw_results": str(raw_path),
                "raw_results_sha256": raw_sha256,
                "datasets": config["datasets"],
                "reconstruction_starts": config["reconstruction_starts"]})
    _status(directory, job, stage="completed", evaluated_scenes=len(rows),
            evaluation_total=count, result_path=str(directory / "results.json"))
    _event(directory, job, "reconstruction_evaluation_completed", scenes=len(rows))


def run_reconstruction_job(directory, job, config, *, resume=False, preflight=None):
    """Dispatch one schema-4 workbench task after the shared CUDA preflight."""
    directory = Path(directory)
    if config["schema_version"] != 4:
        raise ValueError("S1-03 runner requires schema 4")
    if preflight is None:
        raise ValueError("S1-03 formal workbench job requires verified preflight")
    route = config["coverage"]["reconstruction"]["route"]
    _status(directory, job, stage="initializing", preflight=preflight)
    _event(directory, job, "reconstruction_preflight_verified", preflight=preflight)
    if job["kind"] == "train":
        if route != "P1":
            raise ValueError("Only P1 trains a reconstruction policy")
        _train(directory, job, config, resume, preflight)
    elif job["kind"] == "evaluate":
        if route == "prepare":
            _prepare(directory, job, config, resume)
        elif route in ("R0", "R1", "R2", "R3", "P1"):
            _evaluate(directory, job, config, resume)
        else:
            raise ValueError("Unknown S1-03 comparison route")
    else:
        raise ValueError("Unknown S1-03 workbench task kind")
