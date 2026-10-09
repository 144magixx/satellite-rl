"""Schema-5 workbench worker for variable-count coverage experiments.

Formal tasks enter only through the workbench dispatcher.  The local callable
can also be exercised with a tiny synthetic job for correctness checks.
"""

from __future__ import annotations

from dataclasses import fields, replace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .config import config_hash
from .runner import (_event, _jsonl, _load_torch, _now, _read, _restore_rng,
                     _rng_state, _save_torch, _seed, _sha256, _status,
                     _stop_requested, _write_json)
from .variable_config import ROUTES, VariableConfig
from .variable_dataset import load_variable_dataset
from .variable_design import (Layout, VariableAction, apply_action, geometry_audit, initial_layout,
                              metrics, solve)
from .variable_observation import candidate_digest, observation
from .variable_ppo import (Transition, VariableSetPolicy, collate, ppo_update)


POLICY_FORMAT = "variable_set_ppo_v1"
LEARNING_ROUTES = ("R08", "R09")
_CPU_CONTEXT = None


def _core_config(config):
    supplied = config["coverage"]["variable"]
    names = {field.name for field in fields(VariableConfig)}
    values = {name: supplied[name] for name in names if name in supplied}
    if "width_choices_deg" in values:
        values["width_choices_deg"] = tuple(values["width_choices_deg"])
    return VariableConfig(**values)


def _dataset(config, role):
    record = config["datasets"][role]
    path = Path(record["path"]).resolve()
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise ValueError(f"Frozen aircraft dataset absent or SHA mismatch: {role}")
    dataset = load_variable_dataset(path)
    if not dataset.scenes:
        raise ValueError("Frozen aircraft dataset is empty")
    return dataset


def _scaled(scene, multiplier):
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Demand multiplier must be finite and positive")
    return replace(scene, demands=np.asarray(scene.demands, dtype=np.float64) * multiplier,
                   metadata={**scene.metadata, "demand_multiplier": float(multiplier)})


def _route(config):
    route = config["coverage"]["variable"]["route"]
    if route not in ROUTES:
        raise ValueError("Unknown variable-beam route")
    return route


def _actor(config, device):
    network = config["network"]
    if network["kind"] != "variable_set_ppo" or network.get("hidden_size", 128) != 128:
        raise ValueError("Variable learning needs frozen 128-wide set PPO")
    return VariableSetPolicy().to(device)


def _check_training(config, job):
    training = config["training"]
    purpose = job.get("experiment_role", "formal")
    if purpose not in ("formal", "verification"):
        raise ValueError("Unknown variable training experiment role")
    expected_steps = 2048 if purpose == "verification" else 262144
    if training["seed"] != 51001 or training["total_steps"] != expected_steps:
        raise ValueError("Variable training seed or role-specific decision budget differs")
    expected_role = "development" if purpose == "verification" else "validation"
    if set(config["datasets"]) != {"train", expected_role}:
        raise ValueError("Variable training dataset partition differs from experiment role")
    if training["num_envs"] != 32:
        raise ValueError("Variable runner requires 32 independently sampled streams")
    if training["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal variable-beam learning requires DQN/CUDA")
    batch = training["rollout_steps"] * training["num_envs"]
    if training["checkpoint_interval"] % batch or \
            training["validation_interval"] % batch or \
            training["total_steps"] % batch:
        raise ValueError("Rollouts must end exactly at frozen checkpoints and validations")
    return training


def _read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _truncate_lines(path, count):
    lines = _read_lines(path)
    if len(lines) < count:
        raise ValueError(f"Committed ledger is shorter than checkpoint: {path}")
    if len(lines) > count:
        path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                for row in lines[:count]), encoding="utf-8")


def _layout_row(scene, layout, cfg, *, route, actions, init_seconds,
                algorithm_seconds, geometry_seconds, final_seconds, calls,
                actual_calls=None, ray_pairs=None):
    result = metrics(scene, layout, cfg)
    demand = result["beam_demand_bps"]
    users = [{"aircraft_id": identifier, "demand_bps": float(value),
              "assigned_beam": int(owner) if owner >= 0 else None,
              "assigned_beam_stable_id": layout.beam_ids[int(owner)] if owner >= 0 else None}
             for identifier, value, owner in zip(scene.aircraft_ids, scene.demands,
                                                  layout.assignment)]
    return {"scene_id": scene.scene_id, "scene_sha256": scene.source_hash,
            "demand_multiplier": scene.metadata.get("demand_multiplier", 1.0),
            "route": route, "metrics": result, "layout": layout.to_record(),
            "users": users,
            "beams": [{"beam_id": i, "beam_stable_id": layout.beam_ids[i],
                       "demand_bps": float(value),
                       "capacity_bps": None, "carried_bps": None,
                       "unclipped_capacity_demand_ratio": None,
                       "si": None, "sgm": None,
                       "capacity_missing_reason": "downstream_allocation_not_yet_run"}
                      for i, value in enumerate(demand)],
            "actions": actions, "macro_steps": len(actions),
            "geometry_calls": int(actual_calls if actual_calls is not None else calls),
            "geometry_budget_charges": int(calls),
            "ray_pair_checks": int(ray_pairs) if ray_pairs is not None else None,
            "initialization_seconds": float(init_seconds),
            "algorithm_seconds": float(algorithm_seconds),
            "geometry_seconds": float(geometry_seconds),
            "final_check_seconds": float(final_seconds),
            "total_seconds": float(init_seconds + algorithm_seconds + final_seconds),
            "coverage_complete": bool(result["coverage_100"]),
            "demand_conserved": bool(result["demand_conserved"])}


def _summary(rows):
    if not rows:
        return {"scene_count": 0}
    return {"scene_count": len(rows),
            "coverage_100_count": sum(bool(row["metrics"]["coverage_100"]) for row in rows),
            "mean_coverage_users": float(np.mean([row["metrics"]["coverage_users"] for row in rows])),
            "mean_load_cv": float(np.mean([row["metrics"]["load_cv"] for row in rows])),
            "mean_beam_count": float(np.mean([row["metrics"]["beam_count"] for row in rows])),
            "mean_total_seconds": float(np.mean([row["total_seconds"] for row in rows]))}


def _select(scene, layout, cfg, route, policy, *, steps, geometry_calls,
            deterministic, budget=None):
    observation_row, menu = observation(scene, layout, cfg, route=route,
                                        steps=steps, geometry_calls=geometry_calls,
                                        budget=budget)
    batch = collate([observation_row], next(policy.parameters()).device)
    with torch.no_grad():
        choice, logp, value = policy.act(batch, deterministic=deterministic)
    index = int(choice.item())
    return menu[index][0], observation_row, index, float(logp.item()), \
        float(value.item()), candidate_digest(menu)


def _quality(values):
    # Strict full coverage remains a separate hard condition; this shaped
    # diagnostic gives the constructor learning signal before its first full
    # layout.  Reference capacity is a screening surrogate, not measured C.
    return (2.0 * values["coverage_demand"] + values["coverage_users"] -
            .25 * min(values["load_cv"], 10.0) -
            .25 * values["reference_unserved_bps"] /
            max(values["total_demand_bps"], 1.0) -
            .05 * values["beam_count"] / 200.0)


def _new_episode(dataset, cfg, route, sampler, multipliers):
    scene_index = int(sampler.integers(len(dataset.scenes)))
    level = int(sampler.integers(len(multipliers)))
    scene = _scaled(dataset.scenes[scene_index], multipliers[level])
    began = time.monotonic()
    layout, info = initial_layout(scene, cfg, route=route)
    initialization_seconds = time.monotonic() - began
    return scene_index, level, layout, info, initialization_seconds


def _fresh_stream(dataset, cfg, route, seed, multipliers):
    sampler = np.random.default_rng(seed)
    index, level, layout, info, elapsed = _new_episode(dataset, cfg, route,
                                                        sampler, multipliers)
    return {"sampler": sampler, "scene_index": index, "level": level,
            "layout": layout, "actions": [], "episode_return": 0.0,
            "episode_steps": 0, "geometry_calls": int(info.get("geometry_calls", 0)),
            "actual_geometry_calls": int(info.get("geometry_calls", 0)),
            "actual_ray_pair_checks": int(info.get("actual_ray_pair_checks", 0)),
            "init_seconds": elapsed, "algorithm_seconds": 0.0,
            "geometry_seconds": 0.0, "episode_serial": 0}


def _stream_scene(stream, dataset, multipliers):
    return _scaled(dataset.scenes[stream["scene_index"]],
                   multipliers[stream["level"]])


def _stream_record(stream):
    return {key: value for key, value in stream.items() if key not in ("sampler", "layout")} | {
        "layout": stream["layout"].to_record(),
        "sampler_state": stream["sampler"].bit_generator.state}


def _stream_from_record(record):
    sampler = np.random.default_rng()
    sampler.bit_generator.state = record["sampler_state"]
    return {key: value for key, value in record.items() if key not in ("sampler_state", "layout")} | {
        "sampler": sampler, "layout": Layout.from_record(record["layout"])}


def _cpu_worker_init(dataset_record, config, route, multipliers):
    """One immutable, SHA-checked aircraft dataset per spawned CPU process."""
    global _CPU_CONTEXT
    dataset = _dataset({"datasets": {"train": dataset_record}}, "train")
    _CPU_CONTEXT = {"dataset": dataset, "cfg": _core_config(config),
                    "config": config, "route": route,
                    "multipliers": tuple(multipliers)}


def _cpu_scene(packet):
    context = _CPU_CONTEXT
    return _scaled(context["dataset"].scenes[packet["scene_index"]],
                   context["multipliers"][packet["level"]])


def _cpu_packet(stream):
    return {"scene_index": stream["scene_index"], "level": stream["level"],
            "layout": stream["layout"].to_record(),
            "episode_steps": stream["episode_steps"],
            "geometry_calls": stream["geometry_calls"]}


def _cpu_terminal_packet(stream, stream_id):
    trace = stream["actions"]
    return _cpu_packet(stream) | {
        "init_seconds": stream["init_seconds"],
        "algorithm_seconds": stream["algorithm_seconds"],
        "geometry_seconds": stream["geometry_seconds"],
        "actual_geometry_calls": stream["actual_geometry_calls"],
        "actual_ray_pair_checks": stream["actual_ray_pair_checks"],
        "action_trace": {"file": "actions.jsonl", "stream_id": stream_id,
                         "stream_episode_serial": stream["episode_serial"] + 1,
                         "first_env_step": trace[0]["env_step"],
                         "last_env_step": trace[-1]["env_step"],
                         "count": len(trace)}}


def _cpu_observe(packet):
    context = _CPU_CONTEXT
    scene, cfg, route = _cpu_scene(packet), context["cfg"], context["route"]
    layout = Layout.from_record(packet["layout"])
    budget = {"geometry_calls": packet["geometry_calls"],
              "max_geometry_calls": cfg.max_geometry_calls}
    with geometry_audit() as audit:
        obs, menu = observation(scene, layout, cfg, route=route,
                                steps=packet["episode_steps"],
                                geometry_calls=packet["geometry_calls"],
                                budget=budget)
        before = metrics(scene, layout, cfg)
    return obs, [action.to_record() for action, _ in menu], before, \
        budget["geometry_calls"], dict(audit)


def _cpu_execute(item):
    packet, action_record, screened_calls = item
    context = _CPU_CONTEXT
    scene, cfg = _cpu_scene(packet), context["cfg"]
    layout = Layout.from_record(packet["layout"])
    budget = {"geometry_calls": screened_calls,
              "max_geometry_calls": cfg.max_geometry_calls}
    with geometry_audit() as audit:
        next_layout, info = apply_action(scene, layout, cfg,
                                         VariableAction.from_record(action_record), budget)
        after = metrics(scene, next_layout, cfg)
    return next_layout.to_record(), info, after, budget["geometry_calls"], dict(audit)


def _cpu_bootstrap(packet):
    context = _CPU_CONTEXT
    scene, cfg, route = _cpu_scene(packet), context["cfg"], context["route"]
    layout = Layout.from_record(packet["layout"])
    with geometry_audit() as audit:
        obs, _ = observation(scene, layout, cfg, route=route,
                             steps=packet["episode_steps"],
                             geometry_calls=packet["geometry_calls"],
                             value_only=True)
    return obs, dict(audit)


def _cpu_initial_layout(item):
    index, level = item
    context = _CPU_CONTEXT
    scene = _scaled(context["dataset"].scenes[index],
                    context["multipliers"][level])
    began = time.monotonic()
    layout, info = initial_layout(scene, context["cfg"], route=context["route"])
    return layout.to_record(), info, time.monotonic() - began


def _cpu_terminal(packet):
    context = _CPU_CONTEXT
    scene, cfg, route = _cpu_scene(packet), context["cfg"], context["route"]
    layout = Layout.from_record(packet["layout"])
    row = _layout_row(scene, layout, cfg, route=route, actions=[],
                      init_seconds=packet["init_seconds"],
                      algorithm_seconds=packet["algorithm_seconds"],
                      geometry_seconds=packet["geometry_seconds"],
                      final_seconds=0.0, calls=packet["geometry_calls"],
                      actual_calls=packet["actual_geometry_calls"],
                      ray_pairs=packet["actual_ray_pair_checks"])
    row.pop("actions")
    row["macro_steps"] = packet["action_trace"]["count"]
    row["action_trace"] = packet["action_trace"]
    return _downstream(scene, layout, context["config"], route, row, compact=True)


def _checkpoint_parallel(directory, job, config, policy, optimizer, *,
                         streams, step, updates, episode, last_validation,
                         ledger_counts):
    payload = {"format_version": 2, "run_id": job["run_id"],
               "config_hash": config_hash(config), "source_commit": job["git_commit"],
               "policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
               "rng": _rng_state(), "streams": [_stream_record(x) for x in streams],
               "step": step, "updates": updates, "episode": episode,
               "last_validation": last_validation, "ledger_counts": dict(ledger_counts)}
    path = directory / "checkpoint.pt"
    _save_torch(path, payload)
    return _sha256(path)


def _load_parallel(directory, job, config, policy, optimizer):
    payload = _load_torch(directory / "checkpoint.pt", "cpu")
    if (payload.get("format_version") != 2 or payload.get("run_id") != job["run_id"] or
            payload.get("config_hash") != config_hash(config) or
            payload.get("source_commit") != job["git_commit"] or
            len(payload.get("streams", [])) != config["training"]["num_envs"]):
        raise ValueError("Variable parallel checkpoint provenance or stream count differs")
    policy.load_state_dict(payload["policy"])
    optimizer.load_state_dict(payload["optimizer"])
    _restore_rng(payload["rng"])
    for name, count in payload["ledger_counts"].items():
        _truncate_lines(directory / name, count)
    payload["streams"] = [_stream_from_record(x) for x in payload["streams"]]
    return payload


def _downstream(scene, layout, config, route, row, *, compact=False):
    """Require measured downstream allocation for a formal terminal result."""
    from .variable_downstream import evaluate_layout

    measured = evaluate_layout(scene, layout, config, route=route)
    beams = measured.get("beams")
    if not isinstance(beams, list) or len(beams) != len(layout.centers):
        raise ValueError("Downstream allocation lacks one result per actual beam")
    for index, (beam, expected) in enumerate(zip(beams, row["beams"])):
        if beam.get("beam_id") != index or not np.isclose(
                beam.get("demand_bps", np.nan), expected["demand_bps"],
                rtol=1e-9, atol=1e-5):
            raise ValueError("Downstream result beam identity or demand differs")
        for name in ("capacity_bps", "carried_bps", "unclipped_capacity_demand_ratio",
                     "si", "sgm"):
            value = beam.get(name)
            if expected["demand_bps"] == 0 and name in (
                    "unclipped_capacity_demand_ratio", "si", "sgm"):
                if value is None:
                    continue
            if value is None or not np.isfinite(value):
                raise ValueError(f"Downstream beam {index} lacks finite {name}")
        demand = float(expected["demand_bps"])
        capacity = float(beam["capacity_bps"])
        carried = float(beam["carried_bps"])
        if capacity < 0 or not np.isclose(carried, min(capacity, demand),
                                         rtol=1e-8, atol=1e-5):
            raise ValueError("Downstream carried rate must be min(C,D)")
        if demand > 0 and not np.isclose(beam["unclipped_capacity_demand_ratio"],
                                         capacity / demand, rtol=1e-8, atol=1e-8):
            raise ValueError("Downstream C/D ratio differs from raw capacity")
    if (not isinstance(measured.get("metrics"), dict) or
            not isinstance(measured.get("timing"), dict) or
            not isinstance(measured.get("calls"), dict)):
        raise ValueError("Downstream result lacks metrics, timing or call ledger")
    if "total_seconds" not in measured["timing"] or not np.isfinite(
            measured["timing"]["total_seconds"]):
        raise ValueError("Downstream timing lacks measured total_seconds")
    row["beams"] = beams
    for index, beam in enumerate(row["beams"]):
        beam["beam_stable_id"] = layout.beam_ids[index]
    if compact:
        service = measured.get("aircraft_service", {})
        for name in ("capacity_bps", "carried_bps", "si", "airtime_fraction"):
            values = service.get(name)
            if not isinstance(values, list) or len(values) != len(row["users"]):
                raise ValueError(f"Training terminal lacks all-aircraft {name}")
            for user, value in zip(row["users"], values):
                user[name] = value
        row["downstream"] = {
            "metrics": measured["metrics"], "timing": measured["timing"],
            "calls": measured["calls"], "physics_version": measured.get("physics_version"),
            "allocator": measured.get("allocator"),
            "aircraft_service_model": service.get("model"),
            "user_tdma_carried_fraction": service.get("demand_weighted_carried_fraction")}
        row["terminal_physics_status"] = "evaluated_all_actual_beams_and_users"
    else:
        row["downstream"] = {key: value for key, value in measured.items() if key != "beams"}
    row["total_seconds"] += float(measured["timing"]["total_seconds"])
    return row


def _single_evaluation(scene, cfg, route, policy, *, config=None, previous=None):
    init_seconds = 0.0
    actions = []
    geometry_calls = 0
    actual_calls = 0
    ray_pairs = 0
    policy_seconds = 0.0
    geometry_seconds = 0.0
    if route in LEARNING_ROUTES:
        if policy is None:
            raise ValueError("Learning route requires a verified model")
        began = time.monotonic()
        layout, info = initial_layout(scene, cfg, route=route)
        init_seconds = time.monotonic() - began
        geometry_calls = int(info.get("geometry_calls", 0))
        actual_calls = geometry_calls
        ray_pairs = int(info.get("actual_ray_pair_checks", 0))
        for step in range(200 if route == "R09" else cfg.max_actions):
            budget = {"geometry_calls": geometry_calls,
                      "max_geometry_calls": cfg.max_geometry_calls}
            began = time.monotonic()
            action, _, index, logp, _, digest = _select(
                scene, layout, cfg, route, policy, steps=step,
                geometry_calls=geometry_calls, deterministic=True,
                budget=budget)
            policy_seconds += time.monotonic() - began
            proposed, event = apply_action(scene, layout, cfg, action, budget=budget)
            geometry_calls = budget["geometry_calls"]
            screen_calls = 1
            actual_calls += screen_calls + int(event.get("actual_geometry_calls", 0))
            ray_pairs += screen_calls * len(scene.demands) * max(len(layout.centers), 1)
            ray_pairs += int(event.get("actual_ray_pair_checks", 0))
            geometry_seconds += float(event.get("elapsed_s", 0))
            actions.append({**event, "candidate_screen_geometry_calls": 1,
                            "cumulative_geometry_calls": geometry_calls,
                            "candidate_index": index,
                            "candidate_identity_sha256": digest, "log_prob": logp})
            layout = proposed
            if action.operation == "stop" and event["accepted"]:
                break
            if geometry_calls >= cfg.max_geometry_calls:
                break
            if route == "R09" and len(layout.centers) >= cfg.max_beams:
                break
    elif route == "R10":
        from .variable_downstream import solve_feedback

        began = time.monotonic()
        layout, info = solve_feedback(scene, cfg, config=config, route=route)
        geometry_seconds = time.monotonic() - began
        actual_calls = int(info.get("actual_geometry_calls", info.get("geometry_calls", 0)))
        geometry_calls = int(info.get("geometry_budget_charges", actual_calls))
        ray_pairs = int(info.get("actual_ray_pair_checks", 0))
        actions = list(info.get("action_log", info.get("actions", [])))
    else:
        # Core routes provide their own deterministic action/call ledger.
        began = time.monotonic()
        layout, info = solve(scene, cfg, route=route, previous=previous)
        geometry_seconds = time.monotonic() - began
        actual_calls = int(info.get("geometry_calls", 0))
        geometry_calls = int(info.get("geometry_budget_charges", actual_calls))
        ray_pairs = int(info.get("actual_ray_pair_checks", 0))
        actions = list(info.get("action_log", []))
    began = time.monotonic()
    result = _layout_row(scene, layout, cfg, route=route, actions=actions,
                         init_seconds=init_seconds,
                         algorithm_seconds=policy_seconds + geometry_seconds,
                         geometry_seconds=geometry_seconds,
                         final_seconds=0.0, calls=geometry_calls,
                         actual_calls=actual_calls, ray_pairs=ray_pairs)
    result["final_check_seconds"] = time.monotonic() - began
    result["total_seconds"] += result["final_check_seconds"]
    result["initialization_info"] = info
    if config is None:
        raise ValueError("Variable evaluation requires frozen downstream config")
    result = _downstream(scene, layout, config, route, result)
    return result


def _evaluate_rows(directory, job, config, *, role, route, policy, output,
                   resume, multiplier_key="evaluation"):
    dataset = _dataset(config, role)
    cfg = _core_config(config)
    multipliers = config[multiplier_key]["demand_multipliers"]
    expected = [(i, float(multiplier)) for i in range(len(dataset.scenes))
                for multiplier in multipliers]
    if route == "R12" and role not in ("development", "test", "sequence"):
        raise ValueError("R12 requires development/test cold starts or frozen sequence")
    if role == "sequence":
        keys = [(scene.metadata.get("sequence_id"),
                 scene.metadata.get("selected_time_utc")) for scene in dataset.scenes]
        if any(not sequence_id or not timestamp for sequence_id, timestamp in keys):
            raise ValueError("Sequence evaluation requires sequence IDs and timestamps")
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("Sequence scenes must be uniquely time ordered")
    existing = _read_lines(output) if resume else []
    if output.exists() and not resume:
        raise FileExistsError("Existing evaluation ledger requires same-ID resume")
    for row, (index, multiplier) in zip(existing, expected):
        scene = dataset.scenes[index]
        if (row["scene_id"] != scene.scene_id or
                row["scene_sha256"] != scene.source_hash or
                row["demand_multiplier"] != multiplier or row["route"] != route):
            raise ValueError("Evaluation prefix differs from frozen scene order")
        if role == "sequence" and (row.get("sequence_id") != scene.metadata["sequence_id"] or
                                   row.get("selected_time_utc") != scene.metadata["selected_time_utc"]):
            raise ValueError("Evaluation prefix differs from frozen sequence identity")
        if route == "R12" and role != "sequence" and row.get("temporal_mode") != "cold_init":
            raise ValueError("Static R12 prefix lacks explicit cold initialization")
    if len(existing) > len(expected):
        raise ValueError("Evaluation has more rows than frozen scenarios")
    _status(directory, job, stage="evaluating", evaluated_scenes=len(existing),
            evaluation_total=len(expected))
    previous_by_key = {}
    if route == "R12" and role == "sequence":
        for row in existing:
            previous_by_key[(row["sequence_id"], row["demand_multiplier"])] = \
                Layout.from_record(row["layout"])
    for index, multiplier in expected[len(existing):]:
        if _stop_requested(directory):
            break
        scene = _scaled(dataset.scenes[index], multiplier)
        sequence_id = scene.metadata.get("sequence_id")
        previous = (previous_by_key.get((sequence_id, multiplier))
                    if route == "R12" and role == "sequence" else None)
        row = _single_evaluation(scene, cfg, route, policy, config=config,
                                 previous=previous)
        if route == "R12":
            row["temporal_mode"] = ("cold_init" if role != "sequence" else
                                    "sequence_first_frame" if previous is None else
                                    "sequence_warm_start")
            if role == "sequence":
                row["sequence_id"] = sequence_id
                row["selected_time_utc"] = scene.metadata["selected_time_utc"]
                previous_by_key[(sequence_id, multiplier)] = Layout.from_record(row["layout"])
        elif role == "sequence":
            row["temporal_mode"] = "independent_scene"
            row["sequence_id"] = sequence_id
            row["selected_time_utc"] = scene.metadata["selected_time_utc"]
        _jsonl(output, row)
        existing.append(row)
        _status(directory, job, stage="evaluating", evaluated_scenes=len(existing),
                evaluation_total=len(expected))
    return existing, len(expected)


def _load_model(job, config, device):
    directory = Path(job["model_path"])
    metadata = _read(directory / "model.json")
    if (not isinstance(metadata, dict) or metadata.get("policy_format") != POLICY_FORMAT or
            metadata.get("schema_version") != 5 or
            metadata.get("route") != _route(config) or
            metadata.get("network") != config["network"] or
            metadata.get("variable") != config["coverage"]["variable"] or
            metadata.get("experiment_role") != "formal" or
            metadata.get("selection") != "fixed_budget_endpoint" or
            metadata.get("env_step") != 262144 or
            _sha256(directory / "policy.pt") != metadata.get("policy_sha256")):
        raise ValueError("Variable PPO model metadata, endpoint or SHA differs")
    policy = _actor(config, device)
    policy.load_state_dict(_load_torch(directory / "policy.pt", device), strict=True)
    policy.eval()
    return policy, metadata


def _parallel_validation_if_due(directory, job, config, policy, optimizer,
                                streams, *, resume, step, updates, episode,
                                last_validation, ledgers):
    verification = job.get("experiment_role", "formal") == "verification"
    interval = config["training"]["total_steps"] if verification else config["training"]["validation_interval"]
    role = "development" if verification else "validation"
    if (step == 0 or step == last_validation or step % interval or
            role not in config["datasets"]):
        return last_validation, False
    before = _rng_state()
    policy.eval()
    try:
        rows, expected = _evaluate_rows(directory, job, config,
                                        role=role, route=_route(config),
                                        policy=policy,
                                        output=directory / f"validation_{step}.jsonl",
                                        resume=resume)
    finally:
        policy.train()
        _restore_rng(before)
    if expected != 60:
        raise ValueError("Frozen variable validation must contain 20 mothers × 3 demands")
    if len(rows) != expected:
        _status(directory, job, stage="stopped", env_step=step,
                stop_reason="validation_interrupted")
        return last_validation, True
    allocation_calls = sum(int(row["downstream"]["calls"]["layout_allocation_calls"])
                           for row in rows)
    if allocation_calls != expected or any(
            any(beam.get("capacity_bps") is None or beam.get("carried_bps") is None or
                (beam["demand_bps"] > 0 and
                 (beam.get("unclipped_capacity_demand_ratio") is None or
                  beam.get("si") is None or beam.get("sgm") is None))
                for beam in row["beams"]) for row in rows):
        raise ValueError("Variable validation lacks one complete terminal allocation per scene")
    _jsonl(directory / "evaluations.jsonl",
           {"env_step": step, "dataset_role": role, "scene_count": len(rows),
            "downstream_allocation_calls": allocation_calls,
            "summary": _summary(rows),
            "rows_sha256": _sha256(directory / f"validation_{step}.jsonl")})
    ledgers["evaluations.jsonl"] += 1
    last_validation = step
    sha = _checkpoint_parallel(directory, job, config, policy, optimizer,
                               streams=streams, step=step, updates=updates,
                               episode=episode, last_validation=last_validation,
                               ledger_counts=ledgers)
    _status(directory, job, stage="training", env_step=step,
            last_validation_step=step, checkpoint_sha256=sha)
    return last_validation, False


def _train(directory, job, config, resume, preflight):
    """Thirty-two independent CPU streams with one batched CUDA actor update."""
    route = _route(config)
    if route not in LEARNING_ROUTES:
        raise ValueError("Only R08/R09 train a variable-layout policy")
    training = _check_training(config, job)
    cfg = _core_config(config)
    dataset = _dataset(config, "train")
    validation_role = ("development" if job.get("experiment_role") == "verification"
                       else "validation")
    validation_dataset = _dataset(config, validation_role)
    if len(validation_dataset.scenes) != 20:
        raise ValueError("Frozen variable validation requires exactly 20 mother scenes")
    multipliers = training["demand_multipliers"]
    device = training["device"]
    policy = _actor(config, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=training["lr"])
    count = training["num_envs"]
    ledgers = {name: 0 for name in ("actions.jsonl", "episodes.jsonl",
                                     "updates.jsonl", "evaluations.jsonl")}
    if resume:
        saved = _load_parallel(directory, job, config, policy, optimizer)
        streams = saved["streams"]
        step, updates, episode = saved["step"], saved["updates"], saved["episode"]
        last_validation, ledgers = saved["last_validation"], saved["ledger_counts"]
    else:
        if (directory / "checkpoint.pt").exists() or any(
                (directory / name).exists() for name in ledgers):
            raise FileExistsError("Existing variable training ledger requires same-ID resume")
        _seed(training["seed"])
        streams = [_fresh_stream(dataset, cfg, route,
                                 training["seed"] + 101 + i * 1009, multipliers)
                   for i in range(count)]
        step = updates = episode = last_validation = 0
    _status(directory, job, stage="training", env_step=step,
            budget_env_steps=training["total_steps"], update_step=updates,
            episode=episode, num_envs=count)
    cpu_workers = min(8, count)
    dataset_record = config["datasets"].get("train", {})
    if device == "cuda":
        executor = ProcessPoolExecutor(
            max_workers=cpu_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_cpu_worker_init,
            initargs=(dataset_record, config, route, multipliers))
        cpu_backend = "spawn_process"
    else:
        # Isolated CPU correctness tests use threads; formal workbench tasks
        # have already passed the CUDA gate and always choose spawn above.
        _cpu_worker_init(dataset_record, config, route, multipliers)
        executor = ThreadPoolExecutor(max_workers=cpu_workers)
        cpu_backend = "thread_local_verification"
    with executor as pool:
        while step < training["total_steps"]:
            last_validation, interrupted = _parallel_validation_if_due(
                directory, job, config, policy, optimizer, streams,
                resume=resume, step=step, updates=updates, episode=episode,
                last_validation=last_validation, ledgers=ledgers)
            if interrupted:
                return
            transitions = []
            action_rows, episode_rows = [], []
            started = time.monotonic()
            for _ in range(training["rollout_steps"]):
                packets = [_cpu_packet(stream) for stream in streams]
                observed = list(pool.map(_cpu_observe, packets))
                obs_rows, menus, before_rows, screened_calls, observation_audits = map(
                    list, zip(*observed))
                selection_began = time.monotonic()
                with torch.no_grad():
                    choices, logps, values = policy.act(collate(obs_rows, device))
                selection_seconds = (time.monotonic() - selection_began) / count
                indices = choices.cpu().tolist()
                logp_values = logps.cpu().tolist()
                values_values = values.cpu().tolist()
                actions = [menus[i][indices[i]] for i in range(count)]
                results = list(pool.map(_cpu_execute, [(packets[i], actions[i],
                                                       screened_calls[i])
                                                  for i in range(count)]))
                completed = []
                for i, (next_record, info, after, calls, execution_audit) in enumerate(results):
                    stream = streams[i]
                    next_layout = Layout.from_record(next_record)
                    prior_calls = stream["geometry_calls"]
                    screen_calls = screened_calls[i] - prior_calls
                    stream["episode_steps"] += 1
                    stream["geometry_calls"] = calls
                    stream["actual_geometry_calls"] += (observation_audits[i]["matrix_calls"] +
                                                        execution_audit["matrix_calls"])
                    stream["actual_ray_pair_checks"] += (observation_audits[i]["ray_pair_checks"] +
                                                         execution_audit["ray_pair_checks"])
                    limit = 200 if route == "R09" else cfg.max_actions
                    done = (actions[i]["operation"] == "stop" and info["accepted"] or
                            stream["episode_steps"] >= limit or
                            calls >= cfg.max_geometry_calls or
                            (route == "R09" and len(next_layout.centers) >= cfg.max_beams))
                    reward = _quality(after) - _quality(before_rows[i])
                    reward -= .005 * int(info["geometry_calls"])
                    if not info["accepted"]:
                        reward -= .02
                    if done:
                        reward += 2.0 if after["coverage_100"] else -2.0
                    transitions.append(Transition(obs_rows[i], indices[i],
                                                  logp_values[i], values_values[i],
                                                  reward, done))
                    event = {"env_step": step + len(transitions), "stream_id": i,
                             "scene_index": stream["scene_index"],
                             "scene_id": dataset.scenes[stream["scene_index"]].scene_id,
                             "scene_sha256": dataset.scenes[stream["scene_index"]].source_hash,
                             "demand_multiplier": float(multipliers[stream["level"]]),
                             "stream_episode_serial": stream["episode_serial"] + 1,
                             "candidate_index": indices[i],
                             "candidate_identity_sha256": candidate_digest(
                                 [(VariableAction.from_record(action), None)
                                  for action in menus[i]]),
                             "choice_log_prob": logp_values[i],
                             "value": values_values[i], "reward": reward,
                             "requested_action": actions[i],
                             "actual_action": actions[i] if info["accepted"] else None,
                             "before_metrics": before_rows[i], "after_metrics": after,
                             "candidate_screen_geometry_calls":
                             screen_calls,
                             "observation_actual_geometry_calls": observation_audits[i]["matrix_calls"],
                             "observation_actual_ray_pair_checks": observation_audits[i]["ray_pair_checks"],
                             "execution_actual_geometry_calls": execution_audit["matrix_calls"],
                             "execution_actual_ray_pair_checks": execution_audit["ray_pair_checks"],
                             "cumulative_geometry_calls": calls,
                             "cumulative_actual_geometry_calls": stream["actual_geometry_calls"],
                             "cumulative_actual_ray_pair_checks": stream["actual_ray_pair_checks"],
                             "selection_seconds": selection_seconds, **info}
                    stream["actions"].append(event)
                    action_rows.append(event)
                    stream["episode_return"] += reward
                    stream["algorithm_seconds"] += selection_seconds + float(info["elapsed_s"])
                    stream["geometry_seconds"] += float(info["elapsed_s"])
                    stream["layout"] = next_layout
                    if done:
                        completed.append((i, calls,
                                          step + len(transitions),
                                          "accepted_stop" if actions[i]["operation"] == "stop" and info["accepted"]
                                          else "step_or_geometry_budget"))
                terminal_packets = [_cpu_terminal_packet(streams[i], i)
                                    for i, _, _, _ in completed]
                for terminal, row in zip(completed, pool.map(_cpu_terminal, terminal_packets)):
                    i, calls, terminal_step, stop_reason = terminal
                    stream = streams[i]
                    episode += 1
                    row.update(episode=episode,
                               stream_episode_serial=stream["episode_serial"] + 1,
                               terminal_env_step=terminal_step,
                               episode_return=stream["episode_return"],
                               training_seed=training["seed"],
                               stop_reason=stop_reason,
                               completed_episode=True)
                    episode_rows.append(row)
                    stream["episode_serial"] += 1
                starts = [(int(streams[i]["sampler"].integers(len(dataset.scenes))),
                           int(streams[i]["sampler"].integers(len(multipliers))))
                          for i, _, _, _ in completed]
                initial_results = list(pool.map(_cpu_initial_layout, starts))
                for (i, _, _, _), (index, level), (fresh_record, init, init_elapsed) in zip(
                        completed, starts, initial_results):
                    stream = streams[i]
                    fresh = Layout.from_record(fresh_record)
                    stream.update(scene_index=index, level=level, layout=fresh,
                                  actions=[], episode_return=0.0,
                                  episode_steps=0,
                                  geometry_calls=int(init.get("geometry_calls", 0)),
                                  actual_geometry_calls=int(init.get("geometry_calls", 0)),
                                  actual_ray_pair_checks=int(init.get("actual_ray_pair_checks", 0)),
                                  init_seconds=init_elapsed,
                                  algorithm_seconds=0.0, geometry_seconds=0.0)
            bootstrap_rows = list(pool.map(_cpu_bootstrap,
                                           [_cpu_packet(stream) for stream in streams]))
            bootstrap_obs = []
            for stream, (obs, audit) in zip(streams, bootstrap_rows):
                bootstrap_obs.append(obs)
                stream["actual_geometry_calls"] += audit["matrix_calls"]
                stream["actual_ray_pair_checks"] += audit["ray_pair_checks"]
            with torch.no_grad():
                bootstrap = policy(collate(bootstrap_obs, device))[1].cpu().numpy()
            diagnostics = ppo_update(policy, optimizer, transitions, bootstrap,
                                     device=device, gamma=training["gamma"],
                                     gae_lambda=training["gae_lambda"],
                                     clip_ratio=training["clip_ratio"],
                                     entropy_coef=training["entropy_coef"],
                                     value_coef=training["value_coef"],
                                     max_grad_norm=training["max_grad_norm"],
                                     epochs=training["epochs"],
                                     minibatch_size=training["minibatch_size"],
                                     num_envs=count)
            step += len(transitions)
            updates += diagnostics["minibatch_updates"]
            for row in action_rows:
                _jsonl(directory / "actions.jsonl", row)
            for row in episode_rows:
                _jsonl(directory / "episodes.jsonl", row)
            _jsonl(directory / "updates.jsonl", {"env_step": step,
                                                "update_step": updates,
                                                "episode": episode,
                                                "rollout_seconds": time.monotonic() - started,
                                                "num_envs": count, **diagnostics})
            ledgers["actions.jsonl"] += len(action_rows)
            ledgers["episodes.jsonl"] += len(episode_rows)
            ledgers["updates.jsonl"] += 1
            sha = _checkpoint_parallel(directory, job, config, policy, optimizer,
                                       streams=streams, step=step, updates=updates,
                                       episode=episode, last_validation=last_validation,
                                       ledger_counts=ledgers)
            _status(directory, job, stage="training", env_step=step,
                    budget_env_steps=training["total_steps"], update_step=updates,
                    episode=episode, checkpoint_sha256=sha,
                    cpu_worker_backend=cpu_backend, cpu_workers=cpu_workers,
                    latest_update=diagnostics,
                    effective_steps_per_second=len(transitions) /
                    max(time.monotonic() - started, 1e-9))
            if _stop_requested(directory):
                _status(directory, job, stage="stopped", env_step=step,
                        stop_reason="requested", checkpoint_sha256=sha)
                return
    last_validation, interrupted = _parallel_validation_if_due(
        directory, job, config, policy, optimizer, streams,
        resume=resume, step=step, updates=updates, episode=episode,
        last_validation=last_validation, ledgers=ledgers)
    if interrupted:
        return
    for i, stream in enumerate(streams):
        if not stream["actions"]:
            continue
        if any(row.get("terminal_env_step") == step and row.get("stream_id") == i and
               row.get("stop_reason") == "budget_truncated"
               for row in _read_lines(directory / "episodes.jsonl")):
            continue
        scene = _stream_scene(stream, dataset, multipliers)
        row = _layout_row(scene, stream["layout"], cfg, route=route,
                          actions=[],
                          init_seconds=stream["init_seconds"],
                          algorithm_seconds=stream["algorithm_seconds"],
                          geometry_seconds=stream["geometry_seconds"],
                          final_seconds=0.0, calls=stream["geometry_calls"],
                          actual_calls=stream["actual_geometry_calls"],
                          ray_pairs=stream["actual_ray_pair_checks"])
        row["macro_steps"] = len(stream["actions"])
        row.pop("actions")
        row["action_trace"] = {"file": "actions.jsonl", "stream_id": i,
                               "stream_episode_serial": stream["episode_serial"] + 1,
                               "first_env_step": stream["actions"][0]["env_step"],
                               "last_env_step": stream["actions"][-1]["env_step"],
                               "count": len(stream["actions"])}
        row["terminal_physics_status"] = "not_run_incomplete_budget_tail"
        row.update(episode=episode + i + 1, stream_id=i,
                   stream_episode_serial=stream["episode_serial"] + 1,
                   terminal_env_step=step,
                   episode_return=stream["episode_return"],
                   training_seed=training["seed"],
                   stop_reason="budget_truncated", completed_episode=False)
        _jsonl(directory / "episodes.jsonl", row)
        ledgers["episodes.jsonl"] += 1
    _checkpoint_parallel(directory, job, config, policy, optimizer,
                         streams=streams, step=step, updates=updates,
                         episode=episode, last_validation=last_validation,
                         ledger_counts=ledgers)
    final = Path(job["model_dir"]) / "final"
    final.mkdir(parents=True, exist_ok=True)
    _save_torch(final / "policy.pt", policy.state_dict())
    metadata = {"policy_format": POLICY_FORMAT, "schema_version": 5,
                "research_stage": "coverage", "run_id": job["run_id"],
                "route": route, "network": config["network"],
                "variable": config["coverage"]["variable"],
                "training_seed": training["seed"], "env_step": step,
                "selection": ("verification_endpoint" if job.get("experiment_role") == "verification"
                              else "fixed_budget_endpoint"),
                "experiment_role": job.get("experiment_role", "formal"),
                "source_commit": job["git_commit"],
                "config_hash": config_hash(config), "datasets": config["datasets"],
                "policy_sha256": _sha256(final / "policy.pt"),
                "parameter_count": sum(p.numel() for p in policy.parameters()),
                "num_envs": count, "cpu_workers": cpu_workers,
                "cpu_worker_backend": cpu_backend,
                "validation_scene_count_per_node": 60,
                "validation_node_budget": (1 if job.get("experiment_role") == "verification"
                                           else 16),
                "validation_downstream_allocation_call_budget": (
                    60 if job.get("experiment_role") == "verification" else 960),
                "validation_downstream_allocation_calls": ledgers["evaluations.jsonl"] * 60,
                "training_terminal_downstream_allocation_calls": episode,
                "python_prefix": sys.prefix, "gpu": preflight.get("gpu"),
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "created_at": _now()}
    _write_json(final / "model.json", metadata)
    _write_json(directory / "artifacts.json", {"run_id": job["run_id"],
                "checkpoint": str(directory / "checkpoint.pt"),
                "final_policy": str(final), "model": str(final / "policy.pt"),
                "model_metadata": str(final / "model.json"),
                "model_sha256": metadata["policy_sha256"],
                "actions": str(directory / "actions.jsonl"),
                "episodes": str(directory / "episodes.jsonl"),
                "updates": str(directory / "updates.jsonl"),
                "evaluations": str(directory / "evaluations.jsonl")})
    _status(directory, job, stage="completed", env_step=step,
            budget_env_steps=training["total_steps"], update_step=updates,
            episode=episode, final_policy=str(final))
    _event(directory, job, "variable_training_completed", env_step=step)


def _evaluate(directory, job, config, resume):
    route = _route(config)
    policy = None
    if route in LEARNING_ROUTES:
        if config["training"]["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Formal variable policy evaluation requires CUDA")
        policy, _ = _load_model(job, config, "cuda")
    role = next(iter(config["datasets"]))
    rows, expected = _evaluate_rows(directory, job, config, role=role, route=route,
                                    policy=policy,
                                    output=directory / "results.partial.jsonl",
                                    resume=resume)
    if len(rows) < expected:
        _status(directory, job, stage="stopped", evaluated_scenes=len(rows),
                evaluation_total=expected)
        return
    summary = _summary(rows)
    raw = directory / "results.partial.jsonl"
    result = {"run_id": job["run_id"], "route": route, "scene_count": len(rows),
              "summary": summary, "rows_path": str(raw),
              "rows_sha256": _sha256(raw), "source_commit": job["git_commit"],
              "config_hash": config_hash(config), "created_at": _now()}
    _write_json(directory / "results.json", result)
    _write_json(directory / "artifacts.json", {"run_id": job["run_id"],
                "results": str(directory / "results.json"), "raw_results": str(raw),
                "raw_results_sha256": result["rows_sha256"]})
    _status(directory, job, stage="completed", evaluated_scenes=len(rows),
            evaluation_total=expected, result_path=str(directory / "results.json"))
    _event(directory, job, "variable_evaluation_completed", scenes=len(rows))


def _prepare(directory, job, config, resume):
    route = _route(config)
    if route in LEARNING_ROUTES:
        raise ValueError("Learning starts are created in the frozen train task")
    role = next(iter(config["datasets"]))
    dataset = _dataset(config, role)
    cfg = _core_config(config)
    path = directory / "starts.partial.jsonl"
    rows = _read_lines(path) if resume else []
    if path.exists() and not resume:
        raise FileExistsError("Existing starts ledger requires same-ID resume")
    for index, row in enumerate(rows):
        if (row["scene_id"] != dataset.scenes[index].scene_id or
                row["scene_sha256"] != dataset.scenes[index].source_hash):
            raise ValueError("Prepared prefix differs from frozen dataset")
    for scene in dataset.scenes[len(rows):]:
        if _stop_requested(directory):
            _status(directory, job, stage="stopped", prepared_scenes=len(rows))
            return
        began = time.monotonic()
        layout, info = initial_layout(scene, cfg, route=route)
        row = {"scene_id": scene.scene_id, "scene_sha256": scene.source_hash,
               "route": route, "layout": layout.to_record(), "info": info,
               "metrics": metrics(scene, layout, cfg),
               "initialization_seconds": time.monotonic() - began}
        _jsonl(path, row)
        rows.append(row)
        _status(directory, job, stage="preparing", prepared_scenes=len(rows),
                preparation_total=len(dataset.scenes))
    _write_json(directory / "artifacts.json", {"run_id": job["run_id"],
                "starts": str(path), "starts_sha256": _sha256(path)})
    _status(directory, job, stage="completed", prepared_scenes=len(rows),
            preparation_total=len(dataset.scenes))


def run_variable_job(directory, job, config, *, resume=False, preflight=None):
    """Dispatch one frozen schema-5 job after workbench CUDA preflight."""
    directory = Path(directory)
    if config.get("schema_version") != 5 or config.get("research_stage") != "coverage":
        raise ValueError("Variable runner requires coverage schema 5")
    if preflight is None:
        raise ValueError("Formal variable workbench job requires verified CUDA preflight")
    route = _route(config)
    _status(directory, job, stage="initializing", preflight=preflight)
    _event(directory, job, "variable_preflight_verified", preflight=preflight)
    if job["kind"] == "train":
        _train(directory, job, config, resume, preflight)
    elif job["kind"] == "evaluate":
        _evaluate(directory, job, config, resume)
    elif job["kind"] == "prepare":
        _prepare(directory, job, config, resume)
    else:
        raise ValueError("Unknown variable workbench task kind")
