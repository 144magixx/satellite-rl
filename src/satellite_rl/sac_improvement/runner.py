"""Persistent workbench entry for S2-P19 I0 gates, training and evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from satellite_rl.workbench.datasets import assert_scene_profile, scene_set_files
from satellite_rl.workbench.scene_order import VariableBeamSceneOrder

from .agent import ImprovementSAC, MAX_ACTIONS, pack_actions
from .config import validate_config
from .environment import ImprovementEnv, STATE_WIDTH, run_baseline


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False,
                               indent=2), encoding="utf-8")
    os.replace(temp, path)


def _jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _artifacts(directory, job, **extras):
    path = directory / "artifacts.json"
    value = _read(path, {})
    value.update({"run_id": job["run_id"], "config": str(directory / "config.json"),
                  "source_root": job.get("source_root"), **extras})
    _atomic_json(path, value)


def _load_scene(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def _scene_set(job, name):
    record = (job.get("scene_sets") or {}).get(name)
    if record is None:
        raise ValueError(f"I0 task lacks frozen {name} scene set")
    assert_scene_profile(record, "europe_v1")
    files = scene_set_files(record)
    if len(files) != len(record["files"]):
        raise ValueError("I0 scene metadata and files differ")
    return record, files


def _cuda_preflight(job, config):
    # Reuse the workbench's strict DQN, CUDA, source manifest and import-path gate.
    from satellite_rl.workbench.runner import strict_cuda_preflight
    if config["device"] != "cuda":
        raise ValueError("Persistent I0 workbench jobs require CUDA")
    return strict_cuda_preflight(job)


def _config(job, frozen_config):
    raw = (job.get("i0_config") or frozen_config.get("i0")
           or frozen_config.get("improvement"))
    if not isinstance(raw, dict):
        raise ValueError("Frozen I0 controls absent; expected job.i0_config or config.i0")
    config = validate_config(raw)
    # If both are present, a job cannot override the immutable file.
    if job.get("i0_config") is not None and frozen_config.get("i0") is not None \
            and job["i0_config"] != frozen_config["i0"]:
        raise ValueError("I0 job and frozen config controls differ")
    return config


def _status(directory, job, config, stage, *, step=0, episodes=0, **extra):
    row = {"run_id": job["run_id"], "stage": stage, "env_step": step,
           "budget_env_steps": config["total_steps"], "episodes": episodes,
           "budget_episodes": config["max_episodes"], "heartbeat_at": _now(),
           "pid": os.getpid(), "algorithm": "s2p19_i0_candidate_sac", **extra}
    _atomic_json(directory / "status.json", row)
    return row


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _load_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint(directory, job, config, frozen_config_sha, agent, steps, episodes,
                scene_index, environment, pilot_gate_passed=False):
    state = {"version": 1, "run_id": job["run_id"],
             "frozen_config_sha256": frozen_config_sha,
             "i0_config": config, "env_step": steps, "episodes": episodes,
             "scene_index": scene_index,
             "pilot_gate_passed": bool(pilot_gate_passed),
             "environment": environment.checkpoint_state() if environment is not None else None,
             "agent": agent.state_dict(), "rng": _rng_state()}
    path = directory / "checkpoint.pt"
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(state, temp)
    os.replace(temp, path)
    return _sha(path)


def _read_checkpoint(directory, job, config, frozen_config_sha):
    path = directory / "checkpoint.pt"
    if not path.is_file():
        raise ValueError("I0 resume lacks complete checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.get("version") != 1 or state.get("run_id") != job["run_id"]
            or state.get("frozen_config_sha256") != frozen_config_sha
            or state.get("i0_config") != config):
        raise ValueError("I0 checkpoint identity/config/source differs")
    return state, _sha(path)


def _gate_on_resume(directory, job, previous_status, checkpoint_sha, saved, config):
    if saved.get("pilot_gate_passed") is True or previous_status.get("pilot_gate_passed") is True:
        return True
    if previous_status.get("stop_reason") == "user_stop_during_validation":
        # Permit only finishing the already frozen node; _train_job then
        # transitions to pilot_gate_pending before another training action.
        return False
    requires_gate = (previous_status.get("stop_reason") == "s2p19_pilot_gate_pending"
                     or saved["env_step"] >= config["pilot_stop_steps"]
                     or saved["episodes"] >= config["pilot_max_episodes"])
    if not requires_gate:
        return False
    gate = _read(directory / "pilot_gate.json", {})
    if (gate.get("run_id") != job["run_id"] or
            gate.get("checkpoint_sha256") != checkpoint_sha or
            gate.get("pass") is not True):
        raise ValueError("I0 pilot gate pass/run/checkpoint SHA is missing or differs")
    return True


def _episode_result(env, scene_record, source_sha, seed, method):
    result = env.output()
    result.update(scene_sha256=source_sha, scene_metadata=scene_record,
                  seed=seed, action_seed=seed, method=method, completed_at=_now())
    return result


def _case(directory, case_id, result, repeat_index=0):
    path = directory / "cases" / f"{case_id}.json"
    if path.exists():
        raise ValueError(f"I0 case already exists: {case_id}; load it for idempotent resume")
    _atomic_json(path, result)
    return {"case_id": case_id, "case_path": str(path), "case_sha256": _sha(path),
            "repeat_index": repeat_index,
            "action_seed": result["action_seed"],
            "method": result["method"], "scene_sha256": result["scene_sha256"],
            "metrics": result["metrics"], "beam_count": len(result["beams"])}


def _saved_case(directory, case_id, *, scene_sha, method, repeat_index=0,
                action_seed=None, model_sha=None):
    path = directory / "cases" / f"{case_id}.json"
    if not path.exists():
        return None
    row = _read(path)
    if (row.get("scene_sha256") != scene_sha or row.get("method") != method
            or row.get("repeat_index", 0) != repeat_index
            or (action_seed is not None and row.get("action_seed") != action_seed)
            or (model_sha is not None and row.get("model_sha256") != model_sha)):
        raise ValueError(f"I0 existing case identity differs: {case_id}")
    pending_path = directory / "case_attempt.json"
    pending = _read(pending_path)
    if pending is not None and pending.get("case_id") == case_id:
        _jsonl(directory / "interrupted_attempts.jsonl", {
            "case_id": case_id, "scene_sha256": scene_sha,
            "started_at": pending.get("started_at"), "detected_at": _now(),
            "exact_elapsed_seconds": None,
            "reason": "case saved before interruption; timing receipt missing"})
        pending_path.unlink(missing_ok=True)
    return {"case_id": case_id, "case_path": str(path), "case_sha256": _sha(path),
            "repeat_index": repeat_index,
            "action_seed": row["action_seed"],
            "method": method, "scene_sha256": scene_sha,
            "metrics": row["metrics"], "beam_count": len(row["beams"])}


def _attempt_start(directory, case_id, scene_sha):
    path = directory / "case_attempt.json"
    prior = _read(path)
    if prior is not None:
        _jsonl(directory / "interrupted_attempts.jsonl", {
            "case_id": prior.get("case_id"), "scene_sha256": prior.get("scene_sha256"),
            "started_at": prior.get("started_at"), "detected_at": _now(),
            "exact_elapsed_seconds": None,
            "reason": "previous process left an incomplete case; cost cannot be recovered exactly"})
    _atomic_json(path, {"case_id": case_id, "scene_sha256": scene_sha,
                        "started_at": _now(), "pid": os.getpid()})
    return time.perf_counter()


def _attempt_complete(directory, case_id, started):
    _jsonl(directory / "case_attempts.jsonl", {"case_id": case_id,
                                                 "elapsed_seconds": time.perf_counter() - started,
                                                 "completed_at": _now()})
    (directory / "case_attempt.json").unlink(missing_ok=True)


def _baseline_job(directory, job, config, mode):
    role = job.get("partition", "development" if mode == "gate" else "test")
    if mode == "gate":
        if role != "development":
            raise ValueError("I0 mechanism gate requires development scenes")
        methods = ("greedy", "rule", "random")
        repeats_by_method = {name: 1 for name in methods}
        if job.get("methods") not in (None, list(methods)) or job.get("method_repeats") not in (
                None, repeats_by_method):
            raise ValueError("I0 gate methods/repeats differ")
    elif mode == "baselines":
        if role == "validation":
            methods = ("greedy",)
            repeats_by_method = {"greedy": 1}
        elif role == "test":
            methods = ("greedy", "rule", "random")
            repeats_by_method = {"greedy": 1, "rule": 1, "random": 3}
        else:
            raise ValueError("I0 final baselines require validation or test scenes")
        if job.get("methods") != list(methods) or job.get("method_repeats") != repeats_by_method:
            raise ValueError("I0 final methods/repeats differ from frozen 1/1/3 contract")
    else:
        raise ValueError("Unknown I0 baseline task mode")
    action_seed_base = job.get("action_seed")
    if type(action_seed_base) is not int or action_seed_base != 53001:
        raise ValueError("I0 baseline action seed must be 53001")
    record, files = _scene_set(job, role)
    expected_scenes = 880 if role == "test" else 72
    if len(files) != expected_scenes:
        raise ValueError(f"I0 {role} needs {expected_scenes} frozen scenes")
    total_cases = len(files) * sum(repeats_by_method.values())
    rows = []
    for index, (path, digest) in enumerate(files):
        metadata = record["files"][index]
        for method in methods:
            for repeat in range(repeats_by_method[method]):
                action_seed = action_seed_base + repeat
                case_id = f"scene_{index:05d}_{method}_r{repeat}"
                old = _saved_case(directory, case_id, scene_sha=digest, method=method,
                                  repeat_index=repeat, action_seed=action_seed)
                if old is not None:
                    rows.append(old)
                    continue
                if (directory / "stop.request").exists():
                    _status(directory, job, config, "stopped", step=len(rows),
                            budget_env_steps=total_cases, stop_reason="user_stop",
                            completed_cases=len(rows), total_cases=total_cases)
                    return
                started = _attempt_start(directory, case_id, digest)
                result = run_baseline(_load_scene(path), method, action_seed, config)
                result.update(scene_sha256=digest, scene_metadata=metadata,
                              completed_at=_now(), repeat_index=repeat)
                rows.append(_case(directory, case_id, result, repeat_index=repeat))
                _attempt_complete(directory, case_id, started)
        _status(directory, job, config, "evaluating", step=len(rows),
                budget_env_steps=total_cases, completed_cases=len(rows),
                total_cases=total_cases)
    per_scene = {}
    arms = {method: [] for method in methods}
    for row in rows:
        per_scene.setdefault(row["scene_sha256"], {}).setdefault(row["method"], []).append(row)
        arms[row["method"]].append(row)
    _atomic_json(directory / "results.json", {"mode": mode, "scenes": rows,
                                               "per_scene": per_scene, "arms": arms,
                                               "methods": list(methods),
                                               "method_repeats": repeats_by_method,
                                               "partition": role,
                                               "action_seed_base": action_seed_base,
                                               "source_set_id": record.get("id"),
                                               "completed_at": _now()})
    _artifacts(directory, job, results=str(directory / "results.json"),
               case_directory=str(directory / "cases"))
    _status(directory, job, config, "completed", step=len(rows),
            budget_env_steps=total_cases,
            completed_cases=len(rows), total_cases=len(rows),
            results_sha256=_sha(directory / "results.json"))


def _gate_job(directory, job, config):
    return _baseline_job(directory, job, config, "gate")


def _model_episode(agent, scene, config, seed, *, deterministic=True):
    started = time.perf_counter()
    env = ImprovementEnv(config)
    env.reset(scene, seed=seed)
    while not env.done:
        features, mask, identities = pack_actions(env)
        action = agent.choose(env.state(), features, mask, deterministic=deterministic)
        env.step(identities[action])
    env.total_seconds = time.perf_counter() - started
    return env


VALIDATION_NODES = (0, 50000, 100000, 200000, 300000, 400000)


def _validate_node(directory, job, config, agent, step, checkpoint_sha, validation):
    record, files = validation
    node_dir = directory / "validation" / f"step_{step:09d}"
    node_dir.mkdir(parents=True, exist_ok=True)
    summary_path = node_dir / "summary.json"
    previous = _read(summary_path)
    if previous is not None:
        if previous.get("checkpoint_sha256") != checkpoint_sha or previous.get("count") != len(files):
            raise ValueError("I0 fixed validation node provenance differs")
        return True
    saved_rng = _rng_state()
    actor_was_training = agent.actor.training
    agent.actor.eval()
    rows = []
    try:
        for index, (path, digest) in enumerate(files):
            case_id = f"step_{step:09d}_scene_{index:05d}"
            case_path = node_dir / f"scene_{index:05d}.json"
            prior = _read(case_path)
            if prior is not None:
                if (prior.get("scene_sha256") != digest or
                        prior.get("checkpoint_sha256") != checkpoint_sha):
                    raise ValueError("I0 validation case/checkpoint source differs")
                pending_path = directory / "case_attempt.json"
                pending = _read(pending_path)
                if pending is not None and pending.get("case_id") == case_id:
                    _jsonl(directory / "interrupted_attempts.jsonl", {
                        "case_id": case_id, "scene_sha256": digest,
                        "started_at": pending.get("started_at"), "detected_at": _now(),
                        "exact_elapsed_seconds": None,
                        "reason": "validation case saved before interruption; timing receipt missing"})
                    pending_path.unlink(missing_ok=True)
                rows.append(prior)
                continue
            if (directory / "stop.request").exists():
                return False
            seed = int(job.get("action_seed", 53001))
            _seed(seed)
            started = _attempt_start(directory, case_id, digest)
            env = _model_episode(agent, _load_scene(path), config, seed, deterministic=False)
            result = _episode_result(env, record["files"][index], digest, seed, "I0_validation")
            result.update(checkpoint_sha256=checkpoint_sha, env_step=step, repeat_index=0)
            result["metrics"]["total_seconds"] = env.total_seconds
            _atomic_json(case_path, result)
            _attempt_complete(directory, case_id, started)
            rows.append(result)
    finally:
        agent.actor.train(actor_was_training)
        _load_rng(saved_rng)
    summary = {"env_step": step, "checkpoint_sha256": checkpoint_sha,
               "source_set_id": record.get("id"), "count": len(rows),
               "mean_si": float(np.mean([r["metrics"]["si"] for r in rows])),
               "mean_sgm": float(np.mean([r["metrics"]["sgm"] for r in rows])),
               "mean_si_delta_vs_b1": float(np.mean([
                   r["metrics"]["si"] - r["metrics"]["initial_si"] for r in rows])),
               "cases": [{"path": str(node_dir / f"scene_{i:05d}.json"),
                          "sha256": _sha(node_dir / f"scene_{i:05d}.json")}
                         for i in range(len(rows))], "completed_at": _now()}
    _atomic_json(summary_path, summary)
    # Rebuild from immutable node summaries, avoiding duplicates after a crash.
    summaries = [_read(directory / "validation" / f"step_{node:09d}" / "summary.json")
                 for node in VALIDATION_NODES if
                 (directory / "validation" / f"step_{node:09d}" / "summary.json").is_file()]
    extra = [p for p in (directory / "validation").glob("step_*/summary.json")
             if int(p.parent.name.rsplit("_", 1)[-1]) not in VALIDATION_NODES]
    summaries.extend(_read(p) for p in sorted(extra))
    text = "".join(json.dumps(_jsonable(row), ensure_ascii=False, allow_nan=False) + "\n"
                   for row in sorted(summaries, key=lambda item: item["env_step"]))
    temp = directory / f"evaluations.jsonl.{os.getpid()}.tmp"
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, directory / "evaluations.jsonl")
    return True


def _load_evaluation_agent(job, config):
    """Resolve standard final model pair with both SHA and run provenance."""
    model_path = Path(job["model_path"])
    if model_path.is_dir():
        model_path = model_path / "policy.pt"
    references = job.get("model_files", [])
    if references:
        if set(str(Path(row["path"]).resolve()) for row in references) != {
                str(model_path.resolve()), str((model_path.parent / "model.json").resolve())}:
            raise ValueError("I0 evaluation model files differ from final pair")
        for row in references:
            if not Path(row["path"]).is_file() or _sha(row["path"]) != row["sha256"]:
                raise ValueError("I0 evaluation model file checksum differs")
        expected_sha = next(row["sha256"] for row in references
                            if Path(row["path"]).resolve() == model_path.resolve())
    else:
        expected_sha = job.get("model_sha256")
    if not model_path.is_file() or not expected_sha or _sha(model_path) != expected_sha:
        raise ValueError("I0 evaluation model path/SHA differs")
    _seed(config["seed"])
    agent = ImprovementSAC(config)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    source_run_id = job.get("model_run_id", job.get("source_run_id"))
    if model.get("i0_config") != config or model.get("run_id") != source_run_id:
        raise ValueError("I0 model provenance differs")
    agent.actor.load_state_dict(model["actor"], strict=True)
    agent.actor.eval()
    return agent, expected_sha


def _evaluate_job(directory, job, config):
    record, files = _scene_set(job, "test" if "test" in job.get("scene_sets", {}) else "validation")
    if len(files) != 880 or job.get("partition", "test") != "test":
        raise ValueError("I0 model final evaluation requires 880 frozen test derivatives")
    if job.get("action_seed") != 53001 or job.get("repeats") != 3:
        raise ValueError("I0 model final evaluation requires seed53001 and three stochastic actions")
    agent, expected_sha = _load_evaluation_agent(job, config)
    rows = []
    repeats = int(job.get("repeats", 1))
    if repeats != 3:
        raise ValueError("I0 final evaluation requires three actions per scene")
    for index, (path, digest) in enumerate(files):
        for repeat in range(repeats):
            case_id = f"scene_{index:05d}_r{repeat}"
            seed = job["action_seed"] + repeat
            old = _saved_case(directory, case_id, scene_sha=digest, method="model",
                              repeat_index=repeat, action_seed=seed,
                              model_sha=expected_sha)
            if old is not None:
                rows.append(old)
                continue
            if (directory / "stop.request").exists():
                _status(directory, job, config, "stopped", step=len(rows),
                        budget_env_steps=len(files) * repeats,
                        stop_reason="user_stop", completed_cases=len(rows),
                        total_cases=len(files) * repeats)
                return
            scene = _load_scene(path)
            started = _attempt_start(directory, case_id, digest)
            _seed(seed)
            env = _model_episode(agent, scene, config, seed, deterministic=False)
            result = _episode_result(env, record["files"][index], digest, seed, "model")
            result["metrics"]["total_seconds"] = env.total_seconds
            result["model_sha256"] = expected_sha
            result["repeat_index"] = repeat
            rows.append(_case(directory, case_id, result, repeat_index=repeat))
            _attempt_complete(directory, case_id, started)
        _status(directory, job, config, "evaluating", step=len(rows),
                budget_env_steps=len(files) * repeats,
                completed_cases=len(rows), total_cases=len(files) * repeats)
    _atomic_json(directory / "results.json", {"mode": "evaluate", "scenes": rows,
                                               "model_sha256": expected_sha,
                                               "completed_at": _now()})
    _artifacts(directory, job, results=str(directory / "results.json"),
               case_directory=str(directory / "cases"))
    _status(directory, job, config, "completed", step=len(rows),
            budget_env_steps=len(files) * repeats,
            completed_cases=len(rows), total_cases=len(rows),
            results_sha256=_sha(directory / "results.json"))


def _train_job(directory, job, config, resume, frozen_config_sha):
    record, files = _scene_set(job, "train")
    if len(files) != 8064:
        raise ValueError("I0 training requires the frozen 8064 variable-beam derivatives")
    validation = _scene_set(job, "validation")
    if len(validation[1]) != 72:
        raise ValueError("I0 fixed validation requires 72 frozen derivatives")
    order = VariableBeamSceneOrder("mixed_four_buckets", config["seed"],
                                   [digest for _, digest in files], record["files"])
    previous = _read(directory / "status.json", {})
    _seed(config["seed"])
    agent = ImprovementSAC(config)
    if resume:
        saved, digest = _read_checkpoint(directory, job, config, frozen_config_sha)
        gate_passed = _gate_on_resume(directory, job, previous, digest, saved, config)
        agent.load_state_dict(saved["agent"])
        steps, episodes = int(saved["env_step"]), int(saved["episodes"])
        scene_index = saved["scene_index"]
        current = saved["environment"]
        if scene_index is not None:
            selection = order.selection(episodes)
            if selection["source_index"] != scene_index:
                raise ValueError("I0 resume scene order differs")
            env = ImprovementEnv(config)
            env.reset(_load_scene(files[scene_index][0]))
            env.load_checkpoint_state(current)
        else:
            env = None
        _load_rng(saved["rng"])
    else:
        if (directory / "checkpoint.pt").exists():
            raise ValueError("I0 checkpoint exists; resume required")
        gate_passed = False
        steps = episodes = 0
        env = None
        scene_index = None
    _status(directory, job, config, "training", step=steps, episodes=episodes,
            pilot_gate_passed=gate_passed, updates=agent.updates)
    last_checkpoint_step = steps
    if steps == 0 and not (directory / "checkpoint.pt").is_file():
        digest = _checkpoint(directory, job, config, frozen_config_sha, agent,
                             steps, episodes, scene_index, env, pilot_gate_passed=gate_passed)
    else:
        digest = _sha(directory / "checkpoint.pt")
    _artifacts(directory, job, checkpoint=str(directory / "checkpoint.pt"),
               validation_directory=str(directory / "validation"))
    if steps in VALIDATION_NODES or (not gate_passed and
                                   (steps >= config["pilot_stop_steps"] or
                                    episodes >= config["pilot_max_episodes"])):
        if not _validate_node(directory, job, config, agent, steps, digest, validation):
            _status(directory, job, config, "stopped", step=steps, episodes=episodes,
                    stop_reason="user_stop_during_validation", checkpoint_step=steps,
                    checkpoint_sha256=digest, pilot_gate_passed=gate_passed)
            return
    if not gate_passed and (steps >= config["pilot_stop_steps"] or
                            episodes >= config["pilot_max_episodes"]):
        _status(directory, job, config, "stopped", step=steps, episodes=episodes,
                stop_reason="s2p19_pilot_gate_pending", checkpoint_step=steps,
                checkpoint_sha256=digest, pilot_gate_passed=False,
                updates=agent.updates)
        return
    while steps < config["total_steps"] and episodes < config["max_episodes"]:
        if env is None:
            scene_index = order.selection(episodes)["source_index"]
            env = ImprovementEnv(config)
            env.reset(_load_scene(files[scene_index][0]))
        state = env.state()
        features, mask, identifiers = pack_actions(env)
        index = agent.choose(state, features, mask, deterministic=False)
        chosen = identifiers[index]
        next_state, reward, done, info = env.step(chosen)
        next_features, next_mask, _ = pack_actions(env) if not done else (
            np.zeros_like(features), np.eye(1, MAX_ACTIONS, MAX_ACTIONS - 1, dtype=bool)[0], None)
        agent.replay.add(state, features, mask, identifiers, index, reward,
                         next_state, next_features, next_mask, done)
        steps += 1
        diagnostic = None
        if (steps >= config["warmup_steps"] and steps % config["update_every"] == 0):
            for _ in range(config["gradient_steps"]):
                diagnostic = agent.update()
        if diagnostic is not None and steps % 50 == 0:
            _jsonl(directory / "updates.jsonl", {"env_step": steps, **diagnostic})
        _jsonl(directory / "traces.jsonl", {"env_step": steps, "episode": episodes,
                                              "scene_index": scene_index, "action_index": index,
                                              "candidate_id": chosen, "reward": reward,
                                              "done": done, **info})
        if done:
            output = env.output()
            _jsonl(directory / "episodes.jsonl", {"episode": episodes, "env_step": steps,
                                                   "scene_index": scene_index,
                                                   "scene_sha256": files[scene_index][1],
                                                   "metrics": output["metrics"],
                                                   "reward_total": 100 * (output["metrics"]["si"] -
                                                                          output["metrics"]["initial_si"])})
            episodes += 1
            env = None
            scene_index = None
        pilot_reached = (not gate_passed and
                         (steps >= config["pilot_stop_steps"] or
                          episodes >= config["pilot_max_episodes"]))
        requested_stop = (directory / "stop.request").exists()
        checkpoint_due = (steps - last_checkpoint_step >= config["checkpoint_interval"]
                          or pilot_reached or requested_stop or steps >= config["total_steps"]
                          or episodes >= config["max_episodes"]
                          or steps in VALIDATION_NODES)
        if checkpoint_due:
            digest = _checkpoint(directory, job, config, frozen_config_sha, agent, steps,
                                 episodes, scene_index, env, pilot_gate_passed=gate_passed)
            _artifacts(directory, job, checkpoint=str(directory / "checkpoint.pt"),
                       validation_directory=str(directory / "validation"))
            last_checkpoint_step = steps
            if (steps in VALIDATION_NODES or pilot_reached
                    or episodes >= config["max_episodes"]):
                if not _validate_node(directory, job, config, agent, steps, digest, validation):
                    _status(directory, job, config, "stopped", step=steps, episodes=episodes,
                            stop_reason="user_stop_during_validation", checkpoint_step=steps,
                            checkpoint_sha256=digest, pilot_gate_passed=gate_passed,
                            updates=agent.updates)
                    return
            if pilot_reached or requested_stop:
                reason = "s2p19_pilot_gate_pending" if pilot_reached else "user_stop"
                _status(directory, job, config, "stopped", step=steps, episodes=episodes,
                        stop_reason=reason, checkpoint_step=steps,
                        checkpoint_sha256=digest, pilot_gate_passed=gate_passed,
                        updates=agent.updates)
                return
            _status(directory, job, config, "training", step=steps, episodes=episodes,
                    checkpoint_step=steps, checkpoint_sha256=digest,
                    pilot_gate_passed=gate_passed, updates=agent.updates)
    # The fixed endpoint is the final actor, even if the episode cap stops early.
    model_root = Path(job.get("model_dir", directory / "models"))
    model_dir = model_root / "final"
    model_dir.mkdir(parents=True, exist_ok=True)
    model = {"run_id": job["run_id"], "i0_config": config, "actor": agent.actor.state_dict(),
             "env_step": steps, "episodes": episodes, "updates": agent.updates,
             "provenance": {"run_id": job["run_id"], "env_step": steps,
                            "episode": episodes, "algorithm": "s2p19_i0_candidate_sac"},
             "git_commit": job["git_commit"], "source_manifest_sha256":
             _sha(Path(job["source_root"]) / "source_manifest.json")}
    model_path = model_dir / "policy.pt"
    temp = model_dir / f"policy.pt.{os.getpid()}.tmp"
    torch.save(model, temp)
    os.replace(temp, model_path)
    frozen_model_config_path = directory / "config.json"
    if _sha(frozen_model_config_path) != frozen_config_sha:
        raise ValueError("I0 final model config changed after training checkpoint")
    frozen_model_config = _read(frozen_model_config_path)
    _atomic_json(model_dir / "model.json", {"run_id": job["run_id"], "env_step": steps,
                                              "episodes": episodes, "updates": agent.updates,
                                              "format": "satellite_rl_s2p19_improvement_v1",
                                              "algorithm": "sac", "state_dim": STATE_WIDTH,
                                              "parameter_count": sum(p.numel() for p in agent.actor.parameters()),
                                              "config": frozen_model_config,
                                              "policy_sha256": _sha(model_path),
                                              "config_sha256": frozen_config_sha,
                                              "git_commit": job["git_commit"],
                                              "provenance": model["provenance"],
                                              "policy_algorithm": "s2p19_i0_candidate_sac"})
    _artifacts(directory, job, final_policy=str(model_dir),
               checkpoint=str(directory / "checkpoint.pt"),
               validation_directory=str(directory / "validation"))
    _status(directory, job, config, "completed", step=steps, episodes=episodes,
            completed_due_to="env_steps" if steps == config["total_steps"] else "episode_cap",
            model_path=str(model_path), model_sha256=_sha(model_path),
            checkpoint_sha256=_sha(directory / "checkpoint.pt"), updates=agent.updates)


def run_job(jobdir, resume=False):
    directory = Path(jobdir).resolve()
    job = _read(directory / "job.json")
    frozen_path = directory / "config.json"
    frozen = _read(frozen_path)
    if not isinstance(job, dict) or not isinstance(frozen, dict):
        raise ValueError("I0 workbench job/config files are missing")
    if frozen.get("research_stage") != "sac_improvement":
        raise ValueError("I0 requires research_stage=sac_improvement")
    config = _config(job, frozen)
    if job.get("mode") not in ("gate", "baselines", "train", "evaluate"):
        raise ValueError("I0 mode must be gate/baselines/train/evaluate")
    if (job.get("kind") == "train") != (job["mode"] == "train"):
        raise ValueError("I0 kind/mode mismatch")
    frozen_sha = _sha(frozen_path)
    try:
        preflight = _cuda_preflight(job, config)
        _atomic_json(directory / "i0_preflight.json", preflight)
        if job["mode"] == "train":
            _train_job(directory, job, config, resume, frozen_sha)
        elif job["mode"] == "gate":
            _gate_job(directory, job, config)
        elif job["mode"] == "baselines":
            _baseline_job(directory, job, config, "baselines")
        else:
            _evaluate_job(directory, job, config)
    except BaseException as error:
        failure = {"run_id": job.get("run_id"), "time": _now(), "error": str(error),
                   "traceback": traceback.format_exc(), "last_status": _read(directory / "status.json", {}),
                   "checkpoint_sha256": _sha(directory / "checkpoint.pt")
                   if (directory / "checkpoint.pt").is_file() else None}
        _atomic_json(directory / "failure.json", failure)
        previous = failure["last_status"]
        _status(directory, job, config, "failed", step=int(previous.get("env_step", 0)),
                episodes=int(previous.get("episodes", 0)), error=str(error),
                budget_env_steps=(frozen.get("training", {}).get("total_steps")
                                  if job.get("mode") != "train" else config["total_steps"]),
                failure_path=str(directory / "failure.json"))
        raise
