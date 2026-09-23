"""Reproducible, resumable SAC comparison runs with a fixed interaction budget.

Run this module from a clean, frozen checkout using the DQN CUDA interpreter.
The legacy ``train_sac_*`` entry points retain their original defaults.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch

from satellite_rl.paths import project_path
from satellite_rl.training.run_monitor import RunMonitor, atomic_json, utc_now


MODELS = {
    "transformer": ("sac_transformer", 128),
    "cnn_matched": ("sac_cnn_matched", 128),
    "mlp_matched": ("sac_mlp_matched", 512),
}
NETWORKS = ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2")
OPTIMIZERS = ("actor_optimizer", "critic_1_optimizer", "critic_2_optimizer", "log_alpha_optimizer")
SCENE_KEYS = ("beam_location", "beam_G_Tx_max", "UE_G_Rx_max", "distances", "rate", "noise_temperature")
TRAIN_SEEDS = (51001, 51002, 51003)
SCENE_STREAM_SEEDS = (61001, 61002, 61003)
VALIDATION_SCENE_SEEDS = tuple(range(71000, 71020))
TEST_SCENE_SEEDS = tuple(range(81000, 81100))
ACTION_SEEDS = (91001, 91002, 91003)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_id_for(model, formula, seed):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{model}_{formula}_s{seed}_{uuid.uuid4().hex[:10]}"


def _git_result(*args):
    root = Path(__file__).resolve().parents[3]
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def strict_cuda_preflight():
    """Reject CPU, wrong Conda environment, dirty source, or editable-import drift."""
    import satellite_rl

    root = Path(__file__).resolve().parents[3]
    interpreter = Path(sys.executable).resolve()
    parts = [piece.lower() for piece in interpreter.parts]
    if "dqn" not in parts or "envs" not in parts:
        raise RuntimeError(f"必须使用 DQN Conda 的 Python，当前为 {interpreter}")
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError(f"DQN 无可用 CUDA：PyTorch={torch.__version__}, 构建={torch.version.cuda}")
    torch.set_num_threads(1)
    # 各组明确采用同一精度设置，避免被其他模块导入顺序影响。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    package = Path(satellite_rl.__file__).resolve()
    if not package.is_relative_to(root / "src"):
        raise RuntimeError(f"satellite_rl 导入自 {package}，应来自冻结目录 {root / 'src'}")
    if _git_result("status", "--porcelain"):
        raise RuntimeError(f"训练源码工作目录含未提交改动：{root}")
    return {
        "python": str(interpreter), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(0),
        "numpy": np.__version__, "package_file": str(package),
        "pandas": version("pandas"), "matplotlib": version("matplotlib"),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "source_root": str(root), "git_commit": _git_result("rev-parse", "HEAD"),
        "cities_data": {"path": project_path("cities_china.csv"),
                        "sha256": sha256_file(project_path("cities_china.csv"))},
    }


def _rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def isolated_rng(seed):
    """Validation must leave every training random stream unchanged."""
    before = _rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        _restore_rng(before)


def _atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scene_arrays(beam_info):
    arrays = {key: np.array(beam_info.beam_info[key], copy=True) for key in SCENE_KEYS}
    arrays["original_indices"] = np.array(beam_info.shuffle_indices, copy=True)
    return arrays


def _save_scene(path, beam_info):
    arrays = _scene_arrays(beam_info)
    if Path(path).exists():
        with np.load(path) as old:
            if any(not np.array_equal(old[key], arrays[key]) for key in arrays):
                raise RuntimeError(f"场景数据冲突：{path}")
    else:
        _atomic_npz(path, arrays)
    return sha256_file(path)


def _load_scene(path):
    from types import SimpleNamespace
    with np.load(path) as data:
        return SimpleNamespace(beam_info={key: np.array(data[key], copy=True) for key in SCENE_KEYS})


def prepare_fixed_scenes(experiment_dir, group, seeds):
    from satellite_rl.envs.satellite import BeamInfo

    scene_dir = Path(experiment_dir) / "scenes" / group
    scene_dir.mkdir(parents=True, exist_ok=True)
    references = {}
    for seed in seeds:
        path = scene_dir / f"scene_{seed}.npz"
        if not path.exists():
            with isolated_rng(seed):
                _save_scene(path, BeamInfo())
        references[str(seed)] = {"path": str(path), "sha256": sha256_file(path)}
    return references


def _agent_state(agent):
    return {
        "networks": {name: getattr(agent, name).state_dict() for name in NETWORKS},
        "optimizers": {name: getattr(agent, name).state_dict() for name in OPTIMIZERS},
        "log_alpha": agent.log_alpha.detach().clone(),
        "target_entropy": agent.target_entropy,
    }


def _load_agent_state(agent, state):
    for name in NETWORKS:
        getattr(agent, name).load_state_dict(state["networks"][name], strict=True)
    with torch.no_grad():
        agent.log_alpha.copy_(state["log_alpha"].to(agent.log_alpha.device))
    for name in OPTIMIZERS:
        getattr(agent, name).load_state_dict(state["optimizers"][name])
    if float(agent.target_entropy) != float(state["target_entropy"]):
        raise RuntimeError("断点目标熵与当前配置不一致")


def _checkpoint_payload(agent, replay, progress, config_hash):
    return {
        "format_version": 1, "config_hash": config_hash,
        "agent": _agent_state(agent), "replay": list(replay.buffer),
        "replay_capacity": replay.buffer.maxlen, "progress": progress,
        "rng": _rng_state(), "saved_at": utc_now(),
    }


def _ensure_finite_agent(agent):
    for name in NETWORKS:
        network = getattr(agent, name)
        for key, value in network.state_dict().items():
            if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"{name}.{key} 存在 NaN/Inf，保留上一个正常断点")
    if not bool(torch.isfinite(agent.log_alpha).all()):
        raise FloatingPointError("log_alpha 存在 NaN/Inf，保留上一个正常断点")


def save_checkpoint(path, agent, replay, progress, config_hash):
    """Synchronously serialize between completed steps, then atomically replace."""
    _ensure_finite_agent(agent)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(_checkpoint_payload(agent, replay, progress, config_hash), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def save_actor_weights(path, actor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(actor.state_dict(), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def load_checkpoint(path, agent, replay, config_hash):
    # This file is produced locally by this experiment and is intentionally trusted.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1 or payload.get("config_hash") != config_hash:
        raise RuntimeError("断点格式或实验配置不匹配")
    _load_agent_state(agent, payload["agent"])
    if replay.buffer.maxlen != payload["replay_capacity"]:
        raise RuntimeError("断点回放容量不匹配")
    replay.buffer.clear()
    replay.buffer.extend(payload["replay"])
    _restore_rng(payload["rng"])
    return payload["progress"]


def _archive_future_artifacts(run_dir, model_dir, boundary_step):
    """Preserve abandoned evaluation files while preventing stale result reuse."""
    moved = []
    evaluation_dir = Path(run_dir) / "evaluation"
    if evaluation_dir.exists():
        destination = evaluation_dir / ("abandoned_resume_" + uuid.uuid4().hex[:10])
        for path in evaluation_dir.iterdir():
            match = re.search(r"_(\d+)(?:_steps)?\.(?:json|npz|jsonl\.gz)$", path.name)
            if path.is_file() and match and int(match.group(1)) > boundary_step:
                destination.mkdir(exist_ok=True)
                target = destination / path.name
                path.rename(target)
                moved.append({"from": str(path), "to": str(target)})
    best = Path(model_dir) / "best.pt"
    if best.exists():
        payload = torch.load(best, map_location="cpu", weights_only=False)
        if payload["progress"]["env_step"] > boundary_step:
            destination = Path(model_dir) / ("abandoned_resume_" + uuid.uuid4().hex[:10])
            destination.mkdir()
            for path in (best, Path(model_dir) / "best_actor.pt"):
                if path.exists():
                    target = destination / path.name
                    path.rename(target)
                    moved.append({"from": str(path), "to": str(target)})
    return moved


def _create_agent(model, formula, run_dir, session_name):
    module_name, hidden = MODELS[model]
    module = importlib.import_module("satellite_rl.agents." + module_name)
    previous = os.environ.get("SATELLITE_RL_OUTPUT_ROOT")
    os.environ["SATELLITE_RL_OUTPUT_ROOT"] = str(run_dir / "legacy" / session_name)
    try:
        agent = module.SAC(
            n_states=204, n_hiddens=hidden, freq_dim=100, slots_dim=10,
            power_dim=10, actor_lr=1e-5, critic_lr=1e-4, alpha_lr=1e-4,
            target_entropy=-6.9 if formula == "legacy" else 6.9,
            tau=0.005, gamma=0.99, device=torch.device("cuda"),
            formula_variant=formula,
        )
    finally:
        if previous is None:
            os.environ.pop("SATELLITE_RL_OUTPUT_ROOT", None)
        else:
            os.environ["SATELLITE_RL_OUTPUT_ROOT"] = previous
    return agent, module.ReplayBuffer(capacity=20000)


def _new_environment(beam_info):
    from satellite_rl.envs.satellite import SatelliteFreqState, CurrentBeamState, Env
    frequency = SatelliteFreqState()
    current = CurrentBeamState(beam_info.beam_info)
    environment = Env(frequency, beam_info, current)
    return environment, environment.reset(beam_info)


def _terminal_metrics(env):
    from satellite_rl.evaluation.terminal_metrics import terminal_metrics, terminal_beam_arrays
    return terminal_metrics(env), terminal_beam_arrays(env)


@torch.no_grad()
def _evaluation_action(agent, state, mode):
    if mode == "sample":
        return agent.take_action(state)
    inputs = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    return [int(probability.argmax(dim=1).item()) for probability in agent.actor(inputs)]


def evaluate(agent, scene_refs, action_seeds, checkpoint, step, monitor, output_dir,
             *, include_argmax=False):
    """Evaluate saved actual scenes; preserve training RNG and actor mode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{checkpoint}_{step}.json"
    arrays_path = output_dir / f"{checkpoint}_{step}.npz"
    trace_path = output_dir / f"{checkpoint}_{step}_steps.jsonl.gz"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for row in result["rows"]:
            monitor.row("evaluations", row)
        monitor.flush()
        return result
    original_training = agent.actor.training
    rows = []
    arrays = {}
    trace_temp = trace_path.with_name(trace_path.name + "." + uuid.uuid4().hex + ".tmp")
    trace_stream = gzip.open(trace_temp, "wt", encoding="utf-8") if checkpoint in ("final_test", "best_test") else None
    try:
        agent.actor.eval()
        for scene_id, reference in scene_refs.items():
            trials = [(seed, "sample") for seed in action_seeds]
            if include_argmax:
                trials.append(("argmax", "argmax"))
            for action_seed, action_mode in trials:
                with isolated_rng(action_seed if action_mode == "sample" else 0):
                    env, state = _new_environment(_load_scene(reference["path"]))
                    done = False
                    slot_counts = {"requested_slots_total": 0, "actual_slots_total": 0,
                                   "out_slots_total": 0, "occupied_slots_total": 0,
                                   "clipped_steps": 0}
                    while not done:
                        action = _evaluation_action(agent, state, action_mode)
                        requested = [env.beam_idx % 8, action[0], action[1] + 1,
                                     10 * np.log10((action[2] + 1) * 5)]
                        actual = requested.copy()
                        beam_index = env.beam_idx
                        state, reward, done, info = env.step(actual)
                        slot_counts["requested_slots_total"] += requested[2]
                        slot_counts["actual_slots_total"] += actual[2]
                        slot_counts["out_slots_total"] += int(np.asarray(info["out"]).item())
                        slot_counts["occupied_slots_total"] += int(np.asarray(info["occupy"]).item())
                        slot_counts["clipped_steps"] += int(actual[2] < requested[2])
                        if trace_stream:
                            trace_stream.write(json.dumps({
                                "scene_id": scene_id, "action_seed": action_seed,
                                "action_mode": action_mode,
                                "beam_index": beam_index, "requested": requested,
                                "actual": actual, "reward": reward,
                                "info": {key: np.asarray(value).tolist()
                                         for key, value in info.items()},
                            }, ensure_ascii=False, allow_nan=False) + "\n")
                    metrics, beam_arrays = _terminal_metrics(env)
                key = f"scene_{scene_id}_action_{action_seed}"
                for name, value in beam_arrays.items():
                    arrays[f"{key}_{name}"] = np.asarray(value)
                rows.append({"checkpoint": checkpoint, "env_step": step,
                             "scene_id": scene_id, "action_mode": action_mode,
                             "action_seed": action_seed, **metrics, **slot_counts})
    finally:
        agent.actor.train(original_training)
        if trace_stream:
            trace_stream.close()
    if trace_stream:
        os.replace(trace_temp, trace_path)
    _atomic_npz(arrays_path, arrays)
    result = {
        "checkpoint": checkpoint, "env_step": step, "rows": rows,
        "scene_references": scene_refs, "action_seeds": list(action_seeds),
        "include_argmax": include_argmax,
        "arrays_path": str(arrays_path), "arrays_sha256": sha256_file(arrays_path),
        "trace_path": str(trace_path) if trace_stream else None,
        "trace_sha256": sha256_file(trace_path) if trace_stream else None,
        "si_mean": float(np.mean([row["si_mean"] for row in rows
                                   if row["action_mode"] == "sample"])),
        "argmax_si_mean": (float(np.mean([row["si_mean"] for row in rows
                                          if row["action_mode"] == "argmax"]))
                           if include_argmax else None),
    }
    atomic_json(result_path, result)
    for row in rows:
        monitor.row("evaluations", row)
    monitor.flush()
    return result


def _config(args, runtime):
    seed_index = TRAIN_SEEDS.index(args.seed)
    diagnostic = (args.steps != 400000 or args.validation_interval == 0
                  or args.skip_final_evaluation or args.validation_scene_count != 20
                  or args.test_scene_count != 100)
    return {
        "experiment_id": args.experiment_id, "run_id": args.run_id,
        "model": args.model, "formula": args.formula, "train_seed": args.seed,
        "train_scene_seed": SCENE_STREAM_SEEDS[seed_index],
        "budget_env_steps": args.steps, "validation_interval": args.validation_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "validation_scene_count": args.validation_scene_count,
        "test_scene_count": args.test_scene_count,
        "skip_final_evaluation": args.skip_final_evaluation,
        "diagnostic_not_formal_comparison": diagnostic,
        "batch_size": 256, "replay_capacity": 20000, "warmup_size": 2000,
        "actor_lr": 1e-5, "critic_lr": 1e-4, "alpha_lr": 1e-4,
        "gamma": 0.99, "tau": 0.005,
        "target_entropy": -6.9 if args.formula == "legacy" else 6.9,
        "validation_scene_seeds": VALIDATION_SCENE_SEEDS[:args.validation_scene_count],
        "test_scene_seeds": TEST_SCENE_SEEDS[:args.test_scene_count],
        "action_seeds": ACTION_SEEDS,
        "trace_policy": "final and best test every step; training last 50 steps on failure",
        "runtime": runtime,
    }


def _config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _progress_status(progress, replay, started, stage, **other):
    elapsed = time.monotonic() - started
    return {"stage": stage, "env_step": progress["env_step"],
            "update_step": progress["update_step"], "episode": progress["episode"],
            "replay_size": replay.size(), "elapsed_seconds": elapsed,
            "pid": os.getpid(), **other}


def _scalar_metrics(metrics):
    """Move all GPU diagnostics in one transfer at the configured low frequency."""
    aliases = {"critic1_loss": "critic_1_loss", "critic2_loss": "critic_2_loss"}
    gpu = [(key, value.detach()) for key, value in metrics.items()
           if isinstance(value, torch.Tensor) and value.is_cuda]
    result = {aliases.get(key, key): float(value) for key, value in metrics.items()
              if not isinstance(value, torch.Tensor)}
    result.update({aliases.get(key, key): float(value.detach())
                   for key, value in metrics.items()
                   if isinstance(value, torch.Tensor) and not value.is_cuda})
    if gpu:
        values = torch.stack([value.reshape(()) for _, value in gpu]).cpu().tolist()
        result.update({aliases.get(key, key): float(value)
                       for (key, _), value in zip(gpu, values)})
    if any(not np.isfinite(value) for value in result.values()):
        raise FloatingPointError("更新诊断包含 NaN/Inf，训练已停止")
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _tensorboard_writer(run_dir, monitor):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        monitor.event("tensorboard_unavailable", reason="可选 tensorboard 依赖未安装")
        return None
    return SummaryWriter(log_dir=str(Path(run_dir) / "tensorboard"),
                         max_queue=100, flush_secs=30)


def _episode_row(progress, env, elapsed):
    metrics, _ = _terminal_metrics(env)
    return {"episode": progress["episode"], "env_step": progress["env_step"],
            "length": progress["episode_length"], "return": progress["episode_return"],
            "wall_seconds": elapsed, "scene_id": progress["scene_id"],
            "termination_reason": "power_exceeded" if env.overload else "all_beams",
            "requested_slots_total": progress["requested_slots_total"],
            "actual_slots_total": progress["actual_slots_total"],
            "out_slots_total": progress["out_slots_total"],
            "occupied_slots_total": progress["occupied_slots_total"],
            "clipped_steps": progress["clipped_steps"],
            **metrics}


def _initial_progress():
    return {"env_step": 0, "update_step": 0, "episode": 0,
            "episode_length": 0, "episode_return": 0.0, "episode_wall_seconds": 0.0,
            "requested_slots_total": 0, "actual_slots_total": 0,
            "out_slots_total": 0, "occupied_slots_total": 0, "clipped_steps": 0,
            "scene_id": None, "env": None, "state": None,
            "best_si_mean": None, "next_validation": None,
            "next_checkpoint": None}


def run(args):
    runtime = strict_cuda_preflight()
    if args.run_id is None:
        if args.resume:
            raise ValueError("恢复训练必须提供 --run-id")
        args.run_id = run_id_for(args.model, args.formula, args.seed)
    archive_root = Path(args.archive_root).resolve()
    os.environ["SATELLITE_RL_OUTPUT_ROOT"] = str(archive_root)
    experiment_dir = Path(project_path(f"outputs/experiments/{args.experiment_id}"))
    run_dir = experiment_dir / args.run_id
    model_dir = Path(project_path(f"models/runs/{args.experiment_id}/{args.run_id}"))
    if args.resume and not (run_dir / "manifest.json").exists():
        raise FileNotFoundError("恢复目录没有 manifest.json")
    if not args.resume and run_dir.exists():
        raise FileExistsError(f"run_id 已存在：{run_dir}")
    config = _config(args, runtime)
    config_hash = _config_hash(config)
    manifest_path = run_dir / "manifest.json"
    if args.resume:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior["config_hash"] != config_hash:
            raise RuntimeError("恢复配置与原运行清单不一致")
    model_dir.mkdir(parents=True, exist_ok=args.resume)
    monitor = RunMonitor(run_dir, args.run_id, resume=args.resume)
    if not args.resume:
        atomic_json(manifest_path, {"config": config, "config_hash": config_hash,
                                    "command": sys.argv, "created_at": utc_now()})
    started = time.monotonic()
    last_status = started
    last_print = started
    stop_requested = False
    last_traces = deque(maxlen=50)
    tensorboard = _tensorboard_writer(run_dir, monitor)

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    try:
        # All scenario files are concrete arrays and independent of training RNG.
        validation_refs = (prepare_fixed_scenes(
            experiment_dir, "validation", VALIDATION_SCENE_SEEDS[:args.validation_scene_count])
            if args.validation_interval else {})
        test_refs = (prepare_fixed_scenes(
            experiment_dir, "test", TEST_SCENE_SEEDS[:args.test_scene_count])
            if not args.skip_final_evaluation else {})
        if not args.resume:
            atomic_json(manifest_path, {"config": config, "config_hash": config_hash,
                                        "command": sys.argv, "created_at": utc_now(),
                                        "fixed_scenes": {"validation": validation_refs,
                                                         "test": test_refs}})
        random.seed(args.seed)
        np.random.seed(SCENE_STREAM_SEEDS[TRAIN_SEEDS.index(args.seed)])
        torch.manual_seed(args.seed)
        agent, replay = _create_agent(args.model, args.formula, run_dir, uuid.uuid4().hex)
        if args.resume:
            progress = load_checkpoint(model_dir / "latest.pt", agent, replay, config_hash)
            started -= progress.get("elapsed_seconds", 0.0)
            monitor.reconcile(progress)
            abandoned = _archive_future_artifacts(run_dir, model_dir, progress["env_step"])
            monitor.event("resumed", from_checkpoint=str(model_dir / "latest.pt"),
                          env_step=progress["env_step"], abandoned_artifacts=abandoned)
        else:
            progress = _initial_progress()
            progress["next_validation"] = args.validation_interval or None
            progress["next_checkpoint"] = args.checkpoint_interval
            monitor.event("started", runtime=runtime)
        progress_phase = "warmup" if replay.size() <= 2000 else "training"
        window_started = time.monotonic()
        window_step = progress["env_step"]
        window_update = progress["update_step"]

        def checkpoint(name):
            monitor.flush()
            progress["elapsed_seconds"] = time.monotonic() - started
            path = model_dir / (name + ".pt")
            digest = save_checkpoint(path, agent, replay, progress, config_hash)
            monitor.event("checkpoint", name=name, path=str(path), sha256=digest,
                          env_step=progress["env_step"])
            monitor.write_status(latest_checkpoint=str(path), latest_checkpoint_sha256=digest)

        if not args.resume:
            checkpoint("initial")
            checkpoint("latest")

        def validate_if_due():
            if (progress["env"] is not None or progress["next_validation"] is None
                    or progress["env_step"] < progress["next_validation"]):
                return
            # Save before evaluation so a crash can repeat this boundary exactly.
            checkpoint("latest")
            monitor.write_status(**_progress_status(progress, replay, started, "validating"))
            result = evaluate(agent, validation_refs, ACTION_SEEDS,
                              "validation", progress["env_step"], monitor,
                              run_dir / "evaluation")
            score = result["si_mean"]
            if tensorboard:
                tensorboard.add_scalar("validation/terminal_si_mean", score,
                                       progress["env_step"])
            monitor.event("validation", env_step=progress["env_step"], si_mean=score)
            if progress["best_si_mean"] is None or score > progress["best_si_mean"]:
                progress["best_si_mean"] = score
                checkpoint("best")
                save_actor_weights(model_dir / "best_actor.pt", agent.actor)
            while progress["next_validation"] <= progress["env_step"]:
                progress["next_validation"] += args.validation_interval
            # Commit selection bookkeeping before the next training episode.
            checkpoint("latest")

        if args.resume:
            validate_if_due()

        while (progress["env_step"] < args.steps and not stop_requested
               and (args.stop_after_steps is None
                    or progress["env_step"] < args.stop_after_steps)):
            if progress["env"] is None:
                from satellite_rl.envs.satellite import BeamInfo
                beam_info = BeamInfo()
                progress["episode"] += 1
                progress["scene_id"] = f"train_{progress['episode']:06d}"
                path = run_dir / "scenes" / "train" / (progress["scene_id"] + ".npz")
                scene_sha = _save_scene(path, beam_info)
                monitor.event("training_scene", episode=progress["episode"],
                              scene_id=progress["scene_id"], path=str(path), sha256=scene_sha)
                progress["env"], progress["state"] = _new_environment(beam_info)
                progress["episode_length"] = 0
                progress["episode_return"] = 0.0
                for key in ("requested_slots_total", "actual_slots_total", "out_slots_total",
                            "occupied_slots_total", "clipped_steps"):
                    progress[key] = 0
            env = progress["env"]
            state = progress["state"]
            step_started = time.monotonic()
            action = agent.take_action(state)
            requested = [env.beam_idx % 8, action[0], action[1] + 1,
                         10 * np.log10((action[2] + 1) * 5)]
            actual = requested.copy()
            next_state, reward, done, info = env.step(actual)
            replay.add(state, list(action), reward, next_state, done)
            progress["state"] = next_state
            progress["env_step"] += 1
            progress["episode_length"] += 1
            progress["episode_return"] += reward
            progress["requested_slots_total"] += requested[2]
            progress["actual_slots_total"] += actual[2]
            progress["out_slots_total"] += int(np.asarray(info["out"]).item())
            progress["occupied_slots_total"] += int(np.asarray(info["occupy"]).item())
            progress["clipped_steps"] += int(actual[2] < requested[2])
            last_traces.append({"episode": progress["episode"],
                                "env_step": progress["env_step"],
                                "requested": requested, "actual": actual,
                                "reward": reward, "info": {key: np.asarray(value).tolist()
                                                          for key, value in info.items()}})
            if replay.size() > 2000:
                s, a, r, ns, d = replay.sample(256)
                agent.update({"states": s, "actions": a, "rewards": r,
                              "next_states": ns, "dones": d})
                progress["update_step"] += 1
                if progress["update_step"] % 100 == 0:
                    metrics = _scalar_metrics(getattr(agent, "last_metrics", {}))
                    monitor.row("updates", {"env_step": progress["env_step"],
                                            "update_step": progress["update_step"],
                                            "episode": progress["episode"],
                                            "wall_seconds": time.monotonic() - started,
                                            **metrics})
                    if tensorboard:
                        for key, value in metrics.items():
                            tensorboard.add_scalar("learning/" + key, value,
                                                   progress["env_step"])
            progress["episode_wall_seconds"] += time.monotonic() - step_started
            if done:
                episode_row = _episode_row(progress, env, progress["episode_wall_seconds"])
                monitor.row("episodes", episode_row)
                if tensorboard:
                    tensorboard.add_scalar("training/return", episode_row["return"],
                                           progress["env_step"])
                    tensorboard.add_scalar("training/terminal_si_mean",
                                           episode_row["si_mean"], progress["env_step"])
                progress["env"] = None
                progress["state"] = None
                progress["episode_wall_seconds"] = 0.0
                if progress["env_step"] >= progress["next_checkpoint"]:
                    progress["next_checkpoint"] += args.checkpoint_interval
                    checkpoint("latest")
                validate_if_due()
            now = time.monotonic()
            current_phase = "warmup" if replay.size() <= 2000 else "training"
            if current_phase != progress_phase:
                progress_phase = current_phase
                window_started = now
                window_step = progress["env_step"]
                window_update = progress["update_step"]
                monitor.write_status(**_progress_status(
                    progress, replay, started, current_phase,
                    budget_env_steps=args.steps, steps_per_second=None,
                    updates_per_second=None, eta_seconds=None,
                    progress_at=utc_now()))
                last_status = now
            if now - last_status >= 5:
                duration = max(now - window_started, 1e-9)
                speed = (progress["env_step"] - window_step) / duration
                update_speed = (progress["update_step"] - window_update) / duration
                monitor.write_status(**_progress_status(
                    progress, replay, started, current_phase, budget_env_steps=args.steps,
                    steps_per_second=speed,
                    updates_per_second=update_speed,
                    eta_seconds=((args.steps - progress["env_step"]) / speed
                                 if speed and current_phase == "training" else None),
                    best_validation_si_mean=progress["best_si_mean"],
                    progress_at=utc_now(),
                ))
                last_status = now
                window_started = now
                window_step = progress["env_step"]
                window_update = progress["update_step"]
            if now - last_print >= 30:
                print(f"{args.run_id} | 步 {progress['env_step']}/{args.steps} | "
                      f"更新 {progress['update_step']} | 回合 {progress['episode']}", flush=True)
                last_print = now

        checkpoint("latest")
        if stop_requested:
            monitor.event("interrupted", env_step=progress["env_step"])
            monitor.write_status(**_progress_status(
                progress, replay, started, "interrupted", steps_per_second=None,
                updates_per_second=None, eta_seconds=None, progress_at=utc_now()))
            return 130
        if progress["env_step"] < args.steps:
            monitor.event("diagnostic_paused", env_step=progress["env_step"],
                          stop_after_steps=args.stop_after_steps)
            monitor.write_status(**_progress_status(
                progress, replay, started, "paused", steps_per_second=None,
                updates_per_second=None, eta_seconds=None, progress_at=utc_now()))
            return 0
        checkpoint("final")
        save_actor_weights(model_dir / "final_actor.pt", agent.actor)
        if args.skip_final_evaluation:
            monitor.event("completed_diagnostic", env_step=progress["env_step"])
            monitor.write_status(**_progress_status(
                progress, replay, started, "completed", budget_env_steps=args.steps,
                diagnostic_not_formal_comparison=True, completed_at=utc_now(),
                steps_per_second=None, updates_per_second=None, eta_seconds=0,
                progress_at=utc_now()))
            return 0
        monitor.write_status(**_progress_status(progress, replay, started, "final_validation"))
        final_validation = (evaluate(agent, validation_refs, ACTION_SEEDS,
                                     "final_validation", progress["env_step"], monitor,
                                     run_dir / "evaluation") if validation_refs else None)
        monitor.write_status(**_progress_status(progress, replay, started, "testing"))
        final_test = evaluate(agent, test_refs, ACTION_SEEDS,
                              "final_test", progress["env_step"], monitor,
                              run_dir / "evaluation", include_argmax=True)
        best_test = None
        if (model_dir / "best_actor.pt").exists():
            agent.actor.load_state_dict(torch.load(model_dir / "best_actor.pt",
                                                   map_location="cpu", weights_only=True), strict=True)
            try:
                best_test = evaluate(agent, test_refs, ACTION_SEEDS,
                                     "best_test", progress["env_step"], monitor,
                                     run_dir / "evaluation", include_argmax=True)
            finally:
                agent.actor.load_state_dict(torch.load(model_dir / "final_actor.pt",
                                                       map_location="cpu", weights_only=True), strict=True)
        monitor.event("completed", validation_si_mean=(final_validation["si_mean"]
                                                       if final_validation else None),
                      test_si_mean=final_test["si_mean"],
                      best_test_si_mean=best_test["si_mean"] if best_test else None)
        monitor.write_status(**_progress_status(
            progress, replay, started, "completed", budget_env_steps=args.steps,
            final_validation_si_mean=(final_validation["si_mean"]
                                      if final_validation else None),
            final_test_si_mean=final_test["si_mean"],
            final_test_argmax_si_mean=final_test["argmax_si_mean"],
            best_test_si_mean=best_test["si_mean"] if best_test else None,
            best_test_argmax_si_mean=best_test["argmax_si_mean"] if best_test else None,
            completed_at=utc_now(), steps_per_second=None,
            updates_per_second=None, eta_seconds=0, progress_at=utc_now()))
        return 0
    except Exception as error:
        failure = {"at": utc_now(), "type": type(error).__name__,
                   "message": str(error), "traceback": traceback.format_exc(),
                   "last_traces": list(last_traces)}
        atomic_json(run_dir / "failure.json", _json_safe(failure))
        monitor.event("failed", type=failure["type"], message=failure["message"])
        monitor.write_status(stage="failed", failure_path=str(run_dir / "failure.json"))
        raise
    finally:
        if tensorboard:
            tensorboard.close()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        monitor.close()


def build_parser():
    parser = argparse.ArgumentParser(description="18 组 SAC 配对实验的单次 CUDA 作业")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--formula", choices=("legacy", "corrected"), required=True)
    parser.add_argument("--seed", type=int, choices=TRAIN_SEEDS, required=True)
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--experiment-id", default="sac_comparison_20260924")
    parser.add_argument("--archive-root", default=str(Path(project_path("outputs")).parent))
    parser.add_argument("--validation-interval", type=int, default=10000)
    parser.add_argument("--validation-scene-count", type=int, default=20)
    parser.add_argument("--test-scene-count", type=int, default=100)
    parser.add_argument("--skip-final-evaluation", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-steps", type=int,
                        help="诊断时暂停当前进程，保留原总预算及续训配置")
    return parser


def main():
    args = build_parser().parse_args()
    if args.steps <= 0 or args.validation_interval < 0 or args.checkpoint_interval <= 0:
        raise ValueError("训练步数及保存间隔必须为正，验证间隔可为零")
    if not (0 <= args.validation_scene_count <= 20 and 0 <= args.test_scene_count <= 100):
        raise ValueError("验证场景须为0–20，考核场景须为0–100")
    if args.validation_interval and not args.validation_scene_count:
        raise ValueError("启用周期验证时场景数量须大于零")
    if not args.skip_final_evaluation and not args.test_scene_count:
        raise ValueError("启用最终考核时场景数量须大于零")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise ValueError("诊断暂停步数必须为正")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
