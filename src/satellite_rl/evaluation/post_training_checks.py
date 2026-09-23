"""Fixed post-training checks for the approved 18-run SAC comparison.

This module never trains. Every command refuses to run until all eighteen
frozen-source workers and their fixed-budget final checkpoints are complete.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from satellite_rl.evaluation.checkpoints import load_network_checkpoint
from satellite_rl.evaluation.terminal_metrics import terminal_metrics
from satellite_rl.paths import project_path
from satellite_rl.training import experiment_runner as runner
from satellite_rl.training.run_monitor import RunMonitor, atomic_json, utc_now


EXPERIMENT_ID = "sac_comparison_20260924_r2"
TRAIN_COMMIT = "36c14d74cbe6b9cf7410ee0595ee6d73083afdc8"
OLD_ENV_COMMIT = "d6487f27a0f781ceb754a8ee1df23091d53771ac"
EXPECTED_RUN_IDS = {
    f"{model}_{formula}_s{seed}"
    for seed in runner.TRAIN_SEEDS
    for model in runner.MODELS
    for formula in ("legacy", "corrected")
}
SCENE_IDS = tuple(range(81000, 81020))
ACTION_SEEDS = tuple(runner.ACTION_SEEDS)
CONDITIONS = ("demand_075", "demand_125", "within_group_order")
GUARDED_FILES = (
    "src/satellite_rl/paths.py",
    "src/satellite_rl/envs/satellite.py",
    "src/satellite_rl/agents/sac_transformer.py",
    "src/satellite_rl/agents/sac_cnn_matched.py",
    "src/satellite_rl/agents/sac_mlp_matched.py",
    "src/satellite_rl/evaluation/terminal_metrics.py",
    "src/satellite_rl/training/experiment_runner.py",
)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()


def source_guard(frozen_root):
    """Prevent future main-branch algorithm edits from contaminating results."""
    current = Path(__file__).resolve().parents[3]
    frozen_root = Path(frozen_root).resolve()
    if _git(frozen_root, "rev-parse", "HEAD") != TRAIN_COMMIT:
        raise RuntimeError("训练冻结目录的完整提交号不符")
    if _git(frozen_root, "status", "--porcelain"):
        raise RuntimeError("训练冻结目录含未提交修改")
    if _git(current, "status", "--porcelain"):
        raise RuntimeError("后处理源码目录含未提交修改")
    hashes = {}
    for relative in GUARDED_FILES:
        a, b = current / relative, frozen_root / relative
        if not a.is_file() or not b.is_file():
            raise FileNotFoundError(relative)
        left, right = runner.sha256_file(a), runner.sha256_file(b)
        if left != right:
            raise RuntimeError(f"后处理导入的算法/环境文件与训练冻结版不同：{relative}")
        hashes[relative] = left
    imported = Path(importlib.import_module("satellite_rl").__file__).resolve()
    if not imported.is_relative_to(current / "src"):
        raise RuntimeError(f"satellite_rl 导入路径不属于后处理提交：{imported}")
    return {"post_commit": _git(current, "rev-parse", "HEAD"),
            "post_source_root": str(current), "training_commit": TRAIN_COMMIT,
            "frozen_source_root": str(frozen_root), "matched_file_sha256": hashes}


def _final_result_is_complete(path):
    data = _load_json(path)
    if (data.get("checkpoint") != "final_test" or data.get("env_step") != 400000
            or data.get("include_argmax") is not True):
        raise RuntimeError(f"最终考核不是完整预算终点：{path}")
    rows = data.get("rows", [])
    expected = {(str(scene), "sample", str(seed))
                for scene in range(81000, 81100) for seed in ACTION_SEEDS}
    expected.update((str(scene), "argmax", "argmax")
                    for scene in range(81000, 81100))
    observed = [(str(row["scene_id"]), row["action_mode"],
                 str(row["action_seed"])) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError(f"最终考核缺失或重复场景/动作：{path}")
    for file_key, hash_key in (("arrays_path", "arrays_sha256"),
                               ("trace_path", "trace_sha256")):
        artifact = Path(data[file_key])
        if not artifact.is_file() or runner.sha256_file(artifact) != data[hash_key]:
            raise RuntimeError(f"最终考核原始文件缺失或校验值不符：{artifact}")
    return data


def require_completed(archive_root, frozen_root):
    """Read all terminal artifacts and verify full final weights, without GPU."""
    archive_root = Path(archive_root).resolve()
    exp_dir = archive_root / "outputs" / "experiments" / EXPERIMENT_ID
    plan = _load_json(exp_dir / "suite_plan.json")
    summary = _load_json(exp_dir / "suite_status.json")
    if (plan.get("steps") != 400000 or plan["source"]["git_commit"] != TRAIN_COMMIT
            or Path(plan["source"]["source_root"]).resolve() != Path(frozen_root).resolve()
            or {job["run_id"] for job in plan["jobs"]} != EXPECTED_RUN_IDS
            or len(plan["jobs"]) != 18):
        raise RuntimeError("18组正式实验计划与预先约定不符")
    counts = summary.get("counts", {})
    if any(counts.get(key) != wanted for key, wanted in
           (("completed", 18), ("pending", 0), ("running", 0), ("failed", 0))):
        raise RuntimeError(f"18组尚未全部完成，拒绝后处理：{counts}")
    if (exp_dir / "suite.lock").exists():
        raise RuntimeError("监督进程仍持有队列锁，等待正式队列退出")
    final = {}
    for job in plan["jobs"]:
        run_id = job["run_id"]
        if job["state"] != "completed":
            raise RuntimeError(f"作业未完成：{run_id}")
        run_dir = exp_dir / run_id
        status = _load_json(run_dir / "status.json")
        manifest = _load_json(run_dir / "manifest.json")
        if (status.get("stage") != "completed" or status.get("env_step") != 400000
                or manifest["config"]["runtime"]["git_commit"] != TRAIN_COMMIT
                or manifest["config"]["model"] != job["model"]
                or manifest["config"]["formula"] != job["formula"]
                or manifest["config"]["train_seed"] != job["seed"]):
            raise RuntimeError(f"作业终点状态与配置不符：{run_id}")
        model_dir = archive_root / "models" / "runs" / EXPERIMENT_ID / run_id
        full, actor = model_dir / "final.pt", model_dir / "final_actor.pt"
        if not full.is_file() or not actor.is_file():
            raise FileNotFoundError(f"缺少完整终点断点或Actor权重：{run_id}")
        checkpoint = torch.load(full, map_location="cpu", weights_only=False)
        actor_state = torch.load(actor, map_location="cpu", weights_only=True)
        if (checkpoint.get("format_version") != 1
                or checkpoint.get("config_hash") != manifest["config_hash"]
                or checkpoint["progress"]["env_step"] != 400000
                or not all(key in checkpoint for key in ("agent", "replay", "rng"))
                or not all(key in checkpoint["agent"] for key in ("networks", "optimizers", "log_alpha"))
                or set(checkpoint["agent"]["networks"]) != set(runner.NETWORKS)
                or set(checkpoint["agent"]["optimizers"]) != set(runner.OPTIMIZERS)
                or not bool(torch.isfinite(checkpoint["agent"]["log_alpha"]).all())
                or set(actor_state) != set(checkpoint["agent"]["networks"]["actor"])
                or any(not torch.equal(tensor, checkpoint["agent"]["networks"]["actor"][key])
                       for key, tensor in actor_state.items())):
            raise RuntimeError(f"完整终点断点与Actor权重不一致：{run_id}")
        del checkpoint, actor_state
        result = run_dir / "evaluation" / "final_test_400000.json"
        if not result.is_file():
            raise FileNotFoundError(f"缺少终点考核：{run_id}")
        _final_result_is_complete(result)
        final[run_id] = {
            "job": job, "actor": str(actor), "actor_sha256": runner.sha256_file(actor),
            "checkpoint": str(full), "checkpoint_sha256": runner.sha256_file(full),
            "original_test": str(result), "original_test_sha256": runner.sha256_file(result),
        }
    return exp_dir, plan, final


def scenario_arrays(reference):
    with np.load(reference["path"]) as loaded:
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    if runner.sha256_file(reference["path"]) != reference["sha256"]:
        raise RuntimeError(f"固定场景校验值改变：{reference['path']}")
    if "original_indices" not in arrays or len(arrays["original_indices"]) != 100:
        raise RuntimeError("场景缺少100束身份记录")
    return arrays


def transform_scene(arrays, condition, scene_id):
    """Apply exactly one predeclared intervention to every coupled field."""
    result = {key: np.array(value, copy=True) for key, value in arrays.items()}
    order = np.arange(100)
    if condition in ("demand_075", "demand_125"):
        factor = 0.75 if condition == "demand_075" else 1.25
        result["rate"] = result["rate"].astype(np.float64) * factor
    elif condition == "within_group_order":
        rng = np.random.default_rng(int(scene_id) + 100000)
        for group in range(8):
            positions = np.arange(group, 100, 8)
            order[positions] = rng.permutation(positions)
        if not np.array_equal(order % 8, np.arange(100) % 8):
            raise AssertionError("同组重排改变了组归属")
        result = {key: value[order] for key, value in result.items()}
    else:
        raise ValueError(condition)
    result["new_position_to_old_position"] = order
    result["new_position_to_original_identity"] = np.asarray(arrays["original_indices"])[order]
    result["group_by_new_position"] = np.arange(100) % 8
    return result


def _put_immutable_json(path, data):
    path = Path(path)
    if path.exists():
        if _load_json(path) != data:
            raise RuntimeError(f"已有结果与当前固定配置不一致：{path}")
        return
    atomic_json(path, data)


def prepare_condition_scenes(post_root, plan, provenance):
    references = {}
    for condition in CONDITIONS:
        condition_dir = post_root / "scenes" / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        refs = {}
        for scene_id in SCENE_IDS:
            original = plan["test_scenes"][str(scene_id)]
            transformed = transform_scene(scenario_arrays(original), condition, scene_id)
            path = condition_dir / f"scene_{scene_id}.npz"
            if path.exists():
                with np.load(path) as saved:
                    if set(saved.files) != set(transformed) or any(
                        not np.array_equal(saved[key], value) for key, value in transformed.items()
                    ):
                        raise RuntimeError(f"已有条件场景数组不匹配：{path}")
            else:
                runner._atomic_npz(path, transformed)
            refs[str(scene_id)] = {
                "path": str(path), "sha256": runner.sha256_file(path),
                "base_path": original["path"], "base_sha256": original["sha256"],
            }
        _put_immutable_json(condition_dir / "manifest.json", {
            "condition": condition, "training_commit": TRAIN_COMMIT,
            "post_commit": provenance["post_commit"],
            "scene_count": 20, "action_seeds": list(ACTION_SEEDS),
            "transformation": ("rate x0.75" if condition == "demand_075" else
                               "rate x1.25" if condition == "demand_125" else
                               "within-group permutation, seed=scene_id+100000"),
            "scenes": refs,
        })
        references[condition] = refs
    return references


class EvaluationAgent:
    """Use the exact SAC action sampler without creating training optimizers."""

    def __init__(self, model, device):
        module_name, hidden = runner.MODELS[model]
        module = importlib.import_module("satellite_rl.agents." + module_name)
        self.actor = module.PolicyNet(204, hidden, 100, 10, 10).to(device)
        self.device = device
        self._take_action = module.SAC.take_action

    def take_action(self, state):
        return self._take_action(self, state)


def load_final_agent(job, final, device):
    agent = EvaluationAgent(job["model"], device)
    load_network_checkpoint(agent.actor, final["actor"], algorithm=job["run_id"],
                            map_location=device, evaluation_mode=True)
    return agent


def _task_config(kind, label, run_id, actor_path, scene_refs, provenance, *, old_env_sha=None):
    return {
        "kind": kind, "label": label, "run_id": run_id,
        "training_commit": TRAIN_COMMIT, "post_commit": provenance["post_commit"],
        "matched_source_sha256": provenance["matched_file_sha256"],
        "actor_path": str(actor_path), "actor_sha256": runner.sha256_file(actor_path),
        "scene_sha256": {key: value["sha256"] for key, value in scene_refs.items()},
        "scene_ids": list(SCENE_IDS), "action_seeds": list(ACTION_SEEDS),
        "old_environment_sha256": old_env_sha,
    }


def _run_evaluation_task(task_dir, task_id, config, agent, scene_refs,
                         *, environment_module=None):
    """A completed matching result is immutable; failed work resumes in place."""
    task_dir = Path(task_dir)
    manifest_path = task_dir / "manifest.json"
    existed = task_dir.exists()
    if existed and not manifest_path.exists():
        raise RuntimeError(f"有未归档的部分运行且缺少清单：{task_dir}")
    if existed:
        _put_immutable_json(manifest_path, config)
    monitor = RunMonitor(task_dir, task_id, resume=existed)
    if not existed:
        _put_immutable_json(manifest_path, config)
    original_factory = runner._new_environment
    try:
        if environment_module is not None:
            def old_factory(beam_info):
                frequency = environment_module.SatelliteFreqState()
                current = environment_module.CurrentBeamState(beam_info.beam_info)
                environment = environment_module.Env(frequency, beam_info, current)
                return environment, environment.reset(beam_info)
            runner._new_environment = old_factory
        monitor.write_status(stage="evaluating", scene_count=20,
                             action_seeds=list(ACTION_SEEDS), source_commit=TRAIN_COMMIT)
        result = runner.evaluate(
            agent, scene_refs, ACTION_SEEDS, "final_test", 400000, monitor,
            task_dir / "evaluation", include_argmax=False,
        )
        monitor.write_status(stage="completed", result_path=str(
            task_dir / "evaluation" / "final_test_400000.json"),
            sampled_si_mean=result["si_mean"], completed_at=utc_now())
        return result
    except Exception as error:
        atomic_json(task_dir / "failure.json", {
            "at": utc_now(), "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "config": config,
        })
        monitor.write_status(stage="failed", failure_path=str(task_dir / "failure.json"))
        raise
    finally:
        runner._new_environment = original_factory
        monitor.close()


def _post_root(archive_root):
    os.environ["SATELLITE_RL_OUTPUT_ROOT"] = str(Path(archive_root).resolve())
    return Path(project_path(f"outputs/experiments/{EXPERIMENT_ID}/post_training_checks"))


def run_conditions(archive_root, frozen_root):
    provenance = source_guard(frozen_root)
    exp_dir, plan, final = require_completed(archive_root, frozen_root)
    runtime = runner.strict_cuda_preflight()
    post_root = _post_root(archive_root)
    references = prepare_condition_scenes(post_root, plan, provenance)
    _put_immutable_json(post_root / "original_reference.json", {
        "training_commit": TRAIN_COMMIT, "post_commit": provenance["post_commit"],
        "rule": "原条件直接复用训练终点的100场景考核；此处索引其中81000–81019",
        "results": {run_id: {
            "path": item["original_test"], "sha256": item["original_test_sha256"]}
            for run_id, item in final.items()},
    })
    _put_immutable_json(post_root / "runtime.json", {
        "post_commit": provenance["post_commit"], "training_commit": TRAIN_COMMIT,
        "runtime": runtime, "source_hashes": provenance["matched_file_sha256"],
    })
    for condition in CONDITIONS:
        for job in plan["jobs"]:
            run_id = job["run_id"]
            agent = load_final_agent(job, final[run_id], torch.device("cuda"))
            task_dir = post_root / "conditions" / condition / run_id
            config = _task_config("condition", condition, run_id,
                                  final[run_id]["actor"], references[condition], provenance)
            _run_evaluation_task(task_dir, f"{condition}_{run_id}", config,
                                 agent, references[condition])
            del agent


def legacy_environment_source(post_root):
    """Save the actual pre-correction environment source as an immutable input."""
    root = Path(__file__).resolve().parents[3]
    revision = _git(root, "rev-parse", "c4963ba^")
    if revision != OLD_ENV_COMMIT:
        raise RuntimeError(f"旧环境父提交变化：{revision}")
    contents = subprocess.run(
        ["git", "show", OLD_ENV_COMMIT + ":src/satellite_rl/envs/satellite.py"],
        cwd=root, check=True, capture_output=True,
    ).stdout
    path = Path(post_root) / "sources" / "satellite_before_c4963ba.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != contents:
            raise RuntimeError(f"旧环境源码快照不匹配：{path}")
    else:
        path.write_bytes(contents)
    spec = importlib.util.spec_from_file_location("satellite_rl_pre_c4963ba", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path, runner.sha256_file(path)


def run_environment_pair(archive_root, frozen_root):
    provenance = source_guard(frozen_root)
    exp_dir, plan, _ = require_completed(archive_root, frozen_root)
    runtime = runner.strict_cuda_preflight()
    post_root = _post_root(archive_root)
    old_module, old_source, old_sha = legacy_environment_source(post_root)
    old_actor_path = Path(frozen_root) / "models" / "pretrained" / "sac_transformer" / "actor.pth"
    if not old_actor_path.is_file():
        raise FileNotFoundError(f"旧Transformer权重缺失，对照不执行：{old_actor_path}")
    agent = EvaluationAgent("transformer", torch.device("cuda"))
    load_network_checkpoint(agent.actor, old_actor_path, algorithm="pretrained-transformer",
                            map_location="cuda", evaluation_mode=True)
    references = {str(scene): plan["test_scenes"][str(scene)] for scene in SCENE_IDS}
    _put_immutable_json(post_root / "environment_comparison" / "source_manifest.json", {
        "training_commit": TRAIN_COMMIT, "post_commit": provenance["post_commit"],
        "old_environment_commit": OLD_ENV_COMMIT,
        "old_environment_source": str(old_source), "old_environment_sha256": old_sha,
        "old_actor": str(old_actor_path), "old_actor_sha256": runner.sha256_file(old_actor_path),
        "scenes": references, "action_seeds": list(ACTION_SEEDS), "runtime": runtime,
        "interpretation": "旧错误环境仅说明计算变化及旧策略适应，不参与新环境模型排名",
    })
    for label, environment in (("before_c4963ba", old_module), ("corrected_r2", None)):
        task_dir = post_root / "environment_comparison" / label
        config = _task_config("environment", label, "pretrained_transformer",
                              old_actor_path, references, provenance,
                              old_env_sha=old_sha if environment is not None else None)
        _run_evaluation_task(task_dir, "pretrained_transformer_" + label,
                             config, agent, references, environment_module=environment)


TIMING_BEAM_INDICES = (0, 25, 50, 75)
TIMING_WARMUP = 40
TIMING_REPEATS = 240


def make_timing_states(plan, post_root, scene_ids=SCENE_IDS):
    """Real observations reached with one fixed legal reference allocation."""
    states, observed_scene_ids, beam_indices = [], [], []
    requested, actual = [], []
    for scene_id in scene_ids:
        reference = plan["test_scenes"][str(scene_id)]
        beam_info = runner._load_scene(reference["path"])
        env, state = runner._new_environment(beam_info)
        for index in range(max(TIMING_BEAM_INDICES) + 1):
            if env.beam_idx != index:
                raise RuntimeError("参考分配提前终止，计时状态无法重建")
            if index in TIMING_BEAM_INDICES:
                states.append(np.array(state, dtype=np.float64, copy=True))
                observed_scene_ids.append(scene_id)
                beam_indices.append(index)
            group, frequency, slots, power_w = index % 8, (index // 8) * 5, 5, 20.0
            if frequency + slots > 100 or env.total_power + power_w > 3000:
                raise RuntimeError("参考分配违反频率或总功率预算")
            occupied = env.satellite_freq_state.freq_pool_real[group, frequency:frequency + slots]
            if np.any(occupied):
                raise RuntimeError("参考分配占用已有频槽")
            action = [group, frequency, slots, 10 * np.log10(power_w)]
            applied = action.copy()
            state, _, done, info = env.step(applied)
            requested.append([scene_id, index, *action])
            actual.append([scene_id, index, *applied])
            if (done or env.overload or int(np.asarray(info["out"]).item())
                    or int(np.asarray(info["occupy"]).item()) or applied[2] != slots):
                raise RuntimeError(f"参考分配在场景{scene_id}第{index}束被裁剪/提前终止")
    values = np.stack(states)
    if values.shape != (4 * len(scene_ids), 204) or not np.isfinite(values).all():
        raise RuntimeError("参考状态不是每场景4条有限204维实际观测")
    arrays = {
        "states_float64": values, "states_float32": values.astype(np.float32),
        "scene_id": np.asarray(observed_scene_ids), "before_beam_index": np.asarray(beam_indices),
        "requested_reference_actions": np.asarray(requested, dtype=np.float64),
        "actual_reference_actions": np.asarray(actual, dtype=np.float64),
    }
    path = Path(post_root) / "timing" / "reference_states.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[key], value) for key, value in arrays.items()
            ):
                raise RuntimeError("已有计时参考状态或动作与当前生成结果不一致")
    else:
        runner._atomic_npz(path, arrays)
    return path, runner.sha256_file(path), arrays["states_float32"]


def _gpu_snapshot():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    )
    pieces = result.stdout.strip().splitlines()[0].split(",")
    snapshot = {"at": utc_now(), "used_mib": int(pieces[0]),
                "free_mib": int(pieces[1]), "utilization_percent": int(pieces[2])}
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
    )
    snapshot["compute_process_query"] = (processes.stdout.strip() if processes.returncode == 0
                                           else processes.stderr.strip())
    return snapshot


def _distribution(milliseconds):
    return {"mean_ms": statistics.mean(milliseconds),
            "sample_std_ms": statistics.stdev(milliseconds),
            "median_ms": float(np.median(milliseconds)),
            "p90_ms": float(np.quantile(milliseconds, 0.9)),
            "min_ms": min(milliseconds), "max_ms": max(milliseconds)}


def _time_actor(agent, states):
    device_states = torch.as_tensor(states, dtype=torch.float32, device="cuda")
    agent.actor.eval()
    with torch.no_grad():
        for index in range(TIMING_WARMUP):
            position = index % len(states)
            agent.actor(device_states[position:position + 1])
            agent.take_action(states[position])
        torch.cuda.synchronize()
        forward, end_to_end = [], []
        for index in range(TIMING_REPEATS):
            position = index % len(states)
            torch.cuda.synchronize()
            started = time.perf_counter()
            agent.actor(device_states[position:position + 1])
            torch.cuda.synchronize()
            forward.append((time.perf_counter() - started) * 1000)
        for index in range(TIMING_REPEATS):
            position = index % len(states)
            torch.cuda.synchronize()
            started = time.perf_counter()
            agent.take_action(states[position])
            torch.cuda.synchronize()
            end_to_end.append((time.perf_counter() - started) * 1000)
    return {"forward_device_tensor": _distribution(forward),
            "take_action_host_to_action": _distribution(end_to_end)}


def run_timing(archive_root, frozen_root):
    provenance = source_guard(frozen_root)
    _, plan, final = require_completed(archive_root, frozen_root)
    runtime = runner.strict_cuda_preflight()
    post_root = _post_root(archive_root)
    path, state_sha, states = make_timing_states(plan, post_root)
    config = {
        "training_commit": TRAIN_COMMIT, "post_commit": provenance["post_commit"],
        "matched_source_sha256": provenance["matched_file_sha256"],
        "state_path": str(path), "state_sha256": state_sha,
        "state_shape": list(states.shape), "scene_ids": list(SCENE_IDS),
        "before_beam_indices": list(TIMING_BEAM_INDICES),
        "reference_allocation": "group=index%8; frequency=(index//8)*5; slots=5; power=20W",
        "warmup_per_scope": TIMING_WARMUP, "repeats_per_scope": TIMING_REPEATS,
        "timing_action_seed": 93001,
        "dtype": "float32", "runtime": runtime,
        "scope_forward": "GPU-resident one-state tensor through actor; synchronized before/after",
        "scope_take_action": "host state through SAC.take_action to returned Python ints; synchronized before/after",
        "interpretation": "统一实际观测的推理成本，不是训练或策略成绩",
    }
    timing_dir = post_root / "timing"
    _put_immutable_json(timing_dir / "manifest.json", config)
    results = {}
    for job in plan["jobs"]:
        run_id = job["run_id"]
        result_path = timing_dir / "runs" / (run_id + ".json")
        run_config = {"run_id": run_id, "actor_sha256": final[run_id]["actor_sha256"],
                      "state_sha256": state_sha, "post_commit": provenance["post_commit"],
                      "warmup": TIMING_WARMUP, "repeats": TIMING_REPEATS}
        if result_path.exists():
            result = _load_json(result_path)
            if result["config"] != run_config:
                raise RuntimeError(f"已有计时结果配置不一致：{result_path}")
        else:
            agent = load_final_agent(job, final[run_id], torch.device("cuda"))
            try:
                torch.manual_seed(93001)
                result = {"config": run_config,
                          "parameters": sum(parameter.numel() for parameter in agent.actor.parameters()),
                          "gpu_before": _gpu_snapshot(), "timing": _time_actor(agent, states),
                          "gpu_after": _gpu_snapshot(), "completed_at": utc_now()}
                result_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(result_path, result)
            except Exception as error:
                atomic_json(timing_dir / "runs" / (run_id + ".failure.json"), {
                    "at": utc_now(), "config": run_config, "type": type(error).__name__,
                    "message": str(error), "traceback": traceback.format_exc(),
                })
                raise
            finally:
                del agent
                torch.cuda.empty_cache()
        results[run_id] = {"path": str(result_path),
                           "sha256": runner.sha256_file(result_path)}
    _put_immutable_json(timing_dir / "result_index.json", {
        "manifest": str(timing_dir / "manifest.json"),
        "manifest_sha256": runner.sha256_file(timing_dir / "manifest.json"),
        "runs": results,
    })


def build_parser():
    parser = argparse.ArgumentParser(description="十八组训练完成后的固定补充评价")
    parser.add_argument("command", choices=("verify", "conditions", "environment", "timing"))
    parser.add_argument("--archive-root", default="D:/paper")
    parser.add_argument("--frozen-root", default="D:/paper_worktrees/frozen-sac-20260924-r2")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "verify":
        source_guard(args.frozen_root)
        _, plan, _ = require_completed(args.archive_root, args.frozen_root)
        print(f"已核对 {len(plan['jobs'])} 组固定预算终点权重及完整考核")
    elif args.command == "conditions":
        run_conditions(args.archive_root, args.frozen_root)
    elif args.command == "environment":
        run_environment_pair(args.archive_root, args.frozen_root)
    else:
        run_timing(args.archive_root, args.frozen_root)


if __name__ == "__main__":
    main()
