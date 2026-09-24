"""Strict read-only gate for the six approved seed-51001 SAC runs.

The original suite plan retains eighteen jobs, including stopped seeds. This
gate reads each selected run's own terminal artifacts and never treats stale
suite_status.json or suite.lock as completion evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from satellite_rl.training import experiment_runner as runner


EXPERIMENT_ID = "sac_comparison_20260924_r2"
TRAIN_COMMIT = "36c14d74cbe6b9cf7410ee0595ee6d73083afdc8"
SEED = 51001
MODELS = tuple(runner.MODELS)
FORMULAS = ("legacy", "corrected")
RUN_IDS = {f"{model}_{formula}_s{SEED}" for model in MODELS for formula in FORMULAS}
ALL_ORIGINAL_RUN_IDS = {
    f"{model}_{formula}_s{seed}"
    for seed in runner.TRAIN_SEEDS for model in MODELS for formula in FORMULAS
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _frozen_source(frozen_root):
    root = Path(frozen_root).resolve()
    if subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                      text=True, check=True).stdout.strip() != TRAIN_COMMIT:
        raise RuntimeError("训练冻结源码提交不符")
    if subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                      text=True, check=True).stdout.strip():
        raise RuntimeError("训练冻结源码目录含未提交修改")
    return root


def _result(path, checkpoint, test_scenes, action_seeds):
    data = _read(path)
    expected = {(str(scene), "sample", str(seed))
                for scene in test_scenes for seed in action_seeds}
    expected.update((str(scene), "argmax", "argmax") for scene in test_scenes)
    rows = data.get("rows", [])
    actual = [(str(row["scene_id"]), row["action_mode"], str(row["action_seed"]))
              for row in rows]
    if (data.get("checkpoint") != checkpoint or data.get("env_step") != 400000
            or data.get("include_argmax") is not True
            or data.get("scene_references") != test_scenes
            or data.get("action_seeds") != action_seeds
            or len(rows) != 400 or len(set(actual)) != 400 or set(actual) != expected):
        raise RuntimeError(f"{checkpoint} 未包含100场景×3采样及100个最大概率动作：{path}")
    for path_key, sha_key in (("arrays_path", "arrays_sha256"),
                              ("trace_path", "trace_sha256")):
        raw = Path(data[path_key])
        if not raw.is_file() or runner.sha256_file(raw) != data[sha_key]:
            raise RuntimeError(f"{checkpoint} 原始数组或逐步动作校验失败：{raw}")
    return data


def _checkpoint_pair(path, actor_path, config_hash, *, final):
    path, actor_path = Path(path), Path(actor_path)
    if not path.is_file() or not actor_path.is_file():
        raise FileNotFoundError(f"缺少完整断点或独立Actor权重：{path} / {actor_path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor = torch.load(actor_path, map_location="cpu", weights_only=True)
    progress = checkpoint.get("progress", {})
    step = progress.get("env_step")
    if (checkpoint.get("format_version") != 1
            or checkpoint.get("config_hash") != config_hash
            or not isinstance(step, int) or not (0 < step <= 400000)
            or (final and step != 400000)
            or set(checkpoint.get("agent", {}).get("networks", {})) != set(runner.NETWORKS)
            or set(checkpoint["agent"].get("optimizers", {})) != set(runner.OPTIMIZERS)
            or not torch.isfinite(checkpoint["agent"]["log_alpha"]).all()
            or checkpoint.get("replay_capacity") != 20000
            or not isinstance(checkpoint.get("replay"), list)
            or not isinstance(checkpoint.get("rng"), dict)):
        raise RuntimeError(f"断点缺少完整训练状态或步数不符：{path}")
    saved_actor = checkpoint["agent"]["networks"]["actor"]
    if set(actor) != set(saved_actor) or any(
        not torch.equal(value, saved_actor[key]) for key, value in actor.items()
    ):
        raise RuntimeError(f"独立Actor与完整断点不一致：{actor_path}")
    return {"checkpoint": str(path), "checkpoint_sha256": runner.sha256_file(path),
            "actor": str(actor_path), "actor_sha256": runner.sha256_file(actor_path),
            "checkpoint_step": step}


def _completed_run(exp_dir, archive_root, frozen_root, plan, job):
    run_id = job["run_id"]
    run_dir = exp_dir / run_id
    status = _read(run_dir / "status.json")
    manifest = _read(run_dir / "manifest.json")
    config = manifest["config"]
    runtime = config["runtime"]
    if (status.get("stage") != "completed" or status.get("env_step") != 400000
            or status.get("update_step") != 398000
            or config.get("budget_env_steps") != 400000
            or config.get("model") != job["model"]
            or config.get("formula") != job["formula"]
            or config.get("train_seed") != SEED
            or runtime.get("git_commit") != TRAIN_COMMIT
            or Path(runtime.get("source_root", "")).resolve() != frozen_root
            or not Path(runtime.get("package_file", "")).resolve().is_relative_to(frozen_root / "src")
            or not runtime.get("torch_cuda")
            or "dqn" not in (part.lower() for part in Path(runtime.get("python", "")).parts)):
        raise RuntimeError(f"{run_id} 未以固定DQN/CUDA源码完成400000步和398000更新")
    model_dir = archive_root / "models" / "runs" / EXPERIMENT_ID / run_id
    final = _checkpoint_pair(model_dir / "final.pt", model_dir / "final_actor.pt",
                             manifest["config_hash"], final=True)
    best = _checkpoint_pair(model_dir / "best.pt", model_dir / "best_actor.pt",
                            manifest["config_hash"], final=False)
    result_paths = {kind: run_dir / "evaluation" / f"{kind}_400000.json"
                    for kind in ("final_test", "best_test")}
    for kind, path in result_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{run_id} 缺少{kind}原始考核：{path}")
        _result(path, kind, plan["test_scenes"], plan["action_seeds"])
    return {"job": job, **final,
            "best_checkpoint": best["checkpoint"],
            "best_checkpoint_sha256": best["checkpoint_sha256"],
            "best_checkpoint_step": best["checkpoint_step"],
            "best_actor": best["actor"], "best_actor_sha256": best["actor_sha256"],
            "original_test": str(result_paths["final_test"]),
            "original_test_sha256": runner.sha256_file(result_paths["final_test"]),
            "best_test": str(result_paths["best_test"]),
            "best_test_sha256": runner.sha256_file(result_paths["best_test"])}


def require_completed(archive_root, frozen_root):
    """Return only six real completed seed-51001 runs and their immutable inputs.

    Returns (old_r2_experiment_directory, six_job_filtered_plan, final_mapping).
    The mapping preserves the original post_training_checks keys and adds best
    checkpoint/Actor/test references. It never writes to the original suite.
    """
    archive_root = Path(archive_root).resolve()
    frozen_root = _frozen_source(frozen_root)
    exp_dir = archive_root / "outputs" / "experiments" / EXPERIMENT_ID
    original = _read(exp_dir / "suite_plan.json")
    jobs = original.get("jobs", [])
    if (len(jobs) != 18 or len({job["run_id"] for job in jobs}) != 18
            or {job["run_id"] for job in jobs} != ALL_ORIGINAL_RUN_IDS
            or original.get("steps") != 400000
            or original["source"]["git_commit"] != TRAIN_COMMIT
            or Path(original["source"]["source_root"]).resolve() != frozen_root
            or len(original.get("test_scenes", {})) != 100
            or original.get("action_seeds") != list(runner.ACTION_SEEDS)):
        raise RuntimeError("原r2十八组计划或固定场景/动作种子不符")
    selected = [job for job in jobs if job["seed"] == SEED]
    if len(selected) != 6 or {job["run_id"] for job in selected} != RUN_IDS:
        raise RuntimeError("seed51001不构成三网络×原/修公式的精确六组")
    # Read each run directly; stopped suite_status and suite.lock are irrelevant.
    for job in selected:
        status = _read(exp_dir / job["run_id"] / "status.json")
        if (status.get("stage") != "completed" or status.get("env_step") != 400000
                or status.get("update_step") != 398000):
            raise RuntimeError(f"六组尚未全部完成：{job['run_id']}")
    for reference in original["test_scenes"].values():
        path = Path(reference["path"])
        if not path.is_file() or runner.sha256_file(path) != reference["sha256"]:
            raise RuntimeError(f"固定考核场景数组改变：{path}")
    completed = []
    final = {}
    for job in selected:
        verified_job = {**job, "suite_plan_state": job["state"],
                        "state": "completed", "pid": None,
                        "completion_source": "run_status_and_artifacts"}
        completed.append(verified_job)
        final[job["run_id"]] = _completed_run(
            exp_dir, archive_root, frozen_root, original, verified_job)
    filtered = {**original, "jobs": completed, "scope": "single_training_seed",
                "training_seed": SEED,
                "source_plan_path": str(exp_dir / "suite_plan.json"),
                "single_seed_output_dir": str(exp_dir / "single_seed_51001")}
    return exp_dir, filtered, final
