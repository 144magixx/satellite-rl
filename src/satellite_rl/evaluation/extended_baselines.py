"""Same-scene four-color and random baselines for the frozen SAC comparison.

Run this file directly with PYTHONPATH pointing exclusively at the frozen
training checkout. The evaluation file itself lives in a separate worktree.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

import numpy as np


TRAINING_COMMIT = "36c14d74cbe6b9cf7410ee0595ee6d73083afdc8"
SCENE_IDS = tuple(range(81000, 81100))
ACTION_SEEDS = (91001, 91002, 91003)
POLICIES = {"four_color": "satellite_rl.baselines.four_color.four_color_policy",
            "random": "satellite_rl.baselines.random_policy action rule; "
                      "numpy.random.RandomState(MT19937) independently seeded per scene/action_seed"}
SOURCE_FILES = (
    "src/satellite_rl/baselines/four_color.py",
    "src/satellite_rl/baselines/random_policy.py",
    "src/satellite_rl/envs/satellite.py",
    "src/satellite_rl/evaluation/terminal_metrics.py",
    "src/satellite_rl/training/experiment_runner.py",
)
SUMMARY_FIELDS = (
    "si_sum", "si_mean", "sgm_mean", "throughput_bps", "total_power_w",
    "served_beams", "power_exceeded", "completed", "occupancy_fraction",
    "zero_slot_rate", "allocated_slots", "spectral_efficiency",
    "interference_beams", "in_range_interference_beams",
    "requested_slots_total", "actual_slots_total", "out_slots_total",
    "occupied_slots_total", "clipped_steps", "zero_slot_steps",
    "requested_power_w_total", "group_switch_steps",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def random_action(env, rng):
    """The original random-policy draw order and 5 W power grid."""
    group = env.beam_idx % 8
    freq = int(rng.randint(0, 100))
    slots = int(rng.randint(1, 11))
    power_index = int(rng.randint(0, 10))
    return [group, freq, slots, float(10 * np.log10((power_index + 1) * 5))]


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _stats(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("A summary needs at least two finite independent scenes")
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def summarize(rows, *, scene_ids=SCENE_IDS, action_seeds=ACTION_SEEDS):
    """Random action repetitions are averaged within each scene first."""
    expected = {(policy, scene, seed)
                for scene in scene_ids
                for policy in POLICIES
                for seed in ((None,) if policy == "four_color" else action_seeds)}
    by_key = {(row["policy"], row["scene_id"], row["action_seed"]): row for row in rows}
    if len(by_key) != len(rows) or set(by_key) != expected:
        raise ValueError("Incomplete or duplicated policy/scene/action-seed identity")
    summary = {}
    for policy in POLICIES:
        scene_values = {field: [] for field in SUMMARY_FIELDS}
        for scene in scene_ids:
            keys = [(policy, scene, None)] if policy == "four_color" else [
                (policy, scene, seed) for seed in action_seeds]
            for field in SUMMARY_FIELDS:
                scene_values[field].append(float(np.mean([by_key[key][field] for key in keys])))
        summary[policy] = {field: _stats(values) for field, values in scene_values.items()}
        summary[policy]["power_exceeded_rate"] = summary[policy]["power_exceeded"]["mean"]
        summary[policy]["completed_rate"] = summary[policy]["completed"]["mean"]
    return summary


def validate_row(row, *, total_beams=100):
    rates = np.asarray(row["final_rate_bps_per_beam"], dtype=np.float64)
    required = np.asarray(row["required_rate_bps_per_beam"], dtype=np.float64)
    satisfaction = np.asarray(row["satisfaction_per_beam"], dtype=np.float64)
    sgm = np.asarray(row["sgm_per_beam"], dtype=np.float64)
    if any(array.shape != (total_beams,) for array in (rates, required, satisfaction, sgm)):
        raise ValueError("Terminal arrays must include every target beam")
    if not (np.isfinite(rates).all() and np.isfinite(required).all()
            and np.isfinite(satisfaction).all() and np.isfinite(sgm).all()
            and np.all(required > 0) and np.all(rates >= 0)):
        raise ValueError("Invalid terminal rates or requirements")
    expected = np.minimum(rates / required, 1.0)
    if not np.allclose(satisfaction, expected, rtol=1e-12, atol=1e-12):
        raise ValueError("Terminal SI differs from independent rate/requirement calculation")
    if not math.isclose(float(row["si_sum"]), float(expected.sum()), rel_tol=1e-12):
        raise ValueError("Terminal SI sum differs from full target arrays")
    if not math.isclose(float(row["si_mean"]), float(expected.mean()), rel_tol=1e-12):
        raise ValueError("Terminal SI mean differs from full target arrays")
    if not math.isclose(float(row["sgm_mean"]), float(sgm.mean()), rel_tol=1e-12):
        raise ValueError("Terminal SGM mean differs from full target arrays")
    if not math.isclose(float(row["throughput_bps"]), float(rates.sum()), rel_tol=1e-12):
        raise ValueError("Throughput differs from final beam rates")
    if not math.isclose(float(row["total_power_w"]), float(row["requested_power_w_total"]),
                        rel_tol=1e-11, abs_tol=1e-7):
        raise ValueError("Power ledger differs from requested powers")
    if row["requested_slots_total"] != (row["actual_slots_total"]
                                        + row["out_slots_total"]
                                        + row["occupied_slots_total"]):
        raise ValueError("Frequency-slot request ledger does not conserve slots")
    if row["actual_slots_total"] != row["allocated_slots"]:
        raise ValueError("Allocated frequency slots differ from successful requests")
    if row["steps"] != row["served_beams"] or row["completed"] != (row["steps"] == total_beams):
        raise ValueError("Processed beam count is inconsistent")
    if row["power_exceeded"] != (row["total_power_w"] > 3000):
        raise ValueError("Power threshold flag is inconsistent")


def _git_head_clean(root):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if dirty:
        raise RuntimeError(f"Evaluation source is not clean: {root}")
    return head


def _source_guard(frozen_root):
    import satellite_rl
    imported = Path(satellite_rl.__file__).resolve()
    if not imported.is_relative_to(frozen_root / "src"):
        raise RuntimeError(f"satellite_rl imported from wrong checkout: {imported}")
    if _git_head_clean(frozen_root) != TRAINING_COMMIT:
        raise RuntimeError("Frozen training checkout changed")
    evaluation_root = Path(__file__).resolve().parents[3]
    return {"evaluation_root": str(evaluation_root),
            "evaluation_commit": _git_head_clean(evaluation_root),
            "evaluation_script": str(Path(__file__).resolve()),
            "evaluation_script_sha256": sha256_file(__file__),
            "training_root": str(frozen_root),
            "training_commit": TRAINING_COMMIT,
            "imported_package": str(imported),
            "source_sha256": {str(frozen_root / name): sha256_file(frozen_root / name)
                              for name in SOURCE_FILES}}


def _identity(policy, scene_id, action_seed):
    seed = "none" if action_seed is None else str(action_seed)
    return f"{policy}_scene_{scene_id}_action_{seed}"


def _evaluate_one(policy, scene_id, action_seed, scene_path, scene_sha, out_dir):
    from satellite_rl.baselines.four_color import four_color_policy
    from satellite_rl.evaluation.terminal_metrics import terminal_metrics, terminal_beam_arrays
    from satellite_rl.training.experiment_runner import _load_scene, _new_environment
    from satellite_rl.training.run_monitor import atomic_json, replace_with_retry

    identity = _identity(policy, scene_id, action_seed)
    result_path = out_dir / "rollouts" / f"{identity}.json"
    trace_path = out_dir / "traces" / f"{identity}.jsonl.gz"
    if result_path.exists():
        row = json.loads(result_path.read_text(encoding="utf-8"))
        if ((row["policy"], row["scene_id"], row["action_seed"], row["scene_sha256"])
                != (policy, scene_id, action_seed, scene_sha)):
            raise RuntimeError(f"Existing rollout identity changed: {result_path}")
        if row["trace_path"] != str(trace_path) or sha256_file(trace_path) != row["trace_sha256"]:
            raise RuntimeError(f"Existing trace SHA changed: {trace_path}")
        validate_row(row)
        return row
    if trace_path.exists():
        raise RuntimeError(f"Orphan trace exists without a validated rollout: {trace_path}")

    env, state = _new_environment(_load_scene(scene_path))
    rng = np.random.RandomState(action_seed) if policy == "random" else None
    ledger = {key: 0 for key in ("requested_slots_total", "actual_slots_total",
                                  "out_slots_total", "occupied_slots_total",
                                  "clipped_steps", "zero_slot_steps", "group_switch_steps")}
    ledger["requested_power_w_total"] = 0.0
    requested_actions = []
    actual_actions = []
    episode_return = 0.0
    done = False
    temp = trace_path.with_name(trace_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with gzip.open(temp, "wt", encoding="utf-8") as trace:
            while not done:
                beam_index = env.beam_idx
                requested = (four_color_policy(env) if policy == "four_color"
                             else random_action(env, rng))
                requested = [int(requested[0]), int(requested[1]), int(requested[2]),
                             float(requested[3])]
                actual = requested.copy()
                state, reward, done, info = env.step(actual)
                requested_actions.append(requested)
                actual_actions.append(actual)
                episode_return += float(reward)
                ledger["requested_slots_total"] += requested[2]
                ledger["actual_slots_total"] += actual[2]
                ledger["out_slots_total"] += int(np.asarray(info["out"]).item())
                ledger["occupied_slots_total"] += int(np.asarray(info["occupy"]).item())
                ledger["clipped_steps"] += int(actual[2] < requested[2])
                ledger["zero_slot_steps"] += int(actual[2] == 0)
                ledger["group_switch_steps"] += int(requested[0] != beam_index % 8)
                ledger["requested_power_w_total"] += float(10 ** (requested[3] / 10))
                trace.write(json.dumps({"policy": policy, "scene_id": scene_id,
                                        "action_seed": action_seed,
                                        "beam_id": beam_index, "requested": requested,
                                        "actual": actual, "reward": float(reward),
                                        "info": _jsonable(info)},
                                       ensure_ascii=False, allow_nan=False) + "\n")
        replace_with_retry(temp, trace_path)
    finally:
        temp.unlink(missing_ok=True)

    metrics = terminal_metrics(env)
    arrays = terminal_beam_arrays(env)
    row = {"policy": policy, "scene_id": scene_id, "scene_path": str(scene_path),
           "scene_sha256": scene_sha, "action_seed": action_seed,
           "return_sum": episode_return, "steps": len(requested_actions), **metrics,
           **ledger, "final_rate_bps_per_beam": arrays["rates_bps"].tolist(),
           "required_rate_bps_per_beam": arrays["required_bps"].tolist(),
           "satisfaction_per_beam": arrays["satisfaction"].tolist(),
           "sgm_per_beam": arrays["sgm"].tolist(),
           "requested_action_history": requested_actions,
           "actual_action_history": actual_actions,
           "requested_actual_history": np.asarray(env.history_beam).tolist(),
           "trace_path": str(trace_path), "trace_sha256": sha256_file(trace_path)}
    validate_row(row)
    atomic_json(result_path, row)
    return row


def run(archive_root, frozen_root, output_dir):
    archive_root = Path(archive_root).resolve()
    frozen_root = Path(frozen_root).resolve()
    output_dir = Path(output_dir).resolve()
    source = _source_guard(frozen_root)
    from satellite_rl.training.run_monitor import atomic_json, replace_with_retry

    scenes_dir = archive_root / "outputs" / "experiments" / "sac_comparison_20260924_r2" / "scenes" / "test"
    scene_refs = {scene: {"path": str(scenes_dir / f"scene_{scene}.npz"),
                          "sha256": sha256_file(scenes_dir / f"scene_{scene}.npz")}
                  for scene in SCENE_IDS}
    plan = {**source, "archive_root": str(archive_root),
            "scenes_dir": str(scenes_dir), "scene_ids": list(SCENE_IDS),
            "scene_sha256": {str(scene): scene_refs[scene] for scene in SCENE_IDS},
            "action_seeds": list(ACTION_SEEDS),
            "policies": POLICIES, "random_generator": "numpy.random.RandomState(MT19937)",
            "random_draw_order": "freq randint(0,100), slots randint(1,11), power_index randint(0,10)",
            "four_color_group_rule": "Original policy may use beam_idx%8^1; SAC has fixed beam_idx%8",
            "four_color_slot_rule": "Original 10-row[freq] estimate, with actual contiguous clipping by Env.step",
            "evaluation": "one deterministic four_color and three independently seeded random rollouts per saved scene"}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rollouts").mkdir(exist_ok=True)
    (output_dir / "traces").mkdir(exist_ok=True)
    plan_path = output_dir / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("Existing output plan differs from current source or scenes")
    else:
        atomic_json(plan_path, plan)
    rows = []
    for policy in POLICIES:
        for scene in SCENE_IDS:
            seeds = (None,) if policy == "four_color" else ACTION_SEEDS
            for action_seed in seeds:
                row = _evaluate_one(policy, scene, action_seed,
                                    Path(scene_refs[scene]["path"]),
                                    scene_refs[scene]["sha256"], output_dir)
                rows.append(row)
                atomic_json(output_dir / "status.json", {
                    "stage": "running", "completed_rollouts": len(rows),
                    "expected_rollouts": len(SCENE_IDS) * (1 + len(ACTION_SEEDS)),
                    "last_identity": _identity(policy, scene, action_seed),
                    "evaluation_commit": source["evaluation_commit"],
                    "training_commit": TRAINING_COMMIT})
                if len(rows) % 10 == 0:
                    print(f"completed {len(rows)}/400: {policy} scene {scene}", flush=True)
    summary = summarize(rows)
    canonical_path = output_dir / "per_scene_results.jsonl"
    temp = canonical_path.with_name(canonical_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temp, canonical_path)
    finally:
        temp.unlink(missing_ok=True)
    result = {**plan, "python": sys.executable, "python_version": platform.python_version(),
              "numpy_version": np.__version__, "scene_count": len(SCENE_IDS),
              "rollout_count": len(rows), "per_scene_results_path": str(canonical_path),
              "per_scene_results_sha256": sha256_file(canonical_path), "summary": summary,
              "limitations": ["one deterministic four-color rollout per scene",
                              "random action seeds are repeated within each scene then averaged per scene",
                              "original four-color can switch polarization group unlike SAC",
                              "four-color estimates 10-row[freq] as free slots; Env clips contiguous slots",
                              "equal numeric action seeds do not imply the random policy and SAC sample identical actions",
                              "2025 one-episode files are historical and not used"]}
    atomic_json(output_dir / "config_and_summary.json", result)
    atomic_json(output_dir / "status.json", {
        "stage": "completed", "completed_rollouts": len(rows),
        "expected_rollouts": len(rows), "per_scene_results_sha256": sha256_file(canonical_path),
        "config_and_summary_sha256": sha256_file(output_dir / "config_and_summary.json"),
        "evaluation_commit": source["evaluation_commit"],
        "training_commit": TRAINING_COMMIT})
    return result


def main():
    parser = argparse.ArgumentParser(description="100 saved-scene four-color and random baselines")
    parser.add_argument("--archive-root", default=r"D:\paper")
    parser.add_argument("--frozen-root", default=r"D:\paper_worktrees\frozen-sac-20260924-r2")
    parser.add_argument("--output-dir", default=(
        r"D:\paper\outputs\verification\baseline_recheck_sac_r2_20260925_extended"))
    args = parser.parse_args()
    run(args.archive_root, args.frozen_root, args.output_dir)


if __name__ == "__main__":
    main()
