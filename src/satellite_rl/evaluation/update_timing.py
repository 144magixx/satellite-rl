"""Fixed-budget CUDA timing of the complete SAC parameter update.

This is a cost measurement on independent copies of final agents. It never
saves a model or changes the approved eighteen training runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch

from satellite_rl.evaluation.post_training_checks import (
    EXPERIMENT_ID, EXPECTED_RUN_IDS, TRAIN_COMMIT, require_completed,
    source_guard,
)
from satellite_rl.training import experiment_runner as runner
from satellite_rl.training.run_monitor import atomic_json, replace_with_retry, utc_now


SOURCE_RUN = "transformer_corrected_s51001"
BATCH_SEED = 93002
UPDATE_SEED = 93003
BATCH_COUNT = 8
BATCH_SIZE = 256
WARMUP = 20
REPEATS = 100


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise RuntimeError(f"已有不同配置的更新计时文件，拒绝覆盖：{path}")
    else:
        atomic_json(path, value)


def _experiment_gpu_children():
    """Inspect live Windows Python commands, excluding this timing process."""
    if os.name != "nt":
        raise RuntimeError("本次固定GPU进程核验只支持Windows主机")
    query = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -match '^python' -and ("
        "($_.CommandLine -match 'satellite_rl.training.experiment_runner' "
        "-and $_.CommandLine -match 'sac_comparison_20260924_r2') -or "
        "$_.CommandLine -match 'satellite_rl.evaluation.post_training_checks') } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    output = subprocess.run(["powershell", "-NoProfile", "-Command", query],
                            capture_output=True, text=True, check=True).stdout.strip()
    if not output:
        return []
    values = json.loads(output)
    return values if isinstance(values, list) else [values]


def require_ready(archive_root, frozen_root):
    """Refuse all CUDA work before every fixed prerequisite is complete."""
    provenance = source_guard(frozen_root)
    exp_dir, plan, final = require_completed(archive_root, frozen_root)
    handoff = _read(exp_dir / "post_training_checks" / "handoff" / "state.json")
    if (handoff.get("training_commit") != TRAIN_COMMIT
            or handoff.get("stage") not in ("awaiting_state_diagnosis", "complete")
            or any(handoff.get("steps", {}).get(step, {}).get("state") != "completed"
                   for step in ("conditions", "environment", "timing", "final_analysis"))):
        raise RuntimeError("现有四步后处理尚未全部完成，拒绝更新计时")
    if any(job["state"] != "completed" or job.get("pid") is not None
           for job in plan["jobs"]):
        raise RuntimeError("本实验仍有训练作业进程记录，拒绝更新计时")
    active = _experiment_gpu_children()
    if active:
        raise RuntimeError(f"本实验GPU子进程仍在运行，拒绝更新计时：{active}")
    return provenance, exp_dir, plan, final


def fixed_batches(replay, *, seed=BATCH_SEED, batch_count=BATCH_COUNT,
                  batch_size=BATCH_SIZE):
    """Choose disjoint indices once, outside every measured update."""
    needed = batch_count * batch_size
    if len(replay) < needed:
        raise RuntimeError(f"训练回放不足{needed}条，不能固定公共计时批次")
    indices = np.random.default_rng(seed).permutation(len(replay))[:needed]
    indices = indices.reshape(batch_count, batch_size)
    transitions = [replay[int(index)] for index in indices.flat]
    arrays = {
        "indices": indices,
        "states": np.stack([np.asarray(item[0], dtype=np.float64) for item in transitions])
        .reshape(batch_count, batch_size, 204),
        "actions": np.asarray([item[1] for item in transitions], dtype=np.int64)
        .reshape(batch_count, batch_size, 3),
        "rewards": np.asarray([item[2] for item in transitions], dtype=np.float64)
        .reshape(batch_count, batch_size),
        "next_states": np.stack([np.asarray(item[3], dtype=np.float64) for item in transitions])
        .reshape(batch_count, batch_size, 204),
        "dones": np.asarray([item[4] for item in transitions], dtype=np.bool_)
        .reshape(batch_count, batch_size),
    }
    if (not np.isfinite(arrays["states"]).all()
            or not np.isfinite(arrays["next_states"]).all()
            or not np.isfinite(arrays["rewards"]).all()
            or not np.isin(arrays["actions"][:, :, 0], np.arange(100)).all()
            or not np.isin(arrays["actions"][:, :, 1:], np.arange(10)).all()):
        raise RuntimeError("公共计时批次包含非有限状态或非法动作")
    return arrays


def _load_or_save_batches(directory, final):
    directory = Path(directory)
    source = final[SOURCE_RUN]
    checkpoint = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
    replay = checkpoint["replay"]
    arrays = fixed_batches(replay)
    path = directory / "fixed_batches.npz"
    if path.exists():
        with np.load(path) as prior:
            if set(prior.files) != set(arrays) or any(
                not np.array_equal(prior[key], value) for key, value in arrays.items()
            ):
                raise RuntimeError("已有公共批次数组与固定种子/回放不一致")
    else:
        runner._atomic_npz(path, arrays)
    return path, runner.sha256_file(path), arrays, len(replay)


def _batch_dicts(arrays):
    return [dict(states=arrays["states"][index], actions=arrays["actions"][index],
                 rewards=arrays["rewards"][index],
                 next_states=arrays["next_states"][index], dones=arrays["dones"][index])
            for index in range(len(arrays["indices"]))]


def time_updates(agent, batches, *, warmup=WARMUP, repeats=REPEATS,
                 synchronize=None, clock=None):
    """Time update conversion, forward/backward, optimizers, alpha, and soft targets."""
    if not batches or repeats < 2 or warmup < 0:
        raise ValueError("预热/重复次数或公共批次无效")
    synchronize = synchronize or torch.cuda.synchronize
    clock = clock or time.perf_counter
    for index in range(warmup):
        batch = batches[index % len(batches)]
        agent.update(batch)
    synchronize()
    milliseconds = []
    for index in range(repeats):
        batch = batches[index % len(batches)]  # Selection excluded from the timer.
        synchronize()
        started = clock()
        agent.update(batch)
        synchronize()
        milliseconds.append((clock() - started) * 1000)
    return milliseconds


def summarize_ms(milliseconds):
    if len(milliseconds) < 2 or not all(np.isfinite(milliseconds)) or min(milliseconds) <= 0:
        raise ValueError("至少两次有限的正耗时才可报告更新速度")
    total_seconds = sum(milliseconds) / 1000
    return {"mean_ms": statistics.mean(milliseconds),
            "median_ms": float(np.median(milliseconds)),
            "p90_ms": float(np.quantile(milliseconds, 0.9)),
            "sample_std_ms": statistics.stdev(milliseconds),
            "updates_per_second": len(milliseconds) / total_seconds}


def _gpu_snapshot():
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    values = gpu.stdout.strip().splitlines()[0].split(",")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return {"at": utc_now(), "used_mib": int(values[0]),
            "free_mib": int(values[1]), "utilization_percent": int(values[2]),
            "compute_process_query": (processes.stdout.strip() if processes.returncode == 0
                                      else processes.stderr.strip())}


def _write_csv(path, fields, rows):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finish_reports(directory, plan, results):
    raw = []
    grouped = {}
    paired = {}
    for job in plan["jobs"]:
        run_id = job["run_id"]
        result = results[run_id]
        for index, elapsed in enumerate(result["raw_milliseconds"]):
            raw.append({"run_id": run_id, "model": job["model"],
                        "formula": job["formula"], "seed": job["seed"],
                        "iteration": index + 1, "batch_index": index % BATCH_COUNT,
                        "milliseconds": elapsed})
        grouped.setdefault((job["model"], job["formula"]), []).append(result["statistics"])
        paired[(job["model"], job["formula"], job["seed"])] = result["statistics"]
    summary = []
    for (model, formula), values in sorted(grouped.items()):
        if len(values) != 3:
            raise RuntimeError(f"{model}/{formula} 缺少三个独立训练种子的计时")
        means = [item["mean_ms"] for item in values]
        throughputs = [item["updates_per_second"] for item in values]
        summary.append({"model": model, "formula": formula, "training_seeds": 3,
                        "mean_update_ms": statistics.mean(means),
                        "std_update_ms_across_seeds": statistics.stdev(means),
                        "mean_updates_per_second": statistics.mean(throughputs),
                        "std_updates_per_second_across_seeds": statistics.stdev(throughputs)})
    _write_csv(directory / "raw_updates.csv",
               ("run_id", "model", "formula", "seed", "iteration", "batch_index", "milliseconds"), raw)
    _write_csv(directory / "summary.csv",
               ("model", "formula", "training_seeds", "mean_update_ms",
                "std_update_ms_across_seeds", "mean_updates_per_second",
                "std_updates_per_second_across_seeds"), summary)
    differences = []
    for model in sorted(runner.MODELS):
        millis = [paired[(model, "corrected", seed)]["mean_ms"]
                  - paired[(model, "legacy", seed)]["mean_ms"]
                  for seed in runner.TRAIN_SEEDS]
        speeds = [paired[(model, "corrected", seed)]["updates_per_second"]
                  - paired[(model, "legacy", seed)]["updates_per_second"]
                  for seed in runner.TRAIN_SEEDS]
        differences.append({"model": model, "paired_training_seeds": 3,
                            "corrected_minus_legacy_mean_ms": statistics.mean(millis),
                            "corrected_minus_legacy_std_ms": statistics.stdev(millis),
                            "corrected_minus_legacy_mean_updates_per_second": statistics.mean(speeds),
                            "corrected_minus_legacy_std_updates_per_second": statistics.stdev(speeds)})
    _write_csv(directory / "formula_differences.csv",
               ("model", "paired_training_seeds", "corrected_minus_legacy_mean_ms",
                "corrected_minus_legacy_std_ms",
                "corrected_minus_legacy_mean_updates_per_second",
                "corrected_minus_legacy_std_updates_per_second"), differences)
    lines = ["# SAC 完整参数更新耗时（独立短测）", "",
             "同一份训练回放的固定批次用于十八个终点模型的独立副本。"
             "每个副本预热20次，随后同步计时100次完整 `agent.update`；"
             "批次选择不计入，状态转移、环境步和回放抽样不计入。", "",
             "| 网络 | 公式 | 三种子单次更新均值 ± 样本标准差 (ms) | 更新速度均值 ± 样本标准差 (次/秒) |",
             "| --- | --- | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['formula']} | "
                     f"{row['mean_update_ms']:.3f} ± {row['std_update_ms_across_seeds']:.3f} | "
                     f"{row['mean_updates_per_second']:.2f} ± "
                     f"{row['std_updates_per_second_across_seeds']:.2f} |")
    lines += ["", "| 网络 | 修正公式减原公式：单次更新耗时均值 ± 三种子样本标准差 (ms) |",
              "| --- | ---: |"]
    for row in differences:
        lines.append(f"| {row['model']} | "
                     f"{row['corrected_minus_legacy_mean_ms']:+.3f} ± "
                     f"{row['corrected_minus_legacy_std_ms']:.3f} |")
    lines += ["", "上表在相同网络、配对训练种子和公共批次下描述两套公式的实测更新成本差值；"
              "原始配对值与更新速度差见 `formula_differences.csv`。"
              "短测仍受 GPU 负载与时钟状态影响，不将单次差值外推为整段训练耗时。", "",
              "计时包含输入张量转换、网络前后向、Adam 更新、温度更新和目标网络软更新；"
              "前后均同步 CUDA。结果只表示此硬件与公共批次下的计算成本，"
              "不代表需求满足率、训练收敛或独占环境步速度。"
              "原始100次毫秒值见 `raw_updates.csv`。", ""]
    (directory / "中文更新耗时.md").write_text("\n".join(lines), encoding="utf-8")


def run_update_timing(archive_root, frozen_root):
    provenance, exp_dir, plan, final = require_ready(archive_root, frozen_root)
    runtime = runner.strict_cuda_preflight()
    directory = exp_dir / "post_training_checks" / "update_timing"
    directory.mkdir(parents=True, exist_ok=True)
    batch_path, batch_sha, arrays, replay_size = _load_or_save_batches(directory, final)
    config = {
        "training_commit": TRAIN_COMMIT, "post_commit": provenance["post_commit"],
        "matched_source_sha256": provenance["matched_file_sha256"],
        "source_replay_run": SOURCE_RUN,
        "source_checkpoint": final[SOURCE_RUN]["checkpoint"],
        "source_checkpoint_sha256": final[SOURCE_RUN]["checkpoint_sha256"],
        "replay_size": replay_size, "batch_seed": BATCH_SEED,
        "update_seed": UPDATE_SEED, "batch_count": BATCH_COUNT,
        "batch_size": BATCH_SIZE, "batch_file": str(batch_path),
        "batch_sha256": batch_sha, "warmup": WARMUP, "repeats": REPEATS,
        "all_final_checkpoint_sha256": {key: value["checkpoint_sha256"]
                                        for key, value in final.items()},
        "all_final_checkpoint_paths": {key: value["checkpoint"]
                                       for key, value in final.items()},
        "all_final_actor_sha256": {key: value["actor_sha256"]
                                   for key, value in final.items()},
        "all_final_actor_paths": {key: value["actor"]
                                  for key, value in final.items()},
        "runtime": runtime,
        "timer_scope": "synchronize; agent.update(saved batch); synchronize; monotonic wall time",
        "excluded": ["environment transition", "replay sampling", "batch selection", "checkpoint I/O"],
        "interpretation": "独立参数更新计算开销短测，不保存学习权重或推断策略性能",
    }
    _immutable_json(directory / "manifest.json", config)
    batches = _batch_dicts(arrays)
    results = {}
    for job in plan["jobs"]:
        run_id = job["run_id"]
        result_path = directory / "runs" / (run_id + ".json")
        run_config = {"run_id": run_id, "model": job["model"],
                      "formula": job["formula"], "seed": job["seed"],
                      "checkpoint_sha256": final[run_id]["checkpoint_sha256"],
                      "actor_sha256": final[run_id]["actor_sha256"],
                      "batch_sha256": batch_sha, "post_commit": provenance["post_commit"],
                      "warmup": WARMUP, "repeats": REPEATS}
        if result_path.exists():
            result = _read(result_path)
            if (result.get("config") != run_config
                    or len(result.get("raw_milliseconds", [])) != REPEATS
                    or any(not np.isfinite(value) or value <= 0
                           for value in result["raw_milliseconds"])
                    or "statistics" not in result):
                raise RuntimeError(f"已有不同配置或不完整的更新计时结果：{result_path}")
        else:
            agent, replay = runner._create_agent(
                job["model"], job["formula"], directory / "temporary_logs" / run_id,
                uuid.uuid4().hex,
            )
            try:
                progress = runner.load_checkpoint(
                    final[run_id]["checkpoint"], agent, replay,
                    _read(exp_dir / run_id / "manifest.json")["config_hash"])
                if progress["env_step"] != 400000:
                    raise RuntimeError(f"{run_id} 不是四十万步终点断点")
                torch.manual_seed(UPDATE_SEED)
                before = _gpu_snapshot()
                milliseconds = time_updates(agent, batches)
                runner._ensure_finite_agent(agent)
                result = {"config": run_config, "raw_milliseconds": milliseconds,
                          "statistics": summarize_ms(milliseconds),
                          "parameters_actor": sum(item.numel() for item in agent.actor.parameters()),
                          "temporary_log_file": str(agent.log_file),
                          "gpu_before": before, "gpu_after": _gpu_snapshot(),
                          "completed_at": utc_now()}
                result_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(result_path, result)
            except Exception as error:
                failure = directory / "runs" / f"{run_id}.failure.{uuid.uuid4().hex}.json"
                atomic_json(failure, {"at": utc_now(), "config": run_config,
                                      "type": type(error).__name__, "message": str(error),
                                      "traceback": traceback.format_exc()})
                raise
            finally:
                del agent, replay
                torch.cuda.empty_cache()
        results[run_id] = result
    _finish_reports(directory, plan, results)
    _immutable_json(directory / "result_index.json", {
        "manifest_sha256": runner.sha256_file(directory / "manifest.json"),
        "runs": {run_id: {"path": str(directory / "runs" / (run_id + ".json")),
                           "sha256": runner.sha256_file(directory / "runs" / (run_id + ".json"))}
                 for run_id in EXPECTED_RUN_IDS},
        "raw_csv_sha256": runner.sha256_file(directory / "raw_updates.csv"),
        "summary_csv_sha256": runner.sha256_file(directory / "summary.csv"),
        "formula_differences_csv_sha256": runner.sha256_file(directory / "formula_differences.csv"),
        "chinese_report_sha256": runner.sha256_file(directory / "中文更新耗时.md"),
    })


def main():
    parser = argparse.ArgumentParser(description="十八组终点模型的统一CUDA更新耗时短测")
    parser.add_argument("--archive-root", default="D:/paper")
    parser.add_argument("--frozen-root", default="D:/paper_worktrees/frozen-sac-20260924-r2")
    args = parser.parse_args()
    run_update_timing(args.archive_root, args.frozen_root)


if __name__ == "__main__":
    main()
