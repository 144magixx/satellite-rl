"""Synchronized full-update cost for six independent seed-51001 agent copies."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback
import uuid

import torch

from satellite_rl.evaluation import post_training_checks as common
from satellite_rl.evaluation import update_timing as shared
from satellite_rl.evaluation.single_seed_scope import current_six_status, require_completed
from satellite_rl.evaluation.single_seed_postchecks import (
    GUI_BASELINE_SHA256, SCOPE_NAME, exclusive_gpu_snapshot, scope_root,
    verify_prior_scope,
)
from satellite_rl.training import experiment_runner as runner
from satellite_rl.training.run_monitor import atomic_json, utc_now


def require_ready(archive_root, frozen_root):
    provenance = common.source_guard(frozen_root)
    exp_dir, plan, final = require_completed(archive_root, frozen_root)
    if len(plan["jobs"]) != 6 or {job["seed"] for job in plan["jobs"]} != {51001}:
        raise RuntimeError("更新计时要求种子51001的六个终点模型全部完成")
    snapshot = current_six_status(archive_root)
    if snapshot["counts"]["completed"] != 6 or any(
            row["worker_identity_alive"] for row in snapshot["runs"]):
        raise RuntimeError("六组训练状态未完成或原训练进程尚未退出")
    active = shared._experiment_gpu_children()
    if active:
        raise RuntimeError(f"当前仍有实验GPU子进程，拒绝更新计时：{active}")
    runtime = runner.strict_cuda_preflight()
    return provenance, exp_dir, plan, final, runtime


def _verify_existing_index(directory, expected_ids):
    path = directory / "result_index.json"
    if not path.exists():
        return
    index = shared._read(path)
    if index.get("manifest_sha256") != runner.sha256_file(directory / "manifest.json"):
        raise RuntimeError("已有更新计时索引的清单校验值不匹配")
    if set(index.get("runs", {})) != expected_ids:
        raise RuntimeError("已有更新计时索引不是完整六组")
    for run_id, reference in index["runs"].items():
        if runner.sha256_file(reference["path"]) != reference["sha256"]:
            raise RuntimeError(f"已有更新计时文件校验失败：{run_id}")
    for key, name in (("raw_csv_sha256", "raw_updates.csv"),
                      ("summary_csv_sha256", "summary.csv"),
                      ("formula_differences_csv_sha256", "formula_differences.csv"),
                      ("chinese_report_sha256", "中文更新耗时.md")):
        if runner.sha256_file(directory / name) != index[key]:
            raise RuntimeError(f"已有更新计时汇总文件校验失败：{name}")


def _finish_reports(directory, plan, results):
    rows, raw, differences = [], [], []
    indexed = {(job["model"], job["formula"]): results[job["run_id"]]
               for job in plan["jobs"]}
    if len(indexed) != 6:
        raise RuntimeError("更新计时报告缺少三网络×两套公式")
    for job in plan["jobs"]:
        run_id = job["run_id"]
        result = results[run_id]
        statistics = result["statistics"]
        rows.append({"run_id": run_id, "model": job["model"], "formula": job["formula"],
                     "train_seed": 51001, "actor_parameters": result["parameters_actor"],
                     **statistics})
        raw.extend({"run_id": run_id, "iteration": index + 1,
                    "batch_index": index % shared.BATCH_COUNT,
                    "milliseconds": value}
                   for index, value in enumerate(result["raw_milliseconds"]))
    for model in runner.MODELS:
        old, new = indexed[(model, "legacy")], indexed[(model, "corrected")]
        differences.append({"model": model, "train_seed": 51001,
                            "corrected_minus_legacy_mean_ms": (
                                new["statistics"]["mean_ms"] - old["statistics"]["mean_ms"]),
                            "corrected_minus_legacy_updates_per_second": (
                                new["statistics"]["updates_per_second"]
                                - old["statistics"]["updates_per_second"])})
    shared._write_csv(directory / "raw_updates.csv",
                      ("run_id", "iteration", "batch_index", "milliseconds"), raw)
    shared._write_csv(directory / "summary.csv", tuple(rows[0]), rows)
    shared._write_csv(directory / "formula_differences.csv", tuple(differences[0]), differences)
    lines = ["# 六组 SAC 完整更新耗时", "",
             "全部六组来自同一个训练种子 51001；每组独立终点副本预热20次，再同步计时100次。"
             "100次是计时重复，不能当作独立训练种子或效果证据。", "",
             "| 网络 | 公式 | Actor参数量 | 平均/中位/P90 ms | 更新/秒 |",
             "| --- | --- | ---: | --- | ---: |"]
    for row in rows:
        lines.append(f"| {shared.MODEL_LABELS[row['model']]} | "
                     f"{shared.FORMULA_LABELS[row['formula']]} | {row['actor_parameters']} | "
                     f"{row['mean_ms']:.3f}/{row['median_ms']:.3f}/{row['p90_ms']:.3f} | "
                     f"{row['updates_per_second']:.2f} |")
    lines += ["", "每个模型的修正公式减原公式成本差见 `formula_differences.csv`。"
              "固定批次来自同一份真实训练回放；计时包含函数内部数据转换、网络前后向、"
              "优化器、温度与目标网络更新，不含环境步与批次抽取。"
              "计时前后同步 CUDA。短测受显卡负载和时钟状态影响，不外推为整段训练速度。", ""]
    (directory / "中文更新耗时.md").write_text("\n".join(lines), encoding="utf-8")


def run_update_timing(archive_root, frozen_root):
    provenance, exp_dir, plan, final, runtime = require_ready(archive_root, frozen_root)
    prior_scope = verify_prior_scope(scope_root(exp_dir), plan)
    timing_index_path = scope_root(exp_dir) / "timing" / "result_index.json"
    if not timing_index_path.is_file():
        raise RuntimeError("六组推理计时未完成，拒绝完整参数更新计时")
    timing_index = shared._read(timing_index_path)
    if (set(timing_index.get("runs", {})) != set(final) or
            runner.sha256_file(timing_index["manifest"]) != timing_index["manifest_sha256"] or
            runner.sha256_file(timing_index["report"]) != timing_index["report_sha256"] or
            any(runner.sha256_file(item["path"]) != item["sha256"]
                for item in timing_index["runs"].values())):
        raise RuntimeError("六组推理计时原始索引或SHA不一致")
    exclusive_gpu_snapshot()
    directory = scope_root(exp_dir) / "update_timing"
    directory.mkdir(parents=True, exist_ok=True)
    batch_path, batch_sha, arrays, replay_size = shared._load_or_save_batches(directory, final)
    config = {
        "scope": SCOPE_NAME, "one_training_seed": 51001,
        "training_commit": common.TRAIN_COMMIT, "post_commit": provenance["post_commit"],
        "prior_scope": prior_scope,
        "inference_timing_index_sha256": runner.sha256_file(timing_index_path),
        "desktop_gui_baseline_sha256": GUI_BASELINE_SHA256,
        "gpu_exclusivity_policy": "拒绝其他CUDA训练/未知PID；记录经验证的Windows桌面GUI上下文",
        "matched_source_sha256": provenance["matched_file_sha256"],
        "source_replay_run": shared.SOURCE_RUN,
        "source_checkpoint": final[shared.SOURCE_RUN]["checkpoint"],
        "source_checkpoint_sha256": final[shared.SOURCE_RUN]["checkpoint_sha256"],
        "replay_size": replay_size, "batch_seed": shared.BATCH_SEED,
        "update_seed": shared.UPDATE_SEED, "batch_count": shared.BATCH_COUNT,
        "batch_size": shared.BATCH_SIZE, "batch_file": str(batch_path),
        "batch_sha256": batch_sha, "warmup": shared.WARMUP, "repeats": shared.REPEATS,
        "batch_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "update_input_containers": {
            "states": "numpy.ndarray", "next_states": "numpy.ndarray",
            "actions": "tuple[list[int]]", "rewards": "tuple[float]",
            "dones": "tuple[bool]",
        },
        "all_final_checkpoint_sha256": {key: item["checkpoint_sha256"]
                                        for key, item in final.items()},
        "all_final_checkpoint_paths": {key: item["checkpoint"] for key, item in final.items()},
        "all_final_actor_sha256": {key: item["actor_sha256"]
                                   for key, item in final.items()},
        "all_final_actor_paths": {key: item["actor"] for key, item in final.items()},
        "runtime": runtime,
        "no_other_cuda_training_process_required": True,
        "windows_desktop_gpu_contexts_remain": True,
        "timer_scope": "synchronize; agent.update(saved fixed batch); synchronize",
        "excluded": ["environment transition", "replay sampling", "batch selection", "checkpoint I/O"],
        "interpretation": "六个独立模型副本的完整更新计算成本；只有一个训练种子且仍有Windows桌面GPU负载",
    }
    shared._immutable_json(directory / "manifest.json", config)
    _verify_existing_index(directory, set(final))
    batches = shared._batch_dicts(arrays)
    results = {}
    for job in plan["jobs"]:
        run_id = job["run_id"]
        result_path = directory / "runs" / (run_id + ".json")
        run_config = {"run_id": run_id, "model": job["model"],
                      "formula": job["formula"], "seed": 51001,
                      "checkpoint_sha256": final[run_id]["checkpoint_sha256"],
                      "actor_sha256": final[run_id]["actor_sha256"],
                      "batch_sha256": batch_sha, "post_commit": provenance["post_commit"],
                      "warmup": shared.WARMUP, "repeats": shared.REPEATS}
        if result_path.exists():
            result = shared._validate_cached_result(shared._read(result_path), run_config)
        else:
            before = exclusive_gpu_snapshot()
            agent, replay = runner._create_agent(job["model"], job["formula"],
                                                 directory / "temporary_logs" / run_id,
                                                 uuid.uuid4().hex)
            try:
                run_manifest = shared._read(exp_dir / run_id / "manifest.json")
                progress = runner.load_checkpoint(final[run_id]["checkpoint"], agent, replay,
                                                  run_manifest["config_hash"])
                if progress["env_step"] != 400000 or progress["update_step"] != 398000:
                    raise RuntimeError(f"{run_id} 断点不是预定四十万步终点")
                torch.manual_seed(shared.UPDATE_SEED)
                milliseconds = shared.time_updates(agent, batches)
                runner._ensure_finite_agent(agent)
                result = {"config": run_config, "raw_milliseconds": milliseconds,
                          "statistics": shared.summarize_ms(milliseconds),
                          "parameters_actor": sum(item.numel() for item in agent.actor.parameters()),
                          "temporary_log_file": str(agent.log_file),
                          "gpu_before": before, "gpu_after": exclusive_gpu_snapshot(),
                          "completed_at": utc_now()}
                result_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(result_path, result)
            except Exception as error:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(result_path.with_name(run_id + ".failure." + uuid.uuid4().hex + ".json"), {
                    "at": utc_now(), "config": run_config,
                    "type": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc()})
                raise
            finally:
                del agent, replay
                torch.cuda.empty_cache()
        results[run_id] = result
    _finish_reports(directory, plan, results)
    shared._immutable_json(directory / "result_index.json", {
        "manifest_sha256": runner.sha256_file(directory / "manifest.json"),
        "runs": {run_id: {"path": str(directory / "runs" / (run_id + ".json")),
                          "sha256": runner.sha256_file(directory / "runs" / (run_id + ".json"))}
                 for run_id in final},
        "raw_csv_sha256": runner.sha256_file(directory / "raw_updates.csv"),
        "summary_csv_sha256": runner.sha256_file(directory / "summary.csv"),
        "formula_differences_csv_sha256": runner.sha256_file(directory / "formula_differences.csv"),
        "chinese_report_sha256": runner.sha256_file(directory / "中文更新耗时.md"),
    })


def main():
    parser = argparse.ArgumentParser(description="单种子六组SAC模型的统一CUDA更新耗时")
    parser.add_argument("--archive-root", default="D:/paper")
    parser.add_argument("--frozen-root", default="D:/paper_worktrees/frozen-sac-20260924-r2")
    args = parser.parse_args()
    run_update_timing(args.archive_root, args.frozen_root)


if __name__ == "__main__":
    main()
