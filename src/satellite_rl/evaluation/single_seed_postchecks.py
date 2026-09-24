"""Fixed supplementary evaluations for the six completed seed-51001 SAC runs."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import statistics

import numpy as np
import torch

from satellite_rl.evaluation.checkpoints import load_network_checkpoint
from satellite_rl.evaluation import post_training_checks as common
from satellite_rl.evaluation import update_timing as shared
from satellite_rl.evaluation.post_results_summary import (
    CONDITION_NAMES, _original_sample_rows, _sample_rows, paired_means,
)
from satellite_rl.evaluation.single_seed_scope import current_six_status, require_completed
from satellite_rl.training import experiment_runner as runner
from satellite_rl.training.run_monitor import atomic_json, utc_now


SCOPE_NAME = "single_seed_51001"


def scope_root(exp_dir):
    return Path(exp_dir) / SCOPE_NAME


def _validate_timing_result(result, config):
    if result.get("config") != config or not isinstance(result.get("parameters"), int) \
            or result["parameters"] <= 0 or not result.get("completed_at"):
        raise RuntimeError("已有推理计时结果配置或参数量不完整")
    for scope in ("forward_device_tensor", "take_action_host_to_action"):
        values = result.get("timing", {}).get(scope, {})
        if any(not math.isfinite(float(values[key])) or float(values[key]) <= 0
               for key in ("mean_ms", "median_ms", "p90_ms", "min_ms", "max_ms")):
            raise RuntimeError("已有推理计时数值无效")
    for key in ("gpu_before", "gpu_after"):
        snapshot = result.get(key, {})
        if not snapshot.get("at") or "compute_process_query" not in snapshot:
            raise RuntimeError("已有推理计时缺少GPU快照")
    return result


def exclusive_gpu_snapshot():
    """Refuse timing when nvidia-smi sees another CUDA compute process."""
    snapshot = common._gpu_snapshot()
    for line in snapshot["compute_process_query"].splitlines():
        text = line.strip()
        if not text or text.lower().startswith("no running processes"):
            continue
        pid = text.split(",", 1)[0].strip()
        if not pid.isdigit():
            raise RuntimeError(f"不能识别CUDA计算进程快照，拒绝独占计时：{text}")
        if int(pid) != os.getpid():
            raise RuntimeError(f"GPU上仍有其他计算进程PID {pid}，拒绝独占计时")
    return snapshot


def require_cuda_ready(archive_root, frozen_root):
    provenance = common.source_guard(frozen_root)
    exp_dir, plan, final = require_completed(archive_root, frozen_root)
    if len(plan["jobs"]) != 6 or len(final) != 6 or {job["seed"] for job in plan["jobs"]} != {51001}:
        raise RuntimeError("单种子补评要求六个种子51001终点全部完成")
    snapshot = current_six_status(archive_root)
    if snapshot["counts"]["completed"] != 6 or any(
            row["worker_identity_alive"] for row in snapshot["runs"]):
        raise RuntimeError("六组训练状态未完成或原训练进程尚未退出")
    active = shared._experiment_gpu_children()
    if active:
        raise RuntimeError(f"当前仍有实验训练或后处理GPU进程，拒绝启动补评：{active}")
    runtime = runner.strict_cuda_preflight()
    return provenance, exp_dir, plan, final, runtime


def _scope_manifest(post_root, plan, provenance, runtime):
    common._put_immutable_json(post_root / "scope_manifest.json", {
        "scope": SCOPE_NAME, "train_seed": 51001, "training_commit": common.TRAIN_COMMIT,
        "generation_commit": provenance["post_commit"],
        "runtime": runtime, "run_ids": sorted(job["run_id"] for job in plan["jobs"]),
        "test_scene_ids": list(common.SCENE_IDS), "action_seeds": list(common.ACTION_SEEDS),
        "interpretation": "单个独立训练种子；条件评价不代表跨种子稳定性",
    })


def _verify_existing_index(path, stage, expected_runs, post_commit):
    path = Path(path)
    if not path.exists():
        return
    index = common._load_json(path)
    if index.get("scope") != SCOPE_NAME:
        raise RuntimeError("已有结果索引不属于单种子六组")
    if stage == "conditions":
        if (index.get("post_commit") != post_commit or
                set(index.get("conditions", {})) != set(common.CONDITIONS)):
            raise RuntimeError("已有条件评价索引范围或代码版本不符")
        for item in index["conditions"].values():
            if (set(item["runs"]) != expected_runs or
                    runner.sha256_file(item["scene_manifest"]) != item["scene_manifest_sha256"]):
                raise RuntimeError("已有条件场景或六组结果索引不完整")
            for result in item["runs"].values():
                if runner.sha256_file(result["path"]) != result["sha256"]:
                    raise RuntimeError("已有条件评价结果校验值不符")
    else:
        if (index.get("generation_commit") != post_commit or
                set(index.get("runs", {})) != {"before_c4963ba", "corrected_r2"} or
                runner.sha256_file(path.parent / "source_manifest.json")
                != index["source_manifest_sha256"]):
            raise RuntimeError("已有历史权重对照索引范围或版本不符")
        for result in index["runs"].values():
            if runner.sha256_file(result["path"]) != result["sha256"]:
                raise RuntimeError("已有历史权重对照结果校验值不符")
    for name in ("numeric", "report"):
        asset = index["summary"][name]
        if runner.sha256_file(asset) != index["summary"][name + "_sha256"]:
            raise RuntimeError("已有中文摘要或数字结果校验值不符")


def _save_condition_summary(post_root, plan, final):
    rows = []
    for condition in common.CONDITIONS:
        for job in plan["jobs"]:
            run_id = job["run_id"]
            original = _original_sample_rows(common._load_json(final[run_id]["original_test"]))
            path = post_root / "conditions" / condition / run_id / "evaluation" / "final_test_400000.json"
            changed = _sample_rows(common._load_json(path))
            rows.append({"condition": condition, "run_id": run_id,
                         "model": job["model"], "formula": job["formula"],
                         "train_seed": 51001, "paired_scenes": 20,
                         "action_seeds_per_scene": 3,
                         **paired_means(original, changed)})
    if len(rows) != 18:
        raise RuntimeError("固定条件差值不是三条件×六组")
    directory = post_root / "conditions"
    common._put_immutable_json(directory / "paired_differences.json", {
        "scope": SCOPE_NAME, "one_training_seed": 51001,
        "difference_definition": "changed_condition_minus_original",
        "rows": rows})
    lines = ["# 六组单种子的固定条件对照", "",
             "各模型仅训练种子51001；每个差值先在同一20场景×3动作种子上配对，"
             "不报告不存在的训练种子间波动。", "",
             "| 条件 | 网络 | 公式 | SI差/百分点 | SGM差/百分点 | 吞吐差/Gbps | 功率差/W | 超功率率差/百分点 | 完整分配率差/百分点 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {CONDITION_NAMES[row['condition']]} | "
                     f"{shared.MODEL_LABELS[row['model']]} | {shared.FORMULA_LABELS[row['formula']]} | "
                     f"{row['si_mean'] * 100:.2f} | {row['sgm_mean'] * 100:.2f} | "
                     f"{row['throughput_bps'] / 1e9:.2f} | {row['total_power_w']:.2f} | "
                     f"{row['power_exceeded'] * 100:.2f} | {row['completed'] * 100:.2f} |")
    lines += ["", "差值来自已保存终态记录；场景变化下的关联不单独证明策略因果适应。", ""]
    path = directory / "中文条件对照.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"numeric": str(directory / "paired_differences.json"),
            "numeric_sha256": runner.sha256_file(directory / "paired_differences.json"),
            "report": str(path), "report_sha256": runner.sha256_file(path)}


def _save_environment_summary(post_root):
    directory = post_root / "environment_comparison"
    old = _sample_rows(common._load_json(
        directory / "before_c4963ba" / "evaluation" / "final_test_unknown.json"))
    corrected = _sample_rows(common._load_json(
        directory / "corrected_r2" / "evaluation" / "final_test_unknown.json"))
    differences = paired_means(old, corrected)
    numeric = directory / "paired_differences.json"
    common._put_immutable_json(numeric, {"scope": SCOPE_NAME, "weight_training_steps": "unknown",
                          "independent_training_repeats": 1,
                          "difference_definition": "corrected_environment_minus_old_environment",
                          "paired_scenes": 20, "action_seeds_per_scene": 3,
                          "mean_differences": differences})
    path = directory / "中文历史权重环境对照.md"
    path.write_text("\n".join([
        "# 历史 Transformer 单权重环境对照", "",
        "同一份兼容历史权重分别在修正前、修正后环境评价。训练交互步数为 unknown；"
        "这里是一份权重的20场景×3动作种子对照，不是三个训练种子，也不参与六模型排名。", "",
        "| SI差/百分点 | SGM差/百分点 | 吞吐差/Gbps | 功率差/W | 超功率率差/百分点 | 完整分配率差/百分点 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ("| " + " | ".join((f"{differences['si_mean'] * 100:.2f}",
                              f"{differences['sgm_mean'] * 100:.2f}",
                              f"{differences['throughput_bps'] / 1e9:.2f}",
                              f"{differences['total_power_w']:.2f}",
                              f"{differences['power_exceeded'] * 100:.2f}",
                              f"{differences['completed'] * 100:.2f}")) + " |"), "",
        "旧环境结果只说明计算口径改变及该旧策略的适应，不用于修正环境中的模型排名。", "",
    ]), encoding="utf-8")
    return {"numeric": str(numeric), "numeric_sha256": runner.sha256_file(numeric),
            "report": str(path), "report_sha256": runner.sha256_file(path)}


def run_conditions(archive_root, frozen_root):
    provenance, exp_dir, plan, final, runtime = require_cuda_ready(archive_root, frozen_root)
    post_root = scope_root(exp_dir)
    post_root.mkdir(parents=True, exist_ok=True)
    _scope_manifest(post_root, plan, provenance, runtime)
    _verify_existing_index(post_root / "conditions" / "result_index.json", "conditions",
                           set(final), provenance["post_commit"])
    references = common.prepare_condition_scenes(post_root, plan, provenance)
    common._put_immutable_json(post_root / "original_reference.json", {
        "scope": SCOPE_NAME, "training_commit": common.TRAIN_COMMIT,
        "generation_commit": provenance["post_commit"],
        "rule": "从相同100场景终点考核取81000–81019，采样动作种子不变",
        "results": {run_id: {
            "path": item["original_test"], "sha256": item["original_test_sha256"]}
            for run_id, item in final.items()},
    })
    result_index = {}
    for condition in common.CONDITIONS:
        condition_results = {}
        for job in plan["jobs"]:
            run_id = job["run_id"]
            agent = common.load_final_agent(job, final[run_id], torch.device("cuda"))
            try:
                task_dir = post_root / "conditions" / condition / run_id
                config = common._task_config(
                    "condition", condition, run_id, final[run_id]["actor"],
                    references[condition], provenance)
                common._run_evaluation_task(task_dir, f"{condition}_{run_id}",
                                            config, agent, references[condition])
                result_path = task_dir / "evaluation" / "final_test_400000.json"
                condition_results[run_id] = {"path": str(result_path),
                                             "sha256": runner.sha256_file(result_path)}
            finally:
                del agent
                torch.cuda.empty_cache()
        scene_manifest = post_root / "scenes" / condition / "manifest.json"
        result_index[condition] = {"scene_manifest": str(scene_manifest),
                                   "scene_manifest_sha256": runner.sha256_file(scene_manifest),
                                   "runs": condition_results}
    summary_assets = _save_condition_summary(post_root, plan, final)
    common._put_immutable_json(post_root / "conditions" / "result_index.json", {
        "scope": SCOPE_NAME, "training_commit": common.TRAIN_COMMIT,
        "post_commit": provenance["post_commit"], "conditions": result_index,
        "summary": summary_assets})
    atomic_json(post_root / "conditions_complete.json", {
        "at": utc_now(), "scope": SCOPE_NAME, "generation_commit": provenance["post_commit"],
        "conditions": list(common.CONDITIONS), "runs_per_condition": 6,
        "scenes_per_run": 20, "action_seeds_per_scene": 3})


def run_environment_pair(archive_root, frozen_root):
    provenance, exp_dir, plan, final, runtime = require_cuda_ready(archive_root, frozen_root)
    post_root = scope_root(exp_dir)
    post_root.mkdir(parents=True, exist_ok=True)
    _scope_manifest(post_root, plan, provenance, runtime)
    _verify_existing_index(post_root / "environment_comparison" / "result_index.json",
                           "environment", set(final), provenance["post_commit"])
    old_module, old_source, old_sha = common.legacy_environment_source(post_root)
    old_actor = Path(frozen_root) / "models" / "pretrained" / "sac_transformer" / "actor.pth"
    if not old_actor.is_file():
        raise FileNotFoundError(f"历史Transformer权重缺失：{old_actor}")
    references = {str(scene): plan["test_scenes"][str(scene)] for scene in common.SCENE_IDS}
    source_path = post_root / "environment_comparison" / "source_manifest.json"
    common._put_immutable_json(source_path, {
        "scope": SCOPE_NAME, "training_commit": common.TRAIN_COMMIT,
        "post_commit": provenance["post_commit"],
        "old_environment_commit": common.OLD_ENV_COMMIT,
        "old_environment_source": str(old_source), "old_environment_sha256": old_sha,
        "old_actor": str(old_actor), "old_actor_sha256": runner.sha256_file(old_actor),
        "old_actor_training_steps": "unknown", "scenes": references,
        "action_seeds": list(common.ACTION_SEEDS), "runtime": runtime,
        "interpretation": "同一历史权重的计算口径诊断；不参与六模型排名",
    })
    agent = common.EvaluationAgent("transformer", torch.device("cuda"))
    load_network_checkpoint(agent.actor, old_actor, algorithm="historical-transformer",
                            map_location="cuda", evaluation_mode=True)
    results = {}
    try:
        for label, environment in (("before_c4963ba", old_module), ("corrected_r2", None)):
            config = common._task_config(
                "environment", label, "pretrained_transformer", old_actor,
                references, provenance, old_env_sha=old_sha if environment else None,
                training_steps="unknown")
            common._run_evaluation_task(
                post_root / "environment_comparison" / label,
                "historical_transformer_" + label, config, agent, references,
                environment_module=environment, step="unknown")
            result_path = post_root / "environment_comparison" / label / "evaluation" / "final_test_unknown.json"
            results[label] = {"path": str(result_path), "sha256": runner.sha256_file(result_path)}
    finally:
        del agent
        torch.cuda.empty_cache()
    summary_assets = _save_environment_summary(post_root)
    common._put_immutable_json(post_root / "environment_comparison" / "result_index.json", {
        "scope": SCOPE_NAME, "generation_commit": provenance["post_commit"],
        "one_weight": True, "weight_training_steps": "unknown",
        "scenes": 20, "action_seeds_per_scene": 3,
        "source_manifest_sha256": runner.sha256_file(source_path),
        "runs": results, "summary": summary_assets})


def run_inference_timing(archive_root, frozen_root):
    provenance, exp_dir, plan, final, runtime = require_cuda_ready(archive_root, frozen_root)
    exclusive_gpu_snapshot()
    post_root = scope_root(exp_dir)
    post_root.mkdir(parents=True, exist_ok=True)
    _scope_manifest(post_root, plan, provenance, runtime)
    path, state_sha, states = common.make_timing_states(plan, post_root)
    timing_dir = post_root / "timing"
    manifest = {
        "scope": SCOPE_NAME, "training_commit": common.TRAIN_COMMIT,
        "post_commit": provenance["post_commit"],
        "matched_source_sha256": provenance["matched_file_sha256"],
        "state_path": str(path), "state_sha256": state_sha,
        "state_shape": list(states.shape), "scene_ids": list(common.SCENE_IDS),
        "before_beam_indices": list(common.TIMING_BEAM_INDICES),
        "reference_allocation": "group=index%8; frequency=(index//8)*5; slots=5; power=20W",
        "warmup_per_scope": common.TIMING_WARMUP,
        "repeats_per_scope": common.TIMING_REPEATS,
        "dtype": "float32", "runtime": runtime,
        "exclusive_gpu_required": True,
        "scope_forward": "GPU-resident one-state tensor through actor; synchronized before/after",
        "scope_take_action": "host state through SAC.take_action; synchronized before/after",
        "interpretation": "单种子统一实际观测推理成本，不是训练或策略成绩",
    }
    common._put_immutable_json(timing_dir / "manifest.json", manifest)
    existing_index = timing_dir / "result_index.json"
    if existing_index.exists():
        indexed = common._load_json(existing_index)
        if (indexed.get("manifest_sha256") != runner.sha256_file(timing_dir / "manifest.json")
                or set(indexed.get("runs", {})) != set(final)
                or any(runner.sha256_file(value["path"]) != value["sha256"]
                       for value in indexed["runs"].values())
                or runner.sha256_file(indexed["report"]) != indexed["report_sha256"]):
            raise RuntimeError("已有六组推理计时索引或文件校验失败")
    results = {}
    for job in plan["jobs"]:
        run_id = job["run_id"]
        result_path = timing_dir / "runs" / f"{run_id}.json"
        config = {"run_id": run_id, "actor_sha256": final[run_id]["actor_sha256"],
                  "state_sha256": state_sha, "post_commit": provenance["post_commit"],
                  "warmup": common.TIMING_WARMUP, "repeats": common.TIMING_REPEATS}
        if result_path.exists():
            _validate_timing_result(common._load_json(result_path), config)
        else:
            exclusive_gpu_snapshot()
            agent = common.load_final_agent(job, final[run_id], torch.device("cuda"))
            try:
                torch.manual_seed(93001)
                result = {"config": config, "parameters": sum(
                    parameter.numel() for parameter in agent.actor.parameters()),
                    "gpu_before": common._gpu_snapshot(),
                    "timing": common._time_actor(agent, states),
                    "gpu_after": common._gpu_snapshot(), "completed_at": utc_now()}
                result_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(result_path, result)
            finally:
                del agent
                torch.cuda.empty_cache()
        results[run_id] = {"path": str(result_path), "sha256": runner.sha256_file(result_path)}
    lines = ["# 六组单种子推理计时", "",
             "六组模型仅来自训练种子51001；每组重复计时是测量重复，不代表独立训练种子。", "",
             "| 网络 | 公式 | Actor参数量 | 纯前向均值/中位/P90 ms | 完整决策均值/中位/P90 ms |",
             "| --- | --- | ---: | --- | --- |"]
    for job in plan["jobs"]:
        result = common._load_json(results[job["run_id"]]["path"])
        def numbers(scope):
            measured = result["timing"][scope]
            return "/".join(f"{measured[key]:.3f}" for key in ("mean_ms", "median_ms", "p90_ms"))
        lines.append(f"| {shared.MODEL_LABELS[job['model']]} | "
                     f"{shared.FORMULA_LABELS[job['formula']]} | {result['parameters']} | "
                     f"{numbers('forward_device_tensor')} | {numbers('take_action_host_to_action')} |")
    lines += ["", "纯前向从已驻留GPU的204维单状态张量到Actor输出；完整决策从主机状态到动作返回。"
              "两者均同步CUDA。原始每组计时及GPU负载快照保存在runs/，"
              "不能据此推断六组训练的稳定性能或泛化能力。", ""]
    report_path = timing_dir / "中文推理计时.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    common._put_immutable_json(timing_dir / "result_index.json", {
        "manifest": str(timing_dir / "manifest.json"),
        "manifest_sha256": runner.sha256_file(timing_dir / "manifest.json"),
        "runs": results, "report": str(report_path),
        "report_sha256": runner.sha256_file(report_path),
    })


def main():
    parser = argparse.ArgumentParser(description="六组单训练种子的固定补充评价")
    parser.add_argument("command", choices=("conditions", "environment", "timing"))
    parser.add_argument("--archive-root", default="D:/paper")
    parser.add_argument("--frozen-root", default="D:/paper_worktrees/frozen-sac-20260924-r2")
    args = parser.parse_args()
    commands = {"conditions": run_conditions, "environment": run_environment_pair,
                "timing": run_inference_timing}
    commands[args.command](args.archive_root, args.frozen_root)


if __name__ == "__main__":
    main()
