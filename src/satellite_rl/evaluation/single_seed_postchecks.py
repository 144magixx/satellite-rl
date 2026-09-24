"""Fixed supplementary evaluations for the six completed seed-51001 SAC runs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
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
OLD_POST_COMMIT = "376dd3a00882332b8a8578d3790c26f3be163714"
OLD_SCOPE_SHA256 = "0ca75ac522e63a3d3348788d4b881d2c27231ef20ef8f9ddb6a0791d18fe97d1"
OLD_CONDITIONS_SHA256 = "8694d7bdb3806b4ea9196724b3ddb65436fb222386826dd832ca3beaec05020e"
OLD_ENVIRONMENT_SHA256 = "01b2ac67bfcd947f1feda6c8f0068c15842f3104fba9bc2951e2f71f65a0acb8"
GUI_BASELINE = Path(r"D:\paper\outputs\verification\single_seed_timing_dwm_20260924\idle_gpu_and_windows_processes.json")
GUI_BASELINE_SHA256 = "fc95c137579aaefd22c12c9f7636289a3df66b9c734550f624bcdf73054a2a4f"
WPS_GUI_EVIDENCE = Path(r"D:\paper\outputs\verification\single_seed_timing_dwm_20260924\wps_gui_pid22564.json")
WPS_GUI_EVIDENCE_SHA256 = "c2587468b9f4ed36490df3ba71467216d0b672905ddd0ea8599132226f94e43a"


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


def _windows_process(pid):
    command = (
        "$p=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(pid) + "'; "
        "if($p){[pscustomobject]@{ProcessId=$p.ProcessId;Name=$p.Name;"
        "Path=$p.ExecutablePath;CommandLine=$p.CommandLine;SessionId=$p.SessionId;"
        "Created=$p.CreationDate.ToString('o')} | ConvertTo-Json -Compress}"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def _process_rows(raw):
    result = []
    for line in raw.splitlines():
        if not line.strip() or line.strip().casefold() == "no running processes found":
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3 or not fields[0].isdigit():
            raise RuntimeError(f"无法解析nvidia-smi计算进程记录：{line}")
        result.append({"pid": int(fields[0]), "gpu_process_name": fields[1],
                       "used_gpu_memory": fields[2], "raw": line})
    if len({item["pid"] for item in result}) != len(result):
        raise RuntimeError("nvidia-smi中存在重复GPU进程PID")
    return result


def exclusive_gpu_snapshot(*, baseline_path=GUI_BASELINE,
                           baseline_sha256=GUI_BASELINE_SHA256,
                           wps_evidence_path=WPS_GUI_EVIDENCE,
                           wps_evidence_sha256=WPS_GUI_EVIDENCE_SHA256):
    """Allow only this process and exact, previously verified desktop contexts.

    WDDM lists GUI graphics contexts as compute-apps with N/A memory. N/A alone
    is not an exemption: PID, executable identity and creation time must match
    the saved desktop baseline. Any new or ambiguous process stops timing.
    """
    baseline_path = Path(baseline_path)
    if runner.sha256_file(baseline_path).lower() != baseline_sha256.lower():
        raise RuntimeError("Windows桌面GPU进程基线校验值改变，拒绝计时")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    known = {int(item["pid"]): item for item in baseline["listed_processes"]}
    if not known or any("python" in item["process_name"].casefold() for item in known.values()):
        raise RuntimeError("桌面GPU基线缺失或混入Python进程")
    wps_evidence_path = Path(wps_evidence_path)
    if runner.sha256_file(wps_evidence_path).lower() != wps_evidence_sha256.lower():
        raise RuntimeError("WPS图形进程证据SHA改变，拒绝计时")
    wps = json.loads(wps_evidence_path.read_text(encoding="utf-8-sig"))
    if (wps.get("pid") != 22564 or wps.get("process_name") != "promecefpluginhost.exe"
            or wps.get("signature_status") != "Valid"
            or "Kingsoft Office" not in wps.get("signer_subject", "")
            or "--type=gpu-process" not in wps.get("command_line", "")
            or not wps.get("gpu_engine_summary")
            or any("compute" in item["Engine"].casefold()
                   or float(item["MaxPercent"]) > 0
                   for item in wps["gpu_engine_summary"])):
        raise RuntimeError("WPS图形进程证据不支持窄范围豁免")
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    observed = _process_rows(query.stdout)
    classification = []
    for item in observed:
        pid = item["pid"]
        if pid == os.getpid():
            classification.append({**item, "kind": "current_timing_process"})
            continue
        prior = known.get(pid)
        if prior is None and pid == wps["pid"]:
            current = _windows_process(pid)
            path = str(current.get("Path") or "").replace("/", "\\").casefold() if current else ""
            expected_path = wps["path"].replace("/", "\\").casefold()
            if (not current or current.get("ProcessId") != pid
                    or current.get("Name") != wps["process_name"]
                    or datetime.fromisoformat(current["Created"]) != datetime.fromisoformat(wps["created_iso"])
                    or int(current.get("SessionId", -1)) != int(wps["session_id"])
                    or path != expected_path
                    or item["gpu_process_name"].replace("/", "\\").casefold() != expected_path
                    or item["used_gpu_memory"].casefold() != "[n/a]"
                    or current.get("CommandLine") != wps["command_line"]
                    or runner.sha256_file(current["Path"]).lower() != wps["path_sha256"].lower()):
                raise RuntimeError("新增WPS图形进程PID/路径/创建时间/命令不符，拒绝计时")
            classification.append({**item, "kind": "verified_wps_gui_context",
                                   "windows_name": current["Name"],
                                   "windows_path": current["Path"],
                                   "created": current["Created"],
                                   "evidence_sha256": wps_evidence_sha256,
                                   "limit": "采样窗口无正GPU Engine占用；不能排除未来WebGPU活动"})
            continue
        if prior is None or item["used_gpu_memory"].casefold() != "[n/a]":
            raise RuntimeError(f"存在非基线CUDA/未知GPU进程PID {pid}：{item['raw']}")
        current = _windows_process(pid)
        if not current or current.get("ProcessId") != pid or current.get("Name") != prior["process_name"]:
            raise RuntimeError(f"GPU进程PID {pid} 的Windows身份无法核准")
        if datetime.fromisoformat(current["Created"]) != datetime.fromisoformat(prior["created"]):
            raise RuntimeError(f"GPU进程PID {pid} 已复用，拒绝基线豁免")
        prior_gpu_name = prior["gpu_query"].split(",", 2)[1].strip()
        if prior["process_name"].casefold() == "dwm.exe":
            if (item["gpu_process_name"] != "[Insufficient Permissions]"
                    or prior_gpu_name != "[Insufficient Permissions]"
                    or int(current.get("SessionId", -1)) != 1
                    or current.get("Path")):
                raise RuntimeError("受保护的Windows桌面窗口管理器身份不匹配")
        else:
            path = str(current.get("Path") or "").replace("/", "\\").casefold()
            gpu_path = item["gpu_process_name"].replace("/", "\\").casefold()
            if not path or path != gpu_path or gpu_path != prior_gpu_name.replace("/", "\\").casefold():
                raise RuntimeError(f"GPU进程PID {pid} 的可执行路径与桌面基线不符")
        classification.append({**item, "kind": "verified_preexisting_gui_context",
                               "windows_name": current["Name"],
                               "windows_path": current.get("Path"),
                               "created": current["Created"]})
    snapshot = common._gpu_snapshot()
    snapshot_ids = {int(line.split(",", 1)[0].strip())
                    for line in snapshot["compute_process_query"].splitlines()
                    if line.strip() and line.split(",", 1)[0].strip().isdigit()}
    if snapshot_ids != {item["pid"] for item in observed}:
        raise RuntimeError("GPU进程在两次快照之间变化，拒绝独占计时")
    snapshot["process_query_with_names"] = query.stdout.strip()
    snapshot["desktop_baseline_path"] = str(baseline_path)
    snapshot["desktop_baseline_sha256"] = baseline_sha256
    snapshot["wps_gui_evidence_path"] = str(wps_evidence_path)
    snapshot["wps_gui_evidence_sha256"] = wps_evidence_sha256
    snapshot["process_classification"] = classification
    snapshot["timing_limit"] = ("无其他已知CUDA训练进程；Windows桌面/GUI和WPS图形上下文仍在，"
                                "采样窗口未见WPS计算活动，但不能称整个GPU完全独占或排除未来WebGPU")
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


def verify_prior_scope(post_root, plan):
    """Bind new timing to immutable outputs generated by the prior post commit."""
    post_root = Path(post_root)
    paths = {
        "scope_manifest_sha256": (post_root / "scope_manifest.json", OLD_SCOPE_SHA256),
        "conditions_index_sha256": (post_root / "conditions" / "result_index.json",
                                     OLD_CONDITIONS_SHA256),
        "environment_index_sha256": (post_root / "environment_comparison" / "result_index.json",
                                      OLD_ENVIRONMENT_SHA256),
    }
    for label, (path, expected) in paths.items():
        if runner.sha256_file(path).lower() != expected:
            raise RuntimeError(f"旧六组{label}校验值不符，拒绝继续计时")
    prior = common._load_json(paths["scope_manifest_sha256"][0])
    if (prior.get("scope") != SCOPE_NAME or
            prior.get("training_commit") != common.TRAIN_COMMIT or
            prior.get("generation_commit") != OLD_POST_COMMIT or
            set(prior.get("run_ids", [])) != {job["run_id"] for job in plan["jobs"]}):
        raise RuntimeError("旧六组scope_manifest的来源或运行范围不符")
    _verify_existing_index(paths["conditions_index_sha256"][0], "conditions",
                           {job["run_id"] for job in plan["jobs"]}, OLD_POST_COMMIT)
    _verify_existing_index(paths["environment_index_sha256"][0], "environment",
                           {job["run_id"] for job in plan["jobs"]}, OLD_POST_COMMIT)
    return {"old_generation_commit": OLD_POST_COMMIT,
            **{label: expected for label, (_path, expected) in paths.items()}}


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
    post_root = scope_root(exp_dir)
    prior_scope = verify_prior_scope(post_root, plan)
    exclusive_gpu_snapshot()
    path, state_sha, states = common.make_timing_states(plan, post_root)
    timing_dir = post_root / "timing"
    manifest = {
        "scope": SCOPE_NAME, "training_commit": common.TRAIN_COMMIT,
        "post_commit": provenance["post_commit"],
        "prior_scope": prior_scope,
        "desktop_gui_baseline_sha256": GUI_BASELINE_SHA256,
        "wps_gui_evidence_sha256": WPS_GUI_EVIDENCE_SHA256,
        "gpu_exclusivity_policy": "拒绝其他CUDA训练/未知PID；仅允许SHA固定且PID+路径+创建时间匹配的原桌面GUI与唯一已核WPS图形上下文",
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
        "interpretation": "单种子统一实际观测推理成本；仍有Windows桌面与WPS图形负载，并非整个GPU完全独占",
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
            before = exclusive_gpu_snapshot()
            agent = common.load_final_agent(job, final[run_id], torch.device("cuda"))
            try:
                torch.manual_seed(93001)
                result = {"config": config, "parameters": sum(
                    parameter.numel() for parameter in agent.actor.parameters()),
                    "gpu_before": before,
                    "timing": common._time_actor(agent, states),
                    "gpu_after": exclusive_gpu_snapshot(), "completed_at": utc_now()}
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
