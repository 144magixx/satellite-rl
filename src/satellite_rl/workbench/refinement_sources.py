"""Strict reuse of completed workbench allocations for the variable-beam SCA route."""
from copy import deepcopy
import math
from pathlib import Path

from .experiments import VARIABLE_BEAM_EXPERIMENTS, is_variable_beam_experiment
from .storage import digest, read_json, validate_id


EXPERIMENT_ID = "S2-P17"  # 历史默认值（S2-P17 冻结来源链）；常规链路用 EXPERIMENT_IDS 集合判断。
EXPERIMENT_IDS = VARIABLE_BEAM_EXPERIMENTS
SOURCE_GROUPS = ("model", "fixed", "greedy")


def source_rows(reference):
    """Read saved allocations, never regenerate model or baseline actions."""
    path = Path(reference["path"])
    if not path.is_file() or digest(path) != reference["sha256"]:
        raise ValueError("第二阶段原始结果SHA不一致")
    data = read_json(path, {})
    group = reference["group"]
    rows = (data.get("scenes", []) if group == "model" else
            data.get("baselines", {}).get(group, {}).get("scenes", []))
    expected_repeats = reference["repeats"]
    indexed = {}
    frozen_cases = {item["path"]: item["sha256"] for item in reference.get("case_files", [])}
    for original in rows:
        row = deepcopy(original)
        case_path = row.get("case_path")
        if case_path:
            from .variable_evaluation import _sha256
            case = Path(case_path).resolve(strict=True)
            if not case.is_relative_to(path.parent.resolve() / "cases"):
                raise ValueError("原始初解case不属于来源任务")
            if frozen_cases.get(str(case)) != digest(case):
                raise ValueError("原始初解case SHA不一致")
            envelope = read_json(case, {})
            body = envelope.get("body", {})
            full = body.get("result", {})
            if envelope.get("sha256") != _sha256(body):
                raise ValueError("原始初解case内容校验失败")
            for key, value in row.items():
                if key != "case_path" and full.get(key) != value:
                    raise ValueError("原始初解case与已完成结果摘要不一致")
            row = deepcopy(full)
            row["original_case_path"] = str(case)
            row["original_case_file_sha256"] = frozen_cases[str(case)]
        scene = row.get("scene") or {}
        sha = scene.get("source_sha256") or scene.get("sha256")
        repeat = row.get("repeat_index")
        if not sha or type(repeat) is not int or not 0 <= repeat < expected_repeats:
            raise ValueError("初始分配缺少场景SHA或明确动作重复编号")
        key = (sha, repeat)
        if key in indexed:
            raise ValueError("初始分配存在重复场景/动作")
        duration = row.get("initial_allocation_seconds")
        if type(duration) not in (float, int) or not math.isfinite(duration) or duration < 0:
            raise ValueError("初始分配缺少实际完整分配耗时；不能补零")
        if row.get("action_seed") != reference["action_seed"] + repeat:
            raise ValueError("初始动作种子与冻结来源不一致")
        beams = row.get("beams") or []
        if not beams or len(beams) != row.get("beam_count"):
            raise ValueError("初始结果缺失全体目标束")
        row["policy_inference_seconds"] = row.get("inference_seconds")
        # Existing refinement timing calls its full initialization duration
        # inference_seconds. Preserve the pure policy time separately.
        row["inference_seconds"] = float(duration)
        indexed[key] = row
    if len(indexed) != reference["scene_count"] * expected_repeats:
        raise ValueError("初始分配数量与冻结场景/动作预算不一致")
    return indexed


def prepare_source(app, body):
    """Resolve only a completed variable-beam (S2-P17/S2-P18) evaluation in this workbench."""
    run_id = body.get("source_run_id")
    validate_id(run_id)
    group = body.get("source_group", "model")
    if group not in SOURCE_GROUPS:
        raise ValueError("初始分配来源须为model、fixed或greedy")
    directory, readonly = app.resolve_run(run_id)
    job = read_json(directory / "job.json", {})
    status = read_json(directory / "status.json", {})
    experiment_id = job.get("experiment_id")
    if readonly or job.get("kind") != "evaluate" or not is_variable_beam_experiment(experiment_id):
        raise ValueError(f"须选择本工作台可变束数正式评价任务（{'/'.join(EXPERIMENT_IDS)}）")
    if status.get("stage") != "completed":
        raise ValueError("第二阶段初始分配尚未完整完成")
    dataset = (job.get("scene_sets") or {}).get("test")
    if not dataset or (dataset.get("source") or {}).get("partition_role") != "test":
        raise ValueError("初始分配缺少冻结test场景")
    from .datasets import scene_set_files
    files = scene_set_files(dataset)
    path = directory / "results.json"
    reference = {"run_id": run_id, "path": str(path), "sha256": digest(path),
                 "group": group, "scene_count": len(files),
                 "repeats": job["repeats"] if group == "model" else 1,
                 "action_mode": job["action_mode"] if group == "model" else "deterministic",
                 "action_seed": job["action_seed"]}
    summary = read_json(path, {})
    selected = (summary.get("scenes", []) if group == "model" else
                summary.get("baselines", {}).get(group, {}).get("scenes", []))
    if not selected or any(not row.get("case_path") for row in selected):
        raise ValueError(f"{experiment_id}来源须有完整持久初解case")
    reference["case_files"] = [{"path": str(Path(row["case_path"]).resolve(strict=True)),
                                "sha256": digest(row["case_path"])} for row in selected]
    indexed = source_rows(reference)
    expected = {(sha, r) for _, sha in files for r in range(reference["repeats"])}
    if set(indexed) != expected:
        raise ValueError("初始分配来源与冻结场景全集不一致")
    config = deepcopy(job["model_config"])
    config["environment"]["beam_count_max"] = 200
    if group != "model":
        # Use the actual baseline environment contract saved by this task.
        config["environment"]["action_mask"] = "joint_feasible"
    return job, config, dataset, reference
