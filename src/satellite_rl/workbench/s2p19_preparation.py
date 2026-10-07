"""Persistent typed preparation worker, dispatched only by the workbench."""
from __future__ import annotations

import json
import os
from pathlib import Path
import traceback

from satellite_rl.training.run_monitor import atomic_json
from .storage import digest, read_json, now


def _teacher(directory, job, prep, progress, stopped):
    from .datasets import scene_set_files
    from .scene_order import VariableBeamSceneOrder
    from .s2p19_demonstrations import (generate_greedy_trajectory, pack_demonstrations,
                                     write_manifest, validate_manifest)
    record = job["scene_sets"]["train"]
    files = scene_set_files(record)
    refs = {row["sha256"]: (row, path) for row, (path, _) in zip(record["files"], files)}
    selection = prep["selected_sha256"]
    if len(selection) != 576 or len(set(selection)) != 576 or any(sha not in refs for sha in selection):
        raise ValueError("教师集不是冻结训练集的576个唯一选择")
    if len({refs[sha][0]["mother_scene_id"] for sha in selection}) != 576:
        raise ValueError("教师集每母必须恰选一个场景")
    order = VariableBeamSceneOrder("mixed_four_buckets", 51001,
        [sha for _, sha in files], record["files"])
    target = directory / "teacher"
    target.mkdir(exist_ok=True)
    state_path = directory / "preparation_progress.json"
    saved = read_json(state_path, {})
    records = saved.get("teacher_records", [])
    for index, item in enumerate(records):
        if (index >= len(selection) or item["scene_sha256"] != selection[index] or
                digest(target / item["path"]) != item["sha256"]):
            raise ValueError("已保存教师分片与冻结选择不一致")
    for index in range(len(records), len(selection)):
        if stopped():
            raise InterruptedError("教师准备收到停止请求")
        sha = selection[index]
        item = generate_greedy_trajectory(prep["teacher_config"], refs[sha][1], sha,
                                         target / f"teacher_{index:04d}.npz", stop=stopped)
        records.append(item)
        atomic_json(state_path, {"teacher_records": records, "updated_at": now()})
        progress({"env_step": len(records), "completed_mothers": len(records),
                  "teacher_transitions": sum(r["transition_count"] for r in records), "phase": "teacher_shards"})
    packed = pack_demonstrations(records, target, stop=stopped,
        progress=lambda done, total: progress({"phase": "teacher_pack", "packed_shards": done,
                                               "total_shards": total}))
    manifest = target / "manifest.json"
    write_manifest(prep["teacher_config"], order.source_set_sha256, records, manifest, packed=packed)
    validate_manifest(manifest, digest(manifest), prep["teacher_config"], order.source_set_sha256)
    return {"manifest_path": str(manifest), "manifest_sha256": digest(manifest),
            "source_set_sha256": order.source_set_sha256, "scene_count": len(records),
            "teacher_transitions": sum(r["transition_count"] for r in records)}


def run_job(job_dir, *, resume=False):
    from .runner import strict_cuda_preflight
    directory = Path(job_dir).resolve()
    job, config, status = {}, {}, {}
    initial_config_sha = None

    def progress(values):
        if digest(directory / "config.json") != initial_config_sha:
            raise ValueError("准备过程中冻结配置被改变")
        status.update(values, run_id=job["run_id"], pid=os.getpid(), heartbeat_at=now())
        atomic_json(directory / "status.json", status)

    def emergency_status(stage, error, **extra):
        """Never re-enter progress after it failed; preserve the first error."""
        try:
            latest = read_json(directory / "status.json", status) or status
        except BaseException:
            latest = status
        value = {**latest, "run_id": job.get("run_id", latest.get("run_id")),
                 "stage": stage, "pid": os.getpid(), "heartbeat_at": now(),
                 "error": str(error), **extra}
        atomic_json(directory / "status.json", value)
        status.clear()
        status.update(value)

    def stopped():
        return (directory / "stop.request").is_file()

    try:
        job = read_json(directory / "job.json")
        config = read_json(directory / "config.json")
        status = read_json(directory / "status.json", {})
        if not isinstance(job, dict) or not isinstance(config, dict):
            raise ValueError("准备worker缺少冻结job/config")
        if job.get("kind") != "prepare" or job.get("experiment_id") != "S2-P19":
            raise ValueError("准备worker仅接受正式S2-P19准备任务")
        initial_config_sha = digest(directory / "config.json")
        preflight = strict_cuda_preflight(job)
        atomic_json(directory / "manifest.json", {"run_id": job["run_id"], "purpose": "preparation",
            "config": config, "config_sha256": initial_config_sha, "preflight": preflight,
            "resume": resume, "created_at": now()})
        progress({"stage": "running", "phase": "preparation", "preflight": preflight})
        prep = config["preparation"]
        if prep["preparation_type"] == "demonstrations":
            result = _teacher(directory, job, prep, progress, stopped)
        else:
            from .s2p19_data import run_preparation
            result = run_preparation(directory, prep, progress_callback=progress, stop=stopped)
        atomic_json(directory / "preparation_result.json", result)
        atomic_json(directory / "results.json", {"summary": result, "scenes": [],
            "metadata": {"experiment_id": "S2-P19", "purpose": "preparation",
                         "run_id": job["run_id"], "git_commit": job["git_commit"]}})
        progress({"stage": "completed", "phase": "preparation_completed", "completed_at": now(),
                  "env_step": config["training"]["total_steps"],
                  "result_sha256": digest(directory / "preparation_result.json")})
    except InterruptedError as error:
        try:
            progress({"stage": "stopped", "stop_reason": str(error), "stopped_at": now()})
        except BaseException as status_error:
            # If the normal writer cannot publish a stop, this run is
            # interrupted, not safely stopped with a resumable receipt.
            emergency_status("interrupted", error, stop_reason=str(error),
                             status_write_error=str(status_error))
            raise
    except BaseException as error:
        failure = {"run_id": job.get("run_id"), "error": str(error),
                   "traceback": traceback.format_exc(), "last_status": status,
                   "initial_config_sha256": initial_config_sha, "at": now()}
        try:
            atomic_json(directory / "failure.json", failure)
        except BaseException as record_error:
            # Preserve the original traceback in status if failure.json is
            # unavailable. If even status cannot land, the nonzero exit lets
            # Scheduler mark the old running status as interrupted.
            emergency_status("interrupted", error, traceback=failure["traceback"],
                             failure_write_error=str(record_error))
            raise
        emergency_status("failed", error, traceback=failure["traceback"],
                         failed_at=now(), failure_path=str(directory / "failure.json"))
        raise
