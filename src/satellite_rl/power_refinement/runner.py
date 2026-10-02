"""Durable third-stage evaluation using frozen model, scenes and allocations."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from satellite_rl.workbench.runner import (
    _atomic_json, _evaluation_episode, _jsonl, _load_scene, _read_json,
    _sha256, isolated_rng, now, strict_cuda_preflight,
)
from satellite_rl.workbench.storage import clean_json


def _fingerprint(value):
    payload = json.dumps(clean_json(value), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _status(directory, job, **fields):
    previous = _read_json(directory / "status.json", {})
    previous.update(run_id=job["run_id"], heartbeat_at=now(), pid=os.getpid(),
                    research_stage="power_refinement", training_interactions=0,
                    update_step=0, progress_unit="allocation_method_cases", **fields)
    _atomic_json(directory / "status.json", previous)


def _beam_rows(initial, evaluation, power):
    capacities = np.asarray(evaluation.capacity_bps, dtype=float)
    demands = np.asarray([b["required_bps"] for b in initial["beams"]], dtype=float)
    si = np.minimum(capacities / demands, 1.0)
    from satellite_rl.workbench.environment import _sgm, beam_service_fields
    sgm = _sgm(capacities, demands, float(demands.mean()))
    rows = []
    for index, original in enumerate(initial["beams"]):
        rows.append({**original,
                     **beam_service_fields(capacities[index], demands[index], sgm[index]),
                     "power_w": float(power[index]), "si": float(si[index]),
                     "basic_met": bool(capacities[index] >= 0.1 * demands[index]),
                     "initial_capacity_bps": float(original["rate_bps"]),
                     "initial_power_w": float(original["power_w"]),
                     "service_timing": "after_power_refinement_all_interference"})
    return rows


def verify_solution(initial, beams, power_cap, options):
    """Independent metric acceptance, including zero power and all target beams."""
    initial_beams = initial["beams"]
    if len(beams) != len(initial_beams):
        raise ValueError("优化结果缺失目标波束")
    powers = np.array([b["power_w"] for b in beams])
    capacities = np.array([b["rate_bps"] for b in beams])
    demands = np.array([b["required_bps"] for b in beams])
    if (not np.isfinite(powers).all() or not np.isfinite(capacities).all()
            or np.any(powers < -1e-8) or np.any(capacities < -1e-6)):
        raise ValueError("功率或容量无效")
    if powers.sum() > power_cap + 1e-6:
        raise ValueError("优化结果超过总功率预算")
    for old, new in zip(initial_beams, beams):
        for key in ("beam_id", "original_index", "group", "freq", "slots", "required_bps"):
            if old[key] != new[key]:
                raise ValueError("功率细调改变了冻结波束、需求或频率槽")
        if not new["slots"] and (abs(new["power_w"]) > 1e-8 or abs(new["rate_bps"]) > 1e-6):
            raise ValueError("无槽波束不能产生辐射功率或容量")
    old_c = np.array([b["rate_bps"] for b in initial_beams])
    service_floor = np.minimum(old_c, 0.1 * demands)
    if np.any(capacities < service_floor - np.maximum(1e-3, demands * 1e-8)):
        raise ValueError("优化结果损害原基本服务保护")
    si = float(np.minimum(capacities / demands, 1).mean())
    from satellite_rl.workbench.environment import _sgm
    sgm = float(_sgm(capacities, demands, float(demands.mean())).mean())
    if si < float(initial["metrics"]["final_si_mean"]) - options["si_tolerance"] - 1e-10:
        raise ValueError("最终SI低于初始分配")
    if sgm < float(initial["metrics"]["final_sgm_mean"]) - options["sgm_tolerance"] - 1e-10:
        raise ValueError("最终SGM低于初始分配")
    return {"final_si_mean": si, "final_sgm_mean": sgm,
            "basic_met_count": int(np.sum(capacities >= 0.1 * demands)),
            "all_basic_met": bool(np.all(capacities >= 0.1 * demands)),
            "power_w": float(powers.sum()), "power_budget_w": float(power_cap),
            "throughput_bps": float(capacities.sum()),
            "effective_carried_bps": float(np.minimum(capacities, demands).sum()),
            "zero_capacity_count": int(np.sum(capacities <= 0)),
            "allocated_slots": sum(int(b["slots"]) for b in beams),
            "overload": False, "complete": bool(initial["metrics"]["complete"])}


def _write_csv(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    from satellite_rl.training.run_monitor import replace_with_retry
    replace_with_retry(temporary, path)


def _summaries(rows):
    output = {}
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        # Average repeated actions within a scene first; every scene has equal weight.
        scenes = sorted({r["source_scene_sha256"] for r in selected})
        summary = {"scene_count": len(scenes), "allocation_count": len(selected)}
        for key in ("final_si_mean", "final_sgm_mean", "basic_met_count", "all_basic_met",
                    "power_w", "effective_carried_bps", "throughput_bps"):
            summary[key] = float(np.mean([np.mean([r["metrics"][key] for r in selected
                                                  if r["source_scene_sha256"] == scene])
                                         for scene in scenes]))
        durations = [r["solver"]["elapsed_seconds"] for r in selected]
        summary.update(mean_solver_seconds=float(np.mean(durations)),
                       p95_solver_seconds=float(np.percentile(durations, 95)),
                       fallback_count=sum(bool(r["solver"].get("fallback_used")) for r in selected),
                       nonconverged_count=sum(not r["solver"].get("converged", False)
                                              for r in selected if method == "slsqp"))
        output[method] = summary
    return output


def finalize_results(directory, job, config, rows):
    summaries = _summaries(rows)
    table, beams = [], []
    for row in rows:
        common = {key: row[key] for key in ("run_id", "git_commit", "source_model_sha256",
                                           "scene_id", "source_scene_sha256", "repeat", "method",
                                           "initial_allocation_sha256", "power_budget_mode")}
        table.append({**common, **row["metrics"],
                      "solver_seconds": row["solver"]["elapsed_seconds"],
                      "solver_evaluations": row["solver"]["evaluations"],
                      "termination_reason": row["solver"]["termination_reason"],
                      "fallback_used": row["solver"].get("fallback_used", False)})
        beams.extend({**common, **beam} for beam in row["beams"])
    _write_csv(directory / "power_results.csv", table)
    _write_csv(directory / "power_beams.csv", beams)
    # Rebuild the canonical complete-case stream; partial attempts remain in iteration logs.
    temporary = directory / "power_cases.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(clean_json(row), ensure_ascii=False, allow_nan=False) + "\n")
    from satellite_rl.training.run_monitor import replace_with_retry
    replace_with_retry(temporary, directory / "power_cases.jsonl")
    # Only committed initial allocations enter the canonical trace stream.
    # Interrupted attempts remain separately identifiable for diagnosis.
    trace_temp = directory / "traces.jsonl.tmp"
    initial_paths = sorted({r["initial_path"] for r in rows})
    with trace_temp.open("w", encoding="utf-8") as stream:
        for path in initial_paths:
            initial = _read_json(path)
            for step in initial["trace"]:
                stream.write(json.dumps({**step, "initial_identity": initial["identity"]},
                                        ensure_ascii=False, allow_nan=False) + "\n")
    replace_with_retry(trace_temp, directory / "traces.jsonl")
    _atomic_json(directory / "results.json", {
        "research_stage": "power_refinement", "summary": summaries,
        "scenes": rows, "baselines": {}, "metadata": {
            "run_id": job["run_id"], "git_commit": job["git_commit"],
            "model_files": job["model_files"], "model_provenance": job.get("model_provenance"),
            "power_refinement": config["power_refinement"], "training_interactions": 0,
            "scope": "Fixed-frequency power refinement; repeated actions are not independent scenes",
            "capacity_semantics": "Environment link capacity; min(C,D) is estimated effective carriage",
        }})
    from satellite_rl.power_refinement.reporting import write_report
    write_report(directory, rows, summaries, config)
    artifacts = {p.name: {"path": str(p), "sha256": _sha256(p)} for p in
                 [directory / name for name in ("results.json", "power_results.csv", "power_beams.csv",
                                                "power_cases.jsonl", "power_report.md")]
                 if p.is_file()}
    _atomic_json(directory / "artifacts.json", {"files": artifacts, "created_at": now()})
    return summaries


def evaluate_refinement(directory, job, config, *, device="cuda"):
    """Testable worker body; production entry always performs strict CUDA preflight."""
    from satellite_rl.workbench.checkpoints import load_policy
    from satellite_rl.workbench.datasets import assert_scene_profile, scene_set_files
    from satellite_rl.workbench.environment import WorkbenchEnv
    from satellite_rl.power_refinement.physics import FixedAllocationPhysics
    from satellite_rl.power_refinement.optimizer import optimize_power

    directory = Path(directory)
    options = config["power_refinement"]
    for reference in job["model_files"]:
        if _sha256(reference["path"]) != reference["sha256"]:
            raise ValueError("冻结模型SHA不一致")
    policy = load_policy(job["model_path"], config=job["model_config"], device=device)
    dataset = job["scene_sets"]["test"]
    assert_scene_profile(dataset, "europe_v1")
    files = scene_set_files(dataset)
    budget = len(files) * options["repeats"] * len(options["methods"])
    # Scheduler resume flags are operational state, not scientific identity.
    identity = _fingerprint({"job": {k: job.get(k) for k in
        ("run_id", "git_commit", "source_root", "model_files", "model_config", "scene_sets")},
        "config": config})
    manifest_path = directory / "refinement_manifest.json"
    prior = _read_json(manifest_path)
    if prior and prior["identity"] != identity:
        raise ValueError("恢复任务配置或来源与原运行不一致")
    _atomic_json(manifest_path, {"identity": identity, "budget_cases": budget,
                                "scene_files": [(str(p), sha) for p, sha in files]})
    initials_dir = directory / "initial_allocations"
    cases_dir = directory / "power_cases"
    attempts_dir = directory / "initial_attempts"
    initials_dir.mkdir(exist_ok=True)
    cases_dir.mkdir(exist_ok=True)
    attempts_dir.mkdir(exist_ok=True)
    rows = []
    started = time.monotonic()
    stop_requested = lambda: (directory / "stop.request").exists()

    def progress(**fields):
        elapsed = time.monotonic() - started
        if "repeat" in fields:
            fields["repeat_index"] = fields["repeat"]
        _status(directory, job, stage="evaluating", env_step=len(rows), budget_env_steps=budget,
                wall_seconds=elapsed, **fields)

    progress()
    for index, (source, source_sha) in enumerate(files):
        if _sha256(source) != source_sha:
            raise ValueError("冻结场景SHA不一致")
        scene = _load_scene(source)
        for repeat in range(options["repeats"]):
            if stop_requested():
                _status(directory, job, stage="stopped", env_step=len(rows), budget_env_steps=budget)
                return None
            initial_path = initials_dir / f"scene_{index:05d}_r{repeat}.json"
            initial = _read_json(initial_path)
            if initial is None and initial_path.exists():
                raise ValueError("已保存初始分配文件损坏，原文件保留")
            seed = int(options["action_seed"]) + repeat
            initial_identity = _fingerprint([identity, source_sha, index, repeat, seed])
            if initial is None:
                before = time.monotonic()
                attempt_path = attempts_dir / f"scene_{index:05d}_r{repeat}_{time.time_ns()}.jsonl"
                with isolated_rng(seed):
                    if hasattr(policy, "agent") and hasattr(policy.agent, "rng"):
                        policy.agent.rng.seed(seed + 7321)
                    env = WorkbenchEnv(policy.config)
                    initial = _evaluation_episode(env, policy, scene, f"scene_{index:05d}_r{repeat}",
                                                  seed, options["action_mode"],
                                                  attempt_path, len(rows),
                                                  scene_sha256=source_sha, group="stage3_initial")
                initial.update(identity=initial_identity, source_scene_sha256=source_sha,
                               inference_seconds=time.monotonic() - before, action_seed=seed,
                               trace_attempt_path=str(attempt_path))
                _atomic_json(initial_path, initial)
            if initial.get("identity") != initial_identity:
                raise ValueError("初始分配与冻结场景或动作种子不一致")
            problem = FixedAllocationPhysics.from_scene(scene, policy.config, initial["beams"])
            p0 = np.array([b["power_w"] for b in initial["beams"]])
            reproduced = problem.evaluate(p0)
            expected = np.array([b["rate_bps"] for b in initial["beams"]])
            if not np.allclose(reproduced.capacity_bps, expected, rtol=1e-8, atol=1e-3):
                raise ValueError("新容量计算未复现原环境全束终态")
            cap = (min(float(p0.sum()), float(options["total_power_w"])) if options["power_budget_mode"] == "initial"
                   else float(options["total_power_w"]))
            allocation_sha = _fingerprint(initial["beams"])
            for method in options["methods"]:
                case_path = cases_dir / f"scene_{index:05d}_r{repeat}_{method}.json"
                case_identity = _fingerprint([initial_identity, allocation_sha, method, options])
                row = _read_json(case_path)
                if row is None and case_path.exists():
                    raise ValueError("已保存优化记录损坏，原文件保留")
                if row is not None:
                    if row.get("identity") != case_identity:
                        raise ValueError("已保存优化结果来源不一致")
                    restored = problem.evaluate([beam["power_w"] for beam in row["beams"]])
                    if not np.allclose(restored.capacity_bps, [b["rate_bps"] for b in row["beams"]],
                                       rtol=1e-8, atol=1e-3):
                        raise ValueError("已保存优化结果容量与物理复算不一致")
                    restored_metrics = verify_solution(initial, row["beams"], cap, options)
                    for key, value in restored_metrics.items():
                        if key not in row["metrics"] or not np.isclose(row["metrics"][key], value,
                                                                       rtol=1e-10, atol=1e-9):
                            raise ValueError("已保存优化指标与逐束重算不一致")
                    rows.append(row)
                    progress(scene_index=index, repeat=repeat, method=method)
                    continue
                if stop_requested():
                    _status(directory, job, stage="stopped", env_step=len(rows), budget_env_steps=budget)
                    return None
                progress(scene_index=index, repeat=repeat, method=method)

                def callback(entry):
                    _jsonl(directory / "power_iterations.jsonl", {
                        "time": now(), "scene_id": initial["scene_id"], "repeat": repeat,
                        "method": method, "case_identity": case_identity, **clean_json(entry)})
                    progress(scene_index=index, repeat=repeat, method=method,
                             optimizer_progress=clean_json(entry))

                solve_options = {"power_cap_w": cap, "maxiter": options["maxiter"],
                                 "max_evaluations": options["maxeval"],
                                 "time_limit_seconds": options["max_time_seconds"],
                                 "si_tolerance": options["si_tolerance"],
                                 "sgm_tolerance": options["sgm_tolerance"]}
                solved = optimize_power(problem, p0, method=method, options=solve_options,
                                        callback=callback, stop_requested=stop_requested)
                if stop_requested():
                    _status(directory, job, stage="stopped", env_step=len(rows), budget_env_steps=budget)
                    return None
                power = np.asarray(solved["power_w"], dtype=float)
                evaluated = problem.evaluate(power)
                beams = _beam_rows(initial, evaluated, power)
                metrics = verify_solution(initial, beams, cap, options)
                row = {"identity": case_identity, "run_id": job["run_id"],
                       "git_commit": job["git_commit"],
                       "source_model_sha256": next(ref["sha256"] for ref in job["model_files"]
                                                    if Path(ref["path"]).name == "policy.pt"),
                       "scene_id": initial["scene_id"],
                       "source_scene_sha256": source_sha, "source_scene_path": str(source),
                       "repeat": repeat, "method": method, "power_budget_mode": options["power_budget_mode"],
                       "initial_allocation_sha256": allocation_sha, "initial_path": str(initial_path),
                       "initial_metrics": initial["metrics"], "metrics": metrics,
                       "beams": beams, "solver": clean_json(solved),
                       "inference_seconds": initial["inference_seconds"],
                       "service_timing": "after_power_refinement_all_interference"}
                _atomic_json(case_path, row)
                rows.append(row)
                _jsonl(directory / "evaluations.jsonl", {"env_step": len(rows), "group": method,
                       "scene_id": row["scene_id"], "metrics": metrics})
                _atomic_json(directory / "refinement_checkpoint.json", {
                    "identity": identity, "completed_cases": len(rows),
                    "last_case": str(case_path), "last_case_sha256": _sha256(case_path)})
                progress(scene_index=index, repeat=repeat, method=method, metrics=metrics)
                print(f"{now()} {len(rows)}/{budget} {row['scene_id']} {method} "
                      f"SI={metrics['final_si_mean']:.6f} SGM={metrics['final_sgm_mean']:.6f} "
                      f"P={metrics['power_w']:.3f}W {solved['termination_reason']}", flush=True)
    summary = finalize_results(directory, job, config, rows)
    _status(directory, job, stage="completed", env_step=len(rows), budget_env_steps=budget,
            completed_at=now(), metrics=summary, wall_seconds=time.monotonic() - started)
    return summary


def run_refinement_job(job_dir, *, resume=False):
    directory = Path(job_dir).resolve()
    job = _read_json(directory / "job.json", {})
    try:
        from satellite_rl.power_refinement.config import validate_config
        config = _read_json(directory / "config.json", {})
        if config.get("research_stage") != "power_refinement" or job.get("kind") != "refine":
            raise ValueError("第三阶段任务阶段或类型无效")
        config["power_refinement"] = validate_config(config["power_refinement"])
        preflight = strict_cuda_preflight(job)
        import scipy
        preflight.update(scipy=scipy.__version__, optimizer_device="cpu", policy_device="cuda")
        _status(directory, job, stage="initializing", preflight=preflight, resume=bool(resume))
        _jsonl(directory / "events.jsonl", {"time": now(), "message": "refinement_worker_started",
                                             "resume": bool(resume), "preflight": preflight})
        return evaluate_refinement(directory, job, config)
    except BaseException as error:
        failure = {"time": now(), "run_id": job.get("run_id"), "error": str(error),
                   "traceback": traceback.format_exc(), "research_stage": "power_refinement"}
        _atomic_json(directory / "failure.json", failure)
        _status(directory, job, stage="failed", error=str(error), traceback=failure["traceback"])
        raise
