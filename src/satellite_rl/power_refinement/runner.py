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


def _verify_saved_initial(initial, frozen_initial, source_reference):
    if (any(initial.get(key) != value for key, value in frozen_initial.items())
            or initial.get("source_evaluation") != source_reference
            or initial.get("initial_allocation_reused") is not True):
        raise ValueError("本任务初解副本与冻结第二阶段原始case不一致")


_FAILED_B_REPAIR = {
    "repair_type": "sca_basic_threshold_exact_v1",
    "origin_science_commit": "0a4742a9a15af93478501497fac5cd36579fa075",
    "original_run_id": "refine_sac_20261004_215930_e814197c",
    "request_id": "s2p17-v0-jan29-sca",
    "case_key": "scene_00033_r0_sca_si_lp",
    "scene_sha": "d7b8af50073632dcaeaa69b9af229feb4859d74a3e17f17f7b16cf092f3c0e8d",
    "initial_sha": "9fd3d48531cbdd5e29d63ae0cc24f2bdeb491e6996009c740a147b80c49d4e35",
    "A_case_sha256": "55398b6907276b949373712dbb3aa7da6388e96958e20b7a3ed857e6dcf170c9",
    "failure_sha256": "7fbacd63da943f4539df73dd8fd014552b9f0ae1c2646d363557ee710b8a9a41",
    "iterations_sha256": "ad3f1c46e2488d96180aea8d228bc431f85c25f587dbc25e0699b7fd4e00812e",
}


def _failed_attempt_contract(directory, job):
    """Read only the one approved, SHA-bound failed B attempt."""
    path = Path(directory) / "failed_attempt_fallback.json"
    if not path.exists():
        if job.get("boundary_repair") or job.get("failed_attempt_fallback_path"):
            raise ValueError("修复任务缺少已绑定的失败B回退合同")
        return None
    contract = _read_json(path)
    if not isinstance(contract, dict):
        raise ValueError("失败B回退合同损坏")
    if any(contract.get(key) != value for key, value in _FAILED_B_REPAIR.items()):
        raise ValueError("失败B回退合同与唯一批准案例不符")
    repair = job.get("boundary_repair") or {}
    if (job.get("run_id") != contract["original_run_id"] or
            job.get("request_id") != contract["request_id"] or
            job.get("scientific_origin_commit") != contract["origin_science_commit"] or
            job.get("original_git_commit") != contract["origin_science_commit"] or
            job.get("git_commit") != repair.get("repair_commit") or
            job.get("source_root") != repair.get("new_source_root") or
            repair.get("origin_science_commit") != contract["origin_science_commit"] or
            job.get("experiment_id") != "S2-P17"):
        raise ValueError("失败B回退合同与原工作台任务不符")
    if (repair.get("fallback_contract_sha256") != _sha256(path) or
            Path(job.get("failed_attempt_fallback_path", "")).resolve() != path.resolve() or
            _sha256(Path(job["source_root"]) / "source_manifest.json") !=
            repair.get("new_manifest_sha256")):
        raise ValueError("工作台任务未绑定失败B回退合同SHA")
    recovery_root = (Path(directory) / "recoveries").resolve()
    for field, digest in (("failure_path", "failure_sha256"),
                          ("iterations_path", "iterations_sha256")):
        evidence = Path(contract[field]).resolve(strict=True)
        if not evidence.is_relative_to(recovery_root) or _sha256(evidence) != contract[digest]:
            raise ValueError("失败B原始尝试证据路径或SHA不符")
    a_path = Path(contract["A_case_path"]).resolve(strict=True)
    expected_a = (Path(directory) / "power_cases/scene_00033_r0_sca.json").resolve()
    if a_path != expected_a or _sha256(a_path) != contract["A_case_sha256"]:
        raise ValueError("失败B回退所依赖的A案例SHA不符")
    bounds = contract.get("callback_lower_bounds") or {}
    if (bounds.get("capacity_calls") != 684 or bounds.get("accepted_callbacks") != 49 or
            not np.isclose(bounds.get("elapsed_seconds", -1), 1.0373822273686528,
                           rtol=0, atol=1e-12)):
        raise ValueError("失败B原始尝试下界与保留日志不符")
    if (set(contract.get("unknown_exact_fields", [])) !=
            {"original_attempt_capacity_calls", "original_attempt_optimizer_wall_seconds"}
            or not contract.get("missing_reason") or
            contract.get("fallback") != "failed_attempt_fallback_to_A" or
            contract.get("completed_prefix_case_count") != 299):
        raise ValueError("失败B调用/计时缺失或299案例前缀合同不完整")
    return {"path": str(path), "sha256": _sha256(path), "content": contract}


def _optimizer_seed(base, scene_index, repeat, method):
    """Disjoint scene/repeat/method streams for the supported repeat range."""
    if (type(base) is not int or not 0 <= base < 2**32
            or type(scene_index) is not int or scene_index < 0
            or type(repeat) is not int or not 0 <= repeat < 30
            or method not in ("de", "pso")):
        raise ValueError("Invalid optimizer stream identity")
    return (base + 1000 * scene_index + 10 * repeat + {"de": 1, "pso": 2}[method]) % 2**32


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


def _priority_solver_contract(solved, reference, options):
    """Check the saved SI anchor and budget independently of the optimizer."""
    phase1 = solved.get("phase1_best_si_mean")
    best = solved.get("best_feasible_si_mean")
    floor = solved.get("phase2_si_floor")
    if phase1 is None or best is None or not np.isfinite([phase1, best]).all():
        raise ValueError("SI优先求解器缺少有效的阶段最佳SI")
    if phase1 < reference["metrics"]["final_si_mean"] - options["si_tolerance"]:
        raise ValueError("SI阶段最佳值低于公共SCA参照")
    if best < phase1 - 1e-12:
        raise ValueError("SI最佳值随阶段推进下降")
    if floor is not None and (not np.isfinite(floor) or
                              abs(floor - (best - options["si_tolerance"])) > 1e-12):
        raise ValueError("SGM阶段锁定SI门槛与最佳可行SI不一致")
    if (solved["evaluations"] > options["maxeval"] or
            reference["solver"]["evaluations"] + solved["evaluations"] > 2 * options["maxeval"]):
        raise ValueError("SI优先完整路线超过冻结容量调用预算")
    return phase1, floor


def verify_solution(initial, beams, power_cap, options, *, reference=None,
                    phase1_best_si=None, locked_si_floor=None):
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
    initial_allowance = options["si_tolerance"] + (1e-10 if reference is None else 0.0)
    if si < float(initial["metrics"]["final_si_mean"]) - initial_allowance:
        raise ValueError("最终SI低于初始分配")
    if reference is None and sgm < float(initial["metrics"]["final_sgm_mean"]) - options["sgm_tolerance"] - 1e-10:
        raise ValueError("最终SGM低于初始分配")
    if reference is not None:
        reference_beams = reference["beams"]
        if len(reference_beams) != len(beams):
            raise ValueError("公共SCA参照缺失目标波束")
        reference_c = np.asarray([b["rate_bps"] for b in reference_beams], dtype=float)
        reference_si = float(reference["metrics"]["final_si_mean"])
        if si < reference_si - options["si_tolerance"]:
            raise ValueError("SI优先结果低于公共SCA参照")
        if phase1_best_si is not None and si < float(phase1_best_si) - options["si_tolerance"]:
            raise ValueError("最终SI低于第一阶段最佳可行SI")
        if locked_si_floor is not None and si < float(locked_si_floor):
            raise ValueError("最终SI低于SGM阶段最终锁定门槛")
        basic_threshold = 0.1 * demands
        # A beam already below the threshold must retain its reference capacity;
        # a beam above it must not become a new strict basic-service failure.
        reference_floor = np.where(reference_c >= basic_threshold,
                                   basic_threshold, reference_c)
        if np.any((reference_c >= basic_threshold) & (capacities < basic_threshold)):
            raise ValueError("SI优先结果新增严格基本服务跨下波束")
        if np.any(capacities < reference_floor - np.maximum(1e-3, demands * 1e-8)):
            raise ValueError("SI优先结果新增基本服务跨下或损害原不足波束")
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
        durations = [r["solver"].get("elapsed_seconds") for r in selected
                     if isinstance(r["solver"].get("elapsed_seconds"), (int, float))]
        summary.update(mean_solver_seconds=float(np.mean(durations)) if durations else None,
                       p95_solver_seconds=float(np.percentile(durations, 95)) if durations else None,
                       missing_solver_seconds_count=len(selected) - len(durations),
                       fallback_count=sum(bool(r["solver"].get("fallback_used")) for r in selected),
                       nonconverged_count=sum(not r["solver"].get("converged", False)
                                              for r in selected),
                       convergence_scope=("not_certified_for_original_objective" if method in ("sca", "sca_si_lp", "sca_si_lb", "wmmse")
                                          else "method_specific_stopping_rule"))
        from satellite_rl.power_refinement.timing import aggregate_scene_timing
        summary.update(aggregate_scene_timing(selected))
        output[method] = summary
    return output


def finalize_results(directory, job, config, rows):
    summaries = _summaries(rows)
    table, beams = [], []
    for row in rows:
        common = {key: row[key] for key in ("run_id", "git_commit", "source_model_sha256",
                                           "scene_id", "source_scene_sha256", "repeat", "method",
                                           "initial_allocation_sha256", "power_budget_mode")}
        for key in ("initial_source_group", "initializer_kind", "source_evaluation_run_id",
                    "source_workbench_model_sha256", "source_initial_case_path", "source_initial_case_sha256",
                    "beam_count", "scene_metadata"):
            if key in row:
                common[key] = row[key]
        if "reference_allocation_sha256" in row:
            common["reference_allocation_sha256"] = row["reference_allocation_sha256"]
        if "reference_case_sha256" in row:
            common["reference_case_sha256"] = row["reference_case_sha256"]
        common["optimizer_seed"] = row.get("optimizer_seed")
        table.append({**common, **row["metrics"],
                      **row.get("timing", {}),
                      "solver_seconds": row["solver"]["elapsed_seconds"],
                      "solver_evaluations": row["solver"]["evaluations"],
                      "reference_solver_evaluations": row.get("reference_solver_evaluations"),
                      "complete_route_solver_evaluations": row.get("complete_route_solver_evaluations"),
                      "complete_route_solver_evaluations_lower_bound":
                          row.get("complete_route_solver_evaluations_lower_bound"),
                      "original_failed_attempt_capacity_calls_lower_bound":
                          (row.get("failed_attempt_fallback") or {}).get(
                              "original_attempt_capacity_calls_lower_bound"),
                      "original_failed_attempt_optimizer_seconds_lower_bound":
                          (row.get("failed_attempt_fallback") or {}).get(
                              "original_attempt_optimizer_seconds_lower_bound"),
                      "original_failed_attempt_exact_cost_unknown":
                          (row.get("failed_attempt_fallback") or {}).get(
                              "original_attempt_exact_cost_unknown"),
                      "failed_attempt_contract_sha256":
                          (row.get("failed_attempt_fallback") or {}).get("contract_sha256"),
                      "phase1_best_si_mean": row["solver"].get("phase1_best_si_mean"),
                      "best_feasible_si_mean": row["solver"].get("best_feasible_si_mean"),
                      "locked_si_floor": row["solver"].get("phase2_si_floor"),
                      "reference_si_mean": (row.get("reference_metrics") or {}).get("final_si_mean"),
                      "reference_sgm_mean": (row.get("reference_metrics") or {}).get("final_sgm_mean"),
                      "complete_route_capacity_budget": row.get("complete_route_capacity_budget"),
                      "complete_route_optimizer_time_budget_seconds": row.get("complete_route_optimizer_time_budget_seconds"),
                      "termination_reason": row["solver"]["termination_reason"],
                      "fallback_used": row["solver"].get("fallback_used", False)})
        beams.extend({**common, **beam} for beam in row["beams"])
    _write_csv(directory / "power_results.csv", table)
    _write_csv(directory / "power_beams.csv", beams)
    scene_table = []
    for method in summaries:
        method_rows = [row for row in rows if row["method"] == method]
        for scene in sorted({row["source_scene_sha256"] for row in method_rows}):
            group = [row for row in method_rows if row["source_scene_sha256"] == scene]
            entry = {"source_scene_sha256": scene, "method": method, "repeats": len(group),
                     "final_si_mean": float(np.mean([r["metrics"]["final_si_mean"] for r in group])),
                     "final_sgm_mean": float(np.mean([r["metrics"]["final_sgm_mean"] for r in group])),
                     "power_w": float(np.mean([r["metrics"]["power_w"] for r in group]))}
            if all("timing" in r for r in group):
                timing_keys = ["initialization_seconds", "physics_setup_seconds", "optimizer_wall_seconds",
                            "optimizer_callback_io_seconds", "optimizer_excluding_callback_seconds",
                            "verification_seconds", "total_pipeline_seconds"]
                if all("reference_total_seconds" in r["timing"] for r in group):
                    timing_keys += ["reference_optimizer_wall_seconds",
                                    "reference_verification_seconds", "reference_total_seconds"]
                for key in timing_keys:
                    entry[key + "_sum"] = sum(r["timing"][key] for r in group)
                    entry[key + "_mean_per_action"] = entry[key + "_sum"] / len(group)
            scene_table.append(entry)
    _write_csv(directory / "power_scene_times.csv", scene_table)
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
            "initial_source": job.get("initial_source"),
            "scope": "Fixed-frequency power refinement; repeated actions are not independent scenes",
            "capacity_semantics": "Environment link capacity; min(C,D) is estimated effective carriage",
        }})
    from satellite_rl.power_refinement.reporting import write_report
    write_report(directory, rows, summaries, config)
    if any(item.get("missing_solver_seconds_count", 0) for item in summaries.values()):
        with (directory / "power_report.md").open("a", encoding="utf-8") as stream:
            stream.write("\n失败尝试回退案例的原B求解精确调用数和耗时未保存；"
                         "原日志记录的下界在结果案例及CSV中单列，缺项不补零。"
                         "已完成案例的求解耗时统计只包含有完整实测的案例。\n")
    artifacts = {p.name: {"path": str(p), "sha256": _sha256(p)} for p in
                 [directory / name for name in ("results.json", "power_results.csv", "power_beams.csv",
                                                "power_cases.jsonl", "power_scene_times.csv", "power_report.md")]
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
    from satellite_rl.power_refinement.config import SI_PRIORITY_METHODS

    directory = Path(directory)
    options = config["power_refinement"]
    failed_fallback = _failed_attempt_contract(directory, job)
    for reference in job["model_files"]:
        if _sha256(reference["path"]) != reference["sha256"]:
            raise ValueError("冻结模型SHA不一致")
    saved = None
    if job.get("initial_source"):
        from satellite_rl.workbench.refinement_sources import source_rows
        saved = source_rows(job["initial_source"])
        policy = None
        physics_config = job["model_config"]
    else:
        policy = load_policy(job["model_path"], config=job["model_config"], device=device)
        physics_config = policy.config
    source_partition = ((job.get("initial_source") or {}).get("partition", "test")
                        if job.get("experiment_id") == "S2-P19" else "test")
    if source_partition not in ("development", "validation", "test"):
        raise ValueError("S2-P19 SCA来源分区无效")
    dataset = job["scene_sets"][source_partition]
    if (dataset.get("source") or {}).get("partition_role") != source_partition:
        raise ValueError("SCA冻结场景分区与已保存初解分区不一致")
    assert_scene_profile(dataset, "europe_v1")
    files = scene_set_files(dataset)
    budget = len(files) * options["repeats"] * len(options["methods"])
    # Scheduler resume flags are operational state, not scientific identity.
    job_identity = {k: job.get(k) for k in
        ("run_id", "git_commit", "source_root", "model_files", "model_config", "scene_sets")}
    if job.get("boundary_repair"):
        # Source migration changes execution provenance, not the scientific
        # task/case identity used by the first 299 committed results.
        proof = job["boundary_repair"]
        job_identity["git_commit"] = proof["origin_science_commit"]
        job_identity["source_root"] = proof["old_source_root"]
    if saved is not None:
        job_identity["initial_source"] = job["initial_source"]
    if options["methods"] == list(SI_PRIORITY_METHODS):
        job_identity.update(priority_solver_deps_path=job.get("priority_solver_deps_path"),
                            priority_solver_dependency_files=job.get("priority_solver_dependency_files"))
    identity = _fingerprint({"job": job_identity, "config": config})
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
            if initial is None and saved is not None:
                from copy import deepcopy
                initial = deepcopy(saved[(source_sha, repeat)])
                initial.update(identity=initial_identity, source_scene_sha256=source_sha,
                               source_evaluation=job["initial_source"],
                               initial_allocation_reused=True)
                _atomic_json(initial_path, initial)
            if initial is None:
                if str(device).startswith("cuda"):
                    import torch
                    torch.cuda.synchronize()
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
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                initial.update(identity=initial_identity, source_scene_sha256=source_sha,
                               inference_seconds=time.monotonic() - before, action_seed=seed,
                               trace_attempt_path=str(attempt_path))
                _atomic_json(initial_path, initial)
            if initial.get("identity") != initial_identity:
                raise ValueError("初始分配与冻结场景或动作种子不一致")
            if saved is not None:
                frozen_initial = saved[(source_sha, repeat)]
                _verify_saved_initial(initial, frozen_initial, job["initial_source"])
            setup_started = time.perf_counter()
            problem = FixedAllocationPhysics.from_scene(scene, physics_config, initial["beams"])
            p0 = np.array([b["power_w"] for b in initial["beams"]])
            reproduced = problem.evaluate(p0)
            expected = np.array([b["rate_bps"] for b in initial["beams"]])
            if not np.allclose(reproduced.capacity_bps, expected, rtol=1e-8, atol=1e-3):
                raise ValueError("新容量计算未复现原环境全束终态")
            if "physics_setup_seconds" not in initial:
                initial["physics_setup_seconds"] = time.perf_counter() - setup_started
                _atomic_json(initial_path, initial)
            cap = (min(float(p0.sum()), float(options["total_power_w"])) if options["power_budget_mode"] == "initial"
                   else float(options["total_power_w"]))
            allocation_sha = _fingerprint(initial["beams"])
            for method in options["methods"]:
                case_path = cases_dir / f"scene_{index:05d}_r{repeat}_{method}.json"
                reference = None
                reference_sha = None
                reference_case_sha = None
                if method in SI_PRIORITY_METHODS[1:]:
                    reference_path = cases_dir / f"scene_{index:05d}_r{repeat}_sca.json"
                    reference = _read_json(reference_path)
                    if reference is None:
                        raise ValueError("SI优先追加求解必须先完成并保存公共SCA参照")
                    expected_reference_identity = _fingerprint([initial_identity, allocation_sha,
                                                                "sca", options])
                    if reference.get("identity") != expected_reference_identity:
                        raise ValueError("公共SCA参照来源与冻结初始分配不一致")
                    reference_sha = _fingerprint(reference["beams"])
                    reference_case_sha = _sha256(reference_path)
                case_identity = _fingerprint([initial_identity, allocation_sha, method, options]
                                             + ([reference_sha, reference_case_sha] if reference else []))
                case_key = f"scene_{index:05d}_r{repeat}_{method}"
                repair_this_case = (failed_fallback is not None and
                                    case_key == _FAILED_B_REPAIR["case_key"])
                if repair_this_case:
                    contract = failed_fallback["content"]
                    if (len(rows) != contract["completed_prefix_case_count"] or
                            source_sha != contract["scene_sha"] or
                            allocation_sha != contract["initial_sha"] or
                            reference_case_sha != contract["A_case_sha256"] or
                            Path(contract["A_case_path"]).resolve() != reference_path.resolve()):
                        raise ValueError("失败B回退不对应原299案例、场景、初解或A参照")
                    if contract.get("case_identity") not in (None, case_identity):
                        raise ValueError("失败B回退案例身份改变")
                row = _read_json(case_path)
                if row is None and case_path.exists():
                    raise ValueError("已保存优化记录损坏，原文件保留")
                if row is not None:
                    if row.get("identity") != case_identity:
                        raise ValueError("已保存优化结果来源不一致")
                    if (method == "sca" and options["methods"] == list(SI_PRIORITY_METHODS)
                            and row.get("reference_allocation_sha256") != _fingerprint(row["beams"])):
                        raise ValueError("已保存公共SCA参照最终分配SHA不一致")
                    if reference is not None and (row.get("reference_allocation_sha256") != reference_sha
                                                  or row.get("reference_case_sha256") != reference_case_sha):
                        raise ValueError("已保存SI优先结果的公共SCA来源不一致")
                    row_repair = row.get("failed_attempt_fallback")
                    if row_repair is not None:
                        if (not repair_this_case or
                                row_repair.get("contract_sha256") != failed_fallback["sha256"] or
                                row["solver"].get("termination_reason") != "failed_attempt_fallback_to_A" or
                                row["solver"].get("evaluations", "unexpected") is not None or
                                row["solver"].get("elapsed_seconds", "unexpected") is not None or
                                row["solver"].get("additional_optimizer_calls") != 0 or
                                row_repair.get("additional_optimizer_calls") != 0 or
                                row_repair.get("original_attempt_capacity_calls_lower_bound") !=
                                    contract["callback_lower_bounds"]["capacity_calls"] or
                                not np.isclose(
                                    row_repair.get("original_attempt_optimizer_seconds_lower_bound", -1),
                                    contract["callback_lower_bounds"]["elapsed_seconds"],
                                    rtol=0, atol=1e-12) or
                                row_repair.get("original_attempt_exact_cost_unknown") is not True or
                                row.get("git_commit") != job["git_commit"] or
                                "timing" in row or
                                _fingerprint(row["beams"]) != reference_sha):
                            raise ValueError("已保存失败B回退案例与冻结A或合同不一致")
                    if reference is not None and row_repair is None:
                        _priority_solver_contract(row["solver"], reference, options)
                    restored = problem.evaluate([beam["power_w"] for beam in row["beams"]])
                    if not np.allclose(restored.capacity_bps, [b["rate_bps"] for b in row["beams"]],
                                       rtol=1e-8, atol=1e-3):
                        raise ValueError("已保存优化结果容量与物理复算不一致")
                    restored_metrics = verify_solution(initial, row["beams"], cap, options,
                        reference=reference,
                        phase1_best_si=(row["solver"].get("phase1_best_si_mean")
                                        if reference and row_repair is None else None),
                        locked_si_floor=(row["solver"].get("phase2_si_floor")
                                         if reference and row_repair is None else None))
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

                callback_seconds = 0.0
                def callback(entry):
                    nonlocal callback_seconds
                    callback_started = time.perf_counter()
                    _jsonl(directory / "power_iterations.jsonl", {
                        "time": now(), "scene_id": initial["scene_id"], "repeat": repeat,
                        "method": method, "case_identity": case_identity, **clean_json(entry)})
                    progress(scene_index=index, repeat=repeat, method=method,
                             optimizer_progress=clean_json(entry))
                    callback_seconds += time.perf_counter() - callback_started

                solve_options = {"power_cap_w": cap, "maxiter": options["maxiter"],
                                 "max_evaluations": options["maxeval"],
                                 "time_limit_seconds": options["max_time_seconds"],
                                 "si_tolerance": options["si_tolerance"],
                                 "sgm_tolerance": options["sgm_tolerance"]}
                optimizer_seed = None
                if method in ("de", "pso"):
                    optimizer_seed = _optimizer_seed(options["optimizer_seed"], index, repeat, method)
                    solve_options.update(optimizer_seed=optimizer_seed,
                        **{key: options[key] for key in ("population_size", "de_mutation", "de_crossover",
                                                        "pso_inertia", "pso_cognitive", "pso_social")})
                if reference is not None:
                    from satellite_rl.power_refinement.priority import optimize_priority
                    solve_options.update({key: options[key] for key in
                        ("si_stall_window", "si_stall_tolerance", "si_min_evaluations",
                         "si_max_fraction", "service_margin_ratio")})
                    solve_options["baseline_service_floor_bps"] = np.minimum(
                        np.asarray([beam["rate_bps"] for beam in initial["beams"]], dtype=float),
                        0.1 * problem.demand_bps)
                if repair_this_case:
                    # The original B attempt reached the final guard but did
                    # not persist its power vector or exact cost.  Reuse the
                    # already audited A allocation without another optimizer
                    # call; never report the missing original cost as zero.
                    solved = {"method": method,
                              "power_w": [beam["power_w"] for beam in reference["beams"]],
                              "evaluations": None, "elapsed_seconds": None,
                              "termination_reason": "failed_attempt_fallback_to_A",
                              "fallback_used": True, "converged": False,
                              "history": [],
                              "convergence_scope": "original_failed_attempt_not_certified",
                              "phase1_best_si_mean": None,
                              "best_feasible_si_mean": None,
                              "phase2_si_floor": None,
                              "additional_optimizer_calls": 0,
                              "original_attempt_capacity_calls_lower_bound":
                                  contract["callback_lower_bounds"]["capacity_calls"],
                              "original_attempt_optimizer_seconds_lower_bound":
                                  contract["callback_lower_bounds"]["elapsed_seconds"],
                              "original_attempt_capacity_calls_exact": None,
                              "original_attempt_optimizer_seconds_exact": None,
                              "original_attempt_missing_reason": contract["missing_reason"]}
                    solver_wall = None
                else:
                    solver_started = time.perf_counter()
                    if reference is None:
                        solved = optimize_power(problem, p0, method=method, options=solve_options,
                                                callback=callback, stop_requested=stop_requested)
                    else:
                        solved = optimize_priority(problem,
                            np.asarray([beam["power_w"] for beam in reference["beams"]], dtype=float),
                            method=method, options=solve_options, callback=callback,
                            stop_requested=stop_requested)
                    solver_wall = time.perf_counter() - solver_started
                if stop_requested():
                    _status(directory, job, stage="stopped", env_step=len(rows), budget_env_steps=budget)
                    return None
                verify_started = time.perf_counter()
                power = np.asarray(solved["power_w"], dtype=float)
                evaluated = problem.evaluate(power)
                beams = _beam_rows(initial, evaluated, power)
                phase1_best_si = (_priority_solver_contract(solved, reference, options)[0]
                                  if reference is not None and not repair_this_case else None)
                metrics = verify_solution(initial, beams, cap, options,
                                          reference=reference, phase1_best_si=phase1_best_si,
                                          locked_si_floor=solved.get("phase2_si_floor") if reference else None)
                from satellite_rl.power_refinement.timing import build_timing
                verification_seconds = time.perf_counter() - verify_started
                timing = (None if repair_this_case else
                          build_timing(initial["inference_seconds"], initial["physics_setup_seconds"],
                                       solver_wall, callback_seconds, verification_seconds,
                                       **({"reference_optimizer": reference["timing"]["optimizer_wall_seconds"],
                                           "reference_verification": reference["timing"]["verification_seconds"]}
                                          if reference is not None else {})))
                source_model_sha = next((ref["sha256"] for ref in job["model_files"]
                                         if Path(ref["path"]).name == "policy.pt"), None)
                if saved is None and source_model_sha is None:
                    raise ValueError("直接模型SCA缺冻结模型权重")
                row = {"identity": case_identity, "run_id": job["run_id"],
                       "git_commit": job["git_commit"],
                       "source_model_sha256": source_model_sha,
                       "scene_id": initial["scene_id"],
                       "source_scene_sha256": source_sha, "source_scene_path": str(source),
                       "repeat": repeat, "method": method, "power_budget_mode": options["power_budget_mode"],
                       "optimizer_seed": optimizer_seed,
                       "initial_allocation_sha256": allocation_sha, "initial_path": str(initial_path),
                       "initial_metrics": initial["metrics"], "metrics": metrics,
                       "beams": beams, "solver": clean_json(solved),
                       "inference_seconds": initial["inference_seconds"],
                       "initial_source": job.get("initial_source"),
                       "beam_count": len(initial["beams"]),
                       "scene_metadata": initial.get("scene_metadata"),
                       "service_timing": "after_power_refinement_all_interference"}
                if timing is not None:
                    row["timing"] = timing
                else:
                    row["failed_attempt_fallback"] = {
                        "repair_type": contract["repair_type"],
                        "contract_path": failed_fallback["path"],
                        "contract_sha256": failed_fallback["sha256"],
                        "source_A_case_sha256": reference_case_sha,
                        "original_attempt_capacity_calls_lower_bound":
                            contract["callback_lower_bounds"]["capacity_calls"],
                        "original_attempt_optimizer_seconds_lower_bound":
                            contract["callback_lower_bounds"]["elapsed_seconds"],
                        "original_attempt_exact_cost_unknown": True,
                        "additional_optimizer_calls": 0,
                        "missing_reason": contract["missing_reason"],
                    }
                    row["repair_verification_seconds"] = verification_seconds
                if job.get("initial_source"):
                    source_group = job["initial_source"]["group"]
                    row.update(initial_source_group=source_group,
                               initializer_kind="model" if source_group == "model" else "baseline_" + source_group,
                               source_evaluation_run_id=job["initial_source"]["run_id"],
                               source_partition=source_partition,
                               source_workbench_model_sha256=row["source_model_sha256"],
                               source_initial_case_path=initial.get("original_case_path"),
                               source_initial_case_sha256=initial.get("original_case_file_sha256"))
                    if source_group != "model":
                        row["source_model_sha256"] = None
                if reference is not None:
                    row.update(reference_allocation_sha256=reference_sha,
                               reference_case_sha256=reference_case_sha,
                               reference_method="sca",
                               reference_case_path=str(reference_path),
                               reference_metrics=reference["metrics"],
                               phase1_best_si_mean=phase1_best_si,
                               best_feasible_si_mean=solved.get("best_feasible_si_mean"),
                               locked_si_floor=solved.get("phase2_si_floor"),
                               reference_solver_evaluations=reference["solver"]["evaluations"],
                               complete_route_solver_evaluations=(None if repair_this_case else
                                   reference["solver"]["evaluations"] + solved["evaluations"]),
                               complete_route_solver_evaluations_lower_bound=(
                                   reference["solver"]["evaluations"] +
                                   contract["callback_lower_bounds"]["capacity_calls"]
                                   if repair_this_case else None),
                               complete_route_capacity_budget=2 * options["maxeval"],
                               complete_route_optimizer_time_budget_seconds=2 * options["max_time_seconds"],
                               reference_sgm_rule="legacy_sca_protection",
                               priority_sgm_rule="si_first_sgm_may_decline")
                elif method == "sca" and options["methods"] == list(SI_PRIORITY_METHODS):
                    row.update(reference_allocation_sha256=_fingerprint(beams),
                               reference_method="sca", reference_sgm_rule="legacy_sca_protection")
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
        raw_controls = config["power_refinement"]
        config["power_refinement"] = validate_config(raw_controls)
        # Preserve the original identity of jobs frozen before the SI-priority
        # fields existed. Their existing manifest must still be resumable.
        if "si_stall_window" not in raw_controls:
            for key in ("si_stall_window", "si_stall_tolerance", "si_min_evaluations",
                        "si_max_fraction", "service_margin_ratio"):
                config["power_refinement"].pop(key)
        preflight = strict_cuda_preflight(job)
        import scipy
        preflight.update(scipy=scipy.__version__, optimizer_device="cpu",
                         policy_device=("saved_initial_no_policy_inference"
                                        if job.get("experiment_id") == "S2-P19" and
                                           job.get("initial_source") else "cuda"))
        from satellite_rl.power_refinement.config import SI_PRIORITY_METHODS
        if config["power_refinement"]["methods"] == list(SI_PRIORITY_METHODS):
            import cvxpy
            import clarabel
            dependency_root = Path(job.get("priority_solver_deps_path", ""))
            if not dependency_root.is_absolute() or not dependency_root.is_dir():
                raise ValueError("冻结的SI优先求解依赖目录无效")
            dependency_root = dependency_root.resolve(strict=True)
            dependency_files = job.get("priority_solver_dependency_files") or []
            expected_files = {str(dependency_root / f"{package}-{version}.dist-info" / name)
                              for package, version in (("cvxpy", "1.6.5"), ("clarabel", "0.11.1"))
                              for name in ("METADATA", "RECORD")}
            if {item.get("path") for item in dependency_files} != expected_files:
                raise ValueError("冻结求解依赖清单不完整")
            for item in dependency_files:
                file = Path(item["path"])
                if not file.is_file() or _sha256(file) != item["sha256"]:
                    raise ValueError("冻结求解依赖清单SHA不一致")
            if (not Path(cvxpy.__file__).resolve().is_relative_to(dependency_root) or
                    not Path(clarabel.__file__).resolve().is_relative_to(dependency_root)):
                raise ValueError("SI优先依赖未从冻结隔离目录导入")
            if cvxpy.__version__ != "1.6.5" or clarabel.__version__ != "0.11.1":
                raise ValueError("SI优先求解依赖版本与冻结配置不一致")
            if "CLARABEL" not in cvxpy.installed_solvers():
                raise ValueError("SI优先求解缺少CLARABEL求解器")
            preflight["priority_solver"] = {
                "dependency_root": str(dependency_root),
                "distribution_manifest_sha256": _fingerprint(dependency_files),
                "cvxpy_version": cvxpy.__version__, "cvxpy_file": cvxpy.__file__,
                "clarabel_version": clarabel.__version__, "clarabel_file": clarabel.__file__,
                "solver": "CLARABEL", "optimizer_device": "cpu"}
        import platform
        preflight.update(cpu_description=platform.processor(), cpu_logical_count=os.cpu_count(),
            cpu_affinity_count=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            thread_environment={key: os.environ.get(key) for key in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
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
