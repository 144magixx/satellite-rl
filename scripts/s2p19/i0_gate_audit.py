"""Read-only S2-P19 I0 development gate from saved direct and final-B cases.

This module performs no allocation, model inference, or optimization. Run it
against the completed 20418 workbench archive through a separate transport
wrapper; its output is a source-backed scientific decision, never a training
task or approval by itself.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


EXPERIMENT = "S2-P19"
DIRECT_RUN = "evaluate_sac_20261007_153239_e442d472"
B_RUNS = {"greedy": "refine_sac_20261007_154539_38e92936",
          "rule": "refine_sac_20261007_154541_e3178010",
          "random": "refine_sac_20261007_154543_8adc79ad"}
N_VALUES = (50, 100, 125, 150, 175, 200)
LARGE_N = frozenset((125, 150, 175, 200))
LOADS = frozenset((0.75, 1.25))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def physical_beams(beams: list[dict], count: int, *, power_limit: float = 3000.0) -> dict:
    """Recalculate all-target SI and the final common resource account."""
    require(isinstance(beams, list) and len(beams) == count and 50 <= count <= 200,
            "missing all target beams")
    occupied = {group: set() for group in range(8)}
    si, basic, power, slots_total = 0.0, 0, 0.0, 0
    for index, beam in enumerate(beams):
        require(beam.get("beam_id") == index and beam.get("group") == index % 8,
                "beam identity/reuse group differs")
        slots, freq = beam.get("slots"), beam.get("freq")
        require(type(slots) is int and type(freq) is int and 0 <= slots <= 10
                and 0 <= freq < 100 and freq + slots <= 100,
                "slot block illegal")
        demand = float(beam["required_bps"])
        capacity = float(beam["rate_bps"])
        watt = float(beam["power_w"])
        require(all(math.isfinite(value) for value in (demand, capacity, watt))
                and demand > 0 and capacity >= -1e-6 and watt >= -1e-8,
                "beam demand/capacity/power illegal")
        if slots == 0:
            require(abs(capacity) <= 1e-6 and abs(watt) <= 1e-8,
                    "empty-slot beam has capacity/power")
        for frequency in range(freq, freq + slots):
            require(frequency not in occupied[index % 8], "reuse-group frequency collision")
            occupied[index % 8].add(frequency)
        si += min(max(capacity, 0.0) / demand, 1.0)
        basic += capacity >= .1 * demand
        power += max(watt, 0.0)
        slots_total += slots
    require(power <= power_limit + 1e-6, "total power exceeds common budget")
    return {"si": si / count, "basic_met_count": basic,
            "power_w": power, "slots_total": slots_total}


def candidate_thresholds(paired_deltas: dict[str, list[tuple[int, float]]]) -> dict:
    out = {}
    for group in ("rule", "random"):
        rows = paired_deltas[group]
        require(len(rows) == 72, f"{group} paired scene count differs")
        all_delta = mean(delta for _, delta in rows)
        large = [delta for n, delta in rows if n in LARGE_N]
        require(len(large) == 48, f"{group} large-N paired scene count differs")
        large_delta = mean(large)
        out[group] = {"mean_delta": all_delta, "large_n_mean_delta": large_delta,
                      "overall_at_least_half_pp": all_delta >= .005,
                      "large_at_least_one_pp": large_delta >= .01,
                      "passes_performance": all_delta >= .005 and large_delta >= .01}
    return out


def audit_archive(archive_root: Path) -> dict:
    archive_root = Path(archive_root).resolve(strict=True)
    jobs = archive_root / "outputs/workbench/jobs"
    direct_root = jobs / DIRECT_RUN
    direct_job = read_json(direct_root / "job.json")
    direct_status = read_json(direct_root / "status.json")
    require(direct_job.get("experiment_id") == EXPERIMENT and
            direct_job.get("git_commit") == "47da239910485508f7c978461314dff92f8672ec" and
            direct_status.get("stage") == "completed" and direct_status.get("env_step") == 216,
            "direct source not exact completed 47da 216-case gate")
    direct_path = direct_root / "results.json"
    direct_result = read_json(direct_path)
    direct_summaries = direct_result.get("scenes") or []
    require(len(direct_summaries) == 216, "direct source case count differs")
    source_files = [{"path": str(path), "sha256": digest(path)} for path in
                    (direct_root / "job.json", direct_root / "config.json", direct_root / "status.json", direct_path)]
    direct: dict[str, dict[str, tuple[dict, dict]]] = {group: {} for group in B_RUNS}
    for item in direct_summaries:
        group, scene_sha = item.get("method"), item.get("scene_sha256")
        require(group in direct and scene_sha not in direct[group], "direct method/scene duplicate")
        path = Path(item["case_path"]).resolve(strict=True)
        require(path.is_relative_to((direct_root / "cases").resolve()) and
                digest(path) == item["case_sha256"], "direct case path/SHA differs")
        case = read_json(path)
        require(case.get("method") == group and case.get("scene_sha256") == scene_sha
                and case.get("repeat_index") == 0 and case.get("action_seed") == 53001,
                "direct case identity differs")
        n = int(item["beam_count"])
        observed = physical_beams(case["beams"], n)
        require(abs(observed["si"] - float(case["metrics"]["si"])) <= 1e-7 and
                observed["basic_met_count"] == case["metrics"]["basic_met"] and
                observed["slots_total"] == case["metrics"]["slots_total"] and
                abs(observed["power_w"] - case["metrics"]["power_w"]) <= 1e-5,
                "direct physical metrics differ")
        require(case["metrics"]["decisions"] <= 32 and case["metrics"]["probes"] <= 32,
                "direct local-search decision/probe cap exceeded")
        direct[group][scene_sha] = (item, case)
        source_files.append({"path": str(path), "sha256": item["case_sha256"]})
    scene_set = set(direct["greedy"])
    require(len(scene_set) == 72 and all(set(rows) == scene_set for rows in direct.values()),
            "direct three-way scene SHA set differs")
    mothers = set()
    by_n = Counter()
    by_load = Counter()
    for scene_sha in scene_set:
        base = direct["greedy"][scene_sha][1]
        metadata = base["scene_metadata"]
        n = len(base["beams"])
        mothers.add(metadata["mother_scene_id"])
        by_n[n] += 1
        by_load[float(metadata["load_multiplier"])] += 1
        for group in ("rule", "random"):
            case = direct[group][scene_sha][1]
            metrics, baseline = case["metrics"], base["metrics"]
            require(abs(metrics["initial_si"] - baseline["si"]) <= 1e-7 and
                    abs(metrics["initial_sgm"] - baseline["sgm"]) <= 1e-7 and
                    metrics["si"] + 1e-8 >= baseline["si"] and
                    metrics["sgm"] + 1e-8 >= baseline["sgm"],
                    "selected direct output violates shared SI/SGM guard")
            for initial, selected in zip(base["beams"], case["beams"], strict=True):
                require(abs(float(initial["required_bps"]) - float(selected["required_bps"])) <= 1e-6,
                        "paired beam demand differs")
                if float(initial["rate_bps"]) >= .1 * float(initial["required_bps"]):
                    require(float(selected["rate_bps"]) >= .1 * float(selected["required_bps"]),
                            "selected direct output lost a baseline basic-service beam")
    require(len(mothers) == 6 and set(by_n) == set(N_VALUES) and
            all(by_n[n] == 12 for n in N_VALUES) and set(by_load) == LOADS,
            "development mother/N/load distribution differs")

    final_by_group = {}
    for group, run_id in B_RUNS.items():
        root = jobs / run_id
        job, config, status = (read_json(root / name) for name in ("job.json", "config.json", "status.json"))
        reference = job.get("initial_source") or {}
        controls = config.get("power_refinement") or {}
        require(job.get("experiment_id") == EXPERIMENT and
                job.get("git_commit") == direct_job["git_commit"] and
                status.get("stage") == "completed" and status.get("env_step") == 216 and
                reference.get("run_id") == DIRECT_RUN and reference.get("group") == group and
                reference.get("partition") == "development" and
                reference.get("summary_sha256") == digest(direct_path) and
                reference.get("scene_count") == 72 and reference.get("repeats") == 1 and
                len(reference.get("case_files") or []) == 72 and
                controls.get("methods") == ["none", "sca", "sca_si_lp"] and
                controls.get("maxeval") == 2000 and controls.get("max_time_seconds") == 30.0 and
                controls.get("total_power_w") == 3000 and controls.get("action_seed") == 53001 and
                controls.get("repeats") == 1 and
                controls.get("action_mode") == reference.get("action_mode"),
                f"{group} frozen complete-B task/source/config differs")
        for name in ("job.json", "config.json", "status.json", "results.json"):
            path = root / name
            source_files.append({"path": str(path), "sha256": digest(path)})
        case_dir = root / "power_cases"
        all_cases = list(case_dir.glob("*.json"))
        require(len(all_cases) == 216, f"{group} B method-case count differs")
        routes = sorted(case_dir.glob("*_sca_si_lp.json"))
        require(len(routes) == 72, f"{group} final-B case count differs")
        values = {}
        for path in routes:
            case_sha = digest(path)
            case = read_json(path)
            scene_sha = case.get("source_scene_sha256")
            require(case.get("run_id") == run_id and case.get("method") == "sca_si_lp" and
                    case.get("initial_source_group") == group and
                    case.get("source_evaluation_run_id") == DIRECT_RUN and
                    case.get("source_partition") == "development" and scene_sha in direct[group] and
                    scene_sha not in values and case.get("repeat") == 0,
                    f"{group} final-B source identity differs")
            initial_item, initial_case = direct[group][scene_sha]
            require(case.get("source_initial_case_sha256") == initial_item["case_sha256"] and
                    Path(case["source_initial_case_path"]).resolve() == Path(initial_item["case_path"]).resolve(),
                    f"{group} final-B initial-case SHA/path differs")
            n = int(case["beam_count"])
            result = physical_beams(case["beams"], n)
            metrics = case["metrics"]
            require(metrics.get("complete") is True and metrics.get("overload") is False and
                    abs(result["si"] - float(metrics["final_si_mean"])) <= 1e-7 and
                    result["basic_met_count"] == metrics["basic_met_count"] and
                    result["slots_total"] == metrics["allocated_slots"] and
                    abs(result["power_w"] - metrics["power_w"]) <= 1e-5,
                    f"{group} final-B physical metrics differ")
            for direct_beam, refined_beam in zip(initial_case["beams"], case["beams"], strict=True):
                require((direct_beam["freq"], direct_beam["slots"]) ==
                        (refined_beam["freq"], refined_beam["slots"]),
                        "SCA changed fixed initial frequency-slot layout")
            values[scene_sha] = {"n": n, "si": result["si"],
                                 "mother_scene_id": case["scene_metadata"]["mother_scene_id"]}
            source_files.append({"path": str(path), "sha256": case_sha})
        require(set(values) == scene_set, f"{group} final-B scene set incomplete")
        final_by_group[group] = values
    paired_deltas = {group: [(value["n"], value["si"] - final_by_group["greedy"][scene_sha]["si"])
                             for scene_sha, value in final_by_group[group].items()]
                     for group in ("rule", "random")}
    performance = candidate_thresholds(paired_deltas)
    passed_performance = any(item["passes_performance"] for item in performance.values())
    criteria = {"source_completed_and_sha_verified": True,
                "same_72_scene_pairs_six_mothers": True,
                "direct_shared_protection_and_32_probe_budget": True,
                "all_final_B_resource_and_fixed_layout_checks": True,
                "one_candidate_exceeds_half_pp_overall_and_one_pp_large_N": passed_performance}
    return {"experiment_id": EXPERIMENT,
            "source_commit": "47da239910485508f7c978461314dff92f8672ec",
            "purpose": "I0_development_nonlearning_gate",
            "passed": all(criteria.values()), "criteria": criteria,
            "direct_run_id": DIRECT_RUN, "B_run_ids": B_RUNS,
            "independent_mothers": len(mothers), "paired_scenes": len(scene_set),
            "large_N_paired_scenes": 48, "N_counts": dict(sorted(by_n.items())),
            "final_B_mean_si": {group: mean(item["si"] for item in rows.values())
                                for group, rows in final_by_group.items()},
            "performance": performance,
            "source_files": source_files,
            "timing_scope": "development gate concurrent with C0 training; not formal algorithm timing"}
