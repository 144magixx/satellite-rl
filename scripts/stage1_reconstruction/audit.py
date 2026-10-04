"""Independent read-only geometry, demand and gate audit for S1-03 outputs.

Imports scene decoding, but never the reconstruction environment or its metric
implementation. It does not call a policy, fit centres or generate new results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


BEAMS = 100
EARTH_KM, ORBIT_KM, SAT_LON = 6371.0, 42164.0, 13.0
RADIUS_LIMIT = 4 * math.sin(math.radians(.25) / 2) ** 2 + 1e-14
FULL_LIMIT = 4 * math.sin(math.radians(.5) / 2) ** 2 + 1e-14


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_method_contract(config, job, status, result, route, *, mechanism_step=None):
    if (config.get("coverage", {}).get("reconstruction", {}).get("route") != route or
            job.get("reconstruction_route") != route):
        raise ValueError("method label differs from actual frozen reconstruction route")
    if mechanism_step is None:
        if (job.get("kind") != "evaluate" or status.get("stage") != "completed" or
                result.get("run_id") != job.get("run_id") or result.get("route") != route or
                result.get("source_commit") != job.get("git_commit") or
                result.get("config_hash") != config_digest(config)):
            raise ValueError("method identity, completion or result provenance differs")
    else:
        if (route != "P1" or job.get("kind") != "train" or status.get("stage") != "stopped" or
                status.get("stop_reason") != "mechanism_gate_pending" or
                status.get("env_step") != mechanism_step or
                result.get("scheduled_step") != mechanism_step or
                result.get("actual_env_step") != mechanism_step or
                result.get("config_hash") != config_digest(config)):
            raise ValueError("mechanism task or exact validation node binding differs")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def rays(positions):
    a = np.deg2rad(np.asarray(positions, dtype=np.float64))
    ground = EARTH_KM * np.column_stack((np.cos(a[:, 0]) * np.cos(a[:, 1]),
                                       np.cos(a[:, 0]) * np.sin(a[:, 1]), np.sin(a[:, 0])))
    angle = math.radians(SAT_LON)
    value = ground - ORBIT_KM * np.array([math.cos(angle), math.sin(angle), 0.0])
    return value / np.linalg.norm(value, axis=1)[:, None]


def visible(positions):
    p = np.asarray(positions, dtype=np.float64)
    cosine = np.cos(np.deg2rad(p[:, 0])) * np.cos(np.deg2rad(p[:, 1] - SAT_LON))
    elevation = np.rad2deg(np.arctan2(cosine - EARTH_KM / ORBIT_KM,
                                     np.sqrt(np.maximum(0, 1 - cosine**2))))
    return ((p[:, 0] >= 25) & (p[:, 0] <= 80) & (p[:, 1] >= -40) &
            (p[:, 1] <= 50) & (elevation >= 10))


def edge_components(scene, ratio):
    users = rays(scene.positions)
    n = len(users)
    graph = np.empty((n, n), dtype=bool)
    for start in range(0, n, 128):
        graph[start:start+128] = ((users[start:start+128, None] - users[None])**2).sum(-1) <= FULL_LIMIT
    unseen = set(range(n))
    expected = []
    demands = np.asarray(scene.demands)
    while unseen:
        first = min(unseen)
        pending, members = [first], []
        unseen.remove(first)
        while pending:
            current = pending.pop()
            members.append(current)
            neighbors = unseen.intersection(np.flatnonzero(graph[current]).tolist())
            unseen.difference_update(neighbors)
            pending.extend(neighbors)
        members.sort()
        subtotal = demands[members].sum()
        if not (0 < subtotal < demands.sum() / BEAMS * ratio):
            continue
        points = users[members]
        if any(np.all(((points - point)**2).sum(-1) <= RADIUS_LIMIT) for point in points):
            expected.append(members)
    return sorted(expected)


def metrics(scene, centers, assignment, edge_ids, config):
    demand = np.asarray(scene.demands, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    owner = np.asarray(assignment)
    if (centers.shape != (100, 2) or owner.shape != (len(demand),) or
            not np.isfinite(centers).all() or not np.array_equal(owner, owner.astype(np.int64))):
        raise ValueError("invalid centre/association shapes or values")
    owner = owner.astype(np.int64)
    if np.any((owner < -1) | (owner >= 100)) or not visible(centers).all():
        raise ValueError("invalid association ID or GEO centre")
    dist = ((rays(scene.positions)[:, None] - rays(centers)[None])**2).sum(-1)
    covered = (dist <= RADIUS_LIMIT).any(-1)
    served = owner >= 0
    if not np.array_equal(covered, served):
        raise ValueError("geometrically covered user has no association, or inverse")
    if np.any(dist[np.flatnonzero(served), owner[served]] > RADIUS_LIMIT):
        raise ValueError("association points outside its beam")
    loads = np.bincount(owner[served], weights=demand[served], minlength=100)
    subject = np.array([i not in edge_ids for i in range(100)])
    values = loads[subject]
    mean = values.mean() if len(values) else 0
    tolerance = config["range_tolerance"]
    if mean > 0:
        normalized = values / mean
        cv = float(np.std(values) / mean)
        j = float(cv**2 + np.mean(np.maximum(np.abs(normalized - 1) - tolerance, 0)**2))
        count = int(np.count_nonzero(np.abs(normalized - 1) <= tolerance + 1e-12))
    else:
        cv, j, count = None, None, 0
    strict = bool(covered.all())
    bounded = j / (1+j) if j is not None else 1.0
    result = {"coverage_users": float(covered.mean()),
              "coverage_demand": float(demand[covered].sum() / demand.sum()),
              "coverage_100": strict, "uncovered_users": int((~covered).sum()),
              "uncovered_demand_bps": float(demand[~covered].sum()),
              "j": j, "quality": 2.0 * strict - bounded,
              "subject_cv": cv, "load_cv": float(loads.std()/loads.mean()) if loads.mean() > 0 else None,
              "subject_in_band_beams": count, "subject_beams": int(subject.sum()),
              "edge_beam_count": len(edge_ids), "empty_beams": int(np.count_nonzero(loads <= 0))}
    return result, loads, covered, owner


def audit_row(scene, start, row, coverage, *, expected_components=None):
    cfg = coverage["reconstruction"]
    if (row.get("scene_id") != scene.scene_id or row.get("scene_sha256") != scene.source_hash or
            start.get("scene_id") != scene.scene_id or start.get("scene_sha256") != scene.source_hash):
        raise ValueError("scene provenance mismatch")
    if start.get("state") is None:
        if row.get("initialization_valid") or row.get("centers_latlon_deg") is not None:
            raise ValueError("failed initialization was fabricated as a layout")
        return {"scene_id": scene.scene_id, "initialization_valid": False,
                "physical_metrics_defined": False, "initialization_error": start.get("initialization_error")}
    initial = start["state"]
    regions = initial["edge_regions"]
    expected = edge_components(scene, cfg["edge_demand_ratio"]) if expected_components is None else expected_components
    if sorted(sorted(region["aircraft_indices"]) for region in regions) != expected:
        raise ValueError("fixed edge certificate does not follow input geometry")
    edge_ids = set(initial["edge_beam_ids"])
    proven_ids = {region["dedicated_beam_id"] for region in regions if region.get("dedicated_beam_id") is not None}
    if edge_ids != proven_ids:
        raise ValueError("exempt beam IDs not bound to fixed certificates")
    initial_metrics, _, initially_covered, initial_owner = metrics(
        scene, initial["centers"], initial["assignment"], edge_ids, cfg)
    users = row["users"]
    if [u["aircraft_id"] for u in users] != scene.aircraft_ids:
        raise ValueError("user identity/order differs")
    if not np.allclose([u["demand_bps"] for u in users], scene.demands, rtol=1e-12, atol=1e-6):
        raise ValueError("user demand changed")
    owner = [u["assigned_beam"] if u["assigned_beam"] is not None else -1 for u in users]
    actual, loads, final_covered, final_owner = metrics(
        scene, row["centers_latlon_deg"], owner, edge_ids, cfg)
    if np.any(initially_covered & ~final_covered):
        raise ValueError("previously covered user lost")
    if row["initialization_valid"] != start["initialization_valid"]:
        raise ValueError("initialization qualification changed")
    if row["initialization_valid"] and actual["empty_beams"]:
        raise ValueError("qualified route ends with empty beams")
    for region in regions:
        beam = region.get("dedicated_beam_id")
        if beam is not None:
            members = np.asarray(region["aircraft_indices"], dtype=int)
            if (not np.all(final_owner[members] == beam) or
                    np.count_nonzero(final_owner == beam) != len(members) or
                    not np.all(initial_owner[members] == beam)):
                raise ValueError("fixed edge service was reassigned")
    if not np.allclose(loads, row["beam_load_demand_bps"], rtol=1e-12, atol=1e-5):
        raise ValueError("reported beam demand differs from unique assignments")
    actual["certified_edge_region_count"] = len(regions)
    actual["target_met"] = bool(actual["coverage_100"] and not actual["empty_beams"] and
        len(regions) <= cfg["max_edge_beams"] and actual["subject_cv"] is not None and
        actual["subject_cv"] <= cfg["subject_cv_target"] and
        actual["subject_in_band_beams"] >= cfg["min_subject_beams_in_range"])
    for key, expected_value in actual.items():
        recorded = row["metrics"].get(key)
        if expected_value is None:
            if recorded is not None:
                raise ValueError("undefined metric fabricated: " + key)
        elif not isinstance(recorded, (int, float, bool)) or not math.isclose(
                float(recorded), float(expected_value), rel_tol=1e-8, abs_tol=1e-6 if "bps" in key else 1e-8):
            raise ValueError("reported metric differs: " + key)
    if not (0 <= row["macro_steps"] <= cfg["max_macro_steps"] and
            0 <= row["geometry_calls"] <= cfg["max_geometry_calls_total"]):
        raise ValueError("macro/call budget exceeded")
    if len(row["actions"]) != row["macro_steps"]:
        raise ValueError("incomplete selected-action trace")
    parts = [float(row[key]) for key in ("initialization_seconds", "policy_seconds",
                                       "reconstruction_seconds", "final_check_seconds")]
    if any(not math.isfinite(x) or x < 0 for x in parts) or not math.isclose(
            sum(parts), row["total_seconds"], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("recorded timing segments do not sum")
    if not math.isclose(parts[0], start["initialization_seconds"], rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("common preparation duration changed")
    return {"scene_id": scene.scene_id, "initialization_valid": row["initialization_valid"],
            "physical_metrics_defined": True, "metrics": actual,
            "initial_quality": initial_metrics["quality"],
            "quality_gain": actual["quality"] - initial_metrics["quality"],
            "initialization_seconds": parts[0], "total_seconds": row["total_seconds"],
            "macro_steps": row["macro_steps"], "stop_reason": row["stop_reason"]}


def load_method(directory, *, route, node=None, mechanism_step=None):
    from satellite_rl.coverage.scenes import load_dataset
    directory = Path(directory)
    config, job = read(directory / "config.json"), read(directory / "job.json")
    if config.get("schema_version") != 4:
        raise ValueError("audit requires S1-03 schema4")
    role = "validation" if node is not None else next(iter(config["datasets"]))
    dataset_record = config["datasets"][role]
    if sha(dataset_record["path"]) != dataset_record["sha256"]:
        raise ValueError("frozen data SHA mismatch")
    dataset = load_dataset(dataset_record["path"], "cluster", config["coverage"])
    starts_record = config["reconstruction_starts"][role]
    if (sha(starts_record["path"]) != starts_record["sha256"] or
            starts_record["dataset_sha256"] != dataset_record["sha256"]):
        raise ValueError("frozen initialization SHA mismatch")
    starts = jsonl(starts_record["path"])
    path = Path(node) if node is not None else directory / "results.json"
    result = read(path)
    status = read(directory / "status.json")
    validate_method_contract(config, job, status, result, route, mechanism_step=mechanism_step)
    rows = result.get("rows", result.get("scenes"))
    if rows is None and node is None:
        raw_path = Path(result.get("rows_path", "")).resolve()
        if (raw_path.parent != directory.resolve() or raw_path.name != "results.partial.jsonl" or
                not raw_path.is_file() or sha(raw_path) != result.get("rows_sha256")):
            raise ValueError("raw scene JSONL location or completed SHA differs")
        rows = jsonl(raw_path)
        if result.get("scene_count") != len(rows):
            raise ValueError("raw scene JSONL count differs from completed result")
    if not isinstance(rows, list) or not (len(rows) == len(starts) == len(dataset)):
        raise ValueError("missing or duplicated scene results")
    if result.get("scene_count") != len(rows) or any(row.get("route") != route for row in rows):
        raise ValueError("completed scene count or per-scene method route differs")
    if node is not None:
        ledger = [entry for entry in jsonl(directory / "evaluations.jsonl")
                  if entry.get("scheduled_step") == mechanism_step]
        if (len(ledger) != 1 or ledger[0].get("actual_env_step") != mechanism_step or
                ledger[0].get("node_sha256") != sha(path) or
                Path(ledger[0].get("node", "")).resolve() != path.resolve() or
                ledger[0].get("scene_count") != len(rows)):
            raise ValueError("validation ledger node SHA/count/step differs")
    return config, job, dataset, starts, rows, path


def run(context):
    mode = context["mode"]
    count = context["expected_scenes"]
    criteria = context["criteria"]
    output = {"mode": mode, "plan_sha256": context["plan_sha256"],
              "science_commit": context["science_commit"], "expected_scenes": count,
              "checked_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "methods": {}, "issues": [], "gate_checks": {}, "script_sha256": sha(__file__)}
    methods = context["methods"]
    if mode == "mechanism":
        directory = Path(context["training_job_dir"])
        step = int(criteria["mechanism_step"])
        methods = {"P1": str(directory)}
        output.update(run_id=read(directory / "job.json")["run_id"], env_step=step,
                      checkpoint_sha256=sha(directory / "checkpoint.pt"))
    expected_methods = {"R0", "R1", "R2", "R3"} if mode == "development" else {"P1"} if mode == "mechanism" else {"R0", "R1", "R2", "R3", "P1"}
    if set(methods) != expected_methods:
        raise ValueError("gate method set differs from frozen scope")
    source_binding = None
    unique_run_ids = set()
    edge_cache = {}
    for route, entry in methods.items():
        directory = Path(entry if isinstance(entry, str) else entry["job_dir"])
        node = directory / "validation_nodes" / f"{criteria['mechanism_step']}.json" if mode == "mechanism" else None
        config, job, dataset, starts, rows, path = load_method(
            directory, route=route, node=node,
            mechanism_step=criteria["mechanism_step"] if mode == "mechanism" else None)
        if job["run_id"] in unique_run_ids:
            raise ValueError("one job reused as multiple comparison methods")
        unique_run_ids.add(job["run_id"])
        role = "validation" if mode == "mechanism" else next(iter(config["datasets"]))
        binding = (config["datasets"][role]["sha256"], config["reconstruction_starts"][role]["sha256"])
        if source_binding is None:
            source_binding = binding
        if binding != source_binding or len(rows) != count or job["git_commit"] != context["science_commit"]:
            raise ValueError("methods do not share frozen source, input, starts and denominator")
        checked = []
        for scene, start, row in zip(dataset.scenes, starts, rows):
            try:
                if scene.source_hash not in edge_cache:
                    edge_cache[scene.source_hash] = edge_components(scene, config["coverage"]["reconstruction"]["edge_demand_ratio"])
                checked.append(audit_row(scene, start, row, config["coverage"], expected_components=edge_cache[scene.source_hash]))
            except Exception as error:
                output["issues"].append({"route": route, "scene_id": scene.scene_id, "error": str(error)})
        output["methods"][route] = {"run_id": job["run_id"], "result_path": str(path),
            "result_sha256": sha(path), "dataset_sha256": binding[0], "starts_sha256": binding[1],
            "input_scenes": len(rows), "audited_scenes": len(checked), "rows": checked}
        if node is None:
            output["methods"][route].update(raw_rows_path=str(directory / "results.partial.jsonl"),
                                           raw_rows_sha256=sha(directory / "results.partial.jsonl"))
    threshold = float(criteria["min_quality_gain"])
    if mode == "development":
        base = output["methods"]["R0"]["rows"]
        valid = sum(r["initialization_valid"] for r in base)
        improved = {route: sum(r.get("quality_gain", -1) > threshold for r in output["methods"][route]["rows"])
                    for route in ("R1", "R3")}
        output["gate_checks"] = {"minimum_valid_starts": valid >= criteria["development_min_valid_scenes"],
            "action_improvement_exists": max(improved.values(), default=0) >= criteria["development_min_improved_scenes"]}
        output["gate_evidence"] = {"valid_starts": valid, "improved_scenes": improved}
    elif mode == "mechanism":
        import torch
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
        recorded_step = checkpoint.get("env_step", checkpoint.get("step"))
        weights = checkpoint.get("policy", checkpoint.get("policy_state", checkpoint.get("model")))
        # Exact checkpoint keys are validated by the runner restore contract.
        finite = isinstance(weights, dict) and bool(weights) and all(torch.isfinite(x).all().item() for x in weights.values())
        updates = jsonl(directory / "updates.jsonl")
        episodes = jsonl(directory / "episodes.jsonl")
        audited = output["methods"]["P1"]["rows"]
        gains = [r["quality_gain"] for r in audited if r.get("physical_metrics_defined")]
        immediate = sum(row.get("macro_steps") == 1 and row.get("stop_reason") == "stop" for row in episodes)
        accepted = total_proposals = 0
        for episode in episodes:
            for action in episode.get("actions", []):
                fields = action.get("requested_action_fields", {})
                if fields.get("operation") != "stop":
                    total_proposals += 1
                    accepted += bool(action.get("accepted"))
        recent_accepted = recent_proposals = 0
        cutoff = criteria["mechanism_step"] - config["training"]["validation_interval"]
        with (directory / "actions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                action = json.loads(line)
                if (cutoff < action.get("env_step", -1) <= criteria["mechanism_step"] and
                        action.get("requested_action_fields", {}).get("operation") != "stop"):
                    recent_proposals += 1
                    recent_accepted += bool(action.get("accepted"))
        recent_rate = recent_accepted/recent_proposals if recent_proposals else 0.0
        immediate_fraction = immediate / len(episodes) if episodes else 1.0
        output["gate_checks"] = {
            "checkpoint_step_matches": recorded_step == criteria["mechanism_step"],
            "checkpoint_binding": checkpoint.get("run_id") == output["run_id"] and
                checkpoint.get("source_commit") == context["science_commit"] and
                checkpoint.get("config_hash") == config_digest(config) and
                checkpoint.get("last_validation") == criteria["mechanism_step"],
            "weights_finite": finite,
            "updates_recorded": bool(updates) and updates[-1].get("env_step") == criteria["mechanism_step"] and updates[-1].get("update_step", 0) > 0,
            "validation_gain": len(gains) >= criteria["development_min_valid_scenes"] and float(np.mean(gains)) > threshold,
            "validation_improved_scenes": sum(gain > threshold for gain in gains) >= criteria["mechanism_min_improved_scenes"],
            "not_immediate_stop_collapse": immediate_fraction <= criteria["maximum_immediate_stop_fraction"],
            "accepted_actions_recorded": total_proposals > 0 and accepted > 0,
            "recent_action_validity": recent_proposals > 0 and recent_rate >= criteria["minimum_recent_nonstop_acceptance_rate"],
        }
        output["gate_evidence"] = {"validation_gain_mean": float(np.mean(gains)) if gains else None,
            "validation_improved_scenes": sum(gain > threshold for gain in gains),
            "complete_episodes": len(episodes), "immediate_stop_fraction": immediate_fraction,
            "non_stop_proposals": total_proposals, "accepted_actions": accepted,
            "recent_non_stop_proposals": recent_proposals, "recent_accepted_actions": recent_accepted,
            "recent_acceptance_rate": recent_rate,
            "validation_node_sha256": sha(directory / "validation_nodes" / f"{criteria['mechanism_step']}.json"),
            "updates_sha256": sha(directory / "updates.jsonl"), "episodes_sha256": sha(directory / "episodes.jsonl")}
    else:
        output["gate_checks"] = {"all_five_methods_complete": all(
            record["input_scenes"] == record["audited_scenes"] == count for record in output["methods"].values())}
    output["passed"] = not output["issues"] and all(output["gate_checks"].values())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    context = read(args.context)
    output = run(context)
    output["context_path"] = str(args.context)
    output["context_sha256"] = sha(args.context)
    destination = Path(context["output_path"])
    save(destination, output)
    print(json.dumps({"passed": output["passed"], "audit_path": str(destination), "audit_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
