"""Read-only independent audit before reusing the three S1-03 e174 starts.

Run with the remote DQN Python in an isolated ops directory after both source
snapshots exist. This tool reads frozen jobs and NPZ/JSONL bytes; it never
generates starts, evaluates a policy, or writes into either workbench.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


OLD_COMMIT = "e1745eccc1f47aa0b2f4ede678ce7789e030c1ea"
NEW_COMMIT = "c98085acf8457c3f56dd31f7e6de47ecc5fa6b0d"
PREPARES = {
    "validation": ("evaluate_ppo_20261005_042617_987692dd", 20),
    "train": ("evaluate_ppo_20261005_043425_ebaab1e0", 576),
    "test": ("evaluate_ppo_20261005_050939_c90e9d14", 100),
}
SCIENTIFIC_FILES = (
    "src/satellite_rl/coverage/reconstruction_config.py",
    "src/satellite_rl/coverage/reconstruction_runner.py",
    "src/satellite_rl/coverage/scenes.py",
    "src/satellite_rl/coverage/geometry.py",
    "src/satellite_rl/coverage/config.py",
    "src/satellite_rl/envs/regions.py",
)
RADIUS2 = 4.0 * math.sin(math.radians(.25) / 2.0) ** 2 + 1e-14
FULL2 = 4.0 * math.sin(math.radians(.5) / 2.0) ** 2 + 1e-14
SATELLITE = 42164.0 * np.array((math.cos(math.radians(13.0)),
                                 math.sin(math.radians(13.0)), 0.0))


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked_commit(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value) or value == OLD_COMMIT:
        raise ValueError("new science commit must be a distinct full 40-digit lowercase SHA")
    return value


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def under(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"path leaves frozen root: {path}")
    return resolved


def rays(points: np.ndarray) -> np.ndarray:
    a = np.deg2rad(np.asarray(points, dtype=np.float64))
    ground = 6371.0 * np.column_stack((np.cos(a[:, 0]) * np.cos(a[:, 1]),
                                       np.cos(a[:, 0]) * np.sin(a[:, 1]), np.sin(a[:, 0])))
    direction = ground - SATELLITE
    return direction / np.linalg.norm(direction, axis=1)[:, None]


def visible(points: np.ndarray) -> np.ndarray:
    a = np.asarray(points, dtype=np.float64)
    cosine = np.cos(np.deg2rad(a[:, 0])) * np.cos(np.deg2rad(a[:, 1] - 13.0))
    elevation = np.rad2deg(np.arctan2(cosine - 6371.0 / 42164.0,
                                     np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))))
    return ((a[:, 0] >= 25) & (a[:, 0] <= 80) & (a[:, 1] >= -40) &
            (a[:, 1] <= 50) & (elevation >= 10))


def edge_components(user_rays: np.ndarray, demand: np.ndarray, ratio: float) -> list[list[int]]:
    n = len(user_rays)
    parent = np.arange(n, dtype=np.int64)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for lo in range(0, n, 128):
        distances = ((user_rays[lo:lo + 128, None] - user_rays[None]) ** 2).sum(-1)
        for offset, row in enumerate(distances):
            i = lo + offset
            for j in np.flatnonzero(row[i + 1:] <= FULL2) + i + 1:
                a, b = root(i), root(int(j))
                if a != b:
                    parent[b] = a
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    threshold = float(demand.sum()) / 100.0 * ratio
    result = []
    for members in groups.values():
        subtotal = float(demand[members].sum())
        if not 0 < subtotal < threshold:
            continue
        unit = user_rays[members]
        if any(np.all(((unit - candidate) ** 2).sum(-1) <= RADIUS2)
               for candidate in unit):
            result.append(members)
    return sorted(result)


def audit_scene(scene: dict, start: dict, *, edge_ratio: float) -> dict:
    """Verify one untouched initial state without importing reconstruction code."""
    scene_id, source_hash = scene["scene_id"], scene["source_hash"]
    if (start.get("scene_id") != scene_id or start.get("scene_sha256") != source_hash):
        raise ValueError("start scene ID/source hash differs from frozen NPZ")
    elapsed = start.get("initialization_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("initialization duration missing or invalid; never fill with zero")
    state = start.get("state")
    if state is None:
        if start.get("initialization_valid") is not False or not start.get("initialization_error"):
            raise ValueError("failed start lacks explicit reason")
        return {"scene_id": scene_id, "source_hash": source_hash,
                "initialization_valid": False, "failure_reason": start["initialization_error"],
                "initialization_seconds": float(elapsed),
                "initialization_source": None, "initialization_candidates": None,
                "coverage_users": None, "coverage_demand": None,
                "edge_beam_ids": None, "positive_beams": None}
    ids = scene["aircraft_ids"]
    if (state.get("scene_id") != scene_id or state.get("source_hash") != source_hash or
            state.get("aircraft_ids") != ids):
        raise ValueError("state aircraft/source identity differs from NPZ")
    positions, demands = scene["positions"], scene["demands"]
    centers = np.asarray(state["centers"], dtype=np.float64)
    assignment = np.asarray(state["assignment"])
    if (centers.shape != (100, 2) or len(assignment) != len(demands) or
            not np.isfinite(centers).all() or not np.all(visible(centers)) or
            not np.issubdtype(assignment.dtype, np.integer)):
        raise ValueError("illegal GEO centres or association shape/type")
    assignment = assignment.astype(np.int64)
    if np.any((assignment < -1) | (assignment >= 100)):
        raise ValueError("invalid beam identity")
    user_rays = rays(positions)
    distance = ((user_rays[:, None] - rays(centers)[None]) ** 2).sum(-1)
    covered = np.any(distance <= RADIUS2, axis=1)
    assigned = assignment >= 0
    if not np.array_equal(covered, assigned):
        raise ValueError("geometrically covered users and assigned users differ")
    if np.any(distance[np.flatnonzero(assigned), assignment[assigned]] > RADIUS2):
        raise ValueError("assigned user lies outside its actual GEO beam")
    loads = np.bincount(assignment[assigned], weights=demands[assigned], minlength=100)
    total = float(demands.sum())
    if total <= 0 or not np.isclose(loads.sum(), float(demands[assigned].sum()),
                                    rtol=1e-12, atol=1e-5):
        raise ValueError("unique assignment demand ledger differs")
    coverage_users = float(covered.mean())
    coverage_demand = float(demands[covered].sum() / total)
    qualified = bool(np.all(loads > 0) and coverage_users >= .999 and
                     coverage_demand >= .999)
    failure_reasons = []
    if np.any(loads <= 0):
        failure_reasons.append("nonpositive_beam_load")
    if coverage_users < .999:
        failure_reasons.append("user_coverage_below_99_9")
    if coverage_demand < .999:
        failure_reasons.append("demand_coverage_below_99_9")
    if qualified != bool(state.get("initialization_valid")) or qualified != bool(
            start.get("initialization_valid")):
        raise ValueError("recorded initialization qualification differs")
    expected = edge_components(user_rays, demands, edge_ratio)
    regions = state.get("edge_regions")
    if not isinstance(regions, list) or sorted(sorted(r["aircraft_indices"]) for r in regions) != expected:
        raise ValueError("frozen edge components differ from raw geometry")
    dedicated = set()
    for region in regions:
        members = np.asarray(region["aircraft_indices"], dtype=np.int64)
        if len(members) == 0 or not np.isclose(float(region["demand_bps"]),
                                               float(demands[members].sum()), rtol=1e-12, atol=1e-5):
            raise ValueError("edge component demand differs")
        witness = np.asarray(region["witness_center"], dtype=np.float64).reshape(1, 2)
        if (not bool(visible(witness)[0]) or
                not np.any(np.all(positions[members] == witness[0], axis=1)) or
                np.any(((user_rays[members] - rays(witness)[0]) ** 2).sum(-1) > RADIUS2)):
            raise ValueError("edge witness is not a legal one-beam certificate")
        beam = region.get("dedicated_beam_id")
        if beam is not None:
            if (type(beam) is not int or beam not in range(100) or beam in dedicated or
                    not np.all(assignment[members] == beam) or
                    not np.array_equal(np.flatnonzero(assignment == beam), members)):
                raise ValueError("frozen dedicated edge beam identity changed")
            dedicated.add(beam)
    if sorted(dedicated) != sorted(state.get("edge_beam_ids", [])):
        raise ValueError("edge beam list and certificate identities differ")
    source = state.get("initialization_source")
    candidates = state.get("initialization_candidates")
    state_elapsed = state.get("elapsed_initialization_s")
    repair_elapsed = state.get("elapsed_repair_s")
    if (not isinstance(source, str) or not source or not isinstance(candidates, list) or
            {r.get("source") for r in candidates} != {
                "scene_centers_seed", "recomputed_cluster", "recomputed_rule"} or
            not isinstance(state_elapsed, (int, float)) or
            not isinstance(repair_elapsed, (int, float)) or
            not 0 <= repair_elapsed <= state_elapsed <= elapsed + 1e-3):
        raise ValueError("common initialization source or measured timing is missing")
    for candidate in candidates:
        if (not isinstance(candidate.get("seconds"), (int, float)) or
                not math.isfinite(candidate["seconds"]) or candidate["seconds"] < 0):
            raise ValueError("initialization candidate has missing measured time")
    for key in ("initialization_repair_geometry_calls",
                "initialization_repair_ray_pair_checks", "initialization_repair_attempts"):
        if (type(state.get(key)) is not int or state[key] < 0 or
                start.get(key) != state[key]):
            raise ValueError(f"initialization repair ledger differs: {key}")
    if state.get("geometry_calls") != 0 or state.get("steps") != 0:
        raise ValueError("prepare spent training macro-action budget")
    return {"scene_id": scene_id, "source_hash": source_hash,
            "initialization_valid": qualified,
            "failure_reason": None if qualified else ",".join(failure_reasons),
            "coverage_users": coverage_users, "coverage_demand": coverage_demand,
            "covered_users": int(covered.sum()), "total_users": len(covered),
            "served_demand_bps": float(demands[assigned].sum()),
            "total_demand_bps": total, "positive_beams": int(np.count_nonzero(loads > 0)),
            "edge_region_count": len(regions), "edge_beam_ids": sorted(dedicated),
            "initialization_seconds": float(elapsed),
            "initialization_state_seconds": float(state_elapsed),
            "initialization_repair_seconds": float(repair_elapsed),
            "initialization_source": source,
            "initialization_candidates": [{"source": c["source"], "seconds": c["seconds"],
                                           "valid": c.get("valid")}
                                          for c in candidates],
            "initialization_repair_geometry_calls": state[
                "initialization_repair_geometry_calls"],
            "initialization_repair_ray_pair_checks": state[
                "initialization_repair_ray_pair_checks"],
            "initialization_repair_attempts": state["initialization_repair_attempts"]}


def npz_scenes(path: Path) -> tuple[list[dict], dict]:
    with np.load(path, allow_pickle=False) as data:
        if str(data["format_version"]) != "coverage_aircraft_v1":
            raise ValueError("NPZ is not aircraft-level coverage v1")
        lengths = data["lengths"]
        positions, demands = data["positions"], data["demands"]
        ids = json.loads(str(data["aircraft_ids_json"]))
        names = data["scene_ids"].tolist()
        hashes = data["source_hashes"].tolist()
        metadata = json.loads(str(data["metadata_json"]))
        if (lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer) or
                positions.shape != (len(lengths), positions.shape[1], 2) or
                demands.shape != positions.shape[:2] or
                len(ids) != len(names) or len(names) != len(hashes) or
                len(hashes) != len(lengths)):
            raise ValueError("prepared NPZ arrays have incompatible lengths")
        scenes = []
        for i, length in enumerate(lengths):
            count = int(length)
            if (count <= 0 or count > positions.shape[1] or len(ids[i]) != count or
                    np.any(positions[i, count:] != 0) or np.any(demands[i, count:] != 0) or
                    not np.isfinite(positions[i, :count]).all() or
                    not np.isfinite(demands[i, :count]).all() or
                    np.any(demands[i, :count] < 0) or
                    len(set(ids[i])) != count or not np.all(visible(positions[i, :count]))):
                raise ValueError(f"prepared scene {i} has malformed aircraft data")
            scenes.append({"scene_id": names[i], "source_hash": hashes[i],
                           "aircraft_ids": ids[i],
                           "positions": positions[i, :count].copy(),
                           "demands": demands[i, :count].copy()})
    return scenes, metadata


def audit_source_compatibility(old_root: Path, new_root: Path,
                               new_science_commit: str = NEW_COMMIT) -> dict:
    new_science_commit = checked_commit(new_science_commit)
    old_manifest_path = old_root / "source_manifest.json"
    new_manifest_path = new_root / "source_manifest.json"
    old_manifest, new_manifest = read(old_manifest_path), read(new_manifest_path)
    if (old_manifest.get("git_commit") != OLD_COMMIT or
            new_manifest.get("git_commit") != new_science_commit):
        raise ValueError("frozen source manifest commit identity differs")
    for root, manifest in ((old_root, old_manifest), (new_root, new_manifest)):
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("empty source manifest")
        for name, expected in files.items():
            if sha(under(root / name, root)) != expected:
                raise ValueError(f"frozen source file SHA mismatch: {name}")
    same = {}
    for name in SCIENTIFIC_FILES:
        old, new = (under(old_root / name, old_root),
                    under(new_root / name, new_root))
        if old.read_bytes() != new.read_bytes():
            raise ValueError(f"initialization/physical dependency changed: {name}")
        same[name] = sha(old)
    core_name = "src/satellite_rl/coverage/reconstruction.py"
    old_core = under(old_root / core_name, old_root).read_bytes()
    new_core = under(new_root / core_name, new_root).read_bytes()
    inserted = b"                    next_state._distance_cache = proposal._distance_cache\n"
    if new_core.count(inserted) != 1 or new_core.replace(inserted, b"", 1) != old_core:
        raise ValueError("scientific core differs by more than verified accepted-cache propagation")
    return {"old_source_manifest_sha256": sha(old_manifest_path),
            "new_source_manifest_sha256": sha(new_manifest_path),
            "old_core_sha256": hashlib.sha256(old_core).hexdigest(),
            "new_core_sha256": hashlib.sha256(new_core).hexdigest(),
            "only_scientific_change": "accepted_branch_propagates_verified_distance_cache",
            "unchanged_initialization_dependencies": same,
            "old_manifest_file_count": len(old_manifest["files"]),
            "new_manifest_file_count": len(new_manifest["files"])}


def audit_prepare(job_dir: Path, archive_root: Path, old_source: Path,
                  role: str, expected_run: str, expected_count: int) -> dict:
    if job_dir.name != expected_run:
        raise ValueError("prepare task ID differs from frozen contract")
    paths = {name: under(job_dir / f"{name}.json", job_dir)
             for name in ("job", "config", "status", "artifacts")}
    job, config, status, artifacts = (read(paths[name]) for name in paths)
    if (job.get("run_id") != expected_run or job.get("git_commit") != OLD_COMMIT or
            Path(job.get("source_root", "")).resolve() != old_source.resolve() or
            job.get("kind") != "evaluate" or job.get("reconstruction_route") != "prepare" or
            status.get("stage") != "completed" or
            status.get("prepared_scenes") != expected_count or
            status.get("preparation_total") != expected_count or
            set(config.get("datasets", {})) != {role} or
            config.get("coverage", {}).get("reconstruction", {}).get("route") != "prepare"):
        raise ValueError("prepare task status, source or role differs")
    dataset_record = config["datasets"][role]
    dataset_path = under(Path(dataset_record["path"]), archive_root)
    starts_path = under(Path(artifacts["reconstruction_starts"]), archive_root)
    if (sha(dataset_path) != dataset_record["sha256"] or
            sha(starts_path) != artifacts["reconstruction_starts_sha256"] or
            artifacts.get("dataset_sha256") != dataset_record["sha256"] or
            artifacts.get("run_id") != expected_run or
            Path(artifacts.get("source_root", "")).resolve() != old_source.resolve()):
        raise ValueError("NPZ/start SHA or prepare artifact provenance differs")
    record_path = under(dataset_path.parent / "record.json", archive_root)
    record = read(record_path)
    if (record.get("prepared_npz") != dataset_path.name or
            record.get("prepared_npz_sha256") != dataset_record["sha256"] or
            record.get("partition_role") != role or record.get("count") != expected_count):
        raise ValueError("NPZ provenance record differs")
    scenes, metadata = npz_scenes(dataset_path)
    starts = jsonl(starts_path)
    if (len(scenes) != expected_count or len(starts) != expected_count or
            dataset_record.get("id") is None or
            dataset_record.get("partition_role") != role or
            dataset_record.get("count") != expected_count or
            metadata.get("initializer_sha256") != dataset_record.get("initializer_sha256")):
        raise ValueError("NPZ/start count or initializer provenance differs")
    if [(s["scene_id"], s["source_hash"]) for s in scenes] != [
            (r["scene_id"], r["source_hash"]) for r in record["scenes"]]:
        raise ValueError("NPZ scene order/hash differs from provenance record")
    rows, issues = [], []
    ratio = config["coverage"]["reconstruction"]["edge_demand_ratio"]
    for index, (scene, start) in enumerate(zip(scenes, starts)):
        try:
            rows.append(audit_scene(scene, start, edge_ratio=ratio))
        except Exception as error:
            issues.append({"index": index, "scene_id": scene["scene_id"],
                           "reason": str(error)})
            rows.append({"scene_id": scene["scene_id"], "source_hash": scene["source_hash"],
                         "initialization_valid": None, "failure_reason": str(error),
                         "initialization_seconds": start.get("initialization_seconds"),
                         "initialization_source": None})
    qualified = sum(row["initialization_valid"] is True for row in rows)
    return {"role": role, "prepare_run_id": expected_run,
            "source_run_id": expected_run,
            "scene_count": expected_count, "qualified_count": qualified,
            "failed_count": expected_count - qualified,
            "dataset_sha256": dataset_record["sha256"],
            "starts_sha256": artifacts["reconstruction_starts_sha256"],
            "source_dataset_id": dataset_record["id"],
            "dataset_path": str(dataset_path), "starts_path": str(starts_path),
            "source_provenance": {"npz_record_sha256": sha(record_path),
                                  "npz_record_path": str(record_path),
                                  "source_record_sha256": record.get("source_record_sha256"),
                                  "source_dates": record.get("source_dates"),
                                  "dataset_config_sha256": sha(paths["config"]),
                                  "job_sha256": sha(paths["job"]),
                                  "status_sha256": sha(paths["status"]),
                                  "artifacts_sha256": sha(paths["artifacts"]),
                                  "initializer_sha256": dataset_record.get("initializer_sha256")},
            "initializer_config": {key: value for key, value in config["coverage"].items()
                                   if key != "method"},
            "scenes": rows, "issues": issues}


def run(old_plan: Path, old_source: Path, new_source: Path,
        archive_root: Path, new_science_commit: str = NEW_COMMIT,
        source_catalog: Path | None = None) -> dict:
    new_science_commit = checked_commit(new_science_commit)
    plan = read(old_plan)
    if plan.get("experiment_id") != "S1-03" or plan.get("science_commit") != OLD_COMMIT:
        raise ValueError("old frozen S1-03 plan identity differs")
    source = audit_source_compatibility(old_source, new_source, new_science_commit)
    roles = {}
    issues = []
    for role, (run_id, expected) in PREPARES.items():
        directory = under(archive_root / "outputs/workbench/jobs" / run_id, archive_root)
        result = audit_prepare(directory, archive_root, old_source, role, run_id, expected)
        roles[role] = result
        issues.extend({"role": role, **issue} for issue in result["issues"])
    initial = [roles[role]["initializer_config"] for role in PREPARES]
    if not all(value == initial[0] for value in initial[1:]):
        issues.append({"reason": "prepare initialization/physical configs differ by role"})
    total = sum(roles[role]["scene_count"] for role in PREPARES)
    eligible = roles["train"]["qualified_count"]
    if total != 696 or eligible != 572:
        issues.append({"reason": f"denominator/qualification changed: {total}/696, {eligible}/572"})
    catalog_sha = None
    if source_catalog is not None:
        catalog = read(source_catalog)
        catalog_sha = sha(source_catalog)
        if (catalog.get("source_group") != "s1_03" or
                catalog.get("source_target") != "20418" or
                catalog.get("old_science_commit") != OLD_COMMIT or
                catalog.get("old_plan_sha256") != sha(old_plan) or
                set(catalog.get("roles", {})) != set(PREPARES)):
            issues.append({"reason": "independent source catalog identity differs"})
        else:
            for role, audited in roles.items():
                expected = catalog["roles"][role]
                actual_fields = {"source_dataset_id": audited["source_dataset_id"],
                                 "dataset_sha256": audited["dataset_sha256"],
                                 "source_run_id": audited["source_run_id"],
                                 "starts_sha256": audited["starts_sha256"],
                                 "scene_count": audited["scene_count"],
                                 "qualified_count": audited["qualified_count"]}
                if any(expected.get(key) != value for key, value in actual_fields.items()):
                    issues.append({"role": role, "reason": "independent source catalog role differs"})
    datasets = {role: {"source_dataset_id": row["source_dataset_id"],
                       "sha256": row["dataset_sha256"], "scene_count": row["scene_count"]}
                for role, row in roles.items()}
    starts = {role: {"source_run_id": row["prepare_run_id"],
                     "sha256": row["starts_sha256"],
                     "dataset_sha256": row["dataset_sha256"]}
              for role, row in roles.items()}
    return {"passed": not issues, "checked_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": "S1-03-R1", "source_group": "s1_03",
            "source_target": "20418", "purpose": "read_only_reuse_of_e174_common_starts",
            "old_science_commit": OLD_COMMIT, "new_science_commit": new_science_commit,
            "old_plan_sha256": sha(old_plan), "old_plan_path": str(old_plan),
            "source_catalog_sha256": catalog_sha,
            **source, "datasets": datasets, "starts": starts,
            "roles": roles, "total_scenes": total, "training_qualified": eligible,
            "issues": issues, "script_sha256": sha(__file__)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-plan", type=Path, required=True)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--new-source", type=Path, required=True)
    parser.add_argument("--new-science-commit", default=NEW_COMMIT,
                        help="complete 40-character commit recorded in new source_manifest.json")
    parser.add_argument("--source-catalog", type=Path, required=True,
                        help="frozen independent three-role source_reuse_catalog.json")
    parser.add_argument("--old-archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("reuse receipt already exists; preserve immutable audit")
    if args.output.resolve().is_relative_to(args.old_archive_root.resolve()):
        raise ValueError("reuse receipt must be written outside the old S1-03 archive")
    new_commit = checked_commit(args.new_science_commit)
    try:
        receipt = run(args.old_plan, args.old_source, args.new_source,
                      args.old_archive_root, new_commit, args.source_catalog)
    except Exception as error:
        receipt = {"passed": False, "checked_utc": datetime.now(timezone.utc).isoformat(),
                   "experiment_id": "S1-03-R1", "source_group": "s1_03",
                   "source_target": "20418", "old_science_commit": OLD_COMMIT,
                   "new_science_commit": new_commit,
                   "old_plan_sha256": sha(args.old_plan) if args.old_plan.is_file() else None,
                   "old_source_manifest_sha256": sha(args.old_source / "source_manifest.json")
                   if (args.old_source / "source_manifest.json").is_file() else None,
                   "new_source_manifest_sha256": sha(args.new_source / "source_manifest.json")
                   if (args.new_source / "source_manifest.json").is_file() else None,
                   "source_catalog_sha256": sha(args.source_catalog)
                   if args.source_catalog.is_file() else None,
                   "datasets": {}, "starts": {}, "roles": {}, "total_scenes": None,
                   "training_qualified": None, "issues": [{"reason": str(error)}],
                   "script_sha256": sha(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(args.output)
    if not receipt["passed"]:
        raise SystemExit(f"reuse audit failed: {len(receipt['issues'])} issue(s)")


if __name__ == "__main__":
    main()
