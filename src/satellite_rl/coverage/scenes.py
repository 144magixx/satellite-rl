"""Aircraft-level first-stage scenes and reproducible initial layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from satellite_rl.coverage.geometry import assign_numpy, satellite_rays_numpy, visible_numpy


BEAM_COUNT = 100


def _real_numeric(array):
    return np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)


def _validate_aircraft(positions, demands, aircraft_ids, context):
    """Reject malformed aircraft business without clipping or dropping users."""
    if (not isinstance(aircraft_ids, list) or
            any(not isinstance(identifier, str) or not identifier.strip()
                for identifier in aircraft_ids)):
        raise ValueError(f"{context}: aircraft IDs must be nonempty strings")
    canonical_ids = [identifier.strip().lower() for identifier in aircraft_ids]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError(f"{context}: duplicate aircraft ID")
    if (positions.ndim != 2 or positions.shape[1] != 2 or demands.ndim != 1 or
            len(positions) == 0 or len(positions) != len(demands) or
            len(positions) != len(aircraft_ids)):
        raise ValueError(f"{context}: aircraft positions, demands and IDs have incompatible lengths")
    if (not _real_numeric(positions) or not _real_numeric(demands) or
            not np.isfinite(positions).all() or not np.isfinite(demands).all() or
            np.any(demands < 0)):
        raise ValueError(f"{context}: nonfinite position or invalid nonnegative demand")
    if (np.any(np.abs(positions[:, 0]) > 90) or
            np.any(np.abs(positions[:, 1]) > 180)):
        raise ValueError(f"{context}: latitude/longitude outside geographic coordinates")
    visible = visible_numpy(positions, "europe_v1")
    if not np.all(visible):
        raise ValueError(f"{context}: {int(np.count_nonzero(~visible))} aircraft outside europe_v1 bounds or below 10 degree elevation")


def _validate_centers(centers, context):
    if centers.shape != (BEAM_COUNT, 2):
        raise ValueError(f"{context}: initializer must supply exactly 100 latitude/longitude centers")
    if not _real_numeric(centers):
        raise ValueError(f"{context}: beam centers must have real numeric coordinates")
    if np.isnan(centers).all():
        return  # Explicit failed-initialization sentinel; no artificial beams.
    if not np.isfinite(centers).all():
        raise ValueError(f"{context}: partial/nonfinite beam centers")
    if not np.all(visible_numpy(centers, "europe_v1")):
        raise ValueError(f"{context}: initial centers outside europe_v1 visibility")


@dataclass
class CoverageScene:
    scene_id: str
    positions: np.ndarray
    demands: np.ndarray
    centers: np.ndarray
    aircraft_ids: list[str]
    metadata: dict = field(default_factory=dict)
    source_hash: str = ""


@dataclass
class CoverageDataset:
    scenes: list[CoverageScene]
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.scenes)


def _business_hash(records):
    # Excludes old beam_id, processing_index and off_axis_degrees: these are
    # outputs of the previous initializer, not intrinsic scene identity.
    rows = [(str(r["aircraft_id"]), str(r.get("timestamp_utc", "")),
             float(r["lat"]), float(r["lon"]), float(r["demand_bps"])) for r in records]
    rows.sort()
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _rule_centers(positions, demands):
    """Same weighted farthest-positive-position medoid order as aviation._cover."""
    positive = demands > 0
    locations, demand = positions[positive], demands[positive]
    unique, inverse = np.unique(locations, axis=0, return_inverse=True)
    if len(unique) < BEAM_COUNT:
        return None, {"initialization_valid": False,
                      "initialization_error": "unique_positive_positions_below_100",
                      "unique_positive_positions": int(len(unique))}
    weights = np.zeros(len(unique), dtype=np.float64)
    np.add.at(weights, inverse, demand)
    rays = satellite_rays_numpy(unique)
    chosen = [int(np.argmax(weights))]
    nearest = np.full(len(unique), np.inf)
    for _ in range(BEAM_COUNT - 1):
        distance = np.sum((rays - rays[chosen[-1]]) ** 2, axis=1)
        nearest = np.minimum(nearest, distance)
        score = nearest * weights
        score[chosen] = -1
        chosen.append(int(np.argmax(score)))
    centers = unique[chosen]
    # Reproduce aviation._cover's dot/acos association, including its boundary
    # tolerance. This is the reference R initializer; the environment uses
    # stable chord distance to recompute all later assignments.
    dots = np.clip(satellite_rays_numpy(locations) @ rays[chosen].T, -1.0, 1.0)
    assigned = np.argmax(dots, axis=1)
    angles = np.rad2deg(np.arccos(dots[np.arange(len(locations)), assigned]))
    covered = angles <= 0.25 + 1e-10
    loads = np.bincount(assigned[covered], weights=demand[covered], minlength=BEAM_COUNT)
    valid = bool(np.all(loads > 0))
    return centers, {"initialization_valid": valid,
                     "initialization_error": None if valid else "empty_beam_after_rule_assignment",
                     "initial_empty_beams": int(np.count_nonzero(loads <= 0)),
                     "initializer": "aviation_weighted_farthest_medoids_v1"}


def _cluster_centers(positions, demands, iterations=12):
    """Weighted spherical k-means, deterministic medoid projection and repairs.

    Starts from the frozen R medoids, optimizes demand-weighted satellite-view
    chord distance, then projects each centroid to a distinct positive user
    position. Empty beams are repaired from users of the heaviest beam; failure
    stays in scene metadata. No request is added or discarded.
    """
    positive = demands > 0
    unique, inverse = np.unique(positions[positive], axis=0, return_inverse=True)
    if len(unique) < BEAM_COUNT:
        return None, {"initialization_valid": False,
                      "initialization_error": "unique_positive_positions_below_100",
                      "unique_positive_positions": int(len(unique))}
    weights = np.zeros(len(unique), dtype=np.float64)
    np.add.at(weights, inverse, demands[positive])
    rays = satellite_rays_numpy(unique)
    seed, _ = _rule_centers(positions, demands)
    centroid = satellite_rays_numpy(seed)
    for _ in range(iterations):
        assignment = np.argmin(np.sum((rays[:, None] - centroid[None]) ** 2, axis=-1), axis=1)
        sums = np.zeros((BEAM_COUNT, 3), dtype=np.float64)
        np.add.at(sums, assignment, rays * weights[:, None])
        norm = np.linalg.norm(sums, axis=1)
        nonempty = norm > 0
        centroid[nonempty] = sums[nonempty] / norm[nonempty, None]
    # Stable beam ID and unique candidate projection. Greedy ties select lower
    # unique-position index, and hence are independent of input row order.
    used = np.zeros(len(unique), dtype=bool)
    selected = []
    distance = np.sum((centroid[:, None] - rays[None]) ** 2, axis=-1)
    for beam in range(BEAM_COUNT):
        costs = distance[beam].copy()
        costs[used] = np.inf
        choice = int(np.argmin(costs))
        selected.append(choice)
        used[choice] = True
    centers = unique[selected].copy()
    repairs = 0
    for _ in range(BEAM_COUNT):
        association, loads = assign_numpy(positions[positive], demands[positive], centers)
        empty = np.flatnonzero(loads <= 0)
        if not len(empty):
            break
        candidates = np.argsort(-loads, kind="stable")
        changed = False
        for target in empty:
            for donor in candidates:
                members = np.flatnonzero(association == donor)
                if len(members) < 2:
                    continue
                # Move the empty center onto the farthest covered positive
                # position in the heaviest beam; donor keeps its own medoid.
                member_rays = satellite_rays_numpy(positions[positive][members])
                donor_ray = satellite_rays_numpy(centers[donor:donor + 1])[0]
                farthest = members[np.argmax(np.sum((member_rays - donor_ray) ** 2, axis=1))]
                candidate = positions[positive][farthest]
                if np.any(np.all(centers == candidate, axis=1)):
                    continue
                centers[target] = candidate
                repairs += 1
                changed = True
                break
        if not changed:
            break
    _, final_loads = assign_numpy(positions[positive], demands[positive], centers)
    empty_count = int(np.count_nonzero(final_loads <= 0))
    return centers, {"initialization_valid": empty_count == 0,
                     "initialization_error": None if empty_count == 0 else "cluster_empty_beams_after_repair",
                     "initial_empty_beams": empty_count, "cluster_iterations": iterations,
                     "cluster_repairs": repairs, "initializer": "weighted_satellite_ray_kmeans_medoid_v1"}


def _initial_centers(positions, demands, initialization, cluster_iterations=12):
    if initialization == "rule":
        return _rule_centers(positions, demands)
    if initialization == "cluster":
        return _cluster_centers(positions, demands, iterations=cluster_iterations)
    raise ValueError("initialization must be 'rule' or 'cluster'")


def _initializer_settings(initialization, config):
    if initialization not in ("rule", "cluster"):
        raise ValueError("initialization must be 'rule' or 'cluster'")
    options = {} if config is None else config.get("coverage", config)
    if not isinstance(options, dict):
        raise TypeError("coverage configuration must be a mapping")
    iterations = options.get("cluster_iterations", 12)
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or not 1 <= iterations <= 100:
        raise ValueError("cluster_iterations must be an integer in [1,100]")
    # R is scientifically independent of the unused C iteration setting.
    parameters = {"initialization": initialization}
    if initialization == "cluster":
        parameters["cluster_iterations"] = int(iterations)
    digest = hashlib.sha256(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return int(iterations), digest


def _read_source(path):
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"empty aircraft scene: {path}")
    try:
        ids = [r["aircraft_id"] for r in records]
        positions = np.array([[r["lat"], r["lon"]] for r in records], dtype=np.float64)
        demands = np.array([r["demand_bps"] for r in records], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: malformed aircraft business fields: {error}") from error
    _validate_aircraft(positions, demands, ids, str(path))
    return records, positions, demands, ids, sha


def _validate_prepared_arrays(lengths, positions, demands, ids, scene_meta,
                              scene_ids, source_hashes, saved_centers, path):
    context = str(path)
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer) or len(lengths) == 0:
        raise ValueError(f"{context}: prepared aircraft lengths must be a nonempty integer vector")
    count = len(lengths)
    if (positions.ndim != 3 or positions.shape[0] != count or positions.shape[2] != 2 or
            demands.shape != positions.shape[:2] or
            not _real_numeric(positions) or not _real_numeric(demands)):
        raise ValueError(f"{context}: prepared positions/demands shape or dtype mismatch")
    if np.any(lengths <= 0) or np.any(lengths > positions.shape[1]):
        raise ValueError(f"{context}: prepared aircraft length exceeds padded storage or is empty")
    if (not isinstance(ids, list) or len(ids) != count or
            not isinstance(scene_meta, list) or len(scene_meta) != count or
            any(not isinstance(item, dict) for item in scene_meta) or
            not isinstance(scene_ids, list) or len(scene_ids) != count or
            not isinstance(source_hashes, list) or len(source_hashes) != count or
            any(not isinstance(value, str) or not value for value in scene_ids + source_hashes)):
        raise ValueError(f"{context}: prepared scene IDs, aircraft IDs, hashes or metadata mismatch")
    if saved_centers is not None and (
            saved_centers.shape != (count, BEAM_COUNT, 2) or
            not _real_numeric(saved_centers)):
        raise ValueError(f"{context}: prepared beam centers shape or dtype mismatch")
    for i, length in enumerate(lengths):
        length = int(length)
        if (np.any(positions[i, length:] != 0) or
                np.any(demands[i, length:] != 0)):
            raise ValueError(f"{context}: scene {i} contains nonzero/invalid padded aircraft data")
        _validate_aircraft(positions[i, :length], demands[i, :length], ids[i], f"{context}: scene {i}")
        if saved_centers is not None:
            _validate_centers(saved_centers[i], f"{context}: scene {i}")
            if np.isnan(saved_centers[i]).all() and scene_meta[i].get("initialization_valid") is not False:
                raise ValueError(f"{context}: scene {i} has failed centers without a failed initialization record")


def load_dataset(path, initialization="rule", config=None):
    """Load original *.jsonl.gz scenes or this module's portable NPZ archive.

    A path may be one source file, a prepared NPZ, or a directory with a
    record.json and source_observations_*.jsonl.gz files. Existing stage-two
    scene_*.npz files are deliberately rejected: they lack aircraft records.
    """
    cluster_iterations, initializer_hash = _initializer_settings(initialization, config)
    path = Path(path)
    if path.suffix == ".npz":
        prepared_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        prepared_record = path.parent / "record.json"
        if prepared_record.is_file():
            record = json.loads(prepared_record.read_text(encoding="utf-8"))
            if (record.get("prepared_npz") == path.name and
                    record.get("prepared_npz_sha256") != prepared_sha):
                raise ValueError("prepared coverage NPZ checksum differs from record.json")
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["format_version"]) != "coverage_aircraft_v1":
                raise ValueError("NPZ has no aircraft-level coverage data")
            metadata = json.loads(str(archive["metadata_json"]))
            ids = json.loads(str(archive["aircraft_ids_json"]))
            scene_meta = json.loads(str(archive["scene_metadata_json"]))
            lengths = archive["lengths"]
            positions, demands = archive["positions"], archive["demands"]
            saved_centers = archive["centers"] if "centers" in archive else None
            scene_ids = archive["scene_ids"].tolist()
            source_hashes = archive["source_hashes"].tolist()
            if not isinstance(metadata, dict):
                raise ValueError(f"{path}: prepared metadata must be a mapping")
            _validate_prepared_arrays(lengths, positions, demands, ids, scene_meta,
                                      scene_ids, source_hashes, saved_centers, path)
            scenes = []
            for i, length in enumerate(lengths):
                length = int(length)
                p, d = positions[i, :length].copy(), demands[i, :length].copy()
                if (saved_centers is not None and
                        metadata.get("initializer_sha256") == initializer_hash):
                    centers, init_meta = saved_centers[i].copy(), {}
                    original_meta = scene_meta[i]
                else:
                    centers, init_meta = _initial_centers(p, d, initialization, cluster_iterations)
                    initializer_keys = {"initialization_valid", "initialization_error",
                                        "unique_positive_positions", "initial_empty_beams",
                                        "initializer", "cluster_iterations", "cluster_repairs"}
                    original_meta = {key: value for key, value in scene_meta[i].items()
                                     if key not in initializer_keys}
                if centers is not None:
                    _validate_centers(centers, f"{path}: scene {i}")
                scenes.append(CoverageScene(scene_ids[i], p, d,
                                            np.full((BEAM_COUNT, 2), np.nan) if centers is None else centers,
                                            ids[i], {**original_meta, **init_meta}, source_hashes[i]))
            return CoverageDataset(scenes, {**metadata, "initialization": initialization,
                                            "cluster_iterations": cluster_iterations if initialization == "cluster" else None,
                                            "initializer_sha256": initializer_hash,
                                            "prepared_npz_sha256": prepared_sha})
    files = sorted(path.glob("source_observations_*.jsonl.gz")) if path.is_dir() else [path]
    if not files or any(not file.name.endswith(".jsonl.gz") for file in files):
        raise ValueError("expected aircraft source_observations_*.jsonl.gz, not beam summaries")
    record_path = path / "record.json" if path.is_dir() else path.parent / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    source = record.get("source", {})
    scenes = []
    snapshots = source.get("snapshots", [])
    for file in files:
        rows, positions, demands, ids, source_sha = _read_source(file)
        centers, init_meta = _initial_centers(positions, demands, initialization, cluster_iterations)
        if centers is not None:
            _validate_centers(centers, str(file))
        identity = _business_hash(rows)
        try:
            source_index = int(file.name.removeprefix("source_observations_").split(".")[0])
        except ValueError:
            source_index = -1
        snapshot = snapshots[source_index] if 0 <= source_index < len(snapshots) else {}
        scenes.append(CoverageScene(
            identity, positions, demands,
            np.full((BEAM_COUNT, 2), np.nan) if centers is None else centers,
            ids, {"source_file": file.name,
                  "selected_time_utc": snapshot.get("selected_time_utc"),
                  "source_dates": snapshot.get("source_dates", sorted({r.get("source_date", "") for r in rows})),
                  "business_sha256": identity, **init_meta}, source_sha))
    return CoverageDataset(scenes, {
        "format_version": "coverage_aircraft_v1", "initialization": initialization,
        "cluster_iterations": cluster_iterations if initialization == "cluster" else None,
        "initializer_sha256": initializer_hash,
        "partition_role": source.get("partition_role"),
        "source_dates": source.get("source_dates", []),
        "source_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest() if record_path.is_file() else None,
        "source_files": [{"name": f.name, "sha256": s.source_hash} for f, s in zip(files, scenes)]})


def save_dataset(dataset, destination):
    """Save aircraft arrays, original hashes and provenance without pickle."""
    if len(dataset) == 0:
        raise ValueError("cannot save an empty aircraft coverage dataset")
    for index, scene in enumerate(dataset.scenes):
        _validate_aircraft(np.asarray(scene.positions), np.asarray(scene.demands),
                           scene.aircraft_ids, f"prepared output scene {index}")
        _validate_centers(np.asarray(scene.centers), f"prepared output scene {index}")
        if (np.isnan(scene.centers).all() and
                scene.metadata.get("initialization_valid") is not False):
            raise ValueError(f"prepared output scene {index}: failed centers need a failure record")
    destination = Path(destination)
    if destination.suffix != ".npz":
        raise ValueError("prepared coverage destination must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    record_path = destination.parent / "record.json"
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing provenance record: {record_path}")
    count = len(dataset)
    lengths = np.array([len(s.demands) for s in dataset.scenes], dtype=np.int64)
    maximum = int(lengths.max()) if count else 0
    positions = np.zeros((count, maximum, 2), dtype=np.float64)
    demands = np.zeros((count, maximum), dtype=np.float64)
    for i, scene in enumerate(dataset.scenes):
        positions[i, :lengths[i]] = scene.positions
        demands[i, :lengths[i]] = scene.demands
    np.savez_compressed(destination, format_version="coverage_aircraft_v1", lengths=lengths,
                        positions=positions, demands=demands,
                        centers=np.array([s.centers for s in dataset.scenes], dtype=np.float64),
                        scene_ids=np.array([s.scene_id for s in dataset.scenes]),
                        source_hashes=np.array([s.source_hash for s in dataset.scenes]),
                        aircraft_ids_json=json.dumps([s.aircraft_ids for s in dataset.scenes]),
                        scene_metadata_json=json.dumps([s.metadata for s in dataset.scenes]),
                        metadata_json=json.dumps(dataset.metadata))
    record = {"format_version": "coverage_aircraft_v1", "count": count,
              "prepared_npz": destination.name,
              "prepared_npz_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "initialization": dataset.metadata.get("initialization"),
              "cluster_iterations": dataset.metadata.get("cluster_iterations"),
              "initializer_sha256": dataset.metadata.get("initializer_sha256"),
              "partition_role": dataset.metadata.get("partition_role"),
              "source_dates": dataset.metadata.get("source_dates", []),
              "source_record_sha256": dataset.metadata.get("source_record_sha256"),
              "scenes": [{"scene_id": s.scene_id, "source_hash": s.source_hash,
                          "initialization_valid": s.metadata.get("initialization_valid")}
                         for s in dataset.scenes]}
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
