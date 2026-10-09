"""Aircraft-level input for variable-beam layouts; legacy centers are ignored."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .scenes import CoverageDataset, CoverageScene, _business_hash, _read_source, _validate_aircraft


FORMAT = "variable_aircraft_v1"


def _validate_variable_aircraft(positions, demands, ids, context):
    if len(demands) == 0:
        if positions.shape != (0, 2) or demands.shape != (0,) or ids != []:
            raise ValueError(f"{context}: invalid empty-aircraft scene")
        return
    _validate_aircraft(positions, demands, ids, context)


def _loaded_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        version = str(archive["format_version"])
        if version not in (FORMAT, "coverage_aircraft_v1"):
            raise ValueError("input NPZ lacks aircraft-level records")
        metadata = json.loads(str(archive["metadata_json"]))
        ids = json.loads(str(archive["aircraft_ids_json"]))
        scene_meta = json.loads(str(archive["scene_metadata_json"]))
        lengths = archive["lengths"].copy()
        positions = archive["positions"].copy()
        demands = archive["demands"].copy()
        scene_ids = archive["scene_ids"].tolist()
        hashes = archive["source_hashes"].tolist()
    if (lengths.ndim != 1 or positions.ndim != 3 or positions.shape[2] != 2 or
            demands.ndim != 2 or len(lengths) != positions.shape[0] or
            demands.shape != positions.shape[:2] or
            any(len(value) != len(lengths) for value in (ids, scene_meta, scene_ids, hashes))):
        raise ValueError("aircraft NPZ array shapes differ")
    scenes = []
    for index, length in enumerate(lengths):
        if type(length.item()) is not int or not 0 <= int(length) <= demands.shape[1]:
            raise ValueError("invalid aircraft scene length")
        count = int(length)
        p, d = positions[index, :count], demands[index, :count]
        _validate_variable_aircraft(p, d, ids[index], f"{path}: scene {index}")
        meta = {key: value for key, value in scene_meta[index].items()
                if key not in ("initialization_valid", "initialization_error", "initializer",
                               "initial_active_count", "initial_empty_beams", "parking_coordinates")}
        scenes.append(CoverageScene(str(scene_ids[index]), p, d,
                                    np.empty((0, 2), dtype=np.float64),
                                    list(ids[index]), meta, str(hashes[index])))
    if version == "coverage_aircraft_v1":
        previous_initializer = {key: metadata.pop(key) for key in
                                ("initialization", "cluster_iterations", "candidate_capacity",
                                 "initializer_sha256", "prepared_npz_sha256") if key in metadata}
        metadata["previous_initializer_provenance"] = previous_initializer
        metadata["previous_centers_discarded"] = True
    return CoverageDataset(scenes, {**metadata, "format_version": FORMAT,
                                    "source_format_version": version})


def load_variable_dataset(path):
    """Read a prepared aircraft NPZ or original gzipped aircraft observations."""
    path = Path(path)
    if path.is_file() and path.suffix == ".npz":
        dataset = _loaded_arrays(path)
        record = path.parent / "record.json"
        if record.is_file():
            info = json.loads(record.read_text(encoding="utf-8"))
            expected = info.get("prepared_npz_sha256")
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("prepared NPZ SHA differs from record.json")
        return dataset
    files = sorted(path.glob("source_observations_*.jsonl.gz")) if path.is_dir() else [path]
    if not files or any(not f.is_file() or not f.name.endswith(".jsonl.gz") for f in files):
        raise ValueError("expected aircraft observations or prepared aircraft NPZ")
    record_path = path / "record.json" if path.is_dir() else path.parent / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    source = record.get("source", {})
    snapshots = source.get("snapshots", [])
    scenes = []
    for file in files:
        rows, p, d, ids, sha = _read_source(file)
        try:
            index = int(file.name.removeprefix("source_observations_").split(".")[0])
        except ValueError:
            index = -1
        snapshot = snapshots[index] if 0 <= index < len(snapshots) else {}
        dates = snapshot.get("source_dates") or sorted({r.get("source_date") for r in rows
                                                         if r.get("source_date")})
        scenes.append(CoverageScene(_business_hash(rows), p, d,
                                    np.empty((0, 2), dtype=np.float64), ids,
                                    {"source_file": file.name,
                                     "selected_time_utc": snapshot.get("selected_time_utc"),
                                     "source_dates": dates}, sha))
    return CoverageDataset(scenes, {
        "format_version": FORMAT, "partition_role": source.get("partition_role"),
        "source_dates": source.get("source_dates", []),
        "source_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest()
        if record_path.is_file() else None,
        "source_files": [{"name": file.name, "sha256": scene.source_hash}
                         for file, scene in zip(files, scenes)],
    })


def save_variable_dataset(dataset, path):
    path = Path(path)
    if path.suffix != ".npz" or not dataset.scenes:
        raise ValueError("variable dataset must be nonempty NPZ")
    if (path.parent / "record.json").exists() or path.exists():
        raise FileExistsError("refusing to overwrite aircraft provenance")
    for index, scene in enumerate(dataset.scenes):
        _validate_variable_aircraft(np.asarray(scene.positions), np.asarray(scene.demands),
                                    scene.aircraft_ids, f"variable scene {index}")
    lengths = np.asarray([len(s.demands) for s in dataset.scenes], dtype=np.int64)
    positions = np.zeros((len(lengths), int(lengths.max()), 2), dtype=np.float64)
    demands = np.zeros((len(lengths), int(lengths.max())), dtype=np.float64)
    for index, scene in enumerate(dataset.scenes):
        positions[index, :lengths[index]] = scene.positions
        demands[index, :lengths[index]] = scene.demands
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, format_version=FORMAT, lengths=lengths,
                        positions=positions, demands=demands,
                        scene_ids=np.asarray([s.scene_id for s in dataset.scenes]),
                        source_hashes=np.asarray([s.source_hash for s in dataset.scenes]),
                        aircraft_ids_json=json.dumps([s.aircraft_ids for s in dataset.scenes]),
                        scene_metadata_json=json.dumps([s.metadata for s in dataset.scenes]),
                        metadata_json=json.dumps(dataset.metadata))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"format_version": FORMAT, "count": len(dataset.scenes),
              "prepared_npz": path.name, "prepared_npz_sha256": sha,
              "source": dataset.metadata}
    (path.parent / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record
