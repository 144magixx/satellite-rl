"""Import and archive actual scenarios without replacing order, identities or units."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .storage import digest, new_id, now


def _environment():
    from .config import default_config
    from .environment import WorkbenchEnv
    return WorkbenchEnv(default_config())


def _store_scenes(store, name, arrays, source):
    identifier = new_id("scenes")
    directory = store.record_dir("scenes", identifier)
    directory.mkdir()
    references = []
    environment = _environment()
    for index, data in enumerate(arrays):
        environment.reset(scene=data)
        path = directory / f"scene_{index:05d}.npz"
        np.savez_compressed(path, **environment.scene())
        references.append({"path": str(path), "sha256": digest(path), "scene_id": path.stem})
    if not references:
        raise ValueError("场景文件为空")
    return store.save_record("scenes", {"id": identifier, "name": name or identifier,
        "path": str(directory), "count": len(references), "created_at": now(),
        "source": source, "files": references, "units": {"rate": "bit/s", "beam_location": "degrees"}})


def generate_scenes(store, name, seed, count):
    count, seed = int(count), int(seed)
    if not 1 <= count <= 1000 or not 0 <= seed <= 2 ** 32 - count:
        raise ValueError("场景数量须为1–1000，种子须处于可用范围")
    environment = _environment()
    arrays = []
    for index in range(count):
        environment.reset(seed=seed + index)
        arrays.append(environment.scene())
    return _store_scenes(store, name, arrays, {"type": "generated", "seed": seed, "count": count})


def csv_scene(path):
    """CSV requires explicit units and allocation order; no demand-based reordering."""
    from satellite_rl.envs.satellite import BeamInfo, R, r, satellite_lon
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"beam_id", "lat", "lon", "demand_mbps"}
    if len(rows) != 100 or not rows or not required.issubset(rows[0]):
        raise ValueError("CSV须恰有100行，包含beam_id、lat、lon、demand_mbps；行顺序就是处理顺序")
    identifiers = [int(row["beam_id"]) for row in rows]
    if len(set(identifiers)) != 100 or set(identifiers) != set(range(100)):
        raise ValueError("beam_id必须是0–99的唯一编号；保留行顺序")
    locations = np.array([[float(row["lat"]), float(row["lon"])] for row in rows])
    demand = np.array([float(row["demand_mbps"]) * 1e6 for row in rows]).reshape(100, 1)
    if any(row.get("group", "").strip() and int(row["group"]) != index % 8 for index, row in enumerate(rows)):
        raise ValueError("当前环境按处理序号%8分组；CSV group不匹配时不会静默改写")
    lat, lon = np.deg2rad(locations[:, 0]), np.deg2rad(locations[:, 1])
    sat_lon = np.deg2rad(satellite_lon)
    distances = np.sqrt((R * np.cos(sat_lon) - r * np.cos(lat) * np.cos(lon)) ** 2
                        + (R * np.sin(sat_lon) - r * np.cos(lat) * np.sin(lon)) ** 2
                        + (r * np.sin(lat)) ** 2)
    return {"beam_location": locations, "rate": demand, "distances": distances,
            "beam_G_Tx_max": np.full((100, 1), 50.0), "UE_G_Rx_max": np.full((100, 1), 40.0),
            "noise_temperature": np.full((100, 1), 290.0), "original_indices": np.array(identifiers)}


def import_scenes(store, path, name=None):
    path = Path(path).expanduser().resolve()
    files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    if not files or len(files) > 1000:
        raise ValueError("请选择含1–1000个npz场景的目录或一个npz/CSV文件")
    arrays, refs = [], []
    for file in files:
        if not file.is_file() or file.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("场景文件不存在或超过100MB")
        if file.suffix.lower() == ".npz":
            with np.load(file, allow_pickle=False) as data:
                arrays.append({key: data[key].copy() for key in data.files})
        elif file.suffix.lower() == ".csv":
            arrays.append(csv_scene(file))
        else:
            raise ValueError("场景仅支持npz或有明确单位的CSV")
        refs.append({"path": str(file), "sha256": digest(file)})
    return _store_scenes(store, name or path.stem, arrays, {"type": "imported", "files": refs})
