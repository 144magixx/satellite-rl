"""Import and archive actual scenarios without replacing order, identities or units."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import shutil

import numpy as np

from .storage import digest, inside, new_id, now


def _environment(profile="china_legacy"):
    from .config import default_config
    from .environment import WorkbenchEnv
    config = default_config()
    config["environment"]["profile"] = profile
    return WorkbenchEnv(config)


def _store_scenes(store, name, arrays, source, profile="china_legacy"):
    from satellite_rl.envs.regions import profile_metadata
    metadata = profile_metadata(profile)
    identifier = new_id("scenes")
    directory = store.record_dir("scenes", identifier)
    directory.mkdir()
    references = []
    environment = _environment(profile)
    for index, data in enumerate(arrays):
        environment.reset(scene=data)
        path = directory / f"scene_{index:05d}.npz"
        np.savez_compressed(path, **environment.scene())
        references.append({"path": str(path), "sha256": digest(path), "scene_id": path.stem})
    if not references:
        raise ValueError("场景文件为空")
    return store.save_record("scenes", {"id": identifier, "name": name or identifier,
        "path": str(directory), "count": len(references), "created_at": now(),
        "source": source, "files": references, "profile": metadata["id"],
        "environment_version": metadata["environment_version"], "environment_profile": metadata,
        "units": {"rate": "bit/s", "beam_location": "degrees"}})


def generate_scenes(store, name, seed, count, profile="china_legacy"):
    count, seed = int(count), int(seed)
    if not 1 <= count <= 1000 or not 0 <= seed <= 2 ** 32 - count:
        raise ValueError("场景数量须为1–1000，种子须处于可用范围")
    environment = _environment(profile)
    arrays = []
    for index in range(count):
        environment.reset(seed=seed + index)
        arrays.append(environment.scene())
    return _store_scenes(store, name, arrays, {"type": "generated", "seed": seed, "count": count}, profile)


def csv_scene(path, profile="china_legacy"):
    """CSV requires explicit units and allocation order; no demand-based reordering."""
    from satellite_rl.envs.regions import geo_distances_km
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
    distances = geo_distances_km(locations, profile)
    return {"beam_location": locations, "rate": demand, "distances": distances,
            "beam_G_Tx_max": np.full((100, 1), 50.0), "UE_G_Rx_max": np.full((100, 1), 40.0),
            "noise_temperature": np.full((100, 1), 290.0), "original_indices": np.array(identifiers)}


def _sidecar_hashes(record, files, profile):
    """Bind a sidecar to the selected files by unique basename, even after a move."""
    declared = record.get("files")
    if declared is None:
        if profile == "europe_v1":
            raise ValueError("欧洲场景侧车缺少数组文件与SHA-256绑定")
        return None
    if not isinstance(declared, list) or not declared or record.get("count") != len(declared):
        raise ValueError("场景侧车数组文件数量无效")
    expected = {}
    has_hash = []
    for item in declared:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("场景侧车数组文件路径无效")
        filename = Path(item["path"]).name
        if not filename or filename in expected:
            raise ValueError("场景侧车数组文件名重复或无效")
        value = item.get("sha256")
        has_hash.append(value is not None)
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value)):
            raise ValueError("场景侧车SHA-256无效")
        expected[filename] = value.lower() if value is not None else None
    if set(expected) != {file.name for file in files}:
        raise ValueError("场景侧车数组文件与所选实际文件不一致")
    if not all(has_hash) and (profile == "europe_v1" or any(has_hash)):
        raise ValueError("场景侧车数组文件缺少完整SHA-256绑定")
    return expected if all(has_hash) else None


def import_scenes(store, path, name=None, profile="china_legacy"):
    path = Path(path).expanduser().resolve()
    files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    if not files or len(files) > 1000:
        raise ValueError("请选择含1–1000个npz场景的目录或一个npz/CSV文件")
    sidecar = path / "record.json" if path.is_dir() else path.parent / "record.json"
    record = None
    expected_hashes = None
    if sidecar.is_file():
        record = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        assert_scene_profile(record, profile)
        expected_hashes = _sidecar_hashes(record, files, profile)
    arrays, refs = [], []
    for file in files:
        if not file.is_file() or file.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("场景文件不存在或超过100MB")
        actual_hash = digest(file)
        if expected_hashes is not None and actual_hash != expected_hashes[file.name]:
            raise ValueError("场景侧车数组文件SHA-256校验失败：" + file.name)
        if file.suffix.lower() == ".npz":
            with np.load(file, allow_pickle=False) as data:
                arrays.append({key: data[key].copy() for key in data.files})
        elif file.suffix.lower() == ".csv":
            arrays.append(csv_scene(file, profile))
        else:
            raise ValueError("场景仅支持npz或有明确单位的CSV")
        refs.append({"path": str(file), "sha256": actual_hash})
    # Preserve an archived dataset's region identity, including when users import
    # its directory instead of selecting its existing workbench record.
    source = {"type": "imported", "files": refs}
    if record is not None:
        # Reimport must not erase an aircraft dataset's split or observation dates.
        if record.get("source"):
            source = dict(record["source"])
            source["reimport"] = {"record_id": record.get("id"), "files": refs,
                                  "previous": source.get("reimport")}
    return _store_scenes(store, name or path.stem, arrays, source, profile)


def assert_scene_profile(record, profile):
    """Missing metadata denotes the historical China environment, never Europe."""
    from satellite_rl.envs.regions import profile_metadata
    expected = profile_metadata(profile)
    actual = record.get("profile", "china_legacy")
    if actual != expected["id"]:
        raise ValueError("场景环境profile与任务不匹配：" + str(actual) + " / " + expected["id"])
    if actual != "china_legacy" and ("environment_version" not in record or "environment_profile" not in record):
        raise ValueError("欧洲场景记录缺少完整环境版本与参数")
    if record.get("environment_version", expected["environment_version"]) != expected["environment_version"]:
        raise ValueError("场景环境版本与当前任务不匹配")
    if record.get("environment_profile", expected) != expected:
        raise ValueError("场景环境参数与当前profile不匹配")
    return expected


def scene_set_files(record):
    """Use recorded order and verify source hashes before freezing a dataset."""
    directory = Path(record["path"]).resolve()
    references = record.get("files")
    if references is None:
        references = [{"path": str(p)} for p in sorted(directory.glob("*.npz"))]
    if not references or len(references) != record["count"]:
        raise ValueError("场景文件数量与记录不一致或为空")
    result = []
    for reference in references:
        path = inside(reference["path"], directory)
        if not path.is_file() or path.suffix.lower() != ".npz":
            raise ValueError("场景记录缺少NPZ文件")
        actual = digest(path)
        if reference.get("sha256", actual) != actual:
            raise ValueError("场景文件校验失败：" + path.name)
        result.append((path, actual))
    return result


def freeze_scene_set(record, target, profile):
    """Copy validated actual arrays before publishing a runnable task."""
    from satellite_rl.training.run_monitor import atomic_json
    from .environment import validate_scene
    metadata = assert_scene_profile(record, profile)
    files = scene_set_files(record)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    references = []
    for index, (source, expected) in enumerate(files):
        with np.load(source, allow_pickle=False) as data:
            validate_scene({key: data[key] for key in data.files}, profile=profile)
        destination = target / f"scene_{index:07d}.npz"
        shutil.copyfile(source, destination)
        if digest(destination) != expected:
            raise ValueError("冻结场景时文件发生变化")
        references.append({"path": str(destination), "sha256": expected,
                           "source_path": str(source), "scene_id": source.stem})
    frozen = {"id": record["id"], "name": record.get("name", record["id"]),
              "path": str(target), "count": len(references), "files": references,
              "profile": metadata["id"], "environment_version": metadata["environment_version"],
              "environment_profile": metadata, "source": record.get("source"),
              "frozen_at": now()}
    atomic_json(target / "record.json", frozen)
    return frozen
