"""一次读取日包，同时抽取静态和动态网格。"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import gzip
import json
import tarfile

import numpy as np
from satellite_rl.coverage.geometry import visible_numpy

OUT = Path("D:/paper/outputs/experiments/s1_fix_20261017")
HISTORY = OUT / "data" / "history"  # 旧轮次 first_stage_ai_search_20261013 的三份历史对照 CSV 已复制到此
DATES = ("2024-01-08", "2024-01-09", "2024-01-12", "2024-01-15", "2024-01-18")
HIST12 = [0, 6, 12, 18, 24, 36, 42, 48, 54, 60, 72, 78]
HIST12_COUNTS = [293, 216, 204, 178, 175, 185, 235, 288, 318, 423, 946, 1254]
HIST20 = np.floor((np.arange(20) + .5) * 288 / 20).astype(int)


@dataclass
class Frame:
    date: str
    time: float
    index: int
    icao: np.ndarray
    positions: np.ndarray
    gs_kt: np.ndarray
    track_deg: np.ndarray


def raw_path(date, outputs_root="D:/paper/outputs"):
    return Path(outputs_root) / "experiments" / "adsb_raw" / f"v{date.replace('-', '.')}-planes-readsb-prod-0.tar"


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def extract_day(date, raw, data_dir):
    midnight = datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp()
    grids = {"static": midnight + 120 + np.arange(288) * 300,
             "dynamic": midnight + 21600 + np.arange(961) * 60}
    times = np.unique(np.concatenate(list(grids.values())))
    frames = [[] for _ in times]
    with tarfile.open(raw, "r|") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            payload = json.loads(gzip.decompress(archive.extractfile(member).read()))
            trace = payload["trace"]
            stamps = np.array([number(row[0]) for row in trace]) + number(payload["timestamp"])
            valid_stamps = np.isfinite(stamps)
            rows = np.flatnonzero(valid_stamps)
            stamps = stamps[rows]
            indices = np.searchsorted(stamps, times, side="right") - 1
            selected = np.flatnonzero((indices >= 0) & (times - stamps[np.maximum(indices, 0)] <= 60)) if len(stamps) else []
            records, at = [], []
            for i in selected:
                row = trace[rows[indices[i]]]
                if len(row) < 7:
                    continue
                lat, lon, alt = number(row[1]), number(row[2]), number(row[3]) * .3048
                if not np.isfinite([lat, lon, alt, number(row[6])]).all():
                    continue
                if alt <= 0 or alt > 25000 or int(row[6]) & 1:
                    continue
                records.append([lat, lon, number(row[4]), number(row[5])])
                at.append(i)
            if records:
                records = np.asarray(records)
                visible = visible_numpy(records[:, :2], "europe_v1")
                icao = int(payload["icao"].lstrip("~"), 16)
                for i, record in zip(np.asarray(at)[visible], records[visible]):
                    frames[i].append((icao, *record))
    arrays = {}
    for grid, stamps in grids.items():
        parts = [np.asarray(frames[i], dtype=float).reshape(-1, 5) for i in np.searchsorted(times, stamps)]
        parts = [part[np.argsort(part[:, 0])] for part in parts]
        values = np.concatenate(parts)
        arrays[f"{grid}_times"] = stamps
        arrays[f"{grid}_offsets"] = np.r_[0, np.cumsum([len(part) for part in parts])]
        for j, field in enumerate(("icao", "lat", "lon", "gs_kt", "track_deg")):
            arrays[f"{grid}_{field}"] = values[:, j].astype(np.uint32 if j == 0 else np.float64)
    destination = Path(data_dir) / f"frames_{date}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    result = {"date": date, "static_frames": 288, "dynamic_frames": 961,
              "static_aircraft_records": int(arrays["static_offsets"][-1]),
              "dynamic_aircraft_records": int(arrays["dynamic_offsets"][-1]),
              "bytes": destination.stat().st_size}
    print(json.dumps(result), flush=True)
    return result


def load_day(data_dir, date, grid="static"):
    with np.load(Path(data_dir) / f"frames_{date}.npz") as z:
        offsets, times = z[f"{grid}_offsets"], z[f"{grid}_times"]
        ids, lat, lon, gs, track = [z[f"{grid}_{field}"] for field in ("icao", "lat", "lon", "gs_kt", "track_deg")]
    positions = np.column_stack((lat, lon))
    return [Frame(date, float(t), i, ids[a:b], positions[a:b], gs[a:b], track[a:b])
            for i, (t, a, b) in enumerate(zip(times, offsets[:-1], offsets[1:]))]


def stratified(day):
    counts = np.array([len(frame.icao) for frame in day])
    layers = [counts < 300, (counts >= 600) & (counts <= 1000),
              (counts >= 1200) & (counts <= 1600), counts > 1600]
    return [day[i] for mask in layers for i in np.flatnonzero(mask)[
        np.floor((np.arange(6) + .5) * np.count_nonzero(mask) / 6).astype(int)]]


def static_sets(data_dir):
    jan12 = load_day(data_dir, "2024-01-12")
    jan08 = load_day(data_dir, "2024-01-08")
    return {"DEV24": stratified(jan12),
            "CONF48": stratified(load_day(data_dir, "2024-01-15")) + stratified(load_day(data_dir, "2024-01-18")),
            "HIST12": [jan08[i] for i in HIST12], "HIST20": [jan12[i] for i in HIST20]}


def sequence_sets(data_dir, role):
    dates = {"train": DATES[:2], "dev": (DATES[2],), "conf": DATES[3:]}[role]
    return [day[i * 30:i * 30 + 31] for date in dates
            for day in [load_day(data_dir, date, "dynamic")]
            for i in range(0, 32, 1 if role == "train" else 2)]


def verify_history(data_dir, history=HISTORY):
    rows = []
    day = load_day(data_dir, "2024-01-08")
    for i, count in zip(HIST12, HIST12_COUNTS):
        rows.append(dict(scene_set="HIST12", index=i, time=day[i].time, expected=count, actual=len(day[i].icao)))
    with open(Path(history) / "round8/analysis/mechanism_01/mechanism_rows.csv", encoding="utf-8-sig") as stream:
        expected = {int(row["case_index"]): row for row in csv.DictReader(stream)}
    day = load_day(data_dir, "2024-01-12")
    for k, i in enumerate(HIST20):
        rows.append(dict(scene_set="HIST20", index=int(i), time=day[i].time,
                         expected=int(expected[k]["aircraft_count"]), actual=len(day[i].icao)))
    for row in rows:
        row["match"] = row["actual"] == row["expected"]
    return rows
