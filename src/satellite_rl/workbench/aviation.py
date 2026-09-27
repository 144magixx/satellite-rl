"""Read-only ADS-B positions with explicitly synthetic aircraft demand.

The adapter is a static snapshot/coverage approximation, not a flight traffic
measurement or a first-stage DRL coverage planner. No source downloads occur.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


CSV_UNITS = {"timestamp": "utc_iso8601", "lat": "degrees", "lon": "degrees", "altitude_m": "m"}
SCHEMA_URL = "https://github.com/wiedehopf/readsb/blob/dev/README-json.md"
HISTORY_URL = "https://www.adsb.lol/docs/open-data/historical/"
MAX_BYTES = 100 * 1024 * 1024
MAX_OBSERVATIONS = 1_000_000


class AviationSceneError(ValueError):
    """A rejected snapshot keeps its diagnostic counts available to the caller."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def _utc(value):
    if not isinstance(value, str):
        raise ValueError("时刻必须是带UTC时区的ISO8601字符串")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError("时刻必须明确使用UTC（Z或+00:00）")
    return dt.timestamp()


def _iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("数值必须有限")
    return number


def _model(model):
    if not isinstance(model, dict):
        raise ValueError("必须明确提供合成通信需求参数")
    required = {"persons_per_aircraft", "satellite_access_fraction", "active_fraction",
                "rate_mbps_per_active_person", "parameter_source"}
    allowed = required | {"aircraft_base_mbps", "lognormal_sigma", "sampling", "window_seconds"}
    if not required <= set(model) or set(model) - allowed:
        raise ValueError("合成需求参数缺少必填字段或含未知字段")
    out = dict(model)
    out.setdefault("aircraft_base_mbps", 0.0)
    out.setdefault("lognormal_sigma", 0.0)
    out.setdefault("sampling", "expected")
    out.setdefault("window_seconds", 60.0)
    for key in allowed - {"sampling", "parameter_source"}:
        out[key] = _finite(out[key])
    if out["persons_per_aircraft"] < 0 or out["persons_per_aircraft"] != int(out["persons_per_aircraft"]):
        raise ValueError("每机人数必须是非负整数")
    for key in ("satellite_access_fraction", "active_fraction"):
        if not 0 <= out[key] <= 1:
            raise ValueError("接入比例与活跃比例须在0至1之间")
    if any(out[key] < 0 for key in ("rate_mbps_per_active_person", "aircraft_base_mbps", "lognormal_sigma")):
        raise ValueError("业务速率与扰动幅度不能为负")
    if out["lognormal_sigma"] > 3 or out["window_seconds"] <= 0:
        raise ValueError("扰动幅度须不超过3，统计窗口必须为正")
    if out["sampling"] not in ("expected", "binomial") or not str(out["parameter_source"]).strip():
        raise ValueError("需求抽样须为expected/binomial，参数来源说明不能为空")
    if out["persons_per_aircraft"] > 10000:
        raise ValueError("每机人数超过适配器工程上限10000")
    return out


def _read_sources(paths, source_format, units):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not isinstance(paths, (list, tuple)) or not 1 <= len(paths) <= 1000:
        raise ValueError("须显式选择1–1000个航空数据文件，不自动扫描或下载数据")
    files = sorted({Path(p).expanduser().resolve() for p in paths}, key=str)
    observations, references = [], []
    invalid_rows = 0
    if source_format == "csv_snapshot":
        if not isinstance(units, dict) or any(units.get(k) != v for k, v in CSV_UNITS.items()):
            raise ValueError("CSV必须声明UTC时间、经纬度degrees与altitude_m米单位")
        if set(units) - set(CSV_UNITS) - {"persons"} or ("persons" in units and units["persons"] != "count"):
            raise ValueError("CSV人数单位必须为count，不能接受未知单位")
    elif source_format != "readsb_trace":
        raise ValueError("航空数据格式须为csv_snapshot或readsb_trace")
    for path in files:
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError("航空源文件不存在或超过100MB")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        references.append({"path": str(path), "sha256": sha, "bytes": len(raw)})
        if source_format == "csv_snapshot":
            rows = csv.DictReader(raw.decode("utf-8-sig").splitlines())
            if not {"aircraft_id", *CSV_UNITS} <= set(rows.fieldnames or []):
                raise ValueError("航空CSV须含aircraft_id,timestamp,lat,lon,altitude_m")
            if "persons" in (rows.fieldnames or []) and units.get("persons") != "count":
                raise ValueError("CSV含persons列时必须声明persons=count")
            iterator = rows
        else:
            if path.name.endswith(".gz"):
                with gzip.open(path, "rb") as stream:
                    decoded = stream.read(MAX_BYTES + 1)
                if len(decoded) > MAX_BYTES:
                    raise ValueError("航空gzip解压后超过100MB")
            else:
                decoded = raw
            data = json.loads(decoded)
            if not isinstance(data, dict) or not isinstance(data.get("trace"), list) or not data.get("icao"):
                raise ValueError("readsb文件必须是单机icao/timestamp/trace历史JSON")
            base = _finite(data["timestamp"])
            iterator = data["trace"]
        for row_number, row in enumerate(iterator):
            if len(observations) + invalid_rows >= MAX_OBSERVATIONS:
                raise ValueError("航空观测超过100万条，请先选择更小日期/空域文件范围")
            identifier, stamp = None, None
            try:
                if source_format == "csv_snapshot":
                    identifier = str(row["aircraft_id"]).strip().lower()
                    stamp = _utc(row["timestamp"])
                    lat, lon = _finite(row["lat"]), _finite(row["lon"])
                    ground = str(row["altitude_m"]).strip().lower() == "ground"
                    altitude = 0.0 if ground else _finite(row["altitude_m"])
                    persons = None if not row.get("persons", "").strip() else _finite(row["persons"])
                    stale = False
                else:
                    if not isinstance(row, list) or len(row) < 7:
                        raise ValueError("trace记录字段不足")
                    identifier = str(data["icao"]).strip().lower()
                    stamp = base + _finite(row[0])
                    lat, lon = _finite(row[1]), _finite(row[2])
                    ground = row[3] == "ground"
                    altitude = 0.0 if ground else _finite(row[3]) * 0.3048
                    # readsb bit 1 explicitly marks a stale position.
                    stale = bool(int(row[6]) & 1)
                    persons = None
                if not identifier:
                    raise ValueError("航空器编号为空")
                observations.append({"aircraft_id": identifier, "timestamp": stamp, "lat": lat, "lon": lon,
                                     "altitude_m": altitude, "ground": ground, "stale": stale, "persons": persons,
                                     "source_sha256": sha, "row_number": row_number, "invalid": False})
            except (TypeError, ValueError, KeyError, OverflowError):
                if identifier and stamp is not None:
                    # An invalid newest position must block resurrection of an
                    # earlier valid location of the same aircraft.
                    observations.append({"aircraft_id": identifier, "timestamp": stamp,
                                         "invalid": True, "source_sha256": sha, "row_number": row_number})
                else:
                    invalid_rows += 1
    return observations, references, invalid_rows


def _vectors(locations, profile):
    """Satellite-to-ground unit vectors; matches the centre-user ground model."""
    lat, lon = np.deg2rad(np.asarray(locations).T)
    surface = profile.earth_radius_km * np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
    sat_lon = np.deg2rad(profile.satellite_longitude_deg)
    satellite = profile.orbit_radius_km * np.array([np.cos(sat_lon), np.sin(sat_lon), 0.0])
    rays = surface - satellite
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _cover(locations, demand, profile):
    """Demand weighted farthest-point medoids, then exclusive nearest association.

    Centres are existing unique positions, so every selected centre has positive
    demand and lies inside the service region. This avoids fictitious empty beams.
    """
    unique = np.unique(locations, axis=0)
    if len(unique) < 100:
        raise AviationSceneError("可用正需求航空器的唯一位置不足100，不能生成100个非空波束",
                                 {"unique_positive_positions": len(unique)})
    weights = np.zeros(len(unique))
    inverse = np.unique(locations, axis=0, return_inverse=True)[1]
    np.add.at(weights, inverse, demand)
    rays = _vectors(unique, profile)
    chosen = [int(np.argmax(weights))]
    nearest = np.full(len(unique), np.inf)
    for _ in range(99):
        distance = np.sum((rays - rays[chosen[-1]]) ** 2, axis=1)
        nearest = np.minimum(nearest, distance)
        score = nearest * weights
        score[chosen] = -1
        chosen.append(int(np.argmax(score)))
    centres = unique[chosen]
    dots = np.clip(_vectors(locations, profile) @ rays[chosen].T, -1.0, 1.0)
    assigned = np.argmax(dots, axis=1)
    angles = np.rad2deg(np.arccos(dots[np.arange(len(locations)), assigned]))
    covered = angles <= 0.25 + 1e-10
    assigned = np.where(covered, assigned, -1)
    rates = np.bincount(assigned[covered], weights=demand[covered], minlength=100)
    if (rates <= 0).any():
        raise AviationSceneError("覆盖聚合出现空束，不补充人工需求")
    return centres, rates, assigned, angles


def build_aviation_scenes(paths, selected_times, *, source_format="csv_snapshot", units=None,
                          demand_model, seed, max_record_age_seconds=300,
                          environment_profile="europe_v1", partition_role, window_seconds=None):
    """Return canonical scene arrays and JSON provenance, without writing files.

    Explicit times are UTC ISO strings. The caller owns source/date partition
    exclusivity checks against previously archived workbench scene collections.
    """
    from satellite_rl.envs.regions import get_profile, profile_metadata, geo_distances_km, elevation_degrees

    if environment_profile != "europe_v1":
        raise ValueError("航空适配器首版仅支持独立europe_v1环境")
    if partition_role not in ("train", "validation", "test"):
        raise ValueError("每批航空数据须明确train/validation/test用途")
    if not isinstance(selected_times, (list, tuple)) or not 1 <= len(selected_times) <= 1000:
        raise ValueError("须显式提供1–1000个UTC场景时刻")
    times = [_utc(t) for t in selected_times]
    if len(set(times)) != len(times):
        raise ValueError("场景时刻不能重复")
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed <= 2 ** 32 - 1:
        raise ValueError("随机种子须为0至2^32-1整数")
    max_age = _finite(max_record_age_seconds)
    if max_age < 0:
        raise ValueError("最大观测年龄不能为负")
    model = _model(demand_model)
    if window_seconds is not None:
        if "window_seconds" in demand_model and _finite(window_seconds) != model["window_seconds"]:
            raise ValueError("两处统计窗口设置冲突")
        model["window_seconds"] = _finite(window_seconds)
        if model["window_seconds"] <= 0:
            raise ValueError("统计窗口必须为正")
    observations, references, invalid_rows = _read_sources(paths, source_format, units)
    profile = get_profile(environment_profile)
    arrays, snapshots = [], []
    for stamp in times:
        counts = {"raw_records": len(observations) + invalid_rows,
                  "invalid_records": invalid_rows + sum(o["invalid"] for o in observations),
                  "future_records": 0, "duplicate_historical_records": 0, "expired_aircraft": 0,
                  "ground_aircraft": 0, "invalid_aircraft": 0, "stale_aircraft": 0,
                  "outside_region_aircraft": 0, "below_elevation_aircraft": 0}
        latest = {}
        for obs in observations:
            if obs["timestamp"] > stamp:
                counts["future_records"] += 1
                continue
            identifier = obs["aircraft_id"]
            if identifier in latest:
                counts["duplicate_historical_records"] += 1
                old = latest[identifier]
                if obs["timestamp"] == old["timestamp"] and any(obs.get(k) != old.get(k) for k in ("invalid", "lat", "lon", "altitude_m", "ground", "stale", "persons")):
                    raise AviationSceneError("同一飞机同一UTC时刻出现冲突观测，请先核对源文件", counts)
                if obs["timestamp"] <= old["timestamp"]:
                    continue
            latest[identifier] = obs
        selected = []
        for identifier in sorted(latest):
            obs = latest[identifier]
            if stamp - obs["timestamp"] > max_age:
                counts["expired_aircraft"] += 1
            elif obs["invalid"]:
                counts["invalid_aircraft"] += 1
            elif obs["ground"] or obs["altitude_m"] <= 0:
                counts["ground_aircraft"] += 1
            elif not (-90 <= obs["lat"] <= 90 and -180 <= obs["lon"] <= 180) or obs["altitude_m"] > 25000:
                counts["invalid_aircraft"] += 1
            elif obs["stale"]:
                counts["stale_aircraft"] += 1
            elif not (profile.latitude_bounds_deg[0] <= obs["lat"] <= profile.latitude_bounds_deg[1]
                      and profile.longitude_bounds_deg[0] <= obs["lon"] <= profile.longitude_bounds_deg[1]):
                counts["outside_region_aircraft"] += 1
            elif elevation_degrees([[obs["lat"], obs["lon"]]], profile)[0] < profile.min_elevation_deg:
                counts["below_elevation_aircraft"] += 1
            else:
                selected.append(obs)
        counts["selected_aircraft"] = len(selected)
        if len(selected) > 100000:
            raise AviationSceneError("单时刻可用航空器超过10万，须先限定日期与空域", counts)
        if len(selected) < 100:
            raise AviationSceneError("清洗后可用航空器不足100，不能生成100个非空波束", counts)
        # Timestamp-derived stream makes snapshots reproducible even when reordered.
        stream_id = hashlib.sha256(_iso(stamp).encode()).digest()
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), *np.frombuffer(stream_id, dtype="<u4").tolist()]))
        persons = np.array([model["persons_per_aircraft"] if o["persons"] is None else o["persons"] for o in selected])
        if (persons < 0).any() or (persons > 10000).any() or not np.equal(persons, persons.astype(int)).all():
            raise AviationSceneError("每机人数须为0–10000整数", counts)
        fraction = model["satellite_access_fraction"] * model["active_fraction"]
        active = rng.binomial(persons.astype(int), fraction) if model["sampling"] == "binomial" else persons * fraction
        demand = (active * model["rate_mbps_per_active_person"] + model["aircraft_base_mbps"]) * 1e6
        sigma = model["lognormal_sigma"]
        if sigma:
            demand *= np.exp(rng.normal(-sigma ** 2 / 2, sigma, len(selected)))
        if not np.isfinite(demand).all():
            raise AviationSceneError("合成业务需求超过有限数值范围", counts)
        positive = demand > 0
        counts["zero_demand_aircraft"] = int((~positive).sum())
        positions = np.array([[o["lat"], o["lon"]] for o in selected])
        try:
            centres, rates, association, angles = _cover(positions[positive], demand[positive], profile)
        except AviationSceneError as exc:
            raise AviationSceneError(str(exc), {**counts, **exc.diagnostics}) from exc
        full_association = np.full(len(selected), -1, dtype=int)
        full_association[positive] = association
        full_angles = np.full(len(selected), np.nan)
        full_angles[positive] = angles
        counts["covered_aircraft"] = int((full_association >= 0).sum())
        counts["uncovered_positive_aircraft"] = int(((full_association < 0) & positive).sum())
        total, covered = float(demand.sum()), float(rates.sum())
        uncovered = float(demand[full_association < 0].sum())
        if not np.isclose(total, covered + uncovered, rtol=1e-12, atol=1e-6):
            raise AviationSceneError("需求汇总未守恒", counts)
        order = np.argsort(-rates, kind="stable")
        processing_index = np.empty(100, dtype=int)
        processing_index[order] = np.arange(100)
        arrays.append({"beam_location": centres[order], "rate": rates[order].reshape(100, 1),
                       "distances": geo_distances_km(centres[order], profile),
                       "beam_G_Tx_max": np.full((100, 1), 50.0), "UE_G_Rx_max": np.full((100, 1), 40.0),
                       "noise_temperature": np.full((100, 1), 290.0), "original_indices": order})
        selected_records = [{"aircraft_id": obs["aircraft_id"], "timestamp_utc": _iso(obs["timestamp"]),
                             "source_date": _iso(obs["timestamp"])[:10], "source_sha256": obs["source_sha256"],
                             "lat": obs["lat"], "lon": obs["lon"], "altitude_m": obs["altitude_m"],
                             "persons": int(persons[i]), "demand_bps": float(demand[i]),
                             "beam_id": int(full_association[i]),
                             "processing_index": int(processing_index[full_association[i]]) if full_association[i] >= 0 else -1,
                             "off_axis_degrees": float(full_angles[i]) if positive[i] else None}
                            for i, obs in enumerate(selected)]
        source_dates = sorted({_iso(o["timestamp"])[:10] for o in latest.values()
                               if stamp - o["timestamp"] <= max_age})
        snapshots.append({"selected_time_utc": _iso(stamp), "source_dates": source_dates,
                          "counts": counts, "total_demand_bps": total, "covered_demand_bps": covered,
                          "uncovered_demand_bps": uncovered, "beam_order_original_indices": order.tolist(),
                          "source_observations": selected_records})
    provenance = {"type": "aviation_imported_positions_synthetic_demand", "adapter_version": "aviation_v1",
                  "position_source_status": "user_supplied_not_independently_verified",
                  "source_format": source_format, "files": references, "seed": int(seed), "partition_role": partition_role,
                  "selected_times": [_iso(t) for t in times],
                  "environment_profile": environment_profile, "profile": profile_metadata(profile),
                  "source_dates": sorted({date for s in snapshots for date in s["source_dates"]}),
                  "snapshot_dates": sorted({_iso(t)[:10] for t in times}),
                  "units": {"rate": "bit/s", "position": "degrees", "altitude": "m", "persons": "count"},
                  "input_units": units if source_format == "csv_snapshot" else {"timestamp": "unix_seconds", "altitude": "ft"},
                  "demand_model": model, "max_record_age_seconds": max_age, "snapshots": snapshots,
                  "time_selection": "latest observation at or before selected UTC time; no interpolation; stale flag excluded",
                  "duplicate_policy": "latest timestamp per aircraft; equal-time conflicting observations reject; equal records first lexical file",
                  "coverage": {"algorithm": "demand-weighted farthest-point medoids; exclusive nearest beam",
                               "beam_count": 100, "beam_full_width_degrees": 0.5, "coverage_radius_degrees": 0.25,
                               "angle_reference": "satellite-to-ground line of sight", "aircraft_altitude": "ground projection for geometry; altitude retained and airborne filtered",
                               "beam_order": "stable descending aggregated demand; group=processing_index%8",
                               "beam_identity": "original medoid selection index 0..99; original_indices and per-snapshot beam_order_original_indices preserve mapping",
                               "uncovered_policy": "retain IDs and demand; never duplicate or redistribute uncovered demand",
                               "link_approximation": "beam centre represents all assigned users; not per-aircraft throughput"},
                  "source_date_policy": "UTC dates of latest nonfuture observations within maximum age, including filtered aircraft",
                  "demand_interpretation": "synthetic static window-average requested rate; no measured communication traffic or queue",
                  "schema_sources": [SCHEMA_URL, HISTORY_URL]}
    return arrays, provenance
