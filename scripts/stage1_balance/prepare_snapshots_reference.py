"""Convert the pinned ADSB.lol UTC daily trace to European CSV snapshots.

Run in a persistent process. All candidate times are fixed before inspecting data.
Invalid latest records block earlier positions, including when stale/ground.
"""

from bisect import bisect_right
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time


ROOT = Path(__file__).resolve().parent
FROZEN_SRC = Path(os.environ.get("STAGE1_FROZEN_SRC", "/home/u2024110224/first_stage_satellite_rl/outputs/workbench/source_snapshots/b3a11c51e9da6612a43cb005a79c9e89a37e448d/src"))
sys.path.insert(0, str(FROZEN_SRC))
from satellite_rl.envs.regions import elevation_degrees, get_profile  # noqa: E402


DATES = ("2024-01-28",)
ROLES = ("test",)
PROFILE = get_profile("europe_v1")
MAX_AGE = 60.0
SOURCE_INSTANCE = "staging-0"
FIELDNAMES = ("aircraft_id", "timestamp", "lat", "lon", "altitude_m")
STATUS = ROOT / "preparation_status.json"


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_iso(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def candidate_times(day):
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(minutes=2)
    return [start + timedelta(minutes=5 * index) for index in range(288)]


def select_indices(count, total=288):
    result = [int((index + 0.5) * total / count) for index in range(count)]
    assert len(set(result)) == count and max(result) < total
    return result


def inspect_record(row):
    if not isinstance(row, list) or len(row) < 7:
        return "invalid", None
    try:
        lat, lon = float(row[1]), float(row[2])
        flags = int(row[6])
        ground = row[3] == "ground"
        altitude_m = 0.0 if ground else float(row[3]) * 0.3048
        if not all(math.isfinite(x) for x in (lat, lon, altitude_m)):
            return "invalid", None
        if ground or altitude_m <= 0:
            return "ground", None
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or altitude_m > 25000:
            return "invalid", None
        if flags & 1:
            return "stale", None
        if not (25 <= lat <= 80 and -40 <= lon <= 50):
            return "outside_rectangle", None
        if elevation_degrees([[lat, lon]], PROFILE)[0] < 10.0:
            return "below_elevation", None
        return "selected", (lat, lon, altitude_m)
    except (ValueError, TypeError, IndexError, OverflowError):
        return "invalid", None


def process_day(day, status):
    archive = ROOT / "raw" / f"v{day.replace('-', '.')}-planes-readsb-{SOURCE_INSTANCE}.tar"
    while not archive.is_file():
        download = ROOT / "download_status.json"
        if download.is_file() and json.loads(download.read_text(encoding="utf-8")).get("status") == "failed":
            raise RuntimeError(f"Download failed before {archive.name}")
        time.sleep(30)
    times = candidate_times(day)
    stamps = [t.timestamp() for t in times]
    counts = [Counter() for _ in times]
    rows = [[] for _ in times]
    types = Counter()
    seen_icao = set()
    malformed_files = 0
    source_records = 0
    nfiles = 0
    status["current_day"] = day
    write_json(STATUS, status)
    with tarfile.open(archive, mode="r|") as tar:
        for member in tar:
            if member.name.endswith("LICENSE-ODbL.txt") and day == DATES[0]:
                content = tar.extractfile(member).read()
                (ROOT / "LICENSE-ODbL.txt").write_bytes(content)
                continue
            if not member.isfile() or not member.name.startswith("./traces/") or not member.name.endswith(".json"):
                continue
            nfiles += 1
            try:
                content = tar.extractfile(member).read()
                data = json.loads(gzip.decompress(content) if content[:2] == b"\x1f\x8b" else content)
                icao = str(data["icao"]).strip().lower()
                base = float(data["timestamp"])
                trace = data["trace"]
                if not icao or not math.isfinite(base) or not isinstance(trace, list):
                    raise ValueError("Bad ICAO, base time or trace")
                if icao in seen_icao:
                    raise RuntimeError(f"Duplicate ICAO within day: {icao}")
                seen_icao.add(icao)
                offsets = [float(r[0]) for r in trace]
                if not all(math.isfinite(o) for o in offsets):
                    raise ValueError("Nonfinite offset")
                if offsets != sorted(offsets):
                    trace = [r for _, r in sorted(zip(offsets, trace), key=lambda pair: pair[0])]
                    offsets.sort()
                source_records += len(trace)
                aircraft_type = str(data.get("t") or "unknown").strip().upper()
                types[aircraft_type] += 1
                if not trace:
                    continue
                for index, stamp in enumerate(stamps):
                    position = bisect_right(offsets, stamp - base) - 1
                    if position < 0:
                        counts[index]["no_prior_record"] += 1
                        continue
                    age = stamp - (base + offsets[position])
                    if age > MAX_AGE:
                        counts[index]["expired"] += 1
                        continue
                    reason, position_data = inspect_record(trace[position])
                    counts[index][reason] += 1
                    if reason == "selected":
                        lat, lon, altitude_m = position_data
                        rows[index].append((icao, utc_iso(base + offsets[position]), lat, lon, altitude_m))
            except RuntimeError:
                raise
            except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError) as exc:
                malformed_files += 1
                if malformed_files <= 5:
                    status.setdefault("first_malformed", []).append({"member": member.name, "error": repr(exc)})
            if nfiles % 5000 == 0:
                status.update(files_processed=nfiles, source_records=source_records,
                              selected_rows=sum(len(r) for r in rows), updated_utc=utc_iso(time.time()))
                write_json(STATUS, status)

    output = ROOT / "candidates" / day
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for index, (stamp, snapshot_rows, tally) in enumerate(zip(stamps, rows, counts)):
        unique_positions = len({(r[2], r[3]) for r in snapshot_rows})
        candidate = {"index": index, "selected_time_utc": utc_iso(stamp),
                     "selected_aircraft": len(snapshot_rows), "unique_positions": unique_positions,
                     "eligible": unique_positions >= 100,
                     "rejection_reason": None if unique_positions >= 100 else "fewer_than_100_distinct_positive_positions",
                     "counts": dict(tally)}
        candidates.append(candidate)
    candidate_path = output / "candidate_stats.jsonl"
    with candidate_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    # Candidate observations stay local and make later 5-to-10 minute selection auditable.
    candidate_csv = output / "all_5min_observations.csv.gz"
    with gzip.open(candidate_csv, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("selected_time_utc", *FIELDNAMES))
        for index, snapshot_rows in enumerate(rows):
            for row in sorted(snapshot_rows):
                writer.writerow((utc_iso(stamps[index]), *row))
    summary = {"date": day, "role": ROLES[DATES.index(day)], "archive": str(archive),
               "archive_sha256": sha256(archive), "source_files": nfiles,
               "source_trace_records": source_records, "malformed_files": malformed_files,
               "candidate_count": len(candidates),
               "eligible_count": sum(c["eligible"] for c in candidates),
               "selected_rows_all_5min": sum(len(r) for r in rows),
               "selected_aircraft_min": min(len(r) for r in rows),
               "selected_aircraft_max": max(len(r) for r in rows),
               "candidate_stats": str(candidate_path), "candidate_stats_sha256": sha256(candidate_path),
               "candidate_observations": str(candidate_csv), "candidate_observations_sha256": sha256(candidate_csv),
               "aircraft_type_field": "readsb top-level t; identification only, not occupant count",
               "aircraft_type_counts_top20": types.most_common(20),
               "aircraft_type_unknown": types["UNKNOWN"]}
    write_json(output / "summary.json", summary)
    status.setdefault("days", {})[day] = summary
    status.update(files_processed=nfiles, source_records=source_records,
                  selected_rows=summary["selected_rows_all_5min"], updated_utc=utc_iso(time.time()))
    write_json(STATUS, status)
    return rows, candidates


def final_csv(day, role, indices, rows, candidates):
    output = ROOT / "snapshots" / role / day
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in indices:
        candidate = candidates[index]
        if not candidate["eligible"]:
            continue
        path = output / f"t{index:03d}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDNAMES)
            writer.writerows(sorted(rows[index]))
        frames.append({"candidate_index": index, "selected_time_utc": candidate["selected_time_utc"],
                       "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
                       "row_count": len(rows[index]), "unique_positions": candidate["unique_positions"]})
    return {"role": role, "date": day, "preselected_times": [candidates[i]["selected_time_utc"] for i in indices],
            "frames": frames, "accepted_times": [frame["selected_time_utc"] for frame in frames],
            "rejected_times": [candidates[i] for i in indices if not candidates[i]["eligible"]],
            "row_count": sum(frame["row_count"] for frame in frames),
            "total_bytes": sum(frame["bytes"] for frame in frames)}


def main():
    status = {"status": "running", "started_utc": utc_iso(time.time()), "days": {},
              "source": f"ADSB.lol globe_history_2024 planes-readsb-{SOURCE_INSTANCE}", "profile": "europe_v1"}
    write_json(STATUS, status)
    all_rows = {}
    all_candidates = {}
    try:
        for day in DATES:
            all_rows[day], all_candidates[day] = process_day(day, status)
        train_rows_5min = sum(sum(len(all_rows[day][i]) for i in range(288) if all_candidates[day][i]["eligible"])
                              for day in DATES[:2])
        train_indices = list(range(288))
        selections = {"train_2024-01-08": final_csv(DATES[0], "train", train_indices, all_rows[DATES[0]], all_candidates[DATES[0]]),
                      "train_2024-01-09": final_csv(DATES[1], "train", train_indices, all_rows[DATES[1]], all_candidates[DATES[1]]),
                      "validation_2024-01-12": final_csv(DATES[2], "validation", select_indices(20), all_rows[DATES[2]], all_candidates[DATES[2]]),
                      "test_2024-01-15": final_csv(DATES[3], "test", select_indices(100), all_rows[DATES[3]], all_candidates[DATES[3]])}
        manifest = {"status": "awaiting_archive_finalization", "generated_utc": utc_iso(time.time()),
                    "source_repo": "https://github.com/adsblol/globe_history_2024",
                    "source_preferred_releases": "https://raw.githubusercontent.com/adsblol/globe_history_2024/main/PREFERRED_RELEASES.txt",
                    "license": "ODbL-1.0", "license_file": str(ROOT / "LICENSE-ODbL.txt"),
                    "readsb_schema": "https://github.com/wiedehopf/readsb/blob/dev/README-json.md",
                    "instance": f"planes-readsb-{SOURCE_INSTANCE}", "dates_by_role": dict(zip(DATES, ROLES)),
                    "profile": {"id": "europe_v1", "satellite_longitude_deg": 13, "latitude_bounds_deg": [25, 80],
                                "longitude_bounds_deg": [-40, 50], "min_elevation_deg": 10},
                    "snapshot_grid": "00:02 UTC + 5*i minutes, i=0..287, each UTC day",
                    "validation_selection": "floor((i+0.5)*288/20), i=0..19",
                    "test_selection": "floor((i+0.5)*288/100), i=0..99",
                    "train_selection": f"all 576 preselected five-minute frames; accepted row count {train_rows_5min}; per-frame adapter calls avoid 1m combined-observation cap",
                    "max_record_age_seconds": MAX_AGE,
                    "observation_policy": "latest readsb trace at or before snapshot, within 60 seconds; newest invalid/stale/ground record blocks earlier valid record; no interpolation",
                    "demand_status": "positions observed, communication demand synthesized later by workbench",
                    "demand_model_for_import": {"persons_per_aircraft": 180, "satellite_access_fraction": 0.2,
                                                "active_fraction": 0.3, "rate_mbps_per_active_person": 2.0,
                                                "aircraft_base_mbps": 0, "sampling": "binomial", "lognormal_sigma": 0.35,
                                                "window_seconds": 60, "parameter_source": "engineering assumption, not measured occupants or requests"},
                    "seed_for_import": 51001, "source_archives": status["days"], "selections": selections,
                    "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve())}
        write_json(ROOT / "source_manifest.json", manifest)
        status["status"] = "completed"
        status["manifest"] = str(ROOT / "source_manifest.json")
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = repr(exc)
        raise
    finally:
        status["updated_utc"] = utc_iso(time.time())
        write_json(STATUS, status)


if __name__ == "__main__":
    raise SystemExit("Use data/prepare_independent_test.py for the G08 one-day selection")
