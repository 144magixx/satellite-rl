"""Independent SHA, date and demand-conservation check for the 100 test scenes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

FROZEN_SRC = os.environ.get("STAGE3_G07_FROZEN_SRC")
if FROZEN_SRC:
    sys.path.insert(0, FROZEN_SRC)
    from satellite_rl.envs.regions import geo_distances_km, validate_locations  # noqa: E402


DATA = Path(os.environ.get("STAGE3_DATA_ROOT", "D:/paper/outputs/experiments/stage3_all_routes_20261003/data"))
MANIFEST = DATA / "independent_test_manifest.json"
RECORD = DATA / "staged_test" / "record.json"
OUTPUT = DATA / "independent_test_verification.json"
EXPECTED_DAY = os.environ.get("STAGE3_DATE", "2024-01-26")
EXPECTED_BYTES = int(json.loads((DATA / "source_preflight.json").read_text(encoding="utf-8"))["expected_bytes"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Explicit UTC time required")
    return parsed.astimezone(timezone.utc)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    pin = json.loads((DATA / "source_preflight.json").read_text(encoding="utf-8"))
    contract = json.loads((DATA / "selection_contract.json").read_text(encoding="utf-8"))
    download = json.loads((DATA / "download_status.json").read_text(encoding="utf-8"))
    prep = json.loads((DATA / "preparation_status.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "prepared_not_imported_or_evaluated"
            or prep["status"] != "completed_not_imported_or_evaluated"
            or download["status"] != "completed" or record["count"] != 100
            or len(record["files"]) != 100 or len(record["source"]["snapshots"]) != 100
            or len(manifest["csv_frames"]["frames"]) != 100
            or manifest["csv_frames"]["rejected_times"]
            or record["source"]["partition_role"] != "test"
            or record["profile"] != "europe_v1"
            or record["source"]["source_dates"] != [EXPECTED_DAY]
            or record["source"]["snapshot_dates"] != [EXPECTED_DAY]):
        raise RuntimeError("Independent test status/count/profile/date gate failed")
    expected_indices = [int((i + 0.5) * 288 / 100) for i in range(100)]
    actual_indices = [frame["candidate_index"] for frame in manifest["csv_frames"]["frames"]]
    actual_times = [frame["selected_time_utc"] for frame in manifest["csv_frames"]["frames"]]
    if (actual_indices != expected_indices or actual_indices != contract["indices"]
            or actual_times != contract["selected_times_utc"]
            or [utc(value) for value in record["source"]["selected_times"]]
               != [utc(value) for value in contract["selected_times_utc"]]):
        raise RuntimeError("100 test times differ from the predeclared uniform rule")
    archive = DATA / "raw" / f"v{EXPECTED_DAY.replace('-', '.')}-planes-readsb-staging-0.tar"
    archive_hash = sha256(archive)
    asset = download["assets"][EXPECTED_DAY.replace("-", ".")]
    if (archive.stat().st_size != EXPECTED_BYTES or asset["bytes"] != EXPECTED_BYTES
            or pin["expected_bytes"] != EXPECTED_BYTES
            or archive_hash != asset["sha256"] or archive_hash != manifest["source_archive_sha256"]
            or record["source"]["upstream_archive"]["sha256"] != archive_hash):
        raise RuntimeError("Original archive bytes/SHA/provenance differ")
    if (sha256(DATA / "source_preflight.json") != manifest["source_preflight_sha256"]
            or sha256(Path(record["source"]["preparation_reference"]["path"]))
               != manifest["reference_code_sha256"]
            or sha256(DATA / "PREFERRED_RELEASES.txt") != pin["preferred_sha256"]
            or sha256(DATA / "LICENSE-ODbL-official.txt") != pin["license_sha256"]
            or sha256(DATA / "selection_contract.json") != pin["selection_contract_sha256"]
            or sha256(Path(record["source"]["frozen_source"]["root"]) / "src" /
                      "satellite_rl" / "workbench" / "aviation.py")
               != record["source"]["frozen_source"]["aviation_sha256"]
            or sha256(Path(record["source"]["frozen_source"]["root"]) / "src" /
                      "satellite_rl" / "envs" / "regions.py")
               != record["source"]["frozen_source"]["regions_sha256"]):
        raise RuntimeError("Source/preparation/reference code evidence changed")
    if manifest["staged_record_sha256"] != sha256(RECORD):
        raise RuntimeError("Staged sidecar changed")
    snapshots = record["source"]["snapshots"]
    selected, covered, total, served = [], 0, [], []
    for index, (frame, reference, snapshot) in enumerate(zip(
            manifest["csv_frames"]["frames"], record["files"], snapshots)):
        csv_path, scene_path = Path(frame["path"]), Path(reference["path"])
        detail = snapshot["source_observations_ref"]
        if (sha256(csv_path) != frame["sha256"]
                or sha256(scene_path) != reference["sha256"]
                or sha256(Path(detail["path"])) != detail["sha256"]
                or utc(frame["selected_time_utc"]) != utc(snapshot["selected_time_utc"])
                or utc(frame["selected_time_utc"]).date().isoformat() != EXPECTED_DAY):
            raise RuntimeError(f"CSV/NPZ/detail SHA or time mismatch at scene {index}")
        with gzip.open(detail["path"], "rt", encoding="utf-8") as stream:
            if sum(1 for _ in stream) != detail["count"]:
                raise RuntimeError(f"External aircraft detail count mismatch at scene {index}")
        with np.load(scene_path, allow_pickle=False) as scene:
            rates = np.asarray(scene["rate"], dtype=np.float64)
            locations = np.asarray(scene["beam_location"])
            distances = np.asarray(scene["distances"])
            original = np.asarray(scene["original_indices"])
            if rates.shape != (100, 1) or not np.isfinite(rates).all() or (rates <= 0).any():
                raise RuntimeError(f"Nonfinite/nonpositive beam demand at scene {index}")
            if locations.shape != (100, 2) or distances.shape != (100,) or original.shape != (100,):
                raise RuntimeError(f"Scene geometry or identity shape differs at scene {index}")
            if FROZEN_SRC:
                if (not np.array_equal(validate_locations(locations, "europe_v1"), locations)
                        or not np.allclose(geo_distances_km(locations, "europe_v1"), distances,
                                           rtol=1e-12, atol=1e-8)
                        or original.tolist() != snapshot["beam_order_original_indices"]):
                    raise RuntimeError(f"Geometry/distance/beam order differs at scene {index}")
            if not math.isclose(float(rates.sum()), snapshot["covered_demand_bps"],
                                rel_tol=1e-12, abs_tol=1e-5):
                raise RuntimeError(f"Beam and aircraft demand mismatch at scene {index}")
        if not math.isclose(snapshot["total_demand_bps"],
                            snapshot["covered_demand_bps"] + snapshot["uncovered_demand_bps"],
                            rel_tol=1e-12, abs_tol=1e-5):
            raise RuntimeError(f"Coverage demand conservation failed at scene {index}")
        n = snapshot["counts"]["selected_aircraft"]
        if n < 100 or snapshot["counts"]["covered_aircraft"] > n:
            raise RuntimeError(f"Aircraft count gate failed at scene {index}")
        selected.append(n)
        covered += snapshot["counts"]["covered_aircraft"]
        total.append(snapshot["total_demand_bps"])
        served.append(snapshot["covered_demand_bps"])
    summary = {"status": "verified_input_scenes_only_not_model_evaluation",
               "verified_at_utc": datetime.now(timezone.utc).isoformat(),
               "archive_path": str(archive), "archive_bytes": EXPECTED_BYTES,
               "archive_sha256": archive_hash, "scene_count": 100,
               "preselected_indices": expected_indices, "source_date": EXPECTED_DAY,
               "record_path": str(RECORD), "record_sha256": sha256(RECORD),
               "csv_sha_checked": 100, "npz_sha_checked": 100, "detail_sha_and_lines_checked": 100,
               "geometry_distance_beam_order_checked": 100 if FROZEN_SRC else 0,
               "aircraft_min": min(selected), "aircraft_max": max(selected),
               "aircraft_mean": sum(selected) / 100,
               "aircraft_coverage_fraction": covered / sum(selected),
               "total_demand_gbps_mean": sum(total) / 100 / 1e9,
               "demand_coverage_fraction": sum(served) / sum(total),
               "no_training_or_model_evaluation": True}
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "scene_count", "archive_sha256",
                                                   "demand_coverage_fraction")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
