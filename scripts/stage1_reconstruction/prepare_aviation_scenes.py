"""Prepare a frozen S1-03 independent European aviation test date.

Waits for the pinned archive, then applies the existing S1-01 cleaning and
aviation adapter under S1-01's frozen source. Produces 100 test scenes and
compact, auditable provenance. Does not import to the workbench or evaluate.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


DATA = Path(os.environ.get("STAGE1_DATA_ROOT", "D:/paper/outputs/experiments/first_stage_s1_03_20261005/test/data"))
ROOT = DATA.parent
DAY = os.environ.get("STAGE1_DATE", "2024-01-31")
TAG = f"v{DAY.replace('-', '.')}-planes-readsb-staging-0"
ARCHIVE = DATA / "raw" / (TAG + ".tar")
PIN = DATA / "source_preflight.json"
EXPECTED_BYTES = int(json.loads(PIN.read_text(encoding="utf-8"))["expected_bytes"])
CONTRACT = DATA / "selection_contract.json"
DOWNLOAD = DATA / "download_status.json"
STATUS = DATA / "preparation_status.json"
FROZEN_ROOT = Path(os.environ.get("STAGE1_SOURCE_ROOT", "/home/u2024110224/first_stage_satellite_rl/outputs/workbench/source_snapshots/b3a11c51e9da6612a43cb005a79c9e89a37e448d"))
FROZEN_SRC = FROZEN_ROOT / "src"
STAGE1_TRAIN = Path(os.environ.get("STAGE1_TRAIN_RECORD", "/home/u2024110224/first_stage_satellite_rl/data_inputs/train/record.json"))
REFERENCE = Path(__file__).with_name("prepare_snapshots_reference.py")
UNITS = {"timestamp": "utc_iso8601", "lat": "degrees", "lon": "degrees",
         "altitude_m": "m", "persons": "count"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def wait_download(state: dict) -> dict:
    deadline = time.monotonic() + 72 * 3600
    while time.monotonic() < deadline:
        if DOWNLOAD.is_file():
            current = json.loads(DOWNLOAD.read_text(encoding="utf-8"))
            if current.get("status") == "failed":
                raise RuntimeError("Pinned archive download failed: " + str(current.get("error")))
            if current.get("status") == "completed":
                return current
        time.sleep(60)
    raise RuntimeError("Archive not complete within 72 hours")


def verify_archive(download: dict) -> tuple[dict, str]:
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    if (pin["archive_name"] != ARCHIVE.name or pin["expected_bytes"] != EXPECTED_BYTES
            or pin["group"] != "S1-03" or pin["date_utc"] != DAY
            or download["expected_bytes"] != EXPECTED_BYTES
            or download["bytes"] != EXPECTED_BYTES
            or download["url"] != pin["archive_url"]
            or download["asset_id"] != pin["asset_id"] or not ARCHIVE.is_file()
            or ARCHIVE.stat().st_size != EXPECTED_BYTES):
        raise RuntimeError("Pinned release URL, instance or byte count differs")
    for filename, hash_key in (("release_api.json", "release_api_sha256"),
                               ("PREFERRED_RELEASES.txt", "preferred_sha256"),
                               ("LICENSE-ODbL-official.txt", "license_sha256"),
                               ("selection_contract.json", "selection_contract_sha256")):
        if sha256(DATA / filename) != pin[hash_key]:
            raise RuntimeError("Official source or preselected contract changed: " + filename)
    if pin["archive_url"] not in (DATA / "PREFERRED_RELEASES.txt").read_text(encoding="utf-8"):
        raise RuntimeError("Pinned date missing from official preferred release list")
    actual = sha256(ARCHIVE)
    if actual != download["sha256"]:
        raise RuntimeError("Downloaded archive SHA differs from status evidence")
    return pin, actual


def import_frozen_modules():
    if not FROZEN_SRC.is_dir():
        raise RuntimeError("S1-01 frozen source snapshot is missing")
    frozen = json.loads((FROZEN_ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    if frozen["git_commit"] != FROZEN_ROOT.name:
        raise RuntimeError("S1-01 frozen source commit differs")
    sys.path.insert(0, str(FROZEN_SRC))
    import satellite_rl
    if Path(satellite_rl.__file__).resolve() != FROZEN_SRC / "satellite_rl" / "__init__.py":
        raise RuntimeError("Aviation adapter did not import from S1-01 frozen source")
    from satellite_rl.envs.regions import profile_metadata
    from satellite_rl.workbench.aviation import build_aviation_scenes
    spec = importlib.util.spec_from_file_location("s101_preparation_reference", REFERENCE)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    prep.ROOT = DATA
    prep.STATUS = STATUS
    prep.DATES = (DAY,)
    prep.ROLES = ("test",)
    prep.MAX_AGE = 60.0
    return prep, build_aviation_scenes, profile_metadata


def main() -> None:
    state = {"status": "waiting_for_archive", "started_utc": datetime.now(timezone.utc).isoformat(),
             "date_utc": DAY, "role": "independent_test", "planned_scene_count": 100}
    write_json(STATUS, state)
    try:
        download = wait_download(state)
        pin, archive_sha = verify_archive(download)
        state.update(status="processing_288_fixed_grid", archive_sha256=archive_sha)
        write_json(STATUS, state)
        prep, build_aviation_scenes, profile_metadata = import_frozen_modules()
        rows, candidates = prep.process_day(DAY, state)
        indices = prep.select_indices(100)
        selected = prep.final_csv(DAY, "test", indices, rows, candidates)
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        if (indices != contract["indices"] or selected["preselected_times"] != contract["selected_times_utc"]):
            raise RuntimeError("Selected times differ from frozen contract")
        if (len(indices) != 100 or len(selected["frames"]) != 100
                or len(selected["preselected_times"]) != 100 or selected["rejected_times"]):
            raise RuntimeError("Predetermined 100 times did not all produce valid scenes; no replacements allowed")
        train_record = json.loads(STAGE1_TRAIN.read_text(encoding="utf-8"))
        train_source = train_record["source"]
        model = dict(train_source["demand_model"])
        if (train_source["seed"] != 51001 or train_source["max_record_age_seconds"] != 60
                or model["window_seconds"] != 60 or model["persons_per_aircraft"] != 180
                or model["satellite_access_fraction"] != 0.2 or model["active_fraction"] != 0.3
                or model["rate_mbps_per_active_person"] != 2.0 or model["aircraft_base_mbps"] != 0
                or model["sampling"] != "binomial" or model["lognormal_sigma"] != 0.35):
            raise RuntimeError("S1-01 synthetic-demand parameter source differs from approved common setting")
        target = DATA / "staged_test"
        target.mkdir(parents=True, exist_ok=True)
        scenes, snapshots, inputs = [], [], []
        for index, frame in enumerate(selected["frames"]):
            source_file = Path(frame["path"])
            if sha256(source_file) != frame["sha256"]:
                raise RuntimeError(f"Snapshot CSV SHA mismatch at predetermined index {index}")
            arrays, source = build_aviation_scenes(
                [source_file], [frame["selected_time_utc"]], source_format="csv_snapshot",
                units=UNITS, demand_model=model, seed=51001,
                max_record_age_seconds=60, environment_profile="europe_v1",
                partition_role="test", window_seconds=60)
            if len(arrays) != 1 or len(source["snapshots"]) != 1:
                raise RuntimeError("Adapter returned an unexpected number of test scenes")
            snapshot = dict(source["snapshots"][0])
            observations = snapshot.pop("source_observations")
            detail = target / f"source_observations_{index:05d}.jsonl.gz"
            with detail.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                    for aircraft in observations:
                        stream.write(json.dumps(aircraft, ensure_ascii=False, separators=(",", ":")) + "\n")
            snapshot["source_observations_ref"] = {"path": str(detail), "sha256": sha256(detail),
                                                   "count": len(observations), "format": "jsonl.gz"}
            scene = target / f"scene_{index:05d}.npz"
            np.savez_compressed(scene, **arrays[0])
            with np.load(scene, allow_pickle=False) as check:
                rates = np.asarray(check["rate"], dtype=np.float64)
                if rates.shape != (100, 1) or not np.isfinite(rates).all() or (rates <= 0).any():
                    raise RuntimeError(f"Scene {index} has invalid beam demand")
                if not math.isclose(float(rates.sum()), snapshot["covered_demand_bps"],
                                    rel_tol=1e-12, abs_tol=1e-5):
                    raise RuntimeError(f"Scene {index} beam demand is not conserved")
            scenes.append({"path": str(scene), "sha256": sha256(scene), "scene_id": scene.stem})
            snapshots.append(snapshot)
            inputs.append({"path": str(source_file), "sha256": frame["sha256"],
                           "selected_time_utc": frame["selected_time_utc"],
                           "archive_sha256": archive_sha})
            state.update(status="building_100_test_scenes", built_count=index + 1)
            if (index + 1) % 10 == 0 or index == 99:
                write_json(STATUS, state)
        source = {key: value for key, value in source.items()
                  if key not in ("snapshots", "selected_times", "source_dates", "snapshot_dates", "files")}
        source.update(files=inputs, seed=51001, partition_role="test",
                      selected_times=[item["selected_time_utc"] for item in snapshots],
                      source_dates=sorted({day for item in snapshots for day in item["source_dates"]}),
                      snapshot_dates=[DAY], snapshots=snapshots, demand_model=model,
                      upstream_archive={"path": str(ARCHIVE), "sha256": archive_sha,
                                        "bytes": EXPECTED_BYTES, "official_url": pin["archive_url"]},
                      source_preflight={"path": str(PIN), "sha256": sha256(PIN)},
                      preparation_reference={"path": str(REFERENCE), "sha256": sha256(REFERENCE)},
                      frozen_source={"root": str(FROZEN_ROOT), "git_commit": FROZEN_ROOT.name,
                                     "aviation_sha256": sha256(FROZEN_SRC / "satellite_rl" / "workbench" / "aviation.py"),
                                     "regions_sha256": sha256(FROZEN_SRC / "satellite_rl" / "envs" / "regions.py")},
                      demand_seed_rule="base 51001 plus SHA256(UTC selected time) in adapter SeedSequence",
                      detail_storage="per-frame JSONL.gz, with path/SHA/count in each snapshot")
        if source["source_dates"] != [DAY] or source["snapshot_dates"] != [DAY]:
            raise RuntimeError("Independent test dates escaped the predeclared UTC day")
        profile = profile_metadata("europe_v1")
        compact_day = DAY.replace("-", "")
        record = {"id": f"staged_europe_stage1_s103_test_{compact_day}", "name": f"欧洲航空_第一阶段S1-03独立考核_{compact_day}",
                  "path": str(target), "count": len(scenes),
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "source": source, "files": scenes, "profile": "europe_v1",
                  "environment_version": profile["environment_version"],
                  "environment_profile": profile,
                  "units": {"rate": "bit/s", "beam_location": "degrees"}}
        write_json(target / "record.json", record)
        selected_counts = [snap["counts"]["selected_aircraft"] for snap in snapshots]
        demand = [snap["total_demand_bps"] for snap in snapshots]
        coverage = sum(snap["covered_demand_bps"] for snap in snapshots) / sum(demand)
        summary = {
            "status": "prepared_not_imported_or_evaluated",
            "date_utc": DAY, "role": "independent_test", "count": 100,
            "selection_rule": "floor((i+0.5)*288/100), i=0..99; fixed before viewing data",
            "source_archive_sha256": archive_sha,
            "source_preflight_sha256": sha256(PIN),
            "reference_code_sha256": sha256(REFERENCE),
            "frozen_source_commit": FROZEN_ROOT.name,
            "stage1_train_record_sha256": sha256(STAGE1_TRAIN),
            "selected_aircraft_min": min(selected_counts),
            "selected_aircraft_max": max(selected_counts),
            "selected_aircraft_mean": sum(selected_counts) / 100,
            "total_demand_gbps_mean": sum(demand) / 100 / 1e9,
            "demand_coverage_fraction": coverage,
            "csv_frames": selected, "staged_record": str(target / "record.json"),
            "staged_record_sha256": sha256(target / "record.json"),
            "candidate_summary": str(DATA / "candidates" / DAY / "summary.json"),
        }
        write_json(DATA / "independent_test_manifest.json", summary)
        state.update(status="completed_not_imported_or_evaluated", built_count=100,
                     manifest=str(DATA / "independent_test_manifest.json"),
                     ended_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(status="failed", error=repr(error), ended_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write_json(STATUS, state)


if __name__ == "__main__":
    main()
