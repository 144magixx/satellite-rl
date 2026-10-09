"""Durable S1-V10 workbench preparation of frozen aircraft-level scenes."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np

from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene, _business_hash
from satellite_rl.coverage.variable_dataset import load_variable_dataset, save_variable_dataset
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import Store, digest, new_id, now, read_json


DEMAND_MODEL = {"persons_per_aircraft": 180, "satellite_access_fraction": .2,
                "active_fraction": .3, "rate_mbps_per_active_person": 2.,
                "aircraft_base_mbps": 0, "sampling": "binomial",
                "lognormal_sigma": .35, "window_seconds": 60,
                "parameter_source": "engineering assumption, not measured occupants or requests"}
UNITS = {"timestamp": "utc_iso8601", "lat": "degrees", "lon": "degrees",
         "altitude_m": "m", "persons": "count"}


def _check_source(preparation):
    for key in ("archive", "source_preflight", "selection_contract", "preparation_reference",
                "release_api", "preferred_releases", "license"):
        item = preparation[key]
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"frozen S1-V10 {key} SHA differs")
    archive = Path(preparation["archive"]["path"])
    if archive.stat().st_size != preparation["archive"]["bytes"]:
        raise ValueError("frozen source archive byte count differs")
    pin = read_json(preparation["source_preflight"]["path"])
    contract = read_json(preparation["selection_contract"]["path"])
    if (pin["group"] != "S1-V10" or contract["group"] != "S1-V10" or
            pin["date_utc"] != preparation["date_utc"] or
            contract["date_utc"] != preparation["date_utc"] or
            contract["role"] != preparation["role"] or
            pin["asset_id"] != preparation["archive"]["asset_id"]):
        raise ValueError("source release and frozen selection contract differ")
    times_sha = hashlib.sha256(json.dumps(contract["selected_times_utc"],
                              separators=(",", ":")).encode("utf-8")).hexdigest()
    if (times_sha != preparation["selected_times_sha256"] or
            len(contract["selected_times_utc"]) != preparation["selected_time_count"]):
        raise ValueError("frozen selected UTC time list differs")
    if (pin["release_api_sha256"] != preparation["release_api"]["sha256"] or
            pin["preferred_sha256"] != preparation["preferred_releases"]["sha256"] or
            pin["license_sha256"] != preparation["license"]["sha256"]):
        raise ValueError("official source metadata SHA differs")
    return pin, contract


def _reference(path, source_root, work_root, contract):
    os.environ["STAGE1_FROZEN_SRC"] = str(Path(source_root) / "src")
    spec = importlib.util.spec_from_file_location("s1_v10_trace_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = work_root
    module.STATUS = work_root / "trace_progress.json"
    module.DATES = (contract["date_utc"],)
    module.ROLES = (contract["role"],)
    module.MAX_AGE = 60.
    stamps = [datetime.fromisoformat(s.replace("Z", "+00:00"))
              for s in contract["selected_times_utc"]]
    if len(stamps) != contract["count"] or len(set(stamps)) != len(stamps):
        raise ValueError("selected UTC times differ from frozen contract")
    module.candidate_times = lambda day: stamps
    return module


def _scene_from_rows(directory, index, stamp, rows, role, archive_sha):
    from satellite_rl.workbench.aviation import build_aviation_scenes

    csv_path = directory / f"aircraft_{index:05d}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("aircraft_id", "timestamp", "lat", "lon", "altitude_m"))
        writer.writerows(sorted(rows))
    if not rows:
        return CoverageScene(stamp, np.empty((0, 2)), np.empty(0),
                             np.empty((0, 2)), [],
                             {"selected_time_utc": stamp, "source_dates": [stamp[:10]],
                              "empty_scene": True, "source_csv_sha256": digest(csv_path)},
                             digest(csv_path))
    adapter_role = "validation" if role == "development" else "test" if role == "sequence" else role
    _, source = build_aviation_scenes(
        [csv_path], [stamp], source_format="csv_snapshot", units=UNITS,
        demand_model=DEMAND_MODEL, seed=51001,
        max_record_age_seconds=60, environment_profile="europe_v1",
        partition_role=adapter_role, window_seconds=60,
        beam_count=1, load_multiplier=1., min_positive_aircraft=1)
    snapshot = source["snapshots"][0]
    observations = [{key: value for key, value in record.items()
                     if key not in ("beam_id", "processing_index", "off_axis_degrees")}
                    for record in snapshot["source_observations"]]
    parsed_ids = [str(row[0]).strip().lower() for row in rows]
    adapter_ids = [str(row["aircraft_id"]).strip().lower() for row in observations]
    if (len(adapter_ids) != len(parsed_ids) or len(set(adapter_ids)) != len(adapter_ids) or
            set(adapter_ids) != set(parsed_ids)):
        missing = sorted(set(parsed_ids) - set(adapter_ids))
        extra = sorted(set(adapter_ids) - set(parsed_ids))
        raise ValueError(f"aircraft adapter changed selected parser population: "
                         f"parsed={len(parsed_ids)} adapter={len(adapter_ids)} "
                         f"missing={missing[:10]} extra={extra[:10]}")
    details = directory / f"source_observations_{index:05d}.jsonl.gz"
    with details.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as text:
            for row in observations:
                text.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    positions = np.asarray([[o["lat"], o["lon"]] for o in observations], dtype=np.float64)
    demands = np.asarray([o["demand_bps"] for o in observations], dtype=np.float64)
    ids = [o["aircraft_id"] for o in observations]
    if not np.isclose(demands.sum(), snapshot["total_demand_bps"], rtol=1e-12, atol=1e-5):
        raise ValueError("per-aircraft demand differs from adapter total")
    meta = {"selected_time_utc": stamp, "source_dates": snapshot["source_dates"],
            "counts": snapshot["counts"], "total_demand_bps": snapshot["total_demand_bps"],
            "source_csv_sha256": digest(csv_path),
            "source_observations_sha256": digest(details),
            "source_archive_sha256": archive_sha,
            "demand_model": DEMAND_MODEL, "load_multiplier": 1.,
            "adapter_beam_count_for_demand_only": 1,
            "adapter_layout_fields_discarded": True}
    return CoverageScene(_business_hash(observations), positions, demands,
                         np.empty((0, 2)), ids, meta, digest(details))


def _publish_prepared(directory, job, preparation, scenes):
    work = directory / "preparation"
    path = work / "inputs.npz"
    sha = digest(path)
    store = Store(job["archive_root"], source_root=job["source_root"])
    record = next((r for r in store.records("scenes")
                   if r.get("source_run_id") == job["run_id"] and
                   r.get("path") == str(path) and r.get("sha256") == sha), None)
    if record is None:
        identifier = new_id("variable_scenes")
        record = {"id": identifier, "record_type": "variable_aircraft_dataset",
                  "name": f"S1-V10 {preparation['role']} {preparation['date_utc']}",
                  "path": str(path), "sha256": sha,
                  "research_stage": "coverage", "schema_version": 5,
                  "partition_role": preparation["role"], "dates": [preparation["date_utc"]],
                  "profile": "europe_v1", "source": load_variable_dataset(path).metadata,
                  "source_run_id": job["run_id"], "created_at": now(),
                  "count": len(scenes),
                  "prepared_npz_sha256": sha}
        store.save_record("scenes", record)
    artifacts = {"prepared_dataset": str(path), "prepared_dataset_sha256": sha,
                 "prepared_dataset_id": record["id"],
                 "raw_observations_count": len(scenes),
                 "raw_archive_sha256": preparation["archive"]["sha256"],
                 "source_preflight_sha256": preparation["source_preflight"]["sha256"],
                 "selection_contract_sha256": preparation["selection_contract"]["sha256"],
                 "selected_times_sha256": preparation["selected_times_sha256"]}
    atomic_json(directory / "artifacts.json", artifacts)
    atomic_json(directory / "results.json", {"summary": {
        "count": len(scenes), "role": preparation["role"], "date": preparation["date_utc"],
        "min_aircraft": min(len(s.demands) for s in scenes),
        "max_aircraft": max(len(s.demands) for s in scenes),
        "zero_aircraft_scenes": sum(not len(s.demands) for s in scenes),
        "dataset_id": record["id"], "dataset_sha256": sha}})
    return record


def run_variable_preparation_job(directory, job, config, *, resume=False, preflight=None):
    directory = Path(directory)
    status_path = directory / "status.json"
    status = read_json(status_path, {})
    prep = job["variable_preparation"]
    work = directory / "preparation"
    status.update(stage="preparing", heartbeat_at=now(), started_at=now(),
                  prepared_scenes=0, preparation_total=0, preflight=preflight)
    atomic_json(status_path, status)
    try:
        pin, contract = _check_source(prep)
        status["preparation_total"] = contract["count"]
        atomic_json(status_path, status)
        if resume:
            attempt = int(job.get("resume_attempt", 0))
            if attempt < 1:
                raise ValueError("same-ID preparation resume lacks durable attempt number")
            history = directory / "recovery_history" / f"attempt_{attempt:03d}"
            if not (history / "status_before_resume.json").is_file():
                raise ValueError("same-ID preparation resume lacks original status receipt")
            failure = directory / "failure.json"
            if failure.is_file():
                failure.replace(history / "failure_active_before_retry.json")
            existing = work / "inputs.npz"
            if existing.is_file():
                try:
                    loaded = load_variable_dataset(existing)
                    if (len(loaded.scenes) == contract["count"] and
                            loaded.metadata.get("source_archive", {}).get("sha256") ==
                            prep["archive"]["sha256"] and
                            loaded.metadata.get("selection_contract_sha256") ==
                            prep["selection_contract"]["sha256"]):
                        record = _publish_prepared(directory, job, prep, loaded.scenes)
                        status.update(stage="completed", ended_at=now(), heartbeat_at=now(),
                                      prepared_scenes=len(loaded.scenes),
                                      prepared_dataset_id=record["id"],
                                      resume_reused_complete_dataset=True)
                        atomic_json(status_path, status)
                        return
                except (OSError, ValueError, KeyError):
                    pass
            if work.exists():
                work.rename(history / "partial_preparation")
        work.mkdir(exist_ok=True)
        raw = work / "raw"
        raw.mkdir(exist_ok=True)
        linked = raw / pin["archive_name"]
        if not linked.exists():
            try:
                linked.hardlink_to(prep["archive"]["path"])
            except OSError:
                linked.symlink_to(prep["archive"]["path"])
        reference = _reference(prep["preparation_reference"]["path"],
                               job["source_root"], work, contract)
        original_write = reference.write_json
        def progress_write(path, value):
            original_write(path, value)
            if isinstance(value, dict):
                status.update(heartbeat_at=now(),
                              source_files_processed=value.get("files_processed", 0),
                              source_trace_records=value.get("source_records", 0))
                atomic_json(status_path, status)
        reference.write_json = progress_write
        rows, candidates = reference.process_day(prep["date_utc"],
                                                  {"status": "processing", "date": prep["date_utc"]})
        if digest(prep["archive"]["path"]) != prep["archive"]["sha256"]:
            raise ValueError("S1-V10 source archive changed during parsing")
        if len(rows) != contract["count"] or len(candidates) != contract["count"]:
            raise ValueError("frozen selected-time count differs from parsed archive")
        scenes = []
        for index, stamp in enumerate(contract["selected_times_utc"]):
            if candidates[index]["selected_time_utc"] != stamp:
                raise ValueError("parsed UTC time differs from frozen contract")
            scenes.append(_scene_from_rows(work, index, stamp, rows[index],
                                           prep["role"], prep["archive"]["sha256"]))
            if prep["role"] == "sequence":
                scenes[-1].metadata["sequence_id"] = f"{prep['date_utc']}_segment_{index // 6:02d}"
                scenes[-1].metadata["sequence_step"] = index % 6
            status.update(prepared_scenes=index + 1, heartbeat_at=now())
            atomic_json(status_path, status)
        metadata = {"format_version": "variable_aircraft_v1", "partition_role": prep["role"],
                    "source_dates": [prep["date_utc"]], "source_archive": prep["archive"],
                    "source_preflight_sha256": prep["source_preflight"]["sha256"],
                    "selection_contract_sha256": prep["selection_contract"]["sha256"],
                    "preparation_reference_sha256": prep["preparation_reference"]["sha256"],
                    "demand_model": DEMAND_MODEL, "seed": 51001,
                    "demand_multipliers_applied": [1.],
                    "planned_evaluation_multipliers": [.75, 1., 1.25]}
        save_variable_dataset(CoverageDataset(scenes, metadata), work / "inputs.npz")
        record = _publish_prepared(directory, job, prep, scenes)
        status.update(stage="completed", ended_at=now(), heartbeat_at=now(),
                      prepared_scenes=len(scenes), prepared_dataset_id=record["id"])
        atomic_json(status_path, status)
    except BaseException as error:
        status.update(stage="failed", ended_at=now(), heartbeat_at=now(),
                      error=repr(error))
        atomic_json(status_path, status)
        (directory / "failure.json").write_text(json.dumps({"error": repr(error),
            "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
