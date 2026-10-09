"""Durable workbench preparation of S1-D12A official aircraft scenes/teacher."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import traceback

from satellite_rl.coverage.scenes import CoverageDataset
from satellite_rl.coverage.variable_dataset import load_variable_dataset, save_variable_dataset
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import Store, digest, new_id, now, read_json


def _download(directory, prep, status):
    """One shared remote archive per official day; resume by byte range."""
    from .coverage_drl_r04 import PARENT
    source = PARENT / "source" / prep["role"] / prep["date_utc"]
    source.mkdir(parents=True, exist_ok=True)
    archive = source / prep["archive_name"]
    partial = source / (prep["archive_name"] + ".part")
    if archive.is_file():
        if archive.stat().st_size != prep["expected_bytes"]:
            raise ValueError("existing official archive byte count differs")
        return archive, digest(archive)
    # User quota is unavailable on these instances. This group enforces a
    # conservative local soft cap instead of reading global NFS free space as quota.
    size = subprocess.run(["du", "-sb", str(PARENT)], capture_output=True,
                          text=True, timeout=90, check=True)
    used = int(size.stdout.split()[0])
    if used + prep["expected_bytes"] > 24 * (1 << 30):
        raise ValueError("S1-D12A 24 GiB shared-source soft cap would be exceeded")
    for attempt in range(1, 9):
        command = ["curl", "--location", "--fail", "--silent", "--show-error",
                   "--connect-timeout", "20", "--max-time", "1800",
                   "--continue-at", "-", "--output", str(partial), prep["archive_url"]]
        result = subprocess.run(command, capture_output=True, text=True, timeout=1830)
        bytes_now = partial.stat().st_size if partial.is_file() else 0
        status.update(heartbeat_at=now(), download_attempt=attempt,
                      downloaded_bytes=bytes_now, expected_bytes=prep["expected_bytes"],
                      curl_exit=result.returncode, curl_stderr_tail=result.stderr[-500:])
        atomic_json(directory / "status.json", status)
        if result.returncode == 0 and bytes_now == prep["expected_bytes"]:
            partial.replace(archive)
            return archive, digest(archive)
        if bytes_now > prep["expected_bytes"] or attempt == 8:
            raise RuntimeError("official archive download incomplete or oversized")
    raise AssertionError("unreachable")


def _dataset(directory, job, config, resume, preflight):
    from .coverage_drl_r04 import GROUP
    from .variable_preparation import _reference, _scene_from_rows, DEMAND_MODEL
    prep = job["dataset_preparation"]
    files = prep["files"]
    for item in files.values():
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("frozen official source metadata SHA differs")
    pin = read_json(files["source_preflight.json"]["path"], {})
    contract = read_json(files["selection_contract.json"]["path"], {})
    if (pin.get("group") != GROUP or contract.get("group") != GROUP or
            pin.get("date_utc") != prep["date_utc"] or
            contract.get("date_utc") != prep["date_utc"] or
            pin.get("asset_id") != prep["asset_id"] or
            pin.get("expected_bytes") != prep["expected_bytes"] or
            pin.get("selection_contract_sha256") != files["selection_contract.json"]["sha256"]):
        raise ValueError("official release and UTC selection no longer match job")
    status = read_json(directory / "status.json", {})
    status.update(stage="preparing", started_at=status.get("started_at", now()),
                  heartbeat_at=now(), preparation_total=contract["count"],
                  preflight=preflight)
    atomic_json(directory / "status.json", status)
    try:
        archive, archive_sha = _download(directory, prep, status)
        work = directory / "preparation"
        if resume:
            old = read_json(work / "prepared.json", {})
            existing = work / "inputs.npz"
            if (existing.is_file() and old.get("archive_sha256") == archive_sha and
                    old.get("selection_contract_sha256") == files["selection_contract.json"]["sha256"]):
                loaded = load_variable_dataset(existing)
                if len(loaded.scenes) == contract["count"]:
                    return _publish(directory, job, prep, loaded.scenes, archive_sha)
            history = directory / "recovery_history" / f"attempt_{int(job.get('resume_attempt', 0)):03d}"
            if not (history / "status_before_resume.json").is_file():
                raise ValueError("same-ID preparation retry lacks saved prior state")
            if work.exists():
                work.rename(history / "partial_preparation")
        elif work.exists():
            raise FileExistsError("preparation directory exists; use same-ID resume")
        work.mkdir(parents=True)
        raw = work / "raw"
        raw.mkdir()
        linked = raw / prep["archive_name"]
        try:
            linked.hardlink_to(archive)
        except OSError:
            linked.symlink_to(archive)
        reference = _reference(files["parser"]["path"], job["source_root"], work, contract)
        rows, candidates = reference.process_day(prep["date_utc"],
                                                   {"status": "processing", "date": prep["date_utc"]})
        if (len(rows) != contract["count"] or len(candidates) != contract["count"] or
                digest(archive) != archive_sha):
            raise ValueError("parsed aircraft count or official archive SHA changed")
        scenes = []
        for index, stamp in enumerate(contract["selected_times_utc"]):
            if candidates[index]["selected_time_utc"] != stamp:
                raise ValueError("parsed UTC time differs from frozen selection")
            scene = _scene_from_rows(work, index, stamp, rows[index], prep["role"], archive_sha)
            scene.metadata.update(selection_contract_sha256=files["selection_contract.json"]["sha256"],
                                  source_preflight_sha256=files["source_preflight.json"]["sha256"])
            scenes.append(scene)
            status.update(prepared_scenes=index + 1, heartbeat_at=now())
            atomic_json(directory / "status.json", status)
        metadata = {"partition_role": prep["role"], "source_dates": [prep["date_utc"]],
                    "source_archive": {"path": str(archive), "sha256": archive_sha,
                                       "bytes": prep["expected_bytes"], "asset_id": prep["asset_id"]},
                    "source_preflight_sha256": files["source_preflight.json"]["sha256"],
                    "selection_contract_sha256": files["selection_contract.json"]["sha256"],
                    "preparation_reference_sha256": files["parser"]["sha256"],
                    "demand_model": DEMAND_MODEL, "seed": 51001,
                    "demand_multipliers_applied": [1.0],
                    "planned_evaluation_multipliers": [.75, 1., 1.25]}
        save_variable_dataset(CoverageDataset(scenes, metadata), work / "inputs.npz")
        atomic_json(work / "prepared.json", {"archive_sha256": archive_sha,
                    "selection_contract_sha256": files["selection_contract.json"]["sha256"],
                    "prepared_sha256": digest(work / "inputs.npz")})
        return _publish(directory, job, prep, scenes, archive_sha)
    except BaseException as error:
        status.update(stage="failed", ended_at=now(), heartbeat_at=now(), error=repr(error))
        atomic_json(directory / "status.json", status)
        atomic_json(directory / "failure.json", {"error": repr(error),
                    "traceback": traceback.format_exc()})
        raise


def _publish(directory, job, prep, scenes, archive_sha):
    path = directory / "preparation" / "inputs.npz"
    sha = digest(path)
    store = Store(job["archive_root"], source_root=job["source_root"])
    record = next((r for r in store.records("scenes")
                   if r.get("source_run_id") == job["run_id"] and r.get("sha256") == sha), None)
    if record is None:
        record = {"id": new_id("drl_r04_scenes"),
                  "name": f"S1-D12A {prep['role']} {prep['date_utc']}",
                  "path": str(path), "sha256": sha, "schema_version": 6,
                  "research_stage": "coverage", "record_type": "drl_r04_aircraft_dataset",
                  "profile": "europe_v1", "partition_role": prep["role"],
                  "dates": [prep["date_utc"]], "count": len(scenes),
                  "source": load_variable_dataset(path).metadata,
                  "source_run_id": job["run_id"], "created_at": now()}
        store.save_record("scenes", record)
    atomic_json(directory / "artifacts.json", {"prepared_dataset": str(path),
                "prepared_dataset_sha256": sha, "prepared_dataset_id": record["id"],
                "raw_archive_sha256": archive_sha,
                "source_preflight_sha256": prep["files"]["source_preflight.json"]["sha256"],
                "selection_contract_sha256": prep["files"]["selection_contract.json"]["sha256"]})
    atomic_json(directory / "results.json", {"summary": {"count": len(scenes),
                "role": prep["role"], "date": prep["date_utc"],
                "dataset_id": record["id"], "dataset_sha256": sha}})
    status = read_json(directory / "status.json", {})
    status.update(stage="completed", ended_at=now(), heartbeat_at=now(),
                  prepared_scenes=len(scenes), prepared_dataset_id=record["id"])
    atomic_json(directory / "status.json", status)
    return record


def run_drl_r04_preparation_job(directory, job, config, *, resume=False, preflight=None):
    if job.get("kind") != "prepare":
        raise ValueError("unsupported S1-D12A preparation mode")
    directory = Path(directory)
    if job.get("preparation_mode") == "dataset":
        return _dataset(directory, job, config, resume, preflight)
    if job.get("preparation_mode") != "teacher":
        raise ValueError("unsupported S1-D12A preparation mode")
    from satellite_rl.coverage.drl_r04_runner import run_drl_r04_job
    return run_drl_r04_job(directory, job, config, resume=resume, preflight=preflight)
