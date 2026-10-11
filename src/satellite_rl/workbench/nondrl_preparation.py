"""S1-ND12 official daily archive download and frozen UTC scene preparation."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import time

from satellite_rl.coverage.scenes import CoverageDataset
from satellite_rl.coverage.variable_dataset import load_variable_dataset, save_variable_dataset
from satellite_rl.training.run_monitor import atomic_json
from .storage import Store, digest, new_id, now, read_json
from .coverage_nondrl import GROUP, PARENT, _pin, partition_contract


def submit_dataset(application, body, config, port):
    role, day = body.get("dataset_role"), body.get("source_date")
    manifest_path = _pin(body["partition_manifest_path"], body["partition_manifest_sha256"])
    manifest = partition_contract(manifest_path, body["partition_manifest_sha256"])
    if role not in ("validation", "test") or day not in manifest["roles"][role]["dates"]:
        raise ValueError("new source role/date differs from partition manifest")
    root = PARENT / "source" / role / day
    pin_path, contract_path = root / "source_preflight.json", root / "selection_contract.json"
    pin, contract = read_json(pin_path, {}), read_json(contract_path, {})
    if (pin.get("group") != GROUP or contract.get("group") != GROUP or
            pin.get("date_utc") != day or contract.get("date_utc") != day or
            pin.get("role") != role or contract.get("role") != role or
            pin.get("selection_contract_sha256") != digest(contract_path) or
            contract.get("no_outcome_based_replacement") is not True or
            type(contract.get("count")) is not int or
            len(contract.get("selected_times_utc", [])) != contract["count"] or
            len(set(contract["selected_times_utc"])) != contract["count"] or
            any(not stamp.startswith(day) for stamp in contract["selected_times_utc"])):
        raise ValueError("official UTC selection lineage invalid")
    files = {}
    for name, key in (("source_preflight.json", None), ("selection_contract.json", "selection_contract_sha256"),
                      ("release_api.json", "release_api_sha256"), ("PREFERRED_RELEASES.txt", "preferred_sha256"),
                      ("LICENSE-ODbL-official.txt", "license_sha256")):
        path = root / name
        _pin(path, digest(path) if key is None else pin.get(key))
        files[name] = {"path": str(path), "sha256": digest(path)}
    release = read_json(root / "release_api.json", {})
    assets = [a for a in release.get("assets", []) if a.get("name") == pin.get("archive_name")]
    if (len(assets) != 1 or assets[0].get("id") != pin.get("asset_id") or
            assets[0].get("size") != pin.get("expected_bytes") or
            assets[0].get("browser_download_url") != pin.get("archive_url") or
            pin.get("archive_url") not in (root / "PREFERRED_RELEASES.txt").read_text(encoding="utf-8")):
        raise ValueError("official preferred release asset differs")
    parser = Path(body["parser_path"]).resolve()
    _pin(parser, body["parser_sha256"])
    files["parser"] = {"path": str(parser), "sha256": body["parser_sha256"]}
    files["partition_manifest.json"] = {"path": str(manifest_path), "sha256": body["partition_manifest_sha256"]}
    config["datasets"] = {}
    prep = {"role": role, "date_utc": day, "files": files,
            **{key: pin[key] for key in ("asset_id", "expected_bytes", "archive_name", "archive_url")}}
    extra = {"experiment_id": GROUP, "research_stage": "coverage", "variant": "dataset",
             "experiment_role": "formal", "preparation_mode": "dataset", "dataset_role": role,
             "source_date": day, "dataset_preparation": prep, "execution_device": "cpu",
             "execution_target": f"u2024110224@10.160.17.102:{port}"}
    return application.store.create_job(config, kind="prepare", request_id=body.get("request_id"), extra=extra)


def freeze_preparation(extra, destination):
    from copy import deepcopy
    result = deepcopy(extra)
    if not result.get("dataset_preparation"):
        return result
    root = Path(destination) / "official_source"
    root.mkdir(parents=True)
    for name, item in result["dataset_preparation"]["files"].items():
        path = _pin(item["path"], item["sha256"])
        target = root / ("parser.py" if name == "parser" else name)
        shutil.copyfile(path, target)
        item.update(source_path=str(path), path=str(target))
    return result


def _download(directory, prep, progress, stopped):
    root = PARENT / "source" / prep["role"] / prep["date_utc"]
    root.mkdir(parents=True, exist_ok=True)
    archive = root / prep["archive_name"]
    if archive.is_file():
        if archive.stat().st_size != prep["expected_bytes"]:
            raise ValueError("existing official archive size mismatch")
        return archive
    lock = root / "download.lock"
    # Exclusive shared-source lock prevents two lanes writing the same byte stream.
    import os
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        owner = lock.read_text(encoding="utf-8").splitlines()
        if len(owner) < 2 or owner[1] != str(directory):
            raise ValueError("another lane owns shared source download; inspect owner before retry")
        # Same-ID worker is already serialized by its OS lock; an exited attempt
        # may leave this shared-source marker behind after a native crash.
        lock.unlink()
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, f"{os.getpid()}\n{directory}".encode())
    os.close(descriptor)
    partial = archive.with_name(archive.name + ".part")
    try:
        for attempt in range(8):
            if stopped():
                return None
            with (directory / "download.log").open("ab") as stream:
                child = subprocess.Popen(["curl", "--location", "--fail", "--silent", "--show-error",
                    "--connect-timeout", "20", "--max-time", "1800", "--continue-at", "-",
                    "--output", str(partial), prep["archive_url"]], stdout=stream, stderr=stream)
                while child.poll() is None:
                    progress(download_attempt=attempt + 1, downloaded_bytes=partial.stat().st_size if partial.exists() else 0,
                             expected_bytes=prep["expected_bytes"])
                    if stopped():
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        return None
                    time.sleep(2)
            if child.returncode == 0 and partial.stat().st_size == prep["expected_bytes"]:
                partial.replace(archive)
                return archive
        raise RuntimeError("official download failed; partial archive retained for same-ID resume")
    finally:
        lock.unlink(missing_ok=True)


def run_dataset(directory, job, config, *, resume, progress, stopped):
    from .variable_preparation import _reference, _scene_from_rows, DEMAND_MODEL
    prep = job["dataset_preparation"]
    for item in prep["files"].values():
        _pin(item["path"], item["sha256"])
    contract = read_json(prep["files"]["selection_contract.json"]["path"])
    progress(stage="preparing", preparation_total=contract["count"])
    archive = _download(directory, prep, progress, stopped)
    if archive is None:
        return {"stopped": True}
    archive_sha = digest(archive)
    work = directory / "preparation"
    work.mkdir(exist_ok=True)
    prepared = work / "inputs.npz"
    if not prepared.is_file():
        raw = work / "raw"
        raw.mkdir(exist_ok=True)
        linked = raw / archive.name
        if not linked.exists():
            try:
                linked.hardlink_to(archive)
            except OSError:
                linked.symlink_to(archive)
        reference = _reference(prep["files"]["parser"]["path"], job["source_root"], work, contract)
        rows, candidates = reference.process_day(prep["date_utc"], {"status": "processing", "date": prep["date_utc"]})
        if len(rows) != contract["count"] or len(candidates) != contract["count"]:
            raise ValueError("official parser omitted selected UTC snapshots")
        scenes = []
        for index, stamp in enumerate(contract["selected_times_utc"]):
            if stopped():
                return {"stopped": True}
            if candidates[index]["selected_time_utc"] != stamp:
                raise ValueError("parsed selected UTC differs")
            scene = _scene_from_rows(work, index, stamp, rows[index], prep["role"], archive_sha)
            scene.metadata.update(selection_contract_sha256=prep["files"]["selection_contract.json"]["sha256"],
                                  source_preflight_sha256=prep["files"]["source_preflight.json"]["sha256"])
            scenes.append(scene)
            progress(prepared_scenes=index + 1)
        metadata = {"partition_role": prep["role"], "source_dates": [prep["date_utc"]],
                    "source_archive_sha256": archive_sha, "demand_model": DEMAND_MODEL,
                    "seed": 51001, "demand_multipliers_applied": [1.]}
        save_variable_dataset(CoverageDataset(scenes, metadata), prepared)
    loaded = load_variable_dataset(prepared)
    if len(loaded.scenes) != contract["count"]:
        raise ValueError("prepared scene count differs")
    store = Store(job["archive_root"], source_root=job["source_root"])
    record = next((r for r in store.records("scenes") if r.get("source_run_id") == job["run_id"]), None)
    if record is None:
        record = {"id": new_id("nondrl_scenes"), "name": f"S1-ND12 {prep['role']} {prep['date_utc']}",
                  "path": str(prepared), "sha256": digest(prepared), "schema_version": 7,
                  "research_stage": "coverage", "record_type": "nondrl_aircraft_dataset",
                  "profile": "europe_v1", "partition_role": prep["role"], "dates": [prep["date_utc"]],
                  "count": len(loaded.scenes), "source": loaded.metadata, "source_run_id": job["run_id"],
                  "partition_manifest_path": prep["files"]["partition_manifest.json"]["path"],
                  "partition_manifest_sha256": prep["files"]["partition_manifest.json"]["sha256"], "created_at": now()}
        store.save_record("scenes", record)
    atomic_json(directory / "artifacts.json", {"prepared_dataset": str(prepared), "prepared_dataset_sha256": digest(prepared),
                "prepared_dataset_id": record["id"], "raw_archive_sha256": archive_sha})
    return {"summary": {"count": len(loaded.scenes), "dataset_id": record["id"], "role": prep["role"]}}
