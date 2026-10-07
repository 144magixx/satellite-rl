"""Frozen input preparation for S2-P19 development, validation and final sets.

The workbench owns long-running execution. This module accepts only a typed
record of already frozen mother CSVs, verifies every digest, and emits scenes
under its job directory. It never downloads data, loads a policy or evaluates
an allocation.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from typing import Callable
import urllib.request

import numpy as np

from satellite_rl.envs.regions import elevation_degrees, get_profile, profile_metadata
from satellite_rl.workbench.aviation import CSV_UNITS, build_aviation_scenes


SEED = 51001
LOADS = (0.75, 1.25)
DEVELOPMENT_COUNTS = (50, 100, 125, 150, 175, 200)
VALIDATION_COUNTS = (50, 80, 100, 120, 160, 200)
FINAL_COUNTS = (50, 60, 75, 90, 100, 125, 140, 150, 175, 180, 200)
FINAL_DATES = ("2024-02-01", "2024-02-02")
ALLOWED_TYPES = frozenset(("mother_sources", "dev_validation", "final"))
MAX_CSV_BYTES = 8 * 1024 * 1024
MAX_RECORD_BYTES = 16 * 1024 * 1024
OFFICIAL_ASSETS = (
    ("2024-02-01", 149304672, 1784412160),
    ("2024-02-02", 149522532, 1714810880),
)
RECOVERY_PARENT_RUN_ID = "prepare_prepare_20261007_145623_5e8713ea"
RECOVERY_SMALL_KEYS = ("job", "config", "status", "failure", "feb01_download",
                       "feb02_download", "feb02_stderr", "feb01_eligibility", "feb01_selection")
RECOVERY_KEYS = frozenset(("parent_run_id", "feb01_archive", "feb01_csv_files",
                           "frozen_evidence", *RECOVERY_SMALL_KEYS))
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("source record must be a JSON object")
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def _source_reference(value: object, maximum: int) -> dict:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("each source reference requires exactly path and sha256")
    path = Path(value["path"])
    digest = value["sha256"]
    if not path.is_absolute() or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("source path must be absolute with a SHA256 digest")
    if not path.is_file() or path.stat().st_size > maximum or _sha(path) != digest:
        raise ValueError("source file missing, oversized or SHA changed: " + str(path))
    return {"path": str(path.resolve()), "sha256": digest}


def _recovery_expected_paths(parent: Path, selection: dict) -> dict:
    raw = parent / "mother_sources" / "raw"
    day = parent / "mother_sources" / "snapshots" / "final" / "2024-02-01"
    expected = {"job": parent / "job.json", "config": parent / "config.json",
                "status": parent / "status.json", "failure": parent / "failure.json",
                "feb01_archive": raw / "v2024.02.01-planes-readsb-staging-0.tar",
                "feb01_download": raw / "2024-02-01_download.json",
                "feb02_download": raw / "2024-02-02_download.json",
                "feb02_stderr": raw / "2024-02-02_curl.stderr",
                "feb01_eligibility": day / "eligibility.json",
                "feb01_selection": day / "selection_contract.json"}
    expected["feb01_csv_files"] = [day / f"t{index:03d}.csv"
                                     for index in selection["selected_indices"]]
    return expected


def _validate_recovery_source(value: object, source_record_sha: str,
                              official_assets: list[dict]) -> dict:
    """Only the one failed, SHA-bound S2-P19 Feb01/02 preparation is reusable."""
    if not isinstance(value, dict) or set(value) - RECOVERY_KEYS or value.get("parent_run_id") != RECOVERY_PARENT_RUN_ID:
        raise ValueError("recovery_source只能引用登记的S2-P19失败母场景任务")
    if any(key not in value for key in (*RECOVERY_SMALL_KEYS, "feb01_archive", "feb01_csv_files")):
        raise ValueError("recovery_source缺旧任务/原始资产/二十母证据")
    refs = {key: _source_reference(value[key], MAX_RECORD_BYTES) for key in RECOVERY_SMALL_KEYS}
    archive = _source_reference(value["feb01_archive"], MAX_ARCHIVE_BYTES)
    csv_refs = value["feb01_csv_files"]
    if not isinstance(csv_refs, list) or len(csv_refs) != 20:
        raise ValueError("旧Feb01须有恰好20个母场景CSV来源")
    csv_refs = [_source_reference(row, MAX_CSV_BYTES) for row in csv_refs]
    parent = Path(refs["job"]["path"]).parent
    job = _json(Path(refs["job"]["path"]))
    archive_root = Path(job.get("archive_root", "")).resolve()
    if (parent != archive_root / "outputs" / "workbench" / "jobs" / RECOVERY_PARENT_RUN_ID or
            job.get("run_id") != RECOVERY_PARENT_RUN_ID or job.get("kind") != "prepare" or
            job.get("experiment_id") != "S2-P19" or
            job.get("research_stage") != "s2p19_preparation" or
            job.get("request_id") != "s2p19-feb01-feb02-mothers-20424"):
        raise ValueError("旧母场景任务身份、工作台归属或请求编号不符")
    old_config = _json(Path(refs["config"]["path"]))
    old_prep = old_config.get("preparation") or {}
    if (old_prep.get("preparation_type") != "mother_sources" or
            (old_prep.get("source_record") or {}).get("sha256") != source_record_sha or
            old_prep.get("official_assets") != official_assets or
            old_config.get("research_stage") != "s2p19_preparation"):
        raise ValueError("旧失败任务的科学来源、日期或官方资产与新任务不同")
    status, failure = _json(Path(refs["status"]["path"])), _json(Path(refs["failure"]["path"]))
    if (status.get("stage") != "failed" or failure.get("run_id") != RECOVERY_PARENT_RUN_ID or
            "eight resumable attempts" not in str(failure.get("error", ""))):
        raise ValueError("旧任务不是已记录八次失败的母场景准备")
    feb01 = _json(Path(refs["feb01_download"]["path"]))
    feb02 = _json(Path(refs["feb02_download"]["path"]))
    if (feb01.get("status") != "completed" or feb01.get("asset") != official_assets[0] or
            Path(feb01.get("path", "")).resolve() != Path(archive["path"]) or
            feb01.get("sha256") != archive["sha256"] or
            feb01.get("bytes") != OFFICIAL_ASSETS[0][2] or
            Path(archive["path"]).stat().st_size != OFFICIAL_ASSETS[0][2]):
        raise ValueError("旧Feb01官方归档不完整或SHA/大小与回执不符")
    if (feb02.get("asset") != official_assets[1] or len(feb02.get("attempts") or []) != 8 or
            feb02.get("partial_bytes") != 0 or
            any(item.get("partial_bytes") != 0 for item in feb02["attempts"])):
        raise ValueError("旧Feb02八次失败及0字节证据不完整")
    selection = _json(Path(refs["feb01_selection"]["path"]))
    eligibility = _json(Path(refs["feb01_eligibility"]["path"]))
    if (selection.get("date_utc") != "2024-02-01" or
            selection.get("archive_sha256") != archive["sha256"] or
            selection.get("eligibility_sha256") != refs["feb01_eligibility"]["sha256"] or
            eligibility.get("date_utc") != "2024-02-01" or
            eligibility.get("archive_sha256") != archive["sha256"] or
            len(selection.get("selected_indices") or []) != 20 or
            len(selection.get("selected_times_utc") or []) != 20 or
            len(set(selection["selected_times_utc"])) != 20 or
            any(not stamp.startswith("2024-02-01T") for stamp in selection["selected_times_utc"])):
        raise ValueError("旧Feb01二十母选择/资格/日期或上游归档来源不符")
    expected = _recovery_expected_paths(parent, selection)
    for key, ref in {**refs, "feb01_archive": archive}.items():
        if Path(ref["path"]) != expected[key].resolve():
            raise ValueError("恢复证据路径不在旧任务的预期位置：" + key)
    if [Path(row["path"]) for row in csv_refs] != [path.resolve() for path in expected["feb01_csv_files"]]:
        raise ValueError("旧Feb01母场景CSV文件名、顺序或路径不符")
    candidates = eligibility.get("candidates") or []
    if len(candidates) != 288:
        raise ValueError("旧Feb01资格网格不完整")
    indices, substitutions = _selected_indices(candidates)
    if indices != selection["selected_indices"] or substitutions != selection.get("substitutions"):
        raise ValueError("旧Feb01母场景选择与既定前向替代规则不同")
    for index, stamp in zip(indices, selection["selected_times_utc"], strict=True):
        item = candidates[index]
        if (item.get("selected_time_utc") != stamp or item.get("positive_demand_aircraft", 0) < 200 or
                item.get("unique_positions", 0) < 200):
            raise ValueError("旧Feb01母时刻不是合格且同日的冻结场景")
    normalized = {"parent_run_id": RECOVERY_PARENT_RUN_ID, **refs,
                  "feb01_archive": archive, "feb01_csv_files": csv_refs}
    frozen = value.get("frozen_evidence")
    if frozen is not None:
        if not isinstance(frozen, dict) or set(frozen) != set(RECOVERY_SMALL_KEYS) | {"feb01_csv_files"}:
            raise ValueError("冻结的恢复小证据清单不完整")
        for key in RECOVERY_SMALL_KEYS:
            copied = _source_reference(frozen[key], MAX_RECORD_BYTES)
            if copied["sha256"] != refs[key]["sha256"]:
                raise ValueError("冻结恢复证据SHA不同：" + key)
        copied_csv = frozen["feb01_csv_files"]
        if not isinstance(copied_csv, list) or len(copied_csv) != 20:
            raise ValueError("冻结Feb01 CSV证据数不足20")
        for old, item in zip(csv_refs, copied_csv, strict=True):
            if _source_reference(item, MAX_CSV_BYTES)["sha256"] != old["sha256"]:
                raise ValueError("冻结Feb01 CSV证据SHA不同")
        normalized["frozen_evidence"] = frozen
    return normalized


def _extract_mothers(record: dict, kind: str) -> list[dict]:
    source = record.get("source") or {}
    if source.get("seed") != SEED or record.get("profile") != "europe_v1":
        raise ValueError("source seed/profile differs from frozen S2-P19 contract")
    model = source.get("demand_model") or {}
    if model.get("window_seconds") != 60:
        raise ValueError("source demand window differs")
    mothers = []
    if kind == "dev_validation":
        if record.get("count") != 144 or source.get("mother_scene_count") != 12:
            raise ValueError("Jan12 source must be the twelve-mother, 144-scene P17 record")
        rows = source.get("mother_scene_inputs") or []
        if len(rows) != 12:
            raise ValueError("Jan12 source has incomplete mother CSV bindings")
        for item in rows:
            mothers.append({"selected_time_utc": item["selected_time_utc"],
                            "path": item["path"], "sha256": item["sha256"]})
        expected_dates = {"2024-01-12": 12}
    else:
        if record.get("count") != 40:
            raise ValueError("final source must hold exactly forty mothers")
        files, snapshots = source.get("files") or [], source.get("snapshots") or []
        if len(files) != 40 or len(snapshots) != 40:
            raise ValueError("final source CSV and snapshot counts differ")
        for file_item, snapshot in zip(files, snapshots, strict=True):
            if file_item["selected_time_utc"] != snapshot["selected_time_utc"]:
                raise ValueError("final mother CSV/time binding differs")
            counts = snapshot.get("counts") or {}
            selected = counts.get("selected_aircraft")
            zero = counts.get("zero_demand_aircraft")
            if not isinstance(selected, int) or not isinstance(zero, int) or selected - zero < 200:
                raise ValueError("final mother lacks >=200 positive-demand aircraft qualification")
            mothers.append({"selected_time_utc": snapshot["selected_time_utc"],
                            "path": file_item["path"], "sha256": file_item["sha256"]})
        expected_dates = {date: 20 for date in FINAL_DATES}
    dates = Counter(item["selected_time_utc"][:10] for item in mothers)
    if dict(dates) != expected_dates:
        raise ValueError("source mother dates/counts differ from planned split")
    times = [item["selected_time_utc"] for item in mothers]
    if len(set(times)) != len(times):
        raise ValueError("duplicate mother timestamp")
    return sorted(mothers, key=lambda item: item["selected_time_utc"])


def validate_preparation_config(raw: dict) -> dict:
    """Validate a typed preparation request and all referenced source bytes.

    This is read-only. It accepts an original source record or a config frozen
    by :func:`freeze_preparation_config`.
    """
    if not isinstance(raw, dict) or set(raw) - {
        "preparation_type", "source_record", "seed", "environment_profile",
        "mother_sources", "frozen_input_sources", "original_source_record",
        "official_assets", "recovery_source",
    }:
        raise ValueError("unknown S2-P19 preparation fields")
    kind = raw.get("preparation_type")
    if kind not in ALLOWED_TYPES or raw.get("seed") != SEED or raw.get("environment_profile") != "europe_v1":
        raise ValueError("unsupported preparation type, seed or profile")
    record_ref = _source_reference(raw.get("source_record"), MAX_RECORD_BYTES)
    record = _json(Path(record_ref["path"]))
    if kind == "mother_sources":
        source = record.get("source") or {}
        if (source.get("seed") != SEED or record.get("profile") != "europe_v1"
                or (source.get("demand_model") or {}).get("window_seconds") != 60):
            raise ValueError("mother source demand model/profile differs")
        supplied_assets = raw.get("official_assets")
        if not isinstance(supplied_assets, list) or len(supplied_assets) != 2:
            raise ValueError("mother_sources needs exactly two pinned official assets")
        assets = []
        for supplied, (day, asset_id, size) in zip(supplied_assets, OFFICIAL_ASSETS, strict=True):
            tag = f"v{day.replace('-', '.')}-planes-readsb-staging-0"
            expected = {"date_utc": day, "release_tag": tag, "asset_id": asset_id,
                        "asset_name": tag + ".tar",
                        "url": f"https://github.com/adsblol/globe_history_2024/releases/download/{tag}/{tag}.tar",
                        "expected_bytes": size}
            if supplied != expected:
                raise ValueError("official asset contract differs from pinned release")
            assets.append(expected)
        result = {"preparation_type": kind, "source_record": record_ref,
                  "seed": SEED, "environment_profile": "europe_v1",
                  "official_assets": assets}
        if raw.get("recovery_source") is not None:
            result["recovery_source"] = _validate_recovery_source(
                raw["recovery_source"], record_ref["sha256"], assets)
        if raw.get("frozen_input_sources"):
            result["frozen_input_sources"] = True
            result["original_source_record"] = raw.get("original_source_record")
        return result
    if raw.get("recovery_source") is not None:
        raise ValueError("仅mother_sources允许同资产失败恢复")
    extracted = _extract_mothers(record, kind)
    if raw.get("mother_sources") is None:
        mother_sources = [{"selected_time_utc": row["selected_time_utc"],
                           **_source_reference({"path": row["path"], "sha256": row["sha256"]}, MAX_CSV_BYTES)}
                          for row in extracted]
    else:
        supplied = raw["mother_sources"]
        if not isinstance(supplied, list) or len(supplied) != len(extracted):
            raise ValueError("frozen mother source count differs")
        mother_sources = []
        for expected, item in zip(extracted, supplied, strict=True):
            if not isinstance(item, dict) or set(item) != {"selected_time_utc", "path", "sha256"}:
                raise ValueError("malformed frozen mother source")
            if item["selected_time_utc"] != expected["selected_time_utc"] or item["sha256"] != expected["sha256"]:
                raise ValueError("frozen mother source identity differs")
            mother_sources.append({"selected_time_utc": item["selected_time_utc"],
                                   **_source_reference({"path": item["path"], "sha256": item["sha256"]},
                                                       MAX_CSV_BYTES)})
    normalized = {"preparation_type": kind, "source_record": record_ref,
                  "seed": SEED, "environment_profile": "europe_v1",
                  "mother_sources": mother_sources}
    if raw.get("frozen_input_sources"):
        normalized["frozen_input_sources"] = True
        normalized["original_source_record"] = raw.get("original_source_record")
    return normalized


def build_recovery_source(parent_jobdir: Path) -> dict:
    """Build SHA references from the one failed task for the typed workbench API.

    This is read-only. The caller must still pass the result through
    ``validate_preparation_config`` with the new canonical mother request.
    """
    parent = Path(parent_jobdir).resolve(strict=True)
    if parent.name != RECOVERY_PARENT_RUN_ID or parent.parent.name != "jobs":
        raise ValueError("恢复来源不是原S2-P19失败母场景工作台任务")
    selection = _json(parent / "mother_sources/snapshots/final/2024-02-01/selection_contract.json")
    expected = _recovery_expected_paths(parent, selection)
    result = {"parent_run_id": RECOVERY_PARENT_RUN_ID}
    for key in RECOVERY_SMALL_KEYS:
        path = expected[key]
        result[key] = {"path": str(path), "sha256": _sha(path)}
    archive = expected["feb01_archive"]
    result["feb01_archive"] = {"path": str(archive), "sha256": _sha(archive)}
    result["feb01_csv_files"] = [{"path": str(path), "sha256": _sha(path)}
                                  for path in expected["feb01_csv_files"]]
    return result


def input_references(config: dict) -> list[dict]:
    normalized = validate_preparation_config(config)
    if normalized["preparation_type"] == "mother_sources":
        return [normalized["source_record"]]
    return [normalized["source_record"], *(
        {"path": item["path"], "sha256": item["sha256"]}
        for item in normalized["mother_sources"])]


def freeze_preparation_config(raw: dict, destination: Path) -> dict:
    """Copy small JSON/CSV sources into a task's immutable input_sources dir."""
    if raw.get("frozen_input_sources"):
        raise ValueError("a frozen preparation config cannot be frozen twice")
    normalized = validate_preparation_config(raw)
    destination = Path(destination).resolve()
    recovery = normalized.get("recovery_source")
    if recovery is not None:
        old_jobs = Path(recovery["job"]["path"]).parent.parent
        new_jobs = destination.parent.parent
        if old_jobs != new_jobs or old_jobs.name != "jobs":
            raise ValueError("恢复源任务必须属于同一工作台archive的jobs目录")
    destination.mkdir(parents=True, exist_ok=True)

    def copy(source: dict, name: str) -> dict:
        target = destination / name
        if target.exists():
            if _sha(target) != source["sha256"]:
                raise FileExistsError("existing frozen input differs: " + str(target))
        else:
            pending = target.with_name(target.name + ".tmp")
            shutil.copy2(source["path"], pending)
            if _sha(pending) != source["sha256"]:
                pending.unlink()
                raise RuntimeError("copy digest differs")
            pending.replace(target)
        return {"path": str(target), "sha256": source["sha256"]}

    original = normalized["source_record"]
    record_copy = copy(original, "source_record.json")
    if normalized["preparation_type"] == "mother_sources":
        result = {**normalized, "source_record": record_copy,
                  "frozen_input_sources": True, "original_source_record": original}
        if recovery is not None:
            evidence = {key: copy(recovery[key], f"recovery_{key}{Path(recovery[key]['path']).suffix}")
                        for key in RECOVERY_SMALL_KEYS}
            evidence["feb01_csv_files"] = [copy(item, f"recovery_feb01_{index:02d}.csv")
                                            for index, item in enumerate(recovery["feb01_csv_files"])]
            result["recovery_source"] = {**recovery, "frozen_evidence": evidence}
        validate_preparation_config(result)
        return result
    mother_copies = []
    for index, item in enumerate(normalized["mother_sources"]):
        mother_copies.append({"selected_time_utc": item["selected_time_utc"],
                              **copy(item, f"mother_{index:03d}.csv")})
    result = {**normalized, "source_record": record_copy, "mother_sources": mother_copies,
              "frozen_input_sources": True, "original_source_record": original}
    validate_preparation_config(result)
    return result


def _detail_gzip(path: Path, rows: list[dict]) -> str:
    pending = path.with_name(path.name + ".tmp")
    with pending.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as text:
            for row in rows:
                text.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)
    return _sha(path)


def _partitions(config: dict) -> list[tuple[str, list[dict], tuple[int, ...]]]:
    mothers = config["mother_sources"]
    if config["preparation_type"] == "dev_validation":
        return [("development", mothers[::2], DEVELOPMENT_COUNTS),
                ("validation", mothers[1::2], VALIDATION_COUNTS)]
    return [("final", mothers, FINAL_COUNTS)]


def run_preparation(job_dir: Path, config: dict,
                    progress_callback: Callable[[dict], None] | None = None,
                    stop: Callable[[], bool] | None = None) -> dict:
    """Generate only frozen scientific inputs, with per-mother resume checks."""
    config = validate_preparation_config(config)
    if not config.get("frozen_input_sources"):
        raise ValueError("workbench worker requires frozen input_sources")
    job_dir = Path(job_dir).resolve()
    if config["preparation_type"] == "mother_sources":
        return _prepare_mother_sources(job_dir, config, progress_callback, stop)
    record = _json(Path(config["source_record"]["path"]))
    model = record["source"]["demand_model"]
    outputs = {}
    for partition, mothers, counts in _partitions(config):
        out = job_dir / "prepared" / partition
        out.mkdir(parents=True, exist_ok=True)
        builder_role = "test" if partition == "final" else "validation"
        role = "test" if partition == "final" else partition
        contract = {"experiment_id": "S2-P19", "preparation_type": config["preparation_type"],
                    "partition": partition, "source_record_sha256": config["source_record"]["sha256"],
                    "mothers": mothers, "beam_counts": counts, "loads": LOADS,
                    "seed": SEED, "environment_profile": "europe_v1"}
        contract_path = out / "input_contract.json"
        if contract_path.exists():
            if _json(contract_path) != json.loads(json.dumps(contract)):
                raise ValueError("existing preparation contract differs")
        else:
            _write_json(contract_path, contract)
        completed = out / "record.json"
        if completed.exists():
            prior = _json(completed)
            if prior.get("count") != len(mothers) * len(counts) * len(LOADS):
                raise ValueError("existing completed input count differs")
            if len(prior.get("files", [])) != prior["count"]:
                raise ValueError("existing completed input file list differs")
            for item in prior["files"]:
                if _sha(Path(item["path"])) != item["sha256"]:
                    raise ValueError("existing completed scene SHA differs")
            outputs[partition] = {"path": str(completed), "sha256": _sha(completed),
                                  "count": prior["count"], "reused": True}
            continue
        references, snapshots = [], []
        per_n = Counter()
        for index, mother in enumerate(mothers):
            if stop is not None and stop():
                raise InterruptedError("S2-P19 data preparation stopped before next mother")
            mother_id = f"{partition}_{index:05d}_{hashlib.sha256(mother['selected_time_utc'].encode()).hexdigest()[:12]}"
            checkpoint = out / f"mother_{index:05d}.json"
            if checkpoint.exists():
                saved = _json(checkpoint)
                if saved.get("mother_scene_id") != mother_id or saved.get("source_sha256") != mother["sha256"]:
                    raise ValueError("mother checkpoint identity differs")
                for entry in saved["references"]:
                    if _sha(Path(entry["path"])) != entry["sha256"]:
                        raise ValueError("resumed scene SHA differs")
                    detail = entry["observation_detail"]
                    if _sha(Path(detail["path"])) != detail["sha256"]:
                        raise ValueError("resumed observation detail SHA differs")
                    per_n[entry["beam_count"]] += 1
                references.extend(saved["references"])
                snapshots.extend(saved["snapshots"])
                continue
            mother_refs, mother_snapshots = [], []
            for n in counts:
                arrays, provenance = build_aviation_scenes(
                    [Path(mother["path"])], [mother["selected_time_utc"]],
                    source_format="csv_snapshot", units={**CSV_UNITS, "persons": "count"},
                    demand_model=model, seed=SEED, max_record_age_seconds=60,
                    environment_profile="europe_v1", partition_role=builder_role,
                    window_seconds=60, beam_count=n, min_positive_aircraft=200)
                base = arrays[0]
                snapshot = provenance["snapshots"][0]
                detail = out / f"observations_{index:05d}_n{n:03d}.jsonl.gz"
                detail_sha = _detail_gzip(detail, snapshot.pop("source_observations"))
                for load in LOADS:
                    data = dict(base)
                    data["rate"] = np.asarray(base["rate"], dtype=np.float64) * load
                    scene_id = f"scene_{index:05d}_n{n:03d}_l{int(load * 100):03d}"
                    target = out / f"{scene_id}.npz"
                    pending = target.with_name(target.name + ".tmp.npz")
                    np.savez_compressed(pending, **data)
                    pending.replace(target)
                    total = snapshot["total_demand_bps"] * load
                    covered = float(data["rate"].sum())
                    uncovered = snapshot["uncovered_demand_bps"] * load
                    if not np.isclose(total, covered + uncovered, rtol=1e-12, atol=1e-5):
                        raise RuntimeError("scaled demand account differs")
                    entry = {"path": str(target), "sha256": _sha(target), "scene_id": scene_id,
                             "mother_scene_id": mother_id, "beam_count": n, "load_multiplier": load,
                             "partition_role": role, "selected_time_utc": mother["selected_time_utc"],
                             "source_sha256": mother["sha256"],
                             "observation_detail": {"path": str(detail), "sha256": detail_sha,
                                                    "base_load_multiplier": 1.0,
                                                    "rule": "multiply each base aircraft and beam demand by load_multiplier"},
                             "total_demand_bps": total, "covered_demand_bps": covered,
                             "uncovered_demand_bps": uncovered,
                             "covered_demand_fraction": covered / total if total else 0.0,
                             "covered_aircraft_fraction": snapshot["covered_aircraft_fraction"]}
                    mother_refs.append(entry)
                    compact = {key: entry[key] for key in (
                        "scene_id", "mother_scene_id", "beam_count", "load_multiplier",
                        "selected_time_utc", "total_demand_bps", "covered_demand_bps",
                        "uncovered_demand_bps", "covered_demand_fraction", "covered_aircraft_fraction")}
                    mother_snapshots.append(compact)
                    per_n[n] += 1
            _write_json(checkpoint, {"mother_scene_id": mother_id, "source_sha256": mother["sha256"],
                                     "references": mother_refs, "snapshots": mother_snapshots})
            references.extend(mother_refs)
            snapshots.extend(mother_snapshots)
            if progress_callback is not None:
                progress_callback({"partition": partition, "completed_mothers": index + 1,
                                   "total_mothers": len(mothers), "completed_scenes": len(references),
                                   "total_scenes": len(mothers) * len(counts) * len(LOADS)})
        expected = len(mothers) * len(counts) * len(LOADS)
        if len(references) != expected or set(per_n) != set(counts) or any(per_n[n] != len(mothers) * 2 for n in counts):
            raise RuntimeError("prepared scene count/distribution differs")
        profile = profile_metadata("europe_v1")
        dataset = {"id": f"s2p19_{partition}_20261007", "name": f"S2-P19 {partition}",
                   "path": str(out), "count": expected,
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "source": {"type": "aviation_variable_beams_s2p19", "experiment_id": "S2-P19",
                              "partition_role": role,
                              "source_dates": sorted({row["selected_time_utc"][:10] for row in mothers}),
                              "snapshot_dates": sorted({row["selected_time_utc"][:10] for row in mothers}),
                              "seed": SEED, "demand_model": model, "load_multipliers": list(LOADS),
                              "mother_scene_count": len(mothers), "mother_scene_inputs": mothers,
                              "mother_record_path": config["source_record"]["path"],
                              "mother_record_sha256": config["source_record"]["sha256"],
                              "selection_rule": "Jan12 sorted odd/even six mothers; Feb01/02 twenty each by frozen time ranks",
                              "positive_demand_eligibility": 200,
                              "beam_count_distribution": dict(sorted(per_n.items())),
                              "coverage": {"algorithm": "demand-weighted farthest-point medoids; nearest exclusive association",
                                           "beam_full_width_degrees": 0.5},
                              "snapshots": snapshots},
                   "files": references, "profile": "europe_v1",
                   "environment_version": profile["environment_version"],
                   "environment_profile": profile,
                   "units": {"rate": "bit/s", "beam_location": "degrees",
                             "covered_demand_bps": "bit/s"}}
        _write_json(completed, dataset)
        outputs[partition] = {"path": str(completed), "sha256": _sha(completed),
                              "count": expected, "reused": False}
    return {"status": "completed_inputs_only_no_policy_or_training",
            "preparation_type": config["preparation_type"], "outputs": outputs}


def _official_metadata(asset: dict) -> dict:
    api = ("https://api.github.com/repos/adsblol/globe_history_2024/releases/tags/"
           + asset["release_tag"])
    request = urllib.request.Request(
        api, headers={"User-Agent": "S2-P19-frozen-input-preparation",
                      "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=45) as response:
        release = json.load(response)
    matches = [item for item in release.get("assets", [])
               if item.get("id") == asset["asset_id"]
               and item.get("name") == asset["asset_name"]]
    if release.get("tag_name") != asset["release_tag"] or len(matches) != 1:
        raise RuntimeError("official GitHub release/asset identity differs")
    item = matches[0]
    if item.get("browser_download_url") != asset["url"] or item.get("size") != asset["expected_bytes"]:
        raise RuntimeError("official GitHub asset URL/byte count differs")
    return {"release_url": release.get("html_url"), "release_tag": release["tag_name"],
            "asset_id": item["id"], "asset_name": item["name"],
            "asset_url": item["browser_download_url"], "expected_bytes": item["size"],
            "asset_digest": item.get("digest"), "asset_created_at": item.get("created_at"),
            "asset_updated_at": item.get("updated_at")}


def _download_archive(raw_dir: Path, asset: dict, stop: Callable[[], bool] | None,
                      progress_callback: Callable[[dict], None] | None,
                      *, api_asset: bool = False, max_attempts: int = 8) -> dict:
    if api_asset and (asset["date_utc"] != "2024-02-02" or asset["asset_id"] != 149522532 or
                      max_attempts != 4):
        raise ValueError("恢复传输仅允许Feb02同一官方asset_id且最多四次新尝试")
    if not api_asset and max_attempts != 8:
        raise ValueError("原下载合同保持八次上限")
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / asset["asset_name"]
    partial = raw_dir / (asset["asset_name"] + ".part")
    receipt = raw_dir / (asset["date_utc"] + "_download.json")
    state = _json(receipt) if receipt.exists() else {
        "status": "pending", "asset": asset, "attempts": []}
    if state.get("asset") != asset:
        raise ValueError("existing download belongs to another official asset")
    if api_asset:
        if state.get("transport", "github_asset_api_v1") != "github_asset_api_v1":
            raise ValueError("恢复下载回执来自不同传输入口")
        state["transport"] = "github_asset_api_v1"
        state["attempt_limit"] = 4
    elif state.get("transport"):
        raise ValueError("原下载任务不得复用API恢复回执")
    if target.exists():
        if target.stat().st_size != asset["expected_bytes"]:
            raise RuntimeError("existing archive byte count differs")
        digest = _sha(target)
        if state.get("status") == "completed" and state.get("sha256") != digest:
            raise RuntimeError("existing archive SHA differs from saved receipt")
        state.update(status="completed", path=str(target), bytes=target.stat().st_size,
                     sha256=digest)
        _write_json(receipt, state)
        return state
    if partial.exists() and partial.stat().st_size > asset["expected_bytes"]:
        raise RuntimeError("partial official archive exceeds expected byte count")
    if len(state["attempts"]) >= max_attempts and (not partial.exists() or
                                          partial.stat().st_size != asset["expected_bytes"]):
        raise RuntimeError("official archive exhausted " +
                           ("four" if api_asset else "eight") + " recorded download attempts")
    stderr_path = raw_dir / (asset["date_utc"] + "_curl.stderr")
    for attempt in range(len(state["attempts"]) + 1, max_attempts + 1):
        if stop is not None and stop():
            raise InterruptedError("official archive download stopped")
        command = ["curl", "--location", "--fail", "--silent", "--show-error",
                   "--connect-timeout", "20", "--max-time", "1800", "--continue-at", "-",
                   "--output", str(partial)]
        if api_asset:
            command.extend(["--header", "Accept: application/octet-stream",
                            "--header", "X-GitHub-Api-Version: 2022-11-28",
                            "--user-agent", "S2-P19-official-asset-recovery",
                            "https://api.github.com/repos/adsblol/globe_history_2024/releases/assets/149522532"])
        else:
            command.append(asset["url"])
        started = datetime.now(timezone.utc).isoformat()
        with stderr_path.open("ab") as error_stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=error_stream)
            while process.poll() is None:
                if stop is not None and stop():
                    process.terminate()
                    process.wait(timeout=10)
                    raise InterruptedError("official archive download stopped")
                time.sleep(2)
        current = partial.stat().st_size if partial.exists() else 0
        state["attempts"].append({"number": attempt, "started_utc": started,
                                  "ended_utc": datetime.now(timezone.utc).isoformat(),
                                  "curl_exit": process.returncode, "partial_bytes": current})
        state.update(status="downloading", partial_bytes=current)
        _write_json(receipt, state)
        if progress_callback is not None:
            progress_callback({"partition": "mother_sources", "date_utc": asset["date_utc"],
                               "phase": "download", "partial_bytes": current,
                               "expected_bytes": asset["expected_bytes"]})
        if process.returncode == 0 and current == asset["expected_bytes"]:
            break
        if current > asset["expected_bytes"]:
            raise RuntimeError("download exceeded official byte count")
        if attempt == max_attempts:
            raise RuntimeError("official archive did not download after " +
                               ("four API" if api_asset else "eight") + " resumable attempts")
        time.sleep(min(5 * attempt, 30))
    if not partial.exists() or partial.stat().st_size != asset["expected_bytes"]:
        raise RuntimeError("official archive byte count incomplete after download")
    digest = _sha(partial)
    partial.replace(target)
    state.update(status="completed", path=str(target), bytes=target.stat().st_size,
                 sha256=digest, completed_utc=datetime.now(timezone.utc).isoformat())
    _write_json(receipt, state)
    return state


def _iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _trace_position(row: object, profile) -> tuple[str, tuple[float, float, float] | None]:
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
        if elevation_degrees([[lat, lon]], profile)[0] < 10.0:
            return "below_elevation", None
        return "selected", (lat, lon, altitude_m)
    except (ValueError, TypeError, IndexError, OverflowError):
        return "invalid", None


def _scan_day(archive: Path, day: str, model: dict,
              progress_callback: Callable[[dict], None] | None,
              stop: Callable[[], bool] | None) -> tuple[list[list[tuple]], list[dict], dict]:
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(minutes=2)
    stamps = [(start + timedelta(minutes=5 * index)).timestamp() for index in range(288)]
    rows: list[list[tuple]] = [[] for _ in stamps]
    counts = [Counter() for _ in stamps]
    seen_icao = set()
    profile = get_profile("europe_v1")
    nfiles = malformed = source_records = 0
    first_malformed = []
    with tarfile.open(archive, mode="r|") as tar:
        for member in tar:
            if not member.isfile() or not member.name.startswith("./traces/") or not member.name.endswith(".json"):
                continue
            nfiles += 1
            try:
                raw = tar.extractfile(member).read()
                data = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
                icao = str(data["icao"]).strip().lower()
                base = float(data["timestamp"])
                trace = data["trace"]
                if not icao or not math.isfinite(base) or not isinstance(trace, list):
                    raise ValueError("bad ICAO, base timestamp or trace")
                if icao in seen_icao:
                    raise RuntimeError("duplicate ICAO within official UTC day")
                seen_icao.add(icao)
                offsets = [float(entry[0]) for entry in trace]
                if not all(math.isfinite(offset) for offset in offsets):
                    raise ValueError("nonfinite trace offset")
                if offsets != sorted(offsets):
                    trace = [entry for _, entry in sorted(zip(offsets, trace), key=lambda pair: pair[0])]
                    offsets.sort()
                source_records += len(trace)
                for index, stamp in enumerate(stamps):
                    position = bisect_right(offsets, stamp - base) - 1
                    if position < 0:
                        counts[index]["no_prior_record"] += 1
                        continue
                    age = stamp - (base + offsets[position])
                    if age > 60:
                        counts[index]["expired"] += 1
                        continue
                    reason, position_data = _trace_position(trace[position], profile)
                    counts[index][reason] += 1
                    if reason == "selected":
                        lat, lon, altitude_m = position_data
                        rows[index].append((icao, _iso(base + offsets[position]), lat, lon, altitude_m))
            except RuntimeError:
                raise
            except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError) as exc:
                malformed += 1
                if len(first_malformed) < 5:
                    first_malformed.append({"member": member.name, "error": repr(exc)})
            if nfiles % 5000 == 0:
                if stop is not None and stop():
                    raise InterruptedError("official archive scan stopped")
                if progress_callback is not None:
                    progress_callback({"partition": "mother_sources", "date_utc": day,
                                       "phase": "scan", "files_processed": nfiles,
                                       "source_trace_records": source_records})
    candidates = []
    for index, (stamp, selected, tally) in enumerate(zip(stamps, rows, counts, strict=True)):
        stream_id = hashlib.sha256(_iso(stamp).encode()).digest()
        seed_values = np.frombuffer(stream_id, dtype="<u4").tolist()
        rng = np.random.default_rng(np.random.SeedSequence([SEED, *seed_values]))
        persons = np.full(len(selected), int(model["persons_per_aircraft"]))
        fraction = model["satellite_access_fraction"] * model["active_fraction"]
        active = rng.binomial(persons, fraction) if model["sampling"] == "binomial" else persons * fraction
        demand = (active * model["rate_mbps_per_active_person"] + model["aircraft_base_mbps"]) * 1e6
        positive = int((demand > 0).sum())
        unique = len({(row[2], row[3]) for row in selected})
        candidates.append({"index": index, "selected_time_utc": _iso(stamp),
                           "selected_aircraft": len(selected), "unique_positions": unique,
                           "positive_demand_aircraft": positive,
                           "eligible_200": positive >= 200 and unique >= 200,
                           "rejection_reason": None if positive >= 200 and unique >= 200
                           else "fewer_than_200_positive_demand_or_distinct_positions",
                           "counts": dict(tally)})
    return rows, candidates, {"source_files": nfiles, "source_trace_records": source_records,
                              "malformed_files": malformed, "first_malformed": first_malformed}


def _selected_indices(candidates: list[dict]) -> tuple[list[int], list[dict]]:
    targets = [int((i + 0.5) * 288 / 20) for i in range(20)]
    selected = []
    substitutions = []
    used = set()
    for target in targets:
        chosen = next((index for index in range(target, 288)
                       if candidates[index]["eligible_200"] and index not in used), None)
        if chosen is None:
            raise RuntimeError("fewer than twenty qualified mother times under fixed same-date forward rule")
        used.add(chosen)
        selected.append(chosen)
        if chosen != target:
            substitutions.append({"target_index": target, "selected_index": chosen,
                                  "reason": candidates[target]["rejection_reason"]})
    return selected, substitutions


def _reuse_feb01_mothers(job_dir: Path, recovery: dict,
                         progress_callback: Callable[[dict], None] | None) -> tuple[list[dict], list[dict], dict, dict]:
    """Copy only small, SHA-locked Feb01 evidence; never write the parent task."""
    archive = recovery["feb01_archive"]
    archive_path = Path(archive["path"])
    if (archive_path.stat().st_size != OFFICIAL_ASSETS[0][2] or
            _sha(archive_path) != archive["sha256"]):
        raise ValueError("复用前旧Feb01官方归档大小或SHA发生变化")
    frozen = recovery["frozen_evidence"]
    day = "2024-02-01"
    day_dir = job_dir / "mother_sources" / "snapshots" / "final" / day
    day_dir.mkdir(parents=True, exist_ok=True)

    def copy_small(reference: dict, target: Path):
        if target.exists():
            if _sha(target) != reference["sha256"]:
                raise ValueError("恢复任务已保存的小证据与旧Feb01 SHA不同")
            return
        pending = target.with_name(target.name + ".tmp")
        shutil.copy2(reference["path"], pending)
        if _sha(pending) != reference["sha256"]:
            pending.unlink(missing_ok=True)
            raise ValueError("复制Feb01小证据后SHA不同")
        pending.replace(target)

    eligibility_path = day_dir / "eligibility.json"
    selection_path = day_dir / "selection_contract.json"
    copy_small(frozen["feb01_eligibility"], eligibility_path)
    copy_small(frozen["feb01_selection"], selection_path)
    eligibility = _json(eligibility_path)
    selection = _json(selection_path)
    indices = selection["selected_indices"]
    candidates = eligibility["candidates"]
    files, snapshots = [], []
    for position, (index, source_ref) in enumerate(zip(indices, frozen["feb01_csv_files"], strict=True)):
        target = day_dir / f"t{index:03d}.csv"
        copy_small(source_ref, target)
        candidate = candidates[index]
        files.append({"path": str(target), "sha256": source_ref["sha256"],
                      "bytes": target.stat().st_size,
                      "selected_time_utc": candidate["selected_time_utc"]})
        snapshots.append({"selected_time_utc": candidate["selected_time_utc"],
                          "source_dates": [day],
                          "counts": {"selected_aircraft": candidate["selected_aircraft"],
                                     "zero_demand_aircraft": candidate["selected_aircraft"]
                                     - candidate["positive_demand_aircraft"]}})
        if progress_callback is not None:
            progress_callback({"partition": "mother_sources", "date_utc": day,
                               "phase": "reused_verified_csv", "completed_mothers": position + 1,
                               "total_mothers": 40})
    old_download = _json(Path(frozen["feb01_download"]["path"]))
    if old_download.get("sha256") != archive["sha256"]:
        raise ValueError("冻结Feb01下载回执与复用归档SHA不同")
    receipt = {"date_utc": day,
               "official": {"release_tag": old_download["asset"]["release_tag"],
                            "asset_id": old_download["asset"]["asset_id"],
                            "asset_name": old_download["asset"]["asset_name"],
                            "asset_url": old_download["asset"]["url"],
                            "expected_bytes": old_download["asset"]["expected_bytes"],
                            "metadata_source": "parent_frozen_download_receipt"},
               "path": str(archive_path), "sha256": archive["sha256"],
               "bytes": archive_path.stat().st_size,
               "reused_from_parent_run_id": recovery["parent_run_id"],
               "parent_download_receipt_sha256": recovery["feb01_download"]["sha256"]}
    selection_ref = {"date_utc": day, "path": str(selection_path),
                     "sha256": _sha(selection_path),
                     "substitution_count": len(selection.get("substitutions", [])),
                     "reused_from_parent_run_id": recovery["parent_run_id"]}
    return files, snapshots, receipt, selection_ref


def _prepare_mother_sources(job_dir: Path, config: dict,
                            progress_callback: Callable[[dict], None] | None,
                            stop: Callable[[], bool] | None) -> dict:
    root = job_dir / "mother_sources"
    raw_dir = root / "raw"
    root.mkdir(parents=True, exist_ok=True)
    output_record = root / "record.json"
    if output_record.exists():
        previous = _json(output_record)
        if (previous.get("count") != 40 or
                previous.get("source", {}).get("demand_source_record_sha256")
                != config["source_record"]["sha256"]):
            raise ValueError("existing mother record count differs")
        for item in previous["source"]["files"]:
            if _sha(Path(item["path"])) != item["sha256"]:
                raise ValueError("completed mother CSV SHA differs")
        return {"status": "completed_mother_sources_only", "path": str(output_record),
                "sha256": _sha(output_record), "count": 40, "reused": True}
    model = _json(Path(config["source_record"]["path"]))["source"]["demand_model"]
    files, snapshots, archive_receipts, selections = [], [], [], []
    recovery = config.get("recovery_source")
    if recovery is not None and "frozen_evidence" not in recovery:
        raise ValueError("恢复worker缺工作台冻结的小证据")
    for asset in config["official_assets"]:
        if stop is not None and stop():
            raise InterruptedError("mother preparation stopped before next day")
        day = asset["date_utc"]
        if recovery is not None and day == "2024-02-01":
            reused_files, reused_snapshots, reused_archive, reused_selection = _reuse_feb01_mothers(
                job_dir, recovery, progress_callback)
            files.extend(reused_files)
            snapshots.extend(reused_snapshots)
            archive_receipts.append(reused_archive)
            selections.append(reused_selection)
            continue
        metadata = _official_metadata(asset)
        download = _download_archive(raw_dir, asset, stop, progress_callback,
                                     api_asset=recovery is not None, max_attempts=4 if recovery else 8)
        archive_receipts.append({"date_utc": day, "official": metadata,
                                 "path": download["path"], "sha256": download["sha256"],
                                 "bytes": download["bytes"]})
        day_dir = root / "snapshots" / "final" / day
        day_dir.mkdir(parents=True, exist_ok=True)
        rows, candidates, scan = _scan_day(Path(download["path"]), day, model,
                                          progress_callback, stop)
        eligibility = {"date_utc": day, "candidate_grid": "00:02 UTC + 5*i minutes, i=0..287",
                       "eligible_count": sum(item["eligible_200"] for item in candidates),
                       "candidate_count": 288, "archive_sha256": download["sha256"],
                       "scan": scan, "candidates": candidates}
        eligibility_path = day_dir / "eligibility.json"
        if eligibility_path.exists() and _json(eligibility_path) != eligibility:
            raise ValueError("rerun eligibility differs from frozen previous scan")
        _write_json(eligibility_path, eligibility)
        indices, substitutions = _selected_indices(candidates)
        selection = {"date_utc": day, "selection_rule": "20 fixed centered 5-minute ranks; ineligible -> earliest later unused eligible same UTC day",
                     "target_indices": [int((i + 0.5) * 288 / 20) for i in range(20)],
                     "selected_indices": indices, "substitutions": substitutions,
                     "selected_times_utc": [candidates[index]["selected_time_utc"] for index in indices],
                     "eligibility_sha256": _sha(eligibility_path),
                     "archive_sha256": download["sha256"]}
        selection_path = day_dir / "selection_contract.json"
        if selection_path.exists() and _json(selection_path) != selection:
            raise ValueError("rerun mother selection differs")
        _write_json(selection_path, selection)
        selections.append({"date_utc": day, "path": str(selection_path),
                           "sha256": _sha(selection_path), "substitution_count": len(substitutions)})
        for position, index in enumerate(indices):
            if stop is not None and stop():
                raise InterruptedError("mother CSV preparation stopped")
            candidate = candidates[index]
            target = day_dir / f"t{index:03d}.csv"
            pending = target.with_name(target.name + ".tmp")
            with pending.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("aircraft_id", "timestamp", "lat", "lon", "altitude_m"))
                writer.writerows(sorted(rows[index]))
            digest = _sha(pending)
            if target.exists():
                if _sha(target) != digest:
                    raise ValueError("existing mother CSV differs after rescan")
                pending.unlink()
            else:
                pending.replace(target)
            files.append({"path": str(target), "sha256": digest, "bytes": target.stat().st_size,
                          "selected_time_utc": candidate["selected_time_utc"]})
            snapshots.append({"selected_time_utc": candidate["selected_time_utc"],
                              "source_dates": [day],
                              "counts": {"selected_aircraft": candidate["selected_aircraft"],
                                         "zero_demand_aircraft": candidate["selected_aircraft"]
                                         - candidate["positive_demand_aircraft"]}})
            if progress_callback is not None:
                progress_callback({"partition": "mother_sources", "date_utc": day,
                                   "phase": "selected_csv", "completed_mothers": len(files),
                                   "total_mothers": 40, "completed_this_day": position + 1})
    if len(files) != 40 or len({item["selected_time_utc"] for item in files}) != 40:
        raise RuntimeError("mother source count/identity differs")
    profile = profile_metadata("europe_v1")
    record = {"id": "s2p19_final_mothers_20261007", "name": "S2-P19 independent final mothers",
              "path": str(root / "snapshots" / "final"), "count": 40,
              "source": {"type": "official_readsb_positions_synthetic_demand_pending",
                         "experiment_id": "S2-P19", "partition_role": "test", "seed": SEED,
                         "environment_profile": "europe_v1", "source_dates": list(FINAL_DATES),
                         "snapshot_dates": list(FINAL_DATES), "max_record_age_seconds": 60,
                         "demand_source_record_sha256": config["source_record"]["sha256"],
                         "demand_model": model, "files": files, "snapshots": snapshots,
                         "upstream_archives": archive_receipts,
                         "selection_contracts": selections},
              "profile": "europe_v1", "environment_version": profile["environment_version"],
              "environment_profile": profile,
              "units": {"rate": "bit/s", "beam_location": "degrees"}}
    _write_json(output_record, record)
    return {"status": "completed_mother_sources_only", "path": str(output_record),
            "sha256": _sha(output_record), "count": 40,
            "dates": list(FINAL_DATES), "reused": False}
