"""Prepare twenty Jan29 mother snapshots on 20424, without model inference."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(os.environ["VARIABLE_DATA_ROOT"])
SOURCE = Path(os.environ["VARIABLE_DATA_SOURCE"])
REFERENCE = Path(__file__).with_name("prepare_snapshots_reference.py")
DATE = "2024-01-29"
TAG = "v2024.01.29-planes-readsb-staging-0"
ASSET_ID = 148624093
EXPECTED_BYTES = 1719838720
OFFICIAL_URL = f"https://github.com/adsblol/globe_history_2024/releases/download/{TAG}/{TAG}.tar"
ARCHIVE = ROOT / "raw" / f"{TAG}.tar"
STATUS = ROOT / "jan29_mother" / "preparation_status.json"
G11_RECORD = Path(os.environ["VARIABLE_G11_RECORD"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def positive_count(rows: list[tuple], selected_time: str, model: dict, iso) -> int:
    stamp = datetime.fromisoformat(selected_time.replace("Z", "+00:00")).timestamp()
    stream_id = hashlib.sha256(iso(stamp).encode()).digest()
    rng = np.random.default_rng(np.random.SeedSequence([51001, *np.frombuffer(stream_id, dtype="<u4").tolist()]))
    persons = np.full(len(rows), int(model["persons_per_aircraft"]))
    fraction = model["satellite_access_fraction"] * model["active_fraction"]
    active = rng.binomial(persons, fraction) if model["sampling"] == "binomial" else persons * fraction
    demand = (active * model["rate_mbps_per_active_person"] + model["aircraft_base_mbps"]) * 1e6
    return int((demand > 0).sum())


def main() -> None:
    if (ROOT / "jan29_mother/record.json").exists():
        raise FileExistsError("Jan29 mother input record already finalized")
    state = {"status": "checking_official_archive", "started_utc": datetime.now(timezone.utc).isoformat(),
             "date_utc": DATE, "planned_mothers": 20, "model_inference_calls": 0,
             "training_interactions": 0}
    write_json(STATUS, state)
    try:
        download = json.loads((ROOT / "raw/download_status.json").read_text(encoding="utf-8"))
        if (download.get("status") != "completed" or download.get("date_utc") != DATE
                or download.get("url") != OFFICIAL_URL or download.get("asset_id") != ASSET_ID
                or download.get("bytes") != EXPECTED_BYTES or ARCHIVE.stat().st_size != EXPECTED_BYTES):
            raise RuntimeError("Jan29 archive identity, size or download status mismatch")
        archive_sha = sha256(ARCHIVE)
        if archive_sha != download["sha256"]:
            raise RuntimeError("Jan29 archive SHA mismatch")
        source_manifest = json.loads((SOURCE / "source_manifest.json").read_text(encoding="utf-8"))
        if source_manifest.get("status") != "frozen_data_source_no_training":
            raise RuntimeError("Data source is not frozen")
        for name, expected in source_manifest["files"].items():
            if sha256(SOURCE / name) != expected:
                raise RuntimeError("Frozen data source changed: " + name)
        sys.path.insert(0, str(SOURCE / "src"))
        import satellite_rl
        from satellite_rl.envs.regions import profile_metadata
        from satellite_rl.workbench.aviation import _iso
        if Path(satellite_rl.__file__).resolve() != SOURCE / "src/satellite_rl/__init__.py":
            raise RuntimeError("Aviation module imported from unexpected source")
        old = json.loads(G11_RECORD.read_text(encoding="utf-8"))
        g11_source = old["source"]
        if (old.get("count") != 576 or g11_source.get("seed") != 51001
                or g11_source.get("max_record_age_seconds") != 60):
            raise RuntimeError("G11 source model/seed mismatch")
        model = g11_source["demand_model"]
        if model.get("window_seconds") != 60:
            raise RuntimeError("G11 synthetic demand window mismatch")
        spec = importlib.util.spec_from_file_location("jan29_reference", REFERENCE)
        reference = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(reference)
        reference.ROOT = ROOT
        reference.STATUS = STATUS
        reference.DATES = (DATE,)
        reference.ROLES = ("test",)
        reference.MAX_AGE = 60.0
        state.update(status="scanning_288_fixed_times", archive_sha256=archive_sha)
        write_json(STATUS, state)
        rows, candidates = reference.process_day(DATE, state)
        eligibility = []
        for row, candidate in zip(rows, candidates, strict=True):
            positive = positive_count(row, candidate["selected_time_utc"], model, _iso)
            unique = candidate["unique_positions"]
            entry = {"index": candidate["index"], "selected_time_utc": candidate["selected_time_utc"],
                     "selected_aircraft": candidate["selected_aircraft"], "unique_positions": unique,
                     "positive_demand_aircraft": positive,
                     "eligible_200": positive >= 200 and unique >= 200}
            eligibility.append(entry)
        eligible = [entry["index"] for entry in eligibility if entry["eligible_200"]]
        eligibility_path = ROOT / "jan29_mother/eligibility.json"
        write_json(eligibility_path, {"date_utc": DATE, "candidate_count": len(eligibility),
                                      "eligible_count": len(eligible), "candidates": eligibility})
        if len(eligible) < 20:
            raise RuntimeError(f"Jan29仅有{len(eligible)}个合格母场景，不足20")
        indices = [eligible[int((i + 0.5) * len(eligible) / 20)] for i in range(20)]
        if len(set(indices)) != 20:
            raise RuntimeError("Jan29按合格时刻均匀选择产生重复")
        selection = {"status": "frozen_input_selection_before_scene_generation",
                     "date_utc": DATE, "role": "test", "selection_rule": "fixed time ranks among 200-positive eligible 5-minute candidates",
                     "candidate_grid": "00:02 UTC + 5*i minutes, i=0..287",
                     "eligible_indices": eligible, "selected_indices": indices,
                     "selected_times_utc": [candidates[i]["selected_time_utc"] for i in indices],
                     "eligibility_sha256": sha256(eligibility_path),
                     "archive_sha256": archive_sha, "reference_script_sha256": sha256(REFERENCE),
                     "data_source_manifest_sha256": sha256(SOURCE / "source_manifest.json")}
        write_json(ROOT / "jan29_mother/selection_contract.json", selection)
        frames = reference.final_csv(DATE, "test", indices, rows, candidates)
        if len(frames["frames"]) != 20 or frames["rejected_times"]:
            raise RuntimeError("Jan29 20 preselected mother CSVs were not produced")
        by_index = {entry["index"]: entry for entry in eligibility}
        snapshots = []
        files = []
        for frame in frames["frames"]:
            check = by_index[frame["candidate_index"]]
            snapshots.append({"selected_time_utc": frame["selected_time_utc"],
                              "source_dates": [DATE],
                              "counts": {"selected_aircraft": check["selected_aircraft"],
                                         "zero_demand_aircraft": check["selected_aircraft"] - check["positive_demand_aircraft"]}})
            files.append({"path": frame["path"], "sha256": frame["sha256"],
                          "bytes": frame["bytes"], "selected_time_utc": frame["selected_time_utc"]})
        profile = profile_metadata("europe_v1")
        record = {"id": "variable_beams_jan29_mothers_20261004", "name": "S2-P17 Jan29 mother inputs",
                  "path": str(ROOT / "snapshots/test" / DATE), "count": 20,
                  "source": {"type": "official_readsb_positions_synthetic_demand_pending",
                             "experiment_id": "S2-P17", "partition_role": "test", "seed": 51001,
                             "environment_profile": "europe_v1", "source_dates": [DATE], "snapshot_dates": [DATE],
                             "max_record_age_seconds": 60, "demand_model": model,
                             "files": files, "snapshots": snapshots,
                             "upstream_archive": {"path": str(ARCHIVE), "sha256": archive_sha,
                                                  "bytes": EXPECTED_BYTES, "asset_id": ASSET_ID,
                                                  "url": OFFICIAL_URL},
                             "selection_contract_sha256": sha256(ROOT / "jan29_mother/selection_contract.json"),
                             "candidate_eligibility_sha256": sha256(eligibility_path)},
                  "profile": "europe_v1", "environment_version": profile["environment_version"],
                  "environment_profile": profile,
                  "units": {"rate": "bit/s", "beam_location": "degrees"}}
        write_json(ROOT / "jan29_mother/record.json", record)
        state.update(status="completed_mother_inputs_only", mother_count=20,
                     eligible_candidate_count=len(eligible),
                     record_sha256=sha256(ROOT / "jan29_mother/record.json"),
                     selection_contract_sha256=sha256(ROOT / "jan29_mother/selection_contract.json"),
                     ended_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), ended_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write_json(STATUS, state)


if __name__ == "__main__":
    main()
