"""On 20424, replace 56 ineligible G11 mothers using fixed half-grid times.

All eligibility uses input positions and the frozen synthetic demand model.
No policy, allocation, or optimizer is called. The original 520 CSVs are read
and SHA checked but never rewritten.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


BASE = Path(os.environ["VARIABLE_BASE"])
DATA_SOURCE = Path(os.environ["VARIABLE_DATA_SOURCE"])
G11_RECORD = Path(os.environ["VARIABLE_G11_RECORD"])
AUTHORIZATION = Path(os.environ["VARIABLE_AUTHORIZATION"])
REFERENCE = Path(__file__).with_name("prepare_snapshots_reference.py")
ROOT = BASE / "data"
STATUS = ROOT / "replenishment/preparation_status.json"
RULE = ROOT / "replenishment/selection_contract.json"
RESULT = ROOT / "prepared/train_mothers_replaced576/record.json"
TRAIN_INPUTS = ROOT / "source_csv/train"
DATES = ("2024-01-08", "2024-01-09")
SEED = 51001
ASSETS = {
    "2024-01-08": ("v2024.01.08-planes-readsb-prod-0.tar", 1610444800,
                   "a0b35aed430b8cd7b4de3fef3db79ce8efee958a2a1f2daa32056731db9074bd"),
    "2024-01-09": ("v2024.01.09-planes-readsb-prod-0.tar", 1443092480,
                   "df67efea6c4502177480a08e46fe91dc59f9e4c54c31c88bd1301c91f964b12e"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def positive_positions(rows: list[tuple], time_utc: str, model: dict, iso) -> tuple[int, int]:
    ordered = sorted(rows)
    stamp = datetime.fromisoformat(time_utc.replace("Z", "+00:00")).timestamp()
    stream_id = hashlib.sha256(iso(stamp).encode()).digest()
    rng = np.random.default_rng(np.random.SeedSequence([SEED, *np.frombuffer(stream_id, dtype="<u4").tolist()]))
    persons = np.full(len(ordered), int(model["persons_per_aircraft"]))
    fraction = model["satellite_access_fraction"] * model["active_fraction"]
    active = rng.binomial(persons, fraction) if model["sampling"] == "binomial" else persons * fraction
    demand = (active * model["rate_mbps_per_active_person"] + model["aircraft_base_mbps"]) * 1e6
    positive = demand > 0
    locations = {(row[2], row[3]) for row, is_positive in zip(ordered, positive) if is_positive}
    return int(positive.sum()), len(locations)


def original_contract(record: dict) -> tuple[dict, dict, list[dict]]:
    source = record.get("source") or {}
    if (record.get("count") != 576 or source.get("partition_role") != "train"
            or source.get("seed") != SEED or source.get("environment_profile") != "europe_v1"
            or source.get("max_record_age_seconds") != 60
            or len(source.get("files", [])) != 576 or len(source.get("snapshots", [])) != 576):
        raise RuntimeError("G11 576母场景身份、种子或观测窗口不符")
    model = source["demand_model"]
    if (model.get("window_seconds") != 60 or model.get("persons_per_aircraft") != 180
            or model.get("satellite_access_fraction") != 0.2 or model.get("active_fraction") != 0.3
            or model.get("rate_mbps_per_active_person") != 2.0 or model.get("aircraft_base_mbps") != 0
            or model.get("sampling") != "binomial" or model.get("lognormal_sigma") != 0.35):
        raise RuntimeError("原始合成业务模型与G11冻结条件不符")
    excluded = {day: [] for day in DATES}
    retained = []
    for index, (reference, snapshot) in enumerate(zip(source["files"], source["snapshots"], strict=True)):
        time_utc = snapshot["selected_time_utc"]
        day = time_utc[:10]
        counts = snapshot["counts"]
        selected = counts["selected_aircraft"]
        zero = counts["zero_demand_aircraft"]
        item = {"old_index": index, "old_scene_id": f"scene_{index:05d}",
                "old_time_utc": time_utc, "old_source_sha256": reference["sha256"],
                "positive_demand_aircraft": selected - zero}
        if day not in excluded:
            raise RuntimeError("原始训练母场景日期越界")
        if selected - zero >= 200:
            retained.append(item)
        else:
            item["reason"] = "fewer_than_200_positive_demand_aircraft"
            excluded[day].append(item)
    if len(retained) != 520 or sum(map(len, excluded.values())) != 56:
        raise RuntimeError("原始资格核验不再是520合格/56缺额")
    return model, excluded, retained


def main() -> None:
    if RESULT.exists():
        raise FileExistsError("576补足母记录已冻结，不得覆盖")
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if (authorization.get("experiment_id") != "S2-P17"
            or authorization.get("status") != "user_confirmed_decision"
            or authorization.get("decision") != "replace576"
            or authorization.get("replacement_mothers") != 56
            or authorization.get("derived_scene_count") != 8064):
        raise RuntimeError("用户确认的补足授权不匹配")
    source_manifest = json.loads((DATA_SOURCE / "source_manifest.json").read_text(encoding="utf-8"))
    for name, digest in source_manifest["files"].items():
        if sha(DATA_SOURCE / name) != digest:
            raise RuntimeError("冻结数据生成源码SHA不符：" + name)
    sys.path.insert(0, str(DATA_SOURCE / "src"))
    import satellite_rl
    from satellite_rl.envs.regions import profile_metadata
    from satellite_rl.workbench.aviation import _iso
    if Path(satellite_rl.__file__).resolve() != DATA_SOURCE / "src/satellite_rl/__init__.py":
        raise RuntimeError("地区/航空模块未从冻结数据源码导入")
    old = json.loads(G11_RECORD.read_text(encoding="utf-8"))
    model, excluded, retained = original_contract(old)
    missing = {day: len(excluded[day]) for day in DATES}
    # Freeze rule and quotas before scanning the two raw archives.
    contract = {"status": "predeclared_before_halfgrid_scan", "experiment_id": "S2-P17",
                "decision": "replace576", "selection_rule": "same-day quota; fixed time ranks among >=200 positive-demand and unique-position half-grid candidates",
                "candidate_grid": "00:04:30 UTC + 5*i minutes, i=0..287, each Jan08/09 day; original grid +150 seconds",
                "seed": SEED, "original_record_sha256": sha(G11_RECORD),
                "authorization_sha256": sha(AUTHORIZATION),
                "reference_script_sha256": sha(REFERENCE),
                "data_source_manifest_sha256": sha(DATA_SOURCE / "source_manifest.json"),
                "original_eligible_count": 520, "original_excluded_count": 56,
                "date_quota": missing, "original_excluded": excluded,
                "original_retained": retained, "selected": {}}
    write_json(RULE, contract)
    state = {"status": "scanning_halfgrid_raw_inputs", "started_utc": datetime.now(timezone.utc).isoformat(),
             "date_quota": missing, "original_eligible": 520, "original_excluded": 56,
             "training_interactions": 0, "model_inference_calls": 0}
    write_json(STATUS, state)
    try:
        spec = importlib.util.spec_from_file_location("original_adsb_preparation", REFERENCE)
        reference = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(reference)
        reference.ROOT = ROOT
        reference.STATUS = STATUS
        reference.DATES = DATES
        reference.ROLES = ("train", "train")
        reference.MAX_AGE = 60.0
        original_times = reference.candidate_times
        reference.candidate_times = lambda day: [stamp + timedelta(seconds=150)
                                                  for stamp in original_times(day)]
        candidates_by_day = {}
        rows_by_day = {}
        for day in DATES:
            name, size, expected = ASSETS[day]
            archive = ROOT / "raw" / name
            if not archive.is_file() or archive.stat().st_size != size or sha(archive) != expected:
                raise RuntimeError("Jan08/09官方原始包大小或SHA不符：" + day)
            rows, candidates = reference.process_day(day, state)
            if len(rows) != 288 or len(candidates) != 288:
                raise RuntimeError("半格候选不是每日288个固定时刻")
            eligibility = []
            for candidate, positions in zip(candidates, rows, strict=True):
                positive, unique_positive = positive_positions(
                    positions, candidate["selected_time_utc"], model, _iso)
                eligibility.append({"index": candidate["index"],
                                    "selected_time_utc": candidate["selected_time_utc"],
                                    "selected_aircraft": candidate["selected_aircraft"],
                                    "positive_demand_aircraft": positive,
                                    "unique_positive_positions": unique_positive,
                                    "eligible_200": positive >= 200 and unique_positive >= 200})
            path = ROOT / "replenishment" / f"halfgrid_eligibility_{day}.json"
            write_json(path, {"day": day, "count": 288, "candidates": eligibility})
            good = [entry for entry in eligibility if entry["eligible_200"]]
            quota = missing[day]
            if len(good) < quota:
                raise RuntimeError(f"{day}半格合格时刻{len(good)}不足本日缺额{quota}")
            selected = [good[int((i + 0.5) * len(good) / quota)] for i in range(quota)]
            if len({item["index"] for item in selected}) != quota:
                raise RuntimeError("时间秩选择产生重复半格候选")
            contract["selected"][day] = {"eligible_count": len(good),
                                          "indices": [x["index"] for x in selected],
                                          "times_utc": [x["selected_time_utc"] for x in selected],
                                          "eligibility_file": str(path),
                                          "eligibility_sha256": sha(path)}
            candidates_by_day[day] = {item["index"]: item for item in eligibility}
            rows_by_day[day] = rows
            state.update(status="halfgrid_scanned", completed_day=day,
                         halfgrid_eligible_count={d: contract["selected"][d]["eligible_count"]
                                                  for d in contract["selected"]})
            write_json(STATUS, state)
        contract["status"] = "selected_input_only_before_csv_write"
        write_json(RULE, contract)
        replacements = {}
        for day in DATES:
            quota = missing[day]
            selected_indices = contract["selected"][day]["indices"]
            for old_slot, candidate_index in zip(excluded[day], selected_indices, strict=True):
                candidate = candidates_by_day[day][candidate_index]
                target = TRAIN_INPUTS / day / f"half_t{candidate_index:03d}.csv"
                if target.exists():
                    raise FileExistsError("半格CSV目标已存在，拒绝覆盖：" + str(target))
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".csv.tmp")
                with temporary.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(reference.FIELDNAMES)
                    writer.writerows(sorted(rows_by_day[day][candidate_index]))
                temporary.replace(target)
                replacements[old_slot["old_index"]] = {**old_slot,
                    "new_time_utc": candidate["selected_time_utc"],
                    "candidate_index": candidate_index,
                    "positive_demand_aircraft": candidate["positive_demand_aircraft"],
                    "unique_positive_positions": candidate["unique_positive_positions"],
                    "selected_aircraft": candidate["selected_aircraft"],
                    "new_csv_path": str(target), "new_csv_sha256": sha(target)}
            if len(excluded[day]) != quota:
                raise RuntimeError("日期配额映射不完整")
        if len(replacements) != 56:
            raise RuntimeError("补足CSV不是56个")
        write_json(ROOT / "replenishment/replacement_mapping.json",
                   {"status": "mapped_56_inputs_no_allocation", "mapping": [replacements[i]
                                                                      for i in sorted(replacements)]})
        files, snapshots = [], []
        for index, (old_ref, old_snapshot) in enumerate(zip(old["source"]["files"],
                                                           old["source"]["snapshots"], strict=True)):
            if index in replacements:
                replacement = replacements[index]
                path = Path(replacement["new_csv_path"])
                files.append({"path": str(path), "sha256": replacement["new_csv_sha256"],
                              "bytes": path.stat().st_size,
                              "selected_time_utc": replacement["new_time_utc"],
                              "replaces_original_scene_id": replacement["old_scene_id"]})
                snapshots.append({"selected_time_utc": replacement["new_time_utc"],
                                  "source_dates": [replacement["new_time_utc"][:10]],
                                  "counts": {"selected_aircraft": replacement["selected_aircraft"],
                                             "zero_demand_aircraft": replacement["selected_aircraft"] -
                                                                      replacement["positive_demand_aircraft"]}})
            else:
                date = old_snapshot["selected_time_utc"][:10]
                path = TRAIN_INPUTS / date / Path(old_ref["path"]).name
                if not path.is_file() or sha(path) != old_ref["sha256"]:
                    raise RuntimeError("原520场景CSV缺失或SHA变化：" + str(path))
                files.append({"path": str(path), "sha256": old_ref["sha256"],
                              "bytes": path.stat().st_size,
                              "selected_time_utc": old_snapshot["selected_time_utc"],
                              "retained_original_scene_id": f"scene_{index:05d}"})
                snapshots.append({"selected_time_utc": old_snapshot["selected_time_utc"],
                                  "source_dates": [date],
                                  "counts": {"selected_aircraft": old_snapshot["counts"]["selected_aircraft"],
                                             "zero_demand_aircraft": old_snapshot["counts"]["zero_demand_aircraft"]}})
        if len(files) != 576 or len({item["selected_time_utc"] for item in snapshots}) != 576:
            raise RuntimeError("补足后的576母场景时刻数量不唯一")
        if any(s["counts"]["selected_aircraft"] - s["counts"]["zero_demand_aircraft"] < 200
               for s in snapshots):
            raise RuntimeError("补足母场景仍有不足200正需求航空器")
        profile = profile_metadata("europe_v1")
        final = {"id": "variable_beams_train_mothers_replaced576_20261004",
                 "name": "S2-P17 Jan08/09 520原始＋56半格补足母场景",
                 "path": str(TRAIN_INPUTS), "count": 576,
                 "source": {"type": "aviation_mothers_replenished_halfgrid",
                            "experiment_id": "S2-P17", "partition_role": "train",
                            "environment_profile": "europe_v1", "seed": SEED,
                            "max_record_age_seconds": 60, "demand_model": model,
                            "source_dates": list(DATES), "snapshot_dates": list(DATES),
                            "files": files, "snapshots": snapshots,
                            "original_g11_record_sha256": sha(G11_RECORD),
                            "authorization_sha256": sha(AUTHORIZATION),
                            "selection_contract_sha256": sha(RULE),
                            "replacement_mapping_sha256": sha(ROOT / "replenishment/replacement_mapping.json"),
                            "raw_archive_sha256": {day: ASSETS[day][2] for day in DATES}},
                 "profile": "europe_v1", "environment_version": profile["environment_version"],
                 "environment_profile": profile,
                 "units": {"rate": "bit/s", "beam_location": "degrees"}}
        write_json(RESULT, final)
        state.update(status="completed_576_mother_inputs_only", mother_count=576,
                     original_retained=520, replenished=56,
                     selected_contract_sha256=sha(RULE), record_path=str(RESULT),
                     record_sha256=sha(RESULT), ended_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(status="failed", error=repr(error), ended_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write_json(STATUS, state)


if __name__ == "__main__":
    main()
