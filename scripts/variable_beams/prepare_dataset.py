"""Build S2-P17 input scenes from verified aviation CSVs on 20424.

This script only prepares inputs. It never loads a policy or submits a job.
Run with the frozen repository's ``src`` on PYTHONPATH and the DQN interpreter.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from satellite_rl.envs.regions import profile_metadata
from satellite_rl.workbench.aviation import AviationSceneError, build_aviation_scenes, CSV_UNITS


SEED = 51001
LOADS = (0.75, 1.25)
HELD_OUT = frozenset((60, 90, 140, 180))
VALIDATION_COUNTS = (50, 80, 100, 120, 160, 200)
TEST_COUNTS = (50, 60, 75, 90, 100, 125, 140, 150, 175, 180, 200)
TRAIN_BANDS = (
    tuple(n for n in range(50, 100) if n not in HELD_OUT),
    tuple(n for n in range(101, 150) if n not in HELD_OUT),
    tuple(n for n in range(150, 201) if n not in HELD_OUT),
)
EXPECTED_MOTHERS = {"train": 576, "validation": 12, "test": 20}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def chosen_mothers(record: dict, role: str) -> list[tuple[dict, dict]]:
    source = record.get("source", {})
    if source.get("partition_role") != role:
        raise ValueError("母场景分区与请求不匹配")
    files, snapshots = source.get("files", []), source.get("snapshots", [])
    if len(files) != len(snapshots) or len(files) != record.get("count"):
        raise ValueError("母场景源文件、时刻及数量不一致")
    eligible = []
    ineligible = []
    for index, snapshot in enumerate(snapshots):
        counts = snapshot.get("counts", {})
        selected = counts.get("selected_aircraft")
        zero = counts.get("zero_demand_aircraft")
        if not isinstance(selected, int) or not isinstance(zero, int):
            raise ValueError("母场景缺少正需求人数资格字段")
        if selected - zero >= 200:
            eligible.append(index)
        else:
            ineligible.append({"index": index, "selected_time_utc": snapshot["selected_time_utc"],
                               "selected_aircraft": selected,
                               "positive_demand_aircraft": selected - zero,
                               "reason": "fewer_than_200_positive_demand_aircraft"})
    if role == "train":
        if ineligible:
            raise ValueError(f"原576训练母场景中有{len(ineligible)}个不足200正需求航空器；须先修订方案")
        indices = list(range(len(files)))
    elif role == "validation":
        if len(eligible) < 12:
            raise ValueError("Jan12合格验证母场景不足12个")
        # Fixed time ranks among eligible mothers; no model outcomes are consulted.
        indices = [eligible[int((i + 0.5) * len(eligible) / 12)] for i in range(12)]
    else:
        if ineligible:
            raise ValueError(f"Jan29预选20母场景中有{len(ineligible)}个不足200正需求航空器")
        indices = list(range(len(files)))
    if len(indices) != EXPECTED_MOTHERS[role] or len(set(indices)) != len(indices):
        raise ValueError("母场景数量不符合方案17")
    selected = [(files[i], snapshots[i]) for i in indices]
    times = [snap["selected_time_utc"] for _, snap in selected]
    if len(set(times)) != len(times):
        raise ValueError("母场景时刻重复")
    expected_dates = {"train": {"2024-01-08", "2024-01-09"},
                      "validation": {"2024-01-12"}, "test": {"2024-01-29"}}[role]
    if {time[:10] for time in times} != expected_dates:
        raise ValueError("母场景日期不符合方案17")
    return selected


def selected_counts(role: str, mother_index: int) -> tuple[int, ...]:
    if role == "validation":
        return VALIDATION_COUNTS
    if role == "test":
        return TEST_COUNTS
    values = [100]
    for band_index, band in enumerate(TRAIN_BANDS):
        # A fixed permutation for each band, then a balanced cycle over mothers.
        order = np.random.default_rng(SEED + band_index).permutation(band)
        first = (2 * mother_index) % len(order)
        values.extend((int(order[first]), int(order[(first + 1) % len(order)])))
    if len(values) != 7 or len(set(values)) != 7 or HELD_OUT.intersection(values):
        raise AssertionError("训练数量清单生成错误")
    return tuple(values)


def source_file(input_root: Path, reference: dict, role: str, selected_time: str) -> Path:
    date = selected_time[:10]
    path = input_root / role / date / Path(reference["path"]).name
    if not path.is_file() or sha256(path) != reference["sha256"]:
        raise ValueError("航空母快照缺失或SHA不符：" + str(path))
    return path


def detail_gzip(path: Path, rows: list[dict]) -> str:
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as text:
            for row in rows:
                text.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)
    return sha256(path)


def build_dataset(source_record: Path, input_root: Path, output_dir: Path, role: str) -> dict:
    record = json.loads(source_record.read_text(encoding="utf-8"))
    mothers = chosen_mothers(record, role)
    source = record["source"]
    if source.get("seed") != SEED or source.get("environment_profile") != "europe_v1":
        raise ValueError("母场景随机种子或欧洲环境来源不符")
    model = source["demand_model"]
    if source.get("max_record_age_seconds") != 60 or model.get("window_seconds") != 60:
        raise ValueError("母场景观测年龄或业务统计窗口不符")
    contract = {"role": role, "source_record_sha256": sha256(source_record),
                "input_root": str(input_root.resolve()), "seed": SEED,
                "loads": list(LOADS), "held_out": sorted(HELD_OUT),
                "mother_count": len(mothers)}
    if output_dir.exists():
        if (output_dir / "record.json").exists():
            raise FileExistsError("已完成数据集不得覆盖")
        saved_contract = output_dir / "input_contract.json"
        if not saved_contract.is_file() or json.loads(saved_contract.read_text(encoding="utf-8")) != contract:
            raise ValueError("续建输入合同缺失或变化")
    else:
        output_dir.mkdir(parents=True)
        write_json(output_dir / "input_contract.json", contract)
    references, snapshots, failures = [], [], []
    per_n = Counter()
    input_bindings = []
    progress = output_dir / "preparation_status.json"
    write_json(progress, {"status": "building", "role": role, "built_scenes": 0,
                          "planned_scenes": EXPECTED_MOTHERS[role] * (7 if role == "train" else 6 if role == "validation" else 11) * 2})
    try:
        for mother_index, (reference, mother) in enumerate(mothers):
            selected_time = mother["selected_time_utc"]
            csv_path = source_file(input_root, reference, role, selected_time)
            mother_id = f"{role}_{mother_index:05d}_{hashlib.sha256(selected_time.encode()).hexdigest()[:12]}"
            input_bindings.append({"mother_scene_id": mother_id, "selected_time_utc": selected_time,
                                   "path": str(csv_path), "sha256": reference["sha256"]})
            checkpoint = output_dir / f"mother_{mother_index:05d}.json"
            if checkpoint.is_file():
                prior = json.loads(checkpoint.read_text(encoding="utf-8"))
                if (prior.get("mother_scene_id") != mother_id
                        or prior.get("source_sha256") != reference["sha256"]
                        or len(prior.get("references", [])) != len(selected_counts(role, mother_index)) * 2):
                    raise ValueError("续建母场景检查点不匹配")
                for entry in prior["references"]:
                    if sha256(Path(entry["path"])) != entry["sha256"]:
                        raise ValueError("续建场景文件SHA不匹配：" + entry["path"])
                    detail = entry["observation_detail"]
                    if sha256(Path(detail["path"])) != detail["sha256"]:
                        raise ValueError("续建逐机覆盖细节SHA不匹配")
                    per_n[entry["beam_count"]] += 1
                references.extend(prior["references"])
                snapshots.extend(prior["snapshots"])
                continue
            mother_refs, mother_snaps = [], []
            for n in selected_counts(role, mother_index):
                try:
                    arrays, provenance = build_aviation_scenes(
                        [csv_path], [selected_time], source_format="csv_snapshot",
                        units={**CSV_UNITS, "persons": "count"}, demand_model=model, seed=SEED,
                        max_record_age_seconds=60, environment_profile="europe_v1",
                        partition_role=role, window_seconds=60,
                        beam_count=n, min_positive_aircraft=200)
                except AviationSceneError as exc:
                    failures.append({"mother_scene_id": mother_id, "beam_count": n,
                                     "error": str(exc), "diagnostics": exc.diagnostics})
                    raise
                base = arrays[0]
                snapshot = provenance["snapshots"][0]
                observations = snapshot.pop("source_observations")
                detail = output_dir / f"observations_{mother_index:05d}_n{n:03d}.jsonl.gz"
                detail_sha = detail_gzip(detail, observations)
                for load in LOADS:
                    data = dict(base)
                    data["rate"] = np.asarray(base["rate"], dtype=np.float64) * load
                    scene_id = f"scene_{mother_index:05d}_n{n:03d}_l{int(load * 100):03d}"
                    target = output_dir / f"{scene_id}.npz"
                    pending = target.with_suffix(".tmp.npz")
                    np.savez_compressed(pending, **data)
                    pending.replace(target)
                    total = snapshot["total_demand_bps"] * load
                    covered = float(data["rate"].sum())
                    uncovered = snapshot["uncovered_demand_bps"] * load
                    if not np.isclose(total, covered + uncovered, rtol=1e-12, atol=1e-5):
                        raise RuntimeError("负载缩放后需求汇总不守恒")
                    entry = {"path": str(target), "sha256": sha256(target), "scene_id": scene_id,
                             "mother_scene_id": mother_id, "beam_count": n, "load_multiplier": load,
                             "partition_role": role, "selected_time_utc": selected_time,
                             "source_sha256": reference["sha256"],
                             "observation_detail": {"path": str(detail), "sha256": detail_sha,
                                                    "base_load_multiplier": 1.0,
                                                    "rule": "multiply each base aircraft and beam demand by load_multiplier"},
                             "total_demand_bps": total, "covered_demand_bps": covered,
                             "uncovered_demand_bps": uncovered,
                             "covered_demand_fraction": covered / total if total else 0.0,
                             "covered_aircraft_fraction": snapshot["covered_aircraft_fraction"]}
                    references.append(entry)
                    mother_refs.append(entry)
                    compact = {key: entry[key] for key in (
                        "scene_id", "mother_scene_id", "beam_count", "load_multiplier",
                        "selected_time_utc", "total_demand_bps", "covered_demand_bps",
                        "uncovered_demand_bps", "covered_demand_fraction",
                        "covered_aircraft_fraction")}
                    snapshots.append(compact)
                    mother_snaps.append(compact)
                    per_n[n] += 1
            write_json(checkpoint, {"mother_scene_id": mother_id,
                                    "source_sha256": reference["sha256"],
                                    "references": mother_refs, "snapshots": mother_snaps})
            if (mother_index + 1) % 10 == 0 or mother_index + 1 == len(mothers):
                write_json(progress, {"status": "building", "role": role,
                                      "completed_mothers": mother_index + 1,
                                      "total_mothers": len(mothers), "built_scenes": len(references),
                                      "last_checked_utc": datetime.now(timezone.utc).isoformat()})
        expected = EXPECTED_MOTHERS[role] * (7 if role == "train" else 6 if role == "validation" else 11) * 2
        if len(references) != expected:
            raise RuntimeError("场景数量不足，不能发布不完整数据集")
        if role in ("train", "validation") and any(n in HELD_OUT for n in per_n):
            raise RuntimeError("留出束数进入训练或验证")
        profile = profile_metadata("europe_v1")
        dataset = {"id": f"variable_beams_{role}_20261004", "name": f"S2-P17_{role}",
                   "path": str(output_dir), "count": len(references),
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "source": {"type": "aviation_variable_beams_v1", "experiment_id": "S2-P17",
                              "partition_role": role,
                              "source_dates": sorted({item["selected_time_utc"][:10] for item in input_bindings}),
                              "snapshot_dates": sorted({item["selected_time_utc"][:10] for item in input_bindings}),
                              "seed": SEED, "demand_model": model, "load_multipliers": list(LOADS),
                              "mother_scene_count": len(mothers), "mother_scene_inputs": input_bindings,
                              "mother_record_path": str(source_record), "mother_record_sha256": sha256(source_record),
                              "selection_rule": "all 576 eligible train; twelve fixed time ranks among eligible Jan12; fixed 20 eligible Jan29",
                              "positive_demand_eligibility": 200,
                              "held_out_beam_counts": sorted(HELD_OUT),
                              "beam_count_distribution": dict(sorted(per_n.items())),
                              "coverage": {"algorithm": "demand-weighted farthest-point medoids; nearest exclusive association",
                                           "beam_full_width_degrees": 0.5},
                              "snapshots": snapshots},
                   "files": references, "profile": "europe_v1",
                   "environment_version": profile["environment_version"],
                   "environment_profile": profile,
                   "units": {"rate": "bit/s", "beam_location": "degrees",
                             "covered_demand_bps": "bit/s"}}
        write_json(output_dir / "record.json", dataset)
        summary = {"status": "prepared_inputs_only_no_policy_or_training", "role": role,
                   "mother_count": len(mothers), "derived_scene_count": len(references),
                   "beam_count_distribution": dict(sorted(per_n.items())),
                   "record_sha256": sha256(output_dir / "record.json"),
                   "source_record_sha256": sha256(source_record),
                   "source_input_sha256": {item["mother_scene_id"]: item["sha256"] for item in input_bindings},
                   "output_directory": str(output_dir)}
        write_json(output_dir / "data_manifest.json", summary)
        write_json(progress, {"status": "completed_inputs_only", **summary})
        return summary
    except BaseException as exc:
        write_json(progress, {"status": "failed", "role": role,
                              "built_scenes": len(references), "failures": failures,
                              "error": repr(exc), "at_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="directory containing train/validation/test/YYYY-MM-DD/*.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation", "test"), required=True)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.source_record, args.input_root, args.output_dir, args.role),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
