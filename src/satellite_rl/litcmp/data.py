"""Directory-based datasets and the corrected source-object lookup."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os

from satellite_rl.s2fix import data as s2data
from satellite_rl.s2fix.data import INPUTS, FRAMES, read_record, save_record
from satellite_rl.coverage.s1fix.data import load_day, stratified
from satellite_rl.shared_interface.stages import make_case

SETS = dict(s2data.SETS, conf_d432_n150=("test", 43200000, 150),
            conf_d216_n150=("test", 21600000, 150))


def source_cases(role, inputs_dir=INPUTS, frames_dir=FRAMES):
    root = Path(inputs_dir)
    role = "test" if role == "conf" else role
    records = json.loads((root / f"{role}_published.json").read_text(encoding="utf-8"))["rows"]
    cases = [read_record(root / "objects" / row["case_ref"]["sha256"]) for row in records]
    if role == "test":
        for date in ("2024-01-15", "2024-01-18"):
            for frame in stratified(load_day(frames_dir, date)):
                utc = datetime.fromtimestamp(frame.time, timezone.utc).isoformat().replace("+00:00", "Z")
                metadata = dict(case_id=f"conf48:{utc}", actual_role="test", source_role="confirmation",
                    original_mother_id=f"{date}:{frame.index}", UTC=utc,
                    source_asset_SHA=f"frames_{date}.npz", source=dict(frame_index=frame.index))
                cases.append(make_case(metadata, [str(int(i)) for i in frame.icao], frame.positions, plain=True))
    return cases


def records(set_dir):
    directory = Path(set_dir)
    rows = [row for path in sorted(directory.glob("manifest*.json"))
            for row in json.loads(path.read_text(encoding="utf-8"))["rows"]]
    rows.sort(key=lambda row: row['case_path'])
    for row in rows:
        for key in ("case_path", "layout_path"):
            row[key] = str(directory / row[key])
    return rows


def load_set(set_dir):
    return [(read_record(row["case_path"]), read_record(row["layout_path"]),
             row["case_id"], row["date"]) for row in records(set_dir)]


def shard_indices(items, shard=None):
    i, k = map(int, shard.split("/")) if shard else (0, 1)
    return [(index, item) for index, item in enumerate(items) if index % k == i]


def suffix(shard):
    return "_shard" + shard.replace("/", "of") if shard else ""


def worker_init():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def parallel(function, jobs, workers):
    worker_init()
    if workers == 1:
        return list(map(function, jobs))
    with ProcessPoolExecutor(workers, initializer=worker_init) as pool:
        return list(pool.map(function, jobs))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_set(directory, set_name, outputs, shard=None):
    directory = Path(directory) / set_name
    directory.mkdir(parents=True, exist_ok=True)
    save_record(directory / f"manifest{suffix(shard)}.json",
                dict(set_name=set_name, rows=[row for row, _ in outputs]))
    write_csv(directory / f"checks{suffix(shard)}.csv", [check for _, check in outputs])


def build_set(set_name, output_dir, workers=1, limit=None, shard=None,
              inputs_dir=INPUTS, frames_dir=FRAMES):
    s2data.source_cases = source_cases
    s2data.SETS.update(SETS)
    if shard is None:
        return s2data.build_set(set_name, output_dir, workers, limit, inputs_dir, frames_dir)
    role, request, budget = SETS[set_name]
    cases = source_cases(role, inputs_dir, frames_dir)[:limit]
    jobs = [(i, set_name, case, request, budget, str(output_dir))
            for i, case in shard_indices(cases, shard)]
    outputs = parallel(s2data._generate, jobs, workers)
    write_set(output_dir, set_name, outputs, shard)
    return [check for _, check in outputs]
