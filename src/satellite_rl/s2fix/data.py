"""S2-FIX ordinary JSON cases/layouts and FLOW-AO dataset construction."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import time

import numpy as np

from satellite_rl.coverage.s1fix.data import Frame, load_day, stratified
from satellite_rl.coverage.s1fix.geom import rays, validate_layout
from satellite_rl.coverage.s1fix.skeleton import solve
from satellite_rl.shared_interface.content import _decode, ARRAY_DTYPES
from satellite_rl.shared_interface.stages import make_case, make_layout, UniformStage2Input

def cone_omega(widths):
    return float(np.sum(2 * np.pi * (1 - np.cos(np.deg2rad(np.asarray(widths)) / 2))))


INPUTS = Path("D:/paper/outputs/experiments/s2_width26_20261014/collected/inputs")
FRAMES = Path("D:/paper/outputs/experiments/s1_fix_20261017/data")
SETS = {"train_d864_n150": ("train", 86400000, 150),
        "dev_d864_n150": ("development", 86400000, 150),
        "dev_d432_n150": ("development", 43200000, 150),
        "dev_d216_n150": ("development", 21600000, 150),
        "dev_d864_n200": ("development", 86400000, 200),
        "conf_d864_n150": ("test", 86400000, 150)}


def _json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def save_record(path, value):
    Path(path).write_text(json.dumps(value, default=_json_value, ensure_ascii=False), encoding="utf-8")


def read_record(path):
    value = _decode(json.loads(Path(path).read_text(encoding="utf-8")))
    for name, dtype in ARRAY_DTYPES.items():
        if name in value: value[name] = np.asarray(value[name], dtype=dtype)
    if "stack_id" in value: value["stack_id"] = np.asarray(value["stack_id"], dtype=np.int64)
    return value


def read_records(data_dir, set_name):
    path = Path(data_dir) / set_name / "manifest.json"
    records = json.loads(path.read_text(encoding="utf-8"))["rows"]
    for record in records:
        for key in ("case_path", "layout_path"):
            record[key] = str(path.parent / record[key])
    return records


def load_input(record):
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_V2
    return UniformStage2Input(read_record(record["case_path"]),
                              read_record(record["layout_path"]), None, PHYSICS_V2, plain=True)


def source_cases(role, inputs_dir=INPUTS, frames_dir=FRAMES):
    root = Path(inputs_dir)
    records = json.loads((root / f"{role}_published.json").read_text(encoding="utf-8"))["rows"]
    cases = [read_record(root / "objects" / Path(row["case_ref"]["path"]).name) for row in records]
    if role == "test":
        for date in ("2024-01-15", "2024-01-18"):
            for frame in stratified(load_day(frames_dir, date)):
                utc = datetime.fromtimestamp(frame.time, timezone.utc).isoformat().replace("+00:00", "Z")
                metadata = dict(case_id=f"conf48:{utc}", actual_role="test", source_role="confirmation",
                    original_mother_id=f"{date}:{frame.index}", UTC=utc,
                    source_asset_SHA=f"frames_{date}.npz", source=dict(frame_index=frame.index))
                cases.append(make_case(metadata, [str(int(i)) for i in frame.icao], frame.positions, plain=True))
    return cases


def build_scene(case, request_bps, N_budget, *, time_limit=120, max_iters=None):
    """Rebuild aircraft requests and geometry, without retaining the source layout."""
    started = time.perf_counter()
    metadata = {name: case[name] for name in ("case_id", "actual_role", "source_role", "original_mother_id", "UTC", "source_asset_SHA", "source")}
    metadata["case_id"] = f"{case['case_id']}:s2fix:{request_bps}:n{N_budget}"
    case = make_case(metadata, case["aircraft_ids"], case["lat_lon_deg"], request_bps=request_bps, plain=True)
    n = len(case["aircraft_ids"])
    frame = Frame(case["UTC"][:10], 0., 0, np.asarray(case["aircraft_ids"]), case["lat_lon_deg"], np.zeros(n), np.zeros(n))
    cap = int(1125000000 // request_bps)
    result = solve(frame, N_budget, spacing=True, effort=1, time_limit=time_limit,
                   per_beam_cap=cap, objective="narrow", stacking=True, max_iters=max_iters)
    if not result.feasible: raise ValueError("FLOW-AO found no legal layout")
    output = result.layout
    if not validate_layout(output, rays(frame.positions), N_budget): raise ValueError("invalid FLOW-AO layout")
    counts = np.bincount(output.owner, minlength=len(output.widths))
    ids = [f"flow:{i:04d}" for i in output.beam_ids]
    anchors = [case["aircraft_ids"][np.flatnonzero(output.owner == j)[0]] for j in range(len(ids))]
    layout = make_layout(case, beam_ids=ids, source_slot_ids=ids, anchor_aircraft_ids=anchors,
        centres_lat_lon_deg=output.centers_latlon, full_width_deg=output.widths,
        owner_aircraft_to_beam_id=[[aid, ids[j]] for aid, j in zip(case["aircraft_ids"], output.owner)],
        Ncap=N_budget, geometry_Omega_cap_sr=cone_omega(np.ones(N_budget)),
        geometry_Omega_sr=cone_omega(output.widths), geometry_audit=dict(method="FLOW-AO-v2", spacing=True),
        source_and_cost=dict(seconds=result.t_total, reserve=result.reserve, per_beam_cap=cap),
        stack_id=output.stack_id, plain=True)
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_V2
    UniformStage2Input(case, layout, None, PHYSICS_V2, plain=True)
    stacks = output.stack_id[output.stack_id >= 0]
    check = dict(case_id=case["case_id"], date=case["UTC"][:10], n=n, N=len(ids), K=cap,
        stack_count=len(np.unique(stacks)), stacked_beams=len(stacks),
        max_beam_aircraft=int(counts.max()), over_cap_beams=int(np.count_nonzero(counts > cap)),
        mean_width_deg=float(output.widths.mean()), max_width_deg=float(output.widths.max()),
        seconds=time.perf_counter() - started, UniformStage2Input_ok=True)
    return case, layout, check


def _generate(job):
    index, set_name, case, request, budget, destination = job
    case, layout, check = build_scene(case, request, budget)
    directory = Path(destination) / set_name / "objects"
    directory.mkdir(parents=True, exist_ok=True)
    case_name, layout_name = f"{index:04d}_case.json", f"{index:04d}_layout.json"
    save_record(directory / case_name, case)
    save_record(directory / layout_name, layout)
    record = dict(case_id=case["case_id"], date=case["UTC"][:10], set_name=set_name,
                  case_path=f"objects/{case_name}", layout_path=f"objects/{layout_name}")
    return record, check


def build_set(set_name, output_dir, workers=1, limit=None, inputs_dir=INPUTS, frames_dir=FRAMES):
    role, request, budget = SETS[set_name]
    cases = source_cases(role, inputs_dir, frames_dir)
    if limit is not None: cases = cases[:limit]
    jobs = [(i, set_name, case, request, budget, str(output_dir)) for i, case in enumerate(cases)]
    if workers == 1:
        outputs = list(map(_generate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_generate, jobs))
    directory = Path(output_dir) / set_name
    directory.mkdir(parents=True, exist_ok=True)
    save_record(directory / "manifest.json", dict(set_name=set_name, rows=[row for row, _ in outputs]))
    checks = [check for _, check in outputs]
    with (directory / "checks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
        writer.writeheader(); writer.writerows(checks)
    return checks
