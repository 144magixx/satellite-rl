"""Check S1-01 versus G07 aviation adapter source and sampled output arrays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path("D:/paper")
DATA = ROOT / "outputs/experiments/first_stage_s1_02_20261004"
G07 = ROOT / "outputs/workbench/source_snapshots/01457f245bf0ab0e23873616e44975693014e3d0/src"
S101_FILE = DATA / "preflight/s101_aviation_frozen.py"
G07_FILE = G07 / "satellite_rl/workbench/aviation.py"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    old = G07_FILE.read_bytes()
    current = S101_FILE.read_bytes()
    old_normal = old.replace(b"\r\n", b"\n")
    current_normal = current.replace(b"\r\n", b"\n")
    if old_normal != current_normal:
        raise RuntimeError("S1-01 and G07 aviation adapter code differ beyond line endings")
    sys.path.insert(0, str(G07))
    from satellite_rl.workbench.aviation import build_aviation_scenes

    source = json.loads((DATA / "development/data/source_train/record.json").read_text(encoding="utf-8"))
    model = source["source"]["demand_model"]
    units = {"timestamp": "utc_iso8601", "lat": "degrees", "lon": "degrees",
             "altitude_m": "m", "persons": "count"}
    checks = []
    for stage in ("A", "B"):
        base = DATA / stage / "data"
        manifest = json.loads((base / "independent_test_manifest.json").read_text(encoding="utf-8"))
        for index in (0, 99):
            frame = manifest["csv_frames"]["frames"][index]
            csv = base / "snapshots/test" / manifest["date_utc"] / Path(frame["path"]).name
            arrays, _ = build_aviation_scenes(
                [csv], [frame["selected_time_utc"]], source_format="csv_snapshot",
                units=units, demand_model=model, seed=51001,
                max_record_age_seconds=60, environment_profile="europe_v1",
                partition_role="test", window_seconds=60)
            with np.load(base / "staged_test" / f"scene_{index:05d}.npz", allow_pickle=False) as saved:
                if set(saved.files) != set(arrays[0]):
                    raise RuntimeError("Aviation adapter array keys differ")
                differences = {}
                for key in saved.files:
                    stored = np.asarray(saved[key])
                    rebuilt = np.asarray(arrays[0][key])
                    if stored.shape != rebuilt.shape or stored.dtype != rebuilt.dtype:
                        raise RuntimeError("Aviation array shape or dtype differs: " + key)
                    if np.issubdtype(stored.dtype, np.number):
                        maximum = float(np.max(np.abs(stored - rebuilt)))
                        if not np.allclose(stored, rebuilt, rtol=1e-12, atol=1e-7):
                            raise RuntimeError("Aviation adapter numerical result differs: " + key)
                        differences[key] = maximum
                    elif not np.array_equal(stored, rebuilt):
                        raise RuntimeError("Aviation adapter identity differs: " + key)
            checks.append({"stage": stage, "scene_index": index,
                           "selected_time_utc": frame["selected_time_utc"],
                           "maximum_absolute_difference": differences})
    result = {"status": "sampled_adapter_arrays_consistent",
              "g07_file_sha256": sha(old), "s101_file_sha256": sha(current),
              "normalized_source_sha256": sha(old_normal),
              "normalized_source_identical": True, "scene_checks": checks,
              "scope": "Four endpoint scenes only; full 100-scene per-date SHA and conservation are recorded separately."}
    output = DATA / "preflight/adapter_equivalence.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "scene_checks": len(checks),
                      "normalized_source_sha256": result["normalized_source_sha256"]}))


if __name__ == "__main__":
    main()
