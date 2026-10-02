"""Freeze a new stage 3 independent aviation date before inspecting its scenes."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import time

HERE = Path(os.environ.get("STAGE3_DATA_ROOT", "D:/paper/outputs/experiments/stage3_power_20261003/data"))
ROOT = Path("D:/paper")
DAY = os.environ.get("STAGE3_DATE", "2024-01-25")
TAG = f"v{DAY.replace('-', '.')}-planes-readsb-staging-0"
ASSET = TAG + ".tar"
ASSET_URL = f"https://github.com/adsblol/globe_history_2024/releases/download/{TAG}/{ASSET}"
API_URL = f"https://api.github.com/repos/adsblol/globe_history_2024/releases/tags/{TAG}"
PREFERRED_URL = "https://raw.githubusercontent.com/adsblol/globe_history_2024/main/PREFERRED_RELEASES.txt"
LICENSE_URL = "https://raw.githubusercontent.com/adsblol/globe_history_2024/main/LICENSE-ODbL.txt"
G07_RECORD = ROOT / "outputs/workbench/scenes/scenes_20260928_142054_9bc86790/record.json"
G09_REFERENCE = ROOT / "outputs/experiments/europe_natural_state_20260930/prepare_snapshots_reference.py"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        stable = lambda item: {key: content for key, content in item.items()
                               if key not in ("frozen_utc", "checked_utc")}
        if stable(old) != stable(value):
            raise RuntimeError(f"Existing frozen evidence differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    day = datetime.fromisoformat(DAY).replace(tzinfo=timezone.utc)
    indices = [int((j + 0.5) * 288 / 100) for j in range(100)]
    times = [(day + timedelta(minutes=2 + 5 * i)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
             for i in indices]
    if len(set(indices)) != 100 or len(set(times)) != 100 or indices[-1] >= 288:
        raise RuntimeError("Preselected 100-time grid is invalid")
    g07 = json.loads(G07_RECORD.read_text(encoding="utf-8"))["source"]
    model = g07["demand_model"]
    if (g07["seed"] != 51001 or g07["max_record_age_seconds"] != 60
            or model["window_seconds"] != 60 or model["persons_per_aircraft"] != 180
            or model["satellite_access_fraction"] != 0.2 or model["active_fraction"] != 0.3
            or model["rate_mbps_per_active_person"] != 2.0 or model["aircraft_base_mbps"] != 0
            or model["sampling"] != "binomial" or model["lognormal_sigma"] != 0.35):
        raise RuntimeError("Common G07 demand contract differs")
    contract = {
        "status": "frozen_before_stage3_scene_inspection_or_model_results",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "group": "S3-P01", "role": "common_independent_test", "date_utc": DAY,
        "upstream_instance": "planes-readsb-staging-0",
        "candidate_rule": "288 UTC times: day 00:02 plus 5*i minutes, i=0..287",
        "selection_rule": "index=floor((j+0.5)*288/100), j=0..99; no replacements after inspecting data or scores",
        "indices": indices, "selected_times_utc": times,
        "max_record_age_seconds": 60, "seed": 51001, "demand_model": model,
        "environment_profile": "europe_v1", "beam_count": 100,
        "geometry_rule": "G07/G09 adapter: 100 demand-weighted farthest medoids; 0.25 degree half-width, 10 degree elevation",
        "source_format": "readsb_trace to csv_snapshot, then frozen G07 aviation adapter",
        "g07_test_record_sha256": digest(G07_RECORD.read_bytes()),
        "g09_preparation_reference_sha256": digest(G09_REFERENCE.read_bytes()),
        "training_dates_utc": ["2024-01-08", "2024-01-09"],
        "validation_date_utc": "2024-01-12",
        "previously_used_development_or_test_dates": ["2024-01-15", "2024-01-18", "2024-01-19", "2024-01-22", "2024-01-23", "2024-01-24"],
        "gate": "all 100 predetermined times have >=100 distinct eligible positions and pass 100-beam adapter; never replace failed times based on outcome",
    }
    write_json(HERE / "selection_contract.json", contract)
    source_files = ((API_URL, "release_api.json"), (PREFERRED_URL, "PREFERRED_RELEASES.txt"),
                    (LICENSE_URL, "LICENSE-ODbL-official.txt"))
    for url, name in source_files:
        request = urllib.request.Request(url, headers={"User-Agent": "satellite-rl-stage3-source-audit"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read()
                (HERE / name).write_bytes(payload)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)
    release = json.loads((HERE / "release_api.json").read_text(encoding="utf-8"))
    assets = [asset for asset in release["assets"] if asset["name"] == ASSET]
    preferred = (HERE / "PREFERRED_RELEASES.txt").read_text(encoding="utf-8")
    if (len(assets) != 1 or release["tag_name"] != TAG
            or assets[0]["browser_download_url"] != ASSET_URL or ASSET_URL not in preferred):
        raise RuntimeError("Official release or preferred asset differs")
    asset = assets[0]
    pin = {
        "status": "official_source_pinned_not_downloaded", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "date_utc": DAY, "instance": "planes-readsb-staging-0", "release_tag": TAG,
        "archive_name": ASSET, "archive_url": ASSET_URL,
        "expected_bytes": int(asset["size"]), "asset_id": asset["id"],
        "asset_created_at": asset.get("created_at"), "asset_updated_at": asset.get("updated_at"),
        "release_api": API_URL, "release_api_sha256": digest((HERE / "release_api.json").read_bytes()),
        "preferred_url": PREFERRED_URL, "preferred_sha256": digest((HERE / "PREFERRED_RELEASES.txt").read_bytes()),
        "license_url": LICENSE_URL, "license_sha256": digest((HERE / "LICENSE-ODbL-official.txt").read_bytes()),
        "selection_contract_sha256": digest((HERE / "selection_contract.json").read_bytes()),
    }
    write_json(HERE / "source_preflight.json", pin)
    print(json.dumps({"status":pin["status"],"archive_name":ASSET,"expected_bytes":pin["expected_bytes"],
                      "selected_times":len(times)},ensure_ascii=False))


if __name__ == "__main__":
    main()
