"""Freeze S1-02 Jan28/Jan30 source assets and UTC scene times before inspection.

This only records official metadata and the predetermined 100-time grid. It
does not download the large traces, prepare scenes, or inspect results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


SOURCE_REPO = "https://api.github.com/repos/adsblol/globe_history_2024"
PREFERRED_URL = "https://raw.githubusercontent.com/adsblol/globe_history_2024/main/PREFERRED_RELEASES.txt"
LICENSE_URL = "https://raw.githubusercontent.com/adsblol/globe_history_2024/main/LICENSE-ODbL.txt"
DATES = {"A": "2024-01-28", "B": "2024-01-30"}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "satellite-rl-s1-02-source-freeze"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=30) as response:
                return response.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen file has changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def freeze(root: Path, stage: str, preferred: bytes, license_text: bytes) -> dict:
    day = DATES[stage]
    tag = "v" + day.replace("-", ".") + "-planes-readsb-staging-0"
    archive = tag + ".tar"
    asset_url = f"https://github.com/adsblol/globe_history_2024/releases/download/{tag}/{archive}"
    api_url = SOURCE_REPO + "/releases/tags/" + tag
    target = root / stage / "data"
    release_raw = fetch(api_url)
    release = json.loads(release_raw)
    assets = [item for item in release["assets"] if item["name"] == archive]
    if (release["tag_name"] != tag or len(assets) != 1
            or assets[0]["browser_download_url"] != asset_url
            or asset_url.encode() not in preferred):
        raise RuntimeError(f"Official source mismatch for {day}")
    indices = [int((index + 0.5) * 288 / 100) for index in range(100)]
    if len(set(indices)) != 100 or indices[-1] >= 288:
        raise RuntimeError("100-scene time selection is invalid")
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    times = [(start + timedelta(minutes=2 + 5 * index)).isoformat(timespec="milliseconds")
             .replace("+00:00", "Z") for index in indices]
    contract = {
        "status": "frozen_before_source_scene_or_result_inspection",
        "group": "S1-02", "stage": stage,
        "role": "independent_test_A" if stage == "A" else "independent_test_B",
        "date_utc": day, "source_instance": "planes-readsb-staging-0",
        "candidate_rule": "288 UTC times: day 00:02 plus 5*i minutes, i=0..287",
        "selection_rule": "index=floor((j+0.5)*288/100), j=0..99; no replacement after data or score inspection",
        "indices": indices, "selected_times_utc": times,
        "max_record_age_seconds": 60, "seed": 51001,
        "environment_profile": "europe_v1", "beam_count": 100,
        "training_dates_utc": ["2024-01-08", "2024-01-09"],
        "validation_date_utc": "2024-01-12",
        "previous_test_date_utc": "2024-01-22",
        "other_project_reserved_date_utc": "2024-01-29",
        "gate": "all 100 predetermined times pass the frozen G07 aviation adapter; no outcome-based replacement",
    }
    contract_bytes = (json.dumps(contract, ensure_ascii=False, indent=2) + "\n").encode()
    pin = {
        "status": "official_source_pinned_not_downloaded",
        "group": "S1-02", "stage": stage, "date_utc": day,
        "release_tag": tag, "release_api": api_url,
        "release_api_sha256": sha256(release_raw),
        "archive_name": archive, "archive_url": asset_url,
        "expected_bytes": int(assets[0]["size"]), "asset_id": assets[0]["id"],
        "asset_created_at": assets[0].get("created_at"),
        "asset_updated_at": assets[0].get("updated_at"),
        "preferred_url": PREFERRED_URL, "preferred_sha256": sha256(preferred),
        "license_url": LICENSE_URL, "license_sha256": sha256(license_text),
        "selection_contract_sha256": sha256(contract_bytes),
    }
    pin_bytes = (json.dumps(pin, ensure_ascii=False, indent=2) + "\n").encode()
    for name, payload in (("release_api.json", release_raw),
                          ("PREFERRED_RELEASES.txt", preferred),
                          ("LICENSE-ODbL-official.txt", license_text),
                          ("selection_contract.json", contract_bytes),
                          ("source_preflight.json", pin_bytes)):
        write_once(target / name, payload)
    return {"stage": stage, "date_utc": day, "asset_id": pin["asset_id"],
            "expected_bytes": pin["expected_bytes"], "scenes": len(times),
            "contract_sha256": pin["selection_contract_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("D:/paper/outputs/experiments/first_stage_s1_02_20261004"))
    args = parser.parse_args()
    preferred = fetch(PREFERRED_URL)
    license_text = fetch(LICENSE_URL)
    for stage in DATES:
        print(json.dumps(freeze(args.output, stage, preferred, license_text), ensure_ascii=False))


if __name__ == "__main__":
    main()
