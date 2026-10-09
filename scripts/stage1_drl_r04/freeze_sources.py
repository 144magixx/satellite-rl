"""Freeze official Feb07–09 ADS-B release metadata and UTC scene selection.

This reads only public metadata. Large archives are downloaded later by a
workbench prepare task on the shared remote filesystem.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from scripts.stage1_drl_r04.ops import LOCAL


API = "https://api.github.com/repos/adsblol/globe_history_2024"
PREFERRED_URL = "https://raw.githubusercontent.com/adsblol/globe_history_2024/main/PREFERRED_RELEASES.txt"
LICENSE_URL = "https://raw.githubusercontent.com/adsblol/globe_history_2024/main/LICENSE-ODbL.txt"
DATES = {"validation": ("2024-02-07",),
         "test": ("2024-02-08", "2024-02-09")}
COUNT_PER_DAY = {"validation": 20, "test": 50}


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str) -> bytes:
    for attempt in range(5):
        try:
            request = Request(url, headers={"User-Agent": "satellite-rl-s1-d12a-source-freeze"})
            with urlopen(request, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def write_once(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != content:
            raise RuntimeError(f"frozen source metadata changed: {path}")
    else:
        path.write_bytes(content)


def selected_times(day: str, role: str):
    count = COUNT_PER_DAY[role]
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    minutes = [2 + 5 * int((index + .5) * 288 / count) for index in range(count)]
    stamps = [(start + timedelta(minutes=minute)).isoformat(timespec="milliseconds")
              .replace("+00:00", "Z") for minute in minutes]
    if len(set(stamps)) != count or max(minutes) >= 1440:
        raise ValueError("selected UTC times repeat or exceed day")
    return minutes, stamps


def freeze_day(role: str, day: str, preferred: bytes, license_text: bytes):
    tag = "v" + day.replace("-", ".") + "-planes-readsb-staging-0"
    archive = tag + ".tar"
    url = f"https://github.com/adsblol/globe_history_2024/releases/download/{tag}/{archive}"
    if url.encode("utf-8") not in preferred:
        raise RuntimeError(f"official preferred release missing: {day}")
    release_url = API + "/releases/tags/" + tag
    raw_release = fetch(release_url)
    release = json.loads(raw_release)
    assets = [asset for asset in release.get("assets", []) if asset.get("name") == archive]
    if (release.get("tag_name") != tag or len(assets) != 1 or
            assets[0].get("browser_download_url") != url):
        raise RuntimeError(f"official release asset differs: {day}")
    minutes, stamps = selected_times(day, role)
    contract = {"status": "frozen_before_aircraft_or_outcome_inspection",
                "group": "S1-D12A", "role": role, "date_utc": day,
                "source_instance": "planes-readsb-staging-0", "seed": 51001,
                "environment_profile": "europe_v1", "max_record_age_seconds": 60,
                "count": len(stamps), "minute_offsets": minutes,
                "selected_times_utc": stamps,
                "selection_rule": f"floor((j+0.5)*288/{len(stamps)}) on 00:02+5*i minute grid",
                "no_outcome_based_replacement": True,
                "demand_multipliers": [0.75, 1.0, 1.25],
                "excluded_training_dates": ["2024-01-08", "2024-01-09"],
                "excluded_development_dates": ["2024-01-12"],
                "excluded_s1_v10_test_dates": ["2024-02-03", "2024-02-04",
                                                 "2024-02-05", "2024-02-06"]}
    contract_bytes = (json.dumps(contract, ensure_ascii=False, indent=2) + "\n").encode()
    pin = {"status": "official_source_pinned_not_downloaded", "group": "S1-D12A",
           "role": role, "date_utc": day, "release_tag": tag,
           "release_api": release_url, "release_api_sha256": sha(raw_release),
           "archive_name": archive, "archive_url": url,
           "expected_bytes": int(assets[0]["size"]), "asset_id": assets[0]["id"],
           "preferred_url": PREFERRED_URL, "preferred_sha256": sha(preferred),
           "license_url": LICENSE_URL, "license_sha256": sha(license_text),
           "selection_contract_sha256": sha(contract_bytes)}
    files = {"release_api.json": raw_release, "PREFERRED_RELEASES.txt": preferred,
             "LICENSE-ODbL-official.txt": license_text,
             "selection_contract.json": contract_bytes,
             "source_preflight.json":
                 (json.dumps(pin, ensure_ascii=False, indent=2) + "\n").encode()}
    root = LOCAL / "source" / role / day
    for filename, content in files.items():
        write_once(root / filename, content)
    return {"role": role, "date": day, "count": len(stamps),
            "asset_id": pin["asset_id"], "expected_bytes": pin["expected_bytes"],
            "source_preflight_sha256": sha(files["source_preflight.json"]),
            "selection_contract_sha256": sha(contract_bytes)}


def main():
    preferred, license_text = fetch(PREFERRED_URL), fetch(LICENSE_URL)
    print(json.dumps([freeze_day(role, day, preferred, license_text)
                      for role, days in DATES.items() for day in days], ensure_ascii=False))


if __name__ == "__main__":
    main()
