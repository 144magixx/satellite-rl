"""Freeze official ADSB.lol assets and scene UTC times before reading traces."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from scripts.stage1_variable.ops import LOCAL


API = "https://api.github.com/repos/adsblol/globe_history_2024"
PREFERRED_URL = "https://raw.githubusercontent.com/adsblol/globe_history_2024/main/PREFERRED_RELEASES.txt"
LICENSE_URL = "https://raw.githubusercontent.com/adsblol/globe_history_2024/main/LICENSE-ODbL.txt"
DATES = {"validation": ("2024-02-03",), "test": ("2024-02-04", "2024-02-05"),
         "sequence": ("2024-02-06",)}
COUNTS = {"validation": 20, "test": 50, "sequence": 120}


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "satellite-rl-s1-v10-source-freeze"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"frozen source changed: {path}")
    else:
        path.write_bytes(payload)


def selected_times(day: str, role: str) -> tuple[list[int], list[str]]:
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    if role == "sequence":
        # Twenty nonoverlapping six-minute segments, starting at 00:12 UTC.
        minutes = [12 + 72 * segment + offset for segment in range(20)
                   for offset in range(6)]
    else:
        count = COUNTS[role]
        minutes = [2 + 5 * int((index + .5) * 288 / count)
                   for index in range(count)]
    stamps = [(start + timedelta(minutes=minute)).isoformat(timespec="milliseconds")
              .replace("+00:00", "Z") for minute in minutes]
    return minutes, stamps


def freeze_day(role: str, day: str, preferred: bytes, license_text: bytes):
    tag = "v" + day.replace("-", ".") + "-planes-readsb-staging-0"
    archive = tag + ".tar"
    url = f"https://github.com/adsblol/globe_history_2024/releases/download/{tag}/{archive}"
    if url.encode() not in preferred:
        raise RuntimeError(f"official preferred release missing: {day}")
    release_url = API + "/releases/tags/" + tag
    release_raw = fetch(release_url)
    release = json.loads(release_raw)
    assets = [a for a in release.get("assets", []) if a.get("name") == archive]
    if (release.get("tag_name") != tag or len(assets) != 1 or
            assets[0].get("browser_download_url") != url):
        raise RuntimeError(f"official release asset mismatch: {day}")
    minutes, times = selected_times(day, role)
    if len(set(times)) != len(times) or max(minutes) >= 24 * 60:
        raise RuntimeError("preselected UTC times invalid")
    directory = LOCAL / "source" / role / day
    contract = {
        "status": "frozen_before_aircraft_or_outcome_inspection",
        "group": "S1-V10", "role": role, "date_utc": day,
        "source_instance": "planes-readsb-staging-0", "seed": 51001,
        "environment_profile": "europe_v1", "max_record_age_seconds": 60,
        "count": len(times), "minute_offsets": minutes, "selected_times_utc": times,
        "selection_rule": "20 six-minute segments at 00:12+72*j minute" if role == "sequence"
                          else f"floor((j+0.5)*288/{len(times)}) on 00:02+5*i minute grid",
        "no_outcome_based_replacement": True,
        "demand_multipliers": [0.75, 1.0, 1.25],
        "previous_training_dates": ["2024-01-08", "2024-01-09"],
        "previous_development_dates": ["2024-01-12"],
        "date_collision_scan": "2026-10-09 rg over EXPERIMENT_STATUS.md/docs/outputs/experiments: no 2024-02-03..06 hits",
    }
    contract_bytes = (json.dumps(contract, ensure_ascii=False, indent=2) + "\n").encode()
    pin = {
        "status": "official_source_pinned_not_downloaded", "group": "S1-V10",
        "role": role, "date_utc": day, "release_tag": tag,
        "release_api": release_url, "release_api_sha256": sha(release_raw),
        "archive_name": archive, "archive_url": url,
        "expected_bytes": int(assets[0]["size"]), "asset_id": assets[0]["id"],
        "preferred_url": PREFERRED_URL, "preferred_sha256": sha(preferred),
        "license_url": LICENSE_URL, "license_sha256": sha(license_text),
        "selection_contract_sha256": sha(contract_bytes),
    }
    files = {"release_api.json": release_raw, "PREFERRED_RELEASES.txt": preferred,
             "LICENSE-ODbL-official.txt": license_text,
             "selection_contract.json": contract_bytes,
             "source_preflight.json": (json.dumps(pin, ensure_ascii=False, indent=2) + "\n").encode()}
    for name, content in files.items():
        write_once(directory / name, content)
    return {"role": role, "date": day, "scenes": len(times),
            "asset_id": pin["asset_id"], "bytes": pin["expected_bytes"],
            "pin_sha256": sha(files["source_preflight.json"])}


def main():
    preferred, license_text = fetch(PREFERRED_URL), fetch(LICENSE_URL)
    rows = [freeze_day(role, day, preferred, license_text)
            for role, days in DATES.items() for day in days]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
