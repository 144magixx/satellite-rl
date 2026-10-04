"""Resume a frozen ADSB.lol release download and record its full SHA-256.

Only source data is downloaded; this script does not create workbench tasks.
Run it as a persistent detached process for each S1-03 test date.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    pin_file = data_dir / "source_preflight.json"
    contract_file = data_dir / "selection_contract.json"
    pin = json.loads(pin_file.read_text(encoding="utf-8"))
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    for name, field in (("release_api.json", "release_api_sha256"),
                        ("PREFERRED_RELEASES.txt", "preferred_sha256"),
                        ("LICENSE-ODbL-official.txt", "license_sha256"),
                        ("selection_contract.json", "selection_contract_sha256")):
        if sha256(data_dir / name) != pin[field]:
            raise RuntimeError("Frozen source document differs: " + name)
    if (pin["group"] != "S1-03" or pin["stage"] != contract["stage"]
            or pin["date_utc"] != contract["date_utc"]
            or pin["archive_url"].encode() not in (data_dir / "PREFERRED_RELEASES.txt").read_bytes()
            or not pin["archive_url"].endswith("/" + pin["archive_name"])):
        raise RuntimeError("Official source pin and date contract disagree")
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / pin["archive_name"]
    partial = raw / (pin["archive_name"] + ".part")
    status_file = raw / "download_status.json"
    status = {"status": "running", "group": "S1-03", "stage": pin["stage"],
              "date_utc": pin["date_utc"], "url": pin["archive_url"],
              "asset_id": pin["asset_id"], "expected_bytes": pin["expected_bytes"],
              "source_preflight_sha256": sha256(pin_file),
              "path": str(archive), "started_utc": utc_now(), "attempts": []}
    if status_file.exists():
        old = json.loads(status_file.read_text(encoding="utf-8"))
        immutable = ("group", "stage", "date_utc", "url", "asset_id",
                     "expected_bytes", "source_preflight_sha256", "path")
        if any(old.get(key) != status[key] for key in immutable):
            raise RuntimeError("Existing download belongs to another frozen asset")
        status["started_utc"] = old.get("started_utc", status["started_utc"])
        status["attempts"] = old.get("attempts", [])
    save(status_file, status)
    try:
        if archive.exists():
            if archive.stat().st_size != pin["expected_bytes"]:
                raise RuntimeError("Existing archive has unexpected byte count")
        else:
            for attempt in range(1, 9):
                started = utc_now()
                command = ["curl", "--location", "--fail", "--silent", "--show-error",
                           "--connect-timeout", "20", "--max-time", "1800",
                           "--continue-at", "-", "--output", str(partial), pin["archive_url"]]
                result = subprocess.run(command, capture_output=True, text=True, timeout=1830)
                current_bytes = partial.stat().st_size if partial.exists() else 0
                status["attempts"].append({"number": attempt, "started_utc": started,
                                           "ended_utc": utc_now(), "curl_exit": result.returncode,
                                           "partial_bytes": current_bytes,
                                           "stderr_tail": result.stderr[-300:]})
                save(status_file, status)
                if result.returncode == 0 and current_bytes == pin["expected_bytes"]:
                    break
                if current_bytes > pin["expected_bytes"]:
                    raise RuntimeError("Partial source exceeds pinned release size")
                if attempt == 8:
                    raise RuntimeError("Pinned release did not finish after eight attempts")
                time.sleep(min(attempt * 5, 30))
            partial.replace(archive)
        status.update(status="completed", completed_utc=utc_now(),
                      bytes=archive.stat().st_size, sha256=sha256(archive))
    except BaseException as error:
        status.update(status="failed", failed_utc=utc_now(),
                      error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        save(status_file, status)


if __name__ == "__main__":
    main()
