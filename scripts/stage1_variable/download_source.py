"""Resume and verify one frozen S1-V10 daily aircraft archive on remote shared storage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_dir.resolve()
    pin_path = root / "source_preflight.json"
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    contract = json.loads((root / "selection_contract.json").read_text(encoding="utf-8"))
    if (pin["group"] != "S1-V10" or contract["group"] != "S1-V10" or
            pin["date_utc"] != contract["date_utc"] or
            pin["role"] != contract["role"]):
        raise RuntimeError("pin and selection contract differ")
    for name, key in (("release_api.json", "release_api_sha256"),
                      ("PREFERRED_RELEASES.txt", "preferred_sha256"),
                      ("LICENSE-ODbL-official.txt", "license_sha256"),
                      ("selection_contract.json", "selection_contract_sha256")):
        if sha(root / name) != pin[key]:
            raise RuntimeError(f"frozen source document changed: {name}")
    if pin["archive_url"] not in (root / "PREFERRED_RELEASES.txt").read_text(encoding="utf-8"):
        raise RuntimeError("official preferred release differs")
    archive = root / pin["archive_name"]
    partial = root / (pin["archive_name"] + ".part")
    status_path = root / "download_status.json"
    fixed = {"group": "S1-V10", "role": pin["role"], "date_utc": pin["date_utc"],
             "url": pin["archive_url"], "asset_id": pin["asset_id"],
             "expected_bytes": pin["expected_bytes"], "pin_sha256": sha(pin_path),
             "path": str(archive)}
    old = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    if old and any(old.get(key) != value for key, value in fixed.items()):
        raise RuntimeError("existing download belongs to a different frozen asset")
    status = {**fixed, "status": "running", "started_utc": old.get("started_utc", now()),
              "attempts": old.get("attempts", [])}
    save(status_path, status)
    try:
        if not archive.exists():
            for attempt in range(1, 9):
                command = ["curl", "--location", "--fail", "--silent", "--show-error",
                           "--connect-timeout", "20", "--max-time", "1800", "--continue-at", "-",
                           "--output", str(partial), pin["archive_url"]]
                result = subprocess.run(command, capture_output=True, text=True, timeout=1830)
                size = partial.stat().st_size if partial.exists() else 0
                status["attempts"].append({"attempt": attempt, "observed_utc": now(),
                                           "curl_exit": result.returncode,
                                           "partial_bytes": size,
                                           "stderr_tail": result.stderr[-500:]})
                save(status_path, status)
                if result.returncode == 0 and size == pin["expected_bytes"]:
                    break
                if size > pin["expected_bytes"] or attempt == 8:
                    raise RuntimeError("source archive download incomplete or exceeds official byte count")
                time.sleep(min(5 * attempt, 30))
            partial.replace(archive)
        if archive.stat().st_size != pin["expected_bytes"]:
            raise RuntimeError("archive byte count differs")
        status.update(status="completed", completed_utc=now(),
                      bytes=archive.stat().st_size, sha256=sha(archive))
    except BaseException as error:
        status.update(status="failed", failed_utc=now(), error=repr(error))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    main()
