"""Resumably mirror S1-02 official A/B raw archives into D:/paper outputs.

Run as a persistent local background process. Every final tar is checked
against the remote completed download's size and SHA-256 before publication.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from remote_sources import LOCAL, REMOTE, connect


BLOCK = 8 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def remote_json(sftp, path: str) -> dict:
    with sftp.open(path, "rb") as stream:
        return json.loads(stream.read())


def backup(sftp, stage: str, status_file: Path, status: dict) -> dict:
    pin = json.loads((LOCAL / stage / "data/source_preflight.json").read_text(encoding="utf-8"))
    remote_root = REMOTE + "/data/" + stage + "/raw/"
    remote_status = remote_json(sftp, remote_root + "download_status.json")
    if (pin["stage"] != stage or remote_status["status"] != "completed" or
            remote_status["bytes"] != pin["expected_bytes"] or
            remote_status["asset_id"] != pin["asset_id"] or
            remote_status["url"] != pin["archive_url"]):
        raise RuntimeError("Frozen official archive status differs for " + stage)
    remote_path = remote_root + pin["archive_name"]
    local_path = LOCAL / stage / "data/raw" / pin["archive_name"]
    partial = local_path.with_suffix(local_path.suffix + ".part")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = pin["expected_bytes"]
    expected_sha = remote_status["sha256"]
    if local_path.is_file():
        if local_path.stat().st_size != expected_bytes or sha(local_path) != expected_sha:
            raise RuntimeError("Existing local official archive differs for " + stage)
        action = "existing_sha_verified"
    else:
        current = partial.stat().st_size if partial.exists() else 0
        if current > expected_bytes:
            raise RuntimeError("Local partial archive exceeds official size")
        status["stages"][stage] = {"status": "copying", "bytes": current,
                                   "expected_bytes": expected_bytes,
                                   "remote_sha256": expected_sha, "updated_utc": now()}
        save(status_file, status)
        with sftp.open(remote_path, "rb") as remote, partial.open("ab") as local:
            remote.seek(current)
            while current < expected_bytes:
                block = remote.read(min(BLOCK, expected_bytes - current))
                if not block:
                    raise RuntimeError("Remote source ended before expected size")
                local.write(block)
                current += len(block)
                if current % (64 * 1024 * 1024) < len(block) or current == expected_bytes:
                    local.flush()
                    status["stages"][stage]["bytes"] = current
                    status["stages"][stage]["updated_utc"] = now()
                    save(status_file, status)
        if partial.stat().st_size != expected_bytes or sha(partial) != expected_sha:
            raise RuntimeError("Raw archive size/SHA differs after transfer")
        partial.replace(local_path)
        action = "copied_sha_verified"
    receipt = {"experiment_id": "S1-02", "stage": stage, "status": action,
               "checked_utc": now(), "remote_path": remote_path,
               "local_path": str(local_path), "bytes": expected_bytes,
               "sha256": expected_sha, "official_asset_id": pin["asset_id"]}
    save(LOCAL / "preflight" / f"raw_backup_{stage}.json", receipt)
    status["stages"][stage] = {"status": action, "bytes": expected_bytes,
                               "sha256": expected_sha, "updated_utc": now()}
    save(status_file, status)
    return receipt


def main() -> None:
    status_file = LOCAL / "preflight/raw_backup_status.json"
    status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.is_file() else {
        "experiment_id": "S1-02", "status": "running", "started_utc": now(), "stages": {}}
    remaining = sum(json.loads((LOCAL / stage / "data/source_preflight.json").read_text(encoding="utf-8"))
                    ["expected_bytes"] for stage in ("A", "B"))
    if shutil.disk_usage(LOCAL).free < remaining + 2 * 1024 ** 3:
        raise RuntimeError("D: free space is insufficient for two raw source archives")
    save(status_file, status)
    client = connect()
    try:
        with client.open_sftp() as sftp:
            for stage in ("A", "B"):
                backup(sftp, stage, status_file, status)
        status.update(status="completed", completed_utc=now())
    except BaseException as error:
        status.update(status="failed", failed_utc=now(),
                      error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        client.close()
        save(status_file, status)
    print(json.dumps({"status": status["status"], "stages": status["stages"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
