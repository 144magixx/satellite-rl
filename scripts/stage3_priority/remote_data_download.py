"""Resume and verify a pinned official ADSB.lol daily archive on a remote host.

This script only writes under --output-dir and never starts a workbench task.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--asset-id", type=int, required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    name = f"v{args.date.replace('-', '.')}-planes-readsb-staging-0.tar"
    if not args.url.endswith("/" + name):
        raise ValueError("URL does not match the requested date and preferred archive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / name
    partial = target.with_suffix(target.suffix + ".part")
    status_path = args.output_dir / "download_status.json"
    status = {"status": "running", "started_utc": now(), "date_utc": args.date,
              "url": args.url, "asset_id": args.asset_id,
              "expected_bytes": args.expected_bytes, "path": str(target),
              "attempts": []}
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if any(old.get(key) != status[key] for key in ("date_utc", "url", "asset_id", "expected_bytes")):
            raise RuntimeError("Existing download status belongs to a different pinned asset")
        status["started_utc"] = old.get("started_utc", status["started_utc"])
        status["attempts"] = old.get("attempts", [])
    save(status_path, status)
    try:
        if target.exists():
            if target.stat().st_size != args.expected_bytes:
                raise RuntimeError("Existing complete file has the wrong byte count")
        else:
            for attempt in range(1, 9):
                started = now()
                command = ["curl", "--location", "--fail", "--silent", "--show-error",
                           "--connect-timeout", "20", "--max-time", "1800",
                           "--continue-at", "-", "--output", str(partial), args.url]
                result = subprocess.run(command, capture_output=True, text=True, timeout=1830)
                current_bytes = partial.stat().st_size if partial.exists() else 0
                status["attempts"].append({"number": attempt, "started_utc": started,
                                           "ended_utc": now(), "curl_exit": result.returncode,
                                           "partial_bytes": current_bytes,
                                           "stderr_tail": result.stderr[-300:]})
                save(status_path, status)
                if result.returncode == 0 and current_bytes == args.expected_bytes:
                    break
                if current_bytes > args.expected_bytes:
                    raise RuntimeError("Partial download exceeds official release byte count")
                if attempt == 8:
                    raise RuntimeError("Official archive download did not complete after eight resumable attempts")
                time.sleep(min(attempt * 5, 30))
            partial.replace(target)
        status.update(status="completed", completed_utc=now(),
                      bytes=target.stat().st_size, sha256=sha256(target))
    except BaseException as exc:
        status.update(status="failed", failed_utc=now(), error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    main()
