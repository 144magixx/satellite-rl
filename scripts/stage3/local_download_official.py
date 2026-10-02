"""Bounded, resumable official ADSB.lol daily archive download.

Each curl process has a wall-clock limit. Verified byte ranges are appended only
after status, Content-Range, and exact length checks. Run two independent lanes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlparse

import requests


ROOT = Path(os.environ.get("STAGE3_DATA_ROOT", "D:/paper/outputs/experiments/stage3_power_20261003/data"))
RAW = ROOT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
DATE = os.environ.get("STAGE3_DATE", "2024-01-25")
PIN = json.loads((ROOT / "source_preflight.json").read_text(encoding="utf-8"))
ASSETS = {DATE.replace("-", "."): PIN["expected_bytes"]}
CHUNK_SIZE = 32 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_location(browser_url):
    with requests.get(browser_url + f"?cb={time.time_ns()}", headers={"Cache-Control": "no-cache"},
                      allow_redirects=False, stream=True, timeout=(8, 10)) as response:
        location = response.headers.get("Location", "")
        parsed = urlparse(location)
        if response.status_code != 302 or parsed.scheme != "https" or parsed.hostname != "release-assets.githubusercontent.com":
            raise RuntimeError(f"unexpected official release redirect: {response.status_code} {parsed.hostname}")
        return location


def fetch(location, start, end, total, chunk_path, header_path):
    cmd = ["curl.exe", "--silent", "--show-error", "--max-time", "30",
           "--connect-timeout", "8", "--range", f"{start}-{end}",
           "--dump-header", str(header_path), "--output", str(chunk_path), location]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
    if result.returncode:
        raise RuntimeError(f"curl exit {result.returncode}: {result.stderr[-400:]}")
    headers = header_path.read_text(encoding="latin-1", errors="replace")
    sections = re.split(r"\r?\n\r?\n", headers)
    last = [section for section in sections if section.startswith("HTTP/")][-1]
    status_match = re.search(r"^HTTP/\S+\s+(\d+)", last)
    range_match = re.search(r"(?im)^Content-Range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", last)
    got_status = int(status_match.group(1)) if status_match else None
    got_range = tuple(map(int, range_match.groups())) if range_match else None
    if got_status != 206 or got_range != (start, end, total):
        raise RuntimeError(f"bad HTTP range: status={got_status}, range={got_range}, expected={(start,end,total)}")
    if chunk_path.stat().st_size != end - start + 1:
        raise RuntimeError(f"bad chunk length: {chunk_path.stat().st_size} vs {end-start+1}")


def fetch_with_retries(url, start, end, total, chunk_path, header_path):
    failures = []
    for attempt in range(1, 7):
        try:
            fetch(fresh_location(url), start, end, total, chunk_path, header_path)
            return attempt - 1
        except (OSError, RuntimeError, subprocess.TimeoutExpired, requests.RequestException) as exc:
            failures.append(repr(exc))
            if attempt == 6:
                raise RuntimeError(f"range {start}-{end} failed six times: {failures[-1]}") from exc
            time.sleep(min(attempt, 3))


def download(date, state, state_path):
    expected = ASSETS[date]
    name = f"v{date}-planes-readsb-staging-0.tar"
    url = ("https://github.com/adsblol/globe_history_2024/releases/download/"
           f"v{date}-planes-readsb-staging-0/{name}")
    path = RAW / name
    part = RAW / (name + ".part")
    chunk = RAW / (name + ".chunk")
    headers = RAW / (name + ".headers")
    item = {"url": url, "expected_bytes": expected, "started_utc": now()}
    state["assets"][date] = item
    save(state_path, state)
    if path.is_file():
        if path.stat().st_size != expected:
            raise RuntimeError(f"existing file has unexpected size: {path}")
    else:
        if part.is_file() and part.stat().st_size:
            length = part.stat().st_size
            with part.open("rb") as handle:
                for start in sorted({0, max(0, length // 2 - 2048), max(0, length - 4096)}):
                    end = min(start + 4095, length - 1)
                    for attempt in range(1, 5):
                        try:
                            fetch(fresh_location(url), start, end, expected, chunk, headers)
                            handle.seek(start)
                            if handle.read(end - start + 1) != chunk.read_bytes():
                                raise RuntimeError(f"existing partial differs at {start}")
                            break
                        except (OSError, RuntimeError, subprocess.TimeoutExpired, requests.RequestException) as exc:
                            if attempt == 4:
                                raise RuntimeError("resume prefix verification failed") from exc
                            time.sleep(attempt)
        chunk_size, success_streak, failures_at_min = CHUNK_SIZE, 0, 0
        while (part.stat().st_size if part.exists() else 0) < expected:
            start = part.stat().st_size if part.exists() else 0
            end = min(start + chunk_size, expected) - 1
            item.update(inflight_range=[start, end], chunk_bytes=chunk_size, inflight_started_utc=now())
            save(state_path, state)
            try:
                fetch(fresh_location(url), start, end, expected, chunk, headers)
                with part.open("ab") as target, chunk.open("rb") as source:
                    if target.tell() != start:
                        raise RuntimeError("partial file changed before append")
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
                item["partial_bytes"] = end + 1
                item.pop("retry_error", None)
                item.pop("inflight_range", None)
                item.pop("inflight_started_utc", None)
                success_streak += 1
                failures_at_min = 0
                if chunk_size < CHUNK_SIZE and success_streak >= 3:
                    chunk_size = min(CHUNK_SIZE, chunk_size * 2)
                    success_streak = 0
                save(state_path, state)
            except (OSError, RuntimeError, subprocess.TimeoutExpired, requests.RequestException) as exc:
                item["retry_error"] = repr(exc)
                item["retry_count"] = item.get("retry_count", 0) + 1
                success_streak = 0
                chunk_size = max(1024 * 1024, chunk_size // 2)
                if chunk_size == 1024 * 1024:
                    failures_at_min += 1
                save(state_path, state)
                if failures_at_min >= 12:
                    raise RuntimeError(f"one-megabyte range repeatedly failed at {date} byte {start}") from exc
                time.sleep(min(failures_at_min + 1, 4))
        if part.stat().st_size != expected:
            raise RuntimeError("finished partial has unexpected length")
        part.replace(path)
    item.update(bytes=path.stat().st_size, sha256=sha256(path), completed_utc=now())
    item.pop("partial_bytes", None)
    save(state_path, state)
    for temp in (chunk, headers):
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    dates = (DATE.replace("-", "."),)
    state_path = ROOT / "download_status.json"
    state = {"status": "running", "lane": "stage3_common_independent_test",
             "source_repo": "adsblol/globe_history_2024", "started_utc": now(), "assets": {}}
    save(state_path, state)
    try:
        for date in dates:
            download(date, state, state_path)
        state["status"] = "completed"
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        raise
    finally:
        state["updated_utc"] = now()
        save(state_path, state)


if __name__ == "__main__":
    main()
