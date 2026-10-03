"""Fetch exact official asset ranges and assemble only after full SHA/size checks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


BASE = Path(os.environ["S3_BASE"])
RAW = BASE / "shared_data/raw"
URL = os.environ["S3_URL"]
OFFICIAL_URL = os.environ["S3_OFFICIAL_URL"]
NAME = os.environ["S3_ASSET"]
SIZE = int(os.environ["S3_EXPECTED_BYTES"])
PARTS = RAW / "parallel_parts"
STATUS = RAW / "parallel_status.json"
COUNT = 16


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(value: dict) -> None:
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS)


def chunks():
    width = (SIZE + COUNT - 1) // COUNT
    for index in range(COUNT):
        start = index * width
        end = min(SIZE, start + width) - 1
        if start < SIZE:
            yield index, start, end


def transfer(index: int, start: int, end: int) -> dict:
    final = PARTS / f"chunk_{index:02d}.done"
    partial = PARTS / f"chunk_{index:02d}.part"
    target_bytes = end - start + 1
    if final.exists():
        if final.stat().st_size != target_bytes:
            raise RuntimeError(f"Complete range size differs: {index}")
        return {"index": index, "start": start, "end": end,
                "bytes": target_bytes, "sha256": sha(final), "source": "verified_existing"}
    for attempt in range(1, 17):
        got = partial.stat().st_size if partial.exists() else 0
        if got == target_bytes:
            break
        if got > target_bytes:
            raise RuntimeError(f"Range overrun: {index}")
        offset = start + got
        temp = PARTS / f"chunk_{index:02d}.response"
        headers = PARTS / f"chunk_{index:02d}.headers"
        try:
            request = subprocess.run(["curl", "--location", "--fail", "--silent", "--show-error",
                                      "--connect-timeout", "20", "--max-time", "600",
                                      "--range", f"{offset}-{end}",
                                      "--dump-header", str(headers), "--output", str(temp), URL],
                                     capture_output=True, text=True, timeout=610)
            lines = headers.read_text(errors="replace").splitlines() if headers.is_file() else []
            content_range = next((line.split(":", 1)[1].strip() for line in reversed(lines)
                                  if line.lower().startswith("content-range:")), None)
            expected = f"bytes {offset}-{end}/{SIZE}"
            if content_range != expected or not temp.is_file():
                raise RuntimeError(f"Range response mismatch: {index}")
            size = temp.stat().st_size
            if size > end - offset + 1:
                raise RuntimeError(f"Range response overrun: {index}")
            with partial.open("ab") as output, temp.open("rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    output.write(block)
            temp.unlink()
            headers.unlink(missing_ok=True)
            if partial.stat().st_size == target_bytes:
                break
        except (OSError, TimeoutError, RuntimeError) as exc:
            if attempt == 16:
                raise RuntimeError(f"Official range {index} failed after retries") from exc
            time.sleep(min(2 * attempt, 20))
    if not partial.is_file() or partial.stat().st_size != target_bytes:
        raise RuntimeError(f"Official range {index} incomplete")
    partial.replace(final)
    return {"index": index, "start": start, "end": end,
            "bytes": target_bytes, "sha256": sha(final), "source": "official_range"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started_utc": now(), "asset": NAME,
             "source_url": OFFICIAL_URL, "transfer": "official_release_cdn_ranges",
             "expected_bytes": SIZE, "ranges": COUNT,
             "completed": []}
    save(state)
    try:
        with ThreadPoolExecutor(max_workers=COUNT) as pool:
            pending = [pool.submit(transfer, *part) for part in chunks()]
            for future in as_completed(pending):
                state["completed"].append(future.result())
                state["completed"].sort(key=lambda row: row["index"])
                state["checked_utc"] = now()
                save(state)
        if len(state["completed"]) != COUNT:
            raise RuntimeError("Official range grid incomplete")
        assembled = RAW / (NAME + ".range.part")
        digest = hashlib.sha256()
        total = 0
        with assembled.open("wb") as output:
            for row in state["completed"]:
                path = PARTS / f"chunk_{row['index']:02d}.done"
                if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                    raise RuntimeError("Range changed before assembly")
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        total += len(block)
        if total != SIZE or assembled.stat().st_size != SIZE:
            raise RuntimeError("Assembled official archive size differs")
        final = RAW / NAME
        if final.exists():
            if final.stat().st_size != SIZE or sha(final) != digest.hexdigest():
                raise RuntimeError("Existing final official archive differs")
            assembled.unlink()
        else:
            assembled.replace(final)
        direct = RAW / "download_status.json"
        old = json.loads(direct.read_text(encoding="utf-8"))
        if (old["url"] != OFFICIAL_URL or old["expected_bytes"] != SIZE or
                old["status"] not in {"running", "failed"}):
            raise RuntimeError("Official asset provenance differs at completion")
        old.update(status="completed", completed_utc=now(), bytes=SIZE,
                   sha256=digest.hexdigest(), transfer="official_release_parallel_ranges",
                   chunk_manifest=str(STATUS))
        pending = direct.with_suffix(".json.tmp")
        pending.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(direct)
        state.update(status="completed", completed_utc=now(),
                     bytes=SIZE, sha256=digest.hexdigest())
    except BaseException as exc:
        state.update(status="failed", failed_utc=now(),
                     error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        save(state)


if __name__ == "__main__":
    main()
