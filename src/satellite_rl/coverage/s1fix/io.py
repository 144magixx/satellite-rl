"""小型CSV、布局和运行配置读写。"""
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import csv
import json
import os
import platform
import socket
import subprocess

import numpy as np


@contextmanager
def _file_lock(path):
    """One short shared lock for concurrent CSV/manifest updates."""
    with Path(str(path) + ".lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b" ")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_rows(path, rows, append=False):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(path):
        if append and path.exists():
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream)) + rows
        fields = list(dict.fromkeys(key for row in rows for key in row))
        temporary = Path(str(path) + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def save_layout(path, layout, frame, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, centers_latlon=layout.centers_latlon, widths=layout.widths,
                        owner=layout.owner, beam_ids=layout.beam_ids, aircraft_index=frame.icao,
                        **metadata)


def load_layout(path):
    from .geom import Layout
    with np.load(path) as z:
        return Layout(z["centers_latlon"], z["widths"], z["owner"], z["beam_ids"])


@contextmanager
def manifest(out, key, config):
    path = Path(out) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    commit_file = Path("COMMIT")  # 远端冻结快照没有.git，部署时写入提交号
    commit = (commit_file.read_text().strip() if commit_file.exists() else
              subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    entry = dict(config=config, git_commit=commit, python=platform.python_version(),
        packages={name: version(name) for name in ("numpy", "scipy", "torch")},
        host=socket.gethostname(), start=datetime.now(timezone.utc).isoformat())
    def save():
        with _file_lock(path):
            entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            entries.setdefault(key, [])
            if "end" in entry:
                index = next(i for i, previous in enumerate(entries[key]) if previous["start"] == entry["start"])
                entries[key][index] = entry
            else:
                entries[key].append(entry)
            temporary = Path(str(path) + ".tmp")
            temporary.write_text(json.dumps(entries, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            temporary.replace(path)
    save()
    try:
        yield entry
    finally:
        entry["end"] = datetime.now(timezone.utc).isoformat()
        save()
