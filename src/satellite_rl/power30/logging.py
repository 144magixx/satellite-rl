"""Compact append-only evidence with recoverable gzip member boundaries."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"), default=json_default)


def content_sha(value):
    return hashlib.sha256(canonical_json(value).encode("utf8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class EvidenceLogs:
    """One copy of per-step service and one copy of each terminal whole-N array."""
    NAMES = ("steps", "episodes", "evaluations")

    def __init__(self, directory, offsets=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.streams = {}
        self.offsets = dict(offsets or {})
        for name in self.NAMES:
            path = self.directory / (name + ".jsonl.gz")
            if offsets is not None:
                wanted = int(offsets.get(name, 0))
                if not path.exists() and wanted:
                    raise ValueError("Checkpoint evidence log is missing")
                if path.exists():
                    if path.stat().st_size != wanted:
                        raise ValueError("Checkpoint evidence prefix/suffix differs; preserve all bytes and require root accounting")

    def append(self, name, value):
        if name not in self.NAMES:
            raise ValueError("Unknown evidence stream")
        if name not in self.streams:
            raw = (self.directory / (name + ".jsonl.gz")).open("ab")
            gz = gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=3, mtime=0)
            self.streams[name] = raw, gz
        self.streams[name][1].write((canonical_json(value) + "\n").encode("utf8"))

    def barrier(self):
        for raw, gz in self.streams.values():
            gz.close()
            raw.flush()
            os.fsync(raw.fileno())
            raw.close()
        self.streams.clear()
        self.offsets = {name: (self.directory / (name + ".jsonl.gz")).stat().st_size
                        if (self.directory / (name + ".jsonl.gz")).exists() else 0 for name in self.NAMES}
        return dict(self.offsets)

    def close(self):
        return self.barrier()
