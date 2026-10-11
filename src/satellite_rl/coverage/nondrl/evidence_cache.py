"""Immutable derived supervision cache; original audited evidence remains authority.

This engineering candidate is deliberately not wired into training. Build from
SHA-verified gzip rows, retain their (shard,row) identities and mother-balanced
draw order, and mmap only fields consumed by collate/loss. No labels, floating
values, sample eligibility, seeds, RNG calls or dataset.sha256 are changed.
"""
from contextlib import contextmanager
import gzip
import hashlib
import json
import mmap
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote
import uuid

import numpy as np

SCHEMA = "s1_nd12_compact_supervision_v1"
FIELDS = {
    "01": {"aircraft": "<f8", "candidates": "<f8", "legal": "?", "visible": "<i8", "labels": "<i8", "hidden": "?"},
    "02": {"aircraft": "<f8", "beams": "<f8", "actions": "<f8"},
    "03": {"aircraft": "<f8", "positions": "<f8"},
    "04": {"aircraft": "<f8", "beams": "<f8", "actions": "<f8", "known": "?", "utility": "<f8"},
    "05": {"aircraft": "<f8", "beams": "<f8", "actions": "<f8", "known": "?", "utility": "<f8"},
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _source(path, route):
    path = Path(path).resolve()
    if path.is_dir():path /= "manifest.json"
    raw = path.read_bytes(); record = json.loads(raw.decode("utf-8"))
    if record.get("format") != "s1_nondrl_teacher_v1" or record.get("route") != route:
        raise ValueError("cache source schema or route differs")
    if record.get("status") not in ("completed", "stopped"):
        raise ValueError("cache source has no accepted evidence")
    sha = hashlib.sha256(raw).hexdigest()
    identity = {"schema": SCHEMA, "route": route, "original_manifest_sha256": sha,
                "shards": [{"path": r["path"], "sha256": r["sha256"], "rows": r["rows"]}
                           for r in record["shards"]]}
    return path, record, identity, canonical(identity)


@contextmanager
def _build_lock(path):
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:stream.write(b"0");stream.flush()
            stream.seek(0);msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:yield
        finally:
            if os.name == "nt":stream.seek(0);msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _arrays(sample, route):
    result = {name: np.asarray(sample[name], dtype=dtype) for name, dtype in FIELDS[route].items()}
    if route == "03":
        for i, reference in enumerate(sample["references"]):
            result[f"reference_{i}_centers"] = np.asarray(reference["centers"], dtype="<f8")
            result[f"reference_{i}_assignment"] = np.asarray(reference["assignment"], dtype="<i8")
    return result


def build_cache(source_manifest, cache_root, route="01", *, validator=None, progress=None):
    """Stream one source gzip row at a time into an atomic immutable derivative."""
    if route not in FIELDS:raise ValueError("unsupported compact route")
    if validator is None:
        from .learning import validate_sample
        validator = validate_sample
    source, record, identity, key = _source(source_manifest, route)
    root = Path(cache_root).resolve();root.mkdir(parents=True, exist_ok=True)
    destination = root / key
    with _build_lock(root / (key + ".lock")):
        if destination.exists():
            existing = CompactEvidenceDataset(destination, expected_source=source)
            existing.close();return destination / "manifest.json"
        temporary = root / (key + ".partial_" + uuid.uuid4().hex)
        temporary.mkdir()
        started = time.perf_counter();compressed = rows = valid = filtered = 0
        index = sqlite3.connect(temporary / "index.sqlite")
        try:
            # Only the uncommitted derivative is disposable. Avoid thousands
            # of temporary-shard fsyncs; both final binary files are fsynced
            # before the completed directory becomes visible.
            index.execute("PRAGMA journal_mode=OFF")
            index.execute("PRAGMA synchronous=OFF")
            index.execute("CREATE TABLE samples(shard INTEGER,row INTEGER,mother TEXT,fields TEXT,scalars TEXT,PRIMARY KEY(shard,row))")
            with (temporary / "data.bin").open("wb") as stream:
                for shard, entry in enumerate(record["shards"]):
                    path = (source.parent / entry["path"]).resolve()
                    if not path.is_relative_to(source.parent) or digest(path) != entry["sha256"]:
                        raise ValueError("original cache shard path or SHA differs")
                    compressed += path.stat().st_size;count = 0
                    with gzip.open(path, "rt", encoding="utf-8") as packed:
                        for row, line in enumerate(packed):
                            sample = json.loads(line);rows += 1;count += 1
                            if not validator(sample, route):filtered += 1;continue
                            fields = {}
                            for name, array in _arrays(sample, route).items():
                                array = np.ascontiguousarray(array)
                                fields[name] = {"offset": stream.tell(), "shape": list(array.shape), "dtype": array.dtype.str}
                                stream.write(array.tobytes(order="C"))
                            scalars = {"mother_id": sample["mother_id"]}
                            for name in ("target_n", "positive_actions", "stop_positive"):
                                if name in sample:scalars[name] = sample[name]
                            if route == "03":scalars["reference_count"] = len(sample["references"])
                            index.execute("INSERT INTO samples VALUES(?,?,?,?,?)", (shard,row,json.dumps(sample["mother_id"]),
                                           json.dumps(fields, separators=(",", ":")),json.dumps(scalars, separators=(",", ":"))))
                            valid += 1
                    if count != entry["rows"]:raise ValueError("original cache row count differs")
                    index.commit()
                    if progress:progress({"shards":shard+1,"rows":rows,"valid":valid,"elapsed_seconds":time.perf_counter()-started})
                stream.flush();os.fsync(stream.fileno())
            if not valid:raise ValueError("compact source has no valid supervision")
            if digest(source) != identity["original_manifest_sha256"]:raise ValueError("source manifest changed while building")
            index.close();index = None
            with (temporary / "index.sqlite").open("r+b") as f:os.fsync(f.fileno())
            data = {"format":SCHEMA,"cache_key":key,"route":route,"source_path":str(source),"source_identity":identity,
                    "dataset_sha256":identity["original_manifest_sha256"],"rows":rows,"valid_rows":valid,"filtered":filtered,
                    "files":{name:{"sha256":digest(temporary/name),"bytes":(temporary/name).stat().st_size}
                             for name in ("data.bin","index.sqlite")},
                    "build":{"elapsed_seconds":time.perf_counter()-started,"compressed_source_bytes":compressed},
                    "scope":"derived training fields only; original evidence and audits retained"}
            (temporary / "manifest.json").write_text(json.dumps(data, ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
            temporary.rename(destination)
            return destination / "manifest.json"
        except BaseException:
            if index is not None:index.close()
            # Retain failed derivation for diagnosis; no committed cache appears.
            raise


class CompactEvidenceDataset:
    """Original draw identities, read-only mmap arrays and an immutable SQLite index."""
    def __init__(self, path, *, expected_source=None, expected_sha256=None, verify_original=False):
        self.path = Path(path).resolve()
        if self.path.is_dir():self.path /= "manifest.json"
        if expected_sha256 is not None and digest(self.path) != expected_sha256:
            raise ValueError("frozen compact manifest SHA differs")
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        manifest = self.manifest
        if manifest.get("format") != SCHEMA:raise ValueError("compact cache schema differs")
        source = Path(expected_source or manifest["source_path"]).resolve()
        source, original, identity, key = _source(source, manifest["route"])
        if key != manifest["cache_key"] or identity != manifest["source_identity"]:
            raise ValueError("compact cache original source identity differs")
        for name in ("data.bin", "index.sqlite"):
            file = self.path.parent / name
            if digest(file) != manifest["files"][name]["sha256"] or file.stat().st_size != manifest["files"][name]["bytes"]:
                raise ValueError("compact cache derived artifact differs")
        if verify_original:
            for entry in original["shards"]:
                file = (source.parent/entry["path"]).resolve()
                if not file.is_relative_to(source.parent) or digest(file) != entry["sha256"]:
                    raise ValueError("compact cache original shard SHA differs")
        self.route, self.sha256 = manifest["route"], manifest["dataset_sha256"]
        if self.sha256 != identity["original_manifest_sha256"]:raise ValueError("compact dataset sha differs")
        uri = "file:" + quote(str(self.path.parent / "index.sqlite"),safe="/:") + "?mode=ro&immutable=1"
        self.index = sqlite3.connect(uri,uri=True)
        self.stream = (self.path.parent / "data.bin").open("rb")
        self.buffer = mmap.mmap(self.stream.fileno(),0,access=mmap.ACCESS_READ)
        self.mothers, self.filtered = {}, int(manifest["filtered"])
        for shard,row,mother in self.index.execute("SELECT shard,row,mother FROM samples ORDER BY shard,row"):
            mother = json.loads(mother)
            self.mothers.setdefault(mother,[]).append((shard,row))
        if sum(map(len,self.mothers.values())) != manifest["valid_rows"]:raise ValueError("compact indexed row count differs")
        self.keys = sorted(self.mothers)

    def sample(self, shard, row):
        result = self.index.execute("SELECT fields,scalars FROM samples WHERE shard=? AND row=?",(shard,row)).fetchone()
        if result is None:raise KeyError((shard,row))
        fields, sample = json.loads(result[0]), json.loads(result[1])
        for name, entry in fields.items():
            value = np.ndarray(tuple(entry["shape"]),dtype=np.dtype(entry["dtype"]),buffer=self.buffer,offset=entry["offset"])
            value.setflags(write=False);sample[name] = value
        if self.route == "03":
            sample["references"] = [{"centers":sample.pop(f"reference_{i}_centers"),
                                     "assignment":sample.pop(f"reference_{i}_assignment")}
                                    for i in range(sample.pop("reference_count"))]
        return sample

    def draw(self, rng, size):
        samples, positions = [], []
        for _ in range(size):
            mother = self.keys[int(rng.integers(len(self.keys)))];entries = self.mothers[mother]
            shard,row = entries[int(rng.integers(len(entries)))];samples.append(self.sample(shard,row))
            positions.append({"mother_id":mother,"shard":shard,"row":row})
        return samples, positions

    def close(self):
        self.index.close()
        # ndarray.base retains its mmap. Relinquish our reference without
        # invalidating outstanding read-only sample views; GC closes it once
        # the last view is released. Resident memory follows accessed pages.
        self.buffer = None;self.stream.close()
