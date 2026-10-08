"""SHA-pinned plain files and ZIP members for large S2-P19 report inputs."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_member(name):
    path = Path(name)
    if (not isinstance(name, str) or not name or "\\" in name or
            path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("unsafe ZIP member")
    return name


def source_label(ref):
    if isinstance(ref, (str, Path)):
        return str(ref)
    if "zip_path" in ref:
        return "zip:" + str(ref["zip_path"]) + "!" + ref["member"]
    return str(ref["path"])


@lru_cache(maxsize=32)
def _zip_digest(path, size, mtime_ns):
    return digest(path)


def _check_zip(ref):
    path = Path(ref["zip_path"])
    if not path.is_file():
        raise ValueError("archived ZIP volume missing: " + str(path))
    stat = path.stat()
    if _zip_digest(str(path), stat.st_size, stat.st_mtime_ns) != ref["zip_sha256"]:
        raise ValueError("archived ZIP SHA differs: " + str(path))
    return path


class _HashingRaw(io.RawIOBase):
    def __init__(self, source):
        self.source = source
        self.value = hashlib.sha256()
        self.count = 0

    def readable(self):
        return True

    def readinto(self, target):
        block = self.source.read(len(target))
        target[:len(block)] = block
        self.value.update(block)
        self.count += len(block)
        return len(block)


@contextmanager
def open_text_source(ref):
    """Stream UTF-8 and verify member SHA/size after successful consumption."""
    if isinstance(ref, (str, Path)):
        ref = {"path": str(ref)}
    if "zip_path" not in ref:
        path = Path(ref["path"])
        if not path.is_file():
            raise ValueError("plain source missing: " + str(path))
        if ref.get("sha256") and digest(path) != ref["sha256"]:
            raise ValueError("plain source SHA differs: " + str(path))
        with path.open(encoding="utf-8") as stream:
            yield stream
        return
    path = _check_zip(ref)
    member = safe_member(ref["member"])
    with zipfile.ZipFile(path) as bundle:
        info = bundle.getinfo(member)
        if info.is_dir() or info.file_size != ref["bytes"]:
            raise ValueError("archived ZIP member type/size differs: " + member)
        with bundle.open(info) as source:
            hashed = _HashingRaw(source)
            with io.TextIOWrapper(io.BufferedReader(hashed), encoding="utf-8") as stream:
                yield stream
                # The JSON/JSONL consumer must read to EOF. TextIOWrapper may
                # have buffered final bytes, so drain it before comparing SHA.
                while stream.read(8 * 1024 * 1024):
                    pass
                if hashed.count != ref["bytes"] or hashed.value.hexdigest() != ref["sha256"]:
                    raise ValueError("archived ZIP member SHA differs: " + member)


def read_bytes_source(ref):
    if isinstance(ref, (str, Path)):
        ref = {"path": str(ref)}
    if "zip_path" not in ref:
        path = Path(ref["path"])
        payload = path.read_bytes()
    else:
        path = _check_zip(ref)
        member = safe_member(ref["member"])
        with zipfile.ZipFile(path) as bundle:
            info = bundle.getinfo(member)
            if info.is_dir() or info.file_size != ref["bytes"]:
                raise ValueError("archived ZIP member type/size differs: " + member)
            payload = bundle.read(info)
    if ref.get("sha256") and hashlib.sha256(payload).hexdigest() != ref["sha256"]:
        raise ValueError("archived source SHA differs: " + source_label(ref))
    return payload


@lru_cache(maxsize=64)
def receipt_members(receipt_path):
    """Map receipt member names to already downloaded volume references."""
    path = Path(receipt_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    declared = {item["member"]: item for item in receipt["members"]}
    if len(declared) != len(receipt["members"]):
        raise ValueError("duplicate receipt member")
    result = {}
    for volume in receipt["volumes"]:
        zip_path = path.parent / safe_member(volume["name"])
        with zipfile.ZipFile(zip_path) as bundle:
            for info in bundle.infolist():
                name = safe_member(info.filename)
                if name not in declared or name in result or info.file_size != declared[name]["bytes"]:
                    raise ValueError("receipt ZIP member identity differs")
                result[name] = {"zip_path": str(zip_path),
                                "zip_sha256": volume["sha256"],
                                "member": name, "sha256": declared[name]["sha256"],
                                "bytes": declared[name]["bytes"]}
    if set(result) != set(declared):
        raise ValueError("receipt ZIP volume member set incomplete")
    return result


def member_ref(receipt_path, member):
    return receipt_members(str(receipt_path)).get(safe_member(member))
