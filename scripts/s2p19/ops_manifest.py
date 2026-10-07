"""Merge operational evidence into the S2-P19 manifest without replacing scope."""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import ops


MANIFEST = ops.LOCAL / "run_manifest.json"


def _merge(target: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value
    return target


def merge_operations(patch: dict) -> dict:
    if not isinstance(patch, dict):
        raise ValueError("operations patch must be a dictionary")
    for _ in range(5):
        before = MANIFEST.stat()
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if payload.get("experiment_id") != "S2-P19" or not payload.get("authorization"):
            raise ValueError("missing S2-P19 authorization in root manifest")
        payload["operations"] = _merge(payload.get("operations") or {}, patch)
        pending = MANIFEST.with_name(MANIFEST.name + ".ops.tmp")
        pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        current = MANIFEST.stat()
        if (current.st_mtime_ns, current.st_size) != (before.st_mtime_ns, before.st_size):
            pending.unlink()
            continue
        os.replace(pending, MANIFEST)
        return payload["operations"]
    raise RuntimeError("manifest changed during five merge attempts")
