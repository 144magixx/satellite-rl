"""Resume a SHA-pinned Jan08/09 source archive upload to 20424 only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ops_common import BASE, _transport, remote_python, write_json


LOCAL_ROOT = Path("D:/paper/outputs/experiments/europe_adsb_20260928/raw")
OUTPUT = Path("D:/paper/outputs/experiments/variable_beams_20261004/replenishment")
ASSETS = {
    "2024-01-08": {"name": "v2024.01.08-planes-readsb-prod-0.tar",
                   "bytes": 1610444800,
                   "sha256": "a0b35aed430b8cd7b4de3fef3db79ce8efee958a2a1f2daa32056731db9074bd"},
    "2024-01-09": {"name": "v2024.01.09-planes-readsb-prod-0.tar",
                   "bytes": 1443092480,
                   "sha256": "df67efea6c4502177480a08e46fe91dc59f9e4c54c31c88bd1301c91f964b12e"},
}


def sha(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as stream:
        while True:
            size = 8 * 1024 * 1024 if remaining is None else min(8 * 1024 * 1024, remaining)
            if size == 0:
                break
            chunk = stream.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if remaining not in (None, 0):
        raise ValueError("Local prefix shorter than remote partial file")
    return digest.hexdigest()


def upload(day: str) -> dict:
    asset = ASSETS[day]
    source = LOCAL_ROOT / asset["name"]
    if not source.is_file() or source.stat().st_size != asset["bytes"] or sha(source) != asset["sha256"]:
        raise RuntimeError("Original Jan08/09 tar bytes or SHA differ from frozen source manifest")
    root = BASE + "/data/raw"
    target = root + "/" + asset["name"]
    partial = target + ".part"
    remote = remote_python(f'''
import hashlib,json,pathlib
root=pathlib.Path({root!r});root.mkdir(parents=True,exist_ok=True)
target=pathlib.Path({target!r});part=pathlib.Path({partial!r})
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
print(json.dumps({{'target_exists':target.is_file(),
 'target_bytes':target.stat().st_size if target.is_file() else None,
 'target_sha256':sha(target) if target.is_file() else None,
 'partial_bytes':part.stat().st_size if part.is_file() else 0,
 'partial_sha256':sha(part) if part.is_file() else hashlib.sha256(b'').hexdigest()}}))
''', timeout=180)
    if remote["target_exists"]:
        if remote["target_bytes"] != asset["bytes"] or remote["target_sha256"] != asset["sha256"]:
            raise RuntimeError("Existing remote tar differs; refusing replacement")
        result = {"status": "already_verified", "day": day, "target": target,
                  "sha256": asset["sha256"], "bytes": asset["bytes"]}
        write_json(OUTPUT / f"transfer_{day}.json", result)
        return result
    offset = remote["partial_bytes"]
    if offset > asset["bytes"] or sha(source, offset) != remote["partial_sha256"]:
        raise RuntimeError("Remote partial archive prefix differs from frozen local source")
    connect, _, paramiko = _transport()
    transport = connect(20424)
    try:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            with source.open("rb") as local:
                local.seek(offset)
                with sftp.file(partial, "ab") as output:
                    output.set_pipelined(True)
                    while offset < asset["bytes"]:
                        chunk = local.read(min(1024 * 1024, asset["bytes"] - offset))
                        if not chunk:
                            raise RuntimeError("Local source truncated during upload")
                        output.write(chunk)
                        offset += len(chunk)
                        if offset % (64 * 1024 * 1024) < len(chunk):
                            write_json(OUTPUT / f"transfer_{day}_progress.json",
                                       {"status": "uploading", "day": day, "uploaded_bytes": offset,
                                        "total_bytes": asset["bytes"], "updated_utc": datetime.now(timezone.utc).isoformat()})
    finally:
        transport.close()
    verified = remote_python(f'''
import datetime,hashlib,json,pathlib
part=pathlib.Path({partial!r});target=pathlib.Path({target!r});expected={asset['sha256']!r}
if target.exists():raise RuntimeError('Target appeared during upload')
if part.stat().st_size!={asset['bytes']}:raise RuntimeError('Transferred byte count differs')
h=hashlib.sha256()
with part.open('rb') as stream:
 for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
if h.hexdigest()!=expected:raise RuntimeError('Remote transferred tar SHA differs')
part.rename(target)
print(json.dumps({{'status':'verified_source_only','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','path':str(target),
 'bytes':target.stat().st_size,'sha256':expected}}))
''', timeout=180)
    write_json(OUTPUT / f"transfer_{day}.json", verified)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", choices=tuple(ASSETS), required=True)
    print(json.dumps(upload(parser.parse_args().day), ensure_ascii=False))


if __name__ == "__main__":
    main()
