"""Import the frozen S2-P17 scene records into the S2-P18 archives.

The three records (train 8064 / validation 144 / test 440) are reused as-is:
their NPZ files live on the shared NFS and both instances therefore verify the
same SHA-256 set. The script records the imported record identity plus a
content fingerprint so "same inputs" is provable, not assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

SOURCE_ROOT = ("/home/u2024110224/second_stage_satellite_rl/"
               "variable_beams_20261004/20424/archive/outputs/workbench/scenes")
RECORDS = {
    "train": "scenes_20261004_133711_dec915dc",
    "validation": "scenes_20261004_133835_f0592bee",
    "test": "scenes_20261004_133838_d7476a54",
}
EXPECTED_COUNT = {"train": 8064, "validation": 144, "test": 440}

FINGERPRINT_BODY = '''
import hashlib,json,pathlib
base=pathlib.Path({source!r})
name={name!r}
directory=base/name
files=sorted(directory.glob("*.npz"))
digest=hashlib.sha256()
for path in files:
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\\0")
    digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    digest.update(b"\\n")
print(json.dumps({{"files":len(files),"fingerprint":digest.hexdigest()}}))
'''


def fingerprint(port: int, role: str) -> dict:
    body = FINGERPRINT_BODY.format(source=SOURCE_ROOT, name=RECORDS[role])
    result = ops18.remote_python(body, port, timeout=1800)
    if result["exit_code"]:
        raise RuntimeError(f"指纹计算失败 {port} {role}: {result['stderr'][-400:]}")
    return json.loads(result["stdout"].strip().splitlines()[-1])


def import_all(port: int, *, execute: bool) -> dict:
    existing = {item.get("name"): item for item in ops18.api(port, "/scenes", timeout=60).get("scenes", [])}
    report = {"port": port, "imports": {}, "fingerprints": {}}
    for role, name in RECORDS.items():
        report["fingerprints"][role] = fingerprint(port, role)
        if report["fingerprints"][role]["files"] != EXPECTED_COUNT[role]:
            raise RuntimeError(f"{role} 场景文件数不符：{report['fingerprints'][role]['files']}")
        if name in existing:
            report["imports"][role] = {"status": "already_imported", "id": existing[name].get("id"),
                                       "count": existing[name].get("count")}
            continue
        if not execute:
            report["imports"][role] = {"status": "plan_only", "path": f"{SOURCE_ROOT}/{name}"}
            continue
        result = ops18.api(port, "/scenes/import",
                           {"path": f"{SOURCE_ROOT}/{name}", "name": name, "profile": "europe_v1"},
                           method="POST", timeout=3600)
        report["imports"][role] = {"status": "imported", "id": result.get("id"),
                                   "count": result.get("count"), "name": result.get("name")}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=ops18.PORTS, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = import_all(args.port, execute=args.execute)
    path = ops18.LOCAL / "preflight" / f"scene_import_{args.port}.json"
    ops18.write_json(path, {"checked_utc": ops18.utc_now(), "source_root": SOURCE_ROOT, **report})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
