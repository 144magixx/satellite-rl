"""SHA-verified, single-copy transfer of G11 mother CSVs to 20424.

The local step packages original inputs only; all scene generation runs remotely.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile


TRANSPORT = Path("D:/paper/outputs/verification/g11_remote_launch_20261001")
sys.path.insert(0, str(TRANSPORT))
from remote_transport import connect, execute, paramiko  # noqa: E402

LOCAL_CSV = Path("D:/paper/outputs/experiments/europe_adsb_20260928/snapshots")
LOCAL_RECEIPTS = Path("D:/paper/outputs/experiments/variable_beams_20261004/preflight")
REMOTE_G11 = "/home/u2024110224/second_stage_satellite_rl/g11/20424/archive/outputs/workbench/scenes"
REMOTE_RECORDS = {
    "train": REMOTE_G11 + "/scenes_20261001_161621_17ceaa35/record.json",
    "validation": REMOTE_G11 + "/scenes_20261001_161628_819160d7/record.json",
}
REMOTE_ROOT = "/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424/data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_json(transport, code: str) -> dict:
    encoded = base64.b64encode(code.encode()).decode()
    result = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'", timeout=180)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1000:])
    return json.loads(result["stdout"])


def package() -> dict:
    LOCAL_RECEIPTS.mkdir(parents=True, exist_ok=True)
    transport = connect(20424)
    try:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            for role, remote_path in REMOTE_RECORDS.items():
                local = LOCAL_RECEIPTS / f"g11_{role}_mother_record.json"
                sftp.get(remote_path, str(local))
    finally:
        transport.close()
    entries = []
    for role in ("train", "validation"):
        record_path = LOCAL_RECEIPTS / f"g11_{role}_mother_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("count") != (576 if role == "train" else 20):
            raise ValueError("G11 mother record count differs")
        source = record.get("source", {})
        if source.get("partition_role") != role or source.get("seed") != 51001:
            raise ValueError("G11 mother source role/seed differs")
        for item, snapshot in zip(source["files"], source["snapshots"], strict=True):
            date = snapshot["selected_time_utc"][:10]
            local = LOCAL_CSV / role / date / Path(item["path"]).name
            if not local.is_file() or sha256(local) != item["sha256"]:
                raise ValueError("Mother CSV missing or hash mismatch: " + str(local))
            entries.append({"source": str(local), "archive_name": f"{role}/{date}/{local.name}",
                            "sha256": item["sha256"], "bytes": local.stat().st_size,
                            "selected_time_utc": snapshot["selected_time_utc"]})
    if len(entries) != 596 or len({entry["archive_name"] for entry in entries}) != 596:
        raise ValueError("Expected 596 unique mother CSVs")
    archive = LOCAL_RECEIPTS / "g11_mother_csv_596.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for item in entries:
            output.write(item["source"], item["archive_name"])
    manifest = {"status": "original_csvs_packaged_for_20424_only",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "archive": str(archive), "archive_sha256": sha256(archive),
                "archive_bytes": archive.stat().st_size,
                "mother_records": {role: {"path": str(LOCAL_RECEIPTS / f"g11_{role}_mother_record.json"),
                                          "sha256": sha256(LOCAL_RECEIPTS / f"g11_{role}_mother_record.json")}
                                   for role in REMOTE_RECORDS},
                "csv_count": len(entries), "csv_bytes": sum(x["bytes"] for x in entries),
                "files": entries}
    receipt = LOCAL_RECEIPTS / "g11_mother_csv_package.json"
    receipt.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(receipt), "csv_count": len(entries),
            "csv_bytes": manifest["csv_bytes"], "archive_bytes": manifest["archive_bytes"],
            "archive_sha256": manifest["archive_sha256"]}


def upload() -> dict:
    manifest = json.loads((LOCAL_RECEIPTS / "g11_mother_csv_package.json").read_text(encoding="utf-8"))
    archive = Path(manifest["archive"])
    if sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Local transfer archive changed")
    transport = connect(20424)
    try:
        remote_dir = REMOTE_ROOT + "/source_csv"
        init = remote_json(transport, f'''
import json,pathlib
root=pathlib.Path({remote_dir!r});root.mkdir(parents=True,exist_ok=True)
print(json.dumps({{'root':str(root),'exists':root.is_dir()}}))
''')
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            sftp.put(str(archive), remote_dir + "/g11_mother_csv_596.zip.part")
        verify = remote_json(transport, f'''
import datetime,hashlib,json,pathlib,zipfile
root=pathlib.Path({remote_dir!r})
archive=root/'g11_mother_csv_596.zip.part'
expected={manifest['archive_sha256']!r}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for chunk in iter(lambda:s.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
if sha(archive)!=expected:raise RuntimeError('Transferred archive SHA mismatch')
with zipfile.ZipFile(archive) as bundle:
 members=bundle.infolist()
 if len(members)!=596:raise RuntimeError('Transferred CSV count differs')
 for item in members:
  p=pathlib.PurePosixPath(item.filename)
  if p.is_absolute() or '..' in p.parts or len(p.parts)!=3 or p.suffix!='.csv':
   raise RuntimeError('Unsafe or unexpected CSV archive path')
 bundle.extractall(root)
verified=[]
for item in {[(x['archive_name'],x['sha256']) for x in manifest['files']]!r}:
 path=root/item[0]
 if sha(path)!=item[1]:raise RuntimeError('Remote source CSV SHA mismatch: '+item[0])
 verified.append(item[0])
archive.unlink()
receipt={{'status':'verified_source_csv_only_no_generation','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','source_root':str(root),'csv_count':len(verified),
 'archive_sha256':expected,'source_bytes':sum((root/name).stat().st_size for name in verified)}}
(root/'transfer_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
''')
    finally:
        transport.close()
    path = LOCAL_RECEIPTS / "g11_mother_csv_upload.json"
    path.write_text(json.dumps(verify, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **verify}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("package", "upload"))
    action = parser.parse_args().action
    print(json.dumps({"package": package, "upload": upload}[action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
