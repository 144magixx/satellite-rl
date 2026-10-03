"""Copy verified S1-02 staged test inputs from 20418 to the local archive.

The official multi-GB raw tar remains pinned on 20418 and is recorded by SHA.
This copies the 100 selected CSV snapshots, scene NPZs, and per-aircraft detail.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from remote_sources import LOCAL, REMOTE, connect


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy(sftp, remote_file: str, remote_root: str, local_root: Path,
         expected_sha: str | None = None) -> dict:
    if not remote_file.startswith(remote_root + "/"):
        raise ValueError("Scene source escaped the isolated remote data root")
    relative = remote_file[len(remote_root) + 1:]
    target = local_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        current = digest(target)
        if expected_sha and current != expected_sha:
            raise RuntimeError("Existing local scene SHA differs: " + str(target))
        action = "verified_existing"
    else:
        temporary = target.with_suffix(target.suffix + ".tmp")
        sftp.get(remote_file, str(temporary))
        current = digest(temporary)
        if expected_sha and current != expected_sha:
            temporary.unlink()
            raise RuntimeError("Scene transfer SHA differs: " + remote_file)
        temporary.replace(target)
        action = "downloaded_verified"
    return {"remote_path": remote_file, "local_path": str(target),
            "sha256": current, "bytes": target.stat().st_size, "action": action}


def remote_json(sftp, path: str) -> dict:
    with sftp.open(path, "rb") as handle:
        return json.loads(handle.read())


def sync_stage(sftp, stage: str) -> dict:
    remote_root = REMOTE + "/data/" + stage
    local_root = LOCAL / stage / "data"
    manifest = remote_json(sftp, remote_root + "/independent_test_manifest.json")
    record = remote_json(sftp, remote_root + "/staged_test/record.json")
    verification = remote_json(sftp, remote_root + "/independent_test_verification.json")
    if (manifest.get("status") != "prepared_not_imported_or_evaluated" or
            verification.get("status") != "verified_input_scenes_only_not_model_evaluation" or
            manifest.get("count") != 100 or record.get("count") != 100 or
            verification.get("scene_count") != 100 or
            manifest.get("staged_record_sha256") != verification.get("record_sha256")):
        raise RuntimeError("Remote staged test inputs are not independently verified")
    files = [(remote_root + "/independent_test_manifest.json", None),
             (remote_root + "/independent_test_verification.json", None),
             (remote_root + "/preparation_status.json", None),
             (remote_root + "/raw/download_status.json", None),
             (remote_root + "/staged_test/record.json", verification["record_sha256"])]
    frames = manifest["csv_frames"]["frames"]
    if len(frames) != 100 or len(record["files"]) != 100 or len(record["source"]["snapshots"]) != 100:
        raise RuntimeError("Staged 100-scene provenance is incomplete")
    files += [(frame["path"], frame["sha256"]) for frame in frames]
    files += [(scene["path"], scene["sha256"]) for scene in record["files"]]
    files += [(snapshot["source_observations_ref"]["path"],
               snapshot["source_observations_ref"]["sha256"])
              for snapshot in record["source"]["snapshots"]]
    copied = [copy(sftp, path, remote_root, local_root, sha) for path, sha in files]
    receipt = {"experiment_id": "S1-02", "stage": stage,
               "synced_utc": datetime.now(timezone.utc).isoformat(),
               "remote_data_root": remote_root, "local_data_root": str(local_root),
               "staged_record_sha256": verification["record_sha256"],
               "archive_sha256": verification["archive_sha256"],
               "archive_bytes": verification["archive_bytes"],
               "copied_file_count": len(copied), "files": copied,
               "raw_tar_local_copy": False,
               "note": "Absolute paths inside original source records still name the remote source; file SHA and relative local copies are in this receipt."}
    output = LOCAL / "preflight" / f"scene_sync_{stage}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"stage": stage, "files": len(copied), "receipt": str(output),
            "record_sha256": verification["record_sha256"]}


def sync_development(sftp) -> dict:
    remote_root = REMOTE + "/data"
    local_root = LOCAL / "development" / "data"
    source = remote_json(sftp, remote_root + "/source_copy_receipt.json")
    if source.get("experiment_id") != "S1-02" or set(source.get("roles", {})) != {"train", "validation"}:
        raise RuntimeError("Train/validation copy receipt differs")
    copied = [copy(sftp, remote_root + "/source_copy_receipt.json", remote_root, local_root)]
    for role, expected in (("train", 576), ("validation", 20)):
        entry = source["roles"][role]
        if entry["count"] != expected or len(entry["files"]) != expected + 1:
            raise RuntimeError("Staged development scene count differs")
        for item in entry["files"]:
            remote_file = remote_root + "/source_" + role + "/" + item["name"]
            copied.append(copy(sftp, remote_file, remote_root, local_root, item["sha256"]))
    receipt = {"experiment_id": "S1-02", "stage": "development_sources",
               "synced_utc": datetime.now(timezone.utc).isoformat(),
               "remote_data_root": remote_root, "local_data_root": str(local_root),
               "copied_file_count": len(copied), "files": copied}
    output = LOCAL / "preflight" / "scene_sync_development.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"stage": "development_sources", "files": len(copied), "receipt": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("A", "B", "both", "development"), default="both")
    args = parser.parse_args()
    client = connect()
    try:
        with client.open_sftp() as sftp:
            if args.stage == "development":
                print(json.dumps(sync_development(sftp), ensure_ascii=False))
            else:
                for stage in (("A", "B") if args.stage == "both" else (args.stage,)):
                    print(json.dumps(sync_stage(sftp, stage), ensure_ascii=False))
    finally:
        client.close()


if __name__ == "__main__":
    main()
