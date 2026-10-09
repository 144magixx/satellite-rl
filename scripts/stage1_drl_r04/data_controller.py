"""Idempotent S1-D12A source upload and workbench data preparation.

The only large downloads are performed by the frozen remote prepare jobs.
This controller never runs scientific preparation or evaluation directly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.stage1_drl_r04 import ops


ASSIGNMENT = {20473: ("validation", "2024-02-07"),
              20474: ("test", "2024-02-08"),
              20418: ("test", "2024-02-09")}
FILES = ("source_preflight.json", "selection_contract.json", "release_api.json",
         "PREFERRED_RELEASES.txt", "LICENSE-ODbL-official.txt")
RECEIPT = ops.LOCAL / "controller" / "data_preparation.json"
REGISTRY = ops.LOCAL / "controller" / "dataset_registry.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_dir(role, day):
    return ops.LOCAL / "source" / role / day


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def upload_sources():
    rows = {}
    for port, (role, day) in ASSIGNMENT.items():
        local = source_dir(role, day)
        pin = json.loads((local / "source_preflight.json").read_text(encoding="utf-8"))
        if (pin.get("group") != "S1-D12A" or pin.get("role") != role or
                pin.get("date_utc") != day or pin.get("expected_bytes", 0) <= 0):
            raise ValueError("frozen official source pin differs")
        remote = f"{ops.BASE}/source/{role}/{day}"
        command = f"import pathlib,json;p=pathlib.Path({remote!r});p.mkdir(parents=True,exist_ok=True);print(json.dumps({{'path':str(p)}}))"
        created = ops.remote_python(command, port)
        if created["exit_code"]:
            raise RuntimeError(created["stderr"][-900:])
        files = {}
        for name in FILES:
            target = f"{remote}/{name}"
            local_sha = sha(local / name)
            probe = ops.remote_python(f"import pathlib,hashlib,json;p=pathlib.Path({target!r});print(json.dumps({{'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}}))", port)
            remote_sha = json.loads(probe["stdout"].splitlines()[-1])["sha256"] if probe["exit_code"] == 0 else None
            if remote_sha and remote_sha != local_sha:
                raise ValueError(f"remote frozen source metadata differs: {day}/{name}")
            if not remote_sha:
                transfer = ops.upload(local / name, target, port)
                if transfer["local_sha256"] != transfer["remote"]:
                    raise ValueError(f"official source upload SHA differs: {day}/{name}")
            files[name] = local_sha
        rows[day] = {"role": role, "port": port, "remote": remote,
                     "files": files, "asset_id": pin["asset_id"],
                     "expected_bytes": pin["expected_bytes"]}
    result = {"experiment_id": "S1-D12A", "observed_utc": datetime.now(timezone.utc).isoformat(),
              "uploaded_metadata_only": True, "sources": rows}
    save(ops.LOCAL / "preflight" / "source_metadata_upload.json", result)
    return result


def assert_service(port, commit):
    info = ops.instance(port)
    code = f'''
import json,pathlib,subprocess
repo=pathlib.Path({info['repo']!r});reg=pathlib.Path({info['ops']!r})/'service.json'
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
service=json.loads(reg.read_text());p=pathlib.Path('/proc')/str(service['pid']);raw=(p/'stat').read_text()
ticks=int(raw[raw.rfind(')')+2:].split()[19])
print(json.dumps({{'head':head,'service':service['source_commit'],
 'pid_identity_valid':ticks==service['start_ticks'],
 'archive_root':service['archive_root'],'api_port':service['api_port']}}))
'''
    probe = ops.remote_python(code, port)
    if probe["exit_code"]:
        raise RuntimeError(f"{port} service identity: {probe['stderr'][-600:]}")
    value = json.loads(probe["stdout"].splitlines()[-1])
    if (value["head"] != commit or value["service"] != commit or
            not value["pid_identity_valid"] or value["archive_root"] != info["archive"] or
            value["api_port"] != info["api_port"]):
        raise ValueError(f"{port} frozen service identity differs")
    catalog = ops.api(port, "/api/coverage/catalog")
    if catalog.get("drl_r04_defaults", {}).get("schema_version") != 6:
        raise ValueError(f"{port} schema6 API unavailable")
    return catalog


def submit_preparations(commit):
    if len(commit) != 40:
        raise ValueError("explicit frozen forty-character commit required")
    uploaded = json.loads((ops.LOCAL / "preflight" / "source_metadata_upload.json").read_text(encoding="utf-8"))
    if set(uploaded.get("sources", {})) != {day for role, day in ASSIGNMENT.values()}:
        raise ValueError("three official metadata days must be uploaded first")
    result = json.loads(RECEIPT.read_text(encoding="utf-8")) if RECEIPT.is_file() else {
        "experiment_id": "S1-D12A", "scientific_commit": commit, "tasks": {}}
    if result.get("scientific_commit") != commit:
        raise ValueError("preparation receipt belongs to another frozen commit")
    for port, (role, day) in ASSIGNMENT.items():
        catalog = assert_service(port, commit)
        config = catalog["drl_r04_defaults"]
        config["name"] = f"S1-D12A {role} {day} official ADS-B preparation"
        config["coverage"]["method"] = "none"
        request_id = f"s1d12a_prepare_{day.replace('-', '')}_{port}"
        job = ops.api(port, "/api/coverage/runs", {
            "kind": "prepare", "variant": "dataset", "config": config,
            "dataset_role": role, "source_date": day,
            "experiment_role": "formal", "request_id": request_id})
        if job.get("git_commit") != commit or job.get("request_id") != request_id:
            raise ValueError(f"{port} data preparation lost frozen source/request identity")
        result["tasks"][day] = {"run_id": job["run_id"], "request_id": request_id,
                                "port": port, "role": role, "kind": "prepare",
                                "git_commit": commit}
        result["observed_utc"] = datetime.now(timezone.utc).isoformat()
        save(RECEIPT, result)
    return result


def _prepared(port, task, commit):
    path = f"{ops.instance(port)['archive']}/outputs/workbench/jobs/{task['run_id']}"
    code = f'''
import json,pathlib
root=pathlib.Path({path!r})
print(json.dumps({{name:json.loads((root/(name+'.json')).read_text())
 for name in ('job','status','artifacts')}}))
'''
    result = ops.remote_python(code, port)
    if result["exit_code"]:
        raise RuntimeError(f"{port} prepared data unavailable: {result['stderr'][-800:]}")
    records = json.loads(result["stdout"].splitlines()[-1])
    job, status, artifacts = (records[name] for name in ("job", "status", "artifacts"))
    if (job.get("run_id") != task["run_id"] or job.get("git_commit") != commit or
            job.get("experiment_id") != "S1-D12A" or
            job.get("preparation_mode") != "dataset" or status.get("stage") != "completed" or
            not artifacts.get("prepared_dataset_sha256")):
        raise ValueError("new official dataset task is incomplete or source differs")
    return artifacts


def register_datasets(commit):
    """Register source-referenced NPZ on each lane; merge test only once."""
    if len(commit) != 40:
        raise ValueError("explicit frozen commit required")
    preparations = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if preparations.get("scientific_commit") != commit:
        raise ValueError("preparation source commit differs")
    prior = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.is_file() else {
        "experiment_id": "S1-D12A", "scientific_commit": commit, "lanes": {}}
    if prior.get("scientific_commit") != commit:
        raise ValueError("dataset registry belongs to another frozen commit")
    source_probe = json.loads((ops.LOCAL / "preflight" / "source_probe.json").read_text(encoding="utf-8"))
    historical = source_probe["historical_records"]
    prepared = {}
    for port, (role, day) in ASSIGNMENT.items():
        task = preparations["tasks"][day]
        if task["port"] != port or task["role"] != role:
            raise ValueError("official preparation lane/date/role differs")
        prepared[day] = {**_prepared(port, task, commit), "port": port,
                         "run_id": task["run_id"], "role": role}

    def ensure(port, label, payload, count, sha256):
        lane = prior["lanes"].setdefault(str(port), {})
        catalog = assert_service(port, commit)
        saved = lane.get(label)
        if saved:
            matches = [r for r in ops.api(port, "/api/coverage/datasets")["datasets"]
                       if r.get("id") == saved["id"]]
            if len(matches) != 1 or matches[0].get("sha256") != sha256 or matches[0].get("count") != count:
                raise ValueError(f"{port} {label} saved record changed")
            return saved
        record = ops.api(port, "/api/coverage/datasets/import", {
            "schema_version": 6, "name": f"S1-D12A {label}", **payload})
        if record.get("sha256") != sha256 or record.get("count") != count:
            raise ValueError(f"{port} {label} imported SHA/count differs")
        lane[label] = {"id": record["id"], "sha256": sha256,
                       "count": count, "path": record["path"]}
        prior["observed_utc"] = datetime.now(timezone.utc).isoformat()
        save(REGISTRY, prior)
        return lane[label]

    for port in ops.SSH_PORTS:
        for role, count in (("train", 576), ("development", 20)):
            rows = [row for row in historical if row.get("port") == str(port) and
                    row.get("role") == role and row.get("count") == count and row.get("exists")]
            if len(rows) != 1:
                raise ValueError(f"{port} historical {role} source missing or ambiguous")
            source = rows[0]
            ensure(port, role, {"partition_role": role,
                               "path": source["path"], "sha256": source["sha256"]},
                   count, source["sha256"])
        validation = prepared["2024-02-07"]
        ensure(port, "validation", {"partition_role": "validation",
                                    "path": validation["prepared_dataset"],
                                    "sha256": validation["prepared_dataset_sha256"],
                                    "source_port": validation["port"],
                                    "source_run_id": validation["run_id"]},
               20, validation["prepared_dataset_sha256"])

    # Both daily test sources are copied into 20418's scene registry. The
    # merged 100-mother NPZ is generated once and imported by the other lanes.
    for day in ("2024-02-08", "2024-02-09"):
        source = prepared[day]
        ensure(20418, "test_" + day, {"partition_role": "test",
                                      "path": source["prepared_dataset"],
                                      "sha256": source["prepared_dataset_sha256"],
                                      "source_port": source["port"],
                                      "source_run_id": source["run_id"]},
               50, source["prepared_dataset_sha256"])
    lane = prior["lanes"]["20418"]
    if "test" not in lane:
        merged = ops.api(20418, "/api/coverage/datasets/merge", {
            "schema_version": 6, "partition_role": "test",
            "dataset_ids": [lane["test_2024-02-08"]["id"], lane["test_2024-02-09"]["id"]]})
        if merged.get("count") != 100 or merged.get("dates") != ["2024-02-08", "2024-02-09"]:
            raise ValueError("merged 100-mother independent test set differs")
        lane["test"] = {"id": merged["id"], "sha256": merged["sha256"],
                        "count": 100, "path": merged["path"]}
        save(REGISTRY, prior)
    for port in (20424, 20473, 20474):
        source = lane["test"]
        ensure(port, "test", {"partition_role": "test", "path": source["path"],
                              "sha256": source["sha256"]}, 100, source["sha256"])
    return prior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "submit", "register"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action in ("submit", "register") and not args.commit:
        parser.error("--commit is required for workbench submission or registration")
    value = (upload_sources() if args.action == "upload" else
             submit_preparations(args.commit) if args.action == "submit" else
             register_datasets(args.commit))
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
