"""Idempotent workbench preparation and four-lane dataset registration.

Run only after the scientific source commit has passed validation and all four
isolated services are listening. This controller submits no training/evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import subprocess
import hashlib

from scripts.stage1_variable import ops
from scripts.stage1_variable.launch_downloads import ASSIGNMENT
from scripts.stage1_variable.probe_existing_aircraft import SOURCE as HISTORICAL_SOURCE, EXPECTED as HISTORICAL_SHA
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from satellite_rl.workbench.coverage_variable import (PLAN_REVISION, ROLE_COUNTS, ROLE_DATES,
    SOURCE_CONTRACT_SHA, SOURCE_PREFLIGHT_SHA)


RECEIPT = ops.LOCAL / "controller" / "data_preparation.json"
REGISTRY = ops.LOCAL / "controller" / "dataset_registry.json"
CONTRACT = ops.LOCAL / "controller" / "canonical_dataset_contract.json"
PREPARATION_SOURCE_COMMIT = "ff25076e9f98b8111cc9671cb60e57cfc3f99760"
PREPARATION_IDS = {
    20418: ("prepare_ppo_20261009_094017_5797b0d1", "s1v10_prepare_20240203_20418"),
    20424: ("prepare_ppo_20261009_094021_263060b8", "s1v10_prepare_20240204_20424"),
    20473: ("prepare_ppo_20261009_094025_6201b255", "s1v10_prepare_20240205_20473"),
    20474: ("prepare_ppo_20261009_094029_6436bd1f", "s1v10_prepare_20240206_20474"),
}
PREPARATION_ARCHIVE_SHA = {
    "2024-02-03": "e7ebb945107ede71f75ca929a32d9f978c7ab532542f28e182a0201a1611717a",
    "2024-02-04": "c816b523be2da6c47cf1699005dc4cc8dc0195d4593249fcc62c7d4a35e1c626",
    "2024-02-05": "deecc80209df730e35c3bef0ad8b5c2370fea0351ec72d8de14243f344d1d4ee",
    "2024-02-06": "cd67a4f7bf09597cb55d5eb0631b79458005f8ae2ac767d078fad46dbd0e7038",
}
ALLOWED_CONTINUATION_FILES = {
    "src/satellite_rl/workbench/coverage.py",
    "src/satellite_rl/workbench/static/coverage.js",
    "scripts/stage1_variable/data_controller.py",
    "scripts/stage1_variable/controller.py",
    "scripts/stage1_variable/archive_inventory.py",
    "tests/test_stage1_variable_controller.py",
}
SCIENTIFIC_MODULES = (
    "src/satellite_rl/coverage/variable_config.py",
    "src/satellite_rl/coverage/variable_dataset.py",
    "src/satellite_rl/coverage/variable_design.py",
    "src/satellite_rl/coverage/variable_downstream.py",
    "src/satellite_rl/coverage/variable_observation.py",
    "src/satellite_rl/coverage/variable_ppo.py",
    "src/satellite_rl/coverage/variable_runner.py",
    "src/satellite_rl/workbench/coverage_variable.py",
    "src/satellite_rl/workbench/variable_preparation.py",
    "src/satellite_rl/workbench/storage.py",
    "src/satellite_rl/workbench/runner.py",
    "src/satellite_rl/workbench/runtime.py",
    "src/satellite_rl/coverage/config.py",
)


def continuation_audit(commit):
    """Require a UI/ops-only descendant and identical frozen scientific files."""
    source = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.run(["git", "-C", str(source), *args],
                              capture_output=True, check=True).stdout
    if git("rev-parse", "HEAD").decode().strip() != commit:
        raise RuntimeError("continuation checkout does not match requested scientific commit")
    if git("status", "--porcelain", "--", "src", "scripts/stage1_variable", "tests").strip():
        raise RuntimeError("continuation source has uncommitted changes")
    if git("merge-base", PREPARATION_SOURCE_COMMIT, commit).decode().strip() != PREPARATION_SOURCE_COMMIT:
        raise RuntimeError("continuation is not descended from frozen preparation source")
    changed = set(git("diff", "--name-only", PREPARATION_SOURCE_COMMIT, commit).decode().splitlines())
    if changed - ALLOWED_CONTINUATION_FILES:
        raise RuntimeError(f"science or data module changed in UI continuation: {sorted(changed - ALLOWED_CONTINUATION_FILES)}")
    scientific = {}
    for path in SCIENTIFIC_MODULES:
        old = git("show", PREPARATION_SOURCE_COMMIT + ":" + path)
        new = git("show", commit + ":" + path)
        if old != new:
            raise RuntimeError(f"frozen scientific module content changed: {path}")
        scientific[path] = hashlib.sha256(old).hexdigest()
    result = {"preparation_source_commit": PREPARATION_SOURCE_COMMIT,
              "scientific_commit": commit,
              "changed_files": sorted(changed),
              "scientific_module_sha256": scientific}
    save(ops.LOCAL / "preflight" / f"ui_continuation_{commit[:12]}.json", result)
    return result


def existing_prepares(commit):
    continuation_audit(commit)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt["scientific_commit"] != PREPARATION_SOURCE_COMMIT or set(receipt["prepares"]) != set(map(str, PREPARATION_IDS)):
        raise RuntimeError("frozen four preparation IDs are incomplete")
    for port, (run_id, request_id) in PREPARATION_IDS.items():
        role, day = ASSIGNMENT[port]
        row = receipt["prepares"][str(port)]
        if (row.get("run_id") != run_id or row.get("request_id") != request_id or
                row.get("date") != day or row.get("role") != role or
                row.get("git_commit") != PREPARATION_SOURCE_COMMIT):
            raise RuntimeError(f"{port}: preparation identity differs from frozen handoff")
    return receipt["prepares"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def api(port, endpoint, body=None):
    reply = ops.api(port, endpoint, body)
    if reply["status"] != 200:
        raise RuntimeError(f"{port} {endpoint} HTTP {reply['status']}: {reply.get('error')}")
    return reply["body"]


def assert_service(port, commit):
    info = ops.instance(port)
    code = f'''
import json,pathlib,subprocess
repo=pathlib.Path({info['repo']!r});registry=pathlib.Path({info['ops']!r})/'service.json'
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
service=json.loads(registry.read_text())
p=pathlib.Path('/proc')/str(service['pid']);raw=(p/'stat').read_text()
ticks=int(raw[raw.rfind(')')+2:].split()[19])
print(json.dumps({{'head':head,'service':service['source_commit'],
 'pid_identity_valid':ticks==service['start_ticks'],
 'archive_root':service['archive_root'],'api_port':service['api_port']}}))
'''
    result = ops.remote_python(code, port)
    if result["exit_code"]:
        raise RuntimeError(f"{port}: verified service unavailable: {result['stderr'][-600:]}")
    state = json.loads(result["stdout"].splitlines()[-1])
    if (state["head"] != commit or state["service"] != commit or
            not state["pid_identity_valid"] or state["archive_root"] != info["archive"] or
            state["api_port"] != info["api_port"]):
        raise RuntimeError(f"{port}: workbench identity differs from frozen source")
    catalog = api(port, "/api/coverage/catalog")
    if catalog.get("variable_defaults", {}).get("schema_version") != 5:
        raise RuntimeError(f"{port}: schema-5 API unavailable")
    return catalog


def submit_prepares(commit):
    if commit != PREPARATION_SOURCE_COMMIT:
        return existing_prepares(commit)
    rows = {}
    for port, (role, day) in ASSIGNMENT.items():
        catalog = assert_service(port, commit)
        config = catalog["variable_defaults"]
        config["name"] = f"S1-V10 {role} {day} raw aircraft preparation"
        request_id = f"s1v10_prepare_{day.replace('-', '')}_{port}"
        job = api(port, "/api/coverage/runs", {
            "kind": "prepare", "config": config, "dataset_role": role,
            "source_date": day, "request_id": request_id,
            "experiment_role": "formal"})
        if job.get("git_commit") != commit or job.get("request_id") != request_id:
            raise RuntimeError(f"{port}: submitted preparation lost frozen source/request identity")
        rows[str(port)] = {"port": port, "role": role, "date": day,
                           "run_id": job["run_id"], "request_id": request_id,
                           "git_commit": commit}
        save(RECEIPT, {"scientific_commit": commit, "prepares": rows,
                       "observed_utc": datetime.now(timezone.utc).isoformat()})
    return rows


def _dataset_rows(port):
    return api(port, "/api/coverage/datasets")["datasets"]


def _known_dataset(port, role, sha):
    return next((r for r in _dataset_rows(port)
                 if r.get("schema_version") == 5 and r.get("partition_role") == role
                 and r.get("sha256") == sha), None)


def _remote_sha(port, path):
    code = f'''
import hashlib,json,pathlib
path=pathlib.Path({path!r});h=hashlib.sha256()
with path.open('rb') as stream:
 for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
print(json.dumps({{'sha256':h.hexdigest(),'bytes':path.stat().st_size}}))
'''
    response = ops.remote_python(code, port, timeout=120)
    if response["exit_code"]:
        raise RuntimeError(f"{port}: source SHA probe failed: {response['stderr'][-500:]}")
    return json.loads(response["stdout"].splitlines()[-1])["sha256"]


def _import(port, path, role, expected_sha=None, expected_source_sha=None):
    if expected_source_sha and _remote_sha(port, path) != expected_source_sha:
        raise RuntimeError(f"{port} {role}: historical aircraft input differs from fixed source SHA")
    if expected_sha:
        known = _known_dataset(port, role, expected_sha)
        if known:
            return known
    else:
        known = next((r for r in _dataset_rows(port)
                      if r.get("schema_version") == 5 and r.get("partition_role") == role
                      and r.get("source_path") == path and
                      (expected_source_sha is None or r.get("source_sha256") == expected_source_sha)), None)
        if known:
            return known
    row = api(port, "/api/coverage/datasets/import", {
        "schema_version": 5, "partition_role": role, "path": path,
        "name": f"S1-V10 {role} common source"})
    if expected_sha and row["sha256"] != expected_sha:
        raise RuntimeError(f"{port} {role} imported dataset SHA differs")
    if expected_source_sha and row.get("source_sha256") != expected_source_sha:
        raise RuntimeError(f"{port} {role} imported historical provenance SHA differs")
    return row


def register_datasets(commit):
    prepares = existing_prepares(commit)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if set(prepares) != set(map(str, ASSIGNMENT)):
        raise RuntimeError("all four frozen preparation IDs are required")
    prepared = {}
    source_records = {}
    for port, (role, day) in ASSIGNMENT.items():
        assert_service(port, commit)
        task = receipt["prepares"][str(port)]
        detail = api(port, "/api/runs/" + task["run_id"])
        if (detail["status"]["stage"] != "completed" or
                detail["job"].get("git_commit") != PREPARATION_SOURCE_COMMIT or
                detail["job"].get("run_id") != PREPARATION_IDS[port][0] or
                detail["job"].get("request_id") != PREPARATION_IDS[port][1] or
                detail["job"].get("variable_preparation", {}).get("archive", {}).get("sha256") !=
                PREPARATION_ARCHIVE_SHA[day] or
                detail["job"].get("variable_preparation", {}).get("date_utc") != day):
            raise RuntimeError(f"{port} {day}: preparation not completed at frozen commit")
        artifact = detail["artifacts"]
        record = next((r for r in _dataset_rows(port)
                       if r.get("id") == artifact.get("prepared_dataset_id")), None)
        if record is None or record["sha256"] != artifact["prepared_dataset_sha256"]:
            raise RuntimeError(f"{port} {day}: prepared dataset record differs")
        prepared[day] = record
        source_records[day] = {"asset_id": detail["job"]["variable_preparation"]["archive"]["asset_id"],
                               "preparation_source_commit": PREPARATION_SOURCE_COMMIT,
                               "archive_sha256": artifact["raw_archive_sha256"],
                               "source_preflight_sha256": artifact["source_preflight_sha256"],
                               "selection_contract_sha256": artifact["selection_contract_sha256"],
                               "selected_times_sha256": artifact["selected_times_sha256"],
                               "prepared_dataset_sha256": record["sha256"],
                               "prepared_dataset_id": record["id"],
                               "source_run_id": task["run_id"]}
        if (source_records[day]["source_preflight_sha256"] != SOURCE_PREFLIGHT_SHA[day] or
                source_records[day]["selection_contract_sha256"] != SOURCE_CONTRACT_SHA[day]):
            raise RuntimeError(f"{day}: workbench frozen source contract differs from plan revision")
    train = _import(20418, HISTORICAL_SOURCE + "/train/inputs.npz", "train",
                    expected_source_sha=HISTORICAL_SHA["train"])
    development = _import(20418, HISTORICAL_SOURCE + "/validation/inputs.npz", "development",
                          expected_source_sha=HISTORICAL_SHA["validation"])
    validation = prepared["2024-02-03"]
    test_feb04 = prepared["2024-02-04"]
    test_feb05 = _import(20424, prepared["2024-02-05"]["path"], "test",
                         prepared["2024-02-05"]["sha256"])
    expected_sources = [{"id": test_feb04["id"], "sha256": test_feb04["sha256"],
                         "dates": test_feb04["dates"]},
                        {"id": test_feb05["id"], "sha256": test_feb05["sha256"],
                         "dates": test_feb05["dates"]}]
    previous = next((r for r in _dataset_rows(20424)
                     if r.get("schema_version") == 5 and r.get("partition_role") == "test"
                     and r.get("dates") == ["2024-02-04", "2024-02-05"]
                     and r.get("count") == 100 and
                     r.get("source", {}).get("source_datasets") == expected_sources), None)
    test = previous or api(20424, "/api/coverage/datasets/merge", {
        "partition_role": "test", "dataset_ids": [test_feb04["id"], test_feb05["id"]],
        "name": "S1-V10 independent Feb04+05 100 mother scenes"})
    sequence = prepared["2024-02-06"]
    canonical = {"train": train, "development": development,
                 "validation": validation, "test": test, "sequence": sequence}
    if (train["count"] != 576 or development["count"] != 20 or
            validation["count"] != 20 or test["count"] != 100 or
            sequence["count"] != 120):
        raise RuntimeError("S1-V10 canonical mother-scene counts differ")
    lanes = {}
    for port in ops.SSH_PORTS:
        assert_service(port, commit)
        lanes[str(port)] = {}
        for role, source in canonical.items():
            row = _import(port, source["path"], role, source["sha256"])
            lanes[str(port)][role] = {"id": row["id"], "path": row["path"],
                                      "sha256": row["sha256"],
                                      "count": row["count"], "dates": row["dates"]}
        save(REGISTRY, {"scientific_commit": commit, "lanes": lanes,
                        "canonical_sha256": {role: row["sha256"] for role, row in canonical.items()},
                        "observed_utc": datetime.now(timezone.utc).isoformat()})
    if any(lane[role]["sha256"] != canonical[role]["sha256"]
           for lane in lanes.values() for role in canonical):
        raise RuntimeError("four-lane canonical dataset content SHA differs")
    contract = {"experiment_id": "S1-V10", "plan_revision": PLAN_REVISION,
                "scientific_commit": commit,
                "preparation_source_commit": PREPARATION_SOURCE_COMMIT,
                "datasets": {role: {"sha256": row["sha256"], "count": ROLE_COUNTS[role],
                                    "dates": ROLE_DATES[role],
                                    "source_datasets": row.get("source", {}).get("source_datasets")}
                             for role, row in canonical.items()},
                "historical_source_sha256": HISTORICAL_SHA,
                "source_dates": source_records}
    save(CONTRACT, contract)
    payload = CONTRACT.read_bytes()
    import base64
    encoded = base64.b64encode(payload).decode("ascii")
    remote_contract = f"{ops.BASE}/source/canonical_dataset_contract.json"
    code = f'''
import base64,hashlib,json,pathlib
path=pathlib.Path({remote_contract!r});value=base64.b64decode({encoded!r})
if path.exists() and path.read_bytes()!=value:raise RuntimeError('frozen canonical dataset contract changed')
if not path.exists():path.write_bytes(value)
print(json.dumps({{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}))
'''
    written = ops.remote_python(code, 20418, timeout=120)
    if written["exit_code"]:
        raise RuntimeError("canonical dataset contract write failed: " + written["stderr"][-600:])
    remote_receipt = json.loads(written["stdout"].splitlines()[-1])
    if remote_receipt["sha256"] != __import__("hashlib").sha256(payload).hexdigest():
        raise RuntimeError("remote canonical dataset contract SHA differs")
    save(ops.LOCAL / "controller" / "canonical_dataset_contract_receipt.json", remote_receipt)
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def stage_throughput_canonical_contract(repair_commit):
    """Create an immutable same-data, learned-evaluation-only sidecar."""
    from satellite_rl.workbench.variable_throughput_recovery import (
        DATA_CONTRACT_SHA256, ORIGIN_COMMIT, REPAIR_ID)
    if not isinstance(repair_commit, str) or len(repair_commit) != 40:
        raise ValueError("exact throughput repair commit required")
    original = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (hashlib.sha256(CONTRACT.read_bytes()).hexdigest() != DATA_CONTRACT_SHA256 or
            original.get("scientific_commit") != ORIGIN_COMMIT or
            original.get("datasets") is None):
        raise RuntimeError("original S1-V10 canonical dataset contract differs")
    continuation = {**original,
                    "format": "s1v10_throughput_canonical_continuation_v1",
                    "repair_id": REPAIR_ID,
                    "origin_commit": ORIGIN_COMMIT,
                    "repair_commit": repair_commit,
                    "origin_contract_sha256": DATA_CONTRACT_SHA256,
                    "allowed_eval_only": True,
                    "allowed_route_port": {"R08": "20418", "R09": "20424"},
                    "scientific_commit": repair_commit}
    local = CONTRACT.with_name(f"canonical_dataset_contract_throughput_{repair_commit}.json")
    save(local, continuation)
    payload = local.read_bytes()
    remote_path = f"{ops.BASE}/source/{local.name}"
    import base64
    encoded = base64.b64encode(payload).decode("ascii")
    code = f'''
import base64,hashlib,json,pathlib
path=pathlib.Path({remote_path!r});payload=base64.b64decode({encoded!r})
if path.exists() and path.read_bytes()!=payload:raise RuntimeError('throughput canonical sidecar changed')
if not path.exists():path.write_bytes(payload)
print(json.dumps({{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}))
'''
    result = ops.remote_python(code, 20418, timeout=120)
    if result["exit_code"]:
        raise RuntimeError("throughput canonical sidecar write failed: " + result["stderr"][-500:])
    receipt = json.loads(result["stdout"].splitlines()[-1])
    if receipt["sha256"] != hashlib.sha256(payload).hexdigest():
        raise RuntimeError("throughput canonical sidecar remote SHA differs")
    save(ops.LOCAL / "controller" / "throughput_canonical_contract_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit-prepares", "register-datasets"))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if len(args.commit) != 40:
        parser.error("--commit must be an exact frozen SHA")
    result = submit_prepares(args.commit) if args.action == "submit-prepares" else register_datasets(args.commit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
