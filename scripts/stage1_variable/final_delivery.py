"""Gate and ship the small S1-V10 report package to one shared remote path.

The 32 audited job ZIPs remain in their own archives.  This package contains
only the formal report, derived figures/tables, diagnostics, source/index
receipts and the two small final policies.  ``plan`` is safe before publication;
``deliver`` fails closed until every formal report gate has passed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from scripts.stage1_variable import ops


BASE = ops.LOCAL
REPO = Path(__file__).resolve().parents[2]
MAIN_REPO = Path("D:/paper")
DELIVERY = BASE / "delivery"
REPORT_MD = MAIN_REPO / "docs/第一阶段/实验报告/08_可变波束全覆盖与负载均衡十二路线对照.md"
FINAL_REPORT = BASE / "analysis/final_report"
PUBLICATION = BASE / "analysis/publication_figures/final"
GATE = BASE / "analysis/final_report_controller/state.json"
ACCEPTANCE = BASE / "preflight/final_32_archive_acceptance.json"
ROOT_VERIFY = BASE / "preflight/final_32_archive_root_verification.json"
MANIFEST = BASE / "synced_jobs/manifest.json"
SCIENTIFIC_COMMITS = {
    "preparation": "ff25076e9f98b8111cc9671cb60e57cfc3f99760",
    "original": "b31187fcaf79e90b70d85db4cdfec5a4150020de",
    "learning_repair": "13261cd752a06b6ac725705136fe0f017072159a",
}
ARCHIVE_OPS_COMMIT = "64b10efc794495ab03b2e3831953f1748a33ee17"
REPORT_COMMIT = "94b615cb6643b04f7073b704291fd5f1460b60cb"
MAX_FILES = 3000
MAX_FILE_BYTES = 200 * (1 << 20)
MAX_TOTAL_BYTES = 2 * (1 << 30)
LOCAL_RESERVE_BYTES = 8 * (1 << 30)
ALLOWED_SUFFIXES = {".md", ".json", ".csv", ".png", ".svg", ".pdf",
                    ".py", ".txt", ".pt"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".part")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)
    return path


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("S1-V10 delivery source escapes declared root: " + str(path))
    return path


def _file(path, *, root=None):
    path = Path(path)
    if root is not None:
        _inside(path, root)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("S1-V10 delivery source absent, linked or too large: " + str(path))
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("S1-V10 delivery refuses raw/binary archive member: " + str(path))
    return path


def _tree(root, prefix):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("formal result directory absent: " + str(root))
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("linked formal result refused: " + str(path))
        if path.is_file():
            _file(path, root=root)
            name = prefix + "/" + path.relative_to(root).as_posix()
            if name in result:
                raise ValueError("duplicate delivery member: " + name)
            result[name] = path
    if not result:
        raise ValueError("formal result directory empty: " + str(root))
    return result


def gate_state(report_md=REPORT_MD, publication=PUBLICATION):
    """Return independent formal gates; no file is packaged on failure."""
    errors = []
    paths = {"report_markdown": Path(report_md), "strict_summary": FINAL_REPORT / "summary.json",
             "science_audit": FINAL_REPORT / "science_audit/science_audit.json",
             "report_gate": GATE, "archive_acceptance": ACCEPTANCE,
             "root_verification": ROOT_VERIFY, "archive_manifest": MANIFEST,
             "publication_receipt": Path(publication) / "publication_receipt.json"}
    for label, path in paths.items():
        if not path.is_file():
            errors.append(label + " absent: " + str(path))
    if errors:
        return {"ready": False, "errors": errors,
                "expected_paths": {key: str(path) for key, path in paths.items()}}
    try:
        markdown = Path(report_md)
        if len(markdown.read_text(encoding="utf-8")) < 1000:
            errors.append("formal report Markdown is unexpectedly short")
        acceptance, root_check, manifest = read(ACCEPTANCE), read(ROOT_VERIFY), read(MANIFEST)
        report_gate, summary, science = read(GATE), read(FINAL_REPORT / "summary.json"), read(
            FINAL_REPORT / "science_audit/science_audit.json")
        publication_receipt = read(paths["publication_receipt"])
        manifest_sha = sha(MANIFEST)
        if (acceptance.get("passed") is not True or acceptance.get("package_count") != 32 or
                acceptance.get("science_packages") != 28 or
                acceptance.get("preparation_packages") != 4 or
                acceptance.get("manifest_sha256") != manifest_sha or
                acceptance.get("all_job_zip_and_member_shas_reverified") is not True or
                acceptance.get("all_shared_source_member_shas_reverified") is not True):
            errors.append("32-task archive acceptance/manifest SHA differs")
        if (root_check.get("acceptance_sha256") != sha(ACCEPTANCE) or
                root_check.get("acceptance_manifest_matches_actual_and_report_snapshot") is not True or
                root_check.get("package_count") != 32):
            errors.append("root independent archive acceptance differs")
        if (report_gate.get("stage") != "complete" or report_gate.get("formal_ready") is not True or
                report_gate.get("source_commit") != REPORT_COMMIT or
                report_gate.get("manifest_sha256") != manifest_sha or
                report_gate.get("summary_sha256") != sha(FINAL_REPORT / "summary.json") or
                report_gate.get("per_scene_sha256") != sha(FINAL_REPORT / "per_scene.csv") or
                report_gate.get("plots_sha256") != sha(FINAL_REPORT / "plots.json") or
                report_gate.get("science_audit_sha256") != sha(paths["science_audit"])):
            errors.append("strict report gate is incomplete or derived SHA differs")
        if (summary.get("formal_ready") is not True or summary.get("issues") or
                summary.get("task_count") != 28 or summary.get("audited_task_count") != 28 or
                science.get("formal_ready") is not True or science.get("static_rows") != 3600 or
                science.get("temporal_rows") != 720):
            errors.append("formal science report grid/readiness differs")
        if (publication_receipt.get("formal_ready_input") is not True or
                publication_receipt.get("test_rows") != 3600 or
                publication_receipt.get("sequence_rows") != 720 or
                publication_receipt.get("static_sequence_rows_compared") != 4320 or
                publication_receipt.get("input_per_scene_sha256") !=
                sha(FINAL_REPORT / "per_scene.csv") or
                publication_receipt.get("input_summary_sha256") !=
                sha(FINAL_REPORT / "summary.json") or
                publication_receipt.get("script_sha256") !=
                sha(BASE / "analysis/publication_figures/build_figures.py") or
                publication_receipt.get("passed", True) is not True):
            errors.append("publication final receipt does not bind strict formal inputs")
        outputs = publication_receipt.get("outputs") or {}
        if not outputs or not any(name.endswith(".png") for name in outputs):
            errors.append("formal publication figure is absent")
        for name, item in outputs.items():
            path = _inside(item["path"], publication)
            if path.name != name or sha(path) != item["sha256"]:
                errors.append("formal publication output SHA differs: " + name)
        tasks = manifest.get("tasks") or {}
        expected_tasks = {f"prepare_{port}" for port in ops.SSH_PORTS}
        expected_tasks.update(read(BASE / "controller/state.json")["tasks"])
        if (len(tasks) != 32 or manifest.get("shared_source_dependencies_closed") is not True or
                set(tasks) != expected_tasks):
            errors.append("archive manifest task graph/shared source differs")
        if any(not row.get("archive_receipt") for row in tasks.values()):
            errors.append("one archived task lacks a receipt")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        errors.append("formal source inspection failed: " + str(error))
    return {"ready": not errors, "errors": errors,
            "expected_paths": {key: str(path) for key, path in paths.items()},
            "manifest_sha256": sha(MANIFEST) if MANIFEST.is_file() else None}


def _archive_index(manifest):
    index = {"schema": "s1v10.delivery_archive_index.v1",
             "manifest_path": str(MANIFEST), "manifest_sha256": sha(MANIFEST), "tasks": {}}
    for key, task in sorted(manifest["tasks"].items()):
        receipt_path = Path(task["archive_receipt"])
        receipt = read(receipt_path)
        download_path = receipt_path.with_name("download_receipt.json")
        download = read(download_path)
        if (download.get("all_volume_and_member_shas_verified") is not True or
                download.get("remote_receipt_sha256") != sha(receipt_path) or
                receipt.get("task_key") != key or receipt.get("run_id") != task["run_id"] or
                receipt.get("port") != task["port"]):
            raise ValueError("task archive receipt identity differs: " + key)
        index["tasks"][key] = {"run_id": task["run_id"], "port": task["port"],
                              "source_commit": receipt["source_commit"],
                              "source_manifest_sha256": receipt["source_manifest_sha256"],
                              "receipt_path": str(receipt_path),
                              "receipt_sha256": sha(receipt_path),
                              "download_receipt_path": str(download_path),
                              "download_receipt_sha256": sha(download_path),
                              "member_count": receipt["member_count"],
                              "original_bytes": receipt["total_input_bytes"],
                              "volumes": [{"name": row["name"], "bytes": row["bytes"],
                                           "sha256": row["sha256"],
                                           "local_path": (download.get("local_volume_paths") or {}).get(
                                               row["name"], str(receipt_path.parent / row["name"]))}
                                          for row in receipt["volumes"]],
                              "external_dependencies": receipt.get("external_dependencies", [])}
    return index


def _git_index(archive_index):
    """Identify local Git commits, four remote checkouts and archived source refs."""
    commits = dict(SCIENTIFIC_COMMITS, archive_ops=ARCHIVE_OPS_COMMIT,
                   report=REPORT_COMMIT)
    local = {}
    for label, commit in commits.items():
        result = subprocess.run(["git", "-C", str(REPO), "cat-file", "-t", commit],
                                capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != "commit":
            raise RuntimeError("local Git commit absent: " + label)
        local[label] = {"commit": commit, "repository": str(REPO)}
    remote = {}
    for port in ops.SSH_PORTS:
        info = ops.instance(port)
        code = f'''
import json,pathlib,subprocess
root=pathlib.Path({info['root']!r});service=json.loads((root/'ops/service.json').read_text())
head=subprocess.run(['git','-C',str(root/'repo'),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
print(json.dumps({{'port':{port},'repo':str(root/'repo'),'head':head,
                  'service_source_commit':service['source_commit'],
                  'service_pid':service['pid'],'service_start_ticks':service['start_ticks']}}))
'''
        response = ops.remote_python(code, port, timeout=60)
        if response["exit_code"]:
            raise RuntimeError(f"{port}: remote Git source read failed: " + response["stderr"][-600:])
        row = json.loads(response["stdout"].splitlines()[-1])
        if row["head"] != row["service_source_commit"]:
            raise RuntimeError(f"{port}: remote Git/service source differs")
        remote[str(port)] = row
    source_refs = {}
    manifest = read(MANIFEST)
    for key, task in archive_index["tasks"].items():
        commit = task["source_commit"]
        if commit in source_refs:
            continue
        ref = next((row.get("source_ref") for row in manifest["files"]
                    if row["run_id"] == task["run_id"] and row["port"] == task["port"] and
                    row["name"] == "source/source_manifest.json"), None)
        if not ref or ref.get("sha256") != task["source_manifest_sha256"]:
            raise RuntimeError("archived source manifest ref missing for commit " + commit)
        source_refs[commit] = ref
    if not set(SCIENTIFIC_COMMITS.values()) <= set(source_refs):
        raise RuntimeError("one scientific source commit lacks archived source snapshot")
    return {"schema": "s1v10.delivery_git_sources.v1",
            "local_commits": local, "remote_checkout_and_service": remote,
            "archived_source_manifest_refs": source_refs,
            "report_gate_six_source_sha256": read(GATE)["source_sha256"]}


def _members(report_md, publication, metadata):
    files = {"report/" + Path(report_md).name: _file(report_md, root=MAIN_REPO / "docs")}
    for source in (_tree(FINAL_REPORT, "analysis/final_report"),
                   _tree(publication, "analysis/publication_final")):
        for name, path in source.items():
            if name in files:
                raise ValueError("duplicate S1-V10 delivery member: " + name)
            files[name] = path
    fixed = {
        "evidence/final_32_archive_acceptance.json": ACCEPTANCE,
        "evidence/final_32_archive_root_verification.json": ROOT_VERIFY,
        "evidence/run_manifest.json": BASE / "run_manifest.json",
        "evidence/source_bundle_receipt.json": BASE / "delivery/source_bundle_receipt.json",
        "evidence/shared_source_closure_receipt.json": BASE / "job_archives/shared_source_closure_receipt.json",
        "evidence/report_gate_state.json": GATE,
        "scripts/final_science_audit.py": BASE / "analysis/final_science_audit.py",
        "scripts/final_report_controller.py": BASE / "analysis/final_report_controller.py",
        "scripts/final_archive_audit.py": BASE / "preflight/final_archive_audit.py",
        "scripts/publication_figures.py": BASE / "analysis/publication_figures/build_figures.py",
        "scripts/report.py": MAIN_REPO / "scripts/stage1_variable/report.py",
        "scripts/archive_sources.py": MAIN_REPO / "scripts/s2p19/archive_sources.py",
        "scripts/coverage_controller.py": MAIN_REPO / "scripts/stage1_variable/controller.py",
        "scripts/archive_jobs.py": REPO / "scripts/stage1_variable/archive_jobs.py",
        "scripts/archive_background.py": REPO / "scripts/stage1_variable/archive_background.py",
        "scripts/final_delivery.py": Path(__file__).resolve(),
        "scripts/sgm_physics.py": MAIN_REPO / "src/satellite_rl/power_refinement/physics.py",
    }
    for name, path in fixed.items():
        if name in files:
            raise ValueError("duplicate S1-V10 delivery member: " + name)
        files[name] = _file(path)
    archive_index = metadata["archive_index"]
    for key, task in archive_index["tasks"].items():
        for name, path in (("receipt.json", task["receipt_path"]),
                           ("download_receipt.json", task["download_receipt_path"])):
            files[f"archive_receipts/{key}/{name}"] = _file(path)
    for route in ("R08", "R09"):
        model = read(ACCEPTANCE)["models"][route]
        files[f"models/{route}/export_receipt.json"] = _file(model["receipt"])
        for item in model["files"]:
            files[f"models/{route}/{Path(item['path']).name}"] = _file(item["path"])
    for name, path in {
        "diagnostics/gpu_one_rollout_audit.json": BASE / "preflight/gpu_one_rollout_audit_partial_loader.json",
        "diagnostics/throughput_warm8_R09.json": BASE / "preflight/throughput_warm8_R09.json",
        "diagnostics/throughput_diagnosis.json": BASE / "preflight/throughput_diagnosis.json",
        "diagnostics/repair_downtime.json": BASE / "ops/throughput_repair/downtime_summary.json",
    }.items():
        files[name] = _file(path)
    for name, path in metadata["paths"].items():
        files["indexes/" + name] = _file(path)
    if len(files) > MAX_FILES or sum(path.stat().st_size for path in files.values()) > MAX_TOTAL_BYTES:
        raise ValueError("S1-V10 small delivery exceeded file/byte bounds")
    return files


def _verify_zip(path, expected, zip_sha):
    if sha(path) != zip_sha:
        raise ValueError("S1-V10 delivery ZIP SHA differs")
    with zipfile.ZipFile(path) as bundle:
        info = bundle.infolist()
        if len(info) != len(expected) or {row.filename for row in info} != set(expected):
            raise ValueError("S1-V10 delivery member list differs")
        for member in info:
            row = expected[member.filename]
            if member.is_dir() or member.file_size != row["bytes"]:
                raise ValueError("S1-V10 delivery member size differs")
            value = hashlib.sha256()
            with bundle.open(member) as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    value.update(block)
            if value.hexdigest() != row["sha256"]:
                raise ValueError("S1-V10 delivery member SHA differs")


def package(report_md=REPORT_MD, publication=PUBLICATION):
    gate = gate_state(report_md, publication)
    if not gate["ready"]:
        raise RuntimeError("S1-V10 formal delivery gates incomplete: " + "; ".join(gate["errors"]))
    manifest = read(MANIFEST)
    staging = DELIVERY / "final_staging"
    staging.mkdir(parents=True, exist_ok=True)
    archive_index = _archive_index(manifest)
    sources = _git_index(archive_index)
    metadata_paths = {
        "archive_index.json": save(staging / "archive_index.json", archive_index),
        "git_sources.json": save(staging / "git_sources.json", sources),
        "archive_manifest.json": MANIFEST,
    }
    files = _members(report_md, publication,
                     {"archive_index": archive_index, "paths": metadata_paths})
    if shutil.disk_usage(DELIVERY).free < (sum(path.stat().st_size for path in files.values()) +
                                          LOCAL_RESERVE_BYTES):
        raise RuntimeError("S1-V10 small final ZIP lacks D: input bytes plus 8 GiB reserve")
    rows = []
    for member, path in sorted(files.items()):
        if (member.startswith("/") or "\\" in member or
                any(part in ("", ".", "..") for part in member.split("/"))):
            raise ValueError("unsafe S1-V10 delivery member")
        rows.append({"member": member, "source_path": str(path),
                     "bytes": path.stat().st_size, "sha256": sha(path)})
    identity = hashlib.sha256(json.dumps(
        [(row["member"], row["bytes"], row["sha256"]) for row in rows],
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    package_id = identity[:20]
    target = DELIVERY / ("final_" + package_id)
    target.mkdir(parents=True, exist_ok=True)
    inventory = {"schema": "s1v10.final_delivery_inventory.v1",
                 "package_id": package_id,
                 "archive_manifest_sha256": gate["manifest_sha256"],
                 "formal_report_sha256": sha(report_md),
                 "publication_receipt_sha256": sha(Path(publication) / "publication_receipt.json"),
                 "members": rows}
    inventory_path = target / "inventory.json"
    if inventory_path.exists():
        if read(inventory_path) != inventory:
            raise RuntimeError("existing immutable S1-V10 delivery inventory differs")
    else:
        save(inventory_path, inventory)
    inventory_member = {"member": "indexes/delivery_inventory.json",
                        "source_path": str(inventory_path),
                        "bytes": inventory_path.stat().st_size,
                        "sha256": sha(inventory_path)}
    all_rows = [*rows, inventory_member]
    zip_path = target / ("s1v10_final_delivery_" + package_id + ".zip")
    if not zip_path.exists():
        pending = zip_path.with_suffix(".zip.part")
        with zipfile.ZipFile(pending, "w", allowZip64=True,
                             compression=zipfile.ZIP_DEFLATED, compresslevel=5) as bundle:
            for row in rows:
                path = Path(row["source_path"])
                if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                    raise RuntimeError("S1-V10 delivery source changed while packaging")
                mode = zipfile.ZIP_STORED if path.suffix.lower() in (".png", ".pdf", ".pt") else zipfile.ZIP_DEFLATED
                bundle.write(path, row["member"], compress_type=mode,
                             compresslevel=None if mode == zipfile.ZIP_STORED else 5)
            bundle.write(inventory_path, inventory_member["member"],
                         compress_type=zipfile.ZIP_DEFLATED, compresslevel=5)
        pending.replace(zip_path)
    zip_sha = sha(zip_path)
    _verify_zip(zip_path, {row["member"]: row for row in all_rows}, zip_sha)
    receipt = {"schema": "s1v10.final_delivery_local.v1", "package_id": package_id,
               "created_utc": now(), "zip_path": str(zip_path),
               "zip_sha256": zip_sha, "zip_bytes": zip_path.stat().st_size,
               "inventory_path": str(inventory_path),
               "inventory_sha256": sha(inventory_path),
               "member_count": len(all_rows), "all_member_shas_verified": True,
               "archive_acceptance_sha256": sha(ACCEPTANCE),
               "report_gate_sha256": sha(GATE),
               "publication_receipt_sha256": sha(Path(publication) / "publication_receipt.json"),
               "formal_report_sha256": sha(report_md),
               "raw_job_zips_embedded": False}
    save(target / "local_package_receipt.json", receipt)
    return receipt


def _remote_validate(port, remote_zip, expected_sha, expected_bytes, inventory):
    info = ops.instance(port)
    expected = {row["member"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
                for row in inventory["members"]}
    code = f'''
import hashlib,json,pathlib,zipfile
root=pathlib.Path({ops.BASE!r})/'delivery';path=pathlib.Path({remote_zip!r})
if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
 raise RuntimeError('S1-V10 remote delivery escaped its isolated root')
def digest(stream):
 h=hashlib.sha256()
 for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 return h.hexdigest()
with path.open('rb') as stream:actual=digest(stream)
if actual!={expected_sha!r} or path.stat().st_size!={expected_bytes}:
 raise RuntimeError('S1-V10 remote delivery ZIP SHA/bytes differ')
expected={expected!r}
with zipfile.ZipFile(path) as bundle:
 infos=bundle.infolist()
 if len(infos)!=len(expected) or set(x.filename for x in infos)!=set(expected):
  raise RuntimeError('S1-V10 remote delivery ZIP member set differs')
 for item in infos:
  row=expected[item.filename]
  if item.is_dir() or item.file_size!=row['bytes']:
   raise RuntimeError('S1-V10 remote delivery member size differs')
  with bundle.open(item) as stream:
   if digest(stream)!=row['sha256']:
    raise RuntimeError('S1-V10 remote delivery member SHA differs')
st=path.stat()
print(json.dumps({{'port':{port},'remote_path':str(path),'zip_sha256':actual,
 'zip_bytes':st.st_size,'member_count':len(infos),'all_member_shas_verified':True,
 'device':st.st_dev,'inode':st.st_ino}}))
'''
    result = ops.remote_python(code, port, timeout=600)
    if result["exit_code"]:
        raise RuntimeError(f"{port}: remote delivery validation failed: " + result["stderr"][-900:])
    return json.loads(result["stdout"].splitlines()[-1])


def upload(local_receipt):
    local = Path(local_receipt["zip_path"])
    package_id = local_receipt["package_id"]
    target = DELIVERY / ("final_" + package_id)
    inventory = read(local_receipt["inventory_path"])
    if (sha(local) != local_receipt["zip_sha256"] or
            sha(local_receipt["inventory_path"]) != local_receipt["inventory_sha256"] or
            inventory["package_id"] != package_id or
            len(inventory["members"]) + 1 != local_receipt["member_count"]):
        raise RuntimeError("S1-V10 local package/manifest changed before upload")
    inventory_member = {"member": "indexes/delivery_inventory.json",
                        "bytes": Path(local_receipt["inventory_path"]).stat().st_size,
                        "sha256": local_receipt["inventory_sha256"]}
    expected_members = [*inventory["members"], inventory_member]
    _verify_zip(local, {row["member"]: row for row in expected_members},
                local_receipt["zip_sha256"])
    remote_dir = f"{ops.BASE}/delivery/final_{package_id}"
    remote_zip = remote_dir + "/" + local.name
    code = f'''
import hashlib,json,pathlib
base=pathlib.Path({ops.BASE!r})/'delivery';root=pathlib.Path({remote_dir!r})
if not root.resolve().is_relative_to(base.resolve()):raise RuntimeError('delivery path escaped S1-V10')
root.mkdir(parents=True,exist_ok=True)
path=pathlib.Path({remote_zip!r});h=hashlib.sha256()
if path.is_symlink():raise RuntimeError('S1-V10 shared delivery ZIP target is linked')
if path.exists():
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
print(json.dumps({{'exists':path.exists(),'sha256':h.hexdigest() if path.exists() else None}}))
'''
    observed = ops.remote_python(code, 20418, timeout=120)
    if observed["exit_code"]:
        raise RuntimeError("S1-V10 remote delivery directory failed: " + observed["stderr"][-800:])
    current = json.loads(observed["stdout"].splitlines()[-1])
    if current["exists"] and current["sha256"] != local_receipt["zip_sha256"]:
        raise RuntimeError("immutable S1-V10 remote delivery path already contains another ZIP")
    if not current["exists"]:
        sent = ops.upload(local, remote_zip, 20418)
        if sent["local_sha256"] != local_receipt["zip_sha256"] or sent["remote"] != local_receipt["zip_sha256"]:
            raise RuntimeError("S1-V10 remote delivery upload SHA differs")
    remote_inventory = {**inventory, "members": expected_members}
    lane_checks = {str(port): _remote_validate(port, remote_zip,
                    local_receipt["zip_sha256"], local_receipt["zip_bytes"], remote_inventory)
                   for port in ops.SSH_PORTS}
    if len({row["remote_path"] for row in lane_checks.values()}) != 1:
        raise RuntimeError("four S1-V10 lanes do not reference one shared delivery path")
    remote_receipt = {"schema": "s1v10.final_delivery_remote.v1",
                      "package_id": package_id,
                      "shared_remote_path": remote_zip,
                      "zip_sha256": local_receipt["zip_sha256"],
                      "zip_bytes": local_receipt["zip_bytes"],
                      "member_count": local_receipt["member_count"],
                      "single_shared_physical_copy": True,
                      "same_device_inode_observed": len({(row["device"], row["inode"])
                                                         for row in lane_checks.values()}) == 1,
                      "four_lane_access": lane_checks,
                      "raw_job_zips_embedded": False}
    remote_receipt_path = target / "remote_delivery_receipt.json"
    if remote_receipt_path.is_file():
        if read(remote_receipt_path) != remote_receipt:
            raise RuntimeError("existing immutable S1-V10 remote receipt differs")
    else:
        save(remote_receipt_path, remote_receipt)
    remote_receipt_target = remote_dir + "/remote_delivery_receipt.json"
    code = f'''
import hashlib,json,pathlib
p=pathlib.Path({remote_receipt_target!r})
if p.is_symlink() or not p.resolve().is_relative_to((pathlib.Path({ops.BASE!r})/'delivery').resolve()):
 raise RuntimeError('S1-V10 remote delivery receipt escaped isolated root')
print(json.dumps({{'exists':p.exists(),
 'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}}))
'''
    current_receipt = ops.remote_python(code, 20418, timeout=60)
    if current_receipt["exit_code"]:
        raise RuntimeError("S1-V10 remote delivery receipt inspection failed")
    existing_receipt = json.loads(current_receipt["stdout"].splitlines()[-1])
    if existing_receipt["exists"] and existing_receipt["sha256"] != sha(remote_receipt_path):
        raise RuntimeError("immutable S1-V10 remote delivery receipt already differs")
    if not existing_receipt["exists"]:
        sent = ops.upload(remote_receipt_path, remote_receipt_target, 20418)
        if sent["remote"] != sha(remote_receipt_path):
            raise RuntimeError("S1-V10 remote receipt transfer SHA differs")
    for port in ops.SSH_PORTS:
        code = f'''
import hashlib,json,pathlib
p=pathlib.Path({remote_receipt_target!r});h=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps({{'path':str(p),'sha256':h,'bytes':p.stat().st_size}}))
'''
        row = ops.remote_python(code, port, timeout=60)
        if row["exit_code"] or json.loads(row["stdout"].splitlines()[-1])["sha256"] != sha(remote_receipt_path):
            raise RuntimeError(f"{port}: shared delivery receipt is not accessible")
    final = {"schema": "s1v10.final_delivery_complete.v1", "completed_utc": now(),
             "package_id": package_id, "local_zip": str(local),
             "local_zip_sha256": local_receipt["zip_sha256"],
             "remote_zip": remote_zip, "remote_zip_sha256": local_receipt["zip_sha256"],
             "member_count": local_receipt["member_count"],
             "local_receipt": str(target / "local_package_receipt.json"),
             "remote_receipt": remote_receipt_target,
             "remote_receipt_sha256": sha(remote_receipt_path),
             "all_four_lanes_readback_verified": True,
             "report_sha256": local_receipt["formal_report_sha256"],
             "archive_acceptance_sha256": local_receipt["archive_acceptance_sha256"],
             "manifest_sha256": inventory["archive_manifest_sha256"],
             "raw_job_zips_embedded": False}
    return save(target / "final_delivery_receipt.json", final), final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "deliver"))
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--publication-dir", type=Path, default=PUBLICATION)
    args = parser.parse_args()
    if args.action == "plan":
        plan = {"schema": "s1v10.final_delivery_plan.v1", "observed_utc": now(),
                "no_upload_or_packaging": True,
                "gates": gate_state(args.report_md, args.publication_dir),
                "report_md": str(args.report_md),
                "publication_dir": str(args.publication_dir),
                "remote_shared_root": ops.BASE + "/delivery",
                "four_lane_ports": list(ops.SSH_PORTS),
                "deliver_command": "python -m scripts.stage1_variable.final_delivery deliver",
                "file_count_limit": MAX_FILES, "per_file_bytes_limit": MAX_FILE_BYTES,
                "total_input_bytes_limit": MAX_TOTAL_BYTES,
                "local_reserve_bytes": LOCAL_RESERVE_BYTES,
                "required_gates": ["formal report Markdown", "strict summary formal_ready",
                                   "science audit 3600+720", "report controller complete",
                                   "archive acceptance 32/32 and manifest SHA",
                                   "independent root archive verification",
                                   "publication final receipt with 4320 frozen row comparison"],
                "local_receipts": ["delivery/final_<package_id>/inventory.json",
                                   "delivery/final_<package_id>/local_package_receipt.json",
                                   "delivery/final_<package_id>/remote_delivery_receipt.json",
                                   "delivery/final_<package_id>/final_delivery_receipt.json"],
                "excluded": "32 raw job ZIPs and one shared 6.49GB source ZIP remain referenced by SHA only"}
        save(DELIVERY / "final_delivery_plan.json", plan)
        save(DELIVERY / "final_delivery_receipt_template.json", {
            "schema": "s1v10.final_delivery_complete.v1",
            "status": "template_only_not_delivered",
            "package_id": "<deterministic SHA prefix of exact small members>",
            "local_zip": "D:/paper/outputs/experiments/first_stage_s1_v10_20261009/delivery/final_<package_id>/s1v10_final_delivery_<package_id>.zip",
            "local_zip_sha256": "<verified ZIP SHA256>",
            "remote_zip": ops.BASE + "/delivery/final_<package_id>/s1v10_final_delivery_<package_id>.zip",
            "remote_zip_sha256": "<four-lane verified same SHA256>",
            "member_count": "<small report/evidence members; excludes raw job ZIPs>",
            "local_receipt": "<local_package_receipt.json>",
            "remote_receipt": "<one shared remote_delivery_receipt.json>",
            "remote_receipt_sha256": "<SHA256>",
            "all_four_lanes_readback_verified": False,
            "report_sha256": "<formal report 08 SHA256>",
            "archive_acceptance_sha256": "<passed 32/32 acceptance SHA256>",
            "manifest_sha256": "<frozen 32-task manifest SHA256>",
            "raw_job_zips_embedded": False})
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    receipt = package(args.report_md, args.publication_dir)
    path, result = upload(receipt)
    print(json.dumps({"receipt": str(path), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
