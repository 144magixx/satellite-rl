"""Gate and deliver the small S1-D12A report package after formal publication.

The large scientific job and source ZIPs remain in their canonical archives.
Only their receipts and a source index enter this package. No training or
evaluation is started here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.stage1_drl_r04 import ops


REPO = Path("D:/paper")
BASE = REPO / "outputs/experiments/first_stage_s1_d12a_20261010"
REPORT = REPO / "docs/第一阶段/实验报告/09_R04引导的DRL覆盖A阶段对照.md"
SCIENTIFIC_COMMIT = "dfe7c394267e94bd2084ae72add3aa2c855b578f"
REPORT_FILES = ("audit.json", "summary.json", "task_audit.json",
                "per_scene.json", "per_scene.csv", "per_mother.json",
                "curve_metadata.json", "test_carried_distribution.png",
                "train_episode_reward_raw.png", "validation_best_j.png")
PUBLICATION_OUTPUTS = ("01_quality_100_mothers.png", "02_demand_levels.png",
                       "03_timing_100_mothers.png",
                       "04_paired_differences_100_mothers.png",
                       "overall_100_mothers.csv", "by_level_100_mothers.csv",
                       "timing_100_mothers.csv")
MAX_PACKAGE_BYTES = 256 << 20


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def save_new(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _file(files, name, path, errors):
    if path.is_file() and path.suffix.lower() != ".zip":
        files[name] = path
    else:
        errors.append(f"missing or prohibited package input: {name}")


def gate(repo=REPO, base=BASE):
    """Return exact small inputs only after all frozen formal gates pass."""
    repo, base = Path(repo), Path(base)
    errors, files = [], {}
    report = repo / REPORT.relative_to(REPO)
    final = base / "analysis/final_report_verified"
    audit_path = final / "audit.json"
    audit = read(audit_path) if audit_path.is_file() else {}
    if (audit.get("experiment_id") != "S1-D12A" or
            audit.get("scientific_commit") != SCIENTIFIC_COMMIT or
            audit.get("formal_ready") is not True or audit.get("issues") != [] or
            audit.get("test_rows") != 1800 or
            audit.get("strict_replayed_rows") != 1800 or
            audit.get("method_test_counts") != {name: 300 for name in
                                                 ("I0", "H1", "H2", "BC", "P0", "P1")}):
        errors.append("verified formal report gate is incomplete")
    _file(files, "report/" + report.name, report, errors)
    for name in REPORT_FILES:
        _file(files, "analysis/final_report_verified/" + name, final / name, errors)

    publication = base / "analysis/publication"
    metadata_path = publication / "figure_metadata.json"
    metadata = read(metadata_path) if metadata_path.is_file() else {}
    script = publication / "build_figures.py"
    source_inputs = {"per_scene.csv": final / "per_scene.csv",
                     "threshold_audit_mothers.csv": base / "analysis/threshold_audit_mothers.csv",
                     "threshold_audit.json": base / "analysis/threshold_audit.json"}
    declared_inputs = metadata.get("inputs", {})
    if (metadata.get("experiment") != "S1-D12A" or
            not script.is_file() or metadata.get("script_sha256") !=
            (file_sha(script) if script.is_file() else None) or
            len(declared_inputs) != len(source_inputs)):
        errors.append("publication metadata or figure script differs")
    for name, path in source_inputs.items():
        found = [(source, sha) for source, sha in declared_inputs.items()
                 if Path(source).name == name]
        if (len(found) != 1 or not path.is_file() or
                found[0][1] != file_sha(path)):
            errors.append(f"publication input SHA differs: {name}")
    if set(metadata.get("outputs", {})) != set(PUBLICATION_OUTPUTS):
        errors.append("four formal figures and three publication CSVs incomplete")
    for name in PUBLICATION_OUTPUTS:
        path = publication / name
        expected = metadata.get("outputs", {}).get(name, {})
        if (not path.is_file() or path.stat().st_size != expected.get("bytes") or
                file_sha(path) != expected.get("sha256")):
            errors.append(f"publication output SHA or bytes differ: {name}")
        _file(files, "analysis/publication/" + name, path, errors)
    _file(files, "analysis/publication/figure_metadata.json", metadata_path, errors)
    _file(files, "analysis/publication/build_figures.py", script, errors)

    learning_path = base / "preflight/final_learning_audit.json"
    learning = read(learning_path) if learning_path.is_file() else {}
    for port in (20418, 20424):
        row = learning.get(str(port), {})
        checks = row.get("checks", {})
        if (row.get("passed") is not True or not checks or
                not all(value is True for value in checks.values()) or
                row.get("ledger_counts", {}).get("actions.jsonl") != 262144 or
                row.get("ledger_counts", {}).get("budget_tails.jsonl") != 32 or
                not isinstance(row.get("checkpoint_sha256"), str)):
            errors.append(f"{port}: independent final learning audit incomplete")
    _file(files, "evidence/final_learning_audit.json", learning_path, errors)

    state_path = base / "analysis/final_input/state.json"
    state = read(state_path) if state_path.is_file() else {}
    manifest_path = base / "run_manifest.json"
    manifest = read(manifest_path) if manifest_path.is_file() else {}
    frozen_manifest_path = base / "analysis/final_input/run_manifest.json"
    frozen_manifest = (read(frozen_manifest_path)
                       if frozen_manifest_path.is_file() else {})
    plan_path = base / "run_plan.json"
    plan = read(plan_path) if plan_path.is_file() else {}
    preparation_path = base / "analysis/final_input/data_preparation.json"
    preparation = read(preparation_path) if preparation_path.is_file() else {}
    report_tasks = state.get("tasks", {})
    tasks = manifest.get("tasks", {})
    if (state.get("experiment_id") != "S1-D12A" or
            state.get("scientific_commit") != SCIENTIFIC_COMMIT or
            plan.get("experiment_id") != "S1-D12A" or
            plan.get("formal_source_commit") != SCIENTIFIC_COMMIT or
            not plan_path.is_file() or
            manifest.get("run_plan_sha256") != file_sha(plan_path) or
            frozen_manifest.get("run_plan_sha256") != file_sha(plan_path) or
            frozen_manifest.get("tasks") != tasks or
            preparation.get("experiment_id") != "S1-D12A" or
            preparation.get("scientific_commit") != SCIENTIFIC_COMMIT or
            set(preparation.get("tasks", {})) !=
            {"2024-02-07", "2024-02-08", "2024-02-09"} or
            len(report_tasks) != 34 or
            not set(report_tasks) <= set(tasks) or
            len(tasks) != 37 or len({(task.get("port"), task.get("run_id"))
                                    for task in tasks.values()}) != 37 or
            any(task.get("git_commit") != SCIENTIFIC_COMMIT
                for task in tasks.values()) or
            any(report_tasks.get(key, {}).get("run_id") != task.get("run_id")
                for key, task in tasks.items() if key in report_tasks)):
        errors.append("37 frozen task identities unavailable")
    _file(files, "indexes/frozen_state.json", state_path, errors)
    _file(files, "indexes/run_manifest.json", manifest_path, errors)
    _file(files, "indexes/frozen_run_manifest.json", frozen_manifest_path, errors)
    _file(files, "indexes/run_plan.json", plan_path, errors)
    _file(files, "indexes/frozen_data_preparation.json", preparation_path, errors)

    finalizer_path = base / "analysis/finalizer_state.json"
    finalizer = read(finalizer_path) if finalizer_path.is_file() else {}
    verified = finalizer.get("verified_tasks", [])
    if (len(verified) != 37 or
            {row.get("key") for row in verified} != set(tasks) or
            any(row.get("run_id") != tasks[row.get("key", "")].get("run_id") or
                row.get("proof", {}).get("closed") is not True
                for row in verified if row.get("key") in tasks)):
        errors.append("37-task finalizer archive closure evidence incomplete")
    _file(files, "evidence/finalizer_state.json", finalizer_path, errors)
    receipts = {}
    for key, task in sorted(tasks.items()):
        port, run_id = task.get("port"), task.get("run_id")
        if port not in ops.SSH_PORTS or not isinstance(run_id, str) or "/" in run_id:
            errors.append(f"{key}: invalid frozen task identity")
            continue
        path = base / "job_archives" / str(port) / run_id / "receipt.json"
        if not path.is_file():
            errors.append(f"{key}: archive receipt missing")
            continue
        receipt = read(path)
        if (receipt.get("port") != port or receipt.get("run_id") != run_id or
                receipt.get("source_commit") != SCIENTIFIC_COMMIT or
                not receipt.get("volumes") or
                any(not isinstance(v.get("sha256"), str) or
                    len(v["sha256"]) != 64 or v.get("bytes", 0) <= 0 or
                    not isinstance(v.get("members"), dict) or not v["members"]
                    for v in receipt["volumes"])):
            errors.append(f"{key}: archive receipt identity or volumes differ")
        files[f"archive_receipts/{key}.json"] = path
        receipts[key] = {"port": port, "run_id": run_id,
                         "receipt_sha256": file_sha(path),
                         "volumes": [{"file": v["file"], "bytes": v["bytes"],
                                      "sha256": v["sha256"]}
                                     for v in receipt["volumes"]]}
    if len(receipts) != 37:
        errors.append("37 archive receipts incomplete")
    source_receipt = base / "job_archives/20418/shared_sources/receipt.json"
    if source_receipt.is_file():
        source = read(source_receipt)
        if (source.get("experiment_id") != "S1-D12A" or
                source.get("source_commit") != SCIENTIFIC_COMMIT or
                len(source.get("volumes", [])) != 3 or
                any(not volume.get("members") for volume in source["volumes"])):
            errors.append("shared source receipt differs")
        files["archive_receipts/shared_sources.json"] = source_receipt
    else:
        errors.append("shared source receipt missing")

    exports_path = base / "preflight/model_exports.json"
    exports = read(exports_path) if exports_path.is_file() else {}
    if set(exports.get("exports", {})) != {"BC", "P0", "P1"}:
        errors.append("three final model exports missing")
    for key, row in exports.get("exports", {}).items():
        if key not in ("BC", "P0", "P1"):
            continue
        if (row.get("source") != SCIENTIFIC_COMMIT or
                row.get("env_step") != (0 if key == "BC" else 262144) or
                not any(t.get("run_id") == row.get("run_id") and
                        t.get("port") == row.get("port") for t in tasks.values())):
            errors.append(f"{key}: final model provenance differs")
        folder = Path(row.get("path", ""))
        expected_root = repo / "models/runs/first_stage_s1_d12a_20261010"
        if not folder.resolve().is_relative_to(expected_root.resolve()):
            errors.append(f"{key}: model export escaped its expected directory")
            continue
        for name in ("policy.pt", "model.json"):
            path = folder / name
            if not path.is_file() or file_sha(path) != row.get("files_sha256", {}).get(name):
                errors.append(f"{key}: final model {name} SHA differs")
            else:
                files[f"models/{key}/{name}"] = path
        receipt_path = (base / "job_archives" / str(row.get("port")) /
                        str(row.get("run_id")) / "receipt.json")
        if receipt_path.is_file():
            archived_shas = {member.get("sha256")
                             for volume in read(receipt_path).get("volumes", [])
                             for member in volume.get("members", {}).values()}
            if not set(row.get("files_sha256", {}).values()) <= archived_shas:
                errors.append(f"{key}: exported model SHA absent from job archive")
        else:
            errors.append(f"{key}: model job archive receipt missing")
        if key in ("P0", "P1"):
            check = learning.get(str(row.get("port")), {})
            if (check.get("run_id") != row.get("run_id") or
                    check.get("policy_sha256") != row.get("files_sha256", {}).get("policy.pt") or
                    check.get("model_sha256") != row.get("files_sha256", {}).get("model.json")):
                errors.append(f"{key}: independent model audit differs")
    _file(files, "evidence/model_exports.json", exports_path, errors)

    bundle = base / ("deployment/s1d12a_" + SCIENTIFIC_COMMIT + ".bundle")
    fixed = {"sources/frozen_science.bundle": bundle,
             "scripts/threshold_audit.py": base / "analysis/threshold_audit.py",
             "analysis/threshold_audit.json": base / "analysis/threshold_audit.json",
             "analysis/threshold_audit.csv": base / "analysis/threshold_audit.csv",
             "analysis/threshold_audit_mothers.csv": base / "analysis/threshold_audit_mothers.csv",
             "analysis/result_interpretation.md": base / "analysis/result_interpretation.md",
             "evidence/finalizer.py": base / "analysis/finalizer.py",
             "evidence/strict_repair_launch.json": base / "analysis/strict_repair_launch.json",
             "evidence/source_recovery_cache_audit.json": base / "preflight/source_recovery_cache_audit.json",
             "evidence/source_recovery_cache_audit_probe.py": base / "preflight/source_recovery_cache_audit_probe.py",
             "evidence/checkpoint_audit.json": base / "preflight/crash_recovery_20261010/checkpoint_audit.json",
             "evidence/pilot_reconciliation.json": base / "preflight/crash_recovery_20261010/pilot_reconciliation.json"}
    for name, path in fixed.items():
        _file(files, name, path, errors)
    for path in sorted((repo / "scripts/stage1_drl_r04").glob("*.py")):
        _file(files, "scripts/stage1_drl_r04/" + path.name, path, errors)
    for required in ("report.py", "strict_audit.py", "archive.py", "ops.py",
                     "delivery.py"):
        if "scripts/stage1_drl_r04/" + required not in files:
            errors.append(f"report dependency missing: {required}")
    _file(files, "scripts/s2p19/ops.py", repo / "scripts/s2p19/ops.py", errors)
    learning_files = ("learning_evidence_audit.py", "learning_evidence_addendum.json",
                      "learning_evidence_addendum.csv")
    existing_learning = [name for name in learning_files
                         if (base / "analysis" / name).exists()]
    if existing_learning and len(existing_learning) != len(learning_files):
        errors.append("learning evidence addendum is only partly written")
    for name in existing_learning:
        _file(files, "analysis/" + name, base / "analysis" / name, errors)
    for name, path in {
        "analysis/learning_evidence_addendum.md": base / "analysis/learning_evidence_addendum.md",
        "evidence/final_report_numeric_figure_qa.json": base / "preflight/final_report_numeric_figure_qa.json",
        "plans/12_R04引导的DRL波束覆盖优化方案.md": repo / "docs/第一阶段/方案/12_R04引导的DRL波束覆盖优化方案.md",
        "plans/13_DRL覆盖A阶段四实例执行方案.md": repo / "docs/第一阶段/方案/13_DRL覆盖A阶段四实例执行方案.md",
    }.items():
        if path.is_file():
            _file(files, name, path, errors)
    if "learning_evidence_addendum.json" in existing_learning:
        addendum = read(base / "analysis/learning_evidence_addendum.json")
        provenance = addendum.get("provenance", {})
        if not provenance:
            errors.append("learning evidence provenance missing")
        for label, item in provenance.items():
            path = (base / "job_archives" / str(item.get("port")) /
                    str(item.get("run_id")) / "receipt.json")
            if not path.is_file():
                errors.append(f"learning evidence archive missing: {label}")
                continue
            matches = [row for volume in read(path).get("volumes", [])
                       for name, row in volume.get("members", {}).items()
                       if name == item.get("member")]
            if (len(matches) != 1 or
                    matches[0].get("sha256") != item.get("sha256") or
                    matches[0].get("bytes") != item.get("bytes")):
                errors.append(f"learning evidence member SHA differs: {label}")
    source_index = {"schema": "s1d12a.source_index.v1",
                    "scientific_commit": SCIENTIFIC_COMMIT,
                    "science_bundle_sha256": file_sha(bundle) if bundle.is_file() else None,
                    "task_receipts": receipts,
                    "shared_source_receipt_sha256": (file_sha(source_receipt)
                                                     if source_receipt.is_file() else None)}
    return {"ready": not errors, "errors": errors, "files": files,
            "source_index": source_index,
            "formal_audit_sha256": file_sha(audit_path) if audit_path.is_file() else None,
            "report_sha256": file_sha(report) if report.is_file() else None}


def _members(gates):
    members = {name: path.read_bytes() for name, path in gates["files"].items()}
    members["indexes/source_index.json"] = canonical(gates["source_index"])
    members["REPRODUCE.md"] = (
        "S1-D12A small delivery package. The 37 large task ZIP volumes and three "
        "source ZIPs remain at canonical_paths in archive_receipts; only their "
        "SHA/size references are included here. Restore sources/frozen_science.bundle "
        "with git clone, install the project, and overlay the packaged scripts/ tree "
        "for the later report-only audit fixes (science workers remain frozen). Then run "
        "python -m scripts.stage1_drl_r04.report --state indexes/frozen_state.json "
        "--jobs <canonical job_archives directory> --output <new directory> "
        "--terminal-audit evidence/final_learning_audit.json. "
        "The analysis/publication/build_figures.py script expects analysis/final_report_verified "
        "and the threshold files at their packaged relative paths. "
        "Consult indexes/source_index.json before resolving external source ZIPs. "
        "The report Markdown keeps the original D:/paper repository links; use the "
        "packaged analysis/, models/ and archive_receipts/ folders when viewing this "
        "small package independently, or restore files to their repository locations.\n"
    ).encode("utf-8")
    if (sum(len(data) for data in members.values()) > MAX_PACKAGE_BYTES or
            any(name.endswith(".zip") for name in members)):
        raise ValueError("small delivery package exceeds bounds or includes raw ZIP")
    return members


def _inventory(members, gates):
    rows = [{"name": name, "bytes": len(data), "sha256": digest(data)}
            for name, data in sorted(members.items())]
    identity = digest(canonical(rows))[:20]
    return {"schema": "s1d12a.delivery_inventory.v1", "package_id": identity,
            "scientific_commit": SCIENTIFIC_COMMIT,
            "formal_audit_sha256": gates["formal_audit_sha256"],
            "report_sha256": gates["report_sha256"], "members": rows,
            "raw_job_zips_embedded": False}


def verify_zip(path, receipt):
    if not path.is_file() or path.stat().st_size != receipt["zip_bytes"] or file_sha(path) != receipt["zip_sha256"]:
        raise ValueError("local delivery ZIP SHA or bytes differ")
    expected = {row["name"]: row for row in receipt["members"]}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != set(expected):
            raise ValueError("delivery ZIP member inventory differs")
        for name in names:
            data = archive.read(name)
            if len(data) != expected[name]["bytes"] or digest(data) != expected[name]["sha256"]:
                raise ValueError("delivery ZIP member SHA or bytes differ")


def package(repo=REPO, base=BASE):
    gates = gate(repo, base)
    if not gates["ready"]:
        raise RuntimeError("formal delivery gates incomplete: " + "; ".join(gates["errors"]))
    members = _members(gates)
    inventory = _inventory(members, gates)
    package_id = inventory["package_id"]
    target = Path(base) / "delivery" / ("final_" + package_id)
    if target.exists():
        raise FileExistsError("existing package must use resume-upload with its local receipt")
    target.mkdir(parents=True)
    members["indexes/delivery_inventory.json"] = canonical(inventory)
    zip_path = target / ("s1d12a_final_" + package_id + ".zip")
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    rows = [{"name": name, "bytes": len(data), "sha256": digest(data)}
            for name, data in sorted(members.items())]
    receipt = {"schema": "s1d12a.delivery_local.v1", "package_id": package_id,
               "zip_path": str(zip_path.resolve()), "zip_sha256": file_sha(zip_path),
               "zip_bytes": zip_path.stat().st_size, "members": rows,
               "formal_audit_sha256": gates["formal_audit_sha256"],
               "report_sha256": gates["report_sha256"],
               "raw_job_zips_embedded": False}
    verify_zip(zip_path, receipt)
    save_new(target / "local_package_receipt.json", receipt)
    return receipt


def _remote_check(port, remote_path, receipt):
    expected = {row["name"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
                for row in receipt["members"]}
    code = f'''
import hashlib,json,pathlib,zipfile
base=pathlib.Path({(ops.BASE + '/delivery')!r});p=pathlib.Path({remote_path!r})
if p.is_symlink() or not p.resolve().is_relative_to(base.resolve()) or not p.is_file():
 raise RuntimeError('D12A remote package missing, linked or outside delivery root')
def sha(stream):
 h=hashlib.sha256()
 for block in iter(lambda:stream.read(8<<20),b''): h.update(block)
 return h.hexdigest()
with p.open('rb') as stream: got=sha(stream)
if got!={receipt['zip_sha256']!r} or p.stat().st_size!={receipt['zip_bytes']}:
 raise RuntimeError('D12A remote ZIP SHA/bytes differ')
expected={expected!r}
with zipfile.ZipFile(p) as z:
 names=z.namelist()
 if len(names)!=len(expected) or set(names)!=set(expected):
  raise RuntimeError('D12A remote member inventory differs')
 for name in names:
  row=expected[name]
  if z.getinfo(name).file_size!=row['bytes']:
   raise RuntimeError('D12A remote member bytes differ')
  with z.open(name) as stream:
   if sha(stream)!=row['sha256']:
    raise RuntimeError('D12A remote member SHA differs')
st=p.stat()
print(json.dumps({{'port':{port},'path':str(p),'zip_sha256':got,
 'zip_bytes':st.st_size,'member_count':len(names),'all_member_shas_verified':True,
 'device':st.st_dev,'inode':st.st_ino}}))
'''
    result = ops.remote_python(code, port, timeout=600)
    if result["exit_code"]:
        raise RuntimeError(f"{port}: remote package readback failed: " + result["stderr"][-600:])
    return json.loads(result["stdout"].splitlines()[-1])


def upload(local_receipt_path, repo=REPO, base=BASE):
    path = Path(local_receipt_path).resolve()
    delivery = (Path(base) / "delivery").resolve()
    receipt = read(path)
    if (path.name != "local_package_receipt.json" or
            path.parent.parent != delivery or
            path.parent.name != "final_" + str(receipt.get("package_id")) or
            receipt.get("schema") != "s1d12a.delivery_local.v1" or
            receipt.get("raw_job_zips_embedded") is not False):
        raise ValueError("local receipt is outside its immutable package")
    local = Path(receipt["zip_path"]).resolve()
    if (local.parent != path.parent or
            local.name != "s1d12a_final_" + receipt["package_id"] + ".zip"):
        raise ValueError("local ZIP path differs from receipt directory")
    verify_zip(local, receipt)
    gates = gate(repo, base)
    if not gates["ready"]:
        raise RuntimeError("formal delivery gates no longer pass")
    current = _inventory(_members(gates), gates)
    inventory_data = canonical(current)
    expected_members = sorted([*current["members"], {
        "name": "indexes/delivery_inventory.json", "bytes": len(inventory_data),
        "sha256": digest(inventory_data)}], key=lambda row: row["name"])
    if (current["package_id"] != receipt["package_id"] or
            receipt["members"] != expected_members or
            current["formal_audit_sha256"] != receipt["formal_audit_sha256"] or
            current["report_sha256"] != receipt["report_sha256"]):
        raise ValueError("formal package inputs changed after local receipt")
    remote_dir = ops.BASE + "/delivery/final_" + receipt["package_id"]
    remote_path = remote_dir + "/" + local.name
    code = f'''
import hashlib,json,pathlib
base=pathlib.Path({(ops.BASE + '/delivery')!r});folder=pathlib.Path({remote_dir!r})
if not folder.resolve().is_relative_to(base.resolve()): raise RuntimeError('delivery escaped root')
folder.mkdir(parents=True,exist_ok=True)
p=pathlib.Path({remote_path!r})
if p.is_symlink(): raise RuntimeError('remote delivery path is linked')
h=hashlib.sha256()
if p.exists():
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''): h.update(block)
print(json.dumps({{'exists':p.exists(),'sha256':h.hexdigest() if p.exists() else None}}))
'''
    result = ops.remote_python(code, 20418, timeout=120)
    if result["exit_code"]:
        raise RuntimeError("remote delivery destination check failed")
    observed = json.loads(result["stdout"].splitlines()[-1])
    if observed["exists"] and observed["sha256"] != receipt["zip_sha256"]:
        raise ValueError("remote immutable package path already has different bytes")
    if not observed["exists"]:
        sent = ops.upload(local, remote_path, 20418)
        if (sent.get("exit_code") != 0 or
                sent.get("local_sha256") != receipt["zip_sha256"] or
                sent.get("remote") != receipt["zip_sha256"]):
            raise ValueError("remote upload SHA differs")
    lanes = {str(port): _remote_check(port, remote_path, receipt)
             for port in ops.SSH_PORTS}
    if len({(row["device"], row["inode"]) for row in lanes.values()}) != 1:
        raise ValueError("four lanes did not read one shared physical ZIP")
    remote_receipt = {"schema": "s1d12a.delivery_remote.v1",
                      "package_id": receipt["package_id"],
                      "shared_remote_path": remote_path,
                      "zip_sha256": receipt["zip_sha256"],
                      "zip_bytes": receipt["zip_bytes"],
                      "member_count": len(receipt["members"]),
                      "four_lane_access": lanes,
                      "single_shared_physical_copy": True,
                      "raw_job_zips_embedded": False}
    remote_receipt_path = path.parent / "remote_delivery_receipt.json"
    if remote_receipt_path.exists():
        if read(remote_receipt_path) != remote_receipt:
            raise ValueError("existing immutable remote delivery receipt differs")
    else:
        save_new(remote_receipt_path, remote_receipt)
    return remote_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "package", "deliver", "resume-upload"))
    parser.add_argument("--local-receipt", type=Path)
    args = parser.parse_args()
    if args.action == "plan":
        result = gate()
        print(json.dumps({"ready": result["ready"], "errors": result["errors"],
                          "member_count_if_ready": len(result["files"]) + 2 if result["ready"] else None,
                          "would_upload": False}, ensure_ascii=False, indent=2))
        return
    if args.action == "resume-upload":
        if not args.local_receipt:
            parser.error("resume-upload requires --local-receipt")
        result = upload(args.local_receipt)
    else:
        receipt = package()
        result = receipt if args.action == "package" else upload(
            Path(receipt["zip_path"]).parent / "local_package_receipt.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
