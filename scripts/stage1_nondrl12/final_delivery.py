"""Gate, package and resume one small ND12 delivery; never launch scientific work.

plan is read-only and honestly not-ready before publication. deliver uploads one
shared immutable ZIP and verifies all bytes/members through three physical hosts.
resume-upload uses the existing package bytes, never regenerating a package.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from . import ops, placement_transport

REPO = Path(__file__).resolve().parents[2]
RESERVE = 8 << 30
MAX_FILE = 16 << 20
MAX_MODEL = 64 << 20
MAX_TOTAL = 256 << 20
MAX_FILES = 1500
TERMINAL = {"completed", "failed", "stopped", "interrupted", "cancelled"}
SMALL_SUFFIXES = {".md", ".json", ".csv", ".png", ".svg", ".pdf", ".py", ".txt", ".pt"}
SCHEMA = "s1_nd12_final_delivery_v1"


def indexed_members(receipt,directory):
    """Small member-level entry points; cache and raw curves stay in their one ZIP."""
    from . import archive
    prefix=f"{receipt.get('port',20418)}/{receipt['run_id']}/"
    groups={"raw_curves":[],"runtime_members":[],"scientific_sources":[]}
    for volume in receipt["volumes"]:
        remote=volume.get("remote_path",ops.BASE+"/"+str(receipt.get("port",20418))+
                          "/ops/archive_nd12/"+receipt["run_id"]+"/"+volume["file"])
        for name,row in volume["members"].items():
            relative=name[len(prefix):] if name.startswith(prefix) else name
            group=("raw_curves" if relative in ("models/losses.jsonl","models/validation.json") else
                   "runtime_members" if relative.startswith("runtime_extensions/") else
                   "scientific_sources" if name.endswith(".bundle") or relative in (
                       "source/source_manifest.json","source_lineage/contract.json") else None)
            if group:
                groups[group].append({"member":name,"sha256":row["sha256"],"bytes":row["bytes"],
                    "zip_sha256":volume["sha256"],"zip_bytes":volume["bytes"],
                    "canonical_zip":str(archive.volume_path(directory,volume).resolve()),"remote_zip":remote,
                    "physical_port":20473 if receipt.get("port")==20474 else receipt.get("port",20418)})
    return groups


def sha(path):
    with Path(path).open("rb") as source:
        return stream_sha(source)


def stream_sha(source):
    h = hashlib.sha256()
    for block in iter(lambda: source.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".partial")
    pending.write_bytes(encoded(value)); pending.replace(path)


def safe_member(name):
    if (not isinstance(name, str) or name.startswith("/") or "\\" in name or
            any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("unsafe delivery member")
    return name


def pin(path, expected=None, *, small=False):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing/linked delivery input: " + str(path))
    if expected and sha(path) != expected:
        raise ValueError("delivery input SHA differs: " + str(path))
    if small and (path.suffix.lower() not in SMALL_SUFFIXES or
                  path.stat().st_size > (MAX_MODEL if path.suffix == ".pt" else MAX_FILE)):
        raise ValueError("raw/oversized delivery member refused: " + str(path))
    return path


def placement(root, selected=None):
    transport = placement_transport.Transport.from_file(selected or Path(root) / "controller/execution_placement.json")
    if (set(transport.mapping.values()) != {20418, 20424, 20473} or transport.mapping[20474] != 20473):
        raise ValueError("delivery may contact only physical20418/20424/20473")
    return transport


def _archives(root, planned):
    from . import archive, strict_audit
    cache = set()
    source_path = root / "job_archives/20418/shared_sources/receipt.json"
    source = read(source_path)
    archive.verify_source_receipt(source, source_path.parent, cache)
    receipts, paths = [], []
    for entry in planned["tasks"].values():
        path = root / "job_archives" / str(entry["port"]) / entry["run_id"] / "receipt.json"
        receipt = read(path)
        if (receipt.get("port") != entry["port"] or receipt.get("run_id") != entry["run_id"] or
                receipt.get("source_commit") != entry["source_commit"]):
            raise ValueError("planned canonical receipt identity differs")
        evidence = strict_audit.Evidence(path, cache)
        job, status = evidence.json("job.json"), evidence.json("status.json")
        if (job.get("run_id") != entry["run_id"] or job.get("git_commit") != entry["source_commit"] or
                entry.get("kind") is not None and job.get("kind") != entry["kind"] or
                status.get("stage") not in TERMINAL or status.get("stage") != receipt.get("status")):
            raise ValueError("planned job is not actually terminal in canonical evidence")
        receipts.append(receipt); paths.append(path)
    for receipt, path in zip(receipts, paths):
        closed = archive.verify_receipt(receipt, path.parent, source, source_path.parent,
                                        receipts, verified_volume_cache=cache)
        if closed.get("closed") is not True:
            raise ValueError("canonical task dependency closure incomplete")
    refs = []
    for receipt, path in [(source, source_path), *zip(receipts, paths)]:
        refs.append({"run_id": receipt["run_id"], "port": receipt.get("port", 20418),
                     "receipt_path": str(path.resolve()), "receipt_sha256": sha(path),
                     "source_commit": receipt["source_commit"], "status": receipt.get("status"),
                     "origin_commit":receipt.get("origin_commit",receipt["source_commit"]),
                     "runtime_extension":receipt.get("runtime_extension"),
                     **indexed_members(receipt,path.parent),
                     "source_refs": receipt.get("source_refs", []),
                     "volumes": [{"canonical_path": str(archive.volume_path(path.parent, volume).resolve()),
                        "remote_path": volume.get("remote_path", ops.BASE + "/" + str(receipt.get("port",20418)) +
                            "/ops/archive_nd12/" + receipt["run_id"] + "/" + volume["file"]),
                        "sha256": volume["sha256"], "bytes": volume["bytes"],
                        "member_count": len(volume["members"]),
                        "member_manifest_sha256": hashlib.sha256(encoded(volume["members"])).hexdigest(),
                        "source_bundle_members": [{"member":name,**row} for name,row in volume["members"].items()
                                                  if name.endswith(".bundle")]}
                        for volume in receipt["volumes"]]})
    return refs, receipts


def validate_delivery_refs(report,refs):
    """Fail closed if a strict mixed-source result cannot be reached by exact members."""
    by_run={r["run_id"]:r for r in refs}
    if len(by_run)!=len(refs):raise ValueError("duplicate delivery source owner")
    for run_id,task in report.get("tasks",{}).items():
        if task.get("curve_losses"):
            owner=by_run.get(run_id,{})
            names={r["member"].split("/",2)[-1] for r in owner.get("raw_curves",[])}
            if names!={"models/losses.jsonl","models/validation.json"}:
                raise ValueError("training raw JSONL/validation member references are missing")
    for run_id,extension in report.get("runtime_extensions",{}).items():
        owner=by_run.get(run_id)
        if owner is None or owner.get("runtime_extension")!=extension.get("marker"):
            raise ValueError("strict runtime marker differs from canonical delivery owner")
        scientific=[r for r in owner.get("scientific_sources",[]) if r["member"].endswith("/source/source_manifest.json")]
        if not scientific or owner["source_commit"]!=extension["marker"]["base_execution_commit"]:
            raise ValueError("composite result lacks an exact scientific source manifest reference")
        if extension.get("execution_role")=="model_only_consumer":
            if owner.get("runtime_members"):
                raise ValueError("model consumer must not copy runtime/cache evidence")
            producer=report["runtime_extensions"].get(extension["model_source_run_id"],{})
            if producer.get("marker")!=extension["marker"]:
                raise ValueError("a21 model consumer does not close to the composite producer")
        else:
            if not any(r["sha256"]==extension["marker"]["base_manifest_sha256"] for r in scientific):
                raise ValueError("producer base source manifest SHA differs from its runtime marker")
            required={"contract.json","provider.py","bootstrap.py","cache/manifest.json","cache/index.sqlite","cache/data.bin"}
            names={r["member"].split("/runtime_extensions/",1)[-1].split("/",1)[-1]
                   for r in owner.get("runtime_members",[])}
            if not required.issubset(names):raise ValueError("single canonical runtime/cache references are incomplete")
            for name,key in (("provider.py","provider_sha256"),("bootstrap.py","bridge_sha256"),
                             ("cache/manifest.json","cache_manifest_sha256"),("contract.json","contract_sha256")):
                row=next(r for r in owner["runtime_members"] if r["member"].endswith("/"+name))
                if row["sha256"]!=extension["marker"][key]:raise ValueError("runtime code/data member SHA differs")
    for model in report.get("endpoints",[]):
        extension=model.get("runtime_extension")
        if extension and report.get("runtime_extensions",{}).get(model["run_id"],{}).get("marker")!=extension["marker"]:
            raise ValueError("small model runtime export differs from strict producer")


def gate_state(root=ops.LOCAL, report_path=None, placement_file=None):
    root = Path(root).resolve()
    errors, pins = [], {}
    try:
        transport = placement(root, placement_file)
        pins["placement"] = {"path": str(transport.path), "sha256": transport.mapping_sha256}
        status_path = root / "controller/finalizer/status.json"
        status = read(status_path)
        if status.get("phase") != "complete":
            raise ValueError("finalizer has not genuinely completed")
        paths = {"completion": root / "analysis/final_report/completion.json",
                 "strict_report": root / "analysis/final_report/report.json",
                 "snapshot": root / "analysis/final_input/snapshot.json",
                 "archive_plan": root / "analysis/final_input/archive_plan.json",
                 "path_map": root / "analysis/final_input/path_map.json",
                 "closure": root / "analysis/final_input/archive_closure.json", "finalizer": status_path}
        for key, path in paths.items():
            pin(path); pins[key] = {"path": str(path), "sha256": sha(path)}
        complete, report = read(paths["completion"]), read(paths["strict_report"])
        if (complete.get("formal_ready") is not True or report.get("formal_ready") is not True or
                report.get("issues") != [] or complete.get("report_sha256") != sha(paths["strict_report"]) or
                complete.get("markdown_sha256") != sha(root / "analysis/final_report/report.md")):
            raise ValueError("strict formal-ready/issues/report SHA does not close")
        if set(report.get("routes", {})) != {f"{i:02d}" for i in range(1,13)}:
            raise ValueError("strict report must classify all twelve routes")
        # The strict report owns scientific classification. Inconclusive teacher
        # budgets are valid terminal evidence, never asserted trained/evaluated.
        if any(v.get("status") not in {"passed", "science_stopped", "budget_limited_inconclusive", "completed"}
               for v in report["routes"].values()):
            raise ValueError("strict report still contains a pending/attention route")
        snapshot = read(paths["snapshot"])
        for key, item in snapshot["files"].items():
            pin(item["path"], item["sha256"])
            pins["snapshot/" + key] = {"path": item["path"], "sha256": item["sha256"]}
        planned, closure = read(paths["archive_plan"]), read(paths["closure"])
        from . import archive
        expected_plan=archive.plan(read(snapshot["files"]["state"]["path"]),
                                   read(snapshot["files"]["preparations"]["path"]))
        if (planned.get("experiment_id") != "S1-ND12" or closure.get("complete") is not True or
                planned!=expected_plan or report.get("experiment_id")!="S1-ND12" or
                report.get("scientific_commit")!=snapshot.get("scientific_commit") or
                complete.get("scientific_commit")!=snapshot.get("scientific_commit") or
                len(closure.get("tasks", [])) != len(planned["tasks"]) or
                any(t.get("closed") is not True for t in closure.get("tasks", [])) or
                {t["run_id"] for t in closure.get("tasks", [])}!={t["run_id"] for t in planned["tasks"].values()} or
                complete.get("tasks") != len(planned["tasks"]) or
                status.get("report_sha256") != complete["report_sha256"]):
            raise ValueError("finalizer/archive task closure identity differs")
        if report_path is None:
            matches = list(Path("D:/paper/docs/第一阶段/实验报告").glob("10_*.md"))
            if len(matches) != 1:
                raise ValueError("real Chinese report10 has not been uniquely published")
            report_path = matches[0]
        report_path = pin(report_path, small=True)
        text = report_path.read_text(encoding="utf-8-sig")
        if (not report_path.name.startswith("10_") or len(text) < 200 or
                sum("\u4e00" <= c <= "\u9fff" for c in text) < 40 or
                "非DRL" not in text and "非 DRL" not in text):
            raise ValueError("published report10 is absent, a placeholder or unrelated")
        pins["report10"] = {"path": str(report_path.resolve()), "sha256": sha(report_path)}
        source_refs, receipts = _archives(root, planned)
        validate_delivery_refs(report,source_refs)
        for ref in source_refs:
            receipt_file=pin(ref["receipt_path"],ref["receipt_sha256"],small=True)
            pins[f"canonical_receipt/{ref['port']}/{ref['run_id']}"]={"path":str(receipt_file),"sha256":ref["receipt_sha256"]}
        endpoint_ids = {r["run_id"] for r in receipts if r.get("status") == "completed" and
                        any(t["run_id"] == r["run_id"] and t.get("kind") == "train" for t in planned["tasks"].values())}
        endpoints = report.get("endpoints", [])
        if {r["run_id"] for r in endpoints} != endpoint_ids or len(endpoints) != complete.get("endpoint_count"):
            raise ValueError("completed training endpoint export receipts do not close")
        for item in endpoints:
            model = pin(item["path"], item["sha256"], small=True)
            metadata = pin(model.parent / "model.json", item["model_json_sha256"], small=True)
            record = read(metadata)
            if record.get("selection") != "fixed_budget_endpoint" or record.get("policy_sha256") != item["sha256"]:
                raise ValueError("endpoint is not the canonical fixed-budget export")
            canonical=next(r for r in receipts if r["run_id"]==item["run_id"])
            prefix=str(canonical["port"])+"/"+item["run_id"]+"/models/final/"
            members={name:row for v in canonical["volumes"] for name,row in v["members"].items()}
            if (members.get(prefix+"policy.pt",{}).get("sha256")!=item["sha256"] or
                    members.get(prefix+"model.json",{}).get("sha256")!=item["model_json_sha256"]):
                raise ValueError("small endpoint export differs from canonical model members")
        for figure in report.get("figures", []):
            pin(figure["file"], figure["sha256"], small=True)
        if len(report.get("figures", [])) != complete.get("figure_count"):
            raise ValueError("figure export receipt count differs")
        return {"ready": True, "errors": [], "pins": pins, "report": report,
                "source_refs": source_refs, "placement": transport.placement}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as error:
        errors.append(str(error))
    return {"ready": False, "errors": errors, "pins": pins}


def _inputs(root, gate, assets_manifest=None):
    files, generated = {}, {}
    def add(member, path, expected=None):
        safe_member(member)
        if member in files or member in generated:
            raise ValueError("duplicate delivery member")
        files[member] = pin(path, expected, small=True)
    for key, item in gate["pins"].items():
        path = Path(item["path"])
        # Large strict row arrays and historical teacher proof remain in the
        # pinned source index. No raw evidence tree is copied into a small ZIP.
        if key != "strict_report" and path.stat().st_size <= MAX_FILE:
            add("report/report10.md" if key == "report10" else "receipts/" + key.replace("/","_") + ".json",
                path, item["sha256"])
    report = gate["report"]
    validate_delivery_refs(report,gate["source_refs"])
    summary = {"experiment_id": "S1-ND12", "formal_ready": True, "issues": [],
               "scientific_commit": report["scientific_commit"], "routes": {
                   route: {"status": record["status"], "methods": {
                       method: {"summary": data.get("summary"), "run_ids": data.get("run_ids", [])}
                       for method,data in record.get("methods",{}).items()}}
                   for route,record in report["routes"].items()}}
    summary["runtime_extensions"]={run_id:{key:value for key,value in runtime.items() if key in (
        "marker","activation_update","preserved_loss_rows","original_validation_nodes","base_contract_preserved",
        "original_active_seconds","complete_active_seconds","cache_canonical_owner","cache_manifest_sha256",
        "execution_role","model_source_run_id","provider_execution_authorized","equivalence_proofs")}
        for run_id,runtime in report.get("runtime_extensions",{}).items()}
    summary["training_datafeed_segments"]={run_id:task["training_datafeed_segments"] for run_id,task in report.get("tasks",{}).items()
                                           if task.get("training_datafeed_segments")}
    summary["frozen_source_references"]=report.get("frozen_source_references",{})
    generated["summary/summary.json"] = encoded(summary)
    csv_out=io.StringIO(newline="");writer=csv.writer(csv_out)
    writer.writerow(("route","status","method","metric","value_json"))
    for route,record in summary["routes"].items():
        writer.writerow((route,record["status"],"","route_status",json.dumps(record["status"])))
        for method,data in record["methods"].items():
            for metric,value in (data.get("summary") or {}).items():
                writer.writerow((route,record["status"],method,metric,json.dumps(value,ensure_ascii=False)))
    generated["summary/methods.csv"]=csv_out.getvalue().encode("utf-8")
    generated["indexes/source_refs.json"] = encoded({"canonical_raw_source_copies":1,
        "archives":gate["source_refs"], "input_pins":gate["pins"],
        "raw_job_zips_embedded":False, "placement":gate["placement"]})
    for figure in report.get("figures", []):
        add("figures/" + Path(figure["file"]).name, figure["file"], figure["sha256"])
    for model in report.get("endpoints", []):
        for name, expected in (("policy.pt",model["sha256"]), ("model.json",model["model_json_sha256"])):
            add("models/"+model["run_id"]+"/"+name, Path(model["path"]).parent/name, expected)
    code_index=[]
    scripts=("__init__.py","final_delivery.py","report.py","strict_audit.py","execution_sources.py",
             "placement_transport.py","datafeed_evidence.py","archive.py","ops.py","finalizer.py")
    paths=[Path("scripts/__init__.py")]+[Path("scripts/stage1_nondrl12")/name for name in scripts]
    # Metadata/audit overlay absent from the old a21 science snapshot. Scientific
    # core/models are restored from the exact archived source manifest/bundle.
    paths += [Path("src/satellite_rl/workbench")/name for name in (
        "nondrl_source_repair.py","nondrl_datafeed.py","nondrl_datafeed_bootstrap.py")]
    paths += [Path("scripts")/group/"ops.py" for group in ("s2p19",)]
    paths += [Path("scripts/variable_beams/ops18.py"),Path("scripts/s2p19/__init__.py"),Path("scripts/variable_beams/__init__.py")]
    for path in paths:
        member="reproduce/"+path.as_posix()
        if path.name=="__init__.py" and not (REPO/path).exists():
            generated[member]=b"# Delivery package initializer for the original namespace package.\n"
            record={"member":member,"sha256":hashlib.sha256(generated[member]).hexdigest(),"generated":True}
        else:
            add(member,REPO/path)
            record={"member":member,"sha256":sha(REPO/path),"source_path":str((REPO/path).resolve())}
        code_index.append({**record,"role":"postprocessing_and_metadata_overlay"})
    generated["indexes/code_modules.json"]=encoded({"modules":code_index,"scientific_code_from_exact_archive_source":True})
    generated["reproduce/README.md"]=(
        "# S1-ND12 复现与原始曲线回读\n\n"
        "本包保存报告、小模型及逐模块SHA。不能把当前main提交当作全部科学来源。\n\n"
        "1. 按indexes/source_refs.json的scientific_sources回读精确ZIP/bundle及source/source_manifest.json；"
        "逐文件核SHA后恢复对应105或a21科学源码；本小包不是独立完整科学实现，仍须原冻结源码、共享依赖与归档输入。"
        "原始大卷只读复用，不复制Jan/Feb源或训练缓存。\n"
        "2. 将reproduce/scripts与reproduce/src的审计/元数据overlay合入隔离复现目录，"
        "按indexes/code_modules.json核每个模块SHA。需要DQN的torch2.3.1+cu121、NumPy、Matplotlib及SSH时的paramiko。\n"
        "3. 在该目录设置PYTHONPATH=src，运行python -m scripts.stage1_nondrl12.report --help；"
        "用冻结输入和receipts/path_map.json的path-map离线复算；按indexes/source_refs.json更新搬迁后的本地canonical卷入口。"
        "下载回读不启动科学任务，勿运行正式runner或finalizer执行入口。\n"
        "4. 原始loss JSONL和validation JSON在每个train的raw_curves索引内：按remote_zip/canonical_zip、"
        "zip_sha256、member与成员SHA逐项核验；receipts/canonical_receipt_*保存完整成员表与每卷SHA，"
        "可离线重验全部canonical卷。图仅是原始记录的派生展示，不替代原曲线。\n"
        "5. 组合训练的base a21与provider/bridge/contract各自SHA见runtime_extensions；"
        "training_datafeed_segments分开旧gzip和compact供给活动时间。消费者评价执行源码仍a21，"
        "权重保留producer runtime marker，不获provider执行资格。cache仅在train canonical ZIP的runtime_members中一份。\n"
    ).encode("utf-8")
    if assets_manifest is not None:
        specification = read(pin(assets_manifest, small=True))
        if specification.get("experiment_id") != "S1-ND12":
            raise ValueError("wrong small delivery assets manifest")
        for item in specification["files"]:
            if "teacher_reach" in item["member"] or Path(item["path"]).suffix in {".zip", ".tar", ".npz", ".gz", ".jsonl"}:
                raise ValueError("raw/teacher evidence must be a source_ref")
            add(item["member"], item["path"], item["sha256"])
        add("indexes/assets_manifest.json", assets_manifest)
    records = [{"member": name, "source_path": str(path.resolve()), "bytes": path.stat().st_size, "sha256":sha(path)}
               for name,path in sorted(files.items())]
    records += [{"member": name, "generated": True, "bytes":len(raw), "sha256":hashlib.sha256(raw).hexdigest()}
                for name,raw in sorted(generated.items())]
    if len(records)>MAX_FILES or sum(r["bytes"] for r in records)>MAX_TOTAL or any(
            r["bytes"]>MAX_FILE for r in records if r.get("generated")):
        raise ValueError("small delivery inventory exceeds byte/file bound")
    manifest = {"schema":SCHEMA, "experiment_id":"S1-ND12", "input_pins":gate["pins"],
                "placement":gate["placement"], "members":sorted(records,key=lambda r:r["member"])}
    return manifest, files, generated


def free_bytes(path):
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free


def choose_volume(root, byte_count, roots=None):
    candidates = roots or [Path("E:/paper_archives/outputs/experiments/first_stage_nondrl12_20261012/delivery"),
                          Path(root)/"delivery", Path("C:/paper_archives/outputs/experiments/first_stage_nondrl12_20261012/delivery")]
    for candidate in candidates:
        candidate = Path(candidate)
        try:
            if free_bytes(candidate) >= RESERVE + byte_count:
                return candidate
        except OSError:
            continue
    raise RuntimeError("no canonical E/D/C volume has package bytes plus8GiB reserve")


def verify_zip(path, rows, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError("immutable delivery ZIP SHA differs")
    expected = {r["member"]:r for r in rows}
    with zipfile.ZipFile(path) as bundle:
        if len(bundle.namelist()) != len(expected) or set(bundle.namelist()) != set(expected):
            raise ValueError("delivery ZIP member set differs")
        for name,row in expected.items():
            safe_member(name)
            if bundle.getinfo(name).file_size != row["bytes"]:
                raise ValueError("delivery member size differs")
            with bundle.open(name) as source:
                if stream_sha(source) != row["sha256"]:
                    raise ValueError("delivery member SHA differs")


def package(root=ops.LOCAL, report_path=None, placement_file=None, assets_manifest=None, *, roots=None):
    gate = gate_state(root,report_path,placement_file)
    if not gate["ready"]:
        raise RuntimeError("delivery not ready: "+"; ".join(gate["errors"]))
    manifest, files, generated = _inputs(root,gate,assets_manifest)
    raw_manifest=encoded(manifest); identity=hashlib.sha256(raw_manifest).hexdigest()
    destination=choose_volume(root,sum(r["bytes"] for r in manifest["members"])*2+(2<<20),roots)/("final_"+identity[:20])
    destination.mkdir(parents=True,exist_ok=True)
    receipt_path=destination/"local_package_receipt.json"
    if receipt_path.exists():
        return load_package(receipt_path)[0]
    inventory=destination/"input_manifest.json"
    if inventory.exists() and inventory.read_bytes()!=raw_manifest:
        raise ValueError("immutable delivery input manifest differs")
    inventory.write_bytes(raw_manifest)
    rows=manifest["members"]+[{"member":"indexes/input_manifest.json", "bytes":len(raw_manifest), "sha256":identity}]
    target=destination/("nd12_final_"+identity[:20]+".zip")
    partial=target.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as bundle:
        for row in rows:
            name=row["member"]
            if name=="indexes/input_manifest.json":
                payload=raw_manifest
            elif name in generated:
                payload=generated[name]
            else:
                pin(files[name],row["sha256"],small=True);payload=files[name].read_bytes()
            if len(payload)!=row["bytes"] or hashlib.sha256(payload).hexdigest()!=row["sha256"]:
                raise ValueError("input changed during packaging")
            if free_bytes(destination)<RESERVE+len(payload):
                raise RuntimeError("canonical package volume crossed8GiB reserve")
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            bundle.writestr(info,payload)
    partial.replace(target)
    digest=sha(target);verify_zip(target,rows,digest)
    receipt={"schema":SCHEMA,"package_id":identity[:20],"input_manifest_sha256":identity,
             "input_manifest_path":str(inventory.resolve()), "zip_path":str(target.resolve()),
             "zip_sha256":digest,"zip_bytes":target.stat().st_size,"members":rows,
             "local_receipt_path":str(receipt_path.resolve()),"all_member_shas_verified":True,
             "raw_job_zips_embedded":False}
    save(receipt_path,receipt)
    return receipt


def load_package(receipt_path):
    receipt_path=Path(receipt_path).resolve();receipt=read(receipt_path)
    if (receipt.get("schema")!=SCHEMA or receipt_path.name!="local_package_receipt.json" or
            receipt_path.parent.name!="final_"+receipt.get("package_id","") or
            Path(receipt["zip_path"]).resolve().parent!=receipt_path.parent or
            Path(receipt["input_manifest_path"]).resolve().parent!=receipt_path.parent or
            receipt.get("raw_job_zips_embedded") is not False):
        raise ValueError("invalid existing canonical delivery receipt")
    manifest=read(pin(receipt["input_manifest_path"],receipt["input_manifest_sha256"]))
    if hashlib.sha256(encoded(manifest)).hexdigest()!=receipt["input_manifest_sha256"]:
        raise ValueError("input manifest bytes/canonical identity differ")
    pin(receipt["zip_path"],receipt["zip_sha256"])
    verify_zip(receipt["zip_path"],receipt["members"],receipt["zip_sha256"])
    if (receipt["members"][:-1]!=manifest["members"] or
            receipt["package_id"]!=receipt["input_manifest_sha256"][:20]):
        raise ValueError("package inventory identity differs")
    return receipt,manifest


def resume_stream(local_path, remote, remote_path, byte_count):
    """Resume only an equal prefix. No restart/truncate on incompatible partials."""
    try:
        size=remote.stat(remote_path).st_size
    except FileNotFoundError:
        size=0
    if size>byte_count:
        raise ValueError("partial upload larger than immutable package")
    with Path(local_path).open("rb") as local:
        if size:
            with remote.open(remote_path,"rb") as current:
                remaining=size
                while remaining:
                    block=current.read(min(1<<20,remaining))
                    if not block or local.read(len(block))!=block:
                        raise ValueError("partial upload prefix differs")
                    remaining-=len(block)
        with remote.open(remote_path,"r+b" if size else "wb") as target:
            target.seek(size)
            for block in iter(lambda:local.read(1<<20),b""):
                target.write(block)
            target.flush()
    return {"resumed_bytes":size,"uploaded_bytes":byte_count-size}


def _remote_verify(transport,port,remote_path,receipt):
    if port not in (20418,20424,20473):
        raise ValueError("delivery readback must exclude physical20474")
    rows={r["member"]:{"bytes":r["bytes"],"sha256":r["sha256"]} for r in receipt["members"]}
    if len(rows)!=len(receipt["members"]):raise ValueError("duplicate delivery validation member")
    for name in rows:safe_member(name)
    # The receipt's complete member map used to be interpolated into Python
    # code and then base64 into one SSH shell argument. Linux's per-argument
    # limit is independent of the allowed ZIP/member size. Send unchanged
    # expected values as a SHA-pinned SFTP sidecar instead; shell code stays
    # bounded even when the immutable package has thousands of members.
    payload=encoded({"schema":"s1_nd12_remote_member_validation_v1","members":rows})
    manifest_sha=hashlib.sha256(payload).hexdigest()
    target=PurePosixPath(remote_path)
    if not target.is_relative_to(PurePosixPath(ops.BASE)/"delivery") or ".." in target.parts:
        raise ValueError("remote validation path escaped the delivery root")
    remote_manifest=str(target.parent/("validation_"+manifest_sha+".json"))
    with tempfile.TemporaryDirectory(prefix="nd12_delivery_validation_") as temporary:
        local_manifest=Path(temporary)/"validation.json";local_manifest.write_bytes(payload)
        sent=transport.upload(local_manifest,remote_manifest,port)
        if (sent.get("exit_code")!=0 or sent.get("local_sha256")!=manifest_sha or sent.get("remote")!=manifest_sha):
            raise ValueError("remote complete-member validation manifest transfer SHA differs")
    code=f'''import hashlib,json,pathlib,zipfile
p=pathlib.Path({remote_path!r});root=pathlib.Path({ops.BASE!r})/'delivery'
if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('delivery path escaped group')
def sha(s):
 h=hashlib.sha256()
 for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
with p.open('rb') as s:digest=sha(s)
if digest!={receipt['zip_sha256']!r} or p.stat().st_size!={receipt['zip_bytes']}:raise ValueError('whole package differs')
manifest=pathlib.Path({remote_manifest!r})
if manifest.is_symlink() or not manifest.resolve().is_relative_to(root.resolve()):raise ValueError('member manifest escaped group')
if manifest.stat().st_size!={len(payload)}:raise ValueError('member manifest size differs')
with manifest.open('rb') as s:
 if sha(s)!={manifest_sha!r}:raise ValueError('member manifest SHA differs')
expected=json.loads(manifest.read_text(encoding='utf-8'))
if expected.get('schema')!='s1_nd12_remote_member_validation_v1':raise ValueError('member manifest schema differs')
rows=expected['members']
if len(rows)!={len(rows)}:raise ValueError('member manifest count differs')
with zipfile.ZipFile(p) as z:
 if len(z.namelist())!=len(rows) or set(z.namelist())!=set(rows):raise ValueError('member set differs')
 for name,row in rows.items():
  if z.getinfo(name).file_size!=row['bytes']:raise ValueError('member size differs')
  with z.open(name) as s:
   if sha(s)!=row['sha256']:raise ValueError('member SHA differs')
st=p.stat();print(json.dumps({{'physical_port':{port},'path':str(p),'zip_sha256':digest,'member_count':len(rows),'all_member_shas_verified':True,'device':st.st_dev,'inode':st.st_ino}}))
'''
    result=transport.remote_python(code,port,timeout=120)
    if result["exit_code"]:
        raise RuntimeError("physical"+str(port)+" package readback failed: "+result["stderr"][-600:])
    return json.loads(result["stdout"].splitlines()[-1])


def resume_upload(receipt_path,root=ops.LOCAL,report_path=None,placement_file=None):
    receipt,manifest=load_package(receipt_path)
    gate=gate_state(root,report_path,placement_file)
    if not gate["ready"] or gate["pins"]!=manifest["input_pins"]:
        raise RuntimeError("existing package formal input gates/SHA changed; no repackage")
    transport=placement(root,placement_file);transport.verify_unchanged()
    if manifest["placement"]!=transport.placement:
        raise ValueError("immutable package placement changed")
    remote_dir=ops.BASE+"/delivery/final_"+receipt["package_id"]
    remote_path=remote_dir+"/"+Path(receipt["zip_path"]).name
    # Single upload to shared/home through physical18 only. Resume-safe sidecar
    # prefix is verified byte-for-byte, then an atomic rename publishes the ZIP.
    result=transport.remote_python(f"import pathlib,shutil; p=pathlib.Path({remote_dir!r}); p.mkdir(parents=True,exist_ok=True); assert not p.is_symlink() and p.resolve().is_relative_to((pathlib.Path({ops.BASE!r})/'delivery').resolve()); assert not pathlib.Path({remote_path!r}).is_symlink() and not pathlib.Path({remote_path+'.partial'!r}).is_symlink(); assert shutil.disk_usage(p).free >= {RESERVE+receipt['zip_bytes']}",20418,timeout=30)
    if result["exit_code"]:
        raise RuntimeError("shared remote path/capacity gate failed")
    client=transport.connect(20418)
    try:
        with client.open_sftp() as remote:
            try:
                remote.stat(remote_path);exists=True
            except FileNotFoundError:
                exists=False
            if not exists:
                transfer=resume_stream(receipt["zip_path"],remote,remote_path+".partial",receipt["zip_bytes"])
                # Before atomic publication, verify whole/member SHA on partial.
                _remote_verify(transport,20418,remote_path+".partial",receipt)
                remote.posix_rename(remote_path+".partial",remote_path)
            else:
                transfer={"resumed_bytes":receipt["zip_bytes"],"uploaded_bytes":0}
    finally:
        client.close()
    checks={str(port):_remote_verify(transport,port,remote_path,receipt) for port in (20418,20424,20473)}
    if len({(r["device"],r["inode"]) for r in checks.values()})!=1:
        raise ValueError("three physical entry points did not read one shared package")
    result={"schema":SCHEMA,"completed_utc":datetime.now(timezone.utc).isoformat(),
            "package_id":receipt["package_id"],"zip_sha256":receipt["zip_sha256"],
            "shared_remote_path":remote_path,"single_shared_physical_copy":True,
            "physical_access":checks,"logical_to_physical":transport.mapping,
            "placement_sha256":transport.mapping_sha256,"transfer":transfer,"raw_job_zips_embedded":False}
    result["all_three_physical_readback_verified"]=True
    save(Path(receipt_path).parent/"final_delivery_receipt.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("plan","deliver","resume-upload"))
    parser.add_argument("--root",type=Path,default=ops.LOCAL)
    parser.add_argument("--report",type=Path)
    parser.add_argument("--placement-file",type=Path)
    parser.add_argument("--assets-manifest",type=Path)
    parser.add_argument("--local-receipt",type=Path)
    args=parser.parse_args()
    if args.mode=="plan":
        result=gate_state(args.root,args.report,args.placement_file)
        if result["ready"]:
            manifest,_,_=_inputs(args.root,result,args.assets_manifest)
            identity=hashlib.sha256(encoded(manifest)).hexdigest()
            size=sum(r["bytes"] for r in manifest["members"])
            result["package_plan"]={"package_id":identity[:20],"input_manifest_sha256":identity,
                "members":manifest["members"],"uncompressed_bytes":size,
                "canonical_volume":str(choose_volume(args.root,size*2+(2<<20))),"raw_job_zips_embedded":False}
        print(json.dumps({k:v for k,v in result.items() if k!="report"},ensure_ascii=False))
        return 0 if result["ready"] else 2
    if args.mode=="deliver":
        local=package(args.root,args.report,args.placement_file,args.assets_manifest)
        result=resume_upload(local["local_receipt_path"],args.root,args.report,args.placement_file)
    else:
        if not args.local_receipt:parser.error("resume-upload requires --local-receipt; no package regeneration")
        result=resume_upload(args.local_receipt,args.root,args.report,args.placement_file)
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=="__main__":
    raise SystemExit(main())
