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
from pathlib import Path
import shutil
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
    summary = {"experiment_id": "S1-ND12", "formal_ready": True, "issues": [],
               "scientific_commit": report["scientific_commit"], "routes": {
                   route: {"status": record["status"], "methods": {
                       method: {"summary": data.get("summary"), "run_ids": data.get("run_ids", [])}
                       for method,data in record.get("methods",{}).items()}}
                   for route,record in report["routes"].items()}}
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
    for name in ("final_delivery.py", "report.py", "strict_audit.py", "execution_sources.py", "placement_transport.py"):
        add("reproduce/"+name, Path(__file__).parent/name)
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
    rows={r["member"]:{"bytes":r["bytes"],"sha256":r["sha256"]} for r in receipt["members"]}
    code=f'''import hashlib,json,pathlib,zipfile
p=pathlib.Path({remote_path!r});root=pathlib.Path({ops.BASE!r})/'delivery'
if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('delivery path escaped group')
def sha(s):
 h=hashlib.sha256()
 for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
with p.open('rb') as s:digest=sha(s)
if digest!={receipt['zip_sha256']!r} or p.stat().st_size!={receipt['zip_bytes']}:raise ValueError('whole package differs')
rows={rows!r}
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
