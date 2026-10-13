"""Durable labels worker: one prepare/T/EXACT per signed train candidate."""
import os,sys,time,shutil,traceback
from pathlib import Path
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from .storage import read_json,digest,now,Store
from .coverage_ai_search import pin
from .ai_search_runner import load_cases
from .r6_io import read_asset,pin_tree
from .r6_stage_guard import run_stage
from .geoload_runner import exclusive_lock

def guarded(directory,request,limit,stop):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);wal=directory/"started.json";receipt=directory/"guard.json"
 if wal.exists():
  if not receipt.exists():raise ValueError("unknown WAL retained: root reconciliation required; no replay")
  if read_json(directory/"request.json",{})!=request or digest(directory/"request.json")!=read_json(wal,{})["request_sha256"]:raise ValueError("original request changed; no replay")
  stage=read_json(receipt,{})
  pin_tree(stage)
  if stage.get("status")!="completed":raise ValueError("old unsuccessful call retained; no replay")
  output=read_json(pin(stage["output_ref"]),{});return {**stage,"reused_completed_call":True},read_asset(output["asset_ref"]),output["asset_ref"]
 atomic_json(directory/"request.json",request);atomic_json(wal,{"at":now(),"request_sha256":digest(directory/"request.json"),"algorithm_spend_reserved":True})
 stage=run_stage([sys.executable,"-B","-u","-m","satellite_rl.workbench.ai_search_r6_stage","--request",str(directory/"request.json"),"--directory",str(directory)],directory,limit,2147483648,stop)
 atomic_json(receipt,stage)
 if stage.get("status")!="completed":raise ValueError("failed/unknown science stage; successors stopped")
 output=read_json(pin(stage["output_ref"]),{});return stage,read_asset(output["asset_ref"]),output["asset_ref"]

def run_job(directory,resume=False):
 from . import coverage_ai_search_r6 as policy,frozen_round_tasks
 from satellite_rl.coverage.ai_search.core import Scene
 from scripts.stage1_ai_search_r6.label_audit import audit_bundle,audit_exact_proposal
 from scripts.stage1_ai_search_r5.audit import audit_layout
 directory=Path(directory);job=read_json(directory/"job.json",{});cfg=read_json(directory/"config.json",{});status=read_json(directory/"status.json",{})
 def update(**kw):status.update(kw,pid=os.getpid(),heartbeat_at=now());atomic_json(directory/"status.json",status)
 def stop():
  if shutil.disk_usage(directory).free<8589934592:raise OSError("8GiB reserve reached")
  return (directory/"stop.request").exists()
 try:
  with exclusive_lock(directory/"worker.lock"),exclusive_lock(directory.parent.parent/"compute.lock",blocking=True):
   frozen_round_tasks.verify(Store(job["archive_root"],source_root=job["source_root"]),job,policy);plan=read_json(pin(job["references"]["plan"]),{});metadata={v["input_case_index"]:v for v in plan["case_identity_metadata"]};index=read_json(directory/"case_index.json",{})
   update(stage="running",total_cases=len(job["case_indices"]),completed_cases=len(index))
   for i in job["case_indices"]:
    if str(i) in index:
     old=read_json(pin(index[str(i)]),{})
     meta=metadata[i]
     expected_identity={k:meta[k] for k in ("case_id","mother_id","profile","role","identity_sha256","input_sha256")}
     if old["input_case_index"]!=i or old["source_commit"]!=job["scientific_commit"] or old["identity"]!=expected_identity:raise ValueError("recovered row identity/source differs")
     endpoint=read_json(pin(index[str(i)]["cost_endpoint_ref"]),{})
     if endpoint["identity"]!=expected_identity or endpoint["raw_ref"]["sha256"]!=index[str(i)]["sha256"] or not np.isclose(sum(endpoint["components"].values()),endpoint["complete_offline_label_seconds"],rtol=1e-12,atol=1e-9):raise ValueError("cost endpoint mismatch")
     pin(old["scene_ref"])
     for ref in old["guard_refs"]:pin_tree(ref)
     assets=[read_asset(old[name]) for name in ("prepared_ref","original_T_terminal_ref","D5_qualification_ref")]
     with np.load(pin(old["scene_ref"]),allow_pickle=False) as z:
      audit_scene={"positions":z["positions"].tolist(),"demands":z["demands"].tolist(),"aircraft_ids":list(map(str,z["aircraft_ids"])),"profile":str(z["profile"])}
     proposal_checked=audit_exact_proposal(read_asset(assets[2]["proposal_ref"]),assets[1]["result"],expected_identity,plan["inherited_core_sha256"],job["source_manifest_sha256"])
     if proposal_checked!=old["proposal_audit"]:raise ValueError("recovered proposal audit differs")
     checked=audit_bundle(audit_scene,*assets,expected_identity,plan["inherited_core_sha256"],old["cost_closed"])
     if checked!=old["label_audit"]:raise ValueError("recovered independent audit differs")
     continue
    if stop():update(stage="stopped");return
    start=time.monotonic();case=load_cases(cfg["datasets"]["train"],[i])[0];meta=metadata[i]
    if any(case[k]!=meta[v] for k,v in (("case_id","case_id"),("mother_id","mother_id"),("demand_profile","profile"))):raise ValueError("signed case identity mismatch")
    scene=Scene(case["positions"],case["demands"],case["aircraft_ids"],case["profile"]);ident={k:meta[k] for k in ("case_id","mother_id","profile","role","identity_sha256")};ident["input_sha256"]=scene.input_sha256
    from scripts.stage1_ai_search_r6.data_identity import aircraft_sha
    if scene.input_sha256!=meta["input_sha256"] or aircraft_sha(case["aircraft_ids"])!=meta["aircraft_order_sha256"]:raise ValueError("pre-frozen input/order SHA differs")
    target=directory/"bundles"/str(i);target.mkdir(parents=True,exist_ok=True);raw=target/"scene.npz"
    if not raw.exists():np.savez_compressed(raw,positions=case["positions"],demands=case["demands"],aircraft_ids=case["aircraft_ids"],profile=case["profile"])
    manifest={"stage":"labels","role":"train","admitted":True,"identity_sha256":ident["identity_sha256"],"dataset_sha256":cfg["datasets"]["train"]["sha256"],"cases":{ident["case_id"]:ident}}
    request={"identity":ident,"scene_ref":{"path":str(raw),"sha256":digest(raw)},"source_sha256":plan["inherited_core_sha256"],"producer_sha256":job["source_manifest_sha256"],"phase_manifest":manifest}
    input_seconds=time.monotonic()-start
    guard_start=time.monotonic()
    pguard,prep,pref=guarded(target/"preparation",{**request,"stage":"preparation"},20,stop)
    parent_guard_seconds={"preparation":time.monotonic()-guard_start}
    guard_start=time.monotonic()
    tguard,teacher,tref=guarded(target/"teacher",{**request,"stage":"teacher","prepared_ref":pref},45,stop)
    parent_guard_seconds["teacher"]=time.monotonic()-guard_start
    guard_start=time.monotonic()
    dguard,replay,dref=guarded(target/"decoder",{**request,"stage":"decoder","prepared_ref":pref,"teacher_ref":tref},15,stop)
    parent_guard_seconds["decoder"]=time.monotonic()-guard_start
    audit_start=time.monotonic()
    audit_scene={"positions":case["positions"].tolist(),"demands":case["demands"].tolist(),"aircraft_ids":list(case["aircraft_ids"]),"profile":case["profile"]}
    quality=audit_bundle(audit_scene,prep,teacher,replay,ident,request["source_sha256"],not any(g.get("unknown_algorithm_spend") for g in (pguard,tguard,dguard)))
    proposal_checked=audit_exact_proposal(read_asset(replay["proposal_ref"]),teacher["result"],ident,request["source_sha256"],job["source_manifest_sha256"])
    guard_seconds={name:g["actual_stage_wall_seconds"] for name,g in (("preparation",pguard),("teacher",tguard),("decoder",dguard))}
    components={"raw_input_identity_and_scene_publication":input_seconds,**{"guarded_"+k:v for k,v in parent_guard_seconds.items()},**{"reused_original_child_stage_"+k:g["actual_stage_wall_seconds"] for k,g in (("preparation",pguard),("teacher",tguard),("decoder",dguard)) if g.get("reused_completed_call")},"independent_audit":time.monotonic()-audit_start}
    if not all(isinstance(v,(int,float)) and np.isfinite(v) and v>=0 for v in components.values()):raise ValueError("label cost components missing")
    row={"identity":ident,"input_case_index":i,"scene_ref":request["scene_ref"],"prepared_ref":pref,"original_T_terminal_ref":tref,"D5_qualification_ref":dref,"label_audit":quality,"proposal_audit":proposal_checked,"cost_closed":not any(g.get("unknown_algorithm_spend") for g in (pguard,tguard,dguard)),"guard_refs":[{"path":str(target/name/"guard.json"),"sha256":digest(target/name/"guard.json")} for name in ("preparation","teacher","decoder")],"wall_seconds":time.monotonic()-start,"compressed_asset_bytes":sum(r["compressed_bytes"] for r in (pref,tref,dref)),"actual_calls":{"prepare":1,"T":1,"D5":1},"source_commit":job["scientific_commit"]}
    publication_start=time.monotonic();path=target/"label_receipt.json";atomic_json(path,row);index[str(i)]={"path":str(path),"sha256":digest(path)};atomic_json(directory/"case_index.json",index);update(completed_cases=len(index));components["label_row_index_status_publication"]=time.monotonic()-publication_start
    endpoint=target/"label_cost_endpoint.json";atomic_json(endpoint,{"identity":ident,"raw_ref":index[str(i)],"components":components,"child_stage_wall_seconds_description":guard_seconds,"complete_offline_label_seconds":sum(components.values()),"cost_closed":row["cost_closed"],"stage_wall_includes_child_read_algorithm_compression_publication":True,"metadata_scan_and_controller_separate":True,"endpoint_receipt_IO_separate_metadata":True})
    index[str(i)]["cost_endpoint_ref"]={"path":str(endpoint),"sha256":digest(endpoint)};atomic_json(directory/"case_index.json",index)
   atomic_json(directory/"result_manifest.json",{"group":policy.GROUP,"phase":"labels","cases":index,"science_updates":0,"future_dispatch_allowed":False});update(stage="completed",completed_at=now())
 except BaseException as e:atomic_json(directory/"failure.json",{"message":str(e),"traceback":traceback.format_exc(),"unknown_spend_preserved":True});update(stage="failed",error=str(e));raise
