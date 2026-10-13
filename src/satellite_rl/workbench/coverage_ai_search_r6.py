"""Source-frozen R6 label stage; dependent model stages remain closed."""
from copy import deepcopy
from pathlib import Path
import os,sys
from .storage import read_json,digest,validate_id
from .coverage_ai_search import pin
from . import frozen_round_tasks
from scripts.stage1_ai_search_r6.plan_builder import DESIGN_SHA,TRAIN_SHA,audit
GROUP="S1-AI17-R6";MODES=("labels",);CANONICAL_DATASETS=True
FIELDS=("config","mode","dataset_role","case_indices","executor_port","methods");RUNTIME_FIELDS=FIELDS[1:]
def task_contract(b):return {k:deepcopy(b.get(k)) for k in FIELDS}
def run_prefix(c):return "labels_ai_search_r6"
def plan_references(p):return {"identity":p["identity_ref"]}
def external_references(c):return {}
def decorate_job(c,p,refs,commit):return {"phase":"labels","execution_commit":commit,"algorithm":"ai_search_r6","network":"direct_supervised_labels","kind":"prepare","execution_device":"cpu","methods":c["methods"]}
def verify_metadata(j,p):
 if j.get("phase")!="labels" or j.get("algorithm")!="ai_search_r6" or j.get("execution_commit")!=j["scientific_commit"]:raise ValueError("R6 source phase")
def expected_results(c):return len(c["case_indices"])
def deployment(store):
 root=store.archive_root.resolve();port=root.parent.parent.name
 expected=Path("/home/u2024110224/first_stage_satellite_rl/s1_ai17")/port/"round6"/"archive"
 if root!=expected or port not in ("20418","20424","20473","20474") or os.environ.get("SATELLITE_RL_SSH_PORT")!=port:raise ValueError("isolated R6 root required")
 from .coverage import _require_remote_dqn
 _require_remote_dqn();return port

def validate_config(c):
 if c.get("schema_version")!=14 or c.get("algorithm")!="ai_search_r6" or c.get("workers")!=1 or c.get("research_stage")!="coverage":raise ValueError("R6 schema14")
 if set(c["datasets"])!={"train"} or {k:c["datasets"]["train"][k] for k in ("role","count","sha256")}!={"role":"train","count":2304,"sha256":TRAIN_SHA}:raise ValueError("exact real training source")
 if c["resources"]!={"minimum_free_bytes":8589934592,"parent_children_rss_bytes":2147483648} or c["hard_watchdogs"]!={"preparation":20,"teacher":45,"decoder":15}:raise ValueError("frozen limits")
 return deepcopy(c)
def validate_task(c,p,port):
 audit(p);validate_config(c["config"])
 if str(c["executor_port"])!=str(port) or c not in p["tasks"].values():raise ValueError("exact lane ownership")
 if c["config"]["protocol_ref"]!=p["design"]:raise ValueError("protocol differs")
 return c["config"]
def validate_admission(p,plan_ref,review_ref):
 if p.get('scientific_commit')=='7bb5e68456c00846f5a8b249bba19f8800ea13ee':
  from .r6_runtime_dependency import service_validate_admission
  return service_validate_admission(p,plan_ref,review_ref,Path(__file__).resolve().parents[3])
 audit(p);r=read_json(pin(review_ref),{})
 if not r.get("passed") or r.get("group")!=GROUP or r.get("scientific_commit")!=p["scientific_commit"] or r.get("plan_sha256")!=plan_ref["sha256"] or r.get("design_sha256")!=DESIGN_SHA:raise ValueError("root exact source-plan admission")
 pin(p["design"]);pin(p["identity_ref"])
 if not p.get("inherited_core_sha256") or len(p["inherited_core_sha256"])!=64:raise ValueError("b7 immutable core SHA required")
 core=Path(__file__).resolve().parents[1]/"coverage/ai_search_r5/core.py"
 if digest(core)!=p["inherited_core_sha256"]:raise ValueError("b7 core changed")
 if not p["qualification_refs"]:raise ValueError("source-bound qualification required")
 for ref in p["qualification_refs"].values():
  q=read_json(pin(ref),{})
  if q.get("passed")is not True or q.get("scientific_commit")!=p["scientific_commit"] or q.get("design_sha256")!=DESIGN_SHA:raise ValueError("qualification source differs")
  members=q.get("source_file_sha256",{})
  base=Path(__file__).resolve().parents[3]
  if not members or any(digest(base/name)!=sha for name,sha in members.items()):raise ValueError("qualified source members changed")

def resource_admission(store,j):
 from .r6_resources import observe
 from datetime import datetime,timezone
 port=deployment(store);r=read_json(store.archive_root.parent/"ops/resource_route.json",{})
 age=(datetime.now(timezone.utc)-datetime.fromisoformat(r.get("observed_utc","1970-01-01T00:00:00+00:00"))).total_seconds()
 if r.get("group")!=GROUP or str(r.get("ssh_port"))!=port or not r.get("ready") or r.get("scientific_commit")!=j["scientific_commit"] or not -5<=age<=120:raise ValueError("fresh resource receipt required")
 actual=observe(store.archive_root)
 if actual["foreign_workers"] or actual["gpu"].split(",")[0].strip()!=r["gpu_uuid"]:raise ValueError("GPU/worker differs")
 return actual

def api(app,method,parts,body):
 policy=sys.modules[__name__]
 if method=="GET" and parts==["schema"]:return {"group":GROUP,"schema_version":14,"modes":MODES,"labels":768,"training_allowed":False,"test_allowed":False}
 if method=="GET" and parts==["tasks"]:
  from .coverage_ai_search import observed_status
  return {"tasks":[{"job":j,"status":observed_status(app.store,j,s)} for j,s in app.store.jobs() if j.get("experiment_id")==GROUP]}
 if method=="POST" and parts==["tasks"]:return frozen_round_tasks.create(app,body,policy)
 if len(parts)>=2 and parts[0]=="tasks":
  run=validate_id(parts[1]);j=read_json(app.store.job_dir(run)/"job.json",{})
  if j.get("experiment_id")!=GROUP:raise ValueError("foreign group")
  if method=="GET" and len(parts)==2:return app.run_detail(run)
  if method=="POST" and parts[2:]==["resume"]:return frozen_round_tasks.resume(app.scheduler,run,policy)
  if method=="POST" and parts[2:]==["stop"]:return app.scheduler.request_stop(run)
 raise FileNotFoundError("unknown R6 endpoint")
