"""Exact phase plans. Callers must supply real root-verified artifact references."""
from copy import deepcopy
import hashlib
from pathlib import Path
DESIGN_SHA="74246c2f5baa2f40cc88fff248add335a2c7151028dd5193873256f55e7fd892"
GROUP="S1-AI17-R6-MODEL";PORTS=(20418,20424,20473,20474)
NODE_INDICES=[i for m in (0,4,8,11,15,19) for i in range(4*m,4*m+4)]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require_ref(r):
 if not isinstance(r,dict) or not isinstance(r.get("path"),str) or len(r.get("sha256",""))!=64:raise ValueError("actual pinned reference required")
 return deepcopy(r)
def check_qualified_indices(indices):
 candidates={4*((k*575)//191)+profile for k in range(192) for profile in range(4)}
 if not 692<=len(indices)<=768 or indices!=sorted(set(indices)) or any(type(i)is not int or i not in candidates for i in indices):raise ValueError("immutable qualified subset of original768 train candidates")

def base(commit,phase,inputs):
 if len(commit)!=40:raise ValueError("frozen scientific commit")
 if inputs["design_ref"]["sha256"]!=DESIGN_SHA:raise ValueError("signed v2 design")
 return {"group":GROUP,"phase":phase,"authorized":True,"scientific_commit":commit,"design":require_ref(inputs["design_ref"]),"identity_ref":require_ref(inputs["identity_ref"]),"qualification_refs":deepcopy(inputs["qualification_refs"]),"prerequisite_gate_ref":require_ref(inputs["prerequisite_gate_ref"]),"model_identity":inputs["model_identity"],"training_seed":51001,"active_seconds_max":28800,"global_updates_max":6000,"tasks":{}}
def supply_plan(commit,inputs,*,port):
 if port not in PORTS:raise ValueError("approved supply lane")
 check_qualified_indices(inputs["qualified_case_indices"])
 p=base(commit,"train_setup",inputs);refs={k:require_ref(inputs[k+"_ref"]) for k in ("qualified","initialization_receipt","source_equivalence")};refs["labels_gate"]=require_ref(inputs["prerequisite_gate_ref"])
 cfg={"schema_version":15,"research_stage":"coverage","algorithm":"ai_search_r6_model","name":"R6_train_setup","protocol_ref":p["design"],"datasets":{"train":deepcopy(inputs["dataset_ref"])},"workers":1,"resources":{"minimum_free_bytes":8589934592,"parent_children_rss_bytes":4294967296},"hard_watchdogs":{"training_active_seconds":28800}}
 p["tasks"][cfg["name"]]={"config":cfg,"mode":"train_setup","dataset_role":"train","case_indices":list(inputs["qualified_case_indices"]),"executor_port":port,"methods":[],"model_inputs":refs,"model_identity":p["model_identity"]};p["budget"]={"prepare":0,"T":0,"D5":0,"updates":0};p["future_phase_dispatch_allowed"]=False;audit(p);return p

def training_plan(commit,inputs,*,port,start=0,end=2000):
 if port not in PORTS or (start,end) not in ((0,2000),(2000,6000)):raise ValueError("single frozen node segments")
 check_qualified_indices(inputs["qualified_case_indices"])
 phase="train2000" if start==0 else "train6000";p=base(commit,phase,inputs)
 required=("qualified","initial_state","normalization","feature_cache","initialization_receipt","supply_endpoint","source_equivalence")
 refs={k:require_ref(inputs[k+"_ref"]) for k in required}
 refs["labels_gate"]=require_ref(inputs.get("labels_gate_ref",inputs["prerequisite_gate_ref"]))
 if start:
  refs.update(checkpoint=require_ref(inputs["checkpoint_ref"]),recovery_clock=require_ref(inputs["recovery_clock_ref"]),node_gate=require_ref(inputs["node_gate_ref"]))
 cfg={"schema_version":15,"research_stage":"coverage","algorithm":"ai_search_r6_model","name":"R6_"+phase,"protocol_ref":p["design"],"datasets":{"train":deepcopy(inputs["dataset_ref"])},"workers":1,"resources":{"minimum_free_bytes":8589934592,"parent_children_rss_bytes":4294967296},"hard_watchdogs":{"training_active_seconds":28800},"training":{"seed":51001,"batch":16,"microbatch":4,"start_update":start,"end_update":end,"checkpoint_interval":100}}
 p["tasks"][cfg["name"]]={"config":cfg,"mode":phase,"dataset_role":"train","case_indices":list(inputs["qualified_case_indices"]),"executor_port":port,"methods":[],"model_inputs":refs,"model_identity":p["model_identity"]}
 p["budget"]={"prepare":0,"T":0,"D5":0,"updates":end-start};p["future_phase_dispatch_allowed"]=False;audit(p);return p

def node_plan(commit,inputs,routing=None):
 p=base(commit,"node2000",inputs);routing=routing or {str(i):PORTS[j%4] for j,i in enumerate(NODE_INDICES)}
 if set(routing)!=set(map(str,NODE_INDICES)) or any(v not in PORTS for v in routing.values()):raise ValueError("exact node24 routing")
 refs={k:require_ref(inputs[k+"_ref"]) for k in ("checkpoint","normalization","initialization_receipt","registry","source_equivalence")}
 for port in PORTS:
  key="R6_node2000_"+str(port);cfg={"schema_version":15,"research_stage":"coverage","algorithm":"ai_search_r6_model","name":key,"protocol_ref":p["design"],"datasets":{"development":deepcopy(inputs["dataset_ref"])},"workers":1,"resources":{"minimum_free_bytes":8589934592,"parent_children_rss_bytes":4294967296},"hard_watchdogs":{"AI":90,"decoder":15}}
  p["tasks"][key]={"config":cfg,"mode":"node2000","dataset_role":"development","case_indices":[i for i in NODE_INDICES if routing[str(i)]==port],"executor_port":port,"methods":["AI","RULE","RANDOM"],"model_inputs":deepcopy(refs),"model_identity":p["model_identity"]}
 p["budget"]={"prepare":0,"T":0,"D5":72,"updates":0};p["future_phase_dispatch_allowed"]=False;audit(p);return p

def audit(p):
 if p.get("phase") not in ("train_setup","train2000","train6000","node2000") or p.get("training_seed")!=51001 or p.get("global_updates_max")!=6000 or p.get("active_seconds_max")!=28800:raise ValueError("isolated contracted stage")
 if p["phase"]=="train_setup":
  if len(p["tasks"])!=1 or p["budget"]!={"prepare":0,"T":0,"D5":0,"updates":0}:raise ValueError("one bounded input supply stage")
  task=next(iter(p["tasks"].values()))
  if task["dataset_role"]!="train" or task["methods"]:raise ValueError("train supply role")
 elif p["phase"].startswith("train"):
  if len(p["tasks"])!=1:raise ValueError("one model one GPU")
  task=next(iter(p["tasks"].values()));target=(0,2000) if p["phase"]=="train2000" else (2000,6000)
  if (task["config"]["training"]["start_update"],task["config"]["training"]["end_update"])!=target or task["dataset_role"]!="train" or task["methods"]:raise ValueError("exact updates/role")
  if p["budget"]!={"prepare":0,"T":0,"D5":0,"updates":target[1]-target[0]}:raise ValueError("nontransferable training cap")
 else:
  cases=[i for t in p["tasks"].values() for i in t["case_indices"]]
  if sorted(cases)!=NODE_INDICES or len(set(cases))!=24 or p["budget"]!={"prepare":0,"T":0,"D5":72,"updates":0}:raise ValueError("exact node72 calls")
  if any(t["methods"]!=["AI","RULE","RANDOM"] or t["dataset_role"]!="development" for t in p["tasks"].values()):raise ValueError("three methods same role")
 return {"passed":True,"phase":p["phase"],"science_calls_here":0,"budget":p["budget"]}

FUTURE_PHASE_CAPS={"fixed6000_development80":{"role":"development","cases":80,"prepare":80,"T":0,"D5":240,"methods":["AI","RULE","RANDOM"],"prior_gate":"node2000"},"validation80":{"role":"validation","cases":80,"prepare":80,"T":80,"D5":240,"methods":["AI","RULE","RANDOM","T"],"prior_gate":"fixed6000_development80"},"test160":{"role":"test","cases":160,"prepare":160,"T":160,"D5":480,"methods":["AI","RULE","RANDOM","T"],"prior_gate":"validation80"}}
def sealed_phase_interface(phase):
 if phase not in FUTURE_PHASE_CAPS:raise ValueError("unknown later phase")
 return {"phase":phase,**deepcopy(FUTURE_PHASE_CAPS[phase]),"checkpoint_updates":6000,"dispatch_allowed":False,"requires_separate_root_frozen_exact_input_plan":True}
