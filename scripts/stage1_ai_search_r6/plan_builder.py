"""Exact metadata-only labels plan. Never opens scientific input arrays."""
from copy import deepcopy
DESIGN_SHA="74246c2f5baa2f40cc88fff248add335a2c7151028dd5193873256f55e7fd892"
TRAIN_SHA="7bc2b1f8fdd583e136621e3d25bf8aed371fa72e8d4a399f21f4ed3cd27655af"
PORTS=(20418,20424,20473,20474)
CAPS={"labels":(768,768,768,0),"node2000":(0,0,72,2000),"fixed6000_development80":(80,0,240,4000),"validation80":(80,80,240,0),"test160":(160,160,480,0)}
def selected_indices():return [4*((k*575)//191)+profile for k in range(192) for profile in range(4)]
def build(commit,inputs,routing=None):
 ids=inputs["identities"];dataset=inputs["dataset_ref"]
 if ids["role"]!="train" or ids["sha256"]!=TRAIN_SHA or dataset["role"]!="train" or dataset["sha256"]!=TRAIN_SHA or dataset["count"]!=2304:raise ValueError("true train dataset required")
 if any(len(ids[k])!=2304 for k in ("case_ids","mother_ids","profiles")):raise ValueError("complete training identity metadata")
 indices=selected_indices();route=routing or {str(i):PORTS[j%4] for j,i in enumerate(indices)}
 if set(route)!=set(map(str,indices)) or any(v not in PORTS for v in route.values()):raise ValueError("exact unique routing")
 selected=[{"case_id":ids["case_ids"][i],"mother_id":ids["mother_ids"][i],"profile":ids["profiles"][i],"role":"train","identity_sha256":inputs["identity_ref"]["sha256"],"input_case_index":i,"input_sha256":inputs["case_metadata"][str(i)]["input_sha256"],"aircraft_order_sha256":inputs["case_metadata"][str(i)]["aircraft_order_sha256"]} for i in indices]
 if len(set(c["mother_id"] for c in selected))!=192:raise ValueError("192 distinct mothers")
 plan={"group":"S1-AI17-R6","phase":"labels","authorized":True,"scientific_commit":commit,"design":inputs["design_ref"],"identity_ref":inputs["identity_ref"],"qualification_refs":inputs["qualification_refs"],"inherited_core_sha256":inputs["inherited_core_sha256"],"selected_case_indices":indices,"case_identity_metadata":selected,"training_allowed":False,"test_allowed":False,"budget":dict(zip(("prepare","T","D5","updates"),CAPS["labels"])),"tasks":{}}
 for port in PORTS:
  key="R6_labels_"+str(port)
  cfg={"schema_version":14,"research_stage":"coverage","algorithm":"ai_search_r6","name":key,"protocol_ref":plan["design"],"datasets":{"train":deepcopy(dataset)},"workers":1,"resources":{"minimum_free_bytes":8589934592,"parent_children_rss_bytes":2147483648},"hard_watchdogs":{"preparation":20,"teacher":45,"decoder":15}}
  plan["tasks"][key]={"config":cfg,"mode":"labels","dataset_role":"train","case_indices":[i for i in indices if route[str(i)]==port],"executor_port":port,"methods":["T","D5_EXACT"]}
 audit(plan);return plan

def audit(plan):
 if plan["design"]["sha256"]!=DESIGN_SHA or plan["phase"]!="labels" or plan["budget"]!=dict(zip(("prepare","T","D5","updates"),CAPS["labels"])):raise ValueError("exact labels contract")
 allindices=[]
 for task in plan["tasks"].values():
  if task["dataset_role"]!="train" or task["mode"]!="labels" or task["methods"]!=["T","D5_EXACT"] or task["executor_port"] not in PORTS:raise ValueError("role/method")
  allindices+=task["case_indices"]
 if sorted(allindices)!=selected_indices() or len(set(allindices))!=768:raise ValueError("768 exact cases once")
 return {"passed":True,"candidates":768,"mothers":192,"new_prepare":768,"new_T":768,"new_D5":768,"training_updates":0,"geometry_calls_here":0}

def digest(path):
 import hashlib
 from pathlib import Path
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()
