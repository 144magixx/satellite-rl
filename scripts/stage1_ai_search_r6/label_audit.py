"""Label qualification: only original T supplies supervision."""
import math
import numpy as np

def metrics(record):
 m=record.get("metrics",{});vals=[m.get(k) for k in ("maxload","top10_mean","CV")]
 if any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or x<0 for x in vals):raise ValueError("missing/invalid quality")
 return vals

def leq(a,b,ratio):
 return all(x<=r*y+(1e-9 if j==2 else 1e-6)+1e-12*abs(y) for j,(x,y,r) in enumerate(zip(a,b,ratio)))
def case_audit(prepared,teacher,replay,identity,cost_closed):
 for value in (teacher,replay):
  actual=value.get("identity",{k:value.get(k) for k in identity})
  if actual!=identity:raise ValueError("mixed role/input identity")
 if identity["role"]!="train":raise ValueError("train label only")
 t=teacher["result"] if "result" in teacher else teacher;d=replay["result"] if "result" in replay else replay;p=prepared["result"] if "result" in prepared else prepared
 available=t.get("status")=="available"
 qualified=bool(cost_closed and available and d.get("status")=="success" and leq(metrics(d),metrics(t),(1.02,1.05,1.05)))
 nontrivial=bool(qualified and leq(metrics(t),metrics({"metrics":p["Wenv_metrics"]}),(0.99,1.02,1.02)))
 widths=np.asarray(t.get("layout",{}).get("widths_deg",[]));old=p.get("Wenv_metrics",{})
 mechanism=bool(qualified and len(widths)>1 and np.ptp(widths)>1e-12 and (widths>.3+1e-12).any() and t["metrics"]["N"]!=old.get("N"))
 return {"qualified":qualified,"nontrivial":nontrivial,"capability":mechanism,"mother_id":identity["mother_id"],"label_origin":"original_T_terminal","D5_label_forbidden":True}
def gate(rows):
 if len(rows)!=768:raise ValueError("fixed 768 denominator required")
 q=sum(r["qualified"] for r in rows);n=sum(r["nontrivial"] for r in rows);m=len({r["mother_id"] for r in rows if r["capability"]})
 return {"passed":q>=692 and n>=154 and m>=6,"qualified":q,"nontrivial":n,"capable_mothers":m,"denominator":768,"science_updates":0}

def audit_bundle(scene,prepared,teacher,replay,identity,source_sha,cost_closed):
 from scripts.stage1_ai_search_r4.audit import audit_layout,audit_ledger,scene_sha
 from scripts.stage1_ai_search_r5.audit import audit_decoder_ledger
 p=prepared["result"];t=teacher["result"];d=replay["result"];n=len(scene["demands"])
 if scene_sha(scene)!=identity["input_sha256"]:raise ValueError("original input scene SHA")
 for value in (prepared,teacher,replay):
  if value["identity"]!=identity or value["source_sha256"]!=source_sha:raise ValueError("stage identity/source changed")
 if p["identity"]!=identity or p["source_sha256"]!=source_sha:raise ValueError("prepared original source")
 for value in (t,d):
  provenance=value.get("provenance",value)
  if any(provenance.get(k)!=v for k,v in identity.items()):raise ValueError("terminal role/input identity")
 if t.get("source_sha256",t.get("provenance",{}).get("source_sha256"))!=source_sha:raise ValueError("T original source differs")
 if not audit_ledger(p["ledger"],n,"prepare")["valid"] or not audit_ledger(t["cost"],n,"TEACHER")["valid"]:raise ValueError("prepare/T algorithm ledger invalid")
 if not audit_decoder_ledger(d["cost"],n,"D5")["valid"]:raise ValueError("D5 ledger invalid")
 if p["status"]!="available" or t["status"]!="available":return {"qualified":False,"nontrivial":False,"capability":False,"mother_id":identity["mother_id"],"label_origin":"original_T_terminal","typed_unavailable":True}
 witness=audit_layout(scene,p["raw_w"],200,float("inf"),False,False)
 if not witness["valid"] or p["Ncap"]!=min(200,n,math.ceil(1.25*witness["metrics"]["N"])) or abs(p["Omega_cap"]-witness["metrics"]["actual_Omega"])>1e-12:raise ValueError("common prepare caps invalid")
 w=audit_layout(scene,p["Wenv"],p["Ncap"],p["Omega_cap"],False)
 if not w["valid"]:raise ValueError("Wenv constraints")
 for value in (t,d):
  if value.get("status") not in ("available","success","fallback"):continue
  checked=audit_layout(scene,value["layout"],p["Ncap"],p["Omega_cap"])
  if not checked["valid"]:raise ValueError("independent terminal constraints")
  for name in ("maxload","top10_mean","CV","N"):
   x=value["metrics"][name];y=checked["metrics"][name]
   if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:raise ValueError("invalid metric")
   if abs(x-y)>(1e-9 if name=="CV" else 1e-6)+1e-12*abs(y):raise ValueError("independent terminal metrics")
 return case_audit({**p,"Wenv_metrics":w["metrics"]},teacher,replay,identity,cost_closed and sum(scene["demands"])>0)

def audit_exact_proposal(proposal,teacher,identity,prepared_source,producer_source):
 if proposal.get("schema")!="ProposalV2":raise ValueError("EXACT proposal schema")
 prov=proposal.get("provenance",{})
 if prov.get("source_method")!="EXACT" or prov.get("case_identity")!=identity or prov.get("input_sha")!=identity["input_sha256"] or prov.get("role_sha")!=identity["identity_sha256"] or prov.get("prepared_source_sha256")!=prepared_source or prov.get("producer_sha256")!=producer_source:raise ValueError("EXACT proposal provenance")
 if proposal.get("status")=="unavailable":
  if any(k in proposal for k in ("scores_n_by_K","mask_n_by_K","slot_ids","anchor_aircraft_ids")):raise ValueError("unavailable proposal payload")
  return {"passed":True,"status":"unavailable","original_T_label_only":True}
 if proposal.get("status")!="ready" or teacher["status"]!="available":raise ValueError("EXACT proposal/teacher discriminator")
 if proposal.get("case_identity")!=identity or proposal.get("input_sha")!=identity["input_sha256"] or proposal.get("role_sha")!=identity["identity_sha256"]:raise ValueError("EXACT ready identity")
 layout=teacher["layout"];slots=layout["beam_ids"];anchors=layout["anchor_aircraft_ids"];owners=np.asarray(layout["assignment"],int)
 if proposal["slot_ids"]!=slots or proposal["anchor_aircraft_ids"]!=anchors:raise ValueError("EXACT original T anchor/slot mapping")
 scores=np.asarray(proposal["scores_n_by_K"],np.float32);mask=np.asarray(proposal["mask_n_by_K"])
 if scores.shape!=(len(owners),len(slots)) or mask.shape!=scores.shape or mask.dtype!=np.bool_ or not mask.all() or not np.isfinite(scores).all():raise ValueError("EXACT array shape/mask")
 expected=np.zeros_like(scores);expected[np.arange(len(owners)),owners]=1.
 if not np.array_equal(scores,expected):raise ValueError("EXACT original T owner scores")
 if any("offset" in str(k).lower() for k in proposal):raise ValueError("EXACT offset forbidden")
 return {"passed":True,"status":"ready","original_T_label_only":True,"original_anchor_owner_arrays_recomputed":True}
